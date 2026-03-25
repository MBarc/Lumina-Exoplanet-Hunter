"""
MAST sync task — discovers new observations and populates the work queue.

Runs every 24 hours. Issues a single broad query to MAST for ALL timeseries
observations across every mission, then selects only photometric light curve
products using the structured `productSubGroupDescription` field.

Why this works
--------------
MAST uses a controlled vocabulary for productSubGroupDescription. Every
photometric light curve product in the archive — regardless of mission —
ends with "LC":

  LC   →  TESS light curves
  LLC  →  Kepler / K2 long cadence light curves
  SLC  →  Kepler / K2 short cadence light curves

This is not a free-text keyword match. It is a structured field that MAST
applies consistently. Any future photometry mission (Roman, Plato, etc.)
onboarded to MAST will almost certainly follow the same convention, meaning
new missions are picked up automatically on the next 24-hour sync with no
code changes required.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

import httpx
from astroquery.mast import Observations

from scheduler.config import get_settings

log = logging.getLogger(__name__)


def _extract_target_id(uri: str, obs_id: str) -> str:
    """
    Best-effort extraction of a numeric target identifier from a FITS URI.

    Different missions encode the target in their filenames differently:
      Kepler/K2:  kplr009002278_llc.fits   →  "9002278"
      TESS:       tess...-0000000260647166-...fits  →  "260647166"
      Future:     anything else  →  fall back to obs_id from the parent row

    We strip known mission prefixes then pull the first uninterrupted run
    of digits, which is the TIC / KIC / EPIC ID. Leading zeros are removed.
    """
    filename = uri.split("/")[-1]
    stem = filename.split(".")[0]  # drop extension

    # Strip known mission-specific filename prefixes
    for prefix in ("tess", "kplr", "ktwo", "hlsp"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
            break

    # Extract the first digit run after stripping separators
    digits = ""
    for ch in stem.lstrip("0_-"):
        if ch.isdigit():
            digits += ch
        elif digits:
            break  # stop at the first non-digit after we've collected some

    # Fall back to obs_id (the parent observation identifier) for any mission
    # whose filename format we don't recognise yet.
    return digits or obs_id or filename


def _extract_sector(uri: str, obs_collection: str) -> int | None:
    """
    Extract a sector / campaign / quarter number from the FITS URI path.

    MAST encodes the observing segment in the directory structure:
      TESS:  .../s0001/...  (sector)
      K2:    .../c001/...   (campaign)

    Kepler quarters are not encoded in the URI path — return None.
    Unknown future missions also return None rather than guessing.
    """
    for part in uri.lower().split("/"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])   # TESS sector
        if part.startswith("c") and part[1:].isdigit():
            return int(part[1:])   # K2 campaign
    return None


def _query_all_lightcurves() -> list[dict]:
    """
    Query MAST for every timeseries light curve product across all missions.

    Steps:
      1. Single broad MAST query — all dataproduct_type=timeseries, no
         collection filter.
      2. Fetch product lists in batches of 100 observations.
      3. Keep only products where productSubGroupDescription ends with "LC"
         and the file is a FITS file.
      4. Build a lookup from obs_id → obs_collection so each product row
         knows which mission it belongs to, without re-querying MAST.

    Returns a flat list of target dicts ready to POST to /queue/populate.
    """
    log.info("Querying MAST for all timeseries observations (no collection filter)...")
    try:
        table = Observations.query_criteria(dataproduct_type="timeseries")
    except Exception as e:
        log.error("MAST query failed: %s", e)
        return []

    log.info("  Found %d total timeseries observations across all missions", len(table))

    targets:          list[dict] = []
    seen_collections: set[str]   = set()
    batch_sz = 100

    for i in range(0, len(table), batch_sz):
        chunk = table[i : i + batch_sz]

        # Build obs_id → obs_collection map for this chunk so product rows
        # can be attributed to the right mission without an extra MAST call.
        collection_map: dict[str, str] = {
            str(row["obs_id"]): str(row["obs_collection"])
            for row in chunk
        }

        try:
            products = Observations.get_product_list(chunk)
        except Exception as e:
            log.warning("Product list batch %d failed: %s", i // batch_sz, e)
            continue

        for row in products:
            try:
                sub_group = str(row.get("productSubGroupDescription") or "")
                uri       = str(row["dataURI"])
            except (KeyError, TypeError):
                continue

            # The structured photometry check — any value ending in "LC"
            # (LC, LLC, SLC) is a light curve file.
            if not sub_group.upper().endswith("LC"):
                continue
            if not uri.endswith(".fits"):
                continue

            # Map back to the parent observation's collection name.
            # parent_obsid is the link from product → observation.
            parent_id      = str(row.get("parent_obsid") or row.get("obs_id") or "")
            obs_collection = collection_map.get(parent_id, "unknown")

            if obs_collection not in seen_collections:
                log.info("  Discovered collection: %s", obs_collection)
                seen_collections.add(obs_collection)

            # Active missions (TESS releases new data every ~27 days) get
            # higher priority so fresh sectors are processed before backlog.
            priority = 1 if obs_collection.upper() == "TESS" else 0

            targets.append({
                "tic_id":   _extract_target_id(uri, parent_id),
                "mission":  obs_collection.lower(),
                "sector":   _extract_sector(uri, obs_collection),
                "fits_url": uri,
                "priority": priority,
            })

    log.info(
        "  Scan complete: %d light curve targets across %d collections: %s",
        len(targets), len(seen_collections), sorted(seen_collections),
    )
    return targets


async def run_mast_sync() -> dict:
    """
    Main entry point called by the scheduler every 24 hours.

    Queries MAST for all light curve products, sends new targets to
    POST /queue/populate in batches, and returns a summary dict that
    is written to the scheduler_log collection for Mission Control.

    Duplicate targets (same tic_id + mission + sector) are silently skipped
    by the unique index on the work_queue collection — safe to re-run.
    """
    settings   = get_settings()
    started    = datetime.now(timezone.utc)
    total_new  = 0
    errors:    list[str] = []

    log.info("=== MAST sync started ===")

    targets = _query_all_lightcurves()
    if not targets:
        log.warning("No targets returned from MAST — nothing to queue.")

    async with httpx.AsyncClient(
        base_url = settings.api_url,
        headers  = {"X-API-Key": settings.api_key},
        timeout  = 60.0,
    ) as client:

        batch_sz = settings.mast_sync_batch_size
        for i in range(0, len(targets), batch_sz):
            batch = targets[i : i + batch_sz]
            try:
                resp = await client.post("/queue/populate", json={"targets": batch})
                resp.raise_for_status()
                result    = resp.json()
                inserted  = result.get("inserted", 0)
                total_new += inserted
                log.info(
                    "  Batch %d/%d: %d inserted, %d skipped (already queued)",
                    i // batch_sz + 1,
                    -(-len(targets) // batch_sz),  # ceiling division
                    inserted,
                    result.get("skipped", 0),
                )
            except Exception as e:
                log.error("Populate batch %d failed: %s", i // batch_sz, e)
                errors.append(str(e))

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info(
        "=== MAST sync complete: %d new targets queued in %.0fs ===",
        total_new, elapsed,
    )

    return {
        "task":        "mast_sync",
        "started_at":  started,
        "elapsed_s":   elapsed,
        "discovered":  len(targets),
        "inserted":    total_new,
        "errors":      errors,
    }
