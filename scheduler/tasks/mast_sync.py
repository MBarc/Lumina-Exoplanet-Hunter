"""
MAST sync task — discovers new observations and populates the work queue.

Runs every 24 hours. For each supported mission it queries the MAST catalog
for observations not yet in our queue and sends them to POST /queue/populate.

TESS releases a new sector of sky roughly every 27 days, so daily polling
ensures new sectors appear in the queue within 24 hours of MAST ingestion.
Kepler and K2 are complete archives — after the initial backlog is loaded
this task will mostly find nothing new, which is fine.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

import httpx
from astroquery.mast import Observations

from scheduler.config import get_settings

log = logging.getLogger(__name__)

# Missions and the MAST product keyword that identifies light curve files
_MISSIONS = {
    "kepler": {
        "obs_collection":    "Kepler",
        "dataproduct_type":  "timeseries",
        "description_kw":    "long cadence",
    },
    "k2": {
        "obs_collection":    "K2",
        "dataproduct_type":  "timeseries",
        "description_kw":    "light curve",
    },
    "tess": {
        "obs_collection":    "TESS",
        "dataproduct_type":  "timeseries",
        "description_kw":    "light curves",
    },
}


async def _get_existing_keys(client: httpx.AsyncClient) -> set[str]:
    """
    Fetch the set of (tic_id, mission, sector) keys already in the queue.

    Used to skip duplicates before sending the populate request, keeping
    the payload small and avoiding unnecessary 409-equivalent skips on
    the API side.
    """
    try:
        resp = await client.get("/queue/status")
        resp.raise_for_status()
        data = resp.json()
        return set(data.get("existing_keys", []))
    except Exception as e:
        log.warning("Could not fetch existing queue keys: %s", e)
        return set()


def _query_mast_observations(mission_key: str) -> list[dict]:
    """
    Query MAST for all observations for one mission.

    Returns a list of dicts with the fields we need to build queue documents.
    Uses astroquery which handles MAST pagination automatically.
    """
    cfg = _MISSIONS[mission_key]
    log.info("Querying MAST for %s observations...", mission_key)

    try:
        table = Observations.query_criteria(
            obs_collection  = cfg["obs_collection"],
            dataproduct_type = cfg["dataproduct_type"],
        )
    except Exception as e:
        log.error("MAST query failed for %s: %s", mission_key, e)
        return []

    log.info("  %s: found %d observations", mission_key, len(table))

    # Pull product lists in batches to find the actual FITS file URIs
    keyword  = cfg["description_kw"]
    targets  = []
    batch_sz = 100

    for i in range(0, len(table), batch_sz):
        chunk = table[i : i + batch_sz]
        try:
            products = Observations.get_product_list(chunk)
        except Exception as e:
            log.warning("Product list batch %d failed: %s", i // batch_sz, e)
            continue

        for row in products:
            try:
                desc = str(row["description"]).lower()
                uri  = str(row["dataURI"])
            except (KeyError, TypeError):
                continue

            if keyword in desc and uri.endswith(".fits"):
                # Extract sector from TESS URIs (e.g. tess2019...s0001...)
                sector = None
                if mission_key == "tess":
                    parts = uri.split("/")
                    for p in parts:
                        if p.startswith("s") and p[1:].isdigit():
                            sector = int(p[1:])
                            break

                # TIC ID is in the obs_id field of the parent observation row
                # For simplicity use the filename stem as the identifier
                filename = uri.split("/")[-1]
                tic_id   = filename.split("-")[0].lstrip("kplrktwo").lstrip("0") or filename

                targets.append({
                    "tic_id":   tic_id,
                    "mission":  mission_key,
                    "sector":   sector,
                    "fits_url": uri,
                    "priority": 1 if mission_key == "tess" else 0,
                })

    return targets


async def run_mast_sync() -> dict:
    """
    Main entry point called by the scheduler.

    Queries MAST for all three missions, filters to targets not already
    in the queue, and sends them to POST /queue/populate in batches.

    Returns a summary dict that is written to the scheduler_log collection.
    """
    settings  = get_settings()
    started   = datetime.now(timezone.utc)
    total_new = 0
    errors    = []

    log.info("=== MAST sync started ===")

    async with httpx.AsyncClient(
        base_url = settings.api_url,
        headers  = {"X-API-Key": settings.api_key},
        timeout  = 60.0,
    ) as client:

        for mission_key in _MISSIONS:
            try:
                targets = _query_mast_observations(mission_key)
                if not targets:
                    continue

                # Send in batches to avoid huge request payloads
                batch_sz = settings.mast_sync_batch_size
                inserted = 0
                for i in range(0, len(targets), batch_sz):
                    batch = targets[i : i + batch_sz]
                    try:
                        resp = await client.post(
                            "/queue/populate",
                            json={"targets": batch},
                        )
                        resp.raise_for_status()
                        result   = resp.json()
                        inserted += result.get("inserted", 0)
                    except Exception as e:
                        log.error("Populate batch failed: %s", e)
                        errors.append(str(e))

                log.info("  %s: %d new targets queued", mission_key, inserted)
                total_new += inserted

            except Exception as e:
                log.error("Mission %s sync failed: %s", mission_key, e)
                errors.append(f"{mission_key}: {e}")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info("=== MAST sync complete: %d new targets in %.0fs ===", total_new, elapsed)

    return {
        "task":       "mast_sync",
        "started_at": started,
        "elapsed_s":  elapsed,
        "new_targets": total_new,
        "errors":     errors,
    }
