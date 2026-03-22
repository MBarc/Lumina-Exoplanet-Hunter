"""
Full pipeline worker for Lumina data gathering.

Lifecycle per sector
--------------------
1. Claim a sector from sectors.queue → sectors.processing
2. Query MAST for all 2-min cadence targets in that sector
3. For each TIC ID:
   a. Download the FITS light curve to a temp directory
   b. Preprocess → TransitCandidate list
   c. Run ExoNetInference → probability scores
   d. Insert each candidate into lumina.candidates
   e. Write a telemetry heartbeat to lumina.node_telemetry
   f. Upsert lumina.network_stats global counters
   g. Delete the temp FITS file
4. Mark sector complete → sectors.completed
5. Loop back to step 1
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from pymongo import MongoClient

CONFIG_PATH = r"C:\Program Files\Lumina\Data\config\config.json"
_HOSTNAME = socket.gethostname()


# ── Config / startup helpers ──────────────────────────────────────────────────

def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "data_gathering.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def check_config(config_path: str = CONFIG_PATH) -> dict | None:
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Config not found: {config_path}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in config: {config_path}")
    return None


def check_log_directory(log_dir: str) -> bool:
    if not os.path.isdir(log_dir):
        logging.error(f"Log directory missing: {log_dir}")
        return False
    return True


def check_db_connection(config: dict, client: MongoClient) -> bool:
    try:
        client.admin.command("ping")
        logging.info(
            f"MongoDB connected  {config['db_host']}:{config['db_port']}"
        )
        return True
    except Exception as e:
        logging.error(f"MongoDB ping failed: {e}")
        return False


def on_startup() -> None:
    logging.info("Data Gathering Service starting …")


# ── MAST helpers ──────────────────────────────────────────────────────────────

def _query_tic_ids_in_sector(sector_number: int) -> list[str]:
    """Return TIC IDs of 2-min cadence targets in a TESS sector."""
    from astroquery.mast import Observations  # deferred import

    logging.info(f"Querying MAST for TIC IDs in sector {sector_number} …")
    obs_table = Observations.query_criteria(
        obs_collection="TESS",
        sequence_number=sector_number,
        dataproduct_type="timeseries",
    )
    tic_ids: list[str] = []
    seen: set[str] = set()
    for row in obs_table:
        target = str(row["target_name"]).strip()
        if target.startswith("TIC "):
            tid = target[4:].strip()
            if tid and tid not in seen:
                tic_ids.append(tid)
                seen.add(tid)
    logging.info(f"Found {len(tic_ids)} TIC IDs in sector {sector_number}")
    return tic_ids


def _download_lc_fits(tic_id: str, sector: int, download_dir: str) -> str | None:
    """
    Download the 2-min light curve FITS for one TIC ID / sector.
    Returns the local file path, or None if unavailable.
    """
    from astroquery.mast import Observations  # deferred import

    try:
        obs_table = Observations.query_criteria(
            obs_collection="TESS",
            target_name=f"TIC {tic_id}",
            sequence_number=sector,
            dataproduct_type="timeseries",
        )
        if len(obs_table) == 0:
            return None

        products = Observations.get_product_list(obs_table[:1])
        lc_products = Observations.filter_products(
            products,
            productSubGroupDescription="LC",
            extension="fits",
        )
        if len(lc_products) == 0:
            return None

        manifest = Observations.download_products(
            lc_products[:1],
            download_dir=download_dir,
            cache=False,
        )
        return str(manifest["Local Path"][0])
    except Exception as e:
        logging.warning(f"FITS download failed  TIC {tic_id}: {e}")
        return None


# ── Telemetry helpers ─────────────────────────────────────────────────────────

def _write_telemetry(
    db: MongoClient,
    *,
    uptime_s: float,
    stars_analyzed: int,
    candidates_found: int,
    current_tic_id: str,
    current_sector: int,
) -> None:
    try:
        db["lumina"]["node_telemetry"].insert_one({
            "hostname": _HOSTNAME,
            "reported_at": datetime.now(timezone.utc),
            "uptime_seconds": uptime_s,
            "stars_analyzed": stars_analyzed,
            "candidates_found": candidates_found,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "current_tic_id": current_tic_id,
            "current_sector": current_sector,
        })
    except Exception as e:
        logging.warning(f"Telemetry write failed: {e}")


def _upsert_network_stats(db: MongoClient, new_stars: int, new_candidates: int) -> None:
    try:
        db["lumina"]["network_stats"].update_one(
            {"_id": "global"},
            {
                "$inc": {
                    "total_stars_analyzed": new_stars,
                    "total_candidates": new_candidates,
                },
                "$set": {"last_updated": datetime.now(timezone.utc)},
            },
            upsert=True,
        )
    except Exception as e:
        logging.warning(f"network_stats upsert failed: {e}")


# ── Per-TIC pipeline ──────────────────────────────────────────────────────────

def _process_tic(
    tic_id: str,
    sector: int,
    db: MongoClient,
    model,
    tmp_dir: str,
) -> int:
    """
    Download → preprocess → infer → store one TIC ID.
    Returns number of candidates inserted.
    """
    from ml.preprocess import preprocess  # deferred import

    fits_path = _download_lc_fits(tic_id, sector, tmp_dir)
    if fits_path is None:
        logging.debug(f"No FITS  TIC {tic_id}  sector {sector} — skip")
        return 0

    try:
        candidates = preprocess(fits_path)
    except Exception as e:
        logging.warning(f"Preprocess failed  TIC {tic_id}: {e}")
        return 0
    finally:
        try:
            os.remove(fits_path)
        except OSError:
            pass

    if not candidates:
        return 0

    try:
        scores = model.predict_batch(candidates)
    except Exception as e:
        logging.warning(f"Inference failed  TIC {tic_id}: {e}")
        return 0

    now = datetime.now(timezone.utc)
    docs = [
        {
            "tic_id": tic_id,
            "sector": sector,
            "fits_filename": os.path.basename(fits_path),
            "worker_hostname": _HOSTNAME,
            "reported_at": now,
            "period_days": cand.period,
            "t0_btjd": cand.t0,
            "duration_days": cand.duration,
            "depth_ppm": cand.depth * 1_000_000,
            "bls_power": cand.bls_power,
            "exonet_score": score,
            "label": "CANDIDATE" if score >= 0.5 else "FALSE_POSITIVE",
            "global_view": cand.global_view.tolist(),
            "local_view": cand.local_view.tolist(),
        }
        for cand, score in zip(candidates, scores)
    ]

    db["lumina"]["candidates"].insert_many(docs)
    logging.info(
        f"TIC {tic_id}: {len(docs)} candidate(s)  best {max(scores):.3f}"
    )
    return len(docs)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_data_gathering(
    stop_event,
    config_path: str = CONFIG_PATH,
) -> None:
    """Main entrypoint.  Runs until stop_event is set."""

    config = check_config(config_path)
    if config is None:
        logging.error("Cannot start — missing or invalid config")
        return

    log_dir = config.get(
        "log_directory",
        r"C:\Program Files\Lumina\Data\logs",
    )
    _setup_logging(log_dir)

    if not check_log_directory(log_dir):
        return

    db = MongoClient(
        host=config["db_host"],
        port=int(config["db_port"]),
        username=config["db_username"],
        password=config["db_password"],
        authSource=config["db_auth_database"],
        serverSelectionTimeoutMS=10_000,
    )
    if not check_db_connection(config, db):
        return

    # Load ExoNet inference model
    model_path = config.get(
        "model_path",
        r"C:\Program Files\Lumina\Data\models\exonet.onnx",
    )
    try:
        from ml.inference import ExoNetInference
        model = ExoNetInference(model_path)
        logging.info(f"ExoNet model loaded: {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Warm-up psutil (first call always returns 0.0)
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        pass

    service_start = time.monotonic()
    stars_analyzed = 0
    candidates_found = 0

    logging.info("Data Gathering Service ready — entering main loop")

    while not stop_event.is_set():

        # ── Claim a sector ─────────────────────────────────────────────────
        # Check for a sector this machine started but didn't finish (crash recovery)
        processing_doc = db["sectors"]["processing"].find_one({"worker": _HOSTNAME})

        if processing_doc is None:
            # Pull the next sector off the queue atomically
            queue_doc = db["sectors"]["queue"].find_one_and_delete({})
            if queue_doc is None:
                logging.info("Sector queue empty — sleeping 60 s")
                stop_event.wait(60)
                continue

            processing_doc = {
                "data": queue_doc,
                "worker": _HOSTNAME,
                "start_time": time.time(),
                "processed_tic_ids": [],
            }
            db["sectors"]["processing"].insert_one(processing_doc)
            logging.info(f"Claimed sector {queue_doc.get('sector_number')}")
        else:
            logging.info(
                f"Resuming sector {processing_doc['data'].get('sector_number')}"
            )

        sector_number = int(processing_doc["data"]["sector_number"])
        done_tics: set[str] = set(processing_doc.get("processed_tic_ids", []))

        # ── Discover TIC IDs ───────────────────────────────────────────────
        try:
            tic_ids = _query_tic_ids_in_sector(sector_number)
        except Exception as e:
            logging.error(f"MAST sector query failed: {e}")
            stop_event.wait(300)
            continue

        # ── Process each TIC ID ────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp_dir:
            for tic_id in tic_ids:
                if stop_event.is_set():
                    break
                if tic_id in done_tics:
                    continue

                n_cands = _process_tic(tic_id, sector_number, db, model, tmp_dir)
                stars_analyzed += 1
                candidates_found += n_cands

                # Record TIC as done so we can resume after a crash
                db["sectors"]["processing"].update_one(
                    {"_id": processing_doc["_id"]},
                    {"$addToSet": {"processed_tic_ids": tic_id}},
                )
                done_tics.add(tic_id)

                _write_telemetry(
                    db,
                    uptime_s=time.monotonic() - service_start,
                    stars_analyzed=stars_analyzed,
                    candidates_found=candidates_found,
                    current_tic_id=tic_id,
                    current_sector=sector_number,
                )
                _upsert_network_stats(db, new_stars=1, new_candidates=n_cands)

        if stop_event.is_set():
            # Keep the sector in processing so the next startup can resume
            logging.info(
                "Stop event — sector kept in processing for crash-safe resume"
            )
            break

        # ── Mark sector complete ───────────────────────────────────────────
        db["sectors"]["completed"].insert_one({
            "data": processing_doc["data"],
            "worker": _HOSTNAME,
            "start_time": processing_doc.get("start_time"),
            "end_time": time.time(),
            "tic_count": len(done_tics),
        })
        db["sectors"]["processing"].delete_one({"_id": processing_doc["_id"]})
        logging.info(
            f"Sector {sector_number} complete — "
            f"{len(done_tics)} stars  {candidates_found} candidates total"
        )

    logging.info("Data Gathering Service stopped")
    db.close()
