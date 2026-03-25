"""MongoDB read helpers for the Lumina dashboard."""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient, DESCENDING

_client: MongoClient | None = None


def init_db(config: dict) -> None:
    global _client
    try:
        _client = MongoClient(
            host=config["db_host"],
            port=int(config["db_port"]),
            username=config["db_username"],
            password=config["db_password"],
            authSource=config["db_auth_database"],
            serverSelectionTimeoutMS=3000,
        )
        _client.admin.command("ping")
        logging.info("Dashboard DB connected successfully.")
    except Exception as e:
        logging.warning(f"Dashboard DB init: {e}")
        _client = None


# ── Personal / per-machine queries ────────────────────────────────────────────

def get_my_latest_candidate(hostname: str) -> dict | None:
    """Most recent candidate submitted by this machine, including light curve views."""
    try:
        if _client is None:
            return None
        col = _client["lumina"]["candidates"]
        return col.find_one(
            {"worker_hostname": hostname},
            sort=[("reported_at", DESCENDING)],
        )
    except Exception as e:
        logging.warning(f"get_my_latest_candidate: {e}")
        return None


def get_my_best_candidate(hostname: str) -> dict | None:
    """Highest-scoring candidate this machine has ever found, including light curve views."""
    try:
        if _client is None:
            return None
        col = _client["lumina"]["candidates"]
        return col.find_one(
            {"worker_hostname": hostname},
            sort=[("exonet_score", DESCENDING)],
        )
    except Exception as e:
        logging.warning(f"get_my_best_candidate: {e}")
        return None


def get_my_candidates(hostname: str, n: int = 20) -> list[dict]:
    """Most recent candidates from this machine, excluding large array fields."""
    try:
        if _client is None:
            return []
        col = _client["lumina"]["candidates"]
        cursor = col.find(
            {"worker_hostname": hostname},
            projection={"global_view": 0, "local_view": 0},
        ).sort("reported_at", DESCENDING).limit(n)
        return list(cursor)
    except Exception as e:
        logging.warning(f"get_my_candidates: {e}")
        return []


def get_my_stats(hostname: str) -> dict:
    """
    Aggregate contribution stats for one machine.

    Pulls from node_telemetry (uptime, stars, cpu) and candidates collection
    (candidate count, best score). Falls back to zeros on any error so the
    dashboard always renders.
    """
    result = {
        "stars_analyzed":  0,
        "candidates_found": 0,
        "uptime_seconds":  0,
        "best_score":      0.0,
    }
    try:
        if _client is None:
            return result

        # Latest telemetry heartbeat carries cumulative stars + uptime
        telemetry = _client["lumina"]["node_telemetry"].find_one(
            {"hostname": hostname},
            sort=[("reported_at", DESCENDING)],
        )
        if telemetry:
            result["stars_analyzed"] = int(telemetry.get("stars_analyzed", 0) or 0)
            result["uptime_seconds"] = int(telemetry.get("uptime_seconds", 0) or 0)

        # Count and rank candidates from this machine
        col = _client["lumina"]["candidates"]
        result["candidates_found"] = col.count_documents({"worker_hostname": hostname})
        best = col.find_one(
            {"worker_hostname": hostname},
            sort=[("exonet_score", DESCENDING)],
            projection={"exonet_score": 1},
        )
        if best:
            result["best_score"] = float(best.get("exonet_score", 0.0) or 0.0)

    except Exception as e:
        logging.warning(f"get_my_stats: {e}")

    return result


def get_my_telemetry(hostname: str) -> dict | None:
    """Latest heartbeat document for this machine."""
    try:
        if _client is None:
            return None
        return _client["lumina"]["node_telemetry"].find_one(
            {"hostname": hostname},
            sort=[("reported_at", DESCENDING)],
        )
    except Exception as e:
        logging.warning(f"get_my_telemetry: {e}")
        return None
