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


def get_recent_candidates(n: int = 20) -> list[dict]:
    """Return n most recent candidates, excluding global_view and local_view arrays."""
    try:
        if _client is None:
            return []
        col = _client["lumina"]["candidates"]
        cursor = col.find(
            {},
            projection={
                "global_view": 0,
                "local_view": 0,
            },
        ).sort("reported_at", DESCENDING).limit(n)
        return list(cursor)
    except Exception as e:
        logging.warning(f"get_recent_candidates: {e}")
        return []


def get_top_candidate_with_views() -> dict | None:
    """Return the most recent candidate document including global_view and local_view."""
    try:
        if _client is None:
            return None
        col = _client["lumina"]["candidates"]
        doc = col.find_one({}, sort=[("reported_at", DESCENDING)])
        return doc
    except Exception as e:
        logging.warning(f"get_top_candidate_with_views: {e}")
        return None


def get_node_telemetry() -> list[dict]:
    """Return the latest heartbeat per hostname from the last 5 minutes."""
    try:
        if _client is None:
            return []
        col = _client["lumina"]["node_telemetry"]
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)

        # Aggregate: filter to last 5 min, sort descending, group by hostname keeping first doc
        pipeline = [
            {"$match": {"reported_at": {"$gte": cutoff}}},
            {"$sort": {"reported_at": DESCENDING}},
            {
                "$group": {
                    "_id": "$hostname",
                    "hostname": {"$first": "$hostname"},
                    "reported_at": {"$first": "$reported_at"},
                    "uptime_seconds": {"$first": "$uptime_seconds"},
                    "stars_analyzed": {"$first": "$stars_analyzed"},
                    "candidates_found": {"$first": "$candidates_found"},
                    "cpu_percent": {"$first": "$cpu_percent"},
                    "current_tic_id": {"$first": "$current_tic_id"},
                    "current_sector": {"$first": "$current_sector"},
                }
            },
            {"$sort": {"hostname": 1}},
        ]
        return list(col.aggregate(pipeline))
    except Exception as e:
        logging.warning(f"get_node_telemetry: {e}")
        return []


def get_network_stats() -> dict:
    """Return global network stats plus queue/processing/completed sector counts.

    Returned dict keys:
        total_stars_analyzed  int
        total_candidates      int
        active_nodes          int   — hostnames seen in last 5 min
        sectors_queued        int   — docs in sectors.queue
    """
    result = {
        "total_stars_analyzed": 0,
        "total_candidates": 0,
        "active_nodes": 0,
        "sectors_queued": 0,
    }
    try:
        if _client is None:
            return result

        # Global stats document
        stats_doc = _client["lumina"]["network_stats"].find_one({"_id": "global"})
        if stats_doc:
            result["total_stars_analyzed"] = int(stats_doc.get("total_stars_analyzed", 0))
            result["total_candidates"] = int(stats_doc.get("total_candidates", 0))

        # Active nodes: distinct hostnames with a heartbeat in the last 5 minutes
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        active_nodes = _client["lumina"]["node_telemetry"].distinct(
            "hostname", {"reported_at": {"$gte": cutoff}}
        )
        result["active_nodes"] = len(active_nodes)

        # Sectors queued
        result["sectors_queued"] = _client["sectors"]["queue"].count_documents({})

    except Exception as e:
        logging.warning(f"get_network_stats: {e}")

    return result
