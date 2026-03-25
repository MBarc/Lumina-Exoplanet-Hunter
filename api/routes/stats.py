"""
Network stats, leaderboard, and activity endpoints.

GET /stats              — global counters for the public dashboard hero row
GET /stats/leaderboard  — top contributors ranked by stars analyzed
GET /stats/activity     — candidates per hour (sparkline data)
GET /stats/my           — personal stats for one worker node
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Query

from api import database as db
from api.schemas import NetworkStats, LeaderboardEntry, ActivityPoint

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=NetworkStats)
async def get_network_stats():
    """
    Global network counters for the public Mission Control hero row.

    Stars analyzed and candidates found are maintained as atomic counters
    in a single "global" document rather than computed with count() on
    every request — avoids a full collection scan at scale.
    """
    global_doc = await db.network_stats().find_one({"_id": "global"}) or {}

    # Active nodes: distinct hostnames with a heartbeat in the last 5 minutes
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
    active_hostnames = await db.node_telemetry().distinct(
        "hostname", {"reported_at": {"$gte": cutoff}}
    )

    queue_remaining = await db.work_queue().count_documents(
        {"status": {"$in": ["queued", "assigned"]}}
    )

    total_seconds = float(global_doc.get("total_compute_seconds", 0))

    return NetworkStats(
        active_nodes     = len(active_hostnames),
        stars_analyzed   = int(global_doc.get("total_stars_analyzed", 0)),
        candidates_found = int(global_doc.get("total_candidates", 0)),
        compute_hours    = round(total_seconds / 3600, 1),
        queue_remaining  = queue_remaining,
    )


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(limit: int = Query(10, ge=1, le=50)):
    """
    Top contributors ranked by total stars analyzed.

    Aggregates the processed_log collection to sum per-hostname.
    For large collections this could be pre-materialised on a schedule,
    but at current scale a live aggregation is fast enough.
    """
    pipeline = [
        {"$group": {
            "_id":              "$worker_hostname",
            "stars_analyzed":   {"$sum": 1},
            "candidates_found": {"$sum": "$candidates_found"},
        }},
        {"$sort": {"stars_analyzed": -1}},
        {"$limit": limit},
    ]
    docs = await db.processed_log().aggregate(pipeline).to_list(length=limit)

    return [
        LeaderboardEntry(
            rank             = i + 1,
            hostname         = d["_id"],
            stars_analyzed   = d["stars_analyzed"],
            candidates_found = d["candidates_found"],
        )
        for i, d in enumerate(docs)
    ]


@router.get("/activity", response_model=list[ActivityPoint])
async def get_activity(hours: int = Query(24, ge=1, le=168)):
    """
    Candidates found per hour for the last N hours.

    Used to draw the network activity sparkline on the public dashboard.
    Groups by truncated hour using a date aggregation pipeline.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    pipeline = [
        {"$match": {"reported_at": {"$gte": cutoff}}},
        {"$group": {
            "_id": {
                "$dateToString": {
                    "format": "%Y-%m-%dT%H:00:00Z",
                    "date":   "$reported_at",
                }
            },
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]
    docs = await db.candidates().aggregate(pipeline).to_list(length=hours)
    return [ActivityPoint(hour=d["_id"], count=d["count"]) for d in docs]


@router.get("/my")
async def get_my_stats(hostname: str = Query(..., description="Worker hostname")):
    """
    Personal contribution stats for one worker node.

    Combines telemetry (uptime, cpu) with aggregated processed_log counts.
    Used by the MY CONTRIBUTIONS panel in the contributor dashboard.
    """
    # Latest telemetry heartbeat
    telemetry = await db.node_telemetry().find_one(
        {"hostname": hostname},
        sort=[("reported_at", -1)],
    ) or {}

    # Aggregate personal counts from processed_log
    pipeline = [
        {"$match": {"worker_hostname": hostname}},
        {"$group": {
            "_id":              None,
            "stars_analyzed":   {"$sum": 1},
            "candidates_found": {"$sum": "$candidates_found"},
            "compute_seconds":  {"$sum": "$duration_seconds"},
        }},
    ]
    agg = await db.processed_log().aggregate(pipeline).to_list(length=1)
    agg = agg[0] if agg else {}

    # Highest ExoNet score this node has ever found
    best = await db.candidates().find_one(
        {"worker_hostname": hostname},
        sort=[("exonet_score", -1)],
        projection={"exonet_score": 1},
    )

    return {
        "hostname":        hostname,
        "stars_analyzed":  agg.get("stars_analyzed", 0),
        "candidates_found": agg.get("candidates_found", 0),
        "compute_hours":   round(agg.get("compute_seconds", 0) / 3600, 2),
        "best_score":      best["exonet_score"] if best else None,
        "uptime_seconds":  telemetry.get("uptime_seconds", 0),
        "cpu_percent":     telemetry.get("cpu_percent", 0),
        "ram_percent":     telemetry.get("ram_percent", 0),
        "current_tic_id":  telemetry.get("current_tic_id"),
        "current_sector":  telemetry.get("current_sector"),
        "last_seen":       telemetry.get("reported_at"),
    }
