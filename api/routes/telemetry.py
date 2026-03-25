"""
Node telemetry endpoint.

POST /telemetry/heartbeat — worker reports its current status
"""

from __future__ import annotations
from datetime import datetime, timezone

from fastapi import APIRouter

from api import database as db
from api.schemas import HeartbeatRequest

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post("/heartbeat")
async def heartbeat(payload: HeartbeatRequest):
    """
    Accept a status heartbeat from a worker node.

    Workers call this every ~30 seconds while running. The dashboard uses
    these documents to display real-time CPU/RAM usage, current target,
    and to determine whether a node is ONLINE / IDLE / OFFLINE based on
    how long ago the last heartbeat arrived.

    We upsert by hostname so only the latest document per node is kept,
    rather than growing the collection unboundedly.
    """
    doc = payload.model_dump()
    doc["reported_at"] = datetime.now(timezone.utc)

    await db.node_telemetry().update_one(
        {"hostname": payload.hostname},
        {"$set": doc},
        upsert=True,
    )
    return {"status": "ok"}
