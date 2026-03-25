"""
Admin endpoints — protected, scheduler-facing routes.

POST /admin/scheduler/log   — scheduler posts its task results here so
                              Mission Control can show last-run times and errors
POST /admin/model-refresh   — trigger a model reload on the API (future use)
"""

from __future__ import annotations
from datetime import datetime, timezone

from fastapi import APIRouter

from api import database as db
from api.schemas import SchedulerLogEntry

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/scheduler/log", status_code=204)
async def scheduler_log(entry: SchedulerLogEntry):
    """
    Record a scheduler task result.

    The scheduler container calls this after each task run so operators can
    see last-run timestamps and any errors from the Mission Control dashboard
    without tailing container logs.

    Status 204 (No Content) — the scheduler doesn't need a response body.
    """
    doc = entry.model_dump()
    doc["logged_at"] = datetime.now(timezone.utc)
    await db.scheduler_log().insert_one(doc)


@router.get("/scheduler/log", response_model=list[SchedulerLogEntry])
async def get_scheduler_log(limit: int = 20):
    """
    Return the most recent scheduler task results.

    Used by Mission Control to render the scheduler status panel.
    Most-recent first; one entry per task per run.
    """
    docs = await (
        db.scheduler_log()
        .find({}, {"_id": 0})
        .sort("logged_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )
    return docs


@router.post("/model-refresh", status_code=202)
async def model_refresh():
    """
    Signal the API to reload the ExoNet model weights from disk.

    Returns 202 Accepted immediately. The actual reload is async — workers
    will pick up the new model on their next heartbeat cycle.

    Placeholder: model hot-reload logic lives in the worker, not the API.
    This endpoint exists so the scheduler and operators have a stable URL
    to trigger the refresh without SSH access to the API container.
    """
    return {"status": "accepted", "message": "Model refresh queued."}
