"""
Work queue endpoints.

GET  /queue/next        — worker polls for its next batch of jobs
POST /queue/release     — worker releases unfinished jobs on shutdown
POST /queue/populate    — admin/cron adds new targets to the queue
"""

from __future__ import annotations
from datetime import datetime, timezone, timedelta

from bson import ObjectId
from fastapi import APIRouter, Depends, Query, HTTPException

from api import database as db
from api.config import get_settings
from api.schemas import QueueItem, PopulateRequest

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/next", response_model=list[QueueItem])
async def get_next_jobs(
    hostname: str = Query(..., description="Calling worker's hostname"),
    limit:    int = Query(10,  description="Number of jobs to claim", ge=1, le=100),
):
    """
    Assign the next N queued jobs to the requesting worker.

    Jobs are selected by (priority DESC, created_at ASC) — higher-priority
    targets (e.g. new TESS sectors) are processed before old backlog items,
    and within the same priority older jobs go first.

    Claimed jobs are marked status="assigned" with the worker's hostname and
    a timestamp. If the worker goes offline before reporting completion, a
    background cleanup task (run on each request) re-queues jobs whose
    assigned_at is older than job_timeout_seconds.
    """
    settings = get_settings()
    now      = datetime.now(timezone.utc)
    col      = db.work_queue()

    # ── Re-queue stalled jobs from any worker ─────────────────────────────────
    # Do this before assigning new work so freed slots are available immediately.
    cutoff = now - timedelta(seconds=settings.job_timeout_seconds)
    await col.update_many(
        {"status": "assigned", "assigned_at": {"$lt": cutoff}},
        {"$set": {"status": "queued", "assigned_to": None, "assigned_at": None}},
    )

    # ── Claim jobs atomically using find-and-modify ───────────────────────────
    # We claim one job at a time in a loop rather than bulk-updating, so that
    # two workers racing for the same job never both win it.
    claimed: list[QueueItem] = []
    for _ in range(limit):
        doc = await col.find_one_and_update(
            {"status": "queued"},
            {"$set": {
                "status":      "assigned",
                "assigned_to": hostname,
                "assigned_at": now,
            }},
            sort=[("priority", -1), ("created_at", 1)],
            return_document=True,
        )
        if doc is None:
            break   # queue is empty
        claimed.append(QueueItem(
            job_id   = str(doc["_id"]),
            tic_id   = doc["tic_id"],
            mission  = doc["mission"],
            sector   = doc.get("sector"),
            fits_url = doc["fits_url"],
        ))

    return claimed


@router.post("/release")
async def release_jobs(hostname: str = Query(...)):
    """
    Re-queue all jobs currently assigned to this worker.

    Called on graceful worker shutdown so jobs aren't held until the
    stall timeout expires.
    """
    result = await db.work_queue().update_many(
        {"status": "assigned", "assigned_to": hostname},
        {"$set": {"status": "queued", "assigned_to": None, "assigned_at": None}},
    )
    return {"released": result.modified_count}


@router.post("/populate")
async def populate_queue(payload: PopulateRequest):
    """
    Add new observation targets to the work queue.

    Called by the MAST sync cron job when new TESS sectors are released or
    when an operator wants to re-enqueue targets for reprocessing.

    Duplicates (same tic_id + mission + sector) are silently skipped via
    the unique compound index — safe to call repeatedly with the same list.
    """
    now = datetime.now(timezone.utc)
    inserted = 0
    skipped  = 0

    for t in payload.targets:
        doc = {
            "tic_id":      str(t["tic_id"]),
            "mission":     t["mission"],
            "sector":      t.get("sector"),
            "fits_url":    t["fits_url"],
            "priority":    int(t.get("priority", 0)),
            "status":      "queued",
            "assigned_to": None,
            "assigned_at": None,
            "created_at":  now,
        }
        try:
            await db.work_queue().insert_one(doc)
            inserted += 1
        except Exception:
            # Unique index violation — target already exists
            skipped += 1

    return {"inserted": inserted, "skipped": skipped}
