"""
Candidate submission and retrieval endpoints.

POST /candidates              — worker submits a scored transit candidate
GET  /candidates              — dashboard fetches recent candidates (global or per-node)
GET  /candidates/{id}         — fetch one candidate by ID (includes light curve arrays)
POST /candidates/processed    — worker marks a star as done (even with no candidate)
GET  /candidates/history      — per-node processing history
"""

from __future__ import annotations
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Query, HTTPException, Request

from api import database as db
from api.catalog import classify, t0_bjd_of
from api.routes.nodes import finder_names, node_name
from api.schemas import CandidateSubmission, CandidateResponse, ProcessedSubmission

router = APIRouter(prefix="/candidates", tags=["candidates"])

VIEW_FIELDS = ("global_view", "local_view", "odd_view", "even_view", "secondary_view",
               "transit_view_ppm", "secondary_view_ppm")


def _serialize(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serialisable dict."""
    doc["id"] = str(doc.pop("_id"))
    if isinstance(doc.get("reported_at"), datetime):
        doc["reported_at"] = doc["reported_at"]
    return doc


@router.post("", status_code=201)
async def submit_candidate(request: Request, payload: CandidateSubmission):
    """
    Accept a transit candidate from a worker node.

    Also increments the global candidates_found counter in network_stats
    so the public dashboard stat stays current without an expensive
    count() query on every page load.

    Results come from machines we don't control, so every submission is
    stored unverified until it is reproduced centrally.
    """
    now = datetime.now(timezone.utc)
    doc = payload.model_dump()
    doc["worker_hostname"] = node_name(request, payload.worker_hostname)
    doc["reported_at"] = now
    doc["verified"] = False
    doc["t0_bjd"] = t0_bjd_of(payload.mission, payload.t0)
    doc["catalog"] = await classify(payload.mission, payload.tic_id, payload.period_days,
                                    payload.t0, payload.duration_days)

    result = await db.candidates().insert_one(doc)

    # Increment the global counter atomically
    await db.network_stats().update_one(
        {"_id": "global"},
        {"$inc": {"total_candidates": 1}},
        upsert=True,
    )

    return {"id": str(result.inserted_id)}


@router.get("", response_model=list[CandidateResponse])
async def list_candidates(
    hostname: str | None = Query(None,  description="Filter to one worker node"),
    limit:    int        = Query(20,    description="Max results", ge=1, le=200),
    min_score: float     = Query(0.0,   description="Minimum ExoNet score filter"),
    status:   str | None = Query(None, pattern="^(new|unchecked|known_planet|known_candidate|known_false_positive)$",
                                 description="Catalogue cross-match status, e.g. 'new'"),
):
    """
    Return recent candidates, newest first.

    Pass ?hostname=X to scope to a single contributor's findings (used by
    the personal dashboard). Omit it for the global public view.
    Pass ?min_score=0.5 to filter out low-confidence detections.
    Pass ?status=new for signals that match nothing in the KOI/TOI/K2 catalogues.
    """
    query: dict = {}
    if hostname:
        query["worker_hostname"] = hostname
    if min_score > 0:
        query["exonet_score"] = {"$gte": min_score}
    if status:
        query["catalog.status"] = status

    cursor = db.candidates().find(
        query,
        # Exclude large arrays by default — use the /{id} endpoint for those
        projection={v: 0 for v in VIEW_FIELDS},
    ).sort("reported_at", -1).limit(limit)

    docs = await cursor.to_list(length=limit)
    names = await finder_names({d["worker_hostname"] for d in docs})
    return [_serialize(d) | {"finder": names.get(d["worker_hostname"])} for d in docs]


@router.get("/history")
async def processing_history(
    hostname: str = Query(..., description="Worker hostname"),
    limit:    int = Query(50,  description="Max results", ge=1, le=500),
):
    """
    Return the processing history for one worker node.

    Each entry represents one star that was fully processed (regardless of
    whether a candidate was found). Used to populate the MY HISTORY panel.
    """
    cursor = db.processed_log().find(
        {"worker_hostname": hostname},
    ).sort("processed_at", -1).limit(limit)

    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["id"] = str(d.pop("_id"))
    return docs


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: str):
    """
    Fetch one candidate by ID, including the full global_view and local_view
    arrays needed to render the light curve plots.
    """
    try:
        oid = ObjectId(candidate_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid candidate ID format.")

    doc = await db.candidates().find_one({"_id": oid})
    if doc is None:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    names = await finder_names({doc["worker_hostname"]})
    return _serialize(doc) | {"finder": names.get(doc["worker_hostname"])}


@router.post("/processed", status_code=201)
async def mark_processed(request: Request, payload: ProcessedSubmission):
    """
    Record that a worker finished processing a star.

    Called after every star — even ones that produced no candidate — so the
    MY HISTORY panel shows the full processing log, not just detections.
    Also increments the global stars_analyzed counter.
    """
    now = datetime.now(timezone.utc)
    payload.worker_hostname = node_name(request, payload.worker_hostname)
    doc = payload.model_dump()
    doc["processed_at"] = now

    await db.processed_log().insert_one(doc)

    # Keep the global counter in sync
    await db.network_stats().update_one(
        {"_id": "global"},
        {
            "$inc": {
                "total_stars_analyzed": 1,
                "total_compute_seconds": payload.duration_seconds,
            }
        },
        upsert=True,
    )

    # Mark the queue job as done
    await db.work_queue().update_one(
        {
            "tic_id":      payload.tic_id,
            "mission":     payload.mission,
            "sector":      payload.sector,
            "assigned_to": payload.worker_hostname,
            "status":      "assigned",
        },
        {"$set": {"status": "done"}},
    )

    return {"status": "ok"}
