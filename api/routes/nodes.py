"""
Node listing endpoint.

GET /nodes   — returns all nodes that have sent a heartbeat in the last 5 minutes

Used by the scheduler's queue-health task to log the active node count, and
by Mission Control to display the worker grid.
"""

from __future__ import annotations
import hashlib
import hmac
import secrets
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException, Query

from api import database as db
from api.config import get_settings
from api.schemas import EnrollRequest, EnrollResponse, NodeInfo

router = APIRouter(prefix="/nodes", tags=["nodes"])


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def node_name(request, claimed: str) -> str:
    """The enrolled device's name when a device token was used, else the
    hostname the caller claims (operator/API-key callers)."""
    device = getattr(request.state, "device", None)
    return device["name"] if device else claimed


@router.post("/enroll", response_model=EnrollResponse, status_code=201)
async def enroll(payload: EnrollRequest):
    """
    Join the network. Returns a per-device token, shown once; the server keeps
    only its SHA-256. Requires enroll_token when the server sets one.
    """
    required = get_settings().enroll_token
    if required and not hmac.compare_digest(payload.enroll_token, required):
        raise HTTPException(status_code=403, detail="Invalid enrollment token.")

    device_id = secrets.token_hex(8)
    token     = secrets.token_urlsafe(32)
    name      = f"{payload.hostname}-{device_id[:6]}"
    await db.devices().insert_one({
        "device_id":    device_id,
        "name":         name,
        "hostname":     payload.hostname,
        "platform":     payload.platform,
        "token_sha256": token_hash(token),
        "revoked":      False,
        "created_at":   datetime.now(timezone.utc),
    })
    return EnrollResponse(device_id=device_id, name=name, device_token=token)

# A node is considered "active" if its last heartbeat is within this window.
_ACTIVE_WINDOW_MINUTES = 5


@router.get("", response_model=list[NodeInfo])
async def list_nodes(minutes: int = Query(_ACTIVE_WINDOW_MINUTES, ge=1, le=1440)):
    """
    Return all nodes that have sent a heartbeat within the last N minutes.

    Default window is 5 minutes, matching the heartbeat interval that
    workers are expected to use. Callers can widen the window (e.g.
    minutes=60) to see recently active but currently offline nodes.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

    # One document per hostname — the heartbeat endpoint upserts by hostname
    # so this gives us the *latest* telemetry per node.
    docs = await (
        db.node_telemetry()
        .find({"reported_at": {"$gte": cutoff}}, {"_id": 0})
        .sort("reported_at", -1)
        .to_list(length=1000)
    )

    return [
        NodeInfo(
            hostname         = d["hostname"],
            uptime_seconds   = d.get("uptime_seconds", 0),
            stars_analyzed   = d.get("stars_analyzed", 0),
            candidates_found = d.get("candidates_found", 0),
            cpu_percent      = d.get("cpu_percent", 0.0),
            ram_percent      = d.get("ram_percent", 0.0),
            current_tic_id   = d.get("current_tic_id"),
            current_sector   = d.get("current_sector"),
            last_seen        = d["reported_at"],
        )
        for d in docs
    ]
