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

from fastapi import APIRouter, HTTPException, Query, Request

from api import database as db
from api.config import get_settings
from api.schemas import EnrollRequest, EnrollResponse, FinderProfile, NodeInfo

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
    settings = get_settings()
    if not settings.open_enrollment:
        # Invite-only unless open enrollment is explicitly turned on; with no
        # enroll_token configured, nobody can join.
        if not settings.enroll_token or not hmac.compare_digest(payload.enroll_token, settings.enroll_token):
            raise HTTPException(status_code=403, detail="Invalid enrollment token.")

    device_id = secrets.token_hex(8)
    token     = secrets.token_urlsafe(32)
    # Public identity carries no hostname (hostnames often contain real names);
    # the hostname is kept private for operators only.
    name      = f"node-{device_id[:6]}"
    await db.devices().insert_one({
        "device_id":    device_id,
        "name":         name,
        "hostname":     payload.hostname,
        "platform":     payload.platform,
        "token_sha256": token_hash(token),
        "revoked":      False,
        "created_at":   datetime.now(timezone.utc),
        "profile":      _clean(payload.profile.model_dump()),
    })
    return EnrollResponse(device_id=device_id, name=name, device_token=token)


def _clean(p: dict) -> dict:
    """Strip control characters; a changed email must be verified again."""
    p = dict(p)
    for k in ("display_name", "credit_name", "email"):
        p[k] = "".join(ch for ch in p.get(k, "") if ch.isprintable()).strip()
    p["email_verified"] = False
    p["updated_at"] = datetime.now(timezone.utc)
    return p


@router.get("/me/profile")
async def get_profile(request: Request):
    """This node's own finder details (device token required), so the
    installer can show current choices before changing them."""
    device = getattr(request.state, "device", None)
    if device is None:
        raise HTTPException(status_code=401, detail="Device token required.")
    p = device.get("profile") or FinderProfile().model_dump()
    return {k: p.get(k) for k in FinderProfile.model_fields} | {"email_verified": p.get("email_verified", False)}


@router.put("/me/profile")
async def update_profile(request: Request, profile: FinderProfile):
    """Change this node's finder details (device token required).

    Partial update: only fields present in the request change, so e.g.
    {"show_publicly": false} opts out without wiping the stored name.
    """
    device = getattr(request.state, "device", None)
    if device is None:
        raise HTTPException(status_code=401, detail="Device token required.")
    old = device.get("profile") or FinderProfile().model_dump()
    new = _clean({**old, **profile.model_dump(exclude_unset=True)})
    if old.get("email") == new["email"]:
        new["email_verified"] = old.get("email_verified", False)
    await db.devices().update_one({"device_id": device["device_id"]}, {"$set": {"profile": new}})
    return {"status": "ok"}


async def finder_names(node_names: set[str]) -> dict[str, str]:
    """node name -> public display name, only for finders who opted in."""
    out: dict[str, str] = {}
    async for d in db.devices().find({"name": {"$in": list(node_names)},
                                      "profile.show_publicly": True},
                                     {"name": 1, "profile.display_name": 1}):
        out[d["name"]] = d["profile"].get("display_name") or d["name"]
    return out

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
