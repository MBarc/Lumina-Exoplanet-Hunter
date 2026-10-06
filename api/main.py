"""
Lumina REST API — FastAPI application entry point.

Responsibilities:
  - Mount all route modules
  - Handle MongoDB connect / disconnect via the lifespan context
  - Enforce API key authentication on worker-facing endpoints
  - Expose a public /health endpoint for AWS load balancer health checks
"""

from __future__ import annotations
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import database
from api.config import get_settings
from api.routes import queue, candidates, telemetry, stats, stars, admin, nodes
from api.routes.nodes import token_hash


# ── Lifespan: connect to MongoDB on startup, disconnect on shutdown ────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Lumina API",
    description = "Backend API for the Lumina distributed exoplanet hunting platform.",
    version     = "1.0.0",
    lifespan    = lifespan,
    # Disable docs in production by setting DOCS_URL=none via env if desired
)


# ── CORS ───────────────────────────────────────────────────────────────────────
# Allow the GitHub Pages site and the local contributor dashboard to call the API.
# In production, restrict allow_origins to your actual domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to ["https://mbarc.github.io"] in production
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── API key middleware ─────────────────────────────────────────────────────────
# Worker-facing write endpoints (queue, candidates, telemetry) require an
# X-API-Key header. Public read endpoints (stats, stars) are open.

# Routes open to everyone regardless of method
_PUBLIC_PREFIXES = ("/health", "/docs", "/openapi", "/stats", "/stars", "/nodes",
                    "/queue/status", "/admin/scheduler/log")

# GET-only public routes — read access is open, writes still require a key
_PUBLIC_GET_PREFIXES = ("/candidates",)

# Routes a volunteer node may call with its own X-Device-Token. Everything else
# that isn't public (queue populate, admin) needs the operator's X-API-Key.
_DEVICE_ROUTES = {
    ("GET",  "/queue/next"),
    ("POST", "/queue/release"),
    ("POST", "/candidates"),
    ("POST", "/candidates/processed"),
    ("POST", "/telemetry/heartbeat"),
    ("PUT",  "/nodes/me/profile"),
}


def _unauthorized(detail: str) -> JSONResponse:
    # Returned, not raised: an HTTPException raised inside middleware bypasses
    # FastAPI's handlers and reaches the client as a 500.
    return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": detail})


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """
    Authenticate non-public routes.

    Public routes (stats, stars, health check, docs, node enrollment) are open
    so the GitHub Pages site and anonymous browsers can read them. GET
    /candidates is public too. Node routes accept a per-device token; the
    device is attached to request.state so routes record results under the
    enrolled identity instead of a client-supplied hostname.
    """
    path = request.url.path
    request.state.device = None
    # Device routes first: some live under public prefixes (/nodes/me/...),
    # and must still see who is calling.
    token = request.headers.get("X-Device-Token")
    if token and (request.method, path) in _DEVICE_ROUTES:
        device = await database.devices().find_one({"token_sha256": token_hash(token), "revoked": False})
        if device is None:
            return _unauthorized("Invalid or revoked device token.")
        request.state.device = device
        return await call_next(request)

    if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return await call_next(request)

    if request.method == "GET" and any(path.startswith(p) for p in _PUBLIC_GET_PREFIXES):
        return await call_next(request)

    if not hmac.compare_digest(request.headers.get("X-API-Key", ""), get_settings().api_key):
        return _unauthorized("Invalid or missing API key.")
    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(queue.router)
app.include_router(candidates.router)
app.include_router(telemetry.router)
app.include_router(stats.router)
app.include_router(stars.router)
app.include_router(admin.router)
app.include_router(nodes.router)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health():
    """
    AWS load balancer / ECS health check endpoint.

    Returns 200 if the API is running and MongoDB is reachable.
    Returns 503 if the database connection is down.
    """
    try:
        await database.db().command("ping")
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = f"Database unreachable: {e}",
        )
