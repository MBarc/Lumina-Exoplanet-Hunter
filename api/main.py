"""
Lumina REST API — FastAPI application entry point.

Responsibilities:
  - Mount all route modules
  - Handle MongoDB connect / disconnect via the lifespan context
  - Enforce API key authentication on worker-facing endpoints
  - Expose a public /health endpoint for AWS load balancer health checks
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api import database
from api.config import get_settings
from api.routes import queue, candidates, telemetry, stats, stars


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

_PUBLIC_PREFIXES = ("/health", "/docs", "/openapi", "/stats", "/stars")

@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """
    Enforce API key authentication on non-public routes.

    Public routes (stats, stars, health check, docs) are accessible without
    a key so the GitHub Pages site and anonymous browsers can read them.
    Write routes require the X-API-Key header to prevent anyone from
    polluting the database.
    """
    path = request.url.path
    if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return await call_next(request)

    key = request.headers.get("X-API-Key", "")
    if key != get_settings().api_key:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or missing API key.",
        )
    return await call_next(request)


# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(queue.router)
app.include_router(candidates.router)
app.include_router(telemetry.router)
app.include_router(stats.router)
app.include_router(stars.router)


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
