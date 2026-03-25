"""
Async MongoDB client and collection accessors.

Uses Motor (the async MongoDB driver) so FastAPI can handle concurrent
requests from many worker nodes without blocking the event loop.

All collection objects are accessed via module-level helpers so routes
never need to know the database name or import Motor directly.
"""

from __future__ import annotations
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from api.config import get_settings

# Module-level client — initialised once in the FastAPI lifespan handler.
_client: AsyncIOMotorClient | None = None
_db:     AsyncIOMotorDatabase | None = None


async def connect() -> None:
    """Open the MongoDB connection and ensure all indexes exist."""
    global _client, _db
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db     = _client[settings.mongodb_db]
    await _ensure_indexes()


async def disconnect() -> None:
    """Close the MongoDB connection gracefully on shutdown."""
    if _client is not None:
        _client.close()


def db() -> AsyncIOMotorDatabase:
    """Return the active database handle (raises if not connected)."""
    if _db is None:
        raise RuntimeError("Database not initialised — call connect() first.")
    return _db


# ── Collection accessors ───────────────────────────────────────────────────────
# Named functions rather than module-level variables so the connection is
# always established before the collection is referenced.

def work_queue():       return db()["work_queue"]
def candidates():       return db()["candidates"]
def processed_log():    return db()["processed_log"]
def node_telemetry():   return db()["node_telemetry"]
def network_stats():    return db()["network_stats"]
def scheduler_log():    return db()["scheduler_log"]


# ── Index definitions ──────────────────────────────────────────────────────────

async def _ensure_indexes() -> None:
    """
    Create indexes idempotently on startup.

    Idempotent means safe to call every time the container restarts — MongoDB
    skips creation if an identical index already exists.
    """

    # work_queue: fast lookup of available jobs, ordered by priority then age
    await work_queue().create_indexes([
        IndexModel([("status", ASCENDING), ("priority", DESCENDING), ("created_at", ASCENDING)]),
        IndexModel([("tic_id", ASCENDING), ("mission", ASCENDING), ("sector", ASCENDING)],
                   unique=True),
        # Quickly find stalled jobs owned by a specific node
        IndexModel([("assigned_to", ASCENDING), ("assigned_at", ASCENDING)]),
    ])

    # candidates: per-node queries for the dashboard, global score ranking
    await candidates().create_indexes([
        IndexModel([("worker_hostname", ASCENDING), ("reported_at", DESCENDING)]),
        IndexModel([("exonet_score", DESCENDING)]),
        IndexModel([("tic_id", ASCENDING)]),
        IndexModel([("reported_at", DESCENDING)]),
    ])

    # processed_log: dedup check (was this star already done?) + per-node history
    await processed_log().create_indexes([
        IndexModel([("worker_hostname", ASCENDING), ("processed_at", DESCENDING)]),
        IndexModel([("tic_id", ASCENDING), ("mission", ASCENDING), ("sector", ASCENDING)]),
    ])

    # node_telemetry: latest heartbeat per node
    await node_telemetry().create_indexes([
        IndexModel([("hostname", ASCENDING), ("reported_at", DESCENDING)]),
    ])

    # scheduler_log: recent task runs, queried newest-first
    await scheduler_log().create_indexes([
        IndexModel([("logged_at", DESCENDING)]),
        IndexModel([("task", ASCENDING), ("logged_at", DESCENDING)]),
    ])
