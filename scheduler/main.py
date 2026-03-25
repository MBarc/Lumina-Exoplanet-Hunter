"""
Lumina Scheduler — autonomous task runner for the Lumina platform.

Runs three recurring tasks:
  mast_sync     every 24 hours — discovers new MAST observations, populates queue
  queue_health  every 5 minutes — monitors depth, recovers stalled jobs
  (model_refresh is triggered manually via POST /admin/model-refresh on the API)

Each task result is posted to POST /admin/scheduler/log so operators can
see the last run time and any errors from the Mission Control dashboard.

In AWS this container runs as a standalone ECS task alongside the API task.
It needs only the API URL and API key — no direct database access.
"""

from __future__ import annotations
import asyncio
import logging
import sys
from datetime import datetime, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from scheduler.config import get_settings
from scheduler.tasks.mast_sync import run_mast_sync
from scheduler.tasks.queue_health import run_queue_health

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("scheduler")


# ── Log task results to the API ────────────────────────────────────────────────

async def _log_result(result: dict) -> None:
    """Post a task result to the API scheduler log endpoint."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            base_url = settings.api_url,
            headers  = {"X-API-Key": settings.api_key},
            timeout  = 10.0,
        ) as client:
            await client.post("/admin/scheduler/log", json={
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in result.items()
            })
    except Exception as e:
        log.warning("Could not post scheduler log: %s", e)


# ── Wrapped task runners ───────────────────────────────────────────────────────
# Each wrapper logs start/end and posts the result to the API.

async def task_mast_sync():
    log.info("--- MAST sync starting ---")
    result = await run_mast_sync()
    await _log_result(result)

async def task_queue_health():
    log.info("--- Queue health check ---")
    result = await run_queue_health()
    await _log_result(result)


# ── Scheduler setup ────────────────────────────────────────────────────────────

async def wait_for_api(max_attempts: int = 20, delay: float = 5.0) -> None:
    """
    Block until the API is reachable before starting scheduled tasks.

    The scheduler container starts at the same time as the API container.
    Without this wait, the first task run (triggered at startup) would fail
    because the API isn't ready yet.
    """
    settings = get_settings()
    log.info("Waiting for API at %s ...", settings.api_url)
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.api_url}/health")
                if resp.status_code == 200:
                    log.info("API is ready.")
                    return
        except Exception:
            pass
        log.info("  API not ready yet (attempt %d/%d) — retrying in %.0fs",
                 attempt + 1, max_attempts, delay)
        await asyncio.sleep(delay)
    log.error("API did not become ready in time — proceeding anyway.")


async def main() -> None:
    await wait_for_api()

    scheduler = AsyncIOScheduler()

    # Queue health: every 5 minutes
    scheduler.add_job(
        task_queue_health,
        trigger   = IntervalTrigger(minutes=5),
        id        = "queue_health",
        name      = "Queue health check",
        max_instances = 1,       # never run two health checks simultaneously
        misfire_grace_time = 60, # if a run is missed by < 60s, run it anyway
    )

    # MAST sync: every 24 hours
    # Run once immediately on startup so the queue is populated without waiting
    # a full day on first launch.
    scheduler.add_job(
        task_mast_sync,
        trigger   = IntervalTrigger(hours=24),
        id        = "mast_sync",
        name      = "MAST observation sync",
        max_instances    = 1,
        misfire_grace_time = 3600,
        next_run_time = datetime.now(timezone.utc),  # run immediately on start
    )

    scheduler.start()
    log.info("Scheduler running. Press Ctrl+C to stop.")

    # Keep the event loop alive
    try:
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler shutting down.")
        scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
