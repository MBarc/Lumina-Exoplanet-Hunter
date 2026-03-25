"""
Queue health task — monitors queue depth and cleans up stalled jobs.

Runs every 5 minutes. Checks:
  1. Queue depth — warns if fewer than low_water_mark jobs are available
  2. Stalled jobs — finds jobs assigned longer than timeout and re-queues them
     (Belt-and-suspenders: the API already does this on each /queue/next call,
     but the scheduler ensures it happens even during quiet periods when no
     worker is polling.)
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone

import httpx

from scheduler.config import get_settings

log = logging.getLogger(__name__)


async def run_queue_health() -> dict:
    """
    Check queue depth and recover stalled jobs.

    Returns a summary dict for the scheduler_log.
    """
    settings = get_settings()
    started  = datetime.now(timezone.utc)
    errors   = []

    async with httpx.AsyncClient(
        base_url = settings.api_url,
        headers  = {"X-API-Key": settings.api_key},
        timeout  = 30.0,
    ) as client:

        # ── Fetch queue status ─────────────────────────────────────────────
        try:
            resp = await client.get("/queue/status")
            resp.raise_for_status()
            status = resp.json()
        except Exception as e:
            log.error("Could not fetch queue status: %s", e)
            return {"task": "queue_health", "error": str(e)}

        queued   = status.get("queued",   0)
        assigned = status.get("assigned", 0)
        done     = status.get("done",     0)
        total    = queued + assigned + done

        log.info(
            "Queue: %d queued | %d assigned | %d done | %d total",
            queued, assigned, done, total,
        )

        # ── Low-water-mark warning ─────────────────────────────────────────
        if queued < settings.queue_low_water_mark:
            log.warning(
                "Queue depth LOW: only %d jobs available. "
                "Consider running a MAST sync to add more targets.",
                queued,
            )

        # ── Trigger stale job recovery via a no-op queue poll ─────────────
        # The API's /queue/next already re-queues stalled jobs on each call.
        # We trigger it with limit=0 just to run the cleanup logic even when
        # no worker is polling.
        try:
            await client.get("/queue/next", params={"hostname": "__scheduler__", "limit": 0})
        except Exception as e:
            log.warning("Stale job cleanup trigger failed: %s", e)
            errors.append(str(e))

        # ── Fetch active nodes ─────────────────────────────────────────────
        try:
            resp  = await client.get("/nodes")
            resp.raise_for_status()
            nodes = resp.json()
            log.info("Active nodes: %d", len(nodes))
        except Exception as e:
            log.warning("Could not fetch node list: %s", e)
            errors.append(str(e))
            nodes = []

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    return {
        "task":       "queue_health",
        "started_at": started,
        "elapsed_s":  elapsed,
        "queued":     queued,
        "assigned":   assigned,
        "done":       done,
        "active_nodes": len(nodes),
        "errors":     errors,
    }
