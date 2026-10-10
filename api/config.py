"""
API configuration — all values read from environment variables.

Every setting has a sensible default for local Docker development.
In AWS (ECS / Elastic Beanstalk) these are injected as task environment
variables so no secrets ever live in the image or source code.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── MongoDB ───────────────────────────────────────────────────────────────
    # Swap this for an AWS DocumentDB or MongoDB Atlas connection string and
    # nothing else in the codebase needs to change.
    mongodb_uri:  str = "mongodb://mongo:27017"
    mongodb_db:   str = "lumina"

    # ── API security ──────────────────────────────────────────────────────────
    # Worker nodes include this in the X-API-Key header on every request.
    # Set a strong random value in production — e.g. `openssl rand -hex 32`.
    api_key: str = "dev-insecure-key"

    # Volunteer nodes never see api_key. They enroll once (POST /nodes/enroll)
    # and get their own revocable device token. Joining needs enroll_token
    # (invite / pilot) unless open_enrollment is explicitly turned on.
    enroll_token: str = ""
    open_enrollment: bool = False

    # Refuse to start with an empty or default api_key; only for local test stacks.
    allow_insecure_dev: bool = False

    # Most jobs one device may hold at once (stops a node hoarding the queue).
    max_assigned_per_node: int = 50

    # ── Queue behaviour ───────────────────────────────────────────────────────
    # How long (seconds) a job stays "assigned" before the server assumes the
    # worker died and re-queues it.
    job_timeout_seconds: int = 1800   # 30 minutes

    # ── MAST ─────────────────────────────────────────────────────────────────
    # Base URL for the MAST portal — used when building star detail links.
    mast_portal_url: str = "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html"

    # ── Model metadata ────────────────────────────────────────────────────────
    # Human-readable model version string surfaced via GET /stats so the
    # public dashboard can display the current model without a code deploy.
    model_version: str = "ExoNet v2.0"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Allow "model_version" without conflicting with pydantic's "model_" namespace
        "protected_namespaces": (),
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads env vars once at startup)."""
    return Settings()
