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

    # ── Queue behaviour ───────────────────────────────────────────────────────
    # How long (seconds) a job stays "assigned" before the server assumes the
    # worker died and re-queues it.
    job_timeout_seconds: int = 1800   # 30 minutes

    # ── MAST ─────────────────────────────────────────────────────────────────
    # Base URL for the MAST portal — used when building star detail links.
    mast_portal_url: str = "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html"

    class Config:
        # Load from a .env file when running locally outside Docker.
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads env vars once at startup)."""
    return Settings()
