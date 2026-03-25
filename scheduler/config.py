"""
Scheduler configuration — all values from environment variables.
"""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API to talk to — in Docker this is the api service name
    api_url: str = "http://api:8000"
    api_key: str = "dev-insecure-key"

    # How many targets to enqueue per MAST sync batch
    mast_sync_batch_size: int = 500

    # Minimum queue depth before the scheduler logs a warning
    queue_low_water_mark: int = 100

    # Jobs per node cap — scheduler won't assign more than this per node
    max_jobs_per_node: int = 20

    # How many seconds before a job is considered stalled
    job_timeout_seconds: int = 1800

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
