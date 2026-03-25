"""
Pydantic request and response models for all API endpoints.

Keeping schemas in one file makes the API contract easy to read at a glance
and ensures the same model is reused across routes rather than duplicated.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ── Heartbeat ──────────────────────────────────────────────────────────────────

class HeartbeatRequest(BaseModel):
    hostname:          str
    uptime_seconds:    int   = 0
    stars_analyzed:    int   = 0
    candidates_found:  int   = 0
    cpu_percent:       float = 0.0
    ram_percent:       float = 0.0
    current_tic_id:    str | None = None
    current_sector:    int | None = None


# ── Work queue ─────────────────────────────────────────────────────────────────

class QueueItem(BaseModel):
    """One unit of work returned to a worker node."""
    job_id:    str          # MongoDB _id as string
    tic_id:    str
    mission:   str          # "kepler" | "k2" | "tess"
    sector:    int | None   # TESS sector; None for Kepler/K2
    fits_url:  str          # MAST URI to download

class PopulateRequest(BaseModel):
    """
    Admin endpoint payload: add new targets to the work queue.
    Sent by the MAST sync cron job when new observations are available.
    """
    targets: list[dict[str, Any]]   # list of {tic_id, mission, sector, fits_url, priority}


# ── Candidates ────────────────────────────────────────────────────────────────

class CandidateSubmission(BaseModel):
    """Posted by a worker after scoring a transit candidate."""
    worker_hostname:  str
    tic_id:           str
    mission:          str
    sector:           int | None = None
    period_days:      float
    duration_days:    float
    depth_ppm:        float
    bls_power:        float
    exonet_score:     float = Field(ge=0.0, le=1.0)
    secondary_depth:  float = 0.0
    odd_even_diff:    float = 0.0
    # Phase-folded light curve arrays (stored as lists of floats)
    global_view:      list[float] = []
    local_view:       list[float] = []

class CandidateResponse(BaseModel):
    """Candidate as returned to the dashboard / public site."""
    id:               str
    worker_hostname:  str
    tic_id:           str
    mission:          str
    sector:           int | None
    period_days:      float
    duration_days:    float
    depth_ppm:        float
    bls_power:        float
    exonet_score:     float
    reported_at:      datetime
    global_view:      list[float] = []
    local_view:       list[float] = []


# ── Processed log ─────────────────────────────────────────────────────────────

class ProcessedSubmission(BaseModel):
    """Posted by a worker when it finishes processing a star (even if no candidate found)."""
    worker_hostname:    str
    tic_id:             str
    mission:            str
    sector:             int | None = None
    duration_seconds:   float
    candidates_found:   int = 0


# ── Stats ─────────────────────────────────────────────────────────────────────

class NetworkStats(BaseModel):
    active_nodes:     int
    stars_analyzed:   int
    candidates_found: int
    compute_hours:    float
    queue_remaining:  int

class LeaderboardEntry(BaseModel):
    rank:             int
    hostname:         str
    stars_analyzed:   int
    candidates_found: int

class ActivityPoint(BaseModel):
    hour:   str     # ISO 8601 hour string
    count:  int


# ── Star detail ───────────────────────────────────────────────────────────────

class StarDetail(BaseModel):
    """Metadata for a single star, assembled from MAST and our own database."""
    tic_id:           str
    ra:               float | None = None
    dec:              float | None = None
    magnitude:        float | None = None
    effective_temp:   float | None = None   # Kelvin
    stellar_radius:   float | None = None   # Solar radii
    stellar_mass:     float | None = None   # Solar masses
    distance_pc:      float | None = None   # Parsecs
    mast_url:         str                   # Direct link to MAST portal page
    our_candidates:   list[CandidateResponse] = []
    times_processed:  int = 0
