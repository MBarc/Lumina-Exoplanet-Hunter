"""
Pydantic request and response models for all API endpoints.

Keeping schemas in one file makes the API contract easy to read at a glance
and ensures the same model is reused across routes rather than duplicated.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# ── Heartbeat ──────────────────────────────────────────────────────────────────

JOB_ID = r"^[0-9a-f]{24}$"
STAR_ID = r"^\d{1,12}$"


class HeartbeatRequest(BaseModel):
    # Untrusted (volunteer machines): finite, bounded values only.
    model_config = ConfigDict(allow_inf_nan=False)
    hostname:          str   = Field(max_length=64)
    uptime_seconds:    int   = Field(0, ge=0, le=10 * 365 * 86400)
    stars_analyzed:    int   = Field(0, ge=0, le=10**9)
    candidates_found:  int   = Field(0, ge=0, le=10**9)
    cpu_percent:       float = Field(0.0, ge=0.0, le=100.0)
    ram_percent:       float = Field(0.0, ge=0.0, le=100.0)
    current_tic_id:    str | None = Field(None, pattern=STAR_ID)
    current_sector:    int | None = Field(None, ge=0, le=10_000)


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
    """Posted by a worker after scoring a transit candidate.

    Untrusted input from volunteer machines: NaN/inf are rejected (one bad
    value would make every public candidate listing fail to serialise), and
    fits_url must point at MAST because it is shown to reviewers.
    """
    model_config = ConfigDict(allow_inf_nan=False)

    job_id:           str = Field(pattern=JOB_ID)    # must be a job this device holds
    worker_hostname:  str = Field(max_length=64)
    # Star ids end up in links and in code reviewers copy and run (the
    # lightkurve snippet), so only digits; missions from a fixed list.
    tic_id:           str = Field(pattern=STAR_ID)
    mission:          Literal["kepler", "k2", "tess"]
    sector:           int | None = None
    period_days:      float
    duration_days:    float
    depth_ppm:        float
    bls_power:        float
    exonet_score:     float = Field(ge=0.0, le=1.0)
    secondary_depth:  float = 0.0          # model units (normalised flux); the *_ppm fields are physical
    odd_even_diff:    float = 0.0
    secondary_depth_ppm: float | None = None
    odd_even_diff_ppm:   float | None = None
    # Ephemeris: with period + t0 + duration anyone can re-fold the public
    # light curve and redraw the transit. t0 is mid-transit in the mission's
    # clock (BKJD / BTJD); the server stores the BJD_TDB equivalent.
    t0:               float | None = None
    # Vetting diagnostics
    n_transits:       float | None = None
    snr:              float | None = None   # depth / robust out-of-transit noise * sqrt(n in transit)
    centroid_shift:   float | None = None
    # Provenance: exactly which data and which model produced this
    fits_url:         str | None = Field(None, max_length=512,
                                         pattern=r"^(mast:|https://mast\.stsci\.edu/)[^\s<>\"']*$")
    model_sha256:     str | None = Field(None, max_length=64, pattern=r"^[0-9a-f]{64}$")
    # Phase-folded light curve arrays (stored as lists of floats)
    global_view:      list[float] = Field([], max_length=4096)
    local_view:       list[float] = Field([], max_length=512)
    odd_view:         list[float] = Field([], max_length=512)
    even_view:        list[float] = Field([], max_length=512)
    secondary_view:   list[float] = Field([], max_length=512)
    # Reviewer curves in ppm on one physical scale (correctly centred; the model's
    # secondary_view input above repeats the primary until the next retrain)
    transit_view_ppm:   list[float] = Field([], max_length=512)
    secondary_view_ppm: list[float] = Field([], max_length=512)

class CandidateResponse(BaseModel):
    """Candidate as returned to the dashboard / public site."""
    id:               str
    worker_hostname:  str
    finder:           str | None = None   # display name, only if the finder opted in
    tic_id:           str
    mission:          str
    sector:           int | None
    period_days:      float
    duration_days:    float
    depth_ppm:        float
    bls_power:        float
    exonet_score:     float
    reported_at:      datetime
    verified:         bool = False
    t0:               float | None = None
    t0_bjd:           float | None = None
    n_transits:       float | None = None
    snr:              float | None = None
    secondary_depth_ppm: float | None = None
    odd_even_diff_ppm:   float | None = None
    secondary_depth:  float | None = None
    odd_even_diff:    float | None = None
    centroid_shift:   float | None = None
    fits_url:         str | None = None
    model_sha256:     str | None = None
    global_view:      list[float] = []
    local_view:       list[float] = []
    odd_view:         list[float] = []
    even_view:        list[float] = []
    secondary_view:   list[float] = []
    transit_view_ppm:   list[float] = []
    secondary_view_ppm: list[float] = []
    # Catalogue cross-match: {status, known_star, name, catalog_period}
    catalog:          dict[str, Any] | None = None


# ── Processed log ─────────────────────────────────────────────────────────────

class ProcessedSubmission(BaseModel):
    """Posted by a worker when it finishes processing a star (even if no candidate found).

    Untrusted: must name a job this device currently holds; values bounded.
    """
    model_config = ConfigDict(allow_inf_nan=False)
    job_id:             str = Field(pattern=JOB_ID)
    worker_hostname:    str = Field(max_length=64)
    tic_id:             str = Field(pattern=STAR_ID)
    mission:            Literal["kepler", "k2", "tess"]
    sector:             int | None = Field(None, ge=0, le=10_000)
    duration_seconds:   float = Field(ge=0.0, le=86_400.0)
    candidates_found:   int = Field(0, ge=0, le=100)


# ── Stats ─────────────────────────────────────────────────────────────────────

class NetworkStats(BaseModel):
    active_nodes:     int
    stars_analyzed:   int
    candidates_found: int
    compute_hours:    float
    queue_depth:      int     # jobs currently queued (waiting to be claimed)
    queue_remaining:  int     # total unfinished jobs (queued + assigned)
    model_version:    str     # e.g. "ExoNet v2.0"

class LeaderboardEntry(BaseModel):
    rank:             int
    hostname:         str
    stars_analyzed:   int
    candidates_found: int

class ActivityPoint(BaseModel):
    hour:   str     # ISO 8601 hour string
    count:  int


# ── Known-object catalogue ────────────────────────────────────────────────────

class CatalogObject(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    star:    str = Field(max_length=40)          # api.catalog.star_key form, e.g. "kepler:9941662"
    name:    str = Field(max_length=80)          # "Kepler-13 b", "KOI-13.01", "TOI-1062.01"
    status:  Literal["known_planet", "known_candidate", "known_false_positive"]
    period:  float | None = Field(None, gt=0)
    t0_bjd:  float | None = None                 # a mid-transit time, BJD_TDB
    duration_days: float | None = Field(None, gt=0)

class CatalogUpload(BaseModel):
    source:  str = Field(max_length=200)
    objects: list[CatalogObject] = Field(max_length=200_000)


# ── Device enrollment ─────────────────────────────────────────────────────────

class FinderProfile(BaseModel):
    """Optional, opt-in details about the volunteer behind a node.

    Nothing here is public unless show_publicly is true, and the email is
    never public. credit_in_submissions is a separate consent: the credit
    name may be put on a community candidate submission (ExoFOP CTOI / paper
    acknowledgement) — only after the email has been verified.
    """
    display_name:           str = Field("", max_length=40)
    credit_name:            str = Field("", max_length=80)
    email:                  str = Field("", max_length=254, pattern=r"^$|^[^@\s]+@[^@\s]+\.[^@\s]+$")
    show_publicly:          bool = False
    credit_in_submissions:  bool = False

class EnrollRequest(BaseModel):
    """Sent once by the installer to join the network."""
    hostname:      str = Field(min_length=1, max_length=64)
    platform:      str = Field("", max_length=32)
    enroll_token:  str = ""
    profile:       FinderProfile = FinderProfile()

class EnrollResponse(BaseModel):
    device_id:     str
    name:          str      # identity the server records results under
    device_token:  str      # shown once; only its hash is stored


# ── Node info ──────────────────────────────────────────────────────────────────

class NodeInfo(BaseModel):
    """Live telemetry snapshot for one worker node."""
    hostname:         str
    uptime_seconds:   int   = 0
    stars_analyzed:   int   = 0
    candidates_found: int   = 0
    cpu_percent:      float = 0.0
    ram_percent:      float = 0.0
    current_tic_id:   str | None = None
    current_sector:   int | None = None
    last_seen:        datetime


# ── Queue status ───────────────────────────────────────────────────────────────

class QueueStatus(BaseModel):
    """Work queue depth breakdown returned by GET /queue/status."""
    queued:   int   # jobs waiting to be claimed
    assigned: int   # jobs currently held by a worker
    done:     int   # jobs completed (processed_log count)
    total:    int   # queued + assigned + done


# ── Scheduler log ──────────────────────────────────────────────────────────────

class SchedulerLogEntry(BaseModel):
    """One task-run result posted by the scheduler container."""
    task:        str                   # e.g. "mast_sync", "queue_health"
    started_at:  datetime | None = None
    elapsed_s:   float | None   = None
    errors:      list[str]      = []
    # mast_sync fields
    discovered:  int | None = None
    inserted:    int | None = None
    skipped:     int | None = None
    # queue_health fields
    queued:      int | None = None
    assigned:    int | None = None
    done:        int | None = None
    active_nodes: int | None = None


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
