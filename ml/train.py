"""
ExoNet training script.

Downloads labels from the NASA Exoplanet Archive for Kepler KOIs, TESS TOIs,
and K2 candidates, fetches the corresponding FITS light curves via
astroquery.mast, preprocesses each light curve with the Lumina pipeline,
trains ExoNet with binary cross-entropy loss, and exports the best checkpoint
to ONNX.

Quick start
-----------
::

    python -m ml.train \\
        --fits-dir  /data/fits_cache \\
        --output-dir /data/exonet_run \\
        --epochs 50 --batch-size 64

The script saves:
  ``<output_dir>/exonet.pt``        — best PyTorch state dict (by val AUC-ROC)
  ``<output_dir>/exonet.onnx``      — ONNX export of the best checkpoint
  ``<output_dir>/threshold.json``   — optimal classification threshold (val F1)

Dependencies (developer machine only)
--------------------------------------
    torch, sklearn, astroquery, astropy, numpy, requests
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import io
import json
import math
import os
import random
import re
import sys
import time
import warnings

# Ensure stdout/stderr use UTF-8 on Windows so torch's emoji progress
# messages (✅ etc.) don't crash with a cp1252 UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from ml.inference import ExoNetInference
from ml.model import GLOBAL_LEN, LOCAL_LEN, SCALAR_FEATURES, ExoNet
from ml.preprocess import preprocess, preprocess_multi

# Regex to extract the 9-digit Kepler kepid from a filename (e.g. kplr002440757_llc.fits)
_KEPLER_KEPID_RE = re.compile(r'kplr(\d{9})')
_TESS_TIC_RE     = re.compile(r'tess\d+-s\d+-(\d+)-|tic(\d+)')
_K2_STAR_RE      = re.compile(r'ktwo(\d+)|k2_lightcurve_(\d+)')


def _star_key(fits_key: str, mission: str) -> tuple[str, int | None]:
    """(star id stored in the cache, numeric catalogue id) for a light-curve path.

    Star ids are mission-qualified so star-grouped splits never merge stars
    across missions: Kepler keeps the bare 9-digit kepid (compatible with
    existing caches), TESS is "tic<id>", K2 "epic<id>". ("", None) if unknown.
    """
    name = Path(fits_key).name.lower()
    rx, prefix = {"kepler": (_KEPLER_KEPID_RE, ""), "tess": (_TESS_TIC_RE, "tic"),
                  "k2": (_K2_STAR_RE, "epic")}.get(mission.lower(), (None, ""))
    m = rx.search(name) if rx else None
    if not m:
        return "", None
    digits = next(g for g in m.groups() if g)
    num = int(digits)
    return (digits if mission.lower() == "kepler" else f"{prefix}{num}"), num

# Regex to extract the EPIC ID from a K2 HLSP filename
# e.g. hlsp_kegs_k2_lightcurve_205962305-c03_kepler_v2_llc.fits → 205962305
_K2_EPIC_RE = re.compile(r'_lightcurve_(\d+)-')

# C5: positive-label threshold used consistently throughout this module.
_POSITIVE_LABEL_THRESHOLD: float = 0.5


def _preprocess_one(args: tuple) -> tuple:
    """
    Module-level worker for parallel preprocessing — must be at module level
    so it is picklable by ProcessPoolExecutor on all platforms.

    Parameters
    ----------
    args : (fits_paths, label, mission_tag)

    Returns
    -------
    (fits_paths, label, mission_tag, candidates, error_str | None)
    """
    fits_paths, label, mission_tag = args
    try:
        candidates = preprocess_multi(fits_paths, n_candidates=3)
        return fits_paths, label, mission_tag, candidates, None
    except Exception as exc:  # noqa: BLE001
        return fits_paths, label, mission_tag, [], f"{type(exc).__name__}: {exc}"


# CF1: Augmentation hyperparameters in one configurable dataclass.
@dataclasses.dataclass(frozen=True)
class AugmentationConfig:
    noise_std: float = 0.002
    noise_raw_multiplier: float = 2.0
    noise_centroid_multiplier: float = 0.5
    max_phase_shift_frac: float = 0.10
    scale_jitter_low: float = 0.98
    scale_jitter_high: float = 1.02
    dilution_prob: float = 0.20
    dilution_max_frac: float = 0.25
    injection_prob: float = 0.30
    mission_dropout_prob: float = 0.20  # prob of zeroing mission one-hot (indices 13-15)
                                        # to simulate inference where mission may be unknown
    cutout_prob: float = 0.20           # prob of zeroing a random contiguous window
    cutout_max_frac: float = 0.10       # max window size as fraction of view length


# ── Focal loss ────────────────────────────────────────────────────────────────

# ── Module-level worker seed (H4) ────────────────────────────────────────────
# Set at the start of train() before DataLoaders are created.  Workers read this
# variable to seed their own RNG streams.
_GLOBAL_SEED: int = 42


class _WorkerInit:
    """
    Picklable worker_init_fn — must be a top-level callable, not a closure,
    because Windows DataLoader workers spawn via pickle and nested functions
    cannot be pickled.
    """
    def __init__(self, fold_offset: int) -> None:
        self.fold_offset = fold_offset

    def __call__(self, worker_id: int) -> None:
        seed = _GLOBAL_SEED + worker_id + self.fold_offset * 1000
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


def _make_worker_init_fn(fold_offset: int) -> _WorkerInit:
    return _WorkerInit(fold_offset)


class FocalLoss(nn.Module):
    """
    Binary sigmoid focal loss with optional positive-class weighting.

    Downweights the loss contribution of well-classified examples so training
    focuses on hard negatives and hard positives — critical when mixing ~50k
    unlabeled negatives (mostly easy) with a few thousand genuine positives.

    Parameters
    ----------
    gamma :
        Focusing parameter.  0.0 → standard BCE; 2.0 = Lin et al. (2017).
    pos_weight :
        Optional tensor of shape (1,) — same semantics as BCEWithLogitsLoss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        # M5: register pos_weight as a buffer so it moves with the module to any device.
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        # Compute pt directly from sigmoid(logits) so that pos_weight scaling
        # on bce does not corrupt the pt ↔ probability correspondence (Issue 3.3).
        prob    = torch.sigmoid(logits)
        pt      = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_w = (1.0 - pt) ** self.gamma
        return (focal_w * bce).mean()

# ── Logging helpers ───────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Print *msg* and flush stdout immediately (important when piped to a file)."""
    print(msg, flush=True)


def _pbar(current: int, total: int, width: int = 30) -> str:
    """Return a text progress bar string: [████░░░░] 42/100 (42.0%)"""
    frac   = current / max(total, 1)
    filled = int(width * frac)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({frac * 100:.1f}%)"


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _eta(elapsed: float, current: int, total: int) -> str:
    if current <= 0 or elapsed <= 0:
        return "?"
    rate = current / elapsed          # items per second
    remaining = (total - current) / rate
    return _fmt_elapsed(remaining)


def _stream_download(url: str, label: str, timeout: tuple = (15, 60)) -> str:
    """
    Download *url* with streaming and log bytes received as they arrive.

    Reports progress every 256 KB so the user can see the download is alive.
    Returns the full response body as a string, or raises requests.RequestException.
    """
    CHUNK = 256 * 1024   # 256 KB per read
    REPORT_EVERY = 256 * 1024  # log a line every 256 KB received

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    chunks: list[bytes] = []
    total_bytes = 0
    last_reported = 0
    t0 = time.time()

    for chunk in response.iter_content(chunk_size=CHUNK):
        if chunk:
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes - last_reported >= REPORT_EVERY:
                elapsed = time.time() - t0
                rate_kbs = total_bytes / max(elapsed, 0.001) / 1024
                _log(f"  {label}  {total_bytes / 1024:.0f} KB received  "
                     f"({rate_kbs:.0f} KB/s)  elapsed={_fmt_elapsed(elapsed)}")
                last_reported = total_bytes

    elapsed = time.time() - t0
    _log(f"  {label}  {total_bytes / 1024:.0f} KB total  "
         f"elapsed={_fmt_elapsed(elapsed)}  — download complete")
    return b"".join(chunks).decode("utf-8", errors="replace")


# ── TAP URLs ──────────────────────────────────────────────────────────────────

_KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,koi_disposition,koi_score+from+cumulative&format=csv"
)
_KOI_PERIOD_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,koi_disposition,koi_score,koi_period+from+cumulative&format=csv"
)
_TOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+tid,tfopwg_disp,pl_orbper+from+toi&format=csv"
)
_K2_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+epic_hostname%2Cdisposition+from+k2pandc&format=csv"
)


# ── Label helpers ─────────────────────────────────────────────────────────────

def _koi_disposition_to_label(
    disposition: str,
    koi_score: float | None = None,
) -> float | None:
    """Map a Kepler KOI disposition string to a binary label.

    C1: when disposition is "CANDIDATE" and a valid (non-NaN) koi_score is
    provided, the soft koi_score is returned instead of the hard 1.0.  This
    unifies the behaviour of the TAP download path (which uses koi_score) and
    the CSV load path (which previously always returned 1.0 for CANDIDATE).
    If koi_score is None or NaN, 0.7 is used as a default soft label.
    """
    d = disposition.strip().upper()
    if d == "CONFIRMED":
        return 1.0
    if d == "CANDIDATE":
        if koi_score is not None and not np.isnan(koi_score):
            return koi_score
        return 0.7
    if d == "FALSE POSITIVE":
        return 0.0
    return None


def _tess_disposition_to_label(disposition: str) -> float | None:
    """Map a TESS TFOPWG disposition string to a soft label."""
    d = disposition.strip().upper()
    if d == "CP":
        return 1.0   # Confirmed Planet
    if d == "PC":
        return 0.8   # Planet Candidate
    if d == "APC":
        return 0.5   # Ambiguous Planetary Candidate — soft label at decision boundary
    if d in {"FP", "FA"}:
        return 0.0   # False Positive / False Alarm
    return None


def _k2_disposition_to_label(disposition: str) -> float | None:
    """Map a K2 candidate disposition string to a binary label."""
    d = disposition.strip().upper()
    if d in {"CONFIRMED", "CANDIDATE"}:
        return 1.0
    if d == "FALSE POSITIVE":
        return 0.0
    return None


# ── CSV download functions ────────────────────────────────────────────────────

def download_koi_table() -> list[tuple[int, float]]:
    """
    Fetch the cumulative KOI table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (kepid, label) tuples where label is 0.0 or 1.0.
    """
    _log("  Contacting NASA Exoplanet Archive (Kepler KOI table) ...")
    t_start = time.time()
    try:
        text = _stream_download(_KOI_TAP_URL, "Kepler KOI")
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download KOI table: {exc}") from exc

    if not text.strip():
        raise RuntimeError("KOI table response was empty.")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "kepid" not in reader.fieldnames or "koi_disposition" not in reader.fieldnames:
        raise RuntimeError(f"Unexpected KOI table columns: {reader.fieldnames}")

    records: list[tuple[int, float]] = []
    for row in reader:
        try:
            kepid = int(row["kepid"].strip())
        except (ValueError, KeyError):
            continue
        # M2: delegate disposition mapping to _koi_disposition_to_label instead
        # of re-implementing the logic inline.
        disposition = row.get("koi_disposition", "")
        try:
            koi_score_val: float | None = float(row.get("koi_score") or "nan")
            if np.isnan(koi_score_val):
                koi_score_val = None
        except (ValueError, TypeError):
            koi_score_val = None
        label = _koi_disposition_to_label(disposition, koi_score=koi_score_val)
        if label is not None:
            records.append((kepid, label))

    n_pos = sum(1 for _, l in records if l == 1.0)
    n_neg = sum(1 for _, l in records if l == 0.0)
    _log(f"  Parsed {len(records)} labelled KOIs in {_fmt_elapsed(time.time() - t_start)}  "
         f"|  {n_pos} positives  {n_neg} negatives")
    return records


def download_koi_periods() -> dict[int, list[float]]:
    """
    Fetch catalogued planet periods per Kepler star from the cumulative KOI table.

    Returns
    -------
    dict mapping kepid -> list of koi_period values (days), restricted to KOIs
    whose disposition maps to a positive label. Used to demote BLS peaks on
    planet hosts that do not match any catalogued period to hard negatives —
    without this, every peak on a planet host inherits label 1.0 even though
    most contain no transit.
    """
    _log("  Contacting NASA Exoplanet Archive (KOI periods) ...")
    t_start = time.time()
    try:
        text = _stream_download(_KOI_PERIOD_TAP_URL, "Kepler KOI periods")
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download KOI period table: {exc}") from exc

    periods: dict[int, list[float]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            kepid = int(row["kepid"].strip())
        except (ValueError, KeyError):
            continue
        try:
            koi_score_val: float | None = float(row.get("koi_score") or "nan")
            if np.isnan(koi_score_val):
                koi_score_val = None
        except (ValueError, TypeError):
            koi_score_val = None
        label = _koi_disposition_to_label(row.get("koi_disposition", ""), koi_score=koi_score_val)
        if label is None or label < _POSITIVE_LABEL_THRESHOLD:
            continue
        try:
            per = float(row.get("koi_period") or "nan")
        except (ValueError, TypeError):
            continue
        if np.isnan(per) or per <= 0:
            continue
        periods.setdefault(kepid, []).append(per)
    _log(f"  Catalogued periods for {len(periods)} planet-host kepids "
         f"in {_fmt_elapsed(time.time() - t_start)}")
    return periods


# Same criterion validated in scripts/validate_kepid_alignment.py: a BLS period
# matches a catalogued period if within 1% of it or its 2:1 harmonics.
_PERIOD_MATCH_TOL: float = 0.01


def _matches_koi_period(period: float, catalogued: list[float]) -> bool:
    for cat in catalogued:
        for mult in (1.0, 2.0, 0.5):
            target = cat * mult
            if abs(period - target) <= _PERIOD_MATCH_TOL * target:
                return True
    return False


def download_toi_table() -> tuple[list[tuple[int, float]], dict[int, list[float]]]:
    """
    Fetch the TESS TOI table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    (records, periods)
      records: one (tic, label) per star — a star with several TOIs keeps its
               highest label (a planet host stays a host even if it also has a
               false-positive TOI); per-signal labels come from period matching.
      periods: tic -> catalogued periods (days) of its planet-like TOIs, used
               to demote BLS peaks on hosts that match no catalogued planet.
    """
    _log("  Contacting NASA Exoplanet Archive (TESS TOI table) ...")
    t_start = time.time()
    try:
        text = _stream_download(_TOI_TAP_URL, "TESS TOI")
    except requests.RequestException as exc:
        # Must abort, like the KOI period table: a TESS cache built without
        # period matching would silently get wrong-peak positive labels.
        raise RuntimeError(f"Failed to download TESS TOI table: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or not {"tid", "tfopwg_disp", "pl_orbper"} <= set(reader.fieldnames):
        raise RuntimeError(f"Unexpected TESS TOI columns: {reader.fieldnames}")

    best: dict[int, float] = {}
    periods: dict[int, list[float]] = {}
    for row in reader:
        try:
            tid = int(row["tid"].strip())
        except (ValueError, KeyError):
            continue
        label = _tess_disposition_to_label(row.get("tfopwg_disp", ""))
        if label is None:
            continue
        best[tid] = max(label, best.get(tid, 0.0))
        try:
            per = float(row.get("pl_orbper") or "nan")
        except ValueError:
            per = float("nan")
        if label >= _POSITIVE_LABEL_THRESHOLD and per > 0:   # NaN compares False
            periods.setdefault(tid, []).append(per)

    records = list(best.items())
    n_pos = sum(1 for _, l in records if l >= _POSITIVE_LABEL_THRESHOLD)
    _log(f"  Parsed {len(records)} labelled TESS stars in {_fmt_elapsed(time.time() - t_start)}  "
         f"|  {n_pos} planet hosts  {len(records) - n_pos} negatives  |  periods for {len(periods)} hosts")
    return records, periods


def download_k2_table() -> list[tuple[int, float]]:
    """
    Fetch the K2 candidates table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (epic_id, label) tuples where label is 0.0 or 1.0.
    """
    _log("  Contacting NASA Exoplanet Archive (K2 candidates table) ...")
    t_start = time.time()
    try:
        text = _stream_download(_K2_TAP_URL, "K2 candidates")
    except requests.RequestException as exc:
        _log(f"  WARNING: Failed to download K2 candidates table: {exc}. Skipping.")
        return []

    if not text.strip():
        _log("  WARNING: K2 candidates table response was empty. Skipping.")
        return []

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "epic_hostname" not in reader.fieldnames or "disposition" not in reader.fieldnames:
        _log(f"  WARNING: Unexpected K2 candidates columns: {reader.fieldnames}. Skipping.")
        return []

    records: list[tuple[int, float]] = []
    seen: set[int] = set()
    for row in reader:
        try:
            epic_name = row["epic_hostname"].strip()
            epic_id = int(epic_name.replace("EPIC", "").strip())
        except (ValueError, KeyError, AttributeError):
            continue
        if epic_id in seen:
            continue  # k2pandc has one row per planet — deduplicate by host star
        label = _k2_disposition_to_label(row.get("disposition", ""))
        if label is not None:
            records.append((epic_id, label))
            seen.add(epic_id)

    n_pos = sum(1 for _, l in records if l == 1.0)
    n_neg = sum(1 for _, l in records if l == 0.0)
    _log(f"  Parsed {len(records)} labelled K2 candidates in {_fmt_elapsed(time.time() - t_start)}  "
         f"|  {n_pos} positives  {n_neg} negatives")
    return records


def download_stellar_params() -> dict[str, np.ndarray]:
    """
    Downloads the full KIC stellar parameter table from the Kepler Input
    Catalog via NASA TAP.

    Returns a dict mapping zero-padded 9-digit kepid string → float32 array
    of shape (5,):
        [log_teff_norm, logg, log_radius_norm, feh, kepmag_norm]

    where:
        log_teff_norm  = log10(teff) - log10(5778)   (solar-normalised)
        log_radius_norm = log10(radius_solar)
        feh is returned as-is (z-score normalisation done inside ScalarBranch)
        kepmag_norm = (kepmag - 12.0) / 4.0  (normalised Kepler magnitude;
                      brighter stars have lower contamination risk)

    Stars not found in the catalog get zeros (treated as unknown).

    The 13-element scalar vector has kepmag_norm at index 12 (previously
    labelled contamination, which is not available in the cumulative table).

    C3: the kepids parameter has been removed — the function always downloads
    the full catalog.  Partial downloads were never implemented and the
    parameter was silently ignored.
    """
    # M3: query kic_stellar (full KIC) instead of cumulative (KOI-only table) so
    # that non-KOI Kepler negatives can also receive stellar parameters.
    # Try kic_stellar first; fall back to cumulative with a warning if it fails.
    _url_kic_stellar = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "SELECT+kepid,kic_teff,kic_logg,kic_radius,kic_feh,kic_kepmag"
        "+FROM+kic_stellar&format=csv"
    )
    _url_cumulative = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "SELECT+kepid,kic_teff,kic_logg,kic_radius,kic_feh,kic_kepmag"
        "+FROM+cumulative&format=csv"
    )
    try:
        _log("  Downloading KIC stellar parameters from NASA TAP (kic_stellar) ...")
        try:
            text = _stream_download(_url_kic_stellar, "KIC stellar params", timeout=(15, 120))
            if not text.strip():
                raise ValueError("empty response from kic_stellar")
        except Exception as _kic_exc:
            _log(f"  WARNING: kic_stellar query failed ({_kic_exc}), falling back to cumulative table.")
            text = _stream_download(_url_cumulative, "KIC stellar params (cumulative fallback)", timeout=(15, 120))
        reader = csv.DictReader(io.StringIO(text))
        params: dict[str, np.ndarray] = {}
        for row in reader:
            try:
                kid = str(int(row.get("kepid") or 0)).zfill(9)
                teff   = float(row.get("kic_teff")   or "nan")
                logg   = float(row.get("kic_logg")   or "nan")
                radius = float(row.get("kic_radius") or "nan")
                feh    = float(row.get("kic_feh")    or "nan")
                kepmag = float(row.get("kic_kepmag") or "nan")
                log_teff_n  = (np.log10(teff) - np.log10(5778.0)) if teff > 0 and np.isfinite(teff) else 0.0
                log_rad_n   = np.log10(radius) if radius > 0 and np.isfinite(radius) else 0.0
                logg_v      = logg   if np.isfinite(logg)   else 0.0
                feh_v       = feh    if np.isfinite(feh)    else 0.0
                kepmag_n    = (kepmag - 12.0) / 4.0 if np.isfinite(kepmag) else 0.0
                params[kid] = np.array([log_teff_n, logg_v, log_rad_n, feh_v, kepmag_n], dtype=np.float32)
            except (ValueError, TypeError):
                continue
        _log(f"  KIC stellar params: {len(params):,} stars downloaded.")
        return params
    except Exception as exc:
        _log(f"  WARNING: stellar params download failed ({exc}) — using zeros.")
        return {}


# ── FITS download helpers ─────────────────────────────────────────────────────

def _build_fits_index(fits_dir: Path) -> list[tuple[str, Path]]:
    """
    Scan *fits_dir* once and return a list of (lowercase_filename, full_path)
    for every .fits file found.  Used to replace per-KOI rglob calls (O(n²))
    with a single scan + in-memory substring search (O(n + k)).
    """
    _log(f"  Scanning FITS cache: {fits_dir} ...")
    t_start = time.time()
    index = [(p.name.lower(), p) for p in fits_dir.rglob("*.fits")]
    _log(f"  Found {len(index)} FITS files in {_fmt_elapsed(time.time() - t_start)}.")
    return index


def _load_fits_index_file(index_file: Path) -> list[tuple[str, Path]]:
    """
    Build the FITS index from a pre-generated listing file (one path per line)
    instead of scanning fits_dir. Scanning 1.1M files over a CIFS mount takes
    tens of minutes per process start; a listing generated NAS-side with
    `find` loads in seconds and the cache contents are static.
    """
    _log(f"  Loading FITS index file: {index_file} ...")
    t_start = time.time()
    index: list[tuple[str, Path]] = []
    with open(index_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and line.lower().endswith(".fits"):
                p = Path(line)
                index.append((p.name.lower(), p))
    _log(f"  Loaded {len(index)} FITS paths in {_fmt_elapsed(time.time() - t_start)}.")
    return index


def _fits_cache_lookup(index: list[tuple[str, Path]], pattern: str) -> Path | None:
    """Return the first cached FITS path whose filename contains *pattern* (case-insensitive)."""
    pat = pattern.lower()
    for name, path in index:
        if pat in name:
            return path
    return None


def _mast_download(
    target_name: str,
    obs_collection: str,
    fits_dir: Path,
    product_subgroup: str,
    dataproduct_type: str = "timeseries",
    query_key: str = "target_name",
) -> Path | None:
    """
    Generic MAST downloader used by all three mission helpers.

    Parameters
    ----------
    target_name :
        MAST target identifier (e.g. ``"kplr002440757"``, ``"TIC 261136679"``).
    obs_collection :
        MAST collection name (``"Kepler"``, ``"TESS"``, ``"K2"``).
    fits_dir :
        Local directory used as both a download destination and a cache.
    product_subgroup :
        Value for the ``productSubGroupDescription`` filter applied to the
        product list **after** the observation query (not in ``query_criteria``).
    dataproduct_type :
        Product type filter for the observation query (default ``"timeseries"``).
    query_key :
        The MAST query field to use for the target name. Kepler/K2 use
        ``"target_name"``; TESS requires ``"objectname"``.
    """
    try:
        from astroquery.mast import Observations  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for FITS download. "
            "Install it with: pip install astroquery"
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_table = Observations.query_criteria(
            **{query_key: target_name},
            obs_collection=obs_collection,
            dataproduct_type=dataproduct_type,
        )

    if obs_table is None or len(obs_table) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # M1: pass the full obs_table (not just obs_table[0]) to capture all
        # available observations (e.g. multiple Kepler quarters per star).
        products = Observations.get_product_list(obs_table)
        lc_prods = Observations.filter_products(
            products,
            productSubGroupDescription=product_subgroup,
            extension="fits",
        )

    if lc_prods is None or len(lc_prods) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Download all available products (up to 20) so multi-quarter stitching
        # works for targets fetched live from MAST (Issue 6.5).  Cap at 20 to
        # avoid downloading hundreds of files per target in rare edge cases.
        manifest = Observations.download_products(
            lc_prods[:20],
            download_dir=str(fits_dir),
            cache=True,
        )

    if manifest is None or len(manifest) == 0:
        return None

    local_path = Path(manifest["Local Path"][0])
    return local_path if local_path.exists() else None


def _download_fits_kepler(kepid: int, fits_dir: Path) -> Path | None:
    """Download the long-cadence Kepler light curve for *kepid*."""
    return _mast_download(
        target_name=f"kplr{kepid:09d}",
        obs_collection="Kepler",
        fits_dir=fits_dir,
        product_subgroup="LLC",
    )


def _download_fits_tess(tid: int, fits_dir: Path) -> Path | None:
    """Download a TESS light curve for TIC *tid*."""
    return _mast_download(
        target_name=f"TIC {tid}",
        obs_collection="TESS",
        fits_dir=fits_dir,
        product_subgroup="LC",
        query_key="objectname",   # TESS requires objectname, not target_name
    )


def _download_fits_k2(epic_id: int, fits_dir: Path) -> Path | None:
    """Download a K2 light curve for EPIC *epic_id*."""
    return _mast_download(
        target_name=f"EPIC {epic_id}",
        obs_collection="K2",
        fits_dir=fits_dir,
        product_subgroup="LLC",
    )


# ── Multi-mission Dataset ─────────────────────────────────────────────────────

class MultiMissionDataset(Dataset):
    """
    PyTorch Dataset combining Kepler KOIs, TESS TOIs, and K2 candidates.

    For each target the preprocessing pipeline is run once and the top BLS
    candidate is used.  Targets that fail preprocessing are silently skipped.
    Each source (Kepler, TESS, K2) is downloaded independently and combined
    into one flat list of (fits_path, label) pairs.

    Parameters
    ----------
    fits_dir :
        Directory used as both a cache for downloaded FITS files and as a
        search location for files already on disk.
    csv_path :
        Path to a local Kepler KOI CSV file (kepid, koi_disposition columns).
        If ``None``, all three mission tables are downloaded from NASA.
    max_samples :
        If set, cap the combined dataset at this many samples (drawn from the
        top of the combined list after label filtering).
    augment :
        If ``True``, apply random augmentation to positive examples at
        __getitem__ time (50% probability per call).
    """

    def __init__(
        self,
        fits_dir: str | Path,
        csv_path: str | Path | None = None,
        max_samples: int | None = None,
        augment: bool = True,
        cache_only: bool = False,
        cache_file: str | Path | None = None,
        max_unlabeled: int = 50_000,
        aug_cfg: AugmentationConfig | None = None,
        preprocess_workers: int = 1,
        checkpoint_every: int = 100,
        fits_index_file: str | Path | None = None,
        missions: tuple[str, ...] = ("kepler", "tess", "k2"),
    ) -> None:
        self.fits_dir   = Path(fits_dir)
        # Skip mkdir when loading from an existing cache file — avoids requiring
        # the FITS directory (e.g. a NAS) to be reachable when preprocessing is
        # already done.
        _cache_exists = cache_file is not None and Path(cache_file).exists()
        if not _cache_exists:
            self.fits_dir.mkdir(parents=True, exist_ok=True)
        self.augment    = augment
        self.cache_only = cache_only
        self._preprocess_workers = max(1, preprocess_workers)
        self._checkpoint_every = max(1, checkpoint_every)
        self._fits_index_file = fits_index_file
        # CF1: store augmentation config; fall back to defaults if not provided.
        self._aug_cfg   = aug_cfg if aug_cfg is not None else AugmentationConfig()
        # H5 / E4: when True, zero out stellar param scalars (indices 8-12) to match
        # inference behaviour where stellar params are always zero.
        # Default is False here; train() sets it from args.zero_stellar_params (default True).
        self.zero_stellar_params: bool = False

        # ── Preprocessing cache: fast-load if available ───────────────────────
        if cache_file is not None:
            cache_file = Path(cache_file)
            if cache_file.exists():
                _log(f"\n  Loading preprocessing cache: {cache_file}")
                t0_load = time.time()
                # Issue 5.4: use mmap_mode='r' to memory-map the arrays instead
                # of loading everything into RAM (~20 GB → ~2 GB working set).
                # __getitem__ indexes the mmap'd arrays directly; no pre-built list.
                data   = np.load(cache_file, allow_pickle=False, mmap_mode='r')
                gvs    = data["global_views"]
                lvs    = data["local_views"]             # (N, 2, 201) new or (N, 201) old cache
                labels = data["labels"]                  # (N,)
                N      = len(labels)
                # Backwards compat: 1-channel (N,2001) → 2-channel (N,2,2001)
                if gvs.ndim == 2:
                    gvs = np.stack([gvs, gvs], axis=1)
                # Backwards compat: fill new view arrays if absent (old cache)
                ovs  = data["odd_views"]       if "odd_views"       in data else np.zeros((N, 201), dtype=np.float32)
                evs  = data["even_views"]      if "even_views"      in data else np.zeros((N, 201), dtype=np.float32)
                svs  = data["secondary_views"] if "secondary_views" in data else np.zeros((N, 201), dtype=np.float32)
                cvs  = data["centroid_views"]  if "centroid_views"  in data else np.zeros((N, 201), dtype=np.float32)
                raw_sc = data["scalars"]
                if raw_sc.shape[1] < SCALAR_FEATURES:
                    pad = np.zeros((N, SCALAR_FEATURES - raw_sc.shape[1]), dtype=np.float32)
                    raw_sc = np.concatenate([raw_sc, pad], axis=1)
                # Backwards compat: missions array (new field; default "unknown")
                if "missions" in data:
                    self._missions: list[str] = list(data["missions"])
                else:
                    self._missions = ["unknown"] * N
                # Backwards compat: per-sample kepids (new field; "" = unknown).
                # Enables star-grouped splits without post-hoc kepid recovery.
                self._kepids: list[str] = (list(data["kepids"]) if "kepids" in data
                                           else [""] * N)
                # Issue 5.4: store raw arrays and index them in __getitem__
                # instead of pre-building a Python list that prevents GC.
                self._gvs    = gvs
                self._lvs    = lvs
                self._ovs    = ovs
                self._evs    = evs
                self._svs    = svs
                self._cvs    = cvs
                self._raw_sc = raw_sc
                self._labels_arr = labels
                self._N      = N
                # _items is kept as None to signal array-index mode
                self._items  = None  # type: ignore[assignment]
                self._labels = [float(l) for l in labels]
                n_pos = int((labels >= _POSITIVE_LABEL_THRESHOLD).sum())
                n_neg = N - n_pos
                _log(f"  Loaded {N} samples from cache (memory-mapped) in "
                     f"{_fmt_elapsed(time.time() - t0_load)}  "
                     f"|  {n_pos} positives  {n_neg} negatives")
                return   # skip all FITS download + preprocessing

        _log("\n" + "=" * 65)
        _log("  Building MultiMission dataset")
        _log("=" * 65)

        missions = tuple(m.lower() for m in missions)
        _log(f"  Missions: {', '.join(missions)}")
        kepler_pairs = kepler_neg_pairs = tess_pairs = k2_pairs = k2_neg_pairs = []
        # Catalogued planet periods per mission, keyed by numeric star id. Used
        # to period-match labels during integration: only the BLS peak that
        # matches a catalogued planet keeps a planet host's positive label.
        catalog_periods: dict[str, dict[int, list[float]]] = {}
        stellar_params: dict[str, np.ndarray] = {}

        # Build the FITS index once — shared by all resolve steps.
        if self._fits_index_file is not None:
            fits_index = _load_fits_index_file(Path(self._fits_index_file))
        else:
            fits_index = _build_fits_index(self.fits_dir)

        if self.cache_only:
            _log("  --cache-only: MAST downloads disabled. Using local cache only.")

        # ── Step 1-2: Kepler ──────────────────────────────────────────────────
        if "kepler" in missions:
            _log("\n[Step 1/6]  Downloading Kepler label table ...")
            if csv_path is not None:
                koi_records = self._load_koi_csv(Path(csv_path))
                _log(f"  Loaded {len(koi_records)} records from local CSV: {csv_path}")
            else:
                koi_records = download_koi_table()

            # A failure here must abort: rebuilding the cache without period
            # matching would silently reintroduce the ~89% wrong-peak positives.
            catalog_periods["kepler"] = download_koi_periods()

            # Download stellar params (for scalar feature expansion)
            if not cache_only:
                stellar_params = download_stellar_params()

            _log(f"\n[Step 2/6]  Resolving Kepler FITS files  ({len(koi_records)} labeled targets) ...")
            kepler_pairs = self._resolve_kepler(koi_records, fits_index, self.cache_only)

            # Add unlabeled Kepler stars as negatives: non-KOI targets have no known
            # transit signal and are genuine negatives for the classifier.
            labeled_kepids = {str(kepid).zfill(9) for kepid, _ in koi_records}
            kepler_neg_pairs = self._resolve_kepler_unlabeled(
                fits_index, labeled_kepids, max_targets=max_unlabeled
            )
            _log(f"  Kepler unlabeled negatives: {len(kepler_neg_pairs)} targets added (label=0.0)")

        # ── Step 3-4: TESS ────────────────────────────────────────────────────
        if "tess" in missions:
            _log(f"\n[Step 3/6]  Downloading TESS label table ...")
            toi_records, catalog_periods["tess"] = download_toi_table()

            _log(f"\n[Step 4/6]  Resolving TESS FITS files  ({len(toi_records)} targets) ...")
            tess_pairs = self._resolve_tess(toi_records, fits_index, self.cache_only)

        # ── Step 5-6: K2 ──────────────────────────────────────────────────────
        if "k2" in missions:
            _log(f"\n[Step 5/6]  Downloading K2 label table ...")
            k2_records = download_k2_table()

            _log(f"\n[Step 6/6]  Resolving K2 FITS files  ({len(k2_records)} targets) ...")
            k2_pairs = self._resolve_k2(k2_records, fits_index, self.cache_only)

            # Add unlabeled K2 stars as negatives: files in the cache whose EPIC ID
            # is not in k2pandc are ordinary stars with no known transit candidate.
            labeled_epics = {str(epic_id) for epic_id, _ in k2_records}
            k2_neg_pairs = self._resolve_k2_unlabeled(
                fits_index, labeled_epics, max_targets=max_unlabeled
            )
            _log(f"  K2 unlabeled negatives: {len(k2_neg_pairs)} targets added (label=0.0)")

        # ── Merge ─────────────────────────────────────────────────────────────
        # Build tagged list so we can track which mission each sample came from.
        all_pairs_tagged: list[tuple[list[Path], float, str]] = (
            [(ps, lb, "kepler")  for ps, lb in kepler_pairs]
            + [(ps, lb, "kepler") for ps, lb in kepler_neg_pairs]
            + [(ps, lb, "tess")   for ps, lb in tess_pairs]
            + [(ps, lb, "k2")     for ps, lb in k2_pairs]
            + [(ps, lb, "k2")     for ps, lb in k2_neg_pairs]
        )
        all_pairs: list[tuple[list[Path], float]] = [(ps, lb) for ps, lb, _ in all_pairs_tagged]
        all_pairs_missions: list[str]             = [ms      for _, _, ms  in all_pairs_tagged]

        _log(f"\n  All missions resolved:")
        _log(f"    Kepler labeled   : {len(kepler_pairs):>6} targets")
        _log(f"    Kepler negatives : {len(kepler_neg_pairs):>6} targets")
        _log(f"    TESS             : {len(tess_pairs):>6} targets")
        _log(f"    K2               : {len(k2_pairs):>6} targets")
        _log(f"    Total            : {len(all_pairs):>6} targets")

        if max_samples is not None and max_samples < len(all_pairs):
            _log(f"  Capping at max_samples={max_samples}")
            # M8: slice both lists together to keep them in sync.
            # Shuffle before truncating so all missions are represented fairly.
            _combined = list(zip(all_pairs, all_pairs_missions))
            random.Random(_GLOBAL_SEED).shuffle(_combined)
            if _combined:
                all_pairs, all_pairs_missions = zip(*_combined)
                all_pairs = list(all_pairs)[:max_samples]
                all_pairs_missions = list(all_pairs_missions)[:max_samples]
            else:
                all_pairs, all_pairs_missions = [], []

        # ── Preprocess each FITS file ─────────────────────────────────────────
        self._items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] | None = []
        self._labels: list[float] = []
        self._missions: list[str] = []   # parallel list tracking which mission each item came from
        self._kepids: list[str] = []     # parallel list: 9-digit kepid, or "" for non-Kepler
        # Array-index mode attributes (used when loaded from memory-mapped cache).
        # Set to None here to indicate list mode; populated in the cache-load path.
        self._gvs = self._lvs = self._ovs = self._evs = None
        self._svs = self._cvs = self._raw_sc = self._labels_arr = None
        self._N = 0

        # Incremental checkpoint: load already-processed results so we can
        # resume an interrupted preprocessing run without redoing all files.
        checkpoint_file = (Path(cache_file).parent / "preprocess_checkpoint.npz"
                           if cache_file is not None else None)
        already_done: set[str] = set()
        if checkpoint_file is not None and checkpoint_file.exists():
            _log(f"  Loading incremental checkpoint: {checkpoint_file}")
            try:
                ckpt = np.load(checkpoint_file, allow_pickle=False)
                gvs_ckpt     = ckpt["global_views"]
                lvs_ckpt     = ckpt["local_views"]
                labels_ckpt  = ckpt["labels"]
                paths_ckpt   = ckpt["paths"].tolist()
                N_ckpt       = len(labels_ckpt)
                # Backwards compat: 1-channel (N,2001) → 2-channel (N,2,2001)
                if gvs_ckpt.ndim == 2:
                    gvs_ckpt = np.stack([gvs_ckpt, gvs_ckpt], axis=1)
                ovs_ckpt  = ckpt["odd_views"]       if "odd_views"       in ckpt else np.zeros((N_ckpt, 201), dtype=np.float32)
                evs_ckpt  = ckpt["even_views"]      if "even_views"      in ckpt else np.zeros((N_ckpt, 201), dtype=np.float32)
                svs_ckpt  = ckpt["secondary_views"] if "secondary_views" in ckpt else np.zeros((N_ckpt, 201), dtype=np.float32)
                cvs_ckpt  = ckpt["centroid_views"]  if "centroid_views"  in ckpt else np.zeros((N_ckpt, 201), dtype=np.float32)
                raw_sc_ckpt = ckpt["scalars"]
                if raw_sc_ckpt.shape[1] < SCALAR_FEATURES:
                    pad = np.zeros((N_ckpt, SCALAR_FEATURES - raw_sc_ckpt.shape[1]), dtype=np.float32)
                    raw_sc_ckpt = np.concatenate([raw_sc_ckpt, pad], axis=1)
                missions_ckpt = (ckpt["missions"].tolist()
                                 if "missions" in ckpt
                                 else ["unknown"] * N_ckpt)
                kepids_ckpt = (ckpt["kepids"].tolist()
                               if "kepids" in ckpt
                               else [""] * N_ckpt)
                # Samples (N_ckpt) and files (paths_ckpt) are different counts —
                # a file can yield several candidates. Iterating paths here used
                # to silently drop every sample beyond len(paths) on resume.
                for i in range(N_ckpt):
                    self._items.append((
                        gvs_ckpt[i], lvs_ckpt[i], ovs_ckpt[i], evs_ckpt[i],
                        svs_ckpt[i], cvs_ckpt[i], raw_sc_ckpt[i], float(labels_ckpt[i])
                    ))
                    self._labels.append(float(labels_ckpt[i]))
                    self._missions.append(missions_ckpt[i])
                    self._kepids.append(kepids_ckpt[i])
                already_done.update(paths_ckpt)
                _log(f"  Resumed {len(already_done)} already-processed files "
                     f"({len(self._items)} samples).")
            except Exception as exc:
                _log(f"  WARNING: checkpoint load failed ({exc}), starting fresh.")
                self._items.clear()
                self._labels.clear()
                self._missions.clear()
                self._kepids.clear()
                already_done.clear()

        total        = len(all_pairs)
        n_valid      = 0
        n_skipped    = 0
        n_demoted    = 0   # positive-host peaks demoted to hard negatives (period mismatch)
        n_pos_no_period = 0   # positive samples whose kepid has no catalogued period
        # Incremental pos/neg counters — updated each iteration instead of
        # scanning self._labels with an O(N) loop (Issue 5.5).
        n_pos_so_far = sum(1 for l in self._labels if l >= _POSITIVE_LABEL_THRESHOLD)
        n_neg_so_far = len(self._labels) - n_pos_so_far
        report_every = max(1, total // 100)
        # Each checkpoint recompresses the entire accumulated dataset, so the
        # cadence trades power-cut data loss against save overhead late in the
        # run. Tune via --checkpoint-every for slow CPUs.
        checkpoint_every = self._checkpoint_every
        t0_pre       = time.time()

        _log(f"\n{'=' * 65}")
        _log(f"  Preprocessing {total} light curves  (BLS + fold + bin) ...")
        if already_done:
            _log(f"  Skipping {len(already_done)} already-processed files.")
        _log(f"{'=' * 65}")

        def _do_checkpoint(cp_file: Path) -> None:
            """Write incremental checkpoint; called periodically during preprocessing.

            Written to a temp file and renamed into place so a power cut
            mid-save leaves the previous checkpoint intact.
            """
            if cp_file is None or not self._items:
                return
            # Must end in .npz or np.savez appends the extension after the fact.
            tmp_file = cp_file.with_suffix(".tmp.npz")
            try:
                np.savez_compressed(
                    tmp_file,
                    global_views    = np.stack([it[0] for it in self._items]),
                    local_views     = np.stack([it[1] for it in self._items]),
                    odd_views       = np.stack([it[2] for it in self._items]),
                    even_views      = np.stack([it[3] for it in self._items]),
                    secondary_views = np.stack([it[4] for it in self._items]),
                    centroid_views  = np.stack([it[5] for it in self._items]),
                    scalars         = np.stack([it[6] for it in self._items]),
                    labels          = np.array(self._labels, dtype=np.float32),
                    paths           = np.array(list(already_done)),
                    missions        = np.array(self._missions, dtype="U10"),
                    kepids          = np.array(self._kepids, dtype="U16"),
                )
                os.replace(tmp_file, cp_file)
            except Exception as exc:
                _log(f"  WARNING: checkpoint save failed ({exc})")

        def _integrate_result(fits_paths, label, mission_tag, candidates, error_str) -> None:
            """Post-process one preprocess result and append to self._items (sequential)."""
            nonlocal n_skipped, n_valid, n_pos_so_far, n_neg_so_far, n_demoted, n_pos_no_period
            fits_key = str(fits_paths[0])
            already_done.add(fits_key)
            if error_str is not None:
                _log(f"  SKIP {fits_paths[0].name}: {error_str}")
                n_skipped += 1
                return
            if not candidates:
                n_skipped += 1
                return
            n_valid += 1
            mt = mission_tag.lower()
            star_id, star_num = _star_key(fits_key, mt)
            kepid_str = star_id if mt == "kepler" else ""   # stellar params are Kepler-only
            for c in candidates:
                sp = stellar_params.get(kepid_str, np.zeros(5, dtype=np.float32)) if kepid_str else np.zeros(5, dtype=np.float32)
                # Period-matched labels: on a planet host, only the BLS peak
                # matching a catalogued planet period (KOI / TOI) keeps the
                # positive label; other peaks are wrong periods on a real host
                # — hard negatives. Hosts missing from the period table keep
                # their label (counted, so a systematic gap shows in the log).
                eff_label = label
                if label >= _POSITIVE_LABEL_THRESHOLD and mt in catalog_periods and star_num is not None:
                    cat = catalog_periods[mt].get(star_num)
                    if not cat:
                        n_pos_no_period += 1
                    elif not _matches_koi_period(float(c.period), cat):
                        eff_label = 0.0
                        n_demoted += 1
                mission_kepler = 1.0 if mt == "kepler" else 0.0
                mission_tess   = 1.0 if mt == "tess"   else 0.0
                mission_k2     = 1.0 if mt == "k2"     else 0.0
                transit_snr    = float(np.log1p(max(c.depth / max(float(c.noise_floor), 1e-4), 0.0)))
                scalar = np.array(
                    [c.period, c.duration, c.depth, c.bls_power,
                     c.secondary_depth, c.odd_even_diff,
                     c.centroid_shift, c.n_transits,
                     sp[0], sp[1], sp[2], sp[3], sp[4],
                     mission_kepler, mission_tess, mission_k2,
                     transit_snr],
                    dtype=np.float32,
                )
                gv_2ch = np.stack([c.global_view.astype(np.float32),
                                   c.raw_global_view.astype(np.float32)], axis=0)
                lv_2ch = np.stack([c.local_view.astype(np.float32),
                                   c.raw_local_view.astype(np.float32)], axis=0)
                ov_2ch = np.stack([c.odd_view.astype(np.float32),
                                   c.raw_odd_view.astype(np.float32)], axis=0)
                ev_2ch = np.stack([c.even_view.astype(np.float32),
                                   c.raw_even_view.astype(np.float32)], axis=0)
                sv_2ch = np.stack([c.secondary_view.astype(np.float32),
                                   c.raw_secondary_view.astype(np.float32)], axis=0)
                self._items.append((gv_2ch, lv_2ch, ov_2ch, ev_2ch, sv_2ch,
                                    c.centroid_curve.astype(np.float32), scalar, eff_label))
                self._labels.append(eff_label)
                self._missions.append(mission_tag)
                self._kepids.append(star_id)
                if eff_label >= _POSITIVE_LABEL_THRESHOLD:
                    n_pos_so_far += 1
                else:
                    n_neg_so_far += 1

        # Build the list of jobs to submit (skip already-done entries).
        pending: list[tuple] = []
        for (fits_paths, label), mission_tag in zip(all_pairs, all_pairs_missions):
            fits_key = str(fits_paths[0])
            if fits_key in already_done:
                n_skipped += 1
            else:
                pending.append((fits_paths, label, mission_tag))

        _log(f"  Workers: {self._preprocess_workers}  |  pending: {len(pending)}  |  "
             f"already cached: {n_skipped}")

        from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415

        done = n_skipped   # count of files processed (for progress bar)
        with ProcessPoolExecutor(max_workers=self._preprocess_workers) as pool:
            futures = {pool.submit(_preprocess_one, job): job for job in pending}
            for future in as_completed(futures):
                fits_paths, label, mission_tag, candidates, error_str = future.result()
                _integrate_result(fits_paths, label, mission_tag, candidates, error_str)
                done += 1
                if done % report_every == 0 or done == total:
                    elapsed = time.time() - t0_pre
                    _log(
                        f"  Preprocess  {_pbar(done, total)}  "
                        f"samples={len(self._items)}  "
                        f"(pos={n_pos_so_far} neg={n_neg_so_far})  "
                        f"skipped={n_skipped}  "
                        f"elapsed={_fmt_elapsed(elapsed)}  "
                        f"eta={_eta(elapsed, done, total)}"
                    )
                if (checkpoint_file is not None and self._items
                        and done % checkpoint_every == 0):
                    _do_checkpoint(checkpoint_file)


        # C5: use threshold for final pos/neg count (includes soft CANDIDATE labels).
        n_pos = sum(1 for l in self._labels if l >= _POSITIVE_LABEL_THRESHOLD)
        n_neg = len(self._labels) - n_pos
        total_time = _fmt_elapsed(time.time() - t0_pre)
        _log(f"\n  Preprocessing complete in {total_time}.")
        _log(f"  Dataset: {len(self._items)} samples  |  {n_pos} positives  {n_neg} negatives")
        _log(f"  Period matching: {n_demoted} planet-host peaks demoted to hard negatives  |  "
             f"{n_pos_no_period} positives kept without a catalogued period")

        # Save preprocessing cache so the next run can skip the BLS step.
        if cache_file is not None and self._items:
            cache_file = Path(cache_file)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            _log(f"\n  Saving preprocessing cache: {cache_file} ...")
            gvs     = np.stack([it[0] for it in self._items])  # (N, 2, 2001)
            lvs     = np.stack([it[1] for it in self._items])  # (N, 2, 201)
            ovs     = np.stack([it[2] for it in self._items])  # (N, 2, 201)
            evs     = np.stack([it[3] for it in self._items])  # (N, 2, 201)
            svs     = np.stack([it[4] for it in self._items])  # (N, 2, 201)
            cvs     = np.stack([it[5] for it in self._items])  # (N, 201)
            scalars = np.stack([it[6] for it in self._items])  # (N, 17)
            labels  = np.array(self._labels, dtype=np.float32) # (N,)
            # Atomic write: temp file + rename, so a power cut mid-save cannot
            # leave a truncated cache. The incremental checkpoint is deleted
            # only after the full cache is safely in place — deleting it first
            # would make the final save a single point of total loss.
            _cache_tmp = cache_file.with_suffix(".tmp.npz")
            np.savez_compressed(_cache_tmp,
                                global_views=gvs, local_views=lvs,
                                odd_views=ovs, even_views=evs,
                                secondary_views=svs,
                                centroid_views=cvs,
                                scalars=scalars, labels=labels,
                                missions=np.array(self._missions, dtype="U10"),
                                kepids=np.array(self._kepids, dtype="U16"))
            os.replace(_cache_tmp, cache_file)
            if checkpoint_file is not None and checkpoint_file.exists():
                try:
                    checkpoint_file.unlink()
                except Exception:
                    pass
            _log(f"  Cache saved ({cache_file.stat().st_size // 1024:,} KB).")

    # ── Mission-specific resolution helpers ───────────────────────────────────

    def _resolve_kepler(
        self,
        records: list[tuple[int, float]],
        fits_index: list[tuple[str, Path]],
        cache_only: bool = False,
    ) -> list[tuple[list[Path], float]]:
        """
        Locate or download FITS for each Kepler KOI.

        Returns all available quarter files per target grouped together so that
        ``preprocess_multi`` can stitch them into a single 4-year light curve,
        dramatically improving sensitivity to long-period planets.
        """
        # Build kepid → [paths] map from the full FITS index in one O(index) pass.
        kepler_cache: dict[str, list[Path]] = {}
        for name, path in fits_index:
            m = _KEPLER_KEPID_RE.search(name)
            if m:
                kepler_cache.setdefault(m.group(1), []).append(path)

        _log(f"  Kepler FITS index: {len(kepler_cache):,} unique kepids in cache.")

        total   = len(records)
        pairs: list[tuple[list[Path], float]] = []
        n_cached = n_downloaded = n_missing = 0
        t_start = time.time()
        report_every = max(1, total // 20)

        for i, (kepid, label) in enumerate(records, start=1):
            padded = str(kepid).zfill(9)
            cached_paths = kepler_cache.get(padded)
            if cached_paths:
                pairs.append((sorted(cached_paths), label))
                n_cached += 1
            elif not cache_only:
                try:
                    path = _download_fits_kepler(kepid, self.fits_dir)
                except Exception:  # noqa: BLE001
                    path = None
                if path is not None:
                    pairs.append(([path], label))
                    n_downloaded += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

            if i % report_every == 0 or i == total:
                elapsed = time.time() - t_start
                _log(f"  Kepler  {_pbar(i, total)}  "
                     f"cached={n_cached}  downloaded={n_downloaded}  missing={n_missing}  "
                     f"elapsed={_fmt_elapsed(elapsed)}  eta={_eta(elapsed, i, total)}")

        n_files = sum(len(ps) for ps, _ in pairs)
        avg = n_files / max(len(pairs), 1)
        _log(f"  Kepler resolved: {len(pairs)}/{total} targets  "
             f"({n_files} total files, avg {avg:.1f} quarters per target, "
             f"{n_cached} cached, {n_downloaded} downloaded, {n_missing} missing)")
        return pairs

    def _resolve_tess(
        self,
        records: list[tuple[int, float]],
        fits_index: list[tuple[str, Path]],
        cache_only: bool = False,
    ) -> list[tuple[list[Path], float]]:
        """Locate or download FITS for each TESS TOI; return ([paths], label) pairs.

        G1: groups ALL available sector files per TIC ID (same pattern as
        _resolve_kepler) and passes the full list to preprocess_multi so that
        multi-sector light curves are stitched together, dramatically improving
        BLS sensitivity to long-period signals.
        """
        # Build tic_id → [paths] map from the full FITS index in one O(index) pass.
        # Two TESS filename formats exist on MAST:
        #   1. "tic<digits>" prefix  (e.g. tic261136679_lc.fits)
        #   2. MAST bulk format: tess{date}-s{sector}-{ticid_16digits}-{cadence}_lc.fits
        #      e.g. tess2018206045859-s0001-0000000008196285-0120-s_lc.fits
        # In format 2 the TIC ID is zero-padded to 16 chars; normalise by stripping
        # leading zeros so the key matches str(tid) from the label table.
        tess_cache: dict[str, list[Path]] = {}
        for name, path in fits_index:
            tic_str: str | None = None
            m_tic = re.search(r'tic(\d+)', name)
            if m_tic:
                tic_str = m_tic.group(1)
            else:
                # MAST bulk format: third dash-separated field is the zero-padded TIC ID
                m_mast = re.search(r'tess\d+-s\d+-(\d+)-', name)
                if m_mast:
                    tic_str = str(int(m_mast.group(1)))  # strip leading zeros
            if tic_str is not None:
                tess_cache.setdefault(tic_str, []).append(path)

        _log(f"  TESS FITS index: {len(tess_cache):,} unique TIC IDs in cache.")

        total   = len(records)
        pairs: list[tuple[list[Path], float]] = []
        n_cached = n_downloaded = n_missing = 0
        t_start = time.time()
        report_every = max(1, total // 20)

        _log(f"\n  Resolving TESS FITS  (0/{total})  ...")
        for i, (tid, label) in enumerate(records, start=1):
            tic_str = str(tid)
            cached_paths = tess_cache.get(tic_str)
            if cached_paths:
                pairs.append((sorted(cached_paths), label))
                n_cached += 1
            elif not cache_only:
                try:
                    path = _download_fits_tess(tid, self.fits_dir)
                except Exception:  # noqa: BLE001
                    path = None
                if path is not None:
                    pairs.append(([path], label))
                    n_downloaded += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

            if i % report_every == 0 or i == total:
                elapsed = time.time() - t_start
                _log(f"  TESS    {_pbar(i, total)}  "
                     f"cached={n_cached}  downloaded={n_downloaded}  missing={n_missing}  "
                     f"elapsed={_fmt_elapsed(elapsed)}  eta={_eta(elapsed, i, total)}")

        n_files = sum(len(ps) for ps, _ in pairs)
        avg = n_files / max(len(pairs), 1)
        _log(f"  TESS resolved: {len(pairs)}/{total} targets found  "
             f"({n_files} total files, avg {avg:.1f} sectors per target, "
             f"{n_cached} cached, {n_downloaded} downloaded, {n_missing} missing)")
        return pairs

    def _resolve_k2(
        self,
        records: list[tuple[int, float]],
        fits_index: list[tuple[str, Path]],
        cache_only: bool = False,
    ) -> list[tuple[list[Path], float]]:
        """Locate or download FITS for each K2 candidate; return ([path], label) pairs."""
        total   = len(records)
        pairs: list[tuple[list[Path], float]] = []
        n_cached = n_downloaded = n_missing = 0
        t_start = time.time()
        report_every = max(1, total // 20)

        _log(f"\n  Resolving K2 FITS  (0/{total})  ...")
        for i, (epic_id, label) in enumerate(records, start=1):
            cached = _fits_cache_lookup(fits_index, f"epic{epic_id}")
            if cached is None:
                cached = _fits_cache_lookup(fits_index, str(epic_id))
            if cached is not None:
                pairs.append(([cached], label))
                n_cached += 1
            elif not cache_only:
                try:
                    path = _download_fits_k2(epic_id, self.fits_dir)
                except Exception:  # noqa: BLE001
                    path = None
                if path is not None:
                    pairs.append(([path], label))
                    n_downloaded += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

            if i % report_every == 0 or i == total:
                elapsed = time.time() - t_start
                _log(f"  K2      {_pbar(i, total)}  "
                     f"cached={n_cached}  downloaded={n_downloaded}  missing={n_missing}  "
                     f"elapsed={_fmt_elapsed(elapsed)}  eta={_eta(elapsed, i, total)}")

        _log(f"  K2 resolved: {len(pairs)}/{total} targets found  "
             f"({n_cached} cached, {n_downloaded} downloaded, {n_missing} missing)")
        return pairs

    def _resolve_k2_unlabeled(
        self,
        fits_index: list[tuple[str, Path]],
        labeled_epics: set[str],
        max_targets: int | None = 50_000,
    ) -> list[tuple[list[Path], float]]:
        """
        Find K2 FITS files whose EPIC ID does not appear in the labeled k2pandc
        table and assign them label=0.0 (genuine negatives).

        K2 downloaded files are mostly ordinary stars — valid negatives for the
        classifier.  Uses _K2_EPIC_RE to extract EPIC IDs from HLSP filenames
        (e.g. hlsp_kegs_k2_lightcurve_205962305-c03_kepler_v2_llc.fits).
        """
        k2_cache: dict[str, list[Path]] = {}
        for name, path in fits_index:
            m = _K2_EPIC_RE.search(name)
            if m:
                epic_id = m.group(1)
                if epic_id not in labeled_epics:
                    k2_cache.setdefault(epic_id, []).append(path)

        pairs: list[tuple[list[Path], float]] = [
            (sorted(paths), 0.0) for paths in k2_cache.values()
        ]

        if max_targets is not None and len(pairs) > max_targets:
            random.Random(_GLOBAL_SEED).shuffle(pairs)
            pairs = pairs[:max_targets]

        return pairs

    def _resolve_kepler_unlabeled(
        self,
        fits_index: list[tuple[str, Path]],
        labeled_kepids: set[str],
        max_targets: int | None = 50_000,
    ) -> list[tuple[list[Path], float]]:
        """
        Find Kepler FITS files belonging to stars not in the labeled KOI table
        and assign them label=0.0 (genuine negatives).

        Most Kepler targets are ordinary stars with no transiting planet; using
        them as negatives gives the model a much richer and more realistic
        negative class than relying solely on KOI false-positives.
        """
        kepler_cache: dict[str, list[Path]] = {}
        for name, path in fits_index:
            m = _KEPLER_KEPID_RE.search(name)
            if m:
                kepid = m.group(1)
                if kepid not in labeled_kepids:
                    kepler_cache.setdefault(kepid, []).append(path)

        pairs: list[tuple[list[Path], float]] = [
            (sorted(paths), 0.0) for paths in kepler_cache.values()
        ]

        if max_targets is not None and len(pairs) > max_targets:
            # Sort before shuffle is redundant (shuffle makes order irrelevant) and
            # produces OS-dependent ordering when sorting Path lists as strings.
            random.Random(_GLOBAL_SEED).shuffle(pairs)
            pairs = pairs[:max_targets]

        return pairs

    # ── Static CSV loader (Kepler only) ───────────────────────────────────────

    @staticmethod
    def _load_koi_csv(csv_path: Path) -> list[tuple[int, float]]:
        """Parse a local CSV with ``kepid`` and ``koi_disposition`` columns.

        C1: also reads koi_score (if present) and passes it to
        _koi_disposition_to_label so CANDIDATE labels match the TAP path.
        C9: warns and raises if no records were parsed (column name mismatch).
        """
        records: list[tuple[int, float]] = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    kepid = int(row["kepid"])
                except (KeyError, ValueError):
                    continue
                # C1: read koi_score for soft CANDIDATE labels
                koi_score: float | None = None
                raw_score = row.get("koi_score", "")
                if raw_score:
                    try:
                        parsed = float(raw_score)
                        koi_score = None if np.isnan(parsed) else parsed
                    except (ValueError, TypeError):
                        koi_score = None
                label = _koi_disposition_to_label(row.get("koi_disposition", ""), koi_score)
                if label is not None:
                    records.append((kepid, label))
        # L13: raise instead of warning so downstream code does not silently proceed
        # with zero Kepler samples and fail later with a generic "dataset too small" error.
        if not records:
            raise RuntimeError(
                f"No valid KOI records parsed from {csv_path!r}. "
                "Check column names (expected 'kepid' and 'koi_disposition')."
            )
        return records

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        # Issue 5.4: support both array-index mode (from memory-mapped cache)
        # and list mode (from in-process preprocessing).
        if self._items is None:
            return self._N
        return len(self._items)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        global_view    : float32 tensor of shape (2, 2001) — [detrended, prenorm_raw]
        local_view     : float32 tensor of shape (2, 201) — [detrended, prenorm_raw]
        odd_view       : float32 tensor of shape (2, 201) — [detrended, prenorm_raw]
        even_view      : float32 tensor of shape (2, 201) — [detrended, prenorm_raw]
        secondary_view : float32 tensor of shape (2, 201) — [detrended, prenorm_raw]
        centroid_view  : float32 tensor of shape (1, 201)
        scalar_tensor  : float32 tensor of shape (17,)
        label          : float32 tensor of shape (1,)

        Note on global_view channel naming (Issue 4.5):
            Channel 0 — detrended + per-view z-scored global view
            Channel 1 — "prenorm_global_view": phase-binned WITHOUT the per-view
                        z-score (but after per-light-curve z-score).  Named "raw"
                        in older code but that was misleading since the light curve
                        was already z-scored before phase-binning.
        """
        # Issue 5.4: index arrays directly when loaded from memory-mapped cache.
        if self._items is None:
            gv     = np.array(self._gvs[idx],    dtype=np.float32)
            lv     = np.array(self._lvs[idx],    dtype=np.float32)
            ov     = np.array(self._ovs[idx],    dtype=np.float32)
            ev     = np.array(self._evs[idx],    dtype=np.float32)
            sv     = np.array(self._svs[idx],    dtype=np.float32)
            cv     = np.array(self._cvs[idx],    dtype=np.float32)
            scalar = np.array(self._raw_sc[idx], dtype=np.float32)
            label  = float(self._labels_arr[idx])
            # Backwards compat: old 1-channel caches have shape (201,); expand to (2, 201)
            # by duplicating channel 0 so old caches load without reprocessing.
            if lv.ndim == 1:
                lv = np.stack([lv, lv], axis=0)
            if ov.ndim == 1:
                ov = np.stack([ov, ov], axis=0)
            if ev.ndim == 1:
                ev = np.stack([ev, ev], axis=0)
            if sv.ndim == 1:
                sv = np.stack([sv, sv], axis=0)
        else:
            gv, lv, ov, ev, sv, cv, scalar, label = self._items[idx]
            if not self.augment:
                gv     = gv.copy()
                lv     = lv.copy()
                ov     = ov.copy()
                ev     = ev.copy()
                sv     = sv.copy()
                cv     = cv.copy()
                scalar = scalar.copy()

        if self.augment:
            aug = self._aug_cfg   # CF1: use config object for all magic numbers
            gv = gv.copy()
            lv = lv.copy()
            ov = ov.copy()
            ev = ev.copy()
            sv = sv.copy()
            cv = cv.copy()
            scalar = scalar.copy()

            # 1. Transit injection on negative samples (injection_prob probability)
            # H1: use _POSITIVE_LABEL_THRESHOLD (not == 0.0) so soft-labeled CANDIDATE
            # KOIs with koi_score 0.3–0.49 are not treated as pure negatives for injection.
            if label < _POSITIVE_LABEL_THRESHOLD and np.random.random() < aug.injection_prob:
                gv, lv, ov, ev, sv, cv, injected_depth, duration_frac = _inject_transit(
                    gv, lv, ov, ev, sv, cv, period_days=float(scalar[0])
                )
                label = 1.0   # injected transit is a positive
                # C2: update scalar features to reflect the injected transit signal.
                # Zero out secondary_depth and odd_even_diff (now planet-like, not EB),
                # and set duration_days consistent with the injected duration fraction.
                # _inject_transit uses duration in phase-fraction units (≈ duration/period).
                # We approximate duration_days as duration_frac * period_days (scalar[0]).
                scalar[2] = float(injected_depth)  # actual injected fractional depth
                scalar[1] = float(duration_frac * scalar[0])  # duration_days ≈ duration_fraction × period_days
                scalar[3] = max(scalar[3], 5.0)                   # bls_power: at least above noise floor
                scalar[4] = 0.0                                    # C2: zero secondary_depth (planet has none)
                scalar[5] = 0.0                                    # C2: zero odd_even_diff (planet is consistent)
                scalar[6] = 0.0                                    # C2: zero centroid_shift (centroid view already zeroed)
                scalar[7] = max(scalar[7], 1.0)                   # n_transits: at least 1

            # 2. Noise-calibrated Gaussian noise
            # Channel 0 (z-scored) and channel 1 (raw/prenorm) get different std
            # to match their different dynamic ranges (same pattern as global_view).
            noise_std = aug.noise_std
            gv[0] += np.random.normal(0, noise_std, gv[0].shape).astype(np.float32)
            gv[1] += np.random.normal(0, noise_std * aug.noise_raw_multiplier, gv[1].shape).astype(np.float32)
            lv[0] += np.random.normal(0, noise_std, lv[0].shape).astype(np.float32)
            lv[1] += np.random.normal(0, noise_std * aug.noise_raw_multiplier, lv[1].shape).astype(np.float32)
            ov[0] += np.random.normal(0, noise_std, ov[0].shape).astype(np.float32)
            ov[1] += np.random.normal(0, noise_std * aug.noise_raw_multiplier, ov[1].shape).astype(np.float32)
            ev[0] += np.random.normal(0, noise_std, ev[0].shape).astype(np.float32)
            ev[1] += np.random.normal(0, noise_std * aug.noise_raw_multiplier, ev[1].shape).astype(np.float32)
            sv[0] += np.random.normal(0, noise_std, sv[0].shape).astype(np.float32)
            sv[1] += np.random.normal(0, noise_std * aug.noise_raw_multiplier, sv[1].shape).astype(np.float32)
            cv += np.random.normal(0, noise_std * aug.noise_centroid_multiplier, cv.shape).astype(np.float32)

            # 3. Random phase shift on local-scale views (±max_phase_shift_frac)
            # Use zero-padding instead of np.roll: the local view is a zoom on
            # the transit centre and is non-cyclic, so wrapping would introduce
            # artifactual signal at the edges.
            # C4: sv (secondary view) is deliberately NOT shifted because it is
            # centred at the secondary eclipse phase (0.5), not the transit phase
            # (0.0).  Shifting sv would misalign it from its own reference frame.
            max_shift = max(1, int(aug.max_phase_shift_frac * lv.shape[0]))
            shift = np.random.randint(-max_shift, max_shift + 1)
            lv = _shift_pad(lv, shift)
            half_shift = int(round(shift / 2))
            ov = _shift_pad(ov, half_shift)
            ev = _shift_pad(ev, half_shift)
            # Fix 9: cv is a full-orbit phase fold (periodic at ±0.5) — use cyclic
            # roll instead of zero-padded shift to preserve the periodic nature of
            # the centroid curve and avoid introducing artificial zeros at the edges.
            cv = np.roll(cv, shift)

            # 4. Random flux scaling
            scale = float(np.random.uniform(aug.scale_jitter_low, aug.scale_jitter_high))
            gv = (gv * scale).astype(np.float32)
            lv = (lv * scale).astype(np.float32)
            ov = (ov * scale).astype(np.float32)
            ev = (ev * scale).astype(np.float32)
            sv = (sv * scale).astype(np.float32)
            # Fix 4: scale cv alongside flux branches so centroid displacement
            # amplitude stays consistent with the scaled flux signal.
            cv = (cv * scale).astype(np.float32)

            # 5. Cutout augmentation: zero a random contiguous window in all local views
            #    to simulate gaps from Kepler/TESS data download failures or momentum dumps.
            if np.random.random() < aug.cutout_prob:
                max_w = max(1, int(aug.cutout_max_frac * lv.shape[-1]))
                w = np.random.randint(1, max_w + 1)
                start = np.random.randint(0, lv.shape[-1] - w + 1)
                lv[..., start:start + w] = 0.0
                ov[..., start:start + w] = 0.0
                ev[..., start:start + w] = 0.0
                sv[..., start:start + w] = 0.0
                cv[..., start:start + w] = 0.0

            # 6. Dilution augmentation: simulate contamination from nearby star
            if np.random.random() < aug.dilution_prob:
                f_contam = float(np.random.uniform(0.0, aug.dilution_max_frac))
                gv = (gv * (1.0 - f_contam)).astype(np.float32)
                lv = (lv * (1.0 - f_contam)).astype(np.float32)
                ov = (ov * (1.0 - f_contam)).astype(np.float32)
                ev = (ev * (1.0 - f_contam)).astype(np.float32)
                sv = (sv * (1.0 - f_contam)).astype(np.float32)
                # M4: dilution also reduces centroid displacement proportionally.
                cv = (cv * (1.0 - f_contam)).astype(np.float32)
                # Fix 2: keep scalar depth consistent with the diluted light curve.
                # When a transit was injected (label==1.0), the depth scalar must
                # be rescaled to match the reduced signal seen by the model.
                if label == 1.0:
                    scalar[2] = scalar[2] * (1.0 - f_contam)
                # NOTE: if this sample was label-flipped by injection and dilution makes
                # the signal undetectable (depth << noise_floor), the label remains 1.0,
                # creating a small fraction of noisy-positive labels (~5% of injected+diluted
                # samples). This is acceptable as a form of hard-negative regularisation.

        # E4: zero out stellar params (indices 8-12) when requested, to produce a
        # model consistent with inference behaviour (stellar params are always zero
        # at inference time since the KIC is not queried during prediction).
        if self.zero_stellar_params:
            scalar = scalar.copy()
            scalar[8:13] = 0.0

        # Mission dropout: randomly zero mission one-hot (indices 13-15) during training
        # so the model learns to classify without mission context when it is unavailable.
        if self.augment and np.random.random() < self._aug_cfg.mission_dropout_prob:
            scalar = scalar.copy()
            scalar[13:16] = 0.0

        return (
            torch.from_numpy(gv),                           # (2, 2001)
            torch.from_numpy(lv),                           # (2, 201)
            torch.from_numpy(ov),                           # (2, 201)
            torch.from_numpy(ev),                           # (2, 201)
            torch.from_numpy(sv),                           # (2, 201)
            torch.from_numpy(cv).unsqueeze(0),              # (1, 201) — centroid stays 1-channel
            torch.from_numpy(scalar),                       # (17,)
            torch.tensor([label], dtype=torch.float32),     # (1,)
        )

    @property
    def labels(self) -> list[float]:
        """All labels in dataset order (useful for stratified splitting)."""
        return self._labels


# ── Augmented subset wrapper (removed) ───────────────────────────────────────
#
# _AugSubset was removed because it was thread-unsafe (Issue 1.1): with
# num_workers > 0, DataLoader workers share the same dataset object.
# Worker A would set augment=True; Worker B (serving the val loader) could read
# augment=True and return augmented val samples.
#
# Fix: create two shallow copies of the base dataset — one with augment=True
# for the training split, one with augment=False for validation — and wrap
# each with a plain torch.utils.data.Subset.  Shallow copy is sufficient
# because the underlying data arrays (_items list / mmap'd arrays) are shared
# by reference and never mutated.  Only the scalar `augment` attribute differs.


# ── Training loop ─────────────────────────────────────────────────────────────

def _apply_curriculum(
    sampler: WeightedRandomSampler,
    base_weights: list[float],
    per_sample_losses: list[float],
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    labels: list[float],
) -> None:
    """
    Apply curriculum learning by blending class-balance weights with per-sample
    difficulty (loss-based) weights.

    Schedule:
    - epoch ≤ warmup_epochs           : pure class-balance weights (easy curriculum)
    - warmup_epochs < epoch ≤ midpoint : linear blend towards hard-sample emphasis
    - epoch > midpoint                 : full difficulty-weighted sampling

    The blend ensures positives and negatives remain roughly balanced while
    focusing compute on the samples the model currently finds hardest.

    Parameters
    ----------
    sampler         : existing WeightedRandomSampler to update in-place
    base_weights    : original class-balance weights from _make_sampler
    per_sample_losses : per-sample loss values from the previous epoch (same order as train_idx)
    epoch / warmup_epochs / total_epochs : for the curriculum schedule
    labels          : training labels (to preserve class balance in blend)
    """
    if not per_sample_losses or epoch <= warmup_epochs:
        return

    mid = warmup_epochs + (total_epochs - warmup_epochs) // 2
    t = min(1.0, (epoch - warmup_epochs) / max(mid - warmup_epochs, 1))

    losses = np.array(per_sample_losses, dtype=np.float64)
    # Normalise per class so hard negatives don't swamp rare positives
    hard_weights = np.zeros_like(losses)
    for cls_val, threshold in [(1.0, _POSITIVE_LABEL_THRESHOLD), (0.0, None)]:
        if threshold is not None:
            mask = np.array([l >= threshold for l in labels])
        else:
            mask = np.array([l < _POSITIVE_LABEL_THRESHOLD for l in labels])
        if mask.sum() > 0:
            cls_losses = losses[mask]
            cls_min, cls_max = cls_losses.min(), cls_losses.max()
            if cls_max > cls_min:
                hard_weights[mask] = (cls_losses - cls_min) / (cls_max - cls_min)
            else:
                hard_weights[mask] = 0.5

    base = np.array(base_weights, dtype=np.float64)
    blended = (1.0 - t) * base + t * (base * (1.0 + 3.0 * hard_weights))
    blended = blended / blended.sum() * len(blended)
    sampler.weights = torch.as_tensor(blended, dtype=torch.double)


def _make_sampler(labels: list[float]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws each class with equal probability.

    This ensures every mini-batch is ~50/50 pos/neg regardless of the true
    class ratio, which is more effective than loss-weighting alone.
    """
    # C5: use _POSITIVE_LABEL_THRESHOLD so soft-labeled CANDIDATE KOIs
    # (koi_score values like 0.83) are sampled as positives, not negatives.
    n_pos = sum(1 for l in labels if l >= _POSITIVE_LABEL_THRESHOLD)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    weights = [w_pos if l >= _POSITIVE_LABEL_THRESHOLD else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _update_hard_negative_sampler(
    sampler: WeightedRandomSampler,
    model: "ExoNet",
    dataset: "Dataset",
    train_idx: list[int],
    labels: list[float],
    device: torch.device,
    boost_factor: float = 4.0,
    batch_size: int = 512,
) -> None:
    """
    Update sampler weights in-place to oversample hard negatives.

    After each update cycle, negative samples with high model-predicted
    probability (false positives the model is most confused about) are
    upweighted proportionally, increasing training pressure on them.

    The model is briefly set to eval mode (no augmentation, deterministic),
    then restored to train mode.  The DataLoader for this eval pass uses
    no shuffle so scores map directly to train_idx positions.

    Parameters
    ----------
    sampler     : the existing WeightedRandomSampler to update in-place
    model       : current ExoNet model
    dataset     : full MultiMissionDataset (shallow copy made internally)
    train_idx   : indices of training samples in dataset
    labels      : training labels (parallel to train_idx)
    boost_factor: negative samples get weight *= (1 + boost_factor * score)
                  so a negative with pred=0.8 gets 4.2× the base weight.
    """
    was_training = model.training
    model.eval()

    # Eval-only copy: no augmentation, preserving order so scores → indices
    eval_ds = copy.copy(dataset)
    eval_ds.augment = False
    eval_loader = DataLoader(
        Subset(eval_ds, train_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,          # avoid worker init overhead for a short eval pass
        pin_memory=False,
    )

    all_scores: list[float] = []
    with torch.no_grad():
        for batch in eval_loader:
            gv_b, lv_b, ov_b, ev_b, sv_b, cv_b, sc_b, _ = batch
            gv_b = gv_b.to(device)
            lv_b = lv_b.to(device)
            ov_b = ov_b.to(device)
            ev_b = ev_b.to(device)
            sv_b = sv_b.to(device)
            cv_b = cv_b.to(device)
            sc_b = sc_b.to(device)
            probs = torch.sigmoid(model(gv_b, lv_b, ov_b, ev_b, sv_b, cv_b, sc_b)).squeeze(1)
            all_scores.extend(probs.cpu().numpy().tolist())

    model.train(was_training)

    n_pos = max(sum(1 for l in labels if l >= _POSITIVE_LABEL_THRESHOLD), 1)
    n_neg = max(len(labels) - n_pos, 1)
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg

    new_weights: list[float] = []
    for label, score in zip(labels, all_scores):
        if label >= _POSITIVE_LABEL_THRESHOLD:
            new_weights.append(w_pos)
        else:
            # Boost negatives in proportion to model confidence they are positives.
            # This focuses training on the hardest false positives.
            new_weights.append(w_neg * (1.0 + boost_factor * float(score)))

    # Update sampler weights in-place — no need to rebuild the DataLoader.
    sampler.weights = torch.as_tensor(new_weights, dtype=torch.double)


def _label_smooth(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """Apply label smoothing to hard (0/1) labels only; leave soft labels unchanged.

    Soft labels (e.g. CANDIDATE koi_score=0.7, TESS PC=0.8) already encode
    calibrated uncertainty — double-smoothing them is inconsistent with that intent.
    """
    hard = (targets == 0.0) | (targets == 1.0)
    smoothed = targets * (1.0 - smoothing) + 0.5 * smoothing
    return torch.where(hard, smoothed, targets)


def _trapezoid_transit(
    phase_bins: np.ndarray,
    depth: float,
    duration: float,
    ingress_frac: float = 0.15,
) -> np.ndarray:
    """
    Compute a trapezoidal transit signal over phase_bins.

    Parameters
    ----------
    phase_bins : 1-D phase array in [-0.5, 0.5]
    depth      : fractional flux depth (positive = dip)
    duration   : transit duration in phase units
    ingress_frac : fraction of half-duration spent in ingress/egress

    Returns a 1-D array of the same shape as phase_bins (values in [0, depth]).
    """
    half_dur = duration / 2.0
    half_ing = half_dur * ingress_frac
    flat_end = half_dur - half_ing
    signal = np.zeros_like(phase_bins, dtype=np.float32)

    abs_ph = np.abs(phase_bins)
    in_flat    = abs_ph <= flat_end
    in_ingress = (abs_ph > flat_end) & (abs_ph <= half_dur)

    signal[in_flat] = depth
    if half_ing > 0:
        frac = (abs_ph[in_ingress] - flat_end) / half_ing
        signal[in_ingress] = depth * (1.0 - frac)

    return signal


_GLOBAL_PHASE_BINS = np.linspace(-0.5, 0.5, 2001, dtype=np.float32)


def _shift_pad(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift arr by shift bins along the last axis, zero-padding instead of wrapping.

    Used for local-scale view augmentation where the view is a zoom on the
    transit centre — wrapping with np.roll would introduce artifactual signal
    at the edges, since the view is non-cyclic.

    Uses ``...`` (ellipsis) indexing so it works for both 1D ``(201,)`` and
    2D ``(2, 201)`` arrays (2-channel local views).
    """
    out = np.zeros_like(arr)
    if shift > 0:
        out[..., shift:] = arr[..., :-shift]
    elif shift < 0:
        out[..., :shift] = arr[..., -shift:]
    else:
        out[...] = arr
    return out


def _inject_transit(
    gv_2ch: np.ndarray,   # (2, 2001)
    lv: np.ndarray,       # (2, 201) — [detrended, prenorm_raw]; broadcasting handles transit injection
    ov: np.ndarray,       # (2, 201)
    ev: np.ndarray,       # (2, 201)
    sv: np.ndarray,       # (2, 201) — secondary view
    cv: np.ndarray,       # (201,) — centroid view
    period_days: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Inject a synthetic trapezoid transit into a negative-class sample.

    Returns updated copies of all six view arrays plus the injected depth_frac
    as the 7th return value.  The primary transit dip is applied to gv/lv/ov/ev.
    The secondary view is left flat (no secondary eclipse for a planet), and the
    centroid curve is zeroed (no centroid motion for an on-target planet), giving
    the model consistent evidence across all branches.

    Issue 1.2 fix: the local-view phase bins span ±half_window, which is
    derived from duration/period (same formula used during preprocessing).
    Injecting over a fixed ±0.4 range when the cached local view only spans
    e.g. ±0.05 would place the transit mostly outside the visible bins.
    """
    depth_frac = float(np.random.uniform(0.001, 0.025))   # 0.1% – 2.5%
    # Cap duration fraction so the injected transit is physically plausible:
    # max ~12 hours (0.5 days) regardless of period.  Without this cap a 50-day
    # period with duration_frac=0.12 gives a 6-day transit — unphysical for a planet.
    max_dur_frac = min(0.12, 0.5 / max(float(period_days), 0.5))
    duration     = float(np.random.uniform(0.02, max(0.02, max_dur_frac)))  # phase-fraction

    # Issue 1.2: derive the local half_window the same way preprocessing does,
    # so the injected trapezoid lands inside the actual local-view bin range.
    # duration here is already in phase-fraction units (same as duration/period).
    half_window = min(2.0 * duration, 0.4)   # matches _fold_and_bin: min(2*dur/period, 0.4)
    local_bins  = np.linspace(-half_window, half_window, 201, dtype=np.float32)

    transit_g = _trapezoid_transit(_GLOBAL_PHASE_BINS, depth_frac, duration)
    transit_l = _trapezoid_transit(local_bins,          depth_frac, duration)

    # Slight depth jitter per odd/even transit (real planets have ~equal depths)
    depth_jitter = float(np.random.uniform(0.97, 1.03))

    new_gv = gv_2ch.copy()
    new_gv[0] -= transit_g   # detrended channel
    new_gv[1] -= transit_g   # prenorm channel (approximate)
    new_lv = lv.copy() - transit_l
    new_ov = ov.copy() - transit_l * depth_jitter
    depth_jitter_even = np.random.uniform(0.97, 1.03)
    new_ev = ev.copy() - transit_l * depth_jitter_even
    # Secondary view: planets have no secondary eclipse — flatten to remove any signal
    new_sv = np.zeros_like(sv, dtype=np.float32)
    # Centroid view: on-target transit has no centroid displacement — zero it out
    new_cv = np.zeros_like(cv, dtype=np.float32)

    return new_gv, new_lv, new_ov, new_ev, new_sv, new_cv, depth_frac, duration


def _mixup_batch(
    gv: torch.Tensor,
    lv: torch.Tensor,
    ov: torch.Tensor,
    ev: torch.Tensor,
    sv: torch.Tensor,
    cv: torch.Tensor,
    scalar: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mixup augmentation: linearly interpolate between random pairs of samples.

    Uses a Beta(alpha, alpha) distribution for the mixing coefficient λ.
    Applied at batch level so it is fast and does not touch the DataLoader.

    Returns the mixed tensors and mixed labels.
    """
    batch_size = labels.size(0)
    lam = float(np.random.beta(alpha, alpha))
    # Always take max(λ, 1-λ) so the dominant sample contributes at least 50%.
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(batch_size, device=labels.device)
    gv_mix     = lam * gv     + (1.0 - lam) * gv[perm]
    lv_mix     = lam * lv     + (1.0 - lam) * lv[perm]
    ov_mix     = lam * ov     + (1.0 - lam) * ov[perm]
    ev_mix     = lam * ev     + (1.0 - lam) * ev[perm]
    sv_mix     = lam * sv     + (1.0 - lam) * sv[perm]
    cv_mix     = lam * cv     + (1.0 - lam) * cv[perm]
    scalar_mix = lam * scalar + (1.0 - lam) * scalar[perm]
    labels_mix = lam * labels + (1.0 - lam) * labels[perm]
    return gv_mix, lv_mix, ov_mix, ev_mix, sv_mix, cv_mix, scalar_mix, labels_mix


def _run_epoch(
    model: ExoNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    label_smoothing: float = 0.05,
    max_grad_norm: float = 1.0,
    mixup_prob: float = 0.30,
    mixup_alpha: float = 0.4,
    aux_loss_weight: float = 0.05,
    grad_accum_steps: int = 1,
    simple_loss: bool = False,
) -> tuple[float, list[float], list[float], list[float]]:
    """
    Run one epoch (train or eval).

    Parameters
    ----------
    optimizer :
        If ``None`` the model is run in eval mode (validation pass).
    label_smoothing :
        Soft-labels strength (applied during training only).
    max_grad_norm :
        Gradient clipping max norm (applied during training only).
    mixup_prob :
        Probability of applying Mixup to a batch during training (default 0.30).
    mixup_alpha :
        Beta distribution concentration parameter for Mixup λ (default 0.4).
    aux_loss_weight :
        Weight for auxiliary depth/duration regression losses (default 0.05).
        Set to 0.0 to disable multi-task learning.

    Returns
    -------
    mean_loss, all_scores, all_labels, per_sample_losses
        ``per_sample_losses`` is populated during training (empty list during eval).
        It is the per-sample focal loss before mixing with auxiliary losses,
        used by the curriculum learning scheduler.
    """
    is_train = optimizer is not None
    model.train(is_train)
    # --simple-loss: classification (focal) loss only. Drops the auxiliary
    # regression, gate entropy, Kendall uncertainty (log_var sits at its clamp,
    # leaving a fixed ~1100x focal multiplier) and the physics penalty (built
    # from detached scores, so it never had a gradient).
    extras = is_train and not simple_loss
    if simple_loss:
        aux_loss_weight = 0.0

    total_loss = 0.0
    all_scores: list[float] = []
    per_sample_losses: list[float] = []
    all_labels: list[float] = []
    accum_steps = max(1, grad_accum_steps) if is_train else 1
    accum_count = 0   # counts batches within the current accumulation window

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for gv, lv, ov, ev, sv, cv, scalar, labels in loader:
            gv     = gv.to(device)    # (B, 2, 2001)
            lv     = lv.to(device)
            ov     = ov.to(device)
            ev     = ev.to(device)
            sv     = sv.to(device)
            cv     = cv.to(device)
            scalar = scalar.to(device)
            labels = labels.to(device)

            # Mixup: interpolate random pairs of samples (train only, batch ≥ 2)
            if is_train and labels.size(0) >= 2 and np.random.random() < mixup_prob:
                gv, lv, ov, ev, sv, cv, scalar, labels = _mixup_batch(
                    gv, lv, ov, ev, sv, cv, scalar, labels, alpha=mixup_alpha
                )

            if is_train and accum_count % accum_steps == 0:
                optimizer.zero_grad()

            # Multi-task: use forward_with_aux during training to get auxiliary
            # regression predictions for depth and duration.
            if is_train and aux_loss_weight > 0.0 and hasattr(model, "forward_with_aux"):
                scores, aux_depth, aux_duration = model.forward_with_aux(
                    gv, lv, ov, ev, sv, cv, scalar
                )
                # Targets: log(1 + depth) at scalar[:,2], log(1 + duration) at scalar[:,1]
                # Using smooth_l1 loss which is robust to the outliers in NASA archive depths.
                tgt_depth    = torch.log1p(torch.clamp(scalar[:, 2:3], min=0.0))
                tgt_duration = torch.log1p(torch.clamp(scalar[:, 1:2], min=0.0))
                aux_loss = (
                    nn.functional.smooth_l1_loss(aux_depth, tgt_depth)
                    + nn.functional.smooth_l1_loss(aux_duration, tgt_duration)
                )
            else:
                scores = model(gv, lv, ov, ev, sv, cv, scalar)  # (B, 1)
                aux_loss = None

            targets = _label_smooth(labels, label_smoothing) if is_train else labels
            # Compute per-sample focal loss (for curriculum learning) before mixing
            # with auxiliary losses — auxiliary targets are noisy at early epochs.
            per_sample_focal = nn.functional.binary_cross_entropy_with_logits(
                scores, targets, reduction="none"
            ).squeeze(1)   # (B,)
            if is_train:
                per_sample_losses.extend(per_sample_focal.detach().cpu().tolist())

            loss = criterion(scores, targets)
            if aux_loss is not None:
                loss = loss + aux_loss_weight * aux_loss

            # Gate entropy regularisation: encourage the branch-gating network to
            # activate multiple branches (high entropy) rather than collapsing onto one.
            # We subtract entropy from the loss so the optimizer maximises it.
            if extras and hasattr(model, "fusion") and hasattr(model.fusion, "last_gate_weights"):
                gw = model.fusion.last_gate_weights  # (B, n_branches), already Softmax
                gate_entropy = -(gw * torch.log(gw + 1e-8)).sum(dim=1).mean()
                loss = loss - 0.01 * gate_entropy

            # Heteroscedastic (aleatoric) uncertainty loss — Kendall & Gal (2017).
            # Formulation: L_unc = 0.5 * exp(-s) * focal_loss + 0.5 * s
            # where s = log_var.  This trains the model to predict per-sample
            # confidence: high uncertainty → large s → smaller gradient from
            # noisy labels, lower uncertainty → model is confident and focused.
            # Weight 1.0 (not additive) — replaces part of the focal loss gradient.
            if extras and hasattr(model, "fusion") and hasattr(model.fusion, "last_log_var"):
                log_var = model.fusion.last_log_var   # (B, 1)
                # Recompute per-sample focal loss without reduction for Kendall weighting
                bce_ps = nn.functional.binary_cross_entropy_with_logits(
                    scores, targets, reduction="none"
                )  # (B, 1)
                prob_ps = torch.sigmoid(scores.detach())
                pt_ps   = prob_ps * targets + (1.0 - prob_ps) * (1.0 - targets)
                focal_ps = ((1.0 - pt_ps) ** 2) * bce_ps   # (B, 1)
                # Kendall NLL: upweight easy (low log_var) samples, downweight hard/noisy ones
                log_var_clamped = log_var.clamp(min=-10.0, max=10.0)
                kendall_nll = (0.5 * torch.exp(-log_var_clamped) * focal_ps + 0.5 * log_var_clamped).mean()
                loss = loss + 0.10 * kendall_nll

            # Physics-consistency penalty: softly penalise the model for scoring
            # physically-implausible candidates highly.  This integrates vetting
            # physics into the training signal rather than only as post-hoc filtering.
            # Penalty terms (each in [0,1]): duration/period > 10%, depth > 50%,
            # secondary depth > primary depth.  Penalty = mean(prob * violation).
            if extras:
                prob_det = torch.sigmoid(scores.detach()).squeeze(1)  # (B,)
                # Veto 1: duty cycle (duration/period > 10%)
                period_s   = scalar[:, 0].clamp(min=1e-4)
                duration_s = scalar[:, 1].clamp(min=0.0)
                duty       = (duration_s / period_s - 0.10).clamp(min=0.0)
                # Veto 2: depth ceiling (depth > 50%)
                depth_abs  = scalar[:, 2].abs()          # guard against inverted-BLS negatives
                depth_viol = (depth_abs - 0.50).clamp(min=0.0)
                # Veto 3: secondary deeper than primary (use abs depth so inverted BLS
                # doesn't make any secondary depth appear to violate the constraint)
                sec_viol   = (scalar[:, 4] - depth_abs).clamp(min=0.0)
                phys_penalty = (prob_det * (duty + depth_viol + sec_viol)).mean()
                loss = loss + 0.05 * phys_penalty

            if is_train:
                (loss / accum_steps).backward()
                accum_count += 1
                if accum_count % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(scores).detach().cpu().numpy()[:, 0]
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy()[:, 0].tolist())

    mean_loss = total_loss / max(len(all_labels), 1)
    return mean_loss, all_scores, all_labels, per_sample_losses


# ── Threshold sweep ───────────────────────────────────────────────────────────

def _tune_threshold(
    val_scores: list[float],
    val_labels: list[float],
    output_dir: Path,
    min_precision: float = 0.90,
    save_path: Path | None = None,
) -> float:
    """
    Sweep thresholds from 0.1 to 0.99 in steps of 0.01 and record:
      - best_f1_threshold   : maximises F1
      - precision90_threshold : lowest threshold that keeps precision >= min_precision

    H2: when save_path is provided, write to that path instead of
    output_dir / "threshold.json".  The OOF call uses the default path;
    the test-set call passes save_path=output_dir/"threshold_test_reference.json"
    so the biased test-set threshold never overwrites the unbiased OOF threshold.
    Returns the max-F1 threshold.
    """
    # CF5: precision_score and recall_score are now top-level imports.
    scores_arr = np.array(val_scores)
    # Binarise labels before passing to binary classification metrics.
    # Soft labels (CANDIDATE koi_score, TESS PC=0.8, APC=0.5) must not be passed
    # to sklearn's f1_score/precision_score, which treats non-integer values as
    # distinct classes and produces undefined results.
    labels_arr = (np.array(val_labels) >= _POSITIVE_LABEL_THRESHOLD).astype(int)

    best_thresh  = 0.5
    best_f1      = -1.0
    prec90_thresh: float | None = None

    thresholds = np.arange(0.10, 1.00, 0.01)
    for thresh in thresholds:
        preds = (scores_arr >= thresh).astype(int)
        f1    = f1_score(labels_arr, preds, zero_division=0)
        prec  = precision_score(labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = float(thresh)
        if prec >= min_precision and prec90_thresh is None:
            prec90_thresh = float(thresh)
            # Do NOT break: F1 sweep must continue to completion for best_thresh.

    _log(f"Optimal F1 threshold  : {best_thresh:.2f}  (val F1: {best_f1:.4f})")
    if prec90_thresh is not None:
        preds90 = (scores_arr >= prec90_thresh).astype(int)
        recall90 = recall_score(labels_arr, preds90, zero_division=0)
        _log(f"Precision≥{min_precision:.0%} threshold: {prec90_thresh:.2f}  "
             f"(recall at that threshold: {recall90:.4f})")
    else:
        _log(f"Precision≥{min_precision:.0%} threshold: not achievable on this val set")

    # E1: compute PR-AUC for the scores being thresholded.
    try:
        val_pr_auc = float(average_precision_score(labels_arr, scores_arr))
    except ValueError:
        val_pr_auc = float("nan")

    # H2: use save_path when provided (for test-set reference only runs).
    threshold_path = save_path if save_path is not None else (output_dir / "threshold.json")
    payload: dict = {
        "threshold":           round(best_thresh, 2),
        "threshold_max_f1":    round(best_thresh, 2),
        "val_pr_auc":          round(val_pr_auc, 4),
    }
    if prec90_thresh is not None:
        payload[f"threshold_precision{int(min_precision*100)}"] = round(prec90_thresh, 2)
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _log(f"Threshold saved: {threshold_path}  (val PR-AUC={val_pr_auc:.4f})")

    return best_thresh


# ── Full training pipeline ────────────────────────────────────────────────────

def _atomic_torch_save(obj, path: Path) -> None:
    """torch.save via a temp file + os.replace so an interrupted write (power
    loss, kill) can never leave a truncated/corrupt file at *path*."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json_save(obj, path: Path) -> None:
    """JSON dump via a temp file + os.replace (see _atomic_torch_save)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _fold_complete_path(output_dir: Path, fold_num: int) -> Path:
    """Marker recording a fully finished fold (incl. SWA). Holds the fold's
    val AUC and out-of-fold scores/labels so relaunches can skip the fold and
    still feed OOF threshold tuning. Delete it to force a retrain."""
    return output_dir / f"fold_complete_{fold_num}.json"


def _train_fold(
    dataset: MultiMissionDataset,
    train_idx: list[int],
    val_idx: list[int],
    args: argparse.Namespace,
    device: torch.device,
    pt_save_path: Path,
    fold_num: int = 0,
) -> tuple[float, list[float], list[float]]:
    """
    Train one fold and return (best_val_auc, best_val_scores, best_val_labels).

    The best checkpoint is saved to *pt_save_path*.  When ``args.save_all_folds``
    is True, a per-fold copy is also saved alongside it as
    ``exonet_fold_{fold_num}.pt``.

    E1: PR-AUC is computed and logged each epoch alongside ROC-AUC.
    """
    train_labels = [dataset.labels[i] for i in train_idx]
    sampler = _make_sampler(train_labels)
    # Snapshot the original class-balance weights so curriculum learning can
    # blend them with per-sample difficulty without drifting off class balance.
    _base_sampler_weights = list(sampler.weights.tolist())

    # Issue 1.1: create two shallow copies of the base dataset — one with
    # augment=True for training, one with augment=False for validation.
    # Shallow copy is safe: the data arrays are shared by reference and are
    # never mutated.  Only the scalar `augment` attribute differs between the
    # two copies, so DataLoader workers cannot interfere with each other.
    train_dataset = copy.copy(dataset)
    train_dataset.augment = True
    val_dataset = copy.copy(dataset)
    val_dataset.augment = False

    # H4: pass worker_init_fn to all DataLoaders so --seed seeds augmentation in workers.
    # T-3: use fold-specific worker init so workers in different folds have distinct seeds.
    _fold_worker_init = _make_worker_init_fn(fold_num)
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=args.batch_size,
        sampler=sampler,          # balanced batches via WeightedRandomSampler
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=_fold_worker_init,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=_fold_worker_init,
    )

    # Issue 1.7: SWA runs on an unaugmented copy so that label-distribution
    # noise from transit injection does not corrupt the weight average.
    swa_dataset = copy.copy(dataset)
    swa_dataset.augment = False
    swa_loader = DataLoader(
        Subset(swa_dataset, train_idx),
        batch_size=args.batch_size,
        sampler=_make_sampler(train_labels),
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=_fold_worker_init,
    )

    # WeightedRandomSampler already produces ~50/50 pos/neg batches, so the
    # effective class ratio seen by the loss is already balanced.  Setting
    # pos_weight=1.0 avoids double-correcting for imbalance (sampler + pos_weight
    # together would over-upweight positives, biasing toward high recall/low
    # precision and causing scores to cluster near 1.0).
    # CS2: n_pos / n_neg used for logging only — pos_weight=None because
    # WeightedRandomSampler already balances the class distribution.
    n_pos = max(sum(1 for l in train_labels if l >= _POSITIVE_LABEL_THRESHOLD), 1)
    n_neg = max(len(train_labels) - n_pos, 1)
    use_se       = getattr(args, "use_se", True)
    dropout      = getattr(args, "dropout", 0.4)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    lr_schedule  = getattr(args, "lr_schedule", "plateau")
    cosine_t0    = getattr(args, "cosine_t0", 10)

    # CF3: ReduceLROnPlateau constants extracted so they are identical in both
    # the initial creation path and the checkpoint-resume reconstruction path.
    _RLROP_FACTOR   = 0.5
    _RLROP_PATIENCE = 5
    _RLROP_MIN_LR   = 1e-6

    model = ExoNet(use_se=use_se, dropout=dropout).to(device)

    # Load MAE-pretrained GlobalBranch weights if provided
    pretrained_global = getattr(args, "pretrained_global", None)
    if pretrained_global is not None:
        pretrained_global = Path(pretrained_global)
        if pretrained_global.exists():
            gb_state = torch.load(str(pretrained_global), map_location=device, weights_only=True)
            model.global_branch.load_state_dict(gb_state, strict=True)
            _log(f"  Loaded pretrained GlobalBranch from {pretrained_global.name}")
        else:
            _log(f"  WARNING: --pretrained-global path not found: {pretrained_global}")

    pretrained_local = getattr(args, "pretrained_local", None)
    if pretrained_local is not None:
        pretrained_local = Path(pretrained_local)
        if pretrained_local.exists():
            lb_state = torch.load(str(pretrained_local), map_location=device, weights_only=True)
            # Load into all four local-type branches (local, odd, even, secondary)
            for branch_name in ("local_branch", "odd_branch", "even_branch", "secondary_branch"):
                branch = getattr(model, branch_name)
                branch.load_state_dict(lb_state, strict=True)
            _log(f"  Loaded pretrained LocalBranch into local/odd/even/secondary from {pretrained_local.name}")
        else:
            _log(f"  WARNING: --pretrained-local path not found: {pretrained_local}")

    criterion = FocalLoss(
        gamma=2.0,
        pos_weight=None,   # sampler already balances classes; no double-correction
    )
    _log(f"  pos_weight=1.0 (sampler-balanced)  (n_neg={n_neg}, n_pos={n_pos})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    # Issue 3.1: Use LambdaLR for linear LR warmup so that ReduceLROnPlateau's
    # patience counter only starts after warmup completes.  The main scheduler
    # is created later (once warmup ends) to avoid premature patience ticking.
    # Issue 3.2: CosineAnnealingWarmRestarts is called with no arguments (just
    # scheduler.step()) to avoid mixing explicit step indices with the internal
    # step counter, which PyTorch warns against.
    warmup_epochs = getattr(args, "warmup_epochs", 5)
    # Warmup scheduler: ramp LR from 0 → args.lr linearly over warmup_epochs.
    # CF6: divide by warmup_epochs + 1 so the ramp starts near 0
    # (ep=0 → 1/(warmup_epochs+1) ≈ 0.17 for 5 warmup epochs) and reaches
    # 1.0 on the last warmup epoch (ep=warmup_epochs-1 → warmup_epochs/(warmup_epochs+1)).
    # The original lambda (ep+1)/warmup_epochs started at 20% on epoch 0.
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / (max(warmup_epochs, 1) + 1),
    )
    # Main scheduler is None until warmup ends (initialised on the first post-
    # warmup epoch so its patience counter starts fresh — Issue 3.1).
    scheduler = None

    best_val_auc = -1.0
    best_val_scores: list[float] = []
    best_val_labels: list[float] = []
    epochs_no_improve = 0
    start_epoch = 1
    main_loop_done = False

    # ── Epoch-level resume ────────────────────────────────────────────────────
    epoch_ckpt_path = pt_save_path.parent / f"epoch_checkpoint_fold_{fold_num}.pt"
    swa_ckpt_path   = pt_save_path.parent / f"swa_checkpoint_fold_{fold_num}.pt"

    def _finalize_fold(auc: float, scores: list[float], labels: list[float]) -> None:
        """Record fold completion, then clean up resume state.  The marker is
        written before the checkpoints are deleted so a crash between the two
        steps errs on the side of 'fold done' rather than retraining."""
        _atomic_json_save({
            "fold_auc":        float(auc),
            "best_val_scores": [float(s) for s in scores],
            "best_val_labels": [float(l) for l in labels],
        }, _fold_complete_path(pt_save_path.parent, fold_num))
        for p in (epoch_ckpt_path, swa_ckpt_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
    if epoch_ckpt_path.exists():
        try:
            ckpt = torch.load(epoch_ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            # Restore whichever scheduler was active when the checkpoint was saved.
            if "main_scheduler_state" in ckpt and ckpt["main_scheduler_state"] is not None:
                # Main scheduler was already initialised at checkpoint time.
                if lr_schedule == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=cosine_t0, T_mult=2, eta_min=_RLROP_MIN_LR
                    )
                else:
                    # CF3: use extracted constants for consistency with initial creation.
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max",
                        factor=_RLROP_FACTOR,
                        patience=_RLROP_PATIENCE,
                        min_lr=_RLROP_MIN_LR,
                    )
                # Fix 6: skip loading scheduler state if the schedule type changed,
                # to avoid crashing or silently corrupting the scheduler state.
                ckpt_lr_schedule = ckpt.get("lr_schedule", lr_schedule)
                if ckpt_lr_schedule != lr_schedule:
                    _log(f"  WARNING: checkpoint used lr_schedule={ckpt_lr_schedule!r}, "
                         f"but --lr-schedule={lr_schedule!r}. Ignoring saved scheduler state.")
                else:
                    scheduler.load_state_dict(ckpt["main_scheduler_state"])
            if "warmup_scheduler_state" in ckpt and ckpt["warmup_scheduler_state"] is not None:
                warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state"])
            start_epoch       = ckpt["epoch"] + 1
            best_val_auc      = ckpt["best_val_auc"]
            best_val_scores   = ckpt["best_val_scores"]
            best_val_labels   = ckpt["best_val_labels"]
            epochs_no_improve = ckpt["epochs_no_improve"]
            # M11: restore warmup_epochs from checkpoint so scheduler states are
            # consistent when resuming with different --warmup-epochs.
            warmup_epochs = ckpt.get("warmup_epochs", warmup_epochs)
            main_loop_done = ckpt.get("main_done", False)
            if main_loop_done:
                # Main loop finished on a previous launch; the run died during
                # SWA. Skip straight to the SWA phase.
                start_epoch = args.epochs + 1
                _log(f"  Fold {fold_num}: main training loop already complete "
                     f"(best val AUC {best_val_auc:.4f}) — resuming at SWA phase.")
            else:
                _log(f"  Resuming fold {fold_num} from epoch {start_epoch}  "
                     f"(best val AUC so far: {best_val_auc:.4f})")
        except Exception as exc:
            _log(f"  WARNING: epoch checkpoint load failed ({exc}), starting fold from scratch.")
            start_epoch = 1
            main_loop_done = False

    _log(f"  n_pos={n_pos}  n_neg={n_neg}  (pos_weight=1.0, sampler-balanced)")
    _log(f"\n  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>9}  {'LR':>10}  {'Elapsed':>9}  {'ETA':>9}")
    _log("  " + "-" * 72)
    fold_t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, _, _, train_sample_losses = _run_epoch(model, train_loader, criterion, optimizer, device,
                                                            grad_accum_steps=args.grad_accum_steps,
                                                            max_grad_norm=args.max_grad_norm,
                                                            simple_loss=getattr(args, "simple_loss", False))
        val_loss, val_scores, val_labels_ep, _ = _run_epoch(model, val_loader, criterion, None, device)

        # Issue 3.1 / Fix 3: step the warmup scheduler AFTER the optimizer step
        # (i.e. after _run_epoch) so epoch 1 trains with the epoch-1 LR, not the
        # epoch-2 LR.  PyTorch documented usage is .step() after optimizer.step().
        if epoch <= warmup_epochs:
            warmup_scheduler.step()

        # Binarise soft labels (e.g. TESS APC=0.5, PC=0.8) before passing to
        # roc_auc_score — sklearn raises ValueError for >2 unique y_true values.
        val_labels_binary = [float(l >= _POSITIVE_LABEL_THRESHOLD) for l in val_labels_ep]
        try:
            val_auc = roc_auc_score(val_labels_binary, val_scores)
        except ValueError:
            val_auc = float("nan")

        # E1: compute PR-AUC alongside ROC-AUC at each epoch.
        try:
            val_pr_auc = average_precision_score(val_labels_binary, val_scores)
        except ValueError:
            val_pr_auc = float("nan")

        if epoch > warmup_epochs:
            # Initialise the main scheduler fresh on the first post-warmup epoch.
            if scheduler is None:
                if lr_schedule == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=cosine_t0, T_mult=2, eta_min=_RLROP_MIN_LR
                    )
                else:
                    # CF3: use extracted constants.
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="max",
                        factor=_RLROP_FACTOR,
                        patience=_RLROP_PATIENCE,
                        min_lr=_RLROP_MIN_LR,
                    )
            # Issue 3.2: call scheduler.step() with no arguments for
            # CosineAnnealingWarmRestarts to avoid mixing explicit step indices
            # with the internal step counter (PyTorch docs warn against this).
            if lr_schedule == "cosine":
                scheduler.step()
            else:
                # NaN-safe: skip ReduceLROnPlateau step rather than feeding 0.0,
                # which would trigger a spurious LR reduction on NaN epochs.
                if not math.isnan(val_auc):
                    scheduler.step(val_auc)

        # Hard-negative mining: boost sampling weight of false-negative training examples.
        hard_neg_freq = getattr(args, "hard_neg_update_freq", 5)
        if epoch > warmup_epochs and hard_neg_freq > 0 and epoch % hard_neg_freq == 0:
            _update_hard_negative_sampler(
                sampler, model, dataset, train_idx, train_labels, device
            )

        # Curriculum learning: blend class-balance weights with per-sample difficulty.
        # Easy → hard schedule prevents overfitting to hard samples early in training.
        if train_sample_losses:
            _apply_curriculum(
                sampler, _base_sampler_weights, train_sample_losses,
                epoch, warmup_epochs, args.epochs, train_labels,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.time() - fold_t0
        epoch_eta     = _eta(epoch_elapsed, epoch, args.epochs)

        # E1: log PR-AUC alongside ROC-AUC.
        _log(
            f"  {epoch:>5}  {train_loss:>11.5f}  {val_loss:>9.5f}  "
            f"{val_auc:>9.4f}  pr_auc={val_pr_auc:.4f}  {current_lr:>10.2e}  "
            f"{_fmt_elapsed(epoch_elapsed):>9}  {epoch_eta:>9}"
        )

        improved = not math.isnan(val_auc) and val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_val_scores = list(val_scores)
            best_val_labels = list(val_labels_ep)
            epochs_no_improve = 0
            _atomic_torch_save(model.state_dict(), pt_save_path)
            # Per-fold checkpoint alongside the overall best.
            if fold_num > 0 and getattr(args, "save_all_folds", False):
                fold_path = pt_save_path.parent / f"exonet_fold_{fold_num}.pt"
                _atomic_torch_save(model.state_dict(), fold_path)
            _log(f"         >> New best val AUC {best_val_auc:.4f} -- checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                _log(f"\n  Early stopping after {epoch} epochs (patience={args.patience}).")
                break

        # Save epoch-level resume checkpoint after every completed epoch.
        try:
            _atomic_torch_save({
                "epoch":                  epoch,
                "model_state":            model.state_dict(),
                "optimizer_state":        optimizer.state_dict(),
                "warmup_scheduler_state": warmup_scheduler.state_dict(),
                "main_scheduler_state":   scheduler.state_dict() if scheduler is not None else None,
                "best_val_auc":           best_val_auc,
                "best_val_scores":        best_val_scores,
                "best_val_labels":        best_val_labels,
                "epochs_no_improve":      epochs_no_improve,
                "warmup_epochs":          warmup_epochs,
                # Fix 6: persist lr_schedule so resume can detect a scheduler mismatch.
                "lr_schedule":            lr_schedule,
            }, epoch_ckpt_path)
        except Exception as exc:
            _log(f"  WARNING: epoch checkpoint save failed ({exc})")

        # --max-hours: epoch checkpoint just saved, so this is a safe exit point.
        # Next launch will resume at epoch+1.
        max_hours = getattr(args, "max_hours", None)
        run_t_start = getattr(args, "_run_t_start", None)
        if max_hours is not None and run_t_start is not None:
            elapsed_h = (time.time() - run_t_start) / 3600.0
            if elapsed_h >= max_hours:
                _log(f"\n  --max-hours budget reached ({elapsed_h:.2f}h >= {max_hours:.2f}h). "
                     f"Exiting cleanly at fold {fold_num} epoch {epoch}; "
                     f"resume from epoch {epoch + 1} on next launch.")
                sys.exit(0)

    swa_epochs = getattr(args, "swa_epochs", 10)

    # Main training loop complete.  Do NOT delete the resume checkpoint yet —
    # if the run dies during SWA (hours of work), a relaunch must not restart
    # the fold from epoch 1.  Rewrite it with main_done=True so the next
    # launch skips straight to the SWA phase; _finalize_fold cleans it up
    # once the fold is truly done.
    if swa_epochs > 0 and not main_loop_done:
        try:
            _atomic_torch_save({
                "epoch":                  args.epochs,
                "model_state":            model.state_dict(),
                "optimizer_state":        optimizer.state_dict(),
                "warmup_scheduler_state": warmup_scheduler.state_dict(),
                "main_scheduler_state":   scheduler.state_dict() if scheduler is not None else None,
                "best_val_auc":           best_val_auc,
                "best_val_scores":        best_val_scores,
                "best_val_labels":        best_val_labels,
                "epochs_no_improve":      epochs_no_improve,
                "warmup_epochs":          warmup_epochs,
                "lr_schedule":            lr_schedule,
                "main_done":              True,
            }, epoch_ckpt_path)
        except Exception as exc:
            _log(f"  WARNING: main-done checkpoint save failed ({exc})")

    # M12: always write the per-fold checkpoint using the BEST weights (not the current
    # model which may be post-SWA or from the last epoch if no improvement occurred).
    if fold_num > 0 and getattr(args, "save_all_folds", False):
        fold_path = pt_save_path.parent / f"exonet_fold_{fold_num}.pt"
        if not fold_path.exists():
            # Load the best checkpoint from pt_save_path (saved whenever val AUC improved).
            best_state = torch.load(pt_save_path, map_location="cpu", weights_only=True)
            _atomic_torch_save(best_state, fold_path)
        _log(f"  Fold {fold_num} checkpoint: {fold_path.name}")

    # ── SWA phase ─────────────────────────────────────────────────────────────
    if swa_epochs > 0:
        # CF2: SWA hyperparameters now come from CLI args (--swa-lr, --swa-momentum).
        swa_lr       = getattr(args, "swa_lr", 1e-5)
        swa_momentum = getattr(args, "swa_momentum", 0.9)
        _log(f"\n  Running SWA for {swa_epochs} epochs at lr={swa_lr} ...")

        # Fix 7: snapshot the best pre-SWA weights before the SWA loop starts.
        # Guard against the case where no checkpoint was saved (all epochs had NaN
        # AUC) by falling back to the current model weights.
        if pt_save_path.exists():
            _pre_swa_best_state = torch.load(pt_save_path, map_location="cpu", weights_only=True)
        else:
            _pre_swa_best_state = copy.deepcopy(model.state_dict())
            _atomic_torch_save(_pre_swa_best_state, pt_save_path)

        # Fix 11: create an unweighted loader for BN statistics update only.
        # swa_loader uses WeightedRandomSampler (50/50 balanced batches); BN
        # running stats computed on those batches would not reflect the true
        # (heavily imbalanced) data distribution.  bn_loader uses shuffle=False
        # with no sampler so all samples appear with their natural frequency.
        bn_dataset = copy.copy(dataset)
        bn_dataset.augment = False
        bn_loader = DataLoader(
            Subset(bn_dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            worker_init_fn=_fold_worker_init,
        )

        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_opt   = torch.optim.SGD(model.parameters(), lr=swa_lr, momentum=swa_momentum)
        swa_sched = torch.optim.swa_utils.SWALR(swa_opt, swa_lr=swa_lr, anneal_epochs=max(1, swa_epochs // 2))

        # SWA-level resume: each SWA epoch checkpoints its full state below, so
        # a power cut mid-SWA continues from the last completed SWA epoch.
        swa_start = 0
        if swa_ckpt_path.exists():
            try:
                swa_ckpt = torch.load(swa_ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(swa_ckpt["model_state"])
                swa_model.load_state_dict(swa_ckpt["swa_model_state"])
                swa_opt.load_state_dict(swa_ckpt["swa_opt_state"])
                swa_sched.load_state_dict(swa_ckpt["swa_sched_state"])
                swa_start = swa_ckpt["completed_epochs"]
                _log(f"  Resuming SWA from epoch {swa_start + 1}/{swa_epochs}")
            except Exception as exc:
                _log(f"  WARNING: SWA checkpoint load failed ({exc}) — restarting SWA from epoch 1.")
                swa_start = 0

        try:
            for swa_ep in range(swa_start, swa_epochs):
                # Use bn_loader (natural class distribution, no WeightedRandomSampler)
                # for SWA gradient steps so the weight average is not biased toward
                # oversampled minority-class batches.  WeightedRandomSampler with
                # replacement means some samples are never seen in a given epoch,
                # producing a systematically biased SWA average on small datasets.
                _run_epoch(model, bn_loader, criterion, swa_opt, device)  # 4th return unused
                swa_model.update_parameters(model)
                swa_sched.step()

                _atomic_torch_save({
                    "completed_epochs": swa_ep + 1,
                    "model_state":      model.state_dict(),
                    "swa_model_state":  swa_model.state_dict(),
                    "swa_opt_state":    swa_opt.state_dict(),
                    # SWALR.state_dict() may hold a bound method (anneal_func)
                    # depending on the torch version — strip callables so the
                    # checkpoint stays loadable with weights_only=True.
                    # (load_state_dict is a plain __dict__.update, and
                    # anneal_func is rebuilt in __init__, so omitting it is safe.)
                    "swa_sched_state":  {k: v for k, v in swa_sched.state_dict().items()
                                         if not callable(v)},
                }, swa_ckpt_path)

                # --max-hours: SWA progress just checkpointed, so this is a safe
                # exit point.  (Previously SWA always ran to completion, over-
                # shooting the nightly budget by up to swa_epochs hours.)
                max_hours = getattr(args, "max_hours", None)
                run_t_start = getattr(args, "_run_t_start", None)
                if (max_hours is not None and run_t_start is not None
                        and swa_ep + 1 < swa_epochs):
                    elapsed_h = (time.time() - run_t_start) / 3600.0
                    if elapsed_h >= max_hours:
                        _log(f"\n  --max-hours budget reached ({elapsed_h:.2f}h >= {max_hours:.2f}h). "
                             f"Exiting cleanly after fold {fold_num} SWA epoch {swa_ep + 1}/{swa_epochs}; "
                             f"resume on next launch.")
                        sys.exit(0)
        except Exception as exc:
            # Fix 7: restore the pre-SWA best weights on crash so the on-disk
            # checkpoint is not left in a degraded mid-SWA state.  A real
            # exception here (not a power cut) would recur on every relaunch,
            # so give up on SWA and mark the fold complete with the pre-SWA best.
            _log(f"  WARNING: SWA training interrupted ({exc}) — restoring pre-SWA best checkpoint.")
            _atomic_torch_save(_pre_swa_best_state, pt_save_path)
            _finalize_fold(best_val_auc, best_val_scores, best_val_labels)
            return best_val_auc, best_val_scores, best_val_labels

        # Copy averaged weights into a plain ExoNet before BN update.
        # AveragedModel.forward does not reliably dispatch multiple positional
        # args to ExoNet.forward across all PyTorch versions — calling
        # swa_model(gv, lv, ...) ends up reaching ExoNet with only the first
        # tensor.  Extract the averaged state dict and run BN update on a plain
        # ExoNet directly, which has no dispatch ambiguity.
        swa_weights = ExoNet(use_se=getattr(args, "use_se", True), dropout=getattr(args, "dropout", 0.4)).to(device)
        swa_weights.load_state_dict(swa_model.module.state_dict())

        bn_ok = False
        try:
            swa_weights.train()
            with torch.no_grad():
                for gv, lv, ov, ev, sv, cv, scalar, _ in bn_loader:
                    gv = gv.to(device); lv = lv.to(device); ov = ov.to(device)
                    ev = ev.to(device); sv = sv.to(device); cv = cv.to(device)
                    scalar = scalar.to(device)
                    swa_weights(gv, lv, ov, ev, sv, cv, scalar)
            bn_ok = True
        except Exception as exc:
            _log(f"  WARNING: SWA BN update failed ({exc}) — skipping SWA evaluation.")

        if bn_ok:
            swa_weights.eval()
            _, swa_scores, swa_labels_ep, _ = _run_epoch(swa_weights, val_loader, criterion, None, device)
            swa_labels_binary = [float(l >= _POSITIVE_LABEL_THRESHOLD) for l in swa_labels_ep]
            try:
                swa_auc = roc_auc_score(swa_labels_binary, swa_scores)
            except ValueError:
                swa_auc = float("nan")
            _log(f"  SWA val AUC: {swa_auc:.4f}  (best so far: {best_val_auc:.4f})")
            if not math.isnan(swa_auc) and swa_auc > best_val_auc:
                _log("  SWA improved — saving SWA weights as best checkpoint.")
                _atomic_torch_save(swa_weights.state_dict(), pt_save_path)
                best_val_auc    = swa_auc
                best_val_scores = list(swa_scores)
                best_val_labels = list(swa_labels_ep)
                if fold_num > 0 and getattr(args, "save_all_folds", False):
                    fold_path = pt_save_path.parent / f"exonet_fold_{fold_num}.pt"
                    _atomic_torch_save(swa_weights.state_dict(), fold_path)
                    _log(f"  Updated fold-{fold_num} checkpoint with SWA weights")
            else:
                # Fix 7: SWA did not improve — restore the pre-SWA best weights
                # so pt_save_path holds the true best (not the SWA model).
                _atomic_torch_save(_pre_swa_best_state, pt_save_path)
        else:
            # BN update failed — restore pre-SWA best weights so pt_save_path
            # is not left in the degraded post-SWA-SGD state.
            _atomic_torch_save(_pre_swa_best_state, pt_save_path)

    _finalize_fold(best_val_auc, best_val_scores, best_val_labels)
    return best_val_auc, best_val_scores, best_val_labels


def train(args: argparse.Namespace) -> None:
    """Full training pipeline with k-fold cross-validation."""

    # --max-hours: stamp the run start so _train_fold and the fold loop can
    # check elapsed time against args.max_hours and exit cleanly at safe
    # boundaries (post epoch checkpoint, or between folds).
    args._run_t_start = time.time()
    if getattr(args, "max_hours", None) is not None:
        _log(f"  Wall-clock budget: {args.max_hours:.2f}h "
             f"(exits at the next epoch/fold boundary once exceeded).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # E2: set global random seeds for reproducibility.
    seed = getattr(args, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _log(f"  Random seed: {seed}")

    # H4: expose seed to DataLoader worker processes via the module-level variable.
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # L1: ensure deterministic CUDA ops for reproducibility (slight performance cost).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CF4: warn if --val-split was explicitly set (it has no effect in CV mode).
    if getattr(args, "val_split", 0.15) != 0.15:
        _log("WARNING: --val-split has no effect when --folds > 1 (CV mode)")

    # E4: warn if stellar params will be non-zero (train/inference mismatch).
    # H5: default is now True so this warning only fires when --no-zero-stellar-params is used.
    if not getattr(args, "zero_stellar_params", True):
        _log(
            "INFO: stellar params (scalar indices 8-12) are non-zero during training "
            "but zeroed at inference time. Use --zero-stellar-params to train a model "
            "that is consistent with inference behaviour."
        )

    # ── Build dataset ─────────────────────────────────────────────────────────
    # Build the base dataset without augmentation; _AugSubset applies it to
    # training folds only, keeping validation clean.
    # Fix 5: construct AugmentationConfig from CLI args and pass to dataset.
    aug_cfg = AugmentationConfig(
        injection_prob=getattr(args, "injection_prob", 0.30),
        noise_std=getattr(args, "noise_std", 0.002),
        dilution_prob=getattr(args, "dilution_prob", 0.20),
    )
    dataset = MultiMissionDataset(
        fits_dir=args.fits_dir,
        csv_path=args.csv_path,
        max_samples=args.max_samples,
        augment=False,
        cache_only=args.cache_only,
        cache_file=getattr(args, "cache_file", None),
        max_unlabeled=getattr(args, "max_unlabeled_negatives", 50_000),
        aug_cfg=aug_cfg,
        preprocess_workers=getattr(args, "preprocess_workers", 1),
        checkpoint_every=getattr(args, "checkpoint_every", 100),
        fits_index_file=getattr(args, "fits_index_file", None),
        missions=tuple(getattr(args, "missions", "kepler,tess,k2").split(",")),
    )

    if len(dataset) < 10:
        _log("ERROR: dataset contains fewer than 10 usable samples. Aborting.")
        sys.exit(1)

    if getattr(args, "preprocess_only", False):
        _log(f"\n--preprocess-only: cache built ({len(dataset)} samples). Exiting before training.")
        return

    # E4 / H5: apply zero-stellar-params flag on the dataset.
    dataset.zero_stellar_params = getattr(args, "zero_stellar_params", True)
    if dataset.zero_stellar_params:
        _log("  --zero-stellar-params: stellar param scalars (indices 8-12) will be zeroed during training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"\nTraining on {device}.")

    indices = np.array(range(len(dataset)))
    labels  = np.array(dataset.labels)
    pt_save_path   = output_dir / "exonet.pt"
    onnx_save_path = output_dir / "exonet.onnx"

    # ── K-fold cross-validation ───────────────────────────────────────────────
    # Binarise soft labels (koi_score floats) for stratification purposes only.
    # StratifiedKFold treats each unique float as a separate class, which breaks
    # stratification when koi_score is continuous.  Using >= 0.5 as the split
    # gives consistent class counts across folds while keeping soft values for loss.
    strat_labels = (labels >= _POSITIVE_LABEL_THRESHOLD).astype(int)

    # Star-grouped splits: every sample from one star lands on the same side of
    # every split. Each star yields several BLS peaks; splitting them across
    # train and test leaked the star and inflated test AUC by +0.08 last run.
    # Samples with no kepid get a unique group, so an old cache without kepids
    # splits exactly like plain stratification.
    groups = np.array([k if k else f"_{i}" for i, k in enumerate(dataset._kepids)])
    _log(f"  Star-grouped splits: {len(np.unique(groups))} groups over {len(groups)} samples")

    # ── Held-out test set (~10%) ──────────────────────────────────────────────
    # Stratified, star-grouped split on binarized labels. This set is NEVER
    # used for model selection, threshold tuning, or calibration — only for
    # final reporting.
    # E2: use the CLI seed for train/test split reproducibility.
    train_val_idx, test_idx = next(
        StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        .split(indices, strat_labels, groups)
    )
    _log(f"  Held-out test set: {len(test_idx)} samples ({100*len(test_idx)/len(dataset):.1f}%)")

    # Cross-validation runs only on train_val_idx
    indices_cv = train_val_idx
    labels_cv  = labels[train_val_idx]
    strat_cv   = strat_labels[train_val_idx]
    groups_cv  = groups[train_val_idx]

    n_splits = min(args.folds, int(min(np.sum(strat_cv == 0), np.sum(strat_cv == 1))))
    n_splits = max(n_splits, 2)

    # E2: use the CLI seed for cross-validation split reproducibility.
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs: list[float] = []

    overall_best_auc   = -1.0
    best_val_scores_all: list[float] = []
    best_val_labels_all: list[float] = []

    # E6: OOF (out-of-fold) prediction accumulators for unbiased threshold tuning.
    # Each sample appears in exactly one fold's validation set across K-fold CV.
    oof_scores: list[float]  = []
    oof_labels: list[float]  = []

    _log(f"\n  Starting {n_splits}-fold cross-validation  "
         f"({len(indices_cv)} train/val samples, device={device})")

    # One split manifest (absolute dataset indices) for every downstream
    # consumer — eval scripts and calibration must not rebuild their own splits.
    folds = list(skf.split(indices_cv, strat_cv, groups_cv))
    np.savez(output_dir / "split_manifest.npz",
             test_idx=test_idx, train_val_idx=train_val_idx,
             **{f"fold{f}_val_idx": indices_cv[va] for f, (_, va) in enumerate(folds, start=1)})

    cv_t_start = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        # Fully completed folds (incl. SWA) leave a fold_complete_{N}.json
        # marker with their results — skip retraining and reuse the stored
        # OOF scores.  Delete the marker to force a retrain.
        complete_path = _fold_complete_path(output_dir, fold)
        prev_result = None
        if complete_path.exists():
            try:
                with open(complete_path, "r", encoding="utf-8") as fh:
                    prev_result = json.load(fh)
            except Exception as exc:
                _log(f"  WARNING: could not read {complete_path.name} ({exc}) — retraining fold {fold}.")
                prev_result = None

        # --max-hours: don't start a new fold if the wall-clock budget is
        # already exhausted (completed folds are free to "skip through").
        max_hours = getattr(args, "max_hours", None)
        run_t_start = getattr(args, "_run_t_start", None)
        if max_hours is not None and run_t_start is not None and prev_result is None:
            elapsed_h = (time.time() - run_t_start) / 3600.0
            if elapsed_h >= max_hours:
                _log(f"\n  --max-hours budget reached ({elapsed_h:.2f}h >= {max_hours:.2f}h). "
                     f"Skipping fold {fold} and beyond; resume on next launch.")
                sys.exit(0)

        if prev_result is not None:
            fold_auc      = prev_result["fold_auc"]
            val_scores    = prev_result["best_val_scores"]
            val_labels_ep = prev_result["best_val_labels"]
            _log(f"\n  Fold {fold}/{n_splits} already complete (val AUC {fold_auc:.4f}) "
                 f"— skipping.  (Delete {complete_path.name} to retrain.)")
        else:
            _log(f"\n{'='*65}")
            _log(f"  Fold {fold}/{n_splits}  —  train={len(train_idx)}  val={len(val_idx)}  "
                 f"(cv elapsed so far: {_fmt_elapsed(time.time() - cv_t_start)})")
            _log(f"{'='*65}")

            # train_idx / val_idx from skf.split are positions within indices_cv;
            # map them back to absolute dataset indices.
            abs_train_idx = indices_cv[train_idx].tolist()
            abs_val_idx   = indices_cv[val_idx].tolist()

            fold_auc, val_scores, val_labels_ep = _train_fold(
                dataset,
                abs_train_idx,
                abs_val_idx,
                args,
                device,
                pt_save_path,
                fold_num=fold,
            )
        fold_aucs.append(fold_auc)

        # E6: accumulate OOF predictions.
        oof_scores.extend(val_scores)
        oof_labels.extend(val_labels_ep)

        if fold_auc > overall_best_auc:
            overall_best_auc = fold_auc
            best_val_scores_all = val_scores
            best_val_labels_all = val_labels_ep

        # Gate: stop cleanly after fold N so it can be evaluated before more
        # compute is spent. Rerun without the flag to continue from fold N+1.
        stop_after = getattr(args, "stop_after_fold", None)
        if stop_after is not None and fold >= stop_after and fold < n_splits:
            _log(f"\n  --stop-after-fold {stop_after}: fold {fold} done (val AUC {fold_auc:.4f}). "
                 f"Exiting; rerun without the flag to continue.")
            sys.exit(0)

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    _log(f"\n{'='*65}")
    _log(f"  Cross-validation complete in {_fmt_elapsed(time.time() - cv_t_start)}.")
    _log(f"  Fold AUCs : {[f'{a:.4f}' for a in fold_aucs]}")
    _log(f"  Mean AUC  : {mean_auc:.4f} ± {std_auc:.4f}")
    _log(f"  Best fold : {overall_best_auc:.4f}")
    _log(f"  Checkpoint: {pt_save_path}")
    _log(f"{'='*65}")
    # Machine-parseable summary line for the orchestrator.
    _log(f"RESULT: mean_auc={mean_auc:.4f} std_auc={std_auc:.4f} best_auc={overall_best_auc:.4f}")

    # E6: OOF-based threshold tuning — unbiased because each sample was a
    # validation sample exactly once (never seen by its scoring model during training).
    oof_threshold: float | None = None
    if oof_scores and oof_labels:
        _log("\nRunning OOF threshold sweep (unbiased — each sample scored exactly once) ...")
        oof_threshold = _tune_threshold(oof_scores, oof_labels, output_dir)
        _log(f"  OOF threshold (oof_threshold): {oof_threshold:.2f}")

    # ── Final evaluation on held-out test set ────────────────────────────────
    _log("\nEvaluating best checkpoint on held-out test set ...")
    criterion_eval = FocalLoss(gamma=2.0, pos_weight=None)
    test_model = ExoNet(
        use_se=getattr(args, "use_se", True),
        dropout=getattr(args, "dropout", 0.4),
    ).to(device)
    test_model.load_state_dict(torch.load(pt_save_path, map_location=device, weights_only=True))
    # Test set uses unaugmented data (val_dataset already has augment=False).
    test_dataset_eval = copy.copy(dataset)
    test_dataset_eval.augment = False
    # H4: worker_init_fn ensures seeded workers for the test loader too.
    test_loader = DataLoader(
        Subset(test_dataset_eval, test_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_make_worker_init_fn(0),
    )
    _, test_scores, test_labels_arr, _ = _run_epoch(test_model, test_loader, criterion_eval, None, device)
    test_labels_binary = [float(l >= _POSITIVE_LABEL_THRESHOLD) for l in test_labels_arr]
    try:
        test_auc = roc_auc_score(test_labels_binary, test_scores)
    except ValueError:
        test_auc = float("nan")
    # E1: compute PR-AUC on the test set.
    try:
        test_pr_auc = average_precision_score(test_labels_binary, test_scores)
    except ValueError:
        test_pr_auc = float("nan")
    _log(f"  HELD-OUT TEST AUC   : {test_auc:.4f}")
    _log(f"  HELD-OUT TEST PR-AUC: {test_pr_auc:.4f}")
    _log(f"RESULT_TEST: test_auc={test_auc:.4f} test_pr_auc={test_pr_auc:.4f}")

    # G3: save full PR curve and ROC curve data for paper figures.
    test_scores_np = np.array(test_scores)
    test_labels_np = np.array(test_labels_arr)
    # Curves need BINARISED labels: test_labels_np holds soft koi_score floats,
    # which sklearn rejects with "continuous format is not supported" (this
    # silently skipped both curve files for the whole 2026-08-03 run).
    test_labels_bin_np = np.array(test_labels_binary)
    try:
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(test_labels_bin_np, test_scores_np)
        pr_curve_path = output_dir / "pr_curve.npz"
        np.savez(pr_curve_path,
                 precision=precision_arr.astype(np.float32),
                 recall=recall_arr.astype(np.float32),
                 thresholds=pr_thresholds.astype(np.float32))
        _log(f"  PR curve saved: {pr_curve_path}")
    except Exception as exc:
        _log(f"  WARNING: PR curve save failed ({exc})")
    try:
        fpr_arr, tpr_arr, roc_thresholds = roc_curve(test_labels_bin_np, test_scores_np)
        roc_curve_path = output_dir / "roc_curve.npz"
        np.savez(roc_curve_path,
                 fpr=fpr_arr.astype(np.float32),
                 tpr=tpr_arr.astype(np.float32),
                 thresholds=roc_thresholds.astype(np.float32))
        _log(f"  ROC curve saved: {roc_curve_path}")
    except Exception as exc:
        _log(f"  WARNING: ROC curve save failed ({exc})")

    # E6: tune threshold on test set for reference only (labelled accordingly).
    # The OOF threshold above is preferred for production use.
    # H2: pass a different save_path so this biased threshold does NOT overwrite
    # the unbiased OOF threshold.json written by the call above.
    test_threshold: float | None = None
    if test_scores and test_labels_arr:
        _log("\nRunning threshold sweep on held-out test set (for reference only — optimistically biased) ...")
        test_threshold = _tune_threshold(
            test_scores, test_labels_arr, output_dir,
            save_path=output_dir / "threshold_test_reference.json",
        )

    # G4: save confusion matrix and classification report at OOF threshold.
    if oof_threshold is not None:
        try:
            test_labels_arr_binary = (test_labels_np >= _POSITIVE_LABEL_THRESHOLD).astype(int)
            preds = (test_scores_np >= oof_threshold).astype(int)
            cm = confusion_matrix(test_labels_arr_binary, preds)
            report = classification_report(
                test_labels_arr_binary, preds,
                target_names=["false_positive", "planet"],
                output_dict=True,
            )
            eval_report = {
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "threshold_used": float(oof_threshold),
                "n_test_samples": len(test_labels_np),
            }
            with open(output_dir / "evaluation_report.json", "w") as f:
                json.dump(eval_report, f, indent=2)
            _log(f"  Evaluation report saved: {output_dir / 'evaluation_report.json'}")
        except Exception as exc:
            _log(f"  WARNING: evaluation report save failed ({exc})")

    # E5: per-mission test AUC breakdown.
    test_results: dict = {
        "test_auc":    round(test_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
        "n_test":      len(test_idx),
    }
    if oof_threshold is not None:
        test_results["oof_threshold"]  = round(oof_threshold, 2)
    if test_threshold is not None:
        test_results["test_threshold"] = round(test_threshold, 2)
        test_results["test_threshold_note"] = "for reference only — optimistically biased"

    # E5: per-mission AUC if the missions array is available.
    if hasattr(dataset, "_missions") and dataset._missions:
        missions_arr  = np.array(dataset._missions)
        test_idx_list = test_idx.tolist()
        test_missions = missions_arr[test_idx_list]
        test_scores_arr  = np.array(test_scores)
        test_labels_arr2 = np.array(test_labels_arr)
        for mission_name in ["kepler", "tess", "k2"]:
            m_mask = test_missions == mission_name
            if m_mask.sum() < 10:
                _log(f"  Skipping per-mission AUC for '{mission_name}' "
                     f"(only {m_mask.sum()} test samples).")
                continue
            try:
                m_labels_bin = (test_labels_arr2[m_mask] >= _POSITIVE_LABEL_THRESHOLD).astype(float)
                m_auc    = roc_auc_score(m_labels_bin, test_scores_arr[m_mask])
                m_pr_auc = average_precision_score(m_labels_bin, test_scores_arr[m_mask])
            except ValueError:
                m_auc    = float("nan")
                m_pr_auc = float("nan")
            _log(f"  {mission_name.upper():7s} test AUC={m_auc:.4f}  PR-AUC={m_pr_auc:.4f}  (n={m_mask.sum()})")
            test_results[f"test_auc_{mission_name}"]    = round(m_auc, 4)
            test_results[f"test_pr_auc_{mission_name}"] = round(m_pr_auc, 4)

    with (output_dir / "test_results.json").open("w") as fh:
        json.dump(test_results, fh, indent=2)

    # ── ONNX export ───────────────────────────────────────────────────────────
    _log("\nExporting best checkpoint to ONNX ...")
    ExoNetInference.export_from_pytorch(pt_save_path, onnx_save_path)
    _log(f"ONNX model: {onnx_save_path}")

    # Export per-fold ONNX files if fold checkpoints were saved
    if getattr(args, "save_all_folds", False):
        _log("\nExporting per-fold ONNX models ...")
        for k in range(1, n_splits + 1):
            fold_pt   = output_dir / f"exonet_fold_{k}.pt"
            fold_onnx = output_dir / f"exonet_fold_{k}.onnx"
            if fold_pt.exists():
                ExoNetInference.export_from_pytorch(fold_pt, fold_onnx)
                _log(f"  Fold {k}: {fold_onnx.name}")
        _log(f"Use EnsembleInference({[f'exonet_fold_{k}.onnx' for k in range(1, n_splits + 1)]}) for best accuracy.")


# ── New CLI flags ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ExoNet on multi-mission data (Kepler/TESS/K2) and export to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--max-hours",   type=float, default=None,
                        help="Wall-clock budget in hours. Process exits cleanly at the next "
                             "epoch/fold boundary once exceeded. Resume picks up from the last "
                             "saved checkpoint. Default: no limit. SWA phase, if running when "
                             "the budget is hit, is allowed to finish (typically ~1h overshoot).")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping max norm.")
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--folds",       type=int,   default=5,
                        help="Number of stratified k-folds for cross-validation.")
    parser.add_argument("--fits-dir",    type=Path,  required=True)
    parser.add_argument("--output-dir",  type=Path,  required=True)
    parser.add_argument("--val-split",   type=float, default=0.15,
                        help="(Unused when --folds > 1; kept for compatibility.)")
    parser.add_argument("--max-samples", type=int,   default=None)
    parser.add_argument("--csv-path",    type=Path,  default=None)
    parser.add_argument("--num-workers",   type=int,   default=0)
    parser.add_argument("--preprocess-workers", type=int, default=1,
                        help="Number of parallel processes for BLS preprocessing. "
                             "Default 1 (sequential). Set to os.cpu_count()-1 for full parallelism.")
    parser.add_argument("--no-augment",    action="store_true")
    parser.add_argument("--simple-loss",   action="store_true",
                        help="Focal classification loss only (no aux/entropy/Kendall/physics terms).")
    parser.add_argument("--missions", default="kepler,tess,k2",
                        help="Comma-separated missions to build into the cache, e.g. 'tess' for a "
                             "TESS-only cache (merge caches with scripts/merge_caches.py).")
    parser.add_argument("--fits-index-file", type=Path, default=None,
                        help="Pre-generated FITS listing (one path per line) used instead of "
                             "scanning --fits-dir. Avoids rescanning huge network mounts on "
                             "every restart.")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save the incremental preprocessing checkpoint every N files. "
                             "Each save recompresses the whole accumulated dataset — raise "
                             "this on slow CPUs so save time stays small next to work time.")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Build the preprocessing cache and exit without training. "
                             "Use for the standalone cache rebuild.")
    parser.add_argument("--cache-only",    action="store_true",
                        help="Skip MAST downloads; only use FITS files already on disk.")
    parser.add_argument("--cache-file",    type=Path,  default=None,
                        help="Path to a .npz preprocessing cache file. "
                             "Saves on first run, loads on subsequent runs to skip BLS preprocessing.")
    parser.add_argument("--lr-schedule",   type=str,   default="plateau",
                        choices=["plateau", "cosine"],
                        help="LR scheduler: 'plateau' (ReduceLROnPlateau) or 'cosine' (CosineAnnealingWarmRestarts).")
    parser.add_argument("--cosine-t0",     type=int,   default=10,
                        help="T_0 period (epochs) for CosineAnnealingWarmRestarts.")
    parser.add_argument("--use-se",        action="store_true",  default=True,
                        help="Enable Squeeze-and-Excite channel attention (default: on).")
    parser.add_argument("--no-se",         action="store_false", dest="use_se",
                        help="Disable Squeeze-and-Excite channel attention.")
    parser.add_argument("--dropout",       type=float, default=0.4,
                        help="Dropout rate for CNN branch heads (0–1).")
    parser.add_argument("--weight-decay",  type=float, default=1e-4,
                        help="AdamW weight decay.")
    parser.add_argument("--stop-after-fold", type=int, default=None,
                        help="Exit cleanly after this fold completes (evaluation gate).")
    parser.add_argument("--save-all-folds", action="store_true",
                        help="Save a per-fold checkpoint exonet_fold_k.pt alongside the overall best.")
    parser.add_argument("--max-unlabeled-negatives", type=int, default=50_000,
                        help="Max unlabeled Kepler stars to add as negatives. 0 = disabled.")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Number of linear LR warmup epochs before the main scheduler takes over.")
    parser.add_argument("--swa-epochs", type=int, default=10,
                        help="Number of SWA averaging epochs after main training loop (0 = disabled).")
    # CF2: SWA hyperparameters exposed as CLI arguments.
    parser.add_argument("--swa-lr", type=float, default=1e-5,
                        help="Learning rate for SWA SGD optimiser.")
    parser.add_argument("--swa-momentum", type=float, default=0.9,
                        help="Momentum for SWA SGD optimiser.")
    parser.add_argument("--pretrained-global", type=Path, default=None,
                        help="Path to MAE-pretrained GlobalBranch state dict (global_branch_pretrained.pt). "
                             "If provided, initialises GlobalBranch before supervised training.")
    parser.add_argument("--pretrained-local", type=Path, default=None,
                        help="Path to MAE-pretrained LocalBranch state dict (local_branch_pretrained.pt). "
                             "If provided, initialises local/odd/even/secondary branches before supervised training.")
    # E2: global random seed for reproducibility.
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for NumPy, Python random, and PyTorch.")
    # H5 / E4: zero stellar params to match inference behaviour.
    # Default is True so the trained model is consistent with deployment where KIC data is unavailable.
    parser.add_argument("--zero-stellar-params", action="store_true", default=True, dest="zero_stellar_params",
                        help="Zero out stellar parameters (indices 8-12) during training to match "
                             "inference behaviour where KIC data is unavailable. "
                             "Recommended: True (default). "
                             "Set --no-zero-stellar-params to use KIC stellar params during training (experimental).")
    parser.add_argument("--no-zero-stellar-params", action="store_false", dest="zero_stellar_params",
                        help="Use KIC stellar params during training (experimental; creates train/inference mismatch).")
    # Fix 5: expose the most important augmentation knobs as CLI arguments.
    parser.add_argument("--injection-prob", type=float, default=0.30,
                        help="Probability of injecting a synthetic transit into negative samples.")
    parser.add_argument("--noise-std", type=float, default=0.002,
                        help="Standard deviation of Gaussian noise added to all views during augmentation.")
    parser.add_argument("--dilution-prob", type=float, default=0.20,
                        help="Probability of applying dilution (contamination) augmentation.")
    parser.add_argument("--hard-neg-update-freq", type=int, default=5,
                        help="Update hard-negative sampler every N epochs after warmup (0 = disabled).")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps. Effective batch size = batch-size × grad-accum-steps. "
                             "Use 4 with --batch-size 8 to simulate batch-size 32 on a 6 GB GPU.")
    return parser.parse_args(argv)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
