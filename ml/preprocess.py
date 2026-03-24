"""
Mission-agnostic light curve preprocessing pipeline.

Converts raw FITS light curve data (any mission) into fixed-length,
normalised arrays suitable for transit classification by the ML model.

Pipeline
--------
1. Load    — read FITS file, extract time + flux arrays
2. Clean   — remove NaNs, sigma-clip outliers
3. Detrend — flatten stellar variability with a Savitzky-Golay filter
4. Normalise — zero-mean, unit-variance flux
5. BLS search — find candidate transit periods via Box Least Squares
6. Fold & bin — phase-fold around each candidate, bin to fixed length

Output
------
A list of TransitCandidate objects (one per period candidate), each
carrying a global_view (2001-point full-orbit view) and a local_view
(201-point view zoomed on the transit). Both are ready to feed directly
into the classifier.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter


# ── Tuneable constants ────────────────────────────────────────────────────────

N_GLOBAL_BINS: int = 2001       # phase bins spanning full orbit  [-0.5, 0.5]
N_LOCAL_BINS: int  = 201        # phase bins zoomed on the transit window

SIGMA_CLIP_SIGMA: float = 5.0   # outlier rejection threshold (σ)
DETREND_WINDOW_DAYS: float = 3.0  # Savitzky-Golay filter width (days)
MIN_PERIOD_DAYS: float = 0.5    # shortest period to search (days)


# Column names to look for, in priority order.
# Covers Kepler, TESS, K2 (Everest / K2SFF), and CoRoT pipeline formats.
_FLUX_COLS = ("PDCSAP_FLUX", "KSPSAP_FLUX", "FLUX", "SAP_FLUX")
_TIME_COLS = ("TIME",)
_ERR_COLS  = ("PDCSAP_FLUX_ERR", "KSPSAP_FLUX_ERR", "FLUX_ERR", "SAP_FLUX_ERR")


# ── Output data structure ─────────────────────────────────────────────────────

@dataclass
class TransitCandidate:
    """A single period candidate produced by the preprocessing pipeline."""

    period: float        # best-fit period (days)
    t0: float            # epoch of first transit centre (mission time system)
    duration: float      # transit duration (days)
    depth: float         # fractional flux drop (positive = dimming)
    bls_power: float     # BLS signal-detection efficiency (dimensionless)
    global_view: np.ndarray  # shape (N_GLOBAL_BINS,) — full-orbit phase curve
    local_view: np.ndarray   # shape (N_LOCAL_BINS,)  — transit-window zoom
    secondary_depth: float = 0.0  # depth of strongest dip near phase 0.5
    odd_even_diff: float = 0.0    # |mean_odd_depth − mean_even_depth|


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pick_column(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for col in candidates:
        if col in available:
            return col
    return None


def _load_fits(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (time, flux, flux_err) from any mission FITS light curve file.

    Iterates over all binary table extensions and uses the first one that
    contains a TIME column and a recognised flux column.
    """
    with fits.open(path) as hdul:
        table = None
        for ext in hdul[1:]:
            if not hasattr(ext, "columns"):
                continue
            available = {c.name.upper() for c in ext.columns}
            if "TIME" in available and _pick_column(_FLUX_COLS, available):
                table = ext
                break

        if table is None:
            raise ValueError(f"No recognised light curve extension in {path}")

        available = {c.name.upper() for c in table.columns}
        flux_col = _pick_column(_FLUX_COLS, available)
        err_col  = _pick_column(_ERR_COLS,  available)

        time = table.data["TIME"].astype(np.float64)
        flux = table.data[flux_col].astype(np.float64)
        err  = (table.data[err_col].astype(np.float64)
                if err_col else np.ones_like(flux))

    return time, flux, err


def _drop_nans(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return tuple(arr[mask] for arr in arrays)


def _clip_outliers(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped = sigma_clip(flux, sigma=SIGMA_CLIP_SIGMA, maxiters=5)
    keep = ~clipped.mask
    return time[keep], flux[keep], err[keep]


def _detrend(time: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Remove stellar variability using a Savitzky-Golay filter.

    The filter is applied independently to each continuous segment so that
    data gaps (momentum dumps, quarterly breaks, sector boundaries) do not
    smear the trend across the gap.

    Returns the normalised residual:  (flux - trend) / trend
    """
    cadence = np.nanmedian(np.diff(time))
    window_pts = int(DETREND_WINDOW_DAYS / cadence)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, 5)

    # Find segment boundaries: gaps larger than 2× the typical cadence
    gap_indices = np.where(np.diff(time) > 2 * cadence)[0] + 1
    boundaries  = np.concatenate([[0], gap_indices, [len(time)]])

    residual = np.empty_like(flux)
    for i in range(len(boundaries) - 1):
        sl  = slice(boundaries[i], boundaries[i + 1])
        seg = flux[sl]
        w   = min(window_pts, len(seg))
        if w % 2 == 0:
            w -= 1

        if w < 5:
            # Segment too short to filter — use median as a flat trend
            trend = np.full_like(seg, np.median(seg))
        else:
            trend = savgol_filter(seg, window_length=w, polyorder=2)

        # Guard against near-zero trend values
        trend = np.where(np.abs(trend) < 1e-10, 1e-10, trend)
        residual[sl] = (seg - trend) / trend

    return residual


def _normalise(flux: np.ndarray) -> np.ndarray:
    mu, sigma = np.mean(flux), np.std(flux)
    if sigma < 1e-10:
        return flux - mu
    return (flux - mu) / sigma


def _bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    n_candidates: int,
) -> list[dict]:
    """
    Run Box Least Squares and return up to n_candidates period candidates.

    Candidates that are harmonics of a stronger detection are skipped so
    each independent signal is returned only once.
    """
    baseline   = time[-1] - time[0]
    max_period = min(baseline / 2, 13.0)  # never exceed half the baseline

    if max_period <= MIN_PERIOD_DAYS:
        return []

    # Log-uniform period grid gives equal resolution at short and long periods
    periods   = np.exp(np.linspace(np.log(MIN_PERIOD_DAYS), np.log(max_period), 5000))
    durations = np.array([0.05, 0.1, 0.15, 0.2, 0.3])  # days

    bls    = BoxLeastSquares(time, flux)
    result = bls.power(periods, durations, objective="snr")

    candidates   = []
    used_periods = []

    for idx in np.argsort(result.power)[::-1]:
        p = float(result.period[idx])

        # Reject if within 5 % of a known period or its first harmonic
        is_harmonic = any(
            abs(p - up) / up < 0.05 or abs(p - up / 2) / (up / 2) < 0.05
            for up in used_periods
        )
        if is_harmonic:
            continue

        stats = bls.compute_stats(
            result.period[idx],
            result.duration[idx],
            result.transit_time[idx],
        )
        candidates.append({
            "period":   p,
            "t0":       float(result.transit_time[idx]),
            "duration": float(result.duration[idx]),
            "depth":    float(stats["depth"][0]),
            "power":    float(result.power[idx]),
        })
        used_periods.append(p)

        if len(candidates) >= n_candidates:
            break

    return candidates


def _bin_phase(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Bin a phase-folded light curve into n_bins evenly-spaced phase bins.
    Empty bins are filled by linear interpolation. Result is normalised.
    """
    edges   = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    binned  = np.full(n_bins, np.nan)

    for j in range(n_bins):
        in_bin = (phase >= edges[j]) & (phase < edges[j + 1])
        if in_bin.any():
            binned[j] = np.median(flux[in_bin])

    valid = np.isfinite(binned)
    if valid.sum() >= 2:
        binned = np.interp(centres, centres[valid], binned[valid])
    else:
        binned = np.zeros(n_bins)

    return _normalise(binned)


def _secondary_depth(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    period: float,
) -> float:
    """
    Measure the flux depth at the secondary-eclipse position (phase ±0.5).

    Returns the depth as a positive fractional value (dip = positive).
    """
    half_window = min(2.0 * duration / period, 0.1)
    near_half = np.abs(np.abs(phase) - 0.5) < half_window
    if near_half.sum() < 3:
        return 0.0
    return float(-np.median(flux[near_half]))


def _odd_even_diff(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """
    Absolute difference in mean transit depth between odd and even transits.

    Computed by folding at 2× the period: odd transits land near phase 0,
    even transits land near phase 0.5.
    """
    double_period = 2.0 * period
    half_dur = duration / 2.0

    # Phase in [-0.5, 0.5] at twice the period; transit 1 at 0, transit 2 at ±0.5
    phase2 = ((time - t0) / double_period + 0.5) % 1.0 - 0.5

    in_odd  = np.abs(phase2) < (half_dur / double_period + 0.01)
    in_even = np.abs(np.abs(phase2) - 0.5) < (half_dur / double_period + 0.01)

    if in_odd.sum() < 3 or in_even.sum() < 3:
        return 0.0

    depth_odd  = float(-np.median(flux[in_odd]))
    depth_even = float(-np.median(flux[in_even]))
    return float(abs(depth_odd - depth_even))


def _fold_and_bin(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold the light curve and produce both views expected by the model.

    global_view : 2001 bins, phase [-0.5, 0.5] — the full orbit
    local_view  :  201 bins, phase ±2× transit duration — transit detail
    """
    # Phase in [-0.5, 0.5], transit centred at 0
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

    global_view = _bin_phase(phase, flux, N_GLOBAL_BINS, -0.5, 0.5)

    # Local window: ±2× transit duration, capped at ±0.4 to avoid wrapping
    half_window = min(2 * duration / period, 0.4)
    local_view  = _bin_phase(phase, flux, N_LOCAL_BINS, -half_window, half_window)

    return global_view, local_view


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    fits_path: str | Path,
    n_candidates: int = 5,
) -> list[TransitCandidate]:
    """
    Run the full preprocessing pipeline on a single light curve FITS file.

    Parameters
    ----------
    fits_path :
        Path to a light curve FITS file from any supported mission
        (Kepler, TESS, K2, CoRoT, or any file using the same column names).
    n_candidates :
        Maximum number of period candidates to return per star.

    Returns
    -------
    List of TransitCandidate objects sorted by BLS power (strongest first).
    Returns an empty list if the file cannot be processed or no candidates
    are found.

    Raises
    ------
    ValueError
        If the FITS file cannot be read or contains no recognised columns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        time, flux, err = _load_fits(fits_path)
        time, flux, err = _drop_nans(time, flux, err)

        if len(time) < 100:
            return []  # too few cadences to search reliably

        time, flux, err = _clip_outliers(time, flux, err)
        flux = _detrend(time, flux)
        flux = _normalise(flux)

        candidates = _bls_search(time, flux, n_candidates)
        if not candidates:
            return []

        results = []
        for c in candidates:
            period, t0, duration = c["period"], c["t0"], c["duration"]
            phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

            global_view, local_view = _fold_and_bin(time, flux, period, t0, duration)
            sec_depth = _secondary_depth(phase, flux, duration, period)
            oe_diff   = _odd_even_diff(time, flux, period, t0, duration)

            results.append(TransitCandidate(
                period          = period,
                t0              = t0,
                duration        = duration,
                depth           = c["depth"],
                bls_power       = c["power"],
                global_view     = global_view,
                local_view      = local_view,
                secondary_depth = sec_depth,
                odd_even_diff   = oe_diff,
            ))

    return results
