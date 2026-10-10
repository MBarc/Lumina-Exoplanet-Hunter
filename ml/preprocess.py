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
from dataclasses import dataclass, field
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
# Covers Kepler, TESS, K2 (Everest / K2SFF / KEGS), and CoRoT pipeline formats.
# HLSP KEGS files (hlsp_kegs_k2_lightcurve_*) use FCOR1–FCOR5 (GP corrections
# for 5 aperture sizes) and FRAW (raw flux). There are no per-point error columns.
_FLUX_COLS   = ("PDCSAP_FLUX", "KSPSAP_FLUX", "FCOR1", "FCOR2", "FCOR3", "FCOR4", "FCOR5", "FLUX", "SAP_FLUX", "FRAW")
_TIME_COLS   = ("TIME",)
_ERR_COLS    = ("PDCSAP_FLUX_ERR", "KSPSAP_FLUX_ERR", "FLUX_ERR", "SAP_FLUX_ERR", "FRAW_ERR")
_CENTR_COLS_1 = ("MOM_CENTR1", "POS_CORR1")
_CENTR_COLS_2 = ("MOM_CENTR2", "POS_CORR2")


# ── Output data structure ─────────────────────────────────────────────────────

@dataclass
class TransitCandidate:
    """A single period candidate produced by the preprocessing pipeline.

    Attribute naming note (Issue 4.5):
        ``raw_global_view`` is named for historical reasons but is NOT truly
        raw flux.  It is the phase-binned global view WITHOUT the per-view
        z-score (i.e. the output of _bin_phase_raw), but the input flux has
        already been through the per-light-curve z-score (_normalise) before
        phase-folding.  A more accurate name would be ``prenorm_global_view``
        (after light-curve z-score, before per-view z-score).  The name is
        kept for backwards compatibility with existing cache files.
    """

    period: float        # best-fit period (days)
    t0: float            # epoch of first transit centre (mission time system)
    duration: float      # transit duration (days)
    depth: float         # BLS depth in units of the light curve's std (the flux is z-scored
                         # before the search); model inputs use this, so it stays as is
    bls_power: float     # BLS signal-detection efficiency (dimensionless)
    global_view: np.ndarray  # shape (N_GLOBAL_BINS,) — full-orbit phase curve (per-view z-scored)
    local_view: np.ndarray   # shape (N_LOCAL_BINS,)  — transit-window zoom (per-view z-scored)
    secondary_depth: float = 0.0  # depth of strongest dip in phase range [0.3, 0.7]
    odd_even_diff: float = 0.0    # |mean_odd_depth − mean_even_depth|
    odd_view: np.ndarray       = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    even_view: np.ndarray      = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    secondary_view: np.ndarray = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    centroid_shift: float = 0.0   # centroid displacement during transit (pixel units)
    n_transits: float = 0.0       # estimated number of observed transit events
    raw_global_view: np.ndarray  = field(default_factory=lambda: np.zeros(2001, dtype=np.float32))
    # prenorm_global_view (stored as raw_global_view for backwards compat):
    # phase-binned global view WITHOUT the per-view z-score (but after the
    # per-light-curve z-score applied by _normalise).
    #
    # Channel dynamic range note (Fix 3):
    #   global_view (channel 0): double z-scored — first across the full LC,
    #     then across the 2001 phase bins (per-view z-score).
    #   raw_global_view (channel 1): single z-scored — only across the full LC.
    #   These two channels have different dynamic ranges; GlobalBranch BatchNorm
    #   layers normalise them independently per channel.
    centroid_curve: np.ndarray   = field(default_factory=lambda: np.zeros(201,  dtype=np.float32))
    noise_floor: float = 0.001
    depth_frac: float = 0.0       # physical fractional depth on unclipped relative flux; for reporting
    transit_snr: float = 0.0      # depth / robust out-of-transit noise * sqrt(in-transit points)
    # Reviewer diagnostics on unclipped relative flux (see _physical_diagnostics);
    # NOT model inputs.
    secondary_frac: float = 0.0
    odd_even_frac: float = 0.0
    transit_view_rel: np.ndarray   = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    secondary_view_rel: np.ndarray = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    # Raw (prenorm) local-scale views — NOT per-view z-scored, analogous to
    # raw_global_view.  Used as channel 1 of the 2-channel local branch inputs
    # so the model sees both the normalised transit shape and its original scale.
    raw_local_view: np.ndarray      = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    raw_odd_view: np.ndarray        = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    raw_even_view: np.ndarray       = field(default_factory=lambda: np.zeros(201, dtype=np.float32))
    raw_secondary_view: np.ndarray  = field(default_factory=lambda: np.zeros(201, dtype=np.float32))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pick_column(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for col in candidates:
        if col in available:
            return col
    return None


def _load_fits(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (time, flux, flux_err, centr1, centr2) from any mission FITS file.

    Centroid arrays are NaN-filled if the file contains no centroid columns.
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
        flux_col   = _pick_column(_FLUX_COLS,    available)
        err_col    = _pick_column(_ERR_COLS,     available)
        centr1_col = _pick_column(_CENTR_COLS_1, available)
        centr2_col = _pick_column(_CENTR_COLS_2, available)

        time = table.data["TIME"].astype(np.float64)
        flux = table.data[flux_col].astype(np.float64)
        err  = (table.data[err_col].astype(np.float64)
                if err_col else np.ones_like(flux))
        centr1 = (table.data[centr1_col].astype(np.float64)
                  if centr1_col else np.full_like(time, np.nan))
        centr2 = (table.data[centr2_col].astype(np.float64)
                  if centr2_col else np.full_like(time, np.nan))

    return time, flux, err, centr1, centr2


def _drop_nans(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    centr1: np.ndarray,
    centr2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop cadences where time, flux, or err are non-finite.

    Centroid columns are intentionally excluded from the finite mask: TESS FFI
    targets and many 2-min targets lack MOM_CENTR1/2, so those arrays are
    all-NaN.  Including them in the mask would silently discard every cadence
    for such targets.  NaN centroid values are handled gracefully downstream by
    _compute_centroid_curve and _centroid_shift, which check np.isfinite
    independently.
    """
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(err)
    return time[mask], flux[mask], err[mask], centr1[mask], centr2[mask]


def _clip_outliers(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    *extra: np.ndarray,
) -> tuple[np.ndarray, ...]:
    clipped = sigma_clip(flux, sigma=SIGMA_CLIP_SIGMA, maxiters=5)
    keep = ~np.ma.getmaskarray(clipped)
    return (time[keep], flux[keep], err[keep]) + tuple(a[keep] for a in extra)


def _detrend(time: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Remove stellar variability using a Savitzky-Golay filter.

    The filter is applied independently to each continuous segment so that
    data gaps (momentum dumps, quarterly breaks, sector boundaries) do not
    smear the trend across the gap.

    Returns the normalised residual:  (flux - trend) / trend
    """
    cadence = np.nanmedian(np.diff(time))
    if not np.isfinite(cadence) or cadence <= 0:
        # Degenerate time axis (< 2 cadences or all identical timestamps)
        return (flux - np.median(flux)) / max(np.std(flux), 1e-10)
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
        if len(seg) == 0:
            continue
        w   = min(window_pts, len(seg))
        if w % 2 == 0:
            w -= 1
        # M6: savgol_filter requires window_length < x.size; clamp to avoid crash.
        if w >= len(seg):
            w = len(seg) - 1
        if w % 2 == 0:
            w -= 1
        w = max(w, 1)  # ensure w stays positive after odd-correction on tiny segments

        # w < 5 covers w < 1: both cases use a flat median trend.
        if w < 5:
            # Segment too short for SG filter (also handles w==0/negative) — use median.
            trend = np.full_like(seg, np.median(seg))
        else:
            trend = savgol_filter(seg, window_length=w, polyorder=2)

        # Guard against near-zero trend values (applied unconditionally so
        # single-cadence zero-flux segments do not produce 0/0 NaN).
        trend = np.where(np.abs(trend) < 1e-10, 1e-10, trend)
        residual[sl] = (seg - trend) / trend

    return residual


def _detrend_iterative(time: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """
    Iterative Savitzky-Golay detrending with in-transit masking.

    Issue 4.3 fix: a single SG pass with a 3-day window absorbs part of a
    shallow transit signal.  This function runs a two-pass approach:
      1. Initial SG detrend to get a rough residual.
      2. Fast coarse BLS on the residual to locate approximate transit times.
      3. Mask cadences within ±0.6× duration of each transit centre.
      4. Re-run SG on out-of-transit cadences, interpolating over masked gaps.
      5. Use the second-pass trend for the final residual.

    Falls back to the single-pass _detrend if fewer than 100 cadences remain
    after masking (degenerate light curve or very long transit duration).
    """
    # --- Pass 1: rough detrend ------------------------------------------------
    rough_residual = _detrend(time, flux)

    # --- Coarse BLS on the rough residual to find approximate transit times ---
    try:
        baseline = time[-1] - time[0]
        max_p    = min(baseline / 2.0, 50.0)   # cap at 50 d for speed
        if max_p <= MIN_PERIOD_DAYS:
            return rough_residual

        n_periods   = 500   # coarse grid — speed over precision
        periods_c   = np.exp(np.linspace(np.log(MIN_PERIOD_DAYS), np.log(max_p), n_periods))
        durations_c = np.array([0.05, 0.1, 0.2, 0.5])   # days

        bls_c   = BoxLeastSquares(time, rough_residual)
        result_c = bls_c.power(periods_c, durations_c, objective="snr")
        best_idx  = int(np.argmax(result_c.power))
        # Fix 2: require a plausible signal before proceeding with iterative
        # masking.  If the coarse BLS peak is just noise the transit mask would
        # be placed at the wrong location, corrupting pass-2 detrending.
        # BLS power threshold: 15.0 is chosen conservatively because `rough_residual`
        # is a dimensionless SG residual of z-scored flux, where white noise can produce
        # spurious BLS peaks of 5–10 by chance. A threshold of 15.0 reduces the false-
        # alarm rate while still triggering iterative detrending on real transit signals.
        # Adaptive threshold: 15.0 is the minimum, but scale upward on
        # noisy light curves so the iterative mask is not placed on noise peaks.
        adaptive_threshold = max(15.0, 5.0 * rough_residual.std() * np.sqrt(len(rough_residual)))
        if result_c.power[best_idx] < adaptive_threshold:
            return rough_residual
        best_p    = float(result_c.period[best_idx])
        best_dur  = float(result_c.duration[best_idx])
        best_t0   = float(result_c.transit_time[best_idx])
    except Exception as exc:
        # EH2: log the BLS failure before falling back to single-pass detrend.
        # L5: use the module-level warnings import; no local re-import needed.
        warnings.warn(
            f"Iterative detrend BLS failed ({type(exc).__name__}: {exc}), "
            "using single-pass SG"
        )
        return rough_residual

    # --- Build in-transit mask ------------------------------------------------
    phase_c    = ((time - best_t0) / best_p + 0.5) % 1.0 - 0.5
    half_mask  = 0.6 * best_dur / best_p
    in_transit = np.abs(phase_c) < half_mask
    out_transit = ~in_transit

    if out_transit.sum() < 100:
        # Too few out-of-transit cadences to re-fit — return single-pass result.
        return rough_residual

    # --- Pass 2: SG on out-of-transit cadences only, interpolated over gaps ---
    cadence    = np.nanmedian(np.diff(time))
    if not np.isfinite(cadence) or cadence <= 0:
        # Degenerate time axis (< 2 cadences or all identical timestamps)
        return (flux - np.median(flux)) / max(np.std(flux), 1e-10)
    window_pts = int(DETREND_WINDOW_DAYS / cadence)
    if window_pts % 2 == 0:
        window_pts += 1
    window_pts = max(window_pts, 5)

    # Run SG on out-of-transit time/flux; evaluate at all cadences via interp.
    t_oot  = time[out_transit]
    f_oot  = flux[out_transit]

    gap_idx    = np.where(np.diff(t_oot) > 2 * cadence)[0] + 1
    boundaries = np.concatenate([[0], gap_idx, [len(t_oot)]])
    trend_oot  = np.empty_like(f_oot)

    for i in range(len(boundaries) - 1):
        sl  = slice(boundaries[i], boundaries[i + 1])
        seg = f_oot[sl]
        if len(seg) == 0:
            continue
        w   = min(window_pts, len(seg))
        if w % 2 == 0:
            w -= 1
        # M6: savgol_filter requires window_length < x.size; clamp to avoid crash.
        if w >= len(seg):
            w = len(seg) - 1
        if w % 2 == 0:
            w -= 1
        w = max(w, 1)  # ensure w stays positive after odd-correction on tiny segments
        # w < 5 covers w < 1: both cases use a flat median trend.
        if w < 5:
            # Segment too short for SG filter (also handles w==0/negative) — use median.
            trend_oot[sl] = np.full(len(seg), np.median(seg))
        else:
            trend_oot[sl] = savgol_filter(seg, window_length=w, polyorder=2)

    # Interpolate the OOT trend back to all cadences (in-transit included).
    trend_all = np.interp(time, t_oot, trend_oot)
    trend_all = np.where(np.abs(trend_all) < 1e-10, 1e-10, trend_all)

    return (flux - trend_all) / trend_all



def _physical_diagnostics(
    time: np.ndarray, rel: np.ndarray, period: float, t0: float, duration: float,
) -> dict:
    """Vetting diagnostics for people, measured on detrended relative flux.

    The model's secondary/odd-even inputs come from clipped, z-scored flux (and
    its secondary view is centred on the primary, see _compute_secondary_view);
    these are separate, physical measurements for reviewers:
      secondary_frac   depth half an orbit after the transit (circular orbit)
      odd_even_frac    |depth of odd transits - depth of even transits|
      transit_view / secondary_view   201-bin relative-flux curves around
                       phase 0 and phase 0.5 on the same scale (for plots)
    """
    out = {"secondary_frac": 0.0, "odd_even_frac": 0.0,
           "transit_view": np.zeros(N_LOCAL_BINS, dtype=np.float32),
           "secondary_view": np.zeros(N_LOCAL_BINS, dtype=np.float32)}
    if period <= 0 or duration <= 0:
        return out
    dt = ((time - t0 + 0.5 * period) % period) - 0.5 * period          # days from transit
    ds = ((time - t0) % period) - 0.5 * period                          # days from phase 0.5
    outside = (np.abs(dt) > duration) & (np.abs(ds) > duration)
    if outside.sum() < 10:
        return out
    base = float(np.median(rel[outside]))
    sec = np.abs(ds) < 0.5 * duration
    if sec.sum() >= 3:
        out["secondary_frac"] = base - float(np.mean(rel[sec]))
    n = np.floor((time - t0) / period + 0.5).astype(int)               # transit number
    inside = np.abs(dt) < 0.5 * duration
    odd, even = inside & (n % 2 == 1), inside & (n % 2 == 0)
    if odd.sum() >= 3 and even.sum() >= 3:
        out["odd_even_frac"] = abs(float(np.mean(rel[odd])) - float(np.mean(rel[even])))
    w = min(2.0 * duration, 0.4 * period)
    out["transit_view"] = _bin_phase_raw(dt, rel - base, N_LOCAL_BINS, -w, w)
    out["secondary_view"] = _bin_phase_raw(ds, rel - base, N_LOCAL_BINS, -w, w)
    return out


def _physical_depth(
    time: np.ndarray, rel: np.ndarray, period: float, t0: float, duration: float,
) -> tuple[float, float]:
    """(fractional depth, transit SNR) measured on detrended relative flux.

    BLS runs on z-scored, +-5 sigma clipped flux, so its depth is neither
    physical nor reliable for deep transits; this measures directly:
    depth = out-of-transit median - in-transit mean, noise = 1.4826 * MAD of
    out-of-transit points, SNR = depth / noise * sqrt(n in-transit points).
    """
    dt = ((time - t0 + 0.5 * period) % period) - 0.5 * period   # days from mid-transit
    inside = np.abs(dt) < 0.5 * duration
    outside = np.abs(dt) > duration          # skip ingress/egress margins
    if inside.sum() < 3 or outside.sum() < 10:
        return 0.0, 0.0
    base = float(np.median(rel[outside]))
    depth = base - float(np.mean(rel[inside]))
    noise = 1.4826 * float(np.median(np.abs(rel[outside] - base)))
    snr = depth / noise * float(np.sqrt(inside.sum())) if noise > 0 else 0.0
    return depth, snr

def _normalise(flux: np.ndarray) -> np.ndarray:
    # CS3: use float64 intermediates then cast back to float32 to avoid silent
    # upcasting at callsites.
    mu, sigma = np.mean(flux), np.std(flux)
    if sigma < 1e-10:
        return np.zeros_like(flux, dtype=np.float32)
    # Clip to ±5 σ so shallow/narrow transits in mostly-flat binned arrays do not
    # produce extreme activations when the underlying std is very small.
    return np.clip((flux - mu) / sigma, -5.0, 5.0).astype(np.float32)


def _duration_grid_for_period(period_days: float, n_cad_min: float = 0.02) -> np.ndarray:
    """Return duration grid in days scaled to the given period.

    Based on the transit duration scaling T ~ P^(1/3) for solar-type stars.
    Range: from minimum detectable (2 cadences) to maximum physical (25% of period).
    """
    dur_min = max(n_cad_min, 0.02)
    dur_max = min(0.25 * period_days, 1.5)   # cap at 1.5 days
    if dur_max <= dur_min:
        return np.array([dur_min], dtype=np.float64)
    return np.geomspace(dur_min, dur_max, 7)


def _is_harmonic_of(p: float, used: list[float], tol: float = 0.05) -> bool:
    """True if *p* is within *tol* of any used period or its 1/3..3x harmonics."""
    for up in used:
        for cand in (up, up / 2, up * 2, up / 3, up * 3):
            if abs(p - cand) / cand < tol:
                return True
    return False


def _best_harmonic(
    bls: BoxLeastSquares,
    p0: float,
    durations: np.ndarray,
    max_period: float,
    power0: float,
    duration0: float,
    t00: float,
) -> tuple[float, float, float, float]:
    """
    Pick the best-fitting member of a peak's harmonic family.

    Folding at P/n aligns every transit just as well as folding at P, so BLS
    power at sub-harmonics is nearly identical to power at the truth and the
    raw argmax picks an alias most of the time.  Directly comparing the family
    on the same footing resolves it — measured 15% -> 65% true-period recovery
    on real confirmed planets.  Costs one power() call over ~7 trial periods.
    """
    cands = sorted({
        p0 * m for m in (1.0, 2.0, 3.0, 4.0, 0.5, 1.0 / 3.0, 0.25)
        if MIN_PERIOD_DAYS < p0 * m < max_period
    })
    if len(cands) < 2:
        return p0, duration0, t00, power0
    res = bls.power(np.array(cands), durations, objective="snr")
    i = int(np.argmax(res.power))
    return (float(res.period[i]), float(res.duration[i]),
            float(res.transit_time[i]), float(res.power[i]))


def _refine_bls_peak(
    bls: BoxLeastSquares,
    p0: float,
    durations: np.ndarray,
    coarse_log_step: float,
    baseline: float,
) -> tuple[float, float, float, float]:
    """
    Re-search a narrow window around a coarse BLS peak at the resolution the
    baseline actually demands.  Returns (period, duration, transit_time, power).

    The coarse grid (5000 log-spaced points) has dP/P ~ 1.5e-3, but folded
    transits only stay aligned across the baseline while

        dP/P < duration / baseline        (~9e-5 for Kepler + a 3h transit)

    i.e. the coarse grid is ~17x too coarse.  That costs detections outright,
    makes the search lock onto 3:1 harmonics, and — even on a hit — leaves a
    period error that smears the phase-folded transit over 1.3-2.9x its own
    duration, blurring the very views the model is trained on.

    Refining one coarse cell costs ~400 extra evaluations per candidate
    (against 5000 for the coarse scan), so the search stays cheap.
    """
    span = p0 * coarse_log_step
    step = p0 * (float(np.min(durations)) / baseline) * 0.5
    if step <= 0 or not np.isfinite(step):
        return p0, float("nan"), float("nan"), float("-inf")

    grid = np.arange(p0 - span, p0 + span + step, step)
    grid = grid[grid > MIN_PERIOD_DAYS]
    if len(grid) < 2:
        return p0, float("nan"), float("nan"), float("-inf")

    res = bls.power(grid, durations, objective="snr")
    i = int(np.argmax(res.power))
    return (float(res.period[i]), float(res.duration[i]),
            float(res.transit_time[i]), float(res.power[i]))


def _bls_search(
    time: np.ndarray,
    flux: np.ndarray,
    n_candidates: int,
    refine: bool = True,
    probe_harmonics: bool = True,
    max_period_fraction: float = 1.0 / 3.0,
    reject_harmonics: bool = False,
) -> list[dict]:
    """
    Run Box Least Squares and return up to n_candidates period candidates.

    BLS has near-equal power at a transit's true period and at its integer
    sub-harmonics (folding at P/n still aligns every transit), so the raw
    argmax lands on an alias far more often than on the truth.  Measured on 20
    confirmed planets, the old behaviour recovered the *true* period for only
    3/20 (15%) — matching the 15.6% seen across the whole KOI block.

    Three changes take that to 16/20 (80%):

    * ``probe_harmonics`` — evaluate the peak's harmonic family (P, 2P, 3P, 4P,
      P/2, P/3, P/4) and keep whichever genuinely fits best, instead of trusting
      the argmax.  Biggest single win: 15% -> 65%.
    * ``reject_harmonics=False`` — the old code *discarded the true period* as a
      "duplicate" whenever a sub-harmonic had already been accepted, throwing
      away the right answer.  Now only near-identical periods are de-duplicated.
    * ``max_period_fraction=1/3`` — the old baseline/2 ceiling let the search
      latch onto long-period systematics (375-702 d detections).  A third of the
      baseline still guarantees >=3 transits, which any credible detection needs.

    ``refine`` then polishes the winner on a fine local grid; it does not affect
    which signal is found, but removes period error that would otherwise smear
    the phase-folded views by 1.3-2.9x the transit duration.

    Pass ``refine=False, probe_harmonics=False, max_period_fraction=0.5,
    reject_harmonics=True`` to reproduce the original behaviour for A/B.
    """
    baseline   = time[-1] - time[0]
    max_period = baseline * max_period_fraction

    if max_period <= MIN_PERIOD_DAYS:
        return []

    # Log-uniform period grid gives equal resolution at short and long periods
    periods   = np.exp(np.linspace(np.log(MIN_PERIOD_DAYS), np.log(max_period), 5000))
    # G2: period-relative duration grid using the median period being searched.
    # Previously a fixed grid [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0] days
    # was used; for periods > 20 days even 1.0 day can be too short.
    # The period-relative grid scales with the median search period so long-period
    # planets are not missed.
    # Use the 10th-percentile period so the duration grid covers short-period
    # candidates where the median-period grid over-estimates transit duration.
    durations = _duration_grid_for_period(float(np.percentile(periods, 10)))

    bls    = BoxLeastSquares(time, flux)
    result = bls.power(periods, durations, objective="snr")

    coarse_log_step = float(np.log(max_period / MIN_PERIOD_DAYS) / len(periods))

    candidates   = []
    used_periods = []

    def _claimed(period: float) -> bool:
        """Near-identical to something already emitted (not merely a harmonic)."""
        return any(abs(period - u) / u < 0.02 for u in used_periods)

    for idx in np.argsort(result.power)[::-1]:
        p = float(result.period[idx])

        if reject_harmonics:
            if _is_harmonic_of(p, used_periods):
                continue
        elif _claimed(p):
            continue

        duration = float(result.duration[idx])
        t0       = float(result.transit_time[idx])
        power    = float(result.power[idx])

        # Probe BEFORE refining: refining first would polish the wrong
        # harmonic, and the probe would then jump back to a coarse value.
        if probe_harmonics:
            p, duration, t0, power = _best_harmonic(
                bls, p, durations, max_period, power, duration, t0
            )

        if refine:
            r_p, r_dur, r_t0, r_pow = _refine_bls_peak(
                bls, p, durations, coarse_log_step, baseline
            )
            if np.isfinite(r_pow) and r_pow >= power:
                p, duration, t0, power = r_p, r_dur, r_t0, r_pow

        # Probing/refining can land on a signal already emitted.
        if _claimed(p):
            continue

        stats = bls.compute_stats(p, duration, t0)
        candidates.append({
            "period":   p,
            "t0":       t0,
            "duration": duration,
            "depth":    float(stats["depth"][0]),
            "power":    power,
        })
        used_periods.append(p)

        if len(candidates) >= n_candidates:
            break

    return candidates


def _bin_phase_core(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    CF11: shared binning kernel used by both _bin_phase and _bin_phase_raw.

    Bins a phase-folded light curve into n_bins evenly-spaced phase bins.
    Empty bins are filled by linear interpolation.  Returns a raw float32
    array (NOT z-scored).

    Issue 5.1: vectorised with np.digitize — iterates only over populated
    bins (typically much fewer than n_bins) instead of looping over all bins.
    """
    edges   = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    bin_idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    binned  = np.full(n_bins, np.nan)

    # L2: mask.any() is always True since j comes from np.unique(bin_idx);
    # the guard is dead code and has been removed.
    for j in np.unique(bin_idx):
        mask = bin_idx == j
        binned[j] = np.median(flux[mask])

    valid = np.isfinite(binned)
    if valid.sum() >= 2:
        binned = np.interp(centres, centres[valid], binned[valid])
    else:
        binned = np.zeros(n_bins)

    return binned.astype(np.float32)


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

    CF11: delegates shared binning to _bin_phase_core, then applies _normalise.
    """
    return _normalise(_bin_phase_core(phase, flux, n_bins, lo, hi))


def _bin_phase_raw(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Bin a phase-folded light curve into n_bins evenly-spaced phase bins.
    Empty bins are filled by linear interpolation. Result is NOT normalised
    (returns raw median-per-bin values).

    Note on naming (Issue 4.5): this function is called "raw" because the
    result is not z-scored per view.  However the input *flux* has already
    been through the per-light-curve z-score (_normalise) before this call,
    so the output is "prenorm_global_view" — after light-curve z-score but
    before the per-view z-score applied in _bin_phase.

    CF11: delegates shared binning to _bin_phase_core (no normalisation step).
    """
    return _bin_phase_core(phase, flux, n_bins, lo, hi)


def _secondary_depth(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    period: float,
) -> float:
    """
    Measure the flux depth near the secondary-eclipse position.

    Returns the depth as a positive fractional value (dip = positive).

    Assumption (Issue 4.2): the secondary eclipse is assumed to be near
    phase 0.5 (circular orbit).  For eccentric orbits the secondary may
    appear at a different phase; to partially mitigate this the search is
    expanded to the full phase range [0.3, 0.7] and the maximum depth
    found anywhere in that window is returned.  This catches many eccentric
    EBs whose secondary eclipse falls away from exactly 0.5.

    Note: _compute_secondary_view and _centroid_shift also assume circular
    orbits (secondary centred at phase 0.5) — see their docstrings.
    """
    if period <= 0:
        return 0.0
    half_window = min(2.0 * duration / period, 0.1)
    # Search the full [0.3, 0.7] phase band (|phase| in [0.3, 0.7] with
    # wrapping so both +0.5 and −0.5 sides are covered).
    in_secondary_band = (np.abs(phase) >= 0.3) & (np.abs(phase) <= 0.7)
    if in_secondary_band.sum() < 3:
        return 0.0

    # Slide a window of width 2×half_window across [0.3, 0.7] and return
    # the maximum depth (most negative flux) found in any window.
    # This identifies the deepest secondary regardless of its exact phase.
    phase_band = np.abs(phase[in_secondary_band])
    flux_band  = flux[in_secondary_band]
    step       = max(half_window / 2.0, 0.01)
    # L3: phase data only extends to |phase| = 0.5; clamp centres to 0.51 so
    # windows centred at c > 0.5 (which find no data) are never generated.
    centers    = np.arange(0.3, 0.51, step)
    best_depth = 0.0
    for c in centers:
        in_win = np.abs(phase_band - c) < half_window
        if in_win.sum() >= 3:
            d = float(-np.median(flux_band[in_win]))
            if d > best_depth:
                best_depth = d
    return best_depth


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
    if period <= 0:
        return 0.0
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


def _compute_odd_even_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fold at 2× the period and bin odd and even transits into separate 201-pt views.

    Odd transits appear near phase 0; even transits appear near phase 0.5 in
    the doubled-period phase space.  Each is binned over ±2× transit duration.

    Returns (odd_view, even_view, raw_odd_view, raw_even_view) where raw views
    are NOT per-view z-scored (channel 1 of the 2-channel branch inputs).
    """
    _zeros = np.zeros(N_LOCAL_BINS, dtype=np.float32)
    if period <= 0:
        return _zeros, _zeros, _zeros, _zeros
    double_period = 2.0 * period
    phase2 = ((time - t0) / double_period + 0.5) % 1.0 - 0.5
    half_window = min(2 * duration / double_period, 0.4)

    # Issue 4.1: when duration/period is large (>~0.3) the odd and even windows
    # overlap in the 2× folded phase space, making odd/even discrimination
    # unreliable.  Fall back to zero arrays in that case.
    if half_window * 2 > 0.4:
        # Windows would overlap — odd/even discrimination is unreliable at this
        # duration/period ratio.  Return zero arrays so the model sees no signal
        # rather than misleading blended odd+even depth information.
        return _zeros, _zeros, _zeros, _zeros

    odd_mask = np.abs(phase2) < (half_window + 0.05)
    if odd_mask.sum() >= 3:
        raw_odd_view = _bin_phase_raw(phase2[odd_mask], flux[odd_mask], N_LOCAL_BINS, -half_window, half_window)
        odd_view = _normalise(raw_odd_view)
    else:
        odd_view = raw_odd_view = _zeros

    # Even transits: phase2 near ±0.5 — shift so they centre at 0
    # Shift phase2 so the even transit (at phase2=0.5, i.e. phase=+0.5 from odd)
    # is centred at 0. Use -0.25 as the cut to avoid pulling odd-transit cadences
    # (near phase2=0) into the even window.
    phase2_even = np.where(phase2 < -0.25, phase2 + 1.0, phase2) - 0.5
    even_mask = np.abs(phase2_even) < (half_window + 0.05)
    if even_mask.sum() >= 3:
        raw_even_view = _bin_phase_raw(phase2_even[even_mask], flux[even_mask], N_LOCAL_BINS, -half_window, half_window)
        even_view = _normalise(raw_even_view)
    else:
        even_view = raw_even_view = _zeros

    return odd_view, even_view, raw_odd_view, raw_even_view


def _compute_secondary_view(
    phase: np.ndarray,
    flux: np.ndarray,
    duration: float,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 201-pt view centred on the secondary eclipse position (phase 0.5).

    The phase array is shifted by −0.5 so the secondary sits at 0, then
    binned with the same local window used for the primary transit.

    Returns (secondary_view, raw_secondary_view) where raw_secondary_view is
    NOT per-view z-scored (channel 1 of the 2-channel secondary branch input).

    Assumption (Issue 4.2): the secondary eclipse is assumed to be at phase 0.5
    (circular orbit).  For eccentric orbits the secondary will appear at a
    different phase and this view will be centred on the wrong location.
    The scalar _secondary_depth feature uses a wider search window [0.3, 0.7]
    to partially compensate, but this view always centres at phase 0.5.
    """
    _zeros = np.zeros(N_LOCAL_BINS, dtype=np.float32)
    if period <= 0:
        return _zeros, _zeros
    # KNOWN BUG (found by review 2026-10-10): this maps phase 0 (the primary
    # transit) to 0, so the 'secondary' view repeats the primary. Trained
    # models expect it, so it stays until the next cache rebuild + retrain;
    # the correct shift is ((phase + 1.0) % 1.0) - 0.5. Reviewer-facing
    # diagnostics use _physical_diagnostics instead.
    phase_shifted = (phase - 0.5) % 1.0 - 0.5
    half_window = min(2 * duration / period, 0.4)
    raw_secondary_view = _bin_phase_raw(phase_shifted, flux, N_LOCAL_BINS, -half_window, half_window)
    secondary_view = _normalise(raw_secondary_view)
    return secondary_view, raw_secondary_view


def _centroid_shift(
    time: np.ndarray,
    centr1: np.ndarray,
    centr2: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """
    Euclidean centroid displacement between in-transit and out-of-transit cadences.

    Returns 0.0 if centroid arrays are fully NaN or insufficient data exists.
    A non-zero value indicates the photometric signal originates off-target
    (background eclipsing binary).

    Assumption (Issue 4.2): in-transit cadences are identified using the
    primary transit phase (near phase 0).  For eccentric orbits the secondary
    eclipse may also produce a centroid shift that is not captured here.
    """
    if np.all(~np.isfinite(centr1)) and np.all(~np.isfinite(centr2)):
        return 0.0
    if period <= 0:
        return 0.0

    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    half_dur = (duration / 2.0) / period
    in_transit  = np.abs(phase) < half_dur
    out_transit = (~in_transit) & (np.abs(phase) > 2.0 * half_dur)

    if in_transit.sum() < 3 or out_transit.sum() < 3:
        return 0.0

    def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
        vals = arr[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) > 0 else np.nan

    c1_in, c1_out = _safe_mean(centr1, in_transit), _safe_mean(centr1, out_transit)
    c2_in, c2_out = _safe_mean(centr2, in_transit), _safe_mean(centr2, out_transit)

    if not all(np.isfinite([c1_in, c1_out, c2_in, c2_out])):
        return 0.0
    return float(np.sqrt((c1_in - c1_out) ** 2 + (c2_in - c2_out) ** 2))


def _count_transits(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """Count distinct observed transit windows by detecting contiguous in-transit runs."""
    if period <= 0:
        return 0.0
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    in_transit = np.abs(phase) < (duration / 2.0) / period
    if not in_transit.any():
        return 0.0
    changes = np.diff(in_transit.astype(np.int8))
    n = int((changes == 1).sum())
    if in_transit[0]:
        n += 1
    return float(n)


def _compute_centroid_curve(
    time: np.ndarray,
    centr1: np.ndarray,   # always np.ndarray; callers pass np.full_like(time, np.nan) when absent
    centr2: np.ndarray,   # always np.ndarray; callers pass np.full_like(time, np.nan) when absent
    period: float,
    t0: float,
    n_bins: int = 201,
) -> np.ndarray:
    """
    Phase-fold the centroid displacement magnitude and bin into n_bins.

    Returns a 1-D float32 array of shape (n_bins,) representing the
    median centroid displacement (sqrt(centr1² + centr2²)) per phase bin,
    zero-padded for empty bins.  The result is zero-mean, unit-variance
    normalised if std > 0, otherwise returned as-is.

    C7: the centr1/centr2 parameters are always np.ndarray (never None).
    Callers that have no centroid data pass np.full_like(time, np.nan).
    NaN centroid values are handled by the np.isfinite mask below.
    The previous None guard has been removed as it was dead code.
    """
    # Fix 7: guard against invalid period to prevent division by zero in phase fold.
    if period <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    if len(centr1) < 10:
        return np.zeros(n_bins, dtype=np.float32)
    # Remove NaNs
    mask = np.isfinite(centr1) & np.isfinite(centr2)
    if mask.sum() < 10:
        return np.zeros(n_bins, dtype=np.float32)
    t_c = time[mask]
    c1  = centr1[mask]
    c2  = centr2[mask]
    # Median-subtract each axis to remove systematic offset
    c1 = c1 - np.nanmedian(c1)
    c2 = c2 - np.nanmedian(c2)
    mag = np.sqrt(c1**2 + c2**2)
    # Phase fold — L4: use the standard vectorised formula (no in-place mutation)
    # consistent with the rest of the codebase.
    phase = ((t_c - t0) / period + 0.5) % 1.0 - 0.5
    # Issue 5.2: vectorised binning with np.digitize — only iterates over
    # populated bins rather than all n_bins.
    # Issue 1.8: use a dedicated `populated` boolean array to track which bins
    # were genuinely filled.  Using `result != 0` incorrectly treated bins with
    # a true near-zero centroid displacement as empty and overwrote them.
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx   = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    result    = np.zeros(n_bins, dtype=np.float32)
    populated = np.zeros(n_bins, dtype=bool)
    for j in np.unique(bin_idx):
        mask_b = bin_idx == j
        if mask_b.any():
            result[j]    = float(np.median(mag[mask_b]))
            populated[j] = True
    # Interpolate truly empty bins using the populated mask.
    if populated.sum() > 2:
        x = np.arange(n_bins)
        result = np.interp(x, x[populated], result[populated]).astype(np.float32)
    # If fewer than 25 % of bins are genuinely populated the curve is mostly
    # interpolated noise — return zeros rather than amplifying it to unit variance.
    if populated.sum() < n_bins // 4:
        return np.zeros(n_bins, dtype=np.float32)
    # Normalise
    std = result.std()
    if std > 1e-8:
        result = ((result - result.mean()) / std).astype(np.float32)
    return result


def _compute_noise_floor(flux: np.ndarray, flux_err: np.ndarray) -> float:
    """Estimate per-star photometric noise as median |flux_err| / median |flux|.

    L9: flux_err is always a valid ndarray (callers default to np.ones_like(flux)
    when no error column is present), so the None guard has been removed.
    """
    if len(flux_err) > 0 and np.any(np.isfinite(flux_err)):
        err = float(np.nanmedian(np.abs(flux_err)))
        med = float(np.nanmedian(np.abs(flux)))
        if med > 0:
            return float(np.clip(err / med, 1e-4, 0.05))
    return 0.002   # default: ~200 ppm (bright Kepler star)


def _fold_and_bin(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-fold the light curve and produce both views expected by the model,
    plus the raw (pre-normalisation) global view.

    global_view     : 2001 bins, phase [-0.5, 0.5] — the full orbit (normalised)
    local_view      :  201 bins, phase ±2× transit duration — transit detail (normalised)
    raw_global_view : 2001 bins, phase [-0.5, 0.5] — the full orbit (NOT normalised)
    raw_local_view  :  201 bins, phase ±2× transit duration — transit detail (NOT normalised)
    """
    # Phase in [-0.5, 0.5], transit centred at 0
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

    # Two-channel global view construction — dynamic range note (Fix 3):
    #
    # Channel 0 (global_view): double z-scored — first across the full LC
    #   (via _normalise before _fold_and_bin), then across the 2001 phase bins
    #   (via _normalise inside _bin_phase / _normalise(raw_global_view) here).
    # Channel 1 (raw_global_view / prenorm): single z-scored — only across the
    #   full LC.  The per-view z-score is NOT applied to this channel.
    #
    # The two channels therefore have different dynamic ranges.  GlobalBranch
    # BatchNorm layers normalise them independently per channel, so this
    # asymmetry is handled correctly at training and inference time.
    # See also: TransitCandidate.raw_global_view docstring.

    # Raw (un-normalised) global view — median per bin, interpolated but not z-scored
    raw_global_view = _bin_phase_raw(phase, flux, N_GLOBAL_BINS, -0.5, 0.5)

    # Normalised global view
    global_view = _normalise(raw_global_view)

    # Local window: ±2× transit duration, capped at ±0.4 to avoid wrapping.
    # Compute raw (prenorm) local view first, then normalise for channel 0.
    # Reusing the raw binning avoids running _bin_phase_core twice.
    half_window = min(2 * duration / period, 0.4)
    raw_local_view = _bin_phase_raw(phase, flux, N_LOCAL_BINS, -half_window, half_window)
    local_view = _normalise(raw_local_view)

    return global_view, local_view, raw_global_view, raw_local_view


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_multi(
    fits_paths: list[str | Path],
    n_candidates: int = 5,
) -> list[TransitCandidate]:
    """
    Run the preprocessing pipeline on multiple FITS files for the same star.

    All files are loaded, concatenated, and sorted by time before BLS search.
    This is critical for Kepler targets with multiple quarterly files: stitching
    all quarters together extends the baseline from ~90 days to ~4 years,
    dramatically improving sensitivity to long-period planets.

    Parameters
    ----------
    fits_paths :
        List of paths to FITS light curve files from the same star (e.g. all
        Kepler quarters for one KOI).  A single-element list is equivalent to
        calling ``preprocess`` directly.
    n_candidates :
        Maximum number of period candidates to return.
    """
    if not fits_paths:
        return []
    if len(fits_paths) == 1:
        return preprocess(fits_paths[0], n_candidates=n_candidates)

    # Fix 10: suppress only FITS/astropy header warnings during file loading.
    # Each _load_fits call is wrapped individually; preprocessing runs outside
    # the suppression context so our own warnings surface to operators.
    all_time:   list[np.ndarray] = []
    all_flux:   list[np.ndarray] = []
    all_err:    list[np.ndarray] = []
    all_centr1: list[np.ndarray] = []
    all_centr2: list[np.ndarray] = []

    for path in fits_paths:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, f, e, c1, c2 = _load_fits(path)
            t, f, e, c1, c2 = _drop_nans(t, f, e, c1, c2)
            if len(t) >= 20:          # skip tiny fragments
                all_time.append(t)
                all_flux.append(f)
                all_err.append(e)
                all_centr1.append(c1)
                all_centr2.append(c2)
        except Exception as exc:
            # L14: log the exception so operators can diagnose per-file failures.
            warnings.warn(f"Skipping {Path(path).name}: {type(exc).__name__}: {exc}")
            continue

    if not all_time:
        return []

    time   = np.concatenate(all_time)
    flux   = np.concatenate(all_flux)
    err    = np.concatenate(all_err)
    centr1 = np.concatenate(all_centr1)
    centr2 = np.concatenate(all_centr2)

    order = np.argsort(time)
    time, flux, err = time[order], flux[order], err[order]
    centr1, centr2  = centr1[order], centr2[order]

    if len(time) < 100:
        return []

    time, flux, err, centr1, centr2 = _clip_outliers(time, flux, err, centr1, centr2)

    # Compute noise floor before detrending (use raw flux)
    noise_floor_val = _compute_noise_floor(flux, err)

    # Issue 4.3: use iterative detrending with transit masking as the
    # default; _detrend is kept as a fallback inside _detrend_iterative.
    flux = _detrend_iterative(time, flux)
    # Keep the detrended relative flux unclipped: physical depth and SNR are
    # measured on it (the z-scored copy below is clipped at +-5 sigma).
    rel_flux = flux.copy()
    flux = _normalise(flux)

    candidates = _bls_search(time, flux, n_candidates)
    if not candidates:
        return []

    results = []
    for c in candidates:
        period, t0, duration = c["period"], c["t0"], c["duration"]
        # Issue 5.3: compute phase once and pass it to helpers that need it.
        phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

        global_view, local_view, raw_global_view, raw_local_view = _fold_and_bin(time, flux, period, t0, duration)
        sec_depth                                  = _secondary_depth(phase, flux, duration, period)
        oe_diff                                    = _odd_even_diff(time, flux, period, t0, duration)
        odd_view, even_view, raw_odd_view, raw_even_view = _compute_odd_even_views(time, flux, period, t0, duration)
        secondary_view, raw_secondary_view         = _compute_secondary_view(phase, flux, duration, period)
        cshift                                     = _centroid_shift(time, centr1, centr2, period, t0, duration)
        n_tr                                       = _count_transits(time, period, t0, duration)
        phys                                       = _physical_depth(time, rel_flux, period, t0, duration)
        diag                                       = _physical_diagnostics(time, rel_flux, period, t0, duration)
        centroid_curve                             = _compute_centroid_curve(time, centr1, centr2, period, t0)

        results.append(TransitCandidate(
            period             = period,
            t0                 = t0,
            duration           = duration,
            depth              = c["depth"],
            depth_frac         = phys[0],
            transit_snr        = phys[1],
            secondary_frac     = diag["secondary_frac"],
            odd_even_frac      = diag["odd_even_frac"],
            transit_view_rel   = diag["transit_view"],
            secondary_view_rel = diag["secondary_view"],
            bls_power          = c["power"],
            global_view        = global_view,
            local_view         = local_view,
            secondary_depth    = sec_depth,
            odd_even_diff      = oe_diff,
            odd_view           = odd_view,
            even_view          = even_view,
            secondary_view     = secondary_view,
            centroid_shift     = cshift,
            n_transits         = n_tr,
            raw_global_view    = raw_global_view,
            centroid_curve     = centroid_curve,
            noise_floor        = noise_floor_val,
            raw_local_view     = raw_local_view,
            raw_odd_view       = raw_odd_view,
            raw_even_view      = raw_even_view,
            raw_secondary_view = raw_secondary_view,
        ))

    return results


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
    # Fix 10: suppress only FITS/astropy header warnings during file loading.
    # Preprocessing runs outside the suppression context so our own warnings
    # (e.g. the iterative detrend BLS fallback warning) surface to operators.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        time, flux, err, centr1, centr2 = _load_fits(fits_path)

    time, flux, err, centr1, centr2 = _drop_nans(time, flux, err, centr1, centr2)

    if len(time) < 100:
        return []  # too few cadences to search reliably

    time, flux, err, centr1, centr2 = _clip_outliers(time, flux, err, centr1, centr2)

    # Compute noise floor before detrending (use raw flux)
    noise_floor_val = _compute_noise_floor(flux, err)

    # Issue 4.3: use iterative detrending with transit masking as the
    # default; _detrend is kept as a fallback inside _detrend_iterative.
    flux = _detrend_iterative(time, flux)
    # Keep the detrended relative flux unclipped: physical depth and SNR are
    # measured on it (the z-scored copy below is clipped at +-5 sigma).
    rel_flux = flux.copy()
    flux = _normalise(flux)

    candidates = _bls_search(time, flux, n_candidates)
    if not candidates:
        return []

    results = []
    for c in candidates:
        period, t0, duration = c["period"], c["t0"], c["duration"]
        # Issue 5.3: compute phase once and pass it to helpers that need it.
        phase = ((time - t0) / period + 0.5) % 1.0 - 0.5

        global_view, local_view, raw_global_view, raw_local_view = _fold_and_bin(time, flux, period, t0, duration)
        sec_depth                                  = _secondary_depth(phase, flux, duration, period)
        oe_diff                                    = _odd_even_diff(time, flux, period, t0, duration)
        odd_view, even_view, raw_odd_view, raw_even_view = _compute_odd_even_views(time, flux, period, t0, duration)
        secondary_view, raw_secondary_view         = _compute_secondary_view(phase, flux, duration, period)
        cshift                                     = _centroid_shift(time, centr1, centr2, period, t0, duration)
        n_tr                                       = _count_transits(time, period, t0, duration)
        phys                                       = _physical_depth(time, rel_flux, period, t0, duration)
        diag                                       = _physical_diagnostics(time, rel_flux, period, t0, duration)
        centroid_curve                             = _compute_centroid_curve(time, centr1, centr2, period, t0)

        results.append(TransitCandidate(
            period             = period,
            t0                 = t0,
            duration           = duration,
            depth              = c["depth"],
            depth_frac         = phys[0],
            transit_snr        = phys[1],
            secondary_frac     = diag["secondary_frac"],
            odd_even_frac      = diag["odd_even_frac"],
            transit_view_rel   = diag["transit_view"],
            secondary_view_rel = diag["secondary_view"],
            bls_power          = c["power"],
            global_view        = global_view,
            local_view         = local_view,
            secondary_depth    = sec_depth,
            odd_even_diff      = oe_diff,
            odd_view           = odd_view,
            even_view          = even_view,
            secondary_view     = secondary_view,
            centroid_shift     = cshift,
            n_transits         = n_tr,
            raw_global_view    = raw_global_view,
            centroid_curve     = centroid_curve,
            noise_floor        = noise_floor_val,
            raw_local_view     = raw_local_view,
            raw_odd_view       = raw_odd_view,
            raw_even_view      = raw_even_view,
            raw_secondary_view = raw_secondary_view,
        ))

    return results
