"""
Test whether the BLS period grid is too coarse to land on real transit periods.

Theory: for the folded transits to stay aligned across the whole baseline, the
trial period must satisfy

    dP/P  <  duration / baseline

Kepler's 4-year baseline with a 3-hour transit needs dP/P < ~9e-5. The current
grid is 5000 log-spaced points from 0.5 d to baseline/2, giving

    dP/P = ln(max/min) / 5000  ~ 1.5e-3

which is over an order of magnitude too coarse — so the nearest grid point
usually smears the transits and the peak is lost.

This injects known transits into synthetic Kepler-like light curves and
measures recovery with (a) the current grid and (b) a two-stage coarse->refine
search, isolating grid resolution from every other pipeline variable.

Run:  python -m scripts.bls_grid_experiment
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
from astropy.timeseries import BoxLeastSquares

from ml.preprocess import MIN_PERIOD_DAYS, _duration_grid_for_period

CADENCE = 29.4 / 60 / 24     # Kepler long cadence, days
BASELINE = 1460.0            # 4 years
DEPTH = 0.0008               # 800 ppm
DUR_HOURS = 3.0
NOISE = 0.00025              # per-cadence scatter
SEED = 42


def make_lightcurve(period: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, BASELINE, CADENCE)
    keep = rng.random(len(t)) < 0.92          # realistic duty cycle / gaps
    t = t[keep]
    f = 1.0 + rng.normal(0, NOISE, len(t))
    dur = DUR_HOURS / 24
    t0 = period * 0.37
    phase = np.abs(((t - t0 + period / 2) % period) - period / 2)
    f[phase < dur / 2] -= DEPTH
    return t, f


def current_grid(t: np.ndarray) -> np.ndarray:
    max_p = (t[-1] - t[0]) / 2
    return np.exp(np.linspace(np.log(MIN_PERIOD_DAYS), np.log(max_p), 5000))


def search(t, f, periods, durations) -> tuple[float, float]:
    bls = BoxLeastSquares(t, f)
    r = bls.power(periods, durations, objective="snr")
    i = int(np.argmax(r.power))
    return float(r.period[i]), float(r.power[i])


def two_stage(t, f, durations, n_refine: int = 8) -> tuple[float, float]:
    """Coarse scan, then a fine local grid around each of the top peaks."""
    coarse = current_grid(t)
    bls = BoxLeastSquares(t, f)
    r = bls.power(coarse, durations, objective="snr")
    best_p, best_pow = None, -np.inf
    order = np.argsort(r.power)[::-1]
    seen: list[float] = []
    for idx in order:
        p = float(r.period[idx])
        if any(abs(p - s) / s < 0.05 for s in seen):
            continue
        seen.append(p)
        # Refine: span one coarse cell, sampled at the resolution the
        # baseline actually demands.
        span = p * 1.5 * (np.log(coarse[-1] / coarse[0]) / len(coarse))
        step = p * (min(durations) / (t[-1] - t[0])) * 0.5
        fine = np.arange(p - span, p + span, max(step, 1e-6))
        fine = fine[fine > MIN_PERIOD_DAYS]
        if len(fine) < 2:
            continue
        pf, powf = search(t, f, fine, durations)
        if powf > best_pow:
            best_p, best_pow = pf, powf
        if len(seen) >= n_refine:
            break
    return best_p, best_pow


def main() -> int:
    rng = np.random.default_rng(SEED)
    t_demo = np.arange(0, BASELINE, CADENCE)
    max_p = (t_demo[-1] - t_demo[0]) / 2
    spacing = np.log(max_p / MIN_PERIOD_DAYS) / 5000
    required = (DUR_HOURS / 24) / BASELINE
    print("Grid arithmetic")
    print(f"  current grid dP/P : {spacing:.2e}")
    print(f"  required dP/P     : {required:.2e}  (duration/baseline)")
    print(f"  too coarse by     : {spacing/required:.0f}x\n")

    test_periods = [3.5, 8.2, 15.7, 29.3, 55.1, 110.4]
    # Mirror the pipeline exactly: duration grid keyed to the 10th-percentile
    # period of the search grid (ml/preprocess.py:408).
    durations = _duration_grid_for_period(float(np.percentile(current_grid(t_demo), 10)))
    print(f"duration grid (days): {np.round(durations, 4).tolist()}\n")
    print(f"{'true P':>8}  {'current grid':>24}  {'two-stage refine':>24}")
    print(f"{'(days)':>8}  {'found P':>14}{'hit':>10}  {'found P':>14}{'hit':>10}")
    print("-" * 62)
    cur_hits = ref_hits = 0
    for p_true in test_periods:
        t, f = make_lightcurve(p_true, rng)
        p_cur, _ = search(t, f, current_grid(t), durations)
        p_ref, _ = two_stage(t, f, durations)
        ok_c = abs(p_cur - p_true) / p_true < 0.01
        ok_r = p_ref is not None and abs(p_ref - p_true) / p_true < 0.01
        cur_hits += ok_c
        ref_hits += ok_r
        print(f"{p_true:8.2f}  {p_cur:14.4f}{'YES' if ok_c else 'no':>10}  "
              f"{(p_ref if p_ref else float('nan')):14.4f}{'YES' if ok_r else 'no':>10}")
    n = len(test_periods)
    print("-" * 62)
    print(f"recovery: current grid {cur_hits}/{n}   two-stage {ref_hits}/{n}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
