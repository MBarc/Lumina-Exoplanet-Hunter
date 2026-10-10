"""
Automated veto battery for ExoNet preprocessing cache.

Applies a series of physics-based and statistical checks to the positive-labeled
candidates in a preprocessing cache, demoting likely false positives to the
negative class.  A cleaned cache is written that can be passed directly to
ml.train via --cache-file for retraining.

Veto checks applied (in order)
-------------------------------
1. secondary_eclipse  — secondary depth > 35 % of primary depth, indicating
                         a background eclipsing binary with both eclipses visible
2. odd_even           — alternating transit depth difference > 30 % of primary
                         depth, indicating a grazing eclipsing binary at 2× period
3. duration_period    — transit duration > 20 % of the orbital period, which is
                         physically impossible for a planet (transit geometry)
4. depth_ceiling      — BLS depth > 0.5 (50% fractional flux decrease); planets
                         around main-sequence stars cannot produce such deep transits
5. bls_power_floor    — BLS SNR < 4.0; signal too weak to be reliably real
6. centroid_shift     — centroid displacement during transit exceeds the
                         per-mission threshold (Kepler: 0.5 px, TESS: 0.1 px,
                         K2: 0.3 px), indicating the photometric source is
                         off-target
7. duty_cycle         — transit duration > 10% of the orbital period;
                         physically impossible for a planet around a
                         main-sequence star (eclipsing binary signature)
8. secondary_abs      — secondary eclipse depth exceeds primary depth in
                         absolute units, indicating an eclipsing binary

Checks 1–2 require both the ratio AND an absolute floor to avoid vetoing
candidates where the secondary/odd-even measurement is dominated by noise.

Scalar feature order in the cache (13 columns)
-----------------------------------------------
  index 0 : period_days
  index 1 : duration_days
  index 2 : depth  (BLS depth in normalised σ units)
  index 3 : bls_power
  index 4 : secondary_depth  (depth at phase ±0.5, same σ units)
  index 5 : odd_even_diff    (|mean_odd − mean_even|, same σ units)
  index 6 : centroid_shift   (Euclidean centroid displacement, pixels)
  index 7 : n_transits
  index 8 : log_teff_norm    (log10(Teff) − log10(5778), solar-normalised)
  index 9 : logg             (surface gravity, cm s⁻²)
  index 10: log_radius_norm  (log10(radius / R_sun))
  index 11: feh              (metallicity [Fe/H])
  index 12: kepmag_norm      (normalised Kepler magnitude: (kepmag − 12.0) / 4.0)

Usage
-----
::

    python -m ml.vetting \\
        --cache-file  training_runs/preprocess_cache.npz \\
        --output-dir  training_runs/vetting

The cleaned cache is saved as ``<output-dir>/preprocess_cache_vetted.npz``.
Pass it to ml.train with ``--cache-file`` and ``--cache-only`` to retrain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# C5: module-level positive-label threshold constant.
_POSITIVE_LABEL_THRESHOLD: float = 0.5


# ── Default veto thresholds ───────────────────────────────────────────────────

@dataclass
class VetoThresholds:
    """Tuneable thresholds for each veto check."""

    # Secondary eclipse veto
    secondary_ratio:    float = 0.35   # secondary_depth / depth > this → veto
    secondary_abs:      float = 0.05   # secondary depth absolute floor in σ-units
                                   # (ratio check is the binding constraint for real signals)

    # Odd/even veto
    odd_even_ratio:     float = 0.30   # odd_even_diff / depth > this → veto
    odd_even_abs:       float = 0.05   # odd/even diff absolute floor in σ-units
                                   # (ratio check is the binding constraint for real signals)

    # Duration/period veto
    duration_period_max: float = 0.20  # duration / period > this → veto

    # Depth ceiling veto — BLS depth in σ-units of normalised flux.
    # Issue 6.2: the original threshold of 8.0 was intended as "8 σ in
    # normalised flux" but depth is stored as a fractional flux decrease
    # (e.g. 0.01 = 1%), so 8.0 was never reachable and the veto was dead code.
    # Fix: set to 0.5 (50% fractional depth).  A transit deeper than 50%
    # is physically impossible for a planet around a main-sequence star
    # (even a grazing equal-mass EB only just approaches this limit).
    depth_max:          float = 0.5    # max BLS depth in fractional flux units
                                       # (0.5 = 50% dimming; physically impossible for a genuine planet)

    # BLS power floor
    bls_power_min:      float = 4.0    # bls_power < this → veto

    # Duty-cycle ceiling veto
    duty_cycle_max:     float = 0.10   # duration / period > this → veto (10%)

    # Per-mission centroid shift veto thresholds (pixels)
    # Kepler: 4"/px → tight threshold; TESS: 21"/px → much looser threshold;
    # K2: same plate scale as Kepler but noisier centroids → intermediate threshold.
    # L8: expanded inline comments show arcsec equivalent for each mission.
    centroid_abs_kepler:  float = 0.5   # pixels × 4"/px = 2.0" on sky
    centroid_abs_tess:    float = 0.1   # pixels × 21"/px = 2.1" on sky
    centroid_abs_k2:      float = 0.3   # pixels × 4"/px = 1.2" on sky
    centroid_abs_default: float = 0.3   # fallback for unknown mission


# ── Per-sample veto logic ─────────────────────────────────────────────────────

def _apply_vetos(
    scalars: np.ndarray,
    labels: np.ndarray,
    thresholds: VetoThresholds,
    missions: "np.ndarray[Any, np.dtype[np.str_]] | None" = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Evaluate all veto checks against every positive-labeled sample.

    Parameters
    ----------
    scalars  : (N, ≥6) float32 array — scalar feature matrix (up to 13 columns)
    labels   : (N,)   float32 array of 0.0 / 1.0
    thresholds : VetoThresholds
    missions : (N,) string array of mission names ("kepler"/"tess"/"k2"/"unknown"),
               or None.  If None, all samples use the default centroid threshold.

    Returns
    -------
    vetoed_mask : bool array of shape (N,)  — True where a positive sample
                  was demoted to negative by at least one veto.
    veto_breakdown : dict mapping veto name → bool array (which samples it fired on)
    """
    n = len(labels)
    t = thresholds

    period        = scalars[:, 0]
    duration      = scalars[:, 1]

    # C5: use _POSITIVE_LABEL_THRESHOLD so soft-labeled CANDIDATE KOIs
    # (koi_score between 0.5–1.0) are also subject to vetting — they are the
    # highest-risk mislabeled samples.
    is_positive = labels >= _POSITIVE_LABEL_THRESHOLD

    # Veto: inverted BLS solution (brightening, not dimming) — unphysical
    v_negative_depth = is_positive & (scalars[:, 2] < -1e-4)

    depth         = np.abs(scalars[:, 2])   # use absolute value for ratio checks
    bls_power     = scalars[:, 3]
    sec_depth     = scalars[:, 4]
    odd_even_diff = scalars[:, 5]

    # ── Veto 1: secondary eclipse ─────────────────────────────────────────────
    # Guard depth > 1e-6: when depth==0 the ratio check degenerates to
    # sec_depth > 0, which would veto every candidate with any secondary signal.
    # Guard scalars[:,2] >= -1e-4: exclude already-vetoed negative-depth samples
    # so they are not double-counted in the secondary_eclipse veto breakdown.
    v_secondary = (
        is_positive
        & ~v_negative_depth
        & (depth > 1e-6)
        & (sec_depth > t.secondary_abs)
        & (sec_depth > t.secondary_ratio * depth)
    )

    # ── Veto 2: odd/even depth difference ─────────────────────────────────────
    # Guard depth > 1e-6: when depth==0 the ratio check degenerates to
    # odd_even_diff > 0, which would veto every candidate with any odd/even signal.
    # Guard ~v_negative_depth: exclude already-vetoed samples from odd/even counts.
    v_odd_even = (
        is_positive
        & ~v_negative_depth
        & (depth > 1e-6)
        & (odd_even_diff > t.odd_even_abs)
        & (odd_even_diff > t.odd_even_ratio * depth)
    )

    # ── Veto 3: duration/period ratio ─────────────────────────────────────────
    safe_period = np.where(period > 0, period, 1.0)
    v_duration = (
        is_positive
        & (duration / safe_period > t.duration_period_max)
    )

    # ── Veto 4: depth ceiling ─────────────────────────────────────────────────
    v_depth = (
        is_positive
        & (depth > t.depth_max)
    )

    # ── Veto 5: BLS power floor ───────────────────────────────────────────────
    v_bls = (
        is_positive
        & (bls_power < t.bls_power_min)
    )

    # ── Veto 6: centroid shift — per-mission pixel-scale threshold ────────────
    centroid = scalars[:, 6] if scalars.shape[1] > 6 else np.zeros(n)

    # Build a per-sample centroid threshold based on mission pixel scale.
    if missions is not None:
        centroid_thresh = np.where(
            missions == "kepler", t.centroid_abs_kepler,
            np.where(missions == "tess", t.centroid_abs_tess,
            np.where(missions == "k2",   t.centroid_abs_k2,
                                         t.centroid_abs_default))
        )
    else:
        centroid_thresh = np.full(n, t.centroid_abs_default)

    v_centroid = (
        is_positive
        & (centroid > centroid_thresh)
    )

    # ── Veto 7: duty-cycle ceiling ────────────────────────────────────────────
    # Transit duration > 5% of the orbital period is physically implausible
    # for a planet-star geometry around a main-sequence star.  A genuine planet
    # with P=1 d and R_star=R_sun would have duration ~1.8 h = 7.5% P, so the
    # threshold of 10% is conservative.  Values > 10% strongly indicate eclipsing
    # binaries, ellipsoidal variables, or BLS over-fitting.
    safe_period = np.where(period > 0, period, np.inf)
    duty_cycle = duration / safe_period
    v_duty_cycle = (
        is_positive
        & (duty_cycle > t.duty_cycle_max)
        & ~v_negative_depth
    )

    # ── Veto 8: secondary eclipse absolute depth floor ────────────────────────
    # If the secondary eclipse is deeper than the primary in absolute units,
    # this is almost certainly an eclipsing binary (even if the ratio is low
    # because the primary depth is also large).
    secondary_depth = scalars[:, 4]
    depth_safe = np.where(depth > 1e-4, depth, 1.0)
    v_secondary_abs = (
        is_positive
        & (secondary_depth > depth_safe)
        & ~v_negative_depth
    )

    breakdown = {
        "negative_depth":    v_negative_depth,
        "secondary_eclipse": v_secondary,
        "odd_even":          v_odd_even,
        "duration_period":   v_duration,
        "depth_ceiling":     v_depth,
        "bls_power_floor":   v_bls,
        "centroid_shift":    v_centroid,
        "duty_cycle":        v_duty_cycle,
        "secondary_abs":     v_secondary_abs,
    }

    any_veto = (
        v_negative_depth | v_secondary | v_odd_even | v_duration | v_depth
        | v_bls | v_centroid | v_duty_cycle | v_secondary_abs
    )
    return any_veto, breakdown


# ── Report generation ─────────────────────────────────────────────────────────

def _print_report(
    labels_before: np.ndarray,
    labels_after:  np.ndarray,
    vetoed_mask:   np.ndarray,
    breakdown:     dict[str, np.ndarray],
    thresholds:    VetoThresholds,
) -> dict:
    """Print a human-readable vetting summary and return it as a dict."""

    n_total    = len(labels_before)
    # C5: use _POSITIVE_LABEL_THRESHOLD for consistent counting.
    n_pos_before = int((labels_before >= _POSITIVE_LABEL_THRESHOLD).sum())
    n_neg_before = n_total - n_pos_before
    n_vetoed   = int(vetoed_mask.sum())
    n_pos_after  = int((labels_after  >= _POSITIVE_LABEL_THRESHOLD).sum())
    n_neg_after  = n_total - n_pos_after

    lines = [
        "",
        "=" * 60,
        "  Vetting Report",
        "=" * 60,
        f"  Total samples      : {n_total:>6}",
        f"  Positives before   : {n_pos_before:>6}",
        f"  Negatives before   : {n_neg_before:>6}",
        "",
        "  Veto breakdown (positives demoted):",
    ]

    veto_counts = {}
    for name, mask in breakdown.items():
        count = int(mask.sum())
        veto_counts[name] = count
        lines.append(f"    {name:22s}  {count:>5} demoted")

    lines += [
        "",
        f"  Total vetoed       : {n_vetoed:>6}  ({100*n_vetoed/max(n_pos_before,1):.1f}% of positives)",
        f"  Positives after    : {n_pos_after:>6}",
        f"  Negatives after    : {n_neg_after:>6}",
        "=" * 60,
        "",
    ]

    for line in lines:
        print(line, flush=True)

    return {
        "n_total":          n_total,
        "n_pos_before":     n_pos_before,
        "n_neg_before":     n_neg_before,
        "n_vetoed":         n_vetoed,
        "n_pos_after":      n_pos_after,
        "n_neg_after":      n_neg_after,
        "veto_counts":      veto_counts,
        "thresholds": {
            "secondary_ratio":      thresholds.secondary_ratio,
            "secondary_abs":        thresholds.secondary_abs,
            "odd_even_ratio":       thresholds.odd_even_ratio,
            "odd_even_abs":         thresholds.odd_even_abs,
            "duration_period_max":  thresholds.duration_period_max,
            "depth_max":            thresholds.depth_max,
            "bls_power_min":        thresholds.bls_power_min,
            "centroid_abs_kepler":  thresholds.centroid_abs_kepler,
            "centroid_abs_tess":    thresholds.centroid_abs_tess,
            "centroid_abs_k2":      thresholds.centroid_abs_k2,
            "centroid_abs_default": thresholds.centroid_abs_default,
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

# WORKFLOW NOTE: This function modifies labels for ALL samples in the cache,
# including any held-out test set. Always run vetting BEFORE the train/test
# split (i.e., on the raw cache before calling train.py).
def vet_cache(
    cache_path: Path,
    output_dir: Path,
    thresholds: VetoThresholds | None = None,
    missions: np.ndarray | None = None,
) -> dict:
    """
    Load a preprocessing cache, apply veto checks, and save a cleaned version.

    Parameters
    ----------
    cache_path  : Path to the input ``preprocess_cache.npz``.
    output_dir  : Directory for outputs (cleaned cache + report JSON).
    thresholds  : VetoThresholds instance.  Defaults to VetoThresholds().
    missions    : (N,) string array of mission names per sample.  If None, the
                  array is read from the cache (key "missions") when present;
                  otherwise all samples use the default centroid threshold.

    Returns
    -------
    Report dict (same content as the printed summary + veto_report.json).
    """
    if thresholds is None:
        thresholds = VetoThresholds()

    cache_path = Path(cache_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cache: {cache_path}", flush=True)
    # Guard: refuse to vet a cache that already had a train/test split applied.
    # If vetting modifies labels on a post-split cache, the test-set ground truth
    # changes silently, corrupting evaluation.  Detect common split-filename patterns.
    stem_lower = cache_path.stem.lower()
    if any(kw in stem_lower for kw in ("train", "test", "val", "split")):
        raise ValueError(
            f"vet_cache received a cache path that looks post-split: {cache_path}\n"
            "Always run vetting on the RAW (pre-split) cache, then pass the vetted "
            "cache to train.py which performs its own train/test split."
        )
    data    = np.load(cache_path, allow_pickle=False)
    gvs     = data["global_views"]
    lvs     = data["local_views"]
    scalars = data["scalars"]
    labels  = data["labels"]         # (N,)

    print(f"  {len(labels):,} samples loaded.", flush=True)

    # Load or fall back to provided missions array
    if missions is None:
        if "missions" in data:
            missions = data["missions"]
            print(f"  Missions loaded from cache: {np.unique(missions).tolist()}", flush=True)
        else:
            print("  No missions array found in cache — using default centroid threshold.", flush=True)

    # ── Apply vetos ───────────────────────────────────────────────────────────
    vetoed_mask, breakdown = _apply_vetos(scalars, labels, thresholds, missions=missions)

    labels_after = labels.copy()
    labels_after[vetoed_mask] = 0.0   # demote vetoed positives to negative

    # ── Print and save report ─────────────────────────────────────────────────
    report = _print_report(labels, labels_after, vetoed_mask, breakdown, thresholds)

    report_path = output_dir / "vetting_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report saved: {report_path}", flush=True)

    # ── Save cleaned cache (preserve all arrays from source) ─────────────────
    vetted_path = output_dir / "preprocess_cache_vetted.npz"
    save_kwargs: dict = {
        "global_views": gvs,
        "local_views":  lvs,
        "scalars":      scalars,
        "labels":       labels_after,
    }
    # Preserve any extra arrays present in the source cache
    for key in data.files:
        if key not in save_kwargs:
            save_kwargs[key] = data[key]
    np.savez_compressed(vetted_path, **save_kwargs)
    size_kb = vetted_path.stat().st_size // 1024
    print(f"Cleaned cache saved: {vetted_path}  ({size_kb:,} KB)", flush=True)

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply automated veto battery to a preprocessing cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-file",   type=Path, required=True,
                        help="Input preprocess_cache.npz")
    parser.add_argument("--output-dir",   type=Path, required=True,
                        help="Directory for cleaned cache and report")
    parser.add_argument("--secondary-ratio",       type=float, default=0.35)
    parser.add_argument("--secondary-abs",         type=float, default=0.05)
    parser.add_argument("--odd-even-ratio",        type=float, default=0.30)
    parser.add_argument("--odd-even-abs",          type=float, default=0.05)
    parser.add_argument("--duration-period-max",   type=float, default=0.20)
    parser.add_argument("--depth-max",             type=float, default=0.5,
                        help="BLS depth ceiling as fractional flux decrease (0.5 = 50%%). "
                             "Candidates deeper than this are vetoed as unphysical for a planet.")
    parser.add_argument("--bls-power-min",         type=float, default=4.0)
    parser.add_argument("--centroid-abs-kepler",   type=float, default=0.5,
                        help="Centroid shift veto threshold for Kepler (pixels, 4\"/px).")
    parser.add_argument("--centroid-abs-tess",     type=float, default=0.1,
                        help="Centroid shift veto threshold for TESS (pixels, 21\"/px).")
    parser.add_argument("--centroid-abs-k2",       type=float, default=0.3,
                        help="Centroid shift veto threshold for K2 (pixels, noisier centroids).")
    parser.add_argument("--centroid-abs-default",  type=float, default=0.3,
                        help="Centroid veto threshold (pixels) for unknown/unspecified missions (default 0.3)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    thresholds = VetoThresholds(
        secondary_ratio=args.secondary_ratio,
        secondary_abs=args.secondary_abs,
        odd_even_ratio=args.odd_even_ratio,
        odd_even_abs=args.odd_even_abs,
        duration_period_max=args.duration_period_max,
        depth_max=args.depth_max,
        bls_power_min=args.bls_power_min,
        centroid_abs_kepler=args.centroid_abs_kepler,
        centroid_abs_tess=args.centroid_abs_tess,
        centroid_abs_k2=args.centroid_abs_k2,
        centroid_abs_default=args.centroid_abs_default,
    )
    vet_cache(
        cache_path=args.cache_file,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )
