"""
Automated veto battery for ExoNet preprocessing cache.

Applies a series of physics-based and statistical checks to the positive-labeled
candidates in a preprocessing cache, demoting likely false positives to the
negative class.  A cleaned cache is written that can be passed directly to
ml.train via --cache-file for retraining.

Veto checks applied (in order)
-------------------------------
1. secondary_eclipse  — secondary depth > 50 % of primary depth, indicating
                         a background eclipsing binary with both eclipses visible
2. odd_even           — alternating transit depth difference > 40 % of primary
                         depth, indicating a grazing eclipsing binary at 2× period
3. duration_period    — transit duration > 20 % of the orbital period, which is
                         physically impossible for a planet (transit geometry)
4. depth_ceiling      — BLS depth > 8.0 σ in normalised flux; planets around
                         Sun-like stars rarely exceed ~3 σ (Jupiter ≈ 1 %)
5. bls_power_floor    — BLS SNR < 4.0; signal too weak to be reliably real

Checks 1–2 require both the ratio AND an absolute floor to avoid vetoing
candidates where the secondary/odd-even measurement is dominated by noise.

Scalar feature order in the cache
----------------------------------
  index 0 : period_days
  index 1 : duration_days
  index 2 : depth  (BLS depth in normalised σ units)
  index 3 : bls_power
  index 4 : secondary_depth  (depth at phase ±0.5, same σ units)
  index 5 : odd_even_diff    (|mean_odd − mean_even|, same σ units)

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

import numpy as np


# ── Default veto thresholds ───────────────────────────────────────────────────

@dataclass
class VetoThresholds:
    """Tuneable thresholds for each veto check."""

    # Secondary eclipse veto
    secondary_ratio:    float = 0.50   # secondary_depth / depth > this → veto
    secondary_abs:      float = 0.05   # secondary_depth must also exceed this

    # Odd/even veto
    odd_even_ratio:     float = 0.40   # odd_even_diff / depth > this → veto
    odd_even_abs:       float = 0.05   # odd_even_diff must also exceed this

    # Duration/period veto
    duration_period_max: float = 0.20  # duration / period > this → veto

    # Depth ceiling veto (in normalised σ units)
    depth_max:          float = 8.0    # depth > this → veto

    # BLS power floor
    bls_power_min:      float = 4.0    # bls_power < this → veto


# ── Per-sample veto logic ─────────────────────────────────────────────────────

def _apply_vetos(
    scalars: np.ndarray,
    labels: np.ndarray,
    thresholds: VetoThresholds,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Evaluate all veto checks against every positive-labeled sample.

    Parameters
    ----------
    scalars : (N, 6) float32 array
    labels  : (N,)   float32 array of 0.0 / 1.0
    thresholds : VetoThresholds

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
    depth         = np.abs(scalars[:, 2])   # use absolute value for ratio checks
    bls_power     = scalars[:, 3]
    sec_depth     = scalars[:, 4]
    odd_even_diff = scalars[:, 5]

    is_positive = labels == 1.0

    # ── Veto 1: secondary eclipse ─────────────────────────────────────────────
    v_secondary = (
        is_positive
        & (sec_depth > t.secondary_abs)
        & (sec_depth > t.secondary_ratio * depth)
    )

    # ── Veto 2: odd/even depth difference ─────────────────────────────────────
    v_odd_even = (
        is_positive
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

    breakdown = {
        "secondary_eclipse": v_secondary,
        "odd_even":          v_odd_even,
        "duration_period":   v_duration,
        "depth_ceiling":     v_depth,
        "bls_power_floor":   v_bls,
    }

    any_veto = v_secondary | v_odd_even | v_duration | v_depth | v_bls
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
    n_pos_before = int((labels_before == 1.0).sum())
    n_neg_before = int((labels_before == 0.0).sum())
    n_vetoed   = int(vetoed_mask.sum())
    n_pos_after  = int((labels_after  == 1.0).sum())
    n_neg_after  = int((labels_after  == 0.0).sum())

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
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

def vet_cache(
    cache_path: Path,
    output_dir: Path,
    thresholds: VetoThresholds | None = None,
) -> dict:
    """
    Load a preprocessing cache, apply veto checks, and save a cleaned version.

    Parameters
    ----------
    cache_path  : Path to the input ``preprocess_cache.npz``.
    output_dir  : Directory for outputs (cleaned cache + report JSON).
    thresholds  : VetoThresholds instance.  Defaults to VetoThresholds().

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
    data    = np.load(cache_path, allow_pickle=False)
    gvs     = data["global_views"]   # (N, 2001)
    lvs     = data["local_views"]    # (N, 201)
    scalars = data["scalars"]        # (N, 6)
    labels  = data["labels"]         # (N,)

    print(f"  {len(labels):,} samples loaded.", flush=True)

    # ── Apply vetos ───────────────────────────────────────────────────────────
    vetoed_mask, breakdown = _apply_vetos(scalars, labels, thresholds)

    labels_after = labels.copy()
    labels_after[vetoed_mask] = 0.0   # demote vetoed positives to negative

    # ── Print and save report ─────────────────────────────────────────────────
    report = _print_report(labels, labels_after, vetoed_mask, breakdown, thresholds)

    report_path = output_dir / "vetting_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report saved: {report_path}", flush=True)

    # ── Save cleaned cache ────────────────────────────────────────────────────
    vetted_path = output_dir / "preprocess_cache_vetted.npz"
    np.savez_compressed(
        vetted_path,
        global_views=gvs,
        local_views=lvs,
        scalars=scalars,
        labels=labels_after,
    )
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
    parser.add_argument("--secondary-ratio",     type=float, default=0.50)
    parser.add_argument("--secondary-abs",        type=float, default=0.05)
    parser.add_argument("--odd-even-ratio",       type=float, default=0.40)
    parser.add_argument("--odd-even-abs",         type=float, default=0.05)
    parser.add_argument("--duration-period-max",  type=float, default=0.20)
    parser.add_argument("--depth-max",            type=float, default=8.0)
    parser.add_argument("--bls-power-min",        type=float, default=4.0)
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
    )
    vet_cache(
        cache_path=args.cache_file,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )
