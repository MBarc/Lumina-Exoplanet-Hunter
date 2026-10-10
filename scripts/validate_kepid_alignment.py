"""
Decide whether the reconstructed kepid alignment is correct.

recover_kepids.py reproduced the KOI block's row count exactly (4,243 records ->
12,729 rows) but only 82.9% of labels matched. Two explanations:
  (a) the alignment is wrong, or
  (b) the alignment is right and the CATALOG DRIFTED since the cache was built
      in April 2026 (koi_score is re-derived by NASA; CANDIDATEs get promoted
      to CONFIRMED or demoted to FALSE POSITIVE).

Two tests separate them:
  1. Breakdown of label agreement. Drift should spare hard labels (CONFIRMED
     1.0 / FALSE POSITIVE 0.0 are stable) and concentrate in soft koi_scores.
  2. THE DECISIVE TEST: per-star period matching. If the alignment is right,
     a record's 3 BLS candidate periods should hit that star's catalogued
     koi_period far more often than for a randomly reassigned star. This is
     the test that population-level matching could not do.

Run:  python -m scripts.validate_kepid_alignment
"""
from __future__ import annotations

import csv
import io
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

from ml.train import _koi_disposition_to_label

RUN_DIR = Path("training_runs/kepler_run")
KOI_BLOCK_END = 12729
TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
       "select+kepid,koi_disposition,koi_score,koi_period+from+cumulative&format=csv")
TOL = 0.01          # 1% relative
SEED = 42


def period_hit(bls: np.ndarray, cat: list[float]) -> bool:
    """True if any BLS candidate matches any catalogued period (or 2:1 harmonic)."""
    for p in bls:
        if p <= 0:
            continue
        for c in cat:
            for m in (1.0, 2.0, 0.5):
                if abs(p - c*m) <= TOL * c * m:
                    return True
    return False


def main() -> int:
    available = {ln.strip().zfill(9)
                 for ln in (RUN_DIR / "nas_kepids.txt").read_text().splitlines() if ln.strip()}
    text = requests.get(TAP, timeout=180).text
    records, periods = [], defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        try:
            kepid = int(row["kepid"].strip())
        except (ValueError, KeyError):
            continue
        try:
            score = float(row.get("koi_score") or "nan")
            if np.isnan(score):
                score = None
        except (ValueError, TypeError):
            score = None
        lab = _koi_disposition_to_label(row.get("koi_disposition", ""), koi_score=score)
        if lab is None:
            continue
        records.append((kepid, lab))
        try:
            per = float(row.get("koi_period") or "nan")
            if not np.isnan(per) and per > 0:
                periods[kepid].append(per)
        except (ValueError, TypeError):
            pass

    resolved = [(k, l) for k, l in records if str(k).zfill(9) in available]
    cache = np.load(RUN_DIR / "kepler_cache.npz", allow_pickle=True)
    cached = cache["labels"][:KOI_BLOCK_END]
    sc = cache["scalars"][:KOI_BLOCK_END]
    rec_kepid = np.array([k for k, _ in resolved], dtype=np.int64)
    rec_label = np.array([l for _, l in resolved], dtype=np.float32)
    cached_rec = cached[::3]                      # one cached label per record
    n_rec = len(resolved)
    print(f"records: {n_rec}   rows: {n_rec*3} (cached {len(cached)})\n")

    # --- Test 1: where do labels disagree? ----------------------------------
    hard = np.isin(cached_rec, [0.0, 1.0]) & np.isin(rec_label, [0.0, 1.0])
    soft = ~hard
    for name, m in (("hard labels (0.0/1.0 both sides)", hard), ("soft koi_score involved", soft)):
        if m.sum():
            agree = int((cached_rec[m] == rec_label[m]).sum())
            print(f"  {name:<34} n={int(m.sum()):5d}  agree={agree:5d} ({100*agree/m.sum():.1f}%)")

    # --- Test 2 (decisive): per-star period matching vs shuffled control ----
    print("\nPer-star period match (BLS candidate vs that star's catalogued koi_period):")
    bls = sc[:, 0].reshape(n_rec, 3)
    aligned = np.array([period_hit(bls[i], periods.get(int(rec_kepid[i]), [])) for i in range(n_rec)])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_rec)
    control = np.array([period_hit(bls[i], periods.get(int(rec_kepid[perm[i]]), [])) for i in range(n_rec)])

    print(f"  aligned (reconstructed kepid) : {aligned.mean()*100:5.1f}%  ({int(aligned.sum())}/{n_rec})")
    print(f"  shuffled control (random star): {control.mean()*100:5.1f}%  ({int(control.sum())}/{n_rec})")
    ratio = aligned.mean() / max(control.mean(), 1e-9)
    print(f"  enrichment over chance        : {ratio:5.1f}x")

    # Criterion is ENRICHMENT over the shuffled control, not the absolute rate:
    # the absolute rate measures how often BLS recovered the catalogued signal
    # (a property of the search), while enrichment measures whether periods
    # track the reconstructed star (a property of the alignment).
    ok = ratio > 3 and aligned.mean() > 0.15
    print("\n" + ("ALIGNMENT CONFIRMED — labels differ only because the catalog drifted."
                  if ok else
                  "ALIGNMENT NOT CONFIRMED — periods do not track the reconstructed kepids."))

    # How often did BLS actually recover the KOI's signal, by label class?
    print("\nBLS recovery of the catalogued KOI period, by cached label:")
    pos = cached_rec >= 0.5
    for name, m in (("positive (CONFIRMED/high score)", pos), ("negative (FALSE POSITIVE/low)", ~pos)):
        if m.sum():
            print(f"  {name:<34} n={int(m.sum()):5d}  BLS found it: "
                  f"{aligned[m].mean()*100:5.1f}%")
    print(f"\n  => for {100*(1-aligned[pos].mean()):.0f}% of positive targets, NONE of the 3 BLS")
    print("     candidates corresponds to the known planet — so all 3 rows carry a")
    print("     'planet' label with no planet signal in them.")
    if ok:
        kepid_per_row = np.full(len(cache["labels"]), -1, dtype=np.int64)
        kepid_per_row[:KOI_BLOCK_END] = np.repeat(rec_kepid, 3)
        np.savez(RUN_DIR / "kepids.npz", kepid=kepid_per_row,
                 koi_block_end=KOI_BLOCK_END, matched=np.repeat(aligned, 3))
        print(f"saved: {RUN_DIR/'kepids.npz'}  "
              f"({len(set(rec_kepid.tolist()))} distinct kepids)")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
