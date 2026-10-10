"""
Verify the SHIPPED _bls_search (not the harness's reimplementation) against the
cached confirmed-planet light curves.

bls_variant_harness.py explored variants with its own copy of the search loop.
This runs the real ml.preprocess._bls_search that preprocessing will actually
call, so what we measure is what will ship.

Reports true-period recovery, harmonic-tolerant recovery, and the median fold
smear (period error x baseline, in units of transit duration) — the quantity
that blurs the CNN's phase-folded views.

Run:  python -m scripts.verify_bls_shipped
"""
from __future__ import annotations

import csv
import io
import sys
import time as _time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

from ml.preprocess import _bls_search

RUN_DIR = Path("training_runs/kepler_run")
LC_CACHE = RUN_DIR / "bls_testbed_lightcurves.npz"
TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
       "select+kepid,koi_disposition,koi_period+from+cumulative&format=csv")
TOL = 0.01
HARMONICS = (1.0, 2.0, 0.5, 3.0, 1 / 3.0, 4.0, 0.25)

OLD = dict(refine=False, probe_harmonics=False, max_period_fraction=0.5,
           reject_harmonics=True)
NEW = dict()   # shipped defaults


def main() -> int:
    d = np.load(LC_CACHE)
    kepids = sorted({int(n[2:]) for n in d.files if n.startswith("t_")})
    lcs = {k: (d[f"t_{k}"], d[f"f_{k}"]) for k in kepids}

    periods: dict[int, list[float]] = defaultdict(list)
    for row in csv.DictReader(io.StringIO(requests.get(TAP, timeout=180).text)):
        try:
            k = int(row["kepid"].strip())
            p = float(row.get("koi_period") or "nan")
        except (ValueError, KeyError):
            continue
        if row.get("koi_disposition", "").strip().upper() == "CONFIRMED" and p > 0:
            periods[k].append(p)

    def score(kw, n_cand):
        exact = harm = 0
        smears: list[float] = []
        t0 = _time.time()
        for k, (t, f) in lcs.items():
            cat = periods.get(k, [])
            cands = _bls_search(t, f, n_cand, **kw)
            got = [c["period"] for c in cands]
            baseline = t[-1] - t[0]
            if any(abs(p - c) <= TOL * c for p in got for c in cat):
                exact += 1
            if any(abs(p - c * m) <= TOL * c * m
                   for p in got for c in cat for m in HARMONICS):
                harm += 1
            # Fold smear for the closest candidate, in transit durations.
            best = None
            for p in got:
                for c in cat:
                    e = abs(p - c) / c
                    if best is None or e < best[0]:
                        best = (e, c, cands[got.index(p)]["duration"])
            if best and best[0] < 0.05 and best[2] > 0:
                smears.append(best[0] * baseline / best[2])
        return exact, harm, (np.median(smears) if smears else float("nan")), _time.time() - t0

    n = len(lcs)
    print(f"{n} confirmed-planet light curves from cache\n")
    print(f"{'config':<26} {'exact P':>12} {'any harmonic':>14} {'smear':>10} {'secs':>7}")
    print("-" * 74)
    for name, kw, nc in (("OLD (pre-fix)", OLD, 3),
                         ("SHIPPED defaults", NEW, 3),
                         ("SHIPPED, n_cand=5", NEW, 5)):
        e, h, sm, secs = score(kw, nc)
        print(f"{name:<26} {e:>3}/{n} ({100*e/n:4.0f}%) {h:>5}/{n} ({100*h/n:4.0f}%) "
              f"{sm:>8.2f}x {secs:>7.0f}")
    print("\nsmear = period error x baseline, in transit durations (lower is better;")
    print("it is the blurring applied to the phase-folded views the CNN sees)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
