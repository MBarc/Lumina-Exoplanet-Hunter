"""
Injection-recovery test: can Lumina find planets nobody catalogued?

Plants synthetic transits (known period, depth, duration) into real light
curves of stars with NO catalogued object, runs the real preprocessing and the
real model, and measures how often:
  * the BLS search finds the injected period (x1, or a x2 / x1/2 alias), and
  * the model scores that signal at or above --threshold.
Every planet in our test set was already found by Kepler's own pipeline; this
is the measurement that says how Lumina does on ones that were not.

Run on the VM (CPU heavy):
  python -m scripts.injection_recovery --mission kepler \
      --index training_runs/kepler_run_v2/fits_index.txt \
      --model training_runs/kepler_run_v2/exonet_fold_1.onnx --n 300 --workers 6 \
      --out training_runs/injection/kepler_fold1
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import tempfile
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

from ml.train import _star_key

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CATALOGUED = {   # stars with ANY catalogued object are excluded from injection targets
    "kepler": ["select kepid from cumulative"],
    "tess": ["select tid from toi", "select tic_id from pscomppars where tic_id is not null"],
}
DEPTH_BINS = [50, 150, 500, 1500, 5000]     # ppm
PERIOD_BINS = [0.5, 3, 10, 30, 60]          # days
_model = None


def catalogued_stars(mission: str) -> set[int]:
    out: set[int] = set()
    for q in CATALOGUED[mission]:
        r = requests.get(TAP, params={"query": q, "format": "csv"}, timeout=180)
        r.raise_for_status()
        for row in csv.reader(io.StringIO(r.text)):
            d = "".join(ch for ch in (row[0] if row else "") if ch.isdigit())
            if d:
                out.add(int(d))
    return out


def pick_stars(index: Path, mission: str, n: int, seed: int) -> list[list[str]]:
    files: dict[int, list[str]] = defaultdict(list)
    for line in open(index, encoding="utf-8"):
        p = line.strip()
        if p.lower().endswith(".fits"):
            _, num = _star_key(p, mission)
            if num is not None:
                files[num].append(p)
    known = catalogued_stars(mission)
    clean = sorted(s for s in files if s not in known)
    print(f"{len(files)} stars in index, {len(clean)} with no catalogued object", flush=True)
    return [sorted(files[s]) for s in random.Random(seed).sample(clean, min(n, len(clean)))]


def inject(paths: list[str], tmp: Path, period: float, t0: float, dur: float, depth: float) -> list[str]:
    """Copy each file and multiply the flux column by a box transit."""
    from ml.preprocess import _FLUX_COLS
    out = []
    for p in paths:
        dst = tmp / Path(p).name
        shutil.copy(p, dst)
        with fits.open(dst, mode="update") as h:
            d = h[1].data
            col = next(c for c in _FLUX_COLS if c in d.columns.names)
            dt = ((d["TIME"] - t0 + 0.5 * period) % period) - 0.5 * period
            d[col] = np.where(np.abs(dt) < dur / 2, d[col] * (1.0 - depth), d[col])
        out.append(str(dst))
    return out


def _init(model_paths: list[str], threads: int) -> None:
    global _model
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from ml.inference import EnsembleInference, ExoNetInference
    _model = ExoNetInference(model_paths[0]) if len(model_paths) == 1 else EnsembleInference(model_paths)


def trial(args: tuple) -> dict:
    from ml.preprocess import preprocess_multi
    from ml.train import _matches_koi_period
    i, paths, seed, threshold = args
    rng = random.Random(seed * 100_003 + i)
    period = float(np.exp(rng.uniform(np.log(PERIOD_BINS[0]), np.log(PERIOD_BINS[-1]))))
    depth_ppm = float(np.exp(rng.uniform(np.log(DEPTH_BINS[0]), np.log(DEPTH_BINS[-1]))))
    dur = 13 / 24 * (period / 365.25) ** (1 / 3) * rng.uniform(0.7, 1.2)   # Sun-like star, days
    with fits.open(paths[0]) as h:
        t = h[1].data["TIME"]
        t = t[np.isfinite(t)]
    t0 = float(t.min()) + rng.uniform(0, period)
    rec = {"star": Path(paths[0]).name, "files": len(paths), "period": period, "depth_ppm": depth_ppm,
           "duration_h": dur * 24, "found_period": False, "score": None, "flagged": False, "error": None}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cands = preprocess_multi(inject(paths, Path(tmp), period, t0, dur, depth_ppm / 1e6))
            scores = _model.predict_batch(cands) if cands else []
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
            return rec
    hits = [s for c, s in zip(cands, scores) if _matches_koi_period(float(c.period), [period])]
    if hits:
        rec.update(found_period=True, score=float(max(hits)), flagged=max(hits) >= threshold)
    return rec


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["error"] is None]
    def rate(sub, key):
        return round(sum(r[key] for r in sub) / len(sub), 3) if sub else None
    def binned(key, edges):
        out = []
        for lo, hi in zip(edges, edges[1:]):
            sub = [r for r in ok if lo <= r[key] < hi]
            out.append({"range": f"{lo}-{hi}", "n": len(sub), "bls_found": rate(sub, "found_period"),
                        "model_flagged": rate(sub, "flagged")})
        return out
    return {"trials": len(rows), "errors": len(rows) - len(ok),
            "bls_found": rate(ok, "found_period"), "model_flagged": rate(ok, "flagged"),
            "by_depth_ppm": binned("depth_ppm", DEPTH_BINS), "by_period_days": binned("period", PERIOD_BINS)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mission", choices=sorted(CATALOGUED), default="kepler")
    ap.add_argument("--index", type=Path, required=True, help="FITS listing, one path per line")
    ap.add_argument("--model", nargs="+", required=True, help="ONNX model(s); several = ensemble")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.json and .csv)")
    args = ap.parse_args()

    stars = pick_stars(args.index, args.mission, args.n, args.seed)
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.model, 1)) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(trial, [(i, s, args.seed, args.threshold)
                                                          for i, s in enumerate(stars)]), 1):
            rows.append(r)
            if k % 10 == 0 or k == len(stars):
                s = summarise(rows)
                print(f"{k}/{len(stars)}  bls_found={s['bls_found']}  model_flagged={s['model_flagged']}  "
                      f"errors={s['errors']}  {time.time() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows) | {"mission": args.mission, "models": args.model, "threshold": args.threshold}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(args.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
