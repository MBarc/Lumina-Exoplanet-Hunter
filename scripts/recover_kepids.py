"""
Recover the real kepid for every row in the preprocessing cache.

The cache stores no target IDs, but the KOI block's ordering is deterministic:
`download_koi_table()` yields (kepid, label) in catalog CSV row order, keeping
only rows with a mappable disposition; `_resolve_kepler()` then drops any kepid
with no FITS on disk, preserving order; each survivor contributes exactly 3 BLS
candidate rows.

So replaying (catalog order -> disposition filter -> FITS-availability filter)
reproduces the KOI block row-for-row. The reconstruction is *self-validating*:
the replayed label sequence must equal the cached labels exactly. If it does,
the kepid alignment is proven, not assumed.

Requires scripts/nas_fetch_kepids.py to have been run first.

Output: training_runs/kepler_run/kepids.npz  (kepid per row, -1 for unlabelled block)

Run:  python -m scripts.recover_kepids
"""
from __future__ import annotations

import csv
import io
import sys
import traceback
from pathlib import Path

import numpy as np
import requests

from ml.train import _koi_disposition_to_label

RUN_DIR = Path("training_runs/kepler_run")
KEPID_FILE = RUN_DIR / "nas_kepids.txt"
KOI_BLOCK_END = 12729
TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
       "select+kepid,koi_disposition,koi_score+from+cumulative&format=csv")


def main() -> int:
    if not KEPID_FILE.exists():
        print(f"ERROR: {KEPID_FILE} missing — run scripts.nas_fetch_kepids first.")
        return 1
    available = {ln.strip().zfill(9) for ln in KEPID_FILE.read_text().splitlines() if ln.strip()}
    print(f"kepids with FITS on the NAS: {len(available)}")

    print("downloading KOI catalog ...", flush=True)
    text = requests.get(TAP, timeout=180).text
    records: list[tuple[int, float]] = []
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
        label = _koi_disposition_to_label(row.get("koi_disposition", ""), koi_score=score)
        if label is not None:
            records.append((kepid, label))
    print(f"catalog rows with a mappable disposition: {len(records)}")

    # Replay _resolve_kepler: keep records whose kepid has FITS, preserving order.
    resolved = [(k, l) for k, l in records if str(k).zfill(9) in available]
    print(f"resolved (FITS present): {len(resolved)}  -> {3*len(resolved)} rows")

    cache = np.load(RUN_DIR / "kepler_cache.npz", allow_pickle=True)
    cached = cache["labels"][:KOI_BLOCK_END]
    expected = np.repeat(np.array([l for _, l in resolved], dtype=np.float32), 3)

    print(f"\ncached KOI-block rows : {len(cached)}")
    print(f"replayed rows         : {len(expected)}")
    if len(expected) != len(cached):
        print("LENGTH MISMATCH — the catalog has drifted since the cache was built.")
        n = min(len(expected), len(cached))
        agree = int((expected[:n] == cached[:n]).sum())
        print(f"  labels agreeing over first {n}: {agree} ({100*agree/n:.2f}%)")
        first_bad = int(np.argmax(expected[:n] != cached[:n])) if agree < n else -1
        print(f"  first divergence at row {first_bad} (record {first_bad//3})")
        return 1

    agree = int((expected == cached).sum())
    print(f"labels matching       : {agree}/{len(cached)} ({100*agree/len(cached):.2f}%)")
    if agree != len(cached):
        bad = int(np.argmax(expected != cached))
        print(f"VALIDATION FAILED — first divergence at row {bad} (record {bad//3})")
        return 1

    print("\nVALIDATION PASSED — kepid alignment is exact.")
    kepid_per_row = np.full(len(cache["labels"]), -1, dtype=np.int64)
    kepid_per_row[:KOI_BLOCK_END] = np.repeat(
        np.array([k for k, _ in resolved], dtype=np.int64), 3)
    n_stars = len(set(kepid_per_row[:KOI_BLOCK_END].tolist()))
    print(f"distinct kepids in KOI block: {n_stars}")

    np.savez(RUN_DIR / "kepids.npz", kepid=kepid_per_row, koi_block_end=KOI_BLOCK_END)
    print(f"saved: {RUN_DIR/'kepids.npz'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
