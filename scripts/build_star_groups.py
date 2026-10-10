"""
Recover per-sample star grouping from the preprocessing cache, and prove that a
group-wise split eliminates the train/test leakage measured in eval_leakfree.py.

The cache stores no kepid, but star identity is recoverable structurally:
  * preprocessing emits exactly 3 BLS candidates per resolved target, so rows
    3k, 3k+1, 3k+2 are one star-record (verified: 162,726 rows = 54,242 x 3);
  * multi-planet systems add the SAME star once per KOI row, producing
    byte-identical candidate triples, so records sharing a triple fingerprint
    are the same star (428 stars duplicated, up to 7 copies).

Writes star_groups.npz with a star_id per row, suitable for GroupKFold /
GroupShuffleSplit. This is the leakage fix and needs no FITS access.

NOTE: star_id is an arbitrary index, NOT a kepid. Recovering real kepids needs
the FITS index (offline drive) and is required separately for period-matched
relabelling.

Run:  python -m scripts.build_star_groups
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RUN_DIR = Path("training_runs/kepler_run")
CACHE = RUN_DIR / "kepler_cache.npz"
KOI_BLOCK_END = 12729   # rows [0, 12729) are KOI-derived; the rest are unlabelled
SEED = 42


def main() -> int:
    d = np.load(CACHE, allow_pickle=True)
    sc, lab = d["scalars"], d["labels"]
    n = len(lab)
    if n % 3 != 0:
        print(f"ERROR: {n} rows is not divisible by 3 — triple assumption broken.")
        return 1
    n_rec = n // 3

    # Fingerprint each star-record by its 3 candidates' (period, duration, depth).
    fp = [tuple(np.round(sc[i*3:(i+1)*3, :3], 6).ravel().tolist()) for i in range(n_rec)]
    star_of_record: dict[tuple, int] = {}
    rec_star = np.empty(n_rec, dtype=np.int32)
    for i, f in enumerate(fp):
        rec_star[i] = star_of_record.setdefault(f, len(star_of_record))
    star_id = np.repeat(rec_star, 3)

    n_stars = len(star_of_record)
    print(f"rows={n}  star-records={n_rec}  distinct stars={n_stars}")
    print(f"duplicate records merged: {n_rec - n_stars}")

    is_koi = np.zeros(n, dtype=bool)
    is_koi[:KOI_BLOCK_END] = True
    y = (lab >= 0.5).astype(int)
    print(f"KOI block: {is_koi.sum()} rows, {int(y[is_koi].sum())} positives "
          f"({len(np.unique(star_id[is_koi]))} stars)")

    # --- Does a group-wise split remove the leakage? -------------------------
    sig = [tuple(r) for r in np.round(sc[:, :3], 6)]

    def leak_rate(tr: np.ndarray, te: np.ndarray) -> tuple[int, int]:
        train_sigs = {sig[i] for i in tr}
        leaked = [i for i in te if sig[i] in train_sigs]
        pos_leaked = sum(1 for i in leaked if y[i] == 1)
        return len(leaked), pos_leaked

    idx = np.arange(n)
    _, te_row = train_test_split(idx, test_size=max(int(0.10*n), 10),
                                 stratify=y, random_state=SEED)
    tr_row = np.setdiff1d(idx, te_row)
    l, lp = leak_rate(tr_row, te_row)
    print(f"\nROW-wise split (what the finished run used):")
    print(f"   leaked test rows: {l} ({100*l/len(te_row):.1f}%)  | leaked positives: "
          f"{lp}/{int(y[te_row].sum())} ({100*lp/max(int(y[te_row].sum()),1):.1f}%)")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    tr_grp, te_grp = next(gss.split(idx, y, groups=star_id))
    l2, lp2 = leak_rate(tr_grp, te_grp)
    print(f"GROUP-wise split (by star):")
    print(f"   leaked test rows: {l2} ({100*l2/len(te_grp):.1f}%)  | leaked positives: "
          f"{lp2}/{int(y[te_grp].sum())} ({100*lp2/max(int(y[te_grp].sum()),1):.1f}%)")

    np.savez(RUN_DIR / "star_groups.npz",
             star_id=star_id, is_koi=is_koi, y=y, labels=lab)
    print(f"\nSaved: {RUN_DIR/'star_groups.npz'}  (star_id, is_koi, y, labels)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
