"""
Check that ml.train's star-grouped splits never put one star on both sides.

Reproduces the held-out test split and the CV folds exactly as ml/train.py
does (same groups, same splitter, same seed) on a real cache, and asserts the
kepid sets are disjoint everywhere.

Run:  python -m scripts.check_star_split training_runs/kepler_run_v2/kepler_cache_v2.npz
"""
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SEED, FOLDS, POS = 42, 5, 0.5

z = np.load(sys.argv[1], mmap_mode="r")
kepids = np.asarray(z["kepids"])
strat = (np.asarray(z["labels"]) >= POS).astype(int)
groups = np.array([k if k else f"_{i}" for i, k in enumerate(kepids)])
idx = np.arange(len(groups))

t0 = time.time()
trv, test = next(StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(idx, strat, groups))
assert not set(groups[trv]) & set(groups[test]), "star leaks between train/val and test"
print(f"test: {len(test)} samples ({100 * len(test) / len(idx):.1f}%), "
      f"pos rate {strat[test].mean():.4f} vs {strat.mean():.4f} overall")

g_cv, s_cv = groups[trv], strat[trv]
for f, (tr, va) in enumerate(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
                             .split(trv, s_cv, g_cv), start=1):
    assert not set(g_cv[tr]) & set(g_cv[va]), f"star leaks in fold {f}"
    print(f"fold {f}: train={len(tr)} val={len(va)} val pos rate {s_cv[va].mean():.4f}")
print(f"OK: no star on both sides of any split ({time.time() - t0:.0f}s)")
