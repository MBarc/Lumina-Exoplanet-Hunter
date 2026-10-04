"""
Quantify how much train/test duplication inflates the reported ensemble score.

Multi-planet systems add the same kepid's identical BLS candidates once per KOI
row, and the split is by row rather than by star, so some held-out test samples
have an exact duplicate in training.  This scores the existing 5-fold ensemble
on (a) the full test set as reported, and (b) the leak-free subset, so the gap
between them is the measurement error.

No retraining — reuses the exported fold checkpoints.
Per-sample probabilities are saved so future analyses skip the re-scoring.

Run:  python -m scripts.eval_leakfree
"""
from __future__ import annotations

import collections
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from ml.model import ExoNet
from ml.train import _POSITIVE_LABEL_THRESHOLD, AugmentationConfig, MultiMissionDataset

RUN_DIR = Path("training_runs/kepler_run")
SEED = 42


def score(ckpt: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    model = ExoNet(use_se=True, dropout=0.4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for gv, lv, ov, ev, sv, cv, sc, _ in loader:
            y = model(gv.to(device), lv.to(device), ov.to(device), ev.to(device),
                      sv.to(device), cv.to(device), sc.to(device))
            out.extend(torch.sigmoid(y)[:, 0].cpu().numpy().tolist())
    return np.array(out)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MultiMissionDataset(
        fits_dir=Path("E:/fits_cache/kepler"), csv_path=None, max_samples=None,
        augment=False, cache_only=True, cache_file=RUN_DIR / "kepler_cache.npz",
        max_unlabeled=50_000, aug_cfg=AugmentationConfig(), preprocess_workers=1,
    )
    ds.zero_stellar_params = True

    labels = np.array(ds.labels)
    strat = (labels >= _POSITIVE_LABEL_THRESHOLD).astype(int)
    idx = np.array(range(len(labels)))
    _, test_idx = train_test_split(
        idx, test_size=max(int(0.10 * len(labels)), 10), stratify=strat, random_state=SEED
    )
    train_idx = np.array(sorted(set(idx.tolist()) - set(test_idx.tolist())))

    # Identity key = the BLS fit (period, duration, depth); identical rows are the
    # same star's same candidate re-added by a second KOI record.
    raw = np.load(RUN_DIR / "kepler_cache.npz", allow_pickle=True)
    sig = [tuple(r) for r in np.round(raw["scalars"][:, :3], 6)]
    train_keys = collections.Counter(sig[i] for i in train_idx)
    leaked = np.array([train_keys.get(sig[i], 0) > 0 for i in test_idx])
    print(f"test samples: {len(test_idx)}  leaked: {int(leaked.sum())} "
          f"({100*leaked.mean():.1f}%)", flush=True)

    loader = DataLoader(Subset(ds, test_idx.tolist()), batch_size=64,
                        shuffle=False, num_workers=0)
    probs = [score(RUN_DIR / f"exonet_fold_{k}.pt", loader, device) for k in range(1, 6)]
    ens = np.mean(probs, axis=0)
    y = strat[test_idx]

    results: dict = {}
    for name, mask in (("full test set (as reported)", np.ones(len(y), bool)),
                       ("leak-free subset", ~leaked),
                       ("leaked subset only", leaked)):
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        auc = roc_auc_score(y[mask], ens[mask])
        ap = average_precision_score(y[mask], ens[mask])
        print(f"  {name:<28} n={int(mask.sum()):5d} pos={int(y[mask].sum()):4d}  "
              f"AUC={auc:.4f}  PR-AUC={ap:.4f}", flush=True)
        results[name] = {"n": int(mask.sum()), "positives": int(y[mask].sum()),
                         "auc": round(float(auc), 4), "pr_auc": round(float(ap), 4)}

    np.savez(RUN_DIR / "test_probs.npz", test_idx=test_idx, y=y,
             ensemble=ens, folds=np.array(probs), leaked=leaked)
    (RUN_DIR / "leakfree_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {RUN_DIR/'leakfree_eval.json'} and test_probs.npz", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
