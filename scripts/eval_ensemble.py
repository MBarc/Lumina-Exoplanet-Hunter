"""
Evaluate the 5-fold ensemble on the SAME held-out test set train.py uses.

train.py only scores the single best checkpoint (exonet.pt) against the test
set, but it recommends EnsembleInference over the per-fold models for
production.  Nobody had measured whether the ensemble actually beats the single
model, so this reproduces the exact test split (same seed, same stratification)
and reports, on identical data:

  * each individual fold model
  * the single best checkpoint (should reproduce train.py's reported number)
  * the ensemble (mean probability across the 5 fold models)

Also reports precision/recall at the OOF threshold from threshold.json so the
ensemble's operating point is directly comparable to evaluation_report.json.

Run:  python -m scripts.eval_ensemble
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from ml.model import ExoNet
from ml.train import (
    _POSITIVE_LABEL_THRESHOLD,
    AugmentationConfig,
    MultiMissionDataset,
)

RUN_DIR = Path("training_runs/kepler_run")
CACHE   = RUN_DIR / "kepler_cache.npz"
SEED    = 42
BATCH   = 64


def score_model(ckpt: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return per-sample planet probabilities for one checkpoint."""
    model = ExoNet(use_se=True, dropout=0.4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for gv, lv, ov, ev, sv, cv, scalar, _ in loader:
            out = model(gv.to(device), lv.to(device), ov.to(device), ev.to(device),
                        sv.to(device), cv.to(device), scalar.to(device))
            probs.extend(torch.sigmoid(out)[:, 0].cpu().numpy().tolist())
    return np.array(probs)


def report(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    auc    = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    preds  = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    print(f"  {name:<22}  AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
          f"| @{threshold:.2f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}", flush=True)
    return {"auc": round(float(auc), 4), "pr_auc": round(float(pr_auc), 4),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "f1": round(float(f1), 4)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    # Rebuild the dataset exactly as train.py does (cache-only, no augmentation).
    dataset = MultiMissionDataset(
        fits_dir=Path("E:/fits_cache/kepler"),
        csv_path=None,
        max_samples=None,
        augment=False,
        cache_only=True,
        cache_file=CACHE,
        max_unlabeled=50_000,
        aug_cfg=AugmentationConfig(),
        preprocess_workers=1,
    )
    dataset.zero_stellar_params = True   # matches --zero-stellar-params default

    # Reproduce the held-out split: same seed, same stratification on binarised labels.
    labels       = np.array(dataset.labels)
    indices      = np.array(range(len(dataset)))
    strat_labels = (labels >= _POSITIVE_LABEL_THRESHOLD).astype(int)
    test_size    = max(int(0.10 * len(dataset)), 10)
    _, test_idx  = train_test_split(
        indices, test_size=test_size, stratify=strat_labels, random_state=SEED
    )
    print(f"held-out test set: {len(test_idx)} samples "
          f"({int(strat_labels[test_idx].sum())} positives)", flush=True)

    loader = DataLoader(Subset(dataset, test_idx.tolist()),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    y_true = strat_labels[test_idx]

    threshold = 0.77
    thr_file = RUN_DIR / "threshold.json"
    if thr_file.exists():
        threshold = json.loads(thr_file.read_text())["threshold"]
    print(f"OOF threshold from threshold.json: {threshold}\n", flush=True)

    results: dict = {"n_test": len(test_idx), "threshold": threshold}

    fold_probs: list[np.ndarray] = []
    print("Individual fold models:", flush=True)
    for k in range(1, 6):
        ckpt = RUN_DIR / f"exonet_fold_{k}.pt"
        if not ckpt.exists():
            print(f"  fold {k}: MISSING {ckpt.name}", flush=True)
            continue
        p = score_model(ckpt, loader, device)
        fold_probs.append(p)
        results[f"fold_{k}"] = report(f"fold {k}", y_true, p, threshold)

    print("\nSingle best checkpoint (what train.py reports):", flush=True)
    best_probs = score_model(RUN_DIR / "exonet.pt", loader, device)
    results["single_best"] = report("exonet.pt", y_true, best_probs, threshold)

    if len(fold_probs) > 1:
        print("\nEnsemble:", flush=True)
        ens_mean = np.mean(fold_probs, axis=0)
        results["ensemble_mean_prob"] = report("mean probability", y_true, ens_mean, threshold)
        # Rank averaging is scale-free — useful when folds are differently calibrated.
        ranks = np.mean([np.argsort(np.argsort(p)) / (len(p) - 1) for p in fold_probs], axis=0)
        results["ensemble_rank_avg"] = report("rank average", y_true, ranks, threshold)

        cm = confusion_matrix(y_true, (ens_mean >= threshold).astype(int))
        print(f"\n  Ensemble confusion matrix @ {threshold} "
              f"[[TN FP] [FN TP]]:\n  {cm.tolist()}", flush=True)
        results["ensemble_confusion_matrix"] = cm.tolist()

    out = RUN_DIR / "ensemble_eval.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
