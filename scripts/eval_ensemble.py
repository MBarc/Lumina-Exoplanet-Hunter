"""
Score fold models on the held-out test stars from train.py's split_manifest.npz.

The test split is star-grouped and never used for training, model selection,
thresholds or calibration, so these are the numbers to quote. Works with any
subset of folds (e.g. fold 1 alone after --stop-after-fold 1).

Reports, for each fold model and the ensemble (mean calibrated probability):
  * ROC AUC and PR-AUC (average precision), with 95% intervals from a
    star-level bootstrap (resampling stars, not samples)
  * precision / recall in a fixed review budget: the top 100 and top 1% of
    test signals ranked by score — "if astronomers checked the top K, how
    many would be planets, and what share of planets would they see"
  * expected calibration error (10 bins)
  * the same per mission when the cache mixes missions

Temperatures from calibration.json (ml/calibrate.py, out-of-fold) are applied
when present.

Run:  python -m scripts.eval_ensemble --run-dir training_runs/kepler_run_v2 \
          --cache training_runs/kepler_run_v2/kepler_cache_v2.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from ml.model import ExoNet
from ml.train import _POSITIVE_LABEL_THRESHOLD, AugmentationConfig, MultiMissionDataset


def fold_logits(ckpt: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    model = ExoNet(use_se=True, dropout=0.4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for gv, lv, ov, ev, sv, cv, sc, _ in loader:
            y = model(gv.to(device), lv.to(device), ov.to(device), ev.to(device),
                      sv.to(device), cv.to(device), sc.to(device))
            out.extend(y[:, 0].cpu().numpy().tolist())
    return np.array(out)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean()
                     for b in range(bins) if (which == b).any()))


def budget(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    top = np.argsort(-p)[:k]
    hits = int(y[top].sum())
    return {"k": k, "planets_in_top_k": hits, "precision": round(hits / k, 4),
            "recall": round(hits / max(int(y.sum()), 1), 4)}


def bootstrap_ci(y, p, stars, fn, n=1000, seed=0) -> list[float]:
    """95% interval of fn(y, p) when whole stars are resampled with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(stars, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(fn(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def evaluate(name: str, y, p, stars, n_boot: int) -> dict:
    res = {
        "n": len(y), "planets": int(y.sum()),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "auc_95ci": bootstrap_ci(y, p, stars, roc_auc_score, n_boot),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "pr_auc_95ci": bootstrap_ci(y, p, stars, average_precision_score, n_boot),
        "ece": round(ece(y, p), 4),
        "top_100": budget(y, p, min(100, len(y))),
        "top_1pct": budget(y, p, max(1, len(y) // 100)),
    }
    print(f"  {name:<14} n={res['n']:<6} planets={res['planets']:<5} "
          f"AUC={res['auc']:.4f} {res['auc_95ci']}  PR-AUC={res['pr_auc']:.4f} {res['pr_auc_95ci']}  "
          f"ECE={res['ece']:.3f}  top100 P={res['top_100']['precision']:.2f}  "
          f"top1% P={res['top_1pct']['precision']:.2f} R={res['top_1pct']['recall']:.2f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("training_runs/kepler_run_v2"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = MultiMissionDataset(fits_dir=args.run_dir, augment=False, cache_only=True,
                             cache_file=args.cache, aug_cfg=AugmentationConfig())
    ds.zero_stellar_params = True   # matches training default
    test_idx = np.sort(np.load(args.run_dir / "split_manifest.npz")["test_idx"])
    y = (np.asarray(ds.labels)[test_idx] >= _POSITIVE_LABEL_THRESHOLD).astype(int)
    stars = np.asarray(ds._kepids)[test_idx]
    missions = np.asarray(ds._missions)[test_idx]
    print(f"held-out test: {len(test_idx)} samples, {len(np.unique(stars))} stars, {int(y.sum())} planets",
          flush=True)

    temps: dict = {}
    cal = args.run_dir / "calibration.json"
    if cal.exists():
        temps = json.loads(cal.read_text())["fold_temperatures"]
        print(f"applying out-of-fold temperatures from {cal.name}: {temps}", flush=True)

    loader = DataLoader(Subset(ds, test_idx.tolist()), batch_size=64, shuffle=False, num_workers=0)
    probs: dict[str, np.ndarray] = {}
    for k in range(1, args.folds + 1):
        ckpt = args.run_dir / f"exonet_fold_{k}.pt"
        if ckpt.exists():
            probs[f"fold {k}"] = 1 / (1 + np.exp(-fold_logits(ckpt, loader, device) / float(temps.get(str(k), 1.0))))
    if not probs:
        raise SystemExit(f"no exonet_fold_*.pt in {args.run_dir}")
    if len(probs) > 1:
        probs["ensemble"] = np.mean(list(probs.values()), axis=0)

    results: dict = {"test_samples": len(test_idx), "test_stars": int(len(np.unique(stars))), "models": {}}
    for name, p in probs.items():
        results["models"][name] = {"all": evaluate(name, y, p, stars, args.bootstrap)}
        for m in np.unique(missions) if len(np.unique(missions)) > 1 else []:
            sel = missions == m
            if 0 < y[sel].sum() < sel.sum():
                results["models"][name][str(m)] = evaluate(f"  {m}", y[sel], p[sel], stars[sel], args.bootstrap)

    final = probs.get("ensemble", next(iter(probs.values())))
    np.savez(args.run_dir / "test_probs.npz", test_idx=test_idx, y=y, stars=stars, missions=missions,
             final=final, **{k.replace(" ", "_"): v for k, v in probs.items()})
    (args.run_dir / "test_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.run_dir / 'test_eval.json'} and test_probs.npz", flush=True)


if __name__ == "__main__":
    main()
