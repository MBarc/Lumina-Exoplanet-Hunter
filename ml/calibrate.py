"""
Temperature scaling calibration for ExoNet fold checkpoints.

After cross-validation training, each fold model outputs raw logits.
Temperature scaling learns a single scalar T per fold (and a global T
across all folds) such that  prob = sigmoid(logit / T)  produces
well-calibrated probabilities — a predicted score of 0.85 should
correspond to roughly 85 % of candidates actually being planets.

Algorithm
---------
1. Load each fold checkpoint (exonet_fold_1.pt … exonet_fold_5.pt).
2. Run the model on the provided validation set to collect raw logits.
3. Optimise T on [0.1, 10.0] by minimising NLL (BCEWithLogitsLoss).
4. Compute ECE before and after calibration.
5. Persist results to ``calibration.json``.

Usage
-----
::

    from pathlib import Path
    import numpy as np
    import torch
    from ml.calibrate import calibrate_folds

    result = calibrate_folds(
        fold_checkpoint_paths=[Path(f"out/exonet_fold_{k}.pt") for k in range(1, 6)],
        global_views=gv,       # np.ndarray (N, 2, 2001)
        local_views=lv,        # np.ndarray (N, 201)
        odd_views=ov,          # np.ndarray (N, 201)
        even_views=ev,         # np.ndarray (N, 201)
        secondary_views=sv,    # np.ndarray (N, 201)
        centroid_views=cv,     # np.ndarray (N, 201)
        scalars=sc,            # np.ndarray (N, 13)
        labels=y,              # np.ndarray (N,)  binary 0/1
        device=torch.device("cpu"),
        output_dir=Path("out"),
    )
    print(result["global_temperature"])
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


from ml.model import ExoNet

__all__ = ["calibrate_folds"]

# L12 NOTE: array inputs to calibrate_folds are in cache format (N, L) without
# channel dimension.  The channel dimension is added internally in _collect_logits.

# Number of ECE bins used for reliability diagram binning.
_ECE_BINS: int = 15


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_checkpoint(path: Path, device: torch.device) -> ExoNet:
    """
    Load an ExoNet checkpoint, trying use_se=True first then use_se=False.

    Returns the model in eval mode on ``device``.
    """
    state_dict = torch.load(str(path), map_location=device, weights_only=True)

    model = ExoNet(use_se=True)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Key mismatch — first try use_se=False (checkpoint may have been trained
        # without SE blocks); only fall back to strict=False as a last resort to
        # avoid silently zero-initialising SE weights on a use_se=False checkpoint.
        try:
            model = ExoNet(use_se=False)
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model = ExoNet(use_se=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(
                    f"  [calibrate] Warning: {len(missing)} keys missing when loading "
                    f"{path.name} — {missing[:3]}{'...' if len(missing) > 3 else ''}",
                    flush=True,
                )
            if unexpected:
                print(
                    f"  [calibrate] Warning: {len(unexpected)} unexpected keys in "
                    f"{path.name} (likely SE blocks) — ignored.",
                    flush=True,
                )

    model.to(device)
    model.eval()
    return model


# ── Logit collection ──────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_logits(
    model: ExoNet,
    global_views: np.ndarray,    # (N, 2, 2001)
    local_views: np.ndarray,     # (N, 201)
    odd_views: np.ndarray,       # (N, 201)
    even_views: np.ndarray,      # (N, 201)
    secondary_views: np.ndarray, # (N, 201)
    centroid_views: np.ndarray,  # (N, 201)
    scalars: np.ndarray,         # (N, 17) — or (N, 13) for legacy caches, auto-padded to 17
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Run ``model`` over the validation data in mini-batches and return a
    1-D float32 tensor of raw logits on CPU.

    NOTE: inputs are in cache format (N, L) without channel dimension.
    The channel dimension is added internally via ``unsqueeze(1)``.

    API contract (Issue 6.1 / CF12): inputs are expected in cache format — i.e.
    WITHOUT a channel dimension for local/odd/even/secondary/centroid views.
    The channel dimension (dim=1) is added internally via ``unsqueeze(1)``.
    This differs from the internal ``(N, 1, 201)`` convention used by
    ``ExoNetEnsemble`` (ensemble.py), which operates on already-unsqueezed
    tensors.  Callers must pass ``(N, 201)`` arrays here.

    Parameters
    ----------
    global_views    : (N, 2, 2001) — 2-channel global view; if shape is (N, 2001)
                      (legacy 1-channel), it is automatically expanded.
    local_views     : (N, 201)  — cache format, NO channel dimension
    odd_views       : (N, 201)  — cache format, NO channel dimension
    even_views      : (N, 201)  — cache format, NO channel dimension
    secondary_views : (N, 201)  — cache format, NO channel dimension
    centroid_views  : (N, 201)  — cache format, NO channel dimension
    scalars         : (N, 13)   — padded to 13 if fewer columns present
    """
    n = len(global_views)
    all_logits: list[torch.Tensor] = []

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)

        # Handle both 2-channel (N,2,2001) and legacy 1-channel (N,2001) global views
        if global_views.ndim == 2:
            # WARNING: legacy 1-channel cache detected.  Channel 1 is duplicated from
            # channel 0, but the model's BN statistics for channel 1 were learned on
            # prenorm (lower-variance) data.  Calibration logits may be inaccurate for
            # checkpoints trained on 2-channel caches.
            print(
                "[calibrate] WARNING: legacy 1-channel global_views detected — "
                "channel 1 duplicated from channel 0; calibration may be inaccurate "
                "for 2-channel model checkpoints.",
                flush=True,
            )
            gv_batch = torch.from_numpy(global_views[s:e].astype(np.float32)).unsqueeze(1).to(device)
            gv_batch = gv_batch.expand(-1, 2, -1).contiguous()  # repeat channel; contiguous() avoids non-contiguous tensor errors in conv layers
        else:
            gv_batch = torch.from_numpy(global_views[s:e].astype(np.float32)).to(device)  # (B, 2, 2001)

        # Handle both new 2-channel (N,2,201) and legacy 1-channel (N,201) formats.
        # For legacy caches, duplicate channel 0 → channel 1 (prenorm ≈ detrended).
        def _to_2ch(arr: np.ndarray, b_s: int, b_e: int) -> torch.Tensor:
            slc = arr[b_s:b_e].astype(np.float32)
            if slc.ndim == 2:  # (B, 201) legacy
                slc = np.stack([slc, slc], axis=1)  # (B, 2, 201)
            return torch.from_numpy(slc).to(device)

        lv  = _to_2ch(local_views,     s, e)   # (B, 2, 201)
        ov  = _to_2ch(odd_views,       s, e)   # (B, 2, 201)
        ev  = _to_2ch(even_views,      s, e)   # (B, 2, 201)
        sv  = _to_2ch(secondary_views, s, e)   # (B, 2, 201)
        cv  = torch.from_numpy(centroid_views[s:e].astype(np.float32)).unsqueeze(1).to(device)

        # Pad scalars to SCALAR_FEATURES (17) if needed for backward compat with old caches
        sc_batch = scalars[s:e].astype(np.float32)
        if sc_batch.shape[1] < 17:
            sc_batch = np.concatenate([sc_batch, np.zeros((len(sc_batch), 17 - sc_batch.shape[1]))], axis=1)
        sc = torch.from_numpy(sc_batch).to(device)

        logits = model(gv_batch, lv, ov, ev, sv, cv, sc).squeeze(1)
        all_logits.append(logits.cpu())

    return torch.cat(all_logits)            # (N,)


# ── Temperature optimisation ──────────────────────────────────────────────────

def _nll_for_temperature(
    temperature: float,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute binary NLL (BCEWithLogitsLoss) after dividing logits by T.
    """
    scaled = logits / temperature
    loss = nn.functional.binary_cross_entropy_with_logits(
        scaled, labels, reduction="mean"
    )
    return float(loss.item())


def _find_best_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Minimise NLL over T ∈ [0.1, 10.0] using Brent's method.

    Returns the optimal scalar temperature as a Python float.
    """
    result = minimize_scalar(
        _nll_for_temperature,
        bounds=(0.1, 10.0),
        method="bounded",
        args=(logits, labels),
        options={"xatol": 1e-5, "maxiter": 500},
    )
    return float(result.x)


# ── Expected Calibration Error ────────────────────────────────────────────────

def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = _ECE_BINS,
) -> float:
    """
    Compute ECE as the weighted mean absolute difference between predicted
    confidence and empirical accuracy across equal-width probability bins.

    Parameters
    ----------
    probs  : predicted probabilities in [0, 1], shape (N,)
    labels : binary ground-truth labels, shape (N,)
    n_bins : number of equal-width bins

    Returns
    -------
    ECE as a float in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Include the right edge only in the last bin.
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)

        if mask.sum() == 0:
            continue

        bin_n       = mask.sum()
        bin_conf    = probs[mask].mean()
        bin_acc     = labels[mask].mean()
        ece        += (bin_n / n) * abs(bin_conf - bin_acc)

    return float(ece)


# ── Reliability diagram helper ────────────────────────────────────────────────

def _reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute data for a reliability diagram.

    Returns a dict with mean predicted probability, fraction of positives,
    and sample count per bin (empty bins are omitted).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred: list[float] = []
    frac_pos: list[float]  = []
    counts: list[int]      = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() > 0:
            mean_pred.append(float(probs[mask].mean()))
            frac_pos.append(float(labels[mask].mean()))
            counts.append(int(mask.sum()))
    return {
        "mean_predicted_prob": mean_pred,
        "fraction_positive": frac_pos,
        "bin_counts": counts,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def calibrate_folds(
    fold_checkpoint_paths: list[Path],
    global_views: np.ndarray,
    local_views: np.ndarray,
    odd_views: np.ndarray,
    even_views: np.ndarray,
    secondary_views: np.ndarray,
    centroid_views: np.ndarray,
    scalars: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    output_dir: Path,
    manifest_path: Path,
) -> dict:
    """
    Calibrate all ExoNet fold checkpoints via temperature scaling, using only
    data each model never trained on.

    The split comes from train.py's ``split_manifest.npz``. For each fold k:
    1. Loads fold k's model.
    2. Collects raw logits on fold k's validation stars only (out-of-fold).
    3. Optimises a per-fold temperature T_k by minimising NLL on them.

    A global temperature is fit on the pooled out-of-fold logits. ECE is
    reported on the pooled out-of-fold predictions before (T=1) and after
    (per-fold T). The held-out test set is never touched.

    The results are saved to ``output_dir/calibration.json``.

    Parameters
    ----------
    fold_checkpoint_paths :
        Ordered list of checkpoint paths, one per fold
        (e.g. ``[Path("out/exonet_fold_1.pt"), …]``).
    global_views    : np.ndarray, shape (N, 2, 2001)  — 2-channel global view
    local_views     : np.ndarray, shape (N, 201)
    odd_views       : np.ndarray, shape (N, 201)
    even_views      : np.ndarray, shape (N, 201)
    secondary_views : np.ndarray, shape (N, 201)
    centroid_views  : np.ndarray, shape (N, 201)
    scalars         : np.ndarray, shape (N, 17) — or (N, 13) for legacy caches, auto-padded to 17
    labels          : np.ndarray, shape (N,), binary 0/1
    device          : torch.device to run inference on.
    output_dir      : Directory where ``calibration.json`` is written.

    Returns
    -------
    dict with keys:
        ``global_temperature``  — float
        ``fold_temperatures``   — list[float], one per checkpoint
        ``ece_before``          — float
        ``ece_after``           — float
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = np.load(manifest_path)

    # Temperatures keyed by 1-based fold number so missing folds don't shift
    # later folds onto the wrong temperature.
    fold_temperatures: dict[str, float] = {}
    oof_logits: list[torch.Tensor] = []     # each fold's model on its own val stars
    oof_scaled: list[torch.Tensor] = []
    oof_labels: list[np.ndarray] = []

    for k, ckpt_path in enumerate(fold_checkpoint_paths, start=1):
        ckpt_path = Path(ckpt_path)
        key = f"fold{k}_val_idx"
        if not ckpt_path.exists() or key not in manifest.files:
            print(f"[calibrate] Fold {k}: no checkpoint or no {key} in manifest — skipping.", flush=True)
            continue

        idx = np.sort(manifest[key])        # sorted: cheap reads from memory-mapped caches
        y = np.asarray(labels[idx])
        print(f"[calibrate] Fold {k}: {ckpt_path.name} on {len(idx)} out-of-fold samples ...", flush=True)
        model = _load_checkpoint(ckpt_path, device)
        logits = _collect_logits(
            model, global_views[idx], local_views[idx], odd_views[idx], even_views[idx],
            secondary_views[idx], centroid_views[idx], scalars[idx], device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        T = _find_best_temperature(logits, torch.from_numpy(y.astype(np.float32)))
        fold_temperatures[str(k)] = T
        print(f"[calibrate] Fold {k}: optimal T = {T:.4f}", flush=True)
        oof_logits.append(logits)
        oof_scaled.append(logits / T)
        oof_labels.append(y)

    if not oof_logits:
        print("[calibrate] WARNING: no valid fold checkpoints found — returning defaults.", flush=True)
        return {"global_temperature": 1.0, "fold_temperatures": fold_temperatures,
                "ece_before": float("nan"), "ece_after": float("nan")}

    labels = np.concatenate(oof_labels)
    all_logits = torch.cat(oof_logits)
    stacked_probs_before = torch.sigmoid(all_logits).numpy()
    stacked_probs_after  = torch.sigmoid(torch.cat(oof_scaled)).numpy()
    ece_before = _expected_calibration_error(stacked_probs_before, labels, n_bins=_ECE_BINS)
    ece_after  = _expected_calibration_error(stacked_probs_after,  labels, n_bins=_ECE_BINS)
    print(f"[calibrate] Out-of-fold ECE before: {ece_before:.4f}  after: {ece_after:.4f}", flush=True)

    global_T = _find_best_temperature(all_logits, torch.from_numpy(labels.astype(np.float32)))
    print(f"[calibrate] Global temperature T = {global_T:.4f}", flush=True)

    # ── Persist results ───────────────────────────────────────────────────────
    results = {
        "global_temperature": global_T,
        "fold_temperatures":  fold_temperatures,
        "ece_before":         ece_before,
        "ece_after":          ece_after,
    }

    calib_path = output_dir / "calibration.json"
    with open(calib_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"[calibrate] Saved calibration data to {calib_path}", flush=True)

    # G6: save reliability diagram data for paper figures.
    try:
        rd_before = _reliability_diagram_data(stacked_probs_before, labels, n_bins=_ECE_BINS)
        rd_after  = _reliability_diagram_data(stacked_probs_after,  labels, n_bins=_ECE_BINS)
        reliability_data = {
            "before_calibration": rd_before,
            "after_calibration":  rd_after,
            "ece_before": ece_before,
            "ece_after":  ece_after,
            "n_bins": _ECE_BINS,
        }
        rd_path = output_dir / "reliability_diagram.json"
        with open(rd_path, "w", encoding="utf-8") as fh:
            json.dump(reliability_data, fh, indent=2)
        print(f"[calibrate] Saved reliability diagram data to {rd_path}", flush=True)
    except Exception as exc:
        print(f"[calibrate] WARNING: reliability diagram save failed ({exc})", flush=True)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Temperature-scale calibrate ExoNet fold checkpoints")
    p.add_argument("--cache-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--manifest", type=Path, default=None,
                   help="split_manifest.npz written by train.py (default: <output-dir>/split_manifest.npz)")
    args = p.parse_args()
    # load cache arrays and call calibrate_folds(...)
    data = np.load(args.cache_file, mmap_mode="r")
    N = len(data["labels"])
    fold_checkpoint_paths = [
        args.output_dir / f"exonet_fold_{k}.pt"
        for k in range(1, args.folds + 1)
    ]
    calibrate_folds(
        fold_checkpoint_paths=fold_checkpoint_paths,
        global_views=data["global_views"],
        local_views=data["local_views"],
        odd_views=data["odd_views"] if "odd_views" in data else np.zeros((N, 201), dtype=np.float32),
        even_views=data["even_views"] if "even_views" in data else np.zeros((N, 201), dtype=np.float32),
        secondary_views=data["secondary_views"] if "secondary_views" in data else np.zeros((N, 201), dtype=np.float32),
        centroid_views=data["centroid_views"] if "centroid_views" in data else np.zeros((N, 201), dtype=np.float32),
        scalars=data["scalars"],
        labels=data["labels"],
        output_dir=args.output_dir,
        device=torch.device(args.device),
        manifest_path=args.manifest or args.output_dir / "split_manifest.npz",
    )
