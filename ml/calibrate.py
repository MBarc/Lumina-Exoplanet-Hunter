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

from sklearn.model_selection import train_test_split

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
    test_size: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Calibrate all ExoNet fold checkpoints via temperature scaling.

    For each fold checkpoint the function:
    1. Loads the model.
    2. Collects raw logits on the supplied validation data.
    3. Optimises a per-fold temperature T_k by minimising NLL.

    A global temperature is then found by pooling all fold logits.

    ECE is computed once before calibration (using T=1 for all folds) and
    once after (using the per-fold temperatures), and both are reported.

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

    # X-2: exclude held-out test set from calibration — replicate the same
    # train/test split used in train.py so calibration only uses train+val samples.
    # IMPORTANT: train.py computes test_size as max(int(0.10*N), 10) — an integer —
    # so we must do the same; using a float fraction produces a different split on
    # datasets where int(frac*N) != round(frac*N).
    all_indices = np.arange(len(labels))
    n_test = max(int(test_size * len(labels)), 10)
    train_idx, _ = train_test_split(
        all_indices,
        test_size=n_test,
        random_state=seed,
        # IMPORTANT: this 0.5 must match _POSITIVE_LABEL_THRESHOLD in train.py.
        stratify=(labels >= 0.5).astype(int),
    )
    global_views     = global_views[train_idx]
    local_views      = local_views[train_idx]
    odd_views        = odd_views[train_idx]
    even_views       = even_views[train_idx]
    secondary_views  = secondary_views[train_idx]
    centroid_views   = centroid_views[train_idx]
    scalars          = scalars[train_idx]
    labels           = labels[train_idx]

    label_tensor = torch.from_numpy(labels.astype(np.float32))

    fold_logits: list[torch.Tensor] = []
    fold_keys: list[str] = []   # parallel to fold_logits; tracks 1-based fold number string
    # Store temperatures keyed by 1-based fold number so missing folds do not
    # shift indices and cause wrong temperatures to be applied to later folds.
    fold_temperatures: dict[str, float] = {}
    # M9: track which folds were successfully loaded vs skipped.
    valid_fold_mask: list[bool] = []

    for k, ckpt_path in enumerate(fold_checkpoint_paths):
        ckpt_path = Path(ckpt_path)
        fold_num  = k + 1
        fold_key  = str(fold_num)

        if not ckpt_path.exists():
            print(
                f"[calibrate] Fold {fold_num}: checkpoint not found at "
                f"{ckpt_path} — skipping.",
                flush=True,
            )
            # M9: store a placeholder so list lengths stay consistent, but mark as invalid.
            fold_logits.append(torch.zeros(len(labels)))
            fold_keys.append(fold_key)
            valid_fold_mask.append(False)
            continue

        print(f"[calibrate] Fold {fold_num}: loading {ckpt_path.name} ...", flush=True)
        model  = _load_checkpoint(ckpt_path, device)
        logits = _collect_logits(
            model, global_views, local_views, odd_views, even_views,
            secondary_views, centroid_views, scalars, device,
        )
        fold_logits.append(logits)
        fold_keys.append(fold_key)
        valid_fold_mask.append(True)

        # Release GPU memory immediately.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        T = _find_best_temperature(logits, label_tensor)
        fold_temperatures[fold_key] = T
        print(f"[calibrate] Fold {fold_num}: optimal T = {T:.4f}", flush=True)

    # ── ECE before calibration (T = 1 for every fold) ─────────────────────────
    # M9: exclude skipped folds from ECE computation — their zero logits (0.5 prob)
    # would pollute the calibration metrics.
    valid_logits = [lg for lg, v in zip(fold_logits, valid_fold_mask) if v]
    if not valid_logits:
        print("[calibrate] WARNING: no valid fold checkpoints found — returning defaults.", flush=True)
        return {"global_temperature": 1.0, "fold_temperatures": fold_temperatures,
                "ece_before": float("nan"), "ece_after": float("nan")}

    stacked_probs_before = torch.stack(
        [torch.sigmoid(lg) for lg in valid_logits], dim=0
    ).mean(dim=0).numpy()   # (N,)

    ece_before = _expected_calibration_error(
        stacked_probs_before, labels, n_bins=_ECE_BINS
    )
    print(f"[calibrate] ECE before calibration: {ece_before:.4f}", flush=True)

    # ── ECE after per-fold calibration ────────────────────────────────────────
    # M9: only use valid folds for after-calibration ECE.
    # Use fold_keys to look up each fold's temperature from the dict — missing folds
    # have no entry in fold_temperatures and are excluded via valid_fold_mask.
    valid_pairs = [
        (lg, fold_temperatures[fk])
        for lg, fk, v in zip(fold_logits, fold_keys, valid_fold_mask) if v
    ]
    stacked_probs_after = torch.stack(
        [torch.sigmoid(lg / T) for lg, T in valid_pairs],
        dim=0,
    ).mean(dim=0).numpy()   # (N,)

    ece_after = _expected_calibration_error(
        stacked_probs_after, labels, n_bins=_ECE_BINS
    )
    print(f"[calibrate] ECE after  calibration: {ece_after:.4f}", flush=True)

    # ── Global temperature (pooled valid logits only) ─────────────────────────
    # M9: exclude skipped folds' zero logits from global temperature estimation.
    all_logits   = torch.cat(valid_logits)                              # (N * K_valid,)
    all_labels   = label_tensor.repeat(len(valid_logits))               # (N * K_valid,)
    global_T     = _find_best_temperature(all_logits, all_labels)
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
    p.add_argument("--test-size", type=float, default=0.10,
               help="Test fraction held out (must match --test-size in training run; default 0.10). "
                    "Mismatch will cause calibration to run on the wrong subset.")
    p.add_argument("--seed", type=int, default=42,
               help="Random seed for test split (must match --seed in training run; default 42). "
                    "Mismatch will cause calibration to run on the wrong subset.")
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
        test_size=args.test_size,
        seed=args.seed,
    )
