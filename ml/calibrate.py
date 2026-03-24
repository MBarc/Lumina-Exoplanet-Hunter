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
        global_views=gv,    # np.ndarray (N, 2001)
        local_views=lv,     # np.ndarray (N, 201)
        scalars=sc,         # np.ndarray (N, 6)
        labels=y,           # np.ndarray (N,)  binary 0/1
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

# Number of ECE bins used for reliability diagram binning.
_ECE_BINS: int = 15


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_checkpoint(path: Path, device: torch.device) -> ExoNet:
    """
    Load an ExoNet checkpoint, trying use_se=True first then use_se=False.

    The ExoNet constructor in model.py does not expose a ``use_se`` flag in
    its current form, but older checkpoints may have been saved with Squeeze-
    Excitation blocks present.  We handle this by attempting a strict load
    first and, on key-mismatch, falling back to a non-strict load so that
    the architecture in model.py is used as-is.

    Returns the model in eval mode on ``device``.
    """
    state_dict = torch.load(str(path), map_location=device, weights_only=True)

    model = ExoNet()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # Key mismatch — load what we can (e.g. checkpoint trained with SE
        # blocks that are absent in the current architecture definition).
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
    global_views: np.ndarray,
    local_views: np.ndarray,
    scalars: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Run ``model`` over the validation data in mini-batches and return a
    1-D float32 tensor of raw logits on CPU.

    Parameters
    ----------
    global_views : (N, 2001)
    local_views  : (N, 201)
    scalars      : (N, 6)
    """
    n = len(global_views)
    all_logits: list[torch.Tensor] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        gv = torch.from_numpy(
            global_views[start:end].astype(np.float32)
        ).unsqueeze(1).to(device)          # (B, 1, 2001)

        lv = torch.from_numpy(
            local_views[start:end].astype(np.float32)
        ).unsqueeze(1).to(device)          # (B, 1, 201)

        sc = torch.from_numpy(
            scalars[start:end].astype(np.float32)
        ).to(device)                       # (B, 6)

        logits = model(gv, lv, sc).squeeze(1)   # (B,)
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


# ── Public API ────────────────────────────────────────────────────────────────

def calibrate_folds(
    fold_checkpoint_paths: list[Path],
    global_views: np.ndarray,
    local_views: np.ndarray,
    scalars: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    output_dir: Path,
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
    global_views : np.ndarray, shape (N, 2001)
    local_views  : np.ndarray, shape (N, 201)
    scalars      : np.ndarray, shape (N, 6)
    labels       : np.ndarray, shape (N,), binary 0/1
    device       : torch.device to run inference on.
    output_dir   : Directory where ``calibration.json`` is written.

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

    label_tensor = torch.from_numpy(labels.astype(np.float32))

    fold_logits: list[torch.Tensor] = []
    fold_temperatures: list[float] = []

    for k, ckpt_path in enumerate(fold_checkpoint_paths):
        ckpt_path = Path(ckpt_path)
        fold_num  = k + 1

        if not ckpt_path.exists():
            print(
                f"[calibrate] Fold {fold_num}: checkpoint not found at "
                f"{ckpt_path} — skipping.",
                flush=True,
            )
            fold_logits.append(torch.zeros(len(labels)))
            fold_temperatures.append(1.0)
            continue

        print(f"[calibrate] Fold {fold_num}: loading {ckpt_path.name} …", flush=True)
        model  = _load_checkpoint(ckpt_path, device)
        logits = _collect_logits(model, global_views, local_views, scalars, device)
        fold_logits.append(logits)

        # Release GPU memory immediately.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        T = _find_best_temperature(logits, label_tensor)
        fold_temperatures.append(T)
        print(f"[calibrate] Fold {fold_num}: optimal T = {T:.4f}", flush=True)

    # ── ECE before calibration (T = 1 for every fold) ─────────────────────────
    # Average fold probabilities using raw logits (T=1).
    stacked_probs_before = torch.stack(
        [torch.sigmoid(lg) for lg in fold_logits], dim=0
    ).mean(dim=0).numpy()   # (N,)

    ece_before = _expected_calibration_error(
        stacked_probs_before, labels, n_bins=_ECE_BINS
    )
    print(f"[calibrate] ECE before calibration: {ece_before:.4f}", flush=True)

    # ── ECE after per-fold calibration ────────────────────────────────────────
    stacked_probs_after = torch.stack(
        [torch.sigmoid(lg / T) for lg, T in zip(fold_logits, fold_temperatures)],
        dim=0,
    ).mean(dim=0).numpy()   # (N,)

    ece_after = _expected_calibration_error(
        stacked_probs_after, labels, n_bins=_ECE_BINS
    )
    print(f"[calibrate] ECE after  calibration: {ece_after:.4f}", flush=True)

    # ── Global temperature (pooled logits) ────────────────────────────────────
    all_logits   = torch.cat(fold_logits)                           # (N * K,)
    all_labels   = label_tensor.repeat(len(fold_logits))            # (N * K,)
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
    return results
