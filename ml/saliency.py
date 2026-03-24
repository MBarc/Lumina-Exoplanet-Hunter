"""
Gradient-based saliency maps for ExoNet transit detection.

This module computes two complementary explainability signals for a single
TransitCandidate processed by a loaded ExoNet model:

  1. Vanilla gradient saliency
     Backpropagate d(logit)/d(input) through the full network with respect to
     the global_view and local_view tensors.  The absolute gradients are
     smoothed with a 1-D Gaussian filter and normalised to [0, 1].

  2. GradCAM-style saliency on the last conv block
     Register forward/backward hooks on the final ResConvBlock of each CNN
     branch, pool the gradients across the length dimension to obtain channel
     weights, form a weighted average of the feature maps, and upsample the
     resulting 1-D activation map back to the original input length via linear
     interpolation.

Both signals highlight which phase bins most influenced the model's decision,
making them suitable for inclusion in reports submitted to NASA and other
exoplanet science organisations.

Public API
----------
SaliencyResult   — dataclass holding both saliency arrays plus logit/prob
compute_saliency — main entry point for a single TransitCandidate
load_model_for_saliency — convenience loader that returns an eval-mode ExoNet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    # Avoid circular import at runtime; types are only needed for annotations.
    from ml.model import ExoNet
    from ml.preprocess import TransitCandidate


__all__ = [
    "SaliencyResult",
    "compute_saliency",
    "load_model_for_saliency",
]

# Gaussian smoothing sigmas (in phase-bin units)
_GLOBAL_SIGMA: float = 3.0
_LOCAL_SIGMA: float  = 2.0


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class SaliencyResult:
    """
    Saliency maps produced by :func:`compute_saliency` for one transit
    candidate.

    All arrays are normalised to [0, 1]; higher values indicate phase bins
    that most strongly drove the model's transit-detection logit.

    Attributes
    ----------
    global_saliency : np.ndarray, shape (2001,)
        Smoothed absolute vanilla gradient on the global (full-orbit) view.
    local_saliency : np.ndarray, shape (201,)
        Smoothed absolute vanilla gradient on the local (transit-zoom) view.
    global_gradcam : np.ndarray, shape (2001,)
        GradCAM activation map upsampled from the last GlobalBranch conv block.
    local_gradcam : np.ndarray, shape (201,)
        GradCAM activation map upsampled from the last LocalBranch conv block.
    logit : float
        Raw model output (pre-sigmoid).
    probability : float
        Transit probability — sigmoid(logit).
    """

    global_saliency: np.ndarray   # shape (2001,) normalised [0, 1]
    local_saliency:  np.ndarray   # shape (201,)  normalised [0, 1]
    global_gradcam:  np.ndarray   # shape (2001,) normalised [0, 1]
    local_gradcam:   np.ndarray   # shape (201,)  normalised [0, 1]
    logit:       float            # raw model logit
    probability: float            # sigmoid(logit)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_01(arr: np.ndarray) -> np.ndarray:
    """Linearly scale *arr* so that its range is [0, 1].

    If the array is constant (max == min) the function returns an all-zeros
    array of the same shape rather than dividing by zero.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _candidate_to_tensors(
    candidate: "TransitCandidate",
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a TransitCandidate into the three input tensors expected by ExoNet.

    Returns
    -------
    global_t  : (1, 1, 2001) float32 tensor requiring grad
    local_t   : (1, 1, 201)  float32 tensor requiring grad
    scalars_t : (1, 6)       float32 tensor (no grad needed)
    """
    global_t = torch.tensor(
        candidate.global_view[np.newaxis, np.newaxis, :],  # (1, 1, 2001)
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)

    local_t = torch.tensor(
        candidate.local_view[np.newaxis, np.newaxis, :],   # (1, 1, 201)
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)

    scalar_vals = np.array([
        candidate.period,
        candidate.duration,
        candidate.depth,
        candidate.bls_power,
        candidate.secondary_depth,
        candidate.odd_even_diff,
    ], dtype=np.float32)

    scalars_t = torch.tensor(
        scalar_vals[np.newaxis, :],  # (1, 6)
        dtype=torch.float32,
        device=device,
    )

    return global_t, local_t, scalars_t


# ── Vanilla gradient saliency ─────────────────────────────────────────────────

def _vanilla_gradient_saliency(
    model: "ExoNet",
    global_t: torch.Tensor,
    local_t: torch.Tensor,
    scalars_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute d(logit)/d(global_view) and d(logit)/d(local_view) via a single
    backward pass.

    Parameters
    ----------
    model     : ExoNet in eval mode with gradients enabled on input tensors
    global_t  : (1, 1, 2001) tensor with requires_grad=True
    local_t   : (1, 1, 201)  tensor with requires_grad=True
    scalars_t : (1, 6) tensor

    Returns
    -------
    g_sal : raw absolute gradient for global view, shape (2001,)
    l_sal : raw absolute gradient for local view,  shape (201,)
    logit : scalar float
    """
    logit_t = model(global_t, local_t, scalars_t)   # (1, 1)
    logit_scalar = logit_t.squeeze()                  # scalar

    # Backpropagate gradient of the logit w.r.t. both input tensors.
    logit_scalar.backward()

    g_grad = global_t.grad.detach().cpu().numpy()    # (1, 1, 2001)
    l_grad = local_t.grad.detach().cpu().numpy()     # (1, 1, 201)

    g_sal = np.abs(g_grad[0, 0, :])                  # (2001,)
    l_sal = np.abs(l_grad[0, 0, :])                  # (201,)
    logit = float(logit_scalar.detach().cpu().item())

    return g_sal, l_sal, logit


# ── GradCAM on last ResConvBlock ──────────────────────────────────────────────

class _GradCAMHook:
    """
    Attach forward and backward hooks to a single nn.Module.

    Usage::

        hook = _GradCAMHook(layer)
        output = model(...)
        output.backward()
        cam = hook.activation_map()   # numpy array, shape (length,)
        hook.remove()
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._features: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = module.register_forward_hook(self._save_features)
        self._bwd_handle = module.register_full_backward_hook(self._save_gradients)

    # ---- hook callbacks ----

    def _save_features(
        self,
        _module: torch.nn.Module,
        _input: tuple,
        output: torch.Tensor,
    ) -> None:
        # output shape: (batch, channels, length)
        self._features = output.detach()

    def _save_gradients(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        # grad_output[0] shape: (batch, channels, length)
        self._gradients = grad_output[0].detach()

    # ---- public ----

    def activation_map(self, target_length: int) -> np.ndarray:
        """
        Compute the GradCAM activation map and upsample to *target_length*.

        Returns a 1-D numpy array of shape (target_length,) with values in
        their natural (un-normalised) range.  Normalisation is done by the
        caller so that it can be applied uniformly with vanilla saliency.
        """
        if self._features is None or self._gradients is None:
            raise RuntimeError(
                "_GradCAMHook: forward and backward passes must both have "
                "completed before calling activation_map()."
            )

        features  = self._features[0].cpu().numpy()   # (channels, length)
        gradients = self._gradients[0].cpu().numpy()  # (channels, length)

        # Global-average-pool gradients over the length dimension → per-channel weights
        weights = gradients.mean(axis=1)              # (channels,)

        # Weighted sum of feature maps over the channel dimension
        cam = np.einsum("c,cl->l", weights, features) # (length,)

        # ReLU: keep only activations that positively influence the logit
        cam = np.maximum(cam, 0.0)

        # Upsample from the feature-map length to the original input length
        feature_len = cam.shape[0]
        src_x = np.linspace(0, target_length - 1, feature_len)
        dst_x = np.arange(target_length)
        cam_upsampled = np.interp(dst_x, src_x, cam)

        return cam_upsampled

    def remove(self) -> None:
        """Detach both hooks from the module."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _gradcam_for_branch(
    model: "ExoNet",
    global_t: torch.Tensor,
    local_t: torch.Tensor,
    scalars_t: torch.Tensor,
    branch_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run a forward + backward pass with GradCAM hooks on both branches
    simultaneously.

    Parameters
    ----------
    branch_name : str
        One of ``"global_branch"`` or ``"local_branch"``.  The function
        attaches hooks to both branches in a single pass, which is more
        efficient than two separate passes.

    Returns
    -------
    g_cam : GradCAM map for global branch, shape (2001,), un-normalised
    l_cam : GradCAM map for local branch,  shape (201,),  un-normalised
    logit : raw scalar logit
    """
    # The last element of each nn.Sequential is the final ResConvBlock.
    last_global_block = model.global_branch.blocks[-1]
    last_local_block  = model.local_branch.blocks[-1]

    g_hook = _GradCAMHook(last_global_block)
    l_hook = _GradCAMHook(last_local_block)

    try:
        logit_t = model(global_t, local_t, scalars_t)
        logit_scalar = logit_t.squeeze()
        logit_scalar.backward()

        g_cam = g_hook.activation_map(target_length=2001)
        l_cam = l_hook.activation_map(target_length=201)
        logit = float(logit_scalar.detach().cpu().item())
    finally:
        g_hook.remove()
        l_hook.remove()

    return g_cam, l_cam, logit


# ── Public API ────────────────────────────────────────────────────────────────

def compute_saliency(
    model: "ExoNet",
    candidate: "TransitCandidate",
    device: torch.device,
) -> SaliencyResult:
    """
    Compute gradient saliency and GradCAM for a single transit candidate.

    The model is run in evaluation mode with ``torch.no_grad()`` disabled so
    that gradients can flow through BatchNorm and Dropout layers (which are
    frozen in eval mode, making their output deterministic).

    Two separate backward passes are performed:
      - Pass 1: vanilla gradient saliency (d logit / d input)
      - Pass 2: GradCAM with forward/backward hooks on the last conv blocks

    Parameters
    ----------
    model :
        A fully loaded ExoNet instance (PyTorch, not ONNX).  Use
        :func:`load_model_for_saliency` to obtain one from a checkpoint.
    candidate :
        A single :class:`~ml.preprocess.TransitCandidate` produced by the
        preprocessing pipeline.
    device :
        The torch device on which to run inference (CPU or CUDA).

    Returns
    -------
    SaliencyResult
        Dataclass containing four normalised saliency arrays plus the raw
        logit and sigmoid probability.
    """
    model.eval()
    model.to(device)

    # ── Pass 1: vanilla gradient saliency ────────────────────────────────────
    print("[saliency] Pass 1 — vanilla gradient saliency …", flush=True)

    global_t1, local_t1, scalars_t1 = _candidate_to_tensors(candidate, device)

    g_raw, l_raw, logit = _vanilla_gradient_saliency(
        model, global_t1, local_t1, scalars_t1
    )

    # Smooth with Gaussian filter, then normalise
    g_smoothed = gaussian_filter1d(g_raw, sigma=_GLOBAL_SIGMA)
    l_smoothed = gaussian_filter1d(l_raw, sigma=_LOCAL_SIGMA)

    global_saliency = _normalise_01(g_smoothed)
    local_saliency  = _normalise_01(l_smoothed)

    print(
        f"[saliency]   logit={logit:.4f}  "
        f"prob={float(torch.sigmoid(torch.tensor(logit))):.4f}",
        flush=True,
    )

    # ── Pass 2: GradCAM on last conv blocks ───────────────────────────────────
    print("[saliency] Pass 2 — GradCAM …", flush=True)

    # Re-create tensors: gradients from Pass 1 have already been accumulated
    # and the graph was freed by the first backward() call.
    global_t2, local_t2, scalars_t2 = _candidate_to_tensors(candidate, device)

    g_cam_raw, l_cam_raw, _ = _gradcam_for_branch(
        model, global_t2, local_t2, scalars_t2, branch_name="both"
    )

    global_gradcam = _normalise_01(g_cam_raw)
    local_gradcam  = _normalise_01(l_cam_raw)

    print("[saliency] Done.", flush=True)

    probability = float(torch.sigmoid(torch.tensor(logit)).item())

    return SaliencyResult(
        global_saliency=global_saliency,
        local_saliency=local_saliency,
        global_gradcam=global_gradcam,
        local_gradcam=local_gradcam,
        logit=logit,
        probability=probability,
    )


def load_model_for_saliency(
    checkpoint_path: Path,
    device: torch.device,
    use_se: bool = False,
) -> "ExoNet":
    """
    Load a PyTorch checkpoint and return an ExoNet ready for saliency
    computation.

    The checkpoint must have been saved via ``torch.save(model.state_dict(),
    path)`` or as a dict with a ``"model_state_dict"`` key (the format written
    by ``ml/train.py``).

    Parameters
    ----------
    checkpoint_path :
        Path to a ``.pt`` or ``.pth`` checkpoint file.
    device :
        Target device for inference.
    use_se :
        Unused — kept for forward-compatibility with future ExoNet variants
        that add squeeze-excitation blocks.  ExoNet v2.0 always uses
        ``use_se=False``.

    Returns
    -------
    ExoNet
        Model in eval mode, weights loaded, moved to *device*.
    """
    # Lazy import to avoid a hard dependency when the module is imported at
    # the top level (e.g., from the dashboard).
    from ml.model import ExoNet  # noqa: PLC0415

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        f"[saliency] Loading checkpoint: {checkpoint_path}",
        flush=True,
    )

    raw = torch.load(checkpoint_path, map_location=device)

    # Support both bare state-dicts and training checkpoints produced by
    # ml/train.py (which wrap the state dict in a larger dict).
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        epoch = raw.get("epoch", "?")
        print(
            f"[saliency]   Training checkpoint detected (epoch {epoch}).",
            flush=True,
        )
    else:
        state_dict = raw

    model = ExoNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("[saliency]   Model loaded and set to eval mode.", flush=True)
    return model
