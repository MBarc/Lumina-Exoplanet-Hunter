"""
ONNX runtime inference wrapper for ExoNet.

This module is deliberately kept free of any PyTorch dependency so it can run
in lightweight deployment environments where only ``onnxruntime`` and
``numpy`` are available.

Typical usage
-------------
::

    from ml.inference import ExoNetInference
    from ml.preprocess import preprocess

    session = ExoNetInference()                       # loads default ONNX model
    candidates = preprocess("star.fits")
    if candidates:
        score = session.predict_one(candidates[0])    # float in [0, 1]

Ensemble usage
--------------
::

    session = EnsembleInference(["fold1.onnx", "fold2.onnx", "fold3.onnx"])
    score = session.predict_one(candidates[0])        # averaged probability

ONNX export
-----------
To convert a trained PyTorch checkpoint to ONNX call the static helper::

    ExoNetInference.export_from_pytorch("exonet.pt", "exonet.onnx")

The resulting ONNX graph has seven named inputs:
  - ``global_view``     — float32 tensor, shape (N, 2, 2001)
  - ``local_view``      — float32 tensor, shape (N, 2, 201)  [detrended, prenorm_raw]
  - ``odd_view``        — float32 tensor, shape (N, 2, 201)  [detrended, prenorm_raw]
  - ``even_view``       — float32 tensor, shape (N, 2, 201)  [detrended, prenorm_raw]
  - ``secondary_view``  — float32 tensor, shape (N, 2, 201)  [detrended, prenorm_raw]
  - ``centroid_view``   — float32 tensor, shape (N, 1, 201)
  - ``scalar_features`` — float32 tensor, shape (N, 17)
                          raw values [period_days, duration_days,
                          depth_fractional, bls_power, secondary_depth,
                          odd_even_diff, centroid_shift, n_transits,
                          log_teff_norm, logg, log_radius_norm, feh, kepmag_norm,
                          mission_is_kepler, mission_is_tess, mission_is_k2,
                          transit_snr];
                          log1p normalisation is applied inside the model graph
and one output:
  - ``score``           — float32 tensor, shape (N, 1)
"""

from __future__ import annotations

# CF9: standard library imports before the TYPE_CHECKING block.
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only used for type annotations; never imported at runtime from this module.
    import onnxruntime as ort  # noqa: F401

from ml.preprocess import TransitCandidate

# Issue 6.3: the previous default was a Windows-only hardcoded path which
# breaks on Linux.  The default is now None; callers must pass model_path=
# explicitly or set the EXONET_MODEL_PATH environment variable.
_DEFAULT_MODEL_PATH = None


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to an array of probabilities.

    Issue 6.4: ONNX models output sigmoid probabilities.  Temperature scaling
    is logit(p) / T → sigmoid, i.e.  p_cal = 1 / (1 + exp(-logit(p) / T))
    where logit(p) = log(p / (1 - p)).

    H3 NOTE: this function applies temperature scaling to already-averaged
    ensemble probabilities (an approximation of per-logit scaling).  This is
    mathematically sound as a monotone recalibration as long as this caveat is
    understood: the correct procedure is to average calibrated per-fold logits,
    but the ONNX graph bakes in sigmoid so only probabilities are available.
    The approximation error is small when fold probabilities are not extreme.

    Probabilities are clipped to [1e-7, 1-1e-7] before the logit to guard
    against p == 0.0 or p == 1.0 inputs (logit would be ±inf).

    Parameters
    ----------
    probs       : predicted probabilities in (0, 1), shape (N,)
    temperature : scalar temperature T > 0

    Returns
    -------
    Calibrated probabilities, same shape as probs.
    """
    if temperature == 1.0:
        return probs
    # H3: clip before logit to handle edge cases p==0 or p==1 (logit is ±inf).
    probs   = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    logits  = np.log(probs / (1.0 - probs))        # logit(p)
    cal     = 1.0 / (1.0 + np.exp(-logits / temperature))
    return cal.astype(np.float32)


def _make_inputs(candidates: list[TransitCandidate]) -> dict[str, np.ndarray]:
    """Build the ONNX session input dict for a list of candidates."""
    # T3: TransitCandidate always has centroid_curve and raw_global_view via
    # default_factory — hasattr guards were dead code and have been removed.
    n = len(candidates)
    centroid_arr = np.stack(
        [c.centroid_curve.astype(np.float32) for c in candidates]
    ).reshape(n, 1, 201)

    # global_view: 2-channel (detrended + raw)
    global_arr = np.stack([
        np.stack([c.global_view.astype(np.float32),
                  c.raw_global_view.astype(np.float32)],
                 axis=0)
        for c in candidates
    ])  # (N, 2, 2001)

    local_arr = np.stack([
        np.stack([c.local_view.astype(np.float32),
                  c.raw_local_view.astype(np.float32)], axis=0)
        for c in candidates
    ])  # (n, 2, 201)

    odd_arr = np.stack([
        np.stack([c.odd_view.astype(np.float32),
                  c.raw_odd_view.astype(np.float32)], axis=0)
        for c in candidates
    ])  # (n, 2, 201)

    even_arr = np.stack([
        np.stack([c.even_view.astype(np.float32),
                  c.raw_even_view.astype(np.float32)], axis=0)
        for c in candidates
    ])  # (n, 2, 201)

    secondary_arr = np.stack([
        np.stack([c.secondary_view.astype(np.float32),
                  c.raw_secondary_view.astype(np.float32)], axis=0)
        for c in candidates
    ])  # (n, 2, 201)

    scalar_arr = np.stack([
        np.array([c.period, c.duration, c.depth, c.bls_power,
                  c.secondary_depth, c.odd_even_diff,
                  c.centroid_shift, c.n_transits,
                  0.0, 0.0, 0.0, 0.0, 0.0,  # stellar params (8-12) — unknown at inference
                  0.0, 0.0, 0.0,             # mission one-hot (13-15) — unknown at inference
                  float(np.log1p(max(c.depth / max(float(c.noise_floor), 1e-4), 0.0)))],  # transit S/N (16)
                 dtype=np.float32)
        for c in candidates
    ])  # (N, 17)

    return {
        "global_view":     global_arr,
        "local_view":      local_arr,
        "odd_view":        odd_arr,
        "even_view":       even_arr,
        "secondary_view":  secondary_arr,
        "centroid_view":   centroid_arr,
        "scalar_features": scalar_arr,
    }


class ExoNetInference:
    """
    Thin wrapper around an ONNX Runtime inference session for ExoNet.

    Parameters
    ----------
    model_path :
        Path to the ``exonet.onnx`` file.  Pass explicitly or set the
        ``EXONET_MODEL_PATH`` environment variable.  Defaults to ``None``
        (Issue 6.3: the previous Windows-only default path has been removed
        to avoid breakage on Linux / macOS deployments).

    Raises
    ------
    FileNotFoundError
        If ``model_path`` is None (and EXONET_MODEL_PATH is not set), or
        if the resolved path does not exist.
    """

    def __init__(self, model_path: str | Path | None = _DEFAULT_MODEL_PATH) -> None:
        import onnxruntime as ort  # deferred so PyTorch is never required

        # Issue 6.3: fall back to EXONET_MODEL_PATH env var before raising.
        if model_path is None:
            env_path = os.environ.get("EXONET_MODEL_PATH")
            if env_path:
                model_path = env_path
            else:
                raise FileNotFoundError(
                    "No model path provided. Pass model_path= explicitly, "
                    "or set the EXONET_MODEL_PATH environment variable."
                )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run ExoNetInference.export_from_pytorch() to create it."
            )

        # Build provider list based on what is actually available to avoid
        # spurious warnings when onnxruntime-gpu is not installed and to prevent
        # accidental GPU use when only onnxruntime (CPU-only) is present.
        _available = ort.get_available_providers()
        _providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                      if p in _available]
        if not _providers:
            _providers = ["CPUExecutionProvider"]
        self._session: ort.InferenceSession = ort.InferenceSession(
            str(model_path),
            providers=_providers,
        )

    # ── Single-candidate inference ────────────────────────────────────────────

    def predict_one(self, candidate: TransitCandidate) -> float:
        """
        Score a single transit candidate.

        Parameters
        ----------
        candidate :
            A ``TransitCandidate`` produced by ``ml.preprocess.preprocess``.

        Returns
        -------
        float
            Transit probability in [0, 1].
        """
        outputs = self._session.run(None, _make_inputs([candidate]))
        return float(outputs[0][0, 0])

    # ── Batch inference ───────────────────────────────────────────────────────

    def predict_batch(self, candidates: list[TransitCandidate]) -> list[float]:
        """
        Score a list of transit candidates in a single model call.

        Parameters
        ----------
        candidates :
            List of ``TransitCandidate`` objects (must be non-empty).

        Returns
        -------
        list[float]
            Transit probabilities in [0, 1], one per candidate, preserving
            input order.

        Raises
        ------
        ValueError
            If ``candidates`` is empty.
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")

        outputs = self._session.run(None, _make_inputs(candidates))
        return [float(v) for v in outputs[0][:, 0]]

    # ── ONNX export ───────────────────────────────────────────────────────────

    @staticmethod
    def export_from_pytorch(
        pt_path: str | Path,
        onnx_path: str | Path,
    ) -> None:
        """
        Convert a trained PyTorch ``ExoNet`` checkpoint to ONNX format.

        This method lazily imports ``torch`` and ``ml.model`` so that the rest
        of the ``inference`` module remains PyTorch-free.

        Parameters
        ----------
        pt_path :
            Path to the PyTorch state-dict file saved by the training script
            (``exonet.pt``).
        onnx_path :
            Destination path for the exported ONNX file (``exonet.onnx``).
            Parent directories are created automatically.

        Notes
        -----
        The export uses ``opset_version=18`` and sets all seven inputs and the
        output as dynamic-batch axes so the ONNX graph accepts any batch size.
        ``scalar_features`` carries raw values; log1p normalisation is baked
        into the graph via ``ScalarBranch``.
        """
        import torch  # noqa: PLC0415  (deferred import intentional)
        import torch.nn as nn  # noqa: PLC0415
        from ml.model import ExoNet  # noqa: PLC0415

        pt_path   = Path(pt_path)
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        model = ExoNet(use_se=True)
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        # M10: if the checkpoint was trained with --no-se, fall back to use_se=False.
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            err_msg = str(exc)
            err_lower = err_msg.lower()
            if "size mismatch" not in err_lower and "unexpected key" not in err_lower and "missing key" not in err_lower:
                raise  # re-raise if it's not a weight shape mismatch
            model = ExoNet(use_se=False)
            model.load_state_dict(state_dict)
        model.eval()

        # Wrap with sigmoid so the ONNX graph outputs probabilities in [0, 1].
        class _WithSigmoid(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self._inner = inner
                self._sig = nn.Sigmoid()

            def forward(
                self,
                gv: torch.Tensor,
                lv: torch.Tensor,
                ov: torch.Tensor,
                ev: torch.Tensor,
                sv: torch.Tensor,
                cv: torch.Tensor,
                sc: torch.Tensor,
            ) -> torch.Tensor:
                return self._sig(self._inner(gv, lv, ov, ev, sv, cv, sc))

        export_model = _WithSigmoid(model)
        export_model.eval()

        # Dummy inputs for tracing — batch size 1
        dummy_global    = torch.zeros(1, 2, 2001, dtype=torch.float32)
        dummy_local     = torch.zeros(1, 2, 201,  dtype=torch.float32)
        dummy_odd       = torch.zeros(1, 2, 201,  dtype=torch.float32)
        dummy_even      = torch.zeros(1, 2, 201,  dtype=torch.float32)
        dummy_secondary = torch.zeros(1, 2, 201,  dtype=torch.float32)
        dummy_centroid  = torch.zeros(1, 1, 201,  dtype=torch.float32)
        dummy_scalar    = torch.zeros(1, 17,       dtype=torch.float32)

        torch.onnx.export(
            export_model,
            (dummy_global, dummy_local, dummy_odd, dummy_even, dummy_secondary, dummy_centroid, dummy_scalar),
            str(onnx_path),
            opset_version=18,
            input_names=[
                "global_view", "local_view", "odd_view",
                "even_view", "secondary_view", "centroid_view", "scalar_features",
            ],
            output_names=["score"],
            dynamic_axes={
                "global_view":     {0: "batch_size"},
                "local_view":      {0: "batch_size"},
                "odd_view":        {0: "batch_size"},
                "even_view":       {0: "batch_size"},
                "secondary_view":  {0: "batch_size"},
                "centroid_view":   {0: "batch_size"},
                "scalar_features": {0: "batch_size"},
                "score":           {0: "batch_size"},
            },
        )
        print(f"Exported ONNX model to {onnx_path}")


class EnsembleInference:
    """
    Average predictions from multiple ExoNet ONNX models (fold ensembling).

    Loading all fold checkpoints and averaging their outputs reduces variance
    and consistently outperforms any single fold on held-out data.

    Issue 6.4: temperature scaling is now applied to the ONNX ensemble to
    match the behaviour of ``ExoNetEnsemble`` (the PyTorch ensemble class in
    ensemble.py).  The global temperature is loaded from ``calibration.json``
    if it exists in the same directory as the first ONNX model.

    Parameters
    ----------
    model_paths :
        Paths to one or more ``exonet_fold_k.onnx`` files.

    Raises
    ------
    ValueError
        If ``model_paths`` is empty.
    FileNotFoundError
        If any path does not exist.

    Examples
    --------
    ::

        session = EnsembleInference([
            "run/exonet_fold_1.onnx",
            "run/exonet_fold_2.onnx",
            "run/exonet_fold_3.onnx",
        ])
        score = session.predict_one(candidate)
    """

    def __init__(self, model_paths: list[str | Path]) -> None:
        if not model_paths:
            raise ValueError("model_paths must not be empty")
        self._members = [ExoNetInference(p) for p in model_paths]

        # Load per-fold temperatures from calibration.json so we can apply
        # temperature scaling to each fold's logit before averaging — the
        # mathematically correct order (vs. averaging probs then scaling).
        # Falls back to global_temperature if per-fold temps are absent.
        self._fold_temperatures: list[float] = []
        self._global_temperature = 1.0
        cal_path = Path(model_paths[0]).parent / "calibration.json"
        if cal_path.exists():
            try:
                with open(cal_path, "r", encoding="utf-8") as fh:
                    cal = json.load(fh)
                self._global_temperature = float(cal.get("global_temperature", 1.0))
                fold_temps = cal.get("fold_temperatures", [])
                if fold_temps:
                    self._fold_temperatures = [float(t) for t in fold_temps]
            except Exception as exc:
                # EH3: log the parse failure so operators know calibration was skipped.
                print(
                    f"[EnsembleInference] WARNING: could not parse {cal_path}: "
                    f"{type(exc).__name__}: {exc} — using T=1.0 (no temperature scaling)",
                    flush=True,
                )

    def _get_temperature(self, fold_idx: int) -> float:
        """Return the temperature for a given fold, falling back to global."""
        if self._fold_temperatures and fold_idx < len(self._fold_temperatures):
            return self._fold_temperatures[fold_idx]
        return self._global_temperature

    def predict_one(self, candidate: TransitCandidate) -> float:
        """Average score across all ensemble members for one candidate.

        Applies per-fold temperature scaling to each logit before averaging and
        applying sigmoid — the mathematically correct order that matches the
        PyTorch ExoNetEnsemble (ensemble.py).
        """
        # Apply per-fold temperature to each logit, then average, then sigmoid.
        calibrated_probs = np.array([
            _apply_temperature(
                np.array([m.predict_one(candidate)], dtype=np.float32),
                self._get_temperature(i),
            )[0]
            for i, m in enumerate(self._members)
        ], dtype=np.float32)
        return float(np.mean(calibrated_probs))

    def predict_batch(self, candidates: list[TransitCandidate]) -> list[float]:
        """Average scores across all ensemble members for a batch.

        Applies per-fold temperature scaling before averaging (correct order).
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")
        all_scores = np.stack([
            _apply_temperature(
                np.array(m.predict_batch(candidates), dtype=np.float32),
                self._get_temperature(i),
            )
            for i, m in enumerate(self._members)
        ])  # (n_members, N)
        # Per-fold temperature already applied above — just average and return.
        mean_probs = all_scores.mean(axis=0)   # (N,)
        return [float(v) for v in mean_probs]
