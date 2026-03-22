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

ONNX export
-----------
To convert a trained PyTorch checkpoint to ONNX call the static helper::

    ExoNetInference.export_from_pytorch("exonet.pt", "exonet.onnx")

The resulting ONNX graph has three named inputs:
  - ``global_view``     — float32 tensor, shape (N, 1, 2001)
  - ``local_view``      — float32 tensor, shape (N, 1, 201)
  - ``scalar_features`` — float32 tensor, shape (N, 4)
                          raw values [period_days, duration_days,
                          depth_fractional, bls_power]; log1p normalisation
                          is applied inside the model graph
and one output:
  - ``score``           — float32 tensor, shape (N, 1)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only used for type annotations; never imported at runtime from this module.
    import onnxruntime as ort  # noqa: F401

from ml.preprocess import TransitCandidate

# Default model path used by Lumina's installer.
_DEFAULT_MODEL_PATH = Path(r"C:\Program Files\Lumina\Data\models\exonet.onnx")


class ExoNetInference:
    """
    Thin wrapper around an ONNX Runtime inference session for ExoNet.

    Parameters
    ----------
    model_path :
        Path to the ``exonet.onnx`` file.  Defaults to the standard Lumina
        installation path
        ``C:\\Program Files\\Lumina\\Data\\models\\exonet.onnx``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.
    """

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        import onnxruntime as ort  # deferred so PyTorch is never required

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run ExoNetInference.export_from_pytorch() to create it, "
                "or install Lumina to get the pre-built model."
            )

        # Use CPU execution provider by default; GPU is picked up automatically
        # when onnxruntime-gpu is installed and CUDA is available.
        self._session: ort.InferenceSession = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    # ── Single-candidate inference ────────────────────────────────────────────

    def predict_one(self, candidate: TransitCandidate) -> float:
        """
        Score a single transit candidate.

        Parameters
        ----------
        candidate :
            A ``TransitCandidate`` produced by ``ml.preprocess.preprocess``.
            Its ``global_view`` (shape 2001), ``local_view`` (shape 201), and
            scalar attributes ``period``, ``duration``, ``depth``, and
            ``bls_power`` are used as model inputs.

        Returns
        -------
        float
            Transit probability in [0, 1].
        """
        global_arr = candidate.global_view.astype(np.float32).reshape(1, 1, 2001)
        local_arr  = candidate.local_view.astype(np.float32).reshape(1, 1, 201)
        scalar_arr = np.array(
            [candidate.period, candidate.duration, candidate.depth, candidate.bls_power],
            dtype=np.float32,
        ).reshape(1, 4)

        outputs = self._session.run(
            None,
            {
                "global_view":     global_arr,
                "local_view":      local_arr,
                "scalar_features": scalar_arr,
            },
        )
        # outputs[0] has shape (1, 1); extract scalar
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

        global_arr = np.stack(
            [c.global_view.astype(np.float32) for c in candidates]
        ).reshape(-1, 1, 2001)  # (N, 1, 2001)

        local_arr = np.stack(
            [c.local_view.astype(np.float32) for c in candidates]
        ).reshape(-1, 1, 201)   # (N, 1, 201)

        scalar_arr = np.stack(
            [
                np.array(
                    [c.period, c.duration, c.depth, c.bls_power],
                    dtype=np.float32,
                )
                for c in candidates
            ]
        )  # (N, 4)

        outputs = self._session.run(
            None,
            {
                "global_view":     global_arr,
                "local_view":      local_arr,
                "scalar_features": scalar_arr,
            },
        )
        # outputs[0] has shape (N, 1)
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
        The export uses ``opset_version=18`` and sets all three inputs
        (``global_view``, ``local_view``, ``scalar_features``) and the output
        (``score``) as dynamic-batch axes so the ONNX graph accepts any batch
        size.  The ``scalar_features`` input carries raw values; log1p
        normalisation is baked into the exported graph via ``ScalarBranch``.
        """
        import torch  # noqa: PLC0415  (deferred import intentional)
        from ml.model import ExoNet  # noqa: PLC0415

        pt_path   = Path(pt_path)
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        model = ExoNet()
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Dummy inputs for tracing — batch size 1
        dummy_global  = torch.zeros(1, 1, 2001, dtype=torch.float32)
        dummy_local   = torch.zeros(1, 1, 201,  dtype=torch.float32)
        dummy_scalar  = torch.zeros(1, 4,        dtype=torch.float32)

        torch.onnx.export(
            model,
            (dummy_global, dummy_local, dummy_scalar),
            str(onnx_path),
            opset_version=18,
            input_names=["global_view", "local_view", "scalar_features"],
            output_names=["score"],
            dynamic_axes={
                "global_view":     {0: "batch_size"},
                "local_view":      {0: "batch_size"},
                "scalar_features": {0: "batch_size"},
                "score":           {0: "batch_size"},
            },
        )
        print(f"Exported ONNX model to {onnx_path}")
