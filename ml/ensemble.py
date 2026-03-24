"""
Fold-ensemble inference for ExoNet.

Loads all cross-validation fold checkpoints, optionally applies temperature
scaling from a ``calibration.json`` file, and averages calibrated
probabilities across folds to produce a final transit score with an
associated uncertainty estimate (std across folds).

Typical usage
-------------
::

    from pathlib import Path
    from ml.ensemble import ExoNetEnsemble
    from ml.preprocess import preprocess

    ens = ExoNetEnsemble.from_output_dir(Path("training_output"))
    candidates = preprocess("kepler_lc.fits")
    if candidates:
        result = ens.predict_one(candidates[0])
        print(result["score"], result["uncertainty"])

Output dictionary keys
----------------------
score       — float in [0, 1], mean of calibrated fold probabilities
fold_scores — list[float] of per-fold probabilities (len = number of folds)
uncertainty — float, std of fold probabilities (measure of ensemble spread)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.model import ExoNet
from ml.preprocess import TransitCandidate

__all__ = ["ExoNetEnsemble"]


# ── Checkpoint loading helper ─────────────────────────────────────────────────

def _load_fold_model(path: Path, device: torch.device) -> ExoNet | None:
    """
    Load a single fold checkpoint into an ExoNet model.

    Tries strict=True first.  If there is a key-mismatch (e.g. checkpoint
    was trained with Squeeze-Excitation blocks that are absent from the
    current architecture) it falls back to strict=False so that every key
    that *does* match is loaded and the rest is silently ignored.

    Returns ``None`` if the file does not exist, and logs a warning.
    """
    path = Path(path)
    if not path.exists():
        print(
            f"[ensemble] Warning: checkpoint not found at {path} — skipping.",
            flush=True,
        )
        return None

    state_dict = torch.load(str(path), map_location=device, weights_only=True)
    model = ExoNet()

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"[ensemble] Warning: {len(missing)} keys missing in "
                f"{path.name} — {missing[:3]}{'...' if len(missing) > 3 else ''}",
                flush=True,
            )
        if unexpected:
            print(
                f"[ensemble] Warning: {len(unexpected)} unexpected keys in "
                f"{path.name} (ignored — likely SE blocks).",
                flush=True,
            )

    model.to(device)
    model.eval()
    return model


# ── Candidate → tensor helpers ────────────────────────────────────────────────

def _candidate_to_tensors(
    candidate: TransitCandidate,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (global_view, local_view, scalar_features) tensors for one candidate."""
    gv = torch.from_numpy(
        candidate.global_view.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)      # (1, 1, 2001)

    lv = torch.from_numpy(
        candidate.local_view.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)      # (1, 1, 201)

    sc_raw = np.array(
        [
            candidate.period,
            candidate.duration,
            candidate.depth,
            candidate.bls_power,
            candidate.secondary_depth,
            candidate.odd_even_diff,
        ],
        dtype=np.float32,
    )
    sc = torch.from_numpy(sc_raw).unsqueeze(0).to(device)   # (1, 6)

    return gv, lv, sc


def _candidates_to_tensors(
    candidates: list[TransitCandidate],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batched (global_view, local_view, scalar_features) tensors."""
    gv = torch.from_numpy(
        np.stack([c.global_view.astype(np.float32) for c in candidates])
    ).unsqueeze(1).to(device)    # (N, 1, 2001)

    lv = torch.from_numpy(
        np.stack([c.local_view.astype(np.float32) for c in candidates])
    ).unsqueeze(1).to(device)    # (N, 1, 201)

    sc_list = [
        np.array(
            [
                c.period,
                c.duration,
                c.depth,
                c.bls_power,
                c.secondary_depth,
                c.odd_even_diff,
            ],
            dtype=np.float32,
        )
        for c in candidates
    ]
    sc = torch.from_numpy(np.stack(sc_list)).to(device)     # (N, 6)

    return gv, lv, sc


# ── Ensemble class ────────────────────────────────────────────────────────────

class ExoNetEnsemble:
    """
    Ensemble of ExoNet fold models with optional temperature scaling.

    Parameters
    ----------
    checkpoint_dir :
        Directory that contains ``exonet_fold_1.pt`` … ``exonet_fold_5.pt``.
        Folds whose checkpoint files are missing are skipped with a warning.
    calibration_path :
        Path to a ``calibration.json`` file produced by
        ``ml.calibrate.calibrate_folds``.  When *None* no temperature
        scaling is applied (equivalent to T = 1 for all folds).
    device :
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, …).
    """

    # Expected checkpoint filenames inside checkpoint_dir.
    _FOLD_GLOB = "exonet_fold_*.pt"
    _N_FOLDS   = 5

    def __init__(
        self,
        checkpoint_dir: Path,
        calibration_path: Path | None,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._models: list[ExoNet]  = []
        self._temperatures: list[float] = []

        # ── Load calibration data ──────────────────────────────────────────────
        fold_temperatures: list[float] | None = None
        if calibration_path is not None:
            calibration_path = Path(calibration_path)
            if calibration_path.exists():
                with open(calibration_path, "r", encoding="utf-8") as fh:
                    calib = json.load(fh)
                fold_temperatures = calib.get("fold_temperatures")
                print(
                    f"[ensemble] Loaded calibration from {calibration_path} "
                    f"(global T = {calib.get('global_temperature', 'N/A')})",
                    flush=True,
                )
            else:
                print(
                    f"[ensemble] Warning: calibration file not found at "
                    f"{calibration_path} — using T = 1.0 for all folds.",
                    flush=True,
                )

        # ── Load fold checkpoints ──────────────────────────────────────────────
        checkpoint_dir = Path(checkpoint_dir)
        for k in range(1, self._N_FOLDS + 1):
            ckpt_path = checkpoint_dir / f"exonet_fold_{k}.pt"
            model     = _load_fold_model(ckpt_path, self._device)
            if model is None:
                continue    # missing file — skip this fold

            # Temperature for this fold (1-indexed list from calibration.json)
            if fold_temperatures is not None and (k - 1) < len(fold_temperatures):
                T = float(fold_temperatures[k - 1])
            else:
                T = 1.0

            self._models.append(model)
            self._temperatures.append(T)
            print(
                f"[ensemble] Loaded fold {k} from {ckpt_path.name}  T={T:.4f}",
                flush=True,
            )

        if not self._models:
            raise RuntimeError(
                f"No fold checkpoints could be loaded from {checkpoint_dir}. "
                "Ensure exonet_fold_*.pt files are present."
            )

        print(
            f"[ensemble] Ready — {len(self._models)} fold(s) loaded.",
            flush=True,
        )

    # ── Single-candidate prediction ───────────────────────────────────────────

    @torch.no_grad()
    def predict_one(self, candidate: TransitCandidate) -> dict:
        """
        Score a single transit candidate.

        Parameters
        ----------
        candidate :
            A ``TransitCandidate`` produced by ``ml.preprocess.preprocess``.

        Returns
        -------
        dict with keys:
            ``score``       — float in [0, 1], mean calibrated probability
            ``fold_scores`` — list[float], one per loaded fold
            ``uncertainty`` — float, std of fold probabilities
        """
        gv, lv, sc = _candidate_to_tensors(candidate, self._device)
        fold_probs: list[float] = []

        for model, T in zip(self._models, self._temperatures):
            logit = model(gv, lv, sc)           # (1, 1)
            prob  = torch.sigmoid(logit / T)    # (1, 1)
            fold_probs.append(float(prob.item()))

        arr = np.array(fold_probs, dtype=np.float64)
        return {
            "score":       float(arr.mean()),
            "fold_scores": fold_probs,
            "uncertainty": float(arr.std()),
        }

    # ── Batch prediction ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_batch(self, candidates: list[TransitCandidate]) -> list[dict]:
        """
        Score a list of transit candidates.

        Candidates are processed in a single forward pass per fold so that
        batch normalisation statistics are properly computed across the full
        batch.

        Parameters
        ----------
        candidates :
            Non-empty list of ``TransitCandidate`` objects.

        Returns
        -------
        list[dict]
            One result dict per candidate, each with keys ``score``,
            ``fold_scores``, and ``uncertainty``.  Preserves input order.

        Raises
        ------
        ValueError
            If ``candidates`` is empty.
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")

        gv, lv, sc = _candidates_to_tensors(candidates, self._device)
        # fold_probs_matrix[fold_idx][candidate_idx] = probability
        fold_probs_matrix: list[np.ndarray] = []

        for model, T in zip(self._models, self._temperatures):
            logits = model(gv, lv, sc)              # (N, 1)
            probs  = torch.sigmoid(logits / T)      # (N, 1)
            fold_probs_matrix.append(
                probs.squeeze(1).cpu().numpy().astype(np.float64)
            )

        # Stack into (n_folds, N) matrix.
        mat = np.stack(fold_probs_matrix, axis=0)   # (n_folds, N)

        results: list[dict] = []
        for i in range(len(candidates)):
            col = mat[:, i]
            results.append(
                {
                    "score":       float(col.mean()),
                    "fold_scores": col.tolist(),
                    "uncertainty": float(col.std()),
                }
            )
        return results

    # ── Convenience constructor ───────────────────────────────────────────────

    @staticmethod
    def from_output_dir(output_dir: Path, device: str = "cpu") -> "ExoNetEnsemble":
        """
        Construct an ``ExoNetEnsemble`` from a training output directory.

        Looks for ``exonet_fold_*.pt`` files directly inside ``output_dir``
        and for an optional ``calibration.json`` in the same directory.

        Parameters
        ----------
        output_dir :
            Directory produced by the training script, containing fold
            checkpoints and optionally ``calibration.json``.
        device :
            PyTorch device string (``"cpu"``, ``"cuda"``, …).

        Returns
        -------
        ExoNetEnsemble
        """
        output_dir       = Path(output_dir)
        calibration_path = output_dir / "calibration.json"

        return ExoNetEnsemble(
            checkpoint_dir   = output_dir,
            calibration_path = calibration_path if calibration_path.exists() else None,
            device           = device,
        )
