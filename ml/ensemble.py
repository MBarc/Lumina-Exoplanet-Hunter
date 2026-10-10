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

Output dictionary keys (predict_one / predict_batch)
-----------------------------------------------------
score       — float in [0, 1], mean of calibrated fold probabilities
fold_scores — list[float] of per-fold probabilities (len = number of folds)
uncertainty — float, std of fold probabilities (measure of ensemble spread)

Additional methods
------------------
predict_one_mc  — MC Dropout epistemic uncertainty (n stochastic passes per fold)
predict_one_tta — Test-Time Augmentation via circular phase shifts (reduces phase-alignment variance)
"""

from __future__ import annotations

import json
import re
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

    # Fix 8: mirror the three-level fallback from calibrate.py.
    # 1. Try use_se=True strict — matches checkpoints trained with SE.
    # 2. Try use_se=False strict — matches checkpoints trained with --no-se.
    # 3. Fall back to use_se=True non-strict — partial key match as a last resort.
    # This prevents silently zero-initialising SE weights when loading a --no-se checkpoint.
    model = ExoNet(use_se=True)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        try:
            model = ExoNet(use_se=False)
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model = ExoNet(use_se=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[ensemble] WARNING: loaded {path.name} with strict=False", flush=True)
            # Check if any complete model branches are missing — they would be
            # zero-initialised, producing silently wrong inference for those inputs.
            _CRITICAL_BRANCHES = ("global_branch", "local_branch", "centroid_branch",
                                  "secondary_branch", "odd_branch", "even_branch",
                                  "scalar_branch", "fusion")
            uninit = [b for b in _CRITICAL_BRANCHES if any(b in k for k in missing)]
            if uninit:
                print(
                    f"[ensemble] CRITICAL: the following branches are ZERO-INITIALISED "
                    f"because their weights are absent from {path.name}: {uninit}. "
                    "Inference quality will be severely degraded.",
                    flush=True,
                )

    model.to(device)
    model.eval()
    return model


# ── Candidate → tensor helpers ────────────────────────────────────────────────

def _candidate_to_tensors(
    candidate: TransitCandidate,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (global_view, local_view, odd_view, even_view, secondary_view, centroid_view, scalar_features) tensors."""
    # T3: TransitCandidate always has raw_global_view and centroid_curve via
    # default_factory — hasattr guards were dead code and have been removed.
    gv = torch.from_numpy(
        np.stack([candidate.global_view.astype(np.float32),
                  candidate.raw_global_view.astype(np.float32)], axis=0)
    ).unsqueeze(0).to(device)    # (1, 2, 2001)

    lv = torch.from_numpy(
        np.stack([candidate.local_view.astype(np.float32),
                  candidate.raw_local_view.astype(np.float32)], axis=0)
    ).unsqueeze(0).to(device)    # (1, 2, 201)
    ov = torch.from_numpy(
        np.stack([candidate.odd_view.astype(np.float32),
                  candidate.raw_odd_view.astype(np.float32)], axis=0)
    ).unsqueeze(0).to(device)    # (1, 2, 201)
    ev = torch.from_numpy(
        np.stack([candidate.even_view.astype(np.float32),
                  candidate.raw_even_view.astype(np.float32)], axis=0)
    ).unsqueeze(0).to(device)    # (1, 2, 201)
    sv = torch.from_numpy(
        np.stack([candidate.secondary_view.astype(np.float32),
                  candidate.raw_secondary_view.astype(np.float32)], axis=0)
    ).unsqueeze(0).to(device)    # (1, 2, 201)
    cv = torch.from_numpy(candidate.centroid_curve.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    transit_snr = float(np.log1p(
        max(candidate.depth / max(float(candidate.noise_floor), 1e-4), 0.0)
    ))
    sc_raw = np.array([
        candidate.period, candidate.duration, candidate.depth, candidate.bls_power,
        candidate.secondary_depth, candidate.odd_even_diff,
        candidate.centroid_shift, candidate.n_transits,
        0.0, 0.0, 0.0, 0.0, 0.0,  # stellar params (8-12) — unknown at inference
        0.0, 0.0, 0.0,             # mission one-hot (13-15) — unknown at inference
        transit_snr,               # transit S/N (16) — computable from candidate
    ], dtype=np.float32)
    sc = torch.from_numpy(sc_raw).unsqueeze(0).to(device)
    return gv, lv, ov, ev, sv, cv, sc


def _candidates_to_tensors(
    candidates: list[TransitCandidate],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batched (global_view, local_view, odd_view, even_view, secondary_view, centroid_view, scalar_features) tensors."""
    # T3: TransitCandidate always has raw_global_view via default_factory.
    gv = torch.from_numpy(
        np.stack([
            np.stack([
                c.global_view.astype(np.float32),
                c.raw_global_view.astype(np.float32),
            ], axis=0)
            for c in candidates
        ])
    ).to(device)    # (N, 2, 2001)

    lv = torch.from_numpy(
        np.stack([
            np.stack([c.local_view.astype(np.float32),
                      c.raw_local_view.astype(np.float32)], axis=0)
            for c in candidates
        ])
    ).to(device)    # (N, 2, 201)

    ov = torch.from_numpy(
        np.stack([
            np.stack([c.odd_view.astype(np.float32),
                      c.raw_odd_view.astype(np.float32)], axis=0)
            for c in candidates
        ])
    ).to(device)    # (N, 2, 201)

    ev = torch.from_numpy(
        np.stack([
            np.stack([c.even_view.astype(np.float32),
                      c.raw_even_view.astype(np.float32)], axis=0)
            for c in candidates
        ])
    ).to(device)    # (N, 2, 201)

    sv = torch.from_numpy(
        np.stack([
            np.stack([c.secondary_view.astype(np.float32),
                      c.raw_secondary_view.astype(np.float32)], axis=0)
            for c in candidates
        ])
    ).to(device)    # (N, 2, 201)

    # T3: TransitCandidate always has centroid_curve via default_factory.
    cv = torch.from_numpy(
        np.stack([c.centroid_curve.astype(np.float32) for c in candidates])
    ).unsqueeze(1).to(device)    # (N, 1, 201)

    sc_list = [
        np.array(
            [
                c.period, c.duration, c.depth, c.bls_power,
                c.secondary_depth, c.odd_even_diff,
                c.centroid_shift, c.n_transits,
                0.0, 0.0, 0.0, 0.0, 0.0,  # stellar params (8-12) — unknown at inference
                0.0, 0.0, 0.0,             # mission one-hot (13-15) — unknown at inference
                float(np.log1p(max(c.depth / max(float(c.noise_floor), 1e-4), 0.0))),  # transit S/N (16)
            ],
            dtype=np.float32,
        )
        for c in candidates
    ]
    sc = torch.from_numpy(np.stack(sc_list)).to(device)     # (N, 17)

    return gv, lv, ov, ev, sv, cv, sc


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
    # L7: renamed from _N_FOLDS to _MAX_FOLDS — the actual number of loaded folds
    # may be less if some checkpoints are missing.
    _MAX_FOLDS = 5

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
        fold_temperatures: dict[str, float] | list[float] | None = None
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
        # Discover all available fold checkpoints dynamically so that folds
        # beyond _MAX_FOLDS are not silently dropped when more folds were trained.
        fold_paths = sorted(checkpoint_dir.glob(self._FOLD_GLOB))
        if not fold_paths:
            # Fall back to numbered scan up to _MAX_FOLDS for backwards compat.
            fold_paths = [
                checkpoint_dir / f"exonet_fold_{k}.pt"
                for k in range(1, self._MAX_FOLDS + 1)
                if (checkpoint_dir / f"exonet_fold_{k}.pt").exists()
            ]
        for fold_path in fold_paths:
            model = _load_fold_model(fold_path, self._device)
            if model is None:
                continue    # missing file — skip this fold

            # Temperature for this fold: parse fold index from filename using regex.
            # Filenames use 1-based fold numbers (e.g. exonet_fold_1.pt → index 0).
            m = re.search(r'fold[_-]?(\d+)', fold_path.stem)
            if m is None:
                print(f"[ExoNetEnsemble] WARNING: could not parse fold index from '{fold_path.name}'; defaulting to temperature index 0.", flush=True)
                fold_idx = 0
            else:
                fold_idx = int(m.group(1)) - 1
            if fold_temperatures is not None:
                # Support both the new dict format ({"1": T, "3": T, ...}) and the
                # legacy list format ([T0, T1, ...]) produced by older calibrate runs.
                if isinstance(fold_temperatures, dict):
                    fold_key = str(fold_idx + 1)  # fold_idx is 0-based; keys are 1-based
                    T = float(fold_temperatures.get(fold_key, 1.0))
                elif fold_idx < len(fold_temperatures):
                    T = float(fold_temperatures[fold_idx])
                else:
                    T = 1.0
            else:
                T = 1.0

            self._models.append(model)
            self._temperatures.append(T)
            print(
                f"[ensemble] Loaded fold from {fold_path.name}  T={T:.4f}",
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
        gv, lv, ov, ev, sv, cv, sc = _candidate_to_tensors(candidate, self._device)
        fold_probs: list[float] = []
        fold_aleatoric: list[float] = []

        for model, T in zip(self._models, self._temperatures):
            logit = model(gv, lv, ov, ev, sv, cv, sc)  # (1, 1)
            prob  = torch.sigmoid(logit / T)            # (1, 1)
            fold_probs.append(float(prob.item()))
            # Aleatoric uncertainty from heteroscedastic log-variance head
            if hasattr(model, "fusion") and hasattr(model.fusion, "last_log_var"):
                log_var = model.fusion.last_log_var
                fold_aleatoric.append(float(torch.exp(0.5 * log_var).item()))

        arr = np.array(fold_probs, dtype=np.float64)
        result = {
            "score":       float(arr.mean()),
            "fold_scores": fold_probs,
            "uncertainty": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
        if fold_aleatoric:
            result["aleatoric_std"] = float(np.mean(fold_aleatoric))
        return result

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

        gv, lv, ov, ev, sv, cv, sc = _candidates_to_tensors(candidates, self._device)
        # fold_probs_matrix[fold_idx][candidate_idx] = probability
        fold_probs_matrix: list[np.ndarray] = []

        for model, T in zip(self._models, self._temperatures):
            logits = model(gv, lv, ov, ev, sv, cv, sc)  # (N, 1)
            probs  = torch.sigmoid(logits / T)           # (N, 1)
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
                    "uncertainty": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                }
            )
        return results

    # ── MC Dropout uncertainty ────────────────────────────────────────────────

    def predict_one_mc(
        self,
        candidate: TransitCandidate,
        n_passes: int = 20,
    ) -> dict:
        """
        Score a candidate using Monte Carlo Dropout for uncertainty quantification.

        Runs ``n_passes`` stochastic forward passes per fold with dropout active,
        producing an epistemic uncertainty estimate that reflects model uncertainty
        independently of the fold-agreement uncertainty from ``predict_one``.

        Parameters
        ----------
        candidate :
            A ``TransitCandidate`` from ``ml.preprocess.preprocess``.
        n_passes :
            Number of stochastic forward passes per fold (default 20).
            More passes give a better estimate but take longer.

        Returns
        -------
        dict with keys:
            ``score``             — float, mean over all folds × passes
            ``fold_scores``       — list[float], one deterministic score per fold
            ``uncertainty``       — float, fold-agreement std (from predict_one)
            ``mc_uncertainty``    — float, MC dropout std across all stochastic passes
            ``mc_scores``         — list[float], all stochastic sample probabilities
        """
        gv, lv, ov, ev, sv, cv, sc = _candidate_to_tensors(candidate, self._device)

        # First get deterministic fold scores (eval mode, no dropout)
        det_result = self.predict_one(candidate)

        # Stochastic passes with dropout active (train mode)
        mc_probs: list[float] = []
        for model, T in zip(self._models, self._temperatures):
            model.train()   # activates dropout
            with torch.no_grad():
                for _ in range(n_passes):
                    logit = model(gv, lv, ov, ev, sv, cv, sc)
                    prob  = torch.sigmoid(logit / T)
                    mc_probs.append(float(prob.item()))
            model.eval()    # restore eval mode

        arr = np.array(mc_probs, dtype=np.float64)
        return {
            "score":          float(arr.mean()),
            "fold_scores":    det_result["fold_scores"],
            "uncertainty":    det_result["uncertainty"],
            "mc_uncertainty": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "mc_scores":      mc_probs,
        }

    # ── Test-Time Augmentation ────────────────────────────────────────────────

    @torch.no_grad()
    def predict_one_tta(
        self,
        candidate: TransitCandidate,
        n_shifts: int = 7,
        max_shift_frac: float = 0.05,
    ) -> dict:
        """
        Score a candidate with Test-Time Augmentation via circular phase shifts.

        Applies ``n_shifts`` circular shifts to the phase-folded views (global
        and local), runs the ensemble on each augmented version, and averages
        all probabilities.  This reduces variance from imperfect phase-fold
        alignment and produces a smoother, more robust score.

        Parameters
        ----------
        candidate :
            A ``TransitCandidate`` from ``ml.preprocess.preprocess``.
        n_shifts :
            Number of augmented copies including the unshifted original
            (default 7).  Must be >= 1.
        max_shift_frac :
            Maximum shift as a fraction of the view length (default 0.05 = 5%).
            Shifts are drawn uniformly from [-max, +max].

        Returns
        -------
        dict with keys:
            ``score``       — float, mean over folds × TTA passes
            ``fold_scores`` — list[float], deterministic per-fold scores (unshifted)
            ``uncertainty`` — float, std of per-fold unshifted scores
            ``tta_scores``  — list[float], all per-augmentation mean scores
        """
        det_result = self.predict_one(candidate)

        gv_np  = np.stack([candidate.global_view.astype(np.float32),
                            candidate.raw_global_view.astype(np.float32)], axis=0)  # (2, 2001)
        lv_np  = np.stack([candidate.local_view.astype(np.float32),
                            candidate.raw_local_view.astype(np.float32)], axis=0)   # (2, 201)
        ov_np  = np.stack([candidate.odd_view.astype(np.float32),
                            candidate.raw_odd_view.astype(np.float32)], axis=0)
        ev_np  = np.stack([candidate.even_view.astype(np.float32),
                            candidate.raw_even_view.astype(np.float32)], axis=0)
        sv_np  = np.stack([candidate.secondary_view.astype(np.float32),
                            candidate.raw_secondary_view.astype(np.float32)], axis=0)
        cv_np  = candidate.centroid_curve.astype(np.float32)   # (201,)

        transit_snr = float(np.log1p(
            max(candidate.depth / max(float(candidate.noise_floor), 1e-4), 0.0)
        ))
        sc_np = np.array([
            candidate.period, candidate.duration, candidate.depth, candidate.bls_power,
            candidate.secondary_depth, candidate.odd_even_diff,
            candidate.centroid_shift, candidate.n_transits,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            transit_snr,
        ], dtype=np.float32)
        sc = torch.from_numpy(sc_np).unsqueeze(0).to(self._device)

        def _roll(arr: np.ndarray, shift: int) -> np.ndarray:
            """Circular shift along last axis."""
            return np.roll(arr, shift, axis=-1)

        global_len = gv_np.shape[-1]   # 2001
        local_len  = lv_np.shape[-1]   # 201

        # Build shift amounts: 0 (unshifted) + n_shifts-1 random shifts
        rng = np.random.default_rng(seed=0)
        max_g = max(1, int(global_len * max_shift_frac))
        max_l = max(1, int(local_len  * max_shift_frac))
        shifts = [0] + rng.integers(-max_g, max_g + 1, size=n_shifts - 1).tolist()

        tta_scores: list[float] = []
        for shift in shifts:
            shift_g = int(shift)
            shift_l = int(round(shift * local_len / global_len))

            gv = torch.from_numpy(_roll(gv_np, shift_g)).unsqueeze(0).to(self._device)
            lv = torch.from_numpy(_roll(lv_np, shift_l)).unsqueeze(0).to(self._device)
            ov = torch.from_numpy(_roll(ov_np, shift_l)).unsqueeze(0).to(self._device)
            ev = torch.from_numpy(_roll(ev_np, shift_l)).unsqueeze(0).to(self._device)
            sv = torch.from_numpy(_roll(sv_np, shift_l)).unsqueeze(0).to(self._device)
            cv = torch.from_numpy(_roll(cv_np, shift_l)).unsqueeze(0).unsqueeze(0).to(self._device)

            fold_probs: list[float] = []
            for model, T in zip(self._models, self._temperatures):
                logit = model(gv, lv, ov, ev, sv, cv, sc)
                prob  = torch.sigmoid(logit / T)
                fold_probs.append(float(prob.item()))
            tta_scores.append(float(np.mean(fold_probs)))

        return {
            "score":       float(np.mean(tta_scores)),
            "fold_scores": det_result["fold_scores"],
            "uncertainty": det_result["uncertainty"],
            "tta_scores":  tta_scores,
        }

    # ── Bootstrap confidence intervals ───────────────────────────────────────

    @torch.no_grad()
    def predict_one_with_interval(
        self,
        candidate: TransitCandidate,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        rng_seed: int | None = None,
    ) -> dict:
        """
        Score a candidate with bootstrap confidence intervals.

        Resamples fold checkpoints with replacement ``n_bootstrap`` times,
        computes the mean calibrated probability for each resample, and
        returns percentile-based confidence intervals.  This provides
        frequentist coverage guarantees that the fold-agreement std does not.

        Parameters
        ----------
        candidate :
            A ``TransitCandidate`` from ``ml.preprocess.preprocess``.
        n_bootstrap :
            Number of bootstrap resamples (default 200).
        alpha :
            Significance level: returns [alpha/2, 1-alpha/2] quantiles (default 0.05 → 95% CI).
        rng_seed :
            Optional seed for reproducibility.

        Returns
        -------
        dict with keys:
            ``score``       — float, point estimate (mean fold probability)
            ``fold_scores`` — list[float], per-fold probabilities
            ``uncertainty`` — float, std of fold probabilities
            ``ci_lower``    — float, lower bound of (1-alpha)*100% CI
            ``ci_upper``    — float, upper bound of (1-alpha)*100% CI
            ``ci_alpha``    — float, the alpha used (for reference)
        """
        det_result = self.predict_one(candidate)
        fold_probs = np.array(det_result["fold_scores"], dtype=np.float64)
        n_folds = len(fold_probs)

        if n_folds < 2:
            # Single-fold: CI is degenerate; return point estimate with zero width.
            return {
                "score":       det_result["score"],
                "fold_scores": det_result["fold_scores"],
                "uncertainty": det_result["uncertainty"],
                "ci_lower":    det_result["score"],
                "ci_upper":    det_result["score"],
                "ci_alpha":    alpha,
            }

        rng = np.random.default_rng(rng_seed)
        bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            # Sample n_folds fold indices with replacement
            sampled_idx = rng.integers(0, n_folds, size=n_folds)
            bootstrap_means[i] = fold_probs[sampled_idx].mean()

        ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha / 2)))

        return {
            "score":       det_result["score"],
            "fold_scores": det_result["fold_scores"],
            "uncertainty": det_result["uncertainty"],
            "ci_lower":    ci_lower,
            "ci_upper":    ci_upper,
            "ci_alpha":    alpha,
        }

    # ── Convenience constructor ───────────────────────────────────────────────

    @staticmethod
    def from_output_dir(output_dir: Path, device: str = "cpu") -> ExoNetEnsemble:
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
