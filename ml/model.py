"""
ExoNet — Dual-branch 1D CNN for transit classification.

Architecture
------------
Three parallel branches process different aspects of each transit candidate:

  GlobalBranch   — operates on the full-orbit view  (2001 phase bins)
  LocalBranch    — operates on the transit-zoom view  (201 phase bins)
  ScalarBranch   — processes four scalar transit features (period, duration,
                   depth, BLS power) after internal log1p normalisation

The branch outputs are concatenated to form a 416-dimensional joint
representation, which is then passed through a small fusion head that produces
a single probability score in [0, 1].

References
----------
Inspired by the AstroNet architecture described in:
  Shallue & Vanderburg (2018), AJ 155 94, arXiv:1712.05205
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# ── Public constants ──────────────────────────────────────────────────────────

GLOBAL_LEN: int = 2001   # length of the full-orbit phase curve fed to GlobalBranch
LOCAL_LEN: int  = 201    # length of the transit-zoom view fed to LocalBranch
SCALAR_FEATURES: int = 4 # number of scalar features fed to ScalarBranch
MODEL_VERSION: str = "1.0"

__all__ = [
    "GLOBAL_LEN",
    "LOCAL_LEN",
    "SCALAR_FEATURES",
    "MODEL_VERSION",
    "GlobalBranch",
    "LocalBranch",
    "ScalarBranch",
    "ExoNet",
]


# ── Branch definitions ────────────────────────────────────────────────────────

class GlobalBranch(nn.Module):
    """
    1D CNN that processes the full-orbit phase curve.

    Input  : (batch, 1, 2001)
    Output : (batch, 256)

    Three conv+pool blocks progressively compress the 2001-point sequence
    down to a 1280-element flat vector before the linear projection.
    BatchNorm1d is applied after every Conv1d layer.

    Pool factors  : 5 → 5 → 4   (total compression: ×100, 2001 → ~20 → 1280 / 64)
    Channel widths: 1 → 16 → 32 → 64
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: (B, 1, 2001) → (B, 16, 400)
            nn.Conv1d(1,  16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(5),

            # Block 2: (B, 16, 400) → (B, 32, 80)
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(5),

            # Block 3: (B, 32, 80) → (B, 64, 20)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # 64 channels × 20 time-steps = 1280
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, 1, 2001)

        Returns
        -------
        Tensor of shape (batch, 256)
        """
        return self.head(self.conv_blocks(x))


class LocalBranch(nn.Module):
    """
    1D CNN that processes the transit-zoom view.

    Input  : (batch, 1, 201)
    Output : (batch, 128)

    Three conv+pool blocks compress the 201-point sequence.
    BatchNorm1d is applied after every Conv1d layer.

    Pool factors  : 3 → 3 → 2   (total compression: ×18, 201 → 11 → 704 / 64)
    Channel widths: 1 → 16 → 32 → 64
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: (B, 1, 201) → (B, 16, 67)
            nn.Conv1d(1,  16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3),

            # Block 2: (B, 16, 67) → (B, 32, 22)
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3),

            # Block 3: (B, 32, 22) → (B, 64, 11)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # 64 channels × 11 time-steps = 704
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(704, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, 1, 201)

        Returns
        -------
        Tensor of shape (batch, 128)
        """
        return self.head(self.conv_blocks(x))


class ScalarBranch(nn.Module):
    """
    Small MLP that processes four scalar transit features.

    Input  : (batch, 4)  — raw values [period_days, duration_days,
                            depth_fractional, bls_power]
    Output : (batch, 32)

    The log1p normalisation is applied *inside* ``forward`` via the static
    helper ``_normalize_scalars``, so the branch is fully self-contained and
    ONNX export works without any preprocessing steps outside the graph.

    Normalisation
    -------------
    The four raw scalars are transformed before being fed to the linear layers:

      * ``log1p(period)``
      * ``log1p(duration)``
      * ``log1p(max(depth  * 1e6, 1e-3))``  — converts to ppm then log-scales
      * ``log1p(max(bls_power, 1e-3))``

    Architecture
    ------------
    Linear(4, 32) → ReLU → Linear(32, 32) → ReLU
    """

    def __init__(self) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    @staticmethod
    def _normalize_scalars(raw: Tensor) -> Tensor:
        """
        Apply log1p normalisation to raw scalar features in-graph.

        Parameters
        ----------
        raw : Tensor of shape (batch, 4)
            Columns: [period_days, duration_days, depth_fractional, bls_power]

        Returns
        -------
        Tensor of shape (batch, 4)
            Log1p-normalised features ready for the linear layers.
        """
        period    = torch.log1p(raw[:, 0:1])
        duration  = torch.log1p(raw[:, 1:2])
        depth_ppm = torch.log1p(torch.clamp(raw[:, 2:3] * 1e6, min=1e-3))
        power     = torch.log1p(torch.clamp(raw[:, 3:4],        min=1e-3))
        return torch.cat([period, duration, depth_ppm, power], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, 4)
            Raw scalar features [period_days, duration_days,
            depth_fractional, bls_power].

        Returns
        -------
        Tensor of shape (batch, 32)
        """
        return self.mlp(self._normalize_scalars(x))


# ── Full model ────────────────────────────────────────────────────────────────

class ExoNet(nn.Module):
    """
    Tri-branch 1D CNN for exoplanet transit classification.

    All three branches run in parallel; their outputs are concatenated to form
    a 416-dimensional joint representation (256 + 128 + 32), which is then
    processed by a small fusion MLP to produce a single transit probability.

    Parameters
    ----------
    None — the architecture is fully fixed.

    Examples
    --------
    >>> model = ExoNet()
    >>> global_view     = torch.randn(8, 1, 2001)
    >>> local_view      = torch.randn(8, 1, 201)
    >>> scalar_features = torch.zeros(8, 4)          # raw: period, duration, depth, bls_power
    >>> scores = model(global_view, local_view, scalar_features)   # shape: (8, 1)
    """

    def __init__(self) -> None:
        super().__init__()

        self.global_branch = GlobalBranch()   # output: (B, 256)
        self.local_branch  = LocalBranch()    # output: (B, 128)
        self.scalar_branch = ScalarBranch()   # output: (B, 32)

        # Fusion head: 256 + 128 + 32 = 416 → 128 → 1
        self.fusion = nn.Sequential(
            nn.Linear(416, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_view: Tensor,
        local_view: Tensor,
        scalar_features: Tensor,
    ) -> Tensor:
        """
        Compute transit probability scores.

        Parameters
        ----------
        global_view : Tensor of shape (batch, 1, 2001)
            Full-orbit phase curve, one value per phase bin.
        local_view : Tensor of shape (batch, 1, 201)
            Transit-zoom phase curve, centred on the transit.
        scalar_features : Tensor of shape (batch, 4)
            Raw scalar features in column order:
            [period_days, duration_days, depth_fractional, bls_power].
            Log1p normalisation is applied internally by ScalarBranch.

        Returns
        -------
        Tensor of shape (batch, 1)
            Transit probability in [0, 1].  Values close to 1 indicate a
            likely genuine exoplanet transit; values close to 0 indicate a
            likely false positive or no transit.
        """
        g = self.global_branch(global_view)         # (B, 256)
        l = self.local_branch(local_view)           # (B, 128)
        s = self.scalar_branch(scalar_features)     # (B, 32)
        fused = torch.cat([g, l, s], dim=1)         # (B, 416)
        return self.fusion(fused)                   # (B, 1)
