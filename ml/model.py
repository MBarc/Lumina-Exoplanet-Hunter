"""
ExoNet — Tri-branch 1D CNN for exoplanet transit classification.

Architecture
------------
Three parallel branches process different aspects of each transit candidate:

  GlobalBranch   — 4-block residual 1D CNN on the full-orbit view  (2001 bins)
                   channels: 1 → 32 → 64 → 128 → 256, output dim 512
  LocalBranch    — 4-block residual 1D CNN on the transit-zoom view  (201 bins)
                   channels: 1 → 32 → 64 → 128 → 256, output dim 256
  ScalarBranch   — 3-layer MLP + BatchNorm on four scalar transit features
                   (period, duration, depth, BLS power), output dim 64

Branch outputs are concatenated (832-d), then passed through a 4-layer
fusion MLP (832→512→256→64→1) with dropout at each layer.

Residual connections
--------------------
Each conv block has a learned 1×1 skip connection that matches the channel
dimension, allowing the gradient to flow freely to earlier layers and making
it practical to stack four blocks without vanishing gradients.

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

GLOBAL_LEN: int = 2001
LOCAL_LEN: int  = 201
SCALAR_FEATURES: int = 6
MODEL_VERSION: str = "2.0"

__all__ = [
    "GLOBAL_LEN",
    "LOCAL_LEN",
    "SCALAR_FEATURES",
    "MODEL_VERSION",
    "ResConvBlock",
    "GlobalBranch",
    "LocalBranch",
    "ScalarBranch",
    "ExoNet",
]


# ── Squeeze-and-Excite channel attention ─────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excite channel attention for 1D feature maps.

    Globally pools across the time dimension (squeeze), then learns a
    per-channel scale via a small MLP (excite), and multiplies it back
    onto the feature map.  This lets the model weight which frequency
    bands of the light curve matter most per sample.

    Parameters
    ----------
    channels :
        Number of input/output channels.
    reduction :
        Bottleneck reduction factor for the excitation MLP.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L) → squeeze → (B, C) → excite → (B, C, 1)
        scale = self.fc(x.mean(dim=-1)).unsqueeze(-1)
        return x * scale


# ── Residual conv block ───────────────────────────────────────────────────────

class ResConvBlock(nn.Module):
    """
    One residual 1D convolutional block with optional SE attention.

    Structure
    ---------
    ::

        ┌─ Conv1d(in, out, k) ─ BN ─ ReLU ─ Conv1d(out, out, k) ─ BN ─┐
        │                                                                │
        x ── skip: Conv1d(in, out, 1) ─────────────────────────────────(+)── [SE] ── ReLU ── MaxPool1d(p)

    Parameters
    ----------
    in_channels, out_channels :
        Input / output channel counts.
    kernel_size :
        Convolution kernel width (same padding is used).
    pool :
        MaxPool stride after the residual add + activation.
    use_se :
        If True, apply Squeeze-and-Excite channel attention after the
        residual add and before the activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool: int = 2,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.se   = SEBlock(out_channels) if use_se else nn.Identity()
        self.act  = nn.ReLU()
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.act(self.se(self.conv_path(x) + self.skip(x))))


# ── Branch definitions ────────────────────────────────────────────────────────

class GlobalBranch(nn.Module):
    """
    Four-block residual CNN on the full-orbit phase curve.

    Input  : (batch, 1, 2001)
    Output : (batch, 512)

    Block layout
    ------------
    ::

        (B,   1, 2001) ──[32, pool=5]──► (B,  32, 400)
        (B,  32,  400) ──[64, pool=5]──► (B,  64,  80)
        (B,  64,   80) ──[128,pool=4]──► (B, 128,  20)
        (B, 128,   20) ──[256,pool=4]──► (B, 256,   5)
        Flatten → 1280 → Linear(1280, 512) → ReLU → Dropout(0.4)
    """

    def __init__(self, use_se: bool = False) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResConvBlock(  1,  32, kernel_size=5, pool=5, use_se=use_se),
            ResConvBlock( 32,  64, kernel_size=5, pool=5, use_se=use_se),
            ResConvBlock( 64, 128, kernel_size=5, pool=4, use_se=use_se),
            ResConvBlock(128, 256, kernel_size=5, pool=4, use_se=use_se),
        )
        # 256 channels × 5 time-steps = 1280
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.blocks(x))


class LocalBranch(nn.Module):
    """
    Four-block residual CNN on the transit-zoom view.

    Input  : (batch, 1, 201)
    Output : (batch, 256)

    Block layout
    ------------
    ::

        (B,   1, 201) ──[32, pool=3]──► (B,  32, 67)
        (B,  32,  67) ──[64, pool=3]──► (B,  64, 22)
        (B,  64,  22) ──[128,pool=2]──► (B, 128, 11)
        (B, 128,  11) ──[256,pool=2]──► (B, 256,  5)
        Flatten → 1280 → Linear(1280, 256) → ReLU → Dropout(0.4)
    """

    def __init__(self, use_se: bool = False) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResConvBlock(  1,  32, kernel_size=5, pool=3, use_se=use_se),
            ResConvBlock( 32,  64, kernel_size=5, pool=3, use_se=use_se),
            ResConvBlock( 64, 128, kernel_size=5, pool=2, use_se=use_se),
            ResConvBlock(128, 256, kernel_size=5, pool=2, use_se=use_se),
        )
        # 256 channels × 5 time-steps = 1280
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.blocks(x))


class ScalarBranch(nn.Module):
    """
    Three-layer MLP with BatchNorm on the scalar transit features.

    Input  : (batch, 6)  — raw [period_days, duration_days,
                            depth_fractional, bls_power,
                            secondary_depth, odd_even_diff]
    Output : (batch, 64)

    The log1p normalisation is applied inside ``forward`` so the branch is
    fully self-contained for ONNX export.

    Architecture
    ------------
    Linear(6, 64) → BN → ReLU → Linear(64, 64) → BN → ReLU
                 → Linear(64, 64) → BN → ReLU
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    @staticmethod
    def _normalize_scalars(raw: Tensor) -> Tensor:
        period    = torch.log1p(raw[:, 0:1])
        duration  = torch.log1p(raw[:, 1:2])
        depth_ppm = torch.log1p(torch.clamp(raw[:, 2:3] * 1e6, min=1e-3))
        power     = torch.log1p(torch.clamp(raw[:, 3:4],        min=1e-3))
        sec_depth = torch.log1p(torch.clamp(raw[:, 4:5] * 1e6,  min=1e-3))
        oe_diff   = torch.log1p(torch.clamp(raw[:, 5:6] * 1e6,  min=1e-3))
        return torch.cat([period, duration, depth_ppm, power, sec_depth, oe_diff], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self._normalize_scalars(x))


# ── Full model ────────────────────────────────────────────────────────────────

class ExoNet(nn.Module):
    """
    Tri-branch residual 1D CNN for exoplanet transit classification.

    Branch outputs are concatenated (512 + 256 + 64 = 832 dimensions) and
    passed through a four-layer fusion MLP to produce a single transit
    probability in [0, 1].

    Examples
    --------
    >>> model = ExoNet()
    >>> gv = torch.randn(8, 1, 2001)
    >>> lv = torch.randn(8, 1, 201)
    >>> sc = torch.zeros(8, 6)
    >>> model(gv, lv, sc).shape
    torch.Size([8, 1])
    """

    def __init__(self, use_se: bool = False) -> None:
        super().__init__()

        self.global_branch = GlobalBranch(use_se=use_se)   # (B, 512)
        self.local_branch  = LocalBranch(use_se=use_se)    # (B, 256)
        self.scalar_branch = ScalarBranch()                # (B,  64)

        # Fusion: 832 → 512 → 256 → 64 → 1  (outputs raw logits; sigmoid applied externally)
        self.fusion = nn.Sequential(
            nn.Linear(832, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        global_view: Tensor,
        local_view: Tensor,
        scalar_features: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        global_view     : (batch, 1, 2001)
        local_view      : (batch, 1, 201)
        scalar_features : (batch, 6)  — [period_days, duration_days,
                                          depth_fractional, bls_power,
                                          secondary_depth, odd_even_diff]

        Returns
        -------
        Tensor of shape (batch, 1) — raw logit (apply sigmoid for probability)
        """
        g = self.global_branch(global_view)
        l = self.local_branch(local_view)
        s = self.scalar_branch(scalar_features)
        return self.fusion(torch.cat([g, l, s], dim=1))
