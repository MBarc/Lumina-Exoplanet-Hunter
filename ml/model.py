"""
ExoNet — Seven-branch 1D CNN + Transformer fusion for exoplanet transit classification.

Architecture
------------
Eight parallel branches process different aspects of each transit candidate:

  GlobalBranch    — 4-block residual 1D CNN on a 2-channel full-orbit view
                    (detrended + raw, 2001 bins); channels: 2→32→64→128→256,
                    output dim 512; with optional spatial self-attention
  LocalBranch     — 4-block residual 1D CNN on the transit-zoom view (201 bins)
                    2-channel (detrended + raw), output dim 256
  OddBranch       — LocalBranch copy on odd-transit phase-fold, output dim 256
  EvenBranch      — LocalBranch copy on even-transit phase-fold, output dim 256
  SecondaryBranch — LocalBranch copy on secondary-eclipse phase-fold, output dim 256
  CentroidBranch  — LocalBranch copy on phase-folded centroid curve, output dim 256
  ScalarBranch    — 3-layer MLP + BatchNorm on 17 scalar features, output dim 64
  WaveletBranch   — 3-level Haar decomposition + per-level CNNs on global view,
                    captures structure at multiple time scales, output dim 256

Branch outputs are treated as tokens and passed through a Transformer encoder
(BranchFusionTransformer) which learns cross-branch attention patterns before
producing a final logit.

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
ODD_VIEW_LEN: int = 201
EVEN_VIEW_LEN: int = 201
SECONDARY_VIEW_LEN: int = 201
SCALAR_FEATURES: int = 17
MODEL_VERSION: str = "4.0"

__all__ = [
    "GLOBAL_LEN",
    "LOCAL_LEN",
    "ODD_VIEW_LEN",
    "EVEN_VIEW_LEN",
    "SECONDARY_VIEW_LEN",
    "SCALAR_FEATURES",
    "MODEL_VERSION",
    "ResConvBlock",
    "GlobalBranch",
    "LocalBranch",
    "CentroidBranch",
    "ScalarBranch",
    "WaveletBranch",
    "BranchFusionTransformer",
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
    One residual 1D convolutional block with optional SE attention and dilation.

    Structure
    ---------
    ::

        ┌─ Conv1d(in, out, k, dilation=d) ─ BN ─ ReLU ─ Conv1d(out, out, k, dilation=d) ─ BN ─┐
        │                                                                                         │
        x ── skip: Conv1d(in, out, 1) ───────────────────────────────────────────────────────── (+) ── [SE] ── ReLU ── MaxPool1d(p)

    Parameters
    ----------
    in_channels, out_channels :
        Input / output channel counts.
    kernel_size :
        Convolution kernel width.
    pool :
        MaxPool stride after the residual add + activation.  1 = no pooling.
    use_se :
        If True, apply Squeeze-and-Excite channel attention after the
        residual add and before the activation.
    dilation :
        Dilation factor for both convolutions (default 1 = standard conv).
        Using dilation > 1 expands the receptive field without increasing
        parameters or losing time-axis resolution.  Essential for the 2001-bin
        global view where a 4-block network with pool=2 has a receptive field
        of only ~40 bins; dilation=[1,2,4,8] expands this to ~200 bins.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool: int = 2,
        use_se: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation   # 'same' padding for dilated convs

        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.se   = SEBlock(out_channels) if use_se else nn.Identity()
        self.act  = nn.ReLU()
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.act(self.se(self.conv_path(x) + self.skip(x))))


# ── Branch definitions ────────────────────────────────────────────────────────

class GlobalBranch(nn.Module):
    """
    Four-block residual CNN on the 2-channel full-orbit phase curve.

    Input  : (batch, 2, input_len)   default input_len=2001
    Output : (batch, 512)

    Block layout
    ------------
    ::

        (B,   2, 2001) ──[32, pool=5]──► (B,  32, 400)
        (B,  32,  400) ──[64, pool=5]──► (B,  64,  80)
        (B,  64,   80) ──[128,pool=4]──► (B, 128,  20)
        (B, 128,   20) ──[256,pool=4]──► (B, 256,   5)
        Flatten → flat_size → Linear(flat_size, 512) → ReLU → Dropout(dropout)

    channels: 2 → 32 → 64 → 128 → 256

    The flatten size is computed dynamically via a dummy forward pass so this
    branch remains correct if input_len ever changes (Issue 1.5).
    """

    def __init__(
        self,
        use_se: bool = False,
        in_channels: int = 2,
        dropout: float = 0.4,
        input_len: int = 2001,
        use_attn: bool = True,
    ) -> None:
        super().__init__()
        # Dilated convolution schedule: [1, 2, 4, 8]
        # pool=1 on early blocks preserves resolution so later dilations see more context;
        # pooling is applied only at blocks 3 and 4 after sufficient context is captured.
        # Receptive field: block1 ~5 bins, block2 ~15 bins, block3 ~45 bins, block4 ~125 bins
        # vs. standard (no dilation): all blocks ~5-bin receptive field.
        self._conv_layers = nn.Sequential(
            ResConvBlock(in_channels,  32, kernel_size=5, pool=1, use_se=use_se, dilation=1),
            ResConvBlock( 32,  64, kernel_size=5, pool=1, use_se=use_se, dilation=2),
            ResConvBlock( 64, 128, kernel_size=5, pool=5, use_se=use_se, dilation=4),
            ResConvBlock(128, 256, kernel_size=5, pool=5, use_se=use_se, dilation=8),
        )
        # Optional self-attention over the spatial positions that remain after
        # pooling (typically 5 tokens of dim 256).  Each token corresponds to
        # ~400 phase-bins of the orbit, so attention learns orbit-scale
        # correlations such as "if token at phase 0.4-0.6 shows a secondary
        # dip, attend more strongly to the primary at phase 0".
        if use_attn:
            self._spatial_attn: nn.Module = nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512,
                dropout=dropout, batch_first=True,
            )
        else:
            self._spatial_attn = nn.Identity()
        # Compute flatten size dynamically so the branch adapts to any input_len.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_len)
            flat_size = self._conv_layers(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    # Keep attribute name 'blocks' as an alias so that any code loading old
    # checkpoints via 'model.global_branch.blocks' continues to work.
    @property
    def blocks(self) -> nn.Sequential:
        return self._conv_layers

    def forward(self, x: Tensor) -> Tensor:
        features = self._conv_layers(x)                     # (B, 256, L)
        # Transpose to (B, L, 256) for batch_first TransformerEncoderLayer,
        # apply attention, then transpose back to (B, 256, L) for flatten.
        features = self._spatial_attn(
            features.transpose(1, 2)
        ).transpose(1, 2)                                   # (B, 256, L)
        return self.head(features)


class LocalBranch(nn.Module):
    """
    Four-block residual CNN on the transit-zoom view.

    Input  : (batch, 2, input_len)   default input_len=201  — 2-channel: [detrended, prenorm_raw]
    Output : (batch, 256)

    Block layout
    ------------
    ::

        (B,   2, 201) ──[32, pool=3]──► (B,  32, 67)
        (B,  32,  67) ──[64, pool=3]──► (B,  64, 22)
        (B,  64,  22) ──[128,pool=2]──► (B, 128, 11)
        (B, 128,  11) ──[256,pool=2]──► (B, 256,  5)
        Flatten → flat_size → Linear(flat_size, 256) → ReLU → Dropout(dropout)

    The flatten size is computed dynamically via a dummy forward pass so this
    branch remains correct if input_len ever changes (Issue 1.5).
    """

    def __init__(
        self,
        use_se: bool = False,
        in_channels: int = 2,
        dropout: float = 0.4,
        input_len: int = 201,
    ) -> None:
        super().__init__()
        self._conv_layers = nn.Sequential(
            ResConvBlock(in_channels,  32, kernel_size=5, pool=3, use_se=use_se),
            ResConvBlock( 32,  64, kernel_size=5, pool=3, use_se=use_se),
            ResConvBlock( 64, 128, kernel_size=5, pool=2, use_se=use_se),
            ResConvBlock(128, 256, kernel_size=5, pool=2, use_se=use_se),
        )
        # Spatial self-attention over the remaining time tokens after pooling.
        # For the 201-bin local view, the 4 pool layers leave ~5 tokens of dim 256.
        # Attention lets the branch learn transit shape signatures: e.g. "ingress
        # token attends to egress token to verify symmetry" or "the dip token
        # attends to the out-of-transit baseline to normalise depth".
        self._spatial_attn = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512,
            dropout=dropout, batch_first=True,
        )
        # Compute flatten size dynamically so the branch adapts to any input_len.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_len)
            flat_size = self._conv_layers(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @property
    def blocks(self) -> nn.Sequential:
        return self._conv_layers

    def forward(self, x: Tensor) -> Tensor:
        features = self._conv_layers(x)                      # (B, 256, L)
        features = self._spatial_attn(
            features.transpose(1, 2)
        ).transpose(1, 2)                                    # (B, 256, L)
        return self.head(features)


class CentroidBranch(nn.Module):
    """
    Four-block residual CNN on the phase-folded centroid displacement curve.

    Identical architecture to LocalBranch.  Processing centroid motion as a
    dedicated CNN branch rather than a single scalar lets the network learn
    spatial patterns (e.g. centroid offset correlated with transit phase)
    that a single number cannot capture.

    Input  : (batch, 1, input_len)   default input_len=201  — centroid stays 1-channel
    Output : (batch, 256)

    The flatten size is computed dynamically via a dummy forward pass so this
    branch remains correct if input_len ever changes (Issue 1.5).
    """

    def __init__(
        self,
        use_se: bool = False,
        dropout: float = 0.4,
        input_len: int = 201,
    ) -> None:
        super().__init__()
        self._conv_layers = nn.Sequential(
            ResConvBlock(  1,  32, kernel_size=5, pool=3, use_se=use_se),
            ResConvBlock( 32,  64, kernel_size=5, pool=3, use_se=use_se),
            ResConvBlock( 64, 128, kernel_size=5, pool=2, use_se=use_se),
            ResConvBlock(128, 256, kernel_size=5, pool=2, use_se=use_se),
        )
        # Compute flatten size dynamically so the branch adapts to any input_len.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_size = self._conv_layers(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @property
    def blocks(self) -> nn.Sequential:
        return self._conv_layers

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._conv_layers(x))


class ScalarBranch(nn.Module):
    """
    Three-layer MLP with BatchNorm on the scalar transit features.

    Input  : (batch, 17)  — raw scalar features:
                 0  period_days
                 1  duration_days
                 2  depth_fractional
                 3  bls_power
                 4  secondary_depth
                 5  odd_even_diff
                 6  centroid_shift (pixels)
                 7  n_transits
                 8  log_teff_norm  (log10(Teff) - log10(5778))
                 9  logg           (cm s⁻²)
                 10 log_radius_norm (log10(R/R_sun))
                 11 feh            ([Fe/H])
                 12 kepmag_norm    ((kepmag - 12) / 4)
                 13 mission_is_kepler  (one-hot)
                 14 mission_is_tess    (one-hot)
                 15 mission_is_k2     (one-hot)
                 16 transit_snr    (log1p(depth / noise_floor))
    Output : (batch, 64)

    The log1p normalisation is applied inside ``forward`` so the branch is
    fully self-contained for ONNX export.

    Architecture
    ------------
    Linear(17, 128) → BN → ReLU → Linear(128, 64) → BN → ReLU
                   → Linear(64, 64) → BN → ReLU
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(17, 128),       # wider first layer for more features
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    @staticmethod
    def _normalize_scalars(raw: Tensor) -> Tensor:
        period    = torch.log1p(torch.clamp(raw[:, 0:1], min=0.0))
        duration  = torch.log1p(torch.clamp(raw[:, 1:2], min=0.0))
        depth_ppm = torch.log1p(torch.clamp(raw[:, 2:3] * 1e6, min=1e-3))
        power     = torch.log1p(torch.clamp(raw[:, 3:4],        min=1e-3))
        sec_depth = torch.log1p(torch.clamp(raw[:, 4:5] * 1e6,  min=1e-3))
        oe_diff   = torch.log1p(torch.clamp(raw[:, 5:6] * 1e6,  min=1e-3))
        centr     = torch.log1p(torch.clamp(raw[:, 6:7] * 1000., min=1e-3))
        n_tr      = torch.log1p(torch.clamp(raw[:, 7:8],         min=1e-3))
        # Stellar params (indices 8-12): NaN-safe — replace NaN/inf with 0.0
        # so missing stellar parameters (always zero at inference) don't propagate
        # through the network.  torch.nan_to_num is a no-op when values are finite.
        stellar = torch.nan_to_num(raw[:, 8:13], nan=0.0, posinf=0.0, neginf=0.0)
        log_teff   = stellar[:, 0:1]   # already log-normalised by caller
        logg       = stellar[:, 1:2]   # already normalised by caller
        log_radius = stellar[:, 2:3]   # already log-normalised by caller
        feh        = stellar[:, 3:4]   # already normalised by caller (z-scored)
        contam     = stellar[:, 4:5]   # kepmag_norm: (kepmag - 12.0) / 4.0, already normalised
        # Indices 13-15: mission one-hot {kepler, tess, k2} ∈ {0,1} — no transform needed
        mission    = raw[:, 13:16]
        # Index 16: transit S/N = log1p(depth/noise_floor), already log-compressed at
        # construction time.  Apply log1p once more to further compress large SNR values.
        snr        = torch.log1p(torch.clamp(raw[:, 16:17], min=0.0))
        return torch.cat([period, duration, depth_ppm, power, sec_depth, oe_diff, centr, n_tr,
                          log_teff, logg, log_radius, feh, contam, mission, snr], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self._normalize_scalars(x))


# ── Wavelet multi-scale branch ───────────────────────────────────────────────

class WaveletBranch(nn.Module):
    """
    Multi-scale light curve branch using Haar wavelet decomposition.

    Decomposes the global view into approximation (low-freq) and detail
    (high-freq) subbands at 3 levels, then applies a small 1D CNN to each
    subband.  The per-level outputs are concatenated into a 256-d feature
    vector that captures structure at multiple time scales — complementing
    the fixed 2001-bin global view with intermediate-scale features.

    This helps detect transit patterns at periods where the 2001-bin window
    has insufficient resolution and at periods where the transit sits in a
    noisy neighborhood of the fold.

    Input  : (batch, 2, 2001) — 2-channel global view (detrended + raw)
    Output : (batch, 256)
    """

    def __init__(self, levels: int = 3, out_dim: int = 256, dropout: float = 0.1,
                 input_len: int = 2001) -> None:
        super().__init__()
        self._levels = levels
        # Pre-compute the length of each wavelet subband so we can use explicit
        # Conv1d (not LazyConv1d) — LazyConv1d breaks ONNX export and
        # introduces non-determinism across different batch shapes.
        # Level k subband has length ceil(input_len / 2^k).
        self._level_cnns = nn.ModuleList()
        L = input_len
        for _ in range(levels + 1):  # levels detail subbands + 1 approximation
            L = (L + 1) // 2   # length after one Haar step (ceil division)
            self._level_cnns.append(nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),   # explicit, ONNX-safe
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            ))
        combined_dim = 32 * (levels + 1)
        self.head = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _haar_level(signal: Tensor) -> tuple[Tensor, Tensor]:
        """One level of 1D Haar wavelet decomposition along the last axis."""
        # Always pad to even length with a no-op pad when already even.
        # Avoids Python boolean control flow so the ONNX trace is static.
        pad = signal.size(-1) % 2          # 0 or 1 — integer, not a tensor bool
        signal = torch.nn.functional.pad(signal, (0, pad))
        even = signal[..., ::2]
        odd  = signal[..., 1::2]
        approx = (even + odd) / 2.0
        detail = (even - odd) / 2.0
        return approx, detail

    def forward(self, global_view: Tensor) -> Tensor:
        # Use detrended channel (ch 0) for wavelet decomposition
        x = global_view[:, 0:1, :]   # (B, 1, 2001)
        subbands: list[Tensor] = []
        for _ in range(self._levels):
            x, detail = self._haar_level(x)
            subbands.append(detail)
        subbands.append(x)   # final approximation
        # Apply per-level CNNs and concatenate
        features = [cnn(sb) for cnn, sb in zip(self._level_cnns, subbands)]
        return self.head(torch.cat(features, dim=1))


# ── Cross-branch Transformer fusion ──────────────────────────────────────────

class BranchFusionTransformer(nn.Module):
    """
    Cross-branch attention fusion using a Transformer encoder.

    Each branch's output vector is treated as one token in a sequence.
    The Transformer learns which branches' features to attend to for each
    decision, making it possible to model "the local view depth is consistent
    with the odd-transit view" as an attention pattern.

    Parameters
    ----------
    branch_dims :
        List of output dimensionalities from each branch (in order).
    d_model :
        Common token dimension after projection.  Default 256.
    nhead :
        Number of attention heads.  Default 4.
    num_layers :
        Number of Transformer encoder layers.  Default 2.
    """

    def __init__(
        self,
        branch_dims: list[int],
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_branches = len(branch_dims)
        self._n_branches = n_branches
        self.projections = nn.ModuleList([
            nn.Linear(dim, d_model) for dim in branch_dims
        ])
        # Learned positional embedding: one vector per branch token so the
        # Transformer knows which token corresponds to which branch.
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_branches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,   # canonical 4× expansion
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Learnable per-sample branch gating network.
        # Computes a soft weight for each branch token based on the mean-pooled
        # encoded representation, allowing the model to suppress uninformative
        # branches (e.g. centroid when centroid data is missing) on a per-sample basis.
        # An entropy auxiliary loss encourages the gate to activate at least 2-3 branches.
        self.gate = nn.Sequential(
            nn.Linear(d_model, n_branches),
            nn.Softmax(dim=-1),
        )
        # Scratch buffers: reset each forward pass so stale values never bleed
        # across batches.  Stored as non-Parameter attributes so they are not
        # part of state_dict and don't affect checkpointing.
        self.last_gate_weights: Tensor | None = None
        self.last_log_var: Tensor | None = None
        # Learned attention pooling: a single query vector attends over all branch
        # tokens, learning which branches matter most for the final decision.
        # This outperforms mean pooling when different branches contribute
        # unequally (e.g. local+secondary are more diagnostic than global for short-period planets).
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Heteroscedastic (aleatoric) uncertainty head.
        # Outputs log-variance so the model learns per-candidate prediction confidence.
        # During training, Gaussian NLL is used alongside focal loss to jointly optimize
        # both the mean prediction (logit) and its uncertainty (log_var).
        # At inference, aleatoric_std = exp(0.5 * log_var) is reported alongside the score.
        self.log_var_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        branch_outputs: list[Tensor],
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # CF7: validate the number of branch outputs matches the number of projections.
        if len(branch_outputs) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} branch outputs, got {len(branch_outputs)}"
            )
        # Project each branch to d_model, stack as token sequence, add positional embedding
        tokens = [proj(feat) for proj, feat in zip(self.projections, branch_outputs)]
        token_seq = torch.stack(tokens, dim=1)          # (B, n_branches, d_model)
        token_seq = token_seq + self.pos_embedding       # add branch identity
        encoded = self.transformer(token_seq)            # (B, n_branches, d_model)
        # Learnable branch gating: compute per-sample, per-branch scalar weights from
        # the mean-pooled encoded representation.  Scales each branch token before pooling,
        # effectively suppressing branches that carry little signal for this sample.
        # Gate weights are stored as self.last_gate_weights for loss computation in
        # the training loop without changing the return type of this method.
        gate_weights = self.gate(encoded.mean(dim=1))        # (B, n_branches)
        self.last_gate_weights = gate_weights                 # detached copy for loss; set live tensor
        encoded = encoded * gate_weights.unsqueeze(-1)        # (B, n_branches, d_model)
        # Attention pooling: a learned query attends over all gated branch tokens.
        query = self.pool_query.expand(encoded.size(0), -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pool_attn(query, encoded, encoded)       # (B, 1, d_model)
        pooled = pooled.squeeze(1)                                 # (B, d_model)
        # Heteroscedastic uncertainty: store log-variance for training loss and inference.
        # Clamped to [-10, 10] to prevent numerical instability in exp().
        self.last_log_var = torch.clamp(self.log_var_head(pooled), -10.0, 10.0)  # (B, 1)
        if return_features:
            return self.head(pooled), pooled                       # (B, 1), (B, d_model)
        return self.head(pooled)                                   # (B, 1)


# ── Full model ────────────────────────────────────────────────────────────────

class ExoNet(nn.Module):
    """
    Seven-branch residual 1D CNN with Transformer fusion for exoplanet transit classification.

    Branch outputs are treated as tokens and fused via cross-branch attention
    in a BranchFusionTransformer.

    Examples
    --------
    >>> model = ExoNet()
    >>> gv  = torch.randn(8, 2, 2001)
    >>> lv  = torch.randn(8, 2, 201)
    >>> ov  = torch.randn(8, 2, 201)
    >>> ev  = torch.randn(8, 2, 201)
    >>> sv  = torch.randn(8, 2, 201)
    >>> cv  = torch.randn(8, 1, 201)
    >>> sc  = torch.zeros(8, 13)
    >>> model(gv, lv, ov, ev, sv, cv, sc).shape
    torch.Size([8, 1])
    """

    def __init__(self, use_se: bool = True, dropout: float = 0.4, use_global_attn: bool = True) -> None:
        super().__init__()
        self.global_branch    = GlobalBranch(use_se=use_se, in_channels=2, dropout=dropout, use_attn=use_global_attn)  # (B, 512)
        self.local_branch     = LocalBranch(use_se=use_se, in_channels=2, dropout=dropout)   # (B, 256)
        self.odd_branch       = LocalBranch(use_se=use_se, in_channels=2, dropout=dropout)   # (B, 256)
        self.even_branch      = LocalBranch(use_se=use_se, in_channels=2, dropout=dropout)   # (B, 256)
        self.secondary_branch = LocalBranch(use_se=use_se, in_channels=2, dropout=dropout)   # (B, 256)
        self.centroid_branch  = CentroidBranch(use_se=use_se, dropout=dropout) # (B, 256)
        self.scalar_branch    = ScalarBranch()                                 # (B,  64)
        self.wavelet_branch   = WaveletBranch(levels=3, out_dim=256, dropout=dropout)  # (B, 256)
        self.fusion = BranchFusionTransformer(
            branch_dims=[512, 256, 256, 256, 256, 256, 64, 256],  # +wavelet branch
            d_model=256, nhead=4, num_layers=2, dropout=dropout,
        )
        # Auxiliary regression heads — predict log(depth) and log(duration) from
        # the fused representation.  Used only during training as a multi-task
        # regulariser; not exported to ONNX.  Predicting physical quantities
        # forces the trunk to encode photometrically meaningful features.
        self.aux_depth_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.aux_duration_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(
        self,
        global_view: Tensor,      # (batch, 2, 2001)
        local_view: Tensor,       # (batch, 2, 201)
        odd_view: Tensor,         # (batch, 2, 201)
        even_view: Tensor,        # (batch, 2, 201)
        secondary_view: Tensor,   # (batch, 2, 201)
        centroid_view: Tensor,    # (batch, 1, 201)
        scalar_features: Tensor,  # (batch, 17)
    ) -> Tensor:
        """
        Parameters
        ----------
        global_view     : (batch, 2, 2001) — 2-channel: [detrended, raw]
        local_view      : (batch, 2, 201)  — [detrended z-scored, prenorm_raw]
        odd_view        : (batch, 2, 201)  — odd-transit phase-fold, 2-channel
        even_view       : (batch, 2, 201)  — even-transit phase-fold, 2-channel
        secondary_view  : (batch, 2, 201)  — secondary-eclipse phase-fold, 2-channel
        centroid_view   : (batch, 1, 201)  — phase-folded centroid displacement curve
        scalar_features : (batch, 17)  — [period_days, duration_days,
                                          depth_fractional, bls_power,
                                          secondary_depth, odd_even_diff,
                                          centroid_shift, n_transits,
                                          log_teff_norm, logg, log_radius_norm,
                                          feh, kepmag_norm,
                                          mission_is_kepler, mission_is_tess, mission_is_k2,
                                          transit_snr]

        Returns
        -------
        Tensor of shape (batch, 1) — raw logit (apply sigmoid for probability)
        """
        global_feat   = self.global_branch(global_view)
        local_feat    = self.local_branch(local_view)
        odd_feat      = self.odd_branch(odd_view)
        even_feat     = self.even_branch(even_view)
        sec_feat      = self.secondary_branch(secondary_view)
        cen_feat      = self.centroid_branch(centroid_view)
        scalar_feat   = self.scalar_branch(scalar_features)
        wavelet_feat  = self.wavelet_branch(global_view)
        return self.fusion([global_feat, local_feat, odd_feat, even_feat,
                            sec_feat, cen_feat, scalar_feat, wavelet_feat])

    def forward_with_aux(
        self,
        global_view: Tensor,
        local_view: Tensor,
        odd_view: Tensor,
        even_view: Tensor,
        secondary_view: Tensor,
        centroid_view: Tensor,
        scalar_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass returning main logit + auxiliary regression predictions.

        Used during training only for multi-task learning.  The auxiliary heads
        predict log(1 + depth) and log(1 + duration_days) from the fused trunk
        representation, forcing the model to encode physically meaningful signals.

        Returns
        -------
        logit       : (batch, 1) — classification logit
        aux_depth   : (batch, 1) — predicted log(1 + depth)
        aux_duration: (batch, 1) — predicted log(1 + duration_days)
        """
        global_feat   = self.global_branch(global_view)
        local_feat    = self.local_branch(local_view)
        odd_feat      = self.odd_branch(odd_view)
        even_feat     = self.even_branch(even_view)
        sec_feat      = self.secondary_branch(secondary_view)
        cen_feat      = self.centroid_branch(centroid_view)
        scalar_feat   = self.scalar_branch(scalar_features)
        wavelet_feat  = self.wavelet_branch(global_view)
        logit, pooled = self.fusion(
            [global_feat, local_feat, odd_feat, even_feat,
             sec_feat, cen_feat, scalar_feat, wavelet_feat],
            return_features=True,
        )
        return logit, self.aux_depth_head(pooled), self.aux_duration_head(pooled)
