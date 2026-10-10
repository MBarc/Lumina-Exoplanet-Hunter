"""
Masked Autoencoder (MAE) pretraining for ExoNet's GlobalBranch and LocalBranch.

Pretrain the GlobalBranch and/or LocalBranch encoders on unlabeled light curves
before supervised training.  The encoder learns to reconstruct randomly masked
bins, forcing it to build general representations of light curve structure
(transits, stellar variability, instrumental artifacts) without needing labels.

After pretraining, pass --pretrained-global <path> and/or --pretrained-local
<path> to ml.train to initialise the corresponding branch from pretrained weights.

Usage
-----
::

    # Pretrain GlobalBranch only (original behaviour)
    python -m ml.pretrain_mae \\
        --cache-file  training_runs/preprocess_cache.npz \\
        --output-dir  training_runs/mae_pretrain \\
        --epochs 30 --mask-ratio 0.40

    # Pretrain both GlobalBranch and LocalBranch
    python -m ml.pretrain_mae \\
        --cache-file  training_runs/preprocess_cache.npz \\
        --output-dir  training_runs/mae_pretrain \\
        --epochs 30 --mask-ratio 0.40 --pretrain-local
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.model import GlobalBranch, LocalBranch

# ── Dataset ────────────────────────────────────────────────────────────────────

class _MAEDataset(Dataset):
    """
    Wraps global_views from a preprocessing cache for MAE pretraining.

    Each item: (masked_view, original_view, mask) where mask[i]=True means
    bin i was masked and must be reconstructed.
    """

    def __init__(self, cache_path: Path, mask_ratio: float = 0.40, deterministic: bool = False) -> None:
        data = np.load(cache_path, allow_pickle=False)
        gvs  = data["global_views"].astype(np.float32)   # (N, 2001) or (N, 2, 2001)
        # Support both 1-channel and 2-channel caches; use channel 0 (detrended)
        if gvs.ndim == 3:
            self._gvs = gvs        # (N, 2, 2001) — keep both channels
        else:
            self._gvs = np.stack([gvs, gvs], axis=1)  # (N, 2, 2001)
        self.mask_ratio  = mask_ratio
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self._gvs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig = self._gvs[idx].copy()   # (2, 2001)
        n_bins = orig.shape[1]         # 2001

        # Random binary mask over time bins (same mask applied to both channels)
        if self.deterministic:
            # `idx` is the absolute cache position: torch.utils.data.Subset calls
            # dataset[self.indices[i]], passing the remapped absolute index, not the
            # local Subset index.  Using the absolute index as the RNG seed therefore
            # produces a unique, stable mask for each cache row.
            mask = np.random.default_rng(idx).random(n_bins) < self.mask_ratio
        else:
            mask = np.random.random(n_bins) < self.mask_ratio   # True = masked

        masked = orig.copy()
        masked[:, mask] = 0.0   # zero out masked bins

        return (
            torch.from_numpy(masked),                            # (2, 2001)
            torch.from_numpy(orig),                              # (2, 2001)
            torch.from_numpy(mask.astype(np.float32)),           # (2001,)
        )


# ── Model ──────────────────────────────────────────────────────────────────────

class _MAEModel(nn.Module):
    """
    Encoder-decoder for MAE pretraining.

    Encoder : GlobalBranch (outputs 512-d)
    Decoder : Linear(512 → 2 × 2001) — reconstructs both channels so the encoder
              receives supervision for both the detrended (ch 0) and prenorm (ch 1)
              representations.  Reconstructing only ch 0 left ch 1 features underfit.
    """

    def __init__(self, use_se: bool = True) -> None:
        super().__init__()
        self.encoder = GlobalBranch(use_se=use_se, in_channels=2)
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * 2001),   # reconstructs both channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, 2001) — masked input

        Returns
        -------
        (B, 2, 2001) — reconstructed both channels
        """
        encoded = self.encoder(x)              # (B, 512)
        recon   = self.decoder(encoded)         # (B, 2 * 2001)
        return recon.view(recon.size(0), 2, 2001)


# ── Training loop ──────────────────────────────────────────────────────────────

def pretrain_mae(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix 10: seed all RNGs for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MAE pretraining on {device}", flush=True)

    # G5: 90/10 train/val split for validation curve and best-epoch checkpoint.
    # M-2: create separate dataset instances so val uses a deterministic mask per sample.
    # Exclude the same test set that supervised training holds out, to prevent
    # the MAE encoder from seeing test-set light curve structure during pretraining.
    from sklearn.model_selection import train_test_split as _tts
    # Load labels to stratify the split identically to supervised training.
    _cache_labels = np.load(Path(args.cache_file), allow_pickle=False)["labels"]
    _strat = (_cache_labels >= 0.5).astype(int)
    n_total = len(_cache_labels)
    _all_idx = np.arange(n_total)
    _train_val_idx, _ = _tts(_all_idx, test_size=args.exclude_test_frac, random_state=args.exclude_test_seed, stratify=_strat)

    train_ds_full = torch.utils.data.Subset(
        _MAEDataset(Path(args.cache_file), mask_ratio=args.mask_ratio, deterministic=False),
        _train_val_idx.tolist()
    )
    val_ds_full = torch.utils.data.Subset(
        _MAEDataset(Path(args.cache_file), mask_ratio=args.mask_ratio, deterministic=True),
        _train_val_idx.tolist()
    )

    n_mae_val   = max(1, int(0.10 * len(train_ds_full)))
    n_mae_train = len(train_ds_full) - n_mae_val
    indices = torch.randperm(len(train_ds_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:n_mae_train]
    val_indices   = indices[n_mae_train:]
    train_ds = torch.utils.data.Subset(train_ds_full.dataset, [train_ds_full.indices[i] for i in train_indices])
    val_ds   = torch.utils.data.Subset(val_ds_full.dataset,   [val_ds_full.indices[i]   for i in val_indices])

    # M-1: seed numpy per worker for reproducible augmentation.
    _seed = args.seed
    def _worker_init(worker_id: int) -> None:
        np.random.seed(_seed + worker_id)
        random.seed(_seed + worker_id)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        worker_init_fn=_worker_init,
    )

    model = _MAEModel(use_se=args.use_se).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 5-epoch linear LR warmup before cosine decay.
    # The cosine scheduler is created AFTER warmup so its internal step counter
    # starts at 0 when cosine decay begins, preventing a PyTorch UserWarning about
    # stepping schedulers before the optimizer and ensuring the cosine curve spans
    # exactly (epochs - warmup_epochs) steps from args.lr down to 1e-6.
    warmup_epochs = min(5, args.epochs // 4)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0,
    )
    # Placeholder; replaced after warmup completes.
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    print(f"\n{'='*60}", flush=True)
    print(f"  MAE pretraining — {n_mae_train:,} train / {n_mae_val:,} val light curves", flush=True)
    print(f"  mask_ratio={args.mask_ratio}  epochs={args.epochs}  batch={args.batch_size}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    best_val_loss = float("inf")
    out_path = output_dir / "global_branch_pretrained.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for masked, original, mask in loader:
            masked   = masked.to(device)    # (B, 2, 2001)
            original = original.to(device)  # (B, 2, 2001)
            mask     = mask.to(device)      # (B, 2001) — 1 where masked

            recon = model(masked)           # (B, 2, 2001) — reconstructed both channels

            # MSE loss only on masked bins, averaged over both channels.
            # Fix 9: normalise per-sample (divide each sample's loss by its own
            # masked-bin count, then average over the batch).  The previous
            # per-batch normalisation made the effective loss magnitude scale
            # inversely with batch size, coupling learning rate to batch size.
            # mask is (B, 2001); expand to (B, 2, 2001) for both channels.
            mask_2ch = mask.unsqueeze(1).expand_as(recon)   # (B, 2, 2001)
            loss_all = (recon - original) ** 2              # (B, 2, 2001)
            # Sum over channel and bin dimensions; normalise by masked-bin count × 2 channels.
            loss = (
                (loss_all * mask_2ch).sum(dim=(1, 2))
                / (mask.sum(dim=1) * 2 + 1e-8)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
            if epoch == warmup_epochs:
                # Warmup complete — create cosine scheduler now so its step counter
                # starts at 0 and decays over the remaining epochs.
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs, eta_min=1e-6
                )
        elif scheduler is not None:
            scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # G5: compute validation reconstruction loss.
        model.eval()
        val_loss_sum = 0.0
        val_batches  = 0
        with torch.no_grad():
            for masked_v, original_v, mask_v in val_loader:
                masked_v   = masked_v.to(device)
                original_v = original_v.to(device)
                mask_v     = mask_v.to(device)
                recon_v     = model(masked_v)                          # (B, 2, 2001)
                mask_2ch_v  = mask_v.unsqueeze(1).expand_as(recon_v)  # (B, 2, 2001)
                loss_all_v  = (recon_v - original_v) ** 2
                loss_v      = (
                    (loss_all_v * mask_2ch_v).sum(dim=(1, 2))
                    / (mask_v.sum(dim=1) * 2 + 1e-8)
                ).mean()
                val_loss_sum += loss_v.item()
                val_batches  += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  lr={lr_now:.2e}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        # G5: save checkpoint when val loss improves (best epoch, not last epoch).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.encoder.state_dict(), out_path)
            print(f"  >> New best val loss {best_val_loss:.5f} — checkpoint saved.", flush=True)

    print(f"\nBest validation reconstruction loss: {best_val_loss:.5f}", flush=True)
    print(f"Pretrained GlobalBranch saved to: {out_path}", flush=True)
    print("Pass --pretrained-global to ml.train to use these weights.", flush=True)


# ── LocalBranch MAE ────────────────────────────────────────────────────────────

class _LocalMAEDataset(Dataset):
    """
    Wraps local_views from a preprocessing cache for LocalBranch MAE pretraining.

    Each item: (masked_view, original_view, mask) where mask[i]=True means
    bin i was masked and must be reconstructed.
    """

    def __init__(self, cache_path: Path, mask_ratio: float = 0.50, deterministic: bool = False) -> None:
        data = np.load(cache_path, allow_pickle=False)
        lvs = data["local_views"].astype(np.float32)   # (N, 201) or (N, 2, 201)
        if lvs.ndim == 3:
            self._lvs = lvs       # (N, 2, 201) — keep both channels
        else:
            self._lvs = np.stack([lvs, lvs], axis=1)  # (N, 2, 201)
        self.mask_ratio  = mask_ratio
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self._lvs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig = self._lvs[idx].copy()   # (2, 201)
        n_bins = orig.shape[1]         # 201

        if self.deterministic:
            mask = np.random.default_rng(idx).random(n_bins) < self.mask_ratio
        else:
            mask = np.random.random(n_bins) < self.mask_ratio

        masked = orig.copy()
        masked[:, mask] = 0.0

        return (
            torch.from_numpy(masked),                            # (2, 201)
            torch.from_numpy(orig),                              # (2, 201)
            torch.from_numpy(mask.astype(np.float32)),           # (201,)
        )


class _LocalMAEModel(nn.Module):
    """
    Encoder-decoder for LocalBranch MAE pretraining.

    Encoder : LocalBranch (outputs 256-d)
    Decoder : Linear(256 → 2 × 201) — reconstructs both channels
    """

    def __init__(self, use_se: bool = True) -> None:
        super().__init__()
        self.encoder = LocalBranch(use_se=use_se, in_channels=2)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * 201),   # reconstructs both channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)              # (B, 256)
        recon   = self.decoder(encoded)         # (B, 2 * 201)
        return recon.view(recon.size(0), 2, 201)


def pretrain_local_mae(args: argparse.Namespace) -> None:
    """Pretrain the LocalBranch encoder using MAE on local_views from the cache."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LocalBranch MAE pretraining on {device}", flush=True)

    from sklearn.model_selection import train_test_split as _tts
    _cache_labels = np.load(Path(args.cache_file), allow_pickle=False)["labels"]
    _strat = (_cache_labels >= 0.5).astype(int)
    n_total = len(_cache_labels)
    _all_idx = np.arange(n_total)
    _train_val_idx, _ = _tts(_all_idx, test_size=args.exclude_test_frac,
                              random_state=args.exclude_test_seed, stratify=_strat)

    # Use a higher mask_ratio for local views (shorter sequence → need more masking challenge)
    local_mask_ratio = min(args.mask_ratio + 0.10, 0.70)

    train_ds_full = torch.utils.data.Subset(
        _LocalMAEDataset(Path(args.cache_file), mask_ratio=local_mask_ratio, deterministic=False),
        _train_val_idx.tolist()
    )
    val_ds_full = torch.utils.data.Subset(
        _LocalMAEDataset(Path(args.cache_file), mask_ratio=local_mask_ratio, deterministic=True),
        _train_val_idx.tolist()
    )

    n_mae_val   = max(1, int(0.10 * len(train_ds_full)))
    n_mae_train = len(train_ds_full) - n_mae_val
    indices = torch.randperm(len(train_ds_full), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:n_mae_train]
    val_indices   = indices[n_mae_train:]
    train_ds = torch.utils.data.Subset(train_ds_full.dataset, [train_ds_full.indices[i] for i in train_indices])
    val_ds   = torch.utils.data.Subset(val_ds_full.dataset,   [val_ds_full.indices[i]   for i in val_indices])

    _seed = args.seed
    def _worker_init(worker_id: int) -> None:
        np.random.seed(_seed + worker_id)
        random.seed(_seed + worker_id)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        worker_init_fn=_worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False,
                            worker_init_fn=_worker_init)

    model = _LocalMAEModel(use_se=args.use_se).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = min(5, args.epochs // 4)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    print(f"\n{'='*60}", flush=True)
    print(f"  LocalBranch MAE — {n_mae_train:,} train / {n_mae_val:,} val light curves", flush=True)
    print(f"  mask_ratio={local_mask_ratio:.2f}  epochs={args.epochs}  batch={args.batch_size}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    best_val_loss = float("inf")
    out_path = output_dir / "local_branch_pretrained.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for masked, original, mask in loader:
            masked   = masked.to(device)    # (B, 2, 201)
            original = original.to(device)
            mask     = mask.to(device)      # (B, 201)

            recon = model(masked)           # (B, 2, 201)
            mask_2ch = mask.unsqueeze(1).expand_as(recon)
            loss_all = (recon - original) ** 2
            loss = (
                (loss_all * mask_2ch).sum(dim=(1, 2))
                / (mask.sum(dim=1) * 2 + 1e-8)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
            if epoch == warmup_epochs:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs, eta_min=1e-6
                )
        elif scheduler is not None:
            scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches  = 0
        with torch.no_grad():
            for masked_v, original_v, mask_v in val_loader:
                masked_v   = masked_v.to(device)
                original_v = original_v.to(device)
                mask_v     = mask_v.to(device)
                recon_v     = model(masked_v)
                mask_2ch_v  = mask_v.unsqueeze(1).expand_as(recon_v)
                loss_all_v  = (recon_v - original_v) ** 2
                loss_v      = (
                    (loss_all_v * mask_2ch_v).sum(dim=(1, 2))
                    / (mask_v.sum(dim=1) * 2 + 1e-8)
                ).mean()
                val_loss_sum += loss_v.item()
                val_batches  += 1
        val_loss = val_loss_sum / max(val_batches, 1)

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  lr={lr_now:.2e}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.encoder.state_dict(), out_path)
            print(f"  >> New best val loss {best_val_loss:.5f} — checkpoint saved.", flush=True)

    print(f"\nBest val reconstruction loss: {best_val_loss:.5f}", flush=True)
    print(f"Pretrained LocalBranch saved to: {out_path}", flush=True)
    print("Pass --pretrained-local to ml.train to use these weights.", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAE pretraining for ExoNet GlobalBranch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-file",   type=Path, required=True,
                   help="Preprocessing cache .npz (must contain global_views).")
    p.add_argument("--output-dir",   type=Path, required=True,
                   help="Directory for pretrained weights.")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--mask-ratio",   type=float, default=0.40,
                   help="Fraction of bins to mask during pretraining.")
    p.add_argument("--num-workers",  type=int,   default=0)
    # SE blocks are on by default. Use --no-se to disable (must match supervised training config).
    # Note: --use-se is intentionally omitted — it would be a no-op since use_se defaults to True.
    p.add_argument("--no-se",   action="store_false", dest="use_se",
                   help="Disable SE blocks in GlobalBranch (default: SE enabled).")
    p.set_defaults(use_se=True)
    p.add_argument("--seed",    type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--exclude-test-frac", type=float, default=0.10,
                   help="Same test fraction as supervised training to exclude from MAE pretraining. Default: 0.10.")
    p.add_argument("--exclude-test-seed", type=int, default=42,
                   help="Same seed as supervised training for test split exclusion. Default: 42.")
    p.add_argument("--pretrain-local", action="store_true", default=False,
                   help="Also pretrain the LocalBranch encoder using MAE on local_views. "
                        "Saves local_branch_pretrained.pt alongside global_branch_pretrained.pt. "
                        "Requires 'local_views' key in the cache .npz file.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    pretrain_mae(args)
    if args.pretrain_local:
        pretrain_local_mae(args)
