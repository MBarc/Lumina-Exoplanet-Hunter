"""
ExoNet training script.

Downloads Kepler KOI labels from the NASA Exoplanet Archive, fetches the
corresponding FITS light curves via astroquery.mast, preprocesses each light
curve with the Lumina pipeline, trains ExoNet with binary cross-entropy loss,
and exports the best checkpoint to ONNX.

Quick start
-----------
::

    python -m ml.train \\
        --fits-dir  /data/kepler_fits \\
        --output-dir /data/exonet_run \\
        --epochs 30 --batch-size 64

The script saves:
  ``<output_dir>/exonet.pt``    — best PyTorch state dict (by val AUC-ROC)
  ``<output_dir>/exonet.onnx``  — ONNX export of the best checkpoint

Dependencies (developer machine only)
--------------------------------------
    torch, sklearn, astroquery, astropy, numpy, requests
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import warnings

# Ensure stdout/stderr use UTF-8 on Windows so torch's emoji progress
# messages (✅ etc.) don't crash with a cp1252 UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from ml.inference import ExoNetInference
from ml.model import GLOBAL_LEN, LOCAL_LEN, ExoNet
from ml.preprocess import preprocess

# ── Dataset constants ─────────────────────────────────────────────────────────

_KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,koi_disposition+from+cumulative&format=csv"
)

_CONFIRMED_LABELS = {"CONFIRMED", "CANDIDATE"}   # → 1.0
_FP_LABEL         = "FALSE POSITIVE"              # → 0.0


# ── Label helpers ─────────────────────────────────────────────────────────────

def _disposition_to_label(disposition: str) -> float | None:
    """
    Map a KOI disposition string to a binary label.

    Returns ``None`` for unknown/ambiguous dispositions so those rows can be
    skipped cleanly.
    """
    d = disposition.strip().upper()
    if d in _CONFIRMED_LABELS:
        return 1.0
    if d == _FP_LABEL.upper():
        return 0.0
    return None


# ── KOI CSV download ──────────────────────────────────────────────────────────

def download_koi_table() -> list[tuple[int, float]]:
    """
    Fetch the cumulative KOI table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (kepid, label) tuples where label is 0.0 or 1.0.
    """
    print("Downloading KOI table from NASA Exoplanet Archive …")
    try:
        response = requests.get(_KOI_TAP_URL, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download KOI table: {exc}") from exc

    if not response.text.strip():
        raise RuntimeError("KOI table response was empty.")

    # Use csv.DictReader to handle quoted fields (e.g. "FALSE POSITIVE")
    import csv as _csv
    import io as _io
    reader = _csv.DictReader(_io.StringIO(response.text))
    if reader.fieldnames is None or "kepid" not in reader.fieldnames or "koi_disposition" not in reader.fieldnames:
        raise RuntimeError(
            f"Unexpected KOI table columns: {reader.fieldnames}"
        )

    records: list[tuple[int, float]] = []
    for row in reader:
        try:
            kepid = int(row["kepid"].strip())
        except (ValueError, KeyError):
            continue
        label = _disposition_to_label(row.get("koi_disposition", ""))
        if label is not None:
            records.append((kepid, label))

    print(f"  {len(records)} labelled KOIs loaded "
          f"({sum(1 for _, l in records if l == 1.0)} positives, "
          f"{sum(1 for _, l in records if l == 0.0)} negatives).")
    return records


# ── FITS download via astroquery ──────────────────────────────────────────────

def _fits_cache_path(fits_dir: Path, kepid: int) -> Path | None:
    """Return the path to a cached FITS file for ``kepid``, or None if absent."""
    # Accept any FITS file whose name contains the zero-padded kepid
    padded = str(kepid).zfill(9)
    matches = list(fits_dir.glob(f"*{padded}*.fits"))
    return matches[0] if matches else None


def _download_fits(kepid: int, fits_dir: Path) -> Path | None:
    """
    Download the long-cadence Kepler light curve for ``kepid`` using
    astroquery.mast.  Returns the local path, or None if no product is found.

    Only the first available light-curve product is downloaded to keep disk
    usage manageable during training.
    """
    try:
        from astroquery.mast import Observations  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for FITS download. "
            "Install it with: pip install astroquery"
        ) from exc

    # MAST stores Kepler targets as "kplrNNNNNNNNN" (9-digit zero-padded)
    target_name = f"kplr{kepid:09d}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_table = Observations.query_criteria(
            target_name=target_name,
            obs_collection="Kepler",
            dataproduct_type="timeseries",
        )

    if obs_table is None or len(obs_table) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        products = Observations.get_product_list(obs_table[0])
        lc_prods = Observations.filter_products(
            products,
            productSubGroupDescription="LLC",  # long-cadence light curve
            extension="fits",
        )

    if lc_prods is None or len(lc_prods) == 0:
        return None

    # Download only the first product to keep things fast
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = Observations.download_products(
            lc_prods[:1],
            download_dir=str(fits_dir),
            cache=True,
        )

    if manifest is None or len(manifest) == 0:
        return None

    local_path = Path(manifest["Local Path"][0])
    return local_path if local_path.exists() else None


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class KeplerDataset(Dataset):
    """
    PyTorch Dataset for Kepler KOI light curves.

    For each star the preprocessing pipeline is run once and the top BLS
    candidate is used.  Stars that fail preprocessing are silently skipped.

    Parameters
    ----------
    csv_path :
        Path to a CSV file with columns ``kepid`` and ``koi_disposition``.
        If ``None``, the KOI table is downloaded automatically from the
        NASA Exoplanet Archive.
    fits_dir :
        Directory used as both a cache for downloaded FITS files and as a
        search location for files already on disk.
    max_samples :
        If set, cap the dataset at this many samples (drawn from the top of
        the KOI list after label filtering).
    """

    def __init__(
        self,
        fits_dir: str | Path,
        csv_path: str | Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.fits_dir = Path(fits_dir)
        self.fits_dir.mkdir(parents=True, exist_ok=True)

        # Load label table
        if csv_path is not None:
            records = self._load_csv(Path(csv_path))
        else:
            records = download_koi_table()

        if max_samples is not None:
            records = records[:max_samples]

        # Build item list by running preprocessing
        self._items: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._labels: list[float] = []

        print(f"Preprocessing {len(records)} KOIs …")
        for idx, (kepid, label) in enumerate(records):
            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(records)} processed, "
                      f"{len(self._items)} valid so far …")

            fits_path = self._get_fits(kepid)
            if fits_path is None:
                continue

            try:
                candidates = preprocess(fits_path, n_candidates=1)
            except Exception:  # noqa: BLE001
                continue

            if not candidates:
                continue

            c = candidates[0]
            self._items.append((
                c.global_view.astype(np.float32),
                c.local_view.astype(np.float32),
                label,
            ))
            self._labels.append(label)

        print(f"Dataset ready: {len(self._items)} samples "
              f"({sum(1 for l in self._labels if l == 1.0)} positives, "
              f"{sum(1 for l in self._labels if l == 0.0)} negatives).")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_csv(csv_path: Path) -> list[tuple[int, float]]:
        """Parse a local CSV with ``kepid`` and ``koi_disposition`` columns."""
        import csv  # noqa: PLC0415

        records: list[tuple[int, float]] = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    kepid = int(row["kepid"])
                except (KeyError, ValueError):
                    continue
                label = _disposition_to_label(row.get("koi_disposition", ""))
                if label is not None:
                    records.append((kepid, label))
        return records

    def _get_fits(self, kepid: int) -> Path | None:
        """Return a local FITS path for ``kepid``, downloading if necessary."""
        cached = _fits_cache_path(self.fits_dir, kepid)
        if cached is not None:
            return cached
        try:
            return _download_fits(kepid, self.fits_dir)
        except Exception:  # noqa: BLE001
            return None

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        global_view : float32 tensor of shape (1, 2001)
        local_view  : float32 tensor of shape (1, 201)
        label       : float32 tensor of shape (1,)
        """
        gv, lv, label = self._items[idx]
        return (
            torch.from_numpy(gv).unsqueeze(0),           # (1, 2001)
            torch.from_numpy(lv).unsqueeze(0),           # (1, 201)
            torch.tensor([label], dtype=torch.float32),  # (1,)
        )

    @property
    def labels(self) -> list[float]:
        """All labels in dataset order (useful for stratified splitting)."""
        return self._labels


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ExoNet on Kepler KOI data and export to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size for train and validation loaders.",
    )
    parser.add_argument(
        "--fits-dir", type=Path, required=True,
        help="Directory for FITS cache (downloaded and/or pre-existing files).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory where exonet.pt and exonet.onnx will be written.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of data held out for validation (stratified).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap dataset size (useful for quick smoke-tests).",
    )
    parser.add_argument(
        "--csv-path", type=Path, default=None,
        help="Path to a local KOI CSV file. If omitted, downloads from NASA.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes (set >0 for multi-process loading).",
    )
    return parser.parse_args(argv)


# ── Training loop ─────────────────────────────────────────────────────────────

def _pos_weight(labels: list[float], device: torch.device) -> torch.Tensor:
    """
    Compute the positive-class weight for BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive

    A higher weight penalises false negatives more, compensating for class
    imbalance.  Clamped to [0.1, 100] to avoid numerical instability.
    """
    n_pos = sum(1 for l in labels if l == 1.0)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0], device=device)
    weight = n_neg / n_pos
    weight = max(0.1, min(weight, 100.0))
    return torch.tensor([weight], device=device)


def _run_epoch(
    model: ExoNet,
    loader: DataLoader,
    criterion: nn.BCELoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, list[float], list[float]]:
    """
    Run one epoch (train or eval).

    Parameters
    ----------
    optimizer :
        If ``None`` the model is run in eval mode (validation pass).

    Returns
    -------
    mean_loss, all_scores, all_labels
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_scores: list[float] = []
    all_labels: list[float] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for gv, lv, labels in loader:
            gv     = gv.to(device)
            lv     = lv.to(device)
            labels = labels.to(device)

            scores = model(gv, lv)          # (B, 1)
            loss   = criterion(scores, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_scores.extend(scores.detach().cpu().numpy()[:, 0].tolist())
            all_labels.extend(labels.cpu().numpy()[:, 0].tolist())

    mean_loss = total_loss / max(len(all_labels), 1)
    return mean_loss, all_scores, all_labels


def train(args: argparse.Namespace) -> None:
    """Full training pipeline."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = KeplerDataset(
        fits_dir=args.fits_dir,
        csv_path=args.csv_path,
        max_samples=args.max_samples,
    )

    if len(dataset) < 10:
        print("ERROR: dataset contains fewer than 10 usable samples. Aborting.")
        sys.exit(1)

    # Stratified train/val split
    indices    = list(range(len(dataset)))
    val_labels = dataset.labels

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            stratify=val_labels,
            random_state=42,
        )
    except ValueError:
        # Fallback to non-stratified split if a class has too few samples
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            random_state=42,
        )

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model, loss, optimiser, scheduler ────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")

    model = ExoNet().to(device)

    train_labels = [dataset.labels[i] for i in train_idx]
    pw = _pos_weight(train_labels, device)
    # BCELoss is used here; we manually scale the loss rather than using
    # BCEWithLogitsLoss because the final model layer already applies Sigmoid.
    # We achieve class weighting by multiplying the unreduced BCE output.
    criterion_unreduced = nn.BCELoss(reduction="none")

    def weighted_bce(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = criterion_unreduced(scores, targets)
        # Apply pos_weight to positive examples only
        weight = torch.where(targets == 1.0, pw.expand_as(targets), torch.ones_like(targets))
        return (loss * weight).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc   = -1.0
    pt_save_path   = output_dir / "exonet.pt"
    onnx_save_path = output_dir / "exonet.onnx"

    print(
        f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>9}  {'LR':>10}"
    )
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(
            model, train_loader, weighted_bce, optimizer, device
        )
        val_loss, val_scores, val_labels_ep = _run_epoch(
            model, val_loader, weighted_bce, None, device
        )
        scheduler.step()

        # AUC-ROC (only meaningful when both classes are present)
        try:
            val_auc = roc_auc_score(val_labels_ep, val_scores)
        except ValueError:
            val_auc = float("nan")

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"{epoch:>6}  {train_loss:>11.5f}  {val_loss:>9.5f}  "
            f"{val_auc:>9.4f}  {current_lr:>10.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), pt_save_path)
            print(f"         >> New best val AUC {best_val_auc:.4f} -- checkpoint saved.")

    print(f"\nTraining complete.  Best val AUC-ROC: {best_val_auc:.4f}")
    print(f"Best checkpoint: {pt_save_path}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    print("\nExporting best checkpoint to ONNX …")
    ExoNetInference.export_from_pytorch(pt_save_path, onnx_save_path)
    print(f"ONNX model: {onnx_save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
