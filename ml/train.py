"""
ExoNet training script.

Downloads labels from the NASA Exoplanet Archive for Kepler KOIs, TESS TOIs,
and K2 candidates, fetches the corresponding FITS light curves via
astroquery.mast, preprocesses each light curve with the Lumina pipeline,
trains ExoNet with binary cross-entropy loss, and exports the best checkpoint
to ONNX.

Quick start
-----------
::

    python -m ml.train \\
        --fits-dir  /data/fits_cache \\
        --output-dir /data/exonet_run \\
        --epochs 50 --batch-size 64

The script saves:
  ``<output_dir>/exonet.pt``        — best PyTorch state dict (by val AUC-ROC)
  ``<output_dir>/exonet.onnx``      — ONNX export of the best checkpoint
  ``<output_dir>/threshold.json``   — optimal classification threshold (val F1)

Dependencies (developer machine only)
--------------------------------------
    torch, sklearn, astroquery, astropy, numpy, requests
"""

from __future__ import annotations

import argparse
import csv
import io
import json
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from ml.inference import ExoNetInference
from ml.model import GLOBAL_LEN, LOCAL_LEN, ExoNet
from ml.preprocess import preprocess

# ── TAP URLs ──────────────────────────────────────────────────────────────────

_KOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+kepid,koi_disposition+from+cumulative&format=csv"
)
_TOI_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+tid,tfopwg_disp+from+toi&format=csv"
)
_K2_TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+epic_name,disp+from+k2candidates&format=csv"
)


# ── Label helpers ─────────────────────────────────────────────────────────────

def _koi_disposition_to_label(disposition: str) -> float | None:
    """Map a Kepler KOI disposition string to a binary label."""
    d = disposition.strip().upper()
    if d in {"CONFIRMED", "CANDIDATE"}:
        return 1.0
    if d == "FALSE POSITIVE":
        return 0.0
    return None


def _tess_disposition_to_label(disposition: str) -> float | None:
    """Map a TESS TFOPWG disposition string to a binary label."""
    d = disposition.strip().upper()
    if d in {"CP", "PC"}:
        return 1.0
    if d in {"FP", "FA"}:
        return 0.0
    return None


def _k2_disposition_to_label(disposition: str) -> float | None:
    """Map a K2 candidate disposition string to a binary label."""
    d = disposition.strip().upper()
    if d in {"CONFIRMED", "CANDIDATE"}:
        return 1.0
    if d == "FALSE POSITIVE":
        return 0.0
    return None


# ── CSV download functions ────────────────────────────────────────────────────

def download_koi_table() -> list[tuple[int, float]]:
    """
    Fetch the cumulative KOI table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (kepid, label) tuples where label is 0.0 or 1.0.
    """
    print("Downloading Kepler KOI table from NASA Exoplanet Archive ...")
    try:
        response = requests.get(_KOI_TAP_URL, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download KOI table: {exc}") from exc

    if not response.text.strip():
        raise RuntimeError("KOI table response was empty.")

    reader = csv.DictReader(io.StringIO(response.text))
    if reader.fieldnames is None or "kepid" not in reader.fieldnames or "koi_disposition" not in reader.fieldnames:
        raise RuntimeError(f"Unexpected KOI table columns: {reader.fieldnames}")

    records: list[tuple[int, float]] = []
    for row in reader:
        try:
            kepid = int(row["kepid"].strip())
        except (ValueError, KeyError):
            continue
        label = _koi_disposition_to_label(row.get("koi_disposition", ""))
        if label is not None:
            records.append((kepid, label))

    n_pos = sum(1 for _, l in records if l == 1.0)
    n_neg = sum(1 for _, l in records if l == 0.0)
    print(f"  Kepler: {len(records)} labelled KOIs ({n_pos} positives, {n_neg} negatives).")
    return records


def download_toi_table() -> list[tuple[int, float]]:
    """
    Fetch the TESS TOI table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (tid, label) tuples where label is 0.0 or 1.0.
    """
    print("Downloading TESS TOI table from NASA Exoplanet Archive ...")
    try:
        response = requests.get(_TOI_TAP_URL, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: Failed to download TESS TOI table: {exc}. Skipping.")
        return []

    if not response.text.strip():
        print("  WARNING: TESS TOI table response was empty. Skipping.")
        return []

    reader = csv.DictReader(io.StringIO(response.text))
    if reader.fieldnames is None or "tid" not in reader.fieldnames or "tfopwg_disp" not in reader.fieldnames:
        print(f"  WARNING: Unexpected TESS TOI columns: {reader.fieldnames}. Skipping.")
        return []

    records: list[tuple[int, float]] = []
    for row in reader:
        try:
            tid = int(row["tid"].strip())
        except (ValueError, KeyError):
            continue
        label = _tess_disposition_to_label(row.get("tfopwg_disp", ""))
        if label is not None:
            records.append((tid, label))

    n_pos = sum(1 for _, l in records if l == 1.0)
    n_neg = sum(1 for _, l in records if l == 0.0)
    print(f"  TESS: {len(records)} labelled TOIs ({n_pos} positives, {n_neg} negatives).")
    return records


def download_k2_table() -> list[tuple[int, float]]:
    """
    Fetch the K2 candidates table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (epic_id, label) tuples where label is 0.0 or 1.0.
    """
    print("Downloading K2 candidates table from NASA Exoplanet Archive ...")
    try:
        response = requests.get(_K2_TAP_URL, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  WARNING: Failed to download K2 candidates table: {exc}. Skipping.")
        return []

    if not response.text.strip():
        print("  WARNING: K2 candidates table response was empty. Skipping.")
        return []

    reader = csv.DictReader(io.StringIO(response.text))
    if reader.fieldnames is None or "epic_name" not in reader.fieldnames or "disp" not in reader.fieldnames:
        print(f"  WARNING: Unexpected K2 candidates columns: {reader.fieldnames}. Skipping.")
        return []

    records: list[tuple[int, float]] = []
    for row in reader:
        try:
            epic_name = row["epic_name"].strip()
            epic_id = int(epic_name.replace("EPIC", "").strip())
        except (ValueError, KeyError, AttributeError):
            continue
        label = _k2_disposition_to_label(row.get("disp", ""))
        if label is not None:
            records.append((epic_id, label))

    n_pos = sum(1 for _, l in records if l == 1.0)
    n_neg = sum(1 for _, l in records if l == 0.0)
    print(f"  K2: {len(records)} labelled candidates ({n_pos} positives, {n_neg} negatives).")
    return records


# ── FITS download helpers ─────────────────────────────────────────────────────

def _fits_cache_path(fits_dir: Path, pattern: str) -> Path | None:
    """Return the first FITS file matching *pattern* anywhere under fits_dir, or None."""
    matches = list(fits_dir.rglob(f"*{pattern}*.fits"))
    return matches[0] if matches else None


def _download_fits_kepler(kepid: int, fits_dir: Path) -> Path | None:
    """
    Download the long-cadence Kepler light curve for *kepid* via astroquery.mast.
    Returns the local path or None if no product is found.
    """
    try:
        from astroquery.mast import Observations  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for FITS download. "
            "Install it with: pip install astroquery"
        ) from exc

    target_name = f"kplr{kepid:09d}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_table = Observations.query_criteria(
            target_name=target_name,
            obs_collection="Kepler",
            productSubGroupDescription="LLC",
        )

    if obs_table is None or len(obs_table) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        products = Observations.get_product_list(obs_table[0])
        lc_prods = Observations.filter_products(
            products,
            productSubGroupDescription="LLC",
            extension="fits",
        )

    if lc_prods is None or len(lc_prods) == 0:
        return None

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


def _download_fits_tess(tid: int, fits_dir: Path) -> Path | None:
    """
    Download a TESS light curve for TIC *tid* via astroquery.mast.
    Returns the local path or None if no product is found.
    """
    try:
        from astroquery.mast import Observations  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for FITS download. "
            "Install it with: pip install astroquery"
        ) from exc

    target_name = f"TIC {tid}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_table = Observations.query_criteria(
            target_name=target_name,
            obs_collection="TESS",
            dataproduct_type="timeseries",
            productSubGroupDescription="LC",
        )

    if obs_table is None or len(obs_table) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        products = Observations.get_product_list(obs_table[0])
        lc_prods = Observations.filter_products(
            products,
            productSubGroupDescription="LC",
            extension="fits",
        )

    if lc_prods is None or len(lc_prods) == 0:
        return None

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


def _download_fits_k2(epic_id: int, fits_dir: Path) -> Path | None:
    """
    Download a K2 light curve for EPIC *epic_id* via astroquery.mast.
    Returns the local path or None if no product is found.
    """
    try:
        from astroquery.mast import Observations  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for FITS download. "
            "Install it with: pip install astroquery"
        ) from exc

    target_name = f"EPIC {epic_id}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_table = Observations.query_criteria(
            target_name=target_name,
            obs_collection="K2",
            dataproduct_type="timeseries",
            productSubGroupDescription="LLC",
        )

    if obs_table is None or len(obs_table) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        products = Observations.get_product_list(obs_table[0])
        lc_prods = Observations.filter_products(
            products,
            productSubGroupDescription="LLC",
            extension="fits",
        )

    if lc_prods is None or len(lc_prods) == 0:
        return None

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


# ── Multi-mission Dataset ─────────────────────────────────────────────────────

class MultiMissionDataset(Dataset):
    """
    PyTorch Dataset combining Kepler KOIs, TESS TOIs, and K2 candidates.

    For each target the preprocessing pipeline is run once and the top BLS
    candidate is used.  Targets that fail preprocessing are silently skipped.
    Each source (Kepler, TESS, K2) is downloaded independently and combined
    into one flat list of (fits_path, label) pairs.

    Parameters
    ----------
    fits_dir :
        Directory used as both a cache for downloaded FITS files and as a
        search location for files already on disk.
    csv_path :
        Path to a local Kepler KOI CSV file (kepid, koi_disposition columns).
        If ``None``, all three mission tables are downloaded from NASA.
    max_samples :
        If set, cap the combined dataset at this many samples (drawn from the
        top of the combined list after label filtering).
    augment :
        If ``True``, apply random augmentation to positive examples at
        __getitem__ time (50% probability per call).
    """

    def __init__(
        self,
        fits_dir: str | Path,
        csv_path: str | Path | None = None,
        max_samples: int | None = None,
        augment: bool = True,
    ) -> None:
        self.fits_dir = Path(fits_dir)
        self.fits_dir.mkdir(parents=True, exist_ok=True)
        self.augment = augment

        # ── Collect (fits_path, label) pairs from all missions ────────────────
        all_pairs: list[tuple[Path, float]] = []

        # --- Kepler KOIs ---
        if csv_path is not None:
            koi_records = self._load_koi_csv(Path(csv_path))
        else:
            koi_records = download_koi_table()

        kepler_pairs = self._resolve_kepler(koi_records)
        all_pairs.extend(kepler_pairs)

        # --- TESS TOIs ---
        toi_records = download_toi_table()
        tess_pairs = self._resolve_tess(toi_records)
        all_pairs.extend(tess_pairs)

        # --- K2 candidates ---
        k2_records = download_k2_table()
        k2_pairs = self._resolve_k2(k2_records)
        all_pairs.extend(k2_pairs)

        if max_samples is not None:
            all_pairs = all_pairs[:max_samples]

        # ── Preprocess each FITS file ─────────────────────────────────────────
        # Items store: (global_view, local_view, scalar_features, label)
        self._items: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._labels: list[float] = []

        print(f"Preprocessing {len(all_pairs)} targets across all missions ...")
        for idx, (fits_path, label) in enumerate(all_pairs):
            if (idx + 1) % 100 == 0:
                print(f"  {idx + 1}/{len(all_pairs)} processed, "
                      f"{len(self._items)} valid so far ...")

            try:
                candidates = preprocess(fits_path, n_candidates=1)
            except Exception:  # noqa: BLE001
                continue

            if not candidates:
                continue

            c = candidates[0]
            scalar = np.array(
                [c.period, c.duration, c.depth, c.bls_power],
                dtype=np.float32,
            )
            self._items.append((
                c.global_view.astype(np.float32),
                c.local_view.astype(np.float32),
                scalar,
                label,
            ))
            self._labels.append(label)

        n_pos = sum(1 for l in self._labels if l == 1.0)
        n_neg = sum(1 for l in self._labels if l == 0.0)
        print(f"Dataset ready: {len(self._items)} samples "
              f"({n_pos} positives, {n_neg} negatives).")

    # ── Mission-specific resolution helpers ───────────────────────────────────

    def _resolve_kepler(self, records: list[tuple[int, float]]) -> list[tuple[Path, float]]:
        """Locate or download FITS for each Kepler KOI; return (path, label) pairs."""
        pairs: list[tuple[Path, float]] = []
        for kepid, label in records:
            padded = str(kepid).zfill(9)
            cached = _fits_cache_path(self.fits_dir, padded)
            if cached is not None:
                pairs.append((cached, label))
                continue
            try:
                path = _download_fits_kepler(kepid, self.fits_dir)
            except Exception:  # noqa: BLE001
                path = None
            if path is not None:
                pairs.append((path, label))
        return pairs

    def _resolve_tess(self, records: list[tuple[int, float]]) -> list[tuple[Path, float]]:
        """Locate or download FITS for each TESS TOI; return (path, label) pairs."""
        pairs: list[tuple[Path, float]] = []
        for tid, label in records:
            cached = _fits_cache_path(self.fits_dir, f"TIC{tid}")
            if cached is None:
                cached = _fits_cache_path(self.fits_dir, str(tid))
            if cached is not None:
                pairs.append((cached, label))
                continue
            try:
                path = _download_fits_tess(tid, self.fits_dir)
            except Exception:  # noqa: BLE001
                path = None
            if path is not None:
                pairs.append((path, label))
        return pairs

    def _resolve_k2(self, records: list[tuple[int, float]]) -> list[tuple[Path, float]]:
        """Locate or download FITS for each K2 candidate; return (path, label) pairs."""
        pairs: list[tuple[Path, float]] = []
        for epic_id, label in records:
            cached = _fits_cache_path(self.fits_dir, f"EPIC{epic_id}")
            if cached is None:
                cached = _fits_cache_path(self.fits_dir, str(epic_id))
            if cached is not None:
                pairs.append((cached, label))
                continue
            try:
                path = _download_fits_k2(epic_id, self.fits_dir)
            except Exception:  # noqa: BLE001
                path = None
            if path is not None:
                pairs.append((path, label))
        return pairs

    # ── Static CSV loader (Kepler only) ───────────────────────────────────────

    @staticmethod
    def _load_koi_csv(csv_path: Path) -> list[tuple[int, float]]:
        """Parse a local CSV with ``kepid`` and ``koi_disposition`` columns."""
        records: list[tuple[int, float]] = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    kepid = int(row["kepid"])
                except (KeyError, ValueError):
                    continue
                label = _koi_disposition_to_label(row.get("koi_disposition", ""))
                if label is not None:
                    records.append((kepid, label))
        return records

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        global_view    : float32 tensor of shape (1, 2001)
        local_view     : float32 tensor of shape (1, 201)
        scalar_tensor  : float32 tensor of shape (4,)  — [period, duration, depth, bls_power]
        label          : float32 tensor of shape (1,)
        """
        gv, lv, scalar, label = self._items[idx]

        # Runtime augmentation for positive examples
        if self.augment and label == 1.0 and np.random.random() < 0.5:
            gv = gv.copy()
            lv = lv.copy()
            if np.random.random() < 0.5:
                # Random phase shift: simulate different t0
                shift = np.random.randint(-100, 101)
                gv = np.roll(gv, shift)
            else:
                # Gaussian noise injection on both views
                gv = gv + np.random.normal(0, 0.05, gv.shape).astype(np.float32)
                lv = lv + np.random.normal(0, 0.05, lv.shape).astype(np.float32)

        return (
            torch.from_numpy(gv).unsqueeze(0),            # (1, 2001)
            torch.from_numpy(lv).unsqueeze(0),            # (1, 201)
            torch.from_numpy(scalar),                     # (4,)
            torch.tensor([label], dtype=torch.float32),   # (1,)
        )

    @property
    def labels(self) -> list[float]:
        """All labels in dataset order (useful for stratified splitting)."""
        return self._labels


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ExoNet on multi-mission data (Kepler/TESS/K2) and export to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience: stop if val AUC does not improve for this many epochs.",
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
        help="Directory where exonet.pt, exonet.onnx and threshold.json will be written.",
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
        help="Path to a local Kepler KOI CSV file. If omitted, all mission tables are downloaded from NASA.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker processes (set >0 for multi-process loading).",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation.",
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
        for gv, lv, scalar, labels in loader:
            gv     = gv.to(device)
            lv     = lv.to(device)
            scalar = scalar.to(device)
            labels = labels.to(device)

            scores = model(gv, lv, scalar)   # (B, 1)
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


# ── Threshold sweep ───────────────────────────────────────────────────────────

def _tune_threshold(
    val_scores: list[float],
    val_labels: list[float],
    output_dir: Path,
) -> float:
    """
    Sweep thresholds from 0.1 to 0.9 in steps of 0.01, pick the one with the
    highest F1 on the validation set, print a summary, and save threshold.json.

    Returns the optimal threshold.
    """
    scores_arr = np.array(val_scores)
    labels_arr = np.array(val_labels)

    best_thresh = 0.5
    best_f1 = -1.0

    thresholds = np.arange(0.10, 0.91, 0.01)
    for thresh in thresholds:
        preds = (scores_arr >= thresh).astype(int)
        f1 = f1_score(labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    print(f"Optimal threshold: {best_thresh:.2f}  (val F1: {best_f1:.4f})")

    threshold_path = output_dir / "threshold.json"
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump({"threshold": round(best_thresh, 2)}, fh)
    print(f"Threshold saved: {threshold_path}")

    return best_thresh


# ── Full training pipeline ────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Full training pipeline."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = MultiMissionDataset(
        fits_dir=args.fits_dir,
        csv_path=args.csv_path,
        max_samples=args.max_samples,
        augment=not args.no_augment,
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
    best_val_auc      = -1.0
    best_val_scores: list[float] = []
    best_val_labels: list[float] = []
    epochs_no_improve = 0
    pt_save_path      = output_dir / "exonet.pt"
    onnx_save_path    = output_dir / "exonet.onnx"

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

        improved = not (val_auc != val_auc) and val_auc > best_val_auc  # NaN-safe
        if improved:
            best_val_auc = val_auc
            best_val_scores = list(val_scores)
            best_val_labels = list(val_labels_ep)
            epochs_no_improve = 0
            torch.save(model.state_dict(), pt_save_path)
            print(f"         >> New best val AUC {best_val_auc:.4f} -- checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: val AUC has not improved for {args.patience} epochs.")
                break

    print(f"\nTraining complete.  Best val AUC-ROC: {best_val_auc:.4f}")
    print(f"Best checkpoint: {pt_save_path}")

    # ── Threshold tuning ──────────────────────────────────────────────────────
    if best_val_scores and best_val_labels:
        print("\nRunning threshold sweep on validation set ...")
        _tune_threshold(best_val_scores, best_val_labels, output_dir)

    # ── ONNX export ───────────────────────────────────────────────────────────
    print("\nExporting best checkpoint to ONNX ...")
    ExoNetInference.export_from_pytorch(pt_save_path, onnx_save_path)
    print(f"ONNX model: {onnx_save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
