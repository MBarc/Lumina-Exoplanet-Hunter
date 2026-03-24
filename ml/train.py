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
import time
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from ml.inference import ExoNetInference
from ml.model import GLOBAL_LEN, LOCAL_LEN, ExoNet
from ml.preprocess import preprocess

# ── Logging helpers ───────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Print *msg* and flush stdout immediately (important when piped to a file)."""
    print(msg, flush=True)


def _pbar(current: int, total: int, width: int = 30) -> str:
    """Return a text progress bar string: [████░░░░] 42/100 (42.0%)"""
    frac   = current / max(total, 1)
    filled = int(width * frac)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({frac * 100:.1f}%)"


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _eta(elapsed: float, current: int, total: int) -> str:
    if current <= 0 or elapsed <= 0:
        return "?"
    rate = current / elapsed          # items per second
    remaining = (total - current) / rate
    return _fmt_elapsed(remaining)


def _stream_download(url: str, label: str, timeout: tuple = (15, 60)) -> str:
    """
    Download *url* with streaming and log bytes received as they arrive.

    Reports progress every 256 KB so the user can see the download is alive.
    Returns the full response body as a string, or raises requests.RequestException.
    """
    CHUNK = 256 * 1024   # 256 KB per read
    REPORT_EVERY = 256 * 1024  # log a line every 256 KB received

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    chunks: list[bytes] = []
    total_bytes = 0
    last_reported = 0
    t0 = time.time()

    for chunk in response.iter_content(chunk_size=CHUNK):
        if chunk:
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes - last_reported >= REPORT_EVERY:
                elapsed = time.time() - t0
                rate_kbs = total_bytes / max(elapsed, 0.001) / 1024
                _log(f"  {label}  {total_bytes / 1024:.0f} KB received  "
                     f"({rate_kbs:.0f} KB/s)  elapsed={_fmt_elapsed(elapsed)}")
                last_reported = total_bytes

    elapsed = time.time() - t0
    _log(f"  {label}  {total_bytes / 1024:.0f} KB total  "
         f"elapsed={_fmt_elapsed(elapsed)}  — download complete")
    return b"".join(chunks).decode("utf-8", errors="replace")


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
    _log("  Contacting NASA Exoplanet Archive (Kepler KOI table) ...")
    t0 = time.time()
    try:
        text = _stream_download(_KOI_TAP_URL, "Kepler KOI")
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download KOI table: {exc}") from exc

    if not text.strip():
        raise RuntimeError("KOI table response was empty.")

    reader = csv.DictReader(io.StringIO(text))
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
    _log(f"  Parsed {len(records)} labelled KOIs in {_fmt_elapsed(time.time() - t0)}  "
         f"|  {n_pos} positives  {n_neg} negatives")
    return records


def download_toi_table() -> list[tuple[int, float]]:
    """
    Fetch the TESS TOI table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (tid, label) tuples where label is 0.0 or 1.0.
    """
    _log("  Contacting NASA Exoplanet Archive (TESS TOI table) ...")
    t0 = time.time()
    try:
        text = _stream_download(_TOI_TAP_URL, "TESS TOI")
    except requests.RequestException as exc:
        _log(f"  WARNING: Failed to download TESS TOI table: {exc}. Skipping.")
        return []

    if not text.strip():
        _log("  WARNING: TESS TOI table response was empty. Skipping.")
        return []

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "tid" not in reader.fieldnames or "tfopwg_disp" not in reader.fieldnames:
        _log(f"  WARNING: Unexpected TESS TOI columns: {reader.fieldnames}. Skipping.")
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
    _log(f"  Parsed {len(records)} labelled TOIs in {_fmt_elapsed(time.time() - t0)}  "
         f"|  {n_pos} positives  {n_neg} negatives")
    return records


def download_k2_table() -> list[tuple[int, float]]:
    """
    Fetch the K2 candidates table from NASA Exoplanet Archive via TAP.

    Returns
    -------
    list of (epic_id, label) tuples where label is 0.0 or 1.0.
    """
    _log("  Contacting NASA Exoplanet Archive (K2 candidates table) ...")
    t0 = time.time()
    try:
        text = _stream_download(_K2_TAP_URL, "K2 candidates")
    except requests.RequestException as exc:
        _log(f"  WARNING: Failed to download K2 candidates table: {exc}. Skipping.")
        return []

    if not text.strip():
        _log("  WARNING: K2 candidates table response was empty. Skipping.")
        return []

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "epic_name" not in reader.fieldnames or "disp" not in reader.fieldnames:
        _log(f"  WARNING: Unexpected K2 candidates columns: {reader.fieldnames}. Skipping.")
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
    _log(f"  Parsed {len(records)} labelled K2 candidates in {_fmt_elapsed(time.time() - t0)}  "
         f"|  {n_pos} positives  {n_neg} negatives")
    return records


# ── FITS download helpers ─────────────────────────────────────────────────────

def _build_fits_index(fits_dir: Path) -> list[tuple[str, Path]]:
    """
    Scan *fits_dir* once and return a list of (lowercase_filename, full_path)
    for every .fits file found.  Used to replace per-KOI rglob calls (O(n²))
    with a single scan + in-memory substring search (O(n + k)).
    """
    _log(f"  Scanning FITS cache: {fits_dir} ...")
    t0 = time.time()
    index = [(p.name.lower(), p) for p in fits_dir.rglob("*.fits")]
    _log(f"  Found {len(index)} FITS files in {_fmt_elapsed(time.time() - t0)}.")
    return index


def _fits_cache_lookup(index: list[tuple[str, Path]], pattern: str) -> Path | None:
    """Return the first cached FITS path whose filename contains *pattern* (case-insensitive)."""
    pat = pattern.lower()
    for name, path in index:
        if pat in name:
            return path
    return None


def _mast_download(
    target_name: str,
    obs_collection: str,
    fits_dir: Path,
    product_subgroup: str,
    dataproduct_type: str = "timeseries",
    query_key: str = "target_name",
) -> Path | None:
    """
    Generic MAST downloader used by all three mission helpers.

    Parameters
    ----------
    target_name :
        MAST target identifier (e.g. ``"kplr002440757"``, ``"TIC 261136679"``).
    obs_collection :
        MAST collection name (``"Kepler"``, ``"TESS"``, ``"K2"``).
    fits_dir :
        Local directory used as both a download destination and a cache.
    product_subgroup :
        Value for the ``productSubGroupDescription`` filter applied to the
        product list **after** the observation query (not in ``query_criteria``).
    dataproduct_type :
        Product type filter for the observation query (default ``"timeseries"``).
    query_key :
        The MAST query field to use for the target name. Kepler/K2 use
        ``"target_name"``; TESS requires ``"objectname"``.
    """
    try:
        from astroquery.mast import Observations  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for FITS download. "
            "Install it with: pip install astroquery"
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_table = Observations.query_criteria(
            **{query_key: target_name},
            obs_collection=obs_collection,
            dataproduct_type=dataproduct_type,
        )

    if obs_table is None or len(obs_table) == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        products = Observations.get_product_list(obs_table[0])
        lc_prods = Observations.filter_products(
            products,
            productSubGroupDescription=product_subgroup,
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


def _download_fits_kepler(kepid: int, fits_dir: Path) -> Path | None:
    """Download the long-cadence Kepler light curve for *kepid*."""
    return _mast_download(
        target_name=f"kplr{kepid:09d}",
        obs_collection="Kepler",
        fits_dir=fits_dir,
        product_subgroup="LLC",
    )


def _download_fits_tess(tid: int, fits_dir: Path) -> Path | None:
    """Download a TESS light curve for TIC *tid*."""
    return _mast_download(
        target_name=f"TIC {tid}",
        obs_collection="TESS",
        fits_dir=fits_dir,
        product_subgroup="LC",
        query_key="objectname",   # TESS requires objectname, not target_name
    )


def _download_fits_k2(epic_id: int, fits_dir: Path) -> Path | None:
    """Download a K2 light curve for EPIC *epic_id*."""
    return _mast_download(
        target_name=f"EPIC {epic_id}",
        obs_collection="K2",
        fits_dir=fits_dir,
        product_subgroup="LLC",
    )


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
        cache_only: bool = False,
        cache_file: str | Path | None = None,
    ) -> None:
        self.fits_dir   = Path(fits_dir)
        self.fits_dir.mkdir(parents=True, exist_ok=True)
        self.augment    = augment
        self.cache_only = cache_only

        # ── Preprocessing cache: fast-load if available ───────────────────────
        if cache_file is not None:
            cache_file = Path(cache_file)
            if cache_file.exists():
                _log(f"\n  Loading preprocessing cache: {cache_file}")
                t0_load = time.time()
                data = np.load(cache_file, allow_pickle=False)
                gvs     = data["global_views"]   # (N, 2001)
                lvs     = data["local_views"]    # (N, 201)
                scalars = data["scalars"]        # (N, 6)
                labels  = data["labels"]         # (N,)
                self._items = [
                    (gvs[i], lvs[i], scalars[i], float(labels[i]))
                    for i in range(len(labels))
                ]
                self._labels = [float(l) for l in labels]
                n_pos = int((labels == 1.0).sum())
                n_neg = int((labels == 0.0).sum())
                _log(f"  Loaded {len(self._items)} samples from cache in "
                     f"{_fmt_elapsed(time.time() - t0_load)}  "
                     f"|  {n_pos} positives  {n_neg} negatives")
                return   # skip all FITS download + preprocessing

        _log("\n" + "=" * 65)
        _log("  Building MultiMission dataset")
        _log("=" * 65)

        # ── Step 1-2: Kepler ──────────────────────────────────────────────────
        _log("\n[Step 1/6]  Downloading Kepler label table ...")
        if csv_path is not None:
            koi_records = self._load_koi_csv(Path(csv_path))
            _log(f"  Loaded {len(koi_records)} records from local CSV: {csv_path}")
        else:
            koi_records = download_koi_table()

        # Build the FITS index once — shared by all three resolve steps.
        fits_index = _build_fits_index(self.fits_dir)

        if self.cache_only:
            _log("  --cache-only: MAST downloads disabled. Using local cache only.")

        _log(f"\n[Step 2/6]  Resolving Kepler FITS files  ({len(koi_records)} targets) ...")
        kepler_pairs = self._resolve_kepler(koi_records, fits_index, self.cache_only)

        # ── Step 3-4: TESS ────────────────────────────────────────────────────
        _log(f"\n[Step 3/6]  Downloading TESS label table ...")
        toi_records = download_toi_table()

        _log(f"\n[Step 4/6]  Resolving TESS FITS files  ({len(toi_records)} targets) ...")
        tess_pairs = self._resolve_tess(toi_records, fits_index, self.cache_only)

        # ── Step 5-6: K2 ──────────────────────────────────────────────────────
        _log(f"\n[Step 5/6]  Downloading K2 label table ...")
        k2_records = download_k2_table()

        _log(f"\n[Step 6/6]  Resolving K2 FITS files  ({len(k2_records)} targets) ...")
        k2_pairs = self._resolve_k2(k2_records, fits_index, self.cache_only)

        # ── Merge ─────────────────────────────────────────────────────────────
        all_pairs: list[tuple[Path, float]] = kepler_pairs + tess_pairs + k2_pairs

        _log(f"\n  All missions resolved:")
        _log(f"    Kepler : {len(kepler_pairs):>5} files")
        _log(f"    TESS   : {len(tess_pairs):>5} files")
        _log(f"    K2     : {len(k2_pairs):>5} files")
        _log(f"    Total  : {len(all_pairs):>5} files")

        if max_samples is not None and max_samples < len(all_pairs):
            _log(f"  Capping at max_samples={max_samples}")
            all_pairs = all_pairs[:max_samples]

        # ── Preprocess each FITS file ─────────────────────────────────────────
        self._items: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._labels: list[float] = []

        total       = len(all_pairs)
        n_valid     = 0
        n_skipped   = 0
        report_every = 1
        t0_pre      = time.time()

        _log(f"\n{'=' * 65}")
        _log(f"  Preprocessing {total} light curves  (BLS + fold + bin) ...")
        _log(f"{'=' * 65}")

        for idx, (fits_path, label) in enumerate(all_pairs):
            try:
                candidates = preprocess(fits_path, n_candidates=3)
            except Exception:  # noqa: BLE001
                n_skipped += 1
                candidates = []

            if not candidates:
                n_skipped += 1 if not candidates else 0  # already counted above if exception
            else:
                n_valid += 1

            for c in candidates:
                scalar = np.array(
                    [c.period, c.duration, c.depth, c.bls_power,
                     c.secondary_depth, c.odd_even_diff],
                    dtype=np.float32,
                )
                self._items.append((
                    c.global_view.astype(np.float32),
                    c.local_view.astype(np.float32),
                    scalar,
                    label,
                ))
                self._labels.append(label)

            done = idx + 1
            if done % report_every == 0 or done == total:
                elapsed = time.time() - t0_pre
                n_pos_so_far = sum(1 for l in self._labels if l == 1.0)
                n_neg_so_far = len(self._labels) - n_pos_so_far
                _log(
                    f"  Preprocess  {_pbar(done, total)}  "
                    f"samples={len(self._items)}  "
                    f"(pos={n_pos_so_far} neg={n_neg_so_far})  "
                    f"skipped={n_skipped}  "
                    f"elapsed={_fmt_elapsed(elapsed)}  "
                    f"eta={_eta(elapsed, done, total)}"
                )

        n_pos = sum(1 for l in self._labels if l == 1.0)
        n_neg = sum(1 for l in self._labels if l == 0.0)
        total_time = _fmt_elapsed(time.time() - t0_pre)
        _log(f"\n  Preprocessing complete in {total_time}.")
        _log(f"  Dataset: {len(self._items)} samples  |  {n_pos} positives  {n_neg} negatives")

        # Save preprocessing cache so the next run can skip the BLS step.
        if cache_file is not None and self._items:
            cache_file = Path(cache_file)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            _log(f"\n  Saving preprocessing cache: {cache_file} ...")
            gvs     = np.stack([it[0] for it in self._items])  # (N, 2001)
            lvs     = np.stack([it[1] for it in self._items])  # (N, 201)
            scalars = np.stack([it[2] for it in self._items])  # (N, 6)
            labels  = np.array(self._labels, dtype=np.float32) # (N,)
            np.savez_compressed(cache_file,
                                global_views=gvs, local_views=lvs,
                                scalars=scalars, labels=labels)
            _log(f"  Cache saved ({cache_file.stat().st_size // 1024:,} KB).")

    # ── Mission-specific resolution helpers ───────────────────────────────────

    def _resolve_kepler(
        self,
        records: list[tuple[int, float]],
        fits_index: list[tuple[str, Path]],
        cache_only: bool = False,
    ) -> list[tuple[Path, float]]:
        """Locate or download FITS for each Kepler KOI; return (path, label) pairs."""
        total   = len(records)
        pairs: list[tuple[Path, float]] = []
        n_cached = n_downloaded = n_missing = 0
        t0 = time.time()
        report_every = max(1, total // 20)  # ~5% increments

        _log(f"\n  Resolving Kepler FITS  (0/{total})  ...")
        for i, (kepid, label) in enumerate(records, start=1):
            padded = str(kepid).zfill(9)
            cached = _fits_cache_lookup(fits_index, padded)
            if cached is not None:
                pairs.append((cached, label))
                n_cached += 1
            elif not cache_only:
                try:
                    path = _download_fits_kepler(kepid, self.fits_dir)
                except Exception:  # noqa: BLE001
                    path = None
                if path is not None:
                    pairs.append((path, label))
                    n_downloaded += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

            if i % report_every == 0 or i == total:
                elapsed = time.time() - t0
                _log(f"  Kepler  {_pbar(i, total)}  "
                     f"cached={n_cached}  downloaded={n_downloaded}  missing={n_missing}  "
                     f"elapsed={_fmt_elapsed(elapsed)}  eta={_eta(elapsed, i, total)}")

        _log(f"  Kepler resolved: {len(pairs)}/{total} files found  "
             f"({n_cached} cached, {n_downloaded} downloaded, {n_missing} missing)")
        return pairs

    def _resolve_tess(
        self,
        records: list[tuple[int, float]],
        fits_index: list[tuple[str, Path]],
        cache_only: bool = False,
    ) -> list[tuple[Path, float]]:
        """Locate or download FITS for each TESS TOI; return (path, label) pairs."""
        total   = len(records)
        pairs: list[tuple[Path, float]] = []
        n_cached = n_downloaded = n_missing = 0
        t0 = time.time()
        report_every = max(1, total // 20)

        _log(f"\n  Resolving TESS FITS  (0/{total})  ...")
        for i, (tid, label) in enumerate(records, start=1):
            cached = _fits_cache_lookup(fits_index, f"tic{tid}")
            if cached is None:
                cached = _fits_cache_lookup(fits_index, str(tid))
            if cached is not None:
                pairs.append((cached, label))
                n_cached += 1
            elif not cache_only:
                try:
                    path = _download_fits_tess(tid, self.fits_dir)
                except Exception:  # noqa: BLE001
                    path = None
                if path is not None:
                    pairs.append((path, label))
                    n_downloaded += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

            if i % report_every == 0 or i == total:
                elapsed = time.time() - t0
                _log(f"  TESS    {_pbar(i, total)}  "
                     f"cached={n_cached}  downloaded={n_downloaded}  missing={n_missing}  "
                     f"elapsed={_fmt_elapsed(elapsed)}  eta={_eta(elapsed, i, total)}")

        _log(f"  TESS resolved: {len(pairs)}/{total} files found  "
             f"({n_cached} cached, {n_downloaded} downloaded, {n_missing} missing)")
        return pairs

    def _resolve_k2(
        self,
        records: list[tuple[int, float]],
        fits_index: list[tuple[str, Path]],
        cache_only: bool = False,
    ) -> list[tuple[Path, float]]:
        """Locate or download FITS for each K2 candidate; return (path, label) pairs."""
        total   = len(records)
        pairs: list[tuple[Path, float]] = []
        n_cached = n_downloaded = n_missing = 0
        t0 = time.time()
        report_every = max(1, total // 20)

        _log(f"\n  Resolving K2 FITS  (0/{total})  ...")
        for i, (epic_id, label) in enumerate(records, start=1):
            cached = _fits_cache_lookup(fits_index, f"epic{epic_id}")
            if cached is None:
                cached = _fits_cache_lookup(fits_index, str(epic_id))
            if cached is not None:
                pairs.append((cached, label))
                n_cached += 1
            elif not cache_only:
                try:
                    path = _download_fits_k2(epic_id, self.fits_dir)
                except Exception:  # noqa: BLE001
                    path = None
                if path is not None:
                    pairs.append((path, label))
                    n_downloaded += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

            if i % report_every == 0 or i == total:
                elapsed = time.time() - t0
                _log(f"  K2      {_pbar(i, total)}  "
                     f"cached={n_cached}  downloaded={n_downloaded}  missing={n_missing}  "
                     f"elapsed={_fmt_elapsed(elapsed)}  eta={_eta(elapsed, i, total)}")

        _log(f"  K2 resolved: {len(pairs)}/{total} files found  "
             f"({n_cached} cached, {n_downloaded} downloaded, {n_missing} missing)")
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
        scalar_tensor  : float32 tensor of shape (6,)  — [period, duration, depth, bls_power, secondary_depth, odd_even_diff]
        label          : float32 tensor of shape (1,)
        """
        gv, lv, scalar, label = self._items[idx]

        if self.augment:
            gv = gv.copy()
            lv = lv.copy()
            # 1. Gaussian noise
            gv += np.random.normal(0, 0.002, gv.shape).astype(np.float32)
            lv += np.random.normal(0, 0.002, lv.shape).astype(np.float32)
            # 2. Random phase shift on local view (±10% of length)
            max_shift = max(1, int(0.1 * len(lv)))
            shift = np.random.randint(-max_shift, max_shift + 1)
            lv = np.roll(lv, shift)
            # 3. Random flux scaling
            scale = np.random.uniform(0.98, 1.02)
            gv = (gv * scale).astype(np.float32)
            lv = (lv * scale).astype(np.float32)

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


# ── Augmented subset wrapper ──────────────────────────────────────────────────

class _AugSubset(Dataset):
    """
    Wraps a MultiMissionDataset + index list and forces augment=True per item.

    This lets the same underlying dataset be used for training (augmented) and
    validation (unaugmented) without preprocessing data twice.
    """

    def __init__(self, base: "MultiMissionDataset", indices: list[int]) -> None:
        self._base    = base
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        prev = self._base.augment
        self._base.augment = True
        item = self._base[self._indices[i]]
        self._base.augment = prev
        return item


# ── Training loop ─────────────────────────────────────────────────────────────

def _make_sampler(labels: list[float]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws each class with equal probability.

    This ensures every mini-batch is ~50/50 pos/neg regardless of the true
    class ratio, which is more effective than loss-weighting alone.
    """
    n_pos = sum(1 for l in labels if l == 1.0)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / max(n_pos, 1)
    w_neg = 1.0 / max(n_neg, 1)
    weights = [w_pos if l == 1.0 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _label_smooth(targets: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:
    """Apply label smoothing: push labels away from hard 0/1 by *smoothing*/2."""
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _run_epoch(
    model: ExoNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    label_smoothing: float = 0.05,
    max_grad_norm: float = 1.0,
) -> tuple[float, list[float], list[float]]:
    """
    Run one epoch (train or eval).

    Parameters
    ----------
    optimizer :
        If ``None`` the model is run in eval mode (validation pass).
    label_smoothing :
        Soft-labels strength (applied during training only).
    max_grad_norm :
        Gradient clipping max norm (applied during training only).

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

            targets = _label_smooth(labels, label_smoothing) if is_train else labels
            loss = criterion(scores, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(scores).detach().cpu().numpy()[:, 0]
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy()[:, 0].tolist())

    mean_loss = total_loss / max(len(all_labels), 1)
    return mean_loss, all_scores, all_labels


# ── Threshold sweep ───────────────────────────────────────────────────────────

def _tune_threshold(
    val_scores: list[float],
    val_labels: list[float],
    output_dir: Path,
    min_precision: float = 0.90,
) -> float:
    """
    Sweep thresholds from 0.1 to 0.99 in steps of 0.01 and record:
      - best_f1_threshold   : maximises F1
      - precision90_threshold : lowest threshold that keeps precision >= min_precision

    Both are saved to threshold.json.  Returns the max-F1 threshold.
    """
    from sklearn.metrics import precision_score, recall_score  # noqa: PLC0415

    scores_arr = np.array(val_scores)
    labels_arr = np.array(val_labels)

    best_thresh  = 0.5
    best_f1      = -1.0
    prec90_thresh: float | None = None

    thresholds = np.arange(0.10, 1.00, 0.01)
    for thresh in thresholds:
        preds = (scores_arr >= thresh).astype(int)
        f1    = f1_score(labels_arr, preds, zero_division=0)
        prec  = precision_score(labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = float(thresh)
        if prec >= min_precision and prec90_thresh is None:
            prec90_thresh = float(thresh)

    _log(f"Optimal F1 threshold  : {best_thresh:.2f}  (val F1: {best_f1:.4f})")
    if prec90_thresh is not None:
        preds90 = (scores_arr >= prec90_thresh).astype(int)
        recall90 = recall_score(labels_arr, preds90, zero_division=0)
        _log(f"Precision≥{min_precision:.0%} threshold: {prec90_thresh:.2f}  "
             f"(recall at that threshold: {recall90:.4f})")
    else:
        _log(f"Precision≥{min_precision:.0%} threshold: not achievable on this val set")

    threshold_path = output_dir / "threshold.json"
    payload: dict = {
        "threshold":           round(best_thresh, 2),
        "threshold_max_f1":    round(best_thresh, 2),
    }
    if prec90_thresh is not None:
        payload[f"threshold_precision{int(min_precision*100)}"] = round(prec90_thresh, 2)
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    _log(f"Threshold saved: {threshold_path}")

    return best_thresh


# ── Full training pipeline ────────────────────────────────────────────────────

def _train_fold(
    dataset: MultiMissionDataset,
    train_idx: list[int],
    val_idx: list[int],
    args: argparse.Namespace,
    device: torch.device,
    pt_save_path: Path,
    fold_num: int = 0,
) -> tuple[float, list[float], list[float]]:
    """
    Train one fold and return (best_val_auc, best_val_scores, best_val_labels).

    The best checkpoint is saved to *pt_save_path*.  When ``args.save_all_folds``
    is True, a per-fold copy is also saved alongside it as
    ``exonet_fold_{fold_num}.pt``.
    """
    train_labels = [dataset.labels[i] for i in train_idx]
    sampler = _make_sampler(train_labels)

    # Augmentation only on the training split; val uses the base dataset (augment=False).
    train_loader = DataLoader(
        _AugSubset(dataset, train_idx),
        batch_size=args.batch_size,
        sampler=sampler,          # balanced batches via WeightedRandomSampler
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Weighted loss: upweight the minority (positive) class per-fold.
    n_pos = max(sum(1 for l in train_labels if l == 1.0), 1)
    n_neg = max(len(train_labels) - n_pos, 1)
    pw = n_neg / n_pos
    use_se       = getattr(args, "use_se", False)
    dropout      = getattr(args, "dropout", 0.4)
    weight_decay = getattr(args, "weight_decay", 1e-4)
    lr_schedule  = getattr(args, "lr_schedule", "plateau")
    cosine_t0    = getattr(args, "cosine_t0", 10)

    model = ExoNet(use_se=use_se).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_t0, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )

    best_val_auc = -1.0
    best_val_scores: list[float] = []
    best_val_labels: list[float] = []
    epochs_no_improve = 0

    _log(f"  pos_weight={pw:.2f}  (n_neg={n_neg}, n_pos={n_pos})")
    _log(f"\n  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>9}  {'LR':>10}  {'Elapsed':>9}  {'ETA':>9}")
    _log("  " + "-" * 72)
    fold_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_scores, val_labels_ep = _run_epoch(model, val_loader, criterion, None, device)

        try:
            val_auc = roc_auc_score(val_labels_ep, val_scores)
        except ValueError:
            val_auc = float("nan")

        if lr_schedule != "cosine":
            scheduler.step(val_auc if not (val_auc != val_auc) else 0.0)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.time() - fold_t0
        epoch_eta     = _eta(epoch_elapsed, epoch, args.epochs)

        _log(
            f"  {epoch:>5}  {train_loss:>11.5f}  {val_loss:>9.5f}  "
            f"{val_auc:>9.4f}  {current_lr:>10.2e}  "
            f"{_fmt_elapsed(epoch_elapsed):>9}  {epoch_eta:>9}"
        )

        if lr_schedule == "cosine":
            scheduler.step(epoch - 1)  # CosineAnnealingWarmRestarts expects step per epoch
        else:
            scheduler.step(val_auc if not (val_auc != val_auc) else 0.0)

        improved = not (val_auc != val_auc) and val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_val_scores = list(val_scores)
            best_val_labels = list(val_labels_ep)
            epochs_no_improve = 0
            torch.save(model.state_dict(), pt_save_path)
            # Per-fold checkpoint alongside the overall best.
            if fold_num > 0 and getattr(args, "save_all_folds", False):
                fold_path = pt_save_path.parent / f"exonet_fold_{fold_num}.pt"
                torch.save(model.state_dict(), fold_path)
            _log(f"         >> New best val AUC {best_val_auc:.4f} -- checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                _log(f"\n  Early stopping after {epoch} epochs (patience={args.patience}).")
                break

    # Always write the per-fold checkpoint at the best weights found for this fold.
    if fold_num > 0 and getattr(args, "save_all_folds", False):
        fold_path = pt_save_path.parent / f"exonet_fold_{fold_num}.pt"
        if not fold_path.exists():
            # If never improved (degenerate fold), save what we have.
            torch.save(model.state_dict(), fold_path)
        _log(f"  Fold {fold_num} checkpoint: {fold_path.name}")

    return best_val_auc, best_val_scores, best_val_labels


def train(args: argparse.Namespace) -> None:
    """Full training pipeline with k-fold cross-validation."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build dataset ─────────────────────────────────────────────────────────
    # Build the base dataset without augmentation; _AugSubset applies it to
    # training folds only, keeping validation clean.
    dataset = MultiMissionDataset(
        fits_dir=args.fits_dir,
        csv_path=args.csv_path,
        max_samples=args.max_samples,
        augment=False,
        cache_only=args.cache_only,
        cache_file=getattr(args, "cache_file", None),
    )

    if len(dataset) < 10:
        _log("ERROR: dataset contains fewer than 10 usable samples. Aborting.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"\nTraining on {device}.")

    indices = np.array(range(len(dataset)))
    labels  = np.array(dataset.labels)
    pt_save_path   = output_dir / "exonet.pt"
    onnx_save_path = output_dir / "exonet.onnx"

    # ── K-fold cross-validation ───────────────────────────────────────────────
    n_splits = min(args.folds, int(min(np.sum(labels == 0), np.sum(labels == 1))))
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs: list[float] = []

    overall_best_auc   = -1.0
    best_val_scores_all: list[float] = []
    best_val_labels_all: list[float] = []

    _log(f"\n  Starting {n_splits}-fold cross-validation  "
         f"({len(dataset)} samples total, device={device})")

    cv_t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        _log(f"\n{'='*65}")
        _log(f"  Fold {fold}/{n_splits}  —  train={len(train_idx)}  val={len(val_idx)}  "
             f"(cv elapsed so far: {_fmt_elapsed(time.time() - cv_t0)})")
        _log(f"{'='*65}")

        fold_auc, val_scores, val_labels_ep = _train_fold(
            dataset,
            train_idx.tolist(),
            val_idx.tolist(),
            args,
            device,
            pt_save_path,
            fold_num=fold,
        )
        fold_aucs.append(fold_auc)

        if fold_auc > overall_best_auc:
            overall_best_auc = fold_auc
            best_val_scores_all = val_scores
            best_val_labels_all = val_labels_ep

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    _log(f"\n{'='*65}")
    _log(f"  Cross-validation complete in {_fmt_elapsed(time.time() - cv_t0)}.")
    _log(f"  Fold AUCs : {[f'{a:.4f}' for a in fold_aucs]}")
    _log(f"  Mean AUC  : {mean_auc:.4f} ± {std_auc:.4f}")
    _log(f"  Best fold : {overall_best_auc:.4f}")
    _log(f"  Checkpoint: {pt_save_path}")
    _log(f"{'='*65}")
    # Machine-parseable summary line for the orchestrator.
    _log(f"RESULT: mean_auc={mean_auc:.4f} std_auc={std_auc:.4f} best_auc={overall_best_auc:.4f}")

    # ── Threshold tuning on best fold's val set ───────────────────────────────
    if best_val_scores_all and best_val_labels_all:
        _log("\nRunning threshold sweep on best fold's validation set ...")
        _tune_threshold(best_val_scores_all, best_val_labels_all, output_dir)

    # ── ONNX export ───────────────────────────────────────────────────────────
    _log("\nExporting best checkpoint to ONNX ...")
    ExoNetInference.export_from_pytorch(pt_save_path, onnx_save_path)
    _log(f"ONNX model: {onnx_save_path}")


# ── New CLI flags ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ExoNet on multi-mission data (Kepler/TESS/K2) and export to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--folds",       type=int,   default=5,
                        help="Number of stratified k-folds for cross-validation.")
    parser.add_argument("--fits-dir",    type=Path,  required=True)
    parser.add_argument("--output-dir",  type=Path,  required=True)
    parser.add_argument("--val-split",   type=float, default=0.15,
                        help="(Unused when --folds > 1; kept for compatibility.)")
    parser.add_argument("--max-samples", type=int,   default=None)
    parser.add_argument("--csv-path",    type=Path,  default=None)
    parser.add_argument("--num-workers",   type=int,   default=0)
    parser.add_argument("--no-augment",    action="store_true")
    parser.add_argument("--cache-only",    action="store_true",
                        help="Skip MAST downloads; only use FITS files already on disk.")
    parser.add_argument("--cache-file",    type=Path,  default=None,
                        help="Path to a .npz preprocessing cache file. "
                             "Saves on first run, loads on subsequent runs to skip BLS preprocessing.")
    parser.add_argument("--lr-schedule",   type=str,   default="plateau",
                        choices=["plateau", "cosine"],
                        help="LR scheduler: 'plateau' (ReduceLROnPlateau) or 'cosine' (CosineAnnealingWarmRestarts).")
    parser.add_argument("--cosine-t0",     type=int,   default=10,
                        help="T_0 period (epochs) for CosineAnnealingWarmRestarts.")
    parser.add_argument("--use-se",        action="store_true",
                        help="Enable Squeeze-and-Excite channel attention in GlobalBranch and LocalBranch.")
    parser.add_argument("--dropout",       type=float, default=0.4,
                        help="Dropout rate for CNN branch heads (0–1).")
    parser.add_argument("--weight-decay",  type=float, default=1e-4,
                        help="AdamW weight decay.")
    parser.add_argument("--save-all-folds", action="store_true",
                        help="Save a per-fold checkpoint exonet_fold_k.pt alongside the overall best.")
    return parser.parse_args(argv)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
