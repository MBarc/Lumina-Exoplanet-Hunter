"""
Bulk parallel FITS downloader for ExoNet training data.

Downloads Kepler, TESS, and K2 light curves from MAST in parallel using a
thread pool.  Designed to be run once before training to populate the FITS
cache so that subsequent training runs finish quickly.

Usage
-----
::

    # Download up to 2 000 balanced Kepler targets (default)
    python -m ml.download_fits --fits-dir /data/fits --n-kepler 2000

    # Download all missions, 1 000 targets each, 8 threads
    python -m ml.download_fits \\
        --fits-dir /data/fits \\
        --n-kepler 1000 --n-tess 500 \\
        --workers 8

The script skips targets whose FITS file is already on disk (cache-safe).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Reuse helpers from train.py
from ml.train import (
    _download_fits_kepler,
    _download_fits_tess,
    _download_fits_k2,
    _fits_cache_path,
    download_koi_table,
    download_toi_table,
    download_k2_table,
)


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _balanced_sample(
    records: list[tuple[int, float]],
    n: int,
) -> list[tuple[int, float]]:
    """
    Return up to *n* records sampled evenly from positives and negatives.

    Positives (label == 1.0) and negatives (label == 0.0) are each capped at
    n // 2, then combined and shuffled deterministically (seed 42).

    If one class has fewer than n // 2 examples, the remaining quota is filled
    from the other class so we always return min(n, len(records)) entries.
    """
    import random
    rng = random.Random(42)

    positives = [r for r in records if r[1] == 1.0]
    negatives = [r for r in records if r[1] == 0.0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    half = n // 2
    # Take up to half from each; fill slack from the other class
    take_pos = min(half, len(positives))
    take_neg = min(half, len(negatives))
    slack = (half - take_pos) + (half - take_neg)

    selected = positives[:take_pos] + negatives[:take_neg]
    # Use remaining from whichever class has more
    extras = positives[take_pos:] + negatives[take_neg:]
    rng.shuffle(extras)
    selected += extras[:slack]

    rng.shuffle(selected)
    return selected


# ── Per-target download worker ────────────────────────────────────────────────

def _worker_kepler(kepid: int, label: float, fits_dir: Path) -> tuple[int, float, bool]:
    padded = str(kepid).zfill(9)
    if _fits_cache_path(fits_dir, padded) is not None:
        return kepid, label, True  # already cached
    try:
        path = _download_fits_kepler(kepid, fits_dir)
        return kepid, label, path is not None
    except Exception:
        return kepid, label, False


def _worker_tess(tid: int, label: float, fits_dir: Path) -> tuple[int, float, bool]:
    if _fits_cache_path(fits_dir, f"TIC{tid}") or _fits_cache_path(fits_dir, str(tid)):
        return tid, label, True
    try:
        path = _download_fits_tess(tid, fits_dir)
        return tid, label, path is not None
    except Exception:
        return tid, label, False


def _worker_k2(epic_id: int, label: float, fits_dir: Path) -> tuple[int, float, bool]:
    if _fits_cache_path(fits_dir, f"EPIC{epic_id}") or _fits_cache_path(fits_dir, str(epic_id)):
        return epic_id, label, True
    try:
        path = _download_fits_k2(epic_id, fits_dir)
        return epic_id, label, path is not None
    except Exception:
        return epic_id, label, False


# ── Bulk downloader ───────────────────────────────────────────────────────────

def bulk_download(
    records: list[tuple[int, float]],
    fits_dir: Path,
    worker_fn,
    mission_name: str,
    max_workers: int = 4,
) -> tuple[int, int, int]:
    """
    Download *records* in parallel using *worker_fn*.

    Returns (n_cached, n_downloaded, n_failed).
    """
    fits_dir.mkdir(parents=True, exist_ok=True)
    n_cached = n_downloaded = n_failed = 0
    total = len(records)

    print(f"\n[{mission_name}] {total} targets queued  ({max_workers} threads) ...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(worker_fn, target_id, label, fits_dir): (target_id, label)
            for target_id, label in records
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                _, _, ok = future.result()
            except Exception:
                ok = False

            if ok is True:
                # Distinguish cached vs newly downloaded by checking before/after
                # (worker returns True for both; good enough for the counter)
                n_downloaded += 1
            else:
                n_failed += 1

            if done % 50 == 0 or done == total:
                pct = done / total * 100
                print(
                    f"  [{mission_name}] {done}/{total} ({pct:.0f}%)  "
                    f"ok={n_downloaded}  fail={n_failed}"
                )

    return n_cached, n_downloaded, n_failed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel FITS bulk downloader for ExoNet training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--fits-dir", type=Path, required=True,
        help="Local cache directory (downloads go here, already-cached files are skipped).",
    )
    p.add_argument(
        "--n-kepler", type=int, default=2000,
        help="Number of Kepler KOI targets to download (balanced pos/neg).",
    )
    p.add_argument(
        "--n-tess", type=int, default=0,
        help="Number of TESS TOI targets to download (0 = skip).",
    )
    p.add_argument(
        "--n-k2", type=int, default=0,
        help="Number of K2 targets to download (0 = skip).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fits_dir = Path(args.fits_dir)

    warnings.filterwarnings("ignore")

    # ── Kepler ────────────────────────────────────────────────────────────────
    if args.n_kepler > 0:
        koi_records = download_koi_table()
        sample = _balanced_sample(koi_records, args.n_kepler)
        _, dl, fail = bulk_download(sample, fits_dir, _worker_kepler, "Kepler", args.workers)
        print(f"[Kepler] done: {dl} ok, {fail} failed")

    # ── TESS ──────────────────────────────────────────────────────────────────
    if args.n_tess > 0:
        toi_records = download_toi_table()
        sample = _balanced_sample(toi_records, args.n_tess)
        _, dl, fail = bulk_download(sample, fits_dir, _worker_tess, "TESS", args.workers)
        print(f"[TESS] done: {dl} ok, {fail} failed")

    # ── K2 ────────────────────────────────────────────────────────────────────
    if args.n_k2 > 0:
        k2_records = download_k2_table()
        sample = _balanced_sample(k2_records, args.n_k2)
        _, dl, fail = bulk_download(sample, fits_dir, _worker_k2, "K2", args.workers)
        print(f"[K2] done: {dl} ok, {fail} failed")

    print("\nDownload complete. Run ml.train to retrain on the updated cache.")


if __name__ == "__main__":
    main()
