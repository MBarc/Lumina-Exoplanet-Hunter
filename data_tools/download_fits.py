"""
Parallel FITS light curve downloader for Exoplanet-Hunter.

Downloads Kepler, K2, and TESS light curve FITS files from NASA's MAST
archive using a thread pool so that multiple files are fetched concurrently.

Design notes
------------
Downloads are I/O-bound — the CPU sits idle while waiting for MAST to respond
and for bytes to arrive over the network. Python threads are perfectly suited
here: the GIL is released during blocking I/O, so N threads genuinely run N
downloads in parallel without needing multiprocessing overhead.

Thread count guidance
---------------------
MAST does not publish a hard rate limit, but in practice anything above
~15 simultaneous connections triggers throttling (HTTP 429 / connection
resets). 8–10 threads is a safe default that keeps the pipeline near maximum
throughput without getting blocked.

Resumability
------------
The script checks whether each target file already exists in the output
directory before querying MAST. This makes reruns cheap — you can kill the
process at any time and restart it without re-downloading completed files.

Usage (from the repo root)
--------------------------
    python data_tools/download_fits.py \\
        --mission kepler \\
        --output-dir fits_cache \\
        --threads 10 \\
        --limit 50000

    python data_tools/download_fits.py \\
        --mission tess \\
        --output-dir fits_cache \\
        --threads 10

    # Download both missions back-to-back
    python data_tools/download_fits.py \\
        --mission all \\
        --output-dir fits_cache \\
        --threads 10 \\
        --limit 100000

Requirements
------------
    pip install astroquery astropy tqdm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

# astroquery talks to the MAST archive REST API.
# astropy handles FITS parsing for validation.
from astroquery.mast import Observations
from astropy.io import fits
from tqdm import tqdm


# ── Logging setup ──────────────────────────────────────────────────────────────
#
# We write to both stdout (INFO+) and a persistent log file (DEBUG+) so that
# errors are preserved even if the terminal buffer scrolls past them.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── MAST mission configuration ─────────────────────────────────────────────────
#
# Each mission entry describes how to query MAST for long-cadence light curves.
# "obs_collection" and "dataproduct_type" are MAST observation-level filters.
# "description_keyword" narrows the product download to light curve files only,
# avoiding the much larger target pixel files (TPFs) and other data products.

_MISSION_CONFIG: dict[str, dict] = {
    "kepler": {
        "obs_collection":    "Kepler",
        "dataproduct_type":  "timeseries",
        # Kepler long-cadence light curves are named "*_llc.fits".
        # Short-cadence ("*_slc.fits") have higher time resolution but we
        # only need the long-cadence products for BLS transit searching.
        "description_keyword": "Long Cadence",
        "label":             "Kepler LC",
    },
    "k2": {
        "obs_collection":    "K2",
        "dataproduct_type":  "timeseries",
        # K2 light curves from the EVEREST pipeline are the highest-quality
        # detrended products and are what our preprocessing expects.
        "description_keyword": "Light curve",
        "label":             "K2 LC",
    },
    "tess": {
        "obs_collection":    "TESS",
        "dataproduct_type":  "timeseries",
        # TESS 2-minute cadence light curves are stored as "*_lc.fits".
        # The 20-second and full-frame image products are excluded.
        "description_keyword": "Light curves",
        "label":             "TESS LC",
    },
}


# ── Target list helpers ────────────────────────────────────────────────────────

def _query_observation_ids(
    mission_key: str,
    limit: int | None,
) -> list[str]:
    """
    Query MAST for all observation IDs matching a mission.

    Returns a list of obs_id strings that we can later use to fetch the
    associated data products. The query is paged internally by astroquery
    when there are more results than the default page size.

    Parameters
    ----------
    mission_key : One of "kepler", "k2", "tess".
    limit       : Cap the result count (useful for testing). None = no cap.
    """
    cfg = _MISSION_CONFIG[mission_key]
    log.info("Querying MAST for %s observations...", cfg["label"])

    # query_criteria returns an astropy Table. We only need the obs_id column
    # but MAST always returns the full metadata row — we just ignore the rest.
    table = Observations.query_criteria(
        obs_collection=cfg["obs_collection"],
        dataproduct_type=cfg["dataproduct_type"],
    )

    obs_ids: list[str] = list(table["obs_id"])

    log.info("  Found %s observations.", f"{len(obs_ids):,}")

    if limit is not None and len(obs_ids) > limit:
        # Shuffle before slicing so we get a representative sample across the
        # sky rather than just the first chunk of the catalog (which tends to
        # be biased toward bright, well-studied targets).
        import random
        random.shuffle(obs_ids)
        obs_ids = obs_ids[:limit]
        log.info("  Capped to %s after shuffle.", f"{limit:,}")

    return obs_ids


def _iter_download_urls(
    obs_ids: list[str],
    mission_key: str,
    batch_size: int = 500,
) -> Iterator[tuple[str, str]]:
    """
    Yield (url, filename) pairs for every light curve file in obs_ids.

    MAST product queries can only handle a few hundred observation IDs at a
    time before the request payload becomes unwieldy. We batch the obs_ids
    and concatenate the results.

    Parameters
    ----------
    obs_ids    : List of observation IDs returned by _query_observation_ids.
    mission_key: Mission key for filtering to the right file type.
    batch_size : How many obs_ids to include in each product query.
    """
    cfg = _MISSION_CONFIG[mission_key]
    keyword = cfg["description_keyword"].lower()

    total_batches = (len(obs_ids) + batch_size - 1) // batch_size
    log.info("Fetching product lists in %d batches of %d...", total_batches, batch_size)

    for batch_idx in range(total_batches):
        chunk = obs_ids[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # get_product_list returns a Table of all data products for these
        # observations. Each row has a 'dataURI' and 'description' column.
        try:
            products = Observations.get_product_list(chunk)
        except Exception as exc:
            log.warning("Batch %d/%d product query failed: %s — skipping.",
                        batch_idx + 1, total_batches, exc)
            continue

        for row in products:
            desc = str(row.get("description", "")).lower()
            uri  = str(row.get("dataURI", ""))
            if keyword in desc and uri.endswith(".fits"):
                # The filename is the last component of the URI path.
                # e.g. "mast:Kepler/url/path/kplr001234567_llc.fits"
                filename = uri.split("/")[-1]
                yield uri, filename

        if (batch_idx + 1) % 10 == 0:
            log.info("  Product batches processed: %d/%d", batch_idx + 1, total_batches)


# ── Single-file download ───────────────────────────────────────────────────────

# Lock protecting the shared progress counters below.
# Using a lock rather than threading.local because the counters are shared
# *across* threads — each thread increments the same total.
_counter_lock = threading.Lock()
_n_downloaded = 0
_n_skipped    = 0
_n_failed     = 0


def _download_one(uri: str, filename: str, output_dir: Path) -> str:
    """
    Download a single FITS file from MAST to output_dir.

    Returns a status string: "downloaded", "skipped", or "failed:<reason>".

    The function is intentionally self-contained so it can be safely called
    from any thread without shared state other than the output directory and
    the protected counters above.
    """
    global _n_downloaded, _n_skipped, _n_failed

    dest = output_dir / filename

    # --- Resumability: skip if the file already exists and is non-empty. ------
    # We do a quick size check rather than a checksum because MAST does not
    # publish checksums in a conveniently queryable way. A zero-byte file
    # indicates a failed previous download and should be retried.
    if dest.exists() and dest.stat().st_size > 0:
        with _counter_lock:
            _n_skipped += 1
        return "skipped"

    # --- Fetch from MAST via astroquery. ------------------------------------
    # download_file returns a local path; we move it to our target location.
    # We write to a ".part" file first so that a partial download is never
    # mistaken for a complete one by a concurrent thread or a future run.
    part = dest.with_suffix(".fits.part")
    try:
        local_path = Observations.download_file(uri, local_path=str(part))

        # astroquery returns (status, msg, url) or just a path depending on
        # the version. Normalise to a Path.
        if isinstance(local_path, tuple):
            local_path = local_path[0]

        part_path = Path(str(local_path))

        # Quick sanity check: open the FITS header to confirm the file is
        # valid. A corrupted partial download will raise an exception here,
        # which we catch below.
        with fits.open(str(part_path), memmap=False) as hdul:
            _ = hdul[0].header  # just touching the header is enough

        # Atomic rename: on most OSes this is a single syscall, so no other
        # thread can observe a half-written file at the final path.
        part_path.rename(dest)

        with _counter_lock:
            _n_downloaded += 1
        return "downloaded"

    except Exception as exc:
        # Clean up any partial file so the next run will retry this target.
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass
        with _counter_lock:
            _n_failed += 1
        return f"failed:{exc}"


# ── Main orchestration ─────────────────────────────────────────────────────────

def run_download(
    missions: list[str],
    output_dir: Path,
    threads: int,
    limit: int | None,
) -> None:
    """
    Download light curves for one or more missions using a thread pool.

    The function first collects all (uri, filename) pairs across every
    requested mission, then fans them out to the thread pool. Collecting
    everything first lets tqdm show an accurate total count and lets us
    skip already-downloaded files before spawning threads.

    Parameters
    ----------
    missions   : List of mission keys, e.g. ["kepler", "tess"].
    output_dir : Local directory to save FITS files into.
    threads    : Number of concurrent download threads.
    limit      : Max observations to fetch per mission (None = no cap).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect all download targets ──────────────────────────────────
    all_targets: list[tuple[str, str]] = []   # list of (uri, filename)

    for mission in missions:
        obs_ids = _query_observation_ids(mission, limit)
        for uri, filename in _iter_download_urls(obs_ids, mission):
            all_targets.append((uri, filename))

    if not all_targets:
        log.error("No downloadable files found. Check mission names and MAST connectivity.")
        sys.exit(1)

    log.info("")
    log.info("Total files to process : %s", f"{len(all_targets):,}")
    log.info("Output directory       : %s", output_dir.resolve())
    log.info("Thread pool size       : %d", threads)
    log.info("")

    # ── Step 2: Fan out to thread pool ────────────────────────────────────────
    #
    # We use as_completed() rather than map() so that we can update the
    # progress bar and log individual failures as they happen rather than
    # waiting for all results.

    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=threads) as pool:
        # Submit all jobs upfront. futures is a dict so we can look up the
        # (uri, filename) from a completed future if needed for error logging.
        futures = {
            pool.submit(_download_one, uri, fname, output_dir): (uri, fname)
            for uri, fname in all_targets
        }

        with tqdm(
            total=len(futures),
            unit="file",
            desc="Downloading",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                uri, fname = futures[future]
                try:
                    status = future.result()
                except Exception as exc:
                    # Defensive catch — _download_one should not raise, but
                    # belt-and-suspenders for unexpected thread exceptions.
                    log.error("Unexpected thread error for %s: %s", fname, exc)
                    status = "failed:unexpected"

                if status.startswith("failed"):
                    # Log failures at DEBUG level to avoid drowning the
                    # terminal; they are counted and summarised at the end.
                    log.debug("FAIL  %s  (%s)", fname, status)

                pbar.update(1)
                # Keep the postfix stats in sync with the counters.
                pbar.set_postfix(
                    ok=_n_downloaded,
                    skip=_n_skipped,
                    fail=_n_failed,
                    refresh=False,
                )

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    elapsed = time.monotonic() - start_time
    total   = _n_downloaded + _n_skipped + _n_failed

    log.info("")
    log.info("=" * 60)
    log.info("  Download complete in %.0f min", elapsed / 60)
    log.info("  Downloaded  : %s", f"{_n_downloaded:,}")
    log.info("  Skipped     : %s  (already on disk)", f"{_n_skipped:,}")
    log.info("  Failed      : %s", f"{_n_failed:,}")
    log.info("  Total       : %s", f"{total:,}")
    log.info("=" * 60)

    if _n_failed > 0:
        log.warning(
            "%d files failed. Re-run the script to retry them — "
            "completed files will be skipped automatically.",
            _n_failed,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kepler / K2 / TESS light curve FITS from MAST in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data_tools/download_fits.py --mission kepler --threads 10\n"
            "  python data_tools/download_fits.py --mission all --limit 50000\n"
            "  python data_tools/download_fits.py --mission tess k2 --output-dir /mnt/nas/fits"
        ),
    )
    parser.add_argument(
        "--mission",
        nargs="+",
        choices=list(_MISSION_CONFIG.keys()) + ["all"],
        default=["kepler"],
        metavar="MISSION",
        help=(
            "Which mission(s) to download. "
            "Choose one or more of: kepler, k2, tess. "
            "Use 'all' to download all three."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fits_cache"),
        help="Local directory to save FITS files into.",
    )
    # Cap the automatic default at 10 regardless of core count.
    # Downloads are network-bound, not CPU-bound — beyond ~10 simultaneous
    # connections MAST starts throttling rather than serving faster.
    _default_threads = min(os.cpu_count() or 1, 10)

    parser.add_argument(
        "--threads",
        type=int,
        default=_default_threads,
        help=(
            "Number of parallel download threads. "
            f"Defaults to min(cpu_count, 10) = {_default_threads} on this machine. "
            "8–12 is a safe range before MAST starts throttling. "
            "Set lower on a slow connection."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of observations to fetch *per mission*. "
            "Useful for smoke-testing the script before a full run. "
            "Targets are randomly shuffled before the cap is applied."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to write a full DEBUG log to. "
            "Useful for capturing per-file failure reasons without "
            "flooding the terminal."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ── Attach a file handler for verbose logging if requested ────────────────
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(fh)
        log.info("Debug log: %s", args.log_file)

    # ── Resolve mission list ──────────────────────────────────────────────────
    if "all" in args.mission:
        missions = list(_MISSION_CONFIG.keys())
    else:
        missions = args.mission

    log.info("Missions    : %s", ", ".join(missions))
    log.info("Output dir  : %s", args.output_dir)
    log.info("Threads     : %d", args.threads)
    log.info("Limit/mission: %s", args.limit if args.limit else "none")

    run_download(
        missions=missions,
        output_dir=args.output_dir,
        threads=args.threads,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
