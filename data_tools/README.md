# data_tools

Standalone scripts for building and curating the Exoplanet-Hunter training dataset.
These are meant to be run manually — typically on a dedicated machine — and are not
imported by the dashboard or inference pipeline.

---

## download_fits.py

Downloads Kepler, K2, and TESS light curve FITS files from NASA's MAST archive
using a multi-threaded pool. Fully resumable — already-downloaded files are
skipped automatically on reruns.

### Requirements

```
pip install astroquery astropy tqdm
```

### Usage

```bash
# Download all Kepler light curves (up to MAST's full catalog)
python data_tools/download_fits.py --mission kepler --output-dir fits_cache --threads 10

# Download TESS and K2, capped at 50,000 observations each
python data_tools/download_fits.py --mission tess k2 --output-dir fits_cache --threads 10 --limit 50000

# Download all three missions, log failures to a file
python data_tools/download_fits.py --mission all --output-dir fits_cache --threads 10 --log-file download.log

# Smoke test — download 100 files to verify your setup
python data_tools/download_fits.py --mission kepler --output-dir fits_cache --threads 4 --limit 100
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--mission` | `kepler` | Mission(s) to download: `kepler`, `k2`, `tess`, or `all` |
| `--output-dir` | `fits_cache` | Local directory to save FITS files |
| `--threads` | `10` | Parallel download threads (8–12 is safe; higher risks MAST throttling) |
| `--limit` | none | Max observations per mission (useful for testing) |
| `--log-file` | none | Path to write a verbose DEBUG log |

### Thread count guidance

MAST does not publish a hard rate limit but throttles aggressively above ~15
simultaneous connections. **10 threads** is a reliable sweet spot. On a slow
connection (< 50 Mbps) reduce to 5–6 to avoid timeout errors.

### Resuming interrupted downloads

Simply re-run the same command. Files that already exist on disk are detected
and skipped instantly. Only files that failed or were never started are
retried.

---

## Planned tools

| Script | Status | Description |
|---|---|---|
| `download_fits.py` | ✅ Done | Parallel FITS downloader from MAST |
| `inject_transit.py` | Planned | Synthetic transit injection into clean light curves |
| `build_catalog.py` | Planned | Merge Kepler/K2/TESS disposition tables into a unified CSV |
