"""
Lumina node worker. Same code on Windows and Linux.

Loop: claim a batch of jobs from the coordinator API, download each light
curve from MAST, run BLS preprocessing and the ExoNet model, report every
score back, repeat. The node only ever talks to the API with its own device
token (written by the installer); it never holds database credentials.

Run:  python -m node.worker --config <path to config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import socket
import tempfile
import threading
import time
from pathlib import Path

MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"
HEARTBEAT_SECONDS = 30
IDLE_SECONDS = 60


class Api:
    def __init__(self, base_url: str, device_token: str):
        import requests
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-Device-Token"] = device_token

    def get(self, path: str, **params):
        r = self.session.get(self.base + path, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict | None = None, **params):
        r = self.session.post(self.base + path, json=body, params=params, timeout=60)
        r.raise_for_status()
        return r.json()


class Worker:
    def __init__(self, cfg: dict, stop: threading.Event):
        import requests
        from ml.inference import ExoNetInference  # after thread limits are set

        self.cfg = cfg
        self.stop = stop
        self.api = Api(cfg["api_url"], cfg["device_token"])
        self.mast = requests.Session()   # separate: never send the device token to MAST
        self.host = socket.gethostname()
        self.model = ExoNetInference(cfg["model_path"])
        # Fingerprint graph + external weights (newer exports split them), so
        # different weights never look like the same model.
        h = hashlib.sha256()
        for f in (Path(cfg["model_path"]), Path(cfg["model_path"] + ".data")):
            if f.is_file():
                h.update(f.read_bytes())
        self.model_sha256 = h.hexdigest()
        self.tmp = Path(cfg["data_dir"]) / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.stars = 0
        self.found = 0
        self.current: dict | None = None

    # ── one job ──────────────────────────────────────────────────────────────
    def _download(self, uri: str) -> Path:
        url = uri if uri.startswith("http") else f"{MAST_DOWNLOAD}?uri={uri}"
        fd, name = tempfile.mkstemp(suffix=".fits", dir=self.tmp)
        with os.fdopen(fd, "wb") as fh, self.mast.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        return Path(name)

    def process(self, job: dict) -> None:
        from ml.preprocess import preprocess

        t0 = time.monotonic()
        path = self._download(job["fits_url"])
        try:
            candidates = preprocess(path)
        finally:
            path.unlink(missing_ok=True)
        scores = self.model.predict_batch(candidates) if candidates else []

        keep_views = self.cfg["report_threshold"]
        for c, score in zip(candidates, scores):
            big = score >= keep_views   # light-curve arrays only for interesting ones
            view = lambda a: a.tolist() if big else []   # noqa: E731
            self.api.post("/candidates", {
                "worker_hostname": self.host,
                "tic_id": job["tic_id"], "mission": job["mission"], "sector": job.get("sector"),
                # Ephemeris: enough to re-fold the public light curve and redraw the transit.
                "period_days": float(c.period), "duration_days": float(c.duration),
                "t0": float(c.t0),   # mission clock; the server derives BJD
                "depth_ppm": float(c.depth_frac) * 1e6, "bls_power": float(c.bls_power),
                "exonet_score": float(score),
                "n_transits": float(c.n_transits),
                "snr": float(c.transit_snr),
                "secondary_depth": float(c.secondary_depth), "odd_even_diff": float(c.odd_even_diff),
                # Measured directly on the unclipped light curve (not rescaled model values).
                "secondary_depth_ppm": float(c.secondary_frac) * 1e6,
                "odd_even_diff_ppm": float(c.odd_even_frac) * 1e6,
                # Transit and half-an-orbit-later curves on one physical (ppm) scale, for reviewers.
                "transit_view_ppm": (c.transit_view_rel * 1e6).tolist() if big else [],
                "secondary_view_ppm": (c.secondary_view_rel * 1e6).tolist() if big else [],
                "centroid_shift": float(c.centroid_shift),
                "fits_url": job["fits_url"], "model_sha256": self.model_sha256,
                "global_view": view(c.global_view), "local_view": view(c.local_view),
                "odd_view": view(c.odd_view), "even_view": view(c.even_view),
                "secondary_view": view(c.secondary_view),
            })
        self.api.post("/candidates/processed", {
            "worker_hostname": self.host,
            "tic_id": job["tic_id"], "mission": job["mission"], "sector": job.get("sector"),
            "duration_seconds": time.monotonic() - t0, "candidates_found": len(candidates),
        })
        self.stars += 1
        self.found += len(candidates)
        best = max(scores, default=0.0)
        logging.info("%s %s: %d candidate(s), best score %.3f", job["mission"], job["tic_id"], len(candidates), best)

    # ── loops ────────────────────────────────────────────────────────────────
    def heartbeat_loop(self) -> None:
        import psutil

        psutil.cpu_percent(None)
        while not self.stop.wait(HEARTBEAT_SECONDS):
            job = self.current or {}
            try:
                self.api.post("/telemetry/heartbeat", {
                    "hostname": self.host,
                    "uptime_seconds": int(time.monotonic() - self.started),
                    "stars_analyzed": self.stars, "candidates_found": self.found,
                    "cpu_percent": psutil.cpu_percent(None),
                    "ram_percent": psutil.virtual_memory().percent,
                    "current_tic_id": job.get("tic_id"), "current_sector": job.get("sector"),
                })
            except Exception as exc:
                logging.warning("heartbeat failed: %s", exc)

    def run(self) -> None:
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        logging.info("node started; coordinator %s", self.cfg["api_url"])
        while not self.stop.is_set():
            try:
                jobs = self.api.get("/queue/next", hostname=self.host, limit=self.cfg["batch_size"])
            except Exception as exc:
                logging.warning("could not reach coordinator: %s", exc)
                self.stop.wait(IDLE_SECONDS)
                continue
            if not jobs:
                self.stop.wait(IDLE_SECONDS)
                continue
            for job in jobs:
                if self.stop.is_set():
                    break
                self.current = job
                try:
                    self.process(job)
                except Exception as exc:
                    # ponytail: a failed job stays assigned and is re-queued by the
                    # server after job_timeout_seconds; add an explicit /fail call if
                    # poison jobs start looping between nodes.
                    logging.warning("job %s failed: %s", job.get("tic_id"), exc)
                self.current = None
        try:
            self.api.post("/queue/release", hostname=self.host)
        except Exception:
            pass
        logging.info("node stopped")


def _lower_priority() -> None:
    import psutil

    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Lumina volunteer node worker")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    # Thread caps must be set before numpy/onnxruntime load.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(cfg["threads"])
    # The service account has no usable home (systemd ProtectHome, no-home
    # system user); astropy writes its config/cache under $HOME.
    home = Path(cfg["data_dir"]) / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = os.environ["USERPROFILE"] = str(home)

    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.handlers.RotatingFileHandler(log_dir / "node.log", maxBytes=5_000_000, backupCount=3)],
    )

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    _lower_priority()
    Worker(cfg, stop).run()


if __name__ == "__main__":
    main()
