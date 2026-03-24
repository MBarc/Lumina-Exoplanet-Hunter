"""
Autonomous multi-run orchestrator for ExoNet.

Runs a sequence of training configurations back-to-back, parses AUC from each
run's output, promotes improved models, then runs calibration and ensemble
assembly.  The whole session is time-budgeted so the script stops cleanly when
the wall-clock budget is nearly exhausted.

Usage
-----
::

    python -m ml.orchestrate \\
        --fits-dir  /data/fits_cache \\
        --output-dir /data/exonet_runs \\
        --budget-hours 8

Key outputs
-----------
  <output_dir>/best/exonet.pt           — best PyTorch checkpoint
  <output_dir>/best/exonet.onnx         — ONNX export of the best checkpoint
  <output_dir>/best/threshold.json      — classification thresholds
  <output_dir>/best/calibration.json    — temperature-scaling parameters
  <output_dir>/best/exonet_fold_*.pt    — per-fold checkpoints (for ensemble)
  <output_dir>/orchestrate_summary.txt  — human-readable run log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

# Ensure UTF-8 output on Windows when stdout is redirected to a file.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Configuration grid ────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """One training run configuration."""
    name:          str
    epochs:        int   = 75
    lr:            float = 1e-3
    batch_size:    int   = 128
    folds:         int   = 5
    patience:      int   = 12
    lr_schedule:   str   = "plateau"
    cosine_t0:     int   = 10
    use_se:        bool  = False
    dropout:       float = 0.4
    weight_decay:  float = 1e-4
    save_all_folds:bool  = True
    no_augment:    bool  = False
    extra_flags:   list[str] = field(default_factory=list)


# ── Parseable result line ─────────────────────────────────────────────────────

_RESULT_RE = re.compile(
    r"RESULT:\s*mean_auc=([0-9.]+)\s+std_auc=([0-9.]+)\s+best_auc=([0-9.]+)"
)


def _parse_result(log_text: str) -> Optional[tuple[float, float, float]]:
    """
    Extract (mean_auc, std_auc, best_auc) from the training log.
    Returns None if the RESULT line is absent (run crashed or was killed).
    """
    for line in reversed(log_text.splitlines()):
        m = _RESULT_RE.search(line)
        if m:
            return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


# ── Calibration step ──────────────────────────────────────────────────────────

def _run_calibration(run_dir: Path, fits_dir: Path, device: str = "cpu") -> bool:
    """
    Run temperature calibration on fold checkpoints inside *run_dir*.

    Loads the preprocessing cache from run_dir/preprocess_cache.npz (if present)
    to obtain the validation data, then calls ml.calibrate.calibrate_folds.

    Returns True on success, False on failure.
    """
    cache_path = run_dir / "preprocess_cache.npz"
    if not cache_path.exists():
        print(f"[orchestrate] Calibration skipped: no cache at {cache_path}", flush=True)
        return False

    fold_paths = sorted(run_dir.glob("exonet_fold_*.pt"))
    if not fold_paths:
        print("[orchestrate] Calibration skipped: no fold checkpoints found.", flush=True)
        return False

    print(f"[orchestrate] Running temperature calibration on {len(fold_paths)} folds ...",
          flush=True)
    try:
        import numpy as np
        import torch
        from ml.calibrate import calibrate_folds

        data    = np.load(cache_path, allow_pickle=False)
        gvs     = data["global_views"]
        lvs     = data["local_views"]
        scalars = data["scalars"]
        labels  = data["labels"]

        results = calibrate_folds(
            fold_checkpoint_paths=fold_paths,
            global_views=gvs,
            local_views=lvs,
            scalars=scalars,
            labels=labels,
            device=torch.device(device),
            output_dir=run_dir,
        )
        print(f"[orchestrate] Calibration done.  "
              f"ECE {results['ece_before']:.4f} → {results['ece_after']:.4f}  "
              f"(global T={results['global_temperature']:.4f})",
              flush=True)
        return True

    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrate] Calibration failed: {exc}", flush=True)
        return False


# ── ONNX export ───────────────────────────────────────────────────────────────

def _export_onnx(run_dir: Path) -> bool:
    """Export the best PyTorch checkpoint in run_dir to ONNX."""
    pt_path   = run_dir / "exonet.pt"
    onnx_path = run_dir / "exonet.onnx"
    if not pt_path.exists():
        print(f"[orchestrate] ONNX export skipped: {pt_path} not found.", flush=True)
        return False

    print(f"[orchestrate] Exporting ONNX model ...", flush=True)
    try:
        from ml.inference import ExoNetInference
        ExoNetInference.export_from_pytorch(pt_path, onnx_path)
        print(f"[orchestrate] ONNX exported: {onnx_path}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrate] ONNX export failed: {exc}", flush=True)
        return False


# ── Promotion: copy improved model to best/ ───────────────────────────────────

def _promote_run(run_dir: Path, best_dir: Path) -> None:
    """Copy all relevant outputs from *run_dir* into *best_dir*."""
    best_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("exonet.pt", "exonet.onnx", "threshold.json",
                    "calibration.json", "exonet_fold_*.pt"):
        for src in run_dir.glob(pattern) if "*" in pattern else [run_dir / pattern]:
            if src.exists():
                shutil.copy2(src, best_dir / src.name)
    print(f"[orchestrate] Promoted {run_dir.name} → {best_dir}", flush=True)


# ── Single training run ───────────────────────────────────────────────────────

def _launch_run(
    cfg: RunConfig,
    run_dir: Path,
    fits_dir: Path,
    cache_file: Optional[Path],
    cache_only: bool,
    timeout_secs: float,
) -> tuple[Optional[tuple[float, float, float]], str]:
    """
    Launch one training run as a subprocess, wait up to *timeout_secs*, and
    parse the RESULT line from its output.

    Returns ((mean_auc, std_auc, best_auc), log_text) or (None, log_text) if
    the run crashed, timed out, or produced no RESULT line.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    cmd: list[str] = [
        sys.executable, "-u", "-m", "ml.train",
        "--fits-dir",    str(fits_dir),
        "--output-dir",  str(run_dir),
        "--epochs",      str(cfg.epochs),
        "--lr",          str(cfg.lr),
        "--batch-size",  str(cfg.batch_size),
        "--folds",       str(cfg.folds),
        "--patience",    str(cfg.patience),
        "--lr-schedule", cfg.lr_schedule,
        "--cosine-t0",   str(cfg.cosine_t0),
        "--dropout",     str(cfg.dropout),
        "--weight-decay",str(cfg.weight_decay),
        "--num-workers", "0",
    ]
    if cfg.use_se:
        cmd.append("--use-se")
    if cfg.save_all_folds:
        cmd.append("--save-all-folds")
    if cfg.no_augment:
        cmd.append("--no-augment")
    if cache_file is not None:
        cmd += ["--cache-file", str(cache_file)]
    if cache_only:
        cmd.append("--cache-only")
    cmd.extend(cfg.extra_flags)

    print(f"\n[orchestrate] Starting run '{cfg.name}'  ->  {run_dir}", flush=True)
    print(f"[orchestrate] Command: {' '.join(cmd)}", flush=True)

    t_start = time.time()
    log_lines: list[str] = []

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with log_path.open("w", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None

            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    print(line, end="", flush=True)
                    log_fh.write(line)
                    log_lines.append(line)

                # Wall-clock timeout: kill if we exceed budget
                if time.time() - t_start > timeout_secs:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    print(f"\n[orchestrate] Run '{cfg.name}' timed out after "
                          f"{timeout_secs/3600:.1f}h — terminated.", flush=True)
                    break

            proc.wait()
            rc = proc.returncode

    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrate] Run '{cfg.name}' raised exception: {exc}", flush=True)
        rc = -1

    elapsed = time.time() - t_start
    log_text = "".join(log_lines)
    result = _parse_result(log_text)

    status = f"rc={rc}  elapsed={elapsed/60:.1f}m"
    if result:
        mean_auc, std_auc, best_auc = result
        print(f"[orchestrate] Run '{cfg.name}' finished.  "
              f"mean_auc={mean_auc:.4f}±{std_auc:.4f}  best_auc={best_auc:.4f}  {status}",
              flush=True)
    else:
        print(f"[orchestrate] Run '{cfg.name}' finished with no RESULT line.  {status}",
              flush=True)

    return result, log_text


# ── Main orchestration loop ───────────────────────────────────────────────────

def orchestrate(args: argparse.Namespace) -> None:
    """Main entry point: run all configs within the time budget."""

    output_dir  = Path(args.output_dir)
    fits_dir    = Path(args.fits_dir)
    best_dir    = output_dir / "best"
    budget_secs = args.budget_hours * 3600.0
    wall_start  = time.time()
    cache_only  = args.cache_only

    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Shared preprocessing cache: all runs reuse the same BLS output.
    shared_cache = output_dir / "preprocess_cache.npz"

    # ── Run schedule ──────────────────────────────────────────────────────────
    configs: list[RunConfig] = [
        RunConfig(
            name="run01_baseline_cosine",
            epochs=75, lr=1e-3, batch_size=128, folds=5, patience=12,
            lr_schedule="cosine", cosine_t0=15,
            use_se=False, dropout=0.4, weight_decay=1e-4,
            save_all_folds=True,
        ),
        RunConfig(
            name="run02_se_cosine",
            epochs=75, lr=1e-3, batch_size=128, folds=5, patience=12,
            lr_schedule="cosine", cosine_t0=15,
            use_se=True, dropout=0.4, weight_decay=1e-4,
            save_all_folds=True,
        ),
        RunConfig(
            name="run03_se_plateau_lr_low",
            epochs=75, lr=5e-4, batch_size=128, folds=5, patience=12,
            lr_schedule="plateau",
            use_se=True, dropout=0.3, weight_decay=2e-4,
            save_all_folds=True,
        ),
        RunConfig(
            name="run04_se_cosine_t0_10",
            epochs=75, lr=1e-3, batch_size=64, folds=5, patience=12,
            lr_schedule="cosine", cosine_t0=10,
            use_se=True, dropout=0.4, weight_decay=1e-4,
            save_all_folds=True,
        ),
        RunConfig(
            name="run05_se_cosine_long",
            epochs=100, lr=8e-4, batch_size=128, folds=5, patience=15,
            lr_schedule="cosine", cosine_t0=20,
            use_se=True, dropout=0.35, weight_decay=1e-4,
            save_all_folds=True,
        ),
    ]

    summary_lines: list[str] = [
        "ExoNet Orchestration Summary",
        "=" * 60,
        f"Budget:    {args.budget_hours:.1f} h",
        f"Started:   {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Fits dir:  {fits_dir}",
        f"Output:    {output_dir}",
        "",
    ]

    best_mean_auc = -1.0
    best_run_name = "(none)"

    for cfg in configs:
        elapsed_so_far = time.time() - wall_start
        remaining      = budget_secs - elapsed_so_far

        # Reserve 15 minutes for calibration + ONNX export at the end.
        min_run_budget = 5 * 60  # don't start a run with less than 5 min left
        if remaining < min_run_budget:
            msg = (f"[orchestrate] Budget nearly exhausted "
                   f"({remaining/60:.1f} min remaining). Stopping run schedule.")
            print(msg, flush=True)
            summary_lines.append(msg)
            break

        # Give this run at most remaining − 15 min (leave room for post-processing).
        run_budget = remaining - 15 * 60

        run_dir = output_dir / cfg.name
        result, _log_text = _launch_run(
            cfg=cfg,
            run_dir=run_dir,
            fits_dir=fits_dir,
            cache_file=shared_cache,
            cache_only=cache_only,
            timeout_secs=run_budget,
        )

        if result is not None:
            mean_auc, std_auc, best_auc = result
            summary_lines.append(
                f"  {cfg.name:40s}  mean_auc={mean_auc:.4f}±{std_auc:.4f}  "
                f"best_auc={best_auc:.4f}"
            )

            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_run_name = cfg.name
                print(f"\n[orchestrate] New best: {cfg.name}  mean_auc={mean_auc:.4f}",
                      flush=True)

                # Copy preprocessing cache so calibration can find it in best/
                if shared_cache.exists():
                    shutil.copy2(shared_cache, run_dir / "preprocess_cache.npz")

                _run_calibration(run_dir, fits_dir)
                _export_onnx(run_dir)
                _promote_run(run_dir, best_dir)
        else:
            summary_lines.append(f"  {cfg.name:40s}  FAILED / timed out")

        # After first run, subsequent runs can reuse the cache (cache_only=True
        # for the FITS download phase, not for training itself).
        if shared_cache.exists():
            cache_only = True

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - wall_start
    summary_lines += [
        "",
        "=" * 60,
        f"Best run   : {best_run_name}",
        f"Best AUC   : {best_mean_auc:.4f}",
        f"Total time : {total_elapsed / 3600:.2f} h",
        f"Finished   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Best model : {best_dir}",
    ]

    summary_text = "\n".join(summary_lines)
    print(f"\n\n{'='*60}", flush=True)
    print(summary_text, flush=True)

    summary_path = output_dir / "orchestrate_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\n[orchestrate] Summary saved: {summary_path}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous ExoNet training orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fits-dir",     type=Path, required=True,
                        help="Directory containing (or to receive) FITS files.")
    parser.add_argument("--output-dir",   type=Path, required=True,
                        help="Root directory for all run outputs.")
    parser.add_argument("--budget-hours", type=float, default=8.0,
                        help="Total wall-clock budget in hours.")
    parser.add_argument("--cache-only",   action="store_true",
                        help="Skip MAST downloads on all runs.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    orchestrate(_parse_args())
