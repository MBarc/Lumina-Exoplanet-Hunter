"""
Smoke test for the power-loss-resume machinery in ml/train.py.

The house this trains in has real power cuts, so _train_fold must survive
being killed at any point without losing fold progress.  This test simulates
a run that is cut off at every safe exit point (--max-hours 0 makes every
budget check fire immediately) and verifies that successive relaunches walk
one fold forward through its full lifecycle instead of restarting it:

  run 1: main-loop epoch 1 trains, budget exit
         -> epoch_checkpoint_fold_1.pt exists, no main_done flag
  run 2: main loop already done -> main_done checkpoint written,
         SWA epoch 1 trains, budget exit
         -> swa_checkpoint_fold_1.pt has completed_epochs == 1
  run 3: SWA resumes at epoch 2, budget exit
         -> completed_epochs == 2
  run 4: SWA resumes at epoch 3 (the last), BN update + eval run,
         fold finalized
         -> fold_complete_1.json written, both resume checkpoints deleted

Uses a tiny synthetic dataset (no FITS/cache needed) with the full ExoNet.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ml.train import _train_fold
from ml.model import SCALAR_FEATURES


class TinyDataset(Dataset):
    """Minimal stand-in for MultiMissionDataset: deterministic random views,
    alternating labels, and the two attributes _train_fold touches
    (.labels and .augment)."""

    def __init__(self, n: int = 32):
        self.labels = [float(i % 2) for i in range(n)]
        self.augment = False

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        gv     = torch.randn(2, 2001, generator=g)
        lv     = torch.randn(2, 201,  generator=g)
        ov     = torch.randn(2, 201,  generator=g)
        ev     = torch.randn(2, 201,  generator=g)
        sv     = torch.randn(2, 201,  generator=g)
        cv     = torch.randn(1, 201,  generator=g)
        scalar = torch.randn(SCALAR_FEATURES, generator=g)
        label  = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return gv, lv, ov, ev, sv, cv, scalar, label


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        epochs=1,               # main loop is a single epoch so the test is fast
        patience=5,
        batch_size=8,
        lr=1e-3,
        num_workers=0,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        save_all_folds=True,
        swa_epochs=3,
        swa_lr=1e-5,
        swa_momentum=0.9,
        use_se=True,
        dropout=0.4,
        weight_decay=1e-4,
        lr_schedule="plateau",
        warmup_epochs=0,
        cosine_t0=10,
        hard_neg_update_freq=0,  # skip hard-negative mining (needs full dataset API)
        pretrained_global=None,
        pretrained_local=None,
        max_hours=0.0,           # every budget check fires at the next safe exit
        _run_t_start=time.time(),
    )


def _launch(dataset, train_idx, val_idx, args, device, out_dir: Path):
    """One simulated launch. Returns 'exit' if the budget check sys.exit(0)'d,
    'done' if _train_fold returned normally."""
    args._run_t_start = time.time()  # each launch gets a fresh wall clock
    try:
        _train_fold(dataset, train_idx, val_idx, args, device,
                    out_dir / "exonet.pt", fold_num=1)
        return "done"
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise AssertionError(f"unexpected exit code {exc.code}")
        return "exit"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    dataset = TinyDataset(32)
    train_idx = list(range(0, 24))
    val_idx   = list(range(24, 32))
    args = _make_args()

    out_dir = Path(tempfile.mkdtemp(prefix="smoke_resume_"))
    epoch_ckpt = out_dir / "epoch_checkpoint_fold_1.pt"
    swa_ckpt   = out_dir / "swa_checkpoint_fold_1.pt"
    complete   = out_dir / "fold_complete_1.json"
    print(f"out_dir = {out_dir}", flush=True)

    try:
        print("\n[run 1] fresh start — expect budget exit after main epoch 1", flush=True)
        outcome = _launch(dataset, train_idx, val_idx, args, device, out_dir)
        assert outcome == "exit", f"expected budget exit, got {outcome}"
        assert epoch_ckpt.exists(), "epoch checkpoint missing after run 1"
        ck = torch.load(epoch_ckpt, map_location="cpu", weights_only=True)
        assert ck["epoch"] == 1, f"expected epoch 1 checkpoint, got {ck['epoch']}"
        assert not ck.get("main_done", False), "main_done should not be set yet"
        assert not swa_ckpt.exists() and not complete.exists()
        print("    OK: epoch checkpoint at epoch 1, no main_done", flush=True)

        print("\n[run 2] resume — expect main_done + SWA epoch 1, budget exit", flush=True)
        outcome = _launch(dataset, train_idx, val_idx, args, device, out_dir)
        assert outcome == "exit", f"expected budget exit, got {outcome}"
        ck = torch.load(epoch_ckpt, map_location="cpu", weights_only=True)
        assert ck.get("main_done", False), "main_done flag not written"
        assert swa_ckpt.exists(), "SWA checkpoint missing after run 2"
        sck = torch.load(swa_ckpt, map_location="cpu", weights_only=True)
        assert sck["completed_epochs"] == 1, f"expected 1 SWA epoch, got {sck['completed_epochs']}"
        assert not complete.exists()
        print("    OK: main_done set, SWA checkpointed at epoch 1", flush=True)

        print("\n[run 3] resume — expect SWA epoch 2, budget exit", flush=True)
        outcome = _launch(dataset, train_idx, val_idx, args, device, out_dir)
        assert outcome == "exit", f"expected budget exit, got {outcome}"
        sck = torch.load(swa_ckpt, map_location="cpu", weights_only=True)
        assert sck["completed_epochs"] == 2, f"expected 2 SWA epochs, got {sck['completed_epochs']}"
        print("    OK: SWA resumed and checkpointed at epoch 2", flush=True)

        print("\n[run 4] resume — expect SWA epoch 3, BN update, finalize", flush=True)
        outcome = _launch(dataset, train_idx, val_idx, args, device, out_dir)
        assert outcome == "done", f"expected normal completion, got {outcome}"
        assert complete.exists(), "fold_complete_1.json not written"
        with open(complete, "r", encoding="utf-8") as fh:
            result = json.load(fh)
        assert isinstance(result["fold_auc"], float)
        assert len(result["best_val_scores"]) == len(val_idx), (
            f"expected {len(val_idx)} OOF scores, got {len(result['best_val_scores'])}")
        assert not epoch_ckpt.exists(), "epoch checkpoint not cleaned up"
        assert not swa_ckpt.exists(), "SWA checkpoint not cleaned up"
        assert not any(out_dir.glob("*.tmp")), "leftover .tmp files from atomic saves"
        print(f"    OK: fold complete (val AUC {result['fold_auc']:.4f}), "
              f"resume checkpoints cleaned up", flush=True)

        print("\nALL STEPS PASSED — interrupted runs resume mid-fold and mid-SWA "
              "without losing progress.", flush=True)
        return 0
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
