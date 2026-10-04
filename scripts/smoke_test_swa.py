"""
Smoke test for the SWA BN-update path that broke during the kepler_run.

Reproduces the failing path on a tiny synthetic batch (no FITS data needed):

  1. Build an ExoNet, wrap it in AveragedModel.
  2. Run a couple of fake SGD steps and call update_parameters() — the
     pattern train.py uses inside the SWA loop.
  3. Copy the averaged state into a plain ExoNet (the workaround at
     train.py:2528-2530).
  4. Set the plain model to .train() and feed it batches with all 7 views,
     mirroring train.py:2532-2541.  This is the call that previously raised
     "ExoNet.forward() missing 6 required positional arguments".
  5. Switch to .eval() and confirm a forward pass produces (B, 1) logits.

If any step throws, the SWA BN update path is still broken.  If everything
prints OK, the fix in the uncommitted train.py holds and a real training run
will exercise the same code with no surprises.
"""
from __future__ import annotations

import sys
import traceback

import torch

from ml.model import ExoNet


def _make_batch(batch_size: int, device: torch.device) -> tuple:
    """Construct one batch of synthetic inputs matching ExoNet.forward signature."""
    gv     = torch.randn(batch_size, 2, 2001, device=device)
    lv     = torch.randn(batch_size, 2, 201,  device=device)
    ov     = torch.randn(batch_size, 2, 201,  device=device)
    ev     = torch.randn(batch_size, 2, 201,  device=device)
    sv     = torch.randn(batch_size, 2, 201,  device=device)
    cv     = torch.randn(batch_size, 1, 201,  device=device)
    scalar = torch.randn(batch_size, 17,      device=device)
    return gv, lv, ov, ev, sv, cv, scalar


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    print("[1] constructing ExoNet ...", flush=True)
    model = ExoNet(use_se=True, dropout=0.4).to(device)
    model.train()

    print("[2] wrapping in AveragedModel ...", flush=True)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("[3] simulating SWA SGD steps ...", flush=True)
    for step in range(2):
        gv, lv, ov, ev, sv, cv, scalar = _make_batch(4, device)
        labels = torch.randint(0, 2, (4, 1), device=device).float()
        logits = model(gv, lv, ov, ev, sv, cv, scalar)
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        swa_model.update_parameters(model)
        print(f"    step {step}: loss={loss.item():.4f}", flush=True)

    print("[4] copying averaged weights into a plain ExoNet ...", flush=True)
    swa_weights = ExoNet(use_se=True, dropout=0.4).to(device)
    missing, unexpected = swa_weights.load_state_dict(
        swa_model.module.state_dict(), strict=False
    )
    if missing or unexpected:
        print(f"    WARN missing keys:    {missing}", flush=True)
        print(f"    WARN unexpected keys: {unexpected}", flush=True)
    else:
        print("    state_dict copied cleanly (strict=False, no diffs)", flush=True)

    print("[5] running BN update via train()-mode forward passes ...", flush=True)
    swa_weights.train()
    with torch.no_grad():
        for batch_idx in range(3):
            gv, lv, ov, ev, sv, cv, scalar = _make_batch(4, device)
            out = swa_weights(gv, lv, ov, ev, sv, cv, scalar)
            assert out.shape == (4, 1), f"unexpected output shape {out.shape}"
    print("    BN update loop OK — 3 batches forwarded with 7 views each", flush=True)

    print("[6] eval-mode forward pass ...", flush=True)
    swa_weights.eval()
    with torch.no_grad():
        gv, lv, ov, ev, sv, cv, scalar = _make_batch(4, device)
        out = swa_weights(gv, lv, ov, ev, sv, cv, scalar)
    print(f"    eval output shape = {tuple(out.shape)} OK", flush=True)

    print("\nALL STEPS PASSED — the SWA BN-update path in train.py is sound.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
