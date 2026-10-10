"""
Merge preprocessing caches (e.g. the Kepler v2 cache + a TESS-only cache built
with --missions tess) into one cache that ml.train can load.

Every array is concatenated along samples. String fields are widened so no
star id is truncated. Refuses caches whose array shapes disagree.

Run:  python -m scripts.merge_caches OUT.npz IN1.npz IN2.npz [...]
Needs RAM for the combined arrays (~size of the inputs).
"""
import sys
from pathlib import Path

import numpy as np

out, inputs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
if len(inputs) < 2:
    sys.exit("usage: merge_caches OUT.npz IN1.npz IN2.npz [...]")

caches = [np.load(p, mmap_mode="r") for p in inputs]
keys = set(caches[0].files)
for p, c in zip(inputs[1:], caches[1:]):
    if set(c.files) != keys:
        sys.exit(f"{p} has fields {sorted(c.files)}, expected {sorted(keys)}")

merged = {}
for k in sorted(keys):
    parts = [c[k] for c in caches]
    if len({a.shape[1:] for a in parts}) != 1:
        sys.exit(f"field {k}: shapes differ {[a.shape for a in parts]}")
    if parts[0].dtype.kind == "U":
        width = max(16, *(a.dtype.itemsize // 4 for a in parts))
        parts = [a.astype(f"U{width}") for a in parts]
    merged[k] = np.concatenate(parts)

n = len(merged["labels"])
tmp = out.with_name(out.stem + ".tmp.npz")
np.savez(tmp, **merged)
tmp.replace(out)
missions, counts = np.unique(merged["missions"], return_counts=True)
print(f"wrote {out}: {n} samples  " + "  ".join(f"{m}={c}" for m, c in zip(missions, counts)))
