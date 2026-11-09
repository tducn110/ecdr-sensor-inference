"""W4-007 evidence: exhaustive monotonicity check of the frozen executable normalization.

For each channel c with frozen float32 (mean_c, std_c), enumerate EVERY finite float32 raw value in
increasing value order and verify that N_hat(x) = fl32(fl32(x - mean_c) / std_c), evaluated with the same
NumPy float32 ufuncs used by ecdr.week4.reference.execute_normalization, is non-decreasing.

Scope: this verifies the executable on THIS NumPy build / CPU for the frozen constants. It is an exhaustive
finite check, not a proof for other builds; the proof is the IEEE-754 monotonicity argument stated in
ecdr/week4/reference.py::enclose_normalization (premise: correctly rounded elementwise float32 sub/div).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from ecdr.week4.reference import REFERENCE_DIR  # noqa: E402

CHUNK = 1 << 24
OUT = Path(__file__).with_name("exhaustive_monotonicity.json")


def value_ordered_chunks():
    """Yield uint32 bit-pattern chunks of all finite float32 in non-decreasing value order."""
    # negatives: patterns 0xFF7FFFFF down to 0x80000000 (value increasing), then +0 .. 0x7F7FFFFF
    neg_hi, neg_lo = 0xFF7FFFFF, 0x80000000
    start = neg_hi
    while start >= neg_lo:
        stop = max(start - CHUNK + 1, neg_lo)
        yield np.arange(start, stop - 1, -1, dtype=np.uint32)
        start = stop - 1
    pos_hi = 0x7F7FFFFF
    start = 0
    while start <= pos_hi:
        stop = min(start + CHUNK - 1, pos_hi)
        yield np.arange(start, stop + 1, dtype=np.uint32)
        start = stop + 1


def main() -> None:
    with np.load(REFERENCE_DIR / "normalization.npz") as d:
        mean, std = d["mean"].copy(), d["std"].copy()
    assert mean.dtype == np.float32 and std.dtype == np.float32
    report = {
        "numpy_version": np.__version__,
        "constants": {"mean": [float(v) for v in mean], "std": [float(v) for v in std]},
        "values_per_channel": 0,
        "channels": [],
    }
    t0 = time.time()
    for c in range(6):
        m, s = mean[c], std[c]
        prev_last = None
        total = 0
        violations = 0
        nan_count = 0
        inf_count = 0
        first_violation = None
        for bits in value_ordered_chunks():
            x = bits.view(np.float32)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                y = (x - m) / s
            assert y.dtype == np.float32
            nan_count += int(np.count_nonzero(np.isnan(y)))
            inf_count += int(np.count_nonzero(np.isinf(y)))
            bad = y[1:] < y[:-1]
            nbad = int(np.count_nonzero(bad))
            if nbad and first_violation is None:
                i = int(np.argmax(bad))
                first_violation = {"bits": [int(bits[i]), int(bits[i + 1])], "y": [float(y[i]), float(y[i + 1])]}
            violations += nbad
            if prev_last is not None and y[0] < prev_last:
                violations += 1
                first_violation = first_violation or {"chunk_boundary": True}
            prev_last = y[-1]
            total += len(x)
        report["values_per_channel"] = total
        report["channels"].append(
            {
                "channel": c,
                "mean": float(m),
                "std": float(s),
                "values_checked": total,
                "monotonicity_violations": violations,
                "nan_outputs": nan_count,
                "inf_outputs": inf_count,
                "first_violation": first_violation,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        print(f"channel {c}: checked={total} violations={violations} nan={nan_count} inf={inf_count}", flush=True)
    report["all_channels_monotone"] = all(ch["monotonicity_violations"] == 0 and ch["nan_outputs"] == 0 for ch in report["channels"])
    OUT.write_text(json.dumps(report, indent=2))
    print("all_channels_monotone =", report["all_channels_monotone"], flush=True)


if __name__ == "__main__":
    main()
