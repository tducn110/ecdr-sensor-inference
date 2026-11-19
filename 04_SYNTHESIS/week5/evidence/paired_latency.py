"""Recount complete-path latency against the same M00 query and repetition."""
import csv
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[3]
TRACE = ROOT / "output/week4/week4_initial_20261103/timing_trace.csv"


def summarize(pairs):
    if not pairs:
        return None
    baseline = fmean(a for a, _ in pairs)
    method = fmean(b for _, b in pairs)
    return {
        "timed_observations": len(pairs),
        "baseline_mean_ms": baseline,
        "method_mean_ms": method,
        "delta_mean_ms": method - baseline,
        "latency_reduction_pct": 100 * (baseline - method) / baseline,
    }


def main():
    records = defaultdict(dict)
    with TRACE.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["repetition"]), int(row["global_row_id"]))
            method = records[row["method_id"]]
            if key in method or row["is_reuse"] not in ("True", "False"):
                raise ValueError(f"Duplicate or invalid timing row: {row}")
            method[key] = (float(row["query_latency_ms"]), row["is_reuse"] == "True")

    baseline = records["M00"]
    assert len(records) == 15 and len(baseline) == 3 * 5053
    results = {}
    for mid, items in sorted(records.items()):
        assert items.keys() == baseline.keys()
        hit, miss = [], []
        for key, (elapsed, reused) in items.items():
            (hit if reused else miss).append((baseline[key][0], elapsed))
        results[mid] = {"all": summarize(hit + miss), "reuse": summarize(hit),
                        "nonreuse": summarize(miss)}
    assert results["M03"]["reuse"]["timed_observations"] == 3 * 5034
    assert all(results[f"M{i:02}"]["reuse"] is None for i in range(5, 14))
    receipt = {
        "scope": "Saved Week 4 operational query timer, paired by repetition and global row ID. Not a new runtime run.",
        "timing_trace": str(TRACE.relative_to(ROOT)),
        "timing_trace_sha256": sha256(TRACE.read_bytes()).hexdigest(),
        "results": results,
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
