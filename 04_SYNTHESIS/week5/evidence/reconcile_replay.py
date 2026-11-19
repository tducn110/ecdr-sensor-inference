"""Recount saved replay outcomes and compare them to the Week 1 baseline."""
import csv
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
W1 = ROOT / "output/week1_closeout_20261102/reference"
W4 = ROOT / "output/week4/week4_initial_20261103"


def main():
    with (W1 / "test_predictions_manifest.csv").open() as f:
        original = {r["global_row_id"]: r["predicted_label_zero_based"] for r in csv.DictReader(f)}
    count, hits, differences = Counter(), Counter(), Counter()
    m00 = {}
    with (W4 / "replay_trace.csv").open() as f:
        for r in csv.DictReader(f):
            mid = r["method_id"]
            count[mid] += 1
            hits[mid] += r["is_reuse"] == "True"
            differences[mid] += r["predicted_class"] != r["reference_class"]
            if mid == "M00":
                m00[r["global_row_id"]] = r["predicted_class"]
    assert original.keys() == m00.keys() and len(m00) == 5053
    assert len(count) == 15 and set(count.values()) == {5053}
    assert hits["M03"] == 5034 and sum(hits[f"M{i:02}"] for i in range(5, 14)) == 0
    receipt = {
        "m00_vs_week1_top1_disagreements": sum(original[k] != m00[k] for k in original),
        "replay_rows_by_method": dict(count),
        "reuse_flags_by_method": dict(hits),
        "reference_disagreements_by_method": dict(differences),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
