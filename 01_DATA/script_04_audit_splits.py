#!/usr/bin/env python3
"""04_audit_splits.py — subject-disjoint train/val/test, chronological order kept,
leakage checks (no subject overlap, no recording split across sets)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HAPT = ROOT / "data" / "hapt"
OUT = DATA / "split_audit.json"
VAL_FRACTION = 0.2  # of train subjects -> val


def read_subjects(path: Path) -> set[int]:
    return {int(x) for x in path.read_text().split()}


def main() -> None:
    df = pd.read_parquet(DATA / "window_manifest.parquet")
    train_subj = read_subjects(HAPT / "Train" / "subject_id_train.txt")
    test_subj = read_subjects(HAPT / "Test" / "subject_id_test.txt")

    issues = []
    overlap = train_subj & test_subj
    if overlap:
        issues.append(f"train/test subject overlap: {sorted(overlap)}")

    train_sorted = sorted(train_subj)
    n_val = max(1, int(round(len(train_sorted) * VAL_FRACTION)))
    val_subj = set(train_sorted[-n_val:])  # deterministic tail slice
    train_only = set(train_sorted) - val_subj

    all_subj = set(df["subject"].unique())
    unknown = all_subj - (train_subj | test_subj)
    if unknown:
        issues.append(f"subjects not in UCI split files: {sorted(unknown)}")

    df["split"] = df["subject"].map(
        lambda s: "train" if s in train_only else ("val" if s in val_subj else "test"))

    # leakage: each recording (exp) must belong to exactly one split
    rec_split = df.groupby("exp")["split"].nunique()
    recs_split = rec_split[rec_split > 1]
    if len(recs_split):
        issues.append(f"recordings spanning splits: {recs_split.index.tolist()}")

    # chronological order within recording must be preserved in file order
    order_ok = True
    for _, g in df.groupby("exp"):
        if not g["win_idx"].is_monotonic_increasing:
            order_ok = False
            break
    if not order_ok:
        issues.append("window order within a recording is not chronological")

    counts = df["split"].value_counts().to_dict()
    summary = {
        "train_subjects": sorted(train_only),
        "val_subjects": sorted(val_subj),
        "test_subjects": sorted(test_subj),
        "subject_counts": {"train": len(train_only), "val": len(val_subj),
                           "test": len(test_subj)},
        "window_counts": {k: int(v) for k, v in counts.items()},
        "subject_disjoint": not bool(overlap),
        "recording_disjoint": len(recs_split) == 0,
        "chronological_kept": order_ok,
        "total_issues": len(issues),
        "issues": issues,
    }
    df[["exp", "subject", "win_idx", "split"]].to_parquet(
        DATA / "windows_with_split.parquet", index=False)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
