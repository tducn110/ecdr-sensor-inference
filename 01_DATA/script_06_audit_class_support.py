#!/usr/bin/env python3
"""06_audit_class_support.py — G2-A: per-class pure-window support on train/val/test,
transition label durations, and pure-window capacity vs realized counts.

Answers the G1 REVISE question: can K=12 pure-only training be supported by data,
or must the backbone start at K=6? Produces data/g2_class_support.json.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "data" / "hapt" / "RawData"
W, S, FS = 128, 64, 50
MIN_PURE_PER_SPLIT = 50     # below this, per-split recall is unreliable
MIN_SUBJECTS_PER_SPLIT = 3  # below this, subject-specific memorization risk

ACTIVITY_NAMES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
    7: "STAND_TO_SIT", 8: "SIT_TO_STAND", 9: "SIT_TO_LIE",
    10: "LIE_TO_SIT", 11: "STAND_TO_LIE", 12: "LIE_TO_STAND",
}
SPLITS = ("train", "val", "test")


def grid_capacity(a: int, b: int, t_len: int) -> int:
    """Windows (start = 0,64,...,<=t_len-W) fully inside inclusive span [a, b]."""
    lo, hi = a, min(b - W + 1, t_len - W)
    if hi < lo:
        return 0
    first = int(np.ceil(lo / S)) * S
    if first > hi:
        return 0
    return (hi - first) // S + 1


def main() -> None:
    issues: list[str] = []
    man = pd.read_parquet(DATA / "window_manifest.parquet")
    spl = pd.read_parquet(DATA / "windows_with_split.parquet")
    df = man.merge(spl[["exp", "win_idx", "split"]], on=["exp", "win_idx"],
                   how="left", validate="one_to_one")
    if df["split"].isna().any():
        issues.append("windows missing split assignment")

    pure = df[df["label_policy"] == "pure"]

    # ---- 1. pure support: class x split ----
    counts = (pd.crosstab(pure["label_majority"], pure["split"])
              .reindex(index=range(1, 13), columns=list(SPLITS), fill_value=0))
    subjects = (pure.groupby(["label_majority", "split"])["subject"].nunique()
                .unstack().reindex(index=range(1, 13), columns=list(SPLITS), fill_value=0))

    per_class = {}
    for cid in range(1, 13):
        row = {
            "name": ACTIVITY_NAMES[cid],
            "kind": "basic" if cid <= 6 else "transition",
            "pure": {s: int(counts.loc[cid, s]) for s in SPLITS},
            "pure_subjects": {s: int(subjects.loc[cid, s]) for s in SPLITS},
            "pure_total": int(counts.loc[cid].sum()),
        }
        flags = []
        for s in SPLITS:
            if row["pure"][s] == 0:
                flags.append(f"zero_pure_in_{s}")
            elif row["pure"][s] < MIN_PURE_PER_SPLIT:
                flags.append(f"low_pure_in_{s}(<{MIN_PURE_PER_SPLIT})")
            if row["pure_subjects"][s] < MIN_SUBJECTS_PER_SPLIT:
                flags.append(f"few_subjects_in_{s}(<{MIN_SUBJECTS_PER_SPLIT})")
        row["flags"] = flags
        per_class[str(cid)] = row

    # ---- 2. all-policy class distribution (fallback options) ----
    all_policy = {
        str(cid): {p: int(((df["label_majority"] == cid)
                           & (df["label_policy"] == p)).sum())
                   for p in ("pure", "majority", "ambiguous", "unlabeled")}
        for cid in range(1, 13)}

    # ---- 3. label-span durations from labels.txt ----
    labels = np.loadtxt(RAW / "labels.txt", dtype=int)
    dur: dict[int, list[int]] = {}
    span_rows: dict[int, list[tuple[int, int, int]]] = {}
    for row in labels:
        exp, cid, a, b = int(row[0]), int(row[2]), int(row[3]), int(row[4])
        dur.setdefault(cid, []).append(b - a + 1)
        span_rows.setdefault(cid, []).append((exp, a, b))

    durations = {}
    for cid in range(1, 13):
        d = np.array(dur.get(cid, []), dtype=float)
        if len(d) == 0:
            durations[str(cid)] = {"n_spans": 0}
            issues.append(f"class {cid}: no label spans in labels.txt")
            continue
        durations[str(cid)] = {
            "name": ACTIVITY_NAMES[cid],
            "n_spans": int(len(d)),
            "seconds": {
                "min": round(float(d.min()) / FS, 3),
                "p25": round(float(np.percentile(d, 25)) / FS, 3),
                "p50": round(float(np.percentile(d, 50)) / FS, 3),
                "p75": round(float(np.percentile(d, 75)) / FS, 3),
                "max": round(float(d.max()) / FS, 3),
            },
            "spans_ge_window": int((d >= W).sum()),
            "spans_shorter_than_window": int((d < W).sum()),
        }

    # ---- 4. pure capacity (grid) vs realized pure, per class ----
    rec_len = {}
    for ap in sorted(RAW.glob("acc_exp*_user*.txt")):
        exp = int(re.search(r"exp(\d+)", ap.name).group(1))
        with open(ap) as fh:
            rec_len[exp] = sum(1 for _ in fh)

    capacity = {cid: 0 for cid in range(1, 13)}
    for cid, rows in span_rows.items():
        for exp, a, b in rows:
            if exp in rec_len:
                capacity[cid] += grid_capacity(a, b, rec_len[exp])
    for cid in range(1, 13):
        realized = per_class[str(cid)]["pure_total"]
        per_class[str(cid)]["pure_capacity_grid"] = int(capacity[cid])
        if capacity[cid] != realized:
            issues.append(
                f"class {cid}: capacity {capacity[cid]} != realized pure {realized}")

    # ---- 5. decision gates ----
    basic_flags = [f"class {c}: {fl}" for c in range(1, 7)
                   for fl in per_class[str(c)]["flags"]]
    trans_flags = [f"class {c}: {fl}" for c in range(7, 13)
                   for fl in per_class[str(c)]["flags"]]

    summary = {
        "windows_total": int(len(df)),
        "pure_total": int(len(pure)),
        "labeled_total": int((df["label_policy"] != "unlabeled").sum()),
        "policy_counts": {k: int(v) for k, v in df["label_policy"].value_counts().items()},
        "thresholds": {
            "min_pure_per_split": MIN_PURE_PER_SPLIT,
            "min_subjects_per_split": MIN_SUBJECTS_PER_SPLIT,
            "window_samples": W, "stride": S, "sampling_hz": FS,
        },
        "per_class": per_class,
        "all_policy_counts": all_policy,
        "label_durations": durations,
        "k12_ready": len(trans_flags) == 0,
        "k6_ready": len(basic_flags) == 0,
        "k12_blockers": trans_flags,
        "k6_blockers": basic_flags,
        "total_issues": len(issues),
        "issues": issues,
        "note": ("pure = all 128 samples inside one label span; capacity counts "
                 "stride-64 grid starts fully inside each span (must equal realized)"),
    }
    (DATA / "g2_class_support.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

