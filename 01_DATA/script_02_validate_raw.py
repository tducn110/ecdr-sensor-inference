#!/usr/bin/env python3
"""02_validate_raw.py — Pair acc/gyro, shapes, NaN/Inf, annotation validity, timeline gaps."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "hapt" / "RawData"
OUT = ROOT / "data" / "raw_validation.json"


def main() -> None:
    acc_files = sorted(RAW.glob("acc_exp*_user*.txt"))
    gyro_files = sorted(RAW.glob("gyro_exp*_user*.txt"))
    issues, records = [], []
    acc_names = {p.name.replace("acc_", "") for p in acc_files}
    gyro_names = {p.name.replace("gyro_", "") for p in gyro_files}
    for missing in sorted(acc_names - gyro_names):
        issues.append(f"missing gyro pair for {missing}")
    for missing in sorted(gyro_names - acc_names):
        issues.append(f"missing acc pair for {missing}")

    labels = np.loadtxt(RAW / "labels.txt", dtype=int)  # exp,user,act,start,end
    per_rec_labels: dict[int, list] = {}
    for row in labels:
        per_rec_labels.setdefault(int(row[0]), []).append(row)
    label_rows = len(labels)
    if label_rows == 0:
        issues.append("labels.txt empty")

    for ap in acc_files:
        stem = ap.name.replace("acc_", "")  # expXX_userYY.txt
        gp = RAW / ("gyro_" + stem)
        if not gp.exists():
            continue
        a = np.loadtxt(ap)
        g = np.loadtxt(gp)
        if a.ndim == 1:
            a = a.reshape(-1, 3)
        if g.ndim == 1:
            g = g.reshape(-1, 3)
        exp = int(re.search(r"exp(\d+)", stem).group(1))
        ok_pair = a.shape[0] == g.shape[0]
        if not ok_pair:
            issues.append(f"{stem}: acc rows {a.shape[0]} != gyro rows {g.shape[0]}")
        nan_a = int(np.isnan(a).sum() + np.isinf(a).sum())
        nan_g = int(np.isnan(g).sum() + np.isinf(g).sum())
        if nan_a or nan_g:
            issues.append(f"{stem}: non-finite acc={nan_a} gyro={nan_g}")
        rows = a.shape[0]
        # annotation checks for this experiment
        lab = per_rec_labels.get(exp, [])
        if not lab:
            issues.append(f"{stem}: no annotation rows for exp{exp:02d}")
        max_end = 0
        for r in lab:
            if int(r[3]) > int(r[4]):
                issues.append(f"{stem}: start>end row {r.tolist()}")
            if int(r[4]) > rows:
                issues.append(f"{stem}: label end {int(r[4])} > rows {rows}")
            max_end = max(max_end, int(r[4]))
        # timeline gaps: coverage of annotated samples (labels may legitimately
        # leave unannotated gaps -- record, do not auto-label)
        annotated = sum(int(r[4]) - int(r[3]) + 1 for r in lab)
        records.append({
            "file": stem, "exp": exp, "rows": rows,
            "acc_gyro_aligned": bool(ok_pair),
            "nonfinite": nan_a + nan_g,
            "annotated_samples": int(annotated),
            "coverage_ratio": round(annotated / rows, 4) if rows else 0.0,
            "label_rows": len(lab),
        })

    summary = {
        "recordings": len(records),
        "label_rows_total": label_rows,
        "all_aligned": all(r["acc_gyro_aligned"] for r in records),
        "total_issues": len(issues),
        "issues": issues[:50],
        "row_count_min": min(r["rows"] for r in records),
        "row_count_max": max(r["rows"] for r in records),
        "coverage_min": min(r["coverage_ratio"] for r in records),
        "note": "gaps between annotated intervals are unannotated -- never auto-labeled",
    }
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
