#!/usr/bin/env python3
"""03_build_windows.py — 6x128 windows stride 64 per recording (never across
recordings), with label-coverage policy: pure / majority / ambiguous / unlabeled."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "hapt" / "RawData"
DATA = ROOT / "data"
W, S = 128, 64
MAJORITY_FRAC = 0.6


def main() -> None:
    labels = np.loadtxt(RAW / "labels.txt", dtype=int)
    per_rec: dict[int, list] = {}
    for row in labels:
        per_rec.setdefault(int(row[0]), []).append(row)

    rows, arrays = [], []
    for ap in sorted(RAW.glob("acc_exp*_user*.txt")):
        stem = ap.name.replace("acc_", "")
        gp = RAW / ("gyro_" + stem)
        exp = int(re.search(r"exp(\d+)", stem).group(1))
        usr = int(re.search(r"user(\d+)", stem).group(1))
        acc = np.loadtxt(ap, dtype=np.float32).reshape(-1, 3)
        gy = np.loadtxt(gp, dtype=np.float32).reshape(-1, 3)
        sig = np.concatenate([acc, gy], axis=1).T  # (6, T)
        T = sig.shape[1]
        # per-sample label, 0 = unannotated (never auto-labeled)
        plabel = np.zeros(T + 1, dtype=np.int16)
        for r in per_rec.get(exp, []):
            plabel[int(r[3]):int(r[4]) + 1] = int(r[2])
        idx = 0
        for s in range(0, T - W + 1, S):
            e = s + W
            win = sig[:, s:e]
            labs = plabel[s:e]
            annotated = labs[labs > 0]
            frac_ann = len(annotated) / W
            if len(annotated) == 0:
                policy, maj = "unlabeled", 0
            else:
                vals, cnts = np.unique(annotated, return_counts=True)
                dom = int(vals[np.argmax(cnts)])
                dom_frac = int(cnts.max()) / W
                if len(vals) == 1 and dom_frac == 1.0:
                    policy, maj = "pure", dom
                elif dom_frac >= MAJORITY_FRAC:
                    policy, maj = "majority", dom
                else:
                    policy, maj = "ambiguous", dom
            rows.append({
                "exp": exp, "subject": usr, "win_idx": idx,
                "start": int(s), "end": int(e),
                "label_policy": policy, "label_majority": maj,
                "n_distinct_labels": int(len(np.unique(annotated))),
                "annotated_frac": round(float(frac_ann), 4),
                "crosses_boundary": bool(len(np.unique(annotated)) > 1),
            })
            arrays.append(win)
            idx += 1

    df = pd.DataFrame(rows)
    x = np.stack(arrays).astype(np.float32)  # (N, 6, 128)
    np.savez_compressed(DATA / "windows.npz", x=x)
    df.to_parquet(DATA / "window_manifest.parquet", index=False)

    policy_counts = df["label_policy"].value_counts().to_dict()
    summary = {
        "windows": int(len(df)), "array_shape": list(x.shape),
        "stride": S, "window": W,
        "policy_counts": {k: int(v) for k, v in policy_counts.items()},
        "ambiguous_rate": round(float(policy_counts.get("ambiguous", 0)) / len(df), 4),
        "unlabeled_rate": round(float(policy_counts.get("unlabeled", 0)) / len(df), 4),
        "crosses_boundary_rate": round(float(df["crosses_boundary"].sum()) / len(df), 4),
        "subjects": int(df["subject"].nunique()),
        "recordings": int(df["exp"].nunique()),
        "note": "windows never span recordings; policy locked before G2",
    }
    (DATA / "build_windows_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
