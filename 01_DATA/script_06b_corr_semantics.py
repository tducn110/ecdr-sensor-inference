#!/usr/bin/env python3
"""06b_corr_semantics.py — what does "lag-1 correlation" in event_audit.json mean?

G1 F2 reported per-channel lag-1 Pearson ~0.14 computed by 05 over FLATTENED
consecutive windows: corrcoef(win_k, win_{k+1}) pairs raw sample t with t+64
(1.28 s) because windows overlap 50%. This script measures three distinct
quantities so they are not conflated:

  1. raw_sample_lag1    — corr(x[t], x[t+1]) at 50 Hz (20 ms), true sample lag
  2. raw_offset_lag1    — corr(x[t], x[t+64]) at 1.28 s (what window lag-1
                          actually measures under positional pairing)
  3. window_feat_lag1   — corr of per-window channel means (window-level feature
                          dependence between consecutive windows)

Also re-measures the 05 quantity (window_positional_lag1) for consistency.
Produces data/corr_semantics.json. No model, no certificate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    x = np.load(DATA / "windows.npz")["x"].astype(np.float64)  # (N,6,128)
    man = pd.read_parquet(DATA / "window_manifest.parquet")
    off64 = 64

    out: dict = {"per_channel": {}, "method": {
        "raw_sample_lag1": "pooled corr(x[t], x[t+1]) within recording, 50Hz",
        "raw_offset_lag1": "pooled corr(x[t], x[t+64]) = 1.28s within recording",
        "window_positional_lag1": ("mean over k of corrcoef(flatten(win_k), "
                                   "flatten(win_k+1)); pairs t with t+64 "
                                   "(same as 05_audit_temporal_events lag1)"),
        "window_feat_lag1": "corr of per-window channel-mean series, lag 1 window",
        "averaging": "arithmetic mean over recordings (nan-safe)",
    }}

    for ci, cname in enumerate(CHANNELS):
        m_sample, m_offset, m_pos, m_feat = [], [], [], []
        for exp, g in man.groupby("exp"):
            g = g.sort_values("win_idx")
            rows = x[g.index.to_numpy(), ci, :]  # (n_win, 128)
            flat = rows.ravel()
            # 1. true sample lag at 50 Hz
            m_sample.append(safe_corr(flat[:-1], flat[1:]))
            # 2. offset of 64 samples = 1.28 s (positions paired as in window lag1)
            m_offset.append(safe_corr(flat[:-off64], flat[off64:]))
            # 3. replica of 05 window-lag1 (positional pairing)
            if len(rows) > 1:
                m_pos.append(np.nanmean([
                    safe_corr(rows[k], rows[k + 1]) for k in range(len(rows) - 1)]))
            # 4. window-level feature dependence
            feat = rows.mean(axis=1)
            if len(feat) > 1:
                m_feat.append(safe_corr(feat[:-1], feat[1:]))
        out["per_channel"][cname] = {
            "raw_sample_lag1": round(float(np.nanmean(m_sample)), 4),
            "raw_offset_lag1": round(float(np.nanmean(m_offset)), 4),
            "window_positional_lag1": round(float(np.nanmean(m_pos)), 4),
            "window_feat_lag1": round(float(np.nanmean(m_feat)), 4),
        }

    (DATA / "corr_semantics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
