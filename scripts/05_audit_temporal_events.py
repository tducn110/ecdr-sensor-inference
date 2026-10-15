#!/usr/bin/env python3
"""05_audit_temporal_events.py — Q1 temporal distances, Q2 candidate event-region
coverage, correlation structure (diagonal box vs correlated set). Budgets are FIT
ON TRAIN SUBJECTS ONLY, then evaluated on val/test. No CROWN, no backbone.

FIX 2026-10-15 (found by G1 re-verification): win_idx is PER-RECORDING (restarts
at 0), but x rows are in global build order. The pre-fix version indexed x with
win_idx, so 98.2% of pairs read signals from exp01's windows while keeping labels
of the current recording -- all Q1/correlation/coverage numbers were wrong.
Signals are now indexed via df["row_id"] (global position in windows.npz)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
Q_LEVELS = (0.50, 0.75, 0.90)
PCA_VAR = 0.90


def quantiles(v: np.ndarray) -> dict:
    if len(v) == 0:
        return {}
    qs = np.quantile(v, [0.5, 0.9, 0.95, 0.99])
    return {k: round(float(x), 5) for k, x in
            zip(("p50", "p90", "p95", "p99"), qs)}


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    x = np.load(DATA / "windows.npz")["x"].astype(np.float32)
    man = pd.read_parquet(DATA / "window_manifest.parquet")
    spl = pd.read_parquet(DATA / "windows_with_split.parquet")
    df = man.merge(spl[["exp", "win_idx", "split"]], on=["exp", "win_idx"], how="left")
    # global row id == position in windows.npz (03 appends rows in this order)
    assert len(x) == len(df), "windows.npz and window_manifest out of sync"
    df["row_id"] = np.arange(len(df))


    # ---- consecutive pairs within the same recording (never across) ----
    pair_rows = []
    for exp, g in df.groupby("exp"):
        g = g.sort_values("win_idx")
        idx = g.set_index("win_idx")
        order = g["win_idx"].tolist()
        for a, b in zip(order[:-1], order[1:]):
            if b != a + 1:
                continue
            ra, rb = idx.loc[a], idx.loc[b]
            pair_rows.append({
                "exp": exp, "ri": int(ra["row_id"]), "rj": int(rb["row_id"]),
                "split": ra["split"],
                "same_activity": bool(ra["label_majority"] == rb["label_majority"]
                                      and ra["label_policy"] == "pure"
                                      and rb["label_policy"] == "pure"),
                "label_change": bool(ra["label_majority"] != rb["label_majority"]),
            })
    pairs = pd.DataFrame(pair_rows)
    di = x[pairs["ri"].to_numpy()]
    dj = x[pairs["rj"].to_numpy()]

    delta = (dj - di).reshape(len(pairs), -1)
    l2 = np.linalg.norm(delta, axis=1)
    linf = np.abs(delta).max(axis=1)
    pairs["l2"], pairs["linf"] = l2, linf

    # ---- Q1: distance distributions by pair type ----
    q1 = {}
    for name, m in (("same_activity", pairs["same_activity"].to_numpy()),
                    ("label_change", pairs["label_change"].to_numpy())):
        q1[name] = {"n": int(m.sum()),
                    "l2": quantiles(l2[m]), "linf": quantiles(linf[m])}

    # ---- temporal correlation per channel at lags (in windows) ----
    corr = {}
    channel_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for lag in (1, 2, 4, 8):
        cs = {}
        for ci, cname in enumerate(channel_names):
            co = []
            for exp, g in df.groupby("exp"):
                order = g.sort_values("win_idx")["row_id"].to_numpy()

                arr = x[order, :, :]
                for k in range(len(arr) - lag):
                    a = arr[k, ci].ravel()
                    b = arr[k + lag, ci].ravel()
                    co.append(float(np.corrcoef(a, b)[0, 1]))
            cs[cname] = round(float(np.nanmean(co)), 4)
        corr[f"lag{lag}"] = cs

    # ---- Q2: diagonal box budgets fit on TRAIN only ----
    tr = (pairs["split"] == "train").to_numpy()
    tr_same = tr & pairs["same_activity"].to_numpy()
    per_dim_q = {q: np.quantile(np.abs(delta[tr_same]), q, axis=0) for q in Q_LEVELS}
    names = {0.50: "stable", 0.75: "moderate", 0.90: "wide"}
    coverage = {}
    for q, budget in per_dim_q.items():
        ok = (np.abs(delta) <= budget).all(axis=1)
        cov = {}
        for split in ("train", "val", "test"):
            m = (pairs["split"] == split).to_numpy()
            cov[split] = round(float(ok[m].mean()), 4) if m.any() else None
        mt = (pairs["split"] == "test").to_numpy()
        msame = pairs["same_activity"].to_numpy()
        mchg = pairs["label_change"].to_numpy()
        cov["test_same"] = round(float(ok[mt & msame].mean()), 4)
        cov["test_change"] = round(float(ok[mt & mchg].mean()), 4)
        coverage[names[q]] = cov
    stable_ok = (np.abs(delta) <= per_dim_q[0.50]).all(axis=1)
    small_but_change = float(
        (stable_ok & pairs["label_change"].to_numpy()).sum()) / max(int(stable_ok.sum()), 1)

    # ---- correlated-set candidate: PCA + per-component q90 (train fit) ----
    rng = np.random.default_rng(0)
    tr_idx = np.where(tr)[0]
    sub = rng.choice(tr_idx, size=min(6000, len(tr_idx)), replace=False)
    mu = delta[tr_idx].mean(axis=0)
    u, s, vt = np.linalg.svd(delta[sub] - mu, full_matrices=False)
    evr = (s ** 2) / (s ** 2).sum()
    k = int(np.searchsorted(np.cumsum(evr), PCA_VAR) + 1)
    comps = vt[:k]
    scores = (delta - mu) @ comps.T
    resid = np.linalg.norm((delta - mu) - scores @ comps, axis=1)
    thr_score = np.quantile(np.abs(scores[tr_same]), 0.90, axis=0)
    thr_resid = float(np.quantile(resid[tr_same], 0.90))
    ok_corr = ((np.abs(scores) <= thr_score).all(axis=1) & (resid <= thr_resid))
    box_ok90 = (np.abs(delta) <= per_dim_q[0.90]).all(axis=1)
    corr_cov, box_cov = {}, {}
    for split in ("train", "val", "test"):
        m = (pairs["split"] == split).to_numpy()
        corr_cov[split] = round(float(ok_corr[m].mean()), 4) if m.any() else None
        box_cov[split] = round(float(box_ok90[m].mean()), 4) if m.any() else None

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    m_same = pairs["same_activity"].to_numpy()
    m_chg = pairs["label_change"].to_numpy()
    ax[0].hist(l2[m_same], bins=60, alpha=0.6, label="same activity")
    ax[0].hist(l2[m_chg], bins=60, alpha=0.6, label="label change")
    ax[0].set_xlabel("L2 delta consecutive windows"); ax[0].legend()
    ax[1].hist(linf[m_same], bins=60, alpha=0.6, label="same activity")
    ax[1].hist(linf[m_chg], bins=60, alpha=0.6, label="label change")
    ax[1].set_xlabel("Linf delta"); ax[1].legend()
    fig.tight_layout(); fig.savefig(PLOTS / "delta_distributions.png", dpi=120)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(coverage.keys())
    ax.bar(labels, [coverage[n]["test"] for n in labels])
    ax.set_ylabel("coverage (test)"); ax.set_ylim(0, 1)
    ax.set_title("diagonal event-box coverage (fit on train)")
    fig.tight_layout(); fig.savefig(PLOTS / "event_coverage.png", dpi=120)

    # ---- summary ----
    out = {
        "pairs_total": int(len(pairs)),
        "pairs_by_split": {k: int(v) for k, v in pairs["split"].value_counts().items()},
        "q1_distances": q1,
        "temporal_correlation_per_channel": corr,
        "diagonal_box_coverage_fit_on_train": coverage,
        "stable_box_but_label_changed_frac": round(small_but_change, 4),
        "correlated_set_pca": {
            "components_k": k, "variance_explained": round(float(evr[:k].sum()), 4),
            "thresholds_fit": "train same-activity q90 scores + q90 residual",
            "coverage": corr_cov,
            "diagonal_p50_box_coverage": {
                s: round(float(stable_ok[(pairs['split'] == s).to_numpy()].mean()), 4)
                for s in ("train", "val", "test")},
            "diagonal_p90_box_coverage": box_cov,
        },
        "decision_note": ("GO if diagonal box covers a meaningful fraction of "
                          "same-activity pairs while label_change pairs lie mostly "
                          "outside; NO-GO if coverage ~0 or stable box hits change"),
    }
    (DATA / "event_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

