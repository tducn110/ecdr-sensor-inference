#!/usr/bin/env python3
"""11_gate_a_prototype.py — Prototype empirical probe for Gate A (Certifiability)
and Gate B (Stream Replay) on Validation Set.

Step 0: Model verification & 2D equivalence for auto_LiRPA.
Step 1: IBP & CROWN certified-anchor rate p_c(k) across perturbation scales k.
Step 2: Benchmark build latency T_b (p50/p95).
Step 3: Replay Validation stream in chronological order to measure empirical hit rate r.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from backbone import DATA, CHECKPOINTS, load_pure_k6, build_model, apply_norm

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = DATA / "p0_verifier_probe.json"


class SmallCNN2D(nn.Module):
    """Numerically identical 2D replica of SmallCNN1D for auto_LiRPA compatibility."""

    def __init__(self, m1d: nn.Module):
        super().__init__()
        c = [32, 64, 64]
        k = [5, 3, 3]
        hidden = 128
        act = nn.ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(6, c[0], (1, k[0]), padding=(0, k[0] // 2)),
            nn.BatchNorm2d(c[0]), act(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(c[0], c[1], (1, k[1]), padding=(0, k[1] // 2)),
            nn.BatchNorm2d(c[1]), act(),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(c[1], c[2], (1, k[2]), padding=(0, k[2] // 2)),
            nn.BatchNorm2d(c[2]), act(),
        )
        flat = c[2] * (128 // 4)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, hidden), act(), nn.Linear(hidden, 6)
        )
        self.conv[0].weight.data = m1d.conv[0].weight.data.unsqueeze(2)
        self.conv[0].bias.data = m1d.conv[0].bias.data
        self.conv[1].weight.data = m1d.conv[1].weight.data
        self.conv[1].bias.data = m1d.conv[1].bias.data
        self.conv[1].running_mean.data = m1d.conv[1].running_mean.data
        self.conv[1].running_var.data = m1d.conv[1].running_var.data

        self.conv[4].weight.data = m1d.conv[4].weight.data.unsqueeze(2)
        self.conv[4].bias.data = m1d.conv[4].bias.data
        self.conv[5].weight.data = m1d.conv[5].weight.data
        self.conv[5].bias.data = m1d.conv[5].bias.data
        self.conv[5].running_mean.data = m1d.conv[5].running_mean.data
        self.conv[5].running_var.data = m1d.conv[5].running_var.data

        self.conv[8].weight.data = m1d.conv[8].weight.data.unsqueeze(2)
        self.conv[8].bias.data = m1d.conv[8].bias.data
        self.conv[9].weight.data = m1d.conv[9].weight.data
        self.conv[9].bias.data = m1d.conv[9].bias.data
        self.conv[9].running_mean.data = m1d.conv[9].running_mean.data
        self.conv[9].running_var.data = m1d.conv[9].running_var.data

        self.head[1].weight.data = m1d.head[1].weight.data
        self.head[1].bias.data = m1d.head[1].bias.data
        self.head[3].weight.data = m1d.head[3].weight.data
        self.head[3].bias.data = m1d.head[3].bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


def make_c_matrix(pred: int, num_classes: int = 6) -> torch.Tensor:
    other_classes = [j for j in range(num_classes) if j != pred]
    c = torch.zeros(1, num_classes - 1, num_classes)
    for idx, j in enumerate(other_classes):
        c[0, idx, pred] = 1.0
        c[0, idx, j] = -1.0
    return c


def run_prototype() -> dict:
    print("=== Step 0: Loading Checkpoint & Setting up Verifier ===")
    ck = torch.load(CHECKPOINTS / "backbone_best.pt", weights_only=False)
    m1d = build_model(ck["backbone_config"])
    m1d.load_state_dict(ck["state_dict"])
    m1d.eval()

    m2d = SmallCNN2D(m1d)
    m2d.eval()

    # Step 0 check: verify equivalence
    test_x = torch.randn(10, 6, 128)
    with torch.no_grad():
        eq_err = (m1d(test_x) - m2d(test_x.unsqueeze(2))).abs().max().item()
    print(f"Step 0 Verification: 1D vs 2D max error = {eq_err:.6e}")
    assert eq_err < 1e-5, f"Model equivalence check failed: {eq_err}"

    dummy = torch.zeros(1, 6, 1, 128)
    bounded_m = BoundedModule(m2d, dummy)

    # Load data
    data = load_pure_k6()
    x_val, y_val = data["val"]
    mean, std = ck["mean"], ck["std"]
    xn_val = apply_norm(x_val, mean, std)  # shape (N, 6, 128)
    n_val = len(xn_val)
    print(f"Validation pure K=6 samples: {n_val}")

    with torch.no_grad():
        val_logits = m1d(torch.from_numpy(xn_val)).numpy()
    val_preds = val_logits.argmax(axis=1)
    val_acc = float((val_preds == y_val).mean())
    print(f"Validation accuracy on pure K=6: {val_acc*100:.2f}%")

    k_levels = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]

    print("\n=== Step 1 & 2: Testing IBP & CROWN Bounds on Validation Sample ===")
    # Select stratified sample of 100 validation windows (correctly predicted)
    correct_indices = np.where(val_preds == y_val)[0]
    np.random.seed(42)
    sample_indices = np.random.choice(correct_indices, size=100, replace=False)

    results_step1 = {}

    # Warmup BoundedModule
    warmup_x = torch.from_numpy(xn_val[sample_indices[:1]]).unsqueeze(2)
    warmup_pred = val_preds[sample_indices[0]]
    ptb_w = PerturbationLpNorm(norm=float("inf"), eps=0.001)
    _ = bounded_m.compute_bounds(
        x=(BoundedTensor(warmup_x, ptb_w),),
        method="CROWN",
        C=make_c_matrix(warmup_pred),
    )

    for k in k_levels:
        ptb = PerturbationLpNorm(norm=float("inf"), eps=k)
        crown_certified = 0
        ibp_certified = 0
        crown_margins = []
        ibp_margins = []
        crown_latencies = []

        for idx in sample_indices:
            xi = torch.from_numpy(xn_val[idx : idx + 1]).unsqueeze(2)
            c_mat = make_c_matrix(val_preds[idx])
            b_tensor = BoundedTensor(xi, ptb)

            # IBP check
            lb_ibp, _ = bounded_m.compute_bounds(
                x=(b_tensor,), method="IBP", C=c_mat
            )
            min_ibp = float(lb_ibp.min().item())
            ibp_margins.append(min_ibp)
            if min_ibp > 0:
                ibp_certified += 1

            # CROWN check & latency benchmark
            t0 = time.perf_counter()
            lb_crown, _ = bounded_m.compute_bounds(
                x=(b_tensor,), method="CROWN", C=c_mat
            )
            t_ms = (time.perf_counter() - t0) * 1000.0
            crown_latencies.append(t_ms)
            min_crown = float(lb_crown.min().item())
            crown_margins.append(min_crown)
            if min_crown > 0:
                crown_certified += 1

        results_step1[str(k)] = {
            "k": k,
            "ibp_p_c": ibp_certified / len(sample_indices),
            "crown_p_c": crown_certified / len(sample_indices),
            "crown_min_margin_med": float(np.median(crown_margins)),
            "ibp_min_margin_med": float(np.median(ibp_margins)),
            "crown_t_build_ms_p50": float(np.percentile(crown_latencies, 50)),
            "crown_t_build_ms_p95": float(np.percentile(crown_latencies, 95)),
            "crown_t_build_ms_mean": float(np.mean(crown_latencies)),
        }
        print(
            f"k = {k:0.4f} | IBP p_c = {ibp_certified}% | "
            f"CROWN p_c = {crown_certified}% | "
            f"CROWN med margin = {np.median(crown_margins):+.2f} | "
            f"CROWN T_b = {np.median(crown_latencies):.1f} ms"
        )

    print("\n=== Step 3: Stream Replay & Hit Rate Simulation on Val Stream ===")
    # Load metadata to reconstruct chronological order
    x_all = np.load(DATA / "windows.npz")["x"].astype(np.float32)
    man = pd.read_parquet(DATA / "window_manifest.parquet")
    spl = pd.read_parquet(DATA / "windows_with_split.parquet")
    df = man.merge(
        spl[["exp", "win_idx", "split"]], on=["exp", "win_idx"], how="left"
    )

    val_df = df[df["split"] == "val"].copy().reset_index()
    val_df = val_df.sort_values(["exp", "win_idx"]).reset_index(drop=True)

    # Filter to consecutive sequences in validation
    val_pure_consec = (
        (val_df["label_policy"] == "pure")
        & val_df["label_majority"].between(1, 6)
    )
    val_pure_df = val_df[val_pure_consec].copy()

    # Calculate actual distance to previous window
    val_pure_indices = val_pure_df["index"].values
    val_x_raw = x_all[val_pure_indices]
    val_xn = apply_norm(val_x_raw, mean, std)

    # Track hit rates under different box sizes k
    # Condition for hit: |x_{t} - x_{anchor}|_inf <= k AND anchor is certified for k
    stream_results = {}
    for k in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.28]:
        # Simple cache simulation: 1-entry LRU anchor
        hits = 0
        total_queries = 0
        anchor_builds = 0
        anchor_hits_list = []

        curr_anchor = None
        curr_anchor_exp = None
        curr_hits = 0

        for i in range(len(val_pure_df)):
            exp_id = val_pure_df["exp"].iloc[i]
            win_idx = val_pure_df["win_idx"].iloc[i]
            x_curr = val_xn[i]

            if curr_anchor is None or exp_id != curr_anchor_exp:
                # Cache miss / new recording: build anchor
                if curr_anchor is not None:
                    anchor_hits_list.append(curr_hits)
                curr_anchor = x_curr
                curr_anchor_exp = exp_id
                curr_hits = 0
                anchor_builds += 1
            else:
                total_queries += 1
                dist = np.max(np.abs(x_curr - curr_anchor))
                if dist <= k:
                    hits += 1
                    curr_hits += 1
                else:
                    # Miss: rebuild anchor
                    anchor_hits_list.append(curr_hits)
                    curr_anchor = x_curr
                    curr_hits = 0
                    anchor_builds += 1

        if curr_anchor is not None:
            anchor_hits_list.append(curr_hits)

        hit_rate = hits / max(1, (hits + anchor_builds))
        avg_hits_per_anchor = float(np.mean(anchor_hits_list)) if anchor_hits_list else 0.0
        max_hits_per_anchor = int(np.max(anchor_hits_list)) if anchor_hits_list else 0

        stream_results[str(k)] = {
            "k": k,
            "raw_box_hit_rate": hit_rate,
            "total_windows": len(val_pure_df),
            "anchor_builds": anchor_builds,
            "hits": hits,
            "avg_hits_per_anchor": avg_hits_per_anchor,
            "max_hits_per_anchor": max_hits_per_anchor,
        }
        print(
            f"Box k = {k:0.4f} | Raw Stream Hit Rate = {hit_rate*100:5.2f}% | "
            f"Avg Hits/Anchor = {avg_hits_per_anchor:4.2f} | Max Hits = {max_hits_per_anchor}"
        )

    summary = {
        "step0": {"val_samples": n_val, "val_acc": val_acc, "eq_err": eq_err},
        "step1_gate_a": results_step1,
        "step3_gate_b": stream_results,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved probe results to {OUT_JSON}")
    return summary


if __name__ == "__main__":
    run_prototype()
