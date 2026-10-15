#!/usr/bin/env python3
"""10_profile_inference.py — CPU latency BASELINE for the frozen backbone
(batch=1 and batch=32), to compare against G5 later. This is NOT a saving claim.
Outputs: data/g2_cpu_latency.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone import CHECKPOINTS, build_model, load_pure_k6  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "g2_cpu_latency.json"
WARMUP, RUNS = 200, 1000


def timed(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)  # ms
    return out


def stats(v: list[float]) -> dict:
    a = np.array(v)
    return {"mean_ms": round(float(a.mean()), 4),
            "p50_ms": round(float(np.percentile(a, 50)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "min_ms": round(float(a.min()), 4)}


def main() -> None:
    ck = torch.load(CHECKPOINTS / "backbone_best.pt", weights_only=False)
    model = build_model(ck["backbone_config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean, std = ck["mean"], ck["std"]
    xt = load_pure_k6()["test"][0]
    xn = np.ascontiguousarray(((xt - mean[None, :, None]) / std[None, :, None])
                              .astype(np.float32))

    x1 = torch.from_numpy(xn[:1])
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x1)
        lat1 = timed(lambda: model(x1), RUNS)

    x32 = torch.from_numpy(xn[1:33])
    with torch.no_grad():
        for _ in range(50):
            model(x32)
        lat32 = timed(lambda: model(x32), 200)

    out = {
        "scope": "BASELINE for G5 — full backbone forward only, no reuse, no cert",
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "checkpoint": "backbone_best.pt (seed44, selected on val macro-F1)",
        "warmup": WARMUP, "runs_batch1": RUNS, "runs_batch32": 200,
        "batch1": stats(lat1),
        "batch32": stats(lat32),
        "batch32_throughput_ips": round(32 / (np.percentile(lat32, 50) / 1000), 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
