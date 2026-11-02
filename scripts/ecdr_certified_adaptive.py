#!/usr/bin/env python3
"""Benchmark margin/Lipschitz certified adaptive reuse on the K=12 test stream."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.certified_adaptive import CertifiedAdaptiveCache
from ecdr.data import FULL_STREAM_DENOMINATORS, load_full_stream
from ecdr.model import ECDRConv1DNet, OperationalSequentialNumpyCNN
from ecdr.preprocessing import PreprocessingContract


DEFAULT_CHECKPOINT = ROOT / "output" / "ecdr" / "smoke" / "checkpoints_seed20261031" / "ecdr_k12_best.pt"
DEFAULT_OUT = ROOT / "output" / "ecdr" / "certified_adaptive_test.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run(checkpoint: Path, out_json: Path, split: str = "test") -> dict:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    x_stream, manifest = load_full_stream(split)
    if len(x_stream) != FULL_STREAM_DENOMINATORS[split]:
        raise ValueError("full-stream denominator mismatch")

    ck = torch.load(checkpoint, weights_only=False)
    torch_net = ECDRConv1DNet(num_classes=12)
    torch_net.load_state_dict(ck["model_state_dict"])
    prep = PreprocessingContract.create(
        mean=ck["prep_mean"], std=ck["prep_std"], num_channels=6, window_length=128
    )
    net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)

    baseline_predictions = []
    baseline_ms = []
    normalized = []
    for raw in x_stream:
        z = prep.normalize(raw)
        normalized.append(z)
        t0 = time.perf_counter()
        baseline_predictions.append(int(net.forward(z)["pred_class"]))
        baseline_ms.append((time.perf_counter() - t0) * 1000.0)

    cache = CertifiedAdaptiveCache(net)
    predictions = []
    hits = 0
    radii = []
    adaptive_ms = []
    for idx, (z, (_, row)) in enumerate(zip(normalized, manifest.iterrows())):
        t0 = time.perf_counter()
        result = cache.predict(z, recording_id=str(row["exp"]), step_idx=int(row["win_idx"]))
        adaptive_ms.append((time.perf_counter() - t0) * 1000.0)
        predictions.append(int(result["pred_class"]))
        hits += int(result["is_hit"])
        radius = float(result["radius"])
        if np.isfinite(radius):
            radii.append(radius)

    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    disagreement = int(np.sum(predictions != baseline_predictions))
    valid = manifest["label_policy"].isin(["pure", "majority"]) & manifest["label_majority"].between(1, 12)
    labels = manifest.loc[valid, "label_majority"].to_numpy(dtype=np.int64) - 1
    idx_valid = np.flatnonzero(valid.to_numpy())
    adaptive_acc = float(np.mean(predictions[idx_valid] == labels)) if len(labels) else None
    baseline_acc = float(np.mean(baseline_predictions[idx_valid] == labels)) if len(labels) else None

    result = {
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_id": net.model_id,
            "prep_id": prep.prep_id,
            "model_boundary": "ECDRConv1DNet K=12; not SmallCNN1D K=6",
            "split": split,
            "scope": "full_stream",
        },
        "query_counts": {
            "total_queries": len(x_stream),
            "hits": hits,
            "hit_rate_pct": 100.0 * hits / len(x_stream),
            "misses": len(x_stream) - hits,
            "disagreements_vs_full_cnn": disagreement,
            "disagreement_rate_pct": 100.0 * disagreement / len(x_stream),
        },
        "quality_on_valid_labels": {
            "valid_windows": int(len(labels)),
            "full_cnn_accuracy": baseline_acc,
            "certified_adaptive_accuracy": adaptive_acc,
            "accuracy_delta_points": (100.0 * (adaptive_acc - baseline_acc)) if adaptive_acc is not None else None,
        },
        "latency_ms": {
            "full_cnn_mean": float(np.mean(baseline_ms)),
            "certified_adaptive_mean": float(np.mean(adaptive_ms)),
            "speedup_factor": float(np.mean(baseline_ms) / np.mean(adaptive_ms)),
        },
        "certified_radius": {
            "finite_anchor_count": len(radii),
            "min": float(np.min(radii)) if radii else None,
            "median": float(np.median(radii)) if radii else None,
            "max": float(np.max(radii)) if radii else None,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved certified adaptive benchmark to: {out_json}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.checkpoint, args.out_json, args.split)


if __name__ == "__main__":
    main()

