#!/usr/bin/env python3
"""ecdr_stream_replay.py — Chronological Streaming Replay Benchmark.

Executes streaming replay across test stream (5053 windows) or val stream (2422 windows)
with shadow-reference agreement, trusted-anchor lifecycle, and complete cost instrumentation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.data import FULL_STREAM_DENOMINATORS, load_full_stream
from ecdr.model import ECDRConv1DNet, OperationalSequentialNumpyCNN
from ecdr.preprocessing import PreprocessingContract, fit_normalization_stats
from ecdr.replay import StreamingReplayEngine

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "ecdr_k12" / "ecdr_k12_best.pt"
OUT_REPLAY_JSON = ROOT / "data" / "ecdr_replay_summary.json"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_replay(
    split: str = "test",
    domain_type: str = "diagonal_box",
    budget: float = 0.001,
    checkpoint_path: Optional[Path] = None,
    initialized_demo: bool = False,
    max_windows: Optional[int] = None,
    out_json: Path = OUT_REPLAY_JSON,
    measure_baseline: bool = True,
) -> dict:
    print("=" * 80)
    print(f"ECDR STREAMING REPLAY BENCHMARK — Split: {split.upper()} | Domain: {domain_type.upper()}")
    print("=" * 80)

    if max_windows is not None and max_windows <= 0:
        raise ValueError(f"max_windows must be strictly positive (> 0), got {max_windows}")

    # Provenance guard
    checkpoint_sha256 = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Requested checkpoint does not exist: {checkpoint_path}. "
                "Specify a valid checkpoint path or use --initialized-demo."
            )
        trained_status = "trained_checkpoint"
        checkpoint_sha256 = compute_file_sha256(checkpoint_path)
    else:
        if not initialized_demo:
            raise ValueError(
                "No checkpoint provided. You must explicitly pass --initialized-demo "
                "to execute an untrained seeded reference demo."
            )
        trained_status = "untrained_initialized_demo"

    # 1. Load full stream (no label filtering!)
    x_stream, df_stream = load_full_stream(split)
    full_denominator = FULL_STREAM_DENOMINATORS[split]
    actual_count = len(x_stream)

    if max_windows is not None and max_windows < actual_count:
        scope = "prefix"
        x_stream = x_stream[:max_windows]
        df_stream = df_stream.iloc[:max_windows].reset_index(drop=True)
        print(f">> SCOPE: Prefix evaluation on first {max_windows} windows (NOT full stream).")
    else:
        scope = "full_stream"
        print(f">> SCOPE: Full stream evaluation ({actual_count} windows matching expected denominator {full_denominator}).")

    # 2. Setup model and preprocessing
    if trained_status == "trained_checkpoint":
        assert checkpoint_path is not None
        ck = torch.load(checkpoint_path, weights_only=False)
        torch_net = ECDRConv1DNet(num_classes=12)
        torch_net.load_state_dict(ck["model_state_dict"])
        prep = PreprocessingContract.create(
            mean=ck["prep_mean"], std=ck["prep_std"], num_channels=6, window_length=128
        )
        print(f"Loaded trained checkpoint: {checkpoint_path} (SHA-256: {checkpoint_sha256[:12]}...)")
    else:
        print("Using initialized reference model (--initialized-demo enabled).")
        torch.manual_seed(42)
        torch_net = ECDRConv1DNet(num_classes=12)
        from ecdr.data import load_full_stream as load_tr_stream
        x_tr, _ = load_tr_stream("train")
        prep = fit_normalization_stats(x_tr)

    numpy_net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)
    print(f"Bound Operational Reference model_id: {numpy_net.model_id}")
    print(f"Bound Preprocessing Contract prep_id: {prep.prep_id}")

    # 3. Setup Replay Engine
    engine = StreamingReplayEngine(
        net=numpy_net,
        prep=prep,
        domain_type=domain_type,
        budget_norm=budget,
        wave_radius=budget,
        use_incremental_reference=True,
        shadow_reference=True,
    )

    print(f"\nStarting chronological replay (budget={budget})...")
    summary = engine.run_stream(x_stream, df_stream, measure_baseline=measure_baseline)

    # 4. Attach comprehensive provenance metadata
    summary["provenance"] = {
        "trained_status": trained_status,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_sha256,
        "model_id": numpy_net.model_id,
        "prep_id": prep.prep_id,
        "split": split,
        "scope": scope,
        "expected_full_denominator": full_denominator,
        "actual_processed_count": len(x_stream),
        "domain_type": domain_type,
        "budget": budget,
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        },
    }

    print("\n" + "=" * 80)
    print("REPLAY EXECUTION SUMMARY:")
    print("-" * 80)
    qc = summary["query_counts"]
    sa = summary["shadow_agreement"]
    lat = summary["latencies_ms"]
    mac = summary["computation_mac"]

    print(f"  Status / Scope:          {trained_status} | {scope}")
    print(f"  Total Queries Processed: {qc['total_queries']} / {full_denominator}")
    print(f"  Hits (Certified Reuse):  {qc['hits']} ({qc['hit_rate_pct']:.2f}%)")
    print(f"  Misses / Fallbacks:      {qc['misses']}")
    print(f"  Recording Boundary Resets: {qc['recording_resets']}")
    print(f"  Sequence Gap Resets:     {qc['sequence_gap_resets']}")
    print(f"  Certificates Built:      {qc['build_successes']} / {qc['build_attempts']} attempts")
    print("-" * 80)
    if sa["checked_accepted_queries"] > 0:
        print(
            f"  Shadow-Reference Agreement: {sa['agreement_rate']*100:.2f}% "
            f"({sa['agreed_queries']}/{sa['checked_accepted_queries']} checked accepted queries)"
        )
        print(f"  Discrepancies Observed:     {sa['discrepancies']}")
    else:
        print("  Shadow-Reference Agreement: None (0 checked accepted queries)")
    print("-" * 80)
    print(f"  Mean Operational Latency: {lat['operational_query_latency']['mean_ms']:.3f} ms")
    print(f"  P50 Operational Latency:  {lat['operational_query_latency']['p50_ms']:.3f} ms")
    print(f"  P95 Operational Latency:  {lat['operational_query_latency']['p95_ms']:.3f} ms")
    print(f"  Theoretical MAC Saving:   {mac['theoretical_mac_saving_pct']:.2f}%")
    if "empirical_speedup" in summary:
        sp = summary["empirical_speedup"]
        print(
            f"  Measured Reference Baseline: {sp['measured_baseline_mean_ms']:.3f} ms | "
            f"Speedup: {sp['speedup_factor']:.2f}x"
        )
    print("=" * 80)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved replay benchmark artifact to: {out_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ECDR streaming replay benchmark")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--domain",
        type=str,
        default="diagonal_box",
        choices=["diagonal_box", "wave_domain"],
        help="Perturbation domain candidate",
    )
    parser.add_argument("--budget", type=float, default=0.001, help="Perturbation budget radius")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained checkpoint (ecdr_k12_best.pt)",
    )
    parser.add_argument(
        "--initialized-demo",
        action="store_true",
        help="Explicitly permit running seeded untrained demo if no checkpoint is available",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional prefix window count limit (marks scope as prefix)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=OUT_REPLAY_JSON,
        help="Output destination path for replay JSON summary",
    )
    args = parser.parse_args()

    run_replay(
        split=args.split,
        domain_type=args.domain,
        budget=args.budget,
        checkpoint_path=args.checkpoint,
        initialized_demo=args.initialized_demo,
        max_windows=args.max_windows,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
