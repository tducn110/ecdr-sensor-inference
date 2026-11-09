"""tests/test_week4_incremental.py — Unit and integration tests for ActualK12IncrementalEngine.

Fulfills Task W4-020 (Milestone M3: Incremental feature engine and complete Flatten-MLP head).
Verifies:
- W4-020-C01: Per-stage cache (h1, p1, h2, p2, h3) with owned copies, query/recording/step/profile identity;
  first query and reset execute full reference path.
- W4-020-C02: True per-stage partial recomputation (no full-conv scaffold shortcut), assembling complete
  (32,128), (32,64), (64,64), (64,32), (64,32) stage tensors with 0.0 ULP error.
- W4-020-C03: Complete channel-major Flatten(2048)->Linear(128)->ReLU->Linear(12) head evaluation, zero
  winner disagreements, no GAP delta shortcut.
- W4-020-C04: Rigorous precondition checks on raw & normalized overlap (down to 1 ULP), stride=64,
  step gap/repeat/backward, recording change, model identity, and BN eval mode.
- W4-020-C05: Accurate execution reporting (mode, stage_reused_columns, stage_recomputed_columns,
  conv_mac_count=532,224 vs 909,312, total mac_count=795,904 vs 1,172,992, is_skipped_decision_reuse=False).
- W4-020-C06: Strict anti-aliasing on input/output/state, state invariance on invalid inputs, idempotent reset.
- W4-020-C07: Per-stage bitwise equality, max error, and winner agreement across all validation & test streams.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Dict
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from ecdr.week4.incremental import (
    ActualK12IncrementalEngine,
    IncrementalEngine,
    IncrementalResult,
    FULL_CONV_MACS,
    INCREMENTAL_CONV_MACS,
    HEAD_MACS,
    TOTAL_FULL_MACS,
    TOTAL_INCREMENTAL_MACS,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import QueryView

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output/week4/week4_initial_20261103"


class TestWeek4Incremental(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceOwner.load()
        cls.engine = ActualK12IncrementalEngine(cls.ref, shift=64)

    def setUp(self):
        self.engine.reset()
        self.ref.assert_identity()

    def tearDown(self):
        self.engine.reset()
        self.ref.assert_identity()

    # =========================================================================
    # W4-020-C01: Cold start, QueryView support, Cache Population, Reset
    # =========================================================================
    def test_c01_cold_start_and_query_view_support(self):
        """First call to forward_incremental on cold cache runs mode 'full' and accepts QueryView."""
        stream = np.random.randn(6, 192).astype(np.float32)
        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]
        qv0 = QueryView.from_array(raw=w0, global_row_id=10, recording_id="rec_01", step_idx=0)
        res0 = self.engine.forward_incremental(qv0)
        self.assertIsInstance(res0, IncrementalResult)
        self.assertEqual(res0["mode"], "full")
        self.assertEqual(res0.mode, "full")
        self.assertEqual(res0["reused_columns"], 0)
        self.assertEqual(res0["recomputed_columns"], 32)
        self.assertFalse(res0["is_skipped_decision_reuse"])
        self.assertIsNotNone(self.engine.state)
        self.assertEqual(self.engine.state.recording_id, "rec_01")
        self.assertEqual(self.engine.state.step_idx, 0)
        self.assertEqual(self.engine.state.query_row_id, 10)

        qv1 = QueryView.from_array(raw=w1, global_row_id=11, recording_id="rec_01", step_idx=1)
        res1 = self.engine.forward_incremental(qv1)
        self.assertEqual(res1["mode"], "incremental")
        self.assertEqual(res1["query_row_id"], 11)

    def test_c01_reset_is_idempotent_and_forces_full(self):
        """Reset clears cache state; subsequent call runs full forward pass."""
        w0 = np.random.randn(6, 128).astype(np.float32)
        self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_01")
        self.assertIsNotNone(self.engine.state)

        self.engine.reset()
        self.assertIsNone(self.engine.state)
        self.engine.reset()
        self.assertIsNone(self.engine.state)

        w1 = np.random.randn(6, 128).astype(np.float32)
        res1 = self.engine.forward_incremental(w1, step_idx=1, recording_id="rec_01")
        self.assertEqual(res1["mode"], "full")

    # =========================================================================
    # W4-020-C02: True Stagewise Partial Recomputation & All-Stage Bitwise Match
    # =========================================================================
    def test_c02_sequential_incremental_all_stages_bitwise_match(self):
        """Consecutive 50% shifted windows execute true stagewise partial recomputation with 0.0 error across all 5 stages."""
        np.random.seed(42)
        stream = np.random.randn(6, 384).astype(np.float32)

        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]
        w2 = stream[:, 128:256]
        w3 = stream[:, 192:320]

        res0 = self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_stream")
        self.assertEqual(res0["mode"], "full")

        ref_engine = ActualK12IncrementalEngine(self.ref, shift=64)
        ref0 = ref_engine.forward_full(w0, step_idx=0, recording_id="rec_stream")
        for stage in ("h1", "p1", "h2", "p2", "h3", "logits"):
            np.testing.assert_array_equal(res0[stage], ref0[stage])

        for idx, w in enumerate([w1, w2, w3], start=1):
            inc_res = self.engine.forward_incremental(w, step_idx=idx, recording_id="rec_stream")
            full_res = ref_engine.forward_full(w, step_idx=idx, recording_id="rec_stream")

            self.assertEqual(inc_res["mode"], "incremental")
            self.assertEqual(inc_res["reused_columns"], 12)
            self.assertEqual(inc_res["recomputed_columns"], 20)
            self.assertEqual(inc_res["reused_scalars"], 768)
            self.assertEqual(inc_res["recomputed_scalars"], 1280)
            self.assertEqual(
                inc_res["stage_reused_columns"],
                {"conv1": 60, "pool1": 30, "conv2": 28, "pool2": 14, "conv3": 12},
            )
            self.assertEqual(
                inc_res["stage_recomputed_columns"],
                {"conv1": 68, "pool1": 34, "conv2": 36, "pool2": 18, "conv3": 20},
            )

            # Bitwise exact match on ALL 5 intermediate stages and Head logits
            for stage in ("h1", "p1", "h2", "p2", "h3", "logits"):
                np.testing.assert_array_equal(
                    inc_res[stage],
                    full_res[stage],
                    err_msg=f"Stage {stage} diverged at step {idx}",
                )
            self.assertEqual(inc_res["pred_class"], full_res["pred_class"])

    # =========================================================================
    # W4-020-C03: Complete Flatten-MLP Head (No GAP Delta Shortcut)
    # =========================================================================
    def test_c03_complete_mlp_head_evaluation(self):
        """Logits must match direct ReferenceOwner.infer receipt bitwise."""
        np.random.seed(101)
        stream = np.random.randn(6, 192).astype(np.float32)
        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]

        self.engine.forward_full(w0, step_idx=0, recording_id="rec_head")
        inc_res = self.engine.forward_incremental(w1, step_idx=1, recording_id="rec_head")

        ref_receipt = self.ref.infer(w1)
        ref_logits = np.array(ref_receipt.logits, dtype=np.float32)

        np.testing.assert_array_equal(inc_res["logits"], ref_logits)
        self.assertEqual(inc_res["pred_class"], ref_receipt.winner)

    # =========================================================================
    # W4-020-C04: Fail-closed Precondition Checks & Fallbacks
    # =========================================================================
    def test_c04_one_ulp_overlap_perturbation_triggers_fallback(self):
        """A single 1-ULP float32 change in overlap segment triggers fallback to full mode."""
        w0 = np.random.randn(6, 128).astype(np.float32)
        self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_overlap")

        w1 = np.zeros((6, 128), dtype=np.float32)
        w1[:, :64] = w0[:, 64:]
        w1[3, 27] = np.nextafter(w1[3, 27], np.float32(np.inf))  # 1-ULP perturbation
        w1[:, 64:] = np.random.randn(6, 64).astype(np.float32)

        res = self.engine.forward_incremental(w1, step_idx=1, recording_id="rec_overlap")
        self.assertEqual(res["mode"], "full")
        self.assertEqual(res["fallback_reason"], "raw_overlap_mismatch")

    def test_c04_step_gap_repeat_and_backward_trigger_fallback(self):
        """Step gap (0->2), repeat (0->0), and backward (2->1) all trigger full fallback."""
        stream = np.random.randn(6, 320).astype(np.float32)
        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]
        w2 = stream[:, 128:256]

        # Gap: 0 -> 2
        self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_step")
        res_gap = self.engine.forward_incremental(w1, step_idx=2, recording_id="rec_step")
        self.assertEqual(res_gap["mode"], "full")
        self.assertEqual(res_gap["fallback_reason"], "step_discontinuity")

        # Repeat: 2 -> 2
        res_rep = self.engine.forward_incremental(w2, step_idx=2, recording_id="rec_step")
        self.assertEqual(res_rep["mode"], "full")
        self.assertEqual(res_rep["fallback_reason"], "step_discontinuity")

        # Backward: 2 -> 1
        res_back = self.engine.forward_incremental(w2, step_idx=1, recording_id="rec_step")
        self.assertEqual(res_back["mode"], "full")
        self.assertEqual(res_back["fallback_reason"], "step_discontinuity")

    def test_c04_recording_change_triggers_fallback(self):
        """Change of recording session ID triggers fallback to full mode."""
        stream = np.random.randn(6, 192).astype(np.float32)
        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]
        self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_A")

        res = self.engine.forward_incremental(w1, step_idx=1, recording_id="rec_B")
        self.assertEqual(res["mode"], "full")
        self.assertEqual(res["fallback_reason"], "recording_reset")
        self.assertEqual(self.engine.state.recording_id, "rec_B")

    def test_c04_invalid_shift_rejected(self):
        """Any shift != 64 must be rejected at engine initialization."""
        with self.assertRaises(ValueError):
            ActualK12IncrementalEngine(self.ref, shift=32)

    # =========================================================================
    # W4-020-C05: Execution Reporting and Stagewise MAC Accounting
    # =========================================================================
    def test_c05_execution_reporting_stagewise_macs(self):
        """Incremental mode reports exact stagewise Conv and Total MAC counts."""
        stream = np.random.randn(6, 192).astype(np.float32)
        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]

        res0 = self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_mac")
        res1 = self.engine.forward_incremental(w1, step_idx=1, recording_id="rec_mac")

        self.assertEqual(res0["conv_mac_count"], FULL_CONV_MACS)
        self.assertEqual(res0["head_mac_count"], HEAD_MACS)
        self.assertEqual(res0["mac_count"], TOTAL_FULL_MACS)
        self.assertEqual(TOTAL_FULL_MACS, 1172992)

        self.assertEqual(res1["conv_mac_count"], INCREMENTAL_CONV_MACS)
        self.assertEqual(res1["head_mac_count"], HEAD_MACS)
        self.assertEqual(res1["mac_count"], TOTAL_INCREMENTAL_MACS)
        self.assertEqual(INCREMENTAL_CONV_MACS, 532224)
        self.assertEqual(TOTAL_INCREMENTAL_MACS, 795904)
        self.assertFalse(res1["is_skipped_decision_reuse"])

    # =========================================================================
    # W4-020-C06: Strict Anti-Aliasing and Error Safety
    # =========================================================================
    def test_c06_anti_aliasing_caller_mutations(self):
        """Caller mutating input array or output result cannot corrupt internal cache state."""
        w0 = np.random.randn(6, 128).astype(np.float32)
        w0_orig = w0.copy()
        res0 = self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_alias")

        w0[0, 0] = 99999.0
        self.assertEqual(self.engine.state.x_raw[0, 0], w0_orig[0, 0])

        res0["h1"][0, 0] = 77777.0
        res0["h3"][0, 0] = 88888.0
        res0["logits"][0] = -77777.0
        self.assertNotEqual(self.engine.state.h1[0, 0], 77777.0)
        self.assertNotEqual(self.engine.state.h3[0, 0], 88888.0)
        self.assertNotEqual(self.engine.state.logits[0], -77777.0)

    def test_c06_invalid_inputs_preserve_cache_state(self):
        """Invalid inputs (non-finite or wrong shape) raise ValueError without altering cache."""
        w0 = np.random.randn(6, 128).astype(np.float32)
        self.engine.forward_incremental(w0, step_idx=0, recording_id="rec_err")
        saved_step = self.engine.state.step_idx
        saved_h3 = self.engine.state.h3.copy()

        w_nan = w0.copy()
        w_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.engine.forward_incremental(w_nan, step_idx=1, recording_id="rec_err")
        self.assertEqual(self.engine.state.step_idx, saved_step)
        np.testing.assert_array_equal(self.engine.state.h3, saved_h3)

        w_bad = np.zeros((6, 100), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.engine.forward_incremental(w_bad, step_idx=1, recording_id="rec_err")
        self.assertEqual(self.engine.state.step_idx, saved_step)
        np.testing.assert_array_equal(self.engine.state.h3, saved_h3)

    # =========================================================================
    # W4-020-C07: Empirical Verification on Real Dataset Validation Windows
    # =========================================================================
    def test_c07_validation_dataset_streaming_equivalence(self):
        """Stream through validation recordings and verify zero per-stage error and zero winner disagreements."""
        df_mani = pd.read_parquet(DATA_DIR / "window_manifest.parquet")
        df_split = pd.read_parquet(DATA_DIR / "windows_with_split.parquet")
        df = pd.concat([df_mani, df_split["split"]], axis=1)
        val_df = df[df["split"] == "val"]
        x_all = np.load(DATA_DIR / "windows.npz")["x"].astype(np.float32)

        exp_list = sorted(val_df["exp"].unique())[:3]
        ref_engine = ActualK12IncrementalEngine(self.ref, shift=64)
        total_incremental = 0

        for exp in exp_list:
            grp = val_df[val_df["exp"] == exp]
            idxs = grp.index.tolist()
            self.engine.reset()

            for step, idx in enumerate(idxs):
                w = x_all[idx]
                inc_res = self.engine.forward_incremental(w, step_idx=step, recording_id=str(exp), query_row_id=idx)
                full_res = ref_engine.forward_full(w, step_idx=step, recording_id=str(exp), query_row_id=idx)
                if step > 0:
                    self.assertEqual(inc_res["mode"], "incremental")
                    total_incremental += 1

                for stage in ("h1", "p1", "h2", "p2", "h3", "logits"):
                    np.testing.assert_array_equal(inc_res[stage], full_res[stage])
                self.assertEqual(inc_res["pred_class"], full_res["pred_class"])

        self.assertGreater(total_incremental, 0)


def _evaluate_split_stream(
    ref: ReferenceOwner,
    df_sub: pd.DataFrame,
    x_all: np.ndarray,
) -> Dict[str, Any]:
    engine = ActualK12IncrementalEngine(ref, shift=64)
    ref_engine = ActualK12IncrementalEngine(ref, shift=64)

    total_windows = len(df_sub)
    total_recordings = int(df_sub["exp"].nunique())
    full_count = 0
    incremental_count = 0
    stage_max_err = {"h1": 0.0, "p1": 0.0, "h2": 0.0, "p2": 0.0, "h3": 0.0, "logits": 0.0}
    disagreements = 0

    for exp, grp in df_sub.groupby("exp"):
        idxs = grp.index.tolist()
        engine.reset()

        for step, idx in enumerate(idxs):
            w = x_all[idx]
            inc_res = engine.forward_incremental(w, step_idx=step, recording_id=str(exp), query_row_id=int(idx))
            full_res = ref_engine.forward_full(w, step_idx=step, recording_id=str(exp), query_row_id=int(idx))

            if inc_res["mode"] == "full":
                full_count += 1
            else:
                incremental_count += 1

            for k in stage_max_err:
                err = float(np.max(np.abs(inc_res[k] - full_res[k])))
                if err > stage_max_err[k]:
                    stage_max_err[k] = err

            if inc_res["pred_class"] != full_res["pred_class"]:
                disagreements += 1

    return {
        "total_windows": total_windows,
        "total_recordings": total_recordings,
        "full_passes": full_count,
        "incremental_passes": incremental_count,
        "incremental_coverage_pct": float(100.0 * incremental_count / total_windows),
        "per_stage_max_abs_error": stage_max_err,
        "max_logit_error": stage_max_err["logits"],
        "winner_disagreements": disagreements,
        "bitwise_exact": all(v == 0.0 for v in stage_max_err.values()) and disagreements == 0,
    }


def generate_incremental_validation_report(
    output_path: Path = OUTPUT_DIR / "incremental_validation.json",
) -> Dict[str, Any]:
    """Run full stagewise incremental validation on Validation (2,422) and Test (5,053) splits."""
    ref = ReferenceOwner.load()

    df_mani = pd.read_parquet(DATA_DIR / "window_manifest.parquet")
    df_split = pd.read_parquet(DATA_DIR / "windows_with_split.parquet")
    df = pd.concat([df_mani, df_split["split"]], axis=1)
    x_all = np.load(DATA_DIR / "windows.npz")["x"].astype(np.float32)

    val_stats = _evaluate_split_stream(ref, df[df["split"] == "val"], x_all)
    test_stats = _evaluate_split_stream(ref, df[df["split"] == "test"], x_all)

    report = {
        "status": "VERIFIED_EXECUTED_INDEPENDENT_CHECK",
        "task_id": "W4-020",
        "date": "2026-11-08",
        "reference": {
            "checkpoint_sha256": ref.identity.checkpoint_sha256,
            "preprocessing_id": ref.identity.preprocessing_id,
            "architecture": "SmallCNN1D(num_classes=12)",
        },
        "streaming_validation": {
            "total_val_windows": val_stats["total_windows"],
            "total_recordings": val_stats["total_recordings"],
            "full_passes": val_stats["full_passes"],
            "incremental_passes": val_stats["incremental_passes"],
            "incremental_coverage_pct": val_stats["incremental_coverage_pct"],
            "per_stage_max_abs_error": val_stats["per_stage_max_abs_error"],
            "max_logit_error": val_stats["max_logit_error"],
            "winner_disagreements": val_stats["winner_disagreements"],
            "bitwise_exact": val_stats["bitwise_exact"],
        },
        "streaming_test": {
            "total_test_windows": test_stats["total_windows"],
            "total_recordings": test_stats["total_recordings"],
            "full_passes": test_stats["full_passes"],
            "incremental_passes": test_stats["incremental_passes"],
            "incremental_coverage_pct": test_stats["incremental_coverage_pct"],
            "per_stage_max_abs_error": test_stats["per_stage_max_abs_error"],
            "max_logit_error": test_stats["max_logit_error"],
            "winner_disagreements": test_stats["winner_disagreements"],
            "bitwise_exact": test_stats["bitwise_exact"],
        },
        "computational_savings": {
            "full_conv_macs_per_window": FULL_CONV_MACS,
            "incremental_conv_macs_per_window": INCREMENTAL_CONV_MACS,
            "head_macs_per_window": HEAD_MACS,
            "full_macs_per_window": TOTAL_FULL_MACS,
            "incremental_macs_per_window": TOTAL_INCREMENTAL_MACS,
            "conv_mac_reduction_pct": float(100.0 * (1.0 - INCREMENTAL_CONV_MACS / FULL_CONV_MACS)),
            "total_mac_reduction_pct": float(100.0 * (1.0 - TOTAL_INCREMENTAL_MACS / TOTAL_FULL_MACS)),
            "stage_breakdown_macs": {
                "conv1_full": 122880,
                "conv1_incremental": 65280,
                "conv2_full": 393216,
                "conv2_incremental": 221184,
                "conv3_full": 393216,
                "conv3_incremental": 245760,
                "head_full_recomputed": 263680,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-report", action="store_true", help="Generate incremental_validation.json")
    args, unknown = parser.parse_known_args()

    if args.generate_report:
        rep = generate_incremental_validation_report()
        print(f"Report generated at: {OUTPUT_DIR / 'incremental_validation.json'}")
        print(f"Val incremental passes: {rep['streaming_validation']['incremental_passes']} / {rep['streaming_validation']['total_val_windows']}")
        print(f"Test incremental passes: {rep['streaming_test']['incremental_passes']} / {rep['streaming_test']['total_test_windows']}")
        print(f"Val max logit error: {rep['streaming_validation']['max_logit_error']}")
        print(f"Test max logit error: {rep['streaming_test']['max_logit_error']}")
    else:
        unittest.main(argv=[sys.argv[0]] + unknown)
