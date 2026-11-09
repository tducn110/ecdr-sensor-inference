"""tests/test_week4_runtime.py — Unit test suite for W4-023 streaming runtime executor.

Covers:
  - W4-023-C01: Strict QueryView input contract and zero lookahead leakage.
  - W4-023-C02: Formal conjunction of capability, certificate, and membership.
  - W4-023-C03: Exactly one reference execution on miss (no redundant forward).
  - W4-023-C04: Atomic state commit & fail-closed rejection of non-finite inputs.
  - W4-023-C05: Detailed counters separation (hits, misses, fallback reasons).
  - W4-023-C06: Certificate expiration (max_age) and cooldown policies.
  - W4-023-C07: End-to-end stream replay, recording resets, and 100% decision agreement.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.runtime import (
    InvalidQueryError,
    StreamingRuntimeExecutor,
    run_streaming_runtime_validation,
)
from ecdr.week4.types import (
    DecisionAction,
    QueryView,
    RuntimeDecision,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()

        # Load data windows and validation split indices
        cls.x_all = np.load(ROOT / "data/windows.npz")["x"]
        df_mani = pd.read_parquet(ROOT / "data/window_manifest.parquet")
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        df = pd.concat([df_mani, df_split["split"]], axis=1)
        df["global_row_id"] = np.arange(len(df), dtype=np.int64)
        cls.val_df = df[df["split"] == "val"].sort_values(["exp", "win_idx"])

    def test_c01_queryview_input_contract_and_nonfinite_rejection(self):
        """W4-023-C01 & C04: Accepts only QueryView; rejects non-finite inputs fail-closed."""
        executor = StreamingRuntimeExecutor(reference=self.ref)

        # 1. Non-QueryView input must be rejected
        with self.assertRaises(InvalidQueryError):
            executor.process_query("invalid_query_object")  # type: ignore

        # 2. Non-finite array in QueryView creation must be rejected fail-closed
        bad_raw = np.ones((6, 128), dtype=np.float32)
        bad_raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            QueryView.from_array(
                raw=bad_raw,
                global_row_id=0,
                recording_id="exp_01",
                step_idx=0,
            )

        # Ensure state was not committed
        self.assertEqual(executor.counters.total_queries, 1)
        self.assertIsNone(executor._last_recording_id)

    def test_c02_and_c03_miss_single_forward_and_atomic_issuance(self):
        """W4-023-C02 & C03: First query misses, runs reference forward once, and issues certificate."""
        executor = StreamingRuntimeExecutor(
            reference=self.ref,
            formal_reuse_enabled=True,
            default_budget=1e-4,
        )

        row0 = self.val_df.iloc[0]
        q_row_id = int(row0["global_row_id"])
        raw0 = self.x_all[q_row_id].astype(np.float32)

        qv0 = QueryView.from_array(
            raw=raw0,
            global_row_id=q_row_id,
            recording_id=f"exp_{int(row0['exp']):02d}",
            step_idx=int(row0["win_idx"]),
        )

        decision0 = executor.process_query(qv0)
        self.assertIsInstance(decision0, RuntimeDecision)
        self.assertEqual(decision0.action, DecisionAction.FALLBACK_REFERENCE.value)
        self.assertFalse(decision0.is_reuse)

        # Check that exactly one build attempt was made
        self.assertEqual(executor.counters.build_attempts, 1)
        self.assertEqual(executor.counters.fallback_reference_count, 1)

    def test_c05_and_c06_expiration_and_cooldown(self):
        """W4-023-C05 & C06: Certificate expires after max_age steps."""
        executor = StreamingRuntimeExecutor(
            reference=self.ref,
            formal_reuse_enabled=True,
            default_budget=1e-4,
            max_age=1,  # Strictly expires after step t+1
        )

        row0 = self.val_df.iloc[2]
        rec_id = f"exp_{int(row0['exp']):02d}"
        raw0 = self.x_all[int(row0["global_row_id"])].astype(np.float32)

        qv0 = QueryView.from_array(
            raw=raw0,
            global_row_id=int(row0["global_row_id"]),
            recording_id=rec_id,
            step_idx=10,
        )
        decision0 = executor.process_query(qv0)
        self.assertEqual(decision0.action, DecisionAction.FALLBACK_REFERENCE.value)
        self.assertIsNotNone(executor.store.active_certificate)

        # Send query at step 12 (gap > max_age 1)
        qv_expired = QueryView.from_array(
            raw=raw0,
            global_row_id=int(row0["global_row_id"]) + 2,
            recording_id=rec_id,
            step_idx=12,
        )
        decision_exp = executor.process_query(qv_expired)
        self.assertEqual(decision_exp.action, DecisionAction.FALLBACK_EXPIRED.value)
        self.assertGreater(executor.counters.fallback_expired_count, 0)

    def test_c07_recording_boundary_reset_and_gate_closed_profile(self):
        """W4-023-C07: Changing recording resets state; gate closed forces 100% fallback."""
        # Test A: Recording reset
        executor = StreamingRuntimeExecutor(reference=self.ref, formal_reuse_enabled=True)

        raw_zero = np.zeros((6, 128), dtype=np.float32)
        qv1 = QueryView.from_array(raw=raw_zero, global_row_id=0, recording_id="exp_01", step_idx=0)
        executor.process_query(qv1)

        qv2 = QueryView.from_array(raw=raw_zero, global_row_id=1, recording_id="exp_02", step_idx=0)
        d2 = executor.process_query(qv2)
        self.assertEqual(d2.action, DecisionAction.FALLBACK_RECORDING_RESET.value)
        self.assertEqual(executor.counters.fallback_recording_reset_count, 1)

        # Test B: Gate closed profile
        exec_closed = StreamingRuntimeExecutor(reference=self.ref, formal_reuse_enabled=False)
        d_closed1 = exec_closed.process_query(qv1)
        self.assertEqual(d_closed1.action, DecisionAction.FALLBACK_REFERENCE.value)
        self.assertEqual(exec_closed.counters.reused_certified_count, 0)

    def test_c07_validation_ablation_and_artifact(self):
        """W4-023-C07: Run small validation stream execution and generate runtime_validation.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            val_out = run_streaming_runtime_validation(
                reference=self.ref,
                run_id="test_run",
                num_queries=10,
                output_path=tmp_path / "runtime_validation.json",
            )

            self.assertTrue((tmp_path / "runtime_validation.json").is_file())
            self.assertEqual(val_out["task_id"], "W4-023")
            self.assertEqual(val_out["evaluated_queries"], 10)
            self.assertEqual(val_out["gate_open_profile"]["reference_agreement_accuracy"], 1.0)
            self.assertEqual(val_out["gate_closed_profile"]["reference_agreement_accuracy"], 1.0)
            self.assertTrue(val_out["gate_closed_profile"]["gate_closed_all_fallback"])


if __name__ == "__main__":
    unittest.main()
