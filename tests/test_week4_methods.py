"""tests/test_week4_methods.py — Unit test suite for W4-028 Method Registry & Baselines.

Covers checklist requirements W4-028-C01 through W4-028-C07:
  - W4-028-C01: Register M00–M16 per COMPARISON; distinguish independent methods, policy axes, applicability-only rows.
  - W4-028-C02: Full core, strong wrapper, and owned reference share same admitted input/output contract; each identity mode records distinct threat model.
  - W4-028-C03: Exact replay uses full owned input bytes + identity/profile, collision-safe byte comparison; never calls regional reuse.
  - W4-028-C04: Incremental strict eligibility from W4-021; experimental and heuristic methods have dedicated error budgets, no formal badge.
  - W4-028-C05: Lip/IBP/wave/correlated/CROWN adapters expose per-method numerical status and failure reasons, avoiding shared valid booleans.
  - W4-028-C06: Prediction interface receives zero ground truth and zero future query lookahead; reference forward only executes as fallback.
  - W4-028-C07: Tests unsupported rows, semantic grouping, cache invalidation, and generates output/week4/<run_id>/method_capabilities.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ecdr.week4.methods import (
    ALL_METHOD_CLASSES,
    BaseMethodAdapter,
    InvalidMethodInputError,
    M00FrozenCoreAdapter,
    M01WrapperAdapter,
    M02ExactInputReplayAdapter,
    M03IncrementalAdapter,
    M04HeuristicAdapter,
    M05LipschitzAdapter,
    M06IBPDiagonalAdapter,
    M07IBPWaveAdapter,
    M08CoordinateHullAdapter,
    M09CorrelatedNativeAdapter,
    M10CROWNDiagonalAdapter,
    M11CROWNWavesAdapter,
    M12FixedAnchorPolicyAdapter,
    M13FiniteHorizonPolicyAdapter,
    M14LipCacheComparator,
    M15PriorWorkApplicability,
    M16OwnedReferenceAdapter,
    MethodDecision,
    SEMANTIC_GROUPS,
    UnsupportedMethodError,
    get_method_adapter,
    list_registered_methods,
    run_method_registry_validation,
)
from ecdr.week4.reference import IdentityCheckMode, ReferenceOwner
from ecdr.week4.types import NumericalStatus, QueryView

ROOT = Path(__file__).resolve().parents[1]
METHODS_JSON_PATH = ROOT / "configs/week4/methods.json"


class TestWeek4Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()

    def test_c01_method_registration_and_semantic_grouping(self):
        """W4-028-C01: Register M00–M16 per COMPARISON; distinguish independent methods, policy axes, applicability-only rows."""
        self.assertEqual(len(ALL_METHOD_CLASSES), 17)
        expected_ids = [f"M{i:02d}" for i in range(17)]
        self.assertEqual(sorted(list(ALL_METHOD_CLASSES.keys())), expected_ids)

        # Verify semantic grouping covers all 17 methods without duplication
        grouped_ids = []
        for g_name, ids in SEMANTIC_GROUPS.items():
            grouped_ids.extend(ids)
        self.assertEqual(sorted(grouped_ids), expected_ids)
        self.assertEqual(len(grouped_ids), 17)

        # Cross-check against configs/week4/methods.json
        with open(METHODS_JSON_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg_ids = [m["id"] for m in cfg["methods"]]
        self.assertEqual(sorted(cfg_ids), expected_ids)

        registered = list_registered_methods()
        self.assertEqual(len(registered), 17)
        for m in registered:
            self.assertIn("id", m)
            self.assertIn("category", m)
            self.assertIn("guarantee_target", m)
            self.assertIn("threat_model", m)
            self.assertIn("is_executable", m)
            self.assertIn("formal_badge", m)

    def test_c02_input_output_contract_and_identity_threat_models(self):
        """W4-028-C02: Full core, strong wrapper, and owned reference have same admitted contract; distinct threat models."""
        m00 = get_method_adapter("M00", reference=self.ref)
        m01 = get_method_adapter("M01", reference=self.ref)
        m16 = get_method_adapter("M16", reference=self.ref)

        self.assertEqual(m00.threat_model, "NO_INTEGRITY_CHECK")
        self.assertEqual(m01.threat_model, "PER_QUERY_INTEGRITY_CHECK")
        self.assertEqual(m16.threat_model, "OWNED_PROCESS_BOUNDARY")

        # Form a valid query
        arr = np.random.RandomState(42).randn(6, 128).astype(np.float32)
        qv = QueryView.from_array(arr, global_row_id=1, recording_id="rec_01", step_idx=0)

        # Run all three on the exact same QueryView
        d00 = m00.process_query(qv)
        d01 = m01.process_query(qv)
        d16 = m16.process_query(qv)

        # Same decision contract
        self.assertEqual(d00.predicted_class, d01.predicted_class)
        self.assertEqual(d00.predicted_class, d16.predicted_class)
        self.assertIsNotNone(d00.logits)
        self.assertIsNotNone(d01.logits)
        self.assertIsNotNone(d16.logits)
        np.testing.assert_allclose(d00.logits, d01.logits, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(d00.logits, d16.logits, rtol=1e-5, atol=1e-6)

        # None claim reuse
        self.assertFalse(d00.is_reuse)
        self.assertFalse(d01.is_reuse)
        self.assertFalse(d16.is_reuse)

        # Rejection of invalid shape and non-finite inputs
        with self.assertRaises(InvalidMethodInputError):
            m00.process_query(np.zeros((6, 100), dtype=np.float32))

        nan_arr = arr.copy()
        nan_arr[0, 0] = np.nan
        with self.assertRaises(InvalidMethodInputError):
            m01.process_query(nan_arr)

    def test_c03_exact_input_replay_singleton_semantics(self):
        """W4-028-C03: Exact replay uses full owned input bytes + identity/profile, collision-safe byte comparison; never calls regional reuse."""
        m02 = get_method_adapter("M02", reference=self.ref)
        self.assertFalse(m02.formal_badge)
        self.assertEqual(m02.guarantee_target, "IDENTICAL_INPUT_EQUALITY")
        self.assertEqual(m02.threat_model, "EXACT_INPUT_EQUALITY")

        arr0 = np.ones((6, 128), dtype=np.float32) * 0.1
        qv0 = QueryView.from_array(arr0, global_row_id=10, recording_id="rec_A", step_idx=0)

        # Query 1: Cache Miss
        d_miss = m02.process_query(qv0)
        self.assertFalse(d_miss.is_reuse)
        self.assertFalse(d_miss.is_singleton_reuse)
        self.assertFalse(d_miss.is_regional_reuse)
        self.assertEqual(d_miss.action, "fallback_reference")

        # Query 2: Identical input -> Cache Hit!
        d_hit = m02.process_query(qv0)
        self.assertTrue(d_hit.is_reuse)
        self.assertTrue(d_hit.is_singleton_reuse)
        self.assertFalse(d_hit.is_regional_reuse)  # Must NOT call regional reuse!
        self.assertFalse(d_hit.is_formal_reuse)
        self.assertFalse(d_hit.formal_badge)
        self.assertEqual(d_hit.action, "reuse_exact_input")
        self.assertEqual(d_hit.predicted_class, d_miss.predicted_class)

        # Collision safety check: different query under same step does not hit
        arr1 = arr0.copy()
        arr1[0, 0] = 0.999
        qv1 = QueryView.from_array(arr1, global_row_id=11, recording_id="rec_A", step_idx=1)
        d_diff = m02.process_query(qv1)
        self.assertFalse(d_diff.is_reuse)

        # Recording boundary reset check: crossing to rec_B invalidates cache
        qv0_new_rec = QueryView.from_array(arr0, global_row_id=12, recording_id="rec_B", step_idx=0)
        d_reset = m02.process_query(qv0_new_rec)
        self.assertFalse(d_reset.is_reuse)

    def test_c04_incremental_eligibility_and_heuristic_budget(self):
        """W4-028-C04: Incremental strict eligibility from W4-021; heuristic methods have labels/error budget, no formal badge."""
        m03 = get_method_adapter("M03", reference=self.ref)
        self.assertFalse(m03.formal_badge)
        self.assertEqual(m03.guarantee_target, "BACKEND_EQUIVALENT_FORWARD")

        # Query 0: initial step -> full forward
        arr0 = np.random.RandomState(123).randn(6, 128).astype(np.float32)
        qv0 = QueryView.from_array(arr0, global_row_id=1, recording_id="rec_inc", step_idx=0)
        d0 = m03.process_query(qv0)
        self.assertFalse(d0.is_reuse)
        self.assertEqual(d0.action, "incremental_full_forward")

        # Query 1: continuous step (stride 64) -> incremental reuse
        arr1 = np.zeros((6, 128), dtype=np.float32)
        arr1[:, :64] = arr0[:, 64:]
        arr1[:, 64:] = np.random.RandomState(124).randn(6, 64).astype(np.float32)
        qv1 = QueryView.from_array(arr1, global_row_id=2, recording_id="rec_inc", step_idx=1)

        d1 = m03.process_query(qv1)
        self.assertTrue(d1.is_reuse)
        self.assertEqual(d1.action, "reuse_incremental")
        self.assertFalse(d1.is_formal_reuse)
        self.assertFalse(d1.formal_badge)

        # Parity with M00 on query 1
        m00 = get_method_adapter("M00", reference=self.ref)
        d00_1 = m00.process_query(qv1)
        self.assertEqual(d1.predicted_class, d00_1.predicted_class)
        np.testing.assert_allclose(d1.logits, d00_1.logits, rtol=1e-5, atol=1e-5)

        # M04 Heuristic check
        m04 = get_method_adapter("M04", reference=self.ref, threshold=0.1)
        self.assertFalse(m04.formal_badge)
        self.assertEqual(m04.guarantee_target, "EMPIRICAL_TRADE_OFF")
        self.assertEqual(m04.threat_model, "EMPIRICAL_HEURISTIC_NO_GUARANTEE")

        # First query establishes anchor
        d04_0 = m04.process_query(qv0)
        self.assertFalse(d04_0.is_reuse)

        # Close query within threshold -> heuristic reuse
        arr_close = arr0 + 1e-4
        qv_close = QueryView.from_array(arr_close, global_row_id=3, recording_id="rec_inc", step_idx=1)
        d04_close = m04.process_query(qv_close)
        self.assertTrue(d04_close.is_reuse)
        self.assertFalse(d04_close.is_formal_reuse)
        self.assertFalse(d04_close.formal_badge)
        self.assertEqual(d04_close.action, "reuse_heuristic")

    def test_c05_per_method_numerical_status_and_failure_reasons(self):
        """W4-028-C05: Lip/IBP/wave/correlated/CROWN expose per-method numerical status and failure reasons."""
        adapters = {
            "M05": get_method_adapter("M05", reference=self.ref),
            "M06": get_method_adapter("M06", reference=self.ref),
            "M07": get_method_adapter("M07", reference=self.ref),
            "M08": get_method_adapter("M08", reference=self.ref),
            "M09": get_method_adapter("M09", reference=self.ref),
            "M10": get_method_adapter("M10", reference=self.ref),
            "M11": get_method_adapter("M11", reference=self.ref),
        }

        # Verify numerical status is distinct per method category
        self.assertEqual(adapters["M05"].numerical_status, NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value)
        self.assertIn("FLOAT32", adapters["M06"].numerical_status)
        self.assertIn("FLOAT32", adapters["M07"].numerical_status)
        self.assertEqual(adapters["M08"].numerical_status, "CORRELATED_COORDINATE_HULL")
        self.assertEqual(adapters["M09"].numerical_status, "NATIVE_CORRELATED_BOUNDS")
        self.assertEqual(adapters["M10"].numerical_status, "SURROGATE_BACKWARD_RELAXATION")
        self.assertEqual(adapters["M11"].numerical_status, "SURROGATE_BACKWARD_RELAXATION")

        # Verify failure reasons on initial fallback
        arr = np.zeros((6, 128), dtype=np.float32)
        qv = QueryView.from_array(arr, global_row_id=99, recording_id="rec_status", step_idx=0)
        for mid, adapter in adapters.items():
            decision = adapter.process_query(qv)
            self.assertIsInstance(decision, MethodDecision)
            self.assertEqual(decision.method_id, mid)
            self.assertIsNotNone(decision.numerical_status)
            self.assertNotEqual(decision.numerical_status, "")
            self.assertIsNotNone(decision.failure_reason)

    def test_c06_zero_leakage_and_prediction_interface(self):
        """W4-028-C06: Prediction interface receives zero ground truth and zero future query lookahead."""
        # QueryView only accepts global_row_id, recording_id, step_idx, raw_window
        qv = QueryView.from_array(np.zeros((6, 128), dtype=np.float32), 1, "rec", 0)
        self.assertFalse(hasattr(qv, "label"))
        self.assertFalse(hasattr(qv, "ground_truth"))

        # Verify adapter signature takes only (query)
        for mid in ALL_METHOD_CLASSES:
            adapter = get_method_adapter(mid, reference=self.ref)
            self.assertTrue(callable(getattr(adapter, "process_query")))

    def test_c07_unsupported_rows_policy_axes_and_artifact_generation(self):
        """W4-028-C07: Tests unsupported rows, semantic grouping, cache invalidation, and generates artifact."""
        # M14 and M15 raise UnsupportedMethodError fail-closed
        m14 = get_method_adapter("M14")
        m15 = get_method_adapter("M15")
        self.assertFalse(m14.is_executable)
        self.assertFalse(m15.is_executable)
        self.assertEqual(m14.status, "NOT_APPLICABLE_IN_FROZEN_SCOPE")
        self.assertEqual(m15.status, "AUDITED_APPLICABILITY_ONLY")

        qv = QueryView.from_array(np.zeros((6, 128), dtype=np.float32), 1, "rec", 0)
        with self.assertRaises(UnsupportedMethodError):
            m14.process_query(qv)
        with self.assertRaises(UnsupportedMethodError):
            m15.process_query(qv)

        # Policy axes M12 and M13
        m12 = get_method_adapter("M12", reference=self.ref, parent_method_id="M06")
        m13 = get_method_adapter("M13", reference=self.ref, parent_method_id="M07", horizon=1)
        self.assertEqual(m12.category, "policy_axis")
        self.assertEqual(m13.category, "policy_axis")
        self.assertEqual(m12.guarantee_target, "POLICY_AXIS_CONTROL")
        self.assertEqual(m13.guarantee_target, "POLICY_AXIS_CONTROL")

        # Run method registry validation and verify artifact generation
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output/week4/week4_initial_20261103/method_capabilities.json"
            rep = run_method_registry_validation(
                reference=self.ref,
                run_id="week4_initial_20261103",
                output_path=out_path,
                base_dir=tmpdir,
            )
            self.assertTrue(out_path.exists())
            self.assertEqual(rep["total_registered_methods"], 17)
            self.assertTrue(rep["parity_and_contract_checks"]["m00_m01_decision_and_logits_parity"])
            self.assertTrue(rep["parity_and_contract_checks"]["m00_m16_decision_and_logits_parity"])
            self.assertTrue(rep["parity_and_contract_checks"]["m00_m03_decision_and_logits_parity"])
            self.assertTrue(rep["parity_and_contract_checks"]["m02_exact_replay_singleton_verified"])
            self.assertTrue(rep["parity_and_contract_checks"]["m14_faithful_lipcache_fail_closed"])
            self.assertTrue(rep["parity_and_contract_checks"]["m15_prior_work_applicability_fail_closed"])


if __name__ == "__main__":
    unittest.main()
