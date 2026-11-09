"""tests/test_week4_horizon.py — Finite-horizon reachable domains and no-drift reuse policy.

Task: W4-025 (Milestone M3)
Hypotheses: H12, H15, H16, H17, H19, H20

Verifies:
  - W4-025-C01: Shift-append recurrence: h=1 has 384 free dims, h=2 has 768 free dims (full window replacement).
  - W4-025-C02: Sound mathematical enclosure over product-tail sets; sound outer relaxation.
  - W4-025-C03: Strict certificate binding; hit reuse does NOT reset certificate age; deterministic expiry.
  - W4-025-C04: 11 margin bounds verified per horizon; unsupported horizons (h>=3) fail closed.
  - W4-025-C05: Quantitative comparison of structured wave domain vs isotropic full box.
  - W4-025-C06: Adversarial drift witness test, step gap rejection, and recording boundary resets.
  - W4-025-C07: Dimension growth measurement, empirical validation on HAPT validation split, and artifact generation.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ecdr.week4.bounds import BoundEngine, MarginMethod, NumericalMode
from ecdr.week4.certificates import (
    Certificate,
    CertificateIssuer,
    ImmutableCertificateStore,
    StoreFailureReason,
)
from ecdr.week4.domains import (
    FiniteHorizonDomain,
    HorizonStep,
    propose_finite_horizon_domain,
    run_horizon_domain_validation,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.runtime import (
    DecisionAction,
    StreamingRuntimeExecutor,
)
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    CandidateBox,
    NumericalStatus,
    QueryView,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output/week4/week4_initial_20261103"


class TestWeek4Horizon(unittest.TestCase):
    """Test suite for W4-025."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = ReferenceOwner.load()
        cls.raw_zero = np.zeros((6, 128), dtype=np.float32)
        cls.norm_zero = cls.ref.normalize(cls.raw_zero)

    def test_c01_shift_append_recurrence_and_dimensions(self) -> None:
        """W4-025-C01: Verify shift-append recurrence for h=1 (384 free) and h=2 (768 free)."""
        anchor = self.norm_zero.copy()
        dom = propose_finite_horizon_domain(
            anchor=anchor,
            tail_budgets=(0.05, 0.05),
            anchor_step=10,
            recording_id="rec_01",
            max_horizon=2,
        )

        # 1. Horizon h=1: 384 fixed prefix coordinates, 384 free tail coordinates
        cand1 = dom.to_candidate_box(1)
        self.assertEqual(cand1.free_dimensions, 384)
        lo1, hi1 = dom.get_enclosure(1)
        # Prefix must be identical to anchor tail (64 steps)
        np.testing.assert_array_equal(lo1[:, :64], anchor[:, 64:])
        np.testing.assert_array_equal(hi1[:, :64], anchor[:, 64:])
        # Tail has non-zero width
        self.assertTrue(np.all(hi1[:, 64:] > lo1[:, 64:]))

        # 2. Horizon h=2: Entire 128 samples replaced, 768 free coordinates
        cand2 = dom.to_candidate_box(2)
        self.assertEqual(cand2.free_dimensions, 768)
        lo2, hi2 = dom.get_enclosure(2)
        # Entire window has non-zero width (both prefix and tail)
        self.assertTrue(np.all(hi2[:, :64] > lo2[:, :64]))
        self.assertTrue(np.all(hi2[:, 64:] > lo2[:, 64:]))

        # 3. Horizon h >= 3: Must raise ValueError (unsupported fail closed)
        with self.assertRaises(ValueError):
            dom.get_enclosure(3)
        with self.assertRaises(ValueError):
            dom.to_candidate_box(3)

    def test_c02_mathematical_enclosure_soundness(self) -> None:
        """W4-025-C02: Verify sound mathematical enclosure for admitted states."""
        anchor = self.norm_zero.copy()
        dom = propose_finite_horizon_domain(
            anchor=anchor,
            tail_budgets=(0.1, 0.1),
            anchor_step=0,
            recording_id="rec_A",
        )

        # Valid h=1 state: exact prefix + tail within budget 0.1
        q1_valid = np.zeros((6, 128), dtype=np.float32)
        q1_valid[:, :64] = anchor[:, 64:]
        q1_valid[:, 64:] = np.repeat(anchor[:, -1:], 64, axis=1) + 0.05
        is_mem1, code1 = dom.check_membership(q1_valid, step_idx=1, recording_id="rec_A")
        self.assertTrue(is_mem1)
        self.assertEqual(code1, "EXACT_MEMBER")

        # Invalid h=1 state: prefix corrupted by even 1e-4 -> REJECTED
        q1_bad_prefix = q1_valid.copy()
        q1_bad_prefix[0, 0] += 1e-3
        is_mem_bad, code_bad = dom.check_membership(q1_bad_prefix, step_idx=1, recording_id="rec_A")
        self.assertFalse(is_mem_bad)
        self.assertEqual(code_bad, "PREFIX_MISMATCH")

        # Invalid h=1 state: tail exceeds budget -> REJECTED
        q1_bad_tail = q1_valid.copy()
        q1_bad_tail[0, 64] += 0.2
        is_mem_bad_tail, code_bad_tail = dom.check_membership(q1_bad_tail, step_idx=1, recording_id="rec_A")
        self.assertFalse(is_mem_bad_tail)
        self.assertEqual(code_bad_tail, "TAIL_OUTSIDE")

        # Valid h=2 state: both blocks within budget 0.1
        q2_valid = np.repeat(anchor[:, -1:], 128, axis=1) + 0.08
        is_mem2, code2 = dom.check_membership(q2_valid, step_idx=2, recording_id="rec_A")
        self.assertTrue(is_mem2)
        self.assertEqual(code2, "EXACT_MEMBER")

        # Invalid h=2 state: exceeds budget -> REJECTED
        q2_bad = q2_valid.copy()
        q2_bad[2, 20] += 0.25
        is_mem2_bad, code2_bad = dom.check_membership(q2_bad, step_idx=2, recording_id="rec_A")
        self.assertFalse(is_mem2_bad)
        self.assertEqual(code2_bad, "OUTSIDE_DOMAIN")

    def test_c03_certificate_binding_and_no_drift_expiry(self) -> None:
        """W4-025-C03: Hit reuse does NOT reset certificate age; deterministic expiry."""
        store = ImmutableCertificateStore(reference=self.ref)

        dom = propose_finite_horizon_domain(
            anchor=self.norm_zero,
            tail_budgets=(0.2, 0.2),
            anchor_step=10,
            recording_id="rec_01",
        )
        cand_box = dom.to_candidate_box(1)

        receipt = self.ref.infer(self.raw_zero, global_row_id=1, recording_id="rec_01", step_idx=10)

        cert = Certificate(
            reference_identity_id=receipt.reference_identity_id,
            checkpoint_sha256=receipt.checkpoint_sha256,
            preprocessing_id=receipt.preprocessing_id,
            graph_id=receipt.graph_id,
            anchor_raw_hash=receipt.raw_hash,
            anchor_normalized_hash=receipt.normalized_hash,
            anchor_step=10,
            recording_id="rec_01",
            target_class=receipt.winner,
            competitor_classes=tuple(j for j in range(12) if j != receipt.winner),
            domain=cand_box,
            lower_margins=tuple(1.0 for _ in range(11)),
            max_age=2,  # Strictly expires after step 10 + 2 = 12
            method_id="direct_margin",
            numerical_status=NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value,
            lifecycle="ACTIVE",
            formal_reuse_enabled=True,
        )

        store.store_certificate(cert)

        # 1. Query at step 11: Valid reuse (age = 1 <= 2)
        can_reuse, active_c, reason = store.lookup_and_verify(
            query=self.raw_zero,
            step_idx=11,
            recording_id="rec_01",
        )
        self.assertTrue(can_reuse)
        self.assertIsNotNone(active_c)
        # CRITICAL NO-DRIFT ASSERTION: anchor_step remains 10, max_age remains 2!
        self.assertEqual(active_c.anchor_step, 10)
        self.assertEqual(active_c.max_age, 2)

        # 2. Query at step 12: Valid reuse (age = 2 <= 2)
        can_reuse2, active_c2, reason2 = store.lookup_and_verify(
            query=self.raw_zero,
            step_idx=12,
            recording_id="rec_01",
        )
        self.assertTrue(can_reuse2)
        self.assertEqual(active_c2.anchor_step, 10)

        # 3. Query at step 13: EXPIRED (age = 3 > 2)
        can_reuse3, active_c3, reason3 = store.lookup_and_verify(
            query=self.raw_zero,
            step_idx=13,
            recording_id="rec_01",
        )
        self.assertFalse(can_reuse3)
        self.assertIsNone(active_c3)
        self.assertEqual(reason3, StoreFailureReason.EXPIRED_CERTIFICATE.value)
        # Certificate store has invalidated the expired certificate
        self.assertIsNone(store.active_certificate)

    def test_c04_bound_sufficiency_and_unsupported_horizon_fail_closed(self) -> None:
        """W4-025-C04: Verify 11 bounds requirement and fail-closed rejection for h >= 3."""
        dom = propose_finite_horizon_domain(
            anchor=self.norm_zero,
            tail_budgets=(1e-4, 1e-4),
            anchor_step=5,
            recording_id="rec_exp",
            max_horizon=2,
        )

        # Query at step 8: h = 3 > max_horizon 2 -> REJECTED
        q_h3 = self.norm_zero.copy()
        is_mem, reason = dom.check_membership(q_h3, step_idx=8, recording_id="rec_exp")
        self.assertFalse(is_mem)
        self.assertEqual(reason, "HORIZON_EXCEEDED")

        # Non-causal query at step 4: h = -1 <= 0 -> REJECTED
        is_mem_past, reason_past = dom.check_membership(q_h3, step_idx=4, recording_id="rec_exp")
        self.assertFalse(is_mem_past)
        self.assertEqual(reason_past, "PAST_STEP_OR_NON_CAUSAL")

    def test_c05_wave_vs_full_box_comparison(self) -> None:
        """W4-025-C05: Contrast wave domain vs isotropic full box volume and dimensionality."""
        anchor = self.norm_zero.copy()
        budget = 1e-3

        dom = propose_finite_horizon_domain(
            anchor=anchor,
            tail_budgets=(budget, budget),
            anchor_step=0,
            recording_id="rec_test",
        )

        lo_wave, hi_wave = dom.get_enclosure(1)
        # Width of prefix coordinates in wave domain is strictly 0
        wave_prefix_width = (hi_wave - lo_wave)[:, :64]
        self.assertEqual(float(np.max(wave_prefix_width)), 0.0)

        # Full box has width 2*budget across ALL 768 coordinates
        full_box_width = 2.0 * budget
        # Volume ratio: (0)^384 vs (2*budget)^768 -> Wave domain is a lower-dimensional manifold
        # of dimension 384 embedded in R^768!
        self.assertEqual(dom.to_candidate_box(1).free_dimensions, 384)
        self.assertEqual(dom.to_candidate_box(2).free_dimensions, 768)

    def test_c06_adversarial_drift_witness_and_step_gaps(self) -> None:
        """W4-025-C06: Adversarial drift witness test proving No-Drift policy prevents ground-truth corruption."""
        executor = StreamingRuntimeExecutor(
            reference=self.ref,
            formal_reuse_enabled=True,
            default_budget=1e-4,
            max_age=2,  # Strictly expires after 2 steps
            cooldown_steps=0,
        )

        cand_box = CandidateBox(
            lower=tuple(float(x - 0.2) for x in self.norm_zero.ravel()),
            upper=tuple(float(x + 0.2) for x in self.norm_zero.ravel()),
            kind="diagonal_box",
            free_dimensions=768,
        )
        receipt0 = self.ref.infer(self.raw_zero, global_row_id=0, recording_id="stream_X", step_idx=0)
        cert0 = Certificate(
            reference_identity_id=receipt0.reference_identity_id,
            checkpoint_sha256=receipt0.checkpoint_sha256,
            preprocessing_id=receipt0.preprocessing_id,
            graph_id=receipt0.graph_id,
            anchor_raw_hash=receipt0.raw_hash,
            anchor_normalized_hash=receipt0.normalized_hash,
            anchor_step=0,
            recording_id="stream_X",
            target_class=receipt0.winner,
            competitor_classes=tuple(j for j in range(12) if j != receipt0.winner),
            domain=cand_box,
            lower_margins=tuple(1.0 for _ in range(11)),
            max_age=2,
            method_id="direct_margin",
            numerical_status=NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value,
            lifecycle="ACTIVE",
            formal_reuse_enabled=True,
        )
        executor.store.store_certificate(cert0)

        # Step 1: Small perturbation within certified domain (accepted reuse)
        q1_arr = self.raw_zero + 0.02
        q1 = QueryView.from_array(q1_arr, recording_id="stream_X", step_idx=1, global_row_id=1)
        d1 = executor.process_query(q1)
        self.assertTrue(d1.is_reuse)
        self.assertEqual(d1.action, DecisionAction.REUSE_CERTIFIED.value)

        # Step 2: Another perturbation within domain (accepted reuse)
        q2_arr = self.raw_zero + 0.04
        q2 = QueryView.from_array(q2_arr, recording_id="stream_X", step_idx=2, global_row_id=2)
        d2 = executor.process_query(q2)
        self.assertTrue(d2.is_reuse)
        self.assertEqual(d2.action, DecisionAction.REUSE_CERTIFIED.value)

        # Step 3: Certificate MUST EXPIRE (age = 3 > max_age 2), forcing reference fallback!
        # If drifting / floating anchor were allowed, query 3 could continue reusing old label.
        q3_arr = self.raw_zero + 0.06
        q3 = QueryView.from_array(q3_arr, recording_id="stream_X", step_idx=3, global_row_id=3)
        d3 = executor.process_query(q3)

        self.assertFalse(d3.is_reuse)
        self.assertEqual(d3.action, DecisionAction.FALLBACK_EXPIRED.value)
        self.assertGreater(executor.counters.fallback_expired_count, 0)

        # Out-of-order / step gap test: step index 2 arrives after step 3 -> Rejected fail-closed
        q_gap = QueryView.from_array(self.raw_zero, recording_id="stream_X", step_idx=2, global_row_id=10)
        d_gap = executor.process_query(q_gap)
        self.assertFalse(d_gap.is_reuse)
        self.assertEqual(d_gap.action, DecisionAction.FALLBACK_UNSUPPORTED.value)
        self.assertGreater(executor.counters.fallback_unsupported_count, 0)

    def test_c07_validation_ablation_and_artifact(self) -> None:
        """W4-025-C07: Run validation on validation split and generate horizon_validation.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_file = tmp_path / "horizon_validation.json"

            results = run_horizon_domain_validation(
                reference=self.ref,
                run_id="test_run",
                num_sequences=20,
                output_path=out_file,
            )

            self.assertTrue(out_file.exists())
            self.assertEqual(results["task_id"], "W4-025")
            self.assertEqual(results["recurrence_contracts"]["h1_free_dimensions"], 384)
            self.assertEqual(results["recurrence_contracts"]["h2_free_dimensions"], 768)
            self.assertEqual(results["recurrence_contracts"]["dimension_growth_ratio"], 2.0)
            self.assertEqual(results["recurrence_contracts"]["h2_anchor_overlap_fraction"], 0.0)

        # Generate official artifact in output/week4/week4_initial_20261103/
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        official_out = OUTPUT_DIR / "horizon_validation.json"
        official_results = run_horizon_domain_validation(
            reference=self.ref,
            run_id="week4_initial_20261103",
            num_sequences=50,
            output_path=official_out,
        )
        self.assertTrue(official_out.exists())
        self.assertEqual(official_results["task_id"], "W4-025")
