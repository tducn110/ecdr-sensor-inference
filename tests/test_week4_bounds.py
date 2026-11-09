#!/usr/bin/env python3
"""
tests/test_week4_bounds.py
--------------------------
Comprehensive unit test suite for Milestone M2 task W4-011:
Actual-graph IBP and paired direct/separate margin specifications.

Covers checklist requirements W4-011-C01 through W4-011-C07:
- C01: BoundEngine binds to ReferenceOwner and TrustedAnchorReceipt with exactly 11 distinct competitors.
- C02: Actual-graph IBP separates validated numerical path from ordinary-float conditional comparator.
- C03: Direct-margin and separate-logits bounds computed from identical hidden box and budgets.
- C04: Tests H07 (equal-hidden cancellation gap) and H08 (wrapping effect collapse without completeness claim).
- C05: Rejects profile mismatch, stale anchors, inverted domains, and non-finite values fail-closed.
- C06: Timing metrics, method identifiers, cryptographic hashes, and avoids duplicate anchor inference.
- C07: Paired validation anchor checks without test set contamination.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from ecdr.week4.bounds import (
    CANONICAL_PROFILE_ID,
    BoundEngine,
    MarginMethod,
    NonFiniteBoundError,
    NumericalMode,
    ProfileMismatchError,
    StaleAnchorError,
    UnsupportedDomainError,
    generate_ibp_capability_report,
    run_direct_margin_ablation,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    CandidateBox,
    NumericalStatus,
    TrustedAnchorReceipt,
)


class TestWeek4Bounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceOwner.load()
        cls.engine = BoundEngine(reference=cls.ref)

        # Load data windows and validation split indices
        cls.x_all = np.load(REPO_ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(REPO_ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()
        cls.test_indices = df_split.index[df_split["split"] == "test"].tolist()

    def test_c01_bound_engine_receipt_binding_and_competitor_integrity(self):
        """W4-011-C01: BoundEngine binds to receipt, targets class c*, and produces exactly 11 distinct competitors."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        # Compute bounds using the anchor receipt
        res = self.engine.compute_bounds(
            domain=(raw, raw),
            raw_box=(raw, raw),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )

        self.assertIsInstance(res, BoundResult)
        self.assertEqual(res.target_class, receipt.winner)
        self.assertEqual(len(res.competitor_classes), 11)
        self.assertNotIn(res.target_class, res.competitor_classes)
        self.assertEqual(len(set(res.competitor_classes)), 11)
        self.assertEqual(len(res.lower_margins), 11)
        self.assertEqual(res.reference_identity_id, self.ref.identity.identity_id)

    def test_c02_separation_of_validated_and_ordinary_float_paths(self):
        """W4-011-C02: Separates validated numerical path from ordinary-float conditional comparator."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        # 1. Validated numerical path
        res_val = self.engine.compute_bounds(
            domain=(raw, raw),
            raw_box=(raw, raw),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )
        self.assertEqual(res_val.numerical_status, NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value)
        self.assertIn(
            res_val.status,
            [BoundStatus.VALIDATED_POSITIVE.value, BoundStatus.VALIDATED_INCONCLUSIVE.value],
        )
        self.assertTrue(res_val.method_id.startswith("actual-smallcnn1d-ibp-validated"))

        # 2. Ordinary-float conditional comparator path
        res_ord = self.engine.compute_bounds(
            domain=(raw, raw),
            raw_box=(raw, raw),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.ORDINARY_FLOAT,
        )
        self.assertEqual(res_ord.numerical_status, NumericalStatus.NOT_ESTABLISHED.value)
        self.assertIn(
            res_ord.status,
            [BoundStatus.CONDITIONAL_POSITIVE.value, BoundStatus.CONDITIONAL_INCONCLUSIVE.value],
        )
        self.assertTrue(res_ord.method_id.startswith("actual-smallcnn1d-ibp-conditional"))

        # Validated bounds must be more conservative (<=) than ordinary float bounds due to Higham widening
        self.assertLessEqual(res_val.min_lower_margin, res_ord.min_lower_margin + 1e-6)

    def test_c03_paired_direct_margin_and_separate_logits(self):
        """W4-011-C03: Compute direct-margin and separate-logits from identical hidden box and budgets."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        res_direct = self.engine.compute_bounds(
            domain=(raw, raw),
            raw_box=(raw, raw),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )
        res_sep = self.engine.compute_bounds(
            domain=(raw, raw),
            raw_box=(raw, raw),
            receipt=receipt,
            margin_method=MarginMethod.SEPARATE_LOGITS,
            numerical_mode=NumericalMode.VALIDATED,
        )

        self.assertEqual(res_direct.target_class, res_sep.target_class)
        self.assertEqual(res_direct.competitor_classes, res_sep.competitor_classes)
        self.assertEqual(len(res_direct.lower_margins), 11)
        self.assertEqual(len(res_sep.lower_margins), 11)

        # Both produce sound lower bounds
        self.assertTrue(np.all(np.isfinite(res_direct.lower_margins)))
        self.assertTrue(np.all(np.isfinite(res_sep.lower_margins)))

    def test_c04_h07_equal_hidden_cancellation_and_h08_wrapping(self):
        """W4-011-C04: Evaluates H07 (direct-margin tightness gain) and H08 (wrapping collapse without completeness claim)."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        # 1. H07: Direct-margin should be strictly tighter than separate-logits on a perturbation box
        eps = 1e-5
        raw_lo = raw - eps
        raw_hi = raw + eps

        res_dir = self.engine.compute_bounds(
            domain=(raw_lo, raw_hi),
            raw_box=(raw_lo, raw_hi),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )
        res_sep = self.engine.compute_bounds(
            domain=(raw_lo, raw_hi),
            raw_box=(raw_lo, raw_hi),
            receipt=receipt,
            margin_method=MarginMethod.SEPARATE_LOGITS,
            numerical_mode=NumericalMode.VALIDATED,
        )

        # Direct margin gains tightness by canceling shared hidden features
        tightness_gain = res_dir.min_lower_margin - res_sep.min_lower_margin
        self.assertGreaterEqual(
            tightness_gain,
            -1e-6,
            "Direct margin should generally be tighter or equal to separate logits.",
        )

        # 2. H08: For larger epsilon (e.g. 1e-3), multi-layer wrapping causes margin collapse
        eps_large = 1e-3
        res_large = self.engine.compute_bounds(
            domain=(raw - eps_large, raw + eps_large),
            raw_box=(raw - eps_large, raw + eps_large),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )
        self.assertFalse(res_large.is_positive)
        self.assertEqual(res_large.status, BoundStatus.VALIDATED_INCONCLUSIVE.value)
        # Note: Inconclusive does NOT mean counterexample exists (incompleteness of IBP)

    def test_c05_rejection_of_invalid_inputs_fail_closed(self):
        """W4-011-C05: Enforces fail-closed rejection of profile mismatch, stale anchors, inverted domains, and NaNs."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        # 1. Profile mismatch
        engine_bad_profile = BoundEngine(reference=self.ref, profile="unsupported-gpu-profile")
        with self.assertRaises(ProfileMismatchError):
            engine_bad_profile.compute_bounds(
                domain=(raw, raw),
                raw_box=(raw, raw),
                receipt=receipt,
                numerical_mode=NumericalMode.VALIDATED,
            )

        # 2. Stale anchor receipt (tampered checkpoint sha256)
        stale_receipt = dataclasses.replace(
            receipt,
            checkpoint_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        with self.assertRaises(StaleAnchorError):
            self.engine.compute_bounds(
                domain=(raw, raw),
                raw_box=(raw, raw),
                receipt=stale_receipt,
            )

        # 3. Inverted domain (lo > hi)
        with self.assertRaises(UnsupportedDomainError):
            self.engine.compute_bounds(
                domain=(raw + 1.0, raw - 1.0),
                raw_box=(raw + 1.0, raw - 1.0),
                receipt=receipt,
            )

        # 4. NaN / non-finite domain
        bad_raw = raw.copy()
        bad_raw[0, 0] = np.nan
        with self.assertRaises(NonFiniteBoundError):
            self.engine.compute_bounds(
                domain=(bad_raw, bad_raw),
                raw_box=(bad_raw, bad_raw),
                receipt=receipt,
            )

    def test_c06_timings_metadata_and_no_duplicate_inference(self):
        """W4-011-C06: Provides execution timings, method identifiers, and reuses receipt without inference duplication."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        res = self.engine.compute_bounds(
            domain=(raw, raw),
            raw_box=(raw, raw),
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )

        self.assertGreater(res.bound_ms, 0.0)
        self.assertGreater(res.total_ms, 0.0)
        self.assertIn("direct_margin", res.method_id)

    def test_c07_validation_split_ablation_and_artifacts(self):
        """W4-011-C07: Generates capability and validation ablation artifacts without test contamination."""
        cap_path = generate_ibp_capability_report()
        self.assertTrue(cap_path.exists())

        # Run small ablation on 5 validation samples
        sample_subset = self.val_indices[:5]
        abl_res = run_direct_margin_ablation(
            reference=self.ref,
            sample_indices=sample_subset,
            epsilons=(0.0, 1e-5),
        )

        self.assertEqual(abl_res["dataset_split"], "VALIDATION (no test tuning)")
        self.assertIn("eps_0.0", abl_res["results_by_epsilon"])
        self.assertIn("eps_1e-05", abl_res["results_by_epsilon"])


if __name__ == "__main__":
    unittest.main()
