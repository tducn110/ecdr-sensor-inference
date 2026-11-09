"""tests/test_week4_correlated_bounds.py — Unit test suite for W4-017 correlation-preserving bounds.

Covers:
  - W4-017-C01: Concrete affine backend justification (auto-LiRPA CROWN + affine zonotope support evaluation).
  - W4-017-C02: Support and margin bounds derivation for latent and residual variables; numerical status separation.
  - W4-017-C03: Native correlated vs same-domain outer-box controls; outer-box labeled CORRELATED_HULL.
  - W4-017-C04: All 11 margin objectives against winner c*; complete reference and domain identity binding.
  - W4-017-C05: Tiny analytic toy demonstrating cancellation + paired validation study measuring tightness.
  - W4-017-C06: Full latent and residual uncertainty accounted for without dimension truncation; fail-closed rejection.
  - W4-017-C07: Artifact generation (correlated_bound_capability.json, hull_loss_validation.json).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ecdr.week4.bounds import CorrelatedBoundEngine, MarginMethod
from ecdr.week4.domains import AffineTailDomain, fit_train_pca_affine_parameters
from ecdr.week4.lirpa_adapter import (
    CorrelatedBoundMode,
    CrownAdapter,
    evaluate_toy_correlation_cancellation,
    run_correlated_bound_capability_study,
    run_hull_loss_validation,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    NumericalStatus,
    TrustedAnchorReceipt,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4CorrelatedBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()
        cls.adapter = CrownAdapter(cls.ref)
        cls.engine = CorrelatedBoundEngine(cls.ref)
        cls.params = fit_train_pca_affine_parameters(k_components=8)

        # Load data windows and validation split indices
        cls.x_all = np.load(ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()

        # Build a standard validation domain fixture
        cls.val_row = cls.val_indices[0]
        cls.raw0 = cls.x_all[cls.val_row].astype(np.float32)
        cls.receipt0 = cls.ref.infer(cls.raw0, global_row_id=cls.val_row)

        prefix0 = cls.raw0[:, :64]
        c_tail0 = cls.raw0[:, -1:, None].repeat(64, axis=-1).reshape(384).astype(np.float32)
        cls.domain0 = AffineTailDomain(
            center_tail=c_tail0,
            prefix=prefix0,
            params=cls.params,
            latent_scale=1.0,
            residual_scale=1.0,
            step_idx=0,
            recording_id="val",
        )

    def test_c01_backend_justification_and_dimensions(self):
        """W4-017-C01: Concrete affine backend justification and dimension verification."""
        self.assertTrue(self.adapter.conversion_ok)
        self.assertIsNotNone(self.adapter._bounded_module)
        self.assertEqual(self.domain0.params.k_components, 8)
        self.assertEqual(self.domain0.params.basis.shape, (384, 8))
        self.assertEqual(self.domain0.params.residual_bounds.shape, (384,))
        self.assertEqual(self.domain0.prefix.shape, (6, 64))
        self.assertEqual(self.domain0.center_tail.shape, (384,))

    def test_c02_numerical_status_and_margin_derivation(self):
        """W4-017-C02: Numerical status is CONDITIONAL_REAL_ARITHMETIC and margins finite."""
        res_native, diag = self.adapter.compute_correlated_bounds(
            domain=self.domain0,
            receipt=self.receipt0,
            mode=CorrelatedBoundMode.NATIVE_CORRELATED,
        )
        self.assertIsInstance(res_native, BoundResult)
        self.assertEqual(res_native.numerical_status, NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value)
        self.assertEqual(len(res_native.lower_margins), 11)
        self.assertTrue(np.all(np.isfinite(res_native.lower_margins)))

    def test_c03_native_vs_hull_monotonicity_and_labeling(self):
        """W4-017-C03: Native correlated bounds >= outer-box hull bounds; outer-box labeled CORRELATED_HULL."""
        res_native, diag_native = self.adapter.compute_correlated_bounds(
            domain=self.domain0,
            receipt=self.receipt0,
            mode=CorrelatedBoundMode.NATIVE_CORRELATED,
        )
        res_hull, diag_hull = self.adapter.compute_correlated_bounds(
            domain=self.domain0,
            receipt=self.receipt0,
            mode=CorrelatedBoundMode.CORRELATED_HULL,
        )

        # Check labeling
        self.assertEqual(res_native.method_id, "auto-lirpa-crown-native-correlated-v1")
        self.assertEqual(res_hull.method_id, "auto-lirpa-crown-correlated-hull-v1")

        # Check mathematical guarantee: native_margins >= hull_margins
        for m in range(11):
            loss = diag_native["hull_losses"][m]
            self.assertGreaterEqual(loss, -1e-5, f"Margin {m} violated triangle inequality: loss={loss}")
            self.assertGreaterEqual(
                res_native.lower_margins[m],
                res_hull.lower_margins[m] - 1e-5,
                f"Margin {m} native < hull",
            )

        # Confirm strictly positive hull loss exists
        self.assertGreater(diag_native["mean_hull_loss"], 0.0)

    def test_c04_all_11_objectives_and_identity_binding(self):
        """W4-017-C04: Evaluates exactly 11 class margins against winner c* with full identity binding."""
        c_star = self.receipt0.winner
        res, _ = self.engine.compute_correlated_bounds(
            domain=self.domain0,
            receipt=self.receipt0,
            mode="native_correlated",
        )

        self.assertEqual(res.target_class, c_star)
        self.assertEqual(len(res.competitor_classes), 11)
        self.assertNotIn(c_star, res.competitor_classes)
        self.assertEqual(res.reference_identity_id, self.ref.identity.identity_id)
        self.assertEqual(res.domain_id, self.domain0.domain_id)

    def test_c05_toy_correlation_cancellation(self):
        """W4-017-C05: Analytical toy network demonstrates exact 100% correlation cancellation."""
        toy = evaluate_toy_correlation_cancellation()
        self.assertTrue(toy["cancellation_demonstrated"])
        self.assertEqual(toy["native_lower_bound"], 0.0)
        self.assertEqual(toy["native_upper_bound"], 0.0)
        self.assertEqual(toy["hull_lower_bound"], -2.0)
        self.assertEqual(toy["hull_upper_bound"], 2.0)
        self.assertEqual(toy["cancellation_gap"], 2.0)
        self.assertEqual(toy["hull_loss_percentage"], 100.0)

    def test_c06_fail_closed_error_handling_and_no_truncation(self):
        """W4-017-C06: Rejects forged receipts and malformed domains without uncertainty truncation."""
        import dataclasses
        # Test 1: Forged receipt rejection
        bad_receipt = dataclasses.replace(self.receipt0, reference_identity_id="tampered_identity_hash")
        res_fail, _ = self.adapter.compute_correlated_bounds(domain=self.domain0, receipt=bad_receipt)
        self.assertEqual(res_fail.status, BoundStatus.FAILED.value)
        self.assertFalse(res_fail.is_positive)

        # Test 2: Non-finite domain rejection
        bad_c_tail = self.domain0.center_tail.copy()
        bad_c_tail[0] = np.nan
        with self.assertRaises(ValueError):
            AffineTailDomain(
                center_tail=bad_c_tail,
                prefix=self.domain0.prefix,
                params=self.params,
            )

    def test_c07_validation_ablation_and_artifact_generation(self):
        """W4-017-C07: End-to-end validation ablation generates correlated_bound_capability.json and hull_loss_validation.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            cap = run_correlated_bound_capability_study(self.ref, run_id="test_run", output_dir=out_dir)
            hull_val = run_hull_loss_validation(
                self.ref,
                run_id="test_run",
                num_samples=2,
                scales=(0.1, 1.0),
                output_dir=out_dir,
            )

            self.assertTrue((out_dir / "correlated_bound_capability.json").exists())
            self.assertTrue((out_dir / "hull_loss_validation.json").exists())

            self.assertEqual(cap["task_id"], "W4-017")
            self.assertEqual(hull_val["task_id"], "W4-017")
            self.assertTrue(hull_val["summary"]["all_paired_comparisons_pass"])
            self.assertGreater(hull_val["summary"]["overall_mean_hull_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
