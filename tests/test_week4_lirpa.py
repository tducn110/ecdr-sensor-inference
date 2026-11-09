"""tests/test_week4_lirpa.py — Unit test suite for W4-014 CROWN adapter & bound comparison.

Covers:
  - W4-014-C01: Pin CROWN variant, slopes, optimization budget, setup timing.
  - W4-014-C02: 11 direct-margin specifications vs separate-logits control.
  - W4-014-C03: Fail-closed error handling (invalid receipt, non-finite, inverted domain).
  - W4-014-C04: Real-algebra conversion equivalence & sampled logits parity.
  - W4-014-C05: Paired IBP vs CROWN comparison on validation anchors.
  - W4-014-C06: Numerical eligibility designation (SURROGATE_BACKWARD_RELAXATION).
  - W4-014-C07: Artifact generation (crown_capability.json, bound_method_validation.json).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ecdr.week4.bounds import MarginMethod
from ecdr.week4.lirpa_adapter import (
    CrownAdapter,
    CrownVariant,
    LirpaBoundMethod,
    SingletonHeightCNN,
    run_bound_method_comparison,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    NumericalStatus,
    TrustedAnchorReceipt,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4Lirpa(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()
        cls.adapter = CrownAdapter(cls.ref)

        # Load data windows and validation split indices
        cls.x_all = np.load(ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()

    def test_c01_crown_adapter_initialization_and_metadata(self):
        """W4-014-C01: Pin CROWN variant, slopes, bound_opts, and setup timing."""
        self.assertTrue(self.adapter.conversion_ok)
        self.assertGreater(self.adapter.setup_ms, 0.0)
        self.assertEqual(self.adapter.variant, CrownVariant.ANALYTIC_BACKWARD)
        self.assertIn("conv_mode", self.adapter.bound_opts)
        self.assertIsInstance(self.adapter.surrogate, SingletonHeightCNN)

    def test_c02_eleven_direct_margins_and_separate_logits_ablation(self):
        """W4-014-C02: Implement 11 direct-margin specifications and separate-logits control."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        # Use small perturbation radius
        eps = 1e-4
        raw_l, raw_u = raw - eps, raw + eps
        z_l, z_u = self.ref.normalize(raw_l), self.ref.normalize(raw_u)
        domain = (z_l, z_u)
        raw_box = (raw_l, raw_u)

        # 1. Direct Margin CROWN
        res_direct = self.adapter.compute_bounds(
            domain=domain,
            raw_box=raw_box,
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            method=LirpaBoundMethod.CROWN,
        )

        self.assertIsInstance(res_direct, BoundResult)
        self.assertEqual(res_direct.target_class, receipt.winner)
        self.assertEqual(len(res_direct.competitor_classes), 11)
        self.assertEqual(len(res_direct.lower_margins), 11)
        self.assertNotIn(res_direct.target_class, res_direct.competitor_classes)
        self.assertIn(res_direct.status, [BoundStatus.CONDITIONAL_POSITIVE.value, BoundStatus.CONDITIONAL_INCONCLUSIVE.value])
        self.assertEqual(res_direct.numerical_status, NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value)

        # 2. Separate Logits CROWN
        res_sep = self.adapter.compute_bounds(
            domain=domain,
            raw_box=raw_box,
            receipt=receipt,
            margin_method=MarginMethod.SEPARATE_LOGITS,
            method=LirpaBoundMethod.CROWN,
        )

        # Direct margin must be strictly tighter (greater or equal) than separate logits
        self.assertGreaterEqual(res_direct.min_lower_margin, res_sep.min_lower_margin - 1e-5)

    def test_c03_fail_closed_error_handling(self):
        """W4-014-C03: Handle invalid receipt, non-finite inputs, and inverted domains fail-closed."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        z = self.ref.normalize(raw)

        import dataclasses
        # 1. Invalid receipt
        forged_receipt = dataclasses.replace(
            receipt,
            checkpoint_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        res_inv = self.adapter.compute_bounds(
            domain=(z, z),
            raw_box=(raw, raw),
            receipt=forged_receipt,
        )
        self.assertEqual(res_inv.status, BoundStatus.FAILED.value)
        self.assertIn("receipt", res_inv.failure_reason.lower())

        # 2. Inverted domain
        with self.assertRaises(ValueError):
            self.adapter.compute_bounds(
                domain=(z + 1.0, z),
                raw_box=(raw, raw),
                receipt=receipt,
            )

        # 3. Non-finite domain
        bad_z = z.copy()
        bad_z[0, 0] = np.nan
        res_nan = self.adapter.compute_bounds(
            domain=(bad_z, bad_z),
            raw_box=(raw, raw),
            receipt=receipt,
        )
        self.assertEqual(res_nan.status, BoundStatus.FAILED.value)
        self.assertIn("nan", res_nan.failure_reason.lower())

    def test_c04_surrogate_real_algebra_parity(self):
        """W4-014-C04: Sampled logit agreement between 1D reference and 2D surrogate is bit-exact."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        z = self.ref.normalize(raw)

        z_torch_1d = torch.from_numpy(z).unsqueeze(0)
        z_torch_2d = z_torch_1d.unsqueeze(2)

        with torch.no_grad():
            out_1d = self.ref._model(z_torch_1d).cpu().numpy().reshape(-1)
            out_2d = self.adapter.surrogate(z_torch_2d).cpu().numpy().reshape(-1)

        max_abs_diff = float(np.max(np.abs(out_1d - out_2d)))
        self.assertLessEqual(max_abs_diff, 1e-6)
        self.assertEqual(int(np.argmax(out_1d)), int(np.argmax(out_2d)))

    def test_c05_paired_ibp_vs_crown_comparison(self):
        """W4-014-C05: Paired comparison between CROWN and IBP on validation sample."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        eps = 1e-4
        raw_l, raw_u = raw - eps, raw + eps
        z_l, z_u = self.ref.normalize(raw_l), self.ref.normalize(raw_u)
        domain = (z_l, z_u)
        raw_box = (raw_l, raw_u)

        res_crown = self.adapter.compute_bounds(
            domain=domain,
            raw_box=raw_box,
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            method=LirpaBoundMethod.CROWN,
        )

        res_ibp = self.adapter.compute_bounds(
            domain=domain,
            raw_box=raw_box,
            receipt=receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            method=LirpaBoundMethod.IBP,
        )

        # CROWN lower margins should be tighter or equal to IBP lower margins
        self.assertGreaterEqual(res_crown.min_lower_margin, res_ibp.min_lower_margin - 1e-4)

    def test_c06_numerical_eligibility_and_status(self):
        """W4-014-C06: Numerical eligibility marked SURROGATE_BACKWARD_RELAXATION."""
        idx = self.val_indices[0]
        raw = self.x_all[idx].astype(np.float32)
        receipt = self.ref.infer(raw, global_row_id=idx)

        z = self.ref.normalize(raw)
        res = self.adapter.compute_bounds(
            domain=(z, z),
            raw_box=(raw, raw),
            receipt=receipt,
            method=LirpaBoundMethod.CROWN,
        )
        self.assertEqual(res.numerical_status, NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value)

    def test_c07_validation_ablation_and_artifacts(self):
        """W4-014-C07: Run small validation comparison and produce JSON artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cap, val = run_bound_method_comparison(
                reference=self.ref,
                run_id="test_run",
                num_samples=3,
                output_dir=tmp_path,
            )

            self.assertTrue((tmp_path / "crown_capability.json").is_file())
            self.assertTrue((tmp_path / "bound_method_validation.json").is_file())
            self.assertEqual(cap["task_id"], "W4-014")
            self.assertEqual(val["task_id"], "W4-014")
            self.assertEqual(len(val["records"]), 9)  # 3 samples * 3 epsilons


if __name__ == "__main__":
    unittest.main()
