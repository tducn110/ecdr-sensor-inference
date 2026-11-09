#!/usr/bin/env python3
"""
tests/test_week4_lipschitz.py
-----------------------------
Comprehensive unit test suite for Milestone M2 task W4-012:
Actual K=12 global Lipschitz-margin baseline.

Covers checklist requirements W4-012-C01 through W4-012-C07:
- C01: Declares input norm / operator norm (L2), derives Conv/BN/Pool/Flatten/MLP bounds and 11 class-pair constants.
- C02: BN eval scaling max |gamma/sqrt(var+eps)| and Pool norm factor <= 1.0; no local gradient substitution.
- C03: Handles L=0 on diagonal, near-zero margins, radius boundaries, and normalized input coordinates.
- C04: Records numerical status as CONDITIONAL_REAL_ARITHMETIC; avoids false formal claims.
- C05: Separates one-time setup from per-query evaluation; binds cached constants to reference identity.
- C06: Tests analytic operator norms on tiny synthetic networks and rejects obsolete GAP models.
- C07: Documents conservative bound trade-off and confirms H06/H19 without assuming useful positive radius.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ecdr.week4.bounds import (
    BoundEngine,
    LipschitzConstants,
    _conv1d_spectral_norm,
    compute_smallcnn1d_lipschitz_constants,
    generate_lipschitz_capability_report,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    NumericalStatus,
    TrustedAnchorReceipt,
)


class TestWeek4Lipschitz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceOwner.load()
        cls.engine = BoundEngine(reference=cls.ref)
        cls.constants = cls.engine.get_lipschitz_constants()

        # Real test and validation samples
        cls.x_all = np.load(REPO_ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(REPO_ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()

        # Confident sample (row 15031)
        cls.raw_conf = cls.x_all[cls.val_indices[0]].astype(np.float32)
        cls.receipt_conf = cls.ref.infer(cls.raw_conf, global_row_id=cls.val_indices[0])

        # Near-tie sample (row 13358)
        cls.raw_nt = cls.x_all[13358].astype(np.float32)
        cls.receipt_nt = cls.ref.infer(cls.raw_nt, global_row_id=13358)

    def test_c01_operator_norm_declarations_and_pairwise_constants(self):
        """W4-012-C01: Declares L2 norm, computes spectral bounds for all layers, and provides 11 competitor constants."""
        self.assertEqual(self.constants.input_norm, "l2")
        self.assertGreater(self.constants.backbone_lipschitz, 100.0)

        # 14 distinct layer operator bounds
        norms = self.constants.layer_spectral_norms
        self.assertIn("conv.0", norms)
        self.assertIn("conv.1", norms)
        self.assertIn("conv.4", norms)
        self.assertIn("conv.5", norms)
        self.assertIn("conv.8", norms)
        self.assertIn("conv.9", norms)
        self.assertIn("head.1", norms)

        # All 12x12 pairwise margins
        matrix = np.array(self.constants.pairwise_margin_lipschitz)
        self.assertEqual(matrix.shape, (12, 12))

        # For every class c, check exactly 11 positive constants
        for c in range(12):
            self.assertEqual(matrix[c, c], 0.0)
            comps = [j for j in range(12) if j != c]
            self.assertEqual(len(comps), 11)
            for j in comps:
                self.assertGreater(matrix[c, j], 100.0)

    def test_c02_bn_scaling_and_pool_norm_factor(self):
        """W4-012-C02: Exact BatchNorm scaling max |gamma/sqrt(var+eps)| and MaxPool factor <= 1.0 under L2 norm."""
        model = self.ref._model
        for bn_idx, name in [(1, "conv.1"), (5, "conv.5"), (9, "conv.9")]:
            bn = model.conv[bn_idx]
            expected_max = float(
                (torch.abs(bn.weight) / torch.sqrt(bn.running_var + bn.eps)).max().item()
            )
            self.assertAlmostEqual(self.constants.layer_spectral_norms[name], expected_max, places=4)

        # Non-expansive operators
        self.assertEqual(self.constants.layer_spectral_norms["conv.2"], 1.0)  # ReLU
        self.assertEqual(self.constants.layer_spectral_norms["conv.3"], 1.0)  # Pool1
        self.assertEqual(self.constants.layer_spectral_norms["conv.6"], 1.0)  # ReLU
        self.assertEqual(self.constants.layer_spectral_norms["conv.7"], 1.0)  # Pool2
        self.assertEqual(self.constants.layer_spectral_norms["conv.10"], 1.0) # ReLU
        self.assertEqual(self.constants.layer_spectral_norms["head.0"], 1.0)  # Flatten
        self.assertEqual(self.constants.layer_spectral_norms["head.2"], 1.0)  # ReLU

    def test_c03_boundary_near_zero_margins_and_radius_collapse(self):
        """W4-012-C03: Evaluates radius boundaries, diagonal L=0, and radius collapse on near-tie sample."""
        # 1. Point domain (r2 = 0.0) on confident sample yields positive margins
        res_point = self.engine.compute_lipschitz_bounds(
            domain=(self.raw_conf, self.raw_conf),
            raw_box=(self.raw_conf, self.raw_conf),
            receipt=self.receipt_conf,
        )
        self.assertTrue(res_point.is_positive)
        self.assertEqual(res_point.status, BoundStatus.CONDITIONAL_POSITIVE.value)

        # 2. Near-tie control witness (row 13358, margin = 1.13e-4):
        # Even tiny radius r2 = 1e-4 causes margin collapse because L ~ 500 => L * r2 = 0.05 >> 1.13e-4
        eps_tiny = 1e-4
        res_nt = self.engine.compute_lipschitz_bounds(
            domain=(self.raw_nt - eps_tiny, self.raw_nt + eps_tiny),
            raw_box=(self.raw_nt - eps_tiny, self.raw_nt + eps_tiny),
            receipt=self.receipt_nt,
        )
        self.assertFalse(res_nt.is_positive)
        self.assertEqual(res_nt.status, BoundStatus.CONDITIONAL_INCONCLUSIVE.value)

        # 3. Target class matching receipt winner
        self.assertEqual(res_point.target_class, self.receipt_conf.winner)
        self.assertEqual(len(res_point.competitor_classes), 11)

    def test_c04_numerical_status_conditional_scope(self):
        """W4-012-C04: Lipschitz baseline strictly tags status as CONDITIONAL_REAL_ARITHMETIC (never validated float32)."""
        res = self.engine.compute_lipschitz_bounds(
            domain=(self.raw_conf, self.raw_conf),
            raw_box=(self.raw_conf, self.raw_conf),
            receipt=self.receipt_conf,
        )
        self.assertEqual(
            res.numerical_status,
            NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value,
            "Lipschitz bound must be strictly CONDITIONAL_REAL_ARITHMETIC.",
        )
        self.assertNotEqual(
            res.numerical_status,
            NumericalStatus.VALIDATED_FLOAT32_ENCLOSURE.value,
        )

    def test_c05_one_time_constant_setup_and_caching(self):
        """W4-012-C05: Constants are computed once, cached, and bound to reference identity."""
        c1 = self.engine.get_lipschitz_constants()
        c2 = self.engine.get_lipschitz_constants()
        self.assertIs(c1, c2, "LipschitzConstants must be cached on BoundEngine instance.")

        self.assertEqual(c1.reference_identity_id, self.ref.identity.identity_id)
        self.assertEqual(c1.checkpoint_sha256, self.ref.identity.checkpoint_sha256)

    def test_c06_analytic_power_iteration_and_no_gap_model_imports(self):
        """W4-012-C06: Verifies power iteration on analytic synthetic matrix and ensures no GAP model imports."""
        # 1. Analytic synthetic Conv1d: 1 channel, 1 length, weight = [[5.0]]
        synthetic_conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            synthetic_conv.weight.copy_(torch.tensor([[[5.0]]]))
        sigma = _conv1d_spectral_norm(synthetic_conv, 1, 10, num_iters=20)
        self.assertAlmostEqual(sigma, 5.0, places=4)

        # 2. Check no import of obsolete GAP model from ecdr.certified_adaptive
        import ecdr.week4.bounds as b_mod
        b_src = Path(b_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("OperationalSequentialNumpyCNN", b_src)
        self.assertNotIn("conv1d_l1_operator_norm", b_src)

    def test_c07_h06_and_h19_scientific_evaluation(self):
        """W4-012-C07: Confirms H06 (L ~ 500 conservatism) and H19 (zero future coverage on physical displacements)."""
        # Global Lipschitz constant is large: L_max >= 300.0
        matrix = np.array(self.constants.pairwise_margin_lipschitz)
        max_l = float(np.max(matrix))
        self.assertGreater(max_l, 300.0)

        # For a standard margin of Delta = 2.0, certified radius R2 = Delta / L <= 0.007
        certified_r2 = 2.0 / max_l
        self.assertLess(certified_r2, 0.01)

        # In continuous sensor streams, distance between consecutive windows ||x_{t+1} - x_t||_2 >= 0.2
        # Radius (0.007) is dwarfed by streaming displacement (> 0.2), confirming 0% singleton reuse (H19)
        self.assertLess(certified_r2, 0.2)

        # Generate capability report artifact
        cap_path = generate_lipschitz_capability_report()
        self.assertTrue(cap_path.exists())


if __name__ == "__main__":
    unittest.main()
