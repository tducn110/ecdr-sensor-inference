"""test_bounds_all_competitors.py — Test margin bounds across all K-1 competitors and boundary gates.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 8, 9, 11).
"""

import unittest
import numpy as np
import torch

from ecdr.bounds import (
    compute_diagonal_box_certificate,
    compute_wave_domain_certificate,
)
from ecdr.intervals import Interval
from ecdr.model import ECDRConv1DNet, OperationalSequentialNumpyCNN


class TestBoundsAllCompetitors(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch_net = ECDRConv1DNet(num_classes=12)
        self.net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)

    def test_all_11_competitors_computed(self):
        np.random.seed(42)
        z = np.random.randn(6, 128).astype(np.float64)

        # Extremely small budget to ensure positive margins
        res = compute_diagonal_box_certificate(
            net=self.net, z_anchor=z, budget_norm=1e-6, margin_threshold=0.0
        )

        # Target class must have 0 margin, and exactly 11 competitors
        c = res.target_class
        self.assertEqual(res.direct_lower_margins[c], 0.0)
        self.assertEqual(len(res.direct_lower_margins), 12)

        competitors = [j for j in range(12) if j != c]
        self.assertEqual(len(competitors), 11)

    def test_negative_margin_threshold_rejected(self):
        z = np.zeros((6, 128), dtype=np.float64)

        # Negative threshold must be rejected
        res = compute_diagonal_box_certificate(
            net=self.net, z_anchor=z, budget_norm=0.01, margin_threshold=-1.0
        )
        self.assertFalse(res.is_certified)
        self.assertEqual(res.min_direct_margin, -np.inf)
        self.assertIn("error", res.diagnostics)

    def test_invalid_target_class_rejected(self):
        z = np.zeros((6, 128), dtype=np.float64)

        res_neg = compute_diagonal_box_certificate(
            net=self.net, z_anchor=z, budget_norm=0.01, target_class=-1
        )
        self.assertFalse(res_neg.is_certified)

        res_oob = compute_diagonal_box_certificate(
            net=self.net, z_anchor=z, budget_norm=0.01, target_class=12
        )
        self.assertFalse(res_oob.is_certified)

    def test_invalid_shift_rejected(self):
        z = np.zeros((6, 128), dtype=np.float64)
        wave_box = Interval.from_box(np.zeros((6, 64)), 0.01)

        res_zero = compute_wave_domain_certificate(
            net=self.net, z_anchor=z, wave_box=wave_box, shift=0
        )
        self.assertFalse(res_zero.is_certified)

        res_too_large = compute_wave_domain_certificate(
            net=self.net, z_anchor=z, wave_box=wave_box, shift=121
        )
        self.assertFalse(res_too_large.is_certified)

    def test_preactivation_overflow_rejected_before_relu(self):
        # Create an operational network whose weights cause overflow with large input
        w1_huge = np.zeros((32, 6, 5), dtype=np.float64)
        w1_huge[0, 0, 0] = -1e308
        b1 = np.zeros(32, dtype=np.float64)
        w2 = np.zeros((64, 32, 3), dtype=np.float64)
        b2 = np.zeros(64, dtype=np.float64)
        w3 = np.zeros((64, 64, 3), dtype=np.float64)
        b3 = np.zeros(64, dtype=np.float64)
        w_head = np.zeros((12, 64), dtype=np.float64)
        b_head = np.zeros(12, dtype=np.float64)

        overflow_net = OperationalSequentialNumpyCNN(
            w1_huge, b1, w2, b2, w3, b3, w_head, b_head
        )

        x_huge = np.zeros((6, 128), dtype=np.float64)
        x_huge[0, 0] = 1e308

        # In standard forward, -1e308 * 1e308 = -inf.
        # If checked after ReLU, max(0, -inf) = 0.0, hiding the overflow!
        # Our preactivation check MUST catch it before ReLU!
        with self.assertRaises(ValueError) as ctx:
            overflow_net.forward(x_huge)
        self.assertIn("before ReLU", str(ctx.exception))


    def test_diagonal_rounding_counterexample_rejected(self):
        # Anchor all -1, radius 1. Query has 1e-100 on channel 0, others 0.
        # In rounded float distance: abs(1e-100 - (-1)) = 1.0, which would falsely pass <= 1.0!
        # But Interval upper bound is nextafter(-1 + 1, inf) = 5e-324 < 1e-100.
        # Strict endpoint check must reject it!
        from ecdr.membership import check_diagonal_box_membership
        anchor = np.full((6, 128), -1.0)
        query = np.zeros((6, 128))
        query[0, :] = 1e-100

        mem_res = check_diagonal_box_membership(z_query=query, z_anchor=anchor, budget_norm=1.0)
        self.assertFalse(mem_res.is_member)
        self.assertIn("falls outside certified interval endpoints", mem_res.reason)


if __name__ == "__main__":
    unittest.main()
