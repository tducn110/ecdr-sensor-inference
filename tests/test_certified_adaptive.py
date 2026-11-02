"""Tests for the conservative margin/Lipschitz adaptive cache."""

import unittest

import numpy as np
import torch

from ecdr.certified_adaptive import (
    CertifiedAdaptiveCache,
    certified_radius_from_logits,
    conv1d_l1_operator_norm,
    margin_lipschitz_bounds,
)
from ecdr.model import ECDRConv1DNet, OperationalSequentialNumpyCNN


class CertifiedAdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.net = OperationalSequentialNumpyCNN.from_torch_module(ECDRConv1DNet(num_classes=12))

    def test_conv_l1_norm_is_induced_bound(self):
        w = np.zeros((2, 2, 3), dtype=np.float64)
        w[0, 0] = [1.0, -2.0, 3.0]
        w[1, 0] = [-4.0, 0.5, 1.0]
        w[0, 1] = [0.25, 0.0, 0.0]
        self.assertGreaterEqual(conv1d_l1_operator_norm(w), 11.5)
        self.assertAlmostEqual(conv1d_l1_operator_norm(w), 11.5, places=12)

    def test_margin_radius_is_positive_only_for_strict_margin(self):
        bounds = np.ones((2, 2), dtype=np.float64)
        self.assertAlmostEqual(
            certified_radius_from_logits(np.array([2.0, 1.0]), 0, bounds, input_size=2),
            np.nextafter(0.5, 0.0),
        )
        self.assertEqual(certified_radius_from_logits(np.array([1.0, 1.0]), 0, bounds), 0.0)

    def test_global_bounds_and_cache_lifecycle(self):
        bounds = margin_lipschitz_bounds(self.net)
        self.assertEqual(bounds.shape, (12, 12))
        self.assertTrue(np.all(np.isfinite(bounds)))
        np.testing.assert_array_equal(np.diag(bounds), np.zeros(12))

        cache = CertifiedAdaptiveCache(self.net)
        x = np.zeros((6, 128), dtype=np.float64)
        first = cache.predict(x, recording_id="r1", step_idx=0)
        self.assertFalse(first["is_hit"])
        second = cache.predict(x.copy(), recording_id="r1", step_idx=1)
        self.assertTrue(second["is_hit"])
        after_gap = cache.predict(x, recording_id="r1", step_idx=3)
        self.assertFalse(after_gap["is_hit"])
        after_recording = cache.predict(x, recording_id="r2", step_idx=0)
        self.assertFalse(after_recording["is_hit"])


if __name__ == "__main__":
    unittest.main()

