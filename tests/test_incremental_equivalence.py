"""test_incremental_equivalence.py — Test exact incremental shift/cache equivalence and anti-aliasing.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 7, 8, 11).
"""

import unittest
import numpy as np
import torch

from ecdr.incremental import IncrementalEngine
from ecdr.model import ECDRConv1DNet, OperationalSequentialNumpyCNN


class TestIncrementalEquivalence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch_net = ECDRConv1DNet(num_classes=12)
        self.net = OperationalSequentialNumpyCNN.from_torch_module(torch_net)
        self.engine = IncrementalEngine(net=self.net, shift=64)

    def test_bitwise_equivalence_on_shifted_stream(self):
        # Create a continuous time stream of length 128 + 64*4 = 384 samples
        np.random.seed(42)
        stream = np.random.randn(6, 384).astype(np.float64)

        w0 = stream[:, 0:128]
        w1 = stream[:, 64:192]
        w2 = stream[:, 128:256]
        w3 = stream[:, 192:320]

        # 1. First window: full forward
        res0 = self.engine.forward_full(w0, step_idx=0)
        ref0 = self.net.forward(w0)
        np.testing.assert_array_equal(res0["logits"], ref0["logits"])

        # 2. Sequential shifted windows: incremental forward vs full forward
        for idx, w in enumerate([w1, w2, w3], start=1):
            inc_res = self.engine.forward_incremental(w, step_idx=idx)
            full_res = self.net.forward(w)

            # Execution mode must be incremental
            self.assertEqual(inc_res["mode"], "incremental")
            self.assertEqual(inc_res["mac_count"], 1241856)

            # Check bitwise match of intermediate activations H1, H2, H3
            np.testing.assert_array_equal(inc_res["h1"], full_res["h1"])
            np.testing.assert_array_equal(inc_res["h2"], full_res["h2"])
            np.testing.assert_array_equal(inc_res["h3"], full_res["h3"])

            # Check bitwise match of canonical GAP and logits
            np.testing.assert_array_equal(inc_res["m"], full_res["m"])
            np.testing.assert_array_equal(inc_res["logits"], full_res["logits"])
            self.assertEqual(inc_res["pred_class"], full_res["pred_class"])

            # Diagnostic GAP delta discrepancy is bounded by float roundoff
            self.assertLess(inc_res["gap_discrepancy"], 1e-12)

    def test_overlap_perturbation_triggers_fallback(self):
        w0 = np.random.randn(6, 128).astype(np.float64)
        self.engine.forward_full(w0, step_idx=0)

        # Shifted window with tiny 5e-13 perturbation in overlap
        w1 = np.zeros((6, 128), dtype=np.float64)
        w1[:, :64] = w0[:, 64:] + 5e-13
        w1[:, 64:] = np.random.randn(6, 64)

        res = self.engine.forward_incremental(w1, step_idx=1)
        # Must fall back to full forward because bitwise overlap is violated
        self.assertEqual(res["mode"], "full")
        self.assertEqual(res["mac_count"], 2343936)

    def test_nan_overlap_triggers_fallback(self):
        w0 = np.random.randn(6, 128).astype(np.float64)
        self.engine.forward_full(w0, step_idx=0)

        # Shifted window with NaN in overlap
        w1 = np.zeros((6, 128), dtype=np.float64)
        w1[:, :64] = w0[:, 64:]
        w1[0, 0] = np.nan
        w1[:, 64:] = np.random.randn(6, 64)

        # Must be rejected by upfront finiteness check
        with self.assertRaises(ValueError):
            self.engine.forward_incremental(w1, step_idx=1)

    def test_anti_aliasing_detached_payload(self):
        w0 = np.random.randn(6, 128).astype(np.float64)
        res = self.engine.forward_full(w0, step_idx=0)

        # Caller mutates returned dictionary arrays
        res["h3"][0, 0] = 999999.0
        res["logits"][0] = -88888.0

        # Private cache state must NOT be corrupted
        self.assertNotEqual(self.engine.state.h3[0, 0], 999999.0)
        self.assertNotEqual(self.engine.state.logits[0], -88888.0)


if __name__ == "__main__":
    unittest.main()
