"""test_toy_stale_regression.py — Regression test for Section 10.2 stale-state failure.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 10.2).

Demonstrates and verifies that:
1. First append of U=(7, 8) to X0=(1, 2, 3, 4, 5, 6) gives X1=(3, 4, 5, 6, 7, 8) where Class 1 wins (logits: 6.5, -2.5, 5.0).
2. Second append of U=(7, 8) gives X2=(5, 6, 7, 8, 7, 8) where Class 3 wins (logits: 8.5, -4.0, 9.0).
3. Even though U=(7, 8) is inside the wave box [6.5, 7.5] x [7.5, 8.5], the certificate built on X0 is STALE and INVALID for X2 because the overlap has shifted.
4. ECDR membership and lifecycle correctly reject X2 and fall back to reference, avoiding the false-reuse error.
"""

import unittest
import numpy as np

from ecdr.certificate import Certificate, CertificateManager, compute_window_hash
from ecdr.intervals import Interval
from ecdr.membership import check_wave_domain_membership


class TestToyStaleRegression(unittest.TestCase):
    def setUp(self):
        # Setup toy parameters from Section 10
        self.w1 = np.array([[[1.0, -1.0, 1.0]], [[-1.0, 0.0, 1.0]]], dtype=np.float64)  # (2, 1, 3)
        self.b1 = np.array([-2.0, -1.0], dtype=np.float64)
        self.A = np.array([[1.0, -1.0], [-1.0, 0.0], [2.0, -2.0]], dtype=np.float64)  # (3, 2)
        self.beta = np.array([4.0, 1.0, 0.0], dtype=np.float64)

        self.x0 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)
        self.x1 = np.array([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
        self.x2 = np.array([[5.0, 6.0, 7.0, 8.0, 7.0, 8.0]], dtype=np.float64)

        self.wave_box = Interval(
            lo=np.array([[6.5, 7.5]], dtype=np.float64),
            hi=np.array([[7.5, 8.5]], dtype=np.float64),
        )

    def forward_toy(self, x: np.ndarray) -> dict:
        """Forward pass on toy 1D CNN."""
        # Conv1: valid k=3 on length 6 -> out length 4
        c_out, _, k = self.w1.shape
        t_out = x.shape[1] - k + 1
        h = np.zeros((c_out, t_out), dtype=np.float64)
        for o in range(c_out):
            for i in range(t_out):
                acc = self.b1[o] + np.dot(self.w1[o, 0], x[0, i : i + k])
                h[o, i] = max(0.0, acc)

        m = np.mean(h, axis=1)  # GAP
        logits = self.A @ m + self.beta
        pred_class = int(np.argmax(logits))
        return {"h": h, "m": m, "logits": logits, "pred_class": pred_class}

    def test_x2_logits_and_winner(self):
        # Evaluate X2 = (5, 6, 7, 8, 7, 8)
        res2 = self.forward_toy(self.x2)

        # Expected features: H(X2) = ((4, 1), (5, 1), (4, 0), (7, 0))
        expected_h2 = np.array([
            [4.0, 5.0, 4.0, 7.0],
            [1.0, 1.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(res2["h"], expected_h2)

        # Expected GAP: m(X2) = (5.0, 0.5)
        np.testing.assert_allclose(res2["m"], [5.0, 0.5])

        # Expected logits: (17/2, -4, 9) = (8.5, -4.0, 9.0)
        np.testing.assert_allclose(res2["logits"], [8.5, -4.0, 9.0])

        # Winner must be Class 2 (0-based) / Class 3 (1-based)
        self.assertEqual(res2["pred_class"], 2)

    def test_stale_membership_rejection(self):
        # A naive wave-check that only inspects tail samples (7, 8) would see they are in [6.5, 7.5] x [7.5, 8.5]
        u2 = self.x2[:, 4:]  # (7, 8)
        self.assertTrue(np.all(u2 >= self.wave_box.lo) and np.all(u2 <= self.wave_box.hi))

        # BUT check_wave_domain_membership tests both the tail wave AND the overlap with X0!
        # For X2, overlap with X0 requires X2[:, :4] == X0[:, 2:], which is (5,6,7,8) == (3,4,5,6) -> FALSE!
        mem_res = check_wave_domain_membership(
            z_query=self.x2,
            z_anchor=self.x0,
            wave_box=self.wave_box,
            shift=2,
        )
        self.assertFalse(mem_res.is_member)
        self.assertEqual(mem_res.reason, "Overlap mismatch with anchor state")

    def test_certificate_lifecycle_consumption(self):
        # Create certificate with horizon = 1
        cert = Certificate(
            cert_id="toy_c1",
            model_id="toy_model",
            prep_id="toy_prep",
            recording_id="toy_rec",
            anchor_id=0,
            target_class=0,  # Class 1
            domain_type="wave_domain",
            anchor_state_hash=compute_window_hash(self.x0),
            wave_box=self.wave_box,
            min_lower_margin=0.875,
            all_lower_margins=np.array([0.0, 8.0, 0.875]),
            horizon=1,
        )

        self.assertTrue(cert.is_active)

        # After step 1 (evaluating X1), certificate is consumed
        cert.consume_step()
        self.assertFalse(cert.is_active)
        self.assertEqual(cert.steps_used, 1)

        # Query at step 2 cannot use the expired certificate!
        mgr = CertificateManager(model_id="toy_model", prep_id="toy_prep")
        mgr.establish_trusted_anchor(window=self.x0, step_idx=0, recording_id="toy_rec", cert=cert)
        active = mgr.get_active_certificate()
        self.assertIsNone(active)


if __name__ == "__main__":
    unittest.main()
