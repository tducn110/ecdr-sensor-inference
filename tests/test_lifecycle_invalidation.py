"""test_lifecycle_invalidation.py — Test trusted anchor lifecycle and invalidation policies.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 10)
and 07_G3_DOMAIN_CONTRACT_VI.md (Section 6).
"""

import unittest
import numpy as np

from ecdr.certificate import Certificate, CertificateManager, compute_window_hash
from ecdr.intervals import Interval


class TestLifecycleInvalidation(unittest.TestCase):
    def setUp(self):
        self.mgr = CertificateManager(model_id="model_v1", prep_id="prep_v1")
        self.margins12 = np.array([0.0, 1.5, 2.0, 1.8, 3.1, 2.2, 1.9, 2.5, 1.6, 2.8, 3.0, 1.7], dtype=np.float64)
        # target=0, competitor min is 1.5
        self.box128 = Interval.from_box(np.zeros((6, 128)), 0.01)
        self.w0 = np.zeros((6, 128))

    def _make_valid_cert(self, target_class=0, min_margin=1.5, recording_id="rec_01", window=None):
        w = self.w0 if window is None else window
        return Certificate(
            cert_id="c1",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id=recording_id,
            anchor_id=0,
            target_class=target_class,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(w),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=min_margin,
            all_lower_margins=self.margins12.copy(),
            horizon=5,
        )

    def test_recording_boundary_reset(self):
        cert = self._make_valid_cert(recording_id="rec_01")
        w = np.zeros((6, 128))
        self.mgr.establish_trusted_anchor(window=w, step_idx=0, recording_id="rec_01", cert=cert)

        # Active certificate on rec_01
        self.assertIsNotNone(self.mgr.get_active_certificate())

        # Boundary transition to rec_02 -> must reset and invalidate!
        self.mgr.reset_recording("rec_02")
        self.assertIsNone(self.mgr.get_active_certificate())
        self.assertFalse(cert.is_active)
        self.assertIsNone(self.mgr.anchor_window)

    def test_horizon_expiration(self):
        u_box = Interval.from_box(np.zeros((6, 64)), 0.01)
        w = np.zeros((6, 128))
        cert = Certificate(
            cert_id="c1",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="wave_domain",
            anchor_state_hash=compute_window_hash(w),
            wave_box=u_box,
            min_lower_margin=1.5,
            all_lower_margins=self.margins12.copy(),
            horizon=2,
        )

        self.assertTrue(cert.is_active)
        cert.consume_step()
        self.assertTrue(cert.is_active)
        self.assertEqual(cert.steps_used, 1)

        cert.consume_step()
        self.assertFalse(cert.is_active)
        self.assertEqual(cert.steps_used, 2)

    def test_compatibility_invalidation(self):
        cert = self._make_valid_cert()
        self.assertTrue(cert.is_compatible("model_v1", "prep_v1"))
        self.assertFalse(cert.is_compatible("model_v2", "prep_v1"))
        self.assertFalse(cert.is_compatible("model_v1", "prep_v2"))

    def test_validate_all11_competitors_positive(self):
        cert = self._make_valid_cert(target_class=0, min_margin=1.5)
        self.assertTrue(cert.validate(expected_k=12))

    def test_validate_rejects_negative_or_zero_competitor(self):
        # Competitor class 3 margin is zero
        bad_margins = self.margins12.copy()
        bad_margins[3] = 0.0
        cert = Certificate(
            cert_id="c_bad",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(self.w0),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=0.0,
            all_lower_margins=bad_margins,
        )
        self.assertFalse(cert.validate(expected_k=12))
        with self.assertRaises(ValueError) as ctx:
            cert.validate(expected_k=12, raise_on_error=True)
        self.assertIn("Not all K-1 competitor margins are strictly positive", str(ctx.exception))

        # Competitor class 3 margin is negative
        bad_margins[3] = -0.05
        cert2 = Certificate(
            cert_id="c_bad2",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(self.w0),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=-0.05,
            all_lower_margins=bad_margins,
        )
        self.assertFalse(cert2.validate(expected_k=12))
        with self.assertRaises(ValueError) as ctx2:
            cert2.validate(expected_k=12, raise_on_error=True)
        self.assertIn("Not all K-1 competitor margins are strictly positive", str(ctx2.exception))

    def test_validate_rejects_scalar_vector_inconsistency(self):
        # Vector min is 1.5, but scalar min claims 2.0
        cert = Certificate(
            cert_id="c_inconsistent",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(self.w0),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=2.0,  # inconsistent!
            all_lower_margins=self.margins12.copy(),
        )
        self.assertFalse(cert.validate(expected_k=12))
        with self.assertRaises(ValueError) as ctx:
            cert.validate(expected_k=12, raise_on_error=True)
        self.assertIn("inconsistent with vector min", str(ctx.exception))

    def test_validate_rejects_nonfinite_or_missing_margins(self):
        # Missing vector
        cert_none = Certificate(
            cert_id="c_none",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(self.w0),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=1.5,
            all_lower_margins=None,
        )
        self.assertFalse(cert_none.validate(expected_k=12))
        with self.assertRaises(ValueError) as ctx_none:
            cert_none.validate(expected_k=12, raise_on_error=True)
        self.assertIn("all_lower_margins vector is missing", str(ctx_none.exception))

        # NaN in vector
        nan_margins = self.margins12.copy()
        nan_margins[2] = np.nan
        cert_nan = Certificate(
            cert_id="c_nan",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(self.w0),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=1.5,
            all_lower_margins=nan_margins,
        )
        self.assertFalse(cert_nan.validate(expected_k=12))
        with self.assertRaises(ValueError) as ctx_nan:
            cert_nan.validate(expected_k=12, raise_on_error=True)
        self.assertIn("all_lower_margins contains non-finite", str(ctx_nan.exception))

    def test_validate_rejects_tampered_integrity_hash(self):
        cert = self._make_valid_cert()
        self.assertTrue(cert.validate())
        # Simulate tampering with target_class without recomputing hash
        cert.target_class = 2
        self.assertFalse(cert.validate())
        with self.assertRaises(ValueError):
            cert.validate(raise_on_error=True)

    def test_establish_trusted_anchor_rejects_invalid_certificate(self):
        bad_margins = self.margins12.copy()
        bad_margins[1] = -1.0  # negative margin
        cert = Certificate(
            cert_id="c_bad",
            model_id="model_v1",
            prep_id="prep_v1",
            recording_id="rec_01",
            anchor_id=0,
            target_class=0,
            domain_type="diagonal_box",
            anchor_state_hash=compute_window_hash(self.w0),
            domain_box=self.box128,
            budget_norm=np.full(6, 0.01),
            min_lower_margin=-1.0,
            all_lower_margins=bad_margins,
        )
        w = np.zeros((6, 128))
        with self.assertRaises(ValueError):
            self.mgr.establish_trusted_anchor(window=w, step_idx=0, recording_id="rec_01", cert=cert)
        self.assertIsNone(self.mgr.get_active_certificate())

    def test_establish_trusted_anchor_rejects_window_content_mismatch(self):
        cert = self._make_valid_cert(window=self.w0)
        # Attempt to attach cert built on self.w0 (all zeros) to a different window
        w_different = np.ones((6, 128))
        with self.assertRaises(ValueError) as ctx:
            self.mgr.establish_trusted_anchor(window=w_different, step_idx=0, recording_id="rec_01", cert=cert)
        self.assertIn("Anchor window content mismatch", str(ctx.exception))
        self.assertIsNone(self.mgr.get_active_certificate())


if __name__ == "__main__":
    unittest.main()
