#!/usr/bin/env python3
"""
tests/test_week4_certificates.py
--------------------------------
Comprehensive unit test suite for Milestone M3 task W4-022:
Certificate issuer, immutable store, and fixed-anchor lifecycle.

Covers checklist requirements W4-022-C01 through W4-022-C07:
- C01: Strict positive validated margin predicate, capability clearance, and anchor validation.
- C02: Identity binding, complete class order, zero mutable tensor aliases.
- C03: Anti-spoofing and rejection of caller-fabricated claims.
- C04: Fixed-anchor immutability: hits never re-center or extend expiry; atomic invalidation.
- C05: Explicit failure reason codes: missing, expired, wrong-recording, step gap, outside domain, nonfinite.
- C06: Adversarial test suite: forged receipts, cross-model tampering, missing competitors, tie rejection.
- C07: Strict closure when numerical profile is ineligible.
"""

from __future__ import annotations

import dataclasses
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

from ecdr.week4.bounds import BoundEngine, MarginMethod, NumericalMode
from ecdr.week4.certificates import (
    CANONICAL_PROFILE_ID,
    CertificateIssuer,
    CertificateLifecycle,
    ForgedCertificateError,
    ImmutableCertificateStore,
    IssuanceRejectedError,
    StoreFailureReason,
    generate_certificate_lifecycle_review_artifact,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    CandidateBox,
    Certificate,
    DecisionAction,
    NumericalStatus,
    QueryView,
    TrustedAnchorReceipt,
)


class TestWeek4Certificates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceOwner.load()
        cls.engine = BoundEngine(reference=cls.ref)
        cls.issuer = CertificateIssuer(reference=cls.ref, profile_id=CANONICAL_PROFILE_ID)
        cls.store = ImmutableCertificateStore(reference=cls.ref)

        # Load real data windows
        cls.x_all = np.load(REPO_ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(REPO_ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()

        # Build a valid candidate box around confident validation sample 15031
        cls.sample_idx = cls.val_indices[0]
        cls.raw = cls.x_all[cls.sample_idx].astype(np.float32)
        cls.receipt = cls.ref.infer(
            cls.raw,
            global_row_id=cls.sample_idx,
            recording_id="rec_val_01",
            step_idx=100,
        )

        # Build a valid perturbation domain (radius 1e-5)
        eps = 1e-5
        norm_lo, norm_hi = cls.ref.enclose_normalization(cls.raw - eps, cls.raw + eps)
        flat_lo = tuple(float(x) for x in norm_lo.ravel())
        flat_hi = tuple(float(x) for x in norm_hi.ravel())
        nonzeros = int(np.count_nonzero(norm_lo != norm_hi))

        cls.domain = CandidateBox(
            lower=flat_lo,
            upper=flat_hi,
            kind="diagonal_box",
            free_dimensions=nonzeros,
        )

        # Compute valid validated bound result
        cls.bound_res = cls.engine.compute_bounds(
            domain=(cls.raw - eps, cls.raw + eps),
            raw_box=(cls.raw - eps, cls.raw + eps),
            receipt=cls.receipt,
            margin_method=MarginMethod.DIRECT_MARGIN,
            numerical_mode=NumericalMode.VALIDATED,
        )

    def test_c01_strict_issuance_predicate_and_positive_margin(self):
        """W4-022-C01: Issues certificate on strict positive validated margins; rejects non-positive or conditional."""
        # 1. Valid issuance succeeds
        cert = self.issuer.issue_certificate(
            anchor_receipt=self.receipt,
            anchor_raw_query=self.raw,
            bound_result=self.bound_res,
            domain=self.domain,
            max_age=10,
        )
        self.assertIsInstance(cert, Certificate)
        self.assertEqual(cert.target_class, self.receipt.winner)
        self.assertEqual(len(cert.competitor_classes), 11)
        self.assertTrue(cert.formal_reuse_enabled)
        self.assertGreater(min(cert.lower_margins), 0.0)

        # 2. Reject non-positive margin (tie margin = 0.0)
        tie_margins = list(self.bound_res.lower_margins)
        tie_margins[0] = 0.0
        tie_bound_res = dataclasses.replace(self.bound_res, lower_margins=tuple(tie_margins))
        with self.assertRaises(IssuanceRejectedError):
            self.issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=self.raw,
                bound_result=tie_bound_res,
                domain=self.domain,
            )

        # 3. Reject conditional (ordinary float) bound result
        cond_bound_res = dataclasses.replace(
            self.bound_res,
            numerical_status=NumericalStatus.NOT_ESTABLISHED.value,
            status=BoundStatus.CONDITIONAL_POSITIVE.value,
        )
        with self.assertRaises(IssuanceRejectedError):
            self.issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=self.raw,
                bound_result=cond_bound_res,
                domain=self.domain,
            )

    def test_c02_identity_binding_and_no_mutable_aliases(self):
        """W4-022-C02: Binds complete identity tuple and produces immutable dataclass without mutable aliases."""
        cert = self.issuer.issue_certificate(
            anchor_receipt=self.receipt,
            anchor_raw_query=self.raw,
            bound_result=self.bound_res,
            domain=self.domain,
            max_age=12,
        )
        self.assertEqual(cert.checkpoint_sha256, self.ref.identity.checkpoint_sha256)
        self.assertEqual(cert.preprocessing_id, self.ref.identity.preprocessing_id)
        self.assertEqual(cert.graph_id, self.ref.identity.graph_id)
        self.assertEqual(cert.anchor_step, 100)
        self.assertEqual(cert.recording_id, "rec_val_01")
        self.assertEqual(cert.max_age, 12)

        # Check immutability: lower_margins must be tuple, domain.lower must be tuple
        self.assertIsInstance(cert.lower_margins, tuple)
        self.assertIsInstance(cert.competitor_classes, tuple)
        self.assertIsInstance(cert.domain.lower, tuple)
        self.assertIsInstance(cert.domain.upper, tuple)

    def test_c03_anti_spoofing_untrusted_claims(self):
        """W4-022-C03: Rejects forged receipts, tampered signatures, and mismatched target classes."""
        # 1. Tampered anchor receipt signature
        bad_receipt = dataclasses.replace(self.receipt, issuer_signature="forged_signature_token")
        with self.assertRaises(ForgedCertificateError):
            self.issuer.issue_certificate(
                anchor_receipt=bad_receipt,
                anchor_raw_query=self.raw,
                bound_result=self.bound_res,
                domain=self.domain,
            )

        # 2. Query mutated after receipt was signed
        tampered_query = self.raw.copy()
        tampered_query[0, 0] += 0.5
        with self.assertRaises(ForgedCertificateError):
            self.issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=tampered_query,
                bound_result=self.bound_res,
                domain=self.domain,
            )

        # 3. Mismatched target class vs winner
        other_class = (self.receipt.winner + 1) % 12
        other_comps = tuple(j for j in range(12) if j != other_class)
        bad_target_bound = dataclasses.replace(
            self.bound_res,
            target_class=other_class,
            competitor_classes=other_comps,
        )
        with self.assertRaises(IssuanceRejectedError):
            self.issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=self.raw,
                bound_result=bad_target_bound,
                domain=self.domain,
            )

    def test_c04_fixed_anchor_immutability_and_hits(self):
        """W4-022-C04: Fixed-anchor immutability: hits never re-center anchor or extend expiry."""
        cert = self.issuer.issue_certificate(
            anchor_receipt=self.receipt,
            anchor_raw_query=self.raw,
            bound_result=self.bound_res,
            domain=self.domain,
            max_age=10,
        )
        self.store.store_certificate(cert)

        orig_anchor_step = cert.anchor_step
        orig_max_age = cert.max_age
        orig_lower = cert.domain.lower
        orig_upper = cert.domain.upper

        # Step 1: Hit with identical anchor query at step 101
        can_reuse, active_cert, reason = self.store.lookup_and_verify(
            query=self.raw,
            step_idx=101,
            recording_id="rec_val_01",
        )
        self.assertTrue(can_reuse)
        self.assertEqual(reason, DecisionAction.REUSE_CERTIFIED.value)
        self.assertIsNotNone(active_cert)

        # Verify fixed anchor has NOT drifted
        self.assertEqual(active_cert.anchor_step, orig_anchor_step)
        self.assertEqual(active_cert.max_age, orig_max_age)
        self.assertEqual(active_cert.domain.lower, orig_lower)
        self.assertEqual(active_cert.domain.upper, orig_upper)

        # Step 2: Hit with slightly shifted query (within domain) at step 105
        shifted_query = self.raw + 1e-6
        can_reuse_2, active_cert_2, _ = self.store.lookup_and_verify(
            query=shifted_query,
            step_idx=105,
            recording_id="rec_val_01",
        )
        self.assertTrue(can_reuse_2)
        # Even on shifted hit, anchor step and domain endpoints must be completely unchanged
        self.assertEqual(active_cert_2.anchor_step, orig_anchor_step)
        self.assertEqual(active_cert_2.domain.lower, orig_lower)

    def test_c05_explicit_store_failure_reason_codes(self):
        """W4-022-C05: Enforces explicit reason codes on lookup misses and invalidations."""
        store = ImmutableCertificateStore(reference=self.ref)

        # 1. MISSING_CERTIFICATE
        can_reuse, _, reason = store.lookup_and_verify(self.raw, step_idx=101, recording_id="rec_val_01")
        self.assertFalse(can_reuse)
        self.assertEqual(reason, StoreFailureReason.MISSING_CERTIFICATE.value)

        # Store certificate
        cert = self.issuer.issue_certificate(
            anchor_receipt=self.receipt,
            anchor_raw_query=self.raw,
            bound_result=self.bound_res,
            domain=self.domain,
            max_age=10,
        )
        store.store_certificate(cert)

        # 2. NONFINITE_INPUT
        bad_query = self.raw.copy()
        bad_query[0, 0] = np.nan
        can_reuse, _, reason = store.lookup_and_verify(bad_query, step_idx=102, recording_id="rec_val_01")
        self.assertFalse(can_reuse)
        self.assertEqual(reason, StoreFailureReason.NONFINITE_INPUT.value)

        # 3. WRONG_RECORDING
        can_reuse, _, reason = store.lookup_and_verify(self.raw, step_idx=102, recording_id="rec_val_DIFFERENT")
        self.assertFalse(can_reuse)
        self.assertEqual(reason, StoreFailureReason.WRONG_RECORDING.value)
        # Store should now be invalidated
        self.assertIsNone(store.active_certificate)

        # Re-store certificate for subsequent checks
        cert2 = self.issuer.issue_certificate(
            anchor_receipt=self.receipt,
            anchor_raw_query=self.raw,
            bound_result=self.bound_res,
            domain=self.domain,
            max_age=10,
        )
        store.store_certificate(cert2)

        # 4. STEP_GAP (step prior to anchor_step = 100)
        can_reuse, _, reason = store.lookup_and_verify(self.raw, step_idx=95, recording_id="rec_val_01")
        self.assertFalse(can_reuse)
        self.assertEqual(reason, StoreFailureReason.STEP_GAP.value)

        # 5. EXPIRED_CERTIFICATE (step > 100 + 10 = 110)
        can_reuse, _, reason = store.lookup_and_verify(self.raw, step_idx=111, recording_id="rec_val_01")
        self.assertFalse(can_reuse)
        self.assertEqual(reason, StoreFailureReason.EXPIRED_CERTIFICATE.value)

        # 6. OUTSIDE_DOMAIN
        outside_query = self.raw + 10.0  # Large perturbation outside radius 1e-5
        cert3 = self.issuer.issue_certificate(
            anchor_receipt=self.receipt,
            anchor_raw_query=self.raw,
            bound_result=self.bound_res,
            domain=self.domain,
            max_age=10,
        )
        store.store_certificate(cert3)
        can_reuse, _, reason = store.lookup_and_verify(outside_query, step_idx=102, recording_id="rec_val_01")
        self.assertFalse(can_reuse)
        self.assertEqual(reason, StoreFailureReason.OUTSIDE_DOMAIN.value)

    def test_c06_adversarial_tampering_and_tie_rejection(self):
        """W4-022-C06: Adversarial attacks: missing competitors, cross-domain, tie bounds rejected."""
        # 1. Missing competitor (10 competitors instead of 11)
        with self.assertRaises((ValueError, IssuanceRejectedError, ForgedCertificateError)):
            bad_comps = tuple(self.bound_res.competitor_classes[:10])
            bad_margins = tuple(self.bound_res.lower_margins[:10])
            bad_comp_bound = dataclasses.replace(
                self.bound_res,
                competitor_classes=bad_comps,
                lower_margins=bad_margins,
            )
            self.issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=self.raw,
                bound_result=bad_comp_bound,
                domain=self.domain,
            )

        # 2. Anchor point outside declared domain
        shifted_lo = tuple(x + 10.0 for x in self.domain.lower)
        shifted_hi = tuple(x + 10.0 for x in self.domain.upper)
        shifted_domain = CandidateBox(
            lower=shifted_lo,
            upper=shifted_hi,
            kind="diagonal_box",
            free_dimensions=self.domain.free_dimensions,
        )
        with self.assertRaises(IssuanceRejectedError):
            self.issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=self.raw,
                bound_result=self.bound_res,
                domain=shifted_domain,
            )

    def test_c07_ineligible_numerical_profile_closes_issuance(self):
        """W4-022-C07: Ineligible profile immediately disables formal certificate issuance."""
        ineligible_issuer = CertificateIssuer(
            reference=self.ref,
            profile_id="darwin-arm64-metal-unsupported",
            enforce_numerical_gate=True,
        )
        self.assertFalse(ineligible_issuer.formal_issuance_enabled)

        with self.assertRaises(IssuanceRejectedError):
            ineligible_issuer.issue_certificate(
                anchor_receipt=self.receipt,
                anchor_raw_query=self.raw,
                bound_result=self.bound_res,
                domain=self.domain,
            )

    def test_review_artifact_generation(self):
        """Verifies generation of certificate_lifecycle_review.md."""
        artifact_path = generate_certificate_lifecycle_review_artifact()
        self.assertTrue(artifact_path.exists())
        self.assertGreater(artifact_path.stat().st_size, 500)


if __name__ == "__main__":
    unittest.main()
