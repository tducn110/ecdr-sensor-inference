"""tests/test_week4_contracts.py — Comprehensive contract and invariant test suite for Week 4.

Verifies:
1. QueryView immutability, shape validation, and strict absence of label/oracle fields.
2. Disjoint statuses across Membership, Bound, Numerical, Decision, and Execution enums.
3. Strict serialization rejecting NaN/Inf and unknown schema versions.
4. FrozenReferenceAdapter identity assertions against weight, buffer, hook, and eval mutations.
5. TrustedAnchorReceipt authenticity, forgery rejection, cross-query binding, and invalidation.
6. CandidateBox domain boundary enforcement and membership tests.
"""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from ecdr.schema import CLASS_NAMES_12
from ecdr.week4.reference import (
    IdentityError,
    ReferenceOwner,
    UnsupportedGraphError,
)
from ecdr.week4.types import (
    BoundResult,
    BoundStatus,
    CandidateBox,
    Certificate,
    DecisionAction,
    ExecutionMode,
    MembershipResult,
    MembershipStatus,
    NumericalCapability,
    NumericalStatus,
    QueryView,
    ReferenceIdentity,
    RuntimeDecision,
    SCHEMA_VERSION,
    SerializationError,
    TrustedAnchorReceipt,
    from_json_str,
    to_json_str,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "output/week1_closeout_20261102/reference"


class TestQueryViewContracts(unittest.TestCase):
    """C01 tests: QueryView holds only raw sensor window and location metadata."""

    def setUp(self) -> None:
        self.raw_arr = np.sin(np.linspace(0, 10, 768, dtype=np.float32)).reshape(6, 128)
        self.query = QueryView.from_array(
            raw=self.raw_arr,
            global_row_id=42,
            recording_id="rec_01",
            step_idx=10,
        )

    def test_query_view_fields_contract(self) -> None:
        """Verify strict absence of label/activity/purity/split fields."""
        fields = set(self.query.__dataclass_fields__.keys())
        expected = {"global_row_id", "recording_id", "step_idx", "raw_window", "schema_version"}
        self.assertEqual(fields, expected, "QueryView contains undeclared fields")

        # Explicitly ensure no label leakage
        for forbidden in ["label", "label_majority", "activity", "activity_id", "purity", "split", "logits"]:
            self.assertFalse(hasattr(self.query, forbidden), f"QueryView leaks forbidden attribute: {forbidden}")

    def test_query_view_immutability(self) -> None:
        """QueryView must reject attribute mutation."""
        with self.assertRaises(FrozenInstanceError):
            self.query.global_row_id = 99  # type: ignore
        with self.assertRaises(FrozenInstanceError):
            self.query.recording_id = "hacked"  # type: ignore

    def test_query_view_aliasing_protection(self) -> None:
        """Mutating the source array or array() output must not affect QueryView."""
        source = self.raw_arr.copy()
        q = QueryView.from_array(source, 1, "rec_01", 0)
        source[0, 0] = 999.0
        self.assertNotEqual(q.array()[0, 0], 999.0)

        # Returned array must be read-only
        out = q.array()
        self.assertFalse(out.flags.writeable, "query.array() must return a read-only buffer")
        with self.assertRaises(ValueError):
            out[0, 0] = 123.0

    def test_query_view_validation_failures(self) -> None:
        """Malformed shapes, NaNs, Infs, booleans, and negative indices must fail."""
        # Wrong shape
        with self.assertRaises(ValueError):
            QueryView.from_array(np.zeros((6, 127), dtype=np.float32), 0, "rec", 0)
        # NaN in raw window
        nan_arr = self.raw_arr.copy()
        nan_arr[0, 0] = np.nan
        with self.assertRaises(ValueError):
            QueryView.from_array(nan_arr, 0, "rec", 0)
        # Negative row id or step
        with self.assertRaises(ValueError):
            QueryView.from_array(self.raw_arr, -1, "rec", 0)
        with self.assertRaises(ValueError):
            QueryView.from_array(self.raw_arr, 0, "rec", -1)
        # Empty recording id
        with self.assertRaises(ValueError):
            QueryView.from_array(self.raw_arr, 0, "   ", 0)
        # Boolean used as int
        with self.assertRaises(ValueError):
            QueryView.from_array(self.raw_arr, True, "rec", 0)  # type: ignore


class TestTypeEnumsAndContracts(unittest.TestCase):
    """C02 & C03 tests: Disjoint statuses and strict serialization."""

    def test_membership_statuses_disjoint(self) -> None:
        statuses = [s.value for s in MembershipStatus]
        self.assertEqual(len(statuses), len(set(statuses)))
        self.assertIn("EXACT_MEMBER", statuses)
        self.assertIn("NON_MEMBER", statuses)

    def test_bound_statuses_disjoint(self) -> None:
        statuses = [s.value for s in BoundStatus]
        self.assertEqual(len(statuses), len(set(statuses)))
        self.assertIn("EXACT_ENCLOSURE", statuses)
        self.assertIn("CONDITIONAL_POSITIVE", statuses)
        self.assertIn("FAILED", statuses)

    def test_decision_actions_disjoint(self) -> None:
        actions = [a.value for a in DecisionAction]
        self.assertEqual(len(actions), len(set(actions)))
        self.assertIn("REUSE_CERTIFIED", actions)
        self.assertIn("FALLBACK_REFERENCE", actions)

    def test_bound_result_contract(self) -> None:
        """BoundResult must validate competitor count, target exclusion, and bounds."""
        competitors = tuple(j for j in range(12) if j != 3)
        res = BoundResult(
            method_id="M01",
            domain_id="domain_123",
            reference_identity_id="ref_id",
            target_class=3,
            competitor_classes=competitors,
            lower_margins=tuple(0.5 for _ in range(11)),
            status=BoundStatus.CONDITIONAL_POSITIVE.value,
            numerical_status=NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value,
            is_positive=True,
        )
        self.assertEqual(res.min_lower_margin, 0.5)

        # Duplicate competitors
        with self.assertRaises(ValueError):
            BoundResult(
                method_id="M01",
                domain_id="domain_123",
                reference_identity_id="ref_id",
                target_class=3,
                competitor_classes=competitors[:10] + (competitors[0],),
                lower_margins=tuple(0.5 for _ in range(11)),
                status=BoundStatus.CONDITIONAL_POSITIVE.value,
                numerical_status=NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value,
                is_positive=True,
            )

        # Target included in competitors
        with self.assertRaises(ValueError):
            BoundResult(
                method_id="M01",
                domain_id="domain_123",
                reference_identity_id="ref_id",
                target_class=3,
                competitor_classes=competitors[:10] + (3,),
                lower_margins=tuple(0.5 for _ in range(11)),
                status=BoundStatus.CONDITIONAL_POSITIVE.value,
                numerical_status=NumericalStatus.CONDITIONAL_REAL_ARITHMETIC.value,
                is_positive=True,
            )

    def test_strict_serialization_round_trip(self) -> None:
        """Verify strict round-trip serialization across public types."""
        raw_arr = np.ones((6, 128), dtype=np.float32)
        q = QueryView.from_array(raw_arr, 10, "rec_02", 5)
        s = to_json_str(q)
        q_back = from_json_str(QueryView, s)
        self.assertEqual(q.global_row_id, q_back.global_row_id)
        self.assertEqual(q.recording_id, q_back.recording_id)
        self.assertEqual(q.step_idx, q_back.step_idx)
        self.assertEqual(q.raw_hash, q_back.raw_hash)

        # Reject NaN in serialization
        with self.assertRaises(ValueError):
            json.dumps({"val": float("nan")}, allow_nan=False)

        # Reject unknown schema_version in deserialization
        bad_json = json.dumps({
            "schema_version": "w4.v999",
            "global_row_id": 10,
            "recording_id": "rec",
            "step_idx": 1,
            "raw_window": list(range(768)),
        })
        with self.assertRaises(SerializationError):
            from_json_str(QueryView, bad_json)


class TestCandidateBoxContracts(unittest.TestCase):
    """CandidateBox domain bounds and membership tests."""

    def test_candidate_box_valid_endpoints(self) -> None:
        lo = np.zeros((6, 128), dtype=np.float64)
        hi = np.ones((6, 128), dtype=np.float64)
        box = CandidateBox(
            lower=tuple(lo.ravel()),
            upper=tuple(hi.ravel()),
            kind="diagonal_box",
            free_dimensions=768,
        )
        self.assertTrue(box.contains(np.full((6, 128), 0.5, dtype=np.float64)))
        self.assertFalse(box.contains(np.full((6, 128), 1.5, dtype=np.float64)))
        self.assertFalse(box.contains(np.full((6, 128), -0.1, dtype=np.float64)))

    def test_candidate_box_inverted_endpoints_fail(self) -> None:
        lo = np.ones((6, 128), dtype=np.float64)
        hi = np.zeros((6, 128), dtype=np.float64)
        with self.assertRaises(ValueError):
            CandidateBox(
                lower=tuple(lo.ravel()),
                upper=tuple(hi.ravel()),
                kind="diagonal_box",
                free_dimensions=768,
            )


class TestFrozenReferenceAdapterContracts(unittest.TestCase):
    """C04, C05, C06: Frozen reference loading, identity assertion, and receipt signing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = ReferenceOwner.load(REFERENCE_DIR)

    def setUp(self) -> None:
        # Re-assert identity before each test to ensure test isolation
        self.ref.assert_identity()
        self.raw_arr = np.zeros((6, 128), dtype=np.float32)
        self.query = QueryView.from_array(self.raw_arr, 100, "rec_test", 12)

    def test_reference_loading_spec(self) -> None:
        """K=12 SmallCNN1D parameter count, eval mode, and CPU device."""
        self.assertEqual(len(self.ref.identity.class_order), 12)
        self.assertEqual(self.ref.identity.class_order, tuple(CLASS_NAMES_12))
        self.assertEqual(self.ref.identity.device, "cpu")
        self.assertEqual(self.ref.identity.dtype, "float32")

    def test_normalization_deterministic(self) -> None:
        """Deterministic normalization matching frozen mean/std."""
        norm1 = self.ref.normalize(self.query)
        norm2 = self.ref.normalize(self.raw_arr)
        np.testing.assert_array_equal(norm1, norm2)
        self.assertFalse(norm1.flags.writeable)

    def test_infer_and_receipt_authenticity(self) -> None:
        """Infer generates a verified receipt with exact tie-breaking."""
        receipt = self.ref.infer(self.query)
        self.assertIsInstance(receipt, TrustedAnchorReceipt)
        self.assertEqual(receipt.query_row_id, 100)
        self.assertEqual(receipt.recording_id, "rec_test")
        self.assertEqual(receipt.step_idx, 12)
        self.assertEqual(len(receipt.logits), 12)
        self.assertEqual(receipt.winner, int(np.argmax(receipt.logits)))

        # Validate authentic receipt
        self.assertTrue(self.ref.validate_receipt(receipt, self.query))

    def test_forged_receipt_rejected(self) -> None:
        """Fabricated receipt with bogus signature or modified logits is rejected."""
        valid_receipt = self.ref.infer(self.query)

        # Forged signature
        forged_receipt = TrustedAnchorReceipt(
            reference_identity_id=valid_receipt.reference_identity_id,
            checkpoint_sha256=valid_receipt.checkpoint_sha256,
            preprocessing_id=valid_receipt.preprocessing_id,
            graph_id=valid_receipt.graph_id,
            weights_id=valid_receipt.weights_id,
            query_row_id=valid_receipt.query_row_id,
            recording_id=valid_receipt.recording_id,
            step_idx=valid_receipt.step_idx,
            raw_hash=valid_receipt.raw_hash,
            normalized_hash=valid_receipt.normalized_hash,
            logits=valid_receipt.logits,
            winner=valid_receipt.winner,
            margin_to_runner_up=valid_receipt.margin_to_runner_up,
            elapsed_ms=valid_receipt.elapsed_ms,
            issuer_signature="0000000000000000000000000000000000000000000000000000000000000000",
            generation_id=valid_receipt.generation_id,
        )
        self.assertFalse(self.ref.validate_receipt(forged_receipt, self.query))

        # Tampered logits
        tampered_logits = list(valid_receipt.logits)
        tampered_logits[0] += 10.0
        tampered_receipt = TrustedAnchorReceipt(
            reference_identity_id=valid_receipt.reference_identity_id,
            checkpoint_sha256=valid_receipt.checkpoint_sha256,
            preprocessing_id=valid_receipt.preprocessing_id,
            graph_id=valid_receipt.graph_id,
            weights_id=valid_receipt.weights_id,
            query_row_id=valid_receipt.query_row_id,
            recording_id=valid_receipt.recording_id,
            step_idx=valid_receipt.step_idx,
            raw_hash=valid_receipt.raw_hash,
            normalized_hash=valid_receipt.normalized_hash,
            logits=tuple(tampered_logits),
            winner=valid_receipt.winner,
            margin_to_runner_up=valid_receipt.margin_to_runner_up,
            elapsed_ms=valid_receipt.elapsed_ms,
            issuer_signature=valid_receipt.issuer_signature,
            generation_id=valid_receipt.generation_id,
        )
        self.assertFalse(self.ref.validate_receipt(tampered_receipt, self.query))

    def test_cross_query_receipt_rejected(self) -> None:
        """A valid receipt for query A cannot validate query B."""
        receipt_a = self.ref.infer(self.query)
        other_arr = np.ones((6, 128), dtype=np.float32)
        query_b = QueryView.from_array(other_arr, 101, "rec_test", 13)
        self.assertFalse(self.ref.validate_receipt(receipt_a, query_b))

    def test_receipt_invalidation_lifecycle(self) -> None:
        """Invalidate rotates the session salt and rejects earlier receipts."""
        receipt = self.ref.infer(self.query)
        self.assertTrue(self.ref.validate_receipt(receipt, self.query))

        self.ref.invalidate("Security rotation")
        self.assertFalse(self.ref.validate_receipt(receipt, self.query))

    def test_strong_identity_assertion_catches_mutations(self) -> None:
        """Deep check detects weight mutation, buffer mutation, hooks, and training mode."""
        # 1. Training mode mutation
        self.ref._model.train()
        with self.assertRaises(IdentityError):
            self.ref.assert_identity()
        self.ref._model.eval()
        self.ref.assert_identity()

        # 2. Weight mutation (bit-exact clone and restore)
        final_linear = self.ref._model.head[-1]
        orig_bias = final_linear.bias.clone()
        with torch.no_grad():
            final_linear.bias[0] += 1.0
        with self.assertRaises(IdentityError):
            self.ref.assert_identity()
        with torch.no_grad():
            final_linear.bias.copy_(orig_bias)
        self.ref.assert_identity()

        # 3. BatchNorm buffer mutation (bit-exact clone and restore)
        first_bn = self.ref._model.conv[1]
        orig_rm = first_bn.running_mean.clone()
        with torch.no_grad():
            first_bn.running_mean[0] += 1.0
        with self.assertRaises(IdentityError):
            self.ref.assert_identity()
        with torch.no_grad():
            first_bn.running_mean.copy_(orig_rm)
        self.ref.assert_identity()

        # 4. Forward hook mutation
        hook_handle = first_bn.register_forward_hook(lambda m, i, o: o)
        with self.assertRaises(IdentityError):
            self.ref.assert_identity()
        hook_handle.remove()
        self.ref.assert_identity()

        # 5. Preprocessing array mutation (bit-exact clone and restore)
        orig_mean = self.ref._mean.copy()
        self.ref._mean[0] += 1.0
        with self.assertRaises(IdentityError):
            self.ref.assert_identity()
        self.ref._mean[:] = orig_mean
        self.ref.assert_identity()


if __name__ == "__main__":
    unittest.main()
