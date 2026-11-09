"""tests/test_week4_membership.py — W4-007: normalization enclosure and conservative membership.

Covers CoordinateBox validation/immutability/identity, exact membership in normalized and raw
coordinate spaces, the frozen executable-normalization enclosure (monotone argument, boundary,
overflow, one-ULP and neighbourhood checks), and fail-closed verdicts.

Sampled/neighbourhood checks here detect bugs; they are not a proof. The exhaustive all-float32
monotonicity evidence lives in output/week4/<run>/tasks/W4-007/exhaustive_monotonicity.json.
"""

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from ecdr.week4 import domains
from ecdr.week4.domains import (
    CoordinateBox,
    CoordinateSpace,
    contains,
    prefix_mismatches,
    strict_gate_accepts,
)
from ecdr.week4.reference import (
    PREPROCESSING_ID,
    REFERENCE_DIR,
    NormalizationConstants,
    ReferenceOwner,
    enclose_normalization,
    execute_normalization,
)
from ecdr.week4.types import MembershipStatus, QueryView, SCHEMA_VERSION, SerializationError
from scripts.backbone import apply_norm

NORM = CoordinateSpace.NORMALIZED_EXECUTABLE
RAW = CoordinateSpace.RAW_WINDOW
F32_INF = np.float32(np.inf)


def _up(x):
    return np.nextafter(np.float32(x), F32_INF)


def _down(x):
    return np.nextafter(np.float32(x), -F32_INF)


def _frozen_constants() -> NormalizationConstants:
    with np.load(REFERENCE_DIR / "normalization.npz") as d:
        return NormalizationConstants(d["mean"], d["std"])


def _synthetic(mean, std) -> NormalizationConstants:
    return NormalizationConstants(np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32))


def _neighbours(center: float, n: int = 128) -> np.ndarray:
    """n consecutive finite float32 values (in value order) around `center`, both signs included."""
    c = np.float32(center)
    p = int(np.array([abs(c)], dtype=np.float32).view(np.uint32)[0])
    pats = np.arange(max(p - 300, 0), min(p + 300, 0x7F7FFFFF) + 1, dtype=np.uint32)
    pos = pats.view(np.float32)
    xs = np.sort(np.concatenate([pos, -pos]))
    idx = int(np.searchsorted(xs, c))
    start = min(max(idx - n // 2, 0), len(xs) - n)
    return xs[start : start + n].astype(np.float32)


# ==============================================================================
# CoordinateBox
# ==============================================================================

class TestCoordinateBox(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.base = self.rng.standard_normal((6, 128)).astype(np.float32)

    def test_valid_box_and_radius_zero(self):
        box = CoordinateBox.from_center_radius(self.base, 0.25, NORM)
        self.assertEqual(box.free_dimensions, 768)
        self.assertEqual(box.coordinate_space, NORM)
        self.assertEqual(box.schema_version, SCHEMA_VERSION)
        point = CoordinateBox.from_center_radius(self.base, 0.0, NORM)
        self.assertEqual(point.free_dimensions, 0)
        self.assertEqual(point.lower, point.upper)
        self.assertTrue(strict_gate_accepts(contains(point, self.base)))

    def test_negative_or_nonfinite_radius_rejected(self):
        for bad in (-1e-12, -1.0, float("nan"), float("inf")):
            with self.subTest(radius=bad):
                with self.assertRaises(ValueError):
                    CoordinateBox.from_center_radius(self.base, bad, NORM)
        neg = np.full((6, 128), 0.1)
        neg[3, 5] = -1e-9
        with self.assertRaises(ValueError):
            CoordinateBox.from_center_radius(self.base, neg, NORM)
        with self.assertRaises(ValueError):
            CoordinateBox.from_center_radius(self.base, np.ones((6, 127)), NORM)

    def test_inverted_endpoints_rejected(self):
        lo = self.base.astype(np.float64)
        hi = lo.copy()
        hi[2, 9] = lo[2, 9] - 1e-9
        with self.assertRaises(ValueError):
            CoordinateBox.from_endpoints(lo, hi, NORM)

    def test_nonfinite_shape_dtype_space_rejected(self):
        lo = self.base.astype(np.float64)
        for bad_val in (np.nan, np.inf, -np.inf):
            hi = lo + 1.0
            hi[0, 0] = bad_val
            with self.assertRaises(ValueError):
                CoordinateBox.from_endpoints(lo, hi, NORM)
        with self.assertRaises(ValueError):
            CoordinateBox.from_endpoints(lo[:, :100], lo[:, :100], NORM)
        with self.assertRaises(ValueError):
            CoordinateBox.from_endpoints(lo.astype(str), lo.astype(str), NORM)
        with self.assertRaises(ValueError):
            CoordinateBox.from_endpoints(lo, lo, "SOME_OTHER_SPACE")
        for bad_k in (-1, 129, True, 1.5):
            with self.assertRaises(ValueError):
                CoordinateBox.from_endpoints(lo, lo, NORM, fixed_prefix_steps=bad_k)

    def test_fixed_prefix_requires_equal_endpoints(self):
        lo = self.base.astype(np.float64)
        hi = lo + 1.0
        with self.assertRaises(ValueError):
            CoordinateBox.from_endpoints(lo, hi, NORM, fixed_prefix_steps=64)
        hi[:, :64] = lo[:, :64]
        box = CoordinateBox.from_endpoints(lo, hi, NORM, fixed_prefix_steps=64)
        self.assertEqual(box.free_dimensions, 384)

    def test_caller_alias_mutation_cannot_change_box(self):
        lo = self.base.astype(np.float64) - 1.0
        hi = self.base.astype(np.float64) + 1.0
        box = CoordinateBox.from_endpoints(lo, hi, NORM)
        before = box.domain_id
        lo[:] = 1e9
        hi[:] = -1e9
        self.assertEqual(box.domain_id, before)
        self.assertTrue(strict_gate_accepts(contains(box, self.base)))

    def test_arrays_are_copies_and_internal_arrays_read_only(self):
        box = CoordinateBox.from_center_radius(self.base, 1.0, NORM)
        lo, hi = box.arrays()
        lo[:] = 1e9
        hi[:] = -1e9
        self.assertTrue(strict_gate_accepts(contains(box, self.base)))
        self.assertFalse(box._lo.flags.writeable)
        self.assertFalse(box._hi.flags.writeable)
        with self.assertRaises(ValueError):
            box._lo[0, 0] = 5.0

    def test_flatten_order_is_logical_c_order(self):
        lo = self.base.astype(np.float64) - 1.0
        hi = self.base.astype(np.float64) + 1.0
        c_box = CoordinateBox.from_endpoints(lo, hi, NORM)
        f_box = CoordinateBox.from_endpoints(np.asfortranarray(lo), np.asfortranarray(hi), NORM)
        self.assertEqual(c_box.domain_id, f_box.domain_id)
        self.assertEqual(c_box.lower, f_box.lower)
        # a (128, 6)-transposed buffer is a different logical layout and must not be silently accepted
        # as the same domain when raveled
        t_box = CoordinateBox.from_endpoints(lo.T.copy().ravel(), hi.T.copy().ravel(), NORM)
        self.assertNotEqual(c_box.domain_id, t_box.domain_id)

    def test_serialization_roundtrip_and_tamper(self):
        box = CoordinateBox.from_center_radius(self.base, 0.5, RAW, fixed_prefix_steps=0)
        data = box.to_dict()
        again = CoordinateBox.from_dict(data)
        self.assertEqual(again.domain_id, box.domain_id)
        self.assertEqual(again.coordinate_space, RAW)
        tampered = dict(data)
        tampered["lower"] = list(tampered["lower"])
        tampered["lower"][0] -= 1.0
        with self.assertRaises(SerializationError):
            CoordinateBox.from_dict(tampered)
        bad_schema = dict(data, schema_version="w4.v0")
        with self.assertRaises(SerializationError):
            CoordinateBox.from_dict(bad_schema)
        with self.assertRaises(SerializationError):
            CoordinateBox.from_dict(dict(data, type="AffineTailDomain"))

    def test_domain_id_binds_coordinate_space_and_prefix(self):
        lo = self.base.astype(np.float64)
        a = CoordinateBox.from_endpoints(lo, lo, NORM)
        b = CoordinateBox.from_endpoints(lo, lo, RAW)
        c = CoordinateBox.from_endpoints(lo, lo, NORM, fixed_prefix_steps=64)
        self.assertEqual(len({a.domain_id, b.domain_id, c.domain_id}), 3)


# ==============================================================================
# Membership in normalized executable space
# ==============================================================================

class TestNormalizedMembership(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.base = rng.standard_normal((6, 128)).astype(np.float32)
        self.lo32 = (self.base - np.float32(0.5)).astype(np.float32)
        self.hi32 = (self.base + np.float32(0.5)).astype(np.float32)
        self.box = CoordinateBox.from_endpoints(self.lo32, self.hi32, NORM)

    def test_inside_passes_strict_gate(self):
        res = contains(self.box, self.base)
        self.assertEqual(res.status, MembershipStatus.EXACT_MEMBER.value)
        self.assertTrue(res.is_member)
        self.assertTrue(strict_gate_accepts(res))
        self.assertEqual(res.domain_id, self.box.domain_id)

    def test_boundary_equality_is_inside(self):
        self.assertTrue(strict_gate_accepts(contains(self.box, self.lo32)))
        self.assertTrue(strict_gate_accepts(contains(self.box, self.hi32)))

    def test_one_ulp_outside_every_coordinate_both_sides(self):
        for (c, t) in [(0, 0), (5, 127), (3, 64), (2, 63), (1, 100)]:
            up = self.hi32.copy()
            up[c, t] = _up(self.hi32[c, t])
            dn = self.lo32.copy()
            dn[c, t] = _down(self.lo32[c, t])
            for q in (up, dn):
                res = contains(self.box, q)
                self.assertEqual(res.status, MembershipStatus.VALIDATED_OUTSIDE.value)
                self.assertFalse(res.is_member)
                self.assertFalse(strict_gate_accepts(res))

    def test_signed_zero_equals_zero_endpoint(self):
        zero = np.zeros((6, 128), dtype=np.float32)
        box = CoordinateBox.from_endpoints(zero, zero, NORM)
        neg_zero = np.full((6, 128), -0.0, dtype=np.float32)
        self.assertTrue(strict_gate_accepts(contains(box, neg_zero)))
        self.assertTrue(strict_gate_accepts(contains(box, zero)))

    def test_invalid_inputs_fail_closed(self):
        cases = {
            "float64": self.base.astype(np.float64),
            "shape": self.base[:, :100],
            "nan": np.where(np.arange(768).reshape(6, 128) == 5, np.nan, self.base).astype(np.float32),
            "inf": np.where(np.arange(768).reshape(6, 128) == 5, np.inf, self.base).astype(np.float32),
            "list": self.base.tolist(),
        }
        for name, q in cases.items():
            with self.subTest(case=name):
                res = contains(self.box, q)
                self.assertEqual(res.status, MembershipStatus.ERROR.value)
                self.assertFalse(res.is_member)
                self.assertFalse(strict_gate_accepts(res))

    def test_wrong_coordinate_space_query_and_domain_type(self):
        qv = QueryView.from_array(self.base, 1, "rec", 0)
        res = contains(self.box, qv)
        self.assertEqual(res.status, MembershipStatus.ERROR.value)
        self.assertIn("coordinate space", res.reason)
        for not_a_box in (None, {"lower": []}, object(), "box"):
            res = contains(not_a_box, self.base)
            self.assertEqual(res.status, MembershipStatus.ERROR.value)
            self.assertFalse(strict_gate_accepts(res))

    def test_strict_gate_rejects_forged_result_shapes(self):
        self.assertFalse(strict_gate_accepts(None))
        self.assertFalse(strict_gate_accepts({"status": "EXACT_MEMBER", "is_member": True}))
        good = contains(self.box, self.base)
        from ecdr.week4.types import MembershipResult

        non_member_exact = MembershipResult(
            domain_id=good.domain_id,
            status=MembershipStatus.EXACT_MEMBER.value,
            is_member=False,
            reason="x",
            method_id="m",
            elapsed_ms=0.0,
        )
        self.assertFalse(strict_gate_accepts(non_member_exact))
        conditional = MembershipResult(
            domain_id=good.domain_id,
            status=MembershipStatus.CONDITIONAL_MEMBER.value,
            is_member=True,
            reason="x",
            method_id="m",
            elapsed_ms=0.0,
        )
        self.assertFalse(strict_gate_accepts(conditional))

    def test_query_not_mutated_and_result_independent_of_later_mutation(self):
        q = self.base.copy()
        snapshot = q.copy()
        res = contains(self.box, q)
        self.assertTrue(np.array_equal(q, snapshot))
        q[:] = 1e9
        self.assertTrue(res.is_member)


class TestFixedPrefixMembership(unittest.TestCase):
    """C05: all 6 * 64 = 384 prefix coordinates are compared; none are skipped."""

    def setUp(self):
        rng = np.random.default_rng(23)
        self.base = rng.standard_normal((6, 128)).astype(np.float32)
        tail = self.base[:, 64:]
        self.box = CoordinateBox.from_prefix_and_tail(
            self.base[:, :64], tail - np.float32(1.0), tail + np.float32(1.0), NORM
        )

    def test_structure(self):
        self.assertEqual(self.box.fixed_prefix_steps, 64)
        self.assertEqual(self.box.free_dimensions, 384)
        self.assertTrue(strict_gate_accepts(contains(self.box, self.base)))
        self.assertEqual(prefix_mismatches(self.box, self.base), (0, None))

    def test_every_prefix_coordinate_enforced_one_ulp_each_direction(self):
        checked = 0
        for c in range(6):
            for t in range(64):
                for mover in (_up, _down):
                    q = self.base.copy()
                    q[c, t] = mover(self.base[c, t])
                    res = contains(self.box, q)
                    self.assertEqual(
                        res.status, MembershipStatus.VALIDATED_OUTSIDE.value, f"prefix ({c},{t}) not enforced"
                    )
                    self.assertIn("fixed-prefix mismatches=1", res.reason)
                    self.assertEqual(prefix_mismatches(self.box, q), (1, (c, t)))
                    checked += 1
        self.assertEqual(checked, 6 * 64 * 2)

    def test_tail_coordinates_remain_free_within_endpoints(self):
        for c in range(6):
            for t in (64, 100, 127):
                q = self.base.copy()
                q[c, t] = _up(self.base[c, t])
                self.assertTrue(strict_gate_accepts(contains(self.box, q)))
        q = self.base.copy()
        q[2, 70] = self.base[2, 70] + np.float32(1.5)
        self.assertEqual(contains(self.box, q).status, MembershipStatus.VALIDATED_OUTSIDE.value)

    def test_prefix_shape_validation(self):
        with self.assertRaises(ValueError):
            CoordinateBox.from_prefix_and_tail(self.base[:5, :64], self.base[:, 64:], self.base[:, 64:], NORM)
        with self.assertRaises(ValueError):
            CoordinateBox.from_prefix_and_tail(self.base[:, :64], self.base[:, 60:], self.base[:, 60:], NORM)


# ==============================================================================
# Normalization enclosure
# ==============================================================================

class TestNormalizationEnclosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _frozen_constants()

    def test_frozen_constants_identity(self):
        self.assertEqual(self.frozen.preprocessing_id, PREPROCESSING_ID)
        self.assertEqual(self.frozen.mean.dtype, np.float32)
        self.assertTrue(np.all(self.frozen.std > 0))

    def test_constants_validation_and_immutability(self):
        with self.assertRaises(ValueError):
            NormalizationConstants(np.zeros(5), np.ones(6))
        with self.assertRaises(ValueError):
            NormalizationConstants(np.zeros(6), np.zeros(6))
        with self.assertRaises(ValueError):
            NormalizationConstants(np.zeros(6), -np.ones(6))
        with self.assertRaises(ValueError):
            NormalizationConstants(np.array([np.nan] + [0] * 5), np.ones(6))
        m = np.zeros(6, dtype=np.float32)
        s = np.ones(6, dtype=np.float32)
        nc = NormalizationConstants(m, s)
        pid = nc.preprocessing_id
        m[:] = 5.0
        s[:] = 9.0
        self.assertEqual(nc.preprocessing_id, pid)
        with self.assertRaises(ValueError):
            nc.mean[0] = 1.0
        self.assertEqual(float(nc.mean[0]), 0.0)

    def test_executable_matches_pipeline_apply_norm_bitwise(self):
        rng = np.random.default_rng(5)
        raw = (rng.standard_normal((64, 6, 128)) * 3.0 + self.frozen.mean[None, :, None]).astype(np.float32)
        piped = apply_norm(raw, self.frozen.mean, self.frozen.std)
        for i in range(raw.shape[0]):
            ours = execute_normalization(raw[i], self.frozen)
            self.assertEqual(ours.tobytes(), piped[i].tobytes())

    def test_enclosure_contains_every_value_in_consecutive_float_windows(self):
        sets = [
            ([0.0] * 6, [3.0] * 6),
            (self.frozen.mean.tolist(), self.frozen.std.tolist()),
            ([1.0, -2.5, 0.1, 1e-3, 100.0, -0.0], [0.7, 1.3, 2.9, 5.0, 0.3, 1.0]),
        ]
        centers_sets = [
            [0.0, 1e-45, -1e-45, 1e-38, -1e-38, 1.0],
            [-1.0, 123.456, -123.456, 1e10, -1e10, 3e30],
            [0.8043609, 0.0411366, 0.08177005, 0.00913557, -0.00439756, -0.0071832],
        ]
        for mean, std in sets:
            nc = _synthetic(mean, std)
            for centers in centers_sets:
                rows = np.stack([_neighbours(c) for c in centers])
                lo = np.repeat(rows[:, :1], 128, axis=1)
                hi = np.repeat(rows[:, -1:], 128, axis=1)
                try:
                    e_lo, e_hi = enclose_normalization(lo, hi, nc)
                except ValueError:
                    continue  # overflow for this synthetic set is a legitimate fail-closed outcome
                y = execute_normalization(rows, nc)
                self.assertTrue(np.all(y >= e_lo) and np.all(y <= e_hi))
                self.assertTrue(np.all(np.diff(y, axis=1) >= 0), "executable normalization not monotone here")

    def test_float64_real_endpoints_enclose_converted_points(self):
        rng = np.random.default_rng(31)
        center = (rng.standard_normal((6, 128)) * 2.0).astype(np.float64)
        width = 1e-6
        lo = center - width
        hi = center + width
        e_lo, e_hi = enclose_normalization(lo, hi, self.frozen)
        for _ in range(50):
            x64 = rng.uniform(lo, hi)
            y = execute_normalization(x64.astype(np.float32), self.frozen)
            self.assertTrue(np.all(y >= e_lo) and np.all(y <= e_hi))

    def test_zero_width_enclosure_is_the_executable_value(self):
        rng = np.random.default_rng(2)
        raw = (rng.standard_normal((6, 128))).astype(np.float32)
        e_lo, e_hi = enclose_normalization(raw, raw, self.frozen)
        ref = execute_normalization(raw, self.frozen)
        self.assertEqual(e_lo.tobytes(), ref.tobytes())
        self.assertEqual(e_hi.tobytes(), ref.tobytes())

    def test_enclosure_outputs_read_only_float32(self):
        raw = np.zeros((6, 128), dtype=np.float32)
        e_lo, e_hi = enclose_normalization(raw, raw + 1, self.frozen)
        for e in (e_lo, e_hi):
            self.assertEqual(e.dtype, np.float32)
            self.assertFalse(e.flags.writeable)

    def test_conversion_overflow_rejected(self):
        lo = np.zeros((6, 128), dtype=np.float64)
        hi = lo.copy()
        hi[1, 1] = 1e39  # > float32 max
        with self.assertRaises(ValueError) as ctx:
            enclose_normalization(lo, hi, self.frozen)
        self.assertIn("overflow", str(ctx.exception))
        lo2 = lo.copy()
        lo2[0, 0] = -1e39
        with self.assertRaises(ValueError):
            enclose_normalization(lo2, hi * 0, self.frozen)

    def test_normalized_overflow_rejected(self):
        tiny_std = _synthetic([0.0] * 6, [1e-6] * 6)
        hi = np.zeros((6, 128), dtype=np.float32)
        hi[0, 0] = np.float32(3e38)
        with self.assertRaises(ValueError):
            enclose_normalization(np.zeros((6, 128), dtype=np.float32), hi, tiny_std)
        with self.assertRaises(ValueError):
            execute_normalization(hi, tiny_std)
        # subtraction overflow: x - mean overflows float32
        big_mean = _synthetic([-3e38] * 6, [1.0] * 6)
        with self.assertRaises(ValueError):
            execute_normalization(np.full((6, 128), 3e38, dtype=np.float32), big_mean)

    def test_nonfinite_inverted_dtype_shape_rejected(self):
        z = np.zeros((6, 128), dtype=np.float32)
        for bad in (np.nan, np.inf, -np.inf):
            w = z.copy()
            w[2, 2] = bad
            with self.assertRaises(ValueError):
                enclose_normalization(z, w, self.frozen)
            with self.assertRaises(ValueError):
                enclose_normalization(w, z, self.frozen)
        up = z.copy()
        up[0, 0] = 1.0
        with self.assertRaises(ValueError):
            enclose_normalization(up, z, self.frozen)  # inverted
        with self.assertRaises(ValueError):
            enclose_normalization(z.astype(np.int32), z.astype(np.int32), self.frozen)
        with self.assertRaises(ValueError):
            enclose_normalization(z[:, :64], z[:, :64], self.frozen)
        with self.assertRaises(ValueError):
            execute_normalization(z.astype(np.float64), self.frozen)


# ==============================================================================
# Membership in raw space (with normalization bridge)
# ==============================================================================

class TestRawMembership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _frozen_constants()

    def setUp(self):
        rng = np.random.default_rng(101)
        self.mean = self.frozen.mean
        self.raw = (rng.standard_normal((6, 128)) + self.mean[:, None]).astype(np.float32)
        self.lo = (self.raw - np.float32(0.25)).astype(np.float32)
        self.hi = (self.raw + np.float32(0.25)).astype(np.float32)
        self.box = CoordinateBox.from_endpoints(self.lo, self.hi, RAW)

    def test_inside_with_bridge_passes_gate(self):
        res = contains(self.box, self.raw, normalization=self.frozen)
        self.assertTrue(strict_gate_accepts(res), res.reason)
        qv = QueryView.from_array(self.raw, 3, "rec", 1)
        self.assertTrue(strict_gate_accepts(contains(self.box, qv, normalization=self.frozen)))

    def test_inside_without_bridge_is_ambiguous_not_accepted(self):
        res = contains(self.box, self.raw)
        self.assertEqual(res.status, MembershipStatus.CONDITIONAL_MEMBER.value)
        self.assertFalse(res.is_member)
        self.assertFalse(strict_gate_accepts(res))

    def test_clearly_outside(self):
        q = self.raw.copy()
        q[4, 4] += np.float32(2.0)
        for norm in (None, self.frozen):
            res = contains(self.box, q, normalization=norm)
            self.assertEqual(res.status, MembershipStatus.VALIDATED_OUTSIDE.value)
            self.assertFalse(strict_gate_accepts(res))

    def test_raw_boundary_equality_inside(self):
        self.assertTrue(strict_gate_accepts(contains(self.box, self.lo, normalization=self.frozen)))
        self.assertTrue(strict_gate_accepts(contains(self.box, self.hi, normalization=self.frozen)))

    def test_one_ulp_outside_never_accepted(self):
        accepted = 0
        for (c, t) in [(0, 0), (1, 17), (2, 33), (3, 64), (4, 100), (5, 127)]:
            for q0, mover in ((self.hi, _up), (self.lo, _down)):
                q = q0.copy()
                q[c, t] = mover(q0[c, t])
                res = contains(self.box, q, normalization=self.frozen)
                accepted += int(strict_gate_accepts(res))
                self.assertIn(
                    res.status,
                    (MembershipStatus.VALIDATED_OUTSIDE.value, MembershipStatus.CONDITIONAL_MEMBER.value),
                )
        self.assertEqual(accepted, 0)

    def test_non_injective_normalization_gives_ambiguous_not_widened(self):
        nc = _synthetic([0.0] * 6, [3.0] * 6)
        # find adjacent float32 raw values that the executable maps to the SAME normalized value
        pair = None
        for start in np.float32([1.5, 1.75, 2.9, 5.1, 7.7, 100.3]):
            xs = _neighbours(float(start), 128)
            for i in range(len(xs) - 1):
                row = np.tile(xs[i : i + 1], (6, 128))
                row2 = np.tile(xs[i + 1 : i + 2], (6, 128))
                if np.array_equal(execute_normalization(row, nc), execute_normalization(row2, nc)):
                    pair = (xs[i], xs[i + 1])
                    break
            if pair:
                break
        self.assertIsNotNone(pair, "test premise: expected a normalization collision for std=3")
        a, b = np.float32(pair[0]), np.float32(pair[1])
        self.assertLess(a, b)
        lo = np.full((6, 128), a - np.float32(1.0), dtype=np.float32)
        hi = np.full((6, 128), a, dtype=np.float32)
        box = CoordinateBox.from_endpoints(lo, hi, RAW)
        q = np.full((6, 128), b, dtype=np.float32)  # raw outside by one ULP, same executable input as `a`
        res = contains(box, q, normalization=nc)
        self.assertEqual(res.status, MembershipStatus.CONDITIONAL_MEMBER.value)
        self.assertFalse(strict_gate_accepts(res))
        self.assertIn("not widened", res.reason)
        # and the exact upper endpoint itself is accepted
        self.assertTrue(strict_gate_accepts(contains(box, hi, normalization=nc)))

    def test_box_endpoint_conversion_overflow_is_invalid(self):
        lo = np.zeros((6, 128))
        hi = np.zeros((6, 128))
        hi[0, 0] = 1e39
        big = CoordinateBox.from_endpoints(lo, hi, RAW)
        res = contains(big, np.zeros((6, 128), dtype=np.float32), normalization=self.frozen)
        self.assertEqual(res.status, MembershipStatus.ERROR.value)
        self.assertFalse(strict_gate_accepts(res))

    def test_query_normalization_overflow_is_invalid(self):
        nc = _synthetic([0.0] * 6, [1e-6] * 6)
        box = CoordinateBox.from_endpoints(np.full((6, 128), -1.0), np.full((6, 128), 1.0), RAW)
        q = np.zeros((6, 128), dtype=np.float32)
        q[0, 0] = np.float32(3e38)
        res = contains(box, q, normalization=nc)
        # the executable normalization of this query overflows: bridge fails closed (ERROR), never accepted
        self.assertEqual(res.status, MembershipStatus.ERROR.value)
        self.assertIn("non-finite", res.reason)
        self.assertFalse(strict_gate_accepts(res))

    def test_invalid_raw_inputs(self):
        cases = {
            "float64": self.raw.astype(np.float64),
            "shape": self.raw[:, :10],
            "nan": np.where(np.arange(768).reshape(6, 128) == 3, np.nan, self.raw).astype(np.float32),
        }
        for name, q in cases.items():
            with self.subTest(case=name):
                res = contains(self.box, q, normalization=self.frozen)
                self.assertEqual(res.status, MembershipStatus.ERROR.value)
        res = contains(self.box, self.raw, normalization="frozen")
        self.assertEqual(res.status, MembershipStatus.ERROR.value)

    def test_bridge_premise_contradiction_fails_closed(self):
        real_exec = domains.execute_normalization

        def shifted(x, nc):
            return real_exec(x, nc) + np.float32(10.0)

        with mock.patch.object(domains, "execute_normalization", side_effect=shifted):
            res = contains(self.box, self.raw, normalization=self.frozen)
        self.assertEqual(res.status, MembershipStatus.ERROR.value)
        self.assertIn("premise contradicted", res.reason)
        self.assertFalse(strict_gate_accepts(res))

    def test_random_inside_queries_validated_and_consistent_with_normalized_box(self):
        rng = np.random.default_rng(404)
        n_inside = 0
        for _ in range(40):
            q = rng.uniform(self.lo, self.hi).astype(np.float32)
            q = np.minimum(np.maximum(q, self.lo), self.hi)
            res = contains(self.box, q, normalization=self.frozen)
            self.assertTrue(strict_gate_accepts(res), res.reason)
            n_inside += 1
            e_lo, e_hi = enclose_normalization(self.lo, self.hi, self.frozen)
            qn = execute_normalization(q, self.frozen)
            self.assertTrue(np.all(qn >= e_lo) and np.all(qn <= e_hi))
        self.assertEqual(n_inside, 40)


# ==============================================================================
# ReferenceOwner integration (real frozen reference)
# ==============================================================================

@unittest.skipUnless((Path(REFERENCE_DIR) / "selected_best.pt").is_file(), "frozen reference not available")
class TestReferenceOwnerNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner = ReferenceOwner.load()

    def test_normalize_unchanged_semantics(self):
        rng = np.random.default_rng(9)
        raw = (rng.standard_normal((6, 128)) * 2).astype(np.float32)
        out = self.owner.normalize(raw)
        piped = apply_norm(raw[None], self.owner.normalization_constants().mean, self.owner.normalization_constants().std)[0]
        self.assertEqual(out.tobytes(), piped.tobytes())
        self.assertFalse(out.flags.writeable)
        self.assertEqual(self.owner.normalize(QueryView.from_array(raw, 0, "r", 0)).tobytes(), out.tobytes())

    def test_constants_match_frozen_identity_and_are_copies(self):
        nc = self.owner.normalization_constants()
        self.assertEqual(nc.preprocessing_id, PREPROCESSING_ID)
        self.assertEqual(nc.preprocessing_id, self.owner.identity.preprocessing_id)
        mean = nc.mean
        with self.assertRaises(ValueError):
            mean[0] = 99.0
        self.assertEqual(self.owner.normalization_constants().preprocessing_id, PREPROCESSING_ID)

    def test_enclosure_method_and_failed_calls_do_not_mutate_owner(self):
        gen = self.owner.generation_id
        raw = np.zeros((6, 128), dtype=np.float32)
        e_lo, e_hi = self.owner.enclose_normalization(raw - 1, raw + 1)
        self.assertTrue(np.all(e_lo <= e_hi))
        for bad in (np.full((6, 128), np.nan, dtype=np.float32), np.zeros((6, 100), dtype=np.float32)):
            with self.assertRaises(ValueError):
                self.owner.normalize(bad)
        with self.assertRaises(ValueError):
            self.owner.enclose_normalization(raw + 1, raw)
        self.assertEqual(self.owner.generation_id, gen)
        self.owner.assert_identity()

    def test_membership_end_to_end_with_owner_normalization(self):
        rng = np.random.default_rng(77)
        raw = (rng.standard_normal((6, 128)) * 2).astype(np.float32)
        qn = self.owner.normalize(raw)
        nbox = CoordinateBox.from_center_radius(qn, 1e-3, NORM)
        self.assertTrue(strict_gate_accepts(contains(nbox, qn)))
        rbox = CoordinateBox.from_center_radius(raw, 1e-3, RAW)
        res = contains(rbox, raw, normalization=self.owner.normalization_constants())
        self.assertTrue(strict_gate_accepts(res), res.reason)


if __name__ == "__main__":
    unittest.main()
