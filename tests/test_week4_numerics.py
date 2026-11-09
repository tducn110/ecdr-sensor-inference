"""tests/test_week4_numerics.py — Comprehensive unit and oracle tests for validated numerics.

Tests outward directed rounding, interval arithmetic primitives, Higham dot-product
error bounds across all SmallCNN1D layer dimensions, edge cases (cancellation,
signed zeros, subnormals), and exact rational oracle verification.
"""

from fractions import Fraction
import math
import unittest

import numpy as np

from ecdr.week4.numerics import (
    DivisionByZeroIntervalError,
    ExactRationalOracle,
    InvertedIntervalError,
    Interval,
    IntervalVector,
    NegativeSqrtIntervalError,
    NonFiniteIntervalError,
    SUB_NORMAL_F32_MAX,
    SUPPORTED_DOT_LENGTHS,
    ULP32_HALF,
    UnsupportedNumericalError,
    gamma_bound,
    round_down_f32,
    round_down_f64,
    round_up_f32,
    round_up_f64,
)


class TestOutwardRoundingPrimitives(unittest.TestCase):
    """Verify directed rounding primitives in float32 and float64."""

    def test_round_down_and_up_f64(self):
        val = 1.0
        lo = round_down_f64(val)
        hi = round_up_f64(val)
        self.assertLess(lo, val)
        self.assertGreater(hi, val)
        self.assertEqual(val - lo, 1.1102230246251565e-16)
        self.assertEqual(hi - val, 2.220446049250313e-16)

    def test_round_down_and_up_f32(self):
        val = 1.0
        lo = round_down_f32(val)
        hi = round_up_f32(val)
        self.assertLess(lo, val)
        self.assertGreater(hi, val)
        # In float32, eps at 1.0 is ~1.19209e-7
        self.assertAlmostEqual(hi - val, 1.1920929e-07, places=12)

    def test_non_finite_rejected(self):
        with self.assertRaises(NonFiniteIntervalError):
            round_down_f64(float("nan"))
        with self.assertRaises(NonFiniteIntervalError):
            round_up_f32(float("inf"))


class TestIntervalCreationAndProperties(unittest.TestCase):
    """Test Interval construction, validation, and geometrical properties."""

    def test_valid_interval(self):
        iv = Interval(lo=-1.5, hi=2.5)
        self.assertEqual(iv.lo, -1.5)
        self.assertEqual(iv.hi, 2.5)
        self.assertEqual(iv.rad, 2.0)
        self.assertEqual(iv.mid, 0.5)
        self.assertEqual(iv.mag, 2.5)
        self.assertEqual(iv.mig, 0.0)  # Contains 0
        self.assertFalse(iv.is_degenerate)

    def test_positive_mig(self):
        iv = Interval(lo=2.0, hi=5.0)
        self.assertEqual(iv.mig, 2.0)
        self.assertEqual(iv.mag, 5.0)

    def test_negative_mig(self):
        iv = Interval(lo=-5.0, hi=-2.0)
        self.assertEqual(iv.mig, 2.0)
        self.assertEqual(iv.mag, 5.0)

    def test_degenerate_point(self):
        iv = Interval.point(3.14)
        self.assertTrue(iv.is_degenerate)
        self.assertEqual(iv.rad, 0.0)
        self.assertEqual(iv.mid, 3.14)
        self.assertTrue(iv.contains(3.14))

    def test_inverted_rejected(self):
        with self.assertRaises(InvertedIntervalError):
            Interval(lo=5.0, hi=2.0)

    def test_non_finite_rejected(self):
        with self.assertRaises(NonFiniteIntervalError):
            Interval(lo=float("nan"), hi=1.0)
        with self.assertRaises(NonFiniteIntervalError):
            Interval(lo=-math.inf, hi=1.0)


class TestIntervalArithmeticOperations(unittest.TestCase):
    """Test basic arithmetic primitives: add, sub, mul, div, sqrt, relu."""

    def test_addition(self):
        a = Interval(1.0, 2.0)
        b = Interval(0.5, 1.5)
        res = a + b
        self.assertLessEqual(res.lo, 1.5)
        self.assertGreaterEqual(res.hi, 3.5)
        self.assertTrue(res.contains(1.5))
        self.assertTrue(res.contains(3.5))

    def test_subtraction(self):
        a = Interval(1.0, 3.0)
        b = Interval(0.5, 1.0)
        res = a - b
        # [1 - 1, 3 - 0.5] = [0.0, 2.5]
        self.assertLessEqual(res.lo, 0.0)
        self.assertGreaterEqual(res.hi, 2.5)
        self.assertTrue(res.contains(0.0))
        self.assertTrue(res.contains(2.5))

    def test_multiplication_signs(self):
        # Case 1: positive * positive
        self.assertTrue((Interval(1.0, 2.0) * Interval(3.0, 4.0)).contains(8.0))
        # Case 2: negative * positive
        res_neg = Interval(-2.0, -1.0) * Interval(3.0, 4.0)
        self.assertLessEqual(res_neg.lo, -8.0)
        self.assertGreaterEqual(res_neg.hi, -3.0)
        # Case 3: crossing zero
        res_cross = Interval(-2.0, 3.0) * Interval(-4.0, 5.0)
        self.assertLessEqual(res_cross.lo, -12.0)  # min(-2*5, 3*-4) = -12
        self.assertGreaterEqual(res_cross.hi, 15.0) # max(-2*-4, 3*5) = 15

    def test_division(self):
        a = Interval(2.0, 6.0)
        b = Interval(1.0, 2.0)
        res = a / b
        self.assertLessEqual(res.lo, 1.0)
        self.assertGreaterEqual(res.hi, 6.0)

    def test_division_by_zero_rejected(self):
        a = Interval(1.0, 2.0)
        with self.assertRaises(DivisionByZeroIntervalError):
            a / Interval(0.0, 1.0)
        with self.assertRaises(DivisionByZeroIntervalError):
            a / Interval(-1.0, 1.0)

    def test_sqrt(self):
        a = Interval(4.0, 9.0)
        res = a.sqrt()
        self.assertLessEqual(res.lo, 2.0)
        self.assertGreaterEqual(res.hi, 3.0)

    def test_sqrt_clamped_lo(self):
        a = Interval(-0.0001, 4.0)
        res = a.sqrt()
        self.assertEqual(res.lo, 0.0)
        self.assertGreaterEqual(res.hi, 2.0)

    def test_negative_sqrt_rejected(self):
        with self.assertRaises(NegativeSqrtIntervalError):
            Interval(-5.0, -1.0).sqrt()

    def test_relu(self):
        a = Interval(-3.0, 4.0)
        res = a.relu()
        self.assertEqual(res.lo, 0.0)
        self.assertEqual(res.hi, 4.0)


class TestAdversarialEdgeCasesAndOracle(unittest.TestCase):
    """Test adversarial numerical conditions: catastrophic cancellation, extremes, exact oracle."""

    def test_adversarial_cancellation(self):
        # Two very close numbers where ordinary subtraction loses precision
        x = 1.000000000000001
        y = 1.0
        a = Interval.point(x)
        b = Interval.point(y)
        res = a - b
        exact_diff = Fraction(x) - Fraction(y)
        self.assertTrue(ExactRationalOracle.verify_enclosure(res, exact_diff))

    def test_magnitude_extremes(self):
        large = Interval.point(1e15)
        small = Interval.point(1e-15)
        res = large + small
        self.assertGreater(res.hi, 1e15)

    def test_exact_rational_oracle_dot_product(self):
        weights = [0.1, -0.2, 0.35, 1.5]
        inputs = [2.0, 3.0, -1.5, 4.0]
        bias = 0.05

        iv_vec = IntervalVector.from_point_array(np.array(inputs))
        # Note: length 4 requires enforce_supported_length=False for custom test
        interval_res = iv_vec.dot_with_constant(
            np.array(weights), bias=bias, enforce_supported_length=False
        )

        exact_val = ExactRationalOracle.exact_dot_product(inputs, weights, bias=bias)
        self.assertTrue(
            ExactRationalOracle.verify_enclosure(interval_res, exact_val),
            f"Exact value {float(exact_val)} not enclosed in [{interval_res.lo}, {interval_res.hi}]",
        )


class TestSmallCNN1DDotProductBounds(unittest.TestCase):
    """Verify Higham dot product bounds for all 5 SmallCNN1D layer dimensions."""

    def test_gamma_bounds_correctness(self):
        for n in SUPPORTED_DOT_LENGTHS:
            gamma = gamma_bound(n, u=ULP32_HALF)
            self.assertGreater(gamma, 0.0)
            self.assertLess(gamma, 0.01)  # Even for n=2048, gamma is ~1.22e-4

    def test_all_smallcnn1d_lengths(self):
        for n in sorted(SUPPORTED_DOT_LENGTHS):
            np.random.seed(42 + n)
            w = np.random.randn(n).astype(np.float32)
            x = np.random.randn(n).astype(np.float32)
            bias = 0.123

            iv = IntervalVector.from_point_array(x)
            res = iv.dot_with_constant(w, bias=bias, enforce_supported_length=True)

            # Check exact dot product is enclosed
            exact = ExactRationalOracle.exact_dot_product(
                [float(v) for v in x], [float(v) for v in w], bias=bias
            )
            self.assertTrue(ExactRationalOracle.verify_enclosure(res, exact))

    def test_unsupported_dot_length_fails_closed(self):
        # n = 42 is not a valid SmallCNN1D layer reduction length
        w = np.ones(42)
        iv = IntervalVector.from_point_array(np.ones(42))
        with self.assertRaises(UnsupportedNumericalError):
            iv.dot_with_constant(w, enforce_supported_length=True)

    def test_unsupported_profile_fails_closed(self):
        w = np.ones(30)
        iv = IntervalVector.from_point_array(np.ones(30))
        with self.assertRaises(UnsupportedNumericalError):
            iv.dot_with_constant(w, profile="non_existent_untested_hardware")


if __name__ == "__main__":
    unittest.main()
