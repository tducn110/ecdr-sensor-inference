"""test_intervals_soundness.py — Test interval arithmetic soundness and anti-aliasing.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 11)
and independent review numerical recommendations.
"""

from fractions import Fraction
import unittest
import numpy as np

from ecdr.intervals import (
    Interval,
    RationalInterval,
    interval_add_outward,
    interval_div_scalar_pos_outward,
    interval_mul_scalar_outward,
    interval_relu,
    interval_sub_outward,
)


class TestIntervalsSoundness(unittest.TestCase):
    def test_inverted_bounds_rejection(self):
        with self.assertRaises(ValueError):
            Interval(lo=np.array([2.0]), hi=np.array([1.0]))

    def test_nonfinite_bounds_rejection(self):
        with self.assertRaises(ValueError):
            Interval(lo=np.array([np.nan]), hi=np.array([1.0]))
        with self.assertRaises(ValueError):
            Interval(lo=np.array([1.0]), hi=np.array([np.inf]))
        with self.assertRaises(ValueError):
            Interval(lo=np.array([-np.inf]), hi=np.array([1.0]))

    def test_detached_base_alias_immutability(self):
        base_lo = np.array([1.0, 2.0], dtype=np.float64)
        base_hi = np.array([3.0, 4.0], dtype=np.float64)

        # Pass views to Interval
        view_lo = base_lo[0:2]
        view_hi = base_hi[0:2]
        interval = Interval(lo=view_lo, hi=view_hi)

        # Mutate the original base buffers
        base_lo[0] = 999.0
        base_hi[0] = -999.0

        # Interval must remain decoupled and immutable!
        self.assertEqual(interval.lo[0], 1.0)
        self.assertEqual(interval.hi[0], 3.0)

        # Interval fields cannot be directly assigned (frozen dataclass)
        with self.assertRaises(Exception):
            interval.lo = np.array([0.0, 0.0])

        # Underlying arrays have writeable=False
        with self.assertRaises(ValueError):
            interval.lo[0] = 5.0

    def test_outward_addition_enclosure(self):
        a = Interval(lo=np.array([1.0]), hi=np.array([2.0]))
        b = Interval(lo=np.array([3.0]), hi=np.array([4.0]))
        res = interval_add_outward(a, b)

        # Check outward step
        self.assertLessEqual(res.lo[0], 4.0)
        self.assertGreaterEqual(res.hi[0], 6.0)

    def test_outward_subtraction_enclosure(self):
        a = Interval(lo=np.array([5.0]), hi=np.array([6.0]))
        b = Interval(lo=np.array([1.0]), hi=np.array([2.0]))
        res = interval_sub_outward(a, b)

        # lo <= 5 - 2 = 3; hi >= 6 - 1 = 5
        self.assertLessEqual(res.lo[0], 3.0)
        self.assertGreaterEqual(res.hi[0], 5.0)

    def test_outward_scalar_multiplication_enclosure(self):
        a = Interval(lo=np.array([2.0]), hi=np.array([3.0]))

        # Positive multiplier
        res_pos = interval_mul_scalar_outward(a, 2.0)
        self.assertLessEqual(res_pos.lo[0], 4.0)
        self.assertGreaterEqual(res_pos.hi[0], 6.0)

        # Negative multiplier (flips bounds)
        res_neg = interval_mul_scalar_outward(a, -2.0)
        self.assertLessEqual(res_neg.lo[0], -6.0)
        self.assertGreaterEqual(res_neg.hi[0], -4.0)

    def test_relu_enclosure(self):
        a = Interval(lo=np.array([-2.0, 1.0]), hi=np.array([3.0, 5.0]))
        res = interval_relu(a)
        np.testing.assert_array_equal(res.lo, [0.0, 1.0])
        np.testing.assert_array_equal(res.hi, [3.0, 5.0])

    def test_rational_interval(self):
        r1 = RationalInterval.from_bounds("1/3", "2/3")
        r2 = RationalInterval.from_bounds("1/4", "3/4")
        sum_r = r1.add(r2)
        self.assertEqual(sum_r.lo, Fraction(7, 12))
        self.assertEqual(sum_r.hi, Fraction(17, 12))


if __name__ == "__main__":
    unittest.main()
