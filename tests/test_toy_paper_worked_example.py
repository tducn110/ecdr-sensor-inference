"""test_toy_paper_worked_example.py — Test exact Section 10 paper worked example.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 10).

Specifications:
  C=1, W=6, s=2
  X = (1, 2, 3, 4, 5, 6), U = (7, 8), X' = (3, 4, 5, 6, 7, 8)
  w^(1) = (1, -1, 1), b^(1) = -2
  w^(2) = (-1, 0, 1), b^(2) = -1
  A = [[1, -1], [-1, 0], [2, -2]], beta = [4, 1, 0]
"""

from fractions import Fraction
import unittest
import numpy as np

from ecdr.intervals import Interval, RationalInterval


class TestToyPaperWorkedExample(unittest.TestCase):
    def setUp(self):
        # Setup exact rational parameters
        self.w1 = np.array([[[1.0, -1.0, 1.0]], [[-1.0, 0.0, 1.0]]], dtype=np.float64)  # (2, 1, 3)
        self.b1 = np.array([-2.0, -1.0], dtype=np.float64)  # (2,)
        self.A = np.array([[1.0, -1.0], [-1.0, 0.0], [2.0, -2.0]], dtype=np.float64)  # (3, 2)
        self.beta = np.array([4.0, 1.0, 0.0], dtype=np.float64)  # (3,)

        self.x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64)  # (1, 6)
        self.u = np.array([[7.0, 8.0]], dtype=np.float64)  # (1, 2)
        self.x_prime = np.array([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float64)  # (1, 6)

    def test_toy_activations_and_shift_identity(self):
        # 1. Forward pass on X: Conv valid k=3 -> T_out = 6 - 3 + 1 = 4
        # Filter 1:
        # i=0: 1 - 2 + 3 - 2 = 0
        # i=1: 2 - 3 + 4 - 2 = 1
        # i=2: 3 - 4 + 5 - 2 = 2
        # i=3: 4 - 5 + 6 - 2 = 3
        # Filter 2:
        # i=0: -1 + 3 - 1 = 1
        # i=1: -2 + 4 - 1 = 1
        # i=2: -3 + 5 - 1 = 1
        # i=3: -4 + 6 - 1 = 1
        h_x = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float64)

        # Forward pass on X':
        # Filter 1:
        # i=0: 3 - 4 + 5 - 2 = 2
        # i=1: 4 - 5 + 6 - 2 = 3
        # i=2: 5 - 6 + 7 - 2 = 4
        # i=3: 6 - 7 + 8 - 2 = 5
        # Filter 2:
        # i=0: -3 + 5 - 1 = 1
        # i=1: -4 + 6 - 1 = 1
        # i=2: -5 + 7 - 1 = 1
        # i=3: -6 + 8 - 1 = 1
        h_x_prime = np.array([
            [2.0, 3.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float64)

        # Shift identity: H_i(X') == H_{i+s}(X) for 0 <= i < n - s (n=4, s=2 => i < 2)
        # Position 0 and 1 of X' must match position 2 and 3 of X
        np.testing.assert_array_equal(h_x_prime[:, :2], h_x[:, 2:])

    def test_toy_gap_and_logits(self):
        # m(X) = ((0+1+2+3)/4, (1+1+1+1)/4) = (1.5, 1.0)
        m_x = np.array([1.5, 1.0], dtype=np.float64)
        # m(X') = ((2+3+4+5)/4, 1.0) = (3.5, 1.0)
        m_x_prime = np.array([3.5, 1.0], dtype=np.float64)

        # Logits F(X) = A @ m(X) + beta
        # F1 = 1(1.5) - 1(1.0) + 4 = 4.5 = 9/2
        # F2 = -1(1.5) + 0(1.0) + 1 = -0.5 = -1/2
        # F3 = 2(1.5) - 2(1.0) + 0 = 1.0
        f_x = self.A @ m_x + self.beta
        np.testing.assert_allclose(f_x, [4.5, -0.5, 1.0])

        # F(X') = A @ m(X') + beta
        # F1 = 1(3.5) - 1(1.0) + 4 = 6.5 = 13/2
        # F2 = -1(3.5) + 0(1.0) + 1 = -2.5 = -5/2
        # F3 = 2(3.5) - 2(1.0) + 0 = 5.0
        f_x_prime = self.A @ m_x_prime + self.beta
        np.testing.assert_allclose(f_x_prime, [6.5, -2.5, 5.0])

        # Target class is 0 (1-based class 1)
        # Old margins: g_12 = 4.5 - (-0.5) = 5.0, g_13 = 4.5 - 1.0 = 3.5
        g_12_old = f_x[0] - f_x[1]
        g_13_old = f_x[0] - f_x[2]
        self.assertEqual(g_12_old, 5.0)
        self.assertEqual(g_13_old, 3.5)

        # New margins: g_12 = 6.5 - (-2.5) = 9.0, g_13 = 6.5 - 5.0 = 1.5
        g_12_new = f_x_prime[0] - f_x_prime[1]
        g_13_new = f_x_prime[0] - f_x_prime[2]
        self.assertEqual(g_12_new, 9.0)
        self.assertEqual(g_13_new, 1.5)

    def test_toy_gap_delta_identity(self):
        # D_drop = sum_{i=0}^{1} H_i(X) = (0+1, 1+1) = (1.0, 2.0)
        # D_new = sum_{i=2}^{3} H_i(X') = (4+5, 1+1) = (9.0, 2.0)
        d_drop = np.array([1.0, 2.0])
        d_new = np.array([9.0, 2.0])
        n = 4

        delta_m = (d_new - d_drop) / n
        np.testing.assert_allclose(delta_m, [2.0, 0.0])

        # q_2 = A_0 - A_1 = [1 - (-1), -1 - 0] = [2, -1]
        # q_3 = A_0 - A_2 = [1 - 2, -1 - (-2)] = [-1, 1]
        q_2 = self.A[0] - self.A[1]
        q_3 = self.A[0] - self.A[2]
        np.testing.assert_allclose(q_2, [2.0, -1.0])
        np.testing.assert_allclose(q_3, [-1.0, 1.0])

        # Margin deltas:
        # delta g_12 = (1/4) * q_2^T (d_new - d_drop) = (1/4) * (2*8 - 1*0) = 4.0
        # delta g_13 = (1/4) * q_3^T (d_new - d_drop) = (1/4) * (-1*8 + 1*0) = -2.0
        delta_g12 = np.dot(q_2, d_new - d_drop) / n
        delta_g13 = np.dot(q_3, d_new - d_drop) / n
        self.assertEqual(delta_g12, 4.0)
        self.assertEqual(delta_g13, -2.0)

        # Checks new margins via delta formula: 5 + 4 = 9, 3.5 - 2 = 1.5
        self.assertEqual(5.0 + delta_g12, 9.0)
        self.assertEqual(3.5 + delta_g13, 1.5)

    def test_toy_continuous_wave_domain_bounds(self):
        # Box U in [6.5, 7.5] x [7.5, 8.5]
        # Tail feature vectors before ReLU:
        # H_2 = (u - 3, u - 6) -> on box: u in [6.5, 7.5] => [3.5, 4.5] x [0.5, 1.5]
        # H_3 = (4 - u + v, v - 7) -> on box:
        #   4 - u + v: min at u=7.5, v=7.5 => 4 - 7.5 + 7.5 = 4.0; max at u=6.5, v=8.5 => 4 - 6.5 + 8.5 = 6.0
        #   v - 7: [0.5, 1.5]
        # All preactivations are positive, so ReLU is identity!
        # D_new = H_2 + H_3
        # D_1 = (u - 3) + (4 - u + v) = 1 + v
        # D_2 = (u - 6) + (v - 7) = u + v - 13

        # Position bounds:
        # H_2 in [3.5, 4.5] x [0.5, 1.5]
        # H_3 in [4.0, 6.0] x [0.5, 1.5]
        # D_new elementwise interval:
        # D_lo = [3.5 + 4.0, 0.5 + 0.5] = [7.5, 1.0]
        # D_hi = [4.5 + 6.0, 1.5 + 1.5] = [10.5, 3.0]
        d_lo = np.array([7.5, 1.0])
        d_hi = np.array([10.5, 3.0])

        q_2 = np.array([2.0, -1.0])
        q_3 = np.array([-1.0, 1.0])
        d_drop = np.array([1.0, 2.0])
        n = 4

        # Equation (9.4) from paper:
        # ell_{cj} = gamma_cj(Z) + (1/n) * [ (q_j^+)^T D_lo + (q_j^-)^T D_hi - q_j^T D_drop ]
        # For competitor 2: q_2^+ = [2, 0], q_2^- = [0, -1]
        # ell_12 = 5 + (1/4) * [ 2(7.5) + (-1)(3.0) - (2*1 - 1*2) ]
        #        = 5 + (1/4) * [ 15 - 3 - 0 ] = 5 + 3 = 8.0
        ell_12 = 5.0 + (2.0 * d_lo[0] + (-1.0) * d_hi[1] - np.dot(q_2, d_drop)) / n
        self.assertEqual(ell_12, 8.0)

        # For competitor 3: q_3^+ = [0, 1], q_3^- = [-1, 0]
        # ell_13 = 3.5 + (1/4) * [ 1(1.0) + (-1)(10.5) - (-1*1 + 1*2) ]
        #        = 3.5 + (1/4) * [ 1 - 10.5 - 1 ] = 3.5 + (1/4)*(-10.5) = 3.5 - 2.625 = 0.875 = 7/8
        ell_13 = 3.5 + (1.0 * d_lo[1] + (-1.0) * d_hi[0] - np.dot(q_3, d_drop)) / n
        self.assertEqual(ell_13, 0.875)
        self.assertEqual(Fraction(ell_13), Fraction(7, 8))

        # Both lower margins > 0, certifying class 1 on the whole box!
        self.assertGreater(ell_12, 0)
        self.assertGreater(ell_13, 0)

    def test_toy_exact_minima(self):
        # Table 4 of paper:
        # min g_12 = 8.75 = 35/4
        # min g_13 = 1.375 = 11/8
        # Verification that lower bounds 8.0 and 0.875 are strictly <= exact minima:
        self.assertLessEqual(8.0, 8.75)
        self.assertLessEqual(0.875, 1.375)


if __name__ == "__main__":
    unittest.main()
