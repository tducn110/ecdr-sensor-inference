"""test_preprocessing_boundary.py — Test preprocessing contract validations and boundary error guards.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 2.2)
and independent review numerical recommendations.
"""

import unittest
import numpy as np
import torch

from ecdr.preprocessing import PreprocessingContract, fit_normalization_stats


class TestPreprocessingBoundary(unittest.TestCase):
    def test_nan_mean_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessingContract.create(
                mean=[np.nan, 0, 0, 0, 0, 0],
                std=[1, 1, 1, 1, 1, 1],
            )

    def test_zero_std_and_zero_sigma_min_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessingContract.create(
                mean=[0, 0, 0, 0, 0, 0],
                std=[0, 0, 0, 0, 0, 0],
                sigma_min=0.0,
            )

    def test_infinite_std_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessingContract.create(
                mean=[0, 0, 0, 0, 0, 0],
                std=[np.inf, 1, 1, 1, 1, 1],
            )

    def test_negative_std_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessingContract.create(
                mean=[0, 0, 0, 0, 0, 0],
                std=[-1.0, 1, 1, 1, 1, 1],
            )

    def test_invalid_input_shapes_rejected(self):
        prep = PreprocessingContract.create(
            mean=[0, 0, 0, 0, 0, 0],
            std=[1, 1, 1, 1, 1, 1],
            window_length=128,
        )

        # Wrong length: (6, 64)
        with self.assertRaises(ValueError):
            prep.normalize(np.zeros((6, 64)))

        # Batched wrong length: (2, 6, 64)
        with self.assertRaises(ValueError):
            prep.normalize(np.zeros((2, 6, 64)))

        # 4D tensor: (1, 1, 6, 128)
        with self.assertRaises(ValueError):
            prep.normalize(np.zeros((1, 1, 6, 128)))

    def test_negative_and_nonfinite_budgets_rejected(self):
        prep = PreprocessingContract.create(
            mean=[0, 0, 0, 0, 0, 0],
            std=[1, 1, 1, 1, 1, 1],
        )

        with self.assertRaises(ValueError):
            prep.raw_to_norm_budget(-0.5)

        with self.assertRaises(ValueError):
            prep.raw_to_norm_budget(np.array([1, 1, -1, 1, 1, 1]))

        with self.assertRaises(ValueError):
            prep.raw_to_norm_budget(np.array([np.nan, 1, 1, 1, 1, 1]))

        with self.assertRaises(ValueError):
            prep.norm_to_raw_budget(-1.0)

    def test_empty_fit_data_rejected(self):
        with self.assertRaises(ValueError):
            fit_normalization_stats(np.zeros((0, 6, 128)))

    def test_overlap_preservation(self):
        prep = PreprocessingContract.create(
            mean=[0.5, -0.2, 0.1, 0.0, 0.3, -0.4],
            std=[1.2, 0.8, 1.5, 0.9, 1.1, 1.3],
        )

        np.random.seed(42)
        stream = np.random.randn(6, 192)
        x0 = stream[:, :128]
        x1 = stream[:, 64:]

        z0 = prep.normalize(x0)
        z1 = prep.normalize(x1)

        # Exact overlap equality: z1[:, :64] == z0[:, 64:]
        np.testing.assert_array_equal(z1[:, :64], z0[:, 64:])

    def test_train_args_rejected(self):
        from scripts.ecdr_train_k12 import train
        with self.assertRaises(ValueError):
            train(epochs=0)
        with self.assertRaises(ValueError):
            train(lr=-0.01)
        with self.assertRaises(ValueError):
            train(lr=float("nan"))
        with self.assertRaises(ValueError):
            train(batch_size=0)

    def test_replay_args_rejected(self):
        from scripts.ecdr_stream_replay import run_replay
        with self.assertRaises(ValueError):
            run_replay(max_windows=0)
        with self.assertRaises(ValueError):
            run_replay(max_windows=-5)


if __name__ == "__main__":
    unittest.main()
