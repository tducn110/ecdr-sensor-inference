#!/usr/bin/env python3
"""tests/test_post_w4_feasibility.py — Verification of Post-Week 4 feasibility study routines."""

from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "output/week1_closeout_20261102/reference/selected_best.pt"

from scripts.post_w4_feasibility_study import (
    ScaledTemporalModel,
    count_parameters,
    estimate_macs,
    run_break_even_scaling_analysis,
    run_optimized_incremental_microbenchmark
)


class TestPostWeek4Feasibility(unittest.TestCase):
    def test_scaled_temporal_models_instantiation_and_forward(self):
        archs = [
            ("SmallCNN1D (Baseline)", 283_692),
            ("MediumCNN1D (1.2M params)", 1_200_000),
            ("TemporalResNet1D (4.8M params)", 4_800_000),
            ("TemporalTransformer1D (12M params)", 12_000_000)
        ]
        x = torch.randn(2, 6, 128)

        for arch_name, approx_params in archs:
            model = ScaledTemporalModel(arch_name)
            params = count_parameters(model)
            self.assertGreater(params, 100_000)
            y = model(x)
            self.assertEqual(y.shape, (2, 12))

    def test_incremental_microbenchmark_runs(self):
        res = run_optimized_incremental_microbenchmark()
        self.assertIn("conv1_standard_slicing_alloc_ms", res)
        self.assertIn("conv1_inplace_ring_buffer_ms", res)
        self.assertIn("latency_reduction_percent", res)


if __name__ == "__main__":
    unittest.main()
