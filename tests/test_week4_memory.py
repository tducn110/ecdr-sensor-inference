"""tests/test_week4_memory.py — Comprehensive unit test suite for W4-034 memory components, scaling, and membership cost.

Task: W4-034 (Milestone M5)
Hypotheses: H10, H11, H12, H17, H18, H20, H24

Covers all 7 checklist requirements:
  - W4-034-C01: Attributable byte counts without double-counting shared reference model.
  - W4-034-C02: Isolated memory vs process peak RSS containing dataset.
  - W4-034-C03: Membership microbenchmarks (full 768, tail 384, latent, residual) noting Theta(m) worst-case.
  - W4-034-C04: Protocol-declared scaling (rank, horizon, build budgets).
  - W4-034-C05: Coordinates read, workspace allocation, amortization horizon.
  - W4-034-C06: Unmeasured telemetry (GPU, energy, dynamic workspace delta) set to null with reasons.
  - W4-034-C07: Payload accounting de-duplication, correct units, and 17 method metadata records.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ecdr.week4.methods import list_registered_methods
from ecdr.week4.reference import ReferenceOwner
from scripts.week4_run import (
    ProtocolSealMismatchError,
    get_object_deep_size,
    run_memory_analysis,
    verify_protocol_seal,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4Memory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()

    def test_w4_034_c01_baseline_and_owned_byte_separation(self):
        """W4-034-C01: Owned parameter/buffer bytes isolated without double-counting shared baseline model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mem_doc, _, _, _ = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            # Check core float32 tensor payload
            f32_payload = mem_doc["reference_float32_tensor_payload"]
            self.assertEqual(f32_payload["parameter_bytes"], 1134768)
            self.assertEqual(f32_payload["total_tensor_payload_bytes"], 1136072)

            # Check bound snapshot float64 payload
            f64_payload = mem_doc["bound_snapshot_float64_tensor_payload"]
            self.assertEqual(f64_payload["parameter_bytes"], 2269536)
            self.assertEqual(f64_payload["total_tensor_payload_bytes"], 2272120)

            # Verify M01 wrapper does NOT own baseline model weights
            m01_prof = mem_doc["method_memory_profiles"]["M01"]
            self.assertEqual(m01_prof["baseline_owned_model_bytes"], 0)
            self.assertLess(m01_prof["total_attributable_cache_bytes"], 2000)

            # Verify M00 and M16 baseline ownership
            self.assertEqual(mem_doc["method_memory_profiles"]["M00"]["baseline_owned_model_bytes"], 1136072)
            self.assertEqual(mem_doc["method_memory_profiles"]["M16"]["baseline_owned_model_bytes"], 1136072)

    def test_w4_034_c02_process_peak_rss_scope_clarification(self):
        """W4-034-C02: Process peak RSS explicitly documented as containing dataset, not cache overhead."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mem_doc, _, _, _ = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            env = mem_doc["process_memory_environment"]
            self.assertGreater(env["process_peak_rss_kb"], 50000)  # > 50 MB
            self.assertIn("frozen HAPT dataset arrays", env["process_rss_scope"])
            self.assertIn("Not attributable to adapter cache", env["process_rss_scope"])

    def test_w4_034_c03_membership_microbenchmarks(self):
        """W4-034-C03: Full 768, tail 384, latent PCA, and residual checks evaluated with Theta(m) notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mem_doc, _, _, _ = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            mb = mem_doc["membership_microbenchmark"]
            self.assertIn("full_768_coordinate_box", mb)
            self.assertIn("tail_384_fixed_prefix", mb)
            self.assertIn("latent_pca_k16", mb)
            self.assertIn("residual_norm_check", mb)

            # Check coordinates and complexity
            self.assertEqual(mb["full_768_coordinate_box"]["coordinates_checked"], 768)
            self.assertEqual(mb["full_768_coordinate_box"]["worst_case_complexity"], "Theta(m)")
            self.assertGreater(mb["full_768_coordinate_box"]["mean_latency_ns"], 0)

            self.assertEqual(mb["tail_384_fixed_prefix"]["coordinates_checked"], 768)
            self.assertEqual(mb["tail_384_fixed_prefix"]["worst_case_complexity"], "Theta(m)")

            self.assertEqual(mb["latent_pca_k16"]["coordinates_checked"], 384)
            self.assertEqual(mb["latent_pca_k16"]["worst_case_complexity"], "Theta(m * k)")

            self.assertEqual(mb["residual_norm_check"]["coordinates_checked"], 384)
            self.assertEqual(mb["residual_norm_check"]["worst_case_complexity"], "Theta(m)")

    def test_w4_034_c04_scaling_sweeps(self):
        """W4-034-C04: Rank (1..32), horizon (1..32), and build budget scaling adhere to protocol caps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _, scale_doc, _, _ = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            # Rank scaling checks
            r_scale = scale_doc["rank_scaling"]
            for r in [1, 2, 4, 8, 16, 32]:
                k = f"rank_{r}"
                self.assertIn(k, r_scale)
                self.assertEqual(r_scale[k]["rank"], r)
                self.assertEqual(r_scale[k]["basis_matrix_bytes"], 384 * r * 4)

            # Horizon scaling checks
            h_scale = scale_doc["horizon_scaling"]
            for h in [1, 2, 4, 8, 16, 32]:
                k = f"horizon_{h}"
                self.assertIn(k, h_scale)
                self.assertEqual(h_scale[k]["horizon_steps"], h)
                self.assertEqual(h_scale[k]["total_certificate_cache_bytes"], h * 6144)

            # Build budget scaling checks
            b_scale = scale_doc["build_budget_scaling"]
            self.assertIn("budget_1e-4", b_scale)
            self.assertIn("budget_1e-3", b_scale)
            self.assertIn("budget_1e-2", b_scale)
            self.assertIn("budget_1e-1", b_scale)
            self.assertIn("crown_build_time_ms", b_scale["budget_1e-4"])

    def test_w4_034_c05_coordinates_read_and_amortization_scope(self):
        """W4-034-C05: Logs coordinates read, allocation/workspace, and amortization scopes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mem_doc, _, _, _ = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            profiles = mem_doc["method_memory_profiles"]
            # M00 reads full 768
            self.assertEqual(profiles["M00"]["coordinates_read"], 768)
            self.assertEqual(profiles["M00"]["amortization_scope"], "per_query")

            # M03 reads 384 tail coordinates
            self.assertEqual(profiles["M03"]["coordinates_read"], 384)
            self.assertEqual(profiles["M03"]["amortization_scope"], "per_window_stream")

    def test_w4_034_c06_unmeasured_telemetry_null_with_reasons(self):
        """W4-034-C06: GPU memory, energy, and dynamic workspace delta set to null with valid reasons."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mem_doc, _, _, _ = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            env = mem_doc["process_memory_environment"]
            self.assertIsNone(env["gpu_memory_bytes"])
            self.assertIn("Host CPU environment", env["gpu_memory_reason"])

            self.assertIsNone(env["energy_joules"])
            self.assertIn("Hardware power/RAPL", env["energy_reason"])

            self.assertIsNone(env["dynamic_workspace_rss_delta"])
            self.assertIn("heap fragmentation", env["dynamic_workspace_reason"])

    def test_w4_034_c07_all_17_methods_retention_and_no_double_counting(self):
        """W4-034-C07: Retention of all 17 registered methods in memory profiles and recursive de-duplication."""
        # Test deep size de-duplication on shared object
        shared_tensor = torch.zeros((100, 100), dtype=torch.float32)  # 40,000 bytes
        container_a = [shared_tensor, shared_tensor]
        size_a = get_object_deep_size(container_a)
        # Should count tensor exactly once, not twice
        self.assertLess(size_a, 50000)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            mem_doc, _, mem_path, scale_path = run_memory_analysis(
                run_id="week4_initial_20261103",
                output_dir=out_dir,
            )

            self.assertTrue(mem_path.exists())
            self.assertTrue(scale_path.exists())

            profiles = mem_doc["method_memory_profiles"]
            self.assertEqual(len(profiles), 17)

            for mid in ["M14", "M15"]:
                self.assertIn(mid, profiles)
                self.assertEqual(profiles[mid]["execution_status"], "AUDITED_UNSUPPORTED")
                self.assertEqual(profiles[mid]["total_attributable_cache_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
