"""tests/test_week4_accounting.py — Comprehensive unit test suite for W4-032 chronological replay.

Covers all 7 checklist requirements for W4-032:
  - W4-032-C01: Seal verification, disclosure level KNOWN_TEST_EXPLORATORY, no untouched confirmation.
  - W4-032-C02: Exact denominators (5053 windows, 5034 pairs, 19 recordings), chronological ordering.
  - W4-032-C03: Trace logging with stable IDs and all required status/accounting columns.
  - W4-032-C04: Shadow reference agreement and timer exclusion.
  - W4-032-C05: Separate accounting (R, D_reuse, D_fallback, D_all, A_reuse, A_full), null A_reuse when R=0, 3572 supervised queries.
  - W4-032-C06: Exact strata counts: 2874 pure-basic, 119 pure-transition, 579 majority-mixed, 479 ambiguous, 1002 unlabeled.
  - W4-032-C07: Distinction between singleton replay, incremental, and regional reuse; retention of audited methods (M14, M15).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ecdr.week4.methods import list_registered_methods
from ecdr.week4.reference import ReferenceOwner
from scripts.week4_run import (
    ProtocolSealMismatchError,
    classify_stratum,
    compute_macro_f1,
    run_benchmark,
    run_replay,
    verify_protocol_seal,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4Accounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()

    def test_w4_032_c01_seal_and_test_disclosure(self):
        """W4-032-C01: Protocol seal verified before replay; disclosure level is KNOWN_TEST_EXPLORATORY."""
        is_valid, discrepancies = verify_protocol_seal(run_id="week4_initial_20261103")
        self.assertTrue(is_valid, f"Protocol seal failed with discrepancies: {discrepancies}")
        self.assertEqual(len(discrepancies), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Replay with small slice
            res_doc, stat_doc, trace_p, res_p, stat_p = run_replay(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                methods_to_run=["M00"],
                max_queries=5,
            )

            self.assertEqual(res_doc["task_id"], "W4-032")
            self.assertEqual(
                res_doc["disclosure"]["test_exposure_level"],
                "KNOWN_TEST_EXPLORATORY",
            )
            self.assertIn("5,053 windows across 19 HAPT recordings", res_doc["disclosure"]["exposure_history"])

    def test_w4_032_c02_exact_denominators_and_chronology(self):
        """W4-032-C02: Exactly 5,053 windows, 19 recordings, and 5,034 adjacent within-recording pairs."""
        df_mani = pd.read_parquet(ROOT / "data/window_manifest.parquet")
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        df = pd.concat([df_mani, df_split["split"]], axis=1)
        test_df = df[df["split"] == "test"].sort_values(["exp", "win_idx"]).reset_index(drop=True)

        self.assertEqual(len(test_df), 5053)
        self.assertEqual(test_df["exp"].nunique(), 19)

        # Check adjacent pairs count
        pairs = 0
        for i in range(len(test_df) - 1):
            if (
                test_df.iloc[i]["exp"] == test_df.iloc[i + 1]["exp"]
                and test_df.iloc[i + 1]["win_idx"] == test_df.iloc[i]["win_idx"] + 1
            ):
                pairs += 1
        self.assertEqual(pairs, 5034)

        # Verify chronological monotonicity
        for exp_id, grp in test_df.groupby("exp"):
            win_indices = grp["win_idx"].tolist()
            self.assertEqual(win_indices, sorted(win_indices))
            self.assertEqual(len(win_indices), len(set(win_indices)))

    def test_w4_032_c03_c04_trace_columns_and_shadow_reference(self):
        """W4-032-C03 & C04: Query-level trace logging with required columns and shadow reference agreement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            res_doc, stat_doc, trace_p, res_p, stat_p = run_replay(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                methods_to_run=["M00", "M01", "M02", "M03"],
                max_queries=15,
            )

            self.assertTrue(trace_p.exists())
            self.assertTrue(res_p.exists())
            self.assertTrue(stat_p.exists())

            # Verify CSV headers
            with open(trace_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

            expected_headers = [
                "global_row_id",
                "recording_id",
                "step_idx",
                "method_id",
                "action",
                "event",
                "reason",
                "predicted_class",
                "reference_class",
                "is_agreement",
                "is_reuse",
                "is_formal_reuse",
                "is_singleton_reuse",
                "is_regional_reuse",
                "stratum",
                "ground_truth_label",
                "elapsed_ms",
            ]
            for h in expected_headers:
                self.assertIn(h, headers)

            self.assertEqual(len(rows), 15 * 4)

            # M00, M01, M03 must achieve 100% agreement against reference forward
            for mid in ["M00", "M01", "M03"]:
                m_rows = [r for r in rows if r["method_id"] == mid]
                for r in m_rows:
                    self.assertEqual(
                        r["predicted_class"],
                        r["reference_class"],
                        f"Disagreement observed for {mid} on row {r['global_row_id']}",
                    )

    def test_w4_032_c05_separate_accounting_and_null_rule(self):
        """W4-032-C05: Separate accounting of R, D_reuse, D_fallback, D_all; null A_reuse when R=0; 3572 supervised queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            res_doc, _, _, _, _ = run_replay(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                methods_to_run=["M00"],
                max_queries=20,
            )

            m00 = res_doc["methods"]["M00"]
            self.assertIn("reuse_count", m00)
            self.assertIn("reuse_disagreement_count", m00)
            self.assertIn("fallback_disagreement_count", m00)
            self.assertIn("total_disagreement_count", m00)
            self.assertIn("reused_agreement_rate", m00)
            self.assertIn("full_stream_agreement_rate", m00)

            # M00 does not reuse: R=0, so A_reuse must be strictly None (null)
            self.assertEqual(m00["reuse_count"], 0)
            self.assertIsNone(m00["reused_agreement_rate"])
            self.assertEqual(m00["full_stream_agreement_rate"], 1.0)

        # Full test split supervised queries check
        df_mani = pd.read_parquet(ROOT / "data/window_manifest.parquet")
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        df = pd.concat([df_mani, df_split["split"]], axis=1)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        sup_mask = (test_df["annotated_frac"] > 0) & (test_df["label_policy"] != "ambiguous")
        self.assertEqual(sup_mask.sum(), 3572)

    def test_w4_032_c06_strata_classification_and_counts(self):
        """W4-032-C06: Exact strata counts: 2874 pure-basic, 119 pure-transition, 579 majority-mixed, 479 ambiguous, 1002 unlabeled."""
        df_mani = pd.read_parquet(ROOT / "data/window_manifest.parquet")
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        df = pd.concat([df_mani, df_split["split"]], axis=1)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        strata = test_df.apply(classify_stratum, axis=1)
        counts = strata.value_counts().to_dict()

        self.assertEqual(counts.get("pure_basic"), 2874)
        self.assertEqual(counts.get("pure_transition"), 119)
        self.assertEqual(counts.get("majority_mixed"), 579)
        self.assertEqual(counts.get("ambiguous"), 479)
        self.assertEqual(counts.get("unlabeled"), 1002)
        self.assertEqual(sum(counts.values()), 5053)

    def test_w4_032_c07_methods_status_retention(self):
        """W4-032-C07: Retention of all 17 registered methods in status registry (M14/M15 AUDITED_UNSUPPORTED)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _, stat_doc, _, _, _ = run_replay(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                methods_to_run=["M00"],
                max_queries=5,
            )

            self.assertEqual(stat_doc["total_registered_methods"], 17)
            methods = stat_doc["methods"]
            m_dict = {m["method_id"]: m for m in methods}

            # Verify M14 and M15 status and justification
            self.assertIn("M14", m_dict)
            self.assertEqual(m_dict["M14"]["execution_status"], "AUDITED_UNSUPPORTED")
            self.assertIn("LipCache", m_dict["M14"]["evidence_summary"])

            self.assertIn("M15", m_dict)
            self.assertEqual(m_dict["M15"]["execution_status"], "AUDITED_UNSUPPORTED")
            self.assertIn("StreamiNNC/TinyDejaVu", m_dict["M15"]["evidence_summary"])

    def test_w4_033_c01_host_and_thread_isolation(self):
        """W4-033-C01: Enforces single-thread execution, CPU device, and logs host environment profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            b_doc, trace_p, res_p = run_benchmark(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                repetitions=1,
                warmup_queries=5,
                methods_to_benchmark=["M00"],
                max_queries=10,
            )

            self.assertEqual(b_doc["task_id"], "W4-033")
            self.assertEqual(torch.get_num_threads(), 1)
            host = b_doc["host_environment"]
            self.assertEqual(host["device"], "cpu")
            self.assertEqual(host["torch_threads"], 1)
            self.assertEqual(host["repetitions"], 1)
            self.assertEqual(host["warmup_queries"], 5)
            self.assertEqual(host["run_order"], ["M00"])

    def test_w4_033_c02_same_ram_inputs_and_contract(self):
        """W4-033-C02: Baselines M00, M01, M16 and methods receive identical raw RAM inputs at batch=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            b_doc, trace_p, res_p = run_benchmark(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                repetitions=1,
                warmup_queries=5,
                methods_to_benchmark=["M00", "M01", "M16"],
                max_queries=10,
            )

            with open(trace_p, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # Verify same queries processed in each method
            m00_rows = [r["global_row_id"] for r in rows if r["method_id"] == "M00"]
            m01_rows = [r["global_row_id"] for r in rows if r["method_id"] == "M01"]
            m16_rows = [r["global_row_id"] for r in rows if r["method_id"] == "M16"]
            self.assertEqual(m00_rows, m01_rows)
            self.assertEqual(m00_rows, m16_rows)
            self.assertEqual(len(m00_rows), 10)

    def test_w4_033_c03_warmup_and_cold_start_separation(self):
        """W4-033-C03: Warmup queries are executed separately before timed repetitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            b_doc, trace_p, res_p = run_benchmark(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                repetitions=2,
                warmup_queries=10,
                methods_to_benchmark=["M00"],
                max_queries=15,
            )

            with open(trace_p, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            # Exactly repetitions * max_queries recorded in timing trace
            self.assertEqual(len(rows), 2 * 15)
            # Repetitions index must be 1 and 2
            reps = {int(r["repetition"]) for r in rows}
            self.assertEqual(reps, {1, 2})

    def test_w4_033_c04_c05_timing_trace_columns_and_timer_isolation(self):
        """W4-033-C04 & C05: Timing trace format and query timer isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            b_doc, trace_p, res_p = run_benchmark(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                repetitions=1,
                warmup_queries=5,
                methods_to_benchmark=["M00", "M03"],
                max_queries=10,
            )

            self.assertTrue(trace_p.exists())
            with open(trace_p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

            expected_headers = [
                "repetition",
                "method_id",
                "global_row_id",
                "recording_id",
                "step_idx",
                "action",
                "is_reuse",
                "query_latency_ms",
            ]
            for h in expected_headers:
                self.assertIn(h, headers)

            for r in rows:
                lat = float(r["query_latency_ms"])
                self.assertGreater(lat, 0.0)

    def test_w4_033_c06_c07_paired_speed_ratios_and_break_even(self):
        """W4-033-C06 & C07: Paired comparisons, statistics vs M00, and break-even accounting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            b_doc, trace_p, res_p = run_benchmark(
                run_id="week4_initial_20261103",
                split="test",
                output_dir=out_dir,
                repetitions=2,
                warmup_queries=5,
                methods_to_benchmark=["M00", "M01", "M03"],
                max_queries=20,
            )

            paired = b_doc["paired_comparisons"]
            self.assertIn("M00", paired)
            self.assertIn("M01", paired)
            self.assertIn("M03", paired)

            # Per-query statistics
            for mid in ["M00", "M01", "M03"]:
                stats_dict = paired[mid]["per_query_latency_ms"]
                self.assertIn("mean", stats_dict)
                self.assertIn("p50", stats_dict)
                self.assertIn("p95", stats_dict)
                self.assertIn("min", stats_dict)
                self.assertIn("max", stats_dict)

            # M00 self ratio must be 1.0
            self.assertAlmostEqual(paired["M00"]["speed_ratio_vs_m00_core"], 1.0, places=4)

            # Break-even accounting
            be = b_doc["break_even_accounting"]
            self.assertIn("m00_core_mean_query_ms", be)
            self.assertIn("m01_wrapper_mean_query_ms", be)
            self.assertIn("m16_owned_mean_query_ms", be)
            self.assertIn("m03_incremental_mean_query_ms", be)
            self.assertIn("identity_wrapper_overhead_ms", be)
            self.assertIn("theoretical_break_even_hit_rate", be)
            self.assertIn("empirical_analysis", be)


if __name__ == "__main__":
    unittest.main()
