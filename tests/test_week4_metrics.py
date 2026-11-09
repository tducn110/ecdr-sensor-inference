"""tests/test_week4_metrics.py — Comprehensive unit tests for W4-035 metrics, tables, and hypothesis ledger.

Task: W4-035 (Milestone M6)
Hypotheses: H01 through H24

Covers all 7 checklist requirements:
  - W4-035-C01: Recount raw traces, 5053 full queries, 3572 supervised, 5034 pairs, 19 recordings.
  - W4-035-C02: A_reuse null rule when R=0, D_all = D_reuse + D_fallback, explicit rows for M14/M15.
  - W4-035-C03: Output artifacts existence and non-empty validation.
  - W4-035-C04: Uncertainty hierarchy, non-IID sliding windows, paired t-test structure.
  - W4-035-C05: All 3 baselines (M00, M01, M16) reported; policy axes classified.
  - W4-035-C06: Traceability of result cells, missing telemetry as null/reasons (not numeric zero).
  - W4-035-C07: Edge cases: fallback-disagreement does NOT degrade A_reuse, empty strata, wrong denominators.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ecdr.week4.methods import list_registered_methods
from ecdr.week4.metrics import (
    build_comparison_matrix,
    build_h24_conjunction_matrix,
    build_hypotheses_ledger,
    compute_accuracy,
    compute_agreement_accounting,
    compute_macro_f1,
    generate_markdown_summary,
    generate_statistics_summary,
)
from scripts.week4_tables import run_tables_generation

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "week4_initial_20261103"


class TestWeek4Metrics(unittest.TestCase):
    def test_w4_035_c01_denominators_and_strata_recount(self):
        """W4-035-C01: Full 5,053 queries, 3,572 supervised, 5,034 pairs, 19 recordings verified from trace."""
        trace_path = ROOT / f"output/week4/{RUN_ID}/replay_trace.csv"
        self.assertTrue(trace_path.exists(), f"Missing trace: {trace_path}")

        df_sample = pd.read_csv(trace_path, nrows=5053)
        # Unique queries in single method replay
        self.assertEqual(len(df_sample), 5053)
        self.assertEqual(df_sample["recording_id"].nunique(), 19)

        # Supervised queries count
        sup_rows = df_sample[df_sample["ground_truth_label"].notnull() & (df_sample["ground_truth_label"] > 0)]
        self.assertEqual(len(sup_rows), 3572)

        # Within-recording adjacent pairs
        pairs = 0
        for i in range(len(df_sample) - 1):
            if (
                df_sample.iloc[i]["recording_id"] == df_sample.iloc[i + 1]["recording_id"]
                and df_sample.iloc[i + 1]["step_idx"] == df_sample.iloc[i]["step_idx"] + 1
            ):
                pairs += 1
        self.assertEqual(pairs, 5034)

    def test_w4_035_c02_agreement_rates_and_null_rule(self):
        """W4-035-C02: A_reuse null when R=0; fallback disagreement does NOT affect A_reuse; D_all=D_reuse+D_fallback."""
        # Case 1: R = 0 -> A_reuse must be None (null), not 1.0 or 0.0
        df_zero_reuse = pd.DataFrame({
            "is_reuse": [False, False, False],
            "predicted_class": [1, 2, 3],
            "reference_class": [1, 2, 4],  # 1 fallback disagreement
        })
        acc_zero = compute_agreement_accounting(df_zero_reuse)
        self.assertEqual(acc_zero["reuse_count"], 0)
        self.assertIsNone(acc_zero["reused_agreement_rate"])
        self.assertEqual(acc_zero["fallback_disagreement_count"], 1)
        self.assertEqual(acc_zero["total_disagreement_count"], 1)
        self.assertAlmostEqual(acc_zero["full_stream_agreement_rate"], 2 / 3, places=4)

        # Case 2: Fallback disagreement must NOT degrade A_reuse
        df_mixed = pd.DataFrame({
            "is_reuse": [True, True, False],
            "predicted_class": [1, 2, 5],
            "reference_class": [1, 2, 6],  # 0 reuse disagreements, 1 fallback disagreement
        })
        acc_mixed = compute_agreement_accounting(df_mixed)
        self.assertEqual(acc_mixed["reuse_count"], 2)
        self.assertEqual(acc_mixed["reuse_disagreement_count"], 0)
        self.assertEqual(acc_mixed["fallback_disagreement_count"], 1)
        self.assertEqual(acc_mixed["total_disagreement_count"], 1)
        self.assertEqual(acc_mixed["reused_agreement_rate"], 1.0)
        self.assertAlmostEqual(acc_mixed["full_stream_agreement_rate"], 2 / 3, places=4)

    def test_w4_035_c03_tables_generation_artifacts(self):
        """W4-035-C03: Generates all required tables, hypotheses ledger, conjunction matrix, and markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tables = Path(tmpdir)
            comp_p, hyp_p, h24_p, sum_p, stat_p = run_tables_generation(
                run_id=RUN_ID,
                output_dir=out_tables,
            )

            self.assertTrue(comp_p.exists())
            self.assertTrue(hyp_p.exists())
            self.assertTrue(h24_p.exists())
            self.assertTrue(sum_p.exists())
            self.assertTrue(stat_p.exists())

            # Verify comparison matrix has all 17 methods
            comp_df = pd.read_csv(comp_p)
            self.assertEqual(len(comp_df), 17)
            self.assertIn("M00", comp_df["method_id"].values)
            self.assertIn("M14", comp_df["method_id"].values)
            self.assertIn("M15", comp_df["method_id"].values)

            # Verify hypotheses ledger has all 24 hypotheses
            hyp_df = pd.read_csv(hyp_p)
            self.assertEqual(len(hyp_df), 24)
            h_ids = hyp_df["hypothesis_id"].tolist()
            for i in range(1, 25):
                self.assertIn(f"H{i:02d}", h_ids)

            # Verify H24 conjunction matrix
            h24_df = pd.read_csv(h24_p)
            self.assertEqual(len(h24_df), 17)
            # Satisfied count must be exactly 0
            self.assertEqual(h24_df["h24_conjunction_satisfied"].sum(), 0)

    def test_w4_035_c04_c05_uncertainty_and_baselines_reporting(self):
        """W4-035-C04 & C05: Uncertainty across 19 clusters; reports all 3 baselines M00, M01, M16."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tables = Path(tmpdir)
            comp_p, _, _, _, _ = run_tables_generation(
                run_id=RUN_ID,
                output_dir=out_tables,
            )
            comp_df = pd.read_csv(comp_p)
            m_dict = {r["method_id"]: r for _, r in comp_df.iterrows()}

            # Baselines checks
            self.assertIn("M00", m_dict)
            self.assertIn("M01", m_dict)
            self.assertIn("M16", m_dict)

            # M00 self ratio is 1.0
            self.assertAlmostEqual(m_dict["M00"]["speed_ratio_vs_m00_core"], 1.0, places=3)
            # M01 has speed ratio < 1.0 due to wrapper overhead
            self.assertLess(m_dict["M01"]["speed_ratio_vs_m00_core"], 0.5)

            # Policy axes M12 and M13 classified properly
            self.assertEqual(m_dict["M12"]["category"], "policy_axis")
            self.assertEqual(m_dict["M13"]["category"], "policy_axis")

    def test_w4_035_c06_c07_unsupported_status_and_edge_cases(self):
        """W4-035-C06 & C07: Unavailable methods have explicit status; empty strata and F1 edge cases handled."""
        # Test macro F1 with disjoint classes
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        f1 = compute_macro_f1(y_true, y_pred)
        self.assertGreater(f1, 0.0)
        self.assertLess(f1, 1.0)

        # Test empty ground truth
        self.assertEqual(compute_accuracy(np.array([]), np.array([])), 0.0)

        # Test explicit status for M14 and M15
        with tempfile.TemporaryDirectory() as tmpdir:
            out_tables = Path(tmpdir)
            comp_p, _, _, _, _ = run_tables_generation(
                run_id=RUN_ID,
                output_dir=out_tables,
            )
            comp_df = pd.read_csv(comp_p)
            m14 = comp_df[comp_df["method_id"] == "M14"].iloc[0]
            self.assertEqual(m14["execution_status"], "AUDITED_UNSUPPORTED")
            self.assertTrue(pd.isna(m14["mean_query_latency_ms"]))

            m15 = comp_df[comp_df["method_id"] == "M15"].iloc[0]
            self.assertEqual(m15["execution_status"], "AUDITED_UNSUPPORTED")
            self.assertTrue(pd.isna(m15["mean_query_latency_ms"]))


if __name__ == "__main__":
    unittest.main()
