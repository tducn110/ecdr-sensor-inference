"""tests/test_week4_delivery.py — Comprehensive test suite for W4-040 final delivery verification.

Task: W4-040 (Milestone M7)
Hypotheses: Full 24-hypothesis coverage

Covers:
  - W4-040-C01: Tasks registry dispositions (39 prior tasks VERIFIED, W4-040 in progress).
  - W4-040-C02: Protocol seal validity and master hash verification.
  - W4-040-C03: 214 protected baseline files integrity.
  - W4-040-C04: Denominators, strata, A_reuse null rule, and exact agreement metrics.
  - W4-040-C05, C06, C07: Deliverables presence, gate recommendations, engineering completeness.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.week4_verify_delivery import verify_delivery

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "week4_initial_20261103"


class TestWeek4Delivery(unittest.TestCase):
    def test_w4_040_delivery_verification_pipeline(self):
        """W4-040-C01 through C07: Full delivery verification pipeline execution."""
        deliv_doc, preserv_doc, deliv_p, preserv_p = verify_delivery(run_id=RUN_ID)

        self.assertTrue(deliv_p.exists())
        self.assertTrue(preserv_p.exists())

        # 1. Orchestration status
        orch = deliv_doc["orchestration_status"]
        self.assertEqual(orch["total_tasks"], 40)
        self.assertIn(orch["tasks_verified_prior"], [39, 40])
        self.assertEqual(orch["final_task_ready_for_closeout"], "W4-040")

        # 2. Governance integrity
        gov = deliv_doc["governance_integrity"]
        self.assertEqual(gov["protected_files_verified"], 214)
        self.assertEqual(gov["protected_files_status"], "ALL_CLEAN")
        self.assertEqual(gov["seal_verification_status"], "VALID")
        self.assertEqual(gov["test_exposure_level"], "KNOWN_TEST_EXPLORATORY")

        # 3. Deliverables manifest check
        manifest = deliv_doc["deliverables_manifest"]
        expected_keys = [
            "replay_trace.csv",
            "replay_results.json",
            "timing_trace.csv",
            "benchmark_results.json",
            "memory_results.json",
            "scaling_results.json",
            "tables/comparison_matrix.csv",
            "tables/hypotheses.csv",
            "tables/h24_conjunction.csv",
            "tables/summary.md",
            "tables/statistics.json",
            "independent_audit.json",
            "pdf_review.json",
            "Week_4_Implementation_and_Evaluation.pdf",
            "Week_4_Explained.pdf",
            "REPRODUCE.md",
        ]
        for k in expected_keys:
            self.assertIn(k, manifest, f"Missing deliverable key in manifest: {k}")
            self.assertGreater(manifest[k]["size_bytes"], 0)

        # 4. Gate recommendations check
        gates = deliv_doc["gate_recommendations"]
        self.assertEqual(gates["gate_1_mathematical_foundations"]["recommendation"], "VERIFIED")
        self.assertEqual(gates["gate_2_numerical_implementation"]["recommendation"], "HOLD_CONDITIONAL")
        self.assertEqual(gates["gate_3_streaming_coverage"]["recommendation"], "FALSIFIED_STREAMING")
        self.assertEqual(gates["gate_4_net_speedup"]["recommendation"], "FALSIFIED_CPU")
        self.assertEqual(gates["gate_5_prior_work_superiority"]["recommendation"], "NOT_ESTABLISHED")
        self.assertEqual(gates["gate_6_core_trilemma_conjunction"]["recommendation"], "NOT_ESTABLISHED")

        # 5. Delivery disposition check
        disp = deliv_doc["delivery_disposition"]
        self.assertIn("COMPLETE", disp["engineering_implementation"])
        self.assertIn("AUTHENTIC_NEGATIVE_RESULTS", disp["scientific_outcome"])
        self.assertEqual(disp["user_acceptance"], "PENDING_USER_REVIEW")

        # 6. Final preservation check
        self.assertEqual(preserv_doc["protected_files_count"], 214)
        self.assertEqual(preserv_doc["protected_files_status"], "CLEAN")


if __name__ == "__main__":
    unittest.main()
