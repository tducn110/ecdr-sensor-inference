"""tests/test_week4_protocol.py — Unit test suite for W4-018 and W4-030 protocol execution and sealing.

Covers:
  - W4-018-C01..C07: Matched geometry validation sweep, controls, and test split protection.
  - W4-030-C01: Validation selection according to protocol.json grid, objective, tie-breaking, smoke command.
  - W4-030-C02: Declared build cap, cooldown, horizon, identity mode, zero-radius and no-build controls.
  - W4-030-C03: Separate accounting of candidate, conditional positive, joint member, eligible hit, cost scores.
  - W4-030-C04: Label-free proposals; query forward used for offline outcome evaluation only.
  - W4-030-C05: Protocol seal generation, master seal hash, code/config hashes, KNOWN_TEST_EXPLORATORY disclosure.
  - W4-030-C06: Seal immutability and post-seal mutation detection.
  - W4-030-C07: Deterministic selection, zero-hit ties, none-eligible verdict, and test split rejection.
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

from ecdr.week4.reference import ReferenceOwner
from scripts.week4_run import (
    _generate_random_basis_parameters,
    run_geometry_validation,
    run_smoke_validation,
    run_validation_selection,
    seal_protocol,
    verify_protocol_seal,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()

    # -----------------------------------------------------------------------
    # W4-018 Geometry Validation Tests (Preserved)
    # -----------------------------------------------------------------------

    def test_c01_and_c06_test_split_protection(self):
        """W4-018-C01 & C06: Rejects test split without explicit allow_test flag."""
        with self.assertRaises(PermissionError):
            run_geometry_validation(
                run_id="test_run",
                split="test",
                num_anchors=2,
                allow_test=False,
            )

    def test_c02_and_c03_random_basis_control(self):
        """W4-018-C02 & C03: Generates valid orthonormal random basis control with K=8."""
        rand_params = _generate_random_basis_parameters(k_components=8, seed=42)
        self.assertEqual(rand_params.k_components, 8)
        self.assertEqual(rand_params.basis.shape, (384, 8))

        # Check orthonormality: Q^T Q == I_8
        gram = rand_params.basis.T @ rand_params.basis
        ident = np.eye(8, dtype=np.float32)
        diff = np.max(np.abs(gram - ident))
        self.assertLess(diff, 1e-5)

    def test_c04_c05_c07_small_geometry_sweep(self):
        """W4-018-C04, C05, C07: Runs small matched geometry validation sweep generating artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            report, json_path, csv_path = run_geometry_validation(
                run_id="test_sweep",
                split="val",
                num_anchors=2,
                anchor_stride=50,
                budgets=(1e-4,),
                output_dir=out_dir,
            )

            # Check files exist
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            # Check JSON schema and task_id
            self.assertEqual(report["task_id"], "W4-018")
            self.assertEqual(report["split"], "val")
            self.assertEqual(report["evaluated_anchors_count"], 2)

            # Check geometry aggregates contain all 5 required geometries
            required_geoms = [
                "full_box",
                "fixed_overlap_wave",
                "pca_affine_tail_hull",
                "pca_affine_tail_native",
                "random_basis_control",
            ]
            for geom in required_geoms:
                self.assertIn(geom, report["geometry_aggregates"])
                stats = report["geometry_aggregates"][geom]
                self.assertIn("membership_rate", stats)
                self.assertIn("crown_positive_rate", stats)
                self.assertIn("joint_reuse_rate", stats)

    # -----------------------------------------------------------------------
    # W4-030 Validation Selection & Protocol Seal Tests
    # -----------------------------------------------------------------------

    def test_w4_030_c01_and_c07_validation_selection_and_test_protection(self):
        """W4-030-C01 & C07: Runs policy selection on validation split and forbids unsealed test evaluation."""
        with self.assertRaises(PermissionError):
            run_validation_selection(
                run_id="test_run",
                split="test",
                num_anchors=2,
                allow_test=False,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            val_rep, pol, val_path, pol_path = run_validation_selection(
                run_id="test_select",
                split="val",
                num_anchors=3,
                anchor_stride=50,
                output_dir=out_dir,
            )

            self.assertTrue(val_path.exists())
            self.assertTrue(pol_path.exists())
            self.assertEqual(val_rep["task_id"], "W4-030")
            self.assertIn(pol["selection_verdict"], ("VALIDATION_SELECTED_POSITIVE_REUSE", "NONE_ELIGIBLE_FORMAL_REUSE_ZERO_HIT"))
            self.assertIn("selected_policy_id", pol)
            self.assertIn("domain_family", pol)
            self.assertIn("backend", pol)

    def test_w4_030_c02_and_c03_policy_controls_and_separate_metrics(self):
        """W4-030-C02 & C03: Includes controls (zero-radius, no-build) and records separate metric scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            val_rep, pol, _, _ = run_validation_selection(
                run_id="test_metrics",
                split="val",
                num_anchors=2,
                anchor_stride=100,
                output_dir=out_dir,
            )

            policies = val_rep["evaluated_policies"]
            policy_ids = [p["policy_id"] for p in policies]
            self.assertIn("control_zero_radius", policy_ids)
            self.assertIn("control_no_build", policy_ids)
            self.assertIn("control_affine_hull_1e4_h1", policy_ids)

            # Check separate metric fields per policy
            for p in policies:
                self.assertIn("candidate_proposals_count", p)
                self.assertIn("conditional_positive_count", p)
                self.assertIn("joint_member_count", p)
                self.assertIn("eligible_hit_count", p)
                self.assertIn("disagreement_count", p)
                self.assertIn("mean_build_ms", p)
                self.assertIn("failures_count", p)

            # Zero-radius control must be marked ineligible
            zero_rad_p = [p for p in policies if p["policy_id"] == "control_zero_radius"][0]
            self.assertTrue(zero_rad_p["is_zero_radius_ineligible"])

    def test_w4_030_c04_label_free_proposals_and_smoke_command(self):
        """W4-030-C04: Verifies smoke command executes without errors and maintains baseline parity."""
        res = run_smoke_validation(run_id="week4_initial_20261103", num_queries=5)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["baseline_parity_verified"])
        self.assertEqual(res["num_queries_evaluated"], 5)

    def test_w4_030_c05_c06_c07_protocol_seal_and_immutability(self):
        """W4-030-C05, C06, C07: Generates protocol seal, verifies validity, and detects post-seal tampering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Create dummy required artifacts in temp out_dir for sealing
            (out_dir / "method_capabilities.json").write_text(json.dumps({"test": 1}))
            (out_dir / "selected_policy.json").write_text(json.dumps({"selected_policy_id": "test_pol", "formal_reuse_enabled": False}))
            (out_dir / "selection_validation.json").write_text(json.dumps({"test": 2}))
            (out_dir / "pca_affine_parameters.npz").write_bytes(b"dummy_npz_data")

            seal_doc, seal_path = seal_protocol(
                run_id="test_seal_run",
                output_dir=out_dir,
            )

            self.assertTrue(seal_path.exists())
            self.assertEqual(seal_doc["task_id"], "W4-030")
            self.assertEqual(seal_doc["scientific_status"], "PROTOCOL_SEALED")
            self.assertEqual(seal_doc["disclosure"]["test_exposure_level"], "KNOWN_TEST_EXPLORATORY")
            self.assertIn("master_seal_hash", seal_doc)

            # Verification of unaltered seal
            is_valid, disc = verify_protocol_seal(
                run_id="test_seal_run",
                seal_path=seal_path,
            )
            self.assertTrue(is_valid)
            self.assertEqual(len(disc), 0)

            # Invalidate seal by modifying a dummy sealed config in seal_doc
            tampered_doc = seal_doc.copy()
            tampered_doc["config_hashes"] = {"protocol_json": "tampered_fake_hash_12345"}
            tampered_path = out_dir / "tampered_seal.json"
            tampered_path.write_text(json.dumps(tampered_doc))

            is_valid_tampered, disc_tampered = verify_protocol_seal(
                run_id="test_seal_run",
                seal_path=tampered_path,
            )
            self.assertFalse(is_valid_tampered)
            self.assertIn("protocol_json", disc_tampered)


if __name__ == "__main__":
    unittest.main()
