#!/usr/bin/env python3
"""
tests/test_week4_domains.py
---------------------------
Comprehensive unit test suite for Milestone M2 task W4-015:
Causal diagonal/wave proposals and exact prefix contract.

Covers checklist requirements W4-015-C01 through W4-015-C07:
- C01: Generates 768-coordinate full box and 384-free-dimension fixed-overlap wave.
- C02: Uses constant continuation center baseline with zero lookahead or next-input leakage.
- C03: Stores frozen prefix hash, anchor step, recording ID, and tail budgets in domain identity.
- C04: Evaluates independent prefix match, tail bounds, and temporal stream continuity.
- C05: Exposes same-center full-box comparative control.
- C06: Tests H02 ramp counterexample, invalid/negative/nonfinite budgets, and prefix mismatch.
- C07: Validates distance distributions on frozen validation stream without test tuning.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from ecdr.week4.domains import (
    CausalProposalKind,
    CausalWaveProposal,
    check_causal_membership,
    propose_causal_wave_domain,
    propose_same_center_full_box,
    run_wave_domain_validation,
)
from ecdr.week4.reference import ReferenceOwner
from ecdr.week4.types import CandidateBox


class TestWeek4Domains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = ReferenceOwner.load()
        cls.x_all = np.load(REPO_ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(REPO_ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()

        # Anchor window from validation set
        cls.raw_anchor = cls.x_all[cls.val_indices[0]].astype(np.float32)
        cls.norm_anchor = cls.ref.normalize(cls.raw_anchor)

    def test_c01_proposal_dimensions_and_exact_prefix_box(self):
        """W4-015-C01: Wave domain has 384 free tail coordinates; prefix coordinates are exactly fixed."""
        budget = 1e-3
        prop = propose_causal_wave_domain(
            anchor_normalized=self.norm_anchor,
            tail_budget=budget,
            anchor_step=10,
            recording_id="rec_01",
        )

        self.assertEqual(prop.free_dimensions, 384)
        self.assertEqual(prop.kind, CausalProposalKind.FIXED_OVERLAP_WAVE.value)
        self.assertEqual(prop.target_step, 11)

        lo, hi = prop.candidate_box.arrays()
        self.assertEqual(lo.shape, (6, 128))
        self.assertEqual(hi.shape, (6, 128))

        # First 64 steps: lo == hi == anchor suffix (exact prefix)
        self.assertTrue(np.array_equal(lo[:, :64], hi[:, :64]))
        self.assertTrue(np.array_equal(lo[:, :64], self.norm_anchor[:, 64:]))

        # Last 64 steps: free tail interval [center - budget, center + budget]
        expected_center = np.repeat(self.norm_anchor[:, -1:], 64, axis=1)
        np.testing.assert_allclose(lo[:, 64:], expected_center - budget, atol=1e-7)
        np.testing.assert_allclose(hi[:, 64:], expected_center + budget, atol=1e-7)

    def test_c02_constant_continuation_and_no_lookahead_leakage(self):
        """W4-015-C02: Constant continuation baseline depends strictly on observed anchor window."""
        prop = propose_causal_wave_domain(self.norm_anchor, tail_budget=0.05)
        lo, hi = prop.candidate_box.arrays()
        mid = 0.5 * (lo[:, 64:] + hi[:, 64:])

        # Midpoint of tail must exactly equal the last observed step of anchor for each channel
        for c in range(6):
            expected_val = self.norm_anchor[c, -1]
            np.testing.assert_allclose(mid[c, :], expected_val, atol=1e-7)

    def test_c03_provenance_and_domain_identity(self):
        """W4-015-C03: Binds anchor hashes, step, recording ID, and frozen prefix hash into identity."""
        prop = propose_causal_wave_domain(
            anchor_normalized=self.norm_anchor,
            tail_budget=1e-4,
            anchor_step=25,
            recording_id="stream_alpha",
            anchor_raw_hash="hash_raw_123",
            anchor_normalized_hash="hash_norm_456",
        )

        self.assertEqual(prop.anchor_step, 25)
        self.assertEqual(prop.target_step, 26)
        self.assertEqual(prop.recording_id, "stream_alpha")
        self.assertEqual(prop.anchor_raw_hash, "hash_raw_123")
        self.assertEqual(prop.anchor_normalized_hash, "hash_norm_456")
        self.assertTrue(len(prop.frozen_prefix_hash) == 64)
        self.assertTrue(len(prop.domain_id) == 64)

    def test_c04_causal_membership_checks_prefix_and_continuity(self):
        """W4-015-C04: Membership enforces bit-exact prefix, stream continuity, and tail containment."""
        prop = propose_causal_wave_domain(
            anchor_normalized=self.norm_anchor,
            tail_budget=0.1,
            anchor_step=10,
            recording_id="rec_A",
        )

        # 1. Valid exact query next step (perfect continuation)
        lo, hi = prop.candidate_box.arrays()
        query_good = 0.5 * (lo + hi)
        is_member, reason = check_causal_membership(prop, query_good, step_idx=11, recording_id="rec_A")
        self.assertTrue(is_member)
        self.assertEqual(reason, "EXACT_MEMBER")

        # 2. Recording mismatch
        is_member, reason = check_causal_membership(prop, query_good, step_idx=11, recording_id="rec_DIFFERENT")
        self.assertFalse(is_member)
        self.assertEqual(reason, "WRONG_RECORDING")

        # 3. Step gap (jumped to step 12 instead of target 11)
        is_member, reason = check_causal_membership(prop, query_good, step_idx=12, recording_id="rec_A")
        self.assertFalse(is_member)
        self.assertEqual(reason, "STEP_GAP")

        # 4. Prefix mismatch (single bit altered in prefix)
        query_bad_prefix = query_good.copy()
        query_bad_prefix[0, 5] += 1e-4
        is_member, reason = check_causal_membership(prop, query_bad_prefix, step_idx=11, recording_id="rec_A")
        self.assertFalse(is_member)
        self.assertEqual(reason, "PREFIX_MISMATCH")

        # 5. Tail outside (tail value shifted beyond budget)
        query_bad_tail = query_good.copy()
        query_bad_tail[0, 80] += 0.5  # Budget was 0.1
        is_member, reason = check_causal_membership(prop, query_bad_tail, step_idx=11, recording_id="rec_A")
        self.assertFalse(is_member)
        self.assertEqual(reason, "TAIL_OUTSIDE")

    def test_c05_same_center_full_box_control(self):
        """W4-015-C05: Full-box control shares the exact same center but has 768 free coordinates."""
        budget = 1e-3
        prop_wave = propose_causal_wave_domain(self.norm_anchor, tail_budget=budget)
        prop_full = propose_same_center_full_box(self.norm_anchor, budget=budget)

        self.assertEqual(prop_full.free_dimensions, 768)
        self.assertEqual(prop_full.kind, CausalProposalKind.DIAGONAL_BOX.value)

        lo_wave, hi_wave = prop_wave.candidate_box.arrays()
        lo_full, hi_full = prop_full.candidate_box.arrays()

        center_wave = 0.5 * (lo_wave + hi_wave)
        center_full = 0.5 * (lo_full + hi_full)

        # Centers must be identical
        np.testing.assert_allclose(center_wave, center_full, atol=1e-7)

        # Full box widens the prefix, whereas wave fixes it
        self.assertGreater(float(np.max(hi_full[:, :64] - lo_full[:, :64])), 0.0)
        self.assertEqual(float(np.max(hi_wave[:, :64] - lo_wave[:, :64])), 0.0)

    def test_c06_h02_ramp_counterexample_and_invalid_budgets(self):
        """W4-015-C06: H02 Ramp counterexample and fail-closed budget validation."""
        # 1. Budget validation
        with self.assertRaises(ValueError):
            propose_causal_wave_domain(self.norm_anchor, tail_budget=-1e-4)
        with self.assertRaises(ValueError):
            propose_causal_wave_domain(self.norm_anchor, tail_budget=0.0)
        with self.assertRaises(ValueError):
            propose_causal_wave_domain(self.norm_anchor, tail_budget=np.nan)

        # 2. H02 Ramp counterexample:
        # Construct synthetic linear ramp: w_t = [0, 1, 2, ..., 127], w_{t+1} = [64, 65, ..., 191]
        ramp_t = np.tile(np.arange(128, dtype=np.float64), (6, 1))
        ramp_next = np.tile(np.arange(64, 192, dtype=np.float64), (6, 1))

        # Check 50% overlap exact match:
        self.assertTrue(np.array_equal(ramp_next[:, :64], ramp_t[:, 64:]))

        # Despite 50% overlap, distance between full windows is large:
        diff = ramp_next - ramp_t
        l2_dist = float(np.linalg.norm(diff))
        self.assertGreater(l2_dist, 100.0)  # Proves H02: overlap does not imply bounded distance

    def test_c07_validation_stream_evaluation_and_artifact(self):
        """W4-015-C07: Evaluates consecutive pairs in validation split and produces JSON artifact."""
        out = run_wave_domain_validation(self.ref, num_pairs=20)
        self.assertIn("evaluated_consecutive_pairs", out)
        self.assertGreater(out["evaluated_consecutive_pairs"], 0)
        self.assertEqual(out["h01_overlap_exact_match_rate"], 1.0)
        self.assertIn("tail_displacement_statistics", out)
        self.assertIn("membership_rate_by_tail_budget", out)

        artifact_file = Path("output/week4/week4_initial_20261103/wave_domain_validation.json")
        self.assertTrue(artifact_file.exists())


if __name__ == "__main__":
    unittest.main()
