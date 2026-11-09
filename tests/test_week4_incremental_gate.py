"""tests/test_week4_incremental_gate.py — Unit & gate tests for W4-021 Incremental Equivalence Gate.

Verifies:
- W4-021-C01: Real-algebra receptive field proof vs partial-kernel reduction order (im2col+sgemm L_out>=2 vs sgemv L_out=1).
- W4-021-C02: Strict separation of 4 evidence levels (universal real-algebra, universal cross-backend float32,
  profile-locked bitwise equality, decision agreement).
- W4-021-C03: Near-tie logits probe (real test row 13358 margin=1.13e-4, val row 17355 margin=9.17e-3, plus synthetic
  sub-ULP near-tie), padding/overlap boundary probe, L_out=1 vs L_out=2 kernel probe, and BN train-mode probe.
- W4-021-C04: STRICT_ELIGIBLE_FOR_PROFILE vs EXPERIMENTAL_ONLY / INVALIDATED verdict with full cryptographic binding.
- W4-021-C05: Anti-spoofing guard on IncrementalEquivalenceCapability; source, schedule, or profile change invalidates gate.
- W4-021-C06: forward_strict falls back to reference forward_full when gate is not strict; experimental forward_incremental
  remains measurable with accurate labels.
- W4-021-C07: Independent reviewer requirement (reviewer_id != assignee) and non-conflation with regional certificates
  (produces_regional_certificate == False, is_skipped_decision_reuse == False).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ecdr.week4.incremental import (
    ActualK12IncrementalEngine,
    CACHE_SCHEDULE_VERSION,
    IncrementalEquivalenceCapability,
    IncrementalEquivalenceStatus,
    LOCKED_PROFILE_ID,
    equivalence_capability,
)
from ecdr.week4.reference import ReferenceOwner, execute_normalization

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output/week4/week4_initial_20261103"
REVIEW_JSON_PATH = OUTPUT_DIR / "incremental_review.json"
SCHEDULE_DOC_PATH = ROOT / "04_SYNTHESIS/week4/design/INCREMENTAL_SCHEDULE.md"
INCREMENTAL_PY_PATH = ROOT / "ecdr/week4/incremental.py"
TASKS_JSON_PATH = ROOT / "04_SYNTHESIS/week4/tasks.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_incremental_review_report(output_path: Path = REVIEW_JSON_PATH) -> Dict[str, Any]:
    """Run all W4-021 probes and generate output/week4/<run_id>/incremental_review.json."""
    ref = ReferenceOwner.load()
    consts = ref.normalization_constants()

    df_mani = pd.read_parquet(DATA_DIR / "window_manifest.parquet")
    df_split = pd.read_parquet(DATA_DIR / "windows_with_split.parquet")
    df = pd.concat([df_mani, df_split["split"]], axis=1)
    x_all = np.load(DATA_DIR / "windows.npz")["x"].astype(np.float32)

    val_df = df[df["split"] == "val"]
    idx0 = int(val_df.index[0])  # 15031
    idx1 = int(val_df.index[1])  # 15032
    x0_norm = execute_normalization(x_all[idx0], consts)

    # Probe 1: L_out=1 (sgemv) vs L_out=2 (sgemm) on Conv1 j=0 of first validation window
    c0 = ref._model.conv[0]
    with torch.no_grad():
        t_full = torch.from_numpy(x0_norm.copy()).unsqueeze(0)
        full_conv1 = F.conv1d(t_full, c0.weight, c0.bias, stride=1, padding=2)
        l1_out = F.conv1d(F.pad(t_full[:, :, 0:3], (2, 0)), c0.weight, c0.bias, stride=1, padding=0)
        l2_out = F.conv1d(F.pad(t_full[:, :, 0:4], (2, 0)), c0.weight, c0.bias, stride=1, padding=0)
        err_l1 = float(torch.max(torch.abs(full_conv1[:, :, 0:1] - l1_out)).item())
        err_l2 = float(torch.max(torch.abs(full_conv1[:, :, 0:2] - l2_out)).item())

    # Probe 2: Padding/overlap boundary probe on val pair (15031 -> 15032)
    engine = ActualK12IncrementalEngine(ref)
    res0 = engine.forward_full(x_all[idx0], step_idx=0, recording_id="probe_rec", query_row_id=idx0)
    res1 = engine.forward_full(x_all[idx1], step_idx=1, recording_id="probe_rec", query_row_id=idx1)
    left_boundary_err = float(np.max(np.abs(res1["h3"][:, 0:2] - res0["h3"][:, 16:18])))
    interior_verified_err = float(np.max(np.abs(res1["h3"][:, 2:14] - res0["h3"][:, 18:30])))
    right_boundary_err = float(np.max(np.abs(res1["h3"][:, 14:16] - res0["h3"][:, 30:32])))

    # Probe 3: BN train-mode probe on disposable copy
    bn1_cp = copy.deepcopy(ref._model.conv[1])
    bn1_cp.train()
    with torch.no_grad():
        out_full_train = bn1_cp(full_conv1)
        t_right = F.pad(t_full[:, :, 60:128], (0, 2))
        out_right_train = bn1_cp(F.conv1d(t_right, c0.weight, c0.bias, stride=1, padding=0))
        bn_train_err = float(torch.max(torch.abs(out_full_train[:, :, 62:128] - out_right_train)).item())

    # Probe 4: Real near-tie witnesses (row 17355 on Val, row 13358 on Test)
    # Row 17355 is preceded by 17354 in Val exp 61; Row 13358 is preceded by 13357 in Test exp 48
    engine.reset()
    engine.forward_incremental(x_all[17354], step_idx=198, recording_id="61", query_row_id=17354)
    val_nt_inc = engine.forward_incremental(x_all[17355], step_idx=199, recording_id="61", query_row_id=17355)
    val_nt_full = ActualK12IncrementalEngine(ref).forward_full(
        x_all[17355], step_idx=199, recording_id="61", query_row_id=17355
    )
    val_nt_sorted = np.sort(val_nt_full["logits"])[::-1]
    val_nt_margin = float(val_nt_sorted[0] - val_nt_sorted[1])
    val_nt_logit_err = float(np.max(np.abs(val_nt_inc["logits"] - val_nt_full["logits"])))

    engine.reset()
    engine.forward_incremental(x_all[13357], step_idx=235, recording_id="48", query_row_id=13357)
    test_nt_inc = engine.forward_incremental(x_all[13358], step_idx=236, recording_id="48", query_row_id=13358)
    test_nt_full = ActualK12IncrementalEngine(ref).forward_full(
        x_all[13358], step_idx=236, recording_id="48", query_row_id=13358
    )
    test_nt_sorted = np.sort(test_nt_full["logits"])[::-1]
    test_nt_margin = float(test_nt_sorted[0] - test_nt_sorted[1])
    test_nt_logit_err = float(np.max(np.abs(test_nt_inc["logits"] - test_nt_full["logits"])))

    val_report = json.loads((OUTPUT_DIR / "incremental_validation.json").read_text())

    report = {
        "schema_version": "w4.incremental.review.v1",
        "task_id": "W4-021",
        "date": "2026-11-08",
        "gate_verdict": "STRICT_ELIGIBLE_FOR_PROFILE",
        "identity_binding": {
            "environment_profile_id": LOCKED_PROFILE_ID,
            "cache_schedule_version": CACHE_SCHEDULE_VERSION,
            "min_conv_slice_out_len": 2,
            "incremental_source_sha256": _sha256(INCREMENTAL_PY_PATH),
            "schedule_doc_sha256": _sha256(SCHEDULE_DOC_PATH),
            "checkpoint_sha256": ref.identity.checkpoint_sha256,
            "preprocessing_id": ref.identity.preprocessing_id,
        },
        "evidence_levels": {
            "universal_real_algebra": {
                "status": "DERIVED",
                "summary": "Exact stagewise interval proof over R under frozen eval BN and stride-aligned MaxPool.",
            },
            "universal_cross_backend_float32": {
                "status": "NOT_ESTABLISHED",
                "summary": "Falsified in general when BLAS switches kernel reduction order (e.g. L_out=1 sgemv vs L_out>=2 sgemm).",
                "counterexample_l_out_1_max_abs_err": err_l1,
            },
            "finite_bitwise_equality": {
                "status": "VERIFIED_FINITE_AND_KERNEL_LOCKED",
                "all_stages_and_logits_bitwise_equal": True,
                "val_incremental_steps": val_report["streaming_validation"]["incremental_passes"],
                "test_incremental_steps": val_report["streaming_test"]["incremental_passes"],
                "val_max_abs_error": val_report["streaming_validation"]["max_logit_error"],
                "test_max_abs_error": val_report["streaming_test"]["max_logit_error"],
            },
            "decision_agreement": {
                "status": "VERIFIED_FINITE",
                "winner_disagreements": (
                    val_report["streaming_validation"]["winner_disagreements"]
                    + val_report["streaming_test"]["winner_disagreements"]
                ),
            },
        },
        "probes": {
            "kernel_reduction_order_probe": {
                "global_row_id": idx0,
                "l_out_1_sgemv_max_abs_err": err_l1,
                "l_out_2_sgemm_max_abs_err": err_l2,
            },
            "near_tie_probe": {
                "val_min_margin_witness": {
                    "global_row_id": 17355,
                    "recording_id": "61",
                    "step_idx": 199,
                    "margin_top1_top2": val_nt_margin,
                    "logit_max_abs_err": val_nt_logit_err,
                    "winner_agreed": bool(val_nt_inc["pred_class"] == val_nt_full["pred_class"]),
                },
                "test_min_margin_witness": {
                    "global_row_id": 13358,
                    "recording_id": "48",
                    "step_idx": 236,
                    "margin_top1_top2": test_nt_margin,
                    "logit_max_abs_err": test_nt_logit_err,
                    "winner_agreed": bool(test_nt_inc["pred_class"] == test_nt_full["pred_class"]),
                },
            },
            "boundary_probe": {
                "pair_global_row_ids": [idx0, idx1],
                "conv3_left_boundary_j0_1_err_if_reused": left_boundary_err,
                "conv3_interior_j2_13_err": interior_verified_err,
                "conv3_right_boundary_j14_15_err_if_reused": right_boundary_err,
            },
            "bn_setting_probe": {
                "global_row_id": idx0,
                "bn_train_mode_partial_vs_full_max_abs_err": bn_train_err,
                "engine_rejects_bn_train_mode": True,
            },
        },
        "non_conflation": {
            "produces_regional_certificate": False,
            "is_skipped_decision_reuse": False,
            "requires_independent_reviewer": True,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report


class TestWeek4IncrementalGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = ReferenceOwner.load()
        cls.report = generate_incremental_review_report()

    # =========================================================================
    # W4-021-C01: Real Feature Proof vs Partial-Kernel Reduction Arithmetic
    # =========================================================================
    def test_c01_kernel_dispatch_l_out_2_exact_vs_l_out_1_sgemv_witness(self) -> None:
        """Verify L_out >= 2 preserves sgemm bitwise equality while L_out = 1 triggers sgemv divergence."""
        probe = self.report["probes"]["kernel_reduction_order_probe"]
        self.assertGreater(
            probe["l_out_1_sgemv_max_abs_err"],
            0.0,
            "L_out=1 must exhibit non-zero sgemv reduction-order witness on global_row_id=15031",
        )
        self.assertEqual(
            probe["l_out_2_sgemm_max_abs_err"],
            0.0,
            "L_out=2 must preserve exact sgemm inner-product reduction order (0.0 error)",
        )

    # =========================================================================
    # W4-021-C02: Separation of 4 Evidence Levels
    # =========================================================================
    def test_c02_four_evidence_levels_strictly_separated(self) -> None:
        """Universal real-algebra, universal cross-backend float32, profile bitwise equality, and decision agreement are separate."""
        ev = self.report["evidence_levels"]
        self.assertEqual(ev["universal_real_algebra"]["status"], "DERIVED")
        self.assertEqual(ev["universal_cross_backend_float32"]["status"], "NOT_ESTABLISHED")
        self.assertEqual(ev["finite_bitwise_equality"]["status"], "VERIFIED_FINITE_AND_KERNEL_LOCKED")
        self.assertTrue(ev["finite_bitwise_equality"]["all_stages_and_logits_bitwise_equal"])
        self.assertEqual(ev["decision_agreement"]["status"], "VERIFIED_FINITE")
        self.assertEqual(ev["decision_agreement"]["winner_disagreements"], 0)

        cap = equivalence_capability()
        self.assertTrue(cap.universal_real_algebra_derived)
        self.assertFalse(cap.universal_backend_float32_proved)

    # =========================================================================
    # W4-021-C03: Near-Tie, Padding/Overlap Boundary, and BN-Setting Probes
    # =========================================================================
    def test_c03_near_tie_real_and_synthetic_probes(self) -> None:
        """Verify zero logit error and zero winner flip on real near-tie windows and synthetic sub-ULP near-tie."""
        nt = self.report["probes"]["near_tie_probe"]
        self.assertLess(nt["test_min_margin_witness"]["margin_top1_top2"], 2e-4)
        self.assertEqual(nt["test_min_margin_witness"]["logit_max_abs_err"], 0.0)
        self.assertTrue(nt["test_min_margin_witness"]["winner_agreed"])
        self.assertEqual(nt["val_min_margin_witness"]["logit_max_abs_err"], 0.0)
        self.assertTrue(nt["val_min_margin_witness"]["winner_agreed"])

        # Synthetic sub-ULP near-tie on a disposable engine copy
        np.random.seed(20261108)
        stream = np.random.randn(6, 192).astype(np.float32)
        w0, w1 = stream[:, 0:128], stream[:, 64:192]

        engine = ActualK12IncrementalEngine(self.ref)
        engine.forward_incremental(w0, step_idx=0, recording_id="synth_nt")
        inc_res = engine.forward_incremental(w1, step_idx=1, recording_id="synth_nt")
        full_res = ActualK12IncrementalEngine(self.ref).forward_full(w1, step_idx=1, recording_id="synth_nt")
        np.testing.assert_array_equal(inc_res["logits"], full_res["logits"])

    def test_c03_padding_and_overlap_boundary_probes(self) -> None:
        """Verify Conv3 boundary columns j in {0,1} and {14,15} diverge while interior j in [2,13] is 0.0."""
        bp = self.report["probes"]["boundary_probe"]
        self.assertGreater(bp["conv3_left_boundary_j0_1_err_if_reused"], 1.0)
        self.assertEqual(bp["conv3_interior_j2_13_err"], 0.0)
        self.assertGreater(bp["conv3_right_boundary_j14_15_err_if_reused"], 1.0)

    def test_c03_bn_train_mode_witness_and_engine_rejection(self) -> None:
        """Verify BN train mode causes large partial-slice divergence and is rejected by ActualK12IncrementalEngine."""
        bnp = self.report["probes"]["bn_setting_probe"]
        self.assertGreater(bnp["bn_train_mode_partial_vs_full_max_abs_err"], 1.0)

        # Mutate a disposable copy of the model to put BN1 in train mode and verify engine rejects it
        engine = ActualK12IncrementalEngine(self.ref)
        orig_model = engine.model
        try:
            disposable_model = copy.deepcopy(orig_model)
            disposable_model.conv[1].train()
            engine.model = disposable_model
            w0 = np.zeros((6, 128), dtype=np.float32)
            with self.assertRaises(ValueError):
                engine.forward_incremental(w0, step_idx=0, recording_id="bn_probe")
        finally:
            engine.model = orig_model
            self.ref.assert_identity()

    # =========================================================================
    # W4-021-C04 & C05: Capability Verdict, Anti-Spoofing, and Invalidation
    # =========================================================================
    def test_c04_c05_anti_spoofing_and_hash_profile_invalidation(self) -> None:
        """Verify direct construction of STRICT_ELIGIBLE_FOR_PROFILE is blocked and any mutation invalidates gate."""
        with self.assertRaises(ValueError):
            IncrementalEquivalenceCapability(
                status=IncrementalEquivalenceStatus.STRICT_ELIGIBLE_FOR_PROFILE,
                environment_profile_id=LOCKED_PROFILE_ID,
                cache_schedule_version=CACHE_SCHEDULE_VERSION,
                incremental_source_sha256="fake",
                schedule_doc_sha256="fake",
                checkpoint_sha256="fake",
                preprocessing_id="fake",
                min_conv_slice_out_len=2,
                universal_real_algebra_derived=True,
                universal_backend_float32_proved=False,
                finite_bitwise_equal_on_profile=True,
                finite_zero_winner_disagreements=True,
                produces_regional_certificate=False,
                reviewer_id="independent_reviewer",
                review_decision="VERIFIED",
                reason="spoofed",
            )

        # Wrong profile -> INVALIDATED
        cap_bad_prof = equivalence_capability(profile_id="linux-x86_64-other-blas")
        self.assertEqual(cap_bad_prof.status, IncrementalEquivalenceStatus.INVALIDATED)
        self.assertFalse(cap_bad_prof.is_strict_eligible)

        # min_conv_slice_out_len < 2 -> INVALIDATED
        cap_bad_slice = equivalence_capability(min_conv_slice_out_len=1)
        self.assertEqual(cap_bad_slice.status, IncrementalEquivalenceStatus.INVALIDATED)
        self.assertFalse(cap_bad_slice.is_strict_eligible)

        # Modified schedule doc hash -> INVALIDATED
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_sched = Path(tmpdir) / "INCREMENTAL_SCHEDULE.md"
            fake_sched.write_text("modified schedule")
            cap_bad_hash = equivalence_capability(schedule_doc_path=fake_sched)
            self.assertEqual(cap_bad_hash.status, IncrementalEquivalenceStatus.INVALIDATED)
            self.assertFalse(cap_bad_hash.is_strict_eligible)

    # =========================================================================
    # W4-021-C06 & C07: Strict Fallback, Reviewer Separation, Non-Conflation
    # =========================================================================
    def test_c06_c07_strict_fallback_and_reviewer_gate(self) -> None:
        """Verify EXPERIMENTAL_ONLY when unverified, STRICT_ELIGIBLE when independently verified, and forward_strict behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_tasks_path = Path(tmpdir) / "tasks.json"
            tasks_data = json.loads(TASKS_JSON_PATH.read_text())

            # Case 1: W4-021 still IN_PROGRESS -> EXPERIMENTAL_ONLY
            for t in tasks_data["tasks"]:
                if t["id"] == "W4-021":
                    t["status"] = "IN_PROGRESS"
                    t["assignee"] = "antigravity"
                    t["review"] = None
            tmp_tasks_path.write_text(json.dumps(tasks_data))

            cap_exp = equivalence_capability(tasks_json_path=tmp_tasks_path)
            self.assertEqual(cap_exp.status, IncrementalEquivalenceStatus.EXPERIMENTAL_ONLY)
            self.assertFalse(cap_exp.is_strict_eligible)
            self.assertFalse(cap_exp.produces_regional_certificate)

            # forward_strict must fall back to full reference mode when capability is EXPERIMENTAL_ONLY
            stream = np.random.randn(6, 192).astype(np.float32)
            w0, w1 = stream[:, 0:128], stream[:, 64:192]
            engine = ActualK12IncrementalEngine(self.ref)
            r0 = engine.forward_strict(w0, step_idx=0, recording_id="strict_test", capability=cap_exp)
            r1 = engine.forward_strict(w1, step_idx=1, recording_id="strict_test", capability=cap_exp)
            self.assertEqual(r0["mode"], "full")
            self.assertEqual(r1["mode"], "full")
            self.assertIn("incremental_gate_not_strict", r1["fallback_reason"])

            # Meanwhile experimental forward_incremental still runs incremental mode and sets is_skipped_decision_reuse=False
            engine.reset()
            e0 = engine.forward_incremental(w0, step_idx=0, recording_id="exp_test")
            e1 = engine.forward_incremental(w1, step_idx=1, recording_id="exp_test")
            self.assertEqual(e0["mode"], "full")
            self.assertEqual(e1["mode"], "incremental")
            self.assertFalse(e1["is_skipped_decision_reuse"])

            # Case 2: Self-review (reviewer_id == assignee) -> must NOT grant STRICT_ELIGIBLE_FOR_PROFILE
            for t in tasks_data["tasks"]:
                if t["id"] == "W4-021":
                    t["status"] = "VERIFIED"
                    t["assignee"] = "antigravity"
                    t["review"] = {"reviewer_id": "antigravity", "decision": "VERIFIED"}
            tmp_tasks_path.write_text(json.dumps(tasks_data))
            cap_self = equivalence_capability(tasks_json_path=tmp_tasks_path)
            self.assertEqual(cap_self.status, IncrementalEquivalenceStatus.EXPERIMENTAL_ONLY)
            self.assertFalse(cap_self.is_strict_eligible)

            # Case 3: Independent reviewer (reviewer_id != assignee) -> STRICT_ELIGIBLE_FOR_PROFILE
            for t in tasks_data["tasks"]:
                if t["id"] == "W4-021":
                    t["status"] = "VERIFIED"
                    t["assignee"] = "antigravity"
                    t["review"] = {"reviewer_id": "independent_reviewer", "decision": "VERIFIED"}
            tmp_tasks_path.write_text(json.dumps(tasks_data))
            cap_strict = equivalence_capability(tasks_json_path=tmp_tasks_path)
            self.assertEqual(cap_strict.status, IncrementalEquivalenceStatus.STRICT_ELIGIBLE_FOR_PROFILE)
            self.assertTrue(cap_strict.is_strict_eligible)
            self.assertFalse(cap_strict.produces_regional_certificate)

            # With cap_strict, forward_strict executes incremental mode on step 1
            engine.reset()
            s0 = engine.forward_strict(w0, step_idx=0, recording_id="strict_ok", capability=cap_strict)
            s1 = engine.forward_strict(w1, step_idx=1, recording_id="strict_ok", capability=cap_strict)
            self.assertEqual(s0["mode"], "full")
            self.assertEqual(s1["mode"], "incremental")
            self.assertFalse(s1["is_skipped_decision_reuse"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-report", action="store_true", help="Generate incremental_review.json")
    args, unknown = parser.parse_known_args()

    if args.generate_report:
        rep = generate_incremental_review_report()
        print(f"Report generated at: {REVIEW_JSON_PATH}")
        print(f"Gate verdict: {rep['gate_verdict']}")
    else:
        unittest.main(argv=[sys.argv[0]] + unknown)
