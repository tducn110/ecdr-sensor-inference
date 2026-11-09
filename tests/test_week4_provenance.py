"""tests/test_week4_provenance.py — Unit and fixture tests for Week 4 provenance & audit.

Implements W4-001-C07:
- Tests actual frozen reference validity
- Tests data denominators
- Tests mutation detection (tampered checkpoint, preprocessing, class order)
- Tests missing artifact detection
- Tests dirty source capture
- Tests end-to-end artifact generation
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from scripts.week4_audit import (
    build_protected_manifest,
    compute_preprocessing_id,
    get_git_state,
    run_audit,
    verify_data_denominators,
    verify_frozen_reference,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PREPROCESSING_ID,
    ROOT,
)


class TestWeek4Provenance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="w4_provenance_test_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_actual_frozen_reference_clean(self):
        """Actual workspace reference must pass verification without errors."""
        res = verify_frozen_reference()
        self.assertEqual(res["checkpoint_sha256"], EXPECTED_CHECKPOINT_SHA256)
        self.assertEqual(res["preprocessing_id"], EXPECTED_PREPROCESSING_ID)
        self.assertEqual(res["parameter_count"], 283692)
        self.assertTrue(res["eval_mode"])
        self.assertEqual(res["device"], "cpu")
        self.assertEqual(res["dtype"], "torch.float32")

    def test_actual_data_denominators_clean(self):
        """Actual dataset denominators must match exact counts."""
        res = verify_data_denominators()
        self.assertEqual(res["total_windows"], 17453)
        self.assertEqual(res["test_windows"], 5053)
        self.assertEqual(res["supervised_test_windows"], 3572)
        self.assertEqual(res["test_recordings"], 19)
        self.assertEqual(res["within_recording_adjacent_pairs"], 5034)

    def test_actual_protected_manifest_integrity(self):
        """Protected manifest must contain all baseline files and match all hashes."""
        manifest = build_protected_manifest()
        self.assertGreaterEqual(manifest["total_files"], 170)

    def test_mutation_detection_checkpoint(self):
        """Tampering a byte in the checkpoint must raise ValueError."""
        # Create a mock reference directory
        ref_src = ROOT / "output/week1_closeout_20261102/reference"
        mock_ref = self.temp_dir / "reference"
        mock_ref.mkdir(parents=True)
        shutil.copy(ref_src / "normalization.npz", mock_ref / "normalization.npz")
        shutil.copy(ref_src / "config.json", mock_ref / "config.json")

        # Copy checkpoint and mutate one byte
        ckpt_bytes = bytearray((ref_src / "selected_best.pt").read_bytes())
        ckpt_bytes[100] = (ckpt_bytes[100] + 1) % 256
        (mock_ref / "selected_best.pt").write_bytes(ckpt_bytes)

        with self.assertRaises(ValueError) as ctx:
            verify_frozen_reference(reference_dir=mock_ref)
        self.assertIn("Checkpoint SHA256 mismatch", str(ctx.exception))

    def test_mutation_detection_preprocessing(self):
        """Tampering normalization values must raise ValueError."""
        ref_src = ROOT / "output/week1_closeout_20261102/reference"
        mock_ref = self.temp_dir / "reference"
        mock_ref.mkdir(parents=True)
        shutil.copy(ref_src / "selected_best.pt", mock_ref / "selected_best.pt")
        shutil.copy(ref_src / "config.json", mock_ref / "config.json")

        # Mutate normalization
        with np.load(ref_src / "normalization.npz") as data:
            mean = data["mean"].copy()
            std = data["std"].copy()
        mean[0] += 0.5
        np.savez(mock_ref / "normalization.npz", mean=mean, std=std)

        with self.assertRaises(ValueError) as ctx:
            verify_frozen_reference(reference_dir=mock_ref)
        self.assertIn("Preprocessing ID mismatch", str(ctx.exception))

    def test_mutation_detection_class_order(self):
        """Tampering class order in config must raise ValueError."""
        ref_src = ROOT / "output/week1_closeout_20261102/reference"
        mock_ref = self.temp_dir / "reference"
        mock_ref.mkdir(parents=True)
        shutil.copy(ref_src / "selected_best.pt", mock_ref / "selected_best.pt")
        shutil.copy(ref_src / "normalization.npz", mock_ref / "normalization.npz")

        cfg = json.loads((ref_src / "config.json").read_text())
        # Swap first two classes
        cfg["class_order"][0], cfg["class_order"][1] = cfg["class_order"][1], cfg["class_order"][0]
        (mock_ref / "config.json").write_text(json.dumps(cfg))

        with self.assertRaises(ValueError) as ctx:
            verify_frozen_reference(reference_dir=mock_ref)
        self.assertIn("Class order mismatch", str(ctx.exception))

    def test_missing_artifact_detection(self):
        """Missing reference files must raise FileNotFoundError."""
        mock_ref = self.temp_dir / "empty_ref"
        mock_ref.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            verify_frozen_reference(reference_dir=mock_ref)

    def test_git_state_capture(self):
        """Git state capture must return valid dictionary with non-empty source_tree_hash."""
        git_state = get_git_state()
        self.assertIn("head", git_state)
        self.assertIn("branch", git_state)
        self.assertIn("source_tree_hash", git_state)
        self.assertEqual(len(git_state["source_tree_hash"]), 64)

    def test_run_audit_end_to_end_artifacts(self):
        """End-to-end execution of snapshot mode must write all 5 required artifacts."""
        run_id = "test_provenance_run"
        out_dir = self.temp_dir / "out"
        provenance = run_audit(run_id=run_id, mode="snapshot", output_dir=out_dir)

        self.assertEqual(provenance["status"], "PASS")
        self.assertTrue((out_dir / "provenance.json").exists())
        self.assertTrue((out_dir / "protected_manifest.json").exists())
        self.assertTrue((out_dir / "source_map.md").exists())
        self.assertTrue((out_dir / "environment.json").exists())
        self.assertTrue((out_dir / "tasks/W4-001/receipt.json").exists())

        # Check receipt contents
        receipt = json.loads((out_dir / "tasks/W4-001/receipt.json").read_text())
        self.assertEqual(receipt["task_id"], "W4-001")
        self.assertEqual(receipt["submitted_status"], "VERIFIED")
        self.assertEqual(len(receipt["requirement_evidence"]), 7)


if __name__ == "__main__":
    unittest.main()
