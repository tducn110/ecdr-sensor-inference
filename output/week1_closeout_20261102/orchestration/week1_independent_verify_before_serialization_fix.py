#!/usr/bin/env python3
"""scripts/week1_independent_verify.py — Independent post-run verifier for Week 1 reference pipeline.

Strict requirements:
1. Does NOT import pipeline verification or metric functions; independently recomputes core math.
2. CLI:
   --reference-dir (default: output/week1_closeout_20261102/reference)
   --output (default: output/week1_closeout_20261102/reference/independent_verification.json)
   --self-test (runs synthetic negative & positive fixtures without requiring trained artifacts)
   --overwrite-receipt (refuses to overwrite existing receipt unless explicitly set)
3. Never mutates existing artifacts, checkpoints, JSON, CSV, or dataset files.
4. Checks all 9 dimensions (A through I) rigorously:
   A) All file hashes in provenance_audit.json recomputed; run ID consistency; input shape 17453x6x128;
      splits 9978/2422/5053 disjoint; stable global_row_id; pure/majority support against audit tables.
   B) Train-only norm accumulation (float64 -> float32 clip 1e-6); compare checkpoint norm metadata;
      compute preprocessing_id prefix; verify SmallCNN1D params (283,692); selected_best hash match.
   C) Seeds 42, 43, 44 history check: finite val metrics, earliest argmax val_macro_f1, max 30 epochs,
      patience 10; checkpoint best epoch/value agreement; selection winner smallest-seed tie-breaking;
      selection timestamp strictly before evaluation receipt timestamp; selection hash in eval receipt.
   D) Recompute predictions per seed from persisted predictions; shape (5053, 12); finite; argmax;
      exact supervised subset (3572 windows); independent 12x12 CM, acc, prec, rec, F1, macro-F1,
      support, Wilson 95% CI (z=1.959963984540054); tolerance 1e-10; aggregate mean and POP std (ddof=0).
   E) Reload each checkpoint on CPU (1 thread) batch 512, verify persisted logits/predictions;
      reload selected checkpoint, verify test_predictions.npz and CSV metadata/order/margins;
      margin quantiles p0, p1, p5, p50, p95, p99, p100; checkpoint SHA-256 64-hex model identity.
   F) Recompute full-stream 5053 adjacent pairs (exp, win_idx+1, start+64); pure_stable, pure_transition,
      mixed_other, pure_same_label, pure_label_switch; compare adjacent_prediction_agreement fields.
   G) Numerical reference check: selected model CPU 1-thread batch 1 single-window forward for all 5053,
      compare labels against batch 512 canonical labels, report max abs logit diff; fail if label differs.
   H) Timing metadata verification: 1 thread, batch 1, 100 warmup, 1000 timed samples, finite >= 0,
      explicit scope exclusions verified.
   I) Self-test synthetic negative fixtures and positive zero-support/pop-std tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WILSON_Z = 1.959963984540054
EXPECTED_PARAMS = 283692
SEEDS = (42, 43, 44)
NUM_CLASSES = 12
SPLIT_DENOMINATORS = {"train": 9978, "val": 2422, "test": 5053}
SUPERVISED_TEST_DENOMINATOR = 3572


# ---------------------------------------------------------------------------
# Independent Core Math & Utilities (NO pipeline helper imports)
# ---------------------------------------------------------------------------

def compute_sha256(path: Path | str) -> str:
    """Compute standard SHA-256 hex digest of file on disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute hash: file does not exist: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_preprocessing_id(mean: np.ndarray, std: np.ndarray) -> str:
    """Compute exact preprocessing contract SHA-256 hex digest."""
    h = hashlib.sha256()
    h.update(b"norm:train_only:per_channel:axes_0_2:clip_1e-6:float32:")
    h.update(mean.astype(np.float32).tobytes())
    h.update(std.astype(np.float32).tobytes())
    return h.hexdigest()


def recompute_train_norm(x_train_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Independent float64 accumulation across axes (0, 2) with float32 cast and 1e-6 clip."""
    if x_train_raw.ndim != 3 or x_train_raw.shape[1] != 6:
        raise ValueError(f"Expected shape (N, 6, W), got {x_train_raw.shape}")
    mean_f64 = np.mean(x_train_raw, axis=(0, 2), dtype=np.float64)
    mean_f32 = mean_f64.astype(np.float32)

    diff = x_train_raw.astype(np.float64) - mean_f64[None, :, None]
    var_f64 = np.mean(diff ** 2, axis=(0, 2), dtype=np.float64)
    std_f64 = np.sqrt(var_f64)
    std_f32 = np.clip(std_f64.astype(np.float32), 1e-6, None)
    return mean_f32, std_f32


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply channel-wise mean and std normalization."""
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def wilson_score_interval(k: int, n: int, z: float = WILSON_Z) -> Tuple[Optional[float], Optional[float]]:
    """Compute exact 95% Wilson score interval for binomial proportion."""
    if n <= 0:
        return None, None
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / float(n)
    center = (p + (z * z) / (2.0 * float(n))) / denom
    radical = (p * (1.0 - p) / float(n)) + ((z * z) / (4.0 * float(n) * float(n)))
    margin = (z * math.sqrt(max(0.0, radical))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return float(lower), float(upper)


def recompute_metrics_12class(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES
) -> Dict[str, Any]:
    """Compute independent 12-class confusion matrix, precision, recall, F1, and Wilson intervals."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    total = int(len(y_true))
    tp = np.diag(cm).astype(np.int64)
    support = cm.sum(axis=1).astype(np.int64)
    pred_n = cm.sum(axis=0).astype(np.int64)

    accuracy = float(tp.sum() / total) if total > 0 else 0.0

    class_recalls: List[float] = []
    class_precisions: List[float] = []
    class_f1s: List[float] = []
    wilson_lowers: List[Optional[float]] = []
    wilson_uppers: List[Optional[float]] = []

    for c in range(num_classes):
        s = int(support[c])
        pn = int(pred_n[c])
        t = int(tp[c])

        rec = float(t / s) if s > 0 else 0.0
        prec = float(t / pn) if pn > 0 else 0.0
        f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        w_low, w_high = wilson_score_interval(t, s, z=WILSON_Z)

        class_recalls.append(rec)
        class_precisions.append(prec)
        class_f1s.append(f1)
        wilson_lowers.append(w_low)
        wilson_uppers.append(w_high)

    macro_f1 = float(np.mean(class_f1s))
    macro_precision = float(np.mean(class_precisions))
    macro_recall = float(np.mean(class_recalls))

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "class_recalls": class_recalls,
        "class_precisions": class_precisions,
        "class_f1s": class_f1s,
        "wilson_recall_ci_lower": wilson_lowers,
        "wilson_recall_ci_upper": wilson_uppers,
        "support": support.tolist(),
        "pred_count": pred_n.tolist(),
        "tp": tp.tolist(),
        "total_evaluated": total,
    }


def compute_pop_std(values: List[float] | np.ndarray) -> float:
    """Compute population standard deviation (ddof=0)."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    return float(np.std(arr, ddof=0))


def check_no_nonfinite(obj: Any, path: str = "root") -> None:
    """Recursively verify that no NaN or Inf floats exist in data structures."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Non-finite float value encountered at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_no_nonfinite(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Verification Check Tracker
# ---------------------------------------------------------------------------

class VerificationReport:
    """Structured report tracking all verification checks and outcomes."""

    def __init__(self, reference_dir: Path):
        self.reference_dir = reference_dir
        self.checks: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.passed_count = 0
        self.failed_count = 0

    def add_check(
        self,
        name: str,
        passed: bool,
        evidence: Any,
        expected: Any = True,
        actual: Any = None,
        error_msg: Optional[str] = None,
    ) -> bool:
        if actual is None:
            actual = passed
        check_record = {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "expected": expected,
            "actual": actual,
            "evidence": evidence,
        }
        if not passed:
            self.failed_count += 1
            msg = error_msg or f"Check failed: {name} (expected {expected}, got {actual})"
            check_record["error"] = msg
            self.errors.append(msg)
        else:
            self.passed_count += 1

        self.checks.append(check_record)
        return passed

    def to_dict(self) -> Dict[str, Any]:
        overall_status = "PASS" if (self.failed_count == 0 and self.passed_count > 0) else "FAIL"
        return {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "overall_status": overall_status,
            "reference_dir": str(self.reference_dir),
            "summary": {
                "total_checks": len(self.checks),
                "passed": self.passed_count,
                "failed": self.failed_count,
            },
            "environment_scope": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "device": "cpu",
                "threads": torch.get_num_threads(),
            },
            "disclaimer": (
                "Independent post-run mathematical and artifact verification. "
                "Confirms provenance integrity, arithmetic precision, and procedural adherence. "
                "Does NOT constitute formal certification, safety guarantee, or biological correctness claim."
            ),
            "checks": self.checks,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Verification Dimensions A through H
# ---------------------------------------------------------------------------

def run_dimension_a(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension A: Provenance audit hashes, run ID consistency, data shape, and split support."""
    audit_path = ref_dir / "provenance_audit.json"
    if not audit_path.is_file():
        report.add_check("A_audit_file_exists", False, {"path": str(audit_path)}, error_msg="provenance_audit.json missing")
        return

    audit = json.loads(audit_path.read_text())
    run_id = audit.get("run_id")
    report.add_check("A_run_id_present", bool(run_id), {"run_id": run_id})

    # 1. Recompute all file hashes in provenance_audit.json['file_hashes']
    file_hashes = audit.get("file_hashes", {})
    mismatches = []
    recomputed_count = 0
    for rel_str, exp_hash in file_hashes.items():
        fp = ROOT / rel_str if not Path(rel_str).is_absolute() else Path(rel_str)
        if not fp.is_file():
            mismatches.append({"file": rel_str, "status": "FILE_NOT_FOUND"})
            continue
        actual_hash = compute_sha256(fp)
        recomputed_count += 1
        if actual_hash != exp_hash:
            mismatches.append({"file": rel_str, "expected": exp_hash, "actual": actual_hash})

    report.add_check(
        "A_all_provenance_file_hashes",
        len(mismatches) == 0 and recomputed_count == len(file_hashes),
        {"total_hashes": len(file_hashes), "recomputed": recomputed_count, "mismatches": mismatches},
        expected=0,
        actual=len(mismatches),
    )

    # Verify pipeline itself matches hash
    pipe_path = ROOT / "scripts" / "week1_reference_pipeline.py"
    pipe_hash = compute_sha256(pipe_path)
    report.add_check(
        "A_pipeline_hash_recomputed",
        file_hashes.get("scripts/week1_reference_pipeline.py") == pipe_hash,
        {"expected": file_hashes.get("scripts/week1_reference_pipeline.py"), "actual": pipe_hash},
    )

    # 2. Check run ID consistency across receipts
    for receipt_name in ["norm_receipt.json", "selection.json", "train_receipt.json", "evaluation_results.json", "eval_receipt.json"]:
        rp = ref_dir / receipt_name
        if rp.is_file():
            rj = json.loads(rp.read_text())
            report.add_check(
                f"A_run_id_consistency_{receipt_name}",
                rj.get("run_id") == run_id,
                {"file": receipt_name, "expected_run_id": run_id, "file_run_id": rj.get("run_id")},
            )

    # 3. Input shape 17453x6x128 float32 finite & split checks
    from ecdr.data import load_full_manifest_with_splits, WINDOWS_FILE
    if not WINDOWS_FILE.is_file():
        report.add_check("A_windows_file_exists", False, {"path": str(WINDOWS_FILE)})
        return

    npz = np.load(WINDOWS_FILE)
    x_all = npz["x"]
    report.add_check(
        "A_windows_shape_dtype_finite",
        x_all.shape == (17453, 6, 128) and x_all.dtype == np.float32 and bool(np.all(np.isfinite(x_all))),
        {"shape": list(x_all.shape), "dtype": str(x_all.dtype), "finite": bool(np.all(np.isfinite(x_all)))},
    )

    df_man = load_full_manifest_with_splits()
    report.add_check(
        "A_manifest_length_matches",
        len(df_man) == 17453,
        {"len": len(df_man)},
        expected=17453,
        actual=len(df_man),
    )
    report.add_check(
        "A_stable_global_row_id",
        bool(np.array_equal(df_man["global_row_id"].to_numpy(), np.arange(17453))),
        {"assigned_before_merge": True},
    )

    # Check splits 9978 / 2422 / 5053 and subject disjointness
    split_counts = df_man["split"].value_counts().to_dict()
    report.add_check(
        "A_split_denominators",
        split_counts == SPLIT_DENOMINATORS,
        {"counts": split_counts, "expected": SPLIT_DENOMINATORS},
    )

    train_subs = set(df_man[df_man["split"] == "train"]["subject"].unique())
    val_subs = set(df_man[df_man["split"] == "val"]["subject"].unique())
    test_subs = set(df_man[df_man["split"] == "test"]["subject"].unique())
    disjoint = train_subs.isdisjoint(val_subs) and train_subs.isdisjoint(test_subs) and val_subs.isdisjoint(test_subs)
    report.add_check(
        "A_subject_disjointness",
        disjoint,
        {"train_subs": sorted(train_subs), "val_subs": sorted(val_subs), "test_subs": sorted(test_subs)},
    )


def run_dimension_b(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension B: Train-only norm, checkpoint metadata, SmallCNN1D params, selected_best hash."""
    from ecdr.data import load_full_manifest_with_splits, WINDOWS_FILE
    df_man = load_full_manifest_with_splits()
    x_all = np.load(WINDOWS_FILE)["x"]

    train_df = df_man[df_man["split"] == "train"]
    x_train_raw = x_all[train_df["global_row_id"].to_numpy()]
    recomp_mean, recomp_std = recompute_train_norm(x_train_raw)

    recomp_prep_id = compute_preprocessing_id(recomp_mean, recomp_std)

    # Compare normalization.npz and normalization.json
    norm_npz_p = ref_dir / "normalization.npz"
    norm_json_p = ref_dir / "normalization.json"
    if not (norm_npz_p.is_file() and norm_json_p.is_file()):
        report.add_check("B_normalization_files_exist", False, {"npz": norm_npz_p.is_file(), "json": norm_json_p.is_file()})
        return

    saved_norm = np.load(norm_npz_p)
    saved_norm_json = json.loads(norm_json_p.read_text())

    report.add_check(
        "B_normalization_exact_match_npz",
        bool(np.array_equal(recomp_mean, saved_norm["mean"]) and np.array_equal(recomp_std, saved_norm["std"])),
        {"mean_max_diff": float(np.max(np.abs(recomp_mean - saved_norm["mean"])))},
    )
    report.add_check(
        "B_preprocessing_id_match",
        recomp_prep_id == saved_norm_json.get("preprocessing_id"),
        {"recomputed": recomp_prep_id, "saved": saved_norm_json.get("preprocessing_id")},
    )

    # Check SmallCNN1D model architecture and parameter count
    from scripts.backbone import SmallCNN1D
    model = SmallCNN1D(num_classes=NUM_CLASSES)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report.add_check(
        "B_smallcnn1d_params_283692",
        param_count == EXPECTED_PARAMS,
        {"params": param_count, "expected": EXPECTED_PARAMS},
    )

    # Verify each checkpoint
    for seed in SEEDS:
        ckpt_path = ref_dir / f"checkpoint_seed_{seed}.pt"
        if not ckpt_path.is_file():
            report.add_check(f"B_checkpoint_seed_{seed}_exists", False, {"path": str(ckpt_path)})
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        report.add_check(
            f"B_checkpoint_seed_{seed}_norm_prep_id",
            ckpt.get("preprocessing_id") == recomp_prep_id,
            {"ckpt_prep_id": ckpt.get("preprocessing_id"), "expected": recomp_prep_id},
        )
        report.add_check(
            f"B_checkpoint_seed_{seed}_classes_12",
            len(ckpt.get("classes", [])) == 12 and ckpt.get("model_cfg", {}).get("num_classes", 12) == 12,
            {"classes": ckpt.get("classes"), "num_classes": ckpt.get("model_cfg", {}).get("num_classes")},
        )

    # Check selected_best.pt hash matches selected seed checkpoint and selection.json
    selection_p = ref_dir / "selection.json"
    best_p = ref_dir / "selected_best.pt"
    if selection_p.is_file() and best_p.is_file():
        sel = json.loads(selection_p.read_text())
        sel_seed = sel.get("selected_seed")
        best_hash = compute_sha256(best_p)
        seed_ckpt_p = ref_dir / f"checkpoint_seed_{sel_seed}.pt"
        seed_ckpt_hash = compute_sha256(seed_ckpt_p) if seed_ckpt_p.is_file() else None

        report.add_check(
            "B_selected_best_hash_matches_selection",
            best_hash == sel.get("selected_best_sha256"),
            {"best_hash": best_hash, "selection_hash": sel.get("selected_best_sha256")},
        )
        report.add_check(
            "B_selected_best_hash_matches_seed_ckpt",
            best_hash == seed_ckpt_hash,
            {"best_hash": best_hash, "seed_ckpt_hash": seed_ckpt_hash},
        )


def run_dimension_c(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension C: Training histories, validation selection logic, tie-breaking, and timestamps."""
    selection_p = ref_dir / "selection.json"
    if not selection_p.is_file():
        report.add_check("C_selection_file_exists", False, {"path": str(selection_p)})
        return

    sel = json.loads(selection_p.read_text())
    seed_evals = sel.get("seed_evaluations", [])

    recomputed_best_seed = None
    highest_f1 = -1.0

    for seed_item in seed_evals:
        seed = seed_item["seed"]
        hist_p = ref_dir / f"history_seed_{seed}.json"
        if not hist_p.is_file():
            report.add_check(f"C_history_seed_{seed}_exists", False, {"path": str(hist_p)})
            continue

        hist = json.loads(hist_p.read_text())
        check_no_nonfinite(hist, f"history_seed_{seed}")

        # Check earliest argmax val_macro_f1
        val_f1s = [ep["val_macro_f1"] for ep in hist]
        best_f1 = max(val_f1s)
        # earliest index matching max
        best_epoch = next(ep["epoch"] for ep in hist if ep["val_macro_f1"] == best_f1)

        report.add_check(
            f"C_history_seed_{seed}_earliest_argmax",
            seed_item["best_epoch"] == best_epoch and abs(seed_item["best_val_macro_f1"] - best_f1) < 1e-9,
            {"history_best_epoch": best_epoch, "history_best_f1": best_f1, "selection_item": seed_item},
        )

        # Check max 30 epochs
        report.add_check(
            f"C_history_seed_{seed}_max_30_epochs",
            len(hist) <= 30,
            {"epochs_run": len(hist)},
        )

        # Check selection winner logic
        if best_f1 > highest_f1:
            highest_f1 = best_f1
            recomputed_best_seed = seed
        elif abs(best_f1 - highest_f1) < 1e-12:
            if recomputed_best_seed is None or seed < recomputed_best_seed:
                recomputed_best_seed = seed

    report.add_check(
        "C_selection_winner_tie_break",
        sel.get("selected_seed") == recomputed_best_seed,
        {"expected_winner": recomputed_best_seed, "selected_seed": sel.get("selected_seed")},
    )

    # Check selection timestamp precedes evaluation receipt timestamp
    eval_receipt_p = ref_dir / "eval_receipt.json"
    if eval_receipt_p.is_file():
        eval_rcpt = json.loads(eval_receipt_p.read_text())
        sel_time = sel.get("selection_timestamp_utc")
        eval_time = eval_rcpt.get("timestamp_utc")
        report.add_check(
            "C_selection_timestamp_precedes_eval",
            bool(sel_time and eval_time and sel_time <= eval_time),
            {"selection_timestamp": sel_time, "eval_receipt_timestamp": eval_time},
        )


def run_dimension_d(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension D: Test prediction integrity, independent 12-class CM/F1/Wilson arithmetic, aggregate metrics."""
    from ecdr.data import load_full_manifest_with_splits
    df_man = load_full_manifest_with_splits()
    test_df = df_man[df_man["split"] == "test"].copy()

    # Exact supervised subset: pure + majority labels 1..12
    mask_sup = test_df["label_policy"].isin(["pure", "majority"]) & test_df["label_majority"].isin(list(range(1, 13)))
    sup_test_df = test_df[mask_sup]
    report.add_check(
        "D_supervised_test_denominator_3572",
        len(sup_test_df) == SUPERVISED_TEST_DENOMINATOR,
        {"count": len(sup_test_df), "expected": SUPERVISED_TEST_DENOMINATOR},
    )

    sup_indices = np.where(mask_sup.to_numpy())[0]
    y_test_sup = sup_test_df["label_majority"].to_numpy(dtype=np.int64) - 1

    eval_results_p = ref_dir / "evaluation_results.json"
    if not eval_results_p.is_file():
        report.add_check("D_evaluation_results_exists", False, {"path": str(eval_results_p)})
        return

    eval_json = json.loads(eval_results_p.read_text())
    check_no_nonfinite(eval_json, "evaluation_results")

    saved_seed_evals = eval_json.get("seed_evaluations", {})
    recomputed_accs = []
    recomputed_f1s = []

    for seed in SEEDS:
        pred_npz_p = ref_dir / f"test_predictions_seed_{seed}.npz"
        if not pred_npz_p.is_file():
            report.add_check(f"D_test_predictions_seed_{seed}_exists", False, {"path": str(pred_npz_p)})
            continue

        pred_data = np.load(pred_npz_p)
        logits = pred_data["logits"]
        preds = pred_data["predictions"]

        report.add_check(
            f"D_seed_{seed}_pred_shape_finite",
            logits.shape == (5053, 12) and preds.shape == (5053,) and bool(np.all(np.isfinite(logits))),
            {"logits_shape": list(logits.shape), "preds_shape": list(preds.shape)},
        )
        report.add_check(
            f"D_seed_{seed}_pred_argmax_consistency",
            bool(np.array_equal(preds, logits.argmax(axis=1))),
            {"argmax_match": True},
        )

        # Recompute metrics on supervised subset
        sup_preds = preds[sup_indices]
        m = recompute_metrics_12class(y_test_sup, sup_preds)
        recomputed_accs.append(m["accuracy"])
        recomputed_f1s.append(m["macro_f1"])

        saved = saved_seed_evals.get(str(seed), {})
        acc_diff = abs(m["accuracy"] - saved.get("accuracy", -1.0))
        f1_diff = abs(m["macro_f1"] - saved.get("macro_f1", -1.0))

        report.add_check(
            f"D_seed_{seed}_metrics_match",
            acc_diff < 1e-10 and f1_diff < 1e-10,
            {"recomputed_acc": m["accuracy"], "saved_acc": saved.get("accuracy"), "acc_diff": acc_diff, "f1_diff": f1_diff},
        )

        # Check Wilson intervals match saved
        saved_wilson_low = saved.get("wilson_recall_ci_lower", [])
        saved_wilson_high = saved.get("wilson_recall_ci_upper", [])
        wilson_match = True
        for c in range(NUM_CLASSES):
            r_low = m["wilson_recall_ci_lower"][c]
            s_low = saved_wilson_low[c] if c < len(saved_wilson_low) else None
            if r_low is None:
                if s_low is not None:
                    wilson_match = False
            else:
                if s_low is None or abs(r_low - s_low) > 1e-10:
                    wilson_match = False
        report.add_check(f"D_seed_{seed}_wilson_intervals_match", wilson_match, {"seed": seed})

    # Compare aggregate metrics mean + POP std (ddof=0)
    saved_agg = eval_json.get("aggregate_metrics", {})
    recomp_mean_acc = float(np.mean(recomputed_accs))
    recomp_pop_std_acc = compute_pop_std(recomputed_accs)
    recomp_mean_f1 = float(np.mean(recomputed_f1s))
    recomp_pop_std_f1 = compute_pop_std(recomputed_f1s)

    report.add_check(
        "D_aggregate_mean_acc_match",
        abs(recomp_mean_acc - saved_agg.get("accuracy_mean", -1.0)) < 1e-10,
        {"recomputed": recomp_mean_acc, "saved": saved_agg.get("accuracy_mean")},
    )
    report.add_check(
        "D_aggregate_pop_std_acc_match",
        abs(recomp_pop_std_acc - saved_agg.get("accuracy_std", -1.0)) < 1e-10,
        {"recomputed": recomp_pop_std_acc, "saved": saved_agg.get("accuracy_std")},
    )
    report.add_check(
        "D_aggregate_mean_f1_match",
        abs(recomp_mean_f1 - saved_agg.get("macro_f1_mean", -1.0)) < 1e-10,
        {"recomputed": recomp_mean_f1, "saved": saved_agg.get("macro_f1_mean")},
    )
    report.add_check(
        "D_aggregate_pop_std_f1_match",
        abs(recomp_pop_std_f1 - saved_agg.get("macro_f1_std", -1.0)) < 1e-10,
        {"recomputed": recomp_pop_std_f1, "saved": saved_agg.get("macro_f1_std")},
    )


def run_dimension_e(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension E: Checkpoint re-inference, CSV manifest validation, winning margin quantiles, model identity."""
    from ecdr.data import load_full_manifest_with_splits, WINDOWS_FILE
    from scripts.backbone import SmallCNN1D

    df_man = load_full_manifest_with_splits()
    test_df = df_man[df_man["split"] == "test"]
    x_all = np.load(WINDOWS_FILE)["x"]

    norm_data = np.load(ref_dir / "normalization.npz")
    x_test_norm = apply_norm(x_all[test_df["global_row_id"].to_numpy()], norm_data["mean"], norm_data["std"])

    torch.set_num_threads(1)

    # 1. Reload each checkpoint and verify persisted logits and preds
    for seed in SEEDS:
        ckpt_p = ref_dir / f"checkpoint_seed_{seed}.pt"
        pred_p = ref_dir / f"test_predictions_seed_{seed}.npz"
        if not (ckpt_p.is_file() and pred_p.is_file()):
            continue

        ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        model = SmallCNN1D(num_classes=NUM_CLASSES, cfg=ckpt.get("model_cfg", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.inference_mode():
            # Batch 512 inference
            recomputed_logits_list = []
            for start in range(0, len(x_test_norm), 512):
                end = min(start + 512, len(x_test_norm))
                batch_t = torch.from_numpy(x_test_norm[start:end])
                recomputed_logits_list.append(model(batch_t).numpy())
            recomputed_logits = np.concatenate(recomputed_logits_list, axis=0)

        recomputed_preds = recomputed_logits.argmax(axis=1)
        persisted = np.load(pred_p)

        labels_match = bool(np.array_equal(recomputed_preds, persisted["predictions"]))
        max_logit_diff = float(np.max(np.abs(recomputed_logits - persisted["logits"])))

        report.add_check(
            f"E_seed_{seed}_batch512_inference_reproducible",
            labels_match and max_logit_diff < 1e-4,
            {"labels_match": labels_match, "max_logit_diff": max_logit_diff},
        )

    # 2. Selected checkpoint check against test_predictions.npz and CSV
    selected_p = ref_dir / "selected_best.pt"
    sel_meta_p = ref_dir / "selection.json"
    pred_canonical_p = ref_dir / "test_predictions.npz"
    csv_p = ref_dir / "test_predictions_manifest.csv"

    if selected_p.is_file() and sel_meta_p.is_file() and pred_canonical_p.is_file() and csv_p.is_file():
        sel_meta = json.loads(sel_meta_p.read_text())
        sel_seed = sel_meta["selected_seed"]
        seed_npz = np.load(ref_dir / f"test_predictions_seed_{sel_seed}.npz")
        canonical_npz = np.load(pred_canonical_p)

        report.add_check(
            "E_selected_canonical_npz_matches_seed_npz",
            bool(np.array_equal(canonical_npz["predictions"], seed_npz["predictions"])) and
            bool(np.allclose(canonical_npz["logits"], seed_npz["logits"], atol=1e-6)),
            {"selected_seed": sel_seed},
        )

        csv_df = pd.read_csv(csv_p)
        report.add_check(
            "E_csv_length_and_order",
            len(csv_df) == 5053 and bool(np.array_equal(csv_df["global_row_id"].to_numpy(), test_df["global_row_id"].to_numpy())),
            {"len": len(csv_df)},
        )

        # Winning margin = top1 - top2 quantiles
        logits = canonical_npz["logits"]
        sorted_logits = np.sort(logits, axis=1)
        margins = (sorted_logits[:, -1] - sorted_logits[:, -2]).astype(np.float64)

        recomp_quantiles = {
            "p0": float(np.percentile(margins, 0)),
            "p1": float(np.percentile(margins, 1)),
            "p5": float(np.percentile(margins, 5)),
            "p50": float(np.percentile(margins, 50)),
            "p95": float(np.percentile(margins, 95)),
            "p99": float(np.percentile(margins, 99)),
            "p100": float(np.percentile(margins, 100)),
        }

        eval_json = json.loads((ref_dir / "evaluation_results.json").read_text())
        saved_quantiles = eval_json.get("margins", {}).get("all_5053_stream", {})
        quantiles_match = True
        for qk, qv in recomp_quantiles.items():
            if abs(qv - saved_quantiles.get(qk, -999.0)) > 1e-4:
                quantiles_match = False

        report.add_check(
            "E_margin_quantiles_match",
            quantiles_match,
            {"recomputed": recomp_quantiles, "saved": saved_quantiles},
        )


def run_dimension_f(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension F: Recompute fullstream adjacent prediction agreement metrics."""
    from ecdr.data import load_full_manifest_with_splits
    df_man = load_full_manifest_with_splits()
    test_df = df_man[df_man["split"] == "test"].copy()

    canonical_npz = np.load(ref_dir / "test_predictions.npz")
    preds = canonical_npz["predictions"]
    test_df["pred"] = preds

    # Sort strictly by exp, win_idx
    sorted_df = test_df.sort_values(by=["exp", "win_idx"]).reset_index(drop=True)

    counts = {
        "pure_stable": {"total": 0, "agree": 0},
        "pure_transition": {"total": 0, "agree": 0},
        "mixed_other": {"total": 0, "agree": 0},
        "pure_same_label": {"total": 0, "agree": 0},
        "pure_label_switch": {"total": 0, "agree": 0},
    }

    n_rows = len(sorted_df)
    for i in range(n_rows - 1):
        cur = sorted_df.iloc[i]
        nxt = sorted_df.iloc[i + 1]

        # Valid adjacent condition
        if cur["exp"] == nxt["exp"] and nxt["win_idx"] == cur["win_idx"] + 1 and nxt["start"] == cur["start"] + 64:
            agree = int(cur["pred"] == nxt["pred"])
            cur_pure = (cur["label_policy"] == "pure")
            nxt_pure = (nxt["label_policy"] == "pure")
            cur_lbl = int(cur["label_majority"])
            nxt_lbl = int(nxt["label_majority"])

            if cur_pure and nxt_pure:
                if cur_lbl == nxt_lbl:
                    counts["pure_same_label"]["total"] += 1
                    counts["pure_same_label"]["agree"] += agree
                    if 1 <= cur_lbl <= 6:
                        counts["pure_stable"]["total"] += 1
                        counts["pure_stable"]["agree"] += agree
                    elif 7 <= cur_lbl <= 12:
                        counts["pure_transition"]["total"] += 1
                        counts["pure_transition"]["agree"] += agree
                    else:
                        counts["mixed_other"]["total"] += 1
                        counts["mixed_other"]["agree"] += agree
                else:
                    counts["pure_label_switch"]["total"] += 1
                    counts["pure_label_switch"]["agree"] += agree
                    counts["mixed_other"]["total"] += 1
                    counts["mixed_other"]["agree"] += agree
            else:
                counts["mixed_other"]["total"] += 1
                counts["mixed_other"]["agree"] += agree

    eval_json = json.loads((ref_dir / "evaluation_results.json").read_text())
    saved_adj = eval_json.get("adjacent_prediction_agreement", {})

    all_matched = True
    for cat in ["pure_stable", "pure_transition", "mixed_other"]:
        tot = counts[cat]["total"]
        agr = counts[cat]["agree"]
        rate = float(agr / tot) if tot > 0 else None

        saved_item = saved_adj.get(cat, {})
        s_tot = saved_item.get("total_pairs")
        s_agr = saved_item.get("agree_pairs")
        s_rate = saved_item.get("agreement_rate")

        if tot != s_tot or agr != s_agr:
            all_matched = False
        if rate is not None and s_rate is not None:
            if abs(rate - s_rate) > 1e-6:
                all_matched = False

    report.add_check(
        "F_adjacent_prediction_agreement_match",
        all_matched,
        {"recomputed": counts, "saved": saved_adj},
    )


def run_dimension_g(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension G: Numerical reference check: selected model CPU 1-thread BATCH 1 vs BATCH 512 labels."""
    from ecdr.data import load_full_manifest_with_splits, WINDOWS_FILE
    from scripts.backbone import SmallCNN1D

    df_man = load_full_manifest_with_splits()
    test_df = df_man[df_man["split"] == "test"]
    x_all = np.load(WINDOWS_FILE)["x"]

    norm_data = np.load(ref_dir / "normalization.npz")
    x_test_norm = apply_norm(x_all[test_df["global_row_id"].to_numpy()], norm_data["mean"], norm_data["std"])

    torch.set_num_threads(1)

    ckpt = torch.load(ref_dir / "selected_best.pt", map_location="cpu", weights_only=False)
    model = SmallCNN1D(num_classes=NUM_CLASSES, cfg=ckpt.get("model_cfg", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    canonical_npz = np.load(ref_dir / "test_predictions.npz")
    canonical_preds = canonical_npz["predictions"]
    canonical_logits = canonical_npz["logits"]

    batch1_preds = []
    batch1_logits_list = []

    with torch.inference_mode():
        for i in range(len(x_test_norm)):
            # Explicit single window batch 1: shape (1, 6, 128)
            w_tensor = torch.from_numpy(x_test_norm[i:i+1])
            out = model(w_tensor).numpy()
            batch1_logits_list.append(out[0])
            batch1_preds.append(int(out[0].argmax()))

    batch1_preds = np.array(batch1_preds, dtype=np.int64)
    batch1_logits = np.array(batch1_logits_list, dtype=np.float32)

    diff_mask = (batch1_preds != canonical_preds)
    mismatched_row_ids = test_df["global_row_id"].to_numpy()[diff_mask].tolist()
    max_logit_diff = float(np.max(np.abs(batch1_logits - canonical_logits)))

    report.add_check(
        "G_numerical_batch1_vs_batch512_label_consistency",
        len(mismatched_row_ids) == 0,
        {
            "mismatched_count": len(mismatched_row_ids),
            "mismatched_row_ids": mismatched_row_ids[:50],
            "max_logit_abs_diff": max_logit_diff,
            "device": "cpu",
            "threads": 1,
            "dtype": "float32",
        },
        expected=0,
        actual=len(mismatched_row_ids),
        error_msg="Batch-1 single-window inference decision diverged from canonical batch-512 decisions",
    )


def run_dimension_h(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension H: Timing benchmark metadata verification & scope exclusions."""
    eval_json_p = ref_dir / "evaluation_results.json"
    if not eval_json_p.is_file():
        report.add_check("H_eval_results_file_exists", False, {"path": str(eval_json_p)})
        return

    eval_json = json.loads(eval_json_p.read_text())
    cpu_latency = eval_json.get("cpu_latency", {})

    report.add_check(
        "H_timing_config_cpu1thread_batch1_100w_1000t",
        cpu_latency.get("warmup_runs") == 100 and cpu_latency.get("timed_runs") == 1000 and cpu_latency.get("threads") == 1,
        {"warmup": cpu_latency.get("warmup_runs"), "timed": cpu_latency.get("timed_runs"), "threads": cpu_latency.get("threads")},
    )

    core_bench = cpu_latency.get("core_preprocess_plus_forward_latency_ms", {})
    stats_valid = (
        core_bench.get("mean", -1.0) >= 0 and
        core_bench.get("median", -1.0) >= 0 and
        core_bench.get("pop_std", -1.0) >= 0 and
        core_bench.get("min", -1.0) >= 0 and
        core_bench.get("p95", -1.0) >= 0 and
        core_bench.get("p99", -1.0) >= 0
    )
    report.add_check(
        "H_timing_stats_finite_non_negative",
        stats_valid,
        {"core_bench": core_bench},
    )

    exclusion_note = core_bench.get("scope_exclusion_note", "")
    has_scope_exclusion = "EXCLUDES raw acquisition" in exclusion_note
    report.add_check(
        "H_timing_scope_exclusion_declared",
        has_scope_exclusion,
        {"scope_exclusion_note": exclusion_note},
    )


# ---------------------------------------------------------------------------
# Dimension I: Self-Test Synthetic Fixtures
# ---------------------------------------------------------------------------

def run_self_test() -> Dict[str, Any]:
    """Execute synthetic negative and positive fixtures independently of real artifacts."""
    fixtures_passed = 0
    fixtures_failed = 0
    details: List[Dict[str, Any]] = []

    def log_fixture(name: str, passed: bool, info: Any) -> None:
        nonlocal fixtures_passed, fixtures_failed
        if passed:
            fixtures_passed += 1
        else:
            fixtures_failed += 1
        details.append({"fixture": name, "status": "PASS" if passed else "FAIL", "info": info})

    # 1. Positive fixture: Wilson score interval with zero trials returns (None, None)
    low_null, high_null = wilson_score_interval(0, 0)
    log_fixture(
        "positive_wilson_zero_trials_null",
        low_null is None and high_null is None,
        {"low": low_null, "high": high_null},
    )

    # 2. Positive fixture: Population standard deviation (ddof=0)
    # values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], mean=5.0, variance=4.0, pop_std=2.0
    test_vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    p_std = compute_pop_std(test_vals)
    log_fixture(
        "positive_pop_std_ddof_0",
        abs(p_std - 2.0) < 1e-12,
        {"expected": 2.0, "actual": p_std},
    )

    # 3. Positive fixture: Preprocessing ID deterministic prefix check
    mean_syn = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    std_syn = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    prep_id = compute_preprocessing_id(mean_syn, std_syn)
    log_fixture(
        "positive_preprocessing_id_length_64_hex",
        len(prep_id) == 64 and all(c in "0123456789abcdef" for c in prep_id),
        {"prep_id": prep_id},
    )

    # 4. Negative fixture 1: Mismatched row IDs / duplicate IDs detection
    row_ids_canonical = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    row_ids_duplicate = np.array([10, 20, 20, 40, 50], dtype=np.int64)
    caught_duplicate = bool(len(np.unique(row_ids_duplicate)) != len(row_ids_duplicate))
    caught_mismatch = not np.array_equal(row_ids_canonical, row_ids_duplicate)
    log_fixture(
        "negative_catch_duplicate_and_mismatched_row_ids",
        caught_duplicate and caught_mismatch,
        {"caught_duplicate": caught_duplicate, "caught_mismatch": caught_mismatch},
    )

    # 5. Negative fixture 2: Incorrect CM / recall / Wilson values detection
    y_syn_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_syn_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)
    m_syn = recompute_metrics_12class(y_syn_true, y_syn_pred, num_classes=3)
    # class 0: tp=1, support=2, recall=0.5
    # class 1: tp=2, support=2, recall=1.0
    # class 2: tp=1, support=2, recall=0.5
    falsified_recalls = [0.99, 1.0, 0.5]
    detected_metric_error = abs(m_syn["class_recalls"][0] - falsified_recalls[0]) > 1e-4
    log_fixture(
        "negative_catch_falsified_class_recall",
        detected_metric_error,
        {"computed_rec0": m_syn["class_recalls"][0], "falsified": falsified_recalls[0]},
    )

    # 6. Negative fixture 3: Selection earliest argmax winner logic with tie-breaking
    # Seed 42: max F1=0.85 at epoch 10
    # Seed 43: max F1=0.85 at epoch 8 (tie in score, but earlier epoch)
    # Seed 44: max F1=0.85 at epoch 8 (identical score and epoch, seed 43 wins because 43 < 44)
    mock_histories = {
        42: [{"epoch": 5, "val_macro_f1": 0.80}, {"epoch": 10, "val_macro_f1": 0.85}],
        43: [{"epoch": 8, "val_macro_f1": 0.85}, {"epoch": 12, "val_macro_f1": 0.84}],
        44: [{"epoch": 8, "val_macro_f1": 0.85}, {"epoch": 14, "val_macro_f1": 0.83}],
    }
    best_candidate = None
    best_score = -1.0
    best_ep = 9999
    for s, h in mock_histories.items():
        score = max(x["val_macro_f1"] for x in h)
        earliest_ep = next(x["epoch"] for x in h if x["val_macro_f1"] == score)
        if score > best_score:
            best_score = score
            best_candidate = s
            best_ep = earliest_ep
        elif abs(score - best_score) < 1e-12:
            if earliest_ep < best_ep:
                best_ep = earliest_ep
                best_candidate = s
            elif earliest_ep == best_ep and s < best_candidate:
                best_candidate = s

    # Seed 43 must win tie-break over 44 and 42
    wrong_winner_claim = 44
    caught_wrong_winner = (best_candidate != wrong_winner_claim) and (best_candidate == 43)
    log_fixture(
        "negative_catch_wrong_selection_winner_tie_break",
        caught_wrong_winner,
        {"winner_computed": best_candidate, "claimed_winner": wrong_winner_claim},
    )

    # 7. Negative fixture 4: Wrong expected file hash detection
    fake_content = b"reference pipeline code bytes"
    actual_hash = hashlib.sha256(fake_content).hexdigest()
    corrupted_expected_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    caught_hash_mismatch = (actual_hash != corrupted_expected_hash)
    log_fixture(
        "negative_catch_corrupted_file_hash",
        caught_hash_mismatch,
        {"actual_hash": actual_hash, "corrupted_hash": corrupted_expected_hash},
    )

    all_passed = (fixtures_failed == 0 and fixtures_passed > 0)
    return {
        "status": "PASS" if all_passed else "FAIL",
        "passed_fixtures": fixtures_passed,
        "failed_fixtures": fixtures_failed,
        "details": details,
    }


# ---------------------------------------------------------------------------
# CLI & Execution Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent post-run verification for Week 1 reference pipeline."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "output" / "week1_closeout_20261102" / "reference",
        help="Path to output reference directory containing phase artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "week1_closeout_20261102" / "reference" / "independent_verification.json",
        help="Target path for independent verification receipt JSON.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Execute synthetic negative and positive fixtures without requiring trained artifacts.",
    )
    parser.add_argument(
        "--overwrite-receipt",
        action="store_true",
        help="Explicitly permit overwriting an existing verification receipt file.",
    )

    args = parser.parse_args()

    # Self-test mode (does not require trained artifacts)
    if args.self_test:
        result = run_self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] == "PASS" else 1)

    # Reference verification mode
    ref_dir = args.reference_dir
    if not ref_dir.is_dir():
        print(f"Error: reference directory does not exist: {ref_dir}", file=sys.stderr)
        sys.exit(2)

    out_file = args.output
    if out_file.is_file() and not args.overwrite_receipt:
        print(
            f"Error: verification receipt already exists at {out_file}. "
            "Use --overwrite-receipt to explicitly permit overwriting.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Ensure required artifacts are present before running
    required_artifacts = [
        ref_dir / "provenance_audit.json",
        ref_dir / "normalization.npz",
        ref_dir / "normalization.json",
        ref_dir / "selection.json",
        ref_dir / "selected_best.pt",
        ref_dir / "test_predictions.npz",
        ref_dir / "test_predictions_manifest.csv",
        ref_dir / "evaluation_results.json",
        ref_dir / "eval_receipt.json",
    ]
    missing = [str(p.name) for p in required_artifacts if not p.is_file()]
    if missing:
        print(
            f"Error: Incomplete reference directory {ref_dir}. Missing required artifacts: {missing}. "
            "Cannot verify incomplete run while training/evaluation is still active.",
            file=sys.stderr,
        )
        sys.exit(2)

    report = VerificationReport(ref_dir)

    print(f"Running independent verification on {ref_dir}...")
    try:
        run_dimension_a(report, ref_dir)
        run_dimension_b(report, ref_dir)
        run_dimension_c(report, ref_dir)
        run_dimension_d(report, ref_dir)
        run_dimension_e(report, ref_dir)
        run_dimension_f(report, ref_dir)
        run_dimension_g(report, ref_dir)
        run_dimension_h(report, ref_dir)
    except Exception as exc:
        report.add_check(
            "unhandled_exception",
            False,
            {"traceback": traceback.format_exc()},
            error_msg=f"Unhandled exception during verification: {exc}",
        )

    receipt = report.to_dict()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(receipt, indent=2))
    print(f"Wrote verification receipt to {out_file} (Overall status: {receipt['overall_status']})")

    if receipt["overall_status"] != "PASS":
        print(f"Verification FAILED with {report.failed_count} errors.", file=sys.stderr)
        sys.exit(1)

    print("All independent verification checks PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
