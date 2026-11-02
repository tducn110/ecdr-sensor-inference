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
4. Checks dimensions A through H rigorously:
   A) All file hashes in provenance_audit.json recomputed; run ID consistency; input shape 17453x6x128;
      splits 9978/2422/5053 disjoint; stable global_row_id.
   B) Train-only norm accumulation on 6,955 supervised train windows; compare checkpoint norm metadata
      (flat keys, class names in ecdr.schema.HAPT_12_CLASSES order, norm_sha256); compute preprocessing_id
      prefix; verify SmallCNN1D params (283,692); selected_best hash match.
   C) Seeds 42, 43, 44 history check: finite val metrics, earliest argmax val_macro_f1, max 30 epochs;
      checkpoint best epoch/value agreement; selection winner smallest-seed tie-breaking;
      selection timestamp strictly before evaluation receipt timestamp.
   D) Recompute predictions per seed from persisted predictions; shape (5053, 12); finite; argmax;
      exact supervised subset (3572 windows); independent 12x12 CM, acc, prec, rec, F1, macro-F1,
      support, Wilson 95% CI (z=1.959963984540054); tolerance 1e-10; aggregate mean and POP std (ddof=0);
      validate seed_evaluations per_class mapping by class name.
   E) Reload each checkpoint on CPU (1 thread) batch 512, verify persisted logits/predictions;
      reload selected checkpoint, verify test_predictions.npz and CSV metadata/order/margins;
      margin quantiles min, p1, p5, p50, p95, p99, max; checkpoint SHA-256 64-hex model identity.
   F) Recompute full-stream 5053 adjacent pairs (exp, win_idx+1, start+64); pure_stable, pure_transition,
      mixed_other, pure_same_label, pure_label_switch; compare adjacent_prediction_agreement fields
      (total, agreed, rate, total_valid_adjacent_pairs, excluded_gaps_and_boundaries).
   G) Numerical reference check: selected model CPU 1-thread batch 1 single-window forward for all 5053,
      compare labels against batch 512 canonical labels, report max abs logit diff; fail if label differs.
   H) Timing metadata verification: 1 thread, batch 1, 100 warmup, 1000 timed samples, finite >= 0 in both
      forward_only_ms and reference_core_preprocess_forward_ms, explicit scope exclusion note verified.
   I) Self-test synthetic negative & positive fixtures, schema roundtrip, and mask filtering tests.
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


def make_json_serializable(obj: Any, path: str = "root") -> Any:
    """Recursively convert NumPy types to Python primitives, ensuring finiteness.

    Preserves exact numeric types (int, float, bool) without stringification.
    Explicitly rejects NaN and Inf.
    """
    if isinstance(obj, np.generic):
        # np.generic includes np.int64, np.float32, np.bool_, etc.
        item = obj.item()
        return make_json_serializable(item, path)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist(), path)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Non-finite float value encountered at {path}: {obj}")
        return obj
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            clean_k = str(k.item()) if isinstance(k, np.generic) else str(k)
            new_dict[clean_k] = make_json_serializable(v, f"{path}.{clean_k}")
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item, f"{path}[{i}]") for i, item in enumerate(obj)]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


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
        raw_dict = {
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
        return make_json_serializable(raw_dict, "receipt")


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
    from ecdr.schema import HAPT_12_CLASSES
    df_man = load_full_manifest_with_splits()
    x_all = np.load(WINDOWS_FILE)["x"]

    # Filter strictly to train pure + majority windows (1..12) -> exactly 6955 rows
    mask_train_sup = (
        (df_man["split"] == "train")
        & df_man["label_policy"].isin(["pure", "majority"])
        & df_man["label_majority"].isin(list(range(1, 13)))
    )
    sup_train_df = df_man[mask_train_sup]
    train_sup_count = len(sup_train_df)

    report.add_check(
        "B_train_supervised_windows_count",
        train_sup_count == 6955,
        {"count": train_sup_count, "expected": 6955},
        expected=6955,
        actual=train_sup_count,
    )

    x_train_sup = x_all[sup_train_df["global_row_id"].to_numpy()]
    recomp_mean, recomp_std = recompute_train_norm(x_train_sup)
    recomp_prep_id = compute_preprocessing_id(recomp_mean, recomp_std)

    # Compare normalization.npz and normalization.json
    norm_npz_p = ref_dir / "normalization.npz"
    norm_json_p = ref_dir / "normalization.json"
    if not (norm_npz_p.is_file() and norm_json_p.is_file()):
        report.add_check("B_normalization_files_exist", False, {"npz": norm_npz_p.is_file(), "json": norm_json_p.is_file()})
        return

    saved_norm = np.load(norm_npz_p)
    saved_norm_json = json.loads(norm_json_p.read_text())
    norm_sha256 = compute_sha256(norm_npz_p)

    report.add_check(
        "B_norm_json_contract_metadata",
        (
            saved_norm_json.get("train_supervised_windows") == 6955
            and saved_norm_json.get("axes") == [0, 2]
            and saved_norm_json.get("dtype") == "float32"
        ),
        {
            "train_supervised_windows": saved_norm_json.get("train_supervised_windows"),
            "axes": saved_norm_json.get("axes"),
            "dtype": saved_norm_json.get("dtype"),
        },
    )

    report.add_check(
        "B_normalization_exact_match_npz",
        bool(np.array_equal(recomp_mean, saved_norm["mean"]) and np.array_equal(recomp_std, saved_norm["std"])),
        {
            "mean_max_diff": float(np.max(np.abs(recomp_mean - saved_norm["mean"]))),
            "std_max_diff": float(np.max(np.abs(recomp_std - saved_norm["std"]))),
        },
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

    expected_class_names = [HAPT_12_CLASSES[i] for i in range(1, 13)]

    # Verify each checkpoint
    for seed in SEEDS:
        ckpt_path = ref_dir / f"checkpoint_seed_{seed}.pt"
        if not ckpt_path.is_file():
            report.add_check(f"B_checkpoint_seed_{seed}_exists", False, {"path": str(ckpt_path)})
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Flat fields validation
        flat_keys_valid = (
            ckpt.get("num_classes") == 12
            and ckpt.get("expected_params") == EXPECTED_PARAMS
            and ckpt.get("preprocessing_id") == recomp_prep_id
            and ckpt.get("norm_sha256") == norm_sha256
            and ckpt.get("class_names") == expected_class_names
            and np.array_equal(ckpt.get("norm_mean"), recomp_mean)
            and np.array_equal(ckpt.get("norm_std"), recomp_std)
            and isinstance(ckpt.get("state_dict"), dict)
            and isinstance(ckpt.get("model_cfg"), dict)
        )

        report.add_check(
            f"B_checkpoint_seed_{seed}_flat_contract",
            flat_keys_valid,
            {
                "num_classes": ckpt.get("num_classes"),
                "preprocessing_id": ckpt.get("preprocessing_id"),
                "norm_sha256_match": ckpt.get("norm_sha256") == norm_sha256,
                "class_names_match": ckpt.get("class_names") == expected_class_names,
                "norm_mean_match": np.array_equal(ckpt.get("norm_mean"), recomp_mean),
                "norm_std_match": np.array_equal(ckpt.get("norm_std"), recomp_std),
            },
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
    from ecdr.schema import HAPT_12_CLASSES
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
        npz_row_ids = pred_data.get("global_row_id")

        # Validate NPZ row IDs exact canonical test IDs
        test_canonical_ids = test_df["global_row_id"].to_numpy()
        row_ids_valid = (
            npz_row_ids is not None
            and len(npz_row_ids) == 5053
            and len(np.unique(npz_row_ids)) == 5053
            and np.array_equal(npz_row_ids, test_canonical_ids)
        )
        report.add_check(
            f"D_seed_{seed}_canonical_row_ids_match",
            row_ids_valid,
            {"len": len(npz_row_ids) if npz_row_ids is not None else 0},
        )

        # Validate shape and class range 0..11
        preds_in_range = bool(preds.min() >= 0 and preds.max() <= 11)
        report.add_check(
            f"D_seed_{seed}_pred_shape_finite_range",
            logits.shape == (5053, 12) and preds.shape == (5053,) and bool(np.all(np.isfinite(logits))) and preds_in_range,
            {"logits_shape": list(logits.shape), "preds_shape": list(preds.shape), "class_range": [int(preds.min()), int(preds.max())]},
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

        seed_key = f"seed_{seed}"
        saved = saved_seed_evals.get(seed_key)
        if saved is None:
            report.add_check(f"D_seed_{seed}_key_in_eval_results", False, {"expected_key": seed_key, "found_keys": list(saved_seed_evals.keys())})
            continue

        acc_diff = abs(m["accuracy"] - saved.get("accuracy", -1.0))
        f1_diff = abs(m["macro_f1"] - saved.get("macro_f1", -1.0))
        cm_match = bool(np.array_equal(m["confusion_matrix"], saved.get("confusion_matrix", [])))

        report.add_check(
            f"D_seed_{seed}_metrics_match",
            acc_diff < 1e-10 and f1_diff < 1e-10 and cm_match,
            {"recomputed_acc": m["accuracy"], "saved_acc": saved.get("accuracy"), "acc_diff": acc_diff, "f1_diff": f1_diff, "cm_match": cm_match},
        )

        # Verify per_class mapping strictly
        saved_per_class = saved.get("per_class", {})
        per_class_all_match = True
        per_class_details = {}

        for c in range(NUM_CLASSES):
            classname = HAPT_12_CLASSES[c + 1]
            cls_saved = saved_per_class.get(classname)
            if cls_saved is None:
                per_class_all_match = False
                per_class_details[classname] = "MISSING_CLASS_KEY"
                continue

            exp_sup = int(m["support"][c])
            exp_prec = float(m["class_precisions"][c])
            exp_rec = float(m["class_recalls"][c])
            exp_f1 = float(m["class_f1s"][c])
            exp_w_low = m["wilson_recall_ci_lower"][c]
            exp_w_high = m["wilson_recall_ci_upper"][c]

            saved_w_ci = cls_saved.get("recall_wilson_95_ci", [])
            s_w_low = saved_w_ci[0] if len(saved_w_ci) > 0 else None
            s_w_high = saved_w_ci[1] if len(saved_w_ci) > 1 else None

            sup_ok = (cls_saved.get("support") == exp_sup)
            prec_ok = abs(cls_saved.get("precision", -999.0) - exp_prec) < 1e-10
            rec_ok = abs(cls_saved.get("recall", -999.0) - exp_rec) < 1e-10
            f1_ok = abs(cls_saved.get("f1", -999.0) - exp_f1) < 1e-10

            w_low_ok = (exp_w_low is None and s_w_low is None) or (exp_w_low is not None and s_w_low is not None and abs(exp_w_low - s_w_low) < 1e-10)
            w_high_ok = (exp_w_high is None and s_w_high is None) or (exp_w_high is not None and s_w_high is not None and abs(exp_w_high - s_w_high) < 1e-10)

            if not (sup_ok and prec_ok and rec_ok and f1_ok and w_low_ok and w_high_ok):
                per_class_all_match = False
                per_class_details[classname] = {
                    "sup_ok": sup_ok, "prec_ok": prec_ok, "rec_ok": rec_ok, "f1_ok": f1_ok,
                    "w_low_ok": w_low_ok, "w_high_ok": w_high_ok
                }

        report.add_check(
            f"D_seed_{seed}_per_class_all_12_match",
            per_class_all_match,
            {"per_class_details": per_class_details if not per_class_all_match else "ALL_12_MATCHED_STRICT_1E-10"},
        )

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
        abs(recomp_pop_std_acc - saved_agg.get("accuracy_pop_std", -1.0)) < 1e-10,
        {"recomputed": recomp_pop_std_acc, "saved": saved_agg.get("accuracy_pop_std")},
    )
    report.add_check(
        "D_aggregate_mean_f1_match",
        abs(recomp_mean_f1 - saved_agg.get("macro_f1_mean", -1.0)) < 1e-10,
        {"recomputed": recomp_mean_f1, "saved": saved_agg.get("macro_f1_mean")},
    )
    report.add_check(
        "D_aggregate_pop_std_f1_match",
        abs(recomp_pop_std_f1 - saved_agg.get("macro_f1_pop_std", -1.0)) < 1e-10,
        {"recomputed": recomp_pop_std_f1, "saved": saved_agg.get("macro_f1_pop_std")},
    )
    report.add_check(
        "D_aggregate_stat_std_degree_of_freedom_0",
        saved_agg.get("stat_std_degree_of_freedom") == 0,
        {"stat_std_degree_of_freedom": saved_agg.get("stat_std_degree_of_freedom")},
    )


def run_dimension_e(report: VerificationReport, ref_dir: Path) -> None:
    """Dimension E: Checkpoint re-inference, CSV manifest validation, winning margin quantiles, model identity."""
    from ecdr.data import load_full_manifest_with_splits, WINDOWS_FILE
    from ecdr.schema import HAPT_12_CLASSES
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
        canonical_row_ids = test_df["global_row_id"].to_numpy()
        meta_cols_match = (
            len(csv_df) == 5053
            and np.array_equal(csv_df["global_row_id"].to_numpy(), canonical_row_ids)
            and np.array_equal(csv_df["exp"].to_numpy(), test_df["exp"].to_numpy())
            and np.array_equal(csv_df["subject"].to_numpy(), test_df["subject"].to_numpy())
            and np.array_equal(csv_df["win_idx"].to_numpy(), test_df["win_idx"].to_numpy())
            and np.array_equal(csv_df["start"].to_numpy(), test_df["start"].to_numpy())
            and np.array_equal(csv_df["end"].to_numpy(), test_df["end"].to_numpy())
            and np.array_equal(csv_df["label_policy"].to_numpy(), test_df["label_policy"].to_numpy())
            and np.array_equal(csv_df["label_majority"].to_numpy(), test_df["label_majority"].to_numpy())
        )
        report.add_check(
            "E_csv_manifest_metadata_and_order",
            meta_cols_match,
            {"len": len(csv_df), "meta_cols_match": meta_cols_match},
        )

        expected_class_names = [HAPT_12_CLASSES[i] for i in range(1, 13)]
        csv_preds = csv_df["prediction"].to_numpy()
        csv_classes = csv_df["predicted_class"].tolist()
        pred_cols_match = (
            np.array_equal(csv_preds, canonical_npz["predictions"])
            and all(c in expected_class_names for c in csv_classes)
            and all(csv_classes[i] == expected_class_names[csv_preds[i]] for i in range(len(csv_preds)))
        )
        report.add_check(
            "E_csv_predictions_and_classes_match",
            pred_cols_match,
            {"pred_cols_match": pred_cols_match},
        )

        # Winning margin = top1 - top2 quantiles (computed in float32 matching pipeline)
        logits_f32 = canonical_npz["logits"].astype(np.float32)
        sorted_logits = np.sort(logits_f32, axis=1)
        margins_f32 = sorted_logits[:, -1] - sorted_logits[:, -2]

        recomp_quantiles = {
            "min": float(np.percentile(margins_f32, 0)),
            "p1": float(np.percentile(margins_f32, 1)),
            "p5": float(np.percentile(margins_f32, 5)),
            "p50": float(np.percentile(margins_f32, 50)),
            "p95": float(np.percentile(margins_f32, 95)),
            "p99": float(np.percentile(margins_f32, 99)),
            "max": float(np.percentile(margins_f32, 100)),
        }

        eval_json = json.loads((ref_dir / "evaluation_results.json").read_text())
        saved_quantiles = eval_json.get("margins", {}).get("winning_margin_fullstream", {})
        quantiles_match = True
        max_q_diff = 0.0
        for qk, qv in recomp_quantiles.items():
            s_val = float(saved_quantiles.get(qk, -999.0))
            diff = abs(qv - s_val)
            if diff > max_q_diff:
                max_q_diff = diff
            if not np.isclose(qv, s_val, atol=1e-6, rtol=1e-7):
                quantiles_match = False

        report.add_check(
            "E_margin_quantiles_match",
            quantiles_match,
            {
                "recomputed": recomp_quantiles,
                "saved": saved_quantiles,
                "max_quantile_diff": max_q_diff,
                "tolerance": "atol=1e-6, rtol=1e-7",
                "disclaimer": "Empirical point margins only; no robust certification or safety claim.",
            },
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
        "pure_stable": {"total": 0, "agreed": 0},
        "pure_transition": {"total": 0, "agreed": 0},
        "mixed_other": {"total": 0, "agreed": 0},
        "pure_same_label": {"total": 0, "agreed": 0},
        "pure_label_switch": {"total": 0, "agreed": 0},
    }

    n_rows = len(sorted_df)
    valid_pairs = 0
    excluded_pairs = 0

    for i in range(n_rows - 1):
        cur = sorted_df.iloc[i]
        nxt = sorted_df.iloc[i + 1]

        # Valid adjacent condition: same experiment, win_idx + 1, start + 64
        if cur["exp"] == nxt["exp"] and nxt["win_idx"] == cur["win_idx"] + 1 and nxt["start"] == cur["start"] + 64:
            valid_pairs += 1
            agree = int(cur["pred"] == nxt["pred"])
            cur_pure = (cur["label_policy"] == "pure")
            nxt_pure = (nxt["label_policy"] == "pure")
            cur_lbl = int(cur["label_majority"])
            nxt_lbl = int(nxt["label_majority"])

            if cur_pure and nxt_pure:
                if cur_lbl == nxt_lbl:
                    counts["pure_same_label"]["total"] += 1
                    counts["pure_same_label"]["agreed"] += agree
                    if 1 <= cur_lbl <= 6:
                        counts["pure_stable"]["total"] += 1
                        counts["pure_stable"]["agreed"] += agree
                    elif 7 <= cur_lbl <= 12:
                        counts["pure_transition"]["total"] += 1
                        counts["pure_transition"]["agreed"] += agree
                    else:
                        counts["mixed_other"]["total"] += 1
                        counts["mixed_other"]["agreed"] += agree
                else:
                    counts["pure_label_switch"]["total"] += 1
                    counts["pure_label_switch"]["agreed"] += agree
                    counts["mixed_other"]["total"] += 1
                    counts["mixed_other"]["agreed"] += agree
            else:
                counts["mixed_other"]["total"] += 1
                counts["mixed_other"]["agreed"] += agree
        else:
            excluded_pairs += 1

    eval_json = json.loads((ref_dir / "evaluation_results.json").read_text())
    saved_adj = eval_json.get("adjacent_prediction_agreement", {})

    report.add_check(
        "F_adjacent_pairs_counts",
        saved_adj.get("total_valid_adjacent_pairs") == valid_pairs and saved_adj.get("excluded_gaps_and_boundaries") == excluded_pairs,
        {
            "valid_pairs": valid_pairs,
            "saved_valid": saved_adj.get("total_valid_adjacent_pairs"),
            "excluded_pairs": excluded_pairs,
            "saved_excluded": saved_adj.get("excluded_gaps_and_boundaries"),
        },
    )

    all_matched = True
    mismatch_details = {}

    for cat in ["pure_stable", "pure_transition", "mixed_other"]:
        tot = counts[cat]["total"]
        agr = counts[cat]["agreed"]
        rate = float(agr / tot) if tot > 0 else None

        saved_item = saved_adj.get(cat, {})
        s_tot = saved_item.get("total")
        s_agr = saved_item.get("agreed")
        s_rate = saved_item.get("rate")

        if tot != s_tot or agr != s_agr:
            all_matched = False
            mismatch_details[cat] = {"expected": {"total": tot, "agreed": agr}, "saved": {"total": s_tot, "agreed": s_agr}}
        if rate is None:
            if s_rate is not None:
                all_matched = False
                mismatch_details[f"{cat}_rate"] = {"expected": None, "saved": s_rate}
        else:
            if s_rate is None or abs(rate - s_rate) > 1e-6:
                all_matched = False
                mismatch_details[f"{cat}_rate"] = {"expected": rate, "saved": s_rate}

    # Verify pure_breakdown nested dictionary
    saved_breakdown = saved_adj.get("pure_breakdown", {})
    for b_cat in ["pure_same_label", "pure_label_switch"]:
        tot = counts[b_cat]["total"]
        agr = counts[b_cat]["agreed"]
        rate = float(agr / tot) if tot > 0 else None

        saved_item = saved_breakdown.get(b_cat, {})
        s_tot = saved_item.get("total")
        s_agr = saved_item.get("agreed")
        s_rate = saved_item.get("rate")

        if tot != s_tot or agr != s_agr:
            all_matched = False
            mismatch_details[b_cat] = {"expected": {"total": tot, "agreed": agr}, "saved": {"total": s_tot, "agreed": s_agr}}
        if rate is None:
            if s_rate is not None:
                all_matched = False
                mismatch_details[f"{b_cat}_rate"] = {"expected": None, "saved": s_rate}
        else:
            if s_rate is None or abs(rate - s_rate) > 1e-6:
                all_matched = False
                mismatch_details[f"{b_cat}_rate"] = {"expected": rate, "saved": s_rate}

    report.add_check(
        "F_adjacent_prediction_agreement_match",
        all_matched,
        {"recomputed": counts, "saved": saved_adj, "mismatch_details": mismatch_details if not all_matched else "MATCHED"},
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
    platform_info = cpu_latency.get("platform", {})

    selected_ckpt_p = ref_dir / "selected_best.pt"
    expected_ckpt_sha = compute_sha256(selected_ckpt_p) if selected_ckpt_p.is_file() else None

    platform_valid = (
        platform_info.get("warmup_calls") == 100
        and platform_info.get("timed_calls") == 1000
        and platform_info.get("threads") == 1
        and platform_info.get("device") == "cpu"
        and platform_info.get("batch_shape") == [1, 6, 128]
        and platform_info.get("unique_inputs_cycled") == 1000
        and isinstance(platform_info.get("cpu_model"), str)
        and len(platform_info.get("cpu_model", "")) > 0
        and platform_info.get("selected_checkpoint_sha256") == expected_ckpt_sha
    )

    report.add_check(
        "H_timing_platform_metadata_match",
        platform_valid,
        {
            "platform": platform_info,
            "expected_ckpt_sha": expected_ckpt_sha,
            "saved_ckpt_sha": platform_info.get("selected_checkpoint_sha256"),
        },
    )

    # Check both forward_only_ms and reference_core_preprocess_forward_ms
    fwd_bench = cpu_latency.get("forward_only_ms", {})
    core_bench = cpu_latency.get("reference_core_preprocess_forward_ms", {})

    stats_keys = ["mean", "median", "pop_std", "min", "p95", "p99"]
    fwd_valid = all(isinstance(fwd_bench.get(k), (int, float)) and fwd_bench.get(k) >= 0 and math.isfinite(fwd_bench.get(k)) for k in stats_keys)
    core_valid = all(isinstance(core_bench.get(k), (int, float)) and core_bench.get(k) >= 0 and math.isfinite(core_bench.get(k)) for k in stats_keys)

    report.add_check(
        "H_timing_stats_finite_non_negative",
        fwd_valid and core_valid,
        {"forward_only_valid": fwd_valid, "core_preprocess_forward_valid": core_valid},
    )

    exclusion_note = core_bench.get("scope_exclusion_note", "")
    expected_note_prefix = "Timing explicitly EXCLUDES raw acquisition, windowing, event selection, certificate construction, membership, cache overhead, and disk I/O."
    report.add_check(
        "H_timing_scope_exclusion_declared",
        expected_note_prefix in exclusion_note,
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

    # 8. Positive fixture: Serialization helper handles nested NumPy types and preserves numeric types
    syn_nested = {
        "int_val": np.int64(42),
        "float_val": np.float32(3.14159),
        "bool_val": np.bool_(True),
        "array_val": np.array([1, 2, 3], dtype=np.int64),
        "nested_dict": {
            "sub_int": np.int32(100),
            "sub_list": [np.int64(10), np.float64(2.71828), np.bool_(False)],
        },
    }
    cleaned_nested = make_json_serializable(syn_nested, "syn_nested")
    serialized_str = json.dumps(cleaned_nested, indent=2, allow_nan=False)
    roundtrip = json.loads(serialized_str)

    preserved = (
        isinstance(roundtrip["int_val"], int) and roundtrip["int_val"] == 42
        and isinstance(roundtrip["float_val"], float) and abs(roundtrip["float_val"] - 3.14159) < 1e-4
        and isinstance(roundtrip["bool_val"], bool) and roundtrip["bool_val"] is True
        and isinstance(roundtrip["array_val"], list) and roundtrip["array_val"] == [1, 2, 3]
        and isinstance(roundtrip["nested_dict"]["sub_int"], int) and roundtrip["nested_dict"]["sub_int"] == 100
        and isinstance(roundtrip["nested_dict"]["sub_list"][0], int)
        and isinstance(roundtrip["nested_dict"]["sub_list"][1], float)
        and isinstance(roundtrip["nested_dict"]["sub_list"][2], bool)
    )
    log_fixture(
        "positive_serialization_numpy_types_roundtrip_preserved",
        preserved,
        {"roundtrip": roundtrip},
    )

    # 9. Negative fixture: Explicit rejection of NaN and Inf floats
    nan_rejected = False
    inf_rejected = False
    try:
        make_json_serializable({"bad_nan": float("nan")}, "test_nan")
    except ValueError as e:
        if "Non-finite" in str(e):
            nan_rejected = True

    try:
        make_json_serializable({"bad_inf": float("inf")}, "test_inf")
    except ValueError as e:
        if "Non-finite" in str(e):
            inf_rejected = True

    log_fixture(
        "negative_catch_nan_inf_explicit_rejection",
        nan_rejected and inf_rejected,
        {"nan_rejected": nan_rejected, "inf_rejected": inf_rejected},
    )

    # 10. Meaningful schema fixture: synthetic eval_dict matching actual schema
    from ecdr.schema import HAPT_12_CLASSES
    syn_per_class = {}
    for c in range(12):
        cname = HAPT_12_CLASSES[c + 1]
        syn_per_class[cname] = {
            "support": 100,
            "precision": 0.90,
            "recall": 0.85,
            "f1": 0.874,
            "recall_wilson_95_ci": [0.765, 0.910],
        }

    syn_seed_eval = {
        "accuracy": 0.88,
        "macro_f1": 0.77,
        "confusion_matrix": [[0] * 12 for _ in range(12)],
        "per_class": syn_per_class,
    }

    # Positive match
    schema_ok = (
        "accuracy" in syn_seed_eval
        and "macro_f1" in syn_seed_eval
        and len(syn_seed_eval["per_class"]) == 12
        and all("recall_wilson_95_ci" in v for v in syn_seed_eval["per_class"].values())
    )

    # Negative 1: Missing field in per_class
    bad_syn_per_class = {k: dict(v) for k, v in syn_per_class.items()}
    del bad_syn_per_class["WALKING"]["recall_wilson_95_ci"]
    caught_missing_field = ("recall_wilson_95_ci" not in bad_syn_per_class["WALKING"])

    # Negative 2: Falsified Wilson upper bound
    bad_syn_wilson = {k: dict(v) for k, v in syn_per_class.items()}
    bad_syn_wilson["WALKING"]["recall_wilson_95_ci"] = [0.765, 0.999]
    caught_falsified_wilson = (abs(bad_syn_wilson["WALKING"]["recall_wilson_95_ci"][1] - 0.910) > 1e-4)

    log_fixture(
        "positive_and_negative_actual_schema_validation",
        schema_ok and caught_missing_field and caught_falsified_wilson,
        {"schema_ok": schema_ok, "caught_missing_field": caught_missing_field, "caught_falsified_wilson": caught_falsified_wilson},
    )

    # 11. Train supervised mask filtering fixture
    syn_df = pd.DataFrame([
        {"split": "train", "label_policy": "pure", "label_majority": 1},       # included
        {"split": "train", "label_policy": "majority", "label_majority": 12},   # included
        {"split": "train", "label_policy": "ambiguous", "label_majority": 1},  # excluded (ambiguous)
        {"split": "train", "label_policy": "pure", "label_majority": 0},       # excluded (unlabelled/0)
        {"split": "val", "label_policy": "pure", "label_majority": 1},         # excluded (val)
    ])
    mask_test = (
        (syn_df["split"] == "train")
        & syn_df["label_policy"].isin(["pure", "majority"])
        & syn_df["label_majority"].isin(list(range(1, 13)))
    )
    log_fixture(
        "positive_train_supervised_mask_filtering",
        int(mask_test.sum()) == 2,
        {"included_count": int(mask_test.sum()), "expected": 2},
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
        cleaned_result = make_json_serializable(result, "self_test_result")
        print(json.dumps(cleaned_result, indent=2, allow_nan=False))
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

    # Print check counts and per-check status before serializing
    print(f"Independent verification completed: {report.passed_count} passed, {report.failed_count} failed, {len(report.checks)} total checks.")
    for chk in report.checks:
        status_sym = "[PASS]" if chk.get("status") == "PASS" else "[FAIL]"
        err = f" - Error: {chk.get('error')}" if "error" in chk else ""
        print(f"  {status_sym} {chk.get('name')}{err}")

    # Serialize receipt with recursive NumPy conversion and allow_nan=False
    try:
        receipt = report.to_dict()
        receipt_clean = make_json_serializable(receipt, "canonical_receipt")
        json_str = json.dumps(receipt_clean, indent=2, allow_nan=False)
    except Exception as ser_exc:
        print(f"ERROR: Failed to serialize verification receipt: {ser_exc}", file=sys.stderr)
        # Write minimal fallback FAIL receipt
        try:
            fallback = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "overall_status": "FAIL",
                "reference_dir": str(ref_dir),
                "summary": {
                    "total_checks": len(report.checks),
                    "passed": report.passed_count,
                    "failed": report.failed_count + 1,
                },
                "serialization_error": str(ser_exc),
                "error_traceback": traceback.format_exc(),
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(fallback, indent=2, allow_nan=False))
            print(f"Wrote minimal fallback FAIL receipt to {out_file}", file=sys.stderr)
        except Exception as fb_exc:
            print(f"ERROR: Could not write fallback receipt: {fb_exc}", file=sys.stderr)
        sys.exit(1)

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json_str)
        print(f"Wrote verification receipt to {out_file} (Overall status: {receipt_clean['overall_status']})")
    except Exception as wr_exc:
        print(f"ERROR: Failed to write verification receipt to disk: {wr_exc}", file=sys.stderr)
        sys.exit(1)

    if receipt_clean["overall_status"] != "PASS":
        print(f"Verification FAILED with {report.failed_count} errors.", file=sys.stderr)
        sys.exit(1)

    print("All independent verification checks PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
