#!/usr/bin/env python3
"""scripts/week1_reference_pipeline.py — Week 1 Reference Pipeline.

Full-label K=12 SmallCNN1D reference implementation for HAPT.
Preserves existing data, builder conventions, and backbone.py / backbone.yaml bitwise.
Phases: audit, train, evaluate, verify, all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Enforce single thread determinism
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.data import (  # noqa: E402
    DATA_DIR,
    FULL_STREAM_DENOMINATORS,
    HAPT_12_CLASSES,
    MIN_SUPPORT_SUBJECTS_PER_SPLIT,
    MIN_SUPPORT_WINDOWS_PER_SPLIT,
    SPLITS,
    load_full_manifest_with_splits,
    load_raw_windows,
)
from scripts.backbone import SmallCNN1D  # noqa: E402

RAW_DIR = DATA_DIR / "hapt" / "RawData"
CONFIGS_DIR = ROOT / "configs"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "week1_closeout_20261102" / "reference"
EXPECTED_PARAMS = 283692
EXPECTED_SUPERVISED_TEST_DENOM = 3572
CLASS_NAMES_12 = [HAPT_12_CLASSES[i] for i in range(1, 13)]
REQUIRED_SEEDS: Tuple[int, ...] = (42, 43, 44)
REQUIRED_EPOCHS: int = 30


def flush_print(msg: str) -> None:
    print(msg, flush=True)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Two-sided Wilson score confidence interval for a binomial proportion."""
    if trials == 0:
        return None, None
    z = 1.959963984540054  # 95% confidence
    p = successes / trials
    denom = 1.0 + (z**2) / trials
    centre = (p + (z**2) / (2.0 * trials)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) / trials) + (z**2) / (4.0 * (trials**2)))) / denom
    lower = max(0.0, float(centre - spread))
    upper = min(1.0, float(centre + spread))
    return lower, upper


def fit_train_norm(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Train-only normalization across axes (0, 2) with std clipped at 1e-6."""
    mean = x_train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def compute_preprocessing_id(mean: np.ndarray, std: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(b"norm:train_only:per_channel:axes_0_2:clip_1e-6:float32:")
    h.update(mean.tobytes())
    h.update(std.tobytes())
    return h.hexdigest()


def get_environment_info() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cpu_model": get_cpu_model_name(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "torch_num_threads": torch.get_num_threads(),
        "device_configured": "cpu",
    }


def validate_output_directory_guard(out_dir: Path) -> None:
    """Enforces no overlap with source paths, repo ancestors, or symlink escapes."""
    resolved_out = out_dir.resolve()
    repo_root = ROOT.resolve()

    # Reject if output is parent or ancestor of repo
    if resolved_out in repo_root.parents or resolved_out == repo_root:
        raise ValueError(f"Unsafe output directory {out_dir}: cannot be repository root or its ancestor.")

    # Reject if output overlaps protected source directories
    protected_dirs = [
        (ROOT / "checkpoints").resolve(),
        (ROOT / "configs").resolve(),
        (ROOT / "data").resolve(),
        (ROOT / "ecdr").resolve(),
        (ROOT / "scripts").resolve(),
    ]
    for p in protected_dirs:
        if resolved_out == p or p in resolved_out.parents:
            raise ValueError(f"Unsafe output directory {out_dir}: overlaps protected path {p}")

    # Check symlink escapes
    if out_dir.is_symlink():
        raise ValueError(f"Unsafe output directory {out_dir}: symlinks not permitted.")


def create_or_load_config(output_dir: Path, argv: List[str]) -> Dict[str, Any]:
    config_path = output_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())

    cfg_yaml = yaml.safe_load((CONFIGS_DIR / "backbone.yaml").read_text())
    run_id = f"week1_ref_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    config_data = {
        "run_id": run_id,
        "command_argv": argv,
        "architecture": "SmallCNN1D-K12",
        "model_cfg": cfg_yaml.get("model", {}),
        "expected_params": EXPECTED_PARAMS,
        "class_order": CLASS_NAMES_12,
        "input_shape": [6, 128],
        "num_channels": 6,
        "label_policy": "pure_plus_majority_dominant_0.6",
        "training_hyperparameters": {
            "seeds": list(REQUIRED_SEEDS),
            "epochs": REQUIRED_EPOCHS,
            "batch_size": 128,
            "optimizer": "adam",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "min_lr": 0.00001,
            "patience": 10,
            "selection_metric": "val_macro_f1",
            "class_weights": None,
        },
        "normalization_spec": {
            "scope": "train_only",
            "axes": [0, 2],
            "dtype": "float32",
            "std_clip_min": 1e-6,
        },
        "environment": get_environment_info(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_path.write_text(json.dumps(config_data, indent=2))
    return config_data


# ==============================================================================
# PHASE: AUDIT
# ==============================================================================
def run_audit(output_dir: Path, argv: Optional[List[str]] = None) -> Dict[str, Any]:
    flush_print(">>> [Phase: AUDIT] Starting raw data, manifest & provenance audit...")
    validate_output_directory_guard(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_data = create_or_load_config(output_dir, argv or sys.argv)
    run_id = config_data["run_id"]

    # 1. Hashes of tracked source and data files
    tracked_files = [
        ROOT / "data" / "windows.npz",
        ROOT / "data" / "window_manifest.parquet",
        ROOT / "data" / "windows_with_split.parquet",
        CONFIGS_DIR / "backbone.yaml",
        CONFIGS_DIR / "training.yaml",
        ROOT / "scripts" / "backbone.py",
        ROOT / "scripts" / "03_build_windows.py",
        Path(__file__).resolve(),
        RAW_DIR / "labels.txt",
    ]
    raw_sensor_files = sorted(RAW_DIR.glob("acc_exp*_user*.txt")) + sorted(RAW_DIR.glob("gyro_exp*_user*.txt"))
    all_files_to_hash = tracked_files + raw_sensor_files

    hashes: Dict[str, str] = {}
    for p in all_files_to_hash:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for audit: {p}")
        rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        hashes[rel] = compute_sha256(p)

    flush_print(f"Hashed {len(hashes)} files (including {len(raw_sensor_files)} raw sensor recordings).")

    # 2. Verify ORIGINAL npz stored float32 dtype before loader casts
    windows_file = DATA_DIR / "windows.npz"
    with np.load(windows_file) as npz:
        if "x" not in npz.files:
            raise KeyError("data/windows.npz missing key 'x'")
        raw_x_arr = npz["x"]
        if raw_x_arr.dtype != np.float32:
            raise TypeError(f"windows.npz stored dtype is {raw_x_arr.dtype}, expected float32")
        if raw_x_arr.shape != (17453, 6, 128):
            raise ValueError(f"windows.npz shape {raw_x_arr.shape} != (17453, 6, 128)")
        if not np.all(np.isfinite(raw_x_arr)):
            raise ValueError("windows.npz contains non-finite values (NaN/Inf)")

    # 3. Load manifest with stable global_row_id and check denominators
    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    # Check unique global_row_id permutation
    global_ids = df_manifest["global_row_id"].to_numpy()
    if len(np.unique(global_ids)) != 17453 or not np.array_equal(np.sort(global_ids), np.arange(17453)):
        raise ValueError("Manifest global_row_id is not a valid 0..17452 unique permutation")

    # 4. Stream verification against raw sensor recordings & builder convention
    labels_raw = np.loadtxt(RAW_DIR / "labels.txt", dtype=int)
    per_rec_labels: Dict[int, List[np.ndarray]] = {}
    for row in labels_raw:
        per_rec_labels.setdefault(int(row[0]), []).append(row)

    W, S = 128, 64
    majority_thresh = 0.6
    mismatched_slices = 0
    mismatched_builder_policies = 0
    alternate_convention_diffs = 0

    acc_files = sorted(RAW_DIR.glob("acc_exp*_user*.txt"))
    total_recs = len(acc_files)

    for ap in acc_files:
        stem = ap.name.replace("acc_", "")
        gp = RAW_DIR / ("gyro_" + stem)
        exp = int(re.search(r"exp(\d+)", stem).group(1))
        usr = int(re.search(r"user(\d+)", stem).group(1))

        acc = np.loadtxt(ap, dtype=np.float32).reshape(-1, 3)
        gy = np.loadtxt(gp, dtype=np.float32).reshape(-1, 3)
        if acc.shape != gy.shape or acc.shape[1] != 3:
            raise ValueError(f"Corrupted sensor dimensions for exp {exp}: acc={acc.shape}, gy={gy.shape}")
        if not np.all(np.isfinite(acc)) or not np.all(np.isfinite(gy)):
            raise ValueError(f"Non-finite raw readings in exp {exp}")

        sig = np.concatenate([acc, gy], axis=1).T  # shape (6, T)
        T = sig.shape[1]

        # Existing builder convention: plabel size T+1, plabel[start:end+1] = label
        plabel = np.zeros(T + 1, dtype=np.int16)
        # Alternate convention: 0-based start-1:end closed converted
        plabel_alt = np.zeros(T, dtype=np.int16)

        for r in per_rec_labels.get(exp, []):
            st, en, lb = int(r[3]), int(r[4]), int(r[2])
            plabel[st : en + 1] = lb
            alt_st = max(0, st - 1)
            alt_en = min(T, en)
            plabel_alt[alt_st:alt_en] = lb

        exp_df = df_manifest[df_manifest["exp"] == exp].sort_values("win_idx")
        expected_window_count = (T - W) // S + 1
        if len(exp_df) != expected_window_count:
            raise ValueError(f"Manifest window count {len(exp_df)} != expected {expected_window_count} for exp {exp}")

        for idx, (_, row_meta) in enumerate(exp_df.iterrows()):
            if int(row_meta["win_idx"]) != idx:
                raise ValueError(f"Non-sequential win_idx in exp {exp}: {row_meta['win_idx']} != {idx}")
            s = idx * S
            e = s + W
            if int(row_meta["start"]) != s or int(row_meta["end"]) != e:
                raise ValueError(f"Window boundary mismatch in exp {exp}, win_idx {idx}: ({s}, {e}) != ({row_meta['start']}, {row_meta['end']})")
            if int(row_meta["subject"]) != usr:
                raise ValueError(f"Subject mismatch in exp {exp}: {row_meta['subject']} != {usr}")

            win_raw = sig[:, s:e]
            global_id = int(row_meta["global_row_id"])
            win_saved = x_all[global_id]

            if not np.array_equal(win_raw, win_saved):
                mismatched_slices += 1

            # Check overlap consistency with previous window in same recording
            if idx > 0:
                prev_saved = x_all[int(exp_df.iloc[idx - 1]["global_row_id"])]
                if not np.array_equal(win_saved[:, :64], prev_saved[:, 64:]):
                    raise ValueError(f"Stride-64 overlap mismatch in exp {exp}, win_idx {idx}")

            # Builder recreation logic
            labs = plabel[s:e]
            annotated = labs[labs > 0]
            frac_ann = len(annotated) / W
            if len(annotated) == 0:
                pol, maj = "unlabeled", 0
            else:
                vals, cnts = np.unique(annotated, return_counts=True)
                dom = int(vals[np.argmax(cnts)])
                dom_frac = int(cnts.max()) / W
                if len(vals) == 1 and dom_frac == 1.0:
                    pol, maj = "pure", dom
                elif dom_frac >= majority_thresh:
                    pol, maj = "majority", dom
                else:
                    pol, maj = "ambiguous", dom

            if (
                pol != row_meta["label_policy"]
                or maj != row_meta["label_majority"]
                or int(len(np.unique(annotated))) != row_meta["n_distinct_labels"]
                or not math.isclose(round(float(frac_ann), 4), row_meta["annotated_frac"], abs_tol=1e-4)
                or bool(len(np.unique(annotated)) > 1) != row_meta["crosses_boundary"]
            ):
                mismatched_builder_policies += 1

            # Alternate convention plabel_alt[s:e]
            labs_alt = plabel_alt[s:e]
            ann_alt = labs_alt[labs_alt > 0]
            if len(ann_alt) == 0:
                pol_alt, maj_alt = "unlabeled", 0
            else:
                vals_alt, cnts_alt = np.unique(ann_alt, return_counts=True)
                dom_alt = int(vals_alt[np.argmax(cnts_alt)])
                dom_frac_alt = int(cnts_alt.max()) / W
                if len(vals_alt) == 1 and dom_frac_alt == 1.0:
                    pol_alt, maj_alt = "pure", dom_alt
                elif dom_frac_alt >= majority_thresh:
                    pol_alt, maj_alt = "majority", dom_alt
                else:
                    pol_alt, maj_alt = "ambiguous", dom_alt

            if pol_alt != pol or maj_alt != maj:
                alternate_convention_diffs += 1

    flush_print(f"Verified {len(df_manifest)} windows across {total_recs} recordings.")
    if mismatched_slices > 0:
        raise ValueError(f"Audit failure: {mismatched_slices} windows did not match raw sensor signals.")
    if mismatched_builder_policies > 0:
        raise ValueError(f"Audit failure: {mismatched_builder_policies} windows did not match builder label logic.")

    # 5. Persist per-policy support and blocker thresholds per ecdr.schema
    support_table: Dict[str, Any] = {}
    support_failures = []

    pure_basic_total = int(((df_manifest["label_policy"] == "pure") & df_manifest["label_majority"].between(1, 6)).sum())
    pure_transition_total = int(((df_manifest["label_policy"] == "pure") & df_manifest["label_majority"].between(7, 12)).sum())

    if pure_basic_total != 10017:
        raise ValueError(f"Audit failure: pure basic (1..6) count is {pure_basic_total}, expected 10017")
    if pure_transition_total != 344:
        raise ValueError(f"Audit failure: pure transition (7..12) count is {pure_transition_total}, expected 344")

    for split in SPLITS:
        sdf = df_manifest[df_manifest["split"] == split]
        split_support = {
            "total_windows": len(sdf),
            "pure_windows": int((sdf["label_policy"] == "pure").sum()),
            "majority_windows": int((sdf["label_policy"] == "majority").sum()),
            "ambiguous_windows": int((sdf["label_policy"] == "ambiguous").sum()),
            "unlabeled_windows": int((sdf["label_policy"] == "unlabeled").sum()),
            "supervised_pure_majority_k12": int(
                ((sdf["label_policy"].isin(["pure", "majority"])) & sdf["label_majority"].between(1, 12)).sum()
            ),
            "per_class_pure": {},
            "per_class_pure_plus_majority": {},
        }

        # Check all 12 classes support
        for c in range(1, 13):
            c_name = HAPT_12_CLASSES[c]
            # Pure
            pure_mask = (sdf["label_policy"] == "pure") & (sdf["label_majority"] == c)
            pure_cnt = int(pure_mask.sum())
            pure_subjs = int(sdf[pure_mask]["subject"].nunique())
            split_support["per_class_pure"][c_name] = {"windows": pure_cnt, "subjects": pure_subjs}

            # Pure + Majority
            pm_mask = sdf["label_policy"].isin(["pure", "majority"]) & (sdf["label_majority"] == c)
            pm_cnt = int(pm_mask.sum())
            pm_subjs = int(sdf[pm_mask]["subject"].nunique())
            split_support["per_class_pure_plus_majority"][c_name] = {"windows": pm_cnt, "subjects": pm_subjs}

            # Blocker criteria check per ecdr.schema (MIN_SUPPORT_WINDOWS_PER_SPLIT=50, MIN_SUPPORT_SUBJECTS_PER_SPLIT=3)
            if pure_cnt < MIN_SUPPORT_WINDOWS_PER_SPLIT or pure_subjs < MIN_SUPPORT_SUBJECTS_PER_SPLIT:
                support_failures.append({
                    "split": split,
                    "class_id": c,
                    "class_name": c_name,
                    "policy": "pure",
                    "windows": pure_cnt,
                    "subjects": pure_subjs,
                    "min_windows_required": MIN_SUPPORT_WINDOWS_PER_SPLIT,
                    "min_subjects_required": MIN_SUPPORT_SUBJECTS_PER_SPLIT,
                })

        support_table[split] = split_support

    audit_summary = {
        "run_id": run_id,
        "phase": "audit",
        "status": "PASS",
        "file_hashes": hashes,
        "total_windows": len(df_manifest),
        "windows_npz_shape": list(x_all.shape),
        "windows_npz_dtype": str(x_all.dtype),
        "pure_basic_1_6_total": pure_basic_total,
        "pure_transition_7_12_total": pure_transition_total,
        "support_table": support_table,
        "schema_blocker_failures": support_failures,
        "mismatched_slices": mismatched_slices,
        "mismatched_builder_policies": mismatched_builder_policies,
        "alternate_convention_differences": alternate_convention_diffs,
        "annotation_convention_note": (
            "Recorded existing builder convention plabel[start:end+1] slice plabel[s:e]. "
            "Alternate 0-based conversion (start-1:end) evaluated as non-mutating sensitivity comparison only; "
            "no official 1-based origin claim is made absent authoritative specification documentation."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    audit_path = output_dir / "provenance_audit.json"
    audit_path.write_text(json.dumps(audit_summary, indent=2))

    # Phase receipt
    receipt = {
        "phase": "audit",
        "run_id": run_id,
        "status": "PASS",
        "provenance_audit_sha256": compute_sha256(audit_path),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "audit_receipt.json").write_text(json.dumps(receipt, indent=2))

    flush_print(f">>> [Phase: AUDIT] Completed successfully. Summary saved to {audit_path}")
    return audit_summary


# ==============================================================================
# PHASE: TRAIN
# ==============================================================================
def run_train(
    output_dir: Path,
    epochs: int = 30,
    seeds: Tuple[int, ...] = REQUIRED_SEEDS,
    device_str: str = "cpu",
    argv: Optional[List[str]] = None,
) -> Dict[str, Any]:
    flush_print(f">>> [Phase: TRAIN] Starting 3-seed training (epochs={epochs}, seeds={seeds}, device={device_str})...")
    validate_output_directory_guard(output_dir)

    # Phase guard: train requires passed matching audit output
    audit_receipt_path = output_dir / "audit_receipt.json"
    if not audit_receipt_path.exists():
        raise FileNotFoundError(
            f"Phase guard violation: 'train' requires prior passed audit in {output_dir}. "
            "Run 'audit' phase first."
        )

    # Phase guard: no existing training outputs
    training_outputs = [
        output_dir / "selection.json",
        output_dir / "selected_best.pt",
        output_dir / "checkpoint_seed_42.pt",
    ]
    for p in training_outputs:
        if p.exists():
            raise FileExistsError(f"Phase guard violation: existing training artifact {p.name} in {output_dir}. Refusing overwrite.")

    config_data = create_or_load_config(output_dir, argv or sys.argv)
    run_id = config_data["run_id"]

    # Verify architecture parameter count
    cfg_path = CONFIGS_DIR / "backbone.yaml"
    backbone_cfg = yaml.safe_load(cfg_path.read_text())
    test_model = SmallCNN1D(num_classes=12, cfg=backbone_cfg.get("model", {}))
    param_count = sum(p.numel() for p in test_model.parameters())
    if param_count != EXPECTED_PARAMS:
        raise ValueError(f"Model architecture mismatch: expected {EXPECTED_PARAMS} params, got {param_count}")
    del test_model

    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    # NOTE: No TEST INFERENCE or test metrics are used during selection; audit reads test metadata only.
    mask_sup = df_manifest["label_policy"].isin(["pure", "majority"]) & df_manifest["label_majority"].between(1, 12)
    train_df = df_manifest[mask_sup & (df_manifest["split"] == "train")]
    val_df = df_manifest[mask_sup & (df_manifest["split"] == "val")]

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Supervised splits cannot be empty: train={len(train_df)}, val={len(val_df)}")

    # Check all 12 labels exist in train data
    train_labels_unique = np.unique(train_df["label_majority"].to_numpy())
    if len(train_labels_unique) != 12 or not np.array_equal(train_labels_unique, np.arange(1, 13)):
        raise ValueError(f"Train split does not cover all 12 classes: found {train_labels_unique}")

    train_ids = train_df["global_row_id"].to_numpy(dtype=np.int64)
    val_ids = val_df["global_row_id"].to_numpy(dtype=np.int64)

    x_train_raw = x_all[train_ids]
    y_train = (train_df["label_majority"].to_numpy(dtype=np.int64) - 1)  # 0..11
    x_val_raw = x_all[val_ids]
    y_val = (val_df["label_majority"].to_numpy(dtype=np.int64) - 1)  # 0..11

    if not np.all(np.isfinite(x_train_raw)) or not np.all(np.isfinite(x_val_raw)):
        raise ValueError("Non-finite values detected in training or validation arrays.")

    # Train-only normalization
    mean, std = fit_train_norm(x_train_raw)
    preprocessing_id = compute_preprocessing_id(mean, std)

    norm_meta = {
        "run_id": run_id,
        "preprocessing_id": preprocessing_id,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "axes": [0, 2],
        "std_clip_min": 1e-6,
        "dtype": "float32",
        "scope": "train_only",
        "train_supervised_windows": len(x_train_raw),
    }
    (output_dir / "normalization.json").write_text(json.dumps(norm_meta, indent=2))
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, std=std)
    norm_sha256 = compute_sha256(output_dir / "normalization.npz")

    x_train = apply_norm(x_train_raw, mean, std)
    x_val = apply_norm(x_val_raw, mean, std)

    device = torch.device(device_str)
    batch_size = 128
    base_lr = 0.001
    weight_decay = 0.0001
    min_lr = 0.00001
    patience = 10

    seed_results = []
    checkpoint_hashes = {}
    history_hashes = {}

    for seed in seeds:
        flush_print(f"\n--- Training Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SmallCNN1D(num_classes=12, cfg=backbone_cfg.get("model", {})).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_val_macro_f1 = -1.0
        best_epoch = -1
        best_state_dict = None
        no_improve_epochs = 0

        n_train = len(x_train)
        indices = np.arange(n_train)

        for epoch in range(1, epochs + 1):
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))
            current_lr = min_lr + (base_lr - min_lr) * cos_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            rng = np.random.RandomState(seed * 1000 + epoch)
            rng.shuffle(indices)

            model.train()
            train_loss_total = 0.0
            for start_idx in range(0, n_train, batch_size):
                end_idx = min(start_idx + batch_size, n_train)
                batch_x = torch.from_numpy(x_train[indices[start_idx:end_idx]]).to(device)
                batch_y = torch.from_numpy(y_train[indices[start_idx:end_idx]]).to(device)

                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

                train_loss_total += loss.item() * (end_idx - start_idx)

            train_loss = train_loss_total / n_train

            model.eval()
            val_preds = []
            val_loss_total = 0.0
            with torch.no_grad():
                for start_idx in range(0, len(x_val), batch_size):
                    end_idx = min(start_idx + batch_size, len(x_val))
                    bx = torch.from_numpy(x_val[start_idx:end_idx]).to(device)
                    by = torch.from_numpy(y_val[start_idx:end_idx]).to(device)
                    vout = model(bx)
                    vloss = criterion(vout, by)
                    val_loss_total += vloss.item() * (end_idx - start_idx)
                    val_preds.append(vout.argmax(dim=1).cpu().numpy())

            val_loss = val_loss_total / len(x_val)
            y_val_pred = np.concatenate(val_preds)

            cm = np.zeros((12, 12), dtype=np.int64)
            np.add.at(cm, (y_val, y_val_pred), 1)
            tp = np.diag(cm).astype(np.float64)
            support = cm.sum(axis=1)
            pred_n = cm.sum(axis=0)
            recall = np.where(support > 0, tp / support, 0.0)
            precision = np.where(pred_n > 0, tp / pred_n, 0.0)
            f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

            # Store UNROUNDED validation metrics
            val_macro_f1 = float(f1.mean())
            val_acc = float(tp.sum() / len(y_val))

            history.append({
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
            })

            # Earliest epoch tie-breaker
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            flush_print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f} | "
                f"Patience: {no_improve_epochs}/{patience}"
            )

            if no_improve_epochs >= patience:
                flush_print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

        # Save checkpoint with bound identities
        ckpt_path = output_dir / f"checkpoint_seed_{seed}.pt"
        torch.save({
            "run_id": run_id,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "state_dict": best_state_dict,
            "num_classes": 12,
            "class_names": CLASS_NAMES_12,
            "model_cfg": backbone_cfg.get("model", {}),
            "expected_params": EXPECTED_PARAMS,
            "preprocessing_id": preprocessing_id,
            "norm_mean": mean,
            "norm_std": std,
            "norm_sha256": norm_sha256,
        }, ckpt_path)
        ckpt_hash = compute_sha256(ckpt_path)
        checkpoint_hashes[f"seed_{seed}"] = ckpt_hash

        hist_path = output_dir / f"history_seed_{seed}.json"
        hist_text = json.dumps(history, indent=2)
        hist_path.write_text(hist_text)
        history_hashes[f"seed_{seed}"] = compute_bytes_sha256(hist_text.encode("utf-8"))

        seed_results.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "checkpoint_file": ckpt_path.name,
            "checkpoint_sha256": ckpt_hash,
            "history_sha256": history_hashes[f"seed_{seed}"],
        })

    # Selection: best val_macro_f1, tie: smallest seed
    seed_results.sort(key=lambda item: (-item["best_val_macro_f1"], item["seed"]))
    winning_seed_info = seed_results[0]
    winning_seed = winning_seed_info["seed"]

    best_ckpt_src = output_dir / f"checkpoint_seed_{winning_seed}.pt"
    best_ckpt_dst = output_dir / "selected_best.pt"
    best_ckpt_dst.write_bytes(best_ckpt_src.read_bytes())
    selected_best_hash = compute_sha256(best_ckpt_dst)

    selection_meta = {
        "run_id": run_id,
        "selection_metric": "val_macro_f1",
        "selected_seed": winning_seed,
        "selected_best_epoch": winning_seed_info["best_epoch"],
        "selected_best_val_macro_f1": winning_seed_info["best_val_macro_f1"],
        "selected_best_sha256": selected_best_hash,
        "seed_evaluations": seed_results,
        "checkpoint_hashes": checkpoint_hashes,
        "history_hashes": history_hashes,
        "selection_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_timestamp_ns": time.time_ns(),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection_meta, indent=2))

    receipt = {
        "phase": "train",
        "run_id": run_id,
        "status": "PASS",
        "selection_sha256": compute_sha256(output_dir / "selection.json"),
        "selected_best_sha256": selected_best_hash,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "train_receipt.json").write_text(json.dumps(receipt, indent=2))

    flush_print(f"\n>>> [Phase: TRAIN] Completed. Selected Seed {winning_seed} saved to {best_ckpt_dst}")
    return selection_meta


# ==============================================================================
# PHASE: EVALUATE
# ==============================================================================
def run_evaluate(output_dir: Path, device_str: str = "cpu") -> Dict[str, Any]:
    flush_print(">>> [Phase: EVALUATE] Reloading selected model for independent test evaluation...")
    validate_output_directory_guard(output_dir)

    # Phase guard: evaluate requires frozen selection and no existing eval outputs
    selection_path = output_dir / "selection.json"
    norm_path = output_dir / "normalization.npz"
    if not selection_path.exists() or not norm_path.exists():
        raise FileNotFoundError(
            f"Phase guard violation: 'evaluate' requires selection.json and normalization.npz in {output_dir}. "
            "Run 'train' phase first."
        )

    eval_outputs = [
        output_dir / "evaluation_results.json",
        output_dir / "test_predictions.npz",
    ]
    for p in eval_outputs:
        if p.exists():
            raise FileExistsError(f"Phase guard violation: existing evaluation artifact {p.name} in {output_dir}. Refusing overwrite.")

    config_data = json.loads((output_dir / "config.json").read_text())
    run_id = config_data["run_id"]

    selection_meta = json.loads(selection_path.read_text())
    norm_data = np.load(norm_path)
    mean, std = norm_data["mean"], norm_data["std"]

    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    test_df = df_manifest[df_manifest["split"] == "test"].copy()
    if len(test_df) != FULL_STREAM_DENOMINATORS["test"]:
        raise ValueError(f"Expected {FULL_STREAM_DENOMINATORS['test']} test windows, got {len(test_df)}")

    test_global_ids = test_df["global_row_id"].to_numpy(dtype=np.int64)
    x_test_raw = x_all[test_global_ids]
    x_test_norm = apply_norm(x_test_raw, mean, std)

    device = torch.device(device_str)

    # 1. Evaluate all 3 seed checkpoints on supervised test set
    mask_test_sup = test_df["label_policy"].isin(["pure", "majority"]) & test_df["label_majority"].between(1, 12)
    sup_test_df = test_df[mask_test_sup].copy()
    sup_test_indices = np.where(mask_test_sup.to_numpy())[0]
    y_test_sup = sup_test_df["label_majority"].to_numpy(dtype=np.int64) - 1

    # Exact expected supervised test denominator check
    if len(sup_test_df) != EXPECTED_SUPERVISED_TEST_DENOM:
        raise ValueError(
            f"Supervised test denominator mismatch: expected exactly {EXPECTED_SUPERVISED_TEST_DENOM}, "
            f"got {len(sup_test_df)}. Experiment invalid."
        )

    seed_evals = {}
    for seed_item in selection_meta["seed_evaluations"]:
        seed = seed_item["seed"]
        ckpt_path = output_dir / f"checkpoint_seed_{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SmallCNN1D(num_classes=12, cfg=ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.inference_mode():
            batch_size = 512
            logits_list = []
            for i in range(0, len(x_test_norm), batch_size):
                bx = torch.from_numpy(x_test_norm[i : i + batch_size]).to(device)
                logits_list.append(model(bx).cpu().numpy())
            full_logits = np.concatenate(logits_list, axis=0)

        # Persist per-seed full test logits, predictions, and rowids
        seed_preds = full_logits.argmax(axis=1)
        np.savez_compressed(
            output_dir / f"test_predictions_seed_{seed}.npz",
            global_row_id=test_global_ids,
            logits=full_logits,
            predictions=seed_preds,
        )

        sup_logits = full_logits[sup_test_indices]
        sup_preds = sup_logits.argmax(axis=1)

        cm = np.zeros((12, 12), dtype=np.int64)
        np.add.at(cm, (y_test_sup, sup_preds), 1)
        tp = np.diag(cm).astype(np.float64)
        support = cm.sum(axis=1)
        pred_n = cm.sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(support > 0, tp / support, 0.0)
            prec = np.where(pred_n > 0, tp / pred_n, 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

        per_class = {}
        for c_idx, c_name in enumerate(CLASS_NAMES_12):
            w_low, w_high = wilson_score_interval(int(tp[c_idx]), int(support[c_idx]))
            per_class[c_name] = {
                "support": int(support[c_idx]),
                "precision": float(prec[c_idx]),
                "recall": float(rec[c_idx]),
                "f1": float(f1[c_idx]),
                "recall_wilson_95_ci": [w_low, w_high],
            }

        seed_evals[f"seed_{seed}"] = {
            "accuracy": float(tp.sum() / len(y_test_sup)),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }

    # Aggregate metrics across seeds using POPULATION standard deviation (ddof=0)
    accs = [v["accuracy"] for v in seed_evals.values()]
    f1s = [v["macro_f1"] for v in seed_evals.values()]
    aggregate_metrics = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_pop_std": float(np.std(accs, ddof=0)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_pop_std": float(np.std(f1s, ddof=0)),
        "stat_std_degree_of_freedom": 0,
    }

    # 2. Selected best model: fullstream test logits, predictions & point margins
    best_ckpt_path = output_dir / "selected_best.pt"
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    best_model = SmallCNN1D(num_classes=12, cfg=best_ckpt["model_cfg"]).to(device)
    best_model.load_state_dict(best_ckpt["state_dict"])
    best_model.eval()

    with torch.inference_mode():
        logits_list = []
        for i in range(0, len(x_test_norm), 512):
            bx = torch.from_numpy(x_test_norm[i : i + 512]).to(device)
            logits_list.append(best_model(bx).cpu().numpy())
        best_full_logits = np.concatenate(logits_list, axis=0)

    best_full_preds = best_full_logits.argmax(axis=1)

    # Winning point margin on full stream: f_c - max_{j != c} f_j
    sorted_logits = np.sort(best_full_logits, axis=1)
    winning_margins = sorted_logits[:, -1] - sorted_logits[:, -2]

    # Ground truth margins on supervised test subset
    sup_best_logits = best_full_logits[sup_test_indices]
    y_true_onehot = np.eye(12, dtype=bool)[y_test_sup]
    other_logits = sup_best_logits.copy()
    other_logits[y_true_onehot] = -np.inf
    gt_margins = sup_best_logits[np.arange(len(y_test_sup)), y_test_sup] - other_logits.max(axis=1)

    margin_quantiles = {
        "winning_margin_fullstream": {
            "min": float(np.min(winning_margins)),
            "p1": float(np.percentile(winning_margins, 1)),
            "p5": float(np.percentile(winning_margins, 5)),
            "p50": float(np.percentile(winning_margins, 50)),
            "p95": float(np.percentile(winning_margins, 95)),
            "p99": float(np.percentile(winning_margins, 99)),
            "max": float(np.max(winning_margins)),
            "disclaimer": "Empirical point margins only; no robust certification or safety claim.",
        },
        "groundtruth_margin_supervised_test": {
            "min": float(np.min(gt_margins)),
            "p1": float(np.percentile(gt_margins, 1)),
            "p5": float(np.percentile(gt_margins, 5)),
            "p50": float(np.percentile(gt_margins, 50)),
            "p95": float(np.percentile(gt_margins, 95)),
            "p99": float(np.percentile(gt_margins, 99)),
            "max": float(np.max(gt_margins)),
        },
    }

    # Persist predictions npz and CSV
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        global_row_id=test_global_ids,
        logits=best_full_logits,
        predictions=best_full_preds,
        winning_margins=winning_margins,
    )

    test_pred_df = test_df[["global_row_id", "exp", "subject", "win_idx", "start", "end", "label_policy", "label_majority"]].copy()
    test_pred_df["predicted_label_zero_based"] = best_full_preds
    test_pred_df["predicted_label_name"] = [CLASS_NAMES_12[p] for p in best_full_preds]
    test_pred_df["winning_point_margin"] = winning_margins
    test_pred_df.to_csv(output_dir / "test_predictions_manifest.csv", index=False)

    # 3. Full test stream adjacent prediction agreement
    test_pred_df_sorted = test_pred_df.sort_values(["exp", "win_idx"]).reset_index(drop=True)
    pair_records = []
    pure_stable_agrees, pure_stable_total = 0, 0
    pure_trans_agrees, pure_trans_total = 0, 0
    mixed_other_agrees, mixed_other_total = 0, 0
    pure_same_label_agrees, pure_same_label_total = 0, 0
    pure_switch_agrees, pure_switch_total = 0, 0
    excluded_gaps_count = 0

    for i in range(len(test_pred_df_sorted) - 1):
        cur = test_pred_df_sorted.iloc[i]
        nxt = test_pred_df_sorted.iloc[i + 1]

        # Check adjacency within same recording
        is_same_exp = (cur["exp"] == nxt["exp"])
        is_consec_win = (nxt["win_idx"] == cur["win_idx"] + 1)
        is_consec_sample = (nxt["start"] == cur["start"] + 64)

        if not (is_same_exp and is_consec_win and is_consec_sample):
            excluded_gaps_count += 1
            continue

        agree = bool(cur["predicted_label_zero_based"] == nxt["predicted_label_zero_based"])
        pol1, pol2 = cur["label_policy"], nxt["label_policy"]
        l1, l2 = cur["label_majority"], nxt["label_majority"]

        is_pure1 = (pol1 == "pure")
        is_pure2 = (pol2 == "pure")

        if is_pure1 and is_pure2 and (1 <= l1 <= 6) and (1 <= l2 <= 6) and (l1 == l2):
            category = "pure_stable"
            pure_stable_total += 1
            if agree:
                pure_stable_agrees += 1
        elif is_pure1 and is_pure2 and (7 <= l1 <= 12) and (7 <= l2 <= 12) and (l1 == l2):
            category = "pure_transition"
            pure_trans_total += 1
            if agree:
                pure_trans_agrees += 1
        else:
            category = "mixed_other"
            mixed_other_total += 1
            if agree:
                mixed_other_agrees += 1

        # Additional breakdown: pure same-label vs pure switch
        if is_pure1 and is_pure2:
            if l1 == l2:
                pure_same_label_total += 1
                if agree:
                    pure_same_label_agrees += 1
            else:
                pure_switch_total += 1
                if agree:
                    pure_switch_agrees += 1

        pair_records.append({
            "exp": int(cur["exp"]),
            "win_idx_pair": [int(cur["win_idx"]), int(nxt["win_idx"])],
            "pred_pair": [int(cur["predicted_label_zero_based"]), int(nxt["predicted_label_zero_based"])],
            "agreement": agree,
            "category": category,
        })

    adjacent_agreement_summary = {
        "pure_stable": {
            "agreed": pure_stable_agrees,
            "total": pure_stable_total,
            "rate": (pure_stable_agrees / pure_stable_total) if pure_stable_total > 0 else None,
        },
        "pure_transition": {
            "agreed": pure_trans_agrees,
            "total": pure_trans_total,
            "rate": (pure_trans_agrees / pure_trans_total) if pure_trans_total > 0 else None,
        },
        "mixed_other": {
            "agreed": mixed_other_agrees,
            "total": mixed_other_total,
            "rate": (mixed_other_agrees / mixed_other_total) if mixed_other_total > 0 else None,
        },
        "pure_breakdown": {
            "pure_same_label": {
                "agreed": pure_same_label_agrees,
                "total": pure_same_label_total,
                "rate": (pure_same_label_agrees / pure_same_label_total) if pure_same_label_total > 0 else None,
            },
            "pure_label_switch": {
                "agreed": pure_switch_agrees,
                "total": pure_switch_total,
                "rate": (pure_switch_agrees / pure_switch_total) if pure_switch_total > 0 else None,
            },
        },
        "total_valid_adjacent_pairs": len(pair_records),
        "excluded_gaps_and_boundaries": excluded_gaps_count,
        "disclaimer": "Adjacent stream agreement rates characterize temporal stability; no ECDR certification claims implied.",
    }

    # 4. CPU Latency Benchmark (1 thread, batch 1, cycling varied inputs)
    flush_print("Benchmarking single-thread CPU latency (batch=1, cycling varied inputs)...")
    torch.set_num_threads(1)
    cpu_model = best_model.to("cpu")
    cpu_model.eval()

    n_test = len(x_test_norm)
    warmup_n = 100
    timed_n = 1000
    unique_inputs_used = min(timed_n, n_test)

    # a) Forward pass only (pre-normalized)
    for i in range(warmup_n):
        row_x = torch.from_numpy(x_test_norm[[i % n_test]]).to("cpu")
        with torch.inference_mode():
            _ = cpu_model(row_x)

    fwd_times_ms = []
    for i in range(timed_n):
        row_x = torch.from_numpy(x_test_norm[[i % n_test]]).to("cpu")
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            _ = cpu_model(row_x)
        t1 = time.perf_counter_ns()
        fwd_times_ms.append((t1 - t0) / 1e6)

    # b) Reference core preprocessing + forward pass
    for i in range(warmup_n):
        raw_x = x_test_raw[[i % n_test]]
        with torch.inference_mode():
            norm_x = torch.from_numpy(apply_norm(raw_x, mean, std)).to("cpu")
            _ = cpu_model(norm_x)

    core_times_ms = []
    for i in range(timed_n):
        raw_x = x_test_raw[[i % n_test]]
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            norm_x = torch.from_numpy(apply_norm(raw_x, mean, std)).to("cpu")
            _ = cpu_model(norm_x)
        t1 = time.perf_counter_ns()
        core_times_ms.append((t1 - t0) / 1e6)

    latency_summary = {
        "platform": {
            "cpu_model": get_cpu_model_name(),
            "platform_machine": platform.machine(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "threads": 1,
            "device": "cpu",
            "batch_shape": [1, 6, 128],
            "warmup_calls": warmup_n,
            "timed_calls": timed_n,
            "unique_inputs_cycled": unique_inputs_used,
            "selected_checkpoint_sha256": compute_sha256(best_ckpt_path),
        },
        "forward_only_ms": {
            "mean": float(np.mean(fwd_times_ms)),
            "median": float(np.median(fwd_times_ms)),
            "pop_std": float(np.std(fwd_times_ms, ddof=0)),
            "min": float(np.min(fwd_times_ms)),
            "p95": float(np.percentile(fwd_times_ms, 95)),
            "p99": float(np.percentile(fwd_times_ms, 99)),
        },
        "reference_core_preprocess_forward_ms": {
            "mean": float(np.mean(core_times_ms)),
            "median": float(np.median(core_times_ms)),
            "pop_std": float(np.std(core_times_ms, ddof=0)),
            "min": float(np.min(core_times_ms)),
            "p95": float(np.percentile(core_times_ms, 95)),
            "p99": float(np.percentile(core_times_ms, 99)),
            "scope_exclusion_note": (
                "Timing explicitly EXCLUDES raw acquisition, windowing, event selection, "
                "certificate construction, membership, cache overhead, and disk I/O."
            ),
        },
        "disclaimer": "CPU execution benchmark on varied test inputs; no speedup or optimization claims absent ECDR runtime comparisons.",
    }

    eval_results = {
        "run_id": run_id,
        "supervised_test_denominator": len(sup_test_df),
        "seed_evaluations": seed_evals,
        "aggregate_metrics": aggregate_metrics,
        "margins": margin_quantiles,
        "adjacent_prediction_agreement": adjacent_agreement_summary,
        "cpu_latency": latency_summary,
        "descriptive_interval_caveat": (
            "Wilson 95% score intervals are descriptive summaries of sample proportions; "
            "overlapping windows (stride 64) and shared recording subjects violate sample independence assumptions. "
            "No formal population confidence interval or hypothesis certification claim is made."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    eval_out_path = output_dir / "evaluation_results.json"
    eval_out_path.write_text(json.dumps(eval_results, indent=2))

    receipt = {
        "phase": "evaluate",
        "run_id": run_id,
        "status": "PASS",
        "evaluation_results_sha256": compute_sha256(eval_out_path),
        "test_predictions_sha256": compute_sha256(output_dir / "test_predictions.npz"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "eval_receipt.json").write_text(json.dumps(receipt, indent=2))

    flush_print(f">>> [Phase: EVALUATE] Completed. Results saved to {eval_out_path}")
    return eval_results


# ==============================================================================
# PHASE: VERIFY
# ==============================================================================
def run_verify(output_dir: Path) -> Dict[str, Any]:
    flush_print(">>> [Phase: VERIFY] Running independent integrity checks and verification...")
    validate_output_directory_guard(output_dir)

    receipt_path = output_dir / "verification_receipt.json"

    required_artifacts = [
        output_dir / "config.json",
        output_dir / "audit_receipt.json",
        output_dir / "provenance_audit.json",
        output_dir / "normalization.json",
        output_dir / "normalization.npz",
        output_dir / "checkpoint_seed_42.pt",
        output_dir / "checkpoint_seed_43.pt",
        output_dir / "checkpoint_seed_44.pt",
        output_dir / "history_seed_42.json",
        output_dir / "history_seed_43.json",
        output_dir / "history_seed_44.json",
        output_dir / "train_receipt.json",
        output_dir / "selection.json",
        output_dir / "selected_best.pt",
        output_dir / "test_predictions_seed_42.npz",
        output_dir / "test_predictions_seed_43.npz",
        output_dir / "test_predictions_seed_44.npz",
        output_dir / "test_predictions.npz",
        output_dir / "test_predictions_manifest.csv",
        output_dir / "evaluation_results.json",
        output_dir / "eval_receipt.json",
    ]

    for p in required_artifacts:
        if not p.exists():
            raise FileNotFoundError(f"Verification failure: required artifact {p.name} does not exist in {output_dir}")

    config_data = json.loads((output_dir / "config.json").read_text())
    run_id = config_data["run_id"]
    checks_executed = []

    def record_check(name: str, passed: bool, detail: str = "") -> None:
        if not passed:
            raise ValueError(f"Verification failure: check '{name}' failed. Detail: {detail}")
        checks_executed.append({"check": name, "passed": True, "detail": detail})

    # 1. Verify frozen legacy files unchanged
    frozen_files = [
        ROOT / "scripts" / "backbone.py",
        CONFIGS_DIR / "backbone.yaml",
    ]
    audit_json = json.loads((output_dir / "provenance_audit.json").read_text())
    stored_hashes = audit_json["file_hashes"]

    for ff in frozen_files:
        rel = str(ff.relative_to(ROOT))
        current_hash = compute_sha256(ff)
        record_check(f"frozen_file_{rel}", current_hash == stored_hashes[rel], f"Hash: {current_hash}")

    # 2. Verify model parameter count & shape
    best_ckpt = torch.load(output_dir / "selected_best.pt", map_location="cpu", weights_only=False)
    model = SmallCNN1D(num_classes=12, cfg=best_ckpt["model_cfg"])
    model.load_state_dict(best_ckpt["state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    record_check("model_param_count", param_count == EXPECTED_PARAMS, f"Params: {param_count}")
    record_check("model_output_classes", model.head[-1].out_features == 12, "Output dim: 12")

    # 3. Recompute train-only normalization and compare
    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    mask_sup_train = (
        (df_manifest["split"] == "train")
        & df_manifest["label_policy"].isin(["pure", "majority"])
        & df_manifest["label_majority"].between(1, 12)
    )
    x_train_raw = x_all[df_manifest[mask_sup_train]["global_row_id"].to_numpy()]
    recomputed_mean, recomputed_std = fit_train_norm(x_train_raw)
    recomputed_prep_id = compute_preprocessing_id(recomputed_mean, recomputed_std)

    saved_norm = np.load(output_dir / "normalization.npz")
    record_check("norm_mean_recomputed", np.allclose(recomputed_mean, saved_norm["mean"], atol=1e-6))
    record_check("norm_std_recomputed", np.allclose(recomputed_std, saved_norm["std"], atol=1e-6))

    norm_json = json.loads((output_dir / "normalization.json").read_text())
    record_check("preprocessing_id_match", recomputed_prep_id == norm_json["preprocessing_id"])

    # 4. Check selection freeze prior to eval
    selection_meta = json.loads((output_dir / "selection.json").read_text())
    eval_receipt = json.loads((output_dir / "eval_receipt.json").read_text())
    selected_hash = compute_sha256(output_dir / "selected_best.pt")
    record_check("selected_best_hash_proof", selection_meta["selected_best_sha256"] == selected_hash)
    record_check("selection_precedes_eval", selection_meta["selection_timestamp_utc"] <= eval_receipt["timestamp_utc"])

    # 5. Verify prediction reproducibility on all 5053 test windows
    test_df = df_manifest[df_manifest["split"] == "test"]
    x_test_norm = apply_norm(x_all[test_df["global_row_id"].to_numpy()], saved_norm["mean"], saved_norm["std"])

    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(x_test_norm)).numpy()
        preds = logits.argmax(axis=1)

    saved_npz = np.load(output_dir / "test_predictions.npz")
    record_check("all_5053_predictions_reproduced", np.array_equal(preds, saved_npz["predictions"]))
    record_check("all_5053_logits_reproduced", np.allclose(logits, saved_npz["logits"], atol=1e-5))

    # 6. Verify arithmetic of metrics from persisted predictions
    mask_test_sup = test_df["label_policy"].isin(["pure", "majority"]) & test_df["label_majority"].between(1, 12)
    sup_test_indices = np.where(mask_test_sup.to_numpy())[0]
    y_test_sup = (test_df[mask_test_sup]["label_majority"].to_numpy(dtype=np.int64) - 1)

    record_check("supervised_test_denominator", len(y_test_sup) == EXPECTED_SUPERVISED_TEST_DENOM, f"Count: {len(y_test_sup)}")

    eval_json = json.loads((output_dir / "evaluation_results.json").read_text())
    selected_seed = selection_meta["selected_seed"]
    selected_seed_eval = eval_json["seed_evaluations"][f"seed_{selected_seed}"]

    cm_recomputed = np.zeros((12, 12), dtype=np.int64)
    np.add.at(cm_recomputed, (y_test_sup, preds[sup_test_indices]), 1)
    record_check("confusion_matrix_arithmetic", np.array_equal(cm_recomputed, selected_seed_eval["confusion_matrix"]))

    # 7. Check across-seed population standard deviation arithmetic
    accs = [eval_json["seed_evaluations"][f"seed_{s}"]["accuracy"] for s in REQUIRED_SEEDS]
    f1s = [eval_json["seed_evaluations"][f"seed_{s}"]["macro_f1"] for s in REQUIRED_SEEDS]
    record_check("aggregate_acc_mean", math.isclose(np.mean(accs), eval_json["aggregate_metrics"]["accuracy_mean"], abs_tol=1e-6))
    record_check("aggregate_acc_pop_std", math.isclose(np.std(accs, ddof=0), eval_json["aggregate_metrics"]["accuracy_pop_std"], abs_tol=1e-6))
    record_check("aggregate_f1_mean", math.isclose(np.mean(f1s), eval_json["aggregate_metrics"]["macro_f1_mean"], abs_tol=1e-6))
    record_check("aggregate_f1_pop_std", math.isclose(np.std(f1s, ddof=0), eval_json["aggregate_metrics"]["macro_f1_pop_std"], abs_tol=1e-6))

    receipt = {
        "phase": "verify",
        "run_id": run_id,
        "status": "PASS",
        "checks_executed": checks_executed,
        "total_checks_count": len(checks_executed),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    flush_print(f">>> [Phase: VERIFY] All {len(checks_executed)} verification checks passed. Receipt saved to {receipt_path}")
    return receipt


# ==============================================================================
# MAIN CLI DRIVER
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 1 Reference Pipeline: SmallCNN1D K=12 full-label HAPT."
    )
    parser.add_argument(
        "phase",
        choices=["audit", "train", "evaluate", "verify", "all"],
        help="Pipeline phase to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for reference artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (strictly 30 required for reference).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds (strictly '42,43,44' required in order).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training/eval (default: 'cpu').",
    )

    args = parser.parse_args()

    # 1. Strict CLI validation before ANY filesystem writes
    if args.epochs != REQUIRED_EPOCHS:
        raise ValueError(
            f"Strict specification error: --epochs must be exactly {REQUIRED_EPOCHS}, got {args.epochs}."
        )

    try:
        parsed_seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    except Exception:
        raise ValueError(f"Malformed seeds format: '{args.seeds}'. Expected exactly '42,43,44'.")

    if parsed_seeds != REQUIRED_SEEDS:
        raise ValueError(
            f"Strict specification error: seeds must be exactly {REQUIRED_SEEDS} in that order, got {parsed_seeds}."
        )

    out_dir = args.output_dir
    validate_output_directory_guard(out_dir)

    # 2. Phase-specific guards
    if args.phase == "all":
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"Phase guard violation: 'all' requires a new empty target directory: {out_dir}")

    flush_print(f"Executing phase '{args.phase}' on target directory: {out_dir}")

    if args.phase == "audit":
        run_audit(out_dir, argv=sys.argv)
    elif args.phase == "train":
        run_train(out_dir, epochs=args.epochs, seeds=parsed_seeds, device_str=args.device, argv=sys.argv)
    elif args.phase == "evaluate":
        run_evaluate(out_dir, device_str=args.device)
    elif args.phase == "verify":
        run_verify(out_dir)
    elif args.phase == "all":
        run_audit(out_dir, argv=sys.argv)
        run_train(out_dir, epochs=args.epochs, seeds=parsed_seeds, device_str=args.device, argv=sys.argv)
        run_evaluate(out_dir, device_str=args.device)
        run_verify(out_dir)


if __name__ == "__main__":
    main()
