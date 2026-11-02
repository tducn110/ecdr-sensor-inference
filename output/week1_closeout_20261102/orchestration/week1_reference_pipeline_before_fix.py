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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Explicitly ensure single thread determinism
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.data import (  # noqa: E402
    DATA_DIR,
    FULL_STREAM_DENOMINATORS,
    HAPT_12_CLASSES,
    load_full_manifest_with_splits,
    load_raw_windows,
)
from scripts.backbone import SmallCNN1D  # noqa: E402

RAW_DIR = DATA_DIR / "hapt" / "RawData"
CONFIGS_DIR = ROOT / "configs"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "week1_closeout_20261102" / "reference"
EXPECTED_PARAMS = 283692
CLASS_NAMES_12 = [HAPT_12_CLASSES[i] for i in range(1, 13)]


def flush_print(msg: str) -> None:
    print(msg, flush=True)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
    return round(lower, 6), round(upper, 6)


def fit_train_norm(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Train-only normalization across axes (0, 2) clipped at 1e-6."""
    mean = x_train.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.clip(std, 1e-6, None)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


# ==============================================================================
# PHASE: AUDIT
# ==============================================================================
def run_audit(output_dir: Path) -> Dict[str, Any]:
    flush_print(">>> [Phase: AUDIT] Starting raw data, manifest & provenance audit...")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Load manifest with stable global_row_id and check denominators
    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    # 3. Stream verification against raw sensor recordings & builder convention
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

    for i, ap in enumerate(acc_files):
        stem = ap.name.replace("acc_", "")
        gp = RAW_DIR / ("gyro_" + stem)
        exp = int(re.search(r"exp(\d+)", stem).group(1))
        usr = int(re.search(r"user(\d+)", stem).group(1))

        acc = np.loadtxt(ap, dtype=np.float32).reshape(-1, 3)
        gy = np.loadtxt(gp, dtype=np.float32).reshape(-1, 3)
        sig = np.concatenate([acc, gy], axis=1).T  # shape (6, T)
        T = sig.shape[1]

        # Existing builder convention: plabel size T+1, plabel[start:end+1] = label
        plabel = np.zeros(T + 1, dtype=np.int16)
        # Alternate convention: 0-based start-1:end closed converted
        plabel_alt = np.zeros(T, dtype=np.int16)

        for r in per_rec_labels.get(exp, []):
            st, en, lb = int(r[3]), int(r[4]), int(r[2])
            plabel[st : en + 1] = lb
            # Alternate: if r[3], r[4] are 1-based, 0-based slice is (st - 1) : en
            alt_st = max(0, st - 1)
            alt_en = min(T, en)
            plabel_alt[alt_st:alt_en] = lb

        # Manifest subset for this experiment
        exp_df = df_manifest[df_manifest["exp"] == exp].sort_values("win_idx")

        idx = 0
        for s in range(0, T - W + 1, S):
            e = s + W
            win_raw = sig[:, s:e]
            row_meta = exp_df.iloc[idx]
            global_id = int(row_meta["global_row_id"])

            # Verify saved window in windows.npz equals raw window
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

            # Compare with alternate convention plabel_alt[s:e]
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

            idx += 1

    flush_print(f"Verified {len(df_manifest)} windows across {total_recs} recordings.")
    if mismatched_slices > 0:
        raise ValueError(f"Audit failure: {mismatched_slices} windows did not match raw sensor signals.")
    if mismatched_builder_policies > 0:
        raise ValueError(f"Audit failure: {mismatched_builder_policies} windows did not match builder label logic.")

    # 4. Support counts per split and policy
    support_table: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        sdf = df_manifest[df_manifest["split"] == split]
        support_table[split] = {
            "total": len(sdf),
            "pure": int((sdf["label_policy"] == "pure").sum()),
            "majority": int((sdf["label_policy"] == "majority").sum()),
            "ambiguous": int((sdf["label_policy"] == "ambiguous").sum()),
            "unlabeled": int((sdf["label_policy"] == "unlabeled").sum()),
            "supervised_pure_majority_k12": int(
                ((sdf["label_policy"].isin(["pure", "majority"])) & sdf["label_majority"].between(1, 12)).sum()
            ),
        }

    audit_summary = {
        "status": "PASS",
        "file_hashes": hashes,
        "total_windows": len(df_manifest),
        "windows_npz_shape": list(x_all.shape),
        "windows_npz_dtype": str(x_all.dtype),
        "support_table": support_table,
        "mismatched_slices": mismatched_slices,
        "mismatched_builder_policies": mismatched_builder_policies,
        "alternate_convention_differences": alternate_convention_diffs,
        "annotation_convention_note": (
            "Recorded existing builder convention plabel[start:end+1] slice plabel[s:e]. "
            "Alternate 0-based conversion (start-1:end) evaluated as non-mutating audit comparison only; "
            "authoritative indexing resolution deferred absent official specification docs."
        ),
    }

    audit_path = output_dir / "provenance_audit.json"
    audit_path.write_text(json.dumps(audit_summary, indent=2))
    flush_print(f">>> [Phase: AUDIT] Completed successfully. Summary saved to {audit_path}")
    return audit_summary


# ==============================================================================
# PHASE: TRAIN
# ==============================================================================
def run_train(
    output_dir: Path,
    epochs: int = 30,
    seeds: Tuple[int, ...] = (42, 43, 44),
    device_str: str = "cpu",
) -> Dict[str, Any]:
    flush_print(f">>> [Phase: TRAIN] Starting 3-seed training (epochs={epochs}, seeds={seeds}, device={device_str})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if epochs != 30:
        raise ValueError(f"Strict specification: --epochs must be 30 for reference pipeline, got {epochs}")

    # Load frozen backbone config and instantiate model to verify parameter count
    cfg_path = CONFIGS_DIR / "backbone.yaml"
    backbone_cfg = yaml.safe_load(cfg_path.read_text())
    test_model = SmallCNN1D(num_classes=12, cfg=backbone_cfg.get("model", {}))
    param_count = sum(p.numel() for p in test_model.parameters())
    if param_count != EXPECTED_PARAMS:
        raise ValueError(f"Model architecture mismatch: expected {EXPECTED_PARAMS} params, got {param_count}")
    del test_model

    # Load manifest and splits
    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    # Filter supervised windows for train and val splits ONLY (test arrays MUST NOT be loaded or touched)
    mask_sup = df_manifest["label_policy"].isin(["pure", "majority"]) & df_manifest["label_majority"].between(1, 12)
    train_df = df_manifest[mask_sup & (df_manifest["split"] == "train")]
    val_df = df_manifest[mask_sup & (df_manifest["split"] == "val")]

    train_ids = train_df["global_row_id"].to_numpy(dtype=np.int64)
    val_ids = val_df["global_row_id"].to_numpy(dtype=np.int64)

    x_train_raw = x_all[train_ids]
    y_train = (train_df["label_majority"].to_numpy(dtype=np.int64) - 1)  # 0..11
    x_val_raw = x_all[val_ids]
    y_val = (val_df["label_majority"].to_numpy(dtype=np.int64) - 1)  # 0..11

    # Train-only normalization
    mean, std = fit_train_norm(x_train_raw)
    norm_meta = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "axes": [0, 2],
        "std_clip_min": 1e-6,
        "scope": "train_only",
        "train_supervised_windows": len(x_train_raw),
    }
    (output_dir / "normalization.json").write_text(json.dumps(norm_meta, indent=2))
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, std=std)

    x_train = apply_norm(x_train_raw, mean, std)
    x_val = apply_norm(x_val_raw, mean, std)

    device = torch.device(device_str)
    batch_size = 128
    base_lr = 0.001
    weight_decay = 0.0001
    min_lr = base_lr / 100.0
    patience = 10

    seed_results = []
    checkpoint_hashes = {}

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
            # Cosine learning rate schedule per epoch
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / epochs))
            current_lr = min_lr + (base_lr - min_lr) * cos_factor
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # Shuffle training data deterministically with epoch seed
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

            # Validation inference
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

            # Compute validation macro-F1 (12 classes)
            cm = np.zeros((12, 12), dtype=np.int64)
            np.add.at(cm, (y_val, y_val_pred), 1)
            tp = np.diag(cm).astype(np.float64)
            support = cm.sum(axis=1)
            pred_n = cm.sum(axis=0)
            recall = np.where(support > 0, tp / support, 0.0)
            precision = np.where(pred_n > 0, tp / pred_n, 0.0)
            f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
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

            # Early stopping check on validation macro-F1 (earliest epoch tie-breaker)
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

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_seed_{seed}.pt"
        torch.save({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "state_dict": best_state_dict,
            "num_classes": 12,
            "model_cfg": backbone_cfg.get("model", {}),
        }, ckpt_path)
        ckpt_hash = compute_sha256(ckpt_path)
        checkpoint_hashes[f"seed_{seed}"] = ckpt_hash

        hist_path = output_dir / f"history_seed_{seed}.json"
        hist_path.write_text(json.dumps(history, indent=2))

        seed_results.append({
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "checkpoint_file": ckpt_path.name,
            "checkpoint_sha256": ckpt_hash,
        })

    # Select winning seed based on validation macro-F1 (tie: smallest seed number)
    seed_results.sort(key=lambda item: (-item["best_val_macro_f1"], item["seed"]))
    winning_seed_info = seed_results[0]
    winning_seed = winning_seed_info["seed"]

    # Copy winning checkpoint to selected_best.pt
    best_ckpt_src = output_dir / f"checkpoint_seed_{winning_seed}.pt"
    best_ckpt_dst = output_dir / "selected_best.pt"
    best_ckpt_dst.write_bytes(best_ckpt_src.read_bytes())
    selected_best_hash = compute_sha256(best_ckpt_dst)

    selection_meta = {
        "selection_metric": "val_macro_f1",
        "selected_seed": winning_seed,
        "selected_best_epoch": winning_seed_info["best_epoch"],
        "selected_best_val_macro_f1": winning_seed_info["best_val_macro_f1"],
        "selected_best_sha256": selected_best_hash,
        "seed_evaluations": seed_results,
        "selection_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection_meta, indent=2))
    flush_print(f"\n>>> [Phase: TRAIN] Completed. Selected Seed {winning_seed} saved to {best_ckpt_dst}")
    return selection_meta


# ==============================================================================
# PHASE: EVALUATE
# ==============================================================================
def run_evaluate(output_dir: Path, device_str: str = "cpu") -> Dict[str, Any]:
    flush_print(">>> [Phase: EVALUATE] Reloading selected model for independent test evaluation...")

    selection_path = output_dir / "selection.json"
    norm_path = output_dir / "normalization.npz"
    if not selection_path.exists() or not norm_path.exists():
        raise FileNotFoundError("Missing selection.json or normalization.npz. Must run train phase first.")

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

    expected_sup_denominator = 3572
    if len(sup_test_df) != expected_sup_denominator:
        flush_print(f"WARNING: Verified supervised test count is {len(sup_test_df)} (expected reference {expected_sup_denominator}).")

    seed_evals = {}
    for seed_item in selection_meta["seed_evaluations"]:
        seed = seed_item["seed"]
        ckpt_path = output_dir / f"checkpoint_seed_{seed}.pt"
        ckpt = torch.save if False else torch.load(ckpt_path, map_location=device)
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
                "precision": round(float(prec[c_idx]), 4),
                "recall": round(float(rec[c_idx]), 4),
                "f1": round(float(f1[c_idx]), 4),
                "recall_wilson_95_ci": [w_low, w_high],
            }

        seed_evals[f"seed_{seed}"] = {
            "accuracy": round(float(tp.sum() / len(y_test_sup)), 4),
            "macro_f1": round(float(f1.mean()), 4),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }

    # Aggregate metrics across seeds (mean +/- std, ddof=1)
    accs = [v["accuracy"] for v in seed_evals.values()]
    f1s = [v["macro_f1"] for v in seed_evals.values()]
    aggregate_metrics = {
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs, ddof=1)), 4),
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s, ddof=1)), 4),
    }

    # 2. Selected best model: fullstream test logits, predictions & point margins
    best_ckpt_path = output_dir / "selected_best.pt"
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
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
            "min": round(float(np.min(winning_margins)), 4),
            "p1": round(float(np.percentile(winning_margins, 1)), 4),
            "p5": round(float(np.percentile(winning_margins, 5)), 4),
            "p50": round(float(np.percentile(winning_margins, 50)), 4),
            "p95": round(float(np.percentile(winning_margins, 95)), 4),
            "p99": round(float(np.percentile(winning_margins, 99)), 4),
            "max": round(float(np.max(winning_margins)), 4),
            "disclaimer": "Empirical point margins only; no robust certification or safety claim.",
        },
        "groundtruth_margin_supervised_test": {
            "min": round(float(np.min(gt_margins)), 4),
            "p1": round(float(np.percentile(gt_margins, 1)), 4),
            "p5": round(float(np.percentile(gt_margins, 5)), 4),
            "p50": round(float(np.percentile(gt_margins, 50)), 4),
            "p95": round(float(np.percentile(gt_margins, 95)), 4),
            "p99": round(float(np.percentile(gt_margins, 99)), 4),
            "max": round(float(np.max(gt_margins)), 4),
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

    for i in range(len(test_pred_df_sorted) - 1):
        cur = test_pred_df_sorted.iloc[i]
        nxt = test_pred_df_sorted.iloc[i + 1]

        # Check adjacency within same recording
        if cur["exp"] == nxt["exp"] and nxt["win_idx"] == cur["win_idx"] + 1 and nxt["start"] == cur["start"] + 64:
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
            "rate": round(pure_stable_agrees / max(pure_stable_total, 1), 4),
        },
        "pure_transition": {
            "agreed": pure_trans_agrees,
            "total": pure_trans_total,
            "rate": round(pure_trans_agrees / max(pure_trans_total, 1), 4),
        },
        "mixed_other": {
            "agreed": mixed_other_agrees,
            "total": mixed_other_total,
            "rate": round(mixed_other_agrees / max(mixed_other_total, 1), 4),
        },
        "total_valid_pairs": len(pair_records),
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

    # b) End-to-end preprocessing + forward pass
    for i in range(warmup_n):
        raw_x = x_test_raw[[i % n_test]]
        with torch.inference_mode():
            norm_x = torch.from_numpy(apply_norm(raw_x, mean, std)).to("cpu")
            _ = cpu_model(norm_x)

    e2e_times_ms = []
    for i in range(timed_n):
        raw_x = x_test_raw[[i % n_test]]
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            norm_x = torch.from_numpy(apply_norm(raw_x, mean, std)).to("cpu")
            _ = cpu_model(norm_x)
        t1 = time.perf_counter_ns()
        e2e_times_ms.append((t1 - t0) / 1e6)

    latency_summary = {
        "platform": {
            "processor": platform.processor(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "threads": 1,
            "device": "cpu",
            "batch_shape": [1, 6, 128],
            "warmup_calls": warmup_n,
            "timed_calls": timed_n,
            "selected_checkpoint_sha256": compute_sha256(best_ckpt_path),
        },
        "forward_only_ms": {
            "mean": round(float(np.mean(fwd_times_ms)), 4),
            "median": round(float(np.median(fwd_times_ms)), 4),
            "std": round(float(np.std(fwd_times_ms, ddof=1)), 4),
            "min": round(float(np.min(fwd_times_ms)), 4),
            "p95": round(float(np.percentile(fwd_times_ms, 95)), 4),
            "p99": round(float(np.percentile(fwd_times_ms, 99)), 4),
        },
        "end_to_end_preprocess_forward_ms": {
            "mean": round(float(np.mean(e2e_times_ms)), 4),
            "median": round(float(np.median(e2e_times_ms)), 4),
            "std": round(float(np.std(e2e_times_ms, ddof=1)), 4),
            "min": round(float(np.min(e2e_times_ms)), 4),
            "p95": round(float(np.percentile(e2e_times_ms, 95)), 4),
            "p99": round(float(np.percentile(e2e_times_ms, 99)), 4),
        },
        "disclaimer": "CPU execution benchmark on varied test inputs; no speedup or optimization claims absent ECDR runtime comparisons.",
    }

    eval_results = {
        "supervised_test_denominator": len(sup_test_df),
        "seed_evaluations": seed_evals,
        "aggregate_metrics": aggregate_metrics,
        "margins": margin_quantiles,
        "adjacent_prediction_agreement": adjacent_agreement_summary,
        "cpu_latency": latency_summary,
    }

    eval_out_path = output_dir / "evaluation_results.json"
    eval_out_path.write_text(json.dumps(eval_results, indent=2))
    flush_print(f">>> [Phase: EVALUATE] Completed. Results saved to {eval_out_path}")
    return eval_results


# ==============================================================================
# PHASE: VERIFY
# ==============================================================================
def run_verify(output_dir: Path) -> Dict[str, Any]:
    flush_print(">>> [Phase: VERIFY] Running independent integrity checks and verification...")

    required_artifacts = [
        output_dir / "provenance_audit.json",
        output_dir / "normalization.json",
        output_dir / "normalization.npz",
        output_dir / "checkpoint_seed_42.pt",
        output_dir / "checkpoint_seed_43.pt",
        output_dir / "checkpoint_seed_44.pt",
        output_dir / "history_seed_42.json",
        output_dir / "history_seed_43.json",
        output_dir / "history_seed_44.json",
        output_dir / "selection.json",
        output_dir / "selected_best.pt",
        output_dir / "test_predictions.npz",
        output_dir / "test_predictions_manifest.csv",
        output_dir / "evaluation_results.json",
    ]

    for p in required_artifacts:
        if not p.exists():
            raise FileNotFoundError(f"Verification failure: required artifact {p.name} does not exist in {output_dir}")

    # 1. Verify model parameter count & shape
    best_ckpt = torch.load(output_dir / "selected_best.pt", map_location="cpu")
    model = SmallCNN1D(num_classes=12, cfg=best_ckpt["model_cfg"])
    model.load_state_dict(best_ckpt["state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    if param_count != EXPECTED_PARAMS:
        raise ValueError(f"Verification failure: model parameter count {param_count} != {EXPECTED_PARAMS}")

    # 2. Recompute train-only normalization and compare
    df_manifest = load_full_manifest_with_splits()
    x_all = load_raw_windows()

    mask_sup_train = (
        (df_manifest["split"] == "train")
        & df_manifest["label_policy"].isin(["pure", "majority"])
        & df_manifest["label_majority"].between(1, 12)
    )
    x_train_raw = x_all[df_manifest[mask_sup_train]["global_row_id"].to_numpy()]
    recomputed_mean, recomputed_std = fit_train_norm(x_train_raw)

    saved_norm = np.load(output_dir / "normalization.npz")
    if not np.allclose(recomputed_mean, saved_norm["mean"], atol=1e-6):
        raise ValueError("Verification failure: saved normalization mean disagrees with recomputation from train split.")
    if not np.allclose(recomputed_std, saved_norm["std"], atol=1e-6):
        raise ValueError("Verification failure: saved normalization std disagrees with recomputation from train split.")

    # 3. Verify prediction reproducibility on all 5053 test windows
    test_df = df_manifest[df_manifest["split"] == "test"]
    x_test_norm = apply_norm(x_all[test_df["global_row_id"].to_numpy()], saved_norm["mean"], saved_norm["std"])

    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(x_test_norm)).numpy()
        preds = logits.argmax(axis=1)

    saved_npz = np.load(output_dir / "test_predictions.npz")
    if not np.array_equal(preds, saved_npz["predictions"]):
        raise ValueError("Verification failure: recomputed predictions do not match saved predictions on test set.")
    if not np.allclose(logits, saved_npz["logits"], atol=1e-5):
        raise ValueError("Verification failure: recomputed logits do not match saved logits on test set.")

    # 4. Check selection artifact was generated before evaluation
    selection_meta = json.loads((output_dir / "selection.json").read_text())
    selected_hash = compute_sha256(output_dir / "selected_best.pt")
    if selection_meta["selected_best_sha256"] != selected_hash:
        raise ValueError("Verification failure: selected_best.pt hash does not match selection.json proof hash.")

    receipt = {
        "status": "PASS",
        "verified_artifacts_count": len(required_artifacts),
        "verified_model_parameters": param_count,
        "normalization_recomputation_match": True,
        "test_predictions_reproducibility_match": True,
        "selected_best_hash_match": True,
        "verification_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    receipt_path = output_dir / "verification_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    flush_print(f">>> [Phase: VERIFY] All verification checks passed. Receipt saved to {receipt_path}")
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
        help="Number of training epochs (default: 30, strictly 30 permitted for reference).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds (default: '42,43,44').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training/eval (default: 'cpu').",
    )

    args = parser.parse_args()

    # Numeric & parameter sanity validation
    if args.epochs <= 0:
        raise ValueError(f"Invalid epochs: {args.epochs}. Must be positive.")
    try:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    except ValueError:
        raise ValueError(f"Invalid seeds format: {args.seeds}. Expected comma-separated integers.")
    if len(seeds) == 0:
        raise ValueError("Seeds list cannot be empty.")

    # Guard: prevent overlap with legacy outputs or writing outside allowed directory
    out_dir = args.phase_dir = args.output_dir.resolve()
    legacy_dirs = [
        (ROOT / "checkpoints").resolve(),
        (ROOT / "configs").resolve(),
        (ROOT / "data").resolve(),
        (ROOT / "scripts").resolve(),
    ]
    for leg in legacy_dirs:
        if out_dir == leg or leg in out_dir.parents:
            raise ValueError(f"Unsafe output directory {out_dir}: overlaps with source/legacy path {leg}")

    # Check for existing artifacts in output_dir (fail if non-empty, except during verify or evaluate)
    if args.phase in ("audit", "train", "all") and out_dir.exists():
        existing_files = [f for f in out_dir.iterdir() if f.is_file()]
        if existing_files:
            raise FileExistsError(
                f"Output directory {out_dir} already contains {len(existing_files)} files. "
                "Failing safely to prevent accidental overwriting of reference run."
            )

    flush_print(f"Executing phase '{args.phase}' on target directory: {out_dir}")

    if args.phase == "audit":
        run_audit(out_dir)
    elif args.phase == "train":
        run_train(out_dir, epochs=args.epochs, seeds=seeds, device_str=args.device)
    elif args.phase == "evaluate":
        run_evaluate(out_dir, device_str=args.device)
    elif args.phase == "verify":
        run_verify(out_dir)
    elif args.phase == "all":
        run_audit(out_dir)
        run_train(out_dir, epochs=args.epochs, seeds=seeds, device_str=args.device)
        run_evaluate(out_dir, device_str=args.device)
        run_verify(out_dir)


if __name__ == "__main__":
    main()
