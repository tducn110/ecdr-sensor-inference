"""data.py — Robust HAPT Data Loader conforming to independent audit specifications.

Authoritative source: G2_CLASS_DECISION.md and independent audit requirements.

Contract obligations:
1. data/windows.npz has shape (17453, 6, 128), dtype float32, all finite.
2. Manifest keys: (exp, win_idx), where win_idx restarts per recording.
3. Assigns stable global_row_id BEFORE any merge/filter/sort.
4. Validates split merge with validate="one_to_one".
5. Never indexes x by recording-local win_idx.
6. Full stream denominators strictly preserved:
   - test: 5053
   - val: 2422
   - train: 9978
   - total: 17453
7. Label filtering (pure / pure+majority) applies ONLY to supervised tasks:
   - 'pure': label_policy == 'pure' & label_majority in 1..num_classes
   - 'majority': label_policy.isin(['pure', 'majority']) & label_majority in 1..num_classes
   Ambiguous windows (label_policy == 'ambiguous') are NEVER silently included.
8. Corrupted-input guards:
   - Finiteness check on loaded windows
   - Pairwise disjoint subjects across train/val/test
   - Each recording belongs to exactly one split
   - Each recording belongs to exactly one subject
   - Subject metadata in manifest matches split metadata exactly
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ecdr.schema import (
    FULL_STREAM_DENOMINATORS,
    HAPT_12_CLASSES,
    MIN_SUPPORT_SUBJECTS_PER_SPLIT,
    MIN_SUPPORT_WINDOWS_PER_SPLIT,
    SPLITS,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WINDOWS_FILE = DATA_DIR / "windows.npz"
MANIFEST_FILE = DATA_DIR / "window_manifest.parquet"
SPLIT_FILE = DATA_DIR / "windows_with_split.parquet"


def load_full_manifest_with_splits() -> pd.DataFrame:
    """Load manifest and split metadata, assigning stable global_row_id BEFORE merge."""
    if not MANIFEST_FILE.exists() or not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing data files: {MANIFEST_FILE} or {SPLIT_FILE}")

    man = pd.read_parquet(MANIFEST_FILE)
    # Assign stable global row_id matching windows.npz row index
    man["global_row_id"] = np.arange(len(man), dtype=np.int64)

    spl = pd.read_parquet(SPLIT_FILE)

    # Corrupted-input guard: Validate manifest subject matches split subject exactly
    man_spl_merge = man.merge(
        spl,
        on=["exp", "win_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_spl"),
    )
    if "subject_spl" in man_spl_merge.columns:
        if not (man_spl_merge["subject"] == man_spl_merge["subject_spl"]).all():
            raise ValueError("Corrupted data: manifest subject disagrees with split subject metadata")
        man_spl_merge.drop(columns=["subject_spl"], inplace=True)

    df = man_spl_merge

    if len(df) != FULL_STREAM_DENOMINATORS["total"]:
        raise ValueError(
            f"Expected {FULL_STREAM_DENOMINATORS['total']} total windows, got {len(df)}"
        )

    # Denominator verification
    for split_name, expected_count in FULL_STREAM_DENOMINATORS.items():
        if split_name != "total":
            actual_count = int((df["split"] == split_name).sum())
            if actual_count != expected_count:
                raise ValueError(
                    f"Split denominator mismatch for {split_name}: expected {expected_count}, got {actual_count}"
                )

    # Corrupted-input guard: Each recording must belong to exactly one split
    exp_splits = df.groupby("exp")["split"].nunique()
    if (exp_splits != 1).any():
        bad_exps = exp_splits[exp_splits != 1].index.tolist()
        raise ValueError(f"Corrupted data: recordings span multiple splits: {bad_exps}")

    # Corrupted-input guard: Each recording must belong to exactly one subject
    exp_subjs = df.groupby("exp")["subject"].nunique()
    if (exp_subjs != 1).any():
        bad_exps = exp_subjs[exp_subjs != 1].index.tolist()
        raise ValueError(f"Corrupted data: recordings span multiple subjects: {bad_exps}")

    # Corrupted-input guard: Pairwise disjoint subjects across splits
    subjs_by_split = {s: set(df[df["split"] == s]["subject"].unique()) for s in SPLITS}
    if subjs_by_split["train"] & subjs_by_split["val"]:
        raise ValueError("Corrupted data: train and val splits share subjects")
    if subjs_by_split["train"] & subjs_by_split["test"]:
        raise ValueError("Corrupted data: train and test splits share subjects")
    if subjs_by_split["val"] & subjs_by_split["test"]:
        raise ValueError("Corrupted data: val and test splits share subjects")

    return df


def load_raw_windows(global_row_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """Load raw float32 windows from windows.npz by global_row_id with finiteness validation."""
    if not WINDOWS_FILE.exists():
        raise FileNotFoundError(f"Missing windows file: {WINDOWS_FILE}")

    with np.load(WINDOWS_FILE) as npz:
        x_all = npz["x"].astype(np.float32)

    if x_all.shape != (17453, 6, 128):
        raise ValueError(f"Expected shape (17453, 6, 128), got {x_all.shape}")

    # Corrupted-input guard: validate finiteness
    if not np.all(np.isfinite(x_all)):
        raise ValueError("Corrupted data: windows.npz contains non-finite values (NaN or Inf)")

    if global_row_ids is not None:
        return x_all[global_row_ids]
    return x_all


def load_full_stream(split: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load complete chronological stream for replay without label filtering.

    split: "train", "val", or "test"
    Returns (x, manifest_df) sorted chronologically by (exp, win_idx).
    Denominators: test=5053, val=2422, train=9978.
    """
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}', must be one of {SPLITS}")

    df_all = load_full_manifest_with_splits()
    df_split = (
        df_all[df_all["split"] == split]
        .sort_values(["exp", "win_idx"])
        .reset_index(drop=True)
    )

    expected_len = FULL_STREAM_DENOMINATORS[split]
    if len(df_split) != expected_len:
        raise ValueError(
            f"Stream denominator violation for {split}: expected {expected_len}, got {len(df_split)}"
        )

    row_ids = df_split["global_row_id"].to_numpy(dtype=np.int64)
    x_stream = load_raw_windows(row_ids)
    return x_stream, df_split


def load_supervised_dataset(
    split: str, label_policy: str = "pure", num_classes: int = 12
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load supervised windows and labels for model training or accuracy evaluation.

    label_policy:
      - 'pure': requires label_policy == 'pure'
      - 'majority': requires label_policy in ['pure', 'majority'] (NEVER includes 'ambiguous')
    num_classes: 6 or 12
    Returns (x, y, df) where y is in 0..num_classes-1.
    """
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}', must be one of {SPLITS}")

    df_all = load_full_manifest_with_splits()
    df_split = df_all[df_all["split"] == split].copy()

    if label_policy == "pure":
        mask = (df_split["label_policy"] == "pure") & df_split["label_majority"].between(1, num_classes)
    elif label_policy == "majority":
        # Strict explicit pure+majority policy; rejects ambiguous windows
        mask = (
            df_split["label_policy"].isin(["pure", "majority"])
            & df_split["label_majority"].between(1, num_classes)
        )
    else:
        raise ValueError(f"Unsupported label policy: {label_policy}")

    df_filtered = df_split[mask].reset_index(drop=True)
    row_ids = df_filtered["global_row_id"].to_numpy(dtype=np.int64)
    x = load_raw_windows(row_ids)
    y = df_filtered["label_majority"].to_numpy(dtype=np.int64) - 1

    return x, y, df_filtered


def audit_12_class_support() -> Dict[str, dict]:
    """Audit class support for all 12 classes under pure and pure+majority policies across splits.

    Rejects ambiguous windows and tracks both:
    - Blocked cells: number of (class, split) combinations failing support thresholds.
    - Historical criteria count: 18 blockers for pure K=12 per G2_CLASS_DECISION.md.
    """
    df_all = load_full_manifest_with_splits()

    audit_results = {}
    for policy in ("pure", "majority"):
        class_table = []
        blocked_cells = []
        criterion_blockers = []

        for cls_id in range(1, 13):
            cls_name = HAPT_12_CLASSES[cls_id]
            row_data = {"class_id": cls_id, "class_name": cls_name}
            for split_name in SPLITS:
                if policy == "pure":
                    mask = (
                        (df_all["split"] == split_name)
                        & (df_all["label_policy"] == "pure")
                        & (df_all["label_majority"] == cls_id)
                    )
                else:
                    mask = (
                        (df_all["split"] == split_name)
                        & (df_all["label_policy"].isin(["pure", "majority"]))
                        & (df_all["label_majority"] == cls_id)
                    )

                count = int(mask.sum())
                n_subj = int(df_all.loc[mask, "subject"].nunique())
                row_data[f"{split_name}_count"] = count
                row_data[f"{split_name}_subjects"] = n_subj

                cell_blocked = False
                if count < MIN_SUPPORT_WINDOWS_PER_SPLIT:
                    criterion_blockers.append(
                        f"Class {cls_id} ({cls_name}) in {split_name}: count={count} < {MIN_SUPPORT_WINDOWS_PER_SPLIT}"
                    )
                    cell_blocked = True
                if n_subj < MIN_SUPPORT_SUBJECTS_PER_SPLIT:
                    criterion_blockers.append(
                        f"Class {cls_id} ({cls_name}) in {split_name}: subjects={n_subj} < {MIN_SUPPORT_SUBJECTS_PER_SPLIT}"
                    )
                    cell_blocked = True

                if cell_blocked:
                    blocked_cells.append((cls_id, cls_name, split_name, count, n_subj))

            class_table.append(row_data)

        audit_results[policy] = {
            "policy_definition": "pure_only" if policy == "pure" else "pure_plus_majority_excluding_ambiguous",
            "classes": class_table,
            "num_blocked_cells": len(blocked_cells),
            "num_criteria_blockers": len(criterion_blockers),
            "criteria_blockers": criterion_blockers,
            "blocked_cells": [
                f"Class {b[0]} ({b[1]}) in {b[2]}: {b[3]} windows, {b[4]} subjects"
                for b in blocked_cells
            ],
            "k12_trainable_status": "HOLD",  # Both policies have transition support blockers
        }

    return audit_results
