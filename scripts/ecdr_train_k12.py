#!/usr/bin/env python3
"""ecdr_train_k12.py — Training entrypoint for theoretical Conv1D-ReLU-GAP-Affine K=12 backbone.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 11.1)
and G2_CLASS_DECISION.md.

Contract obligations:
1. Architecture: ECDRConv1DNet (6 -> 32 -> 64 -> 64 -> GAP -> Affine 12).
2. Preserves legacy K=6 checkpoint and files; saves to checkpoints/ecdr_k12/.
3. Normalization is fit exclusively on train windows.
4. Exposes label policy (pure vs majority) with explicit HOLD status caveats for both policies.
   Ambiguous windows are strictly excluded.
5. Does not invent metrics; records exact loss, accuracy, and macro-F1.
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.data import load_supervised_dataset
from ecdr.model import ECDRConv1DNet
from ecdr.preprocessing import fit_normalization_stats
from ecdr.schema import HAPT_12_CLASSES

DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "ecdr_k12"
TRAINING_METRICS_FILE = ROOT / "data" / "ecdr_k12_training.json"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, num_classes: int = 12
) -> Tuple[float, float, float]:
    """Evaluate model on loader; returns (loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in loader:
            logits = model(x_b)
            loss = criterion(logits, y_b)
            total_loss += loss.item() * len(y_b)
            total_samples += len(y_b)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = float((all_preds == all_targets).mean()) if total_samples > 0 else 0.0

    # Macro-F1 across all classes with zero-division convention = 0.0
    f1s = []
    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return avg_loss, accuracy, macro_f1


def train(
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    label_policy: str = "majority",
    save_dir: Path = DEFAULT_CHECKPOINT_DIR,
    out_metrics: Path = TRAINING_METRICS_FILE,
) -> dict:
    if epochs <= 0:
        raise ValueError(f"epochs must be strictly positive (> 0), got {epochs}")
    if not (math.isfinite(lr) and lr > 0.0):
        raise ValueError(f"lr must be finite and strictly positive (> 0), got {lr}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be strictly positive (> 0), got {batch_size}")

    set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"ECDR K=12 TRAINING — Policy: {label_policy.upper()}")
    print(">> RESEARCH STATUS CAVEAT: K=12 training is on HOLD per G2_CLASS_DECISION.md.")
    if label_policy == "pure":
        print(">> Pure K=12 has 18 criteria blockers; Class 8 (SIT_TO_STAND) has only 5 train / 1 val windows.")
    else:
        print(">> Pure+Majority K=12 (excluding ambiguous) has 10 criteria blockers on postural transitions.")
    print(">> Proceeding in experimental research mode with support metadata tracked.")
    print("=" * 80)

    # 1. Load data
    x_tr_raw, y_tr, _ = load_supervised_dataset("train", label_policy=label_policy, num_classes=12)
    x_va_raw, y_va, _ = load_supervised_dataset("val", label_policy=label_policy, num_classes=12)
    x_te_raw, y_te, _ = load_supervised_dataset("test", label_policy=label_policy, num_classes=12)

    print(f"Loaded samples: Train={len(x_tr_raw)}, Val={len(x_va_raw)}, Test={len(x_te_raw)}")

    # 2. Fit normalization exclusively on train split
    prep = fit_normalization_stats(x_tr_raw)
    print(f"Fit Preprocessing Contract: prep_id={prep.prep_id}")
    print(f"  mean={np.round(prep.mean, 4)}")
    print(f"  std ={np.round(prep.std, 4)}")

    # Normalize inputs
    x_tr = prep.normalize(x_tr_raw).astype(np.float32)
    x_va = prep.normalize(x_va_raw).astype(np.float32)
    x_te = prep.normalize(x_te_raw).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te)),
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Model, Loss, Optimizer
    model = ECDRConv1DNet(num_classes=12)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_macro_f1 = -1.0
    best_checkpoint_path = save_dir / "ecdr_k12_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            total_samples += len(y_b)

        tr_loss = total_loss / max(1, total_samples)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, num_classes=12)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
            "val_macro_f1": float(va_f1),
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {va_loss:.4f} | Val Acc: {va_acc*100:5.2f}% | Val Macro-F1: {va_f1*100:5.2f}%"
        )

        if va_f1 > best_val_macro_f1:
            best_val_macro_f1 = va_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 12,
                    "label_policy": label_policy,
                    "research_status": "EXPERIMENTAL_HOLD",
                    "prep_mean": prep.mean,
                    "prep_std": prep.std,
                    "prep_id": prep.prep_id,
                    "best_val_macro_f1": float(best_val_macro_f1),
                    "best_val_acc": float(va_acc),
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )

    # Final evaluation of best checkpoint on test set
    best_ck = torch.load(best_checkpoint_path, weights_only=False)
    model.load_state_dict(best_ck["model_state_dict"])
    te_loss, te_acc, te_f1 = evaluate(model, test_loader, criterion, num_classes=12)
    print("=" * 80)
    print(
        f"BEST CHECKPOINT (Epoch {best_ck['epoch']}) EVALUATION ON TEST:\n"
        f"  Test Loss: {te_loss:.4f} | Test Acc: {te_acc*100:.2f}% | Test Macro-F1: {te_f1*100:.2f}%"
    )
    print(f"Saved best checkpoint to: {best_checkpoint_path}")

    summary = {
        "research_status": "EXPERIMENTAL_HOLD",
        "label_policy": label_policy,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "prep_id": prep.prep_id,
        "best_epoch": int(best_ck["epoch"]),
        "best_val_acc": float(best_ck["best_val_acc"]),
        "best_val_macro_f1": float(best_ck["best_val_macro_f1"]),
        "test_acc": float(te_acc),
        "test_macro_f1": float(te_f1),
        "history": history,
    }

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training history to: {out_metrics}")
    print("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train ECDR Conv1D-ReLU-GAP-Affine K=12 backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--label-policy",
        type=str,
        default="majority",
        choices=["pure", "majority"],
        help="Label policy (pure is on HOLD with 18 blockers; majority is pure+majority excluding ambiguous)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints (isolated from legacy checkpoints)",
    )
    parser.add_argument(
        "--out-metrics",
        type=Path,
        default=TRAINING_METRICS_FILE,
        help="Path to save output JSON training metrics",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        label_policy=args.label_policy,
        save_dir=args.save_dir,
        out_metrics=args.out_metrics,
    )


if __name__ == "__main__":
    main()
