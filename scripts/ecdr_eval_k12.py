#!/usr/bin/env python3
"""ecdr_eval_k12.py — Evaluation entrypoint for ECDR Conv1D K=12 backbone.

Evaluates trained checkpoint on test or val split, reporting:
1. Accuracy and Macro-F1 across all 12 classes.
2. Per-class precision, recall, and F1.
3. Confusion matrix.
4. Empirical torch vs sequential float64 reference discrepancy: max |F_torch - F_np|.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.data import load_supervised_dataset
from ecdr.model import ECDRConv1DNet, OperationalSequentialNumpyCNN
from ecdr.preprocessing import PreprocessingContract
from ecdr.schema import HAPT_12_CLASSES

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "ecdr_k12" / "ecdr_k12_best.pt"
OUT_METRICS = ROOT / "data" / "ecdr_k12_eval_metrics.json"


def evaluate_checkpoint(
    checkpoint_path: Path, split: str = "test", out_metrics: Path = OUT_METRICS
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print(f"EVALUATING ECDR K=12 MODEL ON {split.upper()} SPLIT")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    ck = torch.load(checkpoint_path, weights_only=False)
    policy = ck.get("label_policy", "majority")
    model = ECDRConv1DNet(num_classes=12)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    prep = PreprocessingContract.create(
        mean=ck["prep_mean"],
        std=ck["prep_std"],
        num_channels=6,
        window_length=128,
    )
    print(f"Bound Preprocessing Contract: {prep.prep_id}")

    # Load data
    x_raw, y, df = load_supervised_dataset(split, label_policy=policy, num_classes=12)
    x_norm = prep.normalize(x_raw).astype(np.float32)
    n_samples = len(x_norm)
    print(f"Evaluation samples ({policy} policy): {n_samples}")

    # PyTorch evaluation
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y)),
        batch_size=64,
        shuffle=False,
    )

    total_loss = 0.0
    torch_preds = []
    torch_logits_all = []

    with torch.no_grad():
        for x_b, y_b in loader:
            logits = model(x_b)
            loss = criterion(logits, y_b)
            total_loss += loss.item() * len(y_b)
            torch_logits_all.append(logits.cpu().numpy())
            torch_preds.extend(logits.argmax(dim=-1).cpu().numpy())

    torch_logits_all = np.concatenate(torch_logits_all, axis=0)
    torch_preds = np.array(torch_preds)
    accuracy = float((torch_preds == y).mean())
    avg_loss = total_loss / max(1, n_samples)

    # Operational NumPy Sequential Reference Comparison
    numpy_net = OperationalSequentialNumpyCNN.from_torch_module(model)
    print(f"Constructed Operational Reference: model_id={numpy_net.model_id}")

    # Compare first 200 windows (or all if small) for torch vs numpy discrepancy
    sample_size = min(200, n_samples)
    discrepancies = []
    for i in range(sample_size):
        np_res = numpy_net.forward(x_norm[i].astype(np.float64))
        diff = np.max(np.abs(torch_logits_all[i] - np_res["logits"]))
        discrepancies.append(diff)
    max_discrepancy = float(np.max(discrepancies))
    mean_discrepancy = float(np.mean(discrepancies))

    # Confusion matrix and per-class metrics
    conf_mat = np.zeros((12, 12), dtype=np.int64)
    for true_cls, pred_cls in zip(y, torch_preds):
        conf_mat[true_cls, pred_cls] += 1

    per_class = {}
    f1_list = []
    for c in range(12):
        name = HAPT_12_CLASSES[c + 1]
        tp = int(conf_mat[c, c])
        fp = int(conf_mat[:, c].sum() - tp)
        fn = int(conf_mat[c, :].sum() - tp)
        support = int(conf_mat[c, :].sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        per_class[name] = {
            "class_id": c + 1,
            "support": support,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    macro_f1 = float(np.mean(f1_list))

    print("-" * 80)
    print(f"RESULTS on {split.upper()}:")
    print(f"  Accuracy: {accuracy*100:6.2f}% | Macro-F1: {macro_f1*100:6.2f}% | Loss: {avg_loss:.4f}")
    print(f"  Max Torch vs Sequential NumPy Logit Discrepancy: {max_discrepancy:.6e}")
    print(f"  Mean Discrepancy: {mean_discrepancy:.6e}")
    print("-" * 80)
    print(f"{'Class':<22} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    for name, m in per_class.items():
        print(
            f"{name:<22} {m['support']:<10} {m['precision']*100:>8.2f}% "
            f"{m['recall']*100:>10.2f}% {m['f1']*100:>10.2f}%"
        )
    print("=" * 80)

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "label_policy": policy,
        "n_samples": n_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_loss": avg_loss,
        "torch_vs_numpy_discrepancy": {
            "max": max_discrepancy,
            "mean": mean_discrepancy,
            "samples_compared": sample_size,
        },
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
    }

    with open(out_metrics, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation artifact to: {out_metrics}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECDR Conv1D K=12 backbone")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-json", type=Path, default=OUT_METRICS, help="Path to save output JSON metrics")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, split=args.split, out_metrics=args.out_json)


if __name__ == "__main__":
    main()
