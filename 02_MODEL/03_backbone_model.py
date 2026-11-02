#!/usr/bin/env python3
"""backbone.py — shared G2-B helpers (imported by scripts 07-10).

Data: windows.npz + window_manifest.parquet + windows_with_split.parquet under
the LOCKED policy -> pure-only, labels 1..6 (K=6 per G2_CLASS_DECISION.md).
Normalization: per-channel mean/std fit on TRAIN windows only, reused for val/test.
Model: SmallCNN1D, architecture frozen in configs/backbone.yaml.
Metrics are implemented with numpy only (no sklearn in the venv).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG = ROOT / "configs"
CHECKPOINTS = ROOT / "checkpoints"

CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]
SPLITS = ("train", "val", "test")


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())


def load_pure_k6() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {split: (x (n,6,128) f32, y (n,) int64 in 0..5)} for pure K=6."""
    x_all = np.load(DATA / "windows.npz")["x"].astype(np.float32)
    man = pd.read_parquet(DATA / "window_manifest.parquet")
    spl = pd.read_parquet(DATA / "windows_with_split.parquet")
    df = man.merge(spl[["exp", "win_idx", "split"]], on=["exp", "win_idx"],
                   how="left", validate="one_to_one")
    mask = (df["label_policy"] == "pure") & df["label_majority"].between(1, 6)
    df = df[mask].reset_index()  # 'index' column = global row in windows.npz
    rows = df.pop("index").to_numpy()
    x = x_all[rows]
    y = df["label_majority"].to_numpy(dtype=np.int64) - 1
    sp = df["split"].to_numpy()
    return {s: (x[sp == s], y[sp == s]) for s in SPLITS}


def fit_norm(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std on TRAIN windows only (§6.1 of the decision file)."""
    mean = x_train.mean(axis=(0, 2)).astype(np.float32)
    std = np.clip(x_train.std(axis=(0, 2)), 1e-6, None).astype(np.float32)
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


class SmallCNN1D(nn.Module):
    """(B,6,128) -> (B,K). Conv(same pad) x3, MaxPool after conv1+conv2, MLP head."""

    def __init__(self, num_classes: int = 6, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or {}
        c = list(cfg.get("conv_channels", [32, 64, 64]))
        k = list(cfg.get("conv_kernels", [5, 3, 3]))
        hidden = int(cfg.get("fc_hidden", 128))
        act = nn.ReLU
        self.conv = nn.Sequential(
            nn.Conv1d(6, c[0], k[0], padding=k[0] // 2), nn.BatchNorm1d(c[0]), act(),
            nn.MaxPool1d(2),
            nn.Conv1d(c[0], c[1], k[1], padding=k[1] // 2), nn.BatchNorm1d(c[1]), act(),
            nn.MaxPool1d(2),
            nn.Conv1d(c[1], c[2], k[2], padding=k[2] // 2), nn.BatchNorm1d(c[2]), act(),
        )
        flat = c[2] * (128 // 4)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, hidden), act(), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


def build_model(backbone_cfg: dict) -> SmallCNN1D:
    return SmallCNN1D(num_classes=int(backbone_cfg["num_classes"]),
                      cfg=backbone_cfg.get("model", {}))


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, mean: np.ndarray,
                   std: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    xn = apply_norm(x, mean, std)
    out = [model(torch.from_numpy(xn[i:i + batch])).cpu().numpy()
           for i in range(0, len(xn), batch)]
    return np.concatenate(out) if out else np.zeros((0, model.head[-1].out_features))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray,
                        class_names: list[str]) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1 + support from a confusion matrix."""
    k = len(class_names)
    y_pred = logits.argmax(axis=1)
    cm = confusion(y_true, y_pred, k)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    pred_n = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(pred_n > 0, tp / pred_n, 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(tp.sum() / max(len(y_true), 1)), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "per_class": {
            cn: {"support": int(support[i]), "precision": round(float(precision[i]), 4),
                 "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4)}
            for i, cn in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
    }


def class_margins(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """g = f_y - max_{j != y} f_j (correct-class logit margin)."""
    onehot = np.eye(logits.shape[1], dtype=bool)[y]
    other = logits.copy()
    other[onehot] = -np.inf
    return logits[np.arange(len(y)), y] - other.max(axis=1)

