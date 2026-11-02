#!/usr/bin/env python3
"""09_evaluate_backbone.py — G2 metrics on the FROZEN checkpoint: accuracy,
macro-F1, per-class recall/precision, confusion matrix, logit-margin distribution.
Recomputes from checkpoints (independent of 08's in-training numbers).
Outputs: data/g2_metrics.json, plots/g2_confusion_matrix.png,
plots/g2_margin_distribution.png. Denominators: G2_CLASS_DECISION.md §4."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone import (CLASS_NAMES, CHECKPOINTS, class_margins, load_pure_k6,
                      load_yaml, metrics_from_logits, predict_logits,
                      build_model)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "g2_metrics.json"
PLOTS = ROOT / "plots"


def eval_ckpt(path: Path, data: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    ck = torch.load(path, weights_only=False)
    model = build_model(ck["backbone_config"])
    model.load_state_dict(ck["state_dict"])
    res = {"seed": ck["seed"], "best_epoch": ck["best_epoch"]}
    logits_all = {}
    for split in ("val", "test"):
        x, y = data[split]
        lg = predict_logits(model, x, ck["mean"], ck["std"])
        logits_all[split] = (lg, y)
        res[split] = metrics_from_logits(lg, y, CLASS_NAMES)
    return res, logits_all["test"][0], logits_all["test"][1]


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    cfg = load_yaml("backbone.yaml")
    data = load_pure_k6()
    support = json.load(open(ROOT / "data" / "g2_class_support.json"))
    stream = json.load(open(ROOT / "data" / "build_windows_summary.json"))

    per_seed, margins_by_seed, test_logits = [], {}, None
    for s in (42, 43, 44):
        r, lg, y = eval_ckpt(CHECKPOINTS / f"backbone_seed{s}.pt", data)
        per_seed.append(r)
        margins_by_seed[s] = class_margins(lg, y)
        if r["seed"] == json.load(open(ROOT / "data" / "g2_training.json"))["best_seed"]:
            test_logits, test_y, best = lg, y, r

    tmf = [r["test"]["macro_f1"] for r in per_seed]
    tacc = [r["test"]["accuracy"] for r in per_seed]

    # confusion matrix plot (best seed)
    cm = np.array(best["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6), CLASS_NAMES, fontsize=8)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                    fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"test confusion (best seed {best['seed']})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(PLOTS / "g2_confusion_matrix.png", dpi=120); plt.close(fig)

    # margin distribution (best seed, test): g = f_y - max_{j!=y} f_j
    margins = class_margins(test_logits, test_y)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(margins, bins=60, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", ls="--", label="margin=0 (decision boundary)")
    ax.set_xlabel("correct-class logit margin (test, best seed)")
    ax.set_ylabel("count"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "g2_margin_distribution.png", dpi=120)
    plt.close(fig)

    out = {
        "gate": "G2",
        "scope": {
            "statement": "6-class HAPT classifier, pure-only policy",
            "stream_windows": stream["windows"],
            "labeled_windows": support["labeled_total"],
            "trainable_pure_k6": support["pure_total"] - sum(
                support["per_class"][str(c)]["pure_total"] for c in range(7, 13)),
            "test_evaluated_pure_k6": best["test"]["n"],
            "transition_windows_excluded": 1111,
            "denominator_note": ("metrics below use test pure K=6 only; "
                                 "full-stream/labeled denominators reported for scope"),
        },
        "per_seed": per_seed,
        "test_macro_f1_mean": round(float(np.mean(tmf)), 4),
        "test_macro_f1_std": round(float(np.std(tmf)), 4),
        "test_acc_mean": round(float(np.mean(tacc)), 4),
        "test_acc_std": round(float(np.std(tacc)), 4),
        "best_seed": best["seed"],
        "best_val": {"accuracy": best["val"]["accuracy"], "macro_f1": best["val"]["macro_f1"]},
        "best_test": {"accuracy": best["test"]["accuracy"], "macro_f1": best["test"]["macro_f1"],
                      "per_class": best["test"]["per_class"]},
        "margin_stats_test_best": {
            "p1": round(float(np.percentile(margins, 1)), 4),
            "p5": round(float(np.percentile(margins, 5)), 4),
            "p50": round(float(np.percentile(margins, 50)), 4),
            "frac_negative": round(float((margins < 0).mean()), 4),
        },
        "versions": {"torch": torch.__version__},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("scope", "test_macro_f1_mean",
                                          "test_macro_f1_std", "best_seed",
                                          "best_test", "margin_stats_test_best")},
                     indent=2))


if __name__ == "__main__":
    main()
