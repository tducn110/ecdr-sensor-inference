#!/usr/bin/env python3
"""08_train_backbone.py — G2-B: train SmallCNN1D on pure K=6 windows, >=3 seeds,
best epoch selected on VALIDATION macro-F1, test touched only afterwards.

Normalizer fit on train windows only; stored inside each checkpoint.
Outputs: checkpoints/backbone_seed{S}.pt, checkpoints/backbone_best.pt,
data/g2_training.json. Rules: G2_CLASS_DECISION.md §6."""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone import (CLASS_NAMES, apply_norm, build_model, fit_norm,
                      load_pure_k6, load_yaml, metrics_from_logits)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
OUT = ROOT / "data" / "g2_training.json"


def seed_everything(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def val_scores(model, x, y, mean, std) -> tuple[float, float]:
    model.eval()
    logits = model(torch.from_numpy(apply_norm(x, mean, std))).numpy()
    m = metrics_from_logits(logits, y, CLASS_NAMES)
    return m["macro_f1"], m["accuracy"]


def train_one(seed: int, data, mean, std, tcfg: dict, bcfg: dict) -> dict:
    seed_everything(seed)
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xt, yt = data["test"]
    model = build_model(bcfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["optimizer"]["lr"]),
                           weight_decay=float(tcfg["optimizer"]["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(tcfg["epochs"]), eta_min=float(tcfg["optimizer"]["lr"]) / 100)
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(apply_norm(xtr, mean, std)),
                       torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=int(tcfg["batch_size"]), shuffle=True,
                    generator=g, num_workers=0)
    crit = nn.CrossEntropyLoss()

    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        vmf, vac = val_scores(model, xva, yva, mean, std)
        history.append({"epoch": epoch, "val_macro_f1": vmf, "val_acc": vac})
        if vmf > best["macro_f1"]:
            best = {"macro_f1": vmf, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        # early stopping on validation patience
        if epoch - best["epoch"] > int(tcfg["early_stopping"]["patience"]):
            break

    model.load_state_dict(best["state"])
    tmf, tac = val_scores(model, xt, yt, mean, std)  # test once, after selection
    ckpt = {"state_dict": best["state"], "mean": mean, "std": std,
            "seed": seed, "best_epoch": best["epoch"],
            "val_macro_f1": best["macro_f1"],
            "test_macro_f1": tmf, "test_acc": tac,
            "backbone_config": bcfg, "training_config": tcfg,
            "torch_version": torch.__version__}
    CKPT.mkdir(exist_ok=True)
    torch.save(ckpt, CKPT / f"backbone_seed{seed}.pt")
    return {"seed": seed, "best_epoch": best["epoch"],
            "val_macro_f1": best["macro_f1"], "test_macro_f1": tmf,
            "test_acc": tac, "epochs_run": history[-1]["epoch"],
            "history": history}


def main() -> None:
    bcfg = load_yaml("backbone.yaml")
    tcfg = load_yaml("training.yaml")
    t0 = time.time()
    data = load_pure_k6()
    mean, std = fit_norm(data["train"][0])
    runs = [train_one(int(s), data, mean, std, tcfg, bcfg)
            for s in tcfg["seeds"]]
    best = max(runs, key=lambda r: r["val_macro_f1"])
    src = CKPT / f"backbone_seed{best['seed']}.pt"
    torch.save(torch.load(src, weights_only=False), CKPT / "backbone_best.pt")

    out = {
        "policy": "pure_only K=6 (G2_CLASS_DECISION.md)",
        "window_counts": {s: int(len(data[s][1])) for s in ("train", "val", "test")},
        "class_names": CLASS_NAMES,
        "selection_metric": tcfg["selection_metric"],
        "seeds": tcfg["seeds"],
        "runs": runs,
        "best_seed": best["seed"],
        "best_val_macro_f1": best["val_macro_f1"],
        "norm_fit": "train windows only",
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
        "train_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("window_counts", "best_seed", "best_val_macro_f1",
                       "train_seconds")}, indent=2))
    for r in runs:
        print(f"seed {r['seed']}: val_mF1={r['val_macro_f1']:.4f} "
              f"test_mF1={r['test_macro_f1']:.4f} (best epoch {r['best_epoch']})")


if __name__ == "__main__":
    main()
