#!/usr/bin/env python3
"""07_count_params.py — G2 CHECK: parameter count + shape test (B,6,128)->(B,K).
Writes data/g2_model_summary.json. No training."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone import build_model, load_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "g2_model_summary.json"


def main() -> None:
    cfg = load_yaml("backbone.yaml")
    model = build_model(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    b = 4
    with torch.no_grad():
        y = model(torch.zeros(b, cfg["input"]["channels"], cfg["input"]["samples"]))
    shape_ok = tuple(y.shape) == (b, int(cfg["num_classes"]))
    if not shape_ok:
        print(f"SHAPE TEST FAILED: {tuple(y.shape)} != {(b, cfg['num_classes'])}")
        sys.exit(1)

    out = {
        "model": cfg["model"]["name"],
        "num_classes": int(cfg["num_classes"]),
        "input_shape": [b, cfg["input"]["channels"], cfg["input"]["samples"]],
        "output_shape": list(y.shape),
        "shape_test": "PASS (B,6,128)->(B,6)",
        "params_total": int(total),
        "params_trainable": int(trainable),
        "torch_version": torch.__version__,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
