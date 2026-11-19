"""Compare consecutive HAPT windows on complete per-stride grids.

Run from the repository root with .venv/bin/python. Write an exploratory
diagnostic receipt alongside this script.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ecdr.week4.reference import ReferenceOwner
from ecdr.week3_verifier import interval_layer

STRIDES = (1, 4, 16, 32, 64)
RADII = (1e-6, 1e-4, 0.01, 0.1)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def minimum_ibp_margins(model, anchors, targets, radius):
    """Direct all-competitor IBP estimates in ordinary float64."""
    batches = []
    with torch.inference_mode():
        for i in range(0, len(anchors), 256):
            a = torch.from_numpy(anchors[i:i+256]).double()
            c = torch.from_numpy(targets[i:i+256])
            lo, hi = a - radius, a + radius
            for layer in list(model.conv) + list(model.head)[:-1]:
                lo, hi = interval_layer(layer, lo, hi)
            head = model.head[-1]
            w = head.weight[c, None, :] - head.weight[None, :, :]
            b = head.bias[c, None] - head.bias[None, :]
            lower = (w.clamp(min=0) * lo[:, None]).sum(2)
            lower += (w.clamp(max=0) * hi[:, None]).sum(2) + b
            lower[torch.arange(len(c)), c] = float("inf")
            minimum = lower.min(1).values.numpy()
            if not np.isfinite(minimum).all():
                raise ValueError("Nonfinite IBP margins")
            batches.append(minimum)
    return np.concatenate(batches) if batches else np.empty(0)


def main():
    torch.set_num_threads(1)
    reference = ReferenceOwner.load()
    with np.load(ROOT / "output/week1_closeout_20261102/reference/normalization.npz") as f:
        mean, std = f["mean"][:, None], f["std"][:, None]
    manifest_path = ROOT / "data/window_manifest.parquet"
    split_path = ROOT / "data/windows_with_split.parquet"
    windows_path = ROOT / "data/windows.npz"
    manifest = pd.read_parquet(manifest_path)
    manifest["split"] = pd.read_parquet(split_path)["split"]
    saved = np.load(windows_path)["x"]
    raw = ROOT / "data/hapt/RawData"
    distances = {s: [] for s in STRIDES}
    agreements = {s: [] for s in STRIDES}
    norm_l2 = {s: [] for s in STRIDES}
    hashes = {str(p.relative_to(ROOT)): digest(p)
              for p in (manifest_path, split_path, windows_path, Path(__file__).resolve())}
    bound_model = deepcopy(reference._model).double().eval()
    ibp_minima = []
    ibp_agree = []
    recordings = 0
    for acc in sorted(raw.glob("acc_exp*_user*.txt")):
        exp = int(re.search(r"exp(\d+)", acc.name)[1])
        rows = manifest[(manifest.exp == exp) & (manifest.split == "test")].sort_values("win_idx")
        if rows.empty:
            continue
        gyro = acc.with_name("gyro" + acc.name[3:])
        for p in (acc, gyro):
            hashes[str(p.relative_to(ROOT))] = digest(p)
        signal = np.column_stack([np.loadtxt(p, dtype=np.float32) for p in (acc, gyro)]).T
        view = np.lib.stride_tricks.sliding_window_view(signal, 128, axis=1).transpose(1, 0, 2)
        if not np.array_equal(view[::64], saved[rows.index]):
            raise ValueError(f"Canonical grid mismatch for recording {exp}")
        z = np.ascontiguousarray((view - mean) / std)
        with torch.inference_mode():
            preds = np.concatenate([
                reference._model(torch.from_numpy(z[i:i+256])).argmax(1).numpy()
                for i in range(0, len(z), 256)
            ])
        recordings += 1
        for s in STRIDES:
            zs = z[::s].astype(np.float64)
            diff = zs[1:] - zs[:-1]
            distances[s].append(np.max(np.abs(diff), axis=(1, 2)))
            norm_l2[s].append(np.sqrt(np.sum(diff * diff, axis=(1, 2))))
            if s == 1:
                members = np.flatnonzero(distances[s][-1] <= 0.1)
                ibp_minima.append(minimum_ibp_margins(bound_model, z[members], preds[members], 0.1))
                ibp_agree.append(preds[members] == preds[members + 1])
            ps = preds[::s]
            agreements[s].append(ps[1:] == ps[:-1])
        print(f"Recording {exp}: {len(z)} stride-one windows", flush=True)
    results = {}
    for s in STRIDES:
        d = np.concatenate(distances[s])
        l2 = np.concatenate(norm_l2[s])
        agree = np.concatenate(agreements[s])
        results[s] = {
            "within_recording_pairs": int(len(d)),
            "median_linf": float(np.median(d)),
            "median_l2": float(np.median(l2)),
            "reference_top1_agree": int(agree.sum()),
            "box_members": {str(r): int((d <= r).sum()) for r in RADII},
            "reference_agreement_among_box_members": {
                str(r): int((agree & (d <= r)).sum()) for r in RADII
            },
        }
    if recordings != 19 or results[64]["within_recording_pairs"] != 5034:
        raise ValueError("Unexpected recording or pair denominator")
    minima = np.concatenate(ibp_minima)
    ibp_agreements = np.concatenate(ibp_agree)
    if len(minima) != results[1]["box_members"]["0.1"]:
        raise ValueError("IBP did not cover every candidate member")
    results[1]["ibp_radius_0.1"] = {
        "conditional_members_tested": len(minima),
        "positive_all_11": int((minima > 0).sum()),
        "shadow_disagreements_among_positives": int(((minima > 0) & ~ibp_agreements).sum()),
        "largest_minimum_margin_estimate": float(minima.max()),
    }
    receipt = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "checkpoint_sha256": reference.identity.checkpoint_sha256,
        "preprocessing_id": reference.identity.preprocessing_id,
        "test_recordings": recordings,
        "canonical_grid_value_match": True,
        "scope": "Exploratory pairwise geometry and batch-computed top-1 agreement. Ordinary float64 IBP for stride-one box members at radius 0.1. No validated float32 enclosure, adaptive anchor, runtime latency, or formal reuse measurement.",
        "input_hashes": hashes,
        "results": results,
    }
    out = Path(__file__).with_name("stride_audit.json")
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
