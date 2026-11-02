#!/usr/bin/env python3
"""ecdr_audit_support.py — Explicit 12-class label policy and support audit.

Authoritative source: G2_CLASS_DECISION.md and independent audit requirements.

Audits class support for all 12 HAPT classes across train, val, and test splits
under both pure and pure+majority policies. Confirms the 18 criteria blockers
(17 blocked cells) for pure K=12 and class 8 support (train=5, val=1, test=6),
exposing the HOLD research status for both policies on 12 classes.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecdr.data import audit_12_class_support
from ecdr.schema import HAPT_12_CLASSES

OUT_JSON = ROOT / "data" / "ecdr_12_class_support.json"


def run_audit(out_json: Path = OUT_JSON) -> dict:
    print("=" * 80)
    print("ECDR HAPT 12-CLASS SUPPORT AUDIT")
    print("=" * 80)

    results = audit_12_class_support()

    for policy in ("pure", "majority"):
        pol_name = "PURE" if policy == "pure" else "PURE + MAJORITY (Excluding Ambiguous)"
        print(f"\nPolicy: {pol_name}")
        print("-" * 80)
        print(f"{'ID':<4} {'Class Name':<22} {'Train (N/Subj)':<18} {'Val (N/Subj)':<18} {'Test (N/Subj)':<18}")
        print("-" * 80)
        for row in results[policy]["classes"]:
            c_id = row["class_id"]
            c_name = row["class_name"]
            tr_str = f"{row['train_count']:>4} ({row['train_subjects']:>2} subj)"
            va_str = f"{row['val_count']:>4} ({row['val_subjects']:>2} subj)"
            te_str = f"{row['test_count']:>4} ({row['test_subjects']:>2} subj)"
            print(f"{c_id:<4} {c_name:<22} {tr_str:<18} {va_str:<18} {te_str:<18}")

        print("-" * 80)
        status = results[policy]["k12_trainable_status"]
        n_crit = results[policy]["num_criteria_blockers"]
        n_cells = results[policy]["num_blocked_cells"]
        print(
            f"Status: {status} | Blocked Cells: {n_cells} | Total Criteria Blockers: {n_crit}"
        )
        print("Criteria Blockers detail:")
        for b in results[policy]["criteria_blockers"]:
            print(f"  * {b}")

    # Check class 8 specifically
    pure_c8 = [r for r in results["pure"]["classes"] if r["class_id"] == 8][0]
    c8_tr, c8_va, c8_te = pure_c8["train_count"], pure_c8["val_count"], pure_c8["test_count"]
    print("\n" + "=" * 80)
    print(f"Class 8 (SIT_TO_STAND) pure counts: train={c8_tr}, val={c8_va}, test={c8_te}")
    if (c8_tr, c8_va, c8_te) == (5, 1, 6):
        print(">> Verified: Matches exact historical audit (train=5, val=1, test=6).")
    else:
        print(f">> Notice: Class 8 counts differ from (5, 1, 6): ({c8_tr}, {c8_va}, {c8_te})")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved 12-class support audit artifact to: {out_json}")
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audit 12-class support and label policies on HAPT splits"
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=OUT_JSON,
        help="Path to save output JSON audit report",
    )
    args = parser.parse_args()
    run_audit(out_json=args.out_json)


if __name__ == "__main__":
    main()
