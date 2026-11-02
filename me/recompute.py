#!/usr/bin/env python3
"""Recompute the report's arithmetic and audit the existing window arrays.

Run from any directory with the project's NumPy/Pandas environment:
    /home/pro/Downloads/research/.venv/bin/python /home/pro/Downloads/research/me/recompute.py

Writes only me/calculations.json. No training, checkpoint inference, verifier,
latency benchmark, raw-data mutation, or ECDR implementation is performed.
Theoretical examples, current read-only data checks, and archived observations
are separate top-level records. Exact metrics below are recalculated from the
archived integer confusion matrices, not fresh checkpoint predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = Path(__file__).resolve().with_name("calculations.json")


def read_json(name):
    return json.loads((DATA / name).read_text())


def rational(n, d):
    value = Fraction(int(n), int(d))
    return {"numerator": int(n), "denominator": int(d),
            "fraction_reduced": str(value), "value": float(value)}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_std(values):
    return {"values": values, "n": len(values), "mean": statistics.mean(values),
            "population_std_ddof0": statistics.pstdev(values),
            "sample_std_ddof1": statistics.stdev(values)}


def theoretical_calculations():
    channels, width, stride, hz = 6, 128, 64, 50
    cin, length, rf = channels, width, 1
    layers = []
    for i, (cout, kernel) in enumerate(zip([32, 64, 64], [5, 3, 3]), 1):
        length -= kernel - 1
        rf += kernel - 1
        layers.append({"layer": f"conv{i}_valid_relu", "input_channels": cin,
                       "output_channels": cout, "kernel": kernel,
                       "output_length": length,
                       "weight_params": cin * cout * kernel,
                       "bias_params": cout,
                       "parameters": cout * (cin * kernel + 1),
                       "macs": length * cout * cin * kernel,
                       "relu_elements": length * cout,
                       "receptive_field": rf})
        cin = cout
    layers.append({"layer": "GAP", "output_channels": cin,
                   "output_length": 1, "parameters": 0,
                   "additions": cin * (length - 1), "scales": cin})
    layers.append({"layer": "affine_head_K12", "input_channels": cin,
                   "output_channels": 12, "weight_params": cin * 12,
                   "bias_params": 12, "parameters": 12 * (cin + 1),
                   "macs": cin * 12})
    total_macs = sum(layer.get("macs", 0) for layer in layers)
    margin_center = Fraction(1)
    square_halfwidth = Fraction(8, 10)
    tube_halfwidth = Fraction(1, 10)
    tail_q, tail_drop = [2, -1], [1, 2]
    tail_lower, tail_upper = [0, 1], [2, 3]
    tail_n, tail_old = 10, Fraction(4, 10)
    tail_drop_contribution = sum(q * v for q, v in zip(tail_q, tail_drop))
    tail_min = sum(q * (lo if q >= 0 else hi)
                   for q, lo, hi in zip(tail_q, tail_lower, tail_upper))
    tail_valid = tail_old + Fraction(tail_min - tail_drop_contribution, tail_n)
    tail_wrong = tail_old + Fraction(sum(q * lo for q, lo in zip(tail_q, tail_lower)) - tail_drop_contribution, tail_n)
    sensitivity = []
    for build in (1.5, 8.0):
        rows = []
        for hit in (0.50, 0.80, 0.95):
            cost = 0.06 + 0.02 + (1 - hit) * 4 + (1 - hit) * build
            rows.append({"hit_rate": hit, "builds_per_query": 1 - hit,
                         "mean_cost_ms": cost, "speed_ratio": 4 / cost})
        sensitivity.append({"build_ms": build,
                            "strict_hit_rate_threshold": (0.06 + 0.02 + build) / (4 + build),
                            "strict_all_hits_amortization_L_threshold": build / (4 - 0.08),
                            "minimum_integer_all_hits": math.floor(build / (4 - 0.08)) + 1,
                            "rows": rows})
    return {
        "status": "Derived arithmetic for hypothetical frozen valid-Conv/ReLU/GAP architecture; no trained K12 model is asserted.",
        "window": {"channels": channels, "width_samples": width,
                   "dimension": channels * width, "stride_samples": stride,
                   "sampling_hz": hz, "window_duration_convention_s": width / hz,
                   "first_to_last_sample_elapsed_s": (width - 1) / hz,
                   "query_interval_s": stride / hz, "query_rate_hz": hz / stride,
                   "raw_overlap_samples_per_channel": width - stride,
                   "overlap_coordinates": channels * (width - stride),
                   "new_coordinates": channels * stride,
                   "raw_overlap_fraction": rational(width - stride, width),
                   "q_shifts": [{"q": q, "new_since_anchor_per_channel": min(q * stride, width),
                                  "raw_shared_per_channel": max(width - q * stride, 0),
                                  "formula_without_truncation_requires_qs_le_W": q * stride <= width}
                                 for q in range(5)]},
        "marginal_vs_joint_coverage": {
            "status": "Illustrative only, conditional on independent coordinate events with equal marginal probability q; not a HAPT independence assumption or coverage estimate.",
            "union_bound": "P(intersection E_i) >= max(0, 1-sum alpha_i), if P(not E_i)<=alpha_i",
            "examples": [{"q": q, "q_to_d768": q ** 768, "q_to_newwave_d384": q ** 384}
                         for q in (0.9, 0.99, 0.999)]},
        "cnn": {"assumptions": ["valid padding", "stride 1", "dilation 1", "fixed weights and biases", "ReLU", "no BatchNorm", "no intermediate pooling", "global average pooling", "affine K12 head"],
                "layers": layers,
                "parameters_total": sum(layer["parameters"] for layer in layers),
                "conv_and_head_macs": total_macs,
                "approx_conv_and_head_flops_at_2_per_mac": 2 * total_macs,
                "flop_note": "Excludes bias addition, ReLU comparisons, GAP operations, indexing and memory traffic.",
                "relu_elements": sum(layer.get("relu_elements", 0) for layer in layers),
                "final_receptive_field": rf, "final_positions": length,
                "shared_final_positions_one_shift": max(length - stride, 0),
                "shared_final_fraction_one_shift": rational(max(length - stride, 0), length),
                "tail_input_samples_per_channel": stride + rf - 1,
                "final_shared_by_q": [{"q": q, "positions": max(length - q * stride, 0)} for q in range(5)]},
        "break_even": {"equation": "r*t_f > t_v+t_m+b*t_b",
                       "if_build_each_miss": "b=1-r, r>(t_v+t_m+t_b)/(t_f+t_b)",
                       "illustrative_only": {"tf_ms": 4.0, "tv_ms": 0.06, "tm_ms": 0.02,
                                             "build_each_miss": sensitivity},
                       "legacy_timing_sensitivity_only": {"archived_tf_us": 251.1,
                           "assumed_hit_rate": 0.5, "strict_overhead_budget_us": 125.55,
                           "warning": "A recorded forward latency of a different K6 model is used only as a scalar example; no new runtime or K12 baseline claim."}},
        "structured_margin_example": {
            "margin": "1+x1-x2", "center": [0, 0], "direction": [1, -1],
            "square": {"definition": "|x_i|<=0.8", "infimum": float(margin_center - 2 * square_halfwidth),
                       "witness": [-0.8, 0.8]},
            "line": {"definition": "x=(t,t), |t|<=0.8", "infimum": float(margin_center)},
            "tube": {"definition": "x=(t,t)+v, |t|<=0.8, |v_i|<=0.1", "infimum": float(margin_center - 2 * tube_halfwidth),
                     "witness": [-0.1, 0.1]},
            "scope": "Different sets are compared; no matched-volume, matched-coverage or HAPT performance claim."},
        "tail_interval_example": {
            "n": tail_n, "old_margin": float(tail_old), "q": tail_q, "D_drop": tail_drop,
            "D_new_lower": tail_lower, "D_new_upper": tail_upper,
            "q_dot_D_drop": tail_drop_contribution, "q_dot_D_new_min": tail_min,
            "valid_margin_lower_fraction": str(tail_valid), "valid_margin_lower": float(tail_valid),
            "incorrect_use_lower_for_negative_coefficient": float(tail_wrong),
            "counterexample_D_new": [0, 3],
            "requirement": "For K classes, every one of the K-1 target-competitor margins must be certified."}}


def actual_read_only_audit():
    with np.load(DATA / "windows.npz") as archive:
        x = archive["x"]
    manifest = pd.read_parquet(DATA / "window_manifest.parquet")
    splits = pd.read_parquet(DATA / "windows_with_split.parquet")
    manifest = manifest.assign(global_row_id=np.arange(len(manifest)))
    df = manifest.merge(splits[["exp", "win_idx", "split"]],
                        on=["exp", "win_idx"], how="left", validate="one_to_one", sort=False)
    assert len(x) == len(df)
    assert df["global_row_id"].tolist() == list(range(len(df)))
    assert df["split"].notna().all()
    ordered = df.sort_values(["exp", "win_idx"])
    same_recording = ordered["exp"].to_numpy()[1:] == ordered["exp"].to_numpy()[:-1]
    consecutive = np.diff(ordered["win_idx"].to_numpy()) == 1
    mask = same_recording & consecutive
    ids = ordered["global_row_id"].to_numpy()
    left, right = ids[:-1][mask], ids[1:][mask]
    overlap_equal = (x[left, :, 64:] == x[right, :, :64]).all(axis=(1, 2))
    overlap_max_difference = float(np.max(np.abs(x[left, :, 64:] - x[right, :, :64])))
    pure = df["label_policy"].eq("pure")
    class_rows = []
    legacy_support = read_json("g2_class_support.json")
    for c in range(1, 13):
        counts = {s: int((pure & df["label_majority"].eq(c) & df["split"].eq(s)).sum())
                  for s in ("train", "val", "test")}
        assert counts == legacy_support["per_class"][str(c)]["pure"]
        class_rows.append({"class_id": c, "name": legacy_support["per_class"][str(c)]["name"],
                           **counts, "total": sum(counts.values())})
    all_counts = {s: int(df["split"].eq(s).sum()) for s in ("train", "val", "test")}
    pure12_counts = {s: sum(r[s] for r in class_rows) for s in all_counts}
    pure6_counts = {s: sum(r[s] for r in class_rows[:6]) for s in all_counts}
    left_df, right_df = df.iloc[left].reset_index(drop=True), df.iloc[right].reset_index(drop=True)
    same = (left_df["label_majority"].eq(right_df["label_majority"]) &
            left_df["label_policy"].eq("pure") & right_df["label_policy"].eq("pure")).to_numpy()
    change = left_df["label_majority"].ne(right_df["label_majority"]).to_numpy()
    pair_splits = left_df["split"].to_numpy()
    delta = (x[right] - x[left]).reshape(len(left), -1)
    train_same = same & (pair_splits == "train")
    diagonal = {}
    for level, name in [(0.5, "stable"), (0.75, "moderate"), (0.9, "wide")]:
        budgets = np.quantile(np.abs(delta[train_same]), level, axis=0)
        inside = (np.abs(delta) <= budgets).all(axis=1)
        record = {"per_dimension_train_same_quantile": level}
        for s in all_counts:
            subset = pair_splits == s
            record[s] = {"covered": int((inside & subset).sum()),
                         "denominator": int(subset.sum()),
                         "coverage": float(inside[subset].mean())}
        for kind, pair_mask in [("test_same", same), ("test_change", change)]:
            subset = (pair_splits == "test") & pair_mask
            record[kind] = {"covered": int((inside & subset).sum()),
                            "denominator": int(subset.sum()),
                            "coverage": float(inside[subset].mean())}
        record["changed_labels_among_covered_all_splits"] = {
            "count": int((change & inside).sum()), "denominator": int(inside.sum())}
        diagonal[name] = record
    sample_total = int(x.size)
    finite_total = int(np.isfinite(x).sum())
    assert finite_total == sample_total
    assert bool(overlap_equal.all())
    assert int(mask.sum()) == len(df) - df["exp"].nunique()
    return {
        "status": "Fresh read-only audit of existing NPZ and Parquet; no data regenerated.",
        "array_shape": list(x.shape), "array_dtype": str(x.dtype),
        "scalar_values_total": sample_total, "finite_scalar_values": finite_total,
        "nonfinite_scalar_values": sample_total - finite_total,
        "windows": len(df), "recordings": int(df["exp"].nunique()),
        "subjects": int(df["subject"].nunique()),
        "unique_exp_win_idx": not bool(df.duplicated(["exp", "win_idx"]).any()),
        "policy_counts": {k: int(v) for k, v in df["label_policy"].value_counts().items()},
        "split_window_counts": all_counts,
        "subject_counts": {s: int(df.loc[df["split"].eq(s), "subject"].nunique()) for s in all_counts},
        "class_pure_counts": class_rows,
        "pure12_split_counts": pure12_counts, "pure6_split_counts": pure6_counts,
        "pure12_total": sum(pure12_counts.values()), "pure6_total": sum(pure6_counts.values()),
        "transition_pure_total": sum(r["total"] for r in class_rows[6:]),
        "transition_fraction_of_pure12": rational(sum(r["total"] for r in class_rows[6:]), sum(pure12_counts.values())),
        "pure6_fraction_of_all_stream": rational(sum(pure6_counts.values()), len(df)),
        "adjacent_pair_counts": {s: int((pair_splits == s).sum()) for s in all_counts},
        "same_activity_pure_pairs": int(same.sum()), "label_change_pairs": int(change.sum()),
        "exact_raw_overlap": {
            "tested_pairs": int(mask.sum()), "pairs_exactly_equal": int(overlap_equal.sum()),
            "failed_pairs": int((~overlap_equal).sum()),
            "coordinates_compared_per_pair": 6 * 64,
            "coordinates_compared_total": int(mask.sum()) * 6 * 64,
            "max_absolute_difference": overlap_max_difference,
            "start_delta64_pairs": int((np.diff(ordered["start"].to_numpy())[mask] == 64).sum()),
            "comparison": "X_t[:,64:128] == X_(t+1)[:,0:64], componentwise exact float32 equality, only same-recording consecutive win_idx pairs",
            "scope_limit": "This verifies stored raw overlap and the index bridge. It does not prove same-coordinate closeness, feature reuse, labels, certificate validity or runtime savings."},
        "diagonal_box_audit": {
            "status": "Fresh arithmetic from existing raw arrays, same train-fit definition as scripts/05; no verifier or neural model used.",
            "dimension": int(delta.shape[1]), "train_same_fit_pairs": int(train_same.sum()),
            "coverage_counts": diagonal,
            "scope_limit": "Empirical geometry coverage is neither model certifiability nor certified cache hit rate. Raw mixed-unit coordinates are not directly comparable to normalized P0 radii."},
        "annotation_limit": "Reuses the stored policy labels. Does not independently settle the original annotation indexing convention or relabel raw samples.",
        "raw_units_local_source": {"acceleration": "g (README: 1 g = 9.80665 m/s^2)",
                                   "angular_velocity": "rad/s", "source": "data/hapt/README.txt, Notes"}}


def archived_observations():
    metrics = read_json("g2_metrics.json")
    names = read_json("g2_training.json")["class_names"]
    reconstructed = []
    for run in metrics["per_seed"]:
        record = {"seed": run["seed"], "best_epoch": run["best_epoch"]}
        for split in ("val", "test"):
            stored = run[split]
            cm = stored["confusion_matrix"]
            supports = [sum(row) for row in cm]
            predicted = [sum(row[j] for row in cm) for j in range(len(cm))]
            correct = sum(cm[i][i] for i in range(len(cm)))
            total = sum(supports)
            f1s = [Fraction(2 * cm[i][i], supports[i] + predicted[i]) for i in range(len(cm))]
            exact_f1 = sum(f1s, Fraction()) / len(cm)
            record[split] = {
                "n": total, "correct": correct, "errors": total - correct,
                "accuracy": rational(correct, total),
                "macro_f1": {"fraction_reduced": str(exact_f1), "value": float(exact_f1)},
                "stored_rounded_accuracy": stored["accuracy"],
                "stored_rounded_macro_f1": stored["macro_f1"],
                "confusion_matrix_true_rows_predicted_columns": cm,
                "per_class": [{"name": name, "tp": cm[i][i], "support": supports[i],
                               "predicted_count": predicted[i],
                               "precision": rational(cm[i][i], predicted[i]),
                               "recall": rational(cm[i][i], supports[i]),
                               "f1": {"fraction_reduced": str(f1s[i]), "value": float(f1s[i])}}
                              for i, name in enumerate(names)]}
        reconstructed.append(record)
    p0 = read_json("p0_verifier_probe.json")
    p0rows = []
    tf_ms = read_json("g2_cpu_latency.json")["batch1"]["mean_ms"]
    for key, v in p0["step1_gate_a"].items():
        p0rows.append({"normalized_radius_k": v["k"],
                       "anchors_sampled": 100,
                       "ibp_certified": int(round(v["ibp_p_c"] * 100)),
                       "crown_certified": int(round(v["crown_p_c"] * 100)),
                       "archived_crown_build_p50_ms": v["crown_t_build_ms_p50"],
                       "archived_crown_build_p95_ms": v["crown_t_build_ms_p95"],
                       "build_p50_div_archived_forward_mean_only": v["crown_t_build_ms_p50"] / tf_ms})
    exact_test_f1 = [r["test"]["macro_f1"]["value"] for r in reconstructed]
    exact_test_acc = [r["test"]["accuracy"]["value"] for r in reconstructed]
    return {
        "status": "Archived K6 observations; only arithmetic on stored artifacts is recomputed now. No fresh training, inference, verification or latency measurement.",
        "metrics_from_integer_confusions": reconstructed,
        "three_seed_summary": {
            "macro_f1_recomputed_from_confusion": mean_std(exact_test_f1),
            "accuracy_recomputed_from_confusion": mean_std(exact_test_acc),
            "macro_f1_using_archived_4decimal_inputs": mean_std([r["test"]["macro_f1"] for r in metrics["per_seed"]]),
            "accuracy_using_archived_4decimal_inputs": mean_std([r["test"]["accuracy"] for r in metrics["per_seed"]]),
            "rounding_note": "scripts/09 aggregates per-seed metrics already rounded to four decimals, with NumPy std ddof=0. The exact-confusion arithmetic and sample std ddof=1 are separately identified."},
        "P0": {"archived_probe": p0rows, "archived_raw_membership_replay": p0["step3_gate_b"],
               "sample_policy": "np.random.seed(42); uniform sample without replacement of 100 correctly predicted validation pure-K6 windows. Source comment says stratified but no stratification is implemented.",
               "sample_fraction_of_correct_val": rational(100, 1369),
               "sample_fraction_of_all_val": rational(100, 1450),
               "equivalence_check": "Stored eq_err=0 on only 10 random (6,128) tensors; not an exhaustive floating-point equivalence proof.",
               "replay_policy": "1-entry anchor; validation pure-K6 windows sorted by exp/win_idx; reset across recordings. Filtered windows may have gaps because win_idx adjacency is not enforced after filtering.",
               "not_end_to_end_reasons": ["Replay never calls the verifier and does not gate hits on certified status.", "Replay and certifiability sample use different anchor populations; product p_c*r is not measured joint certified-hit probability.", "Radius grids are different; no certifiability observation exists for replay k=0.1 or k=0.28.", "Only membership hits are counted; no ECDR fallback/lookup/state/energy/end-to-end timing is measured.", "The sample excludes incorrectly predicted anchors; sampled p_c is conditional, not a full validation certified fraction.", "Radii are normalized coordinates and cannot be equated to raw-coordinate event-box budgets."]}}


def main():
    sources = ["data/windows.npz", "data/window_manifest.parquet", "data/windows_with_split.parquet",
               "data/g2_class_support.json", "data/g2_metrics.json", "data/g2_training.json",
               "data/g2_cpu_latency.json", "data/p0_verifier_probe.json",
               "data/event_audit.json", "data/corr_semantics.json", "data/hapt/README.txt",
               "scripts/03_build_windows.py", "scripts/05_audit_temporal_events.py",
               "scripts/11_gate_a_prototype.py", "scripts/backbone.py", "configs/backbone.yaml"]
    out = {"schema": "ecdr-theory-calculations-v1", "report_date": "2026-10-28",
           "method": "Deterministic arithmetic + read-only audit; rerun recreates this JSON. Only this file is written.",
           "versions": {"numpy": np.__version__, "pandas": pd.__version__},
           "theoretical": theoretical_calculations(),
           "current_data_audit": actual_read_only_audit(),
           "legacy_appendix": archived_observations(),
           "source_sha256": {name: sha256(ROOT / name) for name in sources}}
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"written": str(OUTPUT),
                      "parameters": out["theoretical"]["cnn"]["parameters_total"],
                      "macs": out["theoretical"]["cnn"]["conv_and_head_macs"],
                      "audit": out["current_data_audit"]["exact_raw_overlap"],
                      "pure_counts": out["current_data_audit"]["pure12_split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
