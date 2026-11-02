#!/usr/bin/env python3
"""Read-only HAPT data audit for theory documents; writes only this QA folder.

Run: .venv/bin/python me/qa/data_cnn/prepare_data_analysis.py
No training, inference, timing benchmark, raw mutation, or policy correction.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA, RAW = ROOT / "data", ROOT / "data/hapt/RawData"
HZ, W, S = 50, 128, 64
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
SPLITS = ["train", "val", "test"]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def summary(v):
    a = np.asarray(v, dtype=np.float64)
    if not a.size:
        return {"n": 0, "note": "No observations in this subset; quantiles undefined."}
    return {"n": int(a.size), "min": float(a.min()), "mean": float(a.mean()),
            "std_population_ddof0": float(a.std()),
            **{k: float(x) for k, x in zip(["p01", "p25", "p50", "p75", "p90", "p95", "p99"],
                  np.quantile(a, [.01, .25, .5, .75, .9, .95, .99]))},
            "max": float(a.max())}


def savefig(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex_escape(t):
    return str(t).replace("_", r"\_")


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def table(name, columns, rows):
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(columns) + r" \\", r"\midrule"]
    lines += [" & ".join(latex_escape(x) for x in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.load(DATA / "windows.npz")["x"]
    man = pd.read_parquet(DATA / "window_manifest.parquet")
    spl = pd.read_parquet(DATA / "windows_with_split.parquet")
    man["global_row_id"] = np.arange(len(man))
    df = man.merge(spl[["exp", "win_idx", "split"]], on=["exp", "win_idx"],
                   validate="one_to_one", how="left", sort=False)
    assert df.global_row_id.tolist() == list(range(len(x)))
    assert df.split.notna().all()
    labels = np.loadtxt(RAW / "labels.txt", dtype=np.int64)
    names = {int(a.split()[0]): a.split()[1] for a in
             (DATA / "hapt/activity_labels.txt").read_text().splitlines() if a.strip()}
    recordings, all_raw, subject_rows, annotation_rows, correlation_rows = [], [], [], [], []
    integrity = {"pair_count": 0, "acc_only": [], "gyro_only": [],
                 "unaligned": [], "nonfinite_raw_coordinates": 0,
                 "label_span_out_of_bounds": [], "overlapping_label_samples": 0,
                 "window_raw_mismatch_coordinates": 0,
                 "window_raw_compared_coordinates": 0,
                 "label_policy_mismatch_windows": 0,
                 "annotation_conflicts": 0}
    raw_files = sorted(RAW.glob("acc_exp*_user*.txt"))
    expected_gyro = {"gyro_" + p.name[4:] for p in raw_files}
    integrity["gyro_only"] = sorted(p.name for p in RAW.glob("gyro_exp*_user*.txt")
                                    if p.name not in expected_gyro)
    representative = None
    multiplicities = {0: 0, 1: 0, 2: 0}
    for ap in raw_files:
        gp = RAW / ("gyro_" + ap.name[4:])
        if not gp.exists():
            integrity["acc_only"].append(ap.name)
            continue
        acc, gyro = np.loadtxt(ap), np.loadtxt(gp)
        exp, subject = map(int, re.search(r"exp(\d+)_user(\d+)", ap.name).groups())
        if acc.shape != gyro.shape or acc.shape[1] != 3:
            integrity["unaligned"].append(exp)
            continue
        sig = np.concatenate([acc, gyro], axis=1)
        T = len(sig)
        all_raw.append(sig)
        integrity["pair_count"] += 1
        integrity["nonfinite_raw_coordinates"] += int((~np.isfinite(sig)).sum())
        rec_labels = labels[labels[:, 0] == exp]
        assert (rec_labels[:, 1] == subject).all()
        plabel, assigned = np.zeros(T + 1, dtype=np.int16), np.zeros(T + 1, dtype=np.int16)
        for e, u, a, start, end in rec_labels:
            if start < 0 or end < start or end > T:
                integrity["label_span_out_of_bounds"].append([int(q) for q in [e, u, a, start, end]])
            selected = slice(int(start), int(end) + 1)
            prior = plabel[selected]
            integrity["annotation_conflicts"] += int(((prior > 0) & (prior != a)).sum())
            plabel[selected] = a
            assigned[selected] += 1
        integrity["overlapping_label_samples"] += int((assigned > 1).sum())
        group = df[df.exp == exp].sort_values("win_idx")
        assert len(group) == (T - W) // S + 1
        counts = np.zeros(T, dtype=np.int16)
        for row in group.itertuples():
            start, end = int(row.start), int(row.end)
            assert end - start == W and start == row.win_idx * S
            raw_window = sig[start:end].T.astype(np.float32)
            integrity["window_raw_mismatch_coordinates"] += int((raw_window != x[row.global_row_id]).sum())
            integrity["window_raw_compared_coordinates"] += raw_window.size
            counts[start:end] += 1
            annotated = plabel[start:end]
            annotated = annotated[annotated > 0]
            vals, cnts = np.unique(annotated, return_counts=True)
            if not len(annotated):
                policy, majority = "unlabeled", 0
            else:
                majority = int(vals[np.argmax(cnts)])
                domfrac = cnts.max() / W
                policy = "pure" if len(vals) == 1 and domfrac == 1 else "majority" if domfrac >= .6 else "ambiguous"
            integrity["label_policy_mismatch_windows"] += int(policy != row.label_policy or majority != row.label_majority)
        uniq, n = np.unique(counts, return_counts=True)
        for c, v in zip(uniq, n):
            multiplicities[int(c)] = multiplicities.get(int(c), 0) + int(v)
        split = str(group.split.iloc[0])
        windows_for_rec = x[group.global_row_id.to_numpy()].astype(np.float64)
        for ci, channel in enumerate(CHANNELS):
            a = windows_for_rec[:-1, ci]
            b = windows_for_rec[1:, ci]
            ac, bc = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
            denom = np.linalg.norm(ac,axis=1)*np.linalg.norm(bc,axis=1)
            local_corr = np.divide(np.sum(ac*bc,axis=1),denom,out=np.full(len(a),np.nan),where=denom!=0)
            feat = windows_for_rec[:,ci].mean(axis=1)
            correlation_rows.append({"exp": exp,"channel": channel,
                "actual_raw_sample_lag1":corr(sig[:-1,ci],sig[1:,ci]),
                "actual_raw_sample_lag64":corr(sig[:-S,ci],sig[S:,ci]),
                "mean_local_window_positional_lag1":float(np.nanmean(local_corr)),
                "window_channel_mean_feature_lag1":corr(feat[:-1],feat[1:])})
        rec = {"exp": exp, "subject": subject, "split": split, "samples": T,
               "acc_file": str(ap.relative_to(ROOT)), "acc_sha256": sha(ap),
               "gyro_file": str(gp.relative_to(ROOT)), "gyro_sha256": sha(gp),
               "duration_sample_count_s": T / HZ,
               "first_to_last_elapsed_s": (T - 1) / HZ,
               "label_spans": len(rec_labels),
               "annotated_samples_inherited_indexing": int((plabel[:T] > 0).sum()),
               "annotation_coverage_inherited": float((plabel[:T] > 0).mean()),
               "windows": len(group), "covered_unique_samples": int((counts > 0).sum()),
               "uncovered_tail_samples": int((counts == 0).sum()),
               "window_sample_occurrences": int(counts.sum())}
        recordings.append(rec)
        subject_rows.append(rec)
        if representative is None:
            representative = (sig[:500].copy(), plabel[:500].copy(), exp, subject)
    raw = np.concatenate(all_raw, axis=0)
    for activity in range(1, 13):
        group = labels[labels[:, 2] == activity]
        lengths = group[:, 4] - group[:, 3] + 1
        # Number of window-grid starts fully inside each inclusive annotation span.
        first_k = (group[:, 3] + S - 1) // S
        last_k = (group[:, 4] - W + 1) // S
        span_grid_counts = np.maximum(0, last_k - first_k + 1)
        annotation_rows.append({"class_id": activity, "name": names[activity],
                                "span_count": len(group), "samples_inclusive": int(lengths.sum()),
                                "duration_sample_count_s": float(lengths.sum() / HZ),
                                "span_duration_s": summary(lengths / HZ),
                                "span_length_samples": summary(lengths),
                                "spans_shorter_than_128": int((lengths < W).sum()),
                                "spans_at_least_128_but_no_aligned_window": int(((lengths >= W) & (span_grid_counts == 0)).sum()),
                                "spans_with_at_least_one_aligned_window": int((span_grid_counts > 0).sum()),
                                "pure_grid_windows_inside_individual_spans": int(span_grid_counts.sum()),
                                "subjects_with_span": int(len(np.unique(group[:, 1])))})
    rd = pd.DataFrame(recordings)
    subjects = []
    for subject, group in rd.groupby("subject"):
        subjects.append({"subject": int(subject), "split": str(group.split.iloc[0]),
                         "recordings": len(group), "samples": int(group.samples.sum()),
                         "duration_sample_count_s": float(group.samples.sum() / HZ),
                         "sum_first_to_last_elapsed_s": float(group.first_to_last_elapsed_s.sum()),
                         "windows": int(group.windows.sum())})
    support = []
    pure = df.label_policy.eq("pure")
    pure6 = pure & df.label_majority.between(1, 6)
    for activity in range(1, 13):
        support.append({"class_id": activity, "name": names[activity],
                        **{s: int((pure & df.label_majority.eq(activity) & df.split.eq(s)).sum())
                           for s in SPLITS}})
    # Exact frozen preprocessing convention from scripts/backbone.py (f32 reduction).
    train_x = x[(pure6 & df.split.eq("train")).to_numpy()]
    mean = train_x.mean(axis=(0, 2)).astype(np.float32)
    std = np.clip(train_x.std(axis=(0, 2)), 1e-6, None).astype(np.float32)
    z = ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    order = df.sort_values(["exp", "win_idx"]).global_row_id.to_numpy()
    adjacent = (df.iloc[order[1:]].exp.to_numpy() == df.iloc[order[:-1]].exp.to_numpy()) & \
               (np.diff(df.iloc[order].win_idx.to_numpy()) == 1)
    left, right = order[:-1][adjacent], order[1:][adjacent]
    ldf, rdf = df.iloc[left].reset_index(drop=True), df.iloc[right].reset_index(drop=True)
    same = (ldf.label_policy.eq("pure") & rdf.label_policy.eq("pure") &
            ldf.label_majority.eq(rdf.label_majority)).to_numpy()
    changed = ldf.label_majority.ne(rdf.label_majority).to_numpy()
    # same and changed are disjoint but not exhaustive: equal non-pure/zero labels remain.
    other = ~(same | changed)
    pair_split = ldf.split.to_numpy()
    overlap = x[left, :, S:] == x[right, :, :W-S]
    delta_raw = (x[right] - x[left]).reshape(len(left), -1)
    delta_z = (z[right] - z[left]).reshape(len(left), -1)
    distances = {}
    arrays = {}
    for coordinates, delta in [("raw_mixed_units", delta_raw), ("normalized_dimensionless", delta_z)]:
        l2 = np.linalg.norm(delta, axis=1)
        linf = np.abs(delta).max(axis=1)
        arrays[coordinates] = {"l2": l2, "linf": linf}
        result = {}
        for split_name, split_mask in [("all", np.ones(len(left), dtype=bool))] + [(s, pair_split == s) for s in SPLITS]:
            result[split_name] = {"all_pairs_denominator": int(split_mask.sum())}
            for kind, mask in [("all", np.ones(len(left), dtype=bool)), ("same_activity_pure", same),
                               ("label_majority_change", changed), ("equal_nonpure_or_unlabeled", other)]:
                m = mask & split_mask
                result[split_name][kind] = {"n": int(m.sum()), "l2": summary(l2[m]), "linf": summary(linf[m])}
        distances[coordinates] = result
    perclass_delta = []
    for activity in range(1, 13):
        m = same & ldf.label_majority.eq(activity).to_numpy()
        perclass_delta.append({"class_id": activity, "name": names[activity], "pairs": int(m.sum()),
                               "normalized_l2": summary(arrays["normalized_dimensionless"]["l2"][m]),
                               "normalized_linf": summary(arrays["normalized_dimensionless"]["linf"][m])})
    split_summary = {}
    for s in SPLITS:
        g = df[df.split.eq(s)]
        r = rd[rd.split.eq(s)]
        split_summary[s] = {"subjects": sorted(int(v) for v in g.subject.unique()),
                            "subject_count": int(g.subject.nunique()), "recordings": int(g.exp.nunique()),
                            "samples_raw": int(r.samples.sum()), "raw_duration_s": float(r.samples.sum()/HZ),
                            "all_windows": len(g), "pure12": int((pure & df.split.eq(s)).sum()),
                            "pure6": int((pure6 & df.split.eq(s)).sum()),
                            "pair_count": int((pair_split == s).sum())}
    split_integrity = {"subjects_disjoint": all(set(split_summary[a]["subjects"]).isdisjoint(split_summary[b]["subjects"])
                     for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]),
                     "each_recording_in_one_split": bool(df.groupby("exp").split.nunique().eq(1).all()),
                     "each_subject_in_one_split": bool(df.groupby("subject").split.nunique().eq(1).all()),
                     "recording_window_indices_consecutive_in_stored_order": all(
                         np.array_equal(g.win_idx.to_numpy(),np.arange(len(g))) for _,g in df.groupby("exp",sort=False))}
    channel_statistics = [{"channel": ch, "unit": "g" if i < 3 else "rad/s",
                           "raw_unique_samples": summary(raw[:, i]),
                           "train_pure6_window_mean": float(mean[i]),
                           "train_pure6_window_std": float(std[i])} for i, ch in enumerate(CHANNELS)]
    temporal_corr = {"aggregation":"Arithmetic mean of 61 within-recording Pearson correlations; positional statistic first averages per adjacent pair within each recording. Actual raw series uses each sensor row once without flattening/repeating overlapping windows.",
                     "per_channel":[{"channel":ch,**{key:float(np.mean([r[key] for r in correlation_rows if r["channel"]==ch]))
                         for key in ["actual_raw_sample_lag1","actual_raw_sample_lag64","mean_local_window_positional_lag1","window_channel_mean_feature_lag1"]}} for ch in CHANNELS]}
    out = {"scope": "Fresh read-only raw/window audit, no training, inference, verifier or runtime claims.",
           "source_files": {str(p.relative_to(ROOT)): sha(p) for p in
                           [DATA / "windows.npz", DATA / "window_manifest.parquet", DATA / "windows_with_split.parquet",
                            RAW / "labels.txt", DATA / "hapt/activity_labels.txt", ROOT / "scripts/03_build_windows.py",
                            ROOT / "scripts/backbone.py", ROOT / "configs/training.yaml"]},
           "conventions": {"sampling_hz": HZ, "window_samples": W, "stride_samples": S,
                           "duration_sample_count": "T/50 represents sample-count convention; no measured timestamp column exists.",
                           "first_to_last_elapsed": "(T-1)/50, assuming a constant 50 Hz sample rate.",
                           "annotation": "Inclusive end-start+1; preserves existing direct Python indices [start:end+1]. README does not explicitly declare 0-based or 1-based indices. Do not infer scientifically confirmed alignment from policy reconstruction.",
                           "normalizer": "Pure K6 train windows, f32 mean/std axes(0,2), clip std>=1e-6. Overlapping sample occurrences count repeatedly, matching scripts/backbone.py.",
                           "same_activity": "both pure and same label_majority; excludes equal mixed or unlabeled windows",
                           "label_change": "label_majority differs, including transitions to/from 0 and nonpure windows; ground-truth metadata, not model prediction change",
                           "distance": "same-coordinate x_next[c,i]-x_prev[c,i] compares raw times separated by 64 samples; aligned overlap equality compares x_prev[c,64+i] with x_next[c,i].",
                           "raw_precision": "raw physical summaries use text->float64; saved-window equality uses raw->float32 as builder does"},
           "integrity": {**integrity, "saved_windows_nonfinite_coordinates": int((~np.isfinite(x)).sum()),
                         "saved_shape": list(x.shape), "saved_dtype": str(x.dtype)},
           "totals": {"recordings": len(recordings), "subjects": len(subjects), "label_spans": len(labels),
                      "unique_raw_time_samples": len(raw), "raw_sensor_coordinates": int(raw.size),
                      "duration_sample_count_s": len(raw)/HZ, "sum_first_to_last_elapsed_s": float(rd.first_to_last_elapsed_s.sum()),
                      "annotation_inclusive_samples": int(sum(r["samples_inclusive"] for r in annotation_rows)),
                      "inherited_labeled_raw_samples": int(rd.annotated_samples_inherited_indexing.sum()),
                      "windows": len(x), "window_coordinates": int(x.size),
                      "label_policy_counts": {str(k): int(v) for k, v in df.label_policy.value_counts().items()},
                      "crosses_boundary_windows": int(df.crosses_boundary.sum()),
                      "grid_sample_multiplicity": multiplicities,
                      "unique_covered_time_samples": int(sum(v for k,v in multiplicities.items() if k>0)),
                      "window_time_sample_occurrences": int(sum(k*v for k,v in multiplicities.items())),
                      "duplicate_time_sample_occurrences": int(sum((k-1)*v for k,v in multiplicities.items() if k>0)),
                      "uncovered_time_samples": multiplicities[0]},
           "channels": channel_statistics, "recordings": recordings, "subjects": subjects,
           "temporal_correlations": temporal_corr,
           "annotation_spans_by_class": annotation_rows, "pure_class_support": support,
           "splits": split_summary, "split_integrity": split_integrity,
           "adjacent_pairs": {"total": len(left), "same_activity_pure": int(same.sum()),
                              "label_majority_change": int(changed.sum()), "equal_nonpure_or_unlabeled": int(other.sum()),
                              "overlap_coordinates_compared": int(overlap.size),
                              "overlap_coordinates_unequal": int((~overlap).sum()),
                              "overlap_all_pairs_exact_equal": bool(overlap.all()),
                              "distances": distances, "same_activity_per_class": perclass_delta},
           "cautions": ["Pure support and annotation summaries use the inherited label policy, not a new annotation validation.",
                        "17,453 windows are not statistically independent because they overlap; use subject-level split and grouped uncertainty estimates.",
                        "Raw norm mixes g and rad/s and is descriptive only; normalized norm is dimensionless with a frozen train-fit map.",
                        "All-split summaries are exploratory descriptive audits. No thresholds, selection, or formal held-out validation outcome is inferred from these plots.",
                        "Historical scripts/06b_corr_semantics.py calls flattened overlapping-window arrays raw series; the fresh actual-raw correlations here instead read unique raw rows and avoid artificial concatenation boundaries.",
                        "Dataset README contains an old noncommercial license statement while saved UCI manifest declares CC BY 4.0; source conflict is unresolved here."]}
    (OUT / "data_analysis.json").write_text(json.dumps(out, indent=2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.2))
    policies = ["pure", "majority", "ambiguous", "unlabeled"]
    vals = [out["totals"]["label_policy_counts"][p] for p in policies]
    axs[0].bar(policies, vals, color=["#315a7d", "#5e8f9e", "#bc8b52", "#9d9d9d"])
    axs[0].set(ylabel="Window count", title="All chronological windows (N=17,453)")
    for i,v in enumerate(vals): axs[0].text(i,v+100,f"{v:,}\n{v/len(x):.1%}",ha="center",va="bottom",fontsize=8)
    axs[0].set_ylim(0, max(vals)*1.18)
    xx = np.arange(12)
    for k,s in enumerate(SPLITS):
        axs[1].bar(xx+(k-1)*.25,[r[s] for r in support],.25,label=s)
    axs[1].set(xticks=xx,xticklabels=range(1,13),xlabel="Activity ID",ylabel="Pure window count",
               title="Pure support by class and subject split")
    axs[1].axvline(5.5,color="0.5",ls="--",lw=.8); axs[1].legend(frameon=False)
    fig.tight_layout(); savefig(fig,"data_policy_support")
    sig, labs, exp, subject = representative
    fig, axs = plt.subplots(3,1,figsize=(10,6.2),sharex=True)
    time=np.arange(len(sig))/HZ
    for i in range(3):
        axs[0].plot(time,sig[:,i],lw=.75,label=CHANNELS[i])
        axs[1].plot(time,sig[:,i+3],lw=.75,label=CHANNELS[i+3])
    axs[0].set(ylabel="Acceleration (g)",title=f"First recording, first 10 s: exp{exp:02d}, subject {subject:02d}")
    axs[1].set_ylabel("Angular velocity (rad/s)")
    axs[0].legend(loc="upper right",ncol=3,frameon=False);axs[1].legend(loc="upper right",ncol=3,frameon=False)
    axs[2].step(time,labs,where="post",color="#315a7d");axs[2].set(xlabel="Time under 50 Hz convention (s)",ylabel="Inherited label ID",yticks=[0,5],ylim=(-.5,6))
    axs[2].text(.02,.8,"0 = unannotated; 5 = STANDING",transform=axs[2].transAxes,fontsize=8)
    fig.tight_layout();savefig(fig,"data_raw_first_recording")
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
    for ax,norm,label in zip(axs,["l2","linf"],["L2 distance","L-infinity distance"]):
        v=arrays["normalized_dimensionless"][norm]
        for mask,col,name in [(same,"#315a7d",f"same activity, pure (n={same.sum():,})"),(changed,"#ba6d38",f"label-majority change (n={changed.sum():,})")]:
            ordered=np.sort(v[mask]); ax.plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),lw=1.4,label=name)
        ax.set(xlabel=f"{label}, normalized coordinates",ylabel="Empirical cumulative probability",xscale="log",ylim=(0,1))
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,loc="lower right")
    fig.suptitle("Same-coordinate adjacent-window distances; pooled descriptive audit")
    fig.tight_layout();savefig(fig,"data_normalized_distances")
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    axs[0].bar(SPLITS,[split_summary[s]["subject_count"] for s in SPLITS],color="#315a7d")
    axs[0].set(ylabel="Subject count",title="Disjoint subject split",ylim=(0,20))
    for i,s in enumerate(SPLITS):axs[0].text(i,split_summary[s]["subject_count"]+.3,str(split_summary[s]["subject_count"]),ha="center")
    keys=[0,1,2];axs[1].bar(keys,[multiplicities[k] for k in keys],color=["#9d9d9d","#5e8f9e","#315a7d"])
    axs[1].set(xticks=keys,xlabel="Window-grid multiplicity of a raw time sample",ylabel="Unique raw time samples",title="Duplication due to 50% overlap")
    for i in keys:axs[1].text(i,multiplicities[i]+9000,f"{multiplicities[i]:,}",ha="center",fontsize=8)
    axs[1].set_ylim(0,max(multiplicities.values())*1.12)
    fig.tight_layout();savefig(fig,"data_split_multiplicity")
    table("data_channel_statistics.tex",["Kênh","Đơn vị","Mean","Std","Min","p01","p50","p99","Max"],
          [[r["channel"],r["unit"]]+[f'{r["raw_unique_samples"][k]:.5f}' for k in ["mean","std_population_ddof0","min","p01","p50","p99","max"]] for r in channel_statistics])
    table("data_class_spans.tex",["ID","Spans","Min (s)","Median (s)","Max (s)","$<128$ mẫu"],
          [[r["class_id"],r["span_count"],*[f'{r["span_duration_s"][k]:.2f}' for k in ["min","p50","max"]],r["spans_shorter_than_128"]] for r in annotation_rows])
    table("data_class_support.tex",["ID","Hoạt động","Train","Val","Test","Tổng"],
          [[r["class_id"],r["name"],r["train"],r["val"],r["test"],sum(r[s] for s in SPLITS)] for r in support])
    inventory = [r"\begin{longtable}{@{}rrlrrrr@{}}",
                 r"\toprule Exp & Người & Split & $T$ & Windows & Tail & Label (\%) \\",
                 r"\midrule\endfirsthead",
                 r"\toprule Exp & Người & Split & $T$ & Windows & Tail & Label (\%) \\",
                 r"\midrule\endhead"]
    for rec in recordings:
        inventory.append(" & ".join([str(rec["exp"]), str(rec["subject"]), rec["split"],
                         f'{rec["samples"]:,}'.replace(',', '.'), f'{rec["windows"]:,}'.replace(',', '.'),
                         str(rec["uncovered_tail_samples"]),
                         f'{100*rec["annotation_coverage_inherited"]:.2f}'.replace('.', ',')]) + r" \\")
    inventory += [r"\bottomrule\end{longtable}"]
    (OUT / "data_recording_inventory.tex").write_text("\n".join(inventory) + "\n")
    print(json.dumps({"output":str(OUT/"data_analysis.json"),"totals":out["totals"],"integrity":out["integrity"],"pairs":{k:v for k,v in out["adjacent_pairs"].items() if k not in ["distances","same_activity_per_class"]}},indent=2))


if __name__ == "__main__":
    main()
