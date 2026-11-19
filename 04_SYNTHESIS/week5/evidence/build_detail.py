"""Render detailed Week 1--4 and follow-up evidence from saved artifacts."""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
W1 = ROOT / "output/week1_closeout_20261102/reference"
W2 = ROOT / "output/week2_foundations_20261103"
W3 = ROOT / "output/week3_certification_20261103"
W4 = ROOT / "output/week4/week4_initial_20261103"
EVIDENCE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def esc(value):
    tokens = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
        "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$", "×": r"\(\times\)",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "→": r"\(\to\)",
        "≥": r"\(\geq\)", "≤": r"\(\leq\)", "±": r"\(\pm\)",
        "Δ": r"\(\Delta\)", "∈": r"\(\in\)", "∞": r"\(\infty\)",
        "−": "-", "≈": r"\(\approx\)",
    }
    return "".join(tokens.get(c, c) for c in str(value))


lines = []


def add(*parts):
    lines.extend(parts)


def table(caption, cols, header, rows):
    add(r"\begin{table}[H]", r"\caption{" + esc(caption) + "}", r"\centering\small",
        r"\begin{tabularx}{\textwidth}{@{}" + cols + r"@{}}\toprule",
        " & ".join(map(esc, header)) + r"\\", r"\midrule")
    for row in rows:
        add(" & ".join(map(esc, row)) + r"\\")
    add(r"\bottomrule\end{tabularx}", r"\end{table}")


add(r"\section{Week 1: Reference Selection and Requirement Evidence}")
selection = read(W1 / "selection.json")
table("Stored validation selection across three training seeds.", "Yrrr",
      ("Fit", "Seed", "Best epoch", "Validation macro-F1"), [
          ("Selected" if x["seed"] == selection["selected_seed"] else "Candidate",
           x["seed"], x["best_epoch"], f'{x["best_val_macro_f1"]:.6f}')
          for x in sorted(selection["seed_evaluations"], key=lambda x: x["seed"])
      ])
add(r"Training used a maximum of 30 epochs, batch size 128, Adam at \(10^{-3}\), weight decay \(10^{-4}\), cosine scheduling and patience 10. The selection criterion was validation macro-F1. The baseline test result uses the selected seed and the frozen supervised denominator of 3,572. The evidence locators below follow the Week 1 requirements and closeout \cite{localw1}.")
w1 = [
    ("Research grounding", "Brief comparison and primary bibliography"),
    ("K=12 lock", "selection.json and SOURCE_OF_TRUTH.md give the selected reference and input identity"),
    ("Architecture", "Executed parameter count and layer shapes, reproduced in Week 2"),
    ("Pipeline execution", "Week 1 train, evaluation and verification receipts"),
    ("Transition support", "G2 hold and support evidence in the brief"),
    ("Research question", "Decision preservation, useful coverage and total cost"),
    ("Evidence register", "EVIDENCE_REGISTER.md and independent baseline check (65/65)"),
    ("Sensor/data contract", "Recording, channel order, window grid, split and label policy"),
    ("Hypotheses", "H1 to H4 in the Week 1 ledger"),
    ("Stratification", "Descriptive stable/transition pair counts and reference agreement"),
    ("Roadmap", "Closeout describes handoff to reconstruction and verification"),
    ("English deliverable", "Editable Week 1 brief and study records"),
]
table("Week 1 requirement-to-evidence map.", "lYY", ("ID", "Requirement", "Recorded evidence"),
      [(f"W1-{i:02}", title, source) for i, (title, source) in enumerate(w1, 1)])
add(r"\textbf{Interpretation.} Seed 43 led the three candidates on validation. High agreement for stable adjacent pairs coexists with much lower agreement for transition pairs. The baseline identifies the decision target and motivates the distinction between temporal similarity and a certificate.")

add(r"\section{Week 2: Dataset and Model Results}",
    r"Raw inputs are ordered accelerometer \(x,y,z\), then gyroscope \(x,y,z\). The accepted majority threshold is 77 of 128 samples. Pure and accepted majority windows enter supervised metrics, while ambiguous and unlabeled windows retain their chronological positions. The local reconstruction establishes the project grid and labels. The upstream annotation-origin sensitivity remains a limitation for comparisons that relabel boundary windows \cite{localweek2}.")
data = read(W2 / "data_audit.json")
support = data["split_supervised_support"]
classes = csv_rows(W2 / "model_per_class_metrics.csv")
table("Frozen class order, supervised support and selected-reference test metrics.", "rYrrrrrr",
      ("ID", "Class", "Train", "Val.", "Test", "Precision", "Recall", "F1"), [
          (i + 1, x["class_name"].replace("_", " ").title(),
           *(support[k]["by_class"][i]["n"] for k in ("train", "val", "test")),
           *(f"{float(x[k]):.4f}" for k in ("precision", "recall", "f1")))
          for i, x in enumerate(classes)
      ])
norm = data["normalization"]
table("Channel normalization fitted only on 6,955 supervised training windows.", "Yrr",
      ("Channel", "Mean", "Std. dev."), [
          (channel, f"{m:.7f}", f"{s:.7f}") for channel, m, s in
          zip(("Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"),
              norm["stored_mean"], norm["stored_std"])
      ])
add(r"\textbf{Interpretation.} Transition validation counts are much smaller than basic-activity counts. The lower transition F1 scores identify where the reference is less accurate on this split. They measure agreement with activity labels. Reuse is evaluated against the reference's decisions, including its mistakes.")

model = read(W2 / "model_audit.json")
macs = {x["module"]: x["macs_per_sample"] for x in model["cost"]["layers"]}
table("Actual module shapes at batch size one and audited cost per sample.", "lYYYrr",
      ("Module", "Operator", "Input shape", "Output shape", "Parameters", "MACs"), [
          (x["module"], x["type"], x["input_shape"], x["output_shape"],
           x["trainable_parameters"], macs[x["module"]])
          for x in csv_rows(W2 / "model_layer_shapes.csv")
      ])
add(r"Total cost is 1,172,992 convolution/dense MACs per sample with the archived convention. Bias, normalization, nonlinear operators and memory traffic are outside that count. The Week 2 mean/p50/p95 forward-only measurements were 0.1892/0.1557/0.3111\,ms on a resident normalized input. These measurements do not include the complete Week 4 decision path.")
with (W2 / "model_confusion_matrix.csv").open(newline="") as f:
    matrix = list(csv.reader(f))
assert len(matrix) == 13 and sum(int(v) for r in matrix[1:] for v in r[1:]) == 3572
table("Confusion counts on 3,572 supervised windows. Rows are true labels, columns are predictions; class IDs follow the support table.",
      "Y" + "r" * 12, ("True/Pred.", *range(1, 13)),
      [(i, *r[1:]) for i, r in enumerate(matrix[1:], 1)])
add(r"The full-stream top-two gap had minimum 0.0001135, median 5.9782 and p95 11.8876 logit units across 5,053 inputs. A sampled near-tie motivates strict margins. A large point gap alone gives no lower bound over a region.")

add(r"\section{Week 3: Domain, Backend and Runtime Evidence}")
domains = read(W3 / "domain_results.json")
table("Offline domain diagnostic at the validation-selected normalized radius 1e-6.",
      "Yrrrrr", ("Domain", "Builds", "Positive estimates", "Next members", "Formal hits", "Mean build (ms)"),
      [(name.replace("_", " "), x["attempted_builds"], x["conditional_positive_builds"],
        x["candidate_next_members"], x["formal_reused_queries"], f'{x["build_ms"]["mean"]:.3f}')
       for name, x in domains.items()])
runs = read(W3 / "runtime_results.json")["runs"]
table("Week 3 operational replay. Each run used 5,053 windows and zero accepted reuse.",
      "Yrrrrrr", ("Domain", "Repeat", "Builds", "Fallback", "Core (ms)", "ECDR (ms)", "Reduction (%)"),
      [(r["kind"].replace("_", " "), r["repeat"] + 1, r["attempted_builds"], r["fallback_count"],
        f'{r["reference_core_ms"]["mean"]:.3f}', f'{r["ecdr_ms"]["mean"]:.3f}',
        f'{100 * (1 - r["ecdr_ms"]["mean"] / r["reference_core_ms"]["mean"]):.2f}')
       for r in runs])
table("Fallback reason counts in the timed Week 3 runs.", "YrY",
      ("Domain", "Repeat", "Reason and query count"), [
          (r["kind"].replace("_", " "), r["repeat"] + 1,
           ", ".join(f'{k.replace("_", " ")}: {v:,}' for k, v in r["fallback_reasons"].items()))
          for r in runs
      ])
probe = read(W3 / "lirpa_probe.json")
table("Saved auto-LiRPA probes on the K=12 graph.", "YYrrY",
      ("Graph", "Conversion", "Bound records", "Positive", "Scope"),
      [(x["graph"].replace("_", " "), x["conversion"], len(x["bounds"]),
        sum(b.get("all_positive", False) for b in x["bounds"]),
        "No native bounds" if not x["bounds"] else "Conditional surrogate diagnostics")
       for x in probe["probes"]])
add(r"\textbf{Interpretation.} The native 1-D operator conversion did not produce bounds in this probe. The singleton-height surrogate matched sampled logits, but the probe did not establish exact executable equivalence in general. Positive interval estimates remained conditional, and no next query entered the tested domains. Builds, state checks and fallback therefore added cost without avoiding reference forwards \cite{localw3}.")

add(r"\section{Week 4: Resource Use and Task Evidence}")
memory = read(W4 / "memory_results.json")
table("Recorded attributable adapter/cache memory and membership coordinate counts.", "Yrrrrr",
      ("Method", "Adapter B", "State B", "Certificate B", "Total B", "Coordinates"),
      [(mid, *(x[key] if x["execution_status"] == "EXECUTED" else "--" for key in
                ("adapter_python_bytes", "state_cache_payload_bytes", "certificate_payload_bytes",
                 "total_attributable_cache_bytes", "coordinates_read")))
       for mid, x in sorted(memory["method_memory_profiles"].items())])
add(r"The separately reported frozen float32 tensor payload is 1,136,072 bytes. The float64 bounds snapshot is 2,272,120 bytes. The attribution table tracks the fields recorded for each method. Process RSS includes the runtime, backend and resident arrays, so it cannot be interpreted as cache cost. The study did not isolate dynamic workspace or energy \cite{localw4}.",
    r"\textbf{Interpretation.} M03 provides actual feature reuse, but its timed hit path is already slower than a complete M00 decision. Regional methods incur costs with no observed useful hits. Low attributed state size does not itself imply low query latency.")
closeout = (ROOT / "04_SYNTHESIS/week4/CLOSEOUT.md").read_text()
tasks = re.findall(r"^\|[^|]*\|\s*(W4-\d{3})\s*\|\s*([^|]+?)\s*\|[^|]*\|\s*\x60(VERIFIED)\x60", closeout, re.M)
assert [t[0] for t in tasks] == [f"W4-{i:03}" for i in range(1, 41)]
english = [
    "Task tracker, schemas and lock governance", "Baseline snapshot, dirty tree guard and integrity audit",
    "Method taxonomy, protocol matrix and task crosswalk", "Runtime, profiling and hardware telemetry",
    "Numerical soundness and floating-point safety bounds", "K=12 competing-class margin analysis",
    "IBP engine and radius calculation", "Global and local Lipschitz bounds", "LP/CROWN adapter",
    "Proof obligations and verification review", "Ball regions", "Box and diagonal regions",
    "Polytope and zonotope adapters", "PCA and low-dimensional projection",
    "Incremental streaming and partial recomputation", "Event triggers and heuristic detectors",
    "Exact and near-neighbor temporal lookup", "Domain integration and compliance",
    "Runtime adapter and lifecycle", "Fail-closed arbitration and fallback", "Cache and eviction",
    "Query instrumentation", "Reference and shadow validation", "Registry and lifecycle checks",
    "Fault injection and assurance", "Benchmark harness", "Workload and split isolation",
    "Reference core and other baseline paths", "Calibration and threshold selection",
    "Synthetic perturbation benchmarks", "Protocol seal and freeze", "Complete stream replay",
    "End-to-end CPU timing", "Memory and scaling profiles", "Synthesis and comparison",
    "Independent recount and audit", "Scientific report", "Explanatory report",
    "Rendered report and layout review", "Delivery verification and handoff",
]
assert len(english) == len(tasks)
assert all((W4 / "tasks" / tid / "receipt.json").is_file() for tid, _, _ in tasks)
for start, end in ((0, 20), (20, 40)):
    table(f"Week 4 closeout tasks {start+1:03}--{end:03}. Receipt locators are relative to the Week 4 run directory.",
          "lYYl", ("Task", "Scope in closeout", "Receipt locator", "Status"), [
              (tid, english[i], f"tasks/{tid}/receipt.json", state)
              for i, (tid, _, state) in enumerate(tasks) if start <= i < end
          ])

add(r"\section{Week 4: Hypothesis Witnesses}")
hs = csv_rows(W4 / "tables/hypotheses.csv")
assert [x["hypothesis_id"] for x in hs] == [f"H{i:02}" for i in range(1, 25)]
witness_clarifications = {
    "H03": "Rows 682 and 683 share true ID 11 (stand-to-lie); zero-based predictions are 10 and 8.",
    "H08": "On the same hidden box, direct margin minimization is no worse than separate extrema.",
    "H09": "The archived probes do not establish universal superiority in cost or tightness.",
    "H10": "Local geometry probes do not establish alignment across certified regions.",
    "H13": "Bitwise aligned features and finite decision agreement; no universal bitwise logit guarantee.",
    "H14": "Paired trace: M03 1.9273 ms, M00 0.3420 ms; reduction -463.47 percent.",
    "H17": "Zero useful hits for the horizon control; containment theorem remains conditional.",
    "H21": "M00 is fastest among timed paths. Faithful LipCache was not timed.",
    "H22": "A justified error enclosure for the executable backend is required.",
    "H24": "No witness among 15 executed out of 17 registered; other regimes remain open.",
}
for start in (0, 8, 16):
    table(f"Evidence recorded in the Week 4 hypothesis ledger, H{start+1:02}--H{start+8:02}.",
          "lYYY", ("ID", "Recorded verdict", "Witness or interpretation", "Governing tasks"), [
              (x["hypothesis_id"], x["week4_verdict"].replace("_", " "),
               witness_clarifications.get(x["hypothesis_id"], x["primary_witness_evidence"]), x["governing_tasks"])
              for x in hs[start:start+8]
          ])
add(r"The witnesses have different logical scope. A finite observation applies to the recorded data. A mathematical counterexample rejects a universal assertion. A conditional theorem needs its domain and numerical assumptions before executable reuse can follow.")

add(r"\section{Follow-up: Transition Quality and Stride Sensitivity}")
post = read(ROOT / "output/post_week4/g2_full_closure.json")
g2 = post["g2b_baseline_vs_upgraded"]["three_seed_summary_ddof0"]
table("Separate post-Week 4 G2 training. Canonical and expanded test policies use different denominators.",
      "Yr" + "r" * 6,
      ("Selection", "Seed", "Canonical acc.", "Canonical F1", "Transition F1",
       "Expanded acc.", "Expanded F1", "Transition F1"), [
           (f'Epoch {x["selected_epoch"]}', x["seed"], *(f"{x[k]:.4f}" for k in (
               "test_canonical_acc", "test_canonical_macro_f1", "test_canonical_transition_f1",
               "test_full_g2_acc", "test_full_g2_macro_f1", "test_full_g2_transition_f1")))
           for x in g2["seeds"]
       ])
add(r"The expanded policy removed all 18 recorded support blockers (18 to zero). Mean canonical test accuracy was 89.85 percent and mean expanded test accuracy was 87.92 percent. Mean transition F1 values were 0.6822 and 0.7917 under those respective policies. The training comparison changes the data policy and objective, so it does not isolate the effect of either change. The frozen Week 4 replay and its timing still refer to the original reference \cite{localg2}.")
stride = read(EVIDENCE / "stride_audit.json")["results"]
table("Stride audit. Counts use chronological pairs within each grid and recording.",
      "Yrrrrrr", ("Stride", "Pairs", "Equal top-1", "Median L2", "Median Linf", "Box hits", "Hit disagreements"), [
          (s, x["within_recording_pairs"], x["reference_top1_agree"],
           f'{x["median_l2"]:.3f}', f'{x["median_linf"]:.3f}',
           x["box_members"]["0.1"], x["box_members"]["0.1"] - x["reference_agreement_among_box_members"]["0.1"])
          for s, x in stride.items()
      ])
add(r"\textbf{Interpretation.} Shorter strides increase empirical neighboring agreement under these grids. Seven stride-one pairs within the radius-0.1 box changed decision. All 49,677 radius-0.1 member pairs were checked with conditional IBP and none had eleven positive lower bounds. This explains why high temporal agreement does not substantiate certified reuse for the tested box \cite{localstride}.",
    r"\textbf{Research conclusion.} The four-week work established the frozen data/model contract, derived a sufficient decision-preservation rule, and tested multiple ways to reuse work. The measured canonical result is negative for acceleration. M03 demonstrated feature reuse at a latency penalty, and the tested regional paths supplied no useful certified hits. The separate G2 study improved classification under its stated support policy. Future claims of net certified acceleration require a valid executable certificate, actual hits and a measured complete-path advantage on the same workload.")
output = ROOT / "04_SYNTHESIS/week5/reports/week5_detail.tex"
output.write_text("\n".join(lines) + "\n")
