"""Generate the frozen result ledgers from the Week 4 CSV and task register."""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output/week4/week4_initial_20261103"
OUT = ROOT / "04_SYNTHESIS/week5/reports/week5_ledgers.tex"


def escape(value):
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
                    "{": r"\{", "}": r"\}", "$": r"\$", "×": r"\(\times\)"}
    return "".join(replacements.get(c, c) for c in value)


def read_csv(name):
    with (RUN / "tables" / name).open() as f:
        return list(csv.DictReader(f))


def row(*values):
    return " & ".join(values) + r" \\"


methods = read_csv("comparison_matrix.csv")
latency = json.loads((Path(__file__).with_name("paired_latency.json")).read_text())["results"]
hypotheses = read_csv("hypotheses.csv")
assert [x["method_id"] for x in methods] == [f"M{i:02}" for i in range(17)]
assert [x["hypothesis_id"] for x in hypotheses] == [f"H{i:02}" for i in range(1, 25)]
lines = [
    r"\section{Replay Method Inventory}",
    r"\begin{table}[H]",
    r"\caption{Week 4 comparison matrix. Times and added delay $\Delta$ are in milliseconds. Reduction is relative to M00. Hits retain each method's semantics. A dash marks an unexecuted comparator.}",
    r"\centering\small",
    r"\begin{tabularx}{\textwidth}{@{}lYrrrrrrrr@{}}\toprule",
    row("ID", "Method", "Hits", "Mean", "p50", "p95", "QPS", "Speed/M00", r"$\Delta$", r"Reduction (\%)"),
    r"\midrule",
]
for x in methods:
    executed = x["execution_status"] == "EXECUTED"
    values = [x[k] for k in ("reuse_count", "mean_query_latency_ms", "p50_query_latency_ms",
                            "p95_query_latency_ms", "throughput_qps", "speed_ratio_vs_m00_core")]
    if executed:
        perf = latency[x["method_id"]]["all"]
        values.extend(f"{perf[key]:+.2f}" for key in ("delta_mean_ms", "latency_reduction_pct"))
    else:
        values.extend(["", ""])
    lines.append(row(x["method_id"], escape(x["method_name"]),
                     *(values if executed else ["--"] * len(values))))
lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}",
              r"\section{Hypothesis Dispositions}",
              "The statements and verdicts below follow the machine-readable Week 4 ledger. "
              "The underlying files retain Week 3 dispositions, governing tasks, and finite witnesses."])
for i in (0, 12):
    lines.extend([
        r"\begin{table}[H]",
        "\\caption{Week 4 ledger, H%02d--H%02d.}" % (i + 1, i + 12),
        r"\centering\small",
        r"\begin{tabularx}{\textwidth}{@{}lYY@{}}\toprule",
        row("ID", "Statement", "Recorded Verdict"),
        r"\midrule",
    ])
    for x in hypotheses[i:i+12]:
        lines.append(row(x["hypothesis_id"], escape(x["statement"]),
                         escape(x["week4_verdict"].replace("_", " "))))
    lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
closeout = (ROOT / "04_SYNTHESIS/week4/CLOSEOUT.md").read_text()
inventory = re.findall(r"^\|[^|]*\|\s*(W4-\d{3})\s*\|[^|]*\|[^|]*\|\s*\x60(VERIFIED)\x60", closeout, re.M)
assert [key for key, _ in inventory] == [f"W4-{i:03}" for i in range(1, 41)]
lines.extend([
    r"\section{Week 4 Task Register}",
    "The closeout records all 40 listed entries as verified engineering tasks. "
    "These dispositions attest to the scoped deliverables and do not imply positive scientific results.",
    r"\begin{table}[H]",
    r"\caption{Week 4 engineering tasks by closeout milestone.}",
    r"\centering\small",
    r"\begin{tabularx}{\columnwidth}{@{}llY@{}}\toprule",
    row("Group", "Task range", "Scope"),
    r"\midrule",
])
for group, span, scope in [
    ("M0", "001--004", "Governance, integrity, specification, environment"),
    ("M1", "005--010", "Numerical contract, margins, IBP, Lipschitz, CROWN, review"),
    ("M2", "011--018", "Domains, incremental computation, event triggers, lookup, integration"),
    ("M3", "019--026", "Runtime, fallback, cache, telemetry, reference, registry, assurance, harness"),
    ("M4", "027--031", "Workload, baselines, calibration, synthetic probes, protocol seal"),
    ("M5", "032--034", "Replay, timing, memory and scaling"),
    ("M6", "035--036", "Comparison and independent audit"),
    ("M7", "037--040", "Scientific and explanatory reports, rendering, delivery verification"),
]:
    lines.append(row(group, span, scope))
lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
OUT.write_text("\n".join(lines) + "\n")
