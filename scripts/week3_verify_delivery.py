#!/usr/bin/env python3
"""Read-only Week 3 delivery consistency checks, not a certification theorem.

Writes only delivery_verification.json and memory_components.json below the
existing Week 3 run directory.  It does not rerun tests, training or benchmarks,
compile PDFs, regenerate wrappers, or modify frozen evidence. --skip-pdf gives
a PARTIAL receipt before final PDF/render evidence exists.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
from dataclasses import is_dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecdr.week3_verifier import ActualGraphIBP, FrozenK12Reference

OUT = ROOT / "output/week3_certification_20261103"
A = ROOT / "03_MATH/03_Week_3_Mathematical_Foundations.tex"
B = ROOT / "03_MATH/04_ECDR_Mathematics_Explained.tex"
EXPECTED_IDS = [f"H{i:02d}" for i in range(1, 25)]
KINDS = ("diagonal_box", "fixed_overlap_wave")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def record(checks, name, condition, evidence=None) -> None:
    checks.append({"name": name, "status": "PASS" if condition else "FAIL", "evidence": evidence or {}})


def verify_hashes(mapping) -> dict:
    failures = []
    for relative, expected in mapping.items():
        path = Path(relative) if Path(relative).is_absolute() else ROOT / relative
        actual = sha(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    return {"checked": len(mapping), "mismatches": failures}


def close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8)


def stats_match(values, recorded) -> bool:
    a = np.asarray(values, dtype=np.float64)
    return (recorded["n"] == len(a) and close(a.mean(), recorded["mean"])
            and close(np.quantile(a, .5), recorded["p50"])
            and close(np.quantile(a, .95), recorded["p95"])
            and close(a.sum(), recorded["total"]))


def boolean(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"unexpected CSV boolean {value!r}")
    return value == "True"


def verify_frozen_evidence(checks, out):
    before = read_json(out / "preservation_before.json")
    result = verify_hashes(before["files"])
    record(checks, "all_170_preexisting_protected_paths_preserved", result["checked"] == 170 and not result["mismatches"], result)
    week2 = read_json(ROOT / "output/week2_foundations_20261103/final_verification.json")
    result = verify_hashes(week2["primary_evidence"])
    record(checks, "week2_primary_evidence_hash_chain_preserved", not result["mismatches"], result)
    data = read_json(ROOT / "output/week2_foundations_20261103/data_audit.json")
    model = read_json(ROOT / "output/week2_foundations_20261103/model_audit.json")
    for name, mapping in (("data_sources", data["source_hashes"]), ("model_inputs", model["input_sha256"]),
                          ("data_audit_script", {"scripts/week2_data_audit.py": data["execution"]["script_sha256"]}),
                          ("model_audit_script", {"scripts/week2_model_audit.py": model["audit_script_sha256"]})):
        result = verify_hashes(mapping)
        record(checks, "week2_" + name + "_hashes_current", not result["mismatches"], result)
    week1 = read_json(out / "week1_preservation.json")
    counts = Counter(item["status"] for item in week1["checks"])
    record(checks, "week1_independent_65_check_receipt_consistency",
           week1["summary"] == {"total_checks": 65, "passed": 65, "failed": 0}
           and counts == {"PASS": 65},
           {"observed_check_statuses": dict(counts), "receipt_sha256": sha(out / "week1_preservation.json"),
            "evidence_type": "consistency of original independent-verification receipt; original check execution was performed separately",
            "not_established_by_this_check": "new model execution, float32 enclosure, mathematical certification or user acceptance"})
    log = (out / "regression_tests.log").read_text(encoding="utf-8")
    matches = re.findall(r"^Ran (\d+) tests in ([0-9.]+)s$", log, flags=re.MULTILINE)
    record(checks, "original_93_test_execution_log_consistency",
           len(matches) == 1 and matches[0][0] == "93" and re.search(r"^OK$", log, flags=re.MULTILINE) is not None
           and not re.search(r"^FAILED", log, flags=re.MULTILINE),
           {"log_sha256": sha(out / "regression_tests.log"), "parsed_test_runs": matches,
            "original_command": ".venv/bin/python -m unittest discover -s tests -v",
            "original_exit_code_reported_by_executing_parent": 0,
            "command_rerun_by_this_verifier": False,
            "scope": "original historical and Week3 tests executed; receipt consistency is not a numerical-soundness proof"})


def verify_experiments(checks, out):
    result = read_json(out / "experiment_results.json")
    checked = verify_hashes(result["source_hashes"])
    record(checks, "experiment_source_and_trace_hash_chain_current", not checked["mismatches"], checked)
    frozen = ROOT / "output/week1_closeout_20261102/reference/test_predictions.npz"
    with np.load(frozen) as data:
        ids, pred = data["global_row_id"].copy(), data["predictions"].copy()
    expected_ids = ids.astype(int).tolist()
    expected_predictions = dict(zip(expected_ids, pred.astype(int).tolist()))
    record(checks, "frozen_full_stream_denominators", len(expected_ids) == 5053 and result["full_test_windows"] == 5053
           and result["supervised_test_windows"] == 3572,
           {"full_stream": len(expected_ids), "supervised_from_original_experiment": result["supervised_test_windows"],
            "supervised_metric_not_recomputed": True})
    with (out / "domain_trace.csv").open(newline="", encoding="utf-8") as handle:
        domains = list(csv.DictReader(handle))
    with (out / "runtime_trace.csv").open(newline="", encoding="utf-8") as handle:
        runtime = list(csv.DictReader(handle))
    domain_json = read_json(out / "domain_results.json")
    runtime_json = read_json(out / "runtime_results.json")
    record(checks, "independent_domain_and_runtime_trace_row_counts", len(domains) == 10106 and len(runtime) == 20212,
           {"domain_rows": len(domains), "runtime_rows": len(runtime)})
    groups = defaultdict(list)
    for row in domains:
        groups[row["kind"]].append(row)
    observed_domain = {}
    for kind in KINDS:
        rows = groups[kind]
        counts = {
            "attempted_builds": len(rows),
            "conditional_positive_builds": sum(boolean(r["conditional_positive"]) for r in rows),
            "candidate_next_members": sum(boolean(r["next_member"]) for r in rows),
            "conditional_positive_member_queries": sum(boolean(r["conditional_member"]) for r in rows),
            "conditional_shadow_disagreements": sum(boolean(r["shadow_disagreement"]) for r in rows),
        }
        observed_domain[kind] = counts
        summary = domain_json[kind]
        proper_order = [int(row["global_row_id"]) for row in rows] == expected_ids
        adjacent = sum(boolean(row["has_next"]) for row in rows)
        derived_logic = True
        for i, row in enumerate(rows):
            next_exists = i+1 < len(rows) and rows[i+1]["exp"] == row["exp"] and int(rows[i+1]["win_idx"]) == int(row["win_idx"])+1
            joint = boolean(row["next_member"]) and boolean(row["conditional_positive"])
            disagreement = joint and expected_predictions[expected_ids[i+1]] != int(row["target"]) if next_exists else False
            derived_logic &= (boolean(row["has_next"]) == next_exists and boolean(row["conditional_member"]) == joint
                              and boolean(row["shadow_disagreement"]) == disagreement
                              and (not boolean(row["next_member"]) or next_exists)
                              and int(row["target"]) == expected_predictions[int(row["global_row_id"])]
                              and not boolean(row["formal_certified"]))
        numerical = all(counts[key] == summary[key] == result["domains"][kind][key] for key in counts)
        numerical &= stats_match([float(r["build_ms"]) for r in rows], summary["build_ms"])
        numerical &= stats_match([float(r["bound_ms"]) for r in rows], summary["bound_ms"])
        numerical &= dict(Counter(row["reason"] for row in rows)) == summary["failure_or_conditional_reasons"]
        numerical &= (summary["formal_reused_queries"] == summary["validated_numerical_certificates"] == 0
                      and summary["r_cert"] == 0 and summary["A_reuse"] is None)
        record(checks, f"{kind}_5053_domain_records_match_independent_recount", proper_order and adjacent == 5034 and derived_logic and numerical,
               {"counts": counts, "adjacent_query_pairs": adjacent, "record_order_matches_frozen": proper_order,
                "interpretation": "positive=float64 conditional estimate only; candidate membership is not certified acceptance"})
    groups = defaultdict(list)
    for row in runtime:
        groups[(int(row["repeat"]), row["kind"])].append(row)
    expected_groups = {(repeat, kind) for repeat in range(2) for kind in KINDS}
    record(checks, "runtime_has_exactly_four_declared_runs", set(groups) == expected_groups and len(runtime_json["runs"]) == 4,
           {"observed_runs": [list(key) for key in sorted(groups)]})
    run_counts = []
    for run in runtime_json["runs"]:
        key = (run["repeat"], run["kind"])
        rows = groups[key]
        reuse = sum(boolean(row["reuse"]) for row in rows)
        disagreed = sum(int(row["pred"]) != expected_predictions[int(row["global_row_id"])] for row in rows)
        builds = sum(bool(row["build_reason"]) for row in rows)
        positive = sum(row["build_reason"] == "numerical_guarantee_unavailable" for row in rows)
        passed = ([int(row["global_row_id"]) for row in rows] == expected_ids
                  and len(rows) == run["windows"] == 5053 and reuse == run["accepted_reuse"] == 0
                  and run["fallback_count"] == 5053 and disagreed == run["disagreements_full_stream"] == 0
                  and builds == run["attempted_builds"] and positive == run["conditional_positive_builds"]
                  and run["A_reuse"] is None
                  and dict(Counter(row["reason"] for row in rows)) == run["fallback_reasons"]
                  and dict(Counter(row["build_reason"] for row in rows if row["build_reason"])) == run["build_reasons"])
        for column, field in (("total_ms", "ecdr_ms"), ("reference_ms", "reference_component_ms"),
                              ("membership_ms", "membership_component_ms"), ("build_ms", "build_component_ms")):
            passed &= stats_match([float(row[column]) for row in rows], run[field])
        passed &= run in result["runtime_runs"]
        count = {"repeat": key[0], "kind": key[1], "windows": len(rows), "reused": reuse,
                 "reference_disagreements": disagreed, "builds": builds, "conditional_positive_builds": positive}
        run_counts.append(count)
        record(checks, f"runtime_repeat{key[0]}_{key[1]}_independent_recount", passed, count)
    return {"independent_domain_recounts": observed_domain, "independent_runtime_recounts": run_counts,
            "empirical_semantics": "5053 valid queries per run; all frozen-reference fallback, zero accepted formal reuse. Zero reference disagreements is consistency of fallback, not proof of certifier soundness. A_reuse is N/A/null."}


def verify_documents_and_wrappers(checks):
    a, b = A.read_text(encoding="utf-8"), B.read_text(encoding="utf-8")
    ids_a = re.findall(r"\\subsubsection\*\{(H\d{2}):", a)
    ids_b = re.findall(r"^(H\d{2})\s*&", b, flags=re.MULTILINE)
    record(checks, "report_A_exactly_one_original_ledger_record_H01_H24", ids_a == EXPECTED_IDS,
           {"observed_ids": ids_a, "scope": "ID traceability/count only; independent mathematical review covers meaning/proofs"})
    record(checks, "report_B_complete_H01_H24_pedagogical_mapping", ids_b == EXPECTED_IDS,
           {"observed_ids": ids_b})
    generated = runpy.run_path(str(ROOT / "me/build_sources.py"), run_name="week3_delivery_readonly_import")
    attempts = []
    def capture_write(path, content, *args, **kwargs):
        attempts.append({"path": str(path), "would_be_sha256": hashlib.sha256(content.encode()).hexdigest()})
        return len(content)
    with mock.patch.object(Path, "write_text", capture_write), contextlib.redirect_stdout(io.StringIO()):
        generated["sync_week3_wrappers"]()
    wrappers = [ROOT / "me/Week_3_Mathematical_Foundations.tex", ROOT / "me/ECDR_Mathematics_Explained.tex"]
    record(checks, "current_generator_week3_wrapper_sync_is_idempotent_read_only", not attempts and all(path.is_file() for path in wrappers),
           {"generator_sha256": sha(ROOT / "me/build_sources.py"), "captured_write_attempts": attempts,
            "method": "executed current sync_week3_wrappers with Path.write_text mocked; no write permitted",
            "wrapper_sha256": {str(path.relative_to(ROOT)): sha(path) for path in wrappers if path.is_file()}})


def verify_pdfs(checks, out):
    review = read_json(out / "pdf_review.json")
    records = review["reports"]
    record(checks, "pdf_review_contains_both_canonical_reports", {r["source"] for r in records} == {str(A.relative_to(ROOT)), str(B.relative_to(ROOT))},
           {"recorded_review_scope": review.get("scope"), "method": review.get("method")})
    observed = {}
    for report in records:
        source, pdf = ROOT / report["source"], ROOT / report["pdf"]
        process = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
        pages = int(re.search(r"^Pages:\s+(\d+)\s*$", process.stdout, flags=re.MULTILINE).group(1))
        renders = verify_hashes(report["rendered_page_hashes"])
        log_path = ROOT / report["final_compile_log"]
        log = log_path.read_text(encoding="utf-8")
        compile_errors = re.findall(r"^!.*|.*(?:Fatal error|Emergency stop|Undefined control sequence|LaTeX Error|Missing character|Overfull).*", log, flags=re.MULTILINE)
        text_process = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True)
        compact_text = re.sub(r"\s+", "", text_process.stdout)
        ids_present = all(identity in compact_text for identity in EXPECTED_IDS)
        passed = (sha(source) == report["source_sha256"] and sha(pdf) == report["pdf_sha256"]
                  and pages == report["pages"] and pages > 0 and renders["checked"] == pages
                  and not renders["mismatches"] and not compile_errors and ids_present and report["visual_review"] == "PASS")
        observed[report["source"]] = {"pages": pages, "pdf_sha256": sha(pdf), "source_sha256": sha(source),
            "rendered_pages_rehashed": renders["checked"], "render_mismatches": renders["mismatches"],
            "compile_log_sha256": sha(log_path), "compile_error_matches": compile_errors,
            "H01_H24_in_extracted_pdf": ids_present,
            "visual_review": "original separate visual-review assertion checked for current inputs; this script does not perform visual inspection"}
        record(checks, "canonical_pdf_source_render_chain:" + pdf.name, passed, observed[report["source"]])
    return observed


def deep_python_size(value, seen=None):
    """Reachable Python representation bytes; not allocator/RSS/peak memory."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if is_dataclass(value):
        size += deep_python_size(vars(value), seen)
    elif isinstance(value, dict):
        size += sum(deep_python_size(k, seen) + deep_python_size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(deep_python_size(v, seen) for v in value)
    return size


def measure_memory_components():
    """Inspect owned persistent tensor payload and one conditional certificate.

    One zero-radius diagnostic build constructs a representative certificate;
    no latency is reported and no test/benchmark is executed.  PyTorch allocator
    workspaces, transient intervals, datasets, Python modules and process RSS are
    intentionally outside these component measurements.
    """
    reference = FrozenK12Reference.load()
    verifier = ActualGraphIBP(reference)
    def tensor_payload(model):
        params = sum(t.numel() * t.element_size() for t in model.parameters())
        buffers = sum(t.numel() * t.element_size() for t in model.buffers())
        return {"parameter_bytes": params, "buffer_bytes": buffers, "total_tensor_payload_bytes": params + buffers}
    build = verifier.build(np.zeros((6, 128), dtype=np.float32), 0., recording_id="memory_example", step_idx=0)
    certificate = build.certificate
    if certificate is None:
        raise RuntimeError("representative zero-radius conditional certificate unavailable")
    return {"schema_version": 1, "measurement_method": "tensor numel*element_size; recursive sys.getsizeof with object-ID de-duplication",
            "source_function": "scripts/week3_verify_delivery.py:measure_memory_components",
            "source_sha256": sha(Path(__file__)), "python_version": sys.version, "torch_version": str(torch.__version__),
            "reference_float32_tensor_payload": tensor_payload(reference.model),
            "bound_snapshot_float64_tensor_payload": tensor_payload(verifier._model),
            "conditional_certificate_reachable_python_bytes": deep_python_size(certificate),
            "domain_reachable_python_bytes_standalone": deep_python_size(certificate.domain),
            "certificate_example": {"domain_kind": certificate.domain.kind, "radius": 0., "coordinates": 768,
                                    "formal_reuse_enabled": False, "numerical_status": "NOT ESTABLISHED"},
            "scope": "persistent CPU tensor payload + standalone reachable certificate representation; domain is included in certificate count and must not be added twice",
            "not_measured": ["peak live interval/workspace memory", "Python model/module object overhead", "allocator/reserved memory", "incremental process RSS", "GPU memory", "energy"],
            "interpretation": "working-memory components only; global experiment process_peak_rss includes data/PCA and is not cache overhead"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_dir():
        raise FileNotFoundError("existing Week 3 run directory required")
    started = datetime.now(timezone.utc).isoformat()
    checks, failures = [], []
    details = {}
    for name, function in (("frozen_evidence", lambda: verify_frozen_evidence(checks, out)),
                           ("experiment_consistency", lambda: verify_experiments(checks, out)),
                           ("document_mapping_and_wrappers", lambda: verify_documents_and_wrappers(checks))):
        try:
            details[name] = function()
        except Exception as error:
            failures.append({"stage": name, "type": type(error).__name__, "error": str(error)})
            record(checks, name + "_completed", False, failures[-1])
    if not args.skip_pdf:
        try:
            details["pdfs"] = verify_pdfs(checks, out)
        except Exception as error:
            failures.append({"stage": "pdf", "type": type(error).__name__, "error": str(error)})
            record(checks, "final_pdf_source_render_chain_completed", False, failures[-1])
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    record(checks, "git_diff_check", diff.returncode == 0,
           {"command": "git diff --check", "exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr,
            "scope": "tracked working-tree diff; does not add/commit untracked implementation files"})
    try:
        memory = measure_memory_components()
        write_json(out / "memory_components.json", memory)
        record(checks, "persistent_memory_components_measured", True, {"artifact": str((out / "memory_components.json").relative_to(ROOT)),
                                                                     "sha256": sha(out / "memory_components.json")})
    except Exception as error:
        record(checks, "persistent_memory_components_measured", False, {"error": str(error)})
    passed = all(check["status"] == "PASS" for check in checks)
    status = "PARTIAL" if passed and args.skip_pdf else "PASS" if passed else "FAIL"
    receipt = {"schema_version": 1, "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "command": " ".join(sys.argv), "script_sha256": sha(Path(__file__)),
               "python": sys.version, "platform": platform.platform(), "overall_status": status,
               "pending": ["final canonical PDFs and source/render/visual-review hash chain"] if args.skip_pdf else [],
               "checks": checks, "details": details, "exceptions": failures,
               "original_execution_records": [
                   {"command": ".venv/bin/python -m unittest discover -s tests -v > output/week3_certification_20261103/regression_tests.log 2>&1",
                    "unified_exec_session_id": 62029, "exit_code": 0},
                   {"command": ".venv/bin/python scripts/week1_independent_verify.py --output output/week3_certification_20261103/week1_preservation.json > output/week3_certification_20261103/week1_verification.log 2>&1",
                    "unified_exec_session_id": 74454, "exit_code": 0},
                   {"command": ".venv/bin/python scripts/week3_experiments.py > output/week3_certification_20261103/experiment.log 2>&1",
                    "unified_exec_session_id": 60096, "exit_code": 0},
               ],
               "original_execution_provenance": "parent executing agent supplied completed unified-exec tool results and exit codes; these commands are not rerun by delivery verification",
               "evidence_boundary": "This execution verifies artifact bytes, trace aggregation, ID mappings and recorded execution evidence. It does not rerun original scientific experiments/tests or establish floating-point certification, useful formal reuse, novelty or user acceptance.",
               "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in (A, B, ROOT / "ecdr/week3_verifier.py", ROOT / "scripts/week3_experiments.py", ROOT / "tests/test_week3_verifier.py")}}
    write_json(out / "delivery_verification.json", receipt)
    print(json.dumps({"overall_status": status, "checks": len(checks), "failed_checks": [c["name"] for c in checks if c["status"] != "PASS"],
                      "pending": receipt["pending"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
