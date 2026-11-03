#!/usr/bin/env python3
"""Compile-free delivery checks for the canonical Week 2 report package.

Run from the repository root with .venv/bin/python scripts/week2_verify_delivery.py.
Writes only document_checks.json and delivery_manifest.json under the Week 2
output directory. Final outputs are excluded from the manifest to avoid hashing
their own changing contents.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/week2_foundations_20261103"
DATA_TEX = ROOT / "01_DATA/03_Week_2_Data_Foundations.tex"
DATA_PDF = ROOT / "01_DATA/03_Week_2_Data_Foundations.pdf"
MODEL_TEX = ROOT / "02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex"
MODEL_PDF = ROOT / "02_MODEL/01_ECDR_CNN_Model_Analysis_EN.pdf"

CHECKS_PATH = OUT / "document_checks.json"
MANIFEST_PATH = OUT / "delivery_manifest.json"
FINAL_RECEIPT = OUT / "final_verification.json"

CHECKPOINT_SHA256 = "8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1"
PREPROCESSING_ID = "aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def record(checks: list[dict], name: str, passed: bool, expected, actual, evidence=None):
    checks.append({
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "expected": expected,
        "actual": actual,
        "evidence": evidence or {},
    })


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def pdf_pages(path: Path) -> int:
    result = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"pdfinfo did not report page count for {path}")
    return int(match.group(1))


def markdown_slug(heading: str) -> str:
    heading = re.sub(r"!?(\[([^\]]+)\])\([^)]*\)", r"\2", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"[`*_~]", "", heading).lower().strip()
    heading = re.sub(r"[^\w\- ]", "", heading, flags=re.UNICODE)
    heading = re.sub(r"\s+", "-", heading)
    return re.sub(r"-+", "-", heading).strip("-")


def markdown_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors = set(re.findall(r"\bid=[\"']([^\"']+)[\"']", text))
    counts: dict[str, int] = {}
    for match in re.finditer(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", text, flags=re.MULTILINE):
        base = markdown_slug(match.group(2))
        occurrence = counts.get(base, 0)
        slug = base if occurrence == 0 else f"{base}-{occurrence}"
        anchors.add(slug)
        counts[base] = occurrence + 1
    return anchors


def check_markdown_links(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    pending: list[dict] = []
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for markdown_path in paths:
        if not markdown_path.exists():
            continue
        text = markdown_path.read_text(encoding="utf-8")
        anchors_cache: dict[Path, set[str]] = {}
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip()
            if target.startswith("<") and ">" in target:
                target = target[1:target.index(">")]
            else:
                target = re.split(r"\s+['\"]", target, maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme.lower() in {"http", "https", "mailto", "ftp", "data"} or target.startswith("//"):
                continue

            rel_path = unquote(parsed.path)
            if rel_path:
                target_path = (markdown_path.parent / rel_path).resolve()
            else:
                target_path = markdown_path.resolve()

            if target_path == FINAL_RECEIPT.resolve() and not target_path.exists():
                pending.append({
                    "source": str(markdown_path.relative_to(ROOT)),
                    "target": str(target_path.relative_to(ROOT)),
                    "reason": "generated final verification receipt has not been written yet",
                })
                continue

            target_exists = target_path.exists()
            fragment_ok = True
            if target_exists and parsed.fragment and target_path.suffix.lower() in {".md", ".markdown"}:
                if target_path not in anchors_cache:
                    anchors_cache[target_path] = markdown_anchors(target_path)
                fragment_ok = unquote(parsed.fragment) in anchors_cache[target_path]
            results.append({
                "source": str(markdown_path.relative_to(ROOT)),
                "target": target,
                "exists": target_exists,
                "fragment_valid": fragment_ok,
                "status": "PASS" if target_exists and fragment_ok else "FAIL",
            })
    return results, pending


def verify_delivery() -> tuple[dict, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    checks: list[dict] = []
    pending: list[dict] = []

    required_paths = [DATA_TEX, DATA_PDF, MODEL_TEX, MODEL_PDF]
    record(checks, "canonical_report_pair_files_exist", all(p.is_file() for p in required_paths),
           [str(p.relative_to(ROOT)) for p in required_paths],
           [str(p.relative_to(ROOT)) for p in required_paths if p.is_file()])

    pdf_texts: dict[Path, str] = {}
    page_counts: dict[str, int] = {}
    for pdf in (DATA_PDF, MODEL_PDF):
        try:
            page_counts[str(pdf.relative_to(ROOT))] = pdf_pages(pdf)
            pdf_texts[pdf] = extract_pdf_text(pdf)
        except Exception as exc:  # report the tool failure as a failed check
            record(checks, f"pdf_readable:{pdf.name}", False, "pdfinfo and pdftotext succeed", str(exc))
    if len(page_counts) == 2:
        record(checks, "canonical_pdfs_have_positive_pages", all(n > 0 for n in page_counts.values()),
               "positive page counts", page_counts,
               {"observed_pages": page_counts, "exact_page_count_is_not_a_gate": True})

    data_source = DATA_TEX.read_text(encoding="utf-8") if DATA_TEX.exists() else ""
    model_source = MODEL_TEX.read_text(encoding="utf-8") if MODEL_TEX.exists() else ""
    data_pdf_text = pdf_texts.get(DATA_PDF, "")
    model_pdf_text = pdf_texts.get(MODEL_PDF, "")
    data_pdf_compact = re.sub(r"\s+", "", data_pdf_text)
    model_pdf_compact = re.sub(r"\s+", "", model_pdf_text)

    unit_ids = [f"W2-{i:02d}" for i in range(1, 9)]
    data_units_source = [unit for unit in unit_ids if unit in data_source]
    data_units_pdf = [unit for unit in unit_ids if unit in data_pdf_compact]
    record(checks, "eight_week2_unit_ids_in_canonical_source", data_units_source == unit_ids,
           unit_ids, data_units_source)
    record(checks, "eight_week2_unit_ids_in_canonical_pdf_text", data_units_pdf == unit_ids,
           unit_ids, data_units_pdf, {"extraction": "pdftotext -layout"})

    # Deep CNN structure and empirical margin evidence belong exclusively to MODEL.
    def math_markers(text: str) -> dict[str, bool]:
        compact_digits = re.sub(r"[\s,]", "", text)
        folded = normalize_text(text).lower()
        folded = re.sub(r"\\[a-zA-Z]+", "", folded)
        folded = re.sub(r"[{}$\\]", "", folded)
        return {
            "parameters_283692": "283692" in compact_digits,
            "macs_1172992": "1172992" in compact_digits,
            "receptive_field_20": bool(re.search(r"receptive\s+field.{0,160}?20\s*[- ]?samples?", folded)),
            "feature_jump_4": bool(re.search(
                r"(?:\bj\s*=\s*4\b|feature\s+spacing.{0,100}?(?:four|4)\s+(?:raw\s+)?samples?|(?:four|4)\s+original\s+samples\s+apart)", folded)),
        }
    required_math_markers = {"parameters_283692": True, "macs_1172992": True,
                             "receptive_field_20": True, "feature_jump_4": True}
    model_math = math_markers(model_source)
    model_pdf_math = math_markers(model_pdf_text)
    for name, actual in [("model_math_markers_in_source", model_math),
                         ("model_math_markers_in_pdf_text", model_pdf_math)]:
        record(checks, name, all(actual.values()), {k: True for k in required_math_markers}, actual)

    margin_markers = {"minimum_0.0001135": "0.0001135", "p95_11.88764": "11.88764"}
    margin_presence = {
        "model_source": {k: v in re.sub(r"\s+", "", model_source) for k, v in margin_markers.items()},
        "model_pdf": {k: v in model_pdf_compact for k, v in margin_markers.items()},
    }
    record(checks, "reported_point_margin_min_p95_match_model_report",
           all(all(result.values()) for result in margin_presence.values()),
           {scope: {key: True for key in margin_markers} for scope in margin_presence}, margin_presence)

    # Both deliverables are independent studies with their own objective, method,
    # and conclusion, present in editable source and exported PDF.
    study_markers = {
        "data": {
            "objective": r"study objective",
            "method": r"method combines source inspection",
            "conclusion": r"conclusions and evidence status",
        },
        "model": {
            "objective": r"independent model-study objective",
            "method": r"the method is a frozen-artifact audit",
            "conclusion": r"independent study conclusion",
        },
    }
    for study, markers in study_markers.items():
        source_text = normalize_text(data_source if study == "data" else model_source).lower()
        pdf_text = normalize_text(data_pdf_text if study == "data" else model_pdf_text).lower()
        source_hits = {key: bool(re.search(pattern, source_text)) for key, pattern in markers.items()}
        pdf_hits = {key: bool(re.search(pattern, pdf_text)) for key, pattern in markers.items()}
        record(checks, f"{study}_independent_objective_method_conclusion_source_and_pdf",
               all(source_hits.values()) and all(pdf_hits.values()),
               {"source": {k: True for k in markers}, "pdf": {k: True for k in markers}},
               {"source": source_hits, "pdf": pdf_hits})

    # Required data-side contract and audit markers. Exact counters are also
    # independently asserted against data_audit.json below.
    def data_contract_markers(text: str) -> dict[str, bool]:
        compact = re.sub(r"\s+", "", text).lower()
        flat = normalize_text(text).lower()
        return {
            "window_count_17453": "17453" in re.sub(r"[\s,]", "", text),
            "supervised_total_12278": "12278" in re.sub(r"[\s,]", "", text),
            "aligned_pairs_17392": "17392" in re.sub(r"[\s,]", "", text),
            "train_supervised_6955": "6955" in re.sub(r"[\s,]", "", text),
            "majority_threshold_77_of_128": bool(re.search(r"77\s*(?:of|/)\s*128", flat)),
            "float32_contract": "float32" in compact,
            "six_by_128_contract": bool(re.search(
                r"6\s*[×x]\s*128|\[6,128\]|6\s*\\times\s*128", text, flags=re.IGNORECASE)),
            "four_label_policies": all(label in flat for label in ("pure", "majority", "ambiguous", "unlabeled")),
        }
    data_contract_source = data_contract_markers(data_source)
    data_contract_pdf = data_contract_markers(data_pdf_text)
    record(checks, "data_input_contract_and_counts_in_source_and_pdf",
           all(data_contract_source.values()) and all(data_contract_pdf.values()),
           {"source": {k: True for k in data_contract_source}, "pdf": {k: True for k in data_contract_pdf}},
           {"source": data_contract_source, "pdf": data_contract_pdf})

    data_forbidden = {
        "no_model_confusion_counts": "model_confusion_matrix" not in data_source and "model_confusion_matrix" not in data_pdf_text,
        "no_model_margin_distribution_values": all(value not in re.sub(r"\s+", "", data_source + data_pdf_text)
                                                    for value in ("0.0001135", "11.88764", "21.46854")),
        "no_model_cost_or_receptive_field_values": all(value not in re.sub(r"[\s,]", "", data_source + data_pdf_text)
                                                        for value in ("1172992", "receptivefield20", "jump4")),
    }
    record(checks, "data_report_does_not_duplicate_model_results", all(data_forbidden.values()),
           {k: True for k in data_forbidden}, data_forbidden)

    # Check that each study hands off at its actual document boundary.
    handoff = {
        "data_title_and_final_contract": "data foundations and audit" in data_source.lower() and
                                         "w2-08: final data contract and handoff" in data_source.lower(),
        "model_title_and_input_contract": "model companion" in model_source.lower() and
                                          "input semantics and preprocessing" in model_source.lower(),
        "data_points_to_model_study": "companion cnn study" in data_source.lower(),
        "model_points_to_data_contract": "data contract" in model_source.lower(),
    }
    record(checks, "data_and_model_contract_boundaries_present", all(handoff.values()),
           {k: True for k in handoff}, handoff)

    docs_for_identity = {
        "01_DATA/03_Week_2_Data_Foundations.tex": data_source,
        "02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex": model_source,
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "",
        "01_DATA/README.md": (ROOT / "01_DATA/README.md").read_text(encoding="utf-8") if (ROOT / "01_DATA/README.md").exists() else "",
        "02_MODEL/README.md": (ROOT / "02_MODEL/README.md").read_text(encoding="utf-8") if (ROOT / "02_MODEL/README.md").exists() else "",
    }
    identity_presence = {}
    identity_integrity = {}
    for label, text in docs_for_identity.items():
        present = {"checkpoint_sha256": CHECKPOINT_SHA256 in text,
                   "preprocessing_id": PREPROCESSING_ID in text}
        identity_presence[label] = present
        # A 16- to 64-hex token beginning with either known ID must be the full ID.
        integrity_errors = []
        for expected_id, key in ((CHECKPOINT_SHA256, "checkpoint_sha256"),
                                 (PREPROCESSING_ID, "preprocessing_id")):
            for match in re.finditer(re.escape(expected_id[:16]) + r"[0-9a-f]*", text, flags=re.IGNORECASE):
                if match.group(0).lower() != expected_id:
                    integrity_errors.append({"field": key, "token": match.group(0)})
        identity_integrity[label] = integrity_errors
    required_identity_fields = {
        "01_DATA/03_Week_2_Data_Foundations.tex": {"preprocessing_id": True},
        "02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex": {"checkpoint_sha256": True, "preprocessing_id": True},
        "02_MODEL/README.md": {"checkpoint_sha256": True, "preprocessing_id": True},
    }
    identity_docs_pass = all(all(identity_presence[path][field] for field in fields)
                             for path, fields in required_identity_fields.items())
    no_truncated_ids = all(not errors for errors in identity_integrity.values())
    record(checks, "required_full_identity_ids_in_data_model_sources_and_model_readme",
           identity_docs_pass, required_identity_fields,
           {p: identity_presence[p] for p in docs_for_identity},
           {"data_requires_preprocessing_id_only": True, "model_requires_checkpoint_and_preprocessing_ids": True})
    record(checks, "no_truncated_known_checkpoint_or_preprocessing_ids", no_truncated_ids,
           "any mentioned known ID is full length", identity_integrity)

    data_boundary_source = normalize_text(data_source).lower()
    data_boundary_pdf = normalize_text(data_pdf_text).lower()
    boundary_patterns = {
        "annotation_origin_open": r"origin\s+open",
        "historical_provenance_open": r"historical\s+(?:majority\s+support\s+)?provenance\s+(?:remains\s+)?open|open\s+historical\s+provenance",
        "g2_hold": r"g2\s+(?:quality\s+gate\s+)?hold",
    }
    boundary_source = {label: bool(re.search(pattern, data_boundary_source))
                       for label, pattern in boundary_patterns.items()}
    boundary_pdf = {label: bool(re.search(pattern, data_boundary_pdf))
                    for label, pattern in boundary_patterns.items()}
    record(checks, "open_annotation_historical_provenance_and_g2_hold_in_data_source",
           all(boundary_source.values()), {label: True for label in boundary_patterns}, boundary_source)
    record(checks, "open_annotation_historical_provenance_and_g2_hold_in_data_pdf",
           all(boundary_pdf.values()), {label: True for label in boundary_patterns}, boundary_pdf)

    data_audit_path = OUT / "data_audit.json"
    model_audit_path = OUT / "model_audit.json"
    preservation_path = OUT / "week1_preservation_verification.json"
    data_audit = read_json(data_audit_path) if data_audit_path.exists() else {}
    model_audit = read_json(model_audit_path) if model_audit_path.exists() else {}
    preservation = read_json(preservation_path) if preservation_path.exists() else {}

    data_assertions = {
        "raw_rebuilt_windows_exact": data_audit.get("raw_rebuilt_windows_exact") is True,
        "raw_rebuilt_manifest_exact": data_audit.get("raw_rebuilt_manifest_exact") is True,
        "shape_is_17453x6x128": data_audit.get("window_shape") == [17453, 6, 128],
        "stream_split_denominators": data_audit.get("split_stream_denominators") == {"train": 9978, "val": 2422, "test": 5053},
        "subject_disjoint_17_4_9": (data_audit.get("subject_disjoint") is True and
                                     data_audit.get("subject_counts_by_split") == {"train": 17, "val": 4, "test": 9}),
        "supervised_support_6955_1751_3572": {s: v.get("supervised_n") for s, v in data_audit.get("split_supervised_support", {}).items()} ==
                                               {"train": 6955, "val": 1751, "test": 3572},
        "one_sample_assignment_delta_98": data_audit.get("one_sample_coordinate_shift_changed_window_policy_or_label_assignments") == 98,
        "normalization_matches_checkpoint": (data_audit.get("normalization", {}).get("mean_exact") is True and
                                               data_audit.get("normalization", {}).get("std_exact") is True and
                                               data_audit.get("normalization", {}).get("checkpoint_norm_matches_saved_json") is True and
                                               data_audit.get("normalization", {}).get("checkpoint_preprocessing_id_matches") is True),
        "data_audit_script_sha_matches_current_source": data_audit.get("execution", {}).get("script_sha256") == sha256(ROOT / "scripts/week2_data_audit.py"),
        "aligned_overlap_pairs_17392_raw_and_normalized_exact": (data_audit.get("overlap", {}).get("aligned_pair_count") == 17392 and
                                                                  data_audit.get("overlap", {}).get("raw_exact_equal_pairs") == 17392 and
                                                                  data_audit.get("overlap", {}).get("normalized_abs_distance_mean") == 0.0),
    }
    record(checks, "data_audit_required_assertions_pass", all(data_assertions.values()),
           {k: True for k in data_assertions}, data_assertions)

    required_model_checks = {
        "checkpoint_sha256_matches_selection", "run_seed_epoch_class_order_match",
        "normalization_arrays_match_checkpoint_exactly", "normalization_preprocessing_id_matches",
        "normalization_file_sha256_matches_checkpoint", "test_row_ids_match_stored_prediction_artifact",
        "stored_logits_allclose_atol_1e-5_rtol_1e-5", "stored_argmax_predictions_exact_match",
        "manifest_row_ids_predictions_and_class_names_exact_match",
        "manifest_margins_match_fresh_top_two_logits_atol_1e-5_rtol_1e-5",
        "parameter_count_is_283692", "leaf_module_count_is_15", "mac_count_is_1172992",
        "receptive_field_is_20_and_stride_is_4", "batchnorm_eval_repeated_forward_is_deterministic",
    }
    verified_names = set(model_audit.get("verified_check_names", []))
    metrics = model_audit.get("metrics", {})
    margins = metrics.get("top_two_logit_margin", {})
    model_assertions = {
        "all_required_named_checks_present": required_model_checks.issubset(verified_names),
        "verified_count_matches_names": model_audit.get("verified_check_count") == len(verified_names) == 18,
        "audit_script_sha_matches_current_source": model_audit.get("audit_script_sha256") == sha256(ROOT / "scripts/week2_model_audit.py"),
        "parameter_count_283692": model_audit.get("trainable_parameters") == 283692,
        "mac_count_1172992": model_audit.get("cost", {}).get("total_macs_per_sample") == 1172992,
        "receptive_field_and_jump_20_4": model_audit.get("receptive_field", {}).get("final_receptive_field_samples") == 20 and
                                          model_audit.get("receptive_field", {}).get("final_stride_samples") == 4,
        "stored_logits_exact_and_argmax_zero_disagreement": metrics.get("stored_logits_max_abs_error") == 0.0 and
                                                               metrics.get("stored_prediction_disagreement_count") == 0,
        "margin_min_p95_and_max_match_reported_artifact": margins.get("min") == 0.00011348724365234375 and
                                                           margins.get("p95") == 11.887640953063965 and
                                                           margins.get("max") == 21.468536376953125,
    }
    record(checks, "model_audit_required_assertions_pass", all(model_assertions.values()),
           {k: True for k in model_assertions}, model_assertions,
           {"required_named_checks": sorted(required_model_checks), "verified_check_count": model_audit.get("verified_check_count"),
            "margin_min": margins.get("min"), "margin_p95": margins.get("p95"),
            "margin_max": margins.get("max")})

    preservation_pass = (preservation.get("overall_status") == "PASS" and
                         preservation.get("summary", {}).get("total_checks") == 65 and
                         preservation.get("summary", {}).get("passed") == 65 and
                         preservation.get("summary", {}).get("failed") == 0)
    record(checks, "week1_preservation_receipt_65_of_65_pass", preservation_pass,
           {"overall_status": "PASS", "total_checks": 65, "passed": 65, "failed": 0},
           preservation.get("summary", {}), {"receipt": str(preservation_path.relative_to(ROOT))})

    if FINAL_RECEIPT.exists():
        final_receipt = read_json(FINAL_RECEIPT)
        receipt_ok = final_receipt.get("overall_status") == "PASS"
        record(checks, "final_generated_verification_receipt_passes", receipt_ok,
               {"overall_status": "PASS"}, {"overall_status": final_receipt.get("overall_status")},
               {"receipt": str(FINAL_RECEIPT.relative_to(ROOT))})
    else:
        pending.append({"artifact": str(FINAL_RECEIPT.relative_to(ROOT)),
                        "reason": "generated final verification receipt; checked automatically when it exists"})

    markdown_paths = [ROOT / "README.md", ROOT / "01_DATA/README.md", ROOT / "02_MODEL/README.md",
                      ROOT / "me/README.md", OUT / "README.md"]
    link_results, pending_links = check_markdown_links(markdown_paths)
    pending.extend(pending_links)
    link_failures = [item for item in link_results if item["status"] != "PASS"]
    record(checks, "markdown_relative_links_and_local_fragments_resolve",
           not link_failures, "all local links resolve and Markdown fragments exist", link_failures,
           {"markdown_files_checked": [str(p.relative_to(ROOT)) for p in markdown_paths if p.exists()],
            "local_links_checked": len(link_results), "external_links_skipped_without_network": True,
            "pending_receipt_links": len(pending_links)})

    # Assert that source and report paths in the final manifest remain within the assigned checkout.
    concept_audit_path = OUT / "concept_audit_64.csv"
    content_review_path = OUT / "content_review.md"
    concept_ids = ([f"P{i:02d}" for i in range(1, 9)] +
                   [f"C{i:02d}" for i in range(1, 57)])
    concept_rows = []
    concept_csv_ok = False
    if concept_audit_path.is_file():
        import csv
        with concept_audit_path.open(newline="", encoding="utf-8") as source:
            concept_rows = list(csv.DictReader(source))
        headers = {key.strip().lower(): key for key in concept_rows[0]} if concept_rows else {}
        id_key = next((headers[key] for key in ("id", "concept_id", "concept") if key in headers), "")
        actual_ids = [row.get(id_key, "").strip() for row in concept_rows] if id_key else []
        concept_csv_ok = len(actual_ids) == 64 and len(set(actual_ids)) == 64 and set(actual_ids) == set(concept_ids)
    record(checks, "64_concept_audit_unique_expected_ids", concept_csv_ok,
           {"count": 64, "unique": True, "ids": concept_ids},
           {"exists": concept_audit_path.is_file(), "row_count": len(concept_rows),
            "unique_count": len(set(actual_ids)),
            "ids_match": concept_csv_ok}, {"csv": str(concept_audit_path.relative_to(ROOT))})
    record(checks, "content_review_artifact_exists", content_review_path.is_file(), True,
           content_review_path.is_file(), {"path": str(content_review_path.relative_to(ROOT))})

    artifact_candidates = [
        DATA_TEX, DATA_PDF, MODEL_TEX, MODEL_PDF, concept_audit_path, content_review_path,
        ROOT / "README.md", ROOT / "01_DATA/README.md", ROOT / "02_MODEL/README.md",
        ROOT / "04_SYNTHESIS/README.md",
        ROOT / "me/README.md", OUT / "README.md",
        ROOT / "scripts/week2_data_audit.py", ROOT / "scripts/week2_model_audit.py",
        ROOT / "scripts/week2_verify_delivery.py", ROOT / "me/build_sources.py",
        ROOT / "me/Week_2_Data_Foundations.tex",
        OUT / "data_findings.md", OUT / "model_findings.md",
        OUT / "data_audit.json", OUT / "data_support_comparison.json", OUT / "model_audit.json",
        OUT / "week1_preservation_verification.json", OUT / "final_verification.json",
        OUT / "visual_review.json",
        OUT / "data_aligned_overlap_pairs.csv", OUT / "model_confusion_matrix.csv",
        OUT / "model_layer_shapes.csv", OUT / "model_per_class_metrics.csv",
        OUT / "model_supervised_test_rows.csv", OUT / "model_confusion_matrix.png",
        OUT / "model_top_two_margins.png", OUT / "model_layer_macs.png",
    ]
    artifacts = []
    for path in artifact_candidates:
        if path.is_file():
            artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "scope": "Week 2 delivery artifact inventory; hashes exclude document_checks.json and delivery_manifest.json themselves",
        "artifacts": artifacts,
        "excluded_outputs": [str(CHECKS_PATH.relative_to(ROOT)), str(MANIFEST_PATH.relative_to(ROOT))],
    }
    manifest_path_contract = all(path not in {CHECKS_PATH, MANIFEST_PATH} for path in artifact_candidates)
    record(checks, "manifest_excludes_its_own_outputs", manifest_path_contract,
           "document_checks.json and delivery_manifest.json are not hashed", True)

    all_actual_pass = all(item["status"] == "PASS" for item in checks)
    finished = datetime.now(timezone.utc).isoformat()
    document_checks = {
        "schema_version": 1,
        "started_at_utc": started,
        "completed_at_utc": finished,
        "command": ".venv/bin/python scripts/week2_verify_delivery.py",
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "pdftotext": shutil.which("pdftotext"), "pdfinfo": shutil.which("pdfinfo")},
        "pdf_page_counts_observed": page_counts,
        "pdf_page_count_policy": "positive page count is required; current page counts are recorded dynamically rather than hard-coded",
        "identity_presence_audit": identity_presence,
        "markdown_link_results": link_results,
        "checks": checks,
        "pending": pending,
        "overall_status": "PASS" if all_actual_pass else "FAIL",
    }
    (CHECKS_PATH).write_text(json.dumps(document_checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document_checks, manifest


def main() -> int:
    checks, manifest = verify_delivery()
    print(json.dumps({
        "overall_status": checks["overall_status"],
        "check_count": len(checks["checks"]),
        "failed_checks": [c["name"] for c in checks["checks"] if c["status"] != "PASS"],
        "pending_count": len(checks["pending"]),
        "manifest_artifact_count": len(manifest["artifacts"]),
        "checks_path": str(CHECKS_PATH.relative_to(ROOT)),
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
    }, indent=2))
    return 0 if checks["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
