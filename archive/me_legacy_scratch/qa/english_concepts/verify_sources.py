#!/usr/bin/env python3
"""Verify English source coverage and XDV-only physical pagination.

This script never generates a PDF or opens an editor tab. Run with --compile
on the first use and after source changes; later runs verify the saved receipt.
"""
from fractions import Fraction as Q
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
DOCUMENTS = [("ECDR_Mathematics_Concepts_EN", "math"),
             ("ECDR_CNN_Model_Analysis", "cnn")]
PROVENANCE = HERE / "typesetting_provenance.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_xdv(executable):
    """Bind every successful XDV/log pair to the exact saved source bytes."""
    compiler = shutil.which(executable)
    if not compiler:
        raise SystemExit(f"Tectonic executable not found: {executable}")
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    PROVENANCE.unlink(missing_ok=True)
    records = {}
    for stem, folder in DOCUMENTS:
        source = REPORT / (stem + ".tex")
        digest = sha(source)
        output = HERE / folder
        output.mkdir(parents=True, exist_ok=True)
        command = [compiler, "-X", "compile", "--outfmt", "xdv",
                   "--keep-intermediates", "--keep-logs", "--untrusted",
                   "--outdir", str(output), str(source)]
        with (output / "compiler_stdout.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        assert sha(source) == digest, (stem, "source changed during compilation")
        records[stem] = {"source_sha256": digest,
                         "log_sha256": sha(output / (stem + ".log")),
                         "xdv_sha256": sha(output / (stem + ".xdv")),
                         "command": command}
    PROVENANCE.write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "compiler": compiler, "version": version,
        "tectonic_cache_dir": os.environ.get("TECTONIC_CACHE_DIR"),
        "outfmt": "xdv", "documents": records}, indent=2))


def one(stem, folder, expected):
    path = REPORT / (stem + ".tex")
    receipt = json.loads(PROVENANCE.read_text())["documents"][stem]
    assert sha(path) == receipt["source_sha256"], (stem, "source newer than typesetting receipt; rerun --compile")
    assert sha(HERE / folder / (stem + ".log")) == receipt["log_sha256"], (stem, "log changed after compilation")
    assert sha(HERE / folder / (stem + ".xdv")) == receipt["xdv_sha256"], (stem, "XDV changed after compilation")
    source = path.read_text()
    log = (HERE / folder / (stem + ".log")).read_text(errors="replace")
    ids = re.findall(r"\\concept\{([A-Z]\d{2})\}\{", source)
    assert ids == expected, (stem, "concept inventory")
    assert r"\usepackage[english]{babel}" in source
    assert not re.search(r"nghiên cứu|định nghĩa|chứng minh|tiếng Việt|Mệnh đề|Tài liệu", source, re.I)
    assert not re.search(r"\bTODO\b|\bTBD\b|\bPLACEHOLDER\b", source)
    starts = re.findall(r"ECDR-CONCEPT-START ([A-Z]\d+) PAGE=(\d+)", log)
    ends = dict(re.findall(r"ECDR-CONCEPT-END ([A-Z]\d+) PAGE=(\d+)", log))
    assert [i for i,p in starts] == expected
    assert len(ends) == len(expected)
    assert all(ends[i] == page for i,page in starts), (stem, "concept spills onto a second page")
    physical = [int(p) for i,p in starts]
    assert all(b-a == 1 for a,b in zip(physical,physical[1:])), (stem, "page gap or overlap")
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    labels.update("concept:" + i for i in expected)
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source))
    refs.update(re.findall(r"\\hyperref\[([^]]+)\]", source))
    assert refs <= labels, refs-labels
    citations = {c for group in re.findall(r"\\cite\{([^}]+)\}", source) for c in group.split(",")}
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", source))
    assert citations <= bibliography, citations-bibliography
    bad = [word for word in ["Overfull", "Missing character", "undefined references",
           "undefined on input", "! LaTeX Error", "! Undefined control sequence"]
           if word.lower() in log.lower()]
    assert not bad, (stem, bad)
    match = re.search(r"Output written on .*?\.xdv \((\d+) pages", log)
    assert match
    return {"stem":stem, "source_sha256":sha(path), "language":"English",
            "concept_count":len(ids), "xdv_typeset_pages":int(match[1]),
            "every_concept_exactly_one_page":True, "forbidden_diagnostics":bad,
            "resolved_citation_keys":len(citations), "resolved_cross_reference_targets":len(refs),
            "concept_pages":{i:int(page) for i,page in starts},
            "validation":"Local Tectonic XDV only; no exported or visually reviewed new PDF."}


def protected_files():
    baseline = json.loads((REPORT / "qa/artifact_checks.json").read_text())
    for item in baseline["documents"]:
        assert sha(REPORT/(item["stem"]+".tex")) == item["source_sha256"]
        assert sha(REPORT/(item["stem"]+".pdf")) == item["pdf_sha256"]
    old_math = json.loads((REPORT / "qa/math_explained/checks.json").read_text())
    assert sha(REPORT/"ECDR_Mathematics_Explained.tex") == old_math["source_sha256"]
    assert sha(REPORT/"ECDR_Mathematics_Explained.pdf") == old_math["pdf_sha256"]
    previous = json.loads((REPORT / "qa/data_cnn/checks.json").read_text())
    old_data, old_cnn = previous["documents"]
    assert old_data["stem"] == "ECDR_Data_Analysis"
    assert sha(REPORT/"ECDR_Data_Analysis.tex") == old_data["tex_sha256"]
    assert sha(REPORT/"ECDR_Data_Analysis.pdf") == old_data["pdf_sha256"]
    assert sha(REPORT/"ECDR_CNN_Model_Analysis_VI.tex") == old_cnn["tex_sha256"]
    assert sha(REPORT/"ECDR_CNN_Model_Analysis_VI.pdf") == old_cnn["pdf_sha256"]
    return {"original_four_week_reports_unchanged":True,
            "previous_Vietnamese_mathematics_unchanged":True,
            "data_report_source_and_pdf_unchanged":True,
            "Vietnamese_CNN_source_and_pdf_preserved_exactly":True}


def arithmetic():
    lengths = [128-5+1,124-3+1,122-3+1]
    params = 32*(6*5+1)+64*(32*3+1)+64*(64*3+1)+12*65
    mac = 124*32*6*5+122*64*32*3+120*64*64*3+12*64
    assert (lengths,params,mac) == ([124,122,120],20332,2343936)
    def toy(x):
        h=[(max(Q(0),a-b+c-2),max(Q(0),-a+c-1)) for a,b,c in zip(x,x[1:],x[2:])]
        a,b=[sum(row[j] for row in h)/4 for j in range(2)]
        return [a-b+4,-a+1,2*a-2*b]
    assert toy(list(map(Q,[1,2,3,4,5,6]))) == [Q(9,2),Q(-1,2),Q(1)]
    assert toy(list(map(Q,[3,4,5,6,7,8]))) == [Q(13,2),Q(-5,2),Q(5)]
    assert toy(list(map(Q,[5,6,7,8,7,8]))) == [Q(17,2),Q(-4),Q(9)]
    corners=[toy([Q(3),Q(4),Q(5),Q(6),u,v]) for u in [Q(13,2),Q(15,2)] for v in [Q(15,2),Q(17,2)]]
    minima=[min(f[0]-f[j] for f in corners) for j in [1,2]]
    assert minima == [Q(35,4),Q(11,8)]
    return {"parameters":params,"MAC":mac,"lengths":lengths,
            "toy_exact_minima":[str(x) for x in minima],
            "note":"Arithmetic checks accompany, not replace, the stable-branch affine proof."}


def app_compiler_evidence():
    """Report saved current diagnostics; never hardcode a previous failure."""
    path = HERE / "app_compiler_results.json"
    entries = json.loads(path.read_text())
    documents = []
    for entry in entries:
        name = entry["document"]
        value = entry.get("value", {})
        blocks = [json.loads(c["text"]) for c in value.get("content", [])
                  if c.get("type") == "text"]
        documents.append({"document": name,
                          "source_matches": entry.get("source_sha256") == sha(REPORT / name),
                          "tool_is_error": value.get("isError", False),
                          "diagnostics": blocks})
    expected = {stem + ".tex" for stem, folder in DOCUMENTS}
    current = {d["document"] for d in documents} == expected and all(d["source_matches"] for d in documents)
    successful = current and all(not d["tool_is_error"] and d["diagnostics"]
                                and all(b.get("kind") == "success" for b in d["diagnostics"])
                                for d in documents)
    return {"status": "success" if successful else "failed_or_stale",
            "evidence_file": str(path.relative_to(REPORT)),
            "evidence_sha256": sha(path), "documents": documents}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", help="Compile XDV and save source/log/XDV provenance before checking")
    parser.add_argument("--tectonic", default="tectonic", help="Tectonic executable used with --compile")
    args = parser.parse_args()
    if args.compile:
        compile_xdv(args.tectonic)
    result = {"scope":"English source revision, concept coverage, arithmetic and XDV typesetting; no new PDF export or HAPT experiment.",
              "protected_files":protected_files(), "arithmetic":arithmetic(),
              "documents":[
                  one("ECDR_Mathematics_Concepts_EN","math",[f"F{i:02}" for i in range(1,47)]+[f"B{i:02}" for i in range(1,31)]+[f"C{i:02}" for i in range(1,29)]),
                  one("ECDR_CNN_Model_Analysis","cnn",[f"P{i:02}" for i in range(1,9)]+[f"C{i:02}" for i in range(1,57)])],
              "app_compiler":app_compiler_evidence(),
              "new_PDF_exported":False, "editor_or_other_tabs_opened":False}
    (HERE / "checks.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"documents":[{k:v for k,v in d.items() if k!="concept_pages"} for d in result["documents"]],"protected_files":result["protected_files"]},indent=2))
