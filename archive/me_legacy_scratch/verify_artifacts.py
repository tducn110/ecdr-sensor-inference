#!/usr/bin/env python3
"""Validate exported research documents and render them for visual QA.

Requires Poppler and Pillow. Uses the project environment; writes only me/qa.
This checks document integrity and numeric artifact consistency, not neural
certificate soundness or empirical ECDR usefulness.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa"
STEMS = ["ECDR_Weeks_1_2_3_Theory", "Week_1_Project_Brief",
         "Week_2_Data_Foundations", "Week_3_Mathematical_Foundations"]

try:
    from build_sources import extract_canonical_week1_body
except ImportError:
    import sys
    sys.path.insert(0, str(ROOT))
    from build_sources import extract_canonical_week1_body


def run(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def strip_generated_comment(text: str) -> str:
    """Ignore only leading English GENERATED comment line(s) without ignoring body content."""
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith("% GENERATED"):
        lines.pop(0)
    return "".join(lines)


def resolve_input_path(base_file, rel_target):
    """Resolve an \\input path relative to canonical source dir or actual TeX working dir."""
    canonical_dir = base_file.resolve().parent
    working_dir = base_file.parent
    cands = [rel_target]
    if not rel_target.endswith(".tex"):
        cands.append(rel_target + ".tex")
    for cand in cands:
        for search_dir in (canonical_dir, working_dir, ROOT, ROOT.parent):
            p = (search_dir / cand).resolve()
            if p.is_file():
                return p
    return None


def collect_source_and_dependencies(entry_file):
    """Recursively collect full source content and tracked \\input dependencies.

    Guards against cycles and reports missing inputs. Does not treat dependency
    content as standalone PDF artifacts.
    """
    visited = set()
    dependencies = []
    missing_inputs = []

    def _recurse(curr_file):
        curr_resolved = curr_file.resolve()
        if curr_resolved in visited:
            return ""
        visited.add(curr_resolved)
        try:
            content = curr_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            missing_inputs.append(str(curr_file))
            return ""

        parts = [content]
        for inp in re.findall(r"\\input\{([^}]+)\}", content):
            inp_clean = inp.strip()
            dep = resolve_input_path(curr_file, inp_clean)
            if dep is None:
                missing_inputs.append(inp_clean)
            else:
                if dep not in dependencies:
                    dependencies.append(dep)
                if dep.resolve() not in visited:
                    parts.append(_recurse(dep))
        return "\n".join(parts)

    full_source = _recurse(entry_file)
    return full_source, dependencies, missing_inputs


def one(stem):
    QA.mkdir(exist_ok=True)
    pdf = ROOT / f"{stem}.pdf"
    
    # Canonical entry paths: for Week 1, canonical compiled entry is 00_OVERVIEW/02_Week_1_Project_Brief.tex
    # and its log sibling 00_OVERVIEW/02_Week_1_Project_Brief.log.
    # The me/ wrapper is tracked separately as a compatibility wrapper (not compiled entry).
    compat_info = None
    generated_body_info = None

    if stem == "Week_1_Project_Brief":
        canonical_tex = ROOT.parent / "00_OVERVIEW" / "02_Week_1_Project_Brief.tex"
        canonical_log = ROOT.parent / "00_OVERVIEW" / "02_Week_1_Project_Brief.log"
        tex = canonical_tex
        log_file = canonical_log if canonical_log.exists() else (ROOT / f"{stem}.log")
        compat_wrapper = ROOT / f"{stem}.tex"
        if compat_wrapper.exists():
            compat_info = {
                "compat_wrapper_path": str(compat_wrapper),
                "compat_wrapper_sha256": hashlib.sha256(compat_wrapper.read_bytes()).hexdigest(),
                "note": "Physical compatibility wrapper in me/; not the canonical compiled entry.",
            }

        # Assert shared generated body equals canonical marker extract (ignoring only leading GENERATED comments)
        generated_body_file = canonical_tex.parent / "week1" / "brief_body.tex"
        if generated_body_file.exists():
            gen_text = generated_body_file.read_text(encoding="utf-8")
            gen_body_stripped = strip_generated_comment(gen_text)
            canonical_text = canonical_tex.read_text(encoding="utf-8")
            canonical_extract = extract_canonical_week1_body(canonical_text)
            assert gen_body_stripped == canonical_extract, (
                stem,
                "Shared generated body does not match canonical marker extract"
            )
            generated_body_info = {
                "path": str(generated_body_file),
                "sha256": hashlib.sha256(generated_body_file.read_bytes()).hexdigest(),
                "matches_canonical_extract": True,
                "role": "derivative_extract_for_combined_master",
            }
    else:
        tex = ROOT / f"{stem}.tex"
        log_file = ROOT / f"{stem}.log"

    log = log_file.read_text(errors="replace") if log_file.exists() else ""
    
    # Collect source and input dependencies with recursion/cycle guards
    full_source, dependencies, missing_inputs = collect_source_and_dependencies(tex)
    assert not missing_inputs, (stem, f"Missing \\input dependencies: {missing_inputs}")

    # Track/hash generated body dependency for Week 1 even if standalone canonical source inlines body without \input
    if stem == "Week_1_Project_Brief":
        gen_body_dep = tex.parent / "week1" / "brief_body.tex"
        if gen_body_dep.exists() and gen_body_dep not in dependencies:
            dependencies.append(gen_body_dep)

    # Dependency hash for artifact identity and readback checks
    if dependencies:
        dep_hasher = hashlib.sha256()
        for d in sorted(dependencies, key=lambda p: str(p)):
            dep_hasher.update(d.read_bytes())
        dependency_hash = dep_hasher.hexdigest()
    else:
        dependency_hash = None

    metadata = run(["pdfinfo", str(pdf)])
    n = int(re.search(r"^Pages:\s+(\d+)", metadata, re.M).group(1))
    textpath = QA / f"{stem}.txt"
    run(["pdftotext", "-layout", str(pdf), str(textpath)])
    text = textpath.read_text()
    problems = []
    for token in ("Missing character", "Overfull", "undefined references",
                  "undefined on input", "! LaTeX Error", "! Undefined control"):
        if token.lower() in log.lower():
            problems.append(token)
            
    # Include \input-body dependencies in source content used for references and citations
    cites = set()
    for block in re.findall(r"\\cite(?:\[[^]]*\])?\{([^}]+)\}", full_source):
        cites.update(block.split(","))
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", full_source))
    missing_cites = sorted(cites - bibliography)
    labels = set(re.findall(r"\\label\{([^}]+)\}", full_source))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", full_source))
    missing_refs = sorted(refs - labels)
    assert not problems, (stem, problems)
    assert not missing_cites, (stem, missing_cites)
    assert not missing_refs, (stem, missing_refs)
    assert "??" not in text, (stem, "unresolved PDF marker")
    
    # English Week 1 contract vs. legacy Vietnamese checks for other 3 PDFs
    if stem == "Week_1_Project_Brief":
        assert "Project Brief" in text and "Research" in text, (stem, "English contract validation")
    else:
        assert "nghiên cứu" in text and "thuyết" in text, (stem, "Vietnamese extraction")
        
    pages = text.split("\f")
    empty = [i + 1 for i, p in enumerate(pages[:n]) if len(p.strip()) < 30]
    assert not empty, (stem, empty)
    renderdir = QA / stem
    renderdir.mkdir(exist_ok=True)
    # A new PDF may have fewer pages after a layout repair. Remove only this
    # script's previously generated page images before rendering the new PDF.
    for old in renderdir.glob("page-*.png"):
        old.unlink()
    for old in QA.glob(f"{stem}_contact_*.jpg"):
        old.unlink()
    run(["pdftoppm", "-r", "85", "-png", str(pdf), str(renderdir / "page")])
    images = sorted(renderdir.glob("page-*.png"),
                    key=lambda p: int(p.stem.split("-")[-1]))
    assert len(images) == n
    sheets = []
    for start in range(0, n, 9):
        subset = images[start:start + 9]
        tile_w, tile_h, gap = 420, 622, 12
        sheet = Image.new("RGB", (3 * tile_w + 4 * gap, 3 * tile_h + 4 * gap), "#d8dce0")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(subset):
            im = Image.open(path).convert("RGB")
            im.thumbnail((tile_w, tile_h - 24))
            x = gap + (i % 3) * (tile_w + gap)
            y = gap + (i // 3) * (tile_h + gap)
            draw.text((x + 4, y + 3), f"{stem}: PDF page {start + i + 1}", fill="black")
            sheet.paste(im, (x + (tile_w - im.width) // 2, y + 24))
        out = QA / f"{stem}_contact_{start // 9 + 1}.jpg"
        sheet.save(out, quality=90)
        sheets.append(str(out.relative_to(ROOT)))
    res = {"stem": stem, "source_path": str(tex), "log_path": str(log_file),
           "pdf_pages": n, "pdf_bytes": pdf.stat().st_size,
           "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
           "source_sha256": hashlib.sha256(tex.read_bytes()).hexdigest(),
           "dependency_hash": dependency_hash,
           "dependencies_sha256": dependency_hash,
           "dependency_files": [str(d) for d in sorted(dependencies, key=lambda p: str(p))],
           "citations_resolved": len(cites), "references_resolved": len(refs),
           "forbidden_compile_diagnostics": problems,
           "empty_pages": empty, "contact_sheets": sheets,
           "visual_review": "Required human/agent inspection of contact sheets and detailed pages; not inferred from these checks."}
    if generated_body_info is not None:
        res["generated_body"] = generated_body_info
    if compat_info is not None:
        res["compat_wrapper"] = compat_info
    return res


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate exported research documents and render them for visual QA."
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        choices=STEMS,
        default=STEMS,
        help="Document stems to verify and render (default: all legacy stems).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=QA / "artifact_checks.json",
        help="Path to write the verification JSON summary (default: me/qa/artifact_checks.json).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    QA.mkdir(exist_ok=True)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    max_workers = min(4, len(args.stems)) if args.stems else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        documents = list(pool.map(one, args.stems))

    # Existing numeric assertions from calculations.json (LEGACY master/data/math snapshot)
    calc = json.loads((ROOT / "calculations.json").read_text())
    data = calc["current_data_audit"]
    assert data["exact_raw_overlap"]["failed_pairs"] == 0
    assert data["array_shape"] == [17453, 6, 128]
    assert calc["theoretical"]["cnn"]["parameters_total"] == 20332
    assert calc["theoretical"]["cnn"]["conv_and_head_macs"] == 2343936

    legacy_hypotheses = re.findall(
        r"^H\d{2}(?= &)", (ROOT / f"{STEMS[0]}.tex").read_text(), re.M
    )
    assert len(legacy_hypotheses) == 24

    legacy_math_snapshot = {
        "scope": "LaTeX/PDF artifact QA, legacy master numeric consistency and data overlap audit; no ECDR benchmark or formal executable verification. Note: 20332 parameters and 24 hypotheses belong to legacy master/data/math snapshot, not new Week 1 Small CNN K12.",
        "calculation_checks": "PASS",
        "legacy_master_parameters_total": calc["theoretical"]["cnn"]["parameters_total"],
        "legacy_master_hypotheses_count": len(legacy_hypotheses),
        "legacy_master_hypotheses": legacy_hypotheses,
    }

    checks = {
        "scope": "LaTeX/PDF artifact QA, numeric consistency and data overlap audit; no ECDR benchmark or formal executable verification.",
        "documents": documents,
        "calculation_checks": "PASS",
        "hypotheses_in_master": legacy_hypotheses,
        "legacy_master_hypotheses_count": len(legacy_hypotheses),
        "legacy_math_snapshot_checks": legacy_math_snapshot,
    }

    output_json.write_text(json.dumps(checks, indent=2) + "\n")
    print(
        json.dumps(
            {
                "documents": [
                    {"name": d["stem"], "pages": d["pdf_pages"]} for d in documents
                ],
                "legacy_master_hypotheses_count": len(legacy_hypotheses),
                "status": "PASS; visual review still separate",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
