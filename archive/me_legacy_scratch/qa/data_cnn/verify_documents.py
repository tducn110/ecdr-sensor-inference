#!/usr/bin/env python3
"""Recheck arithmetic, sources, PDFs and renders for the two new documents.

Run from repository root with .venv/bin/python. This is document QA and
reference arithmetic, not a HAPT CNN inference run or executable verifier.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
ROOT = REPORT.parent
STEMS = ["ECDR_Data_Analysis", "ECDR_CNN_Model_Analysis"]


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_data():
    data = json.loads((HERE / "data_analysis.json").read_text())
    t, integrity = data["totals"], data["integrity"]
    assert sum(t["label_policy_counts"].values()) == 17453
    assert t["unique_raw_time_samples"] == 1122772
    assert t["raw_sensor_coordinates"] == 6 * 1122772
    assert t["window_coordinates"] == 17453 * 128 * 6
    assert sum(t["grid_sample_multiplicity"].values()) == 1122772
    assert sum(int(k) * v for k, v in t["grid_sample_multiplicity"].items()) == 17453 * 128
    assert all(data["split_integrity"].values())
    for key in ["nonfinite_raw_coordinates", "window_raw_mismatch_coordinates",
                "label_policy_mismatch_windows", "saved_windows_nonfinite_coordinates",
                "annotation_conflicts", "overlapping_label_samples"]:
        assert integrity[key] == 0, key
    assert not integrity["unaligned"] and not integrity["acc_only"] and not integrity["gyro_only"]
    for rel, digest in data["source_files"].items():
        assert sha(ROOT / rel) == digest, f"Source drift: {rel}"
    labels = np.loadtxt(ROOT / "data/hapt/RawData/labels.txt", dtype=int)
    recs = {r["exp"]: r for r in data["recordings"]}
    enumeration, short, gridmiss, withgrid = 0, 0, 0, 0
    # Enumerate the recording's actual grid; do not call audit's floor/ceil code.
    for exp, subject, activity, a, b in labels:
        starts = np.arange(0, recs[int(exp)]["samples"] - 128 + 1, 64)
        count = int(((starts >= a) & (starts + 127 <= b)).sum())
        enumeration += count
        short += int(b-a+1 < 128)
        gridmiss += int(b-a+1 >= 128 and count == 0)
        withgrid += int(count > 0)
    assert (enumeration, short, gridmiss, withgrid) == (10361, 40, 91, 1083)
    assert sum(sum(r[s] for s in ["train", "val", "test"]) for r in data["pure_class_support"]) == enumeration
    assert data["adjacent_pairs"]["overlap_coordinates_unequal"] == 0
    assert data["adjacent_pairs"]["total"] == 17453 - 61
    assert sum(data["adjacent_pairs"][k] for k in ["same_activity_pure", "label_majority_change", "equal_nonpure_or_unlabeled"]) == 17392
    return {"source_hashes_current": True, "span_grid_enumeration": enumeration,
            "short_spans": short, "long_spans_without_grid_window": gridmiss,
            "spans_with_grid_windows": withgrid,
            "annotation_percent": 100 * 815614 / 1122772,
            "boundary_percent": 100 * 1273 / 17453,
            "independence_toy_joint_probability": .9 ** 768}


def check_cnn():
    lengths, rfs, params, macs = [], [], [], []
    channels, kernels, length, rf = [6, 32, 64, 64], [5, 3, 3], 128, 1
    for ci, co, kernel in zip(channels, channels[1:], kernels):
        length -= kernel - 1
        rf += kernel - 1
        lengths.append(length)
        rfs.append(rf)
        params.append(co * (ci * kernel + 1))
        macs.append(length * co * ci * kernel)
    params.append(12 * 65)
    macs.append(12 * 64)
    assert lengths == [124, 122, 120] and rfs == [5, 7, 9]
    assert sum(params) == 20332 and sum(macs) == 2343936
    hidden = [c*n for c, n in zip(channels[1:], lengths)]
    assert sum(hidden) == 19456
    assert (768 + sum(hidden) + 64 + 12)*4 == 81200
    assert (hidden[1] + hidden[2])*4 == 61952

    def toy(window):
        h = [(max(Q(0), a-b+c-2), max(Q(0), -a+c-1))
             for a, b, c in zip(window, window[1:], window[2:])]
        pooled = [sum(row[j] for row in h)/4 for j in range(2)]
        a, b = pooled
        logits = [a-b+4, -a+1, 2*a-2*b]
        return h, pooled, logits, [logits[0]-logits[j] for j in [1, 2]]

    old = toy(list(map(Q, [1, 2, 3, 4, 5, 6])))
    new = toy(list(map(Q, [3, 4, 5, 6, 7, 8])))
    stale = toy(list(map(Q, [5, 6, 7, 8, 7, 8])))
    assert old[2] == [Q(9, 2), Q(-1, 2), Q(1)]
    assert new[2] == [Q(13, 2), Q(-5, 2), Q(5)]
    assert stale[2] == [Q(17, 2), Q(-4), Q(9)]
    corners = [toy([Q(3), Q(4), Q(5), Q(6), u, v])
               for u in [Q(13, 2), Q(15, 2)] for v in [Q(15, 2), Q(17, 2)]]
    minima = [min(out[3][j] for out in corners) for j in range(2)]
    assert minima == [Q(35, 4), Q(11, 8)]
    # The document proves stable ReLU branches and an affine objective on
    # this box. The corner check verifies arithmetic, not a generic proof.
    return {"lengths": lengths, "receptive_fields": rfs, "parameters": sum(params),
            "macs": sum(macs), "two_flop_per_mac": 2*sum(macs),
            "relu_scalars": sum(hidden), "gap_additions": 64*119,
            "weight_payload_float32_bytes": 20332*4,
            "listed_activation_payload_bytes": 81200,
            "two_buffer_payload_bytes": 61952,
            "toy_minima": list(map(str, minima)),
            "toy_stale_logits": list(map(str, stale[2]))}


def previous_unchanged():
    original = json.loads((REPORT / "qa/artifact_checks.json").read_text())
    for row in original["documents"]:
        for suffix, field in [("tex", "source_sha256"), ("pdf", "pdf_sha256")]:
            assert sha(REPORT / (row["stem"] + "." + suffix)) == row[field]
    math = json.loads((REPORT / "qa/math_explained/checks.json").read_text())
    assert sha(REPORT / "ECDR_Mathematics_Explained.tex") == math["source_sha256"]
    assert sha(REPORT / "ECDR_Mathematics_Explained.pdf") == math["pdf_sha256"]
    return True


def one_pdf(stem):
    tex, pdf = REPORT / (stem + ".tex"), REPORT / (stem + ".pdf")
    source = tex.read_text()
    log = (REPORT / (stem + ".log")).read_text(errors="replace")
    for forbidden in ["Overfull", "Missing character", "undefined references",
                      "undefined on input", "! LaTeX Error", "! Undefined control"]:
        assert forbidden.lower() not in log.lower(), (stem, forbidden)
    cites = {v for block in re.findall(r"\\cite\{([^}]+)\}", source) for v in block.split(",")}
    bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", source))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source))
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    assert cites <= bibs and refs <= labels
    assert not re.search(r"\b(?:TODO|TBD|PLACEHOLDER)\b", source)
    pages = int(re.search(r"^Pages:\s+(\d+)", run("pdfinfo", str(pdf)), re.M).group(1))
    text_path = HERE / (stem + ".txt")
    run("pdftotext", "-layout", str(pdf), str(text_path))
    text = text_path.read_text()
    assert "??" not in text and "\ufffd" not in text
    assert "chứng" in text and "dữ liệu" in text
    text_pages = text.split("\f")[:pages]
    assert len(text_pages) == pages and all(len(p.strip()) > 70 for p in text_pages)
    fonts = run("pdffonts", str(pdf)).splitlines()[2:]
    assert fonts and all(re.search(r"\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", f) for f in fonts)
    bbox_path = HERE / (stem + "_bbox.html")
    run("pdftotext", "-bbox", str(pdf), str(bbox_path))
    raw_bbox = bbox_path.read_text()
    # Some legacy TeX math glyphs have control-code ToUnicode mappings in
    # Poppler's bbox extraction. Keep all word coordinates, remove only those
    # invalid XML text codepoints for parsing, and report their count. Their
    # visible mathematical glyphs must be checked in the rendered pages.
    control_glyphs = re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", raw_bbox)
    xml = ET.fromstring(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_bbox))
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    outer_words = []
    for i, p in enumerate(xml.findall(".//h:page", ns), 1):
        width, height = float(p.attrib["width"]), float(p.attrib["height"])
        for w in p.findall("h:word", ns):
            a = {k:float(w.attrib[k]) for k in ["xMin", "xMax", "yMin", "yMax"]}
            if a["xMin"] < 20 or a["xMax"] > width-20 or a["yMin"] < 20 or a["yMax"] > height-15:
                outer_words.append([i, w.text, a])
    assert not outer_words, (stem, outer_words[:10])
    render = HERE / stem
    render.mkdir(exist_ok=True)
    for old in render.glob("page-*.png"):
        old.unlink()
    for old in HERE.glob(stem + "_contact_*.jpg"):
        old.unlink()
    run("pdftoppm", "-r", "90", "-png", str(pdf), str(render / "page"))
    imgs = sorted(render.glob("page-*.png"), key=lambda p:int(p.stem.split("-")[-1]))
    assert len(imgs) == pages
    sheets = []
    for start in range(0, pages, 9):
        tw, th, gap = 420, 622, 12
        canvas = Image.new("RGB", (3*tw+4*gap, 3*th+4*gap), "#d8dce0")
        draw = ImageDraw.Draw(canvas)
        for i, path in enumerate(imgs[start:start+9]):
            im = Image.open(path).convert("RGB")
            im.thumbnail((tw, th-24))
            x, y = gap+(i%3)*(tw+gap), gap+(i//3)*(th+gap)
            draw.text((x+4,y+3), f"PDF page {start+i+1}: {stem}", fill="black")
            canvas.paste(im, (x+(tw-im.width)//2,y+24))
        out = HERE / (stem + f"_contact_{start//9+1}.jpg")
        canvas.save(out, quality=90)
        sheets.append(str(out.relative_to(REPORT)))
    return {"stem": stem, "pages": pages, "pdf_sha256": sha(pdf), "tex_sha256": sha(tex),
            "citations_resolved": len(cites), "references_resolved": len(refs),
            "all_fonts_embedded": True, "empty_pages": [], "outer_words": outer_words,
            "bbox_extractor_control_math_glyphs": len(control_glyphs),
            "contact_sheets": sheets,
            "visual_inspection": "Review all contact sheets and detail pages separately."}


if __name__ == "__main__":
    results = {"scope": "Fresh data audit + theoretical CNN arithmetic + document QA; no HAPT inference or ECDR benchmark.",
               "data": check_data(), "cnn": check_cnn(),
               "five_previous_sources_and_pdfs_unchanged": previous_unchanged()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results["documents"] = list(pool.map(one_pdf, STEMS))
    (HERE / "checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False, indent=2))
