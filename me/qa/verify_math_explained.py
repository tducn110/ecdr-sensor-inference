#!/usr/bin/env python3
"""Check companion examples independently and render the standalone PDF.

Run with the project's .venv Python. Writes only me/qa/math_explained/.
Finite checks validate arithmetic; universal claims still rely on the
analytic proofs and stated assumptions in the document.
"""
from pathlib import Path
from fractions import Fraction as Q
from itertools import product
from PIL import Image, ImageDraw
import hashlib
import json
import math
import re
import subprocess

ME = Path(__file__).resolve().parents[1]
OUT = ME / "qa" / "math_explained"
OUT.mkdir(exist_ok=True)
STEM = "ECDR_Mathematics_Explained"
records = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(x):
    return [(max(Q(0), x[i] - x[i + 1] + x[i + 2] - 2),
             max(Q(0), -x[i] + x[i + 2] - 1)) for i in range(len(x) - 2)]


def logits(x):
    h = features(x)
    m = tuple(sum(v[k] for v in h) / len(h) for k in range(2))
    return (m[0] - m[1] + 4, -m[0] + 1, 2 * m[0] - 2 * m[1])


def margins(x):
    f = logits(x)
    return (f[0] - f[1], f[0] - f[2])


x = tuple(map(Q, (1, 2, 3, 4, 5, 6)))
xp = tuple(map(Q, (3, 4, 5, 6, 7, 8)))
xpp = tuple(map(Q, (5, 6, 7, 8, 7, 8)))
assert features(x) == [(0, 1), (1, 1), (2, 1), (3, 1)]
assert features(xp) == [(2, 1), (3, 1), (4, 1), (5, 1)]
assert logits(x) == (Q(9, 2), Q(-1, 2), Q(1))
assert logits(xp) == (Q(13, 2), Q(-5, 2), Q(5))
assert margins(x) == (Q(5), Q(7, 2))
assert margins(xp) == (Q(9), Q(3, 2))
assert features(xpp) == [(4, 1), (5, 1), (4, 0), (7, 0)]
assert logits(xpp) == (Q(17, 2), Q(-4), Q(9))
assert margins(xpp)[1] == Q(-1, 2)
corners = []
for u, v in product((Q(13, 2), Q(15, 2)), (Q(15, 2), Q(17, 2))):
    query = (Q(3), Q(4), Q(5), Q(6), u, v)
    actual = margins(query)
    assert actual == ((35 + v - u) / 4, (u - 1) / 4)
    corners.append({"u": u, "v": v, "margins": actual})
mins = tuple(min(c["margins"][k] for c in corners) for k in range(2))
maxs = tuple(max(c["margins"][k] for c in corners) for k in range(2))
assert mins == (Q(35, 4), Q(11, 8))
assert maxs == (Q(37, 4), Q(13, 8))
records["toy_cnn"] = {"old_features": features(x), "new_features": features(xp),
    "old_logits": logits(x), "new_logits": logits(xp),
    "old_margins": margins(x), "new_margins": margins(xp),
    "wave_box_corners": corners, "affine_exact_minima": mins,
    "affine_exact_maxima": maxs,
    "position_interval_lower_margins": [Q(8), Q(7, 8)],
    "sum_coordinate_interval_lower_margins": [Q(17, 2), Q(9, 8)],
    "stale_certificate_second_step_features": features(xpp),
    "stale_certificate_second_step_logits": logits(xpp),
    "scope": "Corner enumeration confirms affine-example arithmetic, after proving all relevant ReLUs stable on this box. It is not a generic neural-network verifier."}

affine_values = [2 * a - 3 * b + 1 for a, b in product((1, 2), (-1, 3))]
assert min(affine_values) == -6 and max(affine_values) == 8
records["affine_interval"] = {"min": min(affine_values), "max": max(affine_values)}
for i in range(81):
    t = Q(i, 40) - 1
    actual = Q(3, 2) + 2 * max(Q(0), t) - max(Q(0), t + 1)
    assert actual == Q(1, 2) + abs(t)
for alpha in (Q(0), Q(1, 2), Q(1)):
    lower = Q(1, 2) - abs(2 * alpha - 1)
    assert lower == (Q(1, 2) if alpha == Q(1, 2) else Q(-1, 2))
records["crown"] = {"exact_min": Q(1, 2), "independent_ibp_lower": Q(-1, 2),
    "alpha_optimal": Q(1, 2), "optimized_affine_lower": Q(1, 2),
    "sampled_arithmetic_points": 81, "scope": "Analytic proof, not sampling, establishes the whole-domain bound."}

f1 = (Q(16, 19) + Q(6, 7)) / 2
assert f1 == Q(113, 133)
records["metrics"] = {"accuracy": Q(17, 20), "macro_f1": f1,
    "macro_f1_decimal": float(f1), "population_sd_2_4_6": math.sqrt(8 / 3),
    "sample_sd_2_4_6": 2}
assert 1 + (1000 - 128) // 64 == 14
assert list(range(2 * 64, 4 * 64 + 1, 64)) == [128, 192, 256]
records["window_count"] = {"T1000_W128_s64": 14, "span_100_400_starts": [128, 192, 256],
    "majority_count_threshold": math.ceil(.6 * 128), "dx": 6 * 128,
    "overlap_coordinates": 6 * 64, "time_span_s": 128 / 50,
    "first_to_last_s": 127 / 50, "stride_time_s": 64 / 50}
records["geometry"] = {"L2_Linf_L1_lower_for_a3m4_beta2_eps01": [1.5, 1.3, 1.6],
    "square_line_tube_minima": [Q(-3, 5), Q(1), Q(4, 5)],
    "pca_orthogonal_residual_lower": 1 - .1 * math.sqrt(2),
    "pca_query_score": .2 / math.sqrt(2), "pca_query_residual_norm": .1 * math.sqrt(2),
    "normalized_example": ((2.1 - 2) / .5), "raw_affine_example": 6 * 2.1 - 11}
records["probability"] = [{"q": q, "q_pow_768_if_independent": q ** 768,
    "q_pow_384_if_independent": q ** 384} for q in (.9, .99, .999)]
macs = 124 * 32 * 30 + 122 * 64 * 96 + 120 * 64 * 192 + 12 * 64
params = 32 * (30 + 1) + 64 * (96 + 1) + 64 * (192 + 1) + 12 * (64 + 1)
assert (macs, params) == (2343936, 20332)
records["cnn_complexity"] = {"params": params, "mac": macs, "approx_flop": 2 * macs,
    "R": 9, "n": 120, "shared_after_shift64": 56, "tail_input_samples": 72}
full = 100 * Q(4)
cost = 100 * Q(8, 100) + 20 * 4 + 20 * Q(3, 2)
assert cost == 118
records["cost"] = {"full_N100": full, "ecdr_H80_B20": cost,
    "speed_ratio": full / cost, "build1p5_threshold": Q(158, 100) / Q(11, 2),
    "build8_threshold": Q(808, 100) / 12,
    "amortization_L_threshold": Q(8) / Q(392, 100),
    "min_integer_all_hit_count": 3,
    "heterogeneous_forward_full": 20, "heterogeneous_forward_ecdr": 22}
records["numerical"] = {"example_safe_lower": .02 - .003 - .004,
    "example_inconclusive_lower": .005 - .003 - .004,
    "scope": "The error budgets are hypothetical proven bounds in the explanation, not measured numerical error limits."}

pdf = ME / (STEM + ".pdf")
tex = ME / (STEM + ".tex")
source = tex.read_text()
log = (ME / (STEM + ".log")).read_text(errors="replace")
for bad in ("Missing character", "Overfull", "undefined references", "undefined on input", "! LaTeX Error"):
    assert bad.lower() not in log.lower(), bad
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source))
labels = set(re.findall(r"\\label\{([^}]+)\}", source))
assert refs <= labels
cites = set()
for block in re.findall(r"\\cite\{([^}]+)\}", source):
    cites.update(block.split(","))
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", source))
assert cites <= bib
assert len(re.findall(r"^H\d{2} &", source, re.M)) == 24
assert len(re.findall(r"^M\d{2} &", source, re.M)) == 29
main = source.split(r"\appendix")[0]
assert len(re.findall(r"\\section\{", main)) == 24
metadata = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
pages = int(re.search(r"^Pages:\s+(\d+)", metadata, re.M).group(1))
subprocess.run(["pdftotext", "-layout", str(pdf), str(OUT / "document.txt")], check=True)
text = (OUT / "document.txt").read_text()
assert "??" not in text
assert "toán học" in text and "chứng nhận" in text
for old in OUT.glob("page-*.png"):
    old.unlink()
for old in OUT.glob("contact_*.jpg"):
    old.unlink()
subprocess.run(["pdftoppm", "-r", "88", "-png", str(pdf), str(OUT / "page")], check=True)
pngs = sorted(OUT.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
assert len(pngs) == pages
for start in range(0, pages, 9):
    sheet = Image.new("RGB", (1308, 1914), "#d8dce0")
    draw = ImageDraw.Draw(sheet)
    for j, p in enumerate(pngs[start:start + 9]):
        im = Image.open(p).convert("RGB")
        im.thumbnail((420, 594))
        xx, yy = 12 + (j % 3) * 432, 12 + (j // 3) * 634
        draw.text((xx + 4, yy + 3), f"Math explained: PDF page {start + j + 1}", fill="black")
        sheet.paste(im, (xx + (420 - im.width) // 2, yy + 24))
    sheet.save(OUT / f"contact_{start // 9 + 1}.jpg", quality=90)
preserved = json.loads((ME / "qa" / "artifact_checks.json").read_text())["documents"]
assert all(digest(ME / (d["stem"] + ".tex")) == d["source_sha256"]
           and digest(ME / (d["stem"] + ".pdf")) == d["pdf_sha256"] for d in preserved)
result = {"scope": "Companion arithmetic + LaTeX/PDF QA; no ECDR training, verifier or benchmark.",
    "source_sha256": digest(tex), "pdf_sha256": digest(pdf), "pdf_pages": pages,
    "main_sections": 24, "coverage_groups": 29, "hypothesis_explanations": 24,
    "citations_resolved": len(cites), "cross_references_resolved": len(refs),
    "previous_report_sources_and_pdfs_unchanged": True, "examples": records,
    "visual_review": "Separate agent inspection of rendered contact sheets is required."}
(OUT / "checks.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                           default=lambda x: str(x) if isinstance(x, Q) else x) + "\n")
print(json.dumps({"pdf_pages": pages, "main_sections": 24, "coverage_groups": 29,
                  "hypothesis_explanations": 24, "numeric_examples": "PASS",
                  "PDF_checks": "PASS; visual review separate", "old_reports_unchanged": True}, indent=2))
