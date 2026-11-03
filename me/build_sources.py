#!/usr/bin/env python3
"""Extract standalone weekly LaTeX reports and sync canonical extracts.

Canonical Week 1 authority is owned by 00_OVERVIEW/02_Week_1_Project_Brief.tex.
This script extracts the canonical Week 1 body to 00_OVERVIEW/week1/brief_body.tex
for combined master / compat wrappers. Week 2 has its own canonical source in
01_DATA; its compatibility wrapper must never be regenerated from the old master.
Only the historical Week 3 extract still comes from that master.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent

BEGIN_CANONICAL_WEEK1 = "% BEGIN CANONICAL WEEK1 BODY\n"
END_CANONICAL_WEEK1 = "% END CANONICAL WEEK1 BODY"
GENERATED_WEEK1_PREFIX = "% GENERATED from 00_OVERVIEW/02_Week_1_Project_Brief.tex. Do not edit.\n"


def extract_canonical_week1_body(text: str) -> str:
    """Extract Week 1 body from canonical source bounded by exact markers.

    Validates exactly one pair of markers in forward order. Fails clearly if missing
    or ambiguous; never falls back to regenerating Week 1 from legacy master.
    """
    begin_count = text.count(BEGIN_CANONICAL_WEEK1)
    end_count = text.count(END_CANONICAL_WEEK1)
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            f"Canonical Week 1 markers invalid: found {begin_count} begin and "
            f"{end_count} end markers (expected exactly 1 pair)."
        )
    parts = text.split(BEGIN_CANONICAL_WEEK1, 1)
    if END_CANONICAL_WEEK1 not in parts[1]:
        raise ValueError("Canonical Week 1 markers out of order.")
    return parts[1].split(END_CANONICAL_WEEK1, 1)[0]


def sync_generated_week1_body(canonical_path: Path, output_path: Path) -> bool:
    """Sync derivative week1/brief_body.tex from authoritative canonical source."""
    if not canonical_path.exists():
        raise FileNotFoundError(f"Authoritative canonical source missing: {canonical_path}")
    text = canonical_path.read_text(encoding="utf-8")
    body = extract_canonical_week1_body(text)
    content = GENERATED_WEEK1_PREFIX + body
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != content:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"Updated {output_path} from canonical Week 1 markers.")
        return True
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract standalone weekly LaTeX reports and sync canonical extracts."
    )
    parser.add_argument(
        "--week2-only",
        action="store_true",
        help="Sync only the Week 2 compatibility wrapper from its canonical 01_DATA source; preserve other weeks.",
    )
    parser.add_argument(
        "--week1-only",
        action="store_true",
        help="Sync generated Week 1 body extract from canonical authority and return before legacy Week 2/3 extraction.",
    )
    return parser.parse_args(argv)


def sync_week2_wrapper() -> None:
    canonical = ROOT.parent / "01_DATA" / "03_Week_2_Data_Foundations.tex"
    if not canonical.is_file():
        raise FileNotFoundError(f"Authoritative Week 2 source missing: {canonical}")
    wrapper = ROOT / "Week_2_Data_Foundations.tex"
    content = (
        "% GENERATED compatibility wrapper. Edit 01_DATA/03_Week_2_Data_Foundations.tex.\n"
        "\\IfFileExists{01_DATA/03_Week_2_Data_Foundations.tex}{%\n"
        "  \\input{01_DATA/03_Week_2_Data_Foundations.tex}%\n"
        "}{%\n"
        "  \\input{../01_DATA/03_Week_2_Data_Foundations.tex}%\n"
        "}\n"
    )
    if not wrapper.exists() or wrapper.read_text(encoding="utf-8") != content:
        wrapper.write_text(content, encoding="utf-8")
        print(f"Updated {wrapper} to reference canonical Week 2 source.")


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.week1_only and args.week2_only:
        raise ValueError("Choose only one of --week1-only and --week2-only")
    if args.week2_only:
        sync_week2_wrapper()
        return

    # 1. Sync generated Week 1 extract from authoritative canonical source
    canonical_file = ROOT.parent / "00_OVERVIEW" / "02_Week_1_Project_Brief.tex"
    generated_file = ROOT.parent / "00_OVERVIEW" / "week1" / "brief_body.tex"
    sync_generated_week1_body(canonical_file, generated_file)

    if args.week1_only:
        print("Week 1 body sync complete (--week1-only requested; returning before legacy Week 2/3 extraction).")
        return

    sync_week2_wrapper()
    # 2. Extract only historical Week 3 from monolithic master.
    MASTER = ROOT / "ECDR_Weeks_1_2_3_Theory.tex"
    text = MASTER.read_text(encoding="utf-8")
    preamble = text.split(r"\begin{document}", 1)[0]
    bibliography = text.split("% BEGIN BIBLIOGRAPHY\n", 1)[1].split(
        "% END BIBLIOGRAPHY", 1
    )[0]

    titles = {
        1: ("Week_1_Project_Brief", "Week 1: Project Brief và nền tảng nghiên cứu"),
        2: ("Week_2_Data_Foundations", "Week 2: Dữ liệu và quan hệ với ECDR"),
        3: ("Week_3_Mathematical_Foundations", "Week 3: Cơ sở toán học của ECDR"),
    }
    for week, (stem, title) in titles.items():
        if week == 2:
            print(f"Skipping {stem}.tex (canonical Week 2 wrapper preserved)")
            continue
        if week == 1:
            # Preserve canonical Week 1 owned 00_OVERVIEW/02_Week_1_Project_Brief.tex
            # and existing me/Week_1_Project_Brief.tex compatibility wrapper; never regenerate or overwrite.
            print(f"Skipping {stem}.tex (canonical Week 1 preserved)")
            continue
        content = text.split(f"% BEGIN WEEK{week}\n", 1)[1].split(
            f"% END WEEK{week}", 1
        )[0]
        content = content.replace(r"\begin{table}[htbp]", r"\begin{table}[!htbp]")
        if week == 3:
            content = content.replace(
                r"Định lý \ref{th:overlap}",
                r"đẳng thức overlap $X'[:,0:W-s]=X[:,s:W]$ của Week 2",
            )
        weekly_preamble = preamble.replace(
            "ECDR: Project Brief, Data Foundations, Mathematical Foundations", title
        ).replace(r"\fancyhead[R]{\small Week 1--3}",
                  rf"\fancyhead[R]{{\small Week {week}}}")
        front = rf"""\begin{{document}}
\title{{\textcolor{{ink}}{{ECDR}}\\[5pt]\Large {title}}}
\author{{Báo cáo nghiên cứu lý thuyết}}
\date{{28 tháng 10 năm 2026}}
\maketitle
\noindent Bản tuần độc lập, trích từ báo cáo tổng hợp. Scope: lý thuyết mới
theo brief của người nghiên cứu; không dùng implementation cũ làm nguồn
định nghĩa phương pháp. Các tính toán minh họa không phải benchmark.
\tableofcontents
\clearpage
"""
        provenance = r"""
\section*{Nguồn và cách tái kiểm}
Nội dung theo bản chuẩn hóa Week 1--3 do người nghiên cứu cung cấp và các
nguồn gốc trong bibliography. Các số data mô tả snapshot local theo policy
đã ghi; mô hình lý thuyết không phải checkpoint triển khai cũ. Báo cáo tổng
hợp trong cùng thư mục có đầy đủ bảng ký hiệu, provenance và các tuần còn
lại. Phép tính số học được lưu trong \code{recompute.py} và
\code{calculations.json}. Không có training hoặc benchmark ECDR mới.
"""
        (ROOT / f"{stem}.tex").write_text(
            weekly_preamble + front + content + provenance
            + "\n\\begingroup\\small\n" + bibliography + "\n\\endgroup\n"
            + "\n\\end{document}\n", encoding="utf-8"
        )
        print(f"Created {stem}.tex")


if __name__ == "__main__":
    main()
