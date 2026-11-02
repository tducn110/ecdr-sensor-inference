#!/usr/bin/env python3
"""Extract three standalone weekly LaTeX reports from the canonical report.

The master source contains all authored content. This script only packages it;
it neither runs experiments nor changes any source outside this directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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
