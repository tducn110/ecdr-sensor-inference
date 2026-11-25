# ECDR Week 5 Final Report & Evidence

The canonical English 1-column final research report (`\documentclass[11pt,oneside,a4paper]{report}` with IEEE-formatted references) consolidates the entire ECDR implementation, mathematical derivations, Week 1–4 replays, and post-Week 4 diagnostic studies.

- Source: [ECDR_Final_Report_EN.tex](reports/ECDR_Final_Report_EN.tex)
- Final PDF at the repository root: [ECDR_Final_Report_EN.pdf](../../ECDR_Final_Report_EN.pdf)
- New stride experiment: [script](evidence/stride_audit.py), [receipt](evidence/stride_audit.json)
- Saved replay recount: [script](evidence/reconcile_replay.py), [receipt](evidence/reconcile_replay.json)
- Paired end-to-end query timing: [script](evidence/paired_latency.py), [receipt](evidence/paired_latency.json)
- Appendix generators: [replay ledgers](evidence/build_ledgers.py) and [detailed Week 1–4 evidence](evidence/build_detail.py)

The report covers the frozen `SmallCNN1D` (`K=12`) contract, Higham `binary32` enclosure, all 17 registered methods (`M00–M16`), `H01–H24`, G2-Full closure, post-Week 4 stride-1/scaling diagnostics, and `G0–G6`.

Run these commands from the repository root to regenerate the evidence and report.
The stride audit uses the existing Python environment with CPU PyTorch, NumPy,
pandas and a Parquet reader. The LaTeX build uses XeLaTeX and BibTeX.

    .venv/bin/python 04_SYNTHESIS/week5/evidence/stride_audit.py
    python3 04_SYNTHESIS/week5/evidence/reconcile_replay.py
    python3 04_SYNTHESIS/week5/evidence/paired_latency.py
    python3 04_SYNTHESIS/week5/evidence/build_ledgers.py
    python3 04_SYNTHESIS/week5/evidence/build_detail.py
    cd 04_SYNTHESIS/week5/reports
    xelatex -interaction=nonstopmode -halt-on-error ECDR_Final_Report_EN.tex
    bibtex ECDR_Final_Report_EN
    xelatex -interaction=nonstopmode -halt-on-error ECDR_Final_Report_EN.tex
    xelatex -interaction=nonstopmode -halt-on-error ECDR_Final_Report_EN.tex
    cp ECDR_Final_Report_EN.pdf ../../../

