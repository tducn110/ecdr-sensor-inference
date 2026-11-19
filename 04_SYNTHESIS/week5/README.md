# ECDR Week 5 final synthesis

The English IEEEtran journal-style report consolidates Weeks 1–4 and incorporates
the Week 5 evidence audit. The earlier manuscript is retained as a draft.

- Source: [ECDR_Week5_FINAL_EN.tex](reports/ECDR_Week5_FINAL_EN.tex)
- Final PDF at the repository root: [ECDR_Week5_FINAL_EN.pdf](../../ECDR_Week5_FINAL_EN.pdf)
- New stride experiment: [script](evidence/stride_audit.py), [receipt](evidence/stride_audit.json)
- Saved replay recount: [script](evidence/reconcile_replay.py), [receipt](evidence/reconcile_replay.json)
- Paired end-to-end query timing: [script](evidence/paired_latency.py), [receipt](evidence/paired_latency.json)
- Appendix generators: [replay ledgers](evidence/build_ledgers.py) and [detailed Week 1–4 evidence](evidence/build_detail.py)

The PDF includes the frozen K=12 contract, all 17 registered methods, H01–H24,
the recorded 40-task closeout, W4-G01–G10, and the status of G0–G6.

Run these commands from the repository root to regenerate the evidence and report.
The stride audit uses the existing Python environment with CPU PyTorch, NumPy,
pandas and a Parquet reader. The LaTeX build uses pdfLaTeX and BibTeX.

    .venv/bin/python 04_SYNTHESIS/week5/evidence/stride_audit.py
    python3 04_SYNTHESIS/week5/evidence/reconcile_replay.py
    python3 04_SYNTHESIS/week5/evidence/paired_latency.py
    python3 04_SYNTHESIS/week5/evidence/build_ledgers.py
    python3 04_SYNTHESIS/week5/evidence/build_detail.py
    cd 04_SYNTHESIS/week5/reports
    pdflatex -interaction=nonstopmode -halt-on-error ECDR_Week5_FINAL_EN.tex
    bibtex ECDR_Week5_FINAL_EN
    pdflatex -interaction=nonstopmode -halt-on-error ECDR_Week5_FINAL_EN.tex
    pdflatex -interaction=nonstopmode -halt-on-error ECDR_Week5_FINAL_EN.tex
    cp ECDR_Week5_FINAL_EN.pdf ../../../

