# ECDR research repository

Event-Conditioned Certified Decision Reuse for sequential sensor inference.

## Current Week 1 source of truth

Read [the Week 1 project brief](00_OVERVIEW/02_Week_1_Project_Brief.tex) first. This standalone LaTeX file is the only editable Week 1 authority and embeds the complete English body. [00_OVERVIEW/week1/](00_OVERVIEW/week1/) contains generated shared extracts plus the evidence registers; generated PDFs are derivatives.

Week 1 closes the research brief, reference-pipeline plan, training/evaluation protocol, and feasibility baseline. The executed reference run `week1_ref_1793620216_9d6c1d85` selected seed 43 at epoch 8 and passed 17 internal verification checks. Week 2 covers deep-model and data analysis plus component rationale, and Week 3 covers mathematical obligations. Formal certification and net savings remain pending; do not infer them from the verified baseline.

## Reading and evidence navigation

1. [Week 1 brief](00_OVERVIEW/02_Week_1_Project_Brief.tex) — current scope, definitions, requirements, and research questions.
2. [Research questions and gap](01_RQ_GAP.md) — current English RQs and bounded hypotheses.
3. [Targeted prior work](02_LITERATURE.md) — primary-source comparison and collision boundaries.
4. [Reading plan](05_READING_PLAN.md) and [review protocol](06_REVIEW_PROTOCOL.md) — planned reading and evidence rules.
5. [Data audit](03_DATA_AUDIT.md), [model sources](scripts/backbone.py), and [G3 domain contract](07_G3_DOMAIN_CONTRACT.md) — downstream contracts.

`ECDR_Project_Brief_2026-10-22.md`, older Vietnamese documents, saved PDFs, and dated reports are historical snapshots. They remain available for provenance but do not override the current Week 1 source.

The repository contains raw data, checkpoints, scripts, reports, and historical/generated presentation artifacts. The current Week 1 authority is the canonical self-contained LaTeX source, its registers, and the verified PDF delivery; removed presentation folders are not current source.
