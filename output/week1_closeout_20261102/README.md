# Week 1 closeout artifact index

This directory is the reproducible evidence package for the executed Week 1 reference run and its source, navigation, orchestration, and document checks. It is an index of artifacts and evidence boundaries; it is not a second Week 1 source of truth.

## Authority and current delivery

- **Canonical editable source:** [`00_OVERVIEW/02_Week_1_Project_Brief.tex`](../../00_OVERVIEW/02_Week_1_Project_Brief.tex), self-contained and authoritative for the current English Week 1 brief.
- **Canonical standalone PDF:** [`00_OVERVIEW/02_Week_1_Project_Brief.pdf`](../../00_OVERVIEW/02_Week_1_Project_Brief.pdf), generated from the canonical source. The standalone output is 11 pages.
- **Combined report source:** [`me/ECDR_Weeks_1_2_3_Theory.tex`](../../me/ECDR_Weeks_1_2_3_Theory.tex), with the Week 1 block supplied through the generated shared extract. Week 2 and Week 3 sections remain preserved historical scope.
- **Combined PDF:** [`me/ECDR_Weeks_1_2_3_Theory.pdf`](../../me/ECDR_Weeks_1_2_3_Theory.pdf), 36 pages. The corresponding `PDF_LIBRARY` and compatibility PDF paths are delivery derivatives or aliases, not additional authorities.
- **Generated shared extract:** [`00_OVERVIEW/week1/brief_body.tex`](../../00_OVERVIEW/week1/brief_body.tex). It is generated from the canonical body markers for combined-report and compatibility-wrapper use; it is not independently editable authority.

Top-level `02_Week_1_Project_Brief.pdf` and `.log` files are current workspace aliases to the canonical standalone delivery. They are derivatives and must not be treated as competing source files.

The accepted cleanup remains in force. Removed web/presentation applications and removed tier LaTeX aliases are absent and must not be restored.

## Current registers and navigation

The current Week 1 registers are under [`00_OVERVIEW/week1/`](../../00_OVERVIEW/week1/): `SOURCE_OF_TRUTH.md`, `EVIDENCE_REGISTER.md`, `HYPOTHESES.md`, `REQUIREMENTS.md`, and `CLOSEOUT.md`. Root and tier navigation points back to the canonical source and identifies historical Week 2/3 boundaries.

- [`integration_inventory.md`](integration_inventory.md) records ownership, current versus historical artifacts, cleanup boundaries, and evidence status.
- [`navigation_link_audit.md`](navigation_link_audit.md) records the final 27-document navigation audit with zero broken URL-decoded Markdown links.

## Reference run and frozen 25-file package

The [`reference/`](reference/) directory contains exactly 25 files for run `week1_ref_1793620216_9d6c1d85`: configuration, provenance, normalization, three seed checkpoints and histories, selection/evaluation receipts, predictions, manifests, and verification receipts. The run command and environment are evidenced by [`pipeline.log`](pipeline.log), [`reference/provenance_audit.json`](reference/provenance_audit.json), [`reference/train_receipt.json`](reference/train_receipt.json), and [`reference/config.json`](reference/config.json).

The selected checkpoint is seed 43, epoch 8. The retained execution receipt is [`reference/verification_receipt.json`](reference/verification_receipt.json); the independent check receipt is [`reference/independent_verification.json`](reference/independent_verification.json). The former records 17 internal execution checks; the latter records 65/65 independent checks. Quality promotion, formal certificate soundness, strict decision-preservation evidence, and net-savings evidence remain unestablished/HOLD.

## Orchestration and execution provenance

The [`orchestration/`](orchestration/) directory preserves the actual Gemini dialogue and execution provenance, including [`orchestration/orchestration_receipt.json`](orchestration/orchestration_receipt.json). The receipt records the recovered Gemini dialogue, actual model/tool execution, and recovered root PTY session `94879`. The user-facing UI was queued/limited in that session; the recorded PTY and orchestration logs are the execution evidence.

The orchestration package also retains the implementation, review, verifier, and derivative-audit prompts, outputs, stderr/stdout streams, frozen preimplementation hashes, and child-review receipts. These files document ownership and provenance; they do not replace the current source or execution receipts.

## Final verification and document QA

The final manifest and verification artifacts are recorded at the closeout root:

- [`final_verification.json`](final_verification.json) — final consolidated verification receipt.
- [`document_checks.json`](document_checks.json) — PASS for current document checks; canonical standalone 11 pages and combined report 36 pages.
- [`existing_regression.log`](existing_regression.log) — retained regression evidence.
- [`canonical_compile.log`](canonical_compile.log) and [`combined_compile.log`](combined_compile.log) — current compile provenance.

The current QA review covered all 11 standalone pages and all 36 combined pages with contact/detail views. No overlap, clipping, or unresolved glyphs were found. Body-drift checks passed and current PDF aliases were verified against the canonical standalone and combined outputs.

## Failed receipts and draft artifacts

Failed verifier receipts, failed serialization/CSV/schema logs, and earlier failed direct-run logs are retained for auditability. [`document_verification.log`](document_verification.log) is an earlier failed verifier attempt and is historical evidence, not a current PASS receipt. These files show the repair path and must not be read as the final verification result; use [`final_verification.json`](final_verification.json) and [`document_checks.json`](document_checks.json) for current status.

Draft compilation outputs under `drafts/` are historical task drafts and are not current delivery artifacts. They must remain clearly separated from the final canonical PDFs and logs. The command used to reproduce the reference run must write to a **new output directory** and must not overwrite `output/week1_closeout_20261102/reference/` or any retained receipt:

The new directory is required even when reproducing the same run, so prior checkpoints, receipts, and audit logs remain immutable evidence. The pipeline guards reject an existing output directory. Use an unused concrete directory name, for example:

```bash
./.venv/bin/python scripts/week1_reference_pipeline.py all \
  --output-dir output/week1_reproduction_20261102_01 \
  --epochs 30 --seeds 42,43,44 --device cpu
```

## Scope boundary

This package documents the current Week 1 reference-pipeline execution and integration QA. Week 2/3 authored text and older dated reports remain historical dependencies. Formal certificate claims, strict ECDR reference-agreement claims, and net-savings claims require their own evidence and are not established by this closeout package.
