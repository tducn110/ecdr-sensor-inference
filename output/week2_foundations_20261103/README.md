# Week 2 closeout — HAPT data and SmallCNN1D foundations

Date: 3 November 2026 (Asia/Ho_Chi_Minh). Base revision: `main` at `633dc05a80511af4d4580cd985edd494cdc373ff`; the worktree was clean before this work. This package completes the eight-unit data/model foundations analysis and updates repository navigation. It is local research evidence, not external acceptance or a formal ECDR certificate.

## Deliverables and authority

- [Report A: independent Data Foundations — 13 PDF pages](../../01_DATA/03_Week_2_Data_Foundations.pdf), with [editable LaTeX](../../01_DATA/03_Week_2_Data_Foundations.tex): physical signals, window/overlap derivations, annotations, splits/support, dependence, normalization, provenance and the data-side handoff contract.
- [Report B: independent CNN Model Foundations — 16 PDF pages](../../02_MODEL/01_ECDR_CNN_Model_Analysis_EN.pdf), with [editable LaTeX](../../02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex): independent model objective/method/conclusion, 1D-CNN rationale, operator equations, tensor trace, BatchNorm state, pooling/finite boundaries, position-sensitive head, resource accounting, twelve-class metrics, confusion/margins and the Week 3 model contract.
- Updated [root README](../../README.md), [data README](../../01_DATA/README.md), [model README](../../02_MODEL/README.md), and relevant `me` / synthesis navigation. The old combined reports are explicitly historical for Week 2.
- [Data audit source](../../scripts/week2_data_audit.py), [model audit source](../../scripts/week2_model_audit.py), and [delivery validator](../../scripts/week2_verify_delivery.py).
- The [before-edit snapshot](before/) preserves superseded report sources and navigation. The Vietnamese specialist reports, raw data, checkpoint, labels, frozen normalization and Week 1 authority were not changed.

The two standalone LaTeX files are editable Week 2 authorities. `me/Week_2_Data_Foundations.tex` is a compatibility wrapper. `me/build_sources.py --week2-only` synchronizes only that wrapper, and the normal generator no longer extracts Week 2 from the stale monolithic master.

## Independent report boundaries and content audit

[Content review](content_review.md) traces this revision against the two-study requirement. [All 64 earlier concepts](concept_audit_64.csv) retain their original ID, heading and source span, with an explicit current disposition. The new model analysis is a full-source revision of the earlier valid/GAP exposition; it does not merely relabel a few introductory lines.

| Report | Scope and contract | Evidence meaning |
| --- | --- | --- |
| A — Data Foundations | Eight W2 units: physical signals, windowing, labels, splits/support, normalization, dependence/quality, provenance, final data contract | Reconstructed local arrays/counts are VERIFIED; window and affine-overlap identities are DERIVED; original annotation semantics, historical support provenance and population representativeness are NOT ESTABLISHED |
| B — CNN Model Foundations | Rationale, exact actual K=12 graph, operators, shapes/parameters, MAC/FLOP/memory/latency, classwise accuracy/confusion/margins, graph limits and Week 3 contract | Frozen pointwise reproduction and measured scopes are VERIFIED; structural arithmetic and real-valued operator identities are DERIVED; regional invariance, numerical bridge and net reuse benefit are NOT ESTABLISHED |
| Data → Model → Week 3 | Raw X → fixed N → normalized Z `(B,6,128)` → frozen f → twelve logits; later regional certificate must cover all eleven competitors and conservative membership | Labels, model decisions and certificates remain separate objects. Exact weights/preprocessing/class order and domain coordinates must match. No new verifier, cache runtime, training or Week 1 modification is included |

The data report does not duplicate model accuracy, confusion plots or the regional-margin theorem. The model report consumes the defined data contract and owns those analyses. Original training calculus and concrete bound algorithms remain explicitly archived/deferred in the 64-concept audit; a disposition is not a claim that every original lesson was re-taught or implemented.

## Checks actually executed

1. **Data reconstruction:** raw files → float32 windows and ten-field manifest; all required assertions passed. Correctly grouped per-channel lag summaries and dimensionless normalized distances avoid mixing g with rad/s. [Receipt](data_audit.json).
2. **Frozen inference:** loaded selected seed 43/epoch 8; exact logits and 0/5,053 prediction differences; 18 named model checks passed. Fresh winning margins also agree with the saved NPZ exactly, and with CSV rounding within `atol=1e-5, rtol=1e-5`. [Run log](model_audit_run.log).
3. **Week 1 preservation:** existing independent baseline verifier rerun into this package; **65/65 checks PASS**, including frozen provenance and predictions. [Receipt](week1_preservation_verification.json), [log](week1_preservation_verification.log).
4. **Document build:** XeLaTeX builds both canonical reports, with repeated passes for contents/bookmarks. Final logs have no errors or overfull boxes. [Data log](data_compile_final.log), [model log](model_compile_final.log).
5. **Compatibility:** the Week 2 wrapper compiles from both the repository root and `me/`; wrapper sync preserves the canonical sources and unrelated weeks.
6. **Document integrity and visual QA:** PDF text/content, relative navigation links, artifact hashes, the final receipt primary-evidence hash chain, and visual-review binding to the current PDF bytes/page counts are checked by the delivery validator. An isolated [negative hash fixture](hash_chain_regression.json) confirms that a stale evidence checksum is rejected. Every final PDF page was rendered at 90 dpi and inspected through all-page contact sheets, with full-size inspection of dense tables and plots. The exact rendering scope and final PDF hashes are recorded in [visual review](visual_review.json). This is layout QA, not scientific acceptance. The current review covers both independent reports and the 64-concept disposition.
7. **Source hygiene:** `py_compile` on the three new scripts and changed generator; `git diff --check` passed. [Document checks](document_checks.json), [delivery manifest](delivery_manifest.json), [final verification](final_verification.json).

## Reproduce the analysis

From the repository root:

```bash
.venv/bin/python scripts/week2_data_audit.py
.venv/bin/python scripts/week2_model_audit.py
.venv/bin/python me/build_sources.py --week2-only
.venv/bin/python scripts/week2_verify_delivery.py
```

The audits do not retrain or rewrite the frozen corpus. Repeating the model audit changes timing observations; any report quoting them must be synchronized to that new receipt. The final timing here is a repeated-input microbenchmark: same first normalized chronological test window, batch one, CPU, one Torch thread, 100 warmups and 1,000 timed forwards. Background CPU work was not isolated. It excludes acquisition, windowing, normalization, event/membership/certificate/cache/fallback work, transfers and I/O. It is not a controlled speed comparison against Week 1's varied-input run or evidence of net ECDR savings.

Build each canonical report from the root, repeating each command until contents/bookmarks stabilize:

```bash
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=01_DATA 01_DATA/03_Week_2_Data_Foundations.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=02_MODEL 02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex
```

PDFs, their editable sources and evidence files have SHA-256 entries in the delivery manifest. Temporary TeX build files were kept under `qa/tex_aux/`; final compilation logs are retained above.

## Open issues handed to Week 3

- UCI annotation index origin is unresolved. This package reproduces the frozen convention and quantifies 98 offset-sensitive assignments; it does not silently change labels.
- `data/ecdr_12_class_support.json` totals 13,866 under its historical broad heading. The current source predicate produces 12,278; their difference equals 1,588 ambiguous rows, but the generating provenance of the old JSON is not established. The stale artifact is excluded from current supervised evidence.
- Transition undersupport keeps **G2 quality HOLD**. Completing the foundations report is not a quality-gate pass; overlapping windows are not independent subjects.
- The 20,332-parameter valid-convolution/GAP model and its float64 preprocessing are separate from this frozen 283,692-parameter reference. No proof or runtime guarantee is transferred between them.
- Candidate/certified regions, a formal numerical-soundness bridge, verified cache lifecycle and measured net end-to-end benefit remain open. No certified coverage, speedup, energy result or user acceptance is asserted.

## Primary references

[UCI HAPT dataset](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) supplies physical provenance and file descriptions. PyTorch 2.14 operator documentation for [Conv1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html), [BatchNorm1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html), [MaxPool1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool1d.html) and [Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) was checked against the installed 2.14.1 CPU graph. Source/runtime observations govern instantiated arguments and quantitative counts.
