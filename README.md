# ECDR research repository

Event-Conditioned Certified Decision Reuse for sequential sensor inference.

## Current research sources

1. **Week 1 — project brief and frozen baseline:** [canonical English source](00_OVERVIEW/02_Week_1_Project_Brief.tex), [PDF](00_OVERVIEW/02_Week_1_Project_Brief.pdf), and [source-of-truth record](00_OVERVIEW/week1/SOURCE_OF_TRUTH.md).
2. **Week 2 — two independent studies:** Report A, [HAPT Data Foundations source](01_DATA/03_Week_2_Data_Foundations.tex) / [PDF](01_DATA/03_Week_2_Data_Foundations.pdf); Report B, [SmallCNN1D Model Foundations source](02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex) / [PDF](02_MODEL/01_ECDR_CNN_Model_Analysis_EN.pdf). Each has its own objective, method, analysis and conclusion.
3. **Week 2 executed evidence:** [closeout and reproduction](output/week2_foundations_20261103/README.md), [data audit](output/week2_foundations_20261103/data_audit.json), [model audit](output/week2_foundations_20261103/model_audit.json), and [final verification receipt](output/week2_foundations_20261103/final_verification.json).
4. **Week 3 — mathematical obligations:** [domain contract](07_G3_DOMAIN_CONTRACT.md). Older Week 3 reports are prior theory snapshots; they are not a completed verifier for the frozen SmallCNN1D reference.

The two Week 2 standalone LaTeX files are the editable authorities for this stage. Report A owns the data contract; Report B owns the actual model graph, behavior and Week 3 model obligations. The [content review](output/week2_foundations_20261103/content_review.md) and [64-concept disposition audit](output/week2_foundations_20261103/concept_audit_64.csv) document the complete review of the earlier CNN exposition, including material retained only as historical background or deferred training/verifier work. The `me/Week_2_Data_Foundations.tex` compatibility entry points to the canonical Week 2 source; the combined Weeks 1–3 documents retain historical Week 2 content and do not override it. The pre-revision sources are retained in the [before-edit snapshot](output/week2_foundations_20261103/before/).

## Frozen reference and verified Week 2 findings

The reference is **SmallCNN1D K=12**, with same-padded Conv1d, BatchNorm/ReLU, two MaxPool layers, Flatten 2048 → Linear 128 → Linear 12: **283,692 trainable parameters**. Run `week1_ref_1793620216_9d6c1d85` selected seed 43 at epoch 8 using validation macro-F1. The [selected checkpoint](output/week1_closeout_20261102/reference/selected_best.pt) and train-only preprocessing remain unchanged.

On 3 November 2026 the Week 2 scripts rebuilt all **17,453** raw windows and their manifest fields exactly, checked all **17,392** aligned overlap pairs, and reproduced the frozen normalization. The split contains 17/4/9 subjects; full-stream windows are 9,978/2,422/5,053, while supervised pure-plus-majority windows are 6,955/1,751/3,572. Majority requires at least 60% of the full 128-sample window.

Fresh checkpoint inference reproduced all stored logits exactly in the recorded CPU environment and disagreed with stored predictions on **0/5,053** test-stream windows. On the separate **3,572** supervised test windows, accuracy is **0.8905375** and macro-F1 **0.7762645**. This verifies the reference trace, not certified reuse. The graph has a final local-feature receptive field of 20 samples, jump 4, and 1,172,992 Conv/Linear MACs per window under the stated counting convention.

## Open research limits

- **G2 remains HOLD:** transition classes are present but weakly supported. Finishing the Week 2 analysis does not pass the model-quality gate.
- **Annotation origin remains unresolved:** a one-sample conversion changes 98 window policy/label assignments. Current labels are reproduced and preserved.
- **Historical support provenance remains unresolved:** `data/ecdr_12_class_support.json` reports 13,866 broad-support rows; the current source predicate and frozen supervised policy produce 12,278. The 1,588 difference equals ambiguous support, but that equality does not prove the old artifact's generating policy.
- **The GAP branch is a different model:** `ecdr/model.py:ECDRConv1DNet` has 20,332 parameters at K=12; its valid-convolution/GAP identities cannot be used as proof for the frozen Flatten–MLP network. Its preprocessing identity is also separate.
- **Formal certification and complete ECDR cost remain open:** point margins, adjacent agreement, exact trace reproduction and forward-only CPU timing do not establish a certified region or net savings. No training was repeated in Week 2.

## Reproduce Week 2

Run from the repository root using the existing project environment:

```bash
.venv/bin/python scripts/week2_data_audit.py
.venv/bin/python scripts/week2_model_audit.py
```

Both commands write only Week 2 evidence and fail on required consistency checks. The checkpoint, raw corpus and frozen Week 1 artifacts are inputs. The [closeout](output/week2_foundations_20261103/README.md) records PDF compilation, rendering, checks, evidence identities and remaining limitations. Timing varies by run; its fixed-input, forward-only scope is recorded in the model audit.

## Other reading and historical material

[Research questions](01_RQ_GAP.md), [prior work](02_LITERATURE.md), [reading plan](05_READING_PLAN.md) and [review protocol](06_REVIEW_PROTOCOL.md) provide the broader research context. The [data entry point](01_DATA/README.md) and [model entry point](02_MODEL/README.md) distinguish current K=12 evidence from the historical K=6 and GAP artifacts. Older Vietnamese reports, gate snapshots, saved PDFs and `ECDR_Project_Brief_2026-10-22.md` remain historical sources; their metrics and gate labels do not supersede the current Week 1/2 records.
