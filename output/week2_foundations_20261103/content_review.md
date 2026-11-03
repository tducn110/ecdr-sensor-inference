# Independent content review: Week 2 data and model studies

## Scope and evidence boundary

This review compares the 64 complete concept bodies P01–P08 and C01–C56 in `output/week2_foundations_20261103/before/02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex` with the current standalone data study and model companion. The body text, not only titles, was inspected. Exact original source line spans and verbatim titles are recorded in `concept_audit_64.csv`; each ID appears once.

The checkout was on `main` at base SHA `633dc05a80511af4d4580cd985edd494cdc373ff` during review. Current reviewed files are `01_DATA/03_Week_2_Data_Foundations.tex` and `02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex`. This is a content trace and interpretation review. It is not an independent rebuild/render receipt, user acceptance, or a new model/data execution. Final build/render state is recorded separately in `final_verification.json` and `visual_review.json`; this review itself covers source content only.

The trace contains 34 `ADAPTED_TO_K12`, 9 `CONSOLIDATED_BACKGROUND`, 7 `HISTORICAL_GAP_ONLY`, 8 `DEFERRED_WEEK3`, 5 `DEFERRED_TRAINING`, 1 `NOT_ESTABLISHED` concepts. `ADAPTED_TO_K12` means the current report explicitly explains the concept in the selected reference’s terms; it does not imply that every derivation or former toy result was retained. `CONSOLIDATED_BACKGROUND` means only part is used implicitly or as an applied formula. `HISTORICAL_GAP_ONLY` means the source is preserved as archive/provenance and is explicitly not evidence about the selected reference. Deferred categories identify material beyond these two Week 2 studies. `NOT_ESTABLISHED` marks an original claim without a corresponding current teaching or evidence claim.

## Independent findings and gap disposition

The current reports address the initial content gaps identified in review: the data study now explicitly explains overlap identity and its interpretation; the model companion separately states why local temporal filters are a plausible choice while identifying boundary and position sensitivity and the lack of an ablation; the model study includes the per-class 12-class table, confusion matrix, and pointwise top-two-margin evidence; and both reports have explicit, independent study conclusions. The 64-concept trace records what was and was not transferred from the prior theoretical source.

The central model correction is explicit: the prior valid-convolution/GAP exposition describes a separate theoretical branch, while the selected HAPT reference is padded SmallCNN1D with two max-pools and a Flatten-plus-dense head. The current report does not transfer GAP cancellation identities, cache counts, or toy certificates to the selected checkpoint. Historical K=6 material is identified as a different reference. Current model identity is stated as K=12, seed 43, epoch 8, 283,692 trainable parameters, with fixed train-only normalization ID `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

A search of both current TeX files found GAP discussion only in the model companion’s explicit non-transfer section and historical/theoretical branch descriptions. The K=6 mention is explicitly historical. Other “gap” matches refer to logit gaps or stream gaps, not a GAP architecture claim. The data report’s six sensor channels are not six classes. The model report’s present-tense K=12 claims are tied to the frozen reference rather than historical K=6 metrics.

## Contract handoff

| Boundary | Current contract | Evidence/status |
|---|---|---|
| Data contract → model input | Six ordered accelerometer/gyroscope channels, 128 samples, chronological windows, subject-disjoint partitions, pure/accepted-majority supervised labels, and fixed train-only channel normalization. Ambiguous/unlabeled windows remain in the chronological stream but are excluded from supervised metrics. | Local reconstruction and normalization are reported VERIFIED against identified artifacts. Upstream annotation index origin and historical support-file provenance remain OPEN. |
| Model contract → output | The selected SmallCNN1D maps one normalized `(6,128)` window to 12 logits; evaluation state includes frozen weights and normalization. Its actual graph uses padded convolutions, BatchNorm, ReLU, two max-pools, Flatten, and dense layers. | Pointwise reconstruction agrees with stored outputs for the audited 5,053 windows; supervised metrics use 3,572 labeled rows. This does not establish unseen-input or regional behavior. |
| Week 3 boundary | Event conditioning, region definition/membership, sound all-eleven-competitor regional bounds, numerical error containment, anchor/state lifecycle, rejection of invalid inputs, and fallback are system-level obligations. | NOT ESTABLISHED here: no selected-model regional verifier, conservative membership implementation, complete fallback path, formal numerical bridge, or end-to-end reuse savings. |

## Claim classes

**VERIFIED (artifact/setup-bounded):** the data report records exact local reconstruction of 17,453 windows and manifest fields, 17,392 eligible aligned overlap-pair matches, subject-disjoint count/support checks, and six train-only normalization statistics/identity matching the frozen artifacts. The model report records inference-only reconstruction across 5,053 chronological windows with zero argmax disagreements and maximum absolute stored-logit difference 0.0 in that audit, plus supervised evaluation over 3,572 rows and per-class results. These labels describe the reports’ recorded checks; this review did not rerun them.

**DERIVED (under stated formulas and assumptions):** the model companion’s 283,692 parameter ledger, 1,172,992 MAC count, and 20-sample local receptive field with four-sample jump follow from the stated actual graph and counting/recurrence conventions. The data study’s window coordinates, duration distinctions, and deterministic overlap arithmetic are consequences of its declared windowing and sampling contract. These are structural/mathematical results, not runtime performance or a certificate.

**NOT ESTABLISHED:** upstream annotation index-origin semantics; historical support-file generation cause; annotation correctness and deployment representativeness; generalization or calibrated confidence; regional invariance/certification; sound membership and numerical error bounds; certified decision reuse; full event/cache/fallback lifecycle; and net end-to-end latency, memory, or energy savings. The existing transition-support quality status remains HOLD. No external acceptance is claimed.

## Limits and publication gate

This review adds no new model or data claims and does not convert historical concepts into current evidence. It covers source content and the current TeX text only. This semantic review does not itself perform compilation or visual inspection; those distinct checks are recorded in the final build/render receipts after source edits. Treat this review as content-trace evidence, not final artifact acceptance.

Root review refined C12/C21/C22/C35/C36/C56 after the final source edits: groups=1 and decoding/Softmax interpretation are explicit, raw shift-append belongs to the data study, and the archived feature-equality proof is not inherently GAP-dependent but needs new padded/pooling assumptions before use. Original training calculus and concrete verifier methods remain archived/deferred; they are not silently claimed as rewritten lessons.

## Final semantic review

A second source-content review read both complete current TEX files, all 64 dispositions and the relevant archived/source passages. It found and corrected one attribution: the first dense layer (2048 to 128), not the final 128-to-12 layer, dominates the parameter count. The convolution boundary-variable sentence was also clarified. No further substantive content blocker was found within this data/model scope. The delivery validator checks markers and receipts; that automated PASS is distinct from this semantic review. Final PDF build/render evidence is recorded separately in `final_verification.json` and `visual_review.json`; neither represents external acceptance.
