# Week 1 Derivative & Source Alignment Audit Report

**Date:** 2026-11-02  
**Audit Target:** Week 1 ECDR Closeout, Canonical Research Contract, and Derivative Assets  
**Workspace:** `/home/pro/Downloads/research`  

---

## 1. Executive Summary & Audit Scope

This audit evaluates the Week 1 ECDR (Event-Conditioned Certified Decision Reuse) documentation base, derivative reports, presentation layers, and symlink libraries against the authoritative Week 1 Research Contract. All historical evidence confirms that Week 1 must establish the primary research grounding: locking **SmallCNN1D $K=12$** as the frozen reference architecture, formalizing the HAPT data and label policy, isolating historical $K=6$ artifacts as provenance, and enforcing strict decision-preservation semantics ($\hat{h}_{\text{ECDR}}(X) = h_{\text{SmallCNN1D}}(X)$).

---

## 2. Inventory of Week 1 Sources and Derivatives

The Week 1 document ecosystem spans canonical sources, monolithic syntheses, mirrored tiers, and web visualization platforms:

| Category | File Path | Status / Responsibility |
|---|---|---|
| **New Canonical Core** | `00_OVERVIEW/week1/brief_body.tex` | Authoritative English LaTeX body: SmallCNN1D $K=12$, HAPT 17/4/9 split, RQ1–RQ4, H1–H4. |
| **New Canonical Core** | `00_OVERVIEW/week1/SOURCE_OF_TRUTH.md` | Canonical scope record; declares locked semantics, reference identity, and evidence boundary. |
| **New Canonical Core** | `00_OVERVIEW/week1/REQUIREMENTS.md` | 10 normative engineering and certification requirements for Week 1 closeout. |
| **New Canonical Core** | `00_OVERVIEW/week1/EVIDENCE_REGISTER.md` | Evidence ledger classifying claims W1-01–W1-12 as VERIFIED, DERIVED, PROPOSED, or NOT_ESTABLISHED. |
| **New Canonical Core** | `00_OVERVIEW/week1/HYPOTHESES.md` | Hypothesis register H1–H4 defining exact falsification criteria and required measurements. |
| **New Canonical Core** | `00_OVERVIEW/week1/CLOSEOUT.md` | Closeout gate criteria; marks K=12 execution and gate acceptance pending without false claims. |
| **Legacy Week 1 LaTeX** | `00_OVERVIEW/02_Week_1_Project_Brief.tex` | Stale Vietnamese standalone LaTeX; claims "$K=6$ or $K=12$", theory-first, lacks pipeline proof. |
| **Legacy Week 1 LaTeX** | `me/Week_1_Project_Brief.tex` | Byte-identical copy of `00_OVERVIEW/02_Week_1_Project_Brief.tex` (SHA256: `3606f831...`). |
| **Legacy Week 1 PDF** | `00_OVERVIEW/02_Week_1_Project_Brief.pdf` | Stale compiled PDF (SHA256: `80c9ab8f...`). |
| **Legacy Week 1 PDF** | `me/Week_1_Project_Brief.pdf` | Byte-identical copy of `00_OVERVIEW/02_Week_1_Project_Brief.pdf`. |
| **Legacy Week 1 PDF** | `PDF_LIBRARY/01_Week_1_Project_Brief.pdf` | Byte-identical copy in shelf library. |
| **Legacy Week 1 PDF** | `tiers/pdf_shelf/01_Week_1_Project_Brief.pdf` | Byte-identical mirror in tiers tree. |
| **Monolithic Synthesis** | `04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.tex` | Lines 130–303 embed the stale Vietnamese Week 1 brief; mixes GAP theory with SmallCNN1D. |
| **Monolithic Synthesis** | `me/ECDR_Weeks_1_2_3_Theory.tex` | Byte-identical copy of synthesis LaTeX (SHA256: `ce39afe9...`). |
| **Monolithic Synthesis** | `04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.pdf` | Stale compiled PDF of the 3-week synthesis (SHA256: `16ce289c...`). |
| **Monolithic Synthesis** | `me/ECDR_Weeks_1_2_3_Theory.pdf` | Byte-identical PDF copy. |
| **Monolithic Synthesis** | `PDF_LIBRARY/16_ECDR_Weeks_1_2_3_Theory_Full.pdf`| Byte-identical shelf copy. |
| **Monolithic Synthesis** | `tiers/pdf_shelf/16_ECDR_Weeks_1_2_3_Theory_Full.pdf` | Byte-identical tier copy. |
| **Root Markdown** | `PROJECT_BRIEF.md` / `PROJECT_BRIEF_EN.md` | Early project briefs (Vietnamese & English); lack locked SmallCNN1D K=12 architecture contract. |
| **Root Markdown** | `ECDR_Project_Brief_2026-10-22.md` | Stale 2026-10-22 project brief snapshot. |
| **Root Markdown** | `ECDR_Project_Brief_Full_EN_2026-10-22.md` | Byte-identical copy of `PROJECT_BRIEF_EN.md` (SHA256: `4983facf...`). |
| **Root Markdown Stub** | `00_OVERVIEW/01_PROJECT_BRIEF.md` | Empty stub (0 bytes); needs replacement or symlink to canonical English brief. |
| **Web Presentation** | `web/present/src/components/Week1Section.tsx` | Interactive React component; Vietnamese text, references old 8-document layout, lacks K=12 contract. |
| **Web Presentation** | `04_SYNTHESIS/09_presentation_app/src/components/Week1Section.tsx` | Byte-identical source copy of `Week1Section.tsx` (SHA256: `f35a7e72...`). |
| **Web Presentation** | `web/present/src/components/PresentationSlideDeck.tsx` | Slide deck referencing 8-document suite and old Week 1 claims. |
| **Web Presentation** | `web/present/src/utils/figureRegistry.ts` | Registry claiming 100% compilation of 8 documents including old Week 1. |
| **Web Presentation** | `web/js/slides.js` / `web/js/app.js` | Legacy vanilla JS slide presentations echoing stale document metadata. |
| **Symlink Shelf** | `tiers/00_overview/` | Symlinks pointing to root markdown and `me/` files; mirrors stale targets. |
| **Symlink Shelf** | `LATEX_LIBRARY/Week_1_Project_Brief.tex` | Symlink pointing to `../../me/Week_1_Project_Brief.tex`. |
| **Symlink Shelf** | `tiers/latex_sources/Week_1_Project_Brief.tex` | Symlink pointing to `me/Week_1_Project_Brief.tex`. |

---

## 3. Concrete Stale and Contradictory Claims

1. **Reference Architecture and Parameter Conflation:**
   - *Conflict:* Legacy LaTeX (`00_OVERVIEW/02_Week_1_Project_Brief.tex`, line 79) introduces SmallCNN1D with "$K=6$ or $K=12$" without specifying layer dimensions, activation functions, or pooling. Meanwhile, `scripts/backbone.py` and `configs/backbone.yaml` implement a 6-class model (282,918 parameters), and Week 2/3 documents analyze a 20,332-parameter Conv1D-GAP model.
   - *Resolution Contract:* The primary reference is exclusively **SmallCNN1D $K=12$** with 283,692 expected trainable parameters (Conv1D 6$\to$32 k5 p2 BN-ReLU Pool2; Conv1D 32$\to$64 k3 p1 BN-ReLU Pool2; Conv1D 64$\to$64 k3 p1 BN-ReLU; Flatten 2048; Linear 2048$\to$128 ReLU; Linear 128$\to$12). Historical $K=6$ code is preserved strictly as provenance.

2. **Silently Conflating GAP Theorems with SmallCNN1D:**
   - *Conflict:* Synthesis document `04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.tex` (§7) derives exact feature margin deltas specifically for `Conv1D valid-ReLU-GAP`.
   - *Resolution Contract:* SmallCNN1D uses Flatten-MLP (2048 features) and MaxPool, completely invalidating GAP-specific linear summation reuse. Theoretical derivations for GAP must not be asserted as properties of SmallCNN1D.

3. **Classification Accuracy vs. Certified Decision Preservation:**
   - *Conflict:* Early texts blur the distinction between model accuracy against ground truth ($h(X) = y_{\text{true}}$) and decision preservation ($\hat{h}_{\text{ECDR}}(X) = h_{\text{SmallCNN1D}}(X)$).
   - *Resolution Contract:* ECDR does not improve base classifier accuracy. The formal safety target is **0 decision disagreements** with the frozen reference over certified regions. Relative accuracy drops ($\le 5\%$) are experimental tolerance bounds, not certified reuse semantics.

4. **Speedup and Certification Assertions:**
   - *Conflict:* Earlier notes suggest potential speedup from sensor overlap and heuristic similarity without end-to-end accounting.
   - *Resolution Contract:* Heuristic similarity is never certification. Cost savings must be evaluated across the **entire inference pipeline**: input I/O, fixed preprocessing $\mathcal{P}$, event detection, candidate membership checks, certificate construction/reuse, and reference fallback. Forward-only latency cannot claim net speedup.

5. **Language and Presentation Inconsistency:**
   - *Conflict:* `00_OVERVIEW/02_Week_1_Project_Brief.tex`, `04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.tex`, and `web/present/src/components/Week1Section.tsx` are in Vietnamese and pitched as descriptive reviews rather than scope-locking specifications.
   - *Resolution Contract:* Canonical deliverables must be in English, grounded in reproducible run manifests, and clearly segregated by claim status.

---

## 4. Unresolved Evidence Boundaries & Blockers

1. **Pending SmallCNN1D $K=12$ Checkpoint & Metrics:**
   - No validated training/eval run exists for SmallCNN1D $K=12$.
   - Claim status is **PROPOSED / PENDING RUN**. Training must execute under seeds 42, 43, 44, max 30 epochs, Adam lr $10^{-3}$, weight decay $10^{-4}$, cosine scheduler, and patience 10 on the 17/4/9 subject split.
   - Metrics (test accuracy, macro-F1, per-class recall, 12-class confusion matrix, margin distributions) are unverified until JSON artifacts with run IDs and environment hashes are produced.

2. **Transition Class Data Imbalance (The G2 Quality Failure):**
   - HAPT dataset exhibits severe imbalance: **344 pure transition windows** versus **10,017 pure basic-activity windows** out of 10,361 total pure windows.
   - The historical G2 quality gate failed. SmallCNN1D $K=12$ is accepted strictly as an **exploratory feasibility baseline** using the pure-or-dominant $\ge 0.6$ label policy, not as a certified production-ready classifier.

3. **Execution Soundness on Actual Graph:**
   - Interval Bound Propagation (IBP), CROWN, and Lipschitz bounds must be validated on the instantiated PyTorch graph with BatchNorm in evaluation mode (`eval()`) using frozen training statistics.

---

## 5. Derivative Synchronization Manifest & Priority Plan

To align the entire repository without altering frozen backbones (`scripts/backbone.py`, `configs/backbone.yaml`) or unrelated Week 2/3 domain code:

```
[Priority 0: Canonical Core Promotion]
00_OVERVIEW/week1/brief_body.tex           --> Wrap into canonical 00_OVERVIEW/02_Week_1_Project_Brief.tex (English)
00_OVERVIEW/week1/SOURCE_OF_TRUTH.md       --> Establish as repo-wide root contract for Week 1
00_OVERVIEW/01_PROJECT_BRIEF.md            --> Populate with English Canonical Brief

[Priority 1: Symlink & Dual-File Synchronization]
me/Week_1_Project_Brief.tex                --> Re-sync with updated 00_OVERVIEW/02_Week_1_Project_Brief.tex
PDF_LIBRARY/01_Week_1_Project_Brief.pdf   --> Recompile from updated English LaTeX
tiers/pdf_shelf/01_Week_1_Project_Brief.pdf--> Update mirror

[Priority 2: Monolithic Synthesis Update]
04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.tex (Part 1, L130-303)
                                           --> Replace with Week 1 canonical English brief
me/ECDR_Weeks_1_2_3_Theory.tex             --> Sync with updated synthesis LaTeX
PDF_LIBRARY/16_ECDR_Weeks_1_2_3_Theory_Full.pdf --> Recompile synthesis PDF

[Priority 3: Web & Presentation Synchronization]
web/present/src/components/Week1Section.tsx--> Update with English canonical contract & SmallCNN1D K=12
04_SYNTHESIS/09_presentation_app/...      --> Sync with web/present
web/present/src/utils/figureRegistry.ts    --> Align document metadata and verified page counts
```

---

## 6. Audit Conclusion & Final Status

- **Week 1 Status:** Scope-locked and grounded via `00_OVERVIEW/week1/`. Quality gates remain intentionally on hold pending execution of the isolated $K=12$ training pipeline.
- **Backbone Integrity:** `scripts/backbone.py` and `configs/backbone.yaml` remain untouched.
- **Synchronization Readiness:** Full inventory and dependency mapping completed. Derivatives are ready for systematic rebuild upon owner authorization.
