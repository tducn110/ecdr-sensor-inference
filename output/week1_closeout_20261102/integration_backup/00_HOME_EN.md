---
title: ECDR Research Vault Home (English)
tags:
  - ecdr
  - research
  - deep-learning
  - formal-verification
  - time-series
  - trustmebro
created: 2026-10-30
---

# 🌌 ECDR RESEARCH VAULT — HOME & NAVIGATION MAP
## Event-Conditioned Certified Decision Reuse in Sensor Time-Series Streams

> [!abstract] Welcome to the ECDR Research Knowledge Vault!
> The entire **ECDR** research project has been structured into a comprehensive knowledge vault, interconnected with Wikilinks, interactive Canvas mind maps, and a linear knowledge tiering hierarchy:
> **"Read [X] (DONE) $\rightarrow$ Master foundation [Y] $\rightarrow$ Unlock in-depth [Z]"**.

---

> [!warning] PREREQUISITE: Review Notation Before Week 1!
> Before diving into the theoretical monographs in `me/` or Week 1 materials, please keep in mind the core mathematical conventions:
> - **$X \in \mathbb{R}^{6 \times 128} \cong \mathbb{R}^{768}$**: Raw sensor observation window (3 triaxial acceleration axes in $g$, 3 triaxial gyroscope axes in $\text{rad/s}$ over $2.56\text{s}$ at $50\text{Hz}$).
> - **$\mathcal{P}$ (or $N$)**: Channel-wise fixed z-score standardization: $Z = \mathcal{P}(X) = (X - \mu)/\sigma$, with $\mu, \sigma$ frozen from the training set. *(Must remain fixed to prevent certificate drift!)*
> - **$Z$**: Dimensionless normalized input tensor passed into neural network $f_\theta$.
> - **$f_\theta$**: Deep neural backbone [[scripts/backbone.py|SmallCNN1D]] (282,918 parameters, frozen at Gate G2).
> - **$F = f_\theta \circ \mathcal{P}$**: End-to-end reference inference pipeline from raw signal $X$ to logits in $\mathbb{R}^K$.
> - **$h(X) = \arg\max_{k} F_k(X)$**: Final predicted class decision.
> - **$X^r$**: Trusted anchor window genuinely evaluated by full reference pipeline $F$.
> - **$\mathcal{S}_r$**: Certified safe decision domain around the anchor satisfying: $X \in \mathcal{S}_r \implies h(X) = h(X^r)$.
> - See details in: [[README.md#⚠️-instruction-0-bảng-ký-hiệu-tiền-quyết--bản-đồ-tiền-giả-định|Instruction 0 in README]].

---

> [!tip] 3 Quick Ways to Navigate in Obsidian & ECDR Reader
> 1. 🗺️ **Open Interactive Mind Map**: Click [[ECDR_Knowledge_Map.canvas|ECDR Knowledge Map Canvas]] to visualize the 5 tiers.
> 2. 🕸️ **Explore Graph View**: Press `Ctrl + G` (or `Cmd + G`) to inspect color-coded document dependency networks.
> 3. 🔖 **Use Bookmarks & Dual Reader**: Open the left sidebar or the Parallel VI ⇋ EN reading toolbar to inspect bilingual versions side-by-side.

---

## 📚 5-Tier Knowledge Architecture

Each tier corresponds to a progressive mastery phase. Proceed sequentially from Tier 0 to Tier 4:

| Tier | Directory & Guide | Core Research Focus | Gate Audit Status |
| :---: | :--- | :--- | :---: |
| **Tier 0** | [[00_OVERVIEW/README.md\|Tier 0: Overview & Foundations]] | ECDR problem formulation, RQ1 (Decision Reuse), RQ2 (Net Savings), 4-state Anchor Lifecycle | ✅ Approved |
| **Tier 1** | [[01_DATA/README.md\|Tier 1: UCI HAPT Data & Signals]] | 61 recordings, 12 classes vs $K=6$ pure-only, Autocorrelation $r \approx 0.985$, Failure of i.i.d. | ✅ **G1 REVISED** |
| **Tier 2** | [[02_MODEL/README.md\|Tier 2: 1D-CNN Deep Learning Model]] | SmallCNN1D (282k params), Checkpoint seed 44 (F1 0.9354), FLOPs, CPU forward latency ~0.24ms | ✅ **G2 PASS** |
| **Tier 3** | [[03_MATH/README.md\|Tier 3: Mathematical Theory & Verification]] | G3 domain contract, Interval Bound Propagation (IBP), CROWN relaxation, $\mathbb{R}^{768}$ Norm geometry, Wave $T_S$ | ⏳ **G3 PENDING** |
| **Tier 4** | [[04_SYNTHESIS/README.md\|Tier 4: Synthesis, Gates & Publications]] | 31-page theory dossier, comprehensive report, Gate G0–G6 evidence dossiers, international publications | ⏳ **G4–G6 PENDING** |

---

## 📑 21-PDF Linear Reading Library (`PDF_LIBRARY/`)

All PDF documents are numbered sequentially from `01` to `21`. Click any link to open in the integrated viewer:

1. [[PDF_LIBRARY/01_Week_1_Project_Brief.pdf|01. Week 1: Project Brief & Research Context]]
2. [[PDF_LIBRARY/02_SUPP01_ECDR_Primer.pdf|02. SUPP-01: ECDR Primer & Anchor Lifecycle Foundations]]
3. [[PDF_LIBRARY/03_Week_2_Data_Foundations.pdf|03. Week 2: UCI HAPT Sensor Dataset Foundations]]
4. [[PDF_LIBRARY/04_ECDR_Data_Analysis_EN.pdf|04. HAPT Data Analysis & Methodology (28 Pages EN)]]
5. [[PDF_LIBRARY/05_ECDR_Data_Analysis_VI.pdf|05. HAPT Data Analysis (Vietnamese Archive)]]
6. [[PDF_LIBRARY/06_SUPP04_Autocorrelation.pdf|06. SUPP-04: Time-Series Autocorrelation & Gradient Counter-Examples]]
7. [[PDF_LIBRARY/07_SUPP07_Statistical_Dependence.pdf|07. SUPP-07: Collapse of i.i.d. Assumption & Block Bootstrap]]
8. [[PDF_LIBRARY/08_ECDR_CNN_Model_Analysis_EN.pdf|08. 1D-CNN Deconstruction from First Principles (64 Concepts EN)]]
9. [[PDF_LIBRARY/09_ECDR_CNN_Model_Analysis_VI.pdf|09. CNN Model Analysis (Vietnamese Archive)]]
10. [[PDF_LIBRARY/10_Week_3_Mathematical_Foundations.pdf|10. Week 3: Mathematical Foundations & Safety Margins]]
11. [[PDF_LIBRARY/11_SUPP02_Norm_Geometry.pdf|11. SUPP-02: L2 vs L-inf Norm Geometry in 768 Dimensions]]
12. [[PDF_LIBRARY/12_SUPP03_Certification_Methods.pdf|12. SUPP-03: Certification Methods: IBP, CROWN & LiRPA]]
13. [[PDF_LIBRARY/13_SUPP05_Wave_Domain.pdf|13. SUPP-05: Wave Domain Parameterization & TS Phase Shift Operator]]
14. [[PDF_LIBRARY/14_SUPP06_Float_Precision.pdf|14. SUPP-06: IEEE 754 Arithmetic Precision & int8 Quantization]]
15. [[PDF_LIBRARY/15_ECDR_Mathematics_Explained.pdf|15. Comprehensive Mathematics Guide (29 Pages, H01–H24)]]
16. [[PDF_LIBRARY/16_ECDR_Weeks_1_2_3_Theory_Full.pdf|16. Complete 3-Week Theoretical Report (31 Pages)]]
17. [[PDF_LIBRARY/17_Dossier_Scientific_Report.pdf|17. Dossier Scientific Research Report (LaTeX Build)]]
18. [[PDF_LIBRARY/18_LipCache_Xiang2026_Paper.pdf|18. Foundational Paper: LipCache (Xiang et al., arXiv 2026)]]
19. [[PDF_LIBRARY/19_STBP_Varghese2026_Paper.pdf|19. Reference Paper: STBP for Sensor Time-Series (Varghese 2026)]]
20. [[PDF_LIBRARY/20_2605.21724v1_Paper.pdf|20. Extended Reference Paper 1]]
21. [[PDF_LIBRARY/21_got_Paper.pdf|21. Extended Reference Paper 2]]

---

## 🎯 TrustMeBro Evidence Gate Status: [[TMB.md]]

Current evidence status according to **TrustMeBro** rigor standard:
- ✅ **Gate G0 (PASS)**: Problem formulation & 4-axis comparison against LipCache.
- ✅ **Gate G1 (COMPLETE_WITH_REVISIONS)**: UCI HAPT audit, Script 05 index bug correction.
- ✅ **Gate G2 (PASS)**: SmallCNN1D training and freeze (K=6 pure-only, Macro-F1 0.9354).
- ⏳ **Gate G3 (PENDING)**: Standardized domain contract and minimal verified prototype.
- ⏳ **Gate G4 (PENDING)**: Empirical evaluation of Event-Conditioned region benefits.
- ⏳ **Gate G5 (PENDING)**: Measured end-to-end latency on physical target hardware.
- ⏳ **Gate G6 (PENDING)**: Final paper publication and reproducible artifact release.

---
*Optimized for Obsidian and Web with KaTeX LaTeX rendering, Mermaid diagrams, Canvas maps, and bidirectional Wikilinks.*
