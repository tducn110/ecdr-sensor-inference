# ECDR Research Repository

**Event-Conditioned Certified Decision Reuse (ECDR)** for sequential sensor inference.

📖 **Engineering Deep-Dive Blog Post:** [The Illusion of Low-Hanging Fruit in Edge AI: Why Caching Invariant Decisions on Wearables is Harder Than It Looks](BLOG.md)  
📄 **Week 5 Comprehensive Research Paper:** [LaTeX Monograph](04_SYNTHESIS/week5/reports/ECDR_Week5_Comprehensive_Research_Paper_EN.tex) $\cdot$ [Final PDF Report](ECDR_Week5_FINAL_EN.pdf)

---

## 1. Executive Summary & Final Scientific Verdict

The ECDR program investigates whether the 50% temporal overlap between consecutive sensor windows ($W=128$, stride $S=64$ at 50 Hz on HAPT) can be certified mathematically to reuse the reference classifier's decisions, eliminating redundant neural network inference without accuracy degradation.

Through rigorous end-to-end evaluation across 17 methods (M00–M16) on the full test stream ($N=5,053$ windows, $5,034$ adjacent pairs, $3,572$ labeled windows) and Post-Week 4 scaling analyses:
1. **Absolute Decision Preservation via Fail-Closed Design ($100\%$ Soundness):** All certified methods achieve $100\%$ decision agreement ($A_{\text{full}} = 1.0000$, $D_{\text{all}} = 0$) relative to the frozen reference `SmallCNN1D` ($K=12$ classes, 283,692 parameters, 1.17 MMACs).
2. **Phase-Shift Orthogonality ($R_{\text{cert}} = 0\%$ at $S=64$):** On the coarse sliding grid ($S=64$ samples = 1.28 s), human gait periodicity creates a large normalized displacement ($\|\Delta z\|_\infty = 3.6349$, $\|\Delta z\|_2 = 25.49$), which exceeds maximum certified radii by over 1,500$\times$. Static $\ell_p$ perturbation regions cannot cover inter-window drift at $S=64$.
3. **Exact Bit-Level Feature Reuse vs. CPU Memory Slicing Overhead:** Incremental feature reuse (M03) achieves $100\%$ bit-exact feature preservation across 12 internal Conv3 channels, but Python/PyTorch memory slicing at batch=1 costs $1.9273$ ms compared to $0.3420$ ms for monolithic inference ($0.1775\times$ speed ratio).
4. **Viable Operating Regimes (Post-Week 4 Feasibility):** When operating on high-density streams ($S=1$), certified reuse surges to **$87.31\%$** with $100\%$ soundness. With larger neural backbones ($T_f \ge 10$ ms) and lazy on-demand certification, ECDR achieves **$5.88\times\text{--}6.31\times$ net speedup**.

---

## 2. Canonical Research Sources & IEEE Technical Reports

The research documentation is standardized under the **IEEE Transactions (`IEEEtran.cls`)** framework:

1. **Week 5 Comprehensive Research Monograph (English):** [LaTeX Source](04_SYNTHESIS/week5/reports/ECDR_Week5_Comprehensive_Research_Paper_EN.tex) $\cdot$ [Final PDF Report](ECDR_Week5_FINAL_EN.pdf) $\cdot$ [BibTeX](04_SYNTHESIS/week5/reports/refs.bib) — Complete 12-section monograph by student Nguyễn Tâm Đức (Advisor: Trần Thế Vũ) synthesizing mathematical proofs, 17-method taxonomy, 24-hypothesis ledger, and scaling feasibility.
2. **Research & Engineering Blog Post:** [BLOG.md](BLOG.md) — Comprehensive, humanized deep dive into the 5-week research journey, the biomechanical phase-shift discovery, and CPU cache slicing paradox.
3. **IEEE Dossier Scientific Report (Vietnamese):** [LaTeX Source](04_SYNTHESIS/02_Dossier_Scientific_Report.tex) $\cdot$ [Compiled PDF (6 pages)](04_SYNTHESIS/02_Dossier_Scientific_Report.pdf) — Complete 2-column IEEE Transactions monograph synthesizing G0–G6, mathematical theorems, floating-point error contract $\tau_{\text{fp}}$, and G2-Full $K=12$ closure.
4. **IEEE Full Synthesis Technical Monograph (Vietnamese):** [LaTeX Source](04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_VI.tex) $\cdot$ [Compiled PDF (13 pages)](04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_VI.pdf) / [Root Mirror](ECDR_Full_Synthesis_Report_VI.pdf) — Exhaustive evaluation of 17 methods, 24 hypotheses (H01–H24), data denominators, and scaling break-even analysis.
5. **IEEE Mechanism & Future Directions Monograph (Vietnamese):** [LaTeX Source](04_SYNTHESIS/week4/reports/ECDR_Mechanism_Why_and_New_Directions_VI.tex) $\cdot$ [Compiled PDF (13 pages)](04_SYNTHESIS/week4/reports/ECDR_Mechanism_Why_and_New_Directions_VI.pdf) / [Root Mirror](ECDR_Mechanism_Why_and_New_Directions_VI.pdf) — Deep physical, mathematical, and architectural breakdown of successes and limitations.
6. **English Technical Sources (Week 4):** [Full Synthesis EN Source](04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_EN.tex) $\cdot$ [Mechanism EN Source](04_SYNTHESIS/week4/reports/ECDR_Mechanism_Why_and_New_Directions_EN.tex) $\cdot$ [Week 4 Evaluation PDF](04_SYNTHESIS/week4/reports/Week_4_Implementation_and_Evaluation.pdf) $\cdot$ [Week 4 Explained PDF](04_SYNTHESIS/week4/reports/Week_4_Explained.pdf).
7. **Foundations (Weeks 1–3):**
   - Week 1: [02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex) / [PDF](00_OVERVIEW/02_Week_1_Project_Brief.pdf) and [SOURCE_OF_TRUTH.md](00_OVERVIEW/week1/SOURCE_OF_TRUTH.md).
   - Week 2 Data: [01_DATA/03_Week_2_Data_Foundations.tex](01_DATA/03_Week_2_Data_Foundations.tex) / [PDF](01_DATA/03_Week_2_Data_Foundations.pdf).
   - Week 2 Model: [02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex](02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex) / [PDF](02_MODEL/01_ECDR_CNN_Model_Analysis_EN.pdf).
   - Week 3 Math: [03_MATH/03_Week_3_Mathematical_Foundations.tex](03_MATH/03_Week_3_Mathematical_Foundations.tex) / [PDF](03_MATH/03_Week_3_Mathematical_Foundations.pdf) and [03_MATH/README.md](03_MATH/README.md).
   - Theory Monograph: [04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.pdf](04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.pdf).

---

## 3. Cryptographic Seals & Integrity Manifest

All historical foundations are cryptographically sealed and continuously audited:
- **Protocol Seal:** `568067623fdc2b4c193ef1ef8e604fcaebafb72236a2829141be3f1e16fdf9c7` ([output/week4/week4_initial_20261103/protocol_seal.json](output/week4/week4_initial_20261103/protocol_seal.json)).
- **Protected Files Check:** 214 historical files locked by SHA-256 (`scripts/week4_audit.py check` $\to$ `ALL CLEAN`).
- **Frozen Reference Checkpoint ($K=12$, Seed 43, Ep 8):**
  `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1` ([output/week1_closeout_20261102/reference/selected_best.pt](output/week1_closeout_20261102/reference/selected_best.pt)).
- **G2-Full Closure Checkpoint ($K=12$, Seed 43, Ep 6):**
  `c4f892bac5b5852f8070a3032288ef7f7c7b34dc4f487612b4a147ae59615d15` ([output/post_week4/g2_k12_full_best.pt](output/post_week4/g2_k12_full_best.pt)).
- **Preprocessing Identity:** `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

---

## 4. Gate Progression & Final Status

| Gate | Scope | Status | Verified Evidence & Outcomes |
|---|---|---|---|
| **G0** | Problem Formulation | `PASS` | Formal decision preservation framing; fail-closed contract. |
| **G1** | Theoretical Foundations | `PASS / VERIFIED` | Comprehensive literature review and 17-method taxonomy. |
| **G2** | Data & Model Foundations | `PASS (Full Closure)` | Baseline $K=6$ ($93.49\%$ Acc) + Full $K=12$ closure (0/18 blockers; Macro-F1 $0.8537 \pm 0.0020$; Test Acc $89.85\% \pm 0.22\%$). |
| **G3** | Verification & Gate System | `VERIFIED` | 64-bit CROWN with outward rounding $\tau_{\text{fp}} \in [1.17\times 10^{-5}, 4.63\times 10^{-5}]$; 0/600 PGD violations. |
| **G4** | Structured Ablation | `VERIFIED` | 17 methods compared; CROWN achieves $100\%$ certified radius at anchor ($\rho=0$) vs. $0\%$ for global Lipschitz ($L_{\text{net}}=133.16$). |
| **G5** | End-to-End Evaluation | `VERIFIED` | Streaming evaluation ($5,053$ windows); honest rejection of reuse on coarse stride $S=64$ ($\|\Delta z\|_\infty = 3.6349$). |
| **G6** | Synthesis & Reproduction | `VERIFIED` | 376/376 unit tests PASS; 214/214 protected files intact; complete IEEE report suite published. |

---

## 5. Post-Week 4 Feasibility & Gate G2 Full Closure

1. **Gate G2-A & G2-B Full Closure:**
   - On the global sliding grid ($S=64$), postural transition classes (labels 7–12, median length $L \in [131, 246]$ samples) suffered from severe grid phase-mismatch ($P_{\text{hit}}(L) = (L-127)/64$), dropping $93.75\%$ of valid segments and leaving 18 sample-support bottlenecks.
   - We implemented the **Span-Anchored Transition Policy** ($M=9$ interior/midpoint anchor windows per transition span) with **Class-Balanced Focal Loss** ($\gamma=1.5, \beta=0.35$) across 3 independent seeds (`scripts/post_w4_g2_full_closure.py`).
   - Results: 0/18 blockers remaining; labeled samples expanded to 8,809 train / 2,147 val / 4,544 test; Macro-F1 reached $0.8537 \pm 0.0020$ and Transition-6 F1 surged to $0.7917 \pm 0.0024$ ([output/post_week4/g2_full_closure.json](output/post_week4/g2_full_closure.json), [G2_CLASS_DECISION.md](G2_CLASS_DECISION.md)).
2. **Dense Streaming & Large Backbone Scaling:**
   - Reducing window stride to $S=1$ elevates certified reuse to **$87.31\%$** with zero decision errors.
   - For larger models ($T_f \ge 10$ ms), lazy on-demand certification delivers **$5.88\times\text{--}6.31\times$ net speedup** ([output/post_week4/feasibility_results.json](output/post_week4/feasibility_results.json)).

---

## 6. How to Reproduce

Run verification from the repository root:

```bash
# 1. Verify all 214 protected historical files
.venv/bin/python scripts/week4_audit.py check

# 2. Run the complete test suite (376 tests)
.venv/bin/python -m unittest discover -s tests -v

# 3. Reproduce G2-Full ($K=12$) closure
.venv/bin/python scripts/post_w4_g2_full_closure.py

# 4. Reproduce Post-Week 4 feasibility benchmarks
.venv/bin/python scripts/post_w4_feasibility.py
```

---

## 7. Repository Layout

- `00_OVERVIEW/`: Project briefs, frozen baseline specs, and cryptographically preserved evidence registers.
- `01_DATA/`: HAPT sensor data audit, train/val/test splits, and data foundation reports.
- `02_MODEL/`: SmallCNN1D model architecture, training protocols, and weight analysis.
- `03_MATH/`: Mathematical formulations, perturbation bounds, and verification theory.
- `04_SYNTHESIS/`: Canonical IEEE technical reports, synthesis monographs, and reproduction guides.
- `ecdr/`: Core library (bounds, CROWN verifier, reuse gate, data loaders, schemas).
- `scripts/`: Auditing, training, benchmarking, and verification execution scripts.
- `tests/`: 376 automated regression and validation unit tests.
- `archive/`: Historical drafts, legacy stubs, and non-research compilation artifacts.
