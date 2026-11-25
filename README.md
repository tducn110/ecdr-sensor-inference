# ECDR Research Repository

**Event-Conditioned Certified Decision Reuse (ECDR)** for sequential sensor inference.

- **Final Research Report (1-Column Report, IEEE References, EN):** [LaTeX Source](04_SYNTHESIS/week5/reports/ECDR_Final_Report_EN.tex) $\cdot$ [Compiled PDF](ECDR_Final_Report_EN.pdf)
- **Engineering Deep-Dive Blog Post:** [The Illusion of Low-Hanging Fruit in Edge AI: Why Caching Invariant Decisions on Wearables is Harder Than It Looks](BLOG.md)

---

## 1. Executive Summary & Empirical Verdict

This repository investigates whether overlapping temporal windows in human activity recognition ($W=128$, stride $S=64$ at $50\text{ Hz}$ on HAPT) can be mathematically certified to skip redundant neural network forward passes while preserving a frozen classifier's top-1 decision and reducing end-to-end CPU latency.

Through full-stream evaluation of 15 executable methods out of 17 registered entries (`M00`--`M16`) across $5,053$ chronological HAPT test windows ($5,034$ adjacent within-recording pairs, $3,572$ supervised windows, $15,159$ paired timing observations per method) and follow-up diagnostics:

1. **Absolute Decision Preservation via Fail-Closed Design ($100\%$ Agreement):** Every executed method maintains $100\%$ decision agreement ($A_{\text{full}} = 1.0000$, $D_{\text{all}} = 0$) relative to the frozen `SmallCNN1D` reference ($K=12$ classes, $283,692$ parameters, $1,172,992$ Conv/Linear MACs, SHA-256 `8ffc5dd8...`).
2. **Vanishing Regional Coverage at Canonical Stride ($R = 0$ hits):** Because pre-test validation yielded $0$ positive/member/hit proposals across 20 candidates, formal regional reuse remained disabled (`formal_reuse_enabled = false`), and all regional methods (`M05`--`M13`) recorded $0$ hits ($0.4479\text{--}7.1278\text{ ms}$ vs.\ $0.3420\text{ ms}$ baseline). Post-Week 4 pairwise diagnostics show a median normalized $\ell_2$ displacement of $25.492$ (median $\ell_\infty = 3.845$) against a $0.0149$ Lipschitz radius ($1,562.7\times$ median gap ratio).
3. **Exact Bitwise Feature Reuse vs.\ CPU Slicing Overhead (`M03`):** Shift-aligned stagewise convolutional reuse (`M03`) achieves $0.0$ logit error across all $5,034$ adjacent pairs ($A_{\text{reuse}} = 1.0000$), cutting Conv MACs by $41.47\%$ and total MACs by $32.15\%$. However, Python tensor slicing and mandatory recomputation of the `Flatten(2048)->Linear(128)` head ($92.45\%$ of parameters) raise mean latency from $0.3420\text{ ms}$ to $1.9273\text{ ms}$ ($0.1775\times$ speed ratio, $+463.47\%$ overhead).
4. **Stride Sensitivity and Model Scaling Boundaries:** Across finer strides (`04_SYNTHESIS/week5/evidence/stride_audit.json`), $S=1$ ($322,856$ pairs) elevates neighboring top-1 agreement to $97.97\%$, yet boxes of radius $10^{-6}\text{--}10^{-2}$ admit $0$ pairs, a radius-$0.1$ box admits $49,677$ pairs ($7$ decision disagreements), and ordinary float64 IBP finds $0/49,677$ positive 11-margin estimates. On scaled synthetic backbones (`output/post_week4/feasibility_study.json`), borrowing CNN proxy verifier cost $T_v = 1.3346\text{ ms}$ yields a modeled capacity crossover near $27.6\text{ MMACs}$ and a $2.53\times$ ceiling at $R=100\%$ on `TemporalTransformer1D` ($115\text{ MMACs}$, $r_{\min} = 39.5\%$); actual stream speedup remains `NOT_ESTABLISHED`.

---

## 2. Canonical Research Sources & IEEE Reports

1. **Primary Final Research Report (English, 1-Column, IEEE References):** [LaTeX Source](04_SYNTHESIS/week5/reports/ECDR_Final_Report_EN.tex) $\cdot$ [Compiled PDF](ECDR_Final_Report_EN.pdf) $\cdot$ [BibTeX](04_SYNTHESIS/week5/reports/refs.bib) — Complete 51-page synthesis report authored by Nguyễn Tâm Đức (Supervisor: Trần Thế Vũ) covering Weeks 1–4 progression, mathematical formulation, Higham `binary32` enclosure (`ecdr/week4/numerics.py`), domain geometries, 17-method replay, 24-hypothesis ledger, and post-Week 4 diagnostics.
2. **Engineering Deep-Dive Blog Post:** [BLOG.md](BLOG.md) — Accessible narrative examining certified vs.\ heuristic caching, hardware bottlenecks, and empirical lessons.
3. **Foundational & Week 4 Technical Reports:**
   - Week 1: [02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex) / [PDF](00_OVERVIEW/02_Week_1_Project_Brief.pdf) and [SOURCE_OF_TRUTH.md](00_OVERVIEW/week1/SOURCE_OF_TRUTH.md).
   - Week 2 Data & Model: [01_DATA/03_Week_2_Data_Foundations.tex](01_DATA/03_Week_2_Data_Foundations.tex) and [02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex](02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex).
   - Week 3 Math: [03_MATH/03_Week_3_Mathematical_Foundations.tex](03_MATH/03_Week_3_Mathematical_Foundations.tex) and [03_MATH/README.md](03_MATH/README.md).
   - Week 4 Synthesis: [Week_4_Implementation_and_Evaluation.pdf](04_SYNTHESIS/week4/reports/Week_4_Implementation_and_Evaluation.pdf) $\cdot$ [Week_4_Explained.pdf](04_SYNTHESIS/week4/reports/Week_4_Explained.pdf) $\cdot$ [REPRODUCE.md](04_SYNTHESIS/week4/reports/REPRODUCE.md).

---

## 3. Cryptographic Seals & Integrity Manifest

- **Protocol Seal:** `5680676232ddb0103b69368a3200353ef1a3ac31ef80fc24876c0e14628b811b` ([output/week4/week4_initial_20261103/protocol_seal.json](output/week4/week4_initial_20261103/protocol_seal.json)).
- **Protected Files Check:** 214 historical files locked by SHA-256 (`scripts/week4_audit.py check` $\to$ `ALL CLEAN`).
- **Frozen Reference Checkpoint ($K=12$, Seed 43, Ep 8):**
  `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1` ([output/week1_closeout_20261102/reference/selected_best.pt](output/week1_closeout_20261102/reference/selected_best.pt)).
- **G2-Full Closure Checkpoint ($K=12$, Seed 43, Ep 6 — separate support study):**
  `c4f892bac5b5852f8070a3032288ef7f7c7b34dc4f487612b4a147ae59615d15` ([output/post_week4/g2_k12_full_best.pt](output/post_week4/g2_k12_full_best.pt)).
- **Preprocessing Identity:** `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

---

## 4. Gate Progression & Final Dispositions

| Gate | Scope | Status | Verified Evidence & Outcomes |
|---|---|---|---|
| **G0** | Literature & RQ | `PASS` | Scoped primary-source comparisons; fail-closed decision preservation contract. |
| **G1** | Data & Provenance | `PASS` | 61 recordings, $17,453$ windows ($17/4/9$ subjects); 1-sample origin caveat (98 rows) disclosed. |
| **G2** | Reference & Support | `PASS (Separate Study)` | Frozen ref: $89.05\%$ Acc / $0.7763$ F1. Post-W4 Span-Anchored closure: 0/18 blockers; Full-G2 Macro-F1 $0.8537 \pm 0.0020$. |
| **G3** | Numerical & Verifier | `VERIFIED (Soundness Only)` | Higham `binary32` $\gamma_n$ bridge (`ELIGIBLE_FOR_PROFILE`); CROWN 2D surrogate validated conditional; 0 streaming hits. |
| **G4** | Domain Structure | `VERIFIED (Zero Hits)` | Diagonal, Wave ($768 \to 384$), $K=8$ PCA Affine ($9,944$ train pairs), and `M03` bitwise equivalence ($0.0$ err). |
| **G5** | End-to-End Replay | `NOT_ESTABLISHED` | $5,053$ test queries: 0/17 methods pass `H24` trilemma; `M03` slows CPU by $+463.47\%$; regional hits $= 0$. |
| **G6** | Synthesis & Code | `VERIFIED` | **376/376** unit tests PASS; 214/214 protected files intact; final report & receipts published. |

---

## 5. Follow-up Studies

1. **Gate G2-Full $K=12$ Closure (`scripts/post_w4_g2_full_closure.py`):**
   - On the canonical grid ($S=64$), transition classes (labels 7--12) suffered 10 canonical blockers (18 in pure-only) due to grid-phase misalignment across $1.5\text{--}6.0\text{ s}$ transition spans.
   - Span-Anchored Transition Extraction ($M=9$ windows/span) + Class-Balanced Focal Loss ($\gamma=1.5, \beta=0.35$) eliminated all blockers ($0/18$) and raised 3-seed Full-G2 Macro-F1 to $0.8537 \pm 0.0020$ ([g2_full_closure.json](output/post_week4/g2_full_closure.json)).
2. **Stride & Scaling Diagnostics (`stride_audit.py`, `post_w4_feasibility_study.py`):**
   - Finer stride $S=1$ improves empirical neighboring agreement ($97.97\%$), yet tested IBP boxes still return $0/49,677$ positive margins.
   - Forward scaling modeling places proxy break-even around $27.6\text{ MMACs}$ ($R > 39.5\%$ for `TemporalTransformer1D`, $2.53\times$ ceiling at $R=100\%$), while actual streaming acceleration remains open ([feasibility_study.json](output/post_week4/feasibility_study.json)).

---

## 6. How to Reproduce

Run verification from the repository root:

```bash
# 1. Verify all 214 protected historical files
.venv/bin/python scripts/week4_audit.py check

# 2. Run the complete unit and regression test suite (376 tests)
.venv/bin/python tests/run_all_tests.py

# 3. Reproduce Post-Week 4 G2-Full closure and feasibility studies
.venv/bin/python scripts/post_w4_g2_full_closure.py
.venv/bin/python scripts/post_w4_feasibility_study.py

# 4. Reproduce Week 5 stride diagnostic and latency recounts
.venv/bin/python 04_SYNTHESIS/week5/evidence/stride_audit.py
python3 04_SYNTHESIS/week5/evidence/reconcile_replay.py
python3 04_SYNTHESIS/week5/evidence/paired_latency.py
```

---

## 7. Repository Layout

- `00_OVERVIEW/`: Project briefs, frozen baseline specs, and primary-source context audits.
- `01_DATA/`: HAPT sensor data reconstruction, split contracts, and provenance audits.
- `02_MODEL/`: `SmallCNN1D` architecture analysis and historical `ECDRConv1DNet` GAP reference.
- `03_MATH/`: Mathematical formulations, perturbation bounds, and `binary32` verification proofs.
- `04_SYNTHESIS/`: Final research report (`week5/reports/`), evidence scripts (`week5/evidence/`), and synthesis records.
- `ecdr/`: Core verification, numerical containment (`week4/numerics.py`), domain geometry, and replay library.
- `scripts/`: Auditing, training, benchmarking, and post-Week 4 studies.
- `tests/`: 41 unit test modules (376 passing regression checks).
- `archive/`: Legacy scratch drafts and closeout backups.
