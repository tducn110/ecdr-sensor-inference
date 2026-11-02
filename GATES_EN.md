# GATES.md — Research Evidence Gates (English)

> **Current Week 1 scope (2026-11-02):** The executed run `week1_ref_1793620216_9d6c1d85` provides a verified SmallCNN1D K=12 baseline: seed 43 at epoch 8, 17 internal checks passed, 89.0538% supervised test accuracy and 77.6265% macro-F1 on 3,572 windows. This verifies the baseline execution; formal certification and net savings remain pending Week3/G5 evidence. The historical gate body below remains preserved.

> Reconciled 22/10/2026, documentation only. **G0 PASS; G1 COMPLETE_WITH_REVISIONS; G2 PASS; G3–G6 PENDING.** The three completed baseline stages are empirical milestones, not unconditional scientific claims of novelty, certified reuse, or net compute savings. New reports or specifications do not pass a gate without required empirical evidence. Saved measurements were not rerun. Current theory: [04_THEORY.md](04_THEORY.md); minimal next contract: [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).

---

- [x] **G0 — Initial Literature & RQ Baseline (PASS)** *(Historical 15/10/2026 gate)*
  - **EVIDENCE:** Initial 7-entry bibliography and scoped PDF/API checks; comparison ledger, four-axis taxonomy, and gap formulation in `01_RQ_GAP.md` and `02_LITERATURE.md`. The 21/10/2026 pilot added NoScope and refreshed Wei metadata; the bibliography gained CROWN and automatic-LiRPA citations on 22/10/2026, reaching 10 records.
  - **PRINCIPLE:** Every promoted factual claim requires an exact source and scope; GuardNet vs backbone, verification vs forward latency, and certified-accuracy vs runtime metrics remain strictly separated.
  - **OPEN:** Novelty (C11) and net compute savings (C12) remain UNVERIFIED. Expanded literature review must actively search for conceptual collisions rather than inferring novelty from G0.

- [x] **G1 — Dataset Feasibility (COMPLETE_WITH_REVISIONS)** *(Historically REVISE)*
  - **EVIDENCE:** Scripts 01–06b and saved integrity/split/label artifacts; chronological ordering and disjoint subject/recording splits verified. Corrected raw diagonal q90 test membership: same-activity 52.54%, label-change 2.26%, all pairs 31.94%.
  - **CORRECTION:** The Script 05 row-index bug and recomputed F1/F2/F3 findings are documented in `03_DATA_AUDIT.md`; pre-fix values remain historical artifacts.
  - **LIMITATION:** Budget names q50/q75/q90 represent statistical thresholds, not implemented semantic event detectors. Centered PCA audit achieves 35.42% test coverage, but this does not represent a decision certificate or a budget-matched comparison.
  - **OPEN:** Enforce consistent domain definitions and normalization; test multi-step anchor coverage and controlled structured comparisons.

- [x] **G2 — Frozen Backbone Baseline (PASS)** *(Saved 15/10/2026 results)*
  - **EVIDENCE:** `configs/`, `scripts/backbone.py`, scripts 07–10, model checkpoint, and `data/g2_training.json`, `g2_metrics.json`, `g2_model_summary.json`.
  - **SCOPE:** $K=6$ pure-only task scope, test denominator 2,874; seeds 42/43/44. Validation-selected seed 44 achieves test accuracy 0.9349, macro-F1 0.9354; 3-seed macro-F1 is $0.9122 \pm 0.0183$ (population standard deviation on fixed split).
  - **MODEL:** 282,918 parameters with BatchNorm and MaxPool; frozen train-fit normalization. Weights, preprocessing, and evaluation mode must match reference verifier.
  - **LATENCY:** Saved forward-only CPU batch-1 latency: mean 0.2511 ms, p50 0.2357 ms, p95 0.3313 ms on normalized resident RAM tensors (6 Torch threads). Excludes preprocessing and fallback paths.

- [ ] **G3 — Sound Minimal Verifier, Membership & Reuse Gate (PENDING)**
  - **REQUIRED DOMAIN:** Implement normalized diagonal contract in `07_G3_DOMAIN_CONTRACT.md`; raw $x$, fixed $N$, $F = f_\theta \circ N$, normalized box, and raw preimage must remain mathematically and numerically consistent.
  - **REQUIRED VERIFIER:** Compute valid class-pair bounds across the entire declared domain; positive lower margins ($\min_j \ell_{aj} > 0$) are sufficient. Ties, unsupported operators, or inconclusive bounds must fall back to reference inference.
  - **REQUIRED GATE:** Conservative numerical membership independent of heuristics; strict model/anchor/domain identity verification; anchor refresh permitted only after genuine reference forward evaluations; recording-boundary resets.
  - **REQUIRED SOUNDNESS:** Document both mathematical derivations and executable numerical error-containment justifications. Zero observed test mismatches does not constitute proof.
  - **REQUIRED INSTRUMENTATION:** Measure individual timings for event classification, membership testing, reference forwards, and certificate builds/rebuilds.

- [ ] **G4 — Useful Event-Conditioned Structure (PENDING)**
  - **ASSESSMENT:** Distinguish bound-method tightness (fixed domain) from region value (fixed bound method). Compare global, diagonal, and correlated candidates under controlled matching criteria.
  - **BENCHMARKS:** Include certified baseline models and an explicitly labeled heuristic confidence gate. A smaller candidate volume or uncertified PCA coverage does not establish certified reuse superiority.
  - **REQUIREMENT:** Frozen-policy, multi-step anchor ablation with valid bounds; claims of semantic event selection require an implemented, tested online selector.

- [ ] **G5 — Measured End-to-End Net Compute Savings (PENDING)**
  - **ASSESSMENT:** Compare complete reference inference and reuse pathways under identical preprocessing, hardware, workloads, and declared accounting horizons. Include initialization, lookup, fallback, and certificate rebuild costs.
  - **MODEL:** Apply $\mathbb{E}[T] = T_e + T_m + (1 - r) T_f + q T_b$ under constant forward costs, or evaluate conditional query traces when forward costs vary. Include initial construction within counted builds or add $T_\text{init} / N$.
  - **REQUIREMENT:** End-to-end latency tables (mean, p50, p95), throughput, memory consumption, and hardware specifications. Negative results must be reported transparently.

- [ ] **G6 — Scientific Dossier Publication & Reproducibility (PENDING)**
  - **ASSESSMENT:** Source-grounded literature synthesis, formal proofs, empirical data, uncertainty metrics, reproducible scripts, and clear boundary disclosures.
  - **DELIVERABLE:** Fully compiled LaTeX dossier and reproducible code package, clearly delineating verified empirical evidence from ongoing hypotheses.
