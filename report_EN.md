# ECDR — Scientific Research Report and Current State

**Updated:** 22/10/2026. **Type:** Scientific working report.  
**Gate Record:** G0 repo-record PASS; G1 COMPLETE_WITH_REVISIONS; G2 repo-record PASS; G3–G6 PENDING. Novelty and net savings remain unverified.

This document replaces legacy simulation contents with a fully traceable, evidence-grounded assessment. The previous legacy version remains in Git history; earlier simulation results and proposed architectures do not represent current findings. Standalone LaTeX scientific dossier: [latex/dossier.tex](latex/dossier.tex). Executive summary: [PROJECT_BRIEF.md](PROJECT_BRIEF.md). Domain contract: [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).

---

## Executive Summary

ECDR investigates whether repeated full forward evaluations of a neural classifier on consecutive sensor time-series windows can be avoided by soundly reusing decisions from a trusted anchor. A query window may reuse an existing decision only if it lies within a certified domain where the lower bounds of all competing class-pair margins remain strictly positive throughout. Three distinct pillars require independent validation: formal soundness, the practical utility of event-conditioned structure, and end-to-end compute savings. The project currently possesses verified HAPT data pipelines, a frozen baseline model, and exploratory candidate-region audits; it lacks an implemented verifier, runtime reuse measurements, and net savings benchmarks. All reported numbers reflect artifacts saved on 15/10/2026; the documentation update on 22/10/2026 did not rerun models or benchmarks.

---

## 1. Problem Statement and Scope

The reference pipeline classifies triaxial accelerometer and gyroscope segments into six fundamental activities. Evaluating the complete network for every sliding window creates potential computation redundancy, but temporal overlap or similarity metrics do not guarantee that the classifier preserves its decision. ECDR proposes requiring a formal certificate before reusing any previously computed decision.

The targeted guarantee is **decision agreement with the frozen backbone**, not ground-truth correctness or preservation of exact logit values. The project does not investigate LLM context windows or key-value caching; no embedded physical deployment, battery measurements, or hardware energy benchmarks currently exist.

---

## 2. Research Questions and Hypotheses

**RQ1:** Can event-conditioned class-pair bounds certify decision invariance on consecutive inputs more frequently than global Lipschitz bounds, while keeping overall overhead low enough to achieve net inference compute savings?

- **RQ2:** Which domain, membership, identity, and numerical assumptions are necessary and sufficient for sound decision reuse?
- **RQ3:** Does event-conditioned region geometry improve certified coverage or reuse rates under fair, matched baseline comparisons?
- **RQ4:** What reuse rate and amortization horizon are required to offset the costs of certificate construction, event classification, membership testing, fallback inference, and cache rebuilds?

**H1:** Valid lower bounds that remain strictly positive across the entire declared domain, combined with conservative membership testing, provide a sound sufficient condition for decision preservation.  
**H2:** Structured candidate regions yield non-trivial certified coverage advantages in at least one operating regime.  
**H3:** There exists a practical operating regime where net compute savings are positive. Hypotheses H2 and H3 remain open empirical questions; no rigid 50% latency reduction claim is made.

---

## 3. Window, Event, and Region Definitions

| Object | Definition | Current Status |
|---|---|---|
| Time-series window | Raw input $6 \times 128$, 128 time steps per channel; $2.56\text{s}$ at $50\text{Hz}$ | Source pipeline built |
| Event descriptor/selector | Characterizes signal variation and selects template/budget | Online selector remains proposed |
| Candidate region | Set of inputs linked to a trusted anchor under geometric constraints | Audited candidates; no certificate |
| Certified region | Same domain backed by verified positive lower margins and verified identity | Unimplemented |

Stride $s = 64$ ($1.28\text{s}$) produces 50% temporal overlap. However, shared samples shift coordinate positions within the flattened tensor; overlap does not imply decision invariance. The reconstructed dataset contains 17,453 raw windows, differing from the 10,929 handcrafted feature records on the [UCI HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) portal.

---

## 4. Prior Work and Candidate Gap

[CROWN](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html) and [automatic LiRPA](https://papers.nips.cc/paper/2020/hash/0cbc5671ae26f67871cb914d81ef8fc1-Abstract.html) provide linear relaxation bound machinery. The [auto_LiRPA API](https://auto-lirpa.readthedocs.io/en/latest/api.html) supports a specification matrix $C$ mapping logits to margins; this capability has not yet been verified for numerical soundness on the ECDR computation graph.

According to the [literature ledger](02_LITERATURE.md), prior systems such as LipCache, Wei & Liu, NoScope, DeltaCNN, STBP, and time-series smoothing address adjacent efficiency problems under differing guarantees. ECDR claims neither the first temporal reuse, nor the first neural verifier, nor the first class-pair margin formulation. The candidate contribution lies in the combination of a direct-backbone certificate, event-conditioned region geometry, anchor lifecycle management, and rigorously measured end-to-end costs. Claims C11 (novelty) and C12 (net savings) remain UNVERIFIED.

---

## 5. Formulation and Methodology

Let raw input be $x \in \mathbb{R}^{768}$, fixed train-fit normalization be $N(x) = A(x - m)$, normalized tensor be $z = N(x)$, and reference pipeline be $F = f_\theta \circ N$. A trusted anchor $x^r$ must have executed reference inference; $a = \arg\max_k F_k(x^r)$.

The minimal G3 prototype targets a normalized diagonal domain:

$$\mathcal{Z}_e(z^r) = \{ z^r + B_e u : \|u\|_\infty \le 1 \}, \quad B_e = \text{diag}(b), \; b_i \ge 0, \quad \mathcal{S}_e = N^{-1}(\mathcal{Z}_e)$$

For every competing class $j \ne a$, if $\ell_{aj}$ is a valid lower bound of $g_{aj}(z) = f_{\theta, a}(z) - f_{\theta, j}(z)$ over $\mathcal{Z}_e$ and $\min_j \ell_{aj} > 0$, every point in the domain preserves winner $a$. Proof: logit $a$ strictly exceeds every competing logit. This is a sufficient condition, not a proof that an arbitrary floating-point implementation is sound.

If $g_{aj}(z) \ge \alpha^T z + \beta$ is a valid affine relaxation over the domain:

$$\ell_{aj} = \alpha^T z^r + \beta - \|B_e^T \alpha\|_1$$

Global radius $\rho$ uses $B_e = \rho I$. Coefficients valid on a smaller domain must not be applied to a larger candidate without proof.

### Reuse Gate and Lifecycle Protocol

1. Select candidate from available runtime metadata.
2. Retrieve certificate matching anchor, checkpoint, preprocessing, domain, and verifier identity.
3. Verify membership and ensure all stored lower margins are strictly positive.
4. Reuse class $a$ only if all checks pass; otherwise fallback to reference inference.
5. Only inputs completing reference inference can establish a new trusted anchor.
6. Reset at recording boundaries; invalidate certificates upon identity changes.

---

## 6. Current Baseline Empirical Results

All tables are extracted from saved JSON artifacts and source code; no experiments were rerun. Subjects and recordings are strictly disjoint across splits.

| Split Denominator | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Stream windows | 9,978 | 2,422 | 5,053 | 17,453 |
| Pure $K=6$ windows | 5,693 | 1,450 | 2,874 | 10,017 |
| Subjects | 17 | 4 | 9 | 30 |

Evaluated backbone architecture: Conv1D ($6 \to 32 \to 64 \to 64$), kernel sizes 5/3/3, BatchNorm/ReLU, MaxPool after the first two convolutions, Flatten $2048 \to \text{Linear } 128 \to \text{ReLU} \to \text{Linear } 6$; **282,918 parameters**.

| Seed | Selected Epoch | Val Macro-F1 | Test Macro-F1 | Test Accuracy |
|---|---:|---:|---:|---:|
| 42 | 4 | 0.9454 | 0.9105 | 0.9085 |
| 43 | 3 | 0.9418 | 0.8906 | 0.8897 |
| 44 | 25 | 0.9504 | 0.9354 | 0.9349 |

Checkpoints were selected via validation performance. Across 3 seeds on the fixed split, macro-F1 is $0.9122 \pm 0.0183$ and test accuracy is $0.9110 \pm 0.0185$ (mean $\pm$ population standard deviation; not a confidence interval). Evaluated on **2,874 pure $K=6$ test windows**. Full test stream comprises **5,053 windows**.

| Candidate Membership | All Test Pairs | Same-Activity Pairs | Label-Change Pairs |
|---|---:|---:|---:|
| Diagonal q75 | 25.53% | 45.81% | 0.21% |
| Diagonal q90 | 31.94% | 52.54% | 2.26% |
| PCA (108 components) | 35.42% | Not stored separately | Not stored separately |

Evaluated across 17,392 adjacent pairs (5,034 on test). PCA exceeds q90 by **+3.48 percentage points**, but has not been matched for budget or coverage. This does not represent multi-step certified reuse.

CPU batch-1 forward latency (resident normalized RAM tensors): mean $0.2511\text{ms}$, p50 $0.2357\text{ms}$, p95 $0.3313\text{ms}$ (6 Torch threads, 200 warmups, 1,000 runs). Excludes windowing, event classification, membership checks, certificate construction, cache lookups, and I/O.

---

## 7. Cost Model and Analytical Bounds

Under an approximately constant forward-cost model:

$$\mathbb{E}[T] \approx T_e + T_m + (1 - r) T_f + q T_b$$
$$\text{Net saving} \iff r T_f > T_e + T_m + q T_b$$

Where $T_e$ is mean event overhead, $T_m$ is lookup/membership time, $T_f$ is reference forward latency, $r$ is reuse fraction, $q$ is rebuild rate per window, and $T_b$ is mean build time. Initial construction must be accounted for within counted builds or as $T_\text{init} / N$.

Given $T_f \approx 0.2511\text{ms}$ and assuming an illustrative 50% reuse rate, the allowable total overhead budget is $125.55\mu\text{s}$ per window. This is a parametric sensitivity estimate, not an empirical measurement.

---

## 8. Four G3 Deliverables and Evaluation Roadmap

| Deliverable | Completion Requirements |
|---|---|
| Domain Contract | Unified coordinate system, normalization, model/anchor/domain identity |
| Minimal Diagonal Verifier | Sound class-pair lower bounds over frozen CNN domain; documented numerical contract |
| Certified Reuse Gate | Membership verification, valid positive margins, fallback path, anchor lifecycle logging |
| Component Instrumentation | Isolated timing for construction, membership, fallback, reference forward, and rebuilds |

Gate G4 evaluates whether structured regions outperform diagonal boxes under matched constraints. Gate G5 measures net savings using complete end-to-end traces on target hardware. Gate G6 packages reproducible artifacts and the final manuscript.

---

## 9. Limitations and Conclusion

Prior experimental runs have not been reproduced during this documentation cycle; Gate G3 lacks executable implementation and numerical proofs; measurements are limited to a single dataset, fixed split, and $K=6$ model; CPU-only forward numbers do not represent end-to-end device latency; and literature coverage is insufficient to assert novelty.

ECDR possesses valid empirical foundations to investigate feasibility, but has not yet proven a viable acceleration mechanism. The prescribed next step is implementing the domain contract and minimal diagonal verifier before considering complex PCA expansions.

---
*Source Authorities: [metrics](data/g2_metrics.json), [training](data/g2_training.json), [model](data/g2_model_summary.json), [latency](data/g2_cpu_latency.json), [event audit](data/event_audit.json), [split](data/split_audit.json), [theory](04_THEORY.md), [G3 contract](07_G3_DOMAIN_CONTRACT.md), [backbone](scripts/backbone.py), [bibliography](refs.bib).*
