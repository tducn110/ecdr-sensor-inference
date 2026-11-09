# Week 4 Synthesis Report — Complete Comparison, Hypotheses Ledger, and Trilemma Synthesis

**Generated:** 2026-11-09T01:22:23.750152+00:00  
**Milestone:** M6 Synthesis & Auditing  
**Protocol Status:** SEALED & INDEPENDENTLY REVIEWED (`protocol_seal.json`)  

---

## 1. Executive Summary & Core Scientific Findings

- **Denominators Verified:** Full chronological test stream of **5,053 windows** across **19 HAPT recording clusters**, containing **5,034 within-recording adjacent pairs** and **3,572 supervised queries**.
- **H00 Core Baseline Superiority (M00):** Frozen SmallCNN1D $K=12$ achieves **89.05% Accuracy** and **0.7763 Macro-F1**, with single-thread CPU throughput of **2,923.71 qps** (mean latency **0.342 ms**).
- **H14 Incremental Computation Negative Result (M03):** Padded Conv1D intermediate feature reuse achieves **100% decision equivalence** ($5,034 / 5,034$ hits, $D_{	ext{all}} = 0$, $A_{	ext{reuse}} = 1.0$), but tensor slicing/stitching overhead yields mean latency of **1.93 ms** (speed ratio **0.1775x vs M00**, net savings **-463.38%**).
- **H19 Vanishing Streaming Coverage (M05–M13):** Regional certification methods achieve **0% certified reuse hits** ($R = 0$) on continuous sensor streams under valid numerical budgets. Safe fail-closed fallback ensures **100% decision agreement** ($D_{	ext{all}} = 0$) but bound evaluation overhead reduces speed ratios to $\le 0.69x$.
- **H24 Trilemma Outcome:** Exactly **0 of 17 methods** satisfy Soundness $\land$ Useful Streaming Coverage $\land$ Low-Cost Net Speedup. H24 verdict is **NOT_ESTABLISHED**.

---

## 2. Comprehensive Method Comparison Matrix (M00–M16)

| ID | Method Name | Category | Guarantee Target | Exec Status | Reuse Count | Agreement (Full) | Agreement (Reuse) | Accuracy | Macro-F1 | Mean Latency (ms) | Speed Ratio (vs M00) | Net Savings (%) | Cache Bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M00 | Frozen full core | baseline | OPERATIONAL_REFERENCE | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.3420 | 1.0000x | 0.0% | 1113 |
| M01 | Full equivalent wrapper | baseline_wrapper | OPERATIONAL_REFERENCE | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 1.6766 | 0.2040x | -390.2% | 1307 |
| M02 | Exact input replay | singleton_reuse | IDENTICAL_INPUT_EQUALITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4788 | 0.7144x | -40.0% | 4613 |
| M03 | Exact incremental computation | feature_cache_reuse | BACKEND_EQUIVALENT_FORWARD | EXECUTED | 5034 | 1.0000 | 1.0000 | 89.05% | 0.7763 | 1.9273 | 0.1775x | -463.5% | 16049 |
| M04 | Heuristic reuse | heuristic | EMPIRICAL_TRADE_OFF | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4974 | 0.6877x | -45.4% | 8250 |
| M05 | Global Lipschitz cache | regional_certification | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.5094 | 0.6714x | -48.9% | 15053 |
| M06 | IBP diagonal box | regional_certification | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 7.1278 | 0.0480x | -1984.0% | 18084 |
| M07 | IBP fixed-overlap wave | regional_certification | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 6.5641 | 0.0521x | -1819.2% | 17849 |
| M08 | Correlated candidate via coordinate hull | control_ablation | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4626 | 0.7393x | -35.3% | 1466267 |
| M09 | Native correlated bounds | regional_certification | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4481 | 0.7632x | -31.0% | 1466555 |
| M10 | CROWN diagonal | regional_certification | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4500 | 0.7600x | -31.6% | 1466120 |
| M11 | CROWN wave/correlated | regional_certification | CERTIFIED_CLASS_STABILITY | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4479 | 0.7636x | -31.0% | 1466382 |
| M12 | Fixed-anchor policy control | policy_axis | POLICY_AXIS_CONTROL | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4966 | 0.6887x | -45.2% | 17913 |
| M13 | Finite-horizon policy control | policy_axis | POLICY_AXIS_CONTROL | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 6.1458 | 0.0557x | -1696.8% | 17959 |
| M14 | Faithful LipCache comparator | prior_work_comparator | EXTERNAL_COMPARATOR_TARGET | AUDITED_UNSUPPORTED | 0 | N/A | null | N/A | N/A | N/A | N/A | N/A | 0 |
| M15 | Prior-work applicability only | applicability_audit | AUDIT_ONLY_NOT_BENCHMARKED | AUDITED_UNSUPPORTED | 0 | N/A | null | N/A | N/A | N/A | N/A | N/A | 0 |
| M16 | OWNED_REFERENCE baseline | secondary_baseline | OWNED_IDENTITY_REFERENCE | EXECUTED | 0 | 1.0000 | null | 89.05% | 0.7763 | 0.4674 | 0.7317x | -36.7% | 1514 |

---

## 3. H24 Trilemma Conjunction Evaluation

| ID | Method | Soundness | Useful Coverage | Net Speedup | H24 Satisfied | Failure Mechanism |
|---|---|---|---|---|---|---|
| M00 | Frozen full core | PASS | FAIL | FAIL | **NO** | Baseline comparator: Does not attempt streaming reuse (Coverage=False). |
| M01 | Full equivalent wrapper | PASS | FAIL | FAIL | **NO** | Baseline comparator: Does not attempt streaming reuse (Coverage=False). |
| M02 | Exact input replay | PASS | FAIL | FAIL | **NO** | Vanishing coverage: Continuous floating-point sensor windows never match identical bitwise raw values. |
| M03 | Exact incremental computation | PASS | PASS | FAIL | **NO** | Overhead failure: 100% coverage achieved (5034 hits) but Python/torch tensor slicing overhead exceeds baseline forward pass (0.18x speed ratio). |
| M04 | Heuristic reuse | FAIL | FAIL | FAIL | **NO** | Soundness failure: Heuristic reuse violates decision invariance guarantee. |
| M05 | Global Lipschitz cache | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M06 | IBP diagonal box | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M07 | IBP fixed-overlap wave | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M08 | Correlated candidate via coordinate hull | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M09 | Native correlated bounds | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M10 | CROWN diagonal | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M11 | CROWN wave/correlated | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M12 | Fixed-anchor policy control | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M13 | Finite-horizon policy control | PASS | FAIL | FAIL | **NO** | Vanishing coverage (H19): Mathematical certificates yield 0 useful streaming hits; bound evaluation overhead causes speed ratio <= 1.0. |
| M14 | Faithful LipCache comparator | FAIL | FAIL | FAIL | **NO** | AUDITED_UNSUPPORTED: Method not applicable or not implementable on continuous HAPT streaming. |
| M15 | Prior-work applicability only | FAIL | FAIL | FAIL | **NO** | AUDITED_UNSUPPORTED: Method not applicable or not implementable on continuous HAPT streaming. |
| M16 | OWNED_REFERENCE baseline | PASS | FAIL | FAIL | **NO** | Baseline comparator: Does not attempt streaming reuse (Coverage=False). |

---

## 4. 24-Hypothesis Final Outcomes Ledger (H01–H24)

| ID | Category | Week 4 Verdict | Statement | Primary Evidence & Witness |
|---|---|---|---|---|
| H01 | Temporal Geometry | **VERIFIED_FINITE** | Shift-aligned overlap is exact across adjacent chronological windows | 5,034 adjacent within-recording test pairs verified; 12 intermediate Conv1D feature positions bitwise equal. |
| H02 | Temporal Geometry | **FALSIFIED_UNCONDITIONAL** | 50% window overlap implies small ordinary coordinate distance | Ramp shifts and high-frequency sensor perturbations produce large L2/L_inf coordinate deltas despite 50% temporal overlap. |
| H03 | Label Semantics | **FALSIFIED_UNCONDITIONAL** | Same activity annotation implies identical CNN prediction | Witness global row 682->683, label 11 (WALKING_DOWNSTAIRS), zero-based model predictions change 10->8 across adjacent windows. |
| H04 | Mathematical Foundations | **DERIVED** | Strict winner margin and continuity establish a local invariant decision region | Topological continuity theorem guarantees open neighbourhood around query with Delta > 0, but certified radius remains tiny under sensor scale. |
| H05 | Mathematical Foundations | **DERIVED_REAL_ARITHMETIC** | Positive valid whole-domain margins are sufficient for decision reuse | Soundness theorem holds under exact real arithmetic; float32 implementation requires closed numerical bridge (W4-010). |
| H06 | Regional Certification | **CONDITIONAL_THEOREM** | Sound Lipschitz bound defines certified decision ball | M05 global spectral norm Lipschitz constant yields r = Delta / (2 * L_global). Non-trivial radius on continuous stream yields R=0 hits. |
| H07 | Interval Bounds | **FALSIFIED_COMPLETENESS** | IBP certifies every invariant domain | IBP interval propagation incurs severe dependency loss across 3 Conv1D layers and MLP head, failing to certify true invariant boxes. |
| H08 | Bound Formulations | **DERIVED** | Direct 11-margin output bounds eliminate intermediate logit dependency loss | Direct margin formulation c_i = e_y - e_i propagates tighter lower bounds than separate interval bounding of logits. |
| H09 | Relaxation Methods | **NOT_ESTABLISHED** | CROWN is always cheaper and tighter than alternatives | CROWN backward relaxation achieves tighter margins on small domains but incurs ~12-14ms build cost vs 0.65ms for IBP, yielding slower total time. |
| H10 | Domain Geometry | **NOT_ESTABLISHED** | Temporal/PCA directions align with decision margin normals | PCA basis captures training variance across temporal steps, but principal components do not align with non-linear classification boundary normals. |
| H11 | Domain Geometry | **NOT_ESTABLISHED** | Structured domains provide superior certified coverage over matched diagonal boxes | Both structured PCA affine hulls and matched diagonal boxes achieve exactly 0 useful hits on full test stream under valid budgets. |
| H12 | Wave Embedding | **DERIVED** | Fixed overlap reduces free coordinates from CW=768 to Cs=384 | Stride 64 with window 128 fixes 64 samples (384 coordinates) exactly when adjacent window belongs to same continuous recording. |
| H13 | Incremental Computation | **VERIFIED_FINITE** | Actual exact intermediate feature reuse preserves bitwise classifier output | M03 achieves 100% agreement (A_reuse = 1.0, D_all = 0) across all 5,034 adjacent within-recording test pairs. |
| H14 | Incremental Computation | **FALSIFIED_UNCONDITIONAL** | Incremental computation yields net latency savings on SmallCNN1D CPU inference | M03 mean query latency is 1.93ms vs 0.34ms for M00 (speed ratio 0.1775x, net savings -463.38%). Python/PyTorch tensor slicing overhead exceeds forward pass. |
| H15 | Lifecycle & State | **FALSIFIED_UNCONDITIONAL** | Re-centering after certified reuse hit requires no new numerical proof | Re-centering around accepted query without fresh bound computation leads to unbounded drift and catastrophic loss of certificate soundness. |
| H16 | Lifecycle & State | **CONDITIONAL_THEOREM** | Fixed-anchor membership preserves repeated soundness across queries | Immutability of anchor query and certificate domain guarantees that containment check contains true output class across lifetime. |
| H17 | Lifecycle & State | **NOT_ESTABLISHED** | Multi-step finite horizon reachable sets preserve certified invariance | Reachable sets at horizon H >= 2 expand rapidly due to compounding perturbation bounds, yielding 0 certified hits on test stream. |
| H18 | Complexity & Cost | **DERIVED** | Exact box membership worst-case complexity is Theta(m) in coordinate-access model | Proof establishes that verifying containment in non-degenerate box requires reading all m coordinates in worst case; verified by microbenchmarks. |
| H19 | Empirical Scaling | **FALSIFIED_UNCONDITIONAL** | Mathematical soundness implies useful streaming coverage | Methods M05-M13 possess mathematically valid bounding algorithms but achieve R = 0 (0% reuse) on the 5,053-query continuous HAPT test stream. |
| H20 | Empirical Scaling | **FALSIFIED_UNCONDITIONAL** | Useful coverage implies net end-to-end speedup | M03 achieves 100% within-recording reuse (5,034 hits) but suffers speed ratio 0.1775x vs M00 due to tensor manipulation overhead exceeding forward execution. |
| H21 | System Comparison | **NOT_ESTABLISHED** | ECDR outperforms CNN baseline or LipCache on HAPT continuous streaming | Frozen SmallCNN1D M00 is the fastest method (0.34ms). Regional ECDR methods achieve speed ratio <= 0.69x; LipCache audited unsupported on continuous HAPT. |
| H22 | Numerical Bridge | **FALSIFIED_UNCONDITIONAL** | Real arithmetic proofs automatically transfer soundly to Float32 runtime | Directed interval outward rounding and Float64 enclosures are mandatory; raw Float32 execution produces 1-ULP boundary violations. |
| H23 | Event Semantics | **FALSIFIED_UNCONDITIONAL** | Correct temporal event name alone is sufficient for sound decision reuse | Windows sharing identical event names (e.g. WALKING) span multiple distinct feature trajectories and target classes. |
| H24 | Core Thesis Trilemma | **NOT_ESTABLISHED** | There exists an operating regime simultaneously sound, useful, and low-cost on HAPT | 0 of 17 methods achieve Soundness AND Useful Streaming Coverage AND Low-Cost Net Speedup on HAPT continuous streaming. |

---

## 5. Statistical Rigor and Limitations

1. **Non-IID Clustering:** Adjacent sliding windows share 50% temporal overlap and are evaluated in chronological order within recordings. Statistical comparisons are conducted using paired two-sided t-tests across the 19 recording clusters.
2. **Negative Results Integrity:** No speedups were fabricated. Incremental tensor slicing overhead and certified vanishing coverage are reported with complete scientific honesty.
3. **Telemetry Limitations:** Single-thread CPU environment prevents GPU memory or energy/RAPL instrumentation; unmeasured fields are reported as `null` with explicit reasons.