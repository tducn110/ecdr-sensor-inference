# The Illusion of Low-Hanging Fruit in Edge AI: Why Caching Invariant Decisions on Wearables is Harder Than It Looks

*An End-to-End Investigation into Event-Conditioned Certified Decision Reuse (ECDR), From Biomechanical Displacement Gaps to Systems Scaling Limits.*

**Author:** Nguyễn Tâm Đức  
**Academic Supervisor:** Trần Thế Vũ  
**Research Track:** Introduction to Artificial Intelligence  
**Artifacts & Code:** [GitHub Repository](https://github.com/tducn110/ecdr-sensor-inference) $\cdot$ [Final Research Report (LaTeX)](04_SYNTHESIS/week5/reports/ECDR_Final_Report_EN.tex) $\cdot$ [Compiled PDF](ECDR_Final_Report_EN.pdf)

---

## The $50\%$ Redundancy Paradox

Imagine your smartwatch tracking your daily activities. Every 1.28 seconds, an onboard Inertial Measurement Unit (IMU) collects a window of tri-axial acceleration and angular velocity readings spanning the last 2.56 seconds (128 samples at 50 Hz).

To provide responsive tracking, standard wearable protocols slide this window forward by 64 samples (1.28 seconds). This means that **exactly $50\%$ of the raw data points in the newly arriving window are identical to the previous window**.

Yet under standard inference runtimes, what does the microprocessor do? **It executes a complete deep neural network forward pass from scratch on every single window.**

```
Window t-1: [========== Past 64 Samples ==========][========== Shared 64 Samples ==========]
                                                    \                                      /
                                                     \            50% OVERLAP             /
                                                      \                                  /
Window t:                                           [========== Shared 64 Samples ==========][========== New 64 Samples ==========]
```

At first glance, the optimization looks trivial: **why not reuse the previous prediction whenever consecutive windows overlap heavily?**

When we pushed past intuition into formal verification, IEEE-754 floating-point numerics, raw IMU stream reconstruction, and CPU latency profiling across 17 inference methods, that seemingly obvious optimization revealed three non-trivial barriers separating theoretical promise from wall-clock edge acceleration.

---

## 1. The Overlap Fallacy: Why Similarity Heuristics Fail

Two sliding windows sharing half their samples do not guarantee:
1. Small coordinate-space distance ($L_2$ or $L_\infty$),
2. Identical ground-truth labels, or
3. Identical neural network predictions.

Consider a simple linear ramp $u(t) = a t$: shifted consecutive windows share 64 exact samples, yet comparing coordinate index $t$ against $t$ produces a uniform offset $a \cdot S$ across the entire tensor. Worse, in our reconstructed Human Activity and Postural Transitions (HAPT) test stream, consecutive windows `682` and `683` share the exact same ground-truth activity (*Stand-to-Lie*, class 11), yet the frozen neural classifier flips its prediction from index `10` to index `8`.

Unverified heuristic caching (like thresholding $\ell_2$ distance in `M04` without margin proofs) provides **zero invariance guarantees**: a tiny nudge near a ReLU polytope boundary can silently flip a critical medical or activity alert while the cache returns a stale label.

---

## 2. ECDR: Five Layers Between Real Math and Safe Runtime

To avoid heuristic guesswork, **Event-Conditioned Certified Decision Reuse (ECDR)** asks a strict question around a frozen reference classifier $f_\theta$ (`SmallCNN1D`, $K=12$ classes):

> **Given an already-evaluated anchor window $a$ with class $c = \arg\max_k f_k(a)$ and a declared input region $\mathcal{S}$, can a sound verifier prove that all 11 competing pairwise logit margins $g_{c,j}(x) = f_c(x) - f_j(x)$ remain strictly positive for every $x \in \mathcal{S}$?**

Implementing this soundly required separating five distinct assurance layers in [`ecdr/week4/`](ecdr/week4/):

1. **Real-Arithmetic Theorem:** Strict positivity of all 11 contrast margin lower bounds implies unique class dominance across $\mathcal{S}$. Direct-margin subtraction ($W_c - W_j$ fused into the output head) avoids independent logit extrema cancellation loss—boosting validation margin estimates by $+0.5623$ logit units at $\epsilon = 10^{-5}$.
2. **Verifier Graph Soundness:** Implementing actual-graph Interval Bound Propagation (IBP), 50-step power-iteration $\ell_2$ spectral Lipschitz bounds ($L_{\text{backbone}} = 423.72$), and an `auto_LiRPA` 2D-surrogate `CROWN` adapter (`SingletonHeightCNN`, built because native 1D `MaxPool1d` failed conversion in `auto_LiRPA 0.8.0`).
3. **The IEEE-754 Binary32 Numerical Bridge ([`numerics.py`](ecdr/week4/numerics.py)):** Real proofs fail silently on single-precision CPUs—for example, $(2^{24}+1)-2^{24} = 1 > 0.5$ over $\mathbb{R}$, but evaluates to $0 < 0.5$ in `binary32` round-to-nearest ties-to-even! We enclosed all inner-product reductions ($n \in \{30, 96, 128, 192, 2048\}$) using Higham's backward error factor $\gamma_n = \frac{nu}{1-nu}$ and outward directed rounding (`nextafter`), validated against exact rational arithmetic (`fractions.Fraction`).
4. **Conservative Membership ([`domains.py`](ecdr/week4/domains.py)):** Checking axis-aligned diagonal boxes ($768$-D), fixed-overlap wave domains ($384$ fixed prefix + $384$ free tail), and $K=8$ PCA affine tail zones ($\text{tail} = c + B \diag(a) u + \diag(e) v$, fit on $9,944$ training pairs), requiring $\Theta(m)$ exact coordinate accesses.
5. **Cryptographic Lifecycle & Fail-Closed Gate ([`certificates.py`](ecdr/week4/certificates.py), [`runtime.py`](ecdr/week4/runtime.py)):** Every certificate binds checkpoint SHA-256, preprocessing hash, recording ID, `anchor_state_hash`, and step horizon without re-centering on hits. If *any* invariant fails, the runtime defaults immediately to the frozen backbone—guaranteeing **$100\%$ decision agreement ($0/5,053$ errors)**.

---

## 3. Why `SmallCNN1D`? Architecture Design Trade-offs

Selecting the reference model (`SmallCNN1D`, $283,692$ parameters, $1,172,992$ Conv/Linear MACs, Seed 43 checkpoint SHA-256 `8ffc5dd8...`) reflected clear verification and task requirements:
- **1D Convolutions & Verification Tractability:** Conv1D + BatchNorm + ReLU + MaxPool forms a shallow piecewise-linear feedforward graph suitable for IBP and linear relaxations—unlike 128-step unrolled RNNs (whose $\tanh$/sigmoid gates trigger severe interval wrapping explosion) or Transformers (whose $QK^\top$ Softmax attention lacks real-time CPU bounding at batch size 1).
- **Position-Sensitive `Flatten(2048)` Head vs.\ GAP:** Half of the 12 HAPT classes are **directional postural transitions** (*Stand-to-Sit* vs.\ *Sit-to-Stand*, *Sit-to-Lie* vs.\ *Lie-to-Sit*, *Stand-to-Lie* vs.\ *Lie-to-Stand*). Because a terminal Global Average Pooling (GAP) layer averages uniformly across time, it discards feature ordering at the final aggregation step. Retaining `Flatten(2048) -> Linear(128) -> Linear(12)` preserves position sensitivity (note: while motivated directly by transition directionality, we did not run a matched 12-class GAP ablation on the same split/seeds, keeping it framed as an architectural design rationale).
- **A Fast Baseline:** Fitting entirely in CPU L2/L3 cache (`1,136,072` bytes), `SmallCNN1D` runs at **$0.3420\text{ ms}$** mean query latency ($2,923.7\text{ QPS}$) end-to-end—an uncompromising baseline.

---

## 4. Two Empirical Reality Checks Across 5,053 Test Queries

When we evaluated 15 executable configurations out of 17 registered methods (`M00`--`M16`) on $5,053$ chronological test windows across 19 recordings ($15,159$ paired timing observations per method), **zero methods achieved useful streaming acceleration (`H24 = NOT_ESTABLISHED`)**:

### Reality Check 1: Displacement Outstrips Certifiable Radii by $>1,500\times$
All regional verifiers (`M05`--`M13`) recorded **$0$ certified hits ($0\%$ reuse)**, raising latency to $0.4479\text{--}7.1278\text{ ms}$. Pre-test validation across 20 proposals found 0 positive members, closing the formal gate (`formal_reuse_enabled = false`).

Why? Auditing $1,000$ consecutive stride-64 window pairs showed:

| Metric | Median Measured Value |
|---|---|
| Normalized Sensor Displacement ($\|\Delta z\|_2$) | **$25.492$** ($p10=1.148,\; p90=45.675$) |
| Normalized Max Coordinate Shift ($\|\Delta z\|_\infty$) | **$3.845$** (mean $3.966$) |
| Model Separation Margin $\min_{j \neq c} g_{c,j}(a)$ | $6.098$ |
| Max Certified $\ell_2$ Radius $r_{\max}^{(2)}$ (Lipschitz $L=423.72$) | **$0.0149$** (max $0.0401$) |
| **Displacement Gap Ratio ($\|\Delta z\|_2 / r_{\max}^{(2)}$)** | **$1,562.7\times$** |

Sliding by $1.28\text{ s}$ shifts oscillatory human motion signals across an entire gait cycle. In normalized feature space, that phase translation moves inputs over $1,500\times$ farther than the network's Lipschitz margin ball can prove invariant.

### Reality Check 2: The Incremental CPU Slicing Paradox (`M03`)
What about exact intermediate computation reuse (`M03`), inspired by shift-invariant streaming CNNs?

Because padded convolutions satisfy an interior receptive-field shift identity ($j=2\dots 13$ across `Conv3`), `ActualK12IncrementalEngine` reuses 12 `Conv3` columns while enforcing slice width $L_{\text{out}} \ge 2$ (avoiding an $8.94 \times 10^{-8}$ BLAS `sgemv` reduction mismatch). Across all $5,034$ adjacent pairs ($3,866,112$ scalars), `M03` achieved **$0.0$ logit error ($100\%$ decision equivalence)** and reduced total MACs by **$32.15\%$** ($795,904$ vs.\ $1,172,992$; Conv MACs $-41.47\%$).

**Yet `M03` was $5.64\times$ SLOWER than the full baseline:**
- `M00` Full Core: **$0.3420\text{ ms}$** ($2,923.7\text{ QPS}$)
- `M03` Incremental Reuse (all queries): **$1.9273\text{ ms}$** ($+463.47\%$; reuse subset $+463.86\%$)

Why? First, the dense `Linear(2048, 128)` head holds **$92.45\%$ of model parameters** and remains position-sensitive, so it must be evaluated on every query. Second, Python tensor slicing, indexing, and allocation at batch size 1 cost nearly $2\text{ ms}$ compared to a single cache-resident C++ MKL-DNN kernel. Even an in-place pre-allocated `Conv1` ring buffer only shaved Python overhead by $4.06\%$ ($0.03421 \to 0.03282\text{ ms}$), remaining $2.66\times$ slower than direct PyTorch C++ `conv1d` ($0.01233\text{ ms}$).

---

## 5. Stride Sensitivity & Model Scaling: Honest Boundaries

We ran two diagnostic studies to map whether finer strides or heavier models change the picture ([`stride_audit.py`](04_SYNTHESIS/week5/evidence/stride_audit.py), [`post_w4_feasibility_study.py`](scripts/post_w4_feasibility_study.py)):

1. **Shorter Strides ($S=1$) Boost Empirical Agreement, Not IBP Certifiability:**
   Testing strides $S \in \{1, 4, 16, 32, 64\}$ across all 19 test recordings (`322,856` adjacent pairs at $S=1$), neighboring top-1 agreement climbed from $79.10\%$ ($S=64$) to **$97.97\%$ ($S=1$)** and median $\ell_2$ distance dropped to $6.342$. However, **zero** pairs fell inside $\ell_\infty$ boxes of radius $10^{-6}$, $10^{-4}$, or $10^{-2}$ at any stride! At radius $0.1$, $49,677$ stride-1 pairs entered the box ($15.39\%$), **$7$ pairs already disagreed** on top-1 class, and direct `float64` IBP yielded **`0 / 49,677` positive 11-margin estimates** (largest minimum margin: $-1544.86$). High temporal label agreement does not imply IBP box certifiability.

2. **Model Scaling Crossover Sensitivity (`~27.6 MMACs`):**
   Forward-only single-thread CPU benchmarks on scaled synthetic architectures showed `MediumCNN1D` ($0.3769\text{ ms}$) and `TemporalResNet1D` ($1.8677\text{ ms}$, $r_{\min} = 71.5\%$) remain largely unprofitable under Week 4 verifier overhead. On a $4.99\text{M}$-parameter `TemporalTransformer1D` ($115\text{ MMACs}$, $3.3823\text{ ms}$ forward), substituting the CNN verifier's $1.3346\text{ ms}$ check cost gives a modeled break-even crossover near **$27.6\text{ MMACs}$** and a **$2.53\times$ theoretical ceiling** at $100\%$ reuse ($r_{\min} = 39.5\%$). Because native Transformer attention verification was not measured and real $S=64$ streams yielded $R=0\%$, actual streaming speedup on Transformers remains strictly **`NOT_ESTABLISHED`**.

---

## 6. Key Engineering Takeaways

1. **Formal Soundness $\neq$ Coverage $\neq$ Speedup:** All three obligations must be verified on the same workload under the same numerical bridge.
2. **MAC Reductions Do Not Guarantee Latency Wins:** On micro-models fitting in L2/L3 cache, framework memory orchestration dominates FLOP savings; native fused C/CMSIS-NN kernels are essential.
3. **Data Alignment Matters as Much as Model Tuning:** Resolving canonical grid-phase misalignment in a separate transition-support branch (`Span-Anchored M=9 + Class-Balanced Focal Loss`) cleared all 18 pure / 10 canonical support blockers and raised 3-seed Full-G2 Macro-F1 to **$0.8537 \pm 0.0020$** (Transition F1 $0.7917$).
4. **Scientific Honesty Builds Better Systems:** Documenting zero certified stream hits, invalid `LipCache` proxy scope comparisons, and `float32` rounding edge cases reveals the exact mathematical gaps future research must close.

---

## 7. Reproduction and Verification

- **Unit & Regression Suite:** `.venv/bin/python tests/run_all_tests.py` (**376 / 376 tests PASS** across 41 test modules).
- **Protected Integrity Audit:** `.venv/bin/python scripts/week4_audit.py check` (214 historical files `ALL_CLEAN`, Master Seal `56806762...`).
- **Full Technical Reference:** Read [ECDR_Final_Report_EN.tex](04_SYNTHESIS/week5/reports/ECDR_Final_Report_EN.tex) or [ECDR_Final_Report_EN.pdf](ECDR_Final_Report_EN.pdf).
