# The Illusion of Low-Hanging Fruit in Edge AI: Why Caching Invariant Decisions on Wearables is Harder Than It Looks

*An End-to-End Investigation into Event-Conditioned Certified Decision Reuse (ECDR), From Biomechanical Phase Shifts to Systems Scaling.*

**Author:** Nguyễn Tâm Đức  
**Academic Advisor:** Trần Thế Vũ  
**Coursework:** Introduction to Artificial Intelligence  
**Date:** November 2026  
**Artifacts & Code:** [GitHub Repository](https://github.com/tducn110/ecdr-sensor-inference) $\cdot$ [IEEE Technical Monograph](04_SYNTHESIS/week5/reports/ECDR_Week5_Comprehensive_Research_Paper_EN.tex) $\cdot$ [Compiled PDF Report](ECDR_Week5_FINAL_EN.pdf)

---

## The $50\%$ Redundancy Paradox

Imagine your smartwatch tracking your daily activities. Every 1.28 seconds, an onboard Inertial Measurement Unit (IMU) collects a window of acceleration and angular velocity readings spanning the last 2.56 seconds (128 samples at 50 Hz). 

To ensure continuous, responsive tracking, standard wearable protocols slide this window forward by 64 samples (1.28 seconds). This means that **exactly $50\%$ of the raw data points in the newly arriving window are mathematically identical to the previous window**.

Under standard deployment runtimes, what does the onboard microprocessor do? **It evaluates a complete deep neural network forward pass from scratch, for every single incoming window.**

```
Window t-1: [========== Past 64 Samples ==========][========== Shared 64 Samples ==========]
                                                    \                                      /
                                                     \            50% OVERLAP             /
                                                      \                                  /
Window t:                                           [========== Shared 64 Samples ==========][========== New 64 Samples ==========]
```

On battery-constrained edge microcontrollers, computing millions of multiply-accumulate (MAC) operations on inputs that are half-redundant is an egregious waste of thermal and battery budgets.

When I began this research project for my *Introduction to Artificial Intelligence* course, the solution seemed tantalizingly obvious: **Why not just cache the decision?** If the incoming window is largely the same, can't we simply reuse the previous classification and put the CPU to sleep?

It turns out that this seemingly straightforward optimization unravels into a fascinating labyrinth of deep learning theory, formal verification, non-linear biomechanics, and CPU cache micro-architectures. Here is the full story of what we built, where our assumptions were shattered, and where certified reuse actually works.

---

## 1. The Heuristic Trap: Why "Close Enough" Can Be Fatal

In real-world machine learning systems, engineers frequently rely on heuristic semantic caching: if an incoming input $x_t$ is "close" to a cached anchor $x_0$ according to an uncalibrated distance metric (like cosine similarity or Euclidean distance), the system reuses the cached prediction.

In an introductory AI course, this sounds like a clever hack. In safety-critical wearable applications—such as elderly fall detection, post-stroke tremor analysis, or cardiac arrest alerts—**heuristic caching is an accident waiting to happen**.

Deep neural networks with piecewise-linear activations (like ReLU and MaxPool) partition high-dimensional space into an astronomical number of linear polyhedra. The decision boundary between *Walking* and *Falling*, or between *Sitting* and *Collapsing*, can run arbitrarily close to normal data points. 

An unverified heuristic distance threshold offers **zero guarantees**: an imperceptible perturbation can easily push the input across a decision boundary, causing the model to flip its decision silently while the cache blindly outputs the stale label.

---

## 2. Enter ECDR: Mathematics as a Shield

To eliminate heuristic guesswork, we formulated **Event-Conditioned Certified Decision Reuse (ECDR)**. Instead of asking whether $x_t$ is *similar* to $x_0$, ECDR asks a strict formal verification question:

> **Can we evaluate a mathematical certificate over a declared perturbation region $\mathcal{S}$ around an evaluated anchor $x_0$ such that, for any query $x_t \in \mathcal{S}$, the top-1 classification of the neural network is provably guaranteed to remain identical to $x_0$?**

### The 11-Margin Contrast Formulation
In a 12-class problem, a classifier chooses class $c$ if its logit $z_c(x)$ is strictly greater than all other 11 logits $z_j(x)$. We formulate verification directly over the vector of **pairwise contrast margins**:

$$m_{c,j}(x) \triangleq z_c(x) - z_j(x), \quad \forall j \neq c$$

If a formal verifier computes guaranteed lower bounds $\underline{m}_{c,j}(\mathcal{S})$ satisfying:

$$\min_{j \neq c} \underline{m}_{c,j}(\mathcal{S}) > 0$$

then for **any** query $x_t \in \mathcal{S}$, the top-1 decision is mathematically guaranteed to be invariant ($c^*(x_t) = c^*(x_0)$).

### The Four-Barrier Fail-Closed Gate
To prevent floating-point rounding errors from creating false positive certificates in $\mathbb{F}_{32}$ arithmetic, we enforce an outward safety margin $\tau_{\text{fp}} = 10^{-5}$ and route every query through a four-barrier arbitration pipeline:

```
Incoming Query (x_t)
       │
       ▼
┌──────────────┐   Fail
│  Barrier 1   │ ────────> [ Fail-Closed Fallback ]
│Identity Match│                │
└──────────────┘                │ Full Forward Pass
       │ Pass                   │ Establish New Anchor
       ▼                        ▼
┌──────────────┐   Fail         │
│  Barrier 2   │ ───────────────┤
│ Domain Check │                │
└──────────────┘                │
       │ Pass                   │
       ▼                        │
┌──────────────┐   Fail         │
│  Barrier 3   │ ───────────────┤
│Soundness Pass│                │
└──────────────┘                │
       │ Pass                   │
       ▼                        │
┌──────────────┐   Fail         │
│  Barrier 4   │ ───────────────┘
│Margin > tau  │
└──────────────┘
       │ Pass
       ▼
[ CERTIFIED REUSE ] ──> Skip Backbone Forward Pass!
```

If any barrier fails, the runtime safely falls back to evaluating the full neural backbone. Across our entire test stream of **5,053 queries**, this fail-closed architecture achieved an exact **$100\%$ decision agreement rate ($D_{\text{all}} = 0$)**, completely eliminating false reuses.

---

## 3. The Engineering Confession: Why SmallCNN1D?

Selecting the reference neural network was one of the most educational chapters of this project.

### The Inductive Bias of 1D Convolutions
We chose 1D Temporal Convolutions (Conv1D) over fully-connected networks because human physical movement exhibits two fundamental physical properties:
1. **Local Temporal Coherence:** Muscle activations and physical impacts last between $0.10\text{ s}$ and $0.40\text{ s}$. Local convolution kernels ($k=3, 5$) capture these temporal dynamics across multi-axial channels.
2. **Translational Equivariance:** A characteristic motion signature (such as a heel strike) can happen at sample index 10 or sample index 70. Weight sharing detects this primitive anywhere without redundant parameter learning.

### The Global Average Pooling (GAP) Disaster
Early in the project, our mathematical formulations relied on an idealized network (`ECDRConv1DNet`) ending in Global Average Pooling (GAP):

$$g_c = \frac{1}{T} \sum_{t=1}^T F_{c,t}$$

On paper, GAP is mathematically gorgeous: under periodic boundary conditions, shifting a sequence in time leaves the global average unchanged, giving closed-form algebraic bounds on logit drift.

**In reality, GAP failed catastrophically on the 12-class HAPT dataset.**

Why? Because averaging over time is strictly **permutation-invariant**: it treats a sequence played forward identically to a sequence played in reverse! 

In the HAPT dataset, half of the 12 classes are directional postural transitions:
- *Stand-to-Sit* vs. *Sit-to-Stand*
- *Sit-to-Lie* vs. *Lie-to-Sit*
- *Stand-to-Lie* vs. *Lie-to-Stand*

These opposite transitions have nearly identical mean energies and frequency spectra. They differ solely in the **temporal directionality** of the movement. A GAP-based model is completely blind to this distinction.

To recognize postural transitions, we had to adopt a position-sensitive architecture: `SmallCNN1D`, which preserves temporal ordering by flattening the final feature maps into a dense classification head (`Flatten(2048) -> Linear(128) -> Linear(12)`).

### Why Not RNNs or Transformers?
We deliberately avoided recurrent networks (LSTM/GRU) and Transformers for the primary baseline:
1. **Formal Verification Tractability:** Feedforward piecewise-linear networks (Conv1D + ReLU + MaxPool) map directly into linear relaxation frameworks (`auto_LiRPA`). In contrast, unrolling an LSTM across 128 time steps with non-linear $\tanh$ and sigmoid activations causes severe interval wrapping explosion, making verified bounds uselessly loose. Similarly, self-attention in Transformers involves non-linear Softmax operations that are computationally prohibitive to bound in real-time on edge devices.
2. **A Formidable Edge Baseline:** `SmallCNN1D` has only 283,692 parameters ($1.135\text{ MB}$), requires $1.17\text{ MMACs}$, and executes in just **$0.1404\text{ ms}$** on a single CPU core. It fits entirely inside the CPU's L2/L3 cache. A reuse mechanism that only outperforms bloated models is an academic toy; beating a $0.14\text{ ms}$ baseline is a real engineering test.

---

## 4. When Theory Hits Physical Reality: The Two Brutal Bottlenecks

Equipped with formal proofs, the four-barrier gate, and our frozen `SmallCNN1D` baseline, we executed a complete sequential stream replay across all 19 HAPT test recordings ($5,053$ sequential queries, $5,034$ adjacent pairs).

We evaluated 17 distinct configurations (M00 to M16) across 24 formal hypotheses. The results delivered two profound engineering wake-up calls.

### Bottleneck 1: The Biomechanical Phase-Shift Orthogonality ($R_{\text{cert}} = 0\%$)
Every single input-space regional verifier (Global Lipschitz, Interval Bound Propagation, CROWN linear relaxations, Wave-domain bounds, PCA subspace boxes) recorded **exactly zero reuse hits ($R_{\text{cert}} = 0\%$)** on the test stream.

Why did mathematical certification completely fail to admit even a single consecutive window?

We audited the physical displacement between $1,000$ consecutive streaming pairs:

| Metric | Measured Value |
|---|---|
| Median Streaming Sensor Displacement ($\|\Delta z\|_2$) | **$25.492$** |
| Model Margin $\min_{j \neq c} m_{c,j}(x_0)$ | $6.098$ |
| Maximum Certifiable Radius $r_{\max}^{(2)}$ (LiRPA / IBP) | **$0.0149$** |
| **Displacement Gap Ratio ($\|\Delta z\|_2 / r_{\max}$)** | **$1,562.7\times$** |

The root cause was not mathematical looseness—it was **human biomechanics**.

At $50\text{ Hz}$, advancing the window by $S = 64$ samples represents a time jump of $\Delta t = 1.28\text{ seconds}$. This duration is almost exactly equal to a human walking cadence ($1.0\text{ to }1.3\text{ seconds}$ per gait cycle).

Advancing the window shifts an oscillatory sensor wave by an entire period. In 768-dimensional input space, sliding a periodic waveform is **not small additive noise**: it is a **macroscopic phase shift** that rotates the input vector onto an orthogonal trajectory!

The physical displacement ($25.49$) was **$1,562\times$ larger** than the largest radius the neural network's margin could certify ($0.0149$). The inputs were simply nowhere near the certifiable neighborhood.

### Bottleneck 2: The CPU Memory Slicing Paradox ($0.1775\times$ Speedup)
If input certification failed, what about incremental feature computation (M03)?

At $S=64$, the right 64 samples of window $t-1$ match the left 64 samples of window $t$. The first three convolutions maintain shift-aligned receptive fields that produce exactly 12 mathematically identical feature columns.

We implemented M03 to cache these 12 columns, compute convolutions only on the newly arriving 64 samples, concatenate the feature maps, and evaluate the classification head.

The result? **M03 achieved $100\%$ bit-exact feature preservation across all $5,034$ adjacent test pairs.**

**Yet it was $5.63\times$ SLOWER than running the full baseline forward pass!**
- Monolithic Baseline (M00): **$0.3420\text{ ms}$** (2,923 queries/sec)
- Incremental Caching (M03): **$1.9273\text{ ms}$** (518 queries/sec)
- Net Speedup: **$-463.47\%$** (a $0.1775\times$ speed ratio)

How could doing fewer floating-point operations make code $5.6\times$ slower?
1. **CPU Cache Locality:** `SmallCNN1D` is tiny ($1.135\text{ MB}$). A monolithic forward pass executes in a tight C++ loop with SIMD vectorization, reading weights that reside permanently in the CPU's L2/L3 cache.
2. **Framework Memory Allocation Overhead:** At batch size 1, dynamically allocating buffers, indexing non-contiguous memory, and calling `torch.cat` in Python incurs microseconds of runtime dispatch overhead that dwarf the few microseconds saved in arithmetic.
3. **The Dense Head Parameter Concentration:** The classification head `Linear(2048, 128)` contains **$92.45\%$ of all network parameters** ($262,272$ weights). Because it is fully connected, its outputs depend on all 2,048 feature entries. Reusing convolutional features only bypassed $7.55\%$ of the model's parameters while incurring huge tensor stitching penalties!

---

## 5. The Breakthroughs: Where Certified Reuse Actually Wins

Is certified decision reuse dead? Far from it. Once we understood the physical and architectural root causes, we tested two targeted scenarios that revealed where ECDR thrives.

### Breakthrough 1: High-Density Sensor Streams ($S=1$ Stride)
The phase-shift bottleneck was a direct consequence of the coarse $1.28\text{ s}$ stride ($S=64$). In high-density applications—such as continuous robotic reflexes, real-time tremor mitigation, or high-frequency telemetry running at 50 Hz—windows advance with a stride of $S=1$ ($20\text{ ms}$).

At $S=1$, adjacent windows differ by only a single new sample. The sensor displacement collapses:

$$\|\Delta z\|_{S=1} \ll r_{\max} \approx 0.0149$$

Under this high-density operating regime:
- Certified decision reuse surged from $0\%$ to **$87.31\%$ of all queries**!
- Decision agreement remained strictly **$100\%$ ($D_{\text{all}} = 0$)**.
- In high-frequency edge streaming, ECDR mathematically eliminates **$87.31\%$ of all forward inference passes** without risking safety.

### Breakthrough 2: The Model Capacity Scaling Crossover
Why did verification checks fail to save time on `SmallCNN1D`? Because the model was *too fast* ($0.14\text{ ms}$). Running a verification check ($T_v \approx 1.33\text{ ms}$) before a $0.14\text{ ms}$ forward pass is economically irrational.

However, verification overhead ($T_v$) and incremental dispatch overhead ($T_{\text{inc}}$) are relatively fixed costs at batch size 1. As neural networks grow in depth and capacity, forward pass latency ($T_{\text{fwd}}$) scales dramatically:

| Architecture | Parameters | MACs | Mean Forward Latency | Break-Even Feasibility |
|---|---|---|---|---|
| `SmallCNN1D` (Baseline) | $283\text{K}$ | $1.17\text{M}$ | $0.1404\text{ ms}$ | Unprofitable ($0.105\times$) |
| `MediumCNN1D` | $2.23\text{M}$ | $8.50\text{M}$ | $0.3769\text{ ms}$ | Unprofitable ($0.282\times$) |
| `TemporalResNet1D` | $2.70\text{M}$ | $38.20\text{M}$ | $1.8677\text{ ms}$ | Marginal ($r_{\min} = 71.5\%$) |
| `TemporalTransformer1D` | $4.99\text{M}$ | $115.00\text{M}$ | $3.3823\text{ ms}$ | **Highly Profitable ($r_{\min} = 39.5\%$)** |

**The Theoretical Crossover Point:**
- Verifier caching breaks even when $T_{\text{fwd}} \ge 1.3346\text{ ms}$, which occurs at **$27.58\text{ MMACs}$** ($\approx 2.53\text{M}$ parameters).
- On edge Transformers (`TemporalTransformer1D`, 115 MMACs), lazy on-demand verification achieves **$2.53\times\text{--}6.31\times$ net speedup** whenever certified reuse exceeds $39.5\%$.

---

## 6. Four Lessons Learned by a Student AI Researcher

Completing this five-week investigation for *Introduction to Artificial Intelligence* reshaped my understanding of computer science and machine learning. Here are the four takeaways that no textbook could have taught me:

### 1. Never Trust Blind Heuristics in AI Safety
In coursework, thresholding cosine distance seems like an elegant shortcut. In real-world non-linear models, heuristic caching is fundamentally untrustworthy. If a system controls autonomous driving, health monitoring, or industrial safety, formal verification—not empirical similarity—is the only defensible baseline.

### 2. Negative Results are the True Currency of Science
When all our certified verifiers returned $R = 0\%$ at $S=64$, it was tempting to tweak hyperparameters or cherry-pick synthetic data. Resisting that urge and investigating the true root cause led us to discover the biomechanical phase-shift phenomenon and the $1,562\times$ gap. Transparent, verifiable negative results advance engineering far more than inflated positive claims.

### 3. FLOPs Do Not Equal Wall-Clock Latency
Algorithmic theory often counts multiplications and additions as a proxy for speed. Our M03 experiment proved that memory layout, cache hierarchy, tensor stitching, and runtime orchestration dominate execution time at small scales. Real AI engineering requires profiling real hardware.

### 4. Data Engineering Beats Model Tuning Every Single Time
The biggest performance breakthrough of our project—resolving Gate G2 and lifting our 12-class Macro-F1 from $0.7567$ to $0.8537$—did not come from designing a more complex deep neural network. It came from opening raw sensor files, identifying that a rigid stride-64 grid was slicing across the middle of human postural transitions (Grid-Phase Misalignment), and developing a Span-Anchored Extraction policy. Clean, aligned data is the foundation of all artificial intelligence.

---

## 7. Project Artifacts and Reproduction

All experimental records, model checkpoints, raw signal logs, and verification test suites are preserved under cryptographic hash seals:
- **Repository:** [https://github.com/tducn110/ecdr-sensor-inference](https://github.com/tducn110/ecdr-sensor-inference)
- **Primary LaTeX Monograph:** `04_SYNTHESIS/week5/reports/ECDR_Week5_Comprehensive_Research_Paper_EN.tex`
- **Compiled PDF Report:** `ECDR_Week5_FINAL_EN.pdf`
- **Unit Test Suite:** 376 regression tests passing (`pytest tests/`)
- **Protocol Seal:** SHA-256 `568067623fdc2b4c193ef1ef8e604fcaebafb72236a2829141be3f1e16fdf9c7`

If you are working on edge AI, certified robustness, or time-series inference, dive into the code and explore the trade-offs yourself!
