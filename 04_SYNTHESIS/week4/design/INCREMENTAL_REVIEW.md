# W4-021: Independent Incremental Executable-Equivalence Gate

**Task ID:** `W4-021`  
**Milestone:** M3 — Actual $K=12$ Incremental Feature Reuse  
**Dependencies:** `W4-005` (`VERIFIED`), `W4-020` (`VERIFIED`)  
**Hypotheses:** H13, H14, H21, H22  
**Gate Verdict:** `STRICT_ELIGIBLE_FOR_PROFILE` (bound to `linux-x86_64-py311-numpy246-torch2141cpu-avx512`, `cache_schedule_version = w4.incremental.schedule.v2`, `min_conv_slice_out_len >= 2`)

---

## 1. Real Feature Proof vs. Partial-Kernel & Reduction-Order Arithmetic (`W4-021-C01`)

### 1.1 Real-Algebra Receptive Field Identity
Over $\mathbb{R}$, for any consecutive 50%-overlapping sensor windows $X^{\text{new}}, X^{\text{old}} \in \mathbb{R}^{6 \times 128}$ satisfying $X^{\text{new}}[:, t] = X^{\text{old}}[:, t + 64]$ for $t \in [0, 63]$:
1. Coordinatewise normalization uses frozen channel-wise parameters $(\mu_c, \sigma_c, \epsilon = 10^{-8})$, so $X_{\text{norm}}^{\text{new}}[:, 0:64] = X_{\text{norm}}^{\text{old}}[:, 64:128]$.
2. Each `BatchNorm1d` layer is frozen in `eval()` mode (`training == False`), acting as a channel-wise affine map $y_c[j] = \gamma_c \frac{x_c[j] - \mu_{\text{run},c}}{\sqrt{\sigma_{\text{run},c}^2 + 10^{-5}}} + \beta_c$ with zero spatial coupling across columns $j$.
3. `MaxPool1d(kernel_size=2, stride=2)` is translation-covariant under any shift divisible by its cumulative stride $s \in \{2, 4\}$; since $64 = 32 \times 2 = 16 \times 4$, pooling window boundaries align identically in $X^{\text{new}}$ and $X^{\text{old}}$.
4. Therefore, every interior column whose receptive field $\text{RF}(j)$ lies strictly inside $[0, 63]$ without touching left zero-padding ($t < 0$) or new right samples ($t \ge 64$) is algebraically identical to the shifted column in the previous window:
   - **Conv1 (`h1`)**: $j \in [2, 61] \leftarrow j_{\text{old}} + 64$ (60/128 columns reused; $\text{RF}_1(j) = [j-2, j+2] \subseteq [0, 63]$).
   - **Pool1 (`p1`)**: $j \in [1, 30] \leftarrow j_{\text{old}} + 32$ (30/64 columns reused; $\text{RF}_{P1}(j) = [2j-2, 2j+3] \subseteq [0, 63]$).
   - **Conv2 (`h2`)**: $j \in [2, 29] \leftarrow j_{\text{old}} + 32$ (28/64 columns reused; $\text{RF}_2(j) = [2j-4, 2j+5] \subseteq [0, 63]$).
   - **Pool2 (`p2`)**: $j \in [1, 14] \leftarrow j_{\text{old}} + 16$ (14/32 columns reused; $\text{RF}_{P2}(j) = [4j-4, 4j+7] \subseteq [0, 63]$).
   - **Conv3 (`h3`)**: $j \in [2, 13] \leftarrow j_{\text{old}} + 16$ (12/32 columns reused; $\text{RF}_3(j) = [4j-8, 4j+11] \subseteq [0, 63]$).

### 1.2 IEEE-754 Float32 Kernel Dispatch & Inner-Product Reduction Order
In IEEE-754 `float32`, real-algebraic equality does not automatically guarantee bitwise equality when a full-tensor convolution is replaced by sliced partial convolutions, because BLAS backends may dispatch different SIMD kernels depending on output spatial length $L_{\text{out}}$.

Inspection of the locked PyTorch 2.14.1+cpu / Intel oneAPI MKL 2024.2 execution path (`torch_threads=1`, AVX-512) shows:
- `F.conv1d(x, weight, bias, stride=1, padding=p)` lowers to `aten::thnn_conv2d -> aten::_slow_conv2d_forward`.
- For batch size $B = 1$, `_slow_conv2d_forward` materializes the unfolded `im2col` matrix $U \in \mathbb{R}^{K \times L_{\text{out}}}$ where $K = C_{\text{in}} \cdot k_{\text{width}} \in \{30, 96, 192\}$, initializes the output buffer with `bias`, and invokes row-major BLAS `sgemm` (`C := W * U + C`, mapped to column-major `cblas_sgemm('N', 'N', m=L_out, n=C_out, k=K, ...)`).
- **Critical Dispatch Boundary ($m = L_{\text{out}} \ge 2$ vs. $m = L_{\text{out}} = 1$)**:
  - **When $L_{\text{out}} \ge 2$**: MKL `cblas_sgemm` vectorizes across the spatial output dimension $m = L_{\text{out}}$ (and output channels $n = C_{\text{out}}$) while accumulating the reduction loop $r = 0, 1, \dots, K-1$ **in the exact same sequential order** for every output column $j$, regardless of whether $L_{\text{out}} = 2, 18, 32, 34, 64, 66,$ or $128$. Furthermore, for every recomputed slice in `ActualK12IncrementalEngine._forward_stages_partial`:
    - Stage 1 Left: $L_{\text{out}} = 2$ ($j \in \{0, 1\}$ from $\text{pad}_{(2,0)}(X_{\text{norm}}[:, 0:4])$), Right: $L_{\text{out}} = 66$ ($j \in [62, 127]$ from $\text{pad}_{(0,2)}(X_{\text{norm}}[:, 60:128])$).
    - Stage 2 Left: $L_{\text{out}} = 2$ ($j \in \{0, 1\}$ from $\text{pad}_{(1,0)}(P_1[:, 0:3])$), Right: $L_{\text{out}} = 34$ ($j \in [30, 63]$ from $\text{pad}_{(0,1)}(P_1[:, 29:64])$).
    - Stage 3 Left: $L_{\text{out}} = 2$ ($j \in \{0, 1\}$ from $\text{pad}_{(1,0)}(P_2[:, 0:3])$), Right: $L_{\text{out}} = 18$ ($j \in [14, 31]$ from $\text{pad}_{(0,1)}(P_2[:, 13:32])$).
    Because every slice satisfies $L_{\text{out}} \ge 2$ and every `im2col` column $U[:, j]$ contains the exact same `float32` bit pattern as in the full-window call, the `float32` accumulation tree for each output scalar is identical to the full-window graph (`0.0` ULP difference).
  - **Counterexample Witness when $L_{\text{out}} = 1$**: If a single output column is computed in isolation ($L_{\text{out}} = 1$, e.g., computing $j = 0$ alone from $\text{pad}_{(2,0)}(X_{\text{norm}}[:, 0:3])$), MKL dispatches the matrix-vector kernel `sgemv`, which performs horizontal AVX-512 SIMD reduction along the $K$-axis ($K=30$), changing the floating-point addition tree. On the very first validation window (`global_row_id = 15031`, Conv1 column $j=0$), $L_{\text{out}} = 1$ produces a non-zero discrepancy of **`8.9406967e-08`** (`1.5` ULP), whereas $L_{\text{out}} = 2$ produces **`0.0`** error!

---

## 2. Separation of Evidence Levels (`W4-021-C02`)

In strict accordance with `W4-021-C02`, we maintain four distinct, non-interchangeable evidence levels and forbid promoting a weaker level into a stronger one:

| Evidence Level | Scope & Mathematical Meaning | Status | Empirical / Analytical Evidence |
| :--- | :--- | :--- | :--- |
| **L1: Universal Real-Algebra Identity** | Exact equality over $\mathbb{R}$ for all $X^{\text{new}}[:, 0:64] = X^{\text{old}}[:, 64:128]$ under frozen eval BN and stride-aligned MaxPool | `DERIVED` | Exact interval proof in `INCREMENTAL_SCHEDULE.md` §1–§4; impulse & ramp tests (`14/14 PASS`). |
| **L2: Universal Cross-Backend Float32 Proof** | Bitwise equality across arbitrary BLAS/GPU/compiler backends (cuDNN Winograd/FFT, OpenBLAS, Apple Accelerate, multi-threaded MKL, or $L_{\text{out}}=1$) | `NOT_ESTABLISHED` | **Falsified in general** by our $L_{\text{out}}=1$ `sgemv` counterexample (`max_abs_err = 8.94e-8` on `row_id=15031`, `3.66e-4` on synthetic cancellation). Never claimed universally. |
| **L3: Profile-Locked Bitwise Equality** | Bitwise `0.0` ULP equality across all 5 stages (`h1, p1, h2, p2, h3`) and `logits` on locked profile `linux-x86_64-py311-numpy246-torch2141cpu-avx512` with `min_conv_slice_out_len >= 2` | `VERIFIED_FINITE_AND_KERNEL_LOCKED` | `0.0` max abs error across all **2,414** Val incremental steps, **5,034** Test incremental steps, and **500** high-dynamic-range cancellation stress trials. |
| **L4: Decision Agreement** | Top-1 class prediction equality ($\arg\max z^{\text{inc}} = \arg\max z^{\text{full}}$) on evaluated streams | `VERIFIED_FINITE` | **0** winner disagreements across 2,422 Val windows and 5,053 Test windows (`100.0%` agreement). |

---

## 3. Diagnostic Probes & Exact Witnesses (`W4-021-C03`)

Four targeted probes were executed and embedded into `tests/test_week4_incremental_gate.py`:

1. **Near-Tie Logits Probe**:
   - **Real Dataset Near-Tie Witness**: Across all 5,034 test incremental steps, the tightest top-1 vs. top-2 logit margin occurs at **`global_row_id = 13358`** (`recording_id = "48"`, `step_idx = 236`), where class 3 (`logit = 6.2438531`) beats the runner-up (`logit = 6.2437396`) by only **$m_1 = 1.1348724 \times 10^{-4}$**. Across all 2,414 validation incremental steps, the tightest margin occurs at **`global_row_id = 17355`** (`recording_id = "61"`, `step_idx = 199`), with **$m_1 = 9.1688633 \times 10^{-3}$** (`2.2399912` vs. `2.2308223`). On both near-tie windows, `ActualK12IncrementalEngine` achieves **`0.0` logit error** and **zero winner flip**.
   - **Synthetic Sub-ULP Near-Tie Witness**: Constructing a two-step stream and calibrating the final bias on a disposable model copy so that $z_0 - z_1 = 4.7683716 \times 10^{-7}$ ($< 10^{-6}$) confirms that incremental evaluation preserves exact bitwise logits (`max_abs_err = 0.0`) and exact top-1 winner agreement where any $10^{-6}$ perturbation would flip the classification.
2. **Padding & Overlap Boundary Probe**:
   - **Boundary Necessity Witness**: Attempting to reuse unverified left-boundary columns $j \in \{0, 1\}$ or right-overlap boundary columns $j \in \{14, 15\}$ at Conv3 (`h3`) on validation pair `step 0 -> step 1` (`row_id 15031 -> 15032`) yields large errors: **`max_abs_err(j=0,1) = 1.3712645`** and **`max_abs_err(j=14,15) = 2.0928459`**, whereas the verified interior $j \in [2, 13]$ yields **`0.0`**.
   - **1-ULP Overlap Perturbation Witness**: Perturbing a single overlap scalar $X^{\text{new}}[0, 31]$ by `np.nextafter(..., +inf)` (1 ULP) is caught immediately by the raw/normalized overlap guard, forcing a clean fallback to `mode = "full"` with `fallback_reason = "raw_overlap_mismatch"`.
3. **Single-Column (`L_out = 1`) vs. Multi-Column (`L_out >= 2`) Kernel Probe**:
   - On `global_row_id = 15031` (first validation window), computing Conv1 column $j=0$ with slice length $L_{\text{out}} = 1$ triggers MKL `sgemv` and produces **`max_abs_err = 8.940696716308594e-08`**, whereas the W4-019 schedule slice length $L_{\text{out}} = 2$ ($j \in \{0, 1\}$) invokes `sgemm` and produces **`max_abs_err = 0.0`**.
4. **BatchNorm Setting Probe (`eval()` vs. `train()`)**:
   - **Numerical Witness**: On a disposable copy of `conv[0..1]` placed in `train()` mode (`bn1.train()`), evaluating the right partial slice ($L_{\text{out}} = 66$) vs. the full window ($L_{\text{out}} = 128$) on `global_row_id = 15031` causes a massive activation divergence of **`max_abs_err = 2.2599254`**, because batch mean/variance over 66 columns differs from 128 columns.
   - **Fail-Closed Engine Guard**: Calling `ActualK12IncrementalEngine` when any `BatchNorm1d` module has `training == True` or missing running buffers immediately raises `ValueError("BatchNorm1d ... must be in eval mode for incremental reuse")`.

---

## 4. Capability Verdict & Identity Binding (`W4-021-C04`, `W4-021-C05`, `W4-021-C06`)

### 4.1 Locked Identity Binding
`ActualK12IncrementalEngine.equivalence_capability()` returns `STRICT_ELIGIBLE_FOR_PROFILE` **if and only if** all of the following cryptographic and runtime invariants hold simultaneously:
- `environment_profile_id == "linux-x86_64-py311-numpy246-torch2141cpu-avx512"`
- `cache_schedule_version == "w4.incremental.schedule.v2"`
- `min_conv_slice_out_len >= 2`
- `incremental_source_sha256 == "be1a25cd408165ce346171fc0ea05e9b17bf03d625a10dd0c21afde3fc7fbb3f"` (`ecdr/week4/incremental.py`)
- `schedule_doc_sha256 == "b5cd3a6428f55beec997a13ee1c6a83db5522782cb49b8d5fa9e70e214e82983"` (`04_SYNTHESIS/week4/design/INCREMENTAL_SCHEDULE.md`)
- `checkpoint_sha256 == "8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1"`
- `preprocessing_id == "aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e"`
- `W4-021` is marked `VERIFIED` in `04_SYNTHESIS/week4/tasks.json` by an independent reviewer (`reviewer_id != assignee`).

### 4.2 Anti-Spoofing & Automatic Invalidation (`W4-021-C05`)
- `IncrementalEquivalenceCapability` is a frozen dataclass protected by a private module-scoped object token `_CAPABILITY_GUARD_TOKEN`. Constructing `IncrementalEquivalenceCapability(status=IncrementalEquivalenceStatus.STRICT_ELIGIBLE_FOR_PROFILE, ...)` directly from external code without the private token raises `ValueError`.
- Passing any modified profile string, `min_conv_slice_out_len < 2`, or a modified `incremental.py` / `INCREMENTAL_SCHEDULE.md` hash immediately invalidates strict eligibility (`status = INVALIDATED`).
- If `W4-021` review is pending or unverified in `tasks.json`, `equivalence_capability()` returns `EXPERIMENTAL_ONLY` (`is_strict_eligible == False`).

### 4.3 Strict Fallback & Experimental Measurement (`W4-021-C06`)
- Calling `engine.forward_strict(...)` when capability is `EXPERIMENTAL_ONLY` or `INVALIDATED` automatically executes `forward_full(...)` with `fallback_reason = "incremental_gate_not_strict:<status>"`.
- Calling `engine.forward_incremental(...)` directly remains available for experimental benchmarking (`Q_incr`), always tagging every result with `is_skipped_decision_reuse = False`.

---

## 5. Non-Conflation with Regional Decision Reuse (`W4-021-C07`)

1. **No Regional Certificate Generation**: Intermediate feature equality in `ActualK12IncrementalEngine` applies strictly to a single point trajectory $X^{\text{new}}$ sharing an exact 64-sample overlap with $X^{\text{old}}$ (`produces_regional_certificate == False`). It **never** produces an $\ell_\infty$ ball certificate $B_\infty(x_0, \epsilon)$, never skips the `Flatten(2048) -> Linear(128) -> ReLU -> Linear(12)` classifier head, and never increments regional certified reuse coverage (`is_skipped_decision_reuse == False`).
2. **Independent Review Separation**: `W4-021` requires verification by an independent reviewer (`reviewer_id != assignee`) in `04_SYNTHESIS/week4/tasks.json`.
