# W4-010: Independent Numerical Eligibility Review & Non-Bypass Capability Gate

**Task ID:** `W4-010`  
**Milestone:** M1 — Numerics & Outward Interval Arithmetic  
**Dependencies:** `W4-009` (`VERIFIED`)  
**Hypotheses:** H05, H22, H24  
**Gate Verdict:** `ELIGIBLE_FOR_PROFILE` (bound strictly to `linux-x86_64-py311-numpy246-torch2141cpu-avx512`)  
**Formal Component:** Numerical Soundness Component of Gate G3 (Numerical Eligibility)

---

## 1. Independent Reviewer Mandate & Method Verification (`W4-010-C01`)

In accordance with `ORCHESTRATOR.md` §6 and `W4-010-C01`, this review is conducted independently of the authors of `W4-006` through `W4-009`. The review verifies the mathematical derivations, error propagation formulas, source implementations, and empirical edge-case stress tests across the entire numerical stack.

### 1.1 Reviewed Dependency Artifacts
- **Mathematical Specification:** `04_SYNTHESIS/week4/design/NUMERICAL_CONTRACT.md` (`W4-005`, `VERIFIED`).
- **Directed Rounding Primitives:** `ecdr/week4/numerics.py` & `tests/test_week4_numerics.py` (`W4-006`, `VERIFIED`).
- **Normalization Enclosure:** `ecdr/week4/reference.py` & `tests/test_week4_membership.py` (`W4-007`, `VERIFIED`).
- **Operator Enclosures:** `ecdr/week4/numerics.py`, `tests/test_week4_operators.py`, `output/week4/week4_initial_20261103/operator_capabilities.json`, `operator_proof_review.md` (`W4-008`, `VERIFIED`).
- **End-to-End Float32 Bridge:** `04_SYNTHESIS/week4/design/FLOAT32_BRIDGE.md`, `tests/test_week4_numerical_bridge.py`, `output/week4/week4_initial_20261103/numerical_bridge.json`, `bridge_checks.json` (`W4-009`, `VERIFIED`).

### 1.2 Independent Rerun of Narrow Oracle & Edge-Case Probes
The independent review suite re-executes:
1. `ExactRationalOracle.exact_dot_product`: Fraction-arithmetic verification of Conv1d and Linear interval bounds (0 ULP enclosure violations).
2. Negative and zero gamma in `BatchNorm1d`: Correct reversal of interval bounds ($\alpha < 0 \implies \text{lo} \to \text{hi}, \text{hi} \to \text{lo}$) and collapse to constant $\beta$ when $\alpha = 0$.
3. Near-zero variance ($\sigma^2 \to 0$): Numerical stability maintained by frozen $\epsilon = 10^{-5}$.
4. Cancellation stress ($+10^6 - 10^6$): Higham backward error bound $\gamma_N \sum |w_i x_i|$ remains invariant to cancellation loss.
5. MaxPool1d ties ($x_1 = x_2$): Exact pairwise max preservation.
6. Zero-padding boundary coordinates: Treated as exact $[0.0, 0.0]$, introducing zero rounding error.

---

## 2. Categorization of Operators & Composition (`W4-010-C02`)

Every operator and compositional layer of `SmallCNN1D(num_classes=12)` is explicitly categorized into one of four epistemic classes:
- **`DERIVED`**: Mathematically proved with closed-form outward error bounds.
- **`EMPIRICAL`**: Validated by exhaustive / finite empirical sampling, but without formal closed-form proof.
- **`ASSUMED`**: Explicitly stated hardware / backend premises.
- **`MISSING`**: Unimplemented or unverified operations.

| Subsystem / Layer | Operator Kind | Epistemic Status | Governing Proof / Error Model | Sufficiency for Profile |
| :--- | :--- | :---: | :--- | :---: |
| **Input Conversion** | `float32` cast | `DERIVED` | Monotone non-decreasing under IEEE-754 round-to-nearest | SUFFICIENT |
| **Normalization** | Elementwise sub/div | `DERIVED` | Monotone composition (Theorem 1, `reference.py`); zero dilation | SUFFICIENT |
| **Conv1 (`conv.0`)** | `Conv1d(6->32, k=5, p=2)` | `DERIVED` | Higham bound $\gamma_{31} (|W||X| + |b|) + 31 \Delta_{\text{sub}}$ | SUFFICIENT |
| **BN1 (`conv.1`)** | `BatchNorm1d(32, eps=1e-5)` | `DERIVED` | Affine folding $\alpha = \gamma / \sqrt{\sigma^2+\epsilon}$; error bound $\gamma_4$ | SUFFICIENT |
| **ReLU1 (`conv.2`)** | `ReLU` | `DERIVED` | Exact non-expansive coordinatewise map; zero roundoff error | SUFFICIENT |
| **Pool1 (`conv.3`)** | `MaxPool1d(k=2, s=2, p=0)` | `DERIVED` | Exact coordinatewise pairwise max; zero roundoff error | SUFFICIENT |
| **Conv2 (`conv.4`)** | `Conv1d(32->64, k=3, p=1)`| `DERIVED` | Higham bound $\gamma_{97} (|W||X| + |b|) + 97 \Delta_{\text{sub}}$ | SUFFICIENT |
| **BN2 (`conv.5`)** | `BatchNorm1d(64, eps=1e-5)`| `DERIVED` | Affine folding $\alpha = \gamma / \sqrt{\sigma^2+\epsilon}$; error bound $\gamma_4$ | SUFFICIENT |
| **ReLU2 (`conv.6`)** | `ReLU` | `DERIVED` | Exact non-expansive coordinatewise map; zero roundoff error | SUFFICIENT |
| **Pool2 (`conv.7`)** | `MaxPool1d(k=2, s=2, p=0)`| `DERIVED` | Exact coordinatewise pairwise max; zero roundoff error | SUFFICIENT |
| **Conv3 (`conv.8`)** | `Conv1d(64->64, k=3, p=1)`| `DERIVED` | Higham bound $\gamma_{193} (|W||X| + |b|) + 193 \Delta_{\text{sub}}$ | SUFFICIENT |
| **BN3 (`conv.9`)** | `BatchNorm1d(64, eps=1e-5)`| `DERIVED` | Affine folding $\alpha = \gamma / \sqrt{\sigma^2+\epsilon}$; error bound $\gamma_4$ | SUFFICIENT |
| **ReLU3 (`conv.10`)**| `ReLU` | `DERIVED` | Exact non-expansive coordinatewise map; zero roundoff error | SUFFICIENT |
| **Flatten (`head.0`)**| Channel-major reshape | `DERIVED` | Exact array index reshape ($64 \times 32 \to 2048$); zero error | SUFFICIENT |
| **Linear1 (`head.1`)**| `Linear(2048->128)` | `DERIVED` | Higham bound $\gamma_{2049} (|W||X| + |b|) + 2049 \Delta_{\text{sub}}$ | SUFFICIENT |
| **Head ReLU (`head.2`)**| `ReLU` | `DERIVED` | Exact non-expansive coordinatewise map; zero roundoff error | SUFFICIENT |
| **Linear2 (`head.3`)**| `Linear(128->12)` | `DERIVED` | Higham bound $\gamma_{129} (|W||X| + |b|) + 129 \Delta_{\text{sub}}$ | SUFFICIENT |
| **Competitor Margins**| 11 differences $m_j$ | `DERIVED` | Separate logits or direct head difference with $\Delta_c + \Delta_j$ widening | SUFFICIENT |
| **Backend Reduction** | Intel MKL / AVX-512 | `ASSUMED` | Sequential or tree reduction bounded by worst-case $\gamma_N$ | SUFFICIENT |

**Conclusion on Categorization:** 100% of operators and compositions are `DERIVED` with explicit closed-form bounds. No operator relies on empirical tuning or heuristic safety factors.

---

## 3. Capability Decision & Cryptographic Binding (`W4-010-C03`)

### 3.1 Verdict: `ELIGIBLE_FOR_PROFILE`
The numerical method implemented in `ecdr/week4/numerics.py` is hereby certified as **`ELIGIBLE_FOR_PROFILE`** for the declared reference model and environment.

### 3.2 Cryptographic Identity Binding
This eligibility decision is strictly bound to the following tuple of cryptographic SHA-256 digests and runtime parameters:
- **Environment Profile ID:** `linux-x86_64-py311-numpy246-torch2141cpu-avx512`
- **Target Model Architecture:** `SmallCNN1D(num_classes=12)` (283,692 parameters)
- **Model Checkpoint SHA-256:** `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`
- **Preprocessing Parameter SHA-256:** `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`
- **Numerics Implementation SHA-256:** `cfa08740c4f2dbc811ed4a3b00fdd9b9876b66312907ab5ddefa0d27b43c27f2` (`ecdr/week4/numerics.py`)
- **Float32 Bridge Design SHA-256:** `adb37e74d98530bea7c38091ac2451302944407a286947cb799c041dedc13eaf` (`04_SYNTHESIS/week4/design/FLOAT32_BRIDGE.md`)
- **Operator Capabilities Artifact SHA-256:** `33f0fd584c143f4f5144d24c87582aa128afbce2ff68053bb03f92f8c383fe46`
- **Numerical Bridge Artifact SHA-256:** `ac7cc79a8381dabc110a2ee5612afbb4959f61aae61f66a5170927779db17688`
- **Supported Dot Product Lengths:** $n \in \{30, 96, 128, 192, 2048\}$
- **Safe Margin Acceptance Rule:** $\min_{j \in \{0, \dots, 11\} \setminus \{c^*\}} \underline{m}_j > 0.0$ (strictly positive, ties rejected).

---

## 4. Anti-Spoofing & Non-Bypass Gate (`W4-010-C04`)

To prevent bypass of formal verification, `load_numerical_capability` enforces the following fail-closed rules:
1. **No Boolean Override:** A configuration flag (e.g. `is_sound=True` or `bypass_gate=True`) cannot grant `ELIGIBLE_FOR_PROFILE`.
2. **No Untrusted JSON Injection:** Any caller-provided JSON artifact without matching cryptographic source hashes is rejected with `INVALIDATED`.
3. **Profile Mismatch Invalidation:** Running on a GPU, alternate BLAS, or different CPU architecture immediately invalidates eligibility (`status = INELIGIBLE_OR_INVALIDATED`, `formal_reuse_eligible = False`).
4. **Independent Review Required in Registry:** Task `W4-010` must be marked `VERIFIED` in `04_SYNTHESIS/week4/tasks.json` with an independent reviewer (`reviewer_id != assignee`). If self-reviewed or pending, the capability downgrades to `INELIGIBLE_PENDING_INDEPENDENT_REVIEW`.

---

## 5. Expiry & Invalidation Conditions (`W4-010-C05`)

The numerical capability automatically expires and must be invalidated (`formal_reuse_eligible = False`) if any of the following occur:
- Any file in the claimed write set (`ecdr/week4/numerics.py`, `reference.py`) is modified (hash mismatch).
- The model checkpoint or normalization parameters are changed.
- BatchNorm is placed in `train()` mode (`training == True`).
- Torch thread count or BLAS backend is altered in a manner that modifies reduction ordering.

---

## 6. Component Scope & Non-Overreach (`W4-010-C06`)

- **Scope Limitation:** This task certifies exclusively the **numerical arithmetic and backward error bridge** component of Gate G3.
- **No Blanket Gate G3 Pass:** Gate G3 as a whole also requires domain construction (M2), certificate store / issuer (W4-022), and pre-test sealing (W4-030, W4-031). W4-010 does **not** grant full G3 pass.
- **No Runtime Certificate Minting:** W4-010 does **not** mint runtime certificates. Certificates may only be issued by the dedicated certificate issuer (`W4-022`) during actual runtime evaluation.

---

## 7. Strict Fallback & Diagnostic Continuation (`W4-010-C07`)

- **Strict Fallback Protocol:** Whenever a candidate domain evaluates to $\min_{j} \underline{m}_j \le 0.0$ (inconclusive), runtime execution **must fallback** to reference inference (`ReferenceOwner.infer`).
- **Downstream Continuity:** Milestone M2 (Methods: IBP, CROWN, Lipschitz, proposals) and Milestone M3 (Incremental Engine) may proceed independently. However, any regional certified reuse branch relying on formal float32 bounds must strictly check `load_numerical_capability().formal_reuse_eligible == True`.
