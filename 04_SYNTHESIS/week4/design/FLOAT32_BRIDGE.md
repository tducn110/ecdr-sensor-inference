# W4-009: End-to-End Frozen Float32 Bridge & Safe Margin Acceptance

**Task ID:** `W4-009`  
**Milestone:** M1 — Numerics & Outward Interval Arithmetic  
**Dependencies:** `W4-005` (`VERIFIED`), `W4-006` (`VERIFIED`), `W4-007` (`VERIFIED`), `W4-008` (`VERIFIED`)  
**Hypotheses:** H05, H06, H08, H22  
**Target Profile:** `linux-x86_64-py311-numpy246-torch2141cpu-avx512`

---

## 1. End-to-End Composition of Operator Errors (`W4-009-C01`)

### 1.1 Complete Computational Chain
For any raw sensor window $X_{\text{raw}} \in \mathbb{R}^{6 \times 128}$ belonging to a candidate raw domain $\mathcal{D}_{\text{raw}} = [\underline{X}_{\text{raw}}, \overline{X}_{\text{raw}}]$, the reference evaluation chain consists of:
1. **Raw Conversion & Normalization ($N_{\text{exec}}$)**:
   - Coordinatewise: $X_{\text{norm}}[c, t] = \text{fl}_{32}\left( \frac{\text{fl}_{32}(X_{\text{raw}}[c, t] - \mu_c)}{\sigma_c} \right)$.
   - By Theorem 1 (`W4-007` / `reference.py`), monotonicity of round-to-nearest float32 subtraction and division guarantees that $X_{\text{norm}} \in [\underline{X}_{\text{norm}}, \overline{X}_{\text{norm}}]$ with zero dilation.
2. **Backbone Stages 1 to 3 ($f_{\text{conv}}$)**:
   - Stage 1: $\text{Conv1d}(6 \to 32, k=5, p=2) \to \text{BatchNorm1d}(32) \to \text{ReLU} \to \text{MaxPool1d}(k=2, s=2)$.
   - Stage 2: $\text{Conv1d}(32 \to 64, k=3, p=1) \to \text{BatchNorm1d}(64) \to \text{ReLU} \to \text{MaxPool1d}(k=2, s=2)$.
   - Stage 3: $\text{Conv1d}(64 \to 64, k=3, p=1) \to \text{BatchNorm1d}(64) \to \text{ReLU}$.
   - At each Conv layer, Higham backward error analysis bounds reduction rounding error:
     $$\Delta_{\text{conv}} \le \gamma_{n+1} \left( |W| \cdot \max(|U_{\text{lo}}|, |U_{\text{hi}}|) + |b| \right) + (n+1) \Delta_{\text{sub}}$$
   - At each BatchNorm layer, affine parameter folding error is bounded by:
     $$\Delta_{\text{bn}} \le \gamma_4 \left( |x| |\alpha| + |\beta_{\text{eff}}| + |\mu \alpha| + |\beta| \right) + 4 \Delta_{\text{sub}}$$
   - At ReLU and MaxPool layers, operations are exact and non-expansive ($\Delta = 0.0$).
3. **Classifier Head ($f_{\text{head}}$)**:
   - $\text{Flatten}(64 \times 32 \to 2048) \to \text{Linear}(2048 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 12)$.
   - Yields interval enclosure $[\underline{z}_k, \overline{z}_k]$ for every executable logit $\text{fl}(z_k)$ ($k = 0, \dots, 11$).

### 1.2 Soundness of Competitor Margin Lower Bounds
For a declared winning class $c^* \in \{0, \dots, 11\}$ and each competitor $j \in \{0, \dots, 11\} \setminus \{c^*\}$, the true executable logit difference is $m_j(X) = \text{fl}(z_{c^*}(X)) - \text{fl}(z_j(X))$.

Because the verifier guarantees:
$$\underline{z}_{c^*} \le \text{fl}(z_{c^*}(X)) \le \overline{z}_{c^*} \quad \text{and} \quad \underline{z}_j \le \text{fl}(z_j(X)) \le \overline{z}_j \quad \forall X \in \mathcal{D}$$
it immediately follows that:
$$m_j(X) = \text{fl}(z_{c^*}(X)) - \text{fl}(z_j(X)) \ge \underline{z}_{c^*} - \overline{z}_j$$
Applying outward directed rounding `round_down_f32`:
$$\underline{m}_j \triangleq \text{round\_down\_f32}(\underline{z}_{c^*} - \overline{z}_j) \le \text{fl}(z_{c^*}(X)) - \text{fl}(z_j(X)) \quad \forall X \in \mathcal{D}$$
This constitutes a mathematically sound lower bound on the executable margin across the entire admitted domain.

---

## 2. Direct Head Difference vs. Separate Logits (`W4-009-C02`)

In linear verifiers (e.g. CROWN or DeepPoly), it is common to compute a single fused difference weight vector $W_{c^*} - W_j$ at the final linear layer:
$$z_{c^*} - z_j = \sum_{i=1}^{128} (W_{2, c^* i} - W_{2, ji}) h_i + (b_{2, c^*} - b_{2, j})$$

### The Semantic Divergence Hazard
PyTorch executable code **never** computes this fused difference. Instead, PyTorch executes:
1. $z_{c^*} = \text{fl}\left(\sum_{i=1}^{128} W_{2, c^* i} h_i + b_{2, c^*}\right)$
2. $z_j = \text{fl}\left(\sum_{i=1}^{128} W_{2, ji} h_i + b_{2, j}\right)$
3. Decision: $\arg\max_k \text{fl}(z_k)$ (which compares the rounded values directly).

If a verifier bounds $(W_{c^*} - W_j) h$ over $\mathbb{R}$ without accounting for the fact that PyTorch rounds $z_{c^*}$ and $z_j$ separately, coefficient rounding and subtraction cancelation errors can cause the fused real bound to falsely claim a positive margin when $\text{fl}(z_{c^*}) \le \text{fl}(z_j)$ in float32.

### Resolution in `ecdr/week4/numerics.py`
To satisfy `W4-009-C02`, `compute_margin_bounds` provides two verified paths:
1. **`separate_logits` (Default & Preferred)**: Directly uses $[\underline{z}_{c^*}, \overline{z}_{c^*}]$ and $[\underline{z}_j, \overline{z}_j]$ produced by the verified operator enclosures. This preserves the exact two-rounded-dot-products semantics of PyTorch with zero translation error.
2. **`direct_head_difference`**: When fusing weights $W_{c^*} - W_j$, the verifier calculates the fused ideal interval and **explicitly adds the sum of backward rounding errors of both separate reference dot products**:
   $$\underline{m}_j = \text{round\_down\_f32}\left( \underline{d}_j - (\Delta_{c^*} + \Delta_j) \right)$$
   where $\Delta_{c^*} = \gamma_{129} \sum |W_{2, c^* i}| |h_i| + |b_{2, c^*}|$ and $\Delta_j = \gamma_{129} \sum |W_{2, ji}| |h_i| + |b_{2, j}|$. This guarantees that the fused difference bound is provably conservative with respect to the two rounded PyTorch logits.

---

## 3. Safe Margin Acceptance Predicate & Tie Rejection (`W4-009-C03`)

The safe margin acceptance predicate is defined as:
$$\text{IsSafe}(\mathcal{D}, c^*) \iff \min_{j \in \{0, \dots, 11\} \setminus \{c^*\}} \underline{m}_j > 0.0$$

### Strict Positive Requirement
- **No Non-Strict Acceptance:** A margin of $\underline{m}_j = 0.0$ is strictly rejected (`is_positive = False`). In float32, equality between top logits constitutes a tie, which in PyTorch `argmax` is resolved by picking the lowest class index. Any potential tie on the domain invalidates certification.
- **Fail-Closed on Non-Finite Values:** If any logit or margin bound evaluates to `NaN` or `+Inf`/`-Inf`, the verifier immediately halts and returns `is_positive = False` with `failure_reason = "non_finite_margin"`.

---

## 4. Cryptographic & Profile Binding (`W4-009-C04`)

The end-to-end bridge is bound to the following invariants:
- **Locked Profile ID:** `linux-x86_64-py311-numpy246-torch2141cpu-avx512`
- **Reference Model:** `SmallCNN1D(num_classes=12)` (Checkpoint SHA-256: `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`)
- **Normalization Parameters:** Preprocessing ID `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`
- **Supported Dot Product Lengths:** $n \in \{30, 96, 128, 192, 2048\}$.
- **Backend Assumptions Explicitly Listed:**
  1. NumPy float32 subtraction and division follow IEEE-754 round-to-nearest.
  2. PyTorch ATen Conv1d and Linear CPU reductions follow sequential or tree summation whose worst-case forward error is bounded by Higham's $\gamma_N$ bound.
  3. No catastrophic exponent overflow ($> 3.4 \times 10^{38}$) occurs within the sensor dynamic range $[-50, +50]$.

---

## 5. Diagnostic Tightness & Real Near-Tie Controls (`W4-009-C05` & `W4-009-C07`)

### 5.1 Real Dataset Near-Tie Witness
- On test recording `48`, `global_row_id = 13358` (`step_idx = 236`), the top-1 logit is $6.2438531$ and the runner-up is $6.2437396$, yielding a razor-thin margin of **$m_1 = 1.13487 \times 10^{-4}$**.
- When evaluating a point box ($\epsilon = 0$) on this sample:
  - $\underline{z}_{\text{top1}} = 6.2438316$, $\overline{z}_{\text{runner\_up}} = 6.2437611$.
  - The calculated lower margin is $\underline{m}_1 = 7.05 \times 10^{-5} > 0.0$.
  - Thus, even at this extreme near-tie boundary, the Higham error bound is tight enough to prove the decision without false negative tie rejection on point inputs!
- However, as soon as an input perturbation of radius $\epsilon \ge 10^{-5}$ is added to all 768 input dimensions, the interval width expands through 3 Conv and 2 Linear layers by a factor exceeding $10^4$, causing $\underline{m}_1$ to become negative. This explains the 0% coverage of regional IBP certification on continuous streams.
