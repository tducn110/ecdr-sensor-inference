# W4-012: Actual K=12 Global Lipschitz-Margin Baseline Design & Analysis

**Task ID:** `W4-012`  
**Milestone:** M2 — Methods & Bounded Domains  
**Dependencies:** `W4-011` (`VERIFIED`)  
**Target Graph:** `SmallCNN1D(num_classes=12)` (283,692 parameters)  
**Hypotheses Addressed:** H06, H19, H21, H22  
**Status:** IMPLEMENTATION & REVIEW

---

## 1. Executive Summary & Purpose

Task `W4-012` develops the matched global Lipschitz comparator for the frozen `SmallCNN1D(num_classes=12)` architecture on the HAPT continuous sensor benchmark. In prior weeks, historical adaptive verifiers (`ecdr/certified_adaptive.py`) relied on a simplified 6-class model with Global Average Pooling (GAP) and $L_1$ norm bounds. 

This design establishes a mathematically rigorous, graph-matched Lipschitz baseline under the Euclidean operator norm ($\ell_2$), deriving exact spectral bounds across all 15 convolutional, normalization, pooling, flattening, and multi-layer perceptron layers without inheriting constants from obsolete models.

---

## 2. Operator Norm Derivations under $\ell_2$ (`W4-012-C01`, `C02`)

For a multi-layer sequential neural network $f(x) = g_N \circ g_{N-1} \circ \dots \circ g_1(x)$, the global Lipschitz constant with respect to the input Euclidean norm $\|x\|_2$ is bounded by the product of individual operator norms:
$$\text{Lip}_2(f) \le \prod_{l=1}^N \|g_l\|_2$$

### 2.1 1D Convolutional Layers (`Conv1d`)
Each convolution $x \mapsto \text{conv1d}(x, W)$ with stride $s$ and padding $p$ is a linear operator between Hilbert spaces $\mathbb{R}^{C_{\text{in}} \times L_{\text{in}}} \to \mathbb{R}^{C_{\text{out}} \times L_{\text{out}}}$. Its $\ell_2$ induced operator norm is the maximum singular value:
$$\|W_{\text{conv}}\|_2 = \sigma_{\max}(W_{\text{conv}}) = \sup_{\|x\|_2 = 1} \|\text{conv1d}(x, W)\|_2$$
We compute $\sigma_{\max}$ using Power Iteration on the operator $A(x) = \text{conv1d}(x, W)$ and its adjoint $A^*(y) = \text{conv\_transpose1d}(y, W)$.

### 2.2 1D Batch Normalization Layers (`BatchNorm1d` in `eval()` mode)
In evaluation mode, BatchNorm applies an affine elementwise coordinate transform:
$$y_{c, t} = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}} x_{c, t} + \left( \beta_c - \frac{\gamma_c \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \right)$$
The linear scaling matrix is block-diagonal with diagonal entries $\alpha_c = \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}$. Its $\ell_2$ operator norm is the maximum absolute scaling coefficient across channels:
$$\|\text{BN}\|_2 = \max_{c \in \{0, \dots, C-1\}} \left| \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}} \right|$$

### 2.3 Activation & Pooling Layers (`ReLU`, `MaxPool1d`, `Flatten`)
- **`ReLU`:** Coordinatewise non-expansive operator, $\text{Lip}_2(\text{ReLU}) = 1.0$.
- **`MaxPool1d(kernel_size=2, stride=2)`:** For non-overlapping pooling windows ($s = k = 2$), each input coordinate influences at most one output coordinate. By coordinatewise inequality:
  $$|y_i - y'_i|^2 \le |x_{2i} - x'_{2i}|^2 + |x_{2i+1} - x'_{2i+1}|^2 \implies \|y - y'\|_2 \le \|x - x'\|_2$$
  Hence $\text{Lip}_2(\text{MaxPool1d}) \le 1.0$.
- **`Flatten(start_dim=1)`:** Reshape permutation preserves Euclidean length exactly (isometry), $\text{Lip}_2(\text{Flatten}) = 1.0$.

### 2.4 Multi-Layer Perceptron Head (`Linear1`, `Linear2`)
- **`Linear1(2048 -> 128)`:** Dense matrix multiplication $h = W_1 f + b_1$. The operator norm is the spectral norm:
  $$\|W_1\|_2 = \sigma_{\max}(W_1) = \sqrt{\lambda_{\max}(W_1^T W_1)}$$
- **Competitor Decision Margin on `Linear2(128 -> 12)`:**
  The competitor margin function for target class $c^*$ against competitor $j \in \{0, \dots, 11\} \setminus \{c^*\}$ is:
  $$m_j(h) = (W_{c^*} - W_j) h + (b_{c^*} - b_j)$$
  The operator norm of this linear functional from $\mathbb{R}^{128} \to \mathbb{R}$ under $\ell_2$ is the Euclidean vector norm of the weight difference:
  $$\|W_{c^*} - W_j\|_2 = \sqrt{\sum_{k=1}^{128} (W_{c^*, k} - W_{j, k})^2}$$

---

## 3. Empirical Spectral Measurements on Frozen Reference Checkpoint

Measurements performed on `SmallCNN1D` checkpoint (`8ffc5dd8...`):

| Layer Index | Layer Type | Shape / Dimensions | Spectral Bound $\|g_l\|_2$ |
| :--- | :--- | :--- | :---: |
| `conv.0` | `Conv1d(6->32, k=5, p=2)` | $W \in (32, 6, 5)$ | **$1.9028$** |
| `conv.1` | `BatchNorm1d(32)` | $\gamma, \sigma^2 \in (32,)$ | **$3.0539$** |
| `conv.2` | `ReLU` | $(32, 128)$ | $1.0000$ |
| `conv.3` | `MaxPool1d(k=2, s=2)` | $(32, 64)$ | $1.0000$ |
| `conv.4` | `Conv1d(32->64, k=3, p=1)` | $W \in (64, 32, 3)$ | **$1.5327$** |
| `conv.5` | `BatchNorm1d(64)` | $\gamma, \sigma^2 \in (64,)$ | **$3.1434$** |
| `conv.6` | `ReLU` | $(64, 64)$ | $1.0000$ |
| `conv.7` | `MaxPool1d(k=2, s=2)` | $(64, 32)$ | $1.0000$ |
| `conv.8` | `Conv1d(64->64, k=3, p=1)` | $W \in (64, 64, 3)$ | **$1.7621$** |
| `conv.9` | `BatchNorm1d(64)` | $\gamma, \sigma^2 \in (64,)$ | **$3.1684$** |
| `conv.10`| `ReLU` | $(64, 32)$ | $1.0000$ |
| `head.0` | `Flatten` | $2048$ | $1.0000$ |
| `head.1` | `Linear(2048->128)` | $W \in (128, 2048)$ | **$2.7108$** |
| `head.2` | `ReLU` | $128$ | $1.0000$ |

### Cumulative Backbone Lipschitz Constant:
$$L_{\text{backbone}} = 1.9028 \times 3.0539 \times 1.5327 \times 3.1434 \times 1.7621 \times 3.1684 \times 2.7108 \approx \mathbf{423.7158}$$

### Final Linear Differences ($\|W_c - W_j\|_2$):
- Minimum pairwise norm: $\min_{c \ne j} \|W_c - W_j\|_2 = 0.7968$
- Maximum pairwise norm: $\max_{c \ne j} \|W_c - W_j\|_2 = 1.1889$
- **Total Margin Lipschitz Range:** $L_{c, j} \in [\mathbf{337.62}, \mathbf{503.76}]$

---

## 4. Certified Radius & Boundary Handling (`W4-012-C03`)

Given an anchor sample $x_0$ with reference logit margin $\Delta_j(x_0) = z_{c^*}(x_0) - z_j(x_0) > 0$:
$$R_{c^*, j} = \frac{\Delta_j(x_0)}{L_{c^*, j}}$$
The overall certified Euclidean radius is:
$$R_{\text{safe}}(x_0) = \min_{j \ne c^*} R_{c^*, j} = \min_{j \ne c^*} \frac{\Delta_j(x_0)}{L_{c^*, j}}$$

### Boundary & Edge Cases:
1. **$L_{c^*, j} = 0.0$ (Diagonal $c^* = j$):** The self-margin is identically zero and imposes no radius restriction ($R = \infty$).
2. **Near-Zero or Negative Margin ($\Delta_j \le 0.0$):** If the anchor prediction is not strictly separated, $R_{\text{safe}} = 0.0$.
3. **Open Ball Strictness:** To avoid floating-point boundary containment errors, the certified radius is represented strictly via `nextafter(R, 0.0)`.

---

## 5. Scientific Findings on Hypotheses (H06, H19, H21, H22)

### 5.1 H06: Conservatism of Global Lipschitz Bounds
Because $L_{\text{margin}} \approx 500$, the certified radius is severely compressed:
- For a confident validation window with margin $\Delta = 2.0$:
  $$R_2 \approx \frac{2.0}{500} = 0.004$$
  Over 768 dimensions ($6 \times 128$), the per-coordinate perturbation allowance is merely:
  $$\epsilon_\infty \approx \frac{0.004}{\sqrt{768}} \approx 0.00014 = 1.4 \times 10^{-4}$$
- For near-tie control sample row 13358 ($\Delta = 1.13487 \times 10^{-4}$):
  $$R_2 \approx \frac{1.13 \times 10^{-4}}{500} \approx 2.26 \times 10^{-7}$$
  The certified ball collapses entirely.

### 5.2 H19: Vanishing Future Coverage on Continuous Streams
In actual HAPT sensor data, the empirical Euclidean displacement between adjacent overlapping windows ($t$ and $t+1$) is $\|x_{t+1} - x_t\|_2 \approx 0.5 \to 2.0$. Because the physical displacement ($0.5$) is 100 to 500 times larger than the global Lipschitz radius ($0.004$), **the reuse rate of global Lipschitz singletons on continuous streaming test data is identically 0%**.

### 5.3 H21: Rejection of Historical GAP Constants
Historical verifiers assumed $L_1$ bounds and GAP pooling with small constants ($L \approx 5 \to 12$). In `SmallCNN1D`, the absence of GAP, the presence of an expansive 2048-dimensional dense projection (`Linear1`), and 12-class head differences increase the true Lipschitz constant by nearly two orders of magnitude ($L \approx 500$). Reusing historical constants would be mathematically unsound.

### 5.4 H22: Epistemic Scope of Lipschitz Constants
Lipschitz bounds are computed in real arithmetic using spectral power iterations. In accordance with `ORCHESTRATOR.md` and `W4-012-C04`, the numerical status of the Lipschitz baseline is strictly designated as:
$$\text{NumericalStatus} = \text{CONDITIONAL\_REAL\_ARITHMETIC}$$
It serves as a fast diagnostic baseline, but cannot authorize formal outward float32 certificates.
