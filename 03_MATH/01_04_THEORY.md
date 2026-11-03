# 04 THEORY — Event-Conditioned Certified Decision Reuse

> Updated 03/11/2026. **Actual reference: frozen SmallCNN1D K=12; G2 HOLD.**
> This source owns the mathematical specification. Implementation/evaluation
> evidence is reported separately in §7 and the canonical Week 3 reports.
> A real-arithmetic bound is not an executable float32 certificate.
> Contract: [02_07_G3_DOMAIN_CONTRACT.md](02_07_G3_DOMAIN_CONTRACT.md).
> Historical K=6/GAP artifacts remain provenance, not the current reference.

## 1. Objects, coordinates and reference decision

### 1.1 Window, event descriptor and perturbation region

- **Window `x_t ∈ R^{6×128}`:** a raw sensor observation, flattened in the source
  channel/time order to `R^768` when needed. One window spans 2.56 s at 50 Hz;
  stride 64 advances 1.28 s. It is not an event or a region.
- **Event descriptor `e_t`:** proposed information about change in the stream,
  used by a selector to choose candidate template/budgets. Exact features and
  policy are unimplemented. It need not equal a ground-truth activity label.
  Names such as stable/locomotion/transition are illustrative, not implemented
  semantic event categories.
- **Candidate region `S_e(x^r)`:** a set of possible raw windows defined relative
  to a trusted anchor, not a time interval. Existing q75/q90/PCA candidates do
  not establish an online event selector.
- **Certified region:** a candidate with valid bounds proving the required
  decision property. Membership alone is not a certificate. Certification
  need not change the candidate geometry.

### 1.2 Frozen preprocessing and backbone

Use `x` for raw input and `z` for normalized input. Let `m` and positive
`sigma` be the train-fit per-channel statistics repeated over time:

    A = diag(1 / sigma),   N(x) = A(x - m),   z = N(x),
    f_theta: R^768 -> R^12,                 F = f_theta o N.

`F` is the ideal real-arithmetic reference pipeline. The executable reference
is its frozen float32 implementation, denoted `F_hat` when the distinction
matters. `scripts/backbone.py:SmallCNN1D(num_classes=12)` has 283,692 trainable
parameters and this graph:

    [B,6,128] -> Conv(6,32,k5,p2) -> BN(eval) -> ReLU -> Pool2
              -> Conv(32,64,k3,p1) -> BN(eval) -> ReLU -> Pool2
              -> Conv(64,64,k3,p1) -> BN(eval) -> ReLU -> [B,64,32]
              -> Flatten(2048) -> Linear(2048,128) -> ReLU -> Linear(128,12).

Checkpoint: `output/week1_closeout_20261102/reference/selected_best.pt`;
SHA-256 `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
Preprocessing ID: `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.
The ordered 12-class list is read from the frozen reference configuration;
class competitors are all eleven indices other than the anchor winner.
The canonical authority is `00_OVERVIEW/week1/SOURCE_OF_TRUTH.md`, with
Week 2 graph audit `output/week2_foundations_20261103/model_audit.json`.

Historical K=6 `SmallCNN1D` has 282,918 parameters; its pure-only metrics and
checkpoint do not identify this reference. The separate valid-convolution/GAP
family supplies conditional historical theorems, not this graph's head.

These affine equations describe real arithmetic. The float32 pipeline rounds
intermediate values; §3.4 states the additional numerical proof obligation.

A **trusted anchor `x^r`** actually ran the reference pipeline. Store
`z^r=N(x^r)` and `a=argmax_k F_k(x^r)`. The minimal contract rejects ties and
non-finite values. A reused input cannot become a trusted anchor without a new
reference inference or a separate proved rule.

Goal: skip reference inference only if a matching valid certificate and query
membership establish the same decision. This preserves the backbone decision,
not activity-label correctness. A wrong anchor prediction can be preserved.

## 2. Candidate domains and membership

### 2.1 Global and diagonal boxes; minimal G3 target

In normalized coordinates, the minimal candidate is

    Z_e(z^r) = {z^r + B_e u : ||u||_infinity <= 1},
    B_e = diag(b_e),   b_e >= 0,
    S_e(x^r) = N^{-1}(Z_e(z^r)).

Mathematical membership is `|z_i-z_i^r| <= b_e,i` for every coordinate. The
raw preimage has budgets `b_e^raw = sigma * b_e`. Conversely, a raw audit budget
corresponds to `b_e=b_e^raw/sigma` in normalized coordinates. Coordinate order
and scales belong to region identity. These are real-arithmetic equivalences;
executable containment still needs numerical justification.

`B_e=rho I` gives an L-infinity box of radius `rho`; `B_e=I` gives radius 1.
Membership uses O(d) arithmetic operations; actual runtime is unmeasured.

The audit fits per-coordinate raw delta quantiles on train same-activity pairs.
Its q50 level has very low joint coverage; q75/q90 are candidates to evaluate.
Quantiles define a region. They provide neither a decision certificate nor a
statistical guarantee on future coverage, but do not prevent sound verification
of the selected region.

### 2.2 PCA projection domain measured by the audit

`scripts/05_audit_temporal_events.py` uses raw adjacent deltas
`delta=x_{t+1}-x_t`. It fits centered PCA on a train subset, with mean `mu` from
all train pairs and score/residual thresholds from train same-activity pairs.
Write its ideal orthonormal basis as `U ∈ R^{d×q}`, `U^T U=I`, and
`P_perp=I-UU^T`. The measured candidate is

    w = x - x^r - mu,
    S_e^PCA(x^r) = {x : |U^T w| <= b, ||P_perp w||_2 <= epsilon}.

Equivalently, in real arithmetic,

    x = x^r + mu + U diag(b) u + v,
    ||u||_infinity <= 1, U^T v = 0, ||v||_2 <= epsilon.

Projection checks are exact membership for this **projection-defined domain**,
assuming an orthonormal basis and exact arithmetic. Residual uses an L2 budget,
not a coordinate box. Centering by `mu` is essential. This shifted domain need
not contain `x^r`; a certificate can still compare its predictions with the
class obtained at `x^r`, but cannot assume anchor inclusion.

The saved audit reports `q=108`, variance explained 0.9008 and test membership
coverage 0.3542 versus diagonal q90 0.3194. These are adjacent-pair coverage
results, not certified reuse rates; they are not budget/coverage-matched.
The script's float32 decomposition has not been validated as a sound runtime
membership procedure. No PCA certificate implementation is claimed.

Normalizing this raw PCA domain transforms directions and residual geometry;
`AU` is generally not orthonormal. Do not substitute `AU` into an orthogonal
projection formula unchanged. Verify `F` over the raw domain or transform
bounds/domains consistently as in §3.3.

### 2.3 Separate theoretical variant: coordinate residual box

The previous draft's domain remains a possible variant, distinct from the PCA
audit:

    S_e^box = {c + D_e u + v : ||u||_infinity <= 1, |v_i| <= epsilon_i}.

Here `D_e` may be non-diagonal and residual `v` need not be orthogonal. Generic
exact membership is an existential feasibility problem. A particular projection
decomposition satisfying these constraints is a sufficient witness, not
in general an iff membership test for this Minkowski-sum domain. The project
has no measured coverage or implemented verifier for this variant.

### 2.4 Selector versus membership

The selector only proposes a region/template. Membership must be established
independently of that heuristic. A wrong event name does not invalidate an
otherwise valid certificate and membership check. A correct activity label
cannot replace either condition. Conservative membership may reject valid
queries and lose coverage; false acceptance threatens soundness.

## 3. Certificate and proof obligations

### 3.1 Class-pair condition and Proposition 1

For anchor class `a`, define raw-pipeline margins

    g_aj(x) = F_a(x) - F_j(x),   j != a.

**Proposition 1 — sufficient decision preservation condition.** Suppose `a`
is the unique reference winner at a trusted anchor. For a candidate `S`, assume
valid lower bounds `ell_aj` satisfy `g_aj(x) >= ell_aj` for every `x ∈ S` and
`j != a`. If `min_j ell_aj > 0`, every `x ∈ S` has unique reference winner `a`.

*Proof.* For any such `x` and `j`, `F_a(x)-F_j(x) >= ell_aj > 0`; therefore
`F_a(x)` exceeds every competing logit. Query membership establishes that this
result applies to the query. □

Strict positivity is a sufficient acceptance rule. A lower-bound algorithm
need not certify every truly invariant region; deterministic tie-breaking can
also preserve decisions at zero margin. The minimal contract rejects such
inconclusive/tied cases instead of proving a separate tie rule. Failure to
certify does not prove the decision property false.

### 3.2 Affine lower bounds and support functions

Suppose a valid affine relaxation gives `g_aj(x) >= alpha^T x + beta` over the
whole raw domain. Minimizing that affine function yields:

| Domain | Infimum of affine lower bound |
|---|---|
| Raw diagonal `x^r+diag(b^raw)u`, `norm_inf(u)<=1` | `alpha^T x^r+beta-norm_1(diag(b^raw) alpha)` |
| Coordinate residual variant `c+D u+v`, `abs(v)<=epsilon` | `alpha^T c+beta-norm_1(D^T alpha)-sum_i epsilon_i abs(alpha_i)` |
| Audit PCA projection domain, §2.2 | `alpha^T(x^r+mu)+beta-norm_1(diag(b)U^T alpha)-epsilon norm_2(P_perp alpha)` |

`norm_1`, `norm_2` and `norm_inf` denote L1, L2 and L-infinity norms.

PCA's residual term is the support of an L2 ball in the orthogonal complement
of `U`. It differs from the coordinate-box term. With `D=rho I` and zero
residual, the second expression reduces to `alpha^T c+beta-rho||alpha||_1`.

In normalized coordinates, if `f_a(z)-f_j(z) >= h^T z+k` is valid on the entire
normalized diagonal box, its lower margin is

    ell_aj = h^T z^r + k - ||diag(b_e) h||_1.

These are support-function identities in real arithmetic, not evidence that a
CROWN run produced valid coefficients for the frozen model.

### 3.3 Domain validity and coordinate transformation

CROWN/LiRPA is proposed machinery; the first G3 target is the normalized
diagonal box. A bound on an enclosure is also valid on a candidate with proved
containment. Coefficients computed on a smaller or unrelated domain have no
validity guarantee outside that domain. Membership and relaxation must refer
to the same region/version or a documented containing relaxation domain.

If `h^T z+k` is valid for every `z ∈ N(S)`, its raw expression is

    alpha = A^T h,   beta = k - h^T A m,
    h^T N(x)+k = alpha^T x+beta.

One may then use §3.2's raw support functions. This transformation does not
establish relaxation validity over `N(S)`. A PCA support formula does not
supply correlated-domain support in auto_LiRPA. Actual-graph integration and its remaining numerical limitation are recorded in §7; these support identities alone do not validate that implementation.

### 3.4 Mathematical and numerical soundness

Proposition 1 proves an ideal mathematical implication. Agreement with the
executable float32 reference also requires a justified link between that
reference and the computed bounds, normalized inputs and membership checks.
G3 must document operator support, evaluation mode, arithmetic/rounding
assumptions, non-finite handling and conservative boundary checks. An arbitrary
positive tolerance is not a proof of floating-point error control. If numerical
soundness is unresolved, label the guarantee conditional on stated arithmetic
assumptions, rather than end-to-end formal soundness.

Zero observed mismatches is implementation evidence, not proof for all accepted
inputs. Certificate identity includes model, normalization, class order,
anchor, region and relevant numerical semantics. A valid proof on the wrong
checkpoint/domain does not certify the current query.

### 3.5 Toy example: structure can help

Let `g(x_1,x_2)=1+x_1-x_2`, anchor `(0,0)`. On `|x_1|,|x_2|<=0.8`, minimum
margin is `-0.6`. On `x_1=x_2=s`, `|s|<=0.8`, it stays 1. Coordinate residual
budgets `|v_1|,|v_2|<=0.1` leave lower margin 0.8. This illustrates a potential
benefit when directions cancel in a margin; it establishes no HAPT coverage,
learned-model bound quality or runtime saving.

## 4. Proposed G3 mechanism and later cost evaluation

First path: trusted anchor → choose normalized diagonal candidate → construct
valid class-pair bounds → store successful certificate → online identity and
conservative membership check → reuse or fallback. Candidate construction,
membership, verifier and reuse gate are separate responsibilities. Records and
failure behavior: [02_07_G3_DOMAIN_CONTRACT.md](02_07_G3_DOMAIN_CONTRACT.md).
Executable component status and its numerical restrictions are recorded in §7; the implication above remains conditional on valid bounds and containment.

Only real reference inference refreshes an anchor in the minimal policy. Reset
between recordings; reject stale model/preprocessing/domain versions. Failed
or inconclusive construction, failed membership, ties and unsupported
operations cause fallback. Instrument component costs and anchor age in G3;
**G5 decides net saving**, and G4 tests the benefit of structure.

Under a representative approximately constant forward-cost model,

    E[T] = T_e + T_m + (1-r) T_f + q T_b,
    saving iff r T_f > T_e + T_m + q T_b.

`T_e,T_m` are mean event and lookup/membership costs; `r` is reuse fraction;
`q` is mean builds per query; `T_b` is mean cost per build. With path-dependent
forward cost, use measured conditional costs or per-query sums instead of
factorizing expectation without justification. Do not assume `q=1-r`.
Include initial construction in the counted build costs or add `T_init/N`
for the declared evaluation horizon; never omit it.

Saved CPU batch1 forward baseline: mean 0.2511 ms, p50 0.2357 ms, p95 0.3313 ms
(`data/g2_cpu_latency.json`). This times already-normalized RAM tensors, not
preprocessing or a fallback trace. G5 needs comparable complete paths on the
same workload/device, including construction/rebuilds and actual preprocessing. This historical timing is K=6 provenance; it is not a K=12 timing result.

## 5. Metrics and experiments still required

- **Decision agreement:** shadow-reference mismatch count/rate on reused
  queries. Zero observed mismatches is necessary evidence, not proof. Report
  formal guarantees only under the validated mathematical/numerical assumptions.
- **Certified reuse:** reused queries with a valid certificate divided by a
  stated stream denominator. Report construction success, membership hits,
  fallback reasons, anchor age, bound margins and builds separately. No reuse
  means empirical reused-query agreement is undefined, not measured 100%.
- **Task accuracy/macro-F1:** on 3,572 supervised K=12 test windows only:
  frozen reference accuracy 0.8905375 and macro-F1 0.7762645. The historical
  pure-K=6 result (2,874 test windows, best F1 0.9354; three-seed
  0.9122 ± 0.0183) is retained solely as provenance.
- **Runtime:** mean/p50/p95, throughput, build/membership/forward costs, memory
  and environment. Reuse rate alone does not establish saving.

Denominators: full corpus 17,453 windows; full chronological test stream 5,053;
supervised K=12 test subset 3,572. The historical pure-K=6 corpus contains
10,017 windows and its test subset has 2,874. These are distinct populations.
Fit candidate budgets on training and select policy on validation; chronological
test evaluation resets at recording boundaries. Reference-decision agreement
uses all test windows, including those excluded from activity-label metrics.

Existing audit coverage and G2 metrics have already been inspected on the test
subjects. Do not describe this test split as untouched by the whole research
workflow; disclose exploratory use and any subsequent method-selection impact.

Separate two comparisons: fix model+domain and vary bound method; fix
model+bound method and vary global/diagonal/correlated domain. Specify what is
matched (coverage, budget or a defensible region-size measure) and report it.
Tiny regions may certify while covering no useful queries. PCA's adjacent-pair
coverage advantage alone passes neither G3 nor G4/G5.

## 6. Actual-graph wave and feature identities (H12–H14)

For C=6 channels, W=128 samples and shift s=64 samples, a window has 768
scalar values. If the overlap is fixed exactly, only Cs=384 new values vary:

    X_next = [X_anchor[:,64:128], U],   U in R^(6x64).

After the same frozen pointwise normalization this remains a fixed-prefix
normalized box. If raw U budgets are used, transform them with the frozen
channel scales and conservatively enclose normalization arithmetic. Exact
384-variable parameterization does not halve forward or verifier runtime.
A checker without trusted overlap state must also validate all overlap values.

For zero-based final features H_j, the exact inclusive theoretical input
support derived from this graph is:

| Operator output at index i | Inclusive input support | Jump |
|---|---|---|
| Conv1 k5 p2 | [i−2,i+2] | 1 |
| Pool1 k2 stride2 | [2i−2,2i+3] | 2 |
| Conv2 k3 p1 | [2i−4,2i+5] | 2 |
| Pool2 k2 stride2 | [4i−4,4i+7] | 4 |
| Conv3 k3 p1 | [4i−8,4i+11] | 4 |

Thus width is 20 samples and feature jump 4. The shift 64 is 16 final
positions. With frozen BN evaluation statistics, shared kernels and matching
pool phase, RF containment in next-window overlap [0,63] gives
`4j−8>=0` and `4j+11<=63`, hence **j=2,…,13**:

    H_j(X_next) = H_(j+16)(X_anchor),   j=2,...,13.

Proof: trace each used index through the graph. All intermediate ancestors
for these j are within their valid array interiors, so no padding node is used.
Translation by 64 input positions becomes 32 after the first pool and 16
after the second. Corresponding convolution arguments, frozen BN affine maps,
ReLU operands and paired max operands are equal. Induction through those
operators gives equality. This is a real-arithmetic identity for 12 positions
and 64 channels, 768 feature scalars. Bitwise equality is a separate executed
property of a chosen backend, not a universal float32 theorem. The often
quoted RF interval [4j−9,4j+10] has the correct width but the wrong origin.
Boundary features must not be included merely by counting width.

The historical A1–A6 valid/stride-one/no-pool theorem remains valid under
its own assumptions: R=9, n=120, 56 shared positions for s=64. Its GAP delta
identity remains valid only for GAP then affine logits. The actual head is
`W2 ReLU(W1 Flatten(H)+b1)+b2`; shared features at shifted positions can
receive different dense weights. No new-minus-dropped GAP output rule follows.

## 7. Evidence status and reproducibility boundary

The complete H01–H24 original-statement, architecture, proof-obligation and
K=12 disposition ledger is in `03_Week_3_Mathematical_Foundations.tex`;
`04_ECDR_Mathematics_Explained.tex` provides worked examples and the same ID
mapping. Supplementary geometry formulas retain their declared assumptions.
Current execution receipts are under `output/week3_certification_20261103/`.
Only the actual completed receipts support empirical claims; no prior K=6
probe or an unexecuted measurement plan closes a K=12 obligation.

Distinguish DERIVED ideal implications, VERIFIED finite execution evidence,
CONDITIONAL guarantees, FALSIFIED universal claims, and NOT ESTABLISHED
empirical/numerical conclusions. Candidate bounds and positive point margins
cannot be counted as executable certified reuse. If an end-to-end numerical
bridge is unavailable, the formal runtime gate must reject that candidate and
fall back, even if a real-arithmetic diagnostic bound is positive. G2 remains
HOLD; G3/G4/G5 are governed by their own unmet premises and measured evidence.

The executed `experiment_results.json` records 5,053 test windows, selected
validation radius 1e-6, conditional-positive build counts 5,034 diagonal and
4,211 wave, zero next candidate members and zero formal runtime reuse.
All four runtime replays fall back on all 5,053 queries with zero full-stream
disagreements; reused-query agreement is N/A. Complete ECDR means are
3.1069–3.1975 ms/query versus core reference 0.2147–0.2240 ms/query and
identity-rehashing wrapper 1.5885–1.6747 ms/query. These negative savings apply
to the measured fail-closed CPU replay policies. No general impossibility
claim follows. Native auto-LiRPA 1D conversion fails at MaxPool; singleton-height
2D is algebraically equivalent in reals and sampled consistent, but remains a
conditional diagnostic outside the package's torch version contract. The
93 executed tests (60 historical, 27 actual-graph, 6 mathematical) and 65
baseline artifact checks do not establish a universal numerical certificate.
