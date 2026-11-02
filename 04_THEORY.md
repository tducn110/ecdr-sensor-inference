# 04 THEORY — Event-Conditioned Certified Decision Reuse

> Updated 22/10/2026. **Working specification; verifier and certified reuse are
> unimplemented, G3 pending.** Gate record: G0 PASS; G1
> COMPLETE_WITH_REVISIONS; G2 PASS. This revision reconciles notation and candidate
> domains with the existing audit; it reports no new experiment and does not
> freeze a validated implementation. Minimal G3 scope:
> [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).
> Measured results remain in `data/` and [03_DATA_AUDIT.md](03_DATA_AUDIT.md).

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
    f_theta: R^768 -> R^6,                 F = f_theta o N.

`F` is the reference pipeline. `scripts/backbone.py` applies normalization
outside `SmallCNN1D`; the current model includes BatchNorm, ReLU and MaxPool
and has 282,918 parameters. Certificate construction must use the frozen
checkpoint in evaluation mode, including BN running state, class order and
normalization. K=6 is fixed; K=12 remains HOLD for data-support reasons, not
because Proposition 1 requires six classes.

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
supply correlated-domain support in auto_LiRPA. Integration must validate
operators and bounds on the actual BN/MaxPool graph; no result exists yet.

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
failure behavior: [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).
These components remain unimplemented.

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
same workload/device, including construction/rebuilds and actual preprocessing.

## 5. Metrics and experiments still required

- **Decision agreement:** shadow-reference mismatch count/rate on reused
  queries. Zero observed mismatches is necessary evidence, not proof. Report
  formal guarantees only under the validated mathematical/numerical assumptions.
- **Certified reuse:** reused queries with a valid certificate divided by a
  stated stream denominator. Report construction success, membership hits,
  fallback reasons, anchor age, bound margins and builds separately. No reuse
  means empirical reused-query agreement is undefined, not measured 100%.
- **Task accuracy/macro-F1:** only on the declared ground-truth K=6 scope.
  Existing best test macro-F1 is 0.9354; three-seed mean 0.9122 ± 0.0183 on
  2,874 pure K=6 test windows, not full-stream accuracy.
- **Runtime:** mean/p50/p95, throughput, build/membership/forward costs, memory
  and environment. Reuse rate alone does not establish saving.

Denominators: full corpus 17,453 windows, including 5,053 test windows; pure
K=6 corpus 10,017, including 2,874 test windows. The 13,866 windows with some
label are not identical to pure K=6 task scope. Select policy on validation,
then evaluate fixed held-out test recordings; do not pool all splits as test.
Reference-decision agreement is defined on unlabeled/transition windows even
though K=6 task correctness is not established there.

Existing audit coverage and G2 metrics have already been inspected on the test
subjects. Do not describe this test split as untouched by the whole research
workflow; disclose exploratory use and any subsequent method-selection impact.

Separate two comparisons: fix model+domain and vary bound method; fix
model+bound method and vary global/diagonal/correlated domain. Specify what is
matched (coverage, budget or a defensible region-size measure) and report it.
Tiny regions may certify while covering no useful queries. PCA's adjacent-pair
coverage advantage alone passes neither G3 nor G4/G5.

## 6. Current evidence boundary

Evidence establishes data feasibility with revisions and a frozen backbone.
The domains, support identities and sufficient proposition specify a proposed
verifier; they do not close G3. Minimal verifier, conservative numerical
membership, reuse gate, lifecycle and instrumentation remain to implement and
validate. Event selection/structured gains and net savings remain open. Broader
novelty and net saving stay UNVERIFIED in [02_LITERATURE.md](02_LITERATURE.md).
