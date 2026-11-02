# 07 G3 Domain Contract — Minimal Diagonal Decision-Reuse Prototype

> 22/10/2026 · **Proposed contract, documentation only; no verifier or runtime
> implemented. G3 pending.** This file specifies the next prototype. It does
> not change model weights, dataset/label policy, saved results or G4/G5 status.
> Mathematical formulation: [04_THEORY.md](04_THEORY.md).

## 1. Goal, scope and completion boundary

G3 must connect valid class-pair bounds to conservative query membership and
a trusted-anchor reuse gate for the frozen K=6 model. The first target is a
**normalized diagonal candidate box**; global boxes are its isotropic special
case. PCA and semantic event selectors remain later scope, not prerequisites
for the first diagonal certificate.

Required outputs: domain record, minimal verifier, membership/reuse gate,
anchor lifecycle, instrumentation and validation evidence. A positive runtime
result is not a G3 requirement: G4 evaluates structure and G5 evaluates net
saving including builds/rebuilds. Creating this contract does not implement
or validate these outputs.

## 2. Canonical input and frozen reference

| Object | Contract |
|---|---|
| Raw window `x` | `6×128`, source channel/time order, no cross-recording windows; record flatten order |
| Normalization | `z=N(x)=A(x-m)`, `A=diag(1/sigma)`, positive train-fit scales repeated over time |
| Reference `F` | Frozen `SmallCNN1D` composed with frozen normalization: `F=f_theta o N` |
| Executable reference | Matching weights, BN running state, evaluation mode, preprocessing, class order, dtype/device and operator semantics |
| Trusted anchor | Raw `x^r` actually run through reference; store `z^r`, logits and unique winning class `a` |
| Event descriptor | Optional information used to select a region/template; not an activity oracle or proof |

Source authority: `scripts/backbone.py` (`SmallCNN1D`, `apply_norm`), frozen
`configs/`, checkpoint weights and normalization statistics. The current
282,918-parameter model contains BN/MaxPool; actual graph integration must be
validated. A library name does not establish operator support or valid bounds.

Real-valued equations do not model float32 rounding by themselves. Numerical
semantics and containment/error allowances belong to the guarantee (§5).

## 3. Domain record and membership

For coordinate-aligned `b_e ∈ R^768`, all entries nonnegative, define

    Z_e(z^r) = {z^r + diag(b_e) u : ||u||_infinity <= 1},
    S_e(x^r) = {x : N(x) ∈ Z_e(z^r)}.

Mathematical membership is `|N(x)-z^r| <= b_e` coordinatewise. Raw equivalent
budgets are `sigma*b_e`; raw train-fit quantiles transform to normalized
budgets by dividing by `sigma`. These are real-arithmetic equivalences; the
executable checker must establish conservative containment in its arithmetic.

`b_e=rho*1` gives normalized L-infinity radius `rho`; `b_e=1` gives radius 1.
Zero budgets are allowed as degenerate coordinates with conservative boundary
handling. Reject negative/non-finite budgets and non-finite queries.

An immutable domain record must bind:

- Domain/template ID and version, coordinate space/order, anchor ID, complete
  budgets and dtype, construction method/version.
- Checkpoint/graph identity, class order and normalization identity (`m,sigma`).
- Any containing relaxation domain, containment justification, arithmetic
  assumptions and membership error/rounding policy.
- Budget provenance and fit scope: fit on train, select levels/policies on
  validation, freeze before held-out test evaluation.

Empirical quantiles may select candidates. They do not certify decisions or
future population coverage. Valid bounds over such candidates can still yield
sound decision certificates. A selector cannot silently expand/recenter a
certified region; changes need their own matching bound/domain identity.

## 4. Minimal verifier and certificate record

For anchor winner `a`, form all five class-pair specifications

    h_aj(z) = f_theta,a(z) - f_theta,j(z),   j != a.

Produce valid lower bounds over the **whole `Z_e`** or a documented containing
domain. If a valid affine relaxation gives `h_aj(z) >= p_aj^T z+k_aj`, then

    ell_aj = p_aj^T z^r + k_aj - sum_i b_e,i |p_aj,i|.

Direct valid bound outputs may be used without exposing affine coefficients.
State the algorithm, operator support and assumptions justifying validity.
Supported IBP/CROWN machinery is a proposed first integration; no speed or
tightness advantage on this graph is presumed before testing.

A candidate receives a certificate only if every validated safe lower margin
is strictly positive. This rule is sufficient, not necessary for invariance;
failure to certify does not prove a decision change. Zero margins, ties,
unsupported operators, bound failures and uncertain numerical cases cause
construction failure with reason, not successful certification.

Store domain/anchor/model/preprocessing identities, winner `a`, all five safe
lower margins, verifier configuration/version, build status/time and numerical
guarantee. Do not apply bounds from a smaller domain to a larger candidate.

## 5. Numerical proof obligation and validation evidence

Distinguish three levels in every result:

1. **Mathematical implication:** valid bounds over a declared domain plus
   membership and positive margins imply the ideal model's decision.
2. **Executable numerical soundness:** actual floating-point bound/membership
   code conservatively establishes those conditions, and the property applies
   to the chosen executable reference. Justify supported guarantees or validated
   rounding/error enclosures. If normalization error is handled by input
   containment, its enclosure must fit the certified normalized domain.
   Alternatively state and justify the other numerical model used.
3. **Empirical consistency:** a shadow reference agrees on tested accepted
   queries. Report counts and scope; zero observed mismatches cannot replace
   either proof obligation.

An arbitrary comparison epsilon or a small positive margin is not a numerical
proof. If level 2 remains unresolved, label soundness conditional on declared
arithmetic assumptions; do not claim end-to-end formal soundness or close G3's
numerical obligation. Non-finite intermediate values and ambiguous boundaries
must conservatively fail/fallback.

Future validation must include an independently checkable affine/toy case,
class-pair specification checks, zero/small budgets, out-of-domain queries,
boundaries, stale identities, ties/non-finite inputs and unsupported operators.
Add shadow-reference checks on the fixed-model data path. These are required
future checks, not tests performed in this documentation revision.

## 6. Reuse gate and trusted-anchor lifecycle

Proposed per-query gate:

1. Obtain a descriptor/template choice if the policy needs it.
2. Require a successful certificate with compatible anchor/model/preprocessing/
   domain/numerical identities and valid positive margins.
3. Establish conservative membership of the query in that domain.
4. Reuse stored class `a` only when all checks pass. Otherwise run reference
   inference and return its result.
5. Only completed reference inference creates a trusted anchor in this minimal
   policy. A reused query cannot refresh it. Building a certificate is a
   separate policy decision and cost.

Reset at recording boundaries; never compare across recordings. Model,
normalization, class order, region-parameter or relevant numerical-semantics
changes invalidate incompatible certificates. Do not compose local regions
through unverified reused anchors.

Descriptor mistakes can affect coverage/cost without breaking the guarantee
if identities, bounds and membership remain valid. Test ground-truth activity
labels are for offline analysis, not online gating.

## 7. Instrumentation and evaluation

The proposed query trace must record split, recording/window, anchor and
certificate/domain versions, selected template, accepted/fallback reason,
anchor age, membership result, minimum safe margin, reference-call flag,
build/rebuild flag, returned class and optional shadow-reference class.
Measure descriptor, lookup/membership, forward, certificate build/rebuild and
total durations separately, plus cache memory and environment details.

Report all-query/reused-query denominators, builds per query and mismatches.
With no reused queries the empirical reused-query agreement estimate is
undefined; do not report measured 100% agreement from an empty denominator.

Choose policy on validation recordings, then use fixed held-out test streams
chronologically with per-recording reset. Full test stream: 5,053 windows.
Reference agreement is definable without activity ground truth. Accuracy/F1
must retain a declared K=6 ground-truth scope (existing pure K=6 test count
2,874). Do not pool corpus train/val/test as test.

Test coverage and model metrics have already been inspected in earlier stages.
Future fixed-policy evaluation must disclose that exploratory use and any
selection informed by it; do not claim a wholly untouched confirmatory test.

Shadow-reference calls are validation overhead: account for them separately
from operational reuse timing. Include every actual deployment component in
the operational measurement. Saved `g2_cpu_latency.json` times forward on
already-normalized RAM tensors, not the fallback path or net savings.

**G3 records component costs; G5 evaluates end-to-end savings.** For G5 compare
full inference and reuse on the same preprocessing/device/workload and horizon.
Include initial construction in counted build costs or add `T_init/N` over that
horizon. Include rebuild, report mean/p50/p95, and retain negative results.
The simplified `T_e+T_m+(1-r)T_f+qT_b` model assumes roughly constant forward
cost; conditional/per-query measurements govern when that assumption fails.

## 8. PCA and coordinate-residual variants remain later scope

The raw audit uses `w=x-x^r-mu`, score box `|U^T w|<=b` and orthogonal residual
L2 `||(I-UU^T)w||_2<=epsilon`. It is distinct from the prior draft's unrestricted
coordinate-residual box. See theory §2.2–2.3 and §3.2 for their definitions and
different support functions.

Do not import PCA membership coverage as verifier evidence. Any correlated
implementation must bind centering, basis, residual norm/orthogonality,
coordinate transform and sound membership to the correct relaxation domain.
The first diagonal certificate does not certify these other candidates.

## 9. Current completion checklist

- [x] Proposed domain, roles, records and evidence requirements documented.
- [ ] Model/domain/normalization identity loaded and validated in a prototype.
- [ ] Minimal class-pair verifier integrated and its bounds justified.
- [ ] Conservative numerical membership and acceptance implemented/validated.
- [ ] Reuse gate, invalidation and trusted-anchor lifecycle implemented.
- [ ] Instrumentation, negative cases and shadow-reference evidence recorded.
- [ ] Mathematical and executable numerical soundness obligations resolved.

G3 remains **PENDING**. No retraining, new audit, certificate construction,
runtime benchmark or experiment was performed in this documentation update.
