# 07 G3 Domain Contract — Minimal Diagonal Decision-Reuse Prototype

> Updated 03/11/2026 · Contract for the frozen K=12 actual graph.
> Executable evidence and remaining numerical limitations are recorded in §9.
> G2 remains HOLD. A diagnostic positive bound never overrides an unresolved
> executable numerical guarantee. Mathematical owner: [01_04_THEORY.md](01_04_THEORY.md).

## 1. Goal, scope and completion boundary

G3 must connect valid class-pair bounds to conservative query membership and
a trusted-anchor reuse gate for the frozen K=12 model. The first target is a
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
283,692-parameter model contains BN/MaxPool; actual graph integration must be
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

For anchor winner `a`, form all eleven class-pair specifications

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

Store domain/anchor/model/preprocessing identities, winner `a`, all eleven safe
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
Add shadow-reference checks on the fixed-model data path. These are proof obligations; actual execution receipts must identify which checks ran and which remain unresolved.

## 6. Reuse gate and trusted-anchor lifecycle

Required per-query gate (implementation evidence must name the matching version):

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
uses only the 3,572 supervised K=12 test windows. Historical pure-K=6 test
count 2,874 is a different evaluation population. Do not pool corpus train/val/test as test.

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

## 9. Exact identity and evidence boundary

The reference graph is Conv6→32 k5 p2, BN(eval), ReLU, Pool2; Conv32→64
k3 p1, BN(eval), ReLU, Pool2; Conv64→64 k3 p1, BN(eval), ReLU;
Flatten2048→Linear128→ReLU→Linear12. Freeze all running means/variances,
BN epsilon, weights/biases, pooling phase and flatten order.

- Checkpoint: `output/week1_closeout_20261102/reference/selected_best.pt`.
- SHA-256: `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
- Preprocessing: `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.
- Parameter count: 283,692; ordered class list from frozen configuration,
  explicitly 12 logits and eleven distinct competing margins.
- Week 3 receipts: `output/week3_certification_20261103/`; canonical reports
  separate mathematical, executable, empirical and cost findings.

The historical `10_script_11_gate_a_prototype.py` and
`11_p0_verifier_probe.json` target K=6. They remain historical artifacts.
They do not establish current operator support or numerical soundness.
The new verifier must prove or conservatively enclose every operation listed
above; unsupported operators, nonfinite intermediates or insufficient
arithmetic guarantees fail closed. Float64 evaluation or a last-step
`nextafter` alone does not bound all accumulated float32 errors.

Minimal normalized boxes have 768 coordinates. Fixed-overlap wave boxes have
384 free coordinates and an immutable prefix tied to prior normalized input;
verify the prefix unless trusted acquisition state establishes it exactly.
Changing that prefix is a new domain and invalidates the old bound identity.
No automatic recentering after reuse is permitted.

A candidate can pass a real-arithmetic/diagnostic margin calculation while
remaining ineligible for executable certified reuse. Report both fields:
`mathematical_bound_positive` and `executable_certificate_valid`. Formal runtime
acceptance requires the latter and validated membership, all identities and
lifecycle state. Report zero accepted reuse as zero, and reused-query agreement
as N/A when its denominator is zero. Negative coverage/cost findings are valid.

The required closure remains: valid actual-graph bound; conservative executable
normalization/membership; strictly positive all-eleven safe margins; trusted
anchor, identity/invalidation/reset/fallback; executed negative and shadow tests;
complete trace accounting. A checklist or test count is not proof that the
universal numerical obligation is resolved. G3 must stay conditional/pending
when any required premise is missing, regardless of successful diagnostics.

The implementation is `ecdr/week3_verifier.py`; `ActualGraphIBP` explicitly
handles the frozen operators and eleven direct-margin objectives. Its finite
float64 bounds are conditional estimates, with no directed-rounding/error
bridge to the executable float32 reference. `ConditionalCertificate` therefore
cannot enable formal reuse. `RuntimeGate` executes identity/membership/lifecycle
checks and reference fallback. Executed negative checks are in
`tests/test_week3_verifier.py`, with the 93-test full-suite receipt at
`output/week3_certification_20261103/regression_tests.log` (27 tests target the
actual K=12 verifier, 60 are historical, 6 are mathematical counterexamples).
The completed test-stream run accepts zero reuse, so G3 remains PARTIAL and
its numerical obligation unresolved. Candidate/build-positive counts are
reported independently in `domain_results.json`; they are not certificates.
