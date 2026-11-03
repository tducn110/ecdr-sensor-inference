# Week 3 independent mathematical and methodology review

Reviewer role: independent from verifier implementation and report authoring.
Date: 2026-11-03 (Asia/Ho_Chi_Minh). Scope: actual frozen K=12 SmallCNN1D,
the 24 original hypothesis meanings, numerical claim boundaries, and experiment
methodology. This receipt records executed review, not user acceptance.

## Authority and frozen boundary

Inspected the complete execution request attached by the user, the original
H01–H24 ledger in `03_MATH/03_Week_3_Mathematical_Foundations.tex`, the existing
mathematical reports/supplements, `scripts/backbone.py`, the Week 1/2 pipeline
and evidence, and the historical `ecdr/` implementation. The live frozen
reference is `SmallCNN1D(num_classes=12)`, with padded Conv–BN(eval)–ReLU,
two MaxPool1d(2), and Flatten–MLP. The historical valid/ReLU/GAP implementation
is a different classifier and is not evidence of current-reference soundness.

Selected checkpoint identity specified and reviewed:
`8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
Frozen normalization ID:
`aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.
The reviewer did not mutate Week 1/2 sources, weights, normalization, or data.

## Mathematical decisions and corrections

1. **H05:** DERIVED in real arithmetic. For every one of the eleven distinct
   competitors, a valid positive lower bound over the entire domain implies a
   unique winner throughout the domain. A point margin, random samples, or
   source-code presence does not establish that premise.
2. **H04/H06:** DERIVED under continuity/strict-margin or a valid Lipschitz
   upper bound on the complete relevant region. A local gradient is not such an
   upper bound. No useful radius or hit rate follows from existence alone.
3. **H07/H08:** IBP is sound under exact enclosure operations, but incomplete.
   On the same final hidden interval, direct final-head class-difference bounds
   are at least as tight as subtracting separate logit intervals: minimizing a
   difference jointly cannot be less than minimizing the first term and
   subtracting an independently maximized second term. Float64 implementation
   still needs a numerical enclosure argument.
4. **H09:** The original universal cheapness claim remains NOT ESTABLISHED.
   The old supplement's universal CROWN≥IBP assertion also needs qualification.
   For `ReLU(x)`, `x in [-1,1]`, IBP lower bound is 0; the valid CROWN lower
   relaxation `y>=x` has minimum -1. A specified optimized/combined method may
   dominate an alternative, but arbitrary valid lower relaxations need not.
5. **H13:** The proposed feature-index mapping is correct, but its candidate
   receptive-field endpoints `[4j-9,4j+10]` are not. With zero-based indices,
   the exact structural dependency intervals are:

   | Output stage | Raw-input dependency interval |
   |---|---|
   | Conv1 at i | `[i-2,i+2]` |
   | Pool1 at i | `[2i-2,2i+3]` |
   | Conv2 at i | `[2i-4,2i+5]` |
   | Pool2 at i | `[4i-4,4i+7]` |
   | Conv3 / final ReLU at j | `[4j-8,4j+11]` |

   Width is 20 and jump is 4. Containment in the next window's fixed prefix
   `[0,63]` is exactly `2<=j<=13`. Translation by 64 input samples corresponds
   to 16 final positions. Every contributing intermediate convolution is
   interior; pool phase matches because 64 is divisible by both pool strides.
   BN(eval) is the same pointwise channel affine map with fixed running state;
   ReLU and MaxPool preserve equality of their translated inputs. Induction
   over these operators proves, in real arithmetic,
   `H_j(X_next)=H_(j+16)(X_previous)` for those twelve positions. This proof
   requires exact shift-overlap, fixed preprocessing, frozen BN running state,
   evaluation mode, the declared padding/pool semantics and identical graph.
   It does not prove the flattened-head decision unchanged.

   **Correction record:** an initial reviewer message provisionally accepted
   the wrong RF endpoints and suggested j=3..13. Re-deriving every pool phase
   and executing the actual graph corrected that suggestion before adoption:
   the correct endpoints are `[4j-8,4j+11]` and j=2..13. This receipt supersedes
   that provisional message.
6. **H14:** The original GAP new-minus-dropped identity remains a valid
   historical theorem under its stated architecture. Flatten–MLP is position
   sensitive; the GAP identity does not transfer.
7. **H15/H16/H17:** Recursively recentering after a reused query is unsound
   without a separate transfer proof. Fixed-anchor membership and independently
   certified finite-horizon reachable sets are valid sufficient schemes with
   explicit domain identities. None guarantees a useful multi-step lifetime.
8. **H18:** The Theta(m) lower bound concerns worst-case exact membership with
   coordinate access and nontrivial coordinate constraints, not every practical
   vectorized runtime or average-case early rejection.
9. **H19/H20/H24:** Soundness, coverage and net savings are independent proof
   or measurement obligations. For constant modeled costs, comparing
   `N*T_f` with `N*(T_v+T_m)+(N-R)*T_f+B*T_b` gives
   `r*T_f > T_v+T_m+(B/N)*T_b`. This assumes comparable query costs and a
   horizon that includes construction. Actual complete trace timing governs.
10. **H22:** Neither float64 evaluation nor one final `nextafter` controls all
    accumulated errors of float32 reference execution. Exact stored float32
    weights are the defined constants; uncertainty about an unstored ideal
    weight is a different problem. Rounding of normalization, BN folding,
    reduction, margin differences, and membership needs explicit treatment.
    If unavailable, runtime formal reuse must remain disabled.

## Original hypothesis traceability

Every ID below retains its original meaning; the actual-graph extension of
H13 is additional evidence, not a silent redefinition of the old theorem.

| ID | Original object | Independent disposition |
|---|---|---|
| H01 | Shift-aligned overlap equality | DERIVED from window contract; actual data reconstruction evidence applies |
| H02 | Overlap implies small ordinary distance | FALSIFIED as an unconditional claim |
| H03 | Same activity implies same CNN prediction | FALSIFIED as an unconditional claim; annotation does not constrain classifier |
| H04 | Strict winner + continuity gives local invariant region | DERIVED existence; useful size NOT ESTABLISHED |
| H05 | Positive whole-domain margin bounds imply constant decision | DERIVED with valid bounds and matching membership/identity |
| H06 | Lipschitz-margin certified ball | CONDITIONAL on a sound regional Lipschitz upper bound |
| H07 | IBP always certifies invariant domains | FALSIFIED; soundness does not imply completeness |
| H08 | Direct pairwise margins can avoid logit dependency loss | DERIVED under matched hidden intervals; implementation arithmetic conditional |
| H09 | CROWN universally cheaper than alternatives | NOT ESTABLISHED; method/version/graph-specific timing needed |
| H10 | Temporal/PCA directions match margin geometry | Empirical; train-fit coordinates and held-out gradient alignment required |
| H11 | Structured region beats a matched diagonal box | Empirical; criterion must be specified before test and geometry matched |
| H12 | Fixed overlap reduces free coordinates CW to Cs | DERIVED: 768 to 384; not 50 percent runtime saving |
| H13 | Valid–ReLU exact aligned feature reuse | DERIVED historical theorem; separately DERIVED actual-graph twelve-position relation |
| H14 | GAP new-minus-dropped feature sum | DERIVED only for declared GAP head; not current Flatten–MLP |
| H15 | Recenter after every hit without new proof | FALSIFIED by certificate drift |
| H16 | Fixed-anchor membership supports repeated reuse | CONDITIONAL on valid compatible certificate and membership each time |
| H17 | Product-wave/reachable-set multi-step certificate | CONDITIONAL on whole-horizon reachable-set bounds |
| H18 | Exact box membership worst-case Theta(m) | DERIVED in coordinate-access model |
| H19 | Soundness guarantees useful coverage | FALSIFIED by no-future-query certified region |
| H20 | Useful coverage guarantees speedup | FALSIFIED by build/overhead costs |
| H21 | ECDR beats CNN/LipCache on HAPT | Measurement required; LipCache comparator semantic blocker documented below |
| H22 | Real proof automatically transfers to float32 | FALSIFIED as an unconditional claim |
| H23 | Correct event name suffices for reuse | FALSIFIED; valid certificate and membership still required |
| H24 | Sound, useful, cost-saving regime exists | NOT ESTABLISHED until all three obligations and numerical bridge hold |

## Independently executed actual-graph H13 check

Command: `.venv/bin/python /tmp/week3_h13_independent.py` (exit 0). The small
read-only script loaded the frozen checkpoint, explicitly instantiated K=12,
reused checkpoint float32 mean/std, and evaluated `model.conv` batch one on
every neighboring test pair whose recording matched and overlap was exact.
The script was temporary; the canonical experiment driver should contain the
reproducible deliverable implementation of this check.

Result: 5,034 within-recording pairs (5,053 windows minus 19 recording starts).
All 64 channels at final positions j=2..13 were **bitwise equal** to the
previous window at j+16 across all pairs, maximum absolute error 0.0. Maximum
absolute differences at j=0,1,14,15 were respectively
14.924388885498047, 7.022495269775391, 7.839027404785156,
18.572065353393555. This is executed evidence for the specified CPU runtime,
not universal bitwise proof across backends or reduction strategies.

## Independent methodology requirements

- Full stream operational and reference agreement denominator is 5,053;
  supervised activity accuracy/Macro-F1 denominator is 3,572. Pair-level
  adjacent geometry has denominator 5,034, which is a different quantity.
- Sort by recording/window and reset at every recording boundary. Test labels
  may define offline strata; runtime event selection cannot use them as oracle.
- Fit candidate budgets/directions on training data, choose a deterministic
  declared policy on validation, freeze before test. Earlier observations of
  test data must be disclosed rather than calling this wholly untouched.
- H10 must express temporal/PCA direction and margin normals in the same
  normalized coordinate system, specify the competing class, and distinguish
  alignment from usefulness: smaller normal projection means more locally
  tangential motion; local derivatives are not regional certificates.
- H11: matching quantile, dimension, volume, radius or candidate coverage are
  different comparisons. A fixed-prefix wave region is already a degenerate
  diagonal box; IBP sees only coordinate ranges. A smaller wave subset can
  improve bounds without proving an intrinsic structured-geometry advantage.
- Keep attempted builds, positive numeric diagnostic builds, mathematical
  premises, formal runtime certificates, candidate membership, potential hits,
  and accepted runtime reuse separate. With zero accepted reuse, reuse-query
  agreement is N/A, not 100 percent.
- Include every build (including failures), candidate computation, membership,
  preprocessing, fallback inference, state and relevant transfer in latency.
  Shadow reference is validation overhead and must be reported separately from
  the proposed operational path. Interleaved or sequential trials on this host
  are observations, not a broad hardware performance claim.

## Primary-source checks

The following primary sources were opened directly during review:

- [PyTorch 2.14 BatchNorm1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html):
  evaluation uses running state only when tracking is enabled; otherwise input
  batch statistics still apply. This is why eval alone is insufficient.
- [PyTorch 2.14 MaxPool1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool1d.html):
  stride/kernel phase and negative-infinity padding define exact pool semantics.
- [PyTorch 2.14 numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html):
  floating operation ordering/backend/batching can change numeric outputs;
  mathematical equality does not establish universal bitwise equivalence.
- [auto-LiRPA pooling source](https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/auto_LiRPA/operators/pooling.py):
  current public BoundMaxPool is expressed with two-dimensional pooling;
  source presence alone does not establish actual Conv1d graph integration.
- [LipCache v1](https://arxiv.org/html/2608.13144v1), section V-C Remark V.2,
  VI-E and VI-H: the certificate concerns GuardNet consistency; MainNet
  agreement is empirical, proxy-label output is the default, and temporal
  streaming is separately identified future scope. A strict frozen-reference
  MainNet equivalence comparator cannot simply inherit that certificate. No
  LipCache-on-HAPT superiority claim follows without a fair implementation.

## Implementation and final-artifact review

Inspected `ecdr/week3_verifier.py`, `tests/test_week3_verifier.py`,
`scripts/week3_experiments.py`, and `scripts/week3_lirpa_probe.py` directly.
The supported conditional operator transfers cover signed/padded Conv1d,
fixed-state sign-aware BN, monotone ReLU and MaxPool1d, Flatten and Linear.
The final head uses all eleven class-difference rows. The code does not call
the computed float64 margins validated numerical enclosures. Domain endpoints
are immutable tuples and define the candidate itself; no unproved epsilon is
used to enlarge membership acceptance. Membership compares finite normalized
query coordinates with these endpoints. The numerical gate returns `False`
unconditionally, including for forged positive certificate fields.

Independently executed `.venv/bin/python -m unittest tests.test_week3_verifier -v`:
**26 tests, 0.199 s, exit 0** at the reviewed intermediate version. These cover
operator mistakes, exact identity, zero/tiny/tied margin paths, finite/dtype
guards, overflow preserving state, stale/expired/recording resets, immutable
domains, failed builds and forged numerical status. Sample-containment and
allclose tests are implementation consistency evidence, not a numerical proof.

**A concrete identity defect was found and sent for correction.** A registered
forward hook on the final Linear could flip the logits while `assert_identity`
still passed: a zero raw window changed winner from zero-based class 5 to 6.
This demonstrated that parameter bytes and module `repr` alone did not bind
execution semantics. The verifier author subsequently added rejection of
local/global forward/pre hooks and per-module forward/call overrides, hashes
forward code, and reported a 27-test rerun. The reviewer will verify that final
version and update this receipt; the intermediate bug is not hidden.

The experiment source selects radius from a declared validation grid before
loading test data, uses training-only PCA geometry, records same-center wave
controls and explicitly limits H11 to a domain ablation with unequal volume.
It distinguishes conditional builds, candidate members, accepted reuse and
shadow consistency. The runtime keeps an original trusted conditional anchor
until expiry/reset; every query still calls the actual reference.

Two cost qualifications were raised and acknowledged:

- The runtime currently runs reference inference before a build and the build
  runs anchor inference again. This duplicate call is charged but is a prototype
  implementation cost, not a theorem that every ECDR design has that cost.
- The paired baseline includes per-query complete state/graph identity hashing.
  It is a frozen-wrapper baseline, not a raw forward-only CNN benchmark. A
  separate normalization-plus-CNN core measurement is needed to expose that
  distinction. Negative observed speedup concerns the measured policies only.

Independent actual H03 evidence was also extracted from the immutable saved
frozen predictions, checked against the chronological global row identities:
there are **78 prediction-disagreement pairs among 2,661 same-pure-activity
pairs** in the 5,034 within-recording test pairs. One witness is global rows
682 and 683, pure class label 11, with predictions 10 and 8 (zero based).
This is a direct current-reference witness, separate from the universal toy
counterexample and separate from ground-truth model accuracy.

Reproducible read-only H13/H03 check from the repository root (the first
independent H13 execution used the same graph loop in the temporary script
named above; the H03 calculation was executed separately as a heredoc):

```bash
.venv/bin/python - <<'PY'
import json
import numpy as np
import torch
from scripts.backbone import SmallCNN1D
from ecdr.data import load_full_stream

torch.set_num_threads(1)
path = 'output/week1_closeout_20261102/reference/selected_best.pt'
ck = torch.load(path, map_location='cpu', weights_only=False)
model = SmallCNN1D(num_classes=12, cfg=ck['model_cfg'])
model.load_state_dict(ck['state_dict']); model.eval()
raw, df = load_full_stream('test')
z = ((raw - ck['norm_mean'][None, :, None]) /
     ck['norm_std'][None, :, None]).astype(np.float32)
maximum = np.zeros(16); equal_pair_counts = np.zeros(16, dtype=int)
pairs = np.flatnonzero((df.exp.to_numpy()[1:] == df.exp.to_numpy()[:-1]) &
    (df.win_idx.to_numpy()[1:] == df.win_idx.to_numpy()[:-1] + 1))
with torch.inference_mode():
    for i in pairs:
        assert np.array_equal(z[i, :, 64:], z[i+1, :, :64])
        old = model.conv(torch.from_numpy(z[i:i+1])).numpy()[0]
        new = model.conv(torch.from_numpy(z[i+1:i+2])).numpy()[0]
        errors = np.max(np.abs(new[:, :16] - old[:, 16:]), axis=0)
        maximum = np.maximum(maximum, errors)
        equal_pair_counts += np.all(new[:, :16] == old[:, 16:], axis=0)
saved = np.load('output/week1_closeout_20261102/reference/test_predictions.npz')
assert np.array_equal(saved['global_row_id'], df.global_row_id.to_numpy())
left, right = df.iloc[pairs], df.iloc[pairs+1]
same = ((left.label_policy.to_numpy() == 'pure') &
        (right.label_policy.to_numpy() == 'pure') &
        (left.label_majority.to_numpy() == right.label_majority.to_numpy()))
disagree = saved['predictions'][pairs] != saved['predictions'][pairs+1]
print(json.dumps({'pairs': len(pairs), 'max_abs_error': maximum.tolist(),
    'equal_pair_counts': equal_pair_counts.tolist(),
    'same_pure_activity_pairs': int(same.sum()),
    'same_pure_activity_prediction_disagreement_pairs': int((same & disagree).sum())}, indent=2))
PY
```

For H07 the direct-head-safe counterexample is `t in [-1,1]`,
`h1=h2=ReLU(t+1)`, `F0=1+h1-h2`, `F1=0`. The true margin is exactly 1,
whereas hidden intervals `[0,2]` give direct final-head IBP lower bound -1.
This refutes completeness even after the implementation's direct class-row
cancellation improvement. The simpler logits `t+1,t` only demonstrate
separate-logit dependency loss, so it should not be the sole evidence for the
stronger claim about the implemented method.

## Final executed review result

After the hook correction and completion of the benchmark, independently ran
the final 27-test module: **27 tests, 0.197 s, exit 0**. Re-executed the
specific output-changing-hook probe and confirmed `IdentityError` rejection.
These tests executed the new verifier; the following receipt checks separately
recomputed consistency from the original CSVs instead of treating JSON `PASS`
fields as execution evidence.

An independent Python stdlib check read `experiment_results.json`,
`protocol.json`, `domain_results.json`, `runtime_results.json`,
`domain_trace.csv` and `runtime_trace.csv` and asserted:

- 5,053 full windows, 3,572 supervised windows, 5,034 adjacent pairs, 19
  recordings, complete stratum count 5,053; selected validation radius 1e-6.
- 10,106 domain rows (two families times 5,053) and 20,212 runtime rows (two
  families times two repeats times 5,053), unique global row IDs and contiguous
  within-recording window indices per runtime group.
- Direct row recounts equal the summary positive/member/joint/disagreement and
  build/fallback reason counts. Runtime cost sums and means agree with CSV
  durations; all 32 p50/p95 values for runtime/reference/membership/build
  components were independently recomputed using sorted linear interpolation.
  Wrapper/core speed ratios agree with their recorded totals, and
  all four measured savings are negative.
- All six source/artifact SHA-256 entries in `experiment_results.json` match
  current bytes. Important source identities:
  `scripts/week3_experiments.py` =
  `a646e96d7d039cf6f4c20553e665b40a95663a834f34256823166bfe38384fab`;
  `ecdr/week3_verifier.py` =
  `5f86aa2863406b01071e6b49ce5badd28322f1ca6e004b425568f0ad4cab09eb`.
- 31 critical frozen Week 1 reference artifact and Week 1/2 pipeline/source
  hashes match the initial preservation manifest. This is the reviewer's
  bounded preservation check, distinct from any broader root-agent inventory.

Observed final conditional-domain results:

| Family | Builds | Positive numeric diagnostics | Next candidate members | Positive-and-member | Formal reused queries |
|---|---:|---:|---:|---:|---:|
| Diagonal box | 5,053 | 5,034 | 0 | 0 | 0 |
| Fixed-overlap wave | 5,053 | 4,211 | 0 | 0 | 0 |

Observed complete runtime measurements (ms per query; includes failed builds
and fallback; data already resident in RAM):

| Repeat | Family | Runtime builds | Frozen wrapper mean | CNN core mean | ECDR mean | Wrapper/ECDR ratio | Core/ECDR ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Diagonal box | 49 | 1.674681 | 0.224016 | 3.174133 | 0.527603 | 0.070575 |
| 0 | Wave | 69 | 1.674681 | 0.224016 | 3.197501 | 0.523747 | 0.070060 |
| 1 | Diagonal box | 49 | 1.588503 | 0.214741 | 3.106901 | 0.511282 | 0.069117 |
| 1 | Wave | 69 | 1.588503 | 0.214741 | 3.168714 | 0.501308 | 0.067769 |

All four runs have **0 accepted reuse, 5,053 fallback queries, 0 full-stream
reference disagreements, and reused-query agreement N/A**. Zero disagreement
here confirms the fallback execution comparison; it is not reused-query or
formal-certification success. Formal/runtime numerical soundness remains
NOT ESTABLISHED. These measured policies are slower than both declared
baselines. The result does not disprove existence of a different useful
optimized operating regime. G2 remains HOLD; G3's numerical obligation stays
open; H21 LipCache and H24 remain NOT ESTABLISHED.

Final static proof review found and requested correction of the earlier
supplement errors: CROWN universal dominance, real-versus-float32 errors,
reconstruction-versus-provenance, sine-distance lower/upper confusion,
unmeasured PCA dimensions, manifold/coverage assumptions and multi-step drift
using diameters. The revised sources include the strong direct-head H07
counterexample and actual H13 proof. Final static rereading also confirmed
correction of the two additional norm-explanation details sent to the author:
closed ball is distinguished from its sphere boundary; pointwise norm
inequalities induce inequalities of matching quantiles on the same population,
but ratios of marginal medians are not per-pair ratios and do not establish
correlation. The volume-shell probability is explicitly restricted to uniform
volume sampling, rather than arbitrary temporal sensor distributions. The
float-precision supplement now distinguishes reconstruction consistency from
unique provenance and labels historical artifact/count references.

The reviewer has not performed final PDF visual inspection; that belongs to
the separate document QA evidence. This review establishes bounded
mathematical/implementation/methodology checks and preserves unresolved proof
obligations. It is not user ACCEPTED status or an executable formal certificate.
