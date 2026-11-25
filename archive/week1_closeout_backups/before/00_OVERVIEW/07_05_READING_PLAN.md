# 05 READING PLAN — literature expansion, baselines, metrics

> Documentation sync: 2026-10-22. G1/G2 measurements remain dated 2026-10-15; no experiment rerun.
> Status: reading/experiment PLAN. Metadata inspection and bibliography inclusion
> do not establish full-paper reading or support a method/proof/benchmark claim.
> Claims need source evidence and a locator in `02_LITERATURE.md` under the
> [review protocol](06_REVIEW_PROTOCOL.md). C11 (novelty) stays UNVERIFIED and
> C12 (net saving) stays UNVERIFIED until the relevant literature and G4/G5 evidence.
> Reading notes live in `notes/P<ID>_<ShortName>.md` (template §7).

## 0. Reading status at the documentation sync

- **P04 NoScope: SOURCE_CHECKED_SCOPED**, per
  [notes/P04_NoScope.md](notes/P04_NoScope.md). Selected sections/claims have
  locators; this is not full-paper reading or reproduction.
- **P01–P03 and P05–P23: PENDING_FULL_READ**. Existing ledger excerpts for P01/P02,
  selected passages, official records/API pages and verified metadata remain
  useful scoped evidence; they do not mark the full reading task complete.
- CROWN/auto_LiRPA paper records in `refs.bib` are bibliographic evidence, not
  completion of P08/P09 or a tested integration. Keep paper reading, API
  inspection, derivation and code validation as separate activities.
- The current report is [report.md](report.md), with editable
  [LaTeX source](latex/dossier.tex). Neither is a completed systematic review.
  Documentation reconciliation is recorded in [DOCS_SYNC_2026-10-22.md](DOCS_SYNC_2026-10-22.md).

## 1. New collision sources flagged by review (read in Week 1-2)

| ID | Source | Collision / use |
|---|---|---|
| P04 | NoScope (VLDB 2017) https://dawn.cs.stanford.edu/publications/noscope/ | prior art: temporal difference detectors + conditional inference; our distinction must be the mathematical class-pair certificate, not "temporal diff skips inference" |
| P16 | Overlapping vs non-overlapping windows HAR (Sensors 2019) https://www.mdpi.com/1424-8220/19/22/5026 | overlap benefit shrinks subject-independently; forbids using 50% overlap as similarity/performance evidence; shapes G2 eval protocol |
| P10 | Fast-Lin/Fast-Lip (ICML 2018) https://proceedings.mlr.press/v80/weng18a.html | certified baseline B4 — do not benchmark only against a weak global Lipschitz |
| P11 | RecurJac (AAAI 2019) https://cdn.aaai.org/ojs/4522/4522-13-7561-1-10-20190706.pdf | Jacobian bound baseline B4 |

Updated gap map (scope differences only — NOT a novelty proof):

| Study | Solved | Limitation vs our RQ |
|---|---|---|
| LipCache (2026) | certified cache reuse via GuardNet | no certificate over the full domain of the MainNet |
| STBP (2026) | structured spatio-temporal perturbation bounds | no streaming inference-reuse target |
| Wei & Liu (2021) | reuse verification computations online | accelerates verification, not skip-backbone-inference |
| NoScope (2017) | temporal difference + conditional inference in the scoped passages read | scoped evidence does not establish a formal per-input backbone-decision certificate |
| DeltaCNN (2022) | sparse frame-difference computation | no decision-reuse certificate |
| CROWN (2018) | linear-relaxation verification | no runtime reuse policy for streaming |

## 2. Reading list (23 items, 4 groups)

### Group A — novelty / prior-art collision (P0, feed 01+02)

| ID | Paper (official source) | What to extract | Pri |
|---|---|---|---|
| P01 | LipCache, arXiv:2608.13144 (2026) https://arxiv.org/abs/2608.13144 | certified cache region, GuardNet/MainNet split, cost + consistency guarantee | P0 |
| P02 | Hybrid Robustness Verification for Spatio-Temporal NN (STBP), arXiv:2606.09746 (2026) https://arxiv.org/abs/2606.09746 | structured perturbation sets, first-layer bounds, tightness, STBP-LJ caveat | P0 |
| P03 | Wei & Liu, arXiv:2106.12732 (2021) https://arxiv.org/abs/2106.12732 | incremental verification, reuse of previous computations, temporal dependencies | P0 |
| P04 | NoScope (VLDB 2017) https://dawn.cs.stanford.edu/publications/noscope/noscope-optimizing-neural-network-queries-over-video-scale | difference detector, conditional inference, cost model | P0 |
| P05 | DeltaCNN (CVPR 2022) https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html | sparse-difference computation cost | P1 |
| P06 | Skip RNN (ICLR 2018) https://research.google/pubs/skip-rnn-learning-to-skip-state-updates-in-recurrent-neural-networks-2/ | learned skipping policy, accuracy/compute trade-off | P1 |
| P07 | FrameExit (CVPR 2021) https://openaccess.thecvf.com/content/CVPR2021/html/Ghodrati_FrameExit_Conditional_Early_Exiting_for_Efficient_Video_Recognition_CVPR_2021_paper.html | conditional computation, gate design | P2 |

Do NOT collapse certified caching (P01), incremental verification (P03) and
non-certified conditional inference (P04-P07) into one category — they collide
with different parts of the claim.

### Group B — mathematical foundation / formal verification (for 04_THEORY + G3)

| ID | Paper | Focus | Pri |
|---|---|---|---|
| P08 | CROWN (NeurIPS 2018) https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html | affine lower/upper bounds, ReLU relaxation, robustness conditions | P0 |
| P09 | auto_LiRPA (NeurIPS 2020) https://papers.nips.cc/paper/2020/hash/0cbc5671ae26f67871cb914d81ef8fc1-Abstract.html | graph bound propagation, specification matrix C, implementation | P0 |
| P10 | Fast-Lin/Fast-Lip (ICML 2018) https://proceedings.mlr.press/v80/weng18a.html | certified baselines, local/global Lipschitz comparison | P0 |
| P11 | RecurJac (AAAI 2019) https://cdn.aaai.org/ojs/4522/4522-13-7561-1-10-20190706.pdf | Jacobian bounds, Lipschitz tightness | P1 |
| P12 | DeepPoly (POPL 2019) https://www.sri.inf.ethz.ch/publications/singh2019domain | abstract domain, precision vs cost | P1 |
| P13 | IBP (arXiv:1810.12715) https://arxiv.org/abs/1810.12715 | interval bounds, looseness | P1 |
| P14 | Fast-complete verifier (ICLR 2021) https://mlanthology.org/iclr/2021/xu2021iclr-fast/ | branch-and-bound, alpha-CROWN context | P1 |
| P15 | CROWN-IBP (arXiv:1906.06316) https://arxiv.org/abs/1906.06316 | certified training trade-offs | P2 |

Required derivation after P08+P09 (re-derive, do not copy): for a diagonal region
`x = x^r + δ`, `|δ_i| <= d_i`, a valid affine lower relaxation
`g_aj(x) >= α^T x + β` over the **entire region** gives

    g_aj_lower = alpha^T x^r + beta - sum_i d_i * |alpha_i|.

Here `a` is the trusted-anchor predicted class and `j != a` is a competing
class; `g_aj = F_a − F_j` for the fixed reference pipeline. The coordinate
system, normalization and numerical assumptions must follow
[07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md). Correlated-region support
functions, centering and residual norm need their own matching definition and
derivation; do not substitute the exploratory PCA membership check silently.
P08/P09 are mandatory before using auto_LiRPA as the G3 verifier; P10/P11
are required for a stronger certified comparison beyond global Lipschitz.

### Group C — time series, HAPT, structured perturbations (G2 + G4)

| ID | Paper | Why | Pri |
|---|---|---|---|
| P16 | Overlapping vs non-overlapping sliding windows HAR (Sensors 2019) https://www.mdpi.com/1424-8220/19/22/5026 | leakage, subject-independent splits, overlap assumptions | P0 |
| P17 | Franco et al., conformalized randomized smoothing (UAI 2024) https://proceedings.mlr.press/v244/franco24a.html | time-series-native certified robustness | P0 |
| P18 | Self-ensemble time-series certification (arXiv:2409.02802) https://arxiv.org/abs/2409.02802 | margin variance, smoothing-based approach | P1 |
| P19 | Deep Learning for TSC: A Review (2019) https://doi.org/10.1007/s10618-019-00619-1 | CNN baselines, evaluation conventions | P1 |
| P20 | InceptionTime (2020) https://doi.org/10.1007/s10618-020-00710-y | stronger classification reference | P2 |
| P21 | Randomized smoothing (ICML 2019) https://proceedings.mlr.press/v97/cohen19c.html | smoothing vs deterministic bounds — radii NOT directly comparable | P2 |

P17+P18 already exist → never claim time-series certification itself as new.
Dataset citation (required, not a paper): UCI HAPT, Reyes-Ortiz et al. 2015,
DOI 10.24432/C54G7M — already in refs.bib as `reyesortiz2015hapt` (raw signals,
not the 561-feature windows).

### Group D — evaluation, calibration, reproducibility (G2, G5, G6)

| ID | Item | Value | Pri |
|---|---|---|---|
| P22 | Guo et al., calibration (ICML 2017) https://proceedings.mlr.press/v70/guo17a.html | confidence ≠ certificate; builds the B2 confidence-gate baseline | P1 |
| P23 | VNN-COMP 2025 summary arXiv:2512.19007 | verification benchmark organization, runtime comparison practice | P2 (post-G2) |

## 3. Focus five (time budget, not equal coverage)

1. **P01 LipCache — PENDING_FULL_READ.** Read end-to-end (proof, guard-vs-main, hit rate/latency split). Existing C1–C4 excerpts are scoped evidence.
2. **P08 CROWN — PENDING_FULL_READ.** Read end-to-end; hand-derive affine bounds on a toy ReLU net, then the class-pair margin argument.
3. **P02 STBP — PENDING_FULL_READ.** Read end-to-end; separate formal guarantees from empirical tightness. Existing C5–C6 excerpts are scoped evidence.
4. **P03 Wei & Liu — PENDING_FULL_READ.** Method+Experiments+Limitations; write explicitly why faster verification differs from faster inference.
5. **P09 auto_LiRPA — PENDING_FULL_READ.** Theory+API; push `f_a - f_j` margins through `compute_bounds` C; compare IBP/CROWN/alpha-CROWN. API metadata inspection is not this derivation or integration check.

## 4. Paper → gate matrix

| Gate | Required reading | Deliverable |
|---|---|---|
| G0/G1 | P01, P02, P03, P04, P16 | novelty ledger + data feasibility |
| G2, saved baseline complete | P16, P19 (P20 reference; reading still pending) | interpret existing backbone + classification protocol; no automatic retraining |
| G3, pending | P08, P09, P10 | domain contract → minimal diagonal verifier → reuse gate/anchor lifecycle → component instrumentation |
| G4 | P02, P11, P12, P17 | structured-set ablations |
| G5 | P01, P03, P04, P05, P06 | end-to-end inference benchmark |
| G6 | P22, P23 + all related work | paper + reproducibility dossier |

## 5. Required baselines and metrics (design lock before G4/G5)

| ID | Baseline | Meaning |
|---|---|---|
| B0 | Full backbone every window | latency + prediction reference |
| B1 | Naive last-output reuse | cost of uncertified skipping |
| B2 | Confidence gate (P22) | heuristic non-certified competitor |
| B3 | Global Lipschitz certified reuse | simplest certified baseline |
| B4 | Local Lipschitz / Fast-Lip (P10) / RecurJac (P11) | stronger certified baseline |
| B5 | Diagonal CROWN (class-pair) | isolates the value of class-pair bounds |
| B6 | Structured CROWN (event-conditioned) | isolates the value of event sets |
| B7 | B6 + certificate amortization | net runtime saving question |

Minimum viable set for the paper: **B0, B2, B3, B5, B6** (subject to feasibility).

Metrics — every metric MUST state its split, denominator and exclusion policy,
see [G2_CLASS_DECISION.md](G2_CLASS_DECISION.md) §4. **17,453 full-stream** and
**13,866 labeled** describe the entire corpus; **5,053 full-stream test** and
**2,874 pure K=6 test** are different evaluation scopes. Do not pool train/val/test
and label that pool a test benchmark. Report a reuse rate over all evaluated
test-stream windows as well as over a precisely defined eligible subset;
otherwise an eligibility rule can hide exclusions.

    CertifiedReuseRate = N(certified AND reused) / N(eligible windows)
    Agreement           = N(same prediction as backbone) / N(evaluated windows)
    PredictionChangeRate= N{t : argmax F(x_t) != argmax F(x_t^r)} / N(anchor-query pairs evaluated)
    Speedup             = T_full-backbone-total / T_proposed-total

Runtime model (rebuild included):

    E[T] = T_e + T_m + (1-r)*T_f + q*T_b

T_e event extraction · T_m membership/certificate check · r reuse rate ·
T_f full inference · q mean rebuilds per window · T_b rebuild cost.
Net saving claim (C12) only if this beats baseline end-to-end on the same CPU
(median + p95 + mean), including rebuild — mirrors G5 CHECK in GATES.md.
The G2 artifact measures model-forward only (mean 0.2511 ms, p50 0.2357 ms,
p95 0.3313 ms); its correct-class point margins are not certified bounds.
G3 component instrumentation is not a G5 saving result. Include shared
preprocessing in both end-to-end branches and disclose build amortization,
memory and hardware. No energy saving or target-device claim is measured yet.

## 6. Four reading blocks (proposed order, not a calendar commitment)

| Block | Papers | Output |
|---|---|---|
| 1 | P01, P03, P04 | collision matrix, guard-vs-main proof target, updated C11 ledger |
| 2 | P08, P09, P10 | hand-derived class-pair bounds + toy verifier test |
| 3 | P02, P17, P11 | define S_e, membership condition, structured/global ablation design |
| 4 | P16, P05, P22 | review saved G2 protocol, runtime-cost experiment design, metric spec |

Sequencing rule: no gate deliverable waits on all reading items. G2-B is already
complete for K=6; the immediate next gate is G3, beginning with the domain
contract and diagonal case. The present task updates documentation and report
sources, not verifier code or experiments. C11/C12 conclusions require the
relevant literature evidence and appropriate experiments.

## 7. Research note template (one file per paper: notes/P<ID>_<Short>.md)

```markdown
# P<ID> - Title
## 1. Research Question
## 2. Assumptions
## 3. Mathematical Core        (definitions, theorems, proofs, limitations)
## 4. Method                   (Input -> Algorithm -> Output)
## 5. Experiments              (datasets, baselines, metrics, hardware)
## 6. Verified Claims          (claim + exact location of evidence)
## 7. Limitations
## 8. Relation to Our RQ       (OVERLAP / DISTINCT / UNKNOWN)
## 9. Reproduction             (code, parameters, dependencies, tests)
## 10. Open Questions
```

Whenever a paper can affect novelty, update `02_LITERATURE.md` immediately —
do not wait until all 23 are read. Technical docs (not papers):
auto_LiRPA API https://auto-lirpa.readthedocs.io/en/latest/api.html and
PyTorch reproducibility https://docs.pytorch.org/docs/stable/notes/randomness.html.

## 8. Repo layout additions

```
research/
├── refs.bib            # verified bibliographic metadata; claim evidence lives in ledger/notes
├── 02_LITERATURE.md    # ledger updated as reading proceeds
├── 05_READING_PLAN.md  # this file
├── 06_REVIEW_PROTOCOL.md # search, selection, extraction and synthesis rules
├── 07_G3_DOMAIN_CONTRACT.md # domain/preprocessing/certificate contract
├── report.md           # current scientific report; distinct from archived legacy material
├── latex/dossier.tex   # editable scientific LaTeX source
├── papers/             # local PDFs, gitignored; presence is not full-reading evidence
│   ├── core/ verification/ time_series/ systems/   # sort on arrival
└── notes/              # P01..P23 notes, version-controlled
```


