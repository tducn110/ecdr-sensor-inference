# 05 READING PLAN — literature expansion, baselines, metrics (G1 re-review 2026-10-15)

> Status: PLAN. Nothing here is SUPPORTED until it is read, quoted, and added to
> `refs.bib` + the `02_LITERATURE.md` ledger per G0 rules. C11 (novelty) stays
> UNVERIFIED and C12 (CROWN net-saving) stays UNVERIFIED until G4/G5 evidence.
> Reading notes live in `notes/P<ID>_<ShortName>.md` (template §7).

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
| NoScope (2017) | temporal difference + conditional inference | no class-pair guarantee |
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

Required derivations after P08+P09 (re-derive, do not copy):

    g_aj_lower = alpha^T x^r + beta - ||D_e^T alpha||_1
    + residual term  -sum_i eps_i * |alpha_i|   if |v_i| <= eps_i

valid only if the affine relaxations are proven valid on the WHOLE perturbation
domain under consideration. P08/P09 are mandatory before using auto_LiRPA as the
G3 verifier; P10/P11 mandatory so the certified baseline is not a strawman.

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

1. **P01 LipCache** — read end-to-end (proof, guard-vs-main, hit rate/latency split). [DONE partially: full PDF on disk, C1-C4 quoted]
2. **P08 CROWN** — read end-to-end; hand-derive affine bounds on a toy ReLU net, then the class-pair margin argument.
3. **P02 STBP** — read end-to-end; separate formal guarantees from empirical tightness. [DONE partially: full PDF on disk, C5-C6 quoted]
4. **P03 Wei & Liu** — Method+Experiments+Limitations; write explicitly why faster verification != faster inference.
5. **P09 auto_LiRPA** — Theory+API; push `f_a - f_j` margins through `compute_bounds` C; compare IBP/CROWN/alpha-CROWN.

## 4. Paper → gate matrix

| Gate | Required reading | Deliverable |
|---|---|---|
| G0/G1 | P01, P02, P03, P04, P16 | novelty ledger + data feasibility |
| G2 | P16, P19 (P20 reference) | backbone + classification protocol |
| G3 | P08, P09, P10 | sound class-pair verifier |
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

Metrics — every metric MUST state its denominator (labeled 13,866 vs
full-stream 17,453 vs eligible windows), see G2_CLASS_DECISION.md §4:

    CertifiedReuseRate = N(certified AND reused) / N(eligible windows)
    Agreement           = N(same prediction as backbone) / N(evaluated windows)
    PredictionChangeRate= N{t : f(x_t) != f(x_t^r)} / N          (G2-B prerequisite)
    Speedup             = T_full-backbone-total / T_proposed-total

Runtime model (rebuild included):

    E[T] = T_e + T_m + (1-r)*T_f + q*T_b

T_e event extraction · T_m membership/certificate check · r reuse rate ·
T_f full inference · q mean rebuilds per window · T_b rebuild cost.
Net saving claim (C12) only if this beats baseline end-to-end on the same CPU
(median + p95 + mean), including rebuild — mirrors G5 CHECK in GATES.md.

## 6. Four-week reading plan

| Week | Papers | Output |
|---|---|---|
| 1 | P01, P03, P04 | collision matrix, guard-vs-main proof target, updated C11 ledger |
| 2 | P08, P09, P10 | hand-derived class-pair bounds + toy verifier test |
| 3 | P02, P17, P11 | define S_e, membership condition, structured/global ablation design |
| 4 | P16, P05, P22 | final G2 protocol, runtime-cost experiment design, metric spec |

Sequencing rule: no gate deliverable waits on all 23 papers — G2-B training may
start in parallel with reading (K=6, see G2_CLASS_DECISION.md). C11/C12 conclusions
only after the relevant groups are read AND experiments exist.

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
├── refs.bib            # add entries ONLY with verification quotes
├── 02_LITERATURE.md    # ledger updated as reading proceeds
├── 05_READING_PLAN.md  # this file
├── papers/             # local PDFs, gitignored (2 PDFs on disk: LipCache, STBP)
│   ├── core/ verification/ time_series/ systems/   # sort on arrival
└── notes/              # P01..P23 notes, version-controlled
```


