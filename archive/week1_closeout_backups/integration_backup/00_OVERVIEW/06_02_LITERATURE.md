# 02 LITERATURE — Prior Work vs Working Gap (G0 evidence)

> Updated 2026-10-22. Sources: refs.bib has 10 source records: original seven, NoScope, CROWN paper and automatic-LiRPA paper. Metadata/API checks and scoped readings do not mean all papers have been deeply reviewed or reproduced.
> Rule: no turnX citations. Every SUPPORTED row needs URL + page/quote. Numbers without full-PDF quotes stay UNVERIFIED.
> Review protocol: 06_REVIEW_PROTOCOL.md. Search pilot and coverage limits: notes/2026-10-21_SEARCH_LOG.json.

## 1. Claim verification ledger

| # | Claim (from report.md) | Status | Evidence |
|---|---|---|---|
| C1 | LipCache has per-sample certified reuse via GuardNet | SUPPORTED | arXiv:2608.13144 abstract: per-sample certified reuse radius from local margin + spectral norm of head; reuse only inside certified ball, else fallback to MainNet |
| C2 | LipCache speedup max 1.65x | SUPPORTED (scoped) | Same abstract: measured speedup up to 1.65x on CIFAR / Tiny-ImageNet / SVHN with limited accuracy degradation. Scope = image tasks in paper, NOT time series |
| C3 | LipCache certifies GuardNet-side, not MainNet-side | SUPPORTED | Same abstract: all accepted hits satisfy GuardNet-side certified-consistency condition |
| C4 | LipCache authors state MainNet-side + streaming as future work | SUPPORTED | Full PDF v1 Sec 7.1 (p.18): all experiments on i.i.d. test sets; temporal redundancy, concept drift, consecutive-frame reuse in streaming left as future work. Remark 5.2: theoretical certificate is on the GuardNet side, not on the MainNet side; MainNet-side needs stacked Lipschitz/smoothing |
| C5 | STBP studies structured spatio-temporal bounds | SUPPORTED | Full PDF arXiv:2606.09746 v1 (33pp): STBP = exact closed-form first conv layer + propagation after; attacker modifies subset of frames or patches in consecutive frames; ST-Bench benchmark; local copy papers/STBP_Varghese2026_arXiv2606.09746.pdf |
| C6 | STBP reports 1.7x certified robust accuracy | SUPPORTED (scoped) | Prior full-PDF abstract+Sec 6 check: up to 1.7x higher certified robust accuracy vs IBP under identical budgets on the paper's datasets, NOT speed. STBP-LJ empirical-estimator caveat remains from the earlier scoped full-text reading. Current arXiv abstract and local title page inspected 2026-10-22; remote/local byte identity was not established and file-size mismatch is not used to infer scientific content |
| C7 | DeltaCNN reports speedup max 7x | SUPPORTED (scoped) | CVPR22 OA: sparse end-to-end, no retraining, first to beat cuDNN dense ref in practice, up to 7x with marginal accuracy diff. Scope = video workload/HW in paper, NO certificate |
| C8 | Franco et al UAI24 does time-series cert via conformalized randomized smoothing | SUPPORTED | PMLR v244:1371-1388: smoothing generalized to arbitrary transformations + conformal prediction, automotive time-series use case |
| C9 | HAPT fits streaming experiments (30 subjects, 6+6, 50Hz, raw) | SUPPORTED | UCI id=341 page: 30 volunteers 19-48y, 6 basic + 6 transitions, acc+gyro 50Hz waist-mounted, window 2.56s/50% overlap/128 readings, raw signals for online tests, DOI 10.24432/C54G7M, CC BY 4.0 |
| C10 | auto_LiRPA supports C-spec margins for g_aj | SUPPORTED (machinery) | API page: compute_bounds(x, aux, C, method...) with C mapping logits to classification margins; methods IBP/CROWN/CROWN-IBP/Forward/alpha-CROWN; PerturbationLpNorm Linf box. Correlated event-set support still needs prototype |
| C11 | Our problem is certainly novel | UNVERIFIED | Keep as working hypothesis. Evidence of open gap (LipCache C4, STBP) is necessary but NOT sufficient: need broader prior-art audit. New collision found: Wei & Liu 2021 (arXiv:2106.12732) already reuses verification results via temporal dependencies (100x faster VERIFICATION, not inference) |
| C12 | CROWN is fast enough for net saving | UNVERIFIED | Zero benchmarks. G5 experiment required. auto_LiRPA examples show CROWN fast/tight on bound computation but give NO verifier-vs-inference cost comparison |
| C13 | First to use temporal correlation for reuse of certificates | NOT SUPPORTED (do not claim) | Wei & Liu 2021 already exploit temporal dependencies to reuse/incrementalize verification. We may only claim the specific combination: event-conditioned sets + sound class-pair bounds + direct backbone decision preservation + measured inference savings |
| C14 | Temporal difference + conditional inference predates this proposal | SUPPORTED (scoped, 2026-10-21) | NoScope, refs.bib kang2017noscope: https://www.vldb.org/pvldb/vol10/p1586-kang.pdf, Figure 1 p.1586, Sec 5 pp.1589-1590 and Sec 7 p.1591. Evidence/interpretation: notes/P04_NoScope.md; no benchmark reproduction or formal per-input certificate claim imported |
| C15 | Class-pair output-difference construction predates ECDR | SUPPORTED (targeted reading) | CROWN, refs.bib zhang2018crown, PDF p.5, Sec 3.1 footnote 2; source URL https://proceedings.neurips.cc/paper_files/paper/2018/file/d04863f100d59b3eb688a11f95b0ae60-Paper.pdf. Full P08 derivation/reproduction note still pending |

## 2. Prior-work comparison (4 axes)

| Method | Certification target | Perturbation structure | Temporal reuse | Runtime cost |
|---|---|---|---|---|
| LipCache (Xiang et al. 2026) | GuardNet prediction consistency inside reuse ball (SUPPORTED) | Isotropic ball in GuardNet feature space from margin/spectral norm; per-pair r*_i tighter (SUPPORTED, Remark 5.1) | Per-sample cache + MainNet fallback; streaming explicitly future work, i.i.d. only (SUPPORTED, Sec 7.1 p.18) | Measured up to 1.65x on image tasks (SUPPORTED scoped); edge image service |
| STBP (Varghese et al., SAIV26) | Spatio-temporal robustness of 3D CNNs; only STBP-IBP/Lipschitz strictly sound, STBP-LJ empirical (SUPPORTED, Limitations) | Attacker modifies subset of frames or patches in consecutive frames; exact closed-form first conv layer (SUPPORTED) | Video verification benchmark ST-Bench, not decision-reuse cache (SUPPORTED) | Up to 1.7x certified robust accuracy vs IBP same budget (SUPPORTED scoped, NOT speed); 3D CNN moderate size |
| DeltaCNN (Parger et al., CVPR22) | NONE -- empirical sparse reuse (SUPPORTED) | Sparse frame differences, truncated updates (SUPPORTED) | Frame-by-frame sparse propagation, no error accumulation (SUPPORTED) | Up to 7x vs cuDNN dense (SUPPORTED scoped); no certificate |
| Franco et al. (UAI24) | Robustness + conformal guarantees under perturbations/domain shift (SUPPORTED) | Arbitrary time-series-native transformations (SUPPORTED) | Single time-series classifier, automotive case; no anchor-reuse system (SUPPORTED) | Accuracy/reliability retained (SUPPORTED qualitative); no latency-saving claim |
| Wei & Liu (2021, arXiv:2106.12732) | Online NN verification under changing specs/networks (SUPPORTED) | Not input-set structuring; tolerance to spec/network updates (SUPPORTED) | Reuses verification results via temporal dependencies: Branch Management + Perturbation Tolerance + Incremental Computation (SUPPORTED) | Up to 100x acceleration OF VERIFICATION, not of inference (SUPPORTED scoped). Kills any first-temporal-reuse claim |
| CROWN / automatic LiRPA (NeurIPS 2018/2020) | Lower bounds for robustness and class margins; available mathematical/software machinery | Perturbation-domain assumptions must match the bounds | Not evidence that an ECDR trusted-anchor runtime exists | No verifier-versus-forward or net-saving result for this project |
| NoScope (Kang et al., PVLDB 2017) | Approximate reference-model agreement; empirical FP/FN boundary (Sec 3, 6.3) | Video difference detector; sound event-set membership not established by this scoped reading | Difference filtering + specialized-model cascade (C14) | Inference system; no speed number imported in this pilot |
| Ours (working hypothesis) | Direct backbone decision equivalence on reuse (UNVERIFIED -- to prove in G3) | Event-conditioned sets S_e, diagonal baseline then correlated (UNVERIFIED until G1/G3) | Anchor lifecycle + fallback, no transitive reuse without proof (design, UNVERIFIED) | Net saving iff r*T_f > T_e+T_m+q*T_b incl. rebuild (UNVERIFIED until G5) |

## 3. Novelty labels (G0 output)

- SUPPORTED: per-pair certified radius exists (LipCache C1 + Remark 5.1); GuardNet-side scope + MainNet/streaming future work with PDF quotes (C3+C4); structured spatio-temporal verification with 1.7x certified accuracy vs IBP scoped + soundness caveat (C5+C6); sparse temporal reuse without cert (DeltaCNN C7); time-series smoothing cert (Franco C8); temporal reuse of verification results exists (Wei & Liu C13); C-spec margin machinery (C10).
- NOT SUPPORTED / must not claim: first use of temporal correlation for certificate reuse (Wei & Liu 2021 prior art).
- NOT SUPPORTED / must not claim: first use of temporal differences to skip neural inference (NoScope C14).
- NOT SUPPORTED / must not claim: first class-pair margin construction (CROWN C15).
- UNVERIFIED (keep as hypotheses): our novelty as a whole (C11); CROWN net-saving (C12).

## 4. Gap hypothesis (not a proven-new conclusion)

Gap = event-conditioned structured bounds + direct MainNet/backbone decision
certification + sequential reuse + runtime-cost analysis INCLUDING rebuild.
Do NOT claim first structured verifier or first class-pair radius.
G0 passes on documentation of the above ledger + table + labels; G1/G3/G5 decide
whether the hypothesis survives.

## 5. Additions from G1 re-review — scoped update 2026-10-21, other items still pending

Flagged by external review as possible collisions/baselines. Sources must be read
and evidence-located before a claim is promoted. NoScope has now received a scoped
source check; other rows remain TO_READ. C11 stays UNVERIFIED.

| Source | Why it matters | Status |
|---|---|---|
| NoScope (VLDB 2017, Stanford DAWN) — https://dawn.cs.stanford.edu/publications/noscope/ | temporal difference detectors + conditional inference; distinction must go beyond temporal skipping | SOURCE_CHECKED_SCOPED (P04), C14 + notes/P04_NoScope.md; deep reading/reproduction still open |
| Overlapping vs non-overlapping sliding windows for HAR (MDPI Sensors 2019) — https://www.mdpi.com/1424-8220/19/22/5026 | overlap benefit shrinks under subject-independent eval → overlap must never be used as similarity/performance evidence; supports G2 eval protocol | TO_READ (P16) |
| Fast-Lin/Fast-Lip (Weng et al., ICML 2018) — https://proceedings.mlr.press/v80/weng18a.html | certified baseline stronger than plain global Lipschitz → baseline B4 | TO_READ (P10) |
| RecurJac (AAAI 2019) — https://cdn.aaai.org/ojs/4522/4522-13-7561-1-10-20190706.pdf | Jacobian/Lipschitz bound baseline → baseline B4 | TO_READ (P11) |

Full 23-paper reading list, gate mapping, baselines B0-B7 and metric definitions:
`05_READING_PLAN.md`.

## 6. Refresh scope, 2026-10-22

Official UCI HAPT, CROWN proceedings, NeurIPS 2020 automatic-LiRPA proceedings,
auto_LiRPA API and STBP arXiv metadata were reopened. Local LipCache/STBP PDF
title pages were inspected. The LipCache HTML fetch failed; earlier full-text
claims retain their original provenance. No remote/local STBP byte identity was
re-established. This refresh does not close the 23-paper review, numerical
soundness, G3 compatibility, C11 novelty or C12 net-saving questions.

The current report is `report.md`; the standalone scientific LaTeX report is
`latex/dossier.tex`. G3 documentation is `07_G3_DOMAIN_CONTRACT.md`; no verifier
implementation or new empirical result is implied by these documents.

