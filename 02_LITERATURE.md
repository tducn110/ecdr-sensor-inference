# 02 LITERATURE — Prior Work vs Working Gap (G0 evidence)

> Sources: refs.bib (6 entries, verified 2026-10-15 from arXiv / Springer / CVPR OA / PMLR / UCI / auto_LiRPA docs).
> Rule: no turnX citations. Every SUPPORTED row needs URL + page/quote. Numbers without full-PDF quotes stay UNVERIFIED.

## 1. Claim verification ledger

| # | Claim (from report.md) | Status | Evidence |
|---|---|---|---|
| C1 | LipCache has per-sample certified reuse via GuardNet | SUPPORTED | arXiv:2608.13144 abstract: per-sample certified reuse radius from local margin + spectral norm of head; reuse only inside certified ball, else fallback to MainNet |
| C2 | LipCache speedup max 1.65x | SUPPORTED (scoped) | Same abstract: measured speedup up to 1.65x on CIFAR / Tiny-ImageNet / SVHN with limited accuracy degradation. Scope = image tasks in paper, NOT time series |
| C3 | LipCache certifies GuardNet-side, not MainNet-side | SUPPORTED | Same abstract: all accepted hits satisfy GuardNet-side certified-consistency condition |
| C4 | LipCache authors state MainNet-side + streaming as future work | SUPPORTED | Full PDF v1 Sec 7.1 (p.18): all experiments on i.i.d. test sets; temporal redundancy, concept drift, consecutive-frame reuse in streaming left as future work. Remark 5.2: theoretical certificate is on the GuardNet side, not on the MainNet side; MainNet-side needs stacked Lipschitz/smoothing |
| C5 | STBP studies structured spatio-temporal bounds | SUPPORTED | Full PDF arXiv:2606.09746 v1 (33pp): STBP = exact closed-form first conv layer + propagation after; attacker modifies subset of frames or patches in consecutive frames; ST-Bench benchmark; local copy papers/STBP_Varghese2026_arXiv2606.09746.pdf |
| C6 | STBP reports 1.7x certified robust accuracy | SUPPORTED (scoped) | Same PDF abstract+Sec 6: up to 1.7x higher certified robust accuracy vs IBP under identical perturbation budgets on UCF-101/Udacity/MedMNIST. NOT speed. Caveat: STBP-LJ empirical estimator (Remark 1, Sec 4.3.1), only STBP-IBP/Lipschitz strictly sound. Local file verified: md5 07fa78504afb302ecb958822f1fd003b, 3,529,184 bytes = arXiv server content-length at fetch; 2.5MB figure was a stale/different build |
| C7 | DeltaCNN reports speedup max 7x | SUPPORTED (scoped) | CVPR22 OA: sparse end-to-end, no retraining, first to beat cuDNN dense ref in practice, up to 7x with marginal accuracy diff. Scope = video workload/HW in paper, NO certificate |
| C8 | Franco et al UAI24 does time-series cert via conformalized randomized smoothing | SUPPORTED | PMLR v244:1371-1388: smoothing generalized to arbitrary transformations + conformal prediction, automotive time-series use case |
| C9 | HAPT fits streaming experiments (30 subjects, 6+6, 50Hz, raw) | SUPPORTED | UCI id=341 page: 30 volunteers 19-48y, 6 basic + 6 transitions, acc+gyro 50Hz waist-mounted, window 2.56s/50% overlap/128 readings, raw signals for online tests, DOI 10.24432/C54G7M, CC BY 4.0 |
| C10 | auto_LiRPA supports C-spec margins for g_aj | SUPPORTED (machinery) | API page: compute_bounds(x, aux, C, method...) with C mapping logits to classification margins; methods IBP/CROWN/CROWN-IBP/Forward/alpha-CROWN; PerturbationLpNorm Linf box. Correlated event-set support still needs prototype |
| C11 | Our problem is certainly novel | UNVERIFIED | Keep as working hypothesis. Evidence of open gap (LipCache C4, STBP) is necessary but NOT sufficient: need broader prior-art audit. New collision found: Wei & Liu 2021 (arXiv:2106.12732) already reuses verification results via temporal dependencies (100x faster VERIFICATION, not inference) |
| C12 | CROWN is fast enough for net saving | UNVERIFIED | Zero benchmarks. G5 experiment required. auto_LiRPA examples show CROWN fast/tight on bound computation but give NO verifier-vs-inference cost comparison |
| C13 | First to use temporal correlation for reuse of certificates | NOT SUPPORTED (do not claim) | Wei & Liu 2021 already exploit temporal dependencies to reuse/incrementalize verification. We may only claim the specific combination: event-conditioned sets + sound class-pair bounds + direct backbone decision preservation + measured inference savings |

## 2. Prior-work comparison (4 axes)

| Method | Certification target | Perturbation structure | Temporal reuse | Runtime cost |
|---|---|---|---|---|
| LipCache (Xiang et al. 2026) | GuardNet prediction consistency inside reuse ball (SUPPORTED) | Isotropic ball in GuardNet feature space from margin/spectral norm; per-pair r*_i tighter (SUPPORTED, Remark 5.1) | Per-sample cache + MainNet fallback; streaming explicitly future work, i.i.d. only (SUPPORTED, Sec 7.1 p.18) | Measured up to 1.65x on image tasks (SUPPORTED scoped); edge image service |
| STBP (Varghese et al., SAIV26) | Spatio-temporal robustness of 3D CNNs; only STBP-IBP/Lipschitz strictly sound, STBP-LJ empirical (SUPPORTED, Limitations) | Attacker modifies subset of frames or patches in consecutive frames; exact closed-form first conv layer (SUPPORTED) | Video verification benchmark ST-Bench, not decision-reuse cache (SUPPORTED) | Up to 1.7x certified robust accuracy vs IBP same budget (SUPPORTED scoped, NOT speed); 3D CNN moderate size |
| DeltaCNN (Parger et al., CVPR22) | NONE -- empirical sparse reuse (SUPPORTED) | Sparse frame differences, truncated updates (SUPPORTED) | Frame-by-frame sparse propagation, no error accumulation (SUPPORTED) | Up to 7x vs cuDNN dense (SUPPORTED scoped); no certificate |
| Franco et al. (UAI24) | Robustness + conformal guarantees under perturbations/domain shift (SUPPORTED) | Arbitrary time-series-native transformations (SUPPORTED) | Single time-series classifier, automotive case; no anchor-reuse system (SUPPORTED) | Accuracy/reliability retained (SUPPORTED qualitative); no latency-saving claim |
| Wei & Liu (2021, arXiv:2106.12732) | Online NN verification under changing specs/networks (SUPPORTED) | Not input-set structuring; tolerance to spec/network updates (SUPPORTED) | Reuses verification results via temporal dependencies: Branch Management + Perturbation Tolerance + Incremental Computation (SUPPORTED) | Up to 100x acceleration OF VERIFICATION, not of inference (SUPPORTED scoped). Kills any first-temporal-reuse claim |
| Ours (working hypothesis) | Direct backbone decision equivalence on reuse (UNVERIFIED -- to prove in G3) | Event-conditioned sets S_e, diagonal baseline then correlated (UNVERIFIED until G1/G3) | Anchor lifecycle + fallback, no transitive reuse without proof (design, UNVERIFIED) | Net saving iff r*T_f > T_e+T_m+q*T_b incl. rebuild (UNVERIFIED until G5) |

## 3. Novelty labels (G0 output)

- SUPPORTED: per-pair certified radius exists (LipCache C1 + Remark 5.1); GuardNet-side scope + MainNet/streaming future work with PDF quotes (C3+C4); structured spatio-temporal verification with 1.7x certified accuracy vs IBP scoped + soundness caveat (C5+C6); sparse temporal reuse without cert (DeltaCNN C7); time-series smoothing cert (Franco C8); temporal reuse of verification results exists (Wei & Liu C13); C-spec margin machinery (C10).
- NOT SUPPORTED / must not claim: first use of temporal correlation for certificate reuse (Wei & Liu 2021 prior art).
- UNVERIFIED (keep as hypotheses): our novelty as a whole (C11); CROWN net-saving (C12).

## 4. Gap hypothesis (not a proven-new conclusion)

Gap = event-conditioned structured bounds + direct MainNet/backbone decision
certification + sequential reuse + runtime-cost analysis INCLUDING rebuild.
Do NOT claim first structured verifier or first class-pair radius.
G0 passes on documentation of the above ledger + table + labels; G1/G3/G5 decide
whether the hypothesis survives.
