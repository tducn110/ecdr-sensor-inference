# 03 DATA AUDIT — HAPT Feasibility and Candidate-Region Evidence (G1)

> Documentation reconciled 22/10/2026; measurements below remain the saved
> 15/10/2026 post-fix results, not a new run. Current G1 verdict:
> **COMPLETE_WITH_REVISIONS** (historically REVISE). G2 PASS; G3 pending.
> This audit measures raw-window deltas and candidate membership, not certified
> reuse. Domain definitions: [04_THEORY.md](04_THEORY.md); minimal next contract:
> [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).

## 1. Raw source layout (verified from UCI id=341 page)

- RawData/acc_expXX_userYY.txt: triaxial acceleration rows at 50Hz.
- RawData/gyro_expXX_userYY.txt: triaxial angular velocity rows at 50Hz.
- RawData/labels.txt cols: exp ID, user ID, activity ID, start sample, end sample (at 50Hz).
- activity_labels.txt: 12 IDs = 6 basic (standing/sitting/lying/walking/up/down) + 6 transitions (stand-to-sit/sit-to-stand/sit-to-lie/lie-to-sit/stand-to-lie/lie-to-stand).
- Train/X_train.txt (81MB) + Test/X_test.txt (33MB): 561-feature windows + y + subject_id (1-30), features normalized [-1,1]. The raw signals used here have acceleration in g and gyro in rad/s; the handcrafted-feature files are not the streaming inputs.
- Total download 75.9MB, 133 files. License CC BY 4.0, DOI 10.24432/C54G7M.
- KEY for streaming: UCI states raw signals provided instead of pre-windowed version in order to make online tests. Use RawData + labels.txt to rebuild chronological windows; do NOT use Train/Test shuffle for streaming eval.

## 2. Audit questions and evidence scope

These questions retain the original G1 motivation. The actual script implements
quantile-level candidate boxes and centered PCA; it does not implement a semantic
event detector, verifier, certified fallback policy or anchor lifecycle. A window
is the observed `6×128` array; an event descriptor would select a candidate; a
region is a set of possible windows. These are separate objects.

### Q1 — Temporal delta distribution (overlap != similarity)
- Build windows 6x128 stride 64 per experiment in chronological order.
- Measure ||x_{t+1}-x_t||_2 and ||.||_inf across consecutive windows, split by same-activity vs boundary/transition.
- Report delta quantiles p50/p90/p99. G2 now exists, but no full-stream PredictionChangeRate or region-certified margins were produced by this audit.
- PASS bar: non-trivial mass of consecutive deltas small enough that SOME event set could cover them; boundary windows flagged separately.

### Q2 — Candidate-region coverage
- Fit raw diagonal budgets on train same-activity deltas at q50/q75/q90; evaluate membership of consecutive windows on each split.
- The script's names stable/moderate/wide denote budget levels, not classified semantic event types. Periodic/transition/spike-specific regions remain proposals.
- Measure membership coverage and same-vs-change subsets. Non-membership is not yet runtime fallback, and membership alone is not certified reuse.
- Explore centered PCA score-box plus orthogonal residual-L2 membership as a separate candidate; do not equate it with a coordinate-residual-box formulation.

### Q3 — Label balance + K decision
- Count windows per each of 12 IDs from labels.txt start/end spans; report transition durations (expected short, imbalanced).
- Decide K=12 vs K=6 (+boundary rule: majority vote? drop? separate transition class?) BEFORE training backbone.
- Split by subject (e.g. 70/30 by volunteers as UCI, or LOSO), keep chronological order inside each recording. Never randomize overlapping windows across train/test.
- Fit empirical distributions/budgets on train; choose levels/policies on validation subjects and freeze before held-out test evaluation. The existing audit reports all candidate levels; it has not established a final selected reuse policy.

## 3. Existing audit scripts (no verifier)

- `scripts/01_fetch_hapt.py`: download/checksum provenance.
- `scripts/02_validate_raw.py`: raw alignment/integrity.
- `scripts/03_build_windows.py`: chronological windows and label policy.
- `scripts/04_audit_splits.py`: disjoint subjects/recordings and ordering.
- `scripts/05_audit_temporal_events.py`: raw deltas, correlations and candidate membership.
- `scripts/06_audit_class_support.py`: class/transition support for the K decision.
- `scripts/06b_corr_semantics.py`: distinguish correlation scales.

## 4. G1 results (measured 2026-10-15, scripts 01-06b in scripts/)

> **CORRECTION 2026-10-15 (post-G1 re-verification):** the original run of `05`
> indexed `windows.npz` with the per-recording `win_idx` instead of the global row
> id — on **98.2% of consecutive pairs** the signals came from `exp01`'s windows
> while labels came from the current recording. Q1/correlation/coverage numbers
> below were recomputed with the fixed script (`05` + `row_id`; pre-fix values
> preserved in git history at commit `cefd171`). Findings F1–F3 changed
> materially; the historical verdict REVISE is recorded today as
> COMPLETE_WITH_REVISIONS, not an unconditional pass of the event model.



### Integrity + provenance (01, 02)
- Zip: 79,596,192 B, SHA-256 `4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09`, 133 members, DOI 10.24432/C54G7M, CC BY 4.0 (`data/manifest.json`).
- 61 recordings, 1,214 label rows, acc/gyro paired **aligned**, 0 issues, rows 9,898-32,089/recording (`data/raw_validation.json`).
- Annotation coverage min 0.3921 — unannotated gaps exist, never auto-labeled.

### Windows + labels (03)
- 17,453 windows (6x128, stride 64), shape (17453, 6, 128), 30 subjects, 61 recordings.
- Policy: pure 10,361 (59.4%) / unlabeled 3,587 (20.55%) / majority 1,917 (11.0%) / ambiguous 1,588 (9.10%); crosses-boundary 7.29%. Policy LOCKED before G2.

### Splits (04)
- Train 17 / Val 4 / Test 9 subjects (test = UCI Test set), windows 9,978 / 2,422 / 5,053.
- subject_disjoint TRUE, recording_disjoint TRUE, chronological_kept TRUE, 0 issues (`data/split_audit.json`).

### Q1 temporal distances (05, 17,392 pairs — recomputed after index fix)

These norms use raw channels with different physical units, not normalized
model coordinates. They are descriptive distances for this audit; do not apply
them directly as normalized verifier radii.

| Pair type | n | L2 p50/p90 | Linf p50/p90 |
|---|---|---|---|
| same_activity | 9,278 | 1.84 / 17.35 | 0.39 / 2.80 |
| label_change | 1,651 | 13.27 / 22.98 | 2.14 / 3.82 |

**Finding F1 (corrected): raw deltas DO separate same vs change at the median** —
same-activity L2 p50 1.84 vs label-change p50 13.27 (~7x gap); Linf p50 0.39 vs
2.14. Overlap remains in the tails (p90 17.35 vs 22.98), so "overlap != similarity"
still holds for a minority of same-activity pairs, but the pre-fix claim that
distributions "barely separate" was an artifact of the indexing bug.


### Temporal correlation (05, recomputed) + semantics (06b)
Corrected lag1 per-channel Pearson (flattened consecutive windows): acc_z 0.050,
acc_y 0.025, acc_x 0.016, gyro_x -0.029, gyro_y -0.013, gyro_z -0.007; lags 2/4/8 ~0.
Semantics measured by `06b_corr_semantics.py` (`data/corr_semantics.json`):
the 05 quantity pairs raw sample t with t+64 (1.28 s) because windows overlap 50%.
Three distinct scales: **raw sample lag-1 at 50 Hz = 0.82–0.98**, **window-level
channel-mean feature lag-1 = 0.56–0.97**, flattened positional lag-1 = -0.03…0.05.
**Finding F2 (restated): temporal dependence is strong at signal and window-feature
scale, weak only for flattened cross-window positional pairing** — any claim must
name the scale; "the data has weak temporal correlation" is not supported.


### Q2 event-region coverage (budgets fit on TRAIN only; recomputed after index fix)
| Box (train q) | train | val | test | test_same | test_change |
|---|---|---|---|---|---|
| stable p50 | 0.002 | 0.004 | 0.001 | 0.003 | 0.000 |
| moderate p75 | 0.280 | 0.302 | 0.255 | 0.458 | 0.002 |
| wide p90 | 0.330 | 0.353 | 0.319 | 0.525 | 0.023 |

- **p50 box degenerates**: per-dim q50 thresholds over same-activity deltas are
  tiny (median 0.023, max 0.056) because ≥50% of same pairs are near-static, so the
  joint 768-dim box is razor-thin (train same-coverage 0.27%). Budget levels must
  be re-chosen (q75/q90 or per-activity) — do NOT use the q50 level as "stable".
- stable-box-but-label-changed = **0.00%** of stable hits (was 7.68% pre-fix).
- **Wide p90 box separates strongly**: captures 52.5% of same-activity pairs on
  test while admitting only 2.3% of label-change pairs.
- **PCA projection candidate (k=108, explained variance 0.9008): coverage
  0.3663/0.3886/0.3542 vs diagonal q90 0.3296/0.3525/0.3194**
  — +3.48 percentage points on test (pre-fix +8pp is superseded). The script
  centers `delta` by train mean `mu`, uses `scores=(delta-mu)@U`, and accepts
  `|scores|<=b` plus `||delta-mu-U scores||_2<=epsilon`. The residual uses an
  L2 norm, not per-coordinate thresholds. The PCA basis is fit on a train
  subset, `mu` on all train pairs, and score/residual q90 on train same-activity
  pairs. These are empirical candidate parameters, not decision certificates.

**Finding F3 (corrected): raw-delta event boxes work better than the pre-fix audit
showed; PCA's coverage advantage is smaller and NOT budget-matched.** The
structured comparison belongs to G4 after a minimal G3 verifier, with matched
coverage/budget or another justified comparison. Full-stream PredictionChangeRate
is now computable from the frozen G2 backbone but is not in these audit outputs.
Label-change is ground-truth metadata, not a certificate violation.
Plots: `plots/delta_distributions.png`, `plots/event_coverage.png` (regenerated).
Full: `data/event_audit.json`; correlation semantics: `data/corr_semantics.json`.


## 5. Current verdict: **COMPLETE_WITH_REVISIONS**

Historical label: REVISE. Dataset feasibility has evidence; candidate-region
usefulness for certified inference remains a hypothesis. No measurements were
rerun for the 22/10/2026 documentation reconciliation.

- GO criteria met: integrity/alignment/splits/labels all clean; corrected p90-box
  same-activity coverage 52.5% vs 2.3% change-pairs on test is enough signal to
  test event-conditioned reuse. Not NO-GO. (Pre-fix "25-36%" figure superseded.)
- Remaining revision/evaluation obligations:
  1. Budget fitting must avoid the degenerate q50 level (near-zero mode); use
     q75/q90 as candidates, and report same-vs-change separation, not raw coverage
     alone. Per-activity budgets require a separately defined online selector;
     ground-truth test activities cannot drive online gating.
  2. Correlation claims must be scale-qualified (06b: signal 0.82-0.98 / window
     feature 0.56-0.97 / flattened positional ~0.05). PCA and diagonal need a fair
     matched comparison before a structured-benefit claim (G4); this is not a
     prerequisite for the first diagonal G3 prototype. Quantiles can define
     formally verified candidates; G3 needs valid bounds and conservative
     membership on the exact declared region, not a claim that empirical
     quantiles already prove decision preservation.
  3. Unlabeled 20.55% + ambiguous 9.10% stay excluded from training targets per
     LOCKED policy; report labeled (13,866) vs full-stream (17,453) denominators
     in every relevant evaluation. Pure K=6 corpus is 10,017 windows; full
     held-out test stream is 5,053 and pure K=6 test is 2,874. Agreement can be
     checked on all test windows without ground truth; accuracy/F1 needs its
     explicit K=6 label scope. Do not pool train/val/test as an evaluation set.
- K decision for training recorded in `G2_CLASS_DECISION.md` (K=6 baseline GO;
  K=12 pure-only HOLD — transition pure support 5-65/class/split).
- G2-B is complete per `G2_CLASS_DECISION.md` and saved artifacts. G3 remains
  pending; `07_G3_DOMAIN_CONTRACT.md` specifies a proposed normalized diagonal
  prototype with fixed model/preprocessing, bounds, membership, reuse gate and
  instrumentation. The documentation contract is not verifier implementation.
- Absolute conditions verified in code: 03 never spans recordings; unlabeled
  never auto-labeled (02/03 checks passed).

## 6. Domain and inference limits for the next prototype

Use raw `x`, fixed train-fit normalization `N`, normalized `z=N(x)` and
reference `F=f_theta o N`. Either transform raw diagonal budgets by `sigma` or
verify a graph including normalization; mixing raw budgets with normalized
inputs without transformation has no declared-domain guarantee. Real-arithmetic
coordinate identities still need a conservative floating-point implementation.

The PCA audit is the projection-defined centered score-box/residual-L2 domain;
it is not the prior theory's unrestricted coordinate residual box. Their
membership rules/support functions differ; see theory §2–3. Normalization also
changes a raw orthogonal PCA geometry and cannot preserve its projection rule
by merely replacing `U` with a rescaled basis.

All reported coverage is for `x_{t+1}` relative to the preceding window
`x_t`. It is not coverage relative to a trusted anchor retained across several
reuse steps. Certificate construction and such anchor lifecycles are absent
from this audit, as are numerical soundness evidence and runtime savings.


