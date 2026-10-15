# 03 DATA AUDIT — HAPT Feasibility before Freezing Theory (G1)

> Depends on: G0 done (refs.bib + 01 + 02). Blocks: 04_THEORY freeze.
> Goal: answer 3 questions with measured numbers, not assumptions.

## 1. Raw source layout (verified from UCI id=341 page)

- RawData/acc_expXX_userYY.txt: triaxial acceleration rows at 50Hz.
- RawData/gyro_expXX_userYY.txt: triaxial angular velocity rows at 50Hz.
- RawData/labels.txt cols: exp ID, user ID, activity ID, start sample, end sample (at 50Hz).
- activity_labels.txt: 12 IDs = 6 basic (standing/sitting/lying/walking/up/down) + 6 transitions (stand-to-sit/sit-to-stand/sit-to-lie/lie-to-sit/stand-to-lie/lie-to-stand).
- Train/X_train.txt (81MB) + Test/X_test.txt (33MB): 561-feature windows + y + subject_id (1-30). Features normalized [-1,1]; acc in g, gyro in rad/s.
- Total download 75.9MB, 133 files. License CC BY 4.0, DOI 10.24432/C54G7M.
- KEY for streaming: UCI states raw signals provided instead of pre-windowed version in order to make online tests. Use RawData + labels.txt to rebuild chronological windows; do NOT use Train/Test shuffle for streaming eval.

## 2. Three audit questions (G1 exit criteria)

### Q1 — Temporal delta distribution (overlap != similarity)
- Build windows 6x128 stride 64 per experiment in chronological order.
- Measure ||x_{t+1}-x_t||_2 and ||.||_inf across consecutive windows, split by same-activity vs boundary/transition.
- Compare deltas against backbone margins m_aj (needs G2 backbone first; preliminary: report delta quantiles p50/p90/p99).
- PASS bar: non-trivial mass of consecutive deltas small enough that SOME event set could cover them; boundary windows flagged separately.

### Q2 — Event-set coverage
- Candidate diagonal boxes S_e from roadmap Sec 3.1: stable / periodic / transition / spike budgets r_i^(e).
- For each consecutive pair, test membership x_{t+1} in S_e(x_t) per event type.
- Report coverage rate per event type + fallback rate. If coverage ~= 0 for all reasonable budgets, diagonal hypothesis fails early.
- Correlated sets (non-diagonal D_e) only if Q1 shows correlation structure in data.

### Q3 — Label balance + K decision
- Count windows per each of 12 IDs from labels.txt start/end spans; report transition durations (expected short, imbalanced).
- Decide K=12 vs K=6 (+boundary rule: majority vote? drop? separate transition class?) BEFORE training backbone.
- Split by subject (e.g. 70/30 by volunteers as UCI, or LOSO), keep chronological order inside each recording. Never randomize overlapping windows across train/test.
- Thresholds (event budgets, confidence) tuned on validation subjects only.

## 3. Scripts to write (no verifier yet)

- scripts/download_hapt.py -- fetch UCI 341 zip (75.9MB) + checksum.
- scripts/build_windows.py -- RawData + labels.txt -> chronological windows (N,6,128) + meta (exp/user/activity/span).
- scripts/audit_deltas.py -- Q1 histograms + quantiles, same-vs-boundary split.
- scripts/audit_coverage.py -- Q2 membership per event type.
- scripts/audit_labels.py -- Q3 counts + durations + K proposal.

## 4. G1 results (measured 2026-10-15, scripts 01-06b in scripts/)

> **CORRECTION 2026-10-15 (post-G1 re-verification):** the original run of `05`
> indexed `windows.npz` with the per-recording `win_idx` instead of the global row
> id — on **98.2% of consecutive pairs** the signals came from `exp01`'s windows
> while labels came from the current recording. Q1/correlation/coverage numbers
> below were recomputed with the fixed script (`05` + `row_id`; pre-fix values
> preserved in git history at commit `cefd171`). Findings F1–F3 changed
> materially; the G1 verdict stays REVISE.



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
- **Correlated set (PCA, k=108 comps, 90.1% var, train-fit q90 scores + q90
  residual): coverage 0.366/0.389/0.354 vs diagonal p90 0.330/0.353/0.319**
  — +3.5pp on test (was +8pp pre-fix). Residual v has an explicit budget
  (thr_resid), but both budgets are empirical quantiles, not certified radii.

**Finding F3 (corrected): raw-delta event boxes work better than the pre-fix audit
showed; PCA's coverage advantage is smaller and NOT budget-matched.** Before G3:
compare PCA vs diagonal at matched coverage / region size / budget, and add
PredictionChangeRate (needs the frozen G2 backbone) — label-change is ground
truth, not certificate violation.
Plots: `plots/delta_distributions.png`, `plots/event_coverage.png` (regenerated).
Full: `data/event_audit.json`; correlation semantics: `data/corr_semantics.json`.


## 5. Verdict: **REVISE** (dataset feasible; event model under revision)

- GO criteria met: integrity/alignment/splits/labels all clean; corrected p90-box
  same-activity coverage 52.5% vs 2.3% change-pairs on test is enough signal to
  test event-conditioned reuse. Not NO-GO. (Pre-fix "25-36%" figure superseded.)
- REVISE items before G3 (do NOT freeze 04_THEORY with current budgets):
  1. Budget fitting must avoid the degenerate q50 level (near-zero mode); use
     q75/q90 or per-activity budgets, and report same-vs-change separation, not
     raw coverage alone.
  2. Correlation claims must be scale-qualified (06b: signal 0.82-0.98 / window
     feature 0.56-0.97 / flattened positional ~0.05). Correlated PCA set must
     beat diagonal box at matched coverage or matched budget; residual budget
     exists but is empirical — G3 needs formal radii.
  3. Unlabeled 20.55% + ambiguous 9.10% stay excluded from training targets per
     LOCKED policy; report labeled (13,866) vs full-stream (17,453) denominators
     in every evaluation, plus PredictionChangeRate once the G2 backbone exists.
- K decision for training recorded in `G2_CLASS_DECISION.md` (K=6 baseline GO;
  K=12 pure-only HOLD — transition pure support 5-65/class/split).
- G2-B (backbone) may start; G3 verifier waits for revised event model +
  04_THEORY freeze. Backbone and verifier must share one checkpoint.
- Absolute conditions verified in code: 03 never spans recordings; unlabeled
  never auto-labeled (02/03 checks passed).


