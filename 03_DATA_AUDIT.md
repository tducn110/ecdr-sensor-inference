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

## 4. G1 results (measured 2026-10-15, scripts 01-05 in scripts/)

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

### Q1 temporal distances (05, 17,392 pairs)
| Pair type | n | L2 p50/p90 | Linf p50/p90 |
|---|---|---|---|
| same_activity | 9,278 | 9.45 / 19.23 | 1.86 / 3.25 |
| label_change | 1,651 | 11.02 / 19.92 | 2.13 / 3.41 |

**Finding F1: distributions OVERLAP heavily** — raw deltas barely separate same vs change. Confirms "overlap != similarity".

### Temporal correlation (05)
lag1 per-channel Pearson: acc_z 0.140, acc_y 0.052, acc_x 0.040, gyro_y 0.050, gyro_x -0.044, gyro_z -0.022; lags 2/4/8 ~0.
**Finding F2: window-level temporal correlation is WEAK** — 50% overlap does not carry strong raw-signal correlation.

### Q2 event-region coverage (budgets fit on TRAIN only)
| Box (train q) | train | val | test | test_same | test_change |
|---|---|---|---|---|---|
| stable p50 | 0.175 | 0.174 | 0.191 | 0.250 | 0.167 |
| moderate p75 | 0.243 | 0.239 | 0.257 | 0.326 | 0.233 |
| wide p90 | 0.294 | 0.291 | 0.301 | 0.363 | 0.286 |

- stable-box-but-label-changed = 7.68% of stable hits.
- **Correlated set (PCA, k=52 comps, 90.2% var, train-fit q90): coverage 0.375/0.370/0.381 vs diagonal p90 0.294/0.291/0.301** — correlated structure gives ~+8pp coverage.

**Finding F3: coverage is meaningful but modest; correlated set beats diagonal box as hypothesized.**
Plots: `plots/delta_distributions.png`, `plots/event_coverage.png`. Full: `data/event_audit.json`.

## 5. Verdict: **REVISE** (dataset feasible; event model must change)

- GO criteria met: integrity/alignment/splits/labels all clean; same-activity coverage 25-36% is enough signal to test event-conditioned reuse. Not NO-GO.
- REVISE items before G3 (do NOT freeze 04_THEORY with raw diagonal boxes as-is):
  1. Raw-window diagonal boxes separate same vs change poorly (F1: 0.25 vs 0.17 at p50) — event features need more than raw deltas (e.g., per-activity budgets, alignment, low-level features).
  2. Weak lag correlation (F2) argues against naive temporal-correlation event sets on raw windows; correlated-set idea stays but must beat baseline in reuse, not just coverage.
  3. Unlabeled 20.55% + ambiguous 9.10% must be excluded from training targets per locked policy; boundary policy (pure/majority/transition/ambiguous) reported and LOCKED.
- G2 (backbone) may start in parallel; G3 verifier waits for revised event model + 04_THEORY freeze.
- Absolute conditions verified in code: 03 never spans recordings; unlabeled never auto-labeled (02/03 checks passed).

