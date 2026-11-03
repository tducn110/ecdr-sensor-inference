# Week 2 — HAPT data and SmallCNN1D foundations

## Scope and evidence boundary

This is an executed, read-only audit of the frozen HAPT data path for the Week 1 reference run. The audit rebuilt every window from the 122 raw accelerometer/gyroscope recording files plus `labels.txt`, compared rebuilt tensors and manifest rows to the frozen artifacts, reconstructed split and supervised support, recomputed train-only normalization, and compared it to both the saved normalization record and selected checkpoint metadata. The runnable source is [`scripts/week2_data_audit.py`](../../scripts/week2_data_audit.py); machine-readable results and the full aligned overlap-pair inventory are in [`data_audit.json`](data_audit.json) and [`data_aligned_overlap_pairs.csv`](data_aligned_overlap_pairs.csv).

The source/contract inputs were `scripts/03_build_windows.py`, `ecdr/data.py`, `ecdr/preprocessing.py`, `scripts/ecdr_audit_support.py`, `configs/backbone.yaml`, and the frozen artifacts under `output/week1_closeout_20261102/reference/`. This report's checks were executed on 2026-11-03 from the repository's `.venv`; exact UTC timestamps, runtime versions, and the audit-script SHA-256 are recorded in `data_audit.json`. The earlier Week 1 receipts remain stored evidence for that earlier run; this audit does not rerun training or model inference. No formal robustness certificate, confidence interval, or i.i.d. claim is produced here.

## HAPT stream construction

| Check | Observed | Evidence / meaning |
|---|---:|---|
| Raw recordings | 61 (122 sensor files) | 1 accelerometer + 1 gyroscope file per recording; filenames pair by experiment/user |
| Window tensor | `17,453 × 6 × 128`, float32 | All finite; 3 acceleration + 3 gyroscope channels |
| Window / stride | 128 samples / 64 samples | 50% overlap; 50 Hz gives 2.56 s windows and 1.28 s stride |
| Rebuilt raw tensor equality | PASS, exact | Every rebuilt float32 value equals `data/windows.npz` |
| Rebuilt manifest equality | PASS, exact | All 17,453 rows and all 10 manifest fields equal the stored manifest |
| Recording boundary rule | No cross-recording windows | Window index restarts for each recording; adjacent recording endpoints are excluded from overlap pairs |
| Label policy totals | pure 10,361; majority 1,917; ambiguous 1,588; unlabeled 3,587 | Sum = 17,453; majority requires dominant annotated label to cover at least 60% of the full 128-sample window |

There are 17,392 adjacent, stride-aligned window pairs in the same recording. For each pair, the 64 overlapping samples compare exactly equal at their corresponding temporal coordinates (raw and normalized mean absolute difference = 0 for every pair). At the same local coordinates in adjacent windows, per-channel mean absolute difference is:

| Channel | Raw stored-unit MAE | Normalized dimensionless MAE |
|---|---:|---:|
| acc_x | 0.15258 | 0.37602 |
| acc_y | 0.11625 | 0.28552 |
| acc_z | 0.10393 | 0.29283 |
| gyro_x | 0.43151 | 0.96399 |
| gyro_y | 0.34394 | 0.81642 |
| gyro_z | 0.25361 | 0.85550 |

These values are per-channel (raw channels retain their stored units); they are not pooled into a physical norm across accelerometer and gyroscope channels. Per-channel medians, 95th percentiles, and the pair-level values are in `data_audit.json` and `data_aligned_overlap_pairs.csv`. Direct within-recording raw-sample correlation averaged 0.43783 at lag 64 and 0.36346 at lag 128. The means differ by channel: at lag 64, accelerometer x/y/z are 0.72095 / 0.82400 / 0.77693, while gyroscope x/y/z are 0.14840 / 0.06063 / 0.09604. These are descriptive dependence indicators; shared observations and temporal structure mean windows cannot be treated as independent examples. None of these summaries establishes an i.i.d. sampling model or supports confidence intervals.

The one-sample coordinate caveat was independently reproduced: changing annotation indexing by one sample changes **98 window policy/label assignments**. The current tensors and labels follow the existing builder's convention, and this audit leaves them untouched. This identifies sensitivity to a coordinate convention; it does not determine which convention is semantically intended by the upstream HAPT annotation source.

## Split, labels, and supervised support

The frozen split has subject-disjoint train/validation/test groups of 17/4/9 subjects, respectively. Every recording maps to one split, and the three subject sets are pairwise disjoint. The denominator distinction matters: all stream windows remain present for sequential evaluation, while only pure and accepted-majority windows with labels 1–12 enter supervised training/evaluation.

| Split | Full stream | Supervised pure + majority | Ambiguous excluded | Unlabeled excluded |
|---|---:|---:|---:|---:|
| Train | 9,978 | 6,955 | 904 | 2,119 |
| Validation | 2,422 | 1,751 | 205 | 466 |
| Test | 5,053 | 3,572 | 479 | 1,002 |
| Total | 17,453 | 12,278 | 1,588 | 3,587 |

Per-class supervised support (train / validation / test windows) is:

| ID | Class | Train | Val | Test |
|---:|---|---:|---:|---:|
| 1 | WALKING | 1,077 | 246 | 531 |
| 2 | WALKING_UPSTAIRS | 973 | 247 | 537 |
| 3 | WALKING_DOWNSTAIRS | 894 | 231 | 492 |
| 4 | SITTING | 1,103 | 289 | 546 |
| 5 | STANDING | 1,206 | 312 | 597 |
| 6 | LAYING | 1,188 | 316 | 586 |
| 7 | STAND_TO_SIT | 79 | 19 | 45 |
| 8 | SIT_TO_STAND | 53 | 13 | 32 |
| 9 | SIT_TO_LIE | 90 | 26 | 51 |
| 10 | LIE_TO_SIT | 89 | 15 | 46 |
| 11 | STAND_TO_LIE | 119 | 18 | 68 |
| 12 | LIE_TO_STAND | 84 | 19 | 41 |

The historical [`data/ecdr_12_class_support.json`](../../data/ecdr_12_class_support.json) records pure totals 5,878 / 1,490 / 2,993 and majority totals 7,859 / 1,956 / 4,051. Current source `scripts/ecdr_audit_support.py` calls `ecdr.data.audit_12_class_support`; its majority predicate is explicitly `label_policy in {pure, majority}` and `label_majority == class_id`, which recomputes to 6,955 / 1,751 / 3,572 on this frozen manifest. Those current source totals agree with the supervised loader and exclude ambiguous rows. The stored historical majority counts do not reproduce the current source predicate. Their difference numerically equals the current ambiguous counts (904 / 205 / 479), but that coincidence does not establish how the historical artifact was generated. The JSON contains no run provenance sufficient to resolve the mismatch. The artifact is historical evidence only, not current execution evidence or a certificate; neither its name nor its blocker fields establish current model acceptance. See `data_support_comparison.json` for the source predicate and exact comparison.

## Frozen SmallCNN1D input and preprocessing contract

The frozen run config records `SmallCNN1D-K12`, input shape `[6,128]`, class order 1–12, and label policy `pure_plus_majority_dominant_0.6`. Its declared network is Conv1d channels 6→32→64→64, kernels 5/3/3, BatchNorm, ReLU, MaxPool2 after the first two convolutions, flattening MLP with hidden width 128. The checkpoint/config report 283,692 parameters. The selected checkpoint is seed 43, selected at epoch 8 with validation macro-F1 0.8467853. This describes the frozen Week 1 reference artifact; it is not a newly trained result.

Normalization was recomputed independently from the 6,955 supervised training windows over window and time axes `(0,2)`, using float64 accumulation, float32 stored statistics, and standard deviation floor `1e-6`. All six means and standard deviations exactly match `normalization.json` and the `selected_best.pt` metadata:

| Channel | Mean | Std |
|---|---:|---:|
| Acc X | 0.8043609262 | 0.4057796597 |
| Acc Y | 0.0411366001 | 0.4071546793 |
| Acc Z | 0.0817700475 | 0.3549118936 |
| Gyro X | 0.0091355676 | 0.4476351142 |
| Gyro Y | -0.0043975562 | 0.4212706089 |
| Gyro Z | -0.0071832035 | 0.2964415252 |

The checkpoint's preprocessing ID equals the saved normalization ID `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`; checkpoint mean/std also match exactly. The frozen run performs normalization on raw windows in the loader after selecting the supervised subset. Consequently, overlapping raw samples appear in multiple windows and contribute repeatedly to the train-statistic weighting. This is window-weighted normalization, not unique-recording-sample weighting; the contract does not deduplicate shared samples.

There is a separate preprocessing implementation in `ecdr/preprocessing.py`. It computes float64 statistics and returns a float64 transform, uses a default `sigma_min=1e-5`, and hashes a JSON contract (including channel names and dimensions) to a truncated 16-character ID. The frozen Week 1 pipeline instead stores float32 statistics, clips standard deviation at `1e-6`, normalizes into float32, and uses a different byte-based full SHA-256 ID. These paths therefore have different numeric and identity contracts. Treating their IDs, epsilon values, or outputs as interchangeable would be incorrect. This audit does not alter or unify them.

## Interpretation and remaining evidence limits

The raw-to-window reconstruction and frozen artifact alignment are verified by this run. Subject separation and the full-stream/supervised denominator mapping also passed. Transition classes have markedly less support than the six basic activities (test counts 32–68 versus 492–597), so the overall window count alone does not imply balanced class evidence. The 50% overlap further makes window-level counts larger than independent temporal units.

This work is a data and preprocessing audit. It does not re-evaluate the saved checkpoint's predictions, rerun its historical metrics, prove generalization, or establish a robustness/certification claim. Those require their own explicit evidence and protocol. `data_audit.json` includes hashes of the source/data/checkpoint inputs and runtime versions; `data_support_comparison.json` preserves the exact comparison with the historical support snapshot.
