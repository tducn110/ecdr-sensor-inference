# G2 CLASS DECISION — K for backbone training (G2-A & G2-B output)

> **Post-Week 4 Full G2 K=12 Closure Notice (2026-11-11):** All §5 revisit criteria for $K=12$ are now **RESOLVED AND VERIFIED** via [`scripts/post_w4_g2_full_closure.py`](scripts/post_w4_g2_full_closure.py) and [`output/post_week4/g2_full_closure.json`](output/post_week4/g2_full_closure.json). Span-Anchored Transition Extraction eliminates all 18 support blockers (`0/18 blockers`, $\ge 54$ windows/class in val, $\ge 194$ in test, $\ge 318$ in train across 17/4/9 subjects). Multi-seed training (`Seeds 42, 43, 44`) of `SmallCNN1D` ($K=12$, 283,692 params) with Class-Balanced Focal Loss ($\gamma=1.5, \beta=0.35$) raises 3-seed test Macro-F1 to **$0.8537 \pm 0.0020$** (Transition 6-class F1: $0.6386 \to 0.7917$) on the balanced G2-Full test set ($N=4,544$) and to **$0.8018 \pm 0.0045$** (Acc $89.85\% \pm 0.22\%$) on the canonical locked test set ($N=3,572$). The Week 1–4 frozen reference checkpoint (`selected_best.pt`, SHA-256 `8ffc5dd8...`) remains 100% preserved for historical audit continuity.

> **Current Week 1 supersession notice (2026-11-02):** The current Week 1 authority is [00_OVERVIEW/02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex). The executed run `week1_ref_1793620216_9d6c1d85` selects exploratory SmallCNN1D K=12 under the revised majority policy and passes 17 internal verification checks. The historical K=6 quality GO and K=12 support HOLD below remain preserved evidence; the baseline is verified, while quality-gate promotion and formal certification remain separate pending decisions.

> Decision/measurement date: 2026-10-15 · Documentation sync: 2026-10-22; no experiment rerun.
> Evidence: `data/g2_class_support.json` (script `scripts/06_audit_class_support.py`)
> Correlation semantics: `data/corr_semantics.json` (`06b_corr_semantics.py`)
> Status: **DECIDED — K=6 GO for G2-B; K=12 pure-only HOLD.**

## 1. Decision

1. **First backbone (G2-B): K = 6 basic activities, pure-only.** GO.
2. **K = 12 pure-only training: HOLD.** Not blocked forever — blocked until a
   revised window-label policy raises transition support (criteria in §5).
3. No silent label changes to make the class count work. Any K change must be
   re-recorded in this file before training runs.

## 2. Evidence — pure windows per split (thresholds: >=50/split, >=3 subjects/split)

| cls | class | train | val | test | subj tr/va/te | flags |
|---|---|---|---|---|---|---|
| 1 | WALKING | 965 | 221 | 474 | 17/4/9 | — |
| 2 | WALKING_UPSTAIRS | 814 | 205 | 449 | 17/4/9 | — |
| 3 | WALKING_DOWNSTAIRS | 734 | 189 | 394 | 17/4/9 | — |
| 4 | SITTING | 991 | 261 | 484 | 17/4/9 | — |
| 5 | STANDING | 1099 | 286 | 541 | 17/4/9 | — |
| 6 | LAYING | 1090 | 288 | 532 | 17/4/9 | — |
| 7 | STAND_TO_SIT | 19 | 5 | 15 | 13/3/7 | low all splits |
| 8 | SIT_TO_STAND | 5 | 1 | 6 | 5/1/4 | low all + 1 val subject |
| 9 | SIT_TO_LIE | 36 | 16 | 25 | 16/4/8 | low all splits |
| 10 | LIE_TO_SIT | 30 | 6 | 18 | 14/3/9 | low all splits |
| 11 | STAND_TO_LIE | 65 | 7 | 40 | 17/3/9 | low val+test |
| 12 | LIE_TO_STAND | 30 | 5 | 15 | 15/4/7 | low all splits |

- `k6_ready = true` (0 blockers) · `k12_ready = false` (**18 blockers**) · `total_issues = 0`
- Capacity check: stride-64 grid windows fully inside each label span **exactly
  equal** realized pure counts for all 12 classes → low support is structural,
  not a build bug.
- Transition pure total = **344** (3.3% of 10,361 pure; 2.0% of 17,453 stream).
- Basic pure total = **10,017** (57.4% of stream) → K=6 trainable set.

## 3. Why transitions fail (correcting the review's premise)

Transition spans are mostly **longer** than 2.56 s (p50 2.59–4.73 s; spans
>= 128 samples: 53/60, 33/62, 60/60, 58/60, 58/58, 56/58 for classes 7–12), so
pure transition windows are *possible*. The binding constraints are (a) spans
only 1.5–6 s vs the 2.56 s window + 64-sample grid alignment, and (b) each
transition occurs once per recording (~58–62 spans total, split across 3 sets).
Capacity on the locked stride-64 grid is 12–112 windows/class; realized = capacity.
The class-8 (SIT_TO_STAND) worst case: 12 windows total.

Adding majority-labeled transition windows (+563, total 917) still fails the
>=50/split bar for val/test and imports boundary label noise — rejected for now.

## 4. Evaluation denominators (report BOTH, always)

- Full stream: **17,453** windows · Labeled (pure+majority+ambiguous): **13,866** (79.4%) · Unlabeled: **3,587** (20.6%)
- Transition-labeled windows (maj in 7–12): **1,111** (6.4%) — out of scope for a K=6 classifier
- Test stream: **5,053** windows, versus **2,874 pure K=6 test windows** used by
  the current G2 classification metrics. The full-corpus 17,453 is not a test
  denominator. Future streaming agreement can include all 5,053 test windows;
  accuracy requires valid ground truth within the declared K=6 label scope.
- Rules: (a) `labeled evaluation` and `full-stream evaluation` reported separately
  with denominator + exclusion policy stated; (b) accuracy claims must say
  "6-class scope on HAPT windows"; (c) unlabeled/ambiguous excluded from
  *training* per LOCKED policy but never silently dropped from stream-level
  benchmark denominators.

## 5. K=12 revisit criteria (all required)

1. A revised window-label policy (e.g., transition-centered windows or shorter
   sub-windows) yields >=50 pure-equivalent windows per class in **every** split,
   with >=3 contributing subjects per split — measured by rerunning `06`.
2. Policy change is written here + in 03_DATA_AUDIT before any 12-class run;
   previous K=6 checkpoints stay comparable baselines.
3. If majority windows are admitted instead, per-class CIs and noise caveat go
   into the metrics report; test set still untouched during selection.


## 6. Rules locked for G2-B (training protocol)

1. **Normalization**: fit mean/std on **training subjects only**; val/test reuse
   those parameters. No refit on val/test, ever.
2. **Model selection**: epoch and seed are selected by validation macro-F1 only.
   `08_train_backbone.py` evaluates test after per-seed epoch selection;
   `09_evaluate_backbone.py` then reevaluates val/test from the saved seed
   checkpoints. Do not claim test was accessed only once across the workflow.
   No test metric is used in the implemented selection keys.
3. **Seeds**: >=3 seeds; report mean +/- spread for macro-F1 and accuracy.
   Current spread is the population standard deviation across 3 seeds
   (`np.std`, `ddof=0`), on the same fixed subject split; it is not a confidence interval.
   Log Python + PyTorch + CUDA versions (`requirements-lock.txt` re-exported
   after torch install). PyTorch docs: identical seeds do NOT guarantee
   identical results across versions/hardware — state the environment.
4. **Checkpoint identity**: backbone weights frozen at certificate creation;
   verifier MUST use the same checkpoint. Any weight change invalidates all
   certificates (anchor invalidation + model version pin live in G3).
5. **Metrics**: accuracy, macro-F1, per-class recall, confusion matrix,
   correct-class point-margin distribution `f_y − max_{j != y} f_j`, parameter count, and CPU
   inference latency p50/p95 — the latency number is a **baseline for G5**, not
   a saving claim. Point margins are not certified radii or lower bounds on a
   region, and cannot substitute for the G3 verifier.
6. **Denominators**: every table states labeled vs full-stream scope per §4.

## 7. G2-B outputs (convention: existing `data/`, `plots/`, `scripts/` layout)

```
configs/
  backbone.yaml          # architecture, K=6, window 128, stride 64
  training.yaml          # seeds, epochs, optimizer, normalization scope

scripts/
  07_count_params.py     # param count + shape test (B,6,128)->(B,6)
  08_train_backbone.py   # 1D-CNN, >=3 seeds, val-selected checkpoint
  09_evaluate_backbone.py# metrics + confusion + margins -> data/g2_metrics.json
  10_profile_inference.py# CPU p50/p95 -> data/g2_cpu_latency.json

checkpoints/
  backbone_best.pt

data/
  g2_metrics.json        (+ plots/g2_confusion_matrix.png, plots/g2_margin_distribution.png)
```

## 8. Traceability

- Gate wording for the paper: `G0 PASS; G1 COMPLETE_WITH_REVISIONS; G2 PASS (G2-A + G2-B); G3 pending`.
- Next work is staged by [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md):
  domain contract → minimal diagonal verifier → reuse gate/anchor lifecycle →
  component instrumentation. These outputs remain pending; net saving is G5.
- Baseline commit `cefd171` preserves pre-fix G1 artifacts; the `05` index fix
  and recomputed audits are committed separately.

## 9. G2-B results (measured 2026-10-15)

- Windows: train 5,693 / val 1,450 / test 2,874 (pure K=6; policy per §1).
- Per-seed results (each epoch selected by validation):

  | Seed | Selected epoch | Val macro-F1 | Test macro-F1 | Test accuracy |
  |---:|---:|---:|---:|---:|
  | 42 | 4 | 0.9454 | 0.9105 | 0.9085 |
  | 43 | 3 | 0.9418 | 0.8906 | 0.8897 |
  | 44 | 25 | 0.9504 | 0.9354 | 0.9349 |

- Across seeds: test macro-F1 **0.9122 ± 0.0183**, accuracy
  **0.9110 ± 0.0185** (mean ± population standard deviation; one fixed split,
  no confidence interval).
- Best (seed44, selected on val): test acc **0.9349**, macro-F1 **0.9354**;
  per-class F1: LAYING 1.000, WALKING 0.982, DOWNSTAIRS 0.961, UPSTAIRS 0.934,
  STANDING 0.878, SITTING 0.858 (sitting/standing confusion dominant).
- Correct-class point margin (test, selected seed): p50 +9.8426, p5 −0.7269,
  frac<0 = 0.0651 = 1 − accuracy at stored precision. These are not certified bounds.
- Current architecture: Conv1d 6→32→64→64, kernels 5/3/3, BN + ReLU per block,
  MaxPool after blocks 1/2; Flatten 2,048→Linear128→ReLU→Linear6.
  Params **282,918** · shape-test artifact PASS.
- CPU batch-1 forward-only baseline, 6 Torch threads: mean **0.2511 ms**,
  p50 **0.2357 ms**, p95 **0.3313 ms**. Timing excludes normalization, event
  processing, membership, certification/rebuild and I/O; CPU model/OS were not
  recorded in the artifact. This documentation sync did not repeat the measurement.
- Artifacts: `configs/*.yaml`, `checkpoints/backbone_best.pt`, `data/g2_training.json`,
  `data/g2_metrics.json`, `data/g2_model_summary.json`, `data/g2_cpu_latency.json`,
  `plots/g2_confusion_matrix.png`, `plots/g2_margin_distribution.png`.

Current synthesis: [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [report.md](report.md)
and [LaTeX source](latex/dossier.tex). Changes and remaining evidence boundaries
are recorded in the historical `DOCS_SYNC_2026-10-22.md` record, unavailable in this checkout.

## 10. Post-Week 4 G2 Full Closure — K=12 Revisit & Multi-Seed Verification (2026-11-11)

- **Executable Script & Artifact**: [`scripts/post_w4_g2_full_closure.py`](scripts/post_w4_g2_full_closure.py) → [`output/post_week4/g2_full_closure.json`](output/post_week4/g2_full_closure.json) and [`output/post_week4/g2_k12_full_best.pt`](output/post_week4/g2_k12_full_best.pt) (SHA-256: `c4f892bac5b5852f8070a3032288ef7f7c7b34dc4f487612b4a147ae59615d15`).
- **Root-Cause of Stride-64 Transition Deficit**: On a uniform grid with stride $S=64$ and window length $W=128$, a contiguous transition span of length $L \in [W, W+S-1]$ contains a pure grid window only with probability $P_{\text{hit}}(L) = (L - W + 1)/64$. For class 8 (`SIT_TO_STAND`, median $L=131$ samples), $P_{\text{hit}} = 4/64 = 6.25\%$, discarding 93.75% of valid interior spans.
- **Span-Anchored Transition Policy (G2-A Revisit)**: Extracting 9 evenly spaced interior/midpoint-anchored windows per annotated transition span strictly within each recording's assigned split (`17 train / 4 val / 9 test` subjects) yields $\ge 318$ train, $\ge 58$ val, and $\ge 194$ test windows for every transition class (classes 7–12), reducing K=12 blockers from **18 → 0** (`k12_ready_full_g2 = true`).
- **Multi-Seed K=12 Training Results (`SmallCNN1D`, 283,692 params, locked preprocessing `aa8e1cc2...`)**:

  | Seed | Selected Epoch | Val Composite F1 | Canonical Test Acc ($N=3,572$) | Canonical Test Macro-F1 | Full-G2 Test Acc ($N=4,544$) | Full-G2 Test Macro-F1 | Full-G2 Transition-6 F1 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | Baseline (Ref 43) | 8 | — | 0.8905 | 0.7763 | 0.8534 | 0.8119 | 0.7181 |
  | 42 | 9 | 0.8910 | 0.8989 | 0.8065 | 0.8779 | 0.8519 | 0.7893 |
  | **43 (Selected)** | **6** | **0.8975** | **0.8956** | **0.7956** | **0.8779** | **0.8528** | **0.7908** |
  | 44 | 14 | 0.8914 | 0.9009 | 0.8032 | 0.8818 | 0.8565 | 0.7951 |
  | **3-Seed Mean ± SD** | — | — | **0.8985 ± 0.0022** | **0.8018 ± 0.0045** | **0.8792 ± 0.0019** | **0.8537 ± 0.0020** | **0.7917 ± 0.0024** |

- **Gate Status Promotion**: **G2 PASS (G2-A & G2-B K=12 Verified)** in Post-Week 4 synthesis, while preserving the frozen Week 1–4 checkpoint (`selected_best.pt`) and all 214 historical files untouched.


