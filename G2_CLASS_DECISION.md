# G2 CLASS DECISION — K for backbone training (G2-A output)

> Date: 2026-10-15 · Evidence: `data/g2_class_support.json` (script `scripts/06_audit_class_support.py`)
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
| 9 | SIT_TO_LIE | 36 | 16 | 25 | 16/4/8 | low train+val |
| 10 | LIE_TO_SIT | 30 | 6 | 18 | 14/3/9 | low train+val(+test<50) |
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
Max theoretical capacity is 39–112 windows/class; realized = capacity.
The class-8 (SIT_TO_STAND) worst case: 12 windows total.

Adding majority-labeled transition windows (+563, total 917) still fails the
>=50/split bar for val/test and imports boundary label noise — rejected for now.

## 4. Evaluation denominators (report BOTH, always)

- Full stream: **17,453** windows · Labeled (pure+majority+ambiguous): **13,866** (79.4%) · Unlabeled: **3,587** (20.6%)
- Transition-labeled windows (maj in 7–12): **1,111** (6.4%) — out of scope for a K=6 classifier
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
2. **Model selection**: validation only. Test participates exactly once per
   reported run, after checkpoint selection.
3. **Seeds**: >=3 seeds; report mean +/- spread for macro-F1 and accuracy.
   Log Python + PyTorch + CUDA versions (`requirements-lock.txt` re-exported
   after torch install). PyTorch docs: identical seeds do NOT guarantee
   identical results across versions/hardware — state the environment.
4. **Checkpoint identity**: backbone weights frozen at certificate creation;
   verifier MUST use the same checkpoint. Any weight change invalidates all
   certificates (anchor invalidation + model version pin live in G3).
5. **Metrics**: accuracy, macro-F1, per-class recall, confusion matrix, logit
   margin distribution (prerequisite for G3 bounds), parameter count, and CPU
   inference latency p50/p95 — the latency number is a **baseline for G5**, not
   a saving claim.
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
  g2_metrics.json        (+ plots/g2_confusion_matrix.png, plots/g2_margins.png)
```

## 8. Traceability

- Gate wording for the paper: `G0 PASS; G1 COMPLETE_WITH_REVISIONS; G2 PASS (G2-A + G2-B); G3 pending`.
- Baseline commit `cefd171` preserves pre-fix G1 artifacts; the `05` index fix
  and recomputed audits are committed separately.

## 9. G2-B results (measured 2026-10-15)

- Windows: train 5,693 / val 1,450 / test 2,874 (pure K=6; policy per §1).
- 3 seeds (42/43/44): val macro-F1 0.9418 / 0.9454 / 0.9504;
  test macro-F1 0.8906 / 0.9105 / 0.9354 → **mean 0.9122 ± 0.0183**.
- Best (seed44, selected on val): test acc **0.9349**, macro-F1 **0.9354**;
  per-class F1: LAYING 1.000, WALKING 0.982, DOWNSTAIRS 0.961, UPSTAIRS 0.934,
  STANDING 0.878, SITTING 0.858 (sitting/standing confusion dominant).
- Margin (test, best): p50 +9.84, p5 −0.73, frac<0 = 0.0651 = 1 − accuracy.
- Params 282,918 · shape test PASS · CPU p50/p95 = 0.236/0.331 ms (batch1, 6 threads).
- Artifacts: `configs/*.yaml`, `checkpoints/backbone_best.pt`, `data/g2_training.json`,
  `data/g2_metrics.json`, `data/g2_model_summary.json`, `data/g2_cpu_latency.json`,
  `plots/g2_confusion_matrix.png`, `plots/g2_margin_distribution.png`.

