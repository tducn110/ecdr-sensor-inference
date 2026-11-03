# Week 2 model audit: HAPT and the frozen K=12 SmallCNN1D

## Scope and conclusion

This audit reconstructs the frozen Week 1 K=12 `SmallCNN1D` from its selected checkpoint, evaluates it on the existing HAPT test stream, and records its layer shapes, parameter count, receptive field, arithmetic count, point-decision margins, and a narrowly scoped CPU forward timing. It does not train or alter a checkpoint, estimate a candidate-region radius, run a certificate verifier, or measure ECDR end-to-end cost.

The selected reference is `week1_ref_1793620216_9d6c1d85`, seed 43, best epoch 8. The checkpoint SHA-256 is `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`, matching `selection.json`. Its fixed preprocessing ID is `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`. Reconstructed logits on all 5,053 chronological test windows have maximum absolute difference **0.0** from `test_predictions.npz`; argmax differs on **0/5,053** windows, and the CSV prediction manifest agrees in row order, class IDs/names, and winning margins within the declared `atol=1e-5, rtol=1e-5` margin tolerance. The exact normalization arrays and their preprocessing identity were also checked. This verifies a reproduced point-decision trace against frozen artifacts; it does not establish robustness around any input.

## Model structure and computational foundations

The selected model consumes six normalized channels over 128 samples (2.56 seconds at 50 Hz) and emits 12 logits. It uses three same-padded Conv1d layers with channel widths 6→32→64→64 and kernels 5, 3, 3; each convolution is followed by BatchNorm1d and ReLU, with 2× max pooling after the first two blocks. The resulting 64×32 activation is flattened to 2,048 values, followed by Linear 2,048→128, ReLU, and Linear 128→12.

There are **283,692 trainable parameters**. Conv1d parameter counts, including their biases, are 992, 6,208, and 12,352. BatchNorm contributes 64 + 128 + 128 trainable scale/offset parameters. The first Linear contributes 262,272 and the output Linear 1,548. BatchNorm running mean, running variance, and `num_batches_tracked` buffers were present for all three normalization layers. The model was in evaluation mode; two repeated batch-1 forwards produced bitwise identical logits.

For receptive field, let `r` be the input span and `j` the spacing between adjacent features, initially `r=j=1`. Each convolution updates `r ← r+(k−1)j`; each 2-wide pool updates `r ← r+j` and `j ← 2j`. The sequence gives:

| Operation | Receptive field (samples) | Feature jump (samples) |
|---|---:|---:|
| Conv k=5 | 5 | 1 |
| Pool k=2, s=2 | 6 | 2 |
| Conv k=3 | 10 | 2 |
| Pool k=2, s=2 | 12 | 4 |
| Conv k=3 | 20 | 4 |

Thus the final convolutional feature has a **20-sample receptive field** (nominally 0.40 seconds at 50 Hz) and a **4-sample output stride**. This theoretical interior receptive field uses the declared same-padding convolutions and valid 2× pools; boundary positions include padding effects.

The arithmetic estimate counts one multiply-accumulate (MAC) per convolution or Linear weight application, excludes biases, and omits BatchNorm, ReLU, max-pooling, memory traffic, and framework overhead. At batch size one, Conv1d contributes 122,880 + 393,216 + 393,216 MACs; the Linear layers contribute 262,144 + 1,536. Total: **1,172,992 MACs per window** under this convention. It is an operation count, not a latency prediction.

As a static architecture reference, `ECDRConv1DNet(num_classes=12)` has 20,332 trainable parameters in this checkout versus 283,692 for this selected SmallCNN1D. These are different architectures and functions; the parameter comparison does not imply matched accuracy, equivalent outputs, or a speed ratio.

## Test-stream decisions and labeled accuracy

On the full test stream, reconstructed model predictions matched stored reference predictions on **5,053/5,053** windows. The full stream includes unlabeled and ambiguous windows. For ground-truth classification metrics, the audit separately applies the locked supervised policy (`pure` or `majority`, class IDs 1–12) to obtain **3,572** labeled examples. Accuracy is **0.8905375** (3,181/3,572 correct; 391 errors) and macro-F1 is **0.7762645**. Both recomputed values match the selected seed's `evaluation_results.json` values exactly at the stored precision.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| WALKING | 531 | 0.9603 | 0.9567 | 0.9585 |
| WALKING_UPSTAIRS | 537 | 0.9331 | 0.9088 | 0.9208 |
| WALKING_DOWNSTAIRS | 492 | 0.9237 | 0.9837 | 0.9528 |
| SITTING | 546 | 0.7997 | 0.8407 | 0.8196 |
| STANDING | 597 | 0.8798 | 0.8090 | 0.8429 |
| LAYING | 586 | 0.9831 | 0.9949 | 0.9890 |
| STAND_TO_SIT | 45 | 0.7619 | 0.7111 | 0.7356 |
| SIT_TO_STAND | 32 | 0.7778 | 0.6563 | 0.7119 |
| SIT_TO_LIE | 51 | 0.5893 | 0.6471 | 0.6168 |
| LIE_TO_SIT | 46 | 0.7714 | 0.5870 | 0.6667 |
| STAND_TO_LIE | 68 | 0.5714 | 0.5294 | 0.5496 |
| LIE_TO_STAND | 41 | 0.4737 | 0.6585 | 0.5510 |

The large classes have stronger F1 values (0.8196–0.9890) than the transition classes, whose supports are 32–68 and F1 values are 0.5496–0.7356. The confusion matrix records the specific observed transitions: SITTING→STANDING occurs 57 times, STANDING→SITTING 106 times, and STAND_TO_LIE→SIT_TO_LIE 20 times. These are empirical counts on this split, not population error rates. In particular, the 391 ground-truth errors do not conflict with zero disagreement against the frozen model: those quantities answer different questions.

## Point margins, candidate regions, and timing boundary

The reported margin is the largest logit minus the second-largest logit at each test point. It is a nonnegative **pointwise decision gap**, not the correct-class margin for every point and not a bound on logit change under perturbation. Over all 5,053 stream windows, its minimum is **0.0001135**, 5th percentile **0.4120**, median **5.9782**, 95th percentile **11.8876**, and maximum **21.4685**. A small point gap identifies close competing logits at that input; a large gap alone does not guarantee the prediction remains unchanged nearby.

**Candidate-region status: not computed / unavailable in this audit.** There is no candidate-region radius, region-specific lower-bound margin, or certified neighborhood result among these outputs. The empirical point gaps must not be presented as such a result. Formal robustness/certification remains outside this model audit.

Forward-only CPU timing used the frozen model in eval mode, one Torch thread, batch size one, the first chronological test window after frozen normalization (same fixed tensor for every timed call), 100 warm-up calls, then 1,000 calls measured with `perf_counter_ns`. The final recorded run reported p50 **0.1557 ms**, p95 **0.3111 ms**, and mean **0.1892 ms**. It is specific to the runtime and hardware recorded in `model_audit.json`. It excludes window extraction, normalization, event detection, candidate membership/certificate calculations, fallback, data movement, and I/O; it is not end-to-end latency or evidence of net speedup.

## Reproduction and artifacts

From the repository root, reproduce the package with:

```bash
.venv/bin/python scripts/week2_model_audit.py
```

The executable audit is [scripts/week2_model_audit.py](../../scripts/week2_model_audit.py). It writes `model_audit.json`, per-layer shapes, per-class metrics, the confusion matrix and supervised test-row data as CSV, plus the confusion, point-margin, and MAC plots beside this report. The JSON contains its UTC execution timestamp (`2026-11-03T00:21:32Z`), script SHA-256 (`66eb8d8c206d2c72079893a0abf2e66110e6f5a6df4ea1c9cce5f44312d0c81e`), hashes for the model/data/evaluation inputs, 18 named verified checks, environment/timing details, layer/buffer records, numerical tolerances, and the exact point metrics. The run log is `model_audit_run.log`. The source of truth for locked Week 1 semantics is [00_OVERVIEW/week1/SOURCE_OF_TRUTH.md](../../00_OVERVIEW/week1/SOURCE_OF_TRUTH.md); implementation is `scripts/backbone.py:SmallCNN1D`; selected checkpoint and comparison artifacts are under `output/week1_closeout_20261102/reference/`.

Evidence status: model loading, frozen artifact identity, normalization identity, logits/predictions/manifest comparison, supervised metrics, eval repeatability, shapes, parameter count, receptive-field arithmetic, MAC arithmetic, and the stated local timing were executed by this audit. The existing Week 1 record reports its own independent baseline artifact verification as 65/65 checks; that is supporting provenance, not a formal ECDR certificate. No candidate-region analysis, formal soundness result, net-cost evaluation, retraining comparison, or user acceptance is established here.
