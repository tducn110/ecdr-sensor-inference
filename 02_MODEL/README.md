# SmallCNN1D model foundations — current Week 2

Read [Report B: SmallCNN1D Model Foundations](01_ECDR_CNN_Model_Analysis_EN.tex) ([PDF](01_ECDR_CNN_Model_Analysis_EN.pdf)) after the independent [Report A: HAPT Data Foundations](../01_DATA/03_Week_2_Data_Foundations.tex). This model study has its own objective, source/inference-audit method, rationale for 1D CNN, full layer equations and tensor trace, resource accounting, all twelve class metrics, confusion matrix, point-margin analysis, conclusion and model contract for Week 3. The [64-concept audit](../output/week2_foundations_20261103/concept_audit_64.csv) and [content review](../output/week2_foundations_20261103/content_review.md) record the full disposition of the former theoretical exposition.

## Current frozen reference

The implementation is [scripts/backbone.py:SmallCNN1D](../scripts/backbone.py), instantiated using the [K=12 run configuration](../output/week1_closeout_20261102/reference/config.json). The class's default and the older loader helpers are historical K=6 behavior; the frozen K=12 pipeline supplies the explicit class count and its own locked supervised selection.

| Stage | Output shape excluding batch | Trainable parameters |
| --- | --- | ---: |
| Conv 6→32, k5, p2 + BN + ReLU | 32 × 128 | 1,056 |
| MaxPool2 | 32 × 64 | 0 |
| Conv 32→64, k3, p1 + BN + ReLU | 64 × 64 | 6,336 |
| MaxPool2 | 64 × 32 | 0 |
| Conv 64→64, k3, p1 + BN + ReLU | 64 × 32 | 12,480 |
| Flatten → Linear 2048→128 + ReLU | 128 | 262,272 |
| Linear 128→12 | 12 logits | 1,548 |
| **Total** | **12 logits** | **283,692** |

Run `week1_ref_1793620216_9d6c1d85` selected seed **43**, epoch **8**, by validation macro-F1. The [selected checkpoint](../output/week1_closeout_20261102/reference/selected_best.pt) has SHA-256 `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`. Preprocessing ID is `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

## Executed Week 2 verification

The [model audit](../output/week2_foundations_20261103/model_audit.json) and [findings](../output/week2_foundations_20261103/model_findings.md) reproduce all **5,053** test-stream logits exactly in the recorded CPU environment, with **0/5,053** argmax differences against stored outputs. Ground-truth metrics use **3,572** supervised windows: **3,181 correct**, accuracy **0.8905375**, macro-F1 **0.7762645**. [Per-class metrics](../output/week2_foundations_20261103/model_per_class_metrics.csv) and [confusion counts](../output/week2_foundations_20261103/model_confusion_matrix.csv) preserve all twelve classes.

The executed graph has 15 leaf modules, an interior final-feature receptive field of **20 samples** and feature jump **4**. Conv/Linear arithmetic totals **1,172,992 MACs/window**, excluding bias additions, BN, ReLU, pooling, memory traffic and framework overhead. Counts do not predict latency.

```bash
.venv/bin/python scripts/week2_model_audit.py
```

The audit performs inference, not training. Its CPU timing uses batch one, one thread, 100 warmups and 1,000 repetitions of one fixed resident normalized input. The JSON records actual run values. It is forward-only, not complete reference-path latency, ECDR cost, an energy measurement or proof of speedup. See the [closeout](../output/week2_foundations_20261103/README.md) for verified report builds and exact evidence hashes.

## Model boundaries and historical artifacts

- [ecdr/model.py:ECDRConv1DNet](../ecdr/model.py) is the distinct valid-convolution/ReLU/GAP model with **20,332 parameters** at K=12. Its GAP cancellation identities do not apply to this BN/Pool/Flatten–MLP reference. Its separate preprocessing contract must not be substituted by matching output width alone.
- The old [Vietnamese companion](02_ECDR_CNN_Model_Analysis_VI.tex), [copied model source](03_backbone_model.py), [K=6 config](04_backbone_config.yaml), [seed-44 checkpoint](05_checkpoint_seed44_best.pt), numbered G2 metrics, plots and training/profile scripts remain historical artifacts. Their **282,918 parameters**, **2,874 pure test windows** and roughly **0.24 ms** K=6 latency are not current K=12 evidence.
- The previous English GAP exposition is preserved in the [before-edit snapshot](../output/week2_foundations_20261103/before/02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex).
- G2 quality remains **HOLD** because transition support is weak. Pointwise winning margins, repeatable inference and adjacent prediction agreement do not constitute a whole-region certificate.

The next mathematical stage must bound **all eleven competing class margins over the declared region**, verify conservative membership, preserve model/preprocessing/anchor/domain identity, justify floating-point soundness and define reset/fallback behavior. The [domain contract](../07_G3_DOMAIN_CONTRACT.md) is a starting specification; it is not an implemented verifier for this reference.
