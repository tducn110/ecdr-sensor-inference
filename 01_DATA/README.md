# HAPT data foundations — current Week 2

Start with [Report A: HAPT Data Foundations](03_Week_2_Data_Foundations.tex) ([PDF](03_Week_2_Data_Foundations.pdf)). This independent English data study is aligned with the [frozen Week 1 K=12 contract](../00_OVERVIEW/week1/SOURCE_OF_TRUTH.md). Its eight units cover physical signals, windowing, labels, subject splits/support, normalization, dependence/quality, provenance, and the final data contract. It hands a defined normalized tensor and label/stream semantics to the separate [CNN Model Foundations study](../02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex); that report owns architecture, prediction metrics and certification obligations.

## Verified input contract

- HAPT raw accelerometer and gyroscope recordings: six channels, 50 Hz, 30 subjects, 61 recordings. Acceleration is in g; angular velocity in rad/s. The 561-feature tables are a separate dataset representation.
- Input windows are float32 `6 × 128`, stride 64, with no cross-recording slices. All 17,453 stored windows and ten manifest fields were rebuilt exactly; all 17,392 within-recording aligned overlaps match.
- Supervised policy is **K=12, pure or majority with dominant count / 128 ≥ 0.6**. This produces 12,278 examples across all splits. The historical K=6 pure-only choice is not the current reference.
- Full stream train/validation/test: **9,978 / 2,422 / 5,053**. Supervised: **6,955 / 1,751 / 3,572**. Subject groups are disjoint, **17 / 4 / 9**.
- Normalization is fit only on the 6,955 supervised training windows, pooling window/time axes with float64 accumulation, storing float32 means/scales, clipping std at `1e-6`, and producing float32 inputs. Shared raw samples receive repeated weight when they occur in selected overlapping windows.

## Evidence and reproduction

[Data findings](../output/week2_foundations_20261103/data_findings.md), [machine-readable audit](../output/week2_foundations_20261103/data_audit.json), [all aligned pairs](../output/week2_foundations_20261103/data_aligned_overlap_pairs.csv), and [historical support comparison](../output/week2_foundations_20261103/data_support_comparison.json) record the executed 3 November 2026 analysis.

```bash
.venv/bin/python scripts/week2_data_audit.py
```

The command reads raw HAPT and frozen artifacts and fails if required consistency assertions fail. It does not regenerate the frozen corpus or train a model. See the [Week 2 closeout](../output/week2_foundations_20261103/README.md) for final document QA.

## Limitations retained explicitly

The original annotation index origin is unresolved; changing by one sample changes 98 policy/label assignments. The historical class-support JSON does not reproduce the current source predicate: its broad total is 13,866 versus 12,278 supervised. The difference equals ambiguous support, but the artifact's generating provenance is not established. Transition undersupport keeps G2 HOLD. Overlap and within-subject dependence prohibit treating all windows as independent participants or interpreting descriptive windowwise intervals as population guarantees.

`ecdr/preprocessing.py` has a distinct float64/default-floor/identity contract. It is not a drop-in identity match for the selected reference normalizer.

## Historical source shelf

[Earlier data audit](01_03_DATA_AUDIT.md), [K=6 class decision](02_G2_CLASS_DECISION.md), [data analysis EN](04_ECDR_Data_Analysis_EN.tex), [data analysis VI](05_ECDR_Data_Analysis_VI.tex), [autocorrelation supplement](06_SUPP04_Autocorrelation.tex), and [dependence supplement](07_SUPP07_Statistical_Dependence.tex) remain useful historical or conceptual material. Their K=6 choices, old metrics and gate labels do not replace the current K=12 contract. Original numbered scripts are historical audit entry points; use the new Week 2 script for this frozen reproduction.

Continue to the [current model entry point](../02_MODEL/README.md).
