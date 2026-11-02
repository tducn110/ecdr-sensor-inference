# Week 1 ECDR Source Audit & SmallCNN1D K=12 Reference Pipeline Inventory

**Date:** 2026-11-02  
**Auditor:** Gemini 3.8 Flash (Source Audit Worker)  
**Workspace:** `/home/pro/Downloads/research`  
**Status:** COMPLETE (Read-only source inventory; zero edits to frozen files)

---

## 1. Executive Summary & Core Architectural Contrast

A critical architectural and scope dichotomy exists in the repository between the frozen baseline and the formal theory:
1. **Frozen Reference Backbone (`SmallCNN1D`):** Defined in [`scripts/backbone.py`](scripts/backbone.py#L59-L80) and [`configs/backbone.yaml`](configs/backbone.yaml#L10-L18). Architecture: Conv1D (6→32, k=5, pad=2) → BatchNorm1d → ReLU → MaxPool1d(2) → Conv1D (32→64, k=3, pad=1) → BatchNorm1d → ReLU → MaxPool1d(2) → Conv1D (64→64, k=3, pad=1) → BatchNorm1d → ReLU → Flatten(2048) → Linear(2048→128) → ReLU → Linear(128→K).
   - **K=6 Parameter Count:** 282,918 parameters ([`data/g2_model_summary.json`](data/g2_model_summary.json#L12)).
   - **K=12 Parameter Count:** 283,692 parameters (+774 weights/biases in final Linear head).
2. **Theoretical Certification Model (`ECDRConv1DNet`):** Defined in [`ecdr/model.py`](ecdr/model.py#L114-L165) and tested in [`tests/test_model_spec.py`](tests/test_model_spec.py#L14-L63). Architecture: Valid convolutions (pad=0), no BatchNorm, no MaxPool: Conv1 (6→32, k=5) → ReLU → Conv2 (32→64, k=3) → ReLU → Conv3 (64→64, k=3) → ReLU → Global Average Pooling (GAP over 120 positions) → Affine head (64→K).
   - **Parameter Count:** $19,552 + 65 \times K = 20,332$ parameters for K=12 ($19,942$ for K=6).
3. **Research Contract Rule:** Do not conflate the frozen reference `SmallCNN1D` (282k/283k params with BN/MaxPool) with the theoretical `ECDRConv1DNet` (20k params with GAP). The new pipeline must instantiate `SmallCNN1D` with $K=12$ in complete isolation, leaving [`scripts/backbone.py`](scripts/backbone.py) and [`configs/backbone.yaml`](configs/backbone.yaml) strictly untouched.

---

## 2. Verified Current Facts & Exact Source Evidence

### 2.1 Model Dimensions & Forward Contracts
- **Input Tensor Dimensions:** Shape $(B, 6, 128)$, representing $C=6$ channels (3-axis body acc, 3-axis gyro) and $W=128$ time steps ($2.56\text{ s}$ @ $50\text{ Hz}$).
- **Normalization Contract ($\mathcal{P}$):** Formula $Z_{c, i} = (X_{c, i} - \mu_c) / \sigma_c$. Defined in [`scripts/backbone.py`](scripts/backbone.py#L45-L57) and [`ecdr/preprocessing.py`](ecdr/preprocessing.py#L28-L160).
  - **Rule:** $\mu, \sigma$ are fit strictly on **TRAIN windows only** ([`configs/training.yaml`](configs/training.yaml#L14), `G2_CLASS_DECISION.md` §6.1). Val and test splits strictly reuse frozen statistics.
  - Scale clamping $\sigma_c = \max(\sigma_c, \sigma_{\min})$ with $\sigma_{\min} \in [10^{-6}, 10^{-5}]$ to prevent division by zero.
- **Model Identity / Hash Binding:** Implemented in [`ecdr/preprocessing.py`](ecdr/preprocessing.py#L82-L87) (`prep_id` SHA-256) and [`ecdr/model.py`](ecdr/model.py#L225-L236) (`model_id` SHA-256 over raw parameter bytes).

### 2.2 Data Splits, Windowing & Indexing Conventions
- **Windowing Source:** [`scripts/03_build_windows.py`](scripts/03_build_windows.py#L18-L62). Windows are generated per recording `exp` (never cross recording boundaries) with $W=128, S=64$.
- **Window Manifest & Indexing Trap:** [`data/window_manifest.parquet`](data/window_manifest.parquet) has 17,453 rows.
  - **Crucial Invariant:** `win_idx` restarts at 0 for every `exp`. Directly indexing `data/windows.npz` with `win_idx` causes silent corruption.
  - **Solution:** A stable `global_row_id = np.arange(len(df))` must be assigned before any filter, sort, or merge operation, as verified in [`ecdr/data.py`](ecdr/data.py#L38-L43).
  - Split join must validate 1:1 on `['exp', 'win_idx']` ([`ecdr/data.py`](ecdr/data.py#L47-L54)).
- **Subject-Disjoint Splits:** [`scripts/04_audit_splits.py`](scripts/04_audit_splits.py#L18-L70) & [`data/split_audit.json`](data/split_audit.json):
  - **Train (17 subjects):** `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26]` → 9,978 windows.
  - **Val (4 subjects):** `[27, 28, 29, 30]` → 2,422 windows.
  - **Test (9 subjects):** `[2, 4, 9, 10, 12, 13, 18, 20, 24]` → 5,053 windows.
  - Total full-stream corpus: **17,453 windows** (`FULL_STREAM_DENOMINATORS` in [`ecdr/schema.py`](ecdr/schema.py#L49-L54)).

### 2.3 Class Map & Support Behavior (K=6 vs K=12)
- **Class Map (12 HAPT classes, 1-based in raw labels, 0-based in code):**
  - Basic Activities (1..6): `WALKING` (0), `WALKING_UPSTAIRS` (1), `WALKING_DOWNSTAIRS` (2), `SITTING` (3), `STANDING` (4), `LAYING` (5).
  - Postural Transitions (7..12): `STAND_TO_SIT` (6), `SIT_TO_STAND` (7), `SIT_TO_LIE` (8), `LIE_TO_SIT` (9), `STAND_TO_LIE` (10), `LIE_TO_STAND` (11).
- **K=12 Pure-Only Support Reality:**
  - `G2_CLASS_DECISION.md` (§1-§3) and [`ecdr/data.py`](ecdr/data.py#L197-L255) establish that pure K=12 has **18 criteria blockers** (<50 windows or <3 subjects per split).
  - Worst case: Class 8 (`SIT_TO_STAND`) has only 5 train, 1 val, and 6 test pure windows.
  - Pure+majority policy (excluding ambiguous) still has **10 criteria blockers** on transitions.
  - **Status:** Pure K=12 is on **HOLD**. Any K=12 training must be isolated and flagged with explicit status caveats.

---

## 3. Comparison: Requested Reference vs Current State

| Attribute | Frozen Baseline (G2-B) | Theoretical Model (Week 2/3) | Requested SmallCNN1D K=12 Baseline |
|---|---|---|---|
| **Class Symbol** | `SmallCNN1D` ([`scripts/backbone.py`](scripts/backbone.py#L59)) | `ECDRConv1DNet` ([`ecdr/model.py`](ecdr/model.py#L114)) | `SmallCNN1D` (imported from `scripts/backbone.py`) |
| **Output Classes ($K$)** | 6 (pure only) | 12 (or arbitrary $K$) | 12 (full HAPT label set) |
| **Pooling / Regularization** | 2x MaxPool1d(2), 3x BatchNorm1d | No MaxPool, No BatchNorm, GAP(120) | 2x MaxPool1d(2), 3x BatchNorm1d |
| **Classifier Head** | Linear(2048, 128) → Linear(128, 6) | Linear(64, 12) (Affine) | Linear(2048, 128) → Linear(128, 12) |
| **Parameter Count** | 282,918 | 20,332 | 283,692 |
| **FLOPs / MACs** | ~2.55 M MACs | 2,343,936 MACs | ~2.55 M MACs |
| **Formal Verifiability** | Difficult (MaxPool + BatchNorm) | Native (Piecewise affine, IBP/CROWN) | Reference empirical baseline only |
| **Governance Status** | Frozen GO (`backbone_best.pt`) | Theoretical target | Isolated experimental reference |

---

## 4. Pipeline Modules, Exact Commands & Prerequisites

### 4.1 Modules to Add (Zero Edits to Frozen Files)
1. **Isolated Training & Eval Script:** `scripts/week1_reference_pipeline.py`
   - Imports `SmallCNN1D` from `scripts/backbone.py` without modifying `scripts/backbone.py`.
   - Passes `num_classes=12` and modified configuration dictionary.
   - Loads dataset via `ecdr.data.load_supervised_dataset(..., num_classes=12)`.
   - Evaluates full test stream (5,053 windows) and supervised subsets.
2. **Isolated Configuration:** `configs/reference_k12.yaml` (cloned from `training.yaml` and `backbone.yaml` with $K=12$).
3. **Artifact Output Locations:**
   - Checkpoints: `checkpoints/smallcnn1d_k12/backbone_seed{42,43,44}.pt` and `backbone_best.pt`.
   - Training log: `data/smallcnn1d_k12_training.json`.
   - Evaluation metrics: `output/week1_closeout_20261102/reference_k12_metrics.json`.

### 4.2 Exact Commands & Prerequisites
- **Prerequisites:** `data/windows.npz`, `data/window_manifest.parquet`, `data/windows_with_split.parquet` present and verified.
- **Run Command:**
  ```bash
  python3 scripts/week1_reference_pipeline.py \
    --seeds 42 43 44 \
    --epochs 30 \
    --batch-size 128 \
    --label-policy majority \
    --out-dir output/week1_closeout_20261102/reference_k12
  ```

---

## 5. Concrete Risks & Unresolved Evidence Boundaries

1. **Class 8 Transition Starvation:** Under pure policy, validation macro-F1 is dominated by a single window of Class 8 (`SIT_TO_STAND`) from subject 27. Early stopping on `val_macro_f1` will be highly volatile unless pure+majority (dominant $\ge 0.6$) is used with explicit caveats.
2. **Ambiguous Window Contamination:** Ambiguous windows ($<60\%$ label agreement) and unlabeled windows ($20.6\%$ of stream) must NEVER be assigned dummy classes or included in supervised cross-entropy loss.
3. **Verification Fallacy:** Heuristic reuse, test agreement, or cache-hit rate on consecutive windows must **never** be reported as "certified decision reuse" or "formal speedup." Only verified bounding intervals constitute formal certification.
4. **Denominator Ambiguity:** All reports must explicitly distinguish the **5,053 full test stream** windows (for throughput/replay) from the supervised labeled test subset (for accuracy and macro-F1).
