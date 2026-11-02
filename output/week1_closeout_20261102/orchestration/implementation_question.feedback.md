### 1. Current Verified Facts (from Codebase & Configs)

1. **Architecture & Parameter Count:**
   - [`SmallCNN1D`](file:///home/pro/Downloads/research/scripts/backbone.py#L52-L73) in `scripts/backbone.py` implements 3 `Conv1d` layers (channels `[32, 64, 64]`, kernels `[5, 3, 3]`, `same` padding) followed by `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` (after conv1 and conv2), flattening to `64 * 32 = 2048`, followed by an MLP head: `Linear(2048, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128, num_classes)`.
   - Analytical parameter count:
     - Conv block: $992 + 64 + 6,208 + 128 + 12,352 + 128 = 19,872$ parameters.
     - Linear 1: $2048 \times 128 + 128 = 262,272$ parameters.
     - Linear 2 ($K=12$): $128 \times 12 + 12 = 1,548$ parameters.
     - Total for $K=12$: **283,692 parameters** (exact match with proposed spec).
   - `configs/backbone.yaml` hardcodes `num_classes: 6`, `label_policy: pure_only`, and is explicitly locked by G2-B protocol; existing checkpoints in `checkpoints/` are $K=6$ models.

2. **Windowing & Sample Alignment in `scripts/03_build_windows.py`:**
   - Raw recordings are sampled at 50 Hz. Window length $W=128$, stride $S=64$. Windows are strictly within-recording (`range(0, T - W + 1, S)`) and never cross recordings.
   - **Sample Indexing & Off-by-One Risk:** HAPT `labels.txt` contains 1-based sample boundaries `[start_sample, end_sample]`. In `scripts/03_build_windows.py` (lines 34–36):
     ```python
     plabel = np.zeros(T + 1, dtype=np.int16)
     for r in per_rec.get(exp, []):
         plabel[int(r[3]):int(r[4]) + 1] = int(r[2])
     ```
     Because `plabel` has length $T+1$ and the raw signal `sig` has length $T$ indexed from $0$, the slice `plabel[s:e]` (where $s \in [0, T-W]$) associates signal row $0$ with `plabel[0]` (which is permanently $0$ / unannotated). This encodes an explicit 1-sample right shift relative to 0-based signal arrays.
   - **Label Policy Classification:**
     - `pure`: exactly 1 unique label and `dom_frac == 1.0` (all 128 samples have that label).
     - `majority`: `dom_frac >= 0.6` (at least 77 samples have the dominant label `dom`, but may contain unannotated samples or secondary labels).
     - `ambiguous`: has annotations but `dom_frac < 0.6`.
     - `unlabeled`: 0 annotated samples in the window (`maj = 0`).

3. **Window Indexing & Join Convention:**
   - In `scripts/03_build_windows.py`, `windows.npz` (`x`) is saved in the exact insertion order of `window_manifest.parquet`.
   - `window_manifest.parquet` does **not** have an explicit persistent `global_row_id` column; row index in parquet is implicitly array index in `windows.npz`.
   - In `scripts/backbone.py` (`load_pure_k6`), a left merge with `windows_with_split.parquet` is performed on `["exp", "win_idx"]`. If row order shifts or index is reset post-merge, the link to `windows.npz['x']` can be corrupted if not tracked explicitly before merging.

4. **Split Denominators in `ecdr/schema.py`:**
   - Full stream window counts: `train: 9978`, `val: 2422`, `test: 5053`, `total: 17453`.

---

### 2. Alignment & Indexing Convention Encoded by the Data Builder

1. **Per-recording Window Grid:** For an experiment with $T$ samples, window $i$ covers signal indices $[64i, 64i + 128)$.
2. **Label Interval Semantics:** `labels.txt` indices $[r_3, r_4]$ are treated as 1-based closed intervals, written into `plabel[r_3 : r_4 + 1]`.
3. **No Cross-recording Windows:** Every window belongs entirely to a single `(exp, user)`. Window index `win_idx` resets to $0$ at each recording.
4. **Primary Key:** The composite key `(exp, win_idx)` uniquely identifies a window across `window_manifest.parquet`, `windows_with_split.parquet`, and `windows.npz`.

---

### 3. Necessary Source-Data Slice Checks

Before executing the pipeline, the following assertions must run directly against `data/windows.npz`, `data/window_manifest.parquet`, and `data/windows_with_split.parquet`:

1. **Global Row Alignment Check:**
   - Verify `len(windows.npz['x']) == len(window_manifest.parquet) == 17453`.
   - Assign `df_manifest["global_row_id"] = np.arange(len(df_manifest), dtype=np.int64)` **before** joining with `windows_with_split.parquet`.
   - Verify that `windows_with_split.parquet` contains exactly 17,453 rows and merges $1:1$ on `["exp", "win_idx"]` without row loss, duplication, or reordering.

2. **Split Denominator & Label Coverage Check:**
   - Assert split window counts match `ecdr/schema.py`: Train = 9,978, Val = 2,422, Test = 5,053.
   - Check that `label_majority` for supervised windows is strictly within $[1, 12]$ (mapped to zero-based $[0, 11]$).
   - Check the exact window counts for `pure + majority` training/val sets to ensure sufficient support for transition classes (IDs 7–12).

3. **Signal Array Shape & Type Integrity:**
   - Assert `x.shape == (17453, 6, 128)` and `x.dtype == np.float32`.
   - Check for `NaN`, `Inf`, or flat-zero channels across all windows.

---

### 4. Critical Errors That Could Invalidate the Experiment

1. **Test-Set Data Leakage:**
   - Fitting normalization statistics (`mean`, `std`) on all windows or including validation/test splits. Normalization **must** strictly use `fit_norm(x_train_supervised)`.
   - Evaluating test windows or peeking at test metrics during seed/epoch selection. Model selection must rely solely on validation macro-F1.

2. **Evaluation Denominator Ambiguity (Fullstream vs Supervised Test):**
   - In fullstream test ($5,053$ windows), many windows are `unlabeled` ($maj=0$) or `ambiguous` ($dom\_frac < 0.6$).
   - **Fatal error:** Including unlabeled windows in the 12-class confusion matrix, precision, recall, or macro-F1 calculations will corrupt metrics.
   - Confusion matrix, Wilson score recall intervals, and macro-F1 must be evaluated on the well-defined supervised subset (`pure + majority`, $maj \in [1, 12]$) of the test split, while stream continuity / adjacent prediction agreement operates across the full 5,053 stream.

3. **Adjacent Agreement Across Recording Gaps:**
   - Consecutive test windows $i$ and $i+1$ where `exp[i] != exp[i+1]` or `win_idx[i+1] != win_idx[i] + 1` are not temporally adjacent.
   - Calculating transition agreement across recording boundaries creates synthetic mismatch artifacts. Boundary gaps must be masked out.

4. **Label Mapping Mismatch:**
   - Mapping $1$-based raw labels ($1..12$) to zero-based indices ($0..11$) inconsistently between training and inference (e.g., leaving label 0 in the supervised target).

5. **Certification Overclaim:**
   - Point margins ($g = f_y - \max_{j \neq y} f_j$) must remain empirical point observations. Conflating point logit margins with formal robust certification violates G2 constraints.

---

### 5. Architectural Conflicts & Resolution Recommendations

1. **Codebase Preservation:**
   - **Do not modify** `scripts/backbone.py` or `configs/backbone.yaml`.
   - `week1_reference_pipeline.py` should import `SmallCNN1D` from `scripts.backbone` and instantiate it with `SmallCNN1D(num_classes=12, cfg=backbone_cfg.get("model", {}))`.
   - Keep all outputs (checkpoints, metrics JSON, run logs) isolated under a dedicated folder (e.g., `checkpoints/week1_k12/` or `results/week1/`).

2. **Deterministic & Isolated Execution:**
   - Enforce single-thread CPU execution (`torch.set_num_threads(1)`) for latency profiling.
   - Record SHA-256 hashes of input data files (`windows.npz`, `window_manifest.parquet`, `windows_with_split.parquet`) and source scripts in the output metadata for strict provenance.

