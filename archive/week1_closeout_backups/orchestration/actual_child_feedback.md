# Technical Code Review: Week 1 Reference Pipeline & Data Contract

**Review Target:** `/home/pro/Downloads/research`  
**Inspected Files:**
- [`scripts/week1_reference_pipeline.py`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py)
- [`scripts/backbone.py`](file:///home/pro/Downloads/research/scripts/backbone.py)
- [`configs/backbone.yaml`](file:///home/pro/Downloads/research/configs/backbone.yaml)
- [`ecdr/data.py`](file:///home/pro/Downloads/research/ecdr/data.py)

---

### Item (a): K=12 SmallCNN1D 283,692 parameters import and frozen files protected

- **Status:** **PASS (with architecture coupling caveat)**
- **Verification Details:**
  - In [`scripts/week1_reference_pipeline.py#L43`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L43), [`SmallCNN1D`](file:///home/pro/Downloads/research/scripts/backbone.py#L59-L82) is directly imported from `scripts.backbone`.
  - In [`scripts/week1_reference_pipeline.py#L295-L302`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L295-L302) and [`scripts/week1_reference_pipeline.py#L843-L847`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L843-L847), parameter counts are strictly checked against `EXPECTED_PARAMS = 283692`:
    - Conv1: $6 \times 32 \times 5 + 32 = 992$ params, BatchNorm1: $64$ params.
    - Conv2: $32 \times 64 \times 3 + 64 = 6208$ params, BatchNorm2: $128$ params.
    - Conv3: $64 \times 64 \times 3 + 64 = 12352$ params, BatchNorm3: $128$ params.
    - FC1: $(64 \times 32) \times 128 + 128 = 262272$ params.
    - Head: $128 \times 12 + 12 = 1548$ params.
    - Total: $992 + 64 + 6208 + 128 + 12352 + 128 + 262272 + 1548 = 283692$.
  - Frozen file protection:
    - [`configs/backbone.yaml`](file:///home/pro/Downloads/research/configs/backbone.yaml) (locked at $K=6$) is read-only; the pipeline passes `num_classes=12` to `SmallCNN1D` while extracting only `conv_channels`, `conv_kernels`, `fc_hidden` from YAML without mutating it.
    - Output directory validation ([`scripts/week1_reference_pipeline.py#L954-L964`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L954-L964)) explicitly blocks writes overlapping with `checkpoints/`, `configs/`, `data/`, or `scripts/`.
- **Identified Discrepancy / Risk [PROVEN]:**
  - **File:** [`scripts/backbone.py#L84-L86`](file:///home/pro/Downloads/research/scripts/backbone.py#L84-L86), function `build_model`.
  - **Issue:** `build_model(backbone_cfg)` hardcodes `num_classes=int(backbone_cfg["num_classes"])` which yields $K=6$ (282,918 parameters). The pipeline correctly bypasses `build_model` and calls `SmallCNN1D(num_classes=12, ...)`. However, any external caller attempting to construct the model via `build_model(yaml)` will silently instantiate the legacy $K=6$ head.
  - **Minimal Fix:** Keep `build_model` frozen per contract, but document that `SmallCNN1D(num_classes=12, cfg=...)` is the required entry point for $K=12$.

---

### Item (b): Stable global row id and exact supervised vs fullstream denominator

- **Status:** **PARTIAL / WARNING DISCREPANCY**
- **Verification Details:**
  - In [`ecdr/data.py#L55`](file:///home/pro/Downloads/research/ecdr/data.py#L55), `man["global_row_id"] = np.arange(len(man), dtype=np.int64)` is assigned before split merging, sorting, or filtering.
  - Window slices in [`scripts/week1_reference_pipeline.py#L179`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L179), [`#L316`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L316), and [`#L522`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L522) index `x_all` strictly via `global_row_id`.
  - Fullstream denominators are checked in [`scripts/week1_reference_pipeline.py#L518`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L518) against `FULL_STREAM_DENOMINATORS['test']` (5,053 windows).
- **Identified Discrepancy [PROVEN]:**
  - **File:** [`scripts/week1_reference_pipeline.py#L533-L536`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L533-L536), function `run_evaluate`.
  - **Issue:** For the supervised test denominator (pure + majority $K=12$), the code checks:
    ```python
    expected_sup_denominator = 3572
    if len(sup_test_df) != expected_sup_denominator:
        flush_print(f"WARNING: Verified supervised test count is {len(sup_test_df)} (expected reference {expected_sup_denominator}).")
    ```
    This emits only a non-fatal `WARNING` instead of raising a `ValueError`. In contrast, the fullstream denominator strictly raises `ValueError` on mismatch. This violates strict fail-fast validation for the reference evaluation set.
  - **Minimal Recommended Fix:**
    ```python
    if len(sup_test_df) != expected_sup_denominator:
        raise ValueError(f"Supervised test denominator mismatch: expected {expected_sup_denominator}, got {len(sup_test_df)}")
    ```

---

### Item (c): Current snapshot annotation encoding honored without unverified origin claim; raw sensor slices/overlap and majority threshold checks correct

- **Status:** **PASS**
- **Verification Details:**
  - Slicing and overlap: In [`scripts/week1_reference_pipeline.py#L150-L188`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L150-L188), `acc` and `gyro` signals are concatenated, sliced into $W=128, S=64$ windows, and verified against saved windows via `np.array_equal`. 64-sample stride overlap across consecutive windows in the same recording is asserted.
  - Majority threshold: In [`scripts/week1_reference_pipeline.py#L190-L212`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L190-L212), majority threshold $\ge 0.6$ of window length ($W=128$) matches builder policy exactly.
  - Conservative provenance wording: Lines 267–271 explicitly declare:
    `"Recorded existing builder convention plabel[start:end+1] slice plabel[s:e]. Alternate 0-based conversion (start-1:end) evaluated as non-mutating audit comparison only; authoritative indexing resolution deferred absent official specification docs."`
    No unverified origin claims are made.
- **Identified Hypothesis / Operational Risk [HYPOTHESIS]:**
  - **File:** [`scripts/week1_reference_pipeline.py#L150-L152`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L150-L152), function `run_audit`.
  - **Hypothesis:** If any raw experiment recording has an off-by-one or off-by-few length discrepancy between `acc` and `gyro` ($T_{acc} \ne T_{gyro}$), `np.concatenate([acc, gy], axis=1)` will crash with dimension mismatch rather than trimming to `min(len(acc), len(gy))` as `03_build_windows.py` does.
  - **Minimal Recommended Fix:** Align lengths defensively before concatenation:
    ```python
    min_len = min(len(acc), len(gy))
    sig = np.concatenate([acc[:min_len], gy[:min_len]], axis=1).T
    ```

---

### Item (d): Train-only norm, seeds 42/43/44, 30 epochs, Adam lr=0.001 wd=0.0001 cosine lr, patience 10, validation-only selection/checkpoint freeze before ANY test inference

- **Status:** **PASS**
- **Verification Details:**
  - Train-only norm: [`fit_train_norm`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L82-L87) is fit solely on `x_train_raw` across axes `(0, 2)` with clipping `std = np.clip(std, 1e-6, None)`. Test data is never observed or passed during normalization fitting.
  - Hyperparameters: Seeds `(42, 43, 44)` ([`#L286`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L286)), `epochs = 30` enforced ([`#L292`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L292)), Adam `lr=0.001, weight_decay=0.0001` ([`#L339-L353`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L339-L353)), Cosine LR schedule with `min_lr = base_lr / 100` ([`#L366-L370`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L366-L370)), and early stopping `patience = 10` on validation macro-F1 ([`#L342`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L342), [`#L432-L448`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L432-L448)).
  - Validation-only checkpoint freeze: In `run_train`, test windows are not loaded or evaluated. Checkpoints are frozen to disk, the best seed is selected purely by `val_macro_f1` (earliest epoch and lowest seed index break ties), copied to `selected_best.pt`, and its SHA-256 recorded in `selection.json` before any test inference in `run_evaluate`.

---

### Item (e): Phase reentry/output guards and checkpoint reload handling

- **Status:** **FAIL / PROVEN BUGS**
- **Identified Bugs [PROVEN]:**
  1. **Reentry Lockout Bug:**
     - **File:** [`scripts/week1_reference_pipeline.py#L966-L972`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L966-L972), function `main`.
     - **Code:**
       ```python
       if args.phase in ("audit", "train", "all") and out_dir.exists():
           existing_files = [f for f in out_dir.iterdir() if f.is_file()]
           if existing_files:
               raise FileExistsError(
                   f"Output directory {out_dir} already contains {len(existing_files)} files. "
                   "Failing safely to prevent accidental overwriting of reference run."
               )
       ```
     - **Impact:** If a user runs sequential phases to the same `--output-dir` (e.g. `python scripts/week1_reference_pipeline.py audit` followed by `python scripts/week1_reference_pipeline.py train`), the `train` command **crashes immediately** with `FileExistsError` because `out_dir` already contains `provenance_audit.json` from the `audit` phase! This prevents phase-by-phase execution of the reference pipeline.
     - **Minimal Recommended Fix:** Scope collision detection to phase-specific artifacts instead of any arbitrary file:
       ```python
       phase_targets = {
           "audit": ["provenance_audit.json"],
           "train": ["selection.json", "selected_best.pt"],
           "all": ["provenance_audit.json", "selection.json", "selected_best.pt"],
       }
       conflicts = [f for f in phase_targets.get(args.phase, []) if (out_dir / f).exists()]
       if conflicts:
           raise FileExistsError(f"Output directory {out_dir} already contains phase artifacts: {conflicts}.")
       ```
  2. **Dead-Code Ternary in Checkpoint Loading:**
     - **File:** [`scripts/week1_reference_pipeline.py#L541`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L541), function `run_evaluate`.
     - **Code:**
       ```python
       ckpt = torch.save if False else torch.load(ckpt_path, map_location=device)
       ```
     - **Impact:** `torch.save if False else` is an unremoved debug remnant.
     - **Minimal Recommended Fix:**
       ```python
       ckpt = torch.load(ckpt_path, map_location=device)
       ```

---

### Item (f): 12-class confusion matrix, recall, F1, Wilson score interval arithmetic, zero-denominator handling

- **Status:** **PARTIAL / WARNING BUG IN TRAIN LOOP**
- **Verification Details:**
  - Wilson score interval arithmetic in [`wilson_score_interval`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L68-L79) strictly matches the standard binomial confidence formulation $\frac{p + \frac{z^2}{2n} \pm z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$ with $z=1.959964$. Zero-trial cases safely return `(None, None)`.
  - Confusion matrix accumulation uses `np.add.at(cm, (y_true, y_pred), 1)`.
- **Identified Bug / Runtime Warning [PROVEN]:**
  - **File:** [`scripts/week1_reference_pipeline.py#L416-L418`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L416-L418), function `run_train`.
  - **Code:**
    ```python
    recall = np.where(support > 0, tp / support, 0.0)
    precision = np.where(pred_n > 0, tp / pred_n, 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    ```
  - **Issue:** Unlike [`#L563`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L563) in `run_evaluate` where `with np.errstate(divide="ignore", invalid="ignore"):` is used, `run_train` omits `np.errstate`. Because NumPy evaluates both arguments to `np.where` before selecting, any class with zero predictions (`pred_n == 0`) or zero support triggers an unsuppressed `RuntimeWarning: invalid value encountered in divide` on every epoch.
  - **Minimal Recommended Fix:**
    ```python
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(pred_n > 0, tp / pred_n, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    ```

---

### Item (g): Adjacency stratification / gap mask and empirical margin wording

- **Status:** **PASS**
- **Verification Details:**
  - In [`scripts/week1_reference_pipeline.py#L671-L695`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L671-L695), adjacency requires consecutive index (`nxt["win_idx"] == cur["win_idx"] + 1`), identical experiment (`cur["exp"] == nxt["exp"]`), and exactly 64-sample offset (`nxt["start"] == cur["start"] + 64`). Any recording boundary or stride break is excluded from adjacent pairs.
  - Stratification properly isolates `pure_stable` (classes 1..6, identical label), `pure_transition` (classes 7..12, identical label), and `mixed_other`.
  - Margin metrics and adjacency disclaimers ([`#L632`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L632), [`#L721`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L721), [`#L796`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L796)) explicitly state:
    `"Empirical point margins only; no robust certification or safety claim."` and
    `"Adjacent stream agreement rates characterize temporal stability; no ECDR certification claims implied."`

---

### Item (h): Varied CPU 1-thread batch 1 timing with 100 warmup and 1000 timed samples

- **Status:** **PASS**
- **Verification Details:**
  - In [`scripts/week1_reference_pipeline.py#L726-L765`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L726-L765), `torch.set_num_threads(1)` is set.
  - Inputs are indexed as `x_test_norm[[i % n_test]]` (batch size 1, shape `(1, 6, 128)`).
  - Cycling index `i % n_test` ensures varied, distinct inputs rather than reusing a cached memory buffer.
  - Two independent suites are benchmarked:
    1. Pre-normalized forward-pass latency: 100 warmup iterations, 1,000 timed calls (`time.perf_counter_ns()`).
    2. End-to-end preprocessing + forward pass (`apply_norm` on raw window + tensor allocation + forward): 100 warmup iterations, 1,000 timed calls.
  - Detailed summary reports mean, median, std (ddof=1), min, p95, and p99.

---

### Item (i): Verify phase truly recomputes persisted predictions/metrics and checks provenance hashes

- **Status:** **FAIL / MISSING REQUIREMENT COVERAGE**
- **Identified Deficiencies [PROVEN]:**
  1. **Provenance Hashes Omitted from Verification:**
     - **File:** [`scripts/week1_reference_pipeline.py#L820-L887`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L820-L887), function `run_verify`.
     - **Issue:** While `provenance_audit.json` is checked for file existence ([`#L821`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L821)), the SHA-256 hashes inside it (for `windows.npz`, `manifest.parquet`, `backbone.py`, `backbone.yaml`, `labels.txt`, raw sensor files) are **never verified** against the disk in `run_verify`. `run_verify` only verifies `selected_best.pt` against `selection.json`.
     - **Minimal Recommended Fix:** Add a verification loop in `run_verify`:
       ```python
       audit_data = json.loads((output_dir / "provenance_audit.json").read_text())
       for rel_path, expected_hash in audit_data.get("file_hashes", {}).items():
           target_path = ROOT / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
           if compute_sha256(target_path) != expected_hash:
               raise ValueError(f"Provenance hash mismatch for {rel_path}")
       ```
  2. **Persisted Metrics Recomputation Omitted:**
     - **File:** [`scripts/week1_reference_pipeline.py#L867-L881`](file:///home/pro/Downloads/research/scripts/week1_reference_pipeline.py#L867-L881), function `run_verify`.
     - **Issue:** `run_verify` recomputes predictions and logits and compares them against `test_predictions.npz`. However, it does not recompute or verify the summary metrics in `evaluation_results.json` (such as test accuracy, macro-F1, confusion matrix, or Wilson score intervals) to ensure that the persisted JSON matches the recomputed predictions.
     - **Minimal Recommended Fix:** In `run_verify`, compute accuracy and macro-F1 on `preds[sup_test_indices]` and assert equality with `evaluation_results.json["seed_evaluations"][winning_seed_key]`.

---

### Summary Checklist

| Item | Topic | Result | Primary Finding / Fix |
|---|---|---|---|
| **(a)** | $K=12$ SmallCNN1D 283,692 Params & Frozen Protection | **PASS** | `EXPECTED_PARAMS` verified; frozen files protected; `build_model` bypassed. |
| **(b)** | Stable `global_row_id` & Denominators | **PARTIAL** | Supervised test denominator check only warns instead of raising `ValueError`. |
| **(c)** | Current Snapshot Annotation & Slicing | **PASS** | 64-sample stride overlap and $\ge 0.6$ majority verified; origin claims deferred. |
| **(d)** | Train-only Norm, 30 Ep, Cosine LR, Val Selection | **PASS** | Strict train-only norm; early stopping patience 10; test kept strictly blind. |
| **(e)** | Phase Reentry Guards & Checkpoint Reload | **FAIL** | Line 966 blocks running `train` after `audit` in same dir; dead code in line 541. |
| **(f)** | Confusion Matrix, F1, Wilson Score Arithmetic | **PARTIAL** | Exact Wilson arithmetic; missing `np.errstate` in `run_train` emits divide-by-zero warnings. |
| **(g)** | Adjacency Stratification & Margin Wording | **PASS** | Stride-64 gap mask; explicit disclaimers against safety/ECDR certification claims. |
| **(h)** | Varied CPU 1-Thread Latency Benchmark | **PASS** | 1 thread, batch 1, 100 warmup, 1000 timed samples, forward-only + end-to-end. |
| **(i)** | Verify Phase Provenance Hash & Metric Checks | **FAIL** | `provenance_audit.json` hashes and `evaluation_results.json` metrics are not re-verified. |
