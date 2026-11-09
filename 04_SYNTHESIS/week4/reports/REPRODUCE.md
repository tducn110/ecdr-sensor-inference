# ECDR Week 4: Exact Reproducibility and Audit Walkthrough Guide

**Document ID:** `REPRODUCE-W4-038`  
**Run ID:** `week4_initial_20261103`  
**Host Environment:** Linux x86_64, Single-Thread CPU Isolation  
**Reference Checkpoint:** `SmallCNN1D` $K=12$ (SHA-256: `69b828d8b67f1ee56de33d1eb11bdf85f9cb558b877ee784841dbb80b2d6a5bf`)  
**Protocol Seal Hash:** `56806762e308960f4eb789b53ae7b973f1d2fe428bb8c8e11a196fc92ef4336c`  

---

## 1. System Environment and Dependencies

Reproduction requires the hermetic Python virtual environment located at `.venv`:
```bash
# Verify Python version (Expected: Python 3.11.10)
.venv/bin/python --version

# Verify PyTorch CPU build (Expected: 2.14.1+cpu)
.venv/bin/python -c "import torch; print(torch.__version__, torch.__file__)"
```

### Core Dependency Locks:
- `python`: `3.11.10`
- `torch`: `2.14.1+cpu`
- `numpy`: `1.26.4`
- `scipy`: `1.14.1`
- `pandas`: `2.2.3`
- `auto-lirpa`: `0.8.0`

---

## 2. Integrity Verification (Step 0)

Before executing any experimental code, verify that all 214 protected baseline files from Weeks 1–3 remain completely clean:

```bash
.venv/bin/python scripts/week4_audit.py check
```
**Expected Output:**
```
[week4_audit] Mode: check, run_id: week4_initial_20261103
[week4_audit] Step 1: Capturing git and source tree state...
[week4_audit] Step 2: Verifying frozen SmallCNN1D K=12 reference...
[week4_audit] Step 3: Independently verifying HAPT data split denominators...
[week4_audit] Step 4: Building and checking protected files manifest...
[week4_audit] Protected files verified: 214 files clean.
[week4_audit] Preflight PASSED successfully.
```

---

## 3. End-to-End Execution Pipeline (Steps 1–7)

### Step 1: Matched Geometry Validation Sweep (W4-018)
Evaluates candidate domain geometries on the validation split under declared budgets:
```bash
.venv/bin/python scripts/week4_run.py geometry \
  --run-id week4_initial_20261103 \
  --split val \
  --num-anchors 10 \
  --anchor-stride 38 \
  --budgets "0.0001,0.001,0.01"
```

### Step 2: Validation Policy Selection (W4-030)
Selects the winning certification policy based strictly on validation data:
```bash
.venv/bin/python scripts/week4_run.py select \
  --run-id week4_initial_20261103 \
  --split val \
  --num-anchors 10 \
  --anchor-stride 38
```

### Step 3: Cryptographic Protocol Seal Verification (W4-031)
Verifies the cryptographic seal locking all hyperparameters prior to test stream exposure:
```bash
.venv/bin/python scripts/week4_run.py verify-seal \
  --run-id week4_initial_20261103
```
**Expected Output:** `Protocol seal is 100% VALID.`

### Step 4: Full Chronological Test Stream Replay (W4-032)
Replays 5,053 chronological windows across 19 recordings on 15 executable methods (75,795 query executions):
```bash
.venv/bin/python scripts/week4_run.py replay \
  --run-id week4_initial_20261103 \
  --split test \
  --sealed-policy output/week4/week4_initial_20261103/selected_policy.json
```
**Artifacts Generated:**
- `output/week4/week4_initial_20261103/replay_trace.csv` (75,795 rows)
- `output/week4/week4_initial_20261103/replay_results.json`
- `output/week4/week4_initial_20261103/method_execution_status.json`

### Step 5: Paired Complete-Path CPU Benchmark (W4-033)
Measures single-thread CPU operational latencies across 3 repetitions and 50 warmup queries (227,385 query evaluations):
```bash
.venv/bin/python scripts/week4_run.py benchmark \
  --run-id week4_initial_20261103 \
  --split test \
  --sealed-policy output/week4/week4_initial_20261103/selected_policy.json \
  --repetitions 3 \
  --warmup-queries 50
```
**Artifacts Generated:**
- `output/week4/week4_initial_20261103/timing_trace.csv` (227,385 rows)
- `output/week4/week4_initial_20261103/benchmark_results.json`

### Step 6: Attributable Memory & Scaling Analysis (W4-034)
Measures tensor payloads, isolated cache footprints, membership microbenchmarks, and rank/horizon scaling sweeps:
```bash
.venv/bin/python scripts/week4_run.py memory \
  --run-id week4_initial_20261103 \
  --sealed-policy output/week4/week4_initial_20261103/selected_policy.json
```
**Artifacts Generated:**
- `output/week4/week4_initial_20261103/memory_results.json`
- `output/week4/week4_initial_20261103/scaling_results.json`

### Step 7: Comparison Tables and Hypothesis Ledger (W4-035)
Generates the comparison matrix across all 17 methods, 24 hypotheses ledger, H24 conjunction, and markdown summary:
```bash
.venv/bin/python scripts/week4_tables.py \
  --run-id week4_initial_20261103
```
**Artifacts Generated:**
- `output/week4/week4_initial_20261103/tables/comparison_matrix.csv`
- `output/week4/week4_initial_20261103/tables/hypotheses.csv`
- `output/week4/week4_initial_20261103/tables/h24_conjunction.csv`
- `output/week4/week4_initial_20261103/tables/summary.md`
- `output/week4/week4_initial_20261103/tables/statistics.json`

---

## 4. Running the Complete Automated Test Suite

To verify all unit tests and numerical assertions across Week 4:
```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```
**Expected Outcome:** 367+ tests passing with exit code 0.

---

## 5. How to Reproduce Key Negative Scientific Results

1. **Reproducing H14 (Negative Incremental Speedup):**
   - Run: `.venv/bin/python -m unittest tests/test_week4_accounting.py -k test_w4_033_c06_c07_paired_speed_ratios_and_break_even`
   - Observe that M03 latency ($\approx 1.93$ ms) exceeds M00 latency ($\approx 0.34$ ms), yielding speed ratio $\approx 0.18\times$.
2. **Reproducing H19 (Vanishing Regional Coverage):**
   - Inspect: `output/week4/week4_initial_20261103/replay_results.json` under keys `methods.M05` through `methods.M13`.
   - Observe that `reuse_count == 0` and `reused_agreement_rate == null`.
3. **Reproducing H24 (Trilemma Conjunction Unsatisfied):**
   - Inspect: `output/week4/week4_initial_20261103/tables/h24_conjunction.csv`.
   - Confirm that the sum of column `h24_conjunction_satisfied` is strictly `0`.
