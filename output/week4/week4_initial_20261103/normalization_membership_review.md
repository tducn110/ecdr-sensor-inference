# W4-007 — Normalization enclosure & conservative membership: implementation evidence

**Task:** [W4-007](../../../04_SYNTHESIS/week4/tasks/W4-007.md) · **Run:** `week4_initial_20261103` · **Owner:** `antigravity`
**Status of this document:** author's evidence and self-assessment. **It is not an independent review.**
Registry state after submit: `REVIEW`. No `VERIFIED` has been assigned by the author.

Base: `main` @ `eede68e4…`, dirty tree (Week 4 files untracked). Env: Python 3.11.15, NumPy 2.4.6, PyTorch 2.14.1+cpu, Linux x86_64,
Intel Xeon W-11855M (AVX-512, FMA). Frozen identity re-verified: preprocessing ID `aa8e1cc2…2fd6e`, 214/214 protected files clean.

## 1. Changed files

| File | Change |
|---|---|
| `ecdr/week4/reference.py` | Added `NormalizationConstants`, `execute_normalization`, `enclose_normalization`; `ReferenceOwner.normalize` now delegates to the shared `execute_normalization` (same op sequence, same errors); added `ReferenceOwner.normalization_constants()` / `.enclose_normalization()`. Diff: `tasks/W4-007/reference_py.diff`. |
| `ecdr/week4/domains.py` (new) | `CoordinateBox`, `CoordinateSpace`, `MembershipVerdict`, `contains`, `strict_gate_accepts`, `prefix_mismatches`. |
| `tests/test_week4_membership.py` (new) | 47 tests. |
| `tasks/W4-007/exhaustive_monotonicity.{py,json}`, `mutation_check.py`, `reference_py.diff` | Evidence artifacts. |

## 2. Design

**Executable normalization** is `N̂(x) = fl32(fl32(x − μ_c) / σ_c)`: two elementwise float32 NumPy ufuncs with the stored constants,
identical to `scripts/backbone.apply_norm` (checked bitwise on 64 random windows). It is *not* replaced by the real formula.

**Enclosure (C01/C04).** For a raw box `[a, b]`: `N̂(fl32(x)) ∈ [N̂(fl32(a)), N̂(fl32(b))]`.
Argument: round-to-nearest float32 conversion is monotone; for fixed μ_c, `t ↦ fl32(t − μ_c)` is monotone; for fixed σ_c > 0,
`u ↦ fl32(u / σ_c)` is monotone; a composition of monotone maps is monotone, so a coordinate interval maps between its endpoint images.
No error term is added because endpoints are evaluated by the same executable. `NormalizationConstants` rejects σ ≤ 0.
**Premises (not proved by this task):** the running NumPy build performs float32 subtract/divide as correctly rounded elementwise IEEE-754
operations (no fusion; no ordering-breaking flush). The exhaustive check below tests the premise on this build; it is not a proof for others.

**Membership (C02/C03).**

| Verdict (task) | `MembershipStatus` (frozen W4-003 enum) | `is_member` |
|---|---|---|
| INSIDE_VALIDATED | EXACT_MEMBER | True — only value accepted by `strict_gate_accepts` |
| OUTSIDE | VALIDATED_OUTSIDE | False |
| AMBIGUOUS | CONDITIONAL_MEMBER | False |
| INVALID | ERROR | False |

- `NORMALIZED_EXECUTABLE` box: query must be float32 `(6,128)`; exact float64 comparison against stored endpoints; boundary inclusive; no tolerance.
- `RAW_WINDOW` box: raw exact comparison **and** the executable normalized query must lie in the enclosure. Raw-inside without constants → AMBIGUOUS.
  Raw-outside but normalized-inside (non-injective normalization / one-ULP) → AMBIGUOUS ("not widened"). Raw-inside but normalized-outside → INVALID (premise contradicted).
- Fixed prefix (C05): `fixed_prefix_steps = k` forces `lower == upper` on all `6k` prefix coordinates; membership compares all 768 coordinates, and `prefix_mismatches` reports count and first index.

## 3. Evidence per checklist item

| ID | Evidence | Result |
|---|---|---|
| C01 | `execute_normalization`/`enclose_normalization`; bitwise match with `apply_norm` (`test_executable_matches_pipeline_apply_norm_bitwise`) | PASS |
| C02 | `CoordinateBox`: owned tuple endpoints, domain hash binds space/prefix/schema, radius 0 valid, negative/NaN radius, inverted, non-finite, wrong shape/space rejected (`TestCoordinateBox`) | PASS |
| C03 | Four verdicts + mapping; `strict_gate_accepts` rejects forged/non-EXACT results (`test_strict_gate_rejects_forged_result_shapes`) | PASS (mapping onto frozen enum — see limitation 2) |
| C04 | Argument in §2; **exhaustive**: every finite float32 (4,278,190,080 patterns) × 6 channels, 0 monotonicity violations, 0 NaN (`exhaustive_monotonicity.json`, ~1.5 min); boundary equality, conversion overflow, normalized overflow, subtraction overflow, one-ULP, neighbourhood windows incl. subnormals and ±0 | PASS (premise-conditional) |
| C05 | `test_every_prefix_coordinate_enforced_one_ulp_each_direction`: 6×64×2 = 768 single-coordinate perturbations, each rejected; tail stays free | PASS |
| C06 | alias mutation of caller arrays, read-only internals, F-order vs C-order, transposed layout ≠ same domain, serialization round-trip and tamper, float64/NaN/inf/wrong-shape/wrong-space queries | PASS |
| C07 | `contains` is pure; failed `normalize`/`enclose_normalization` leave `generation_id` and identity unchanged; no PCA/affine membership implemented | PASS |

**Test sensitivity.** 5 mutants applied to `domains.py` (1e-6 tolerance; skip coordinate (0,0); enclosure hi=lo; swapped lo/hi; float64 normalization): 5/5 killed
(`mutation_check.py`). Test-vacuity check: the neighbourhood test evaluated 9/9 parameter sets (none skipped by the overflow `continue`) and saw 2,494 adjacent
collision pairs, so the non-injective branch is exercised.

**Commands run (all exit 0):** `unittest … test_week4_membership.py` (47 OK); `… test_week4_contracts.py` (18 OK); `unittest discover tests` (198 OK, was 151);
`scripts/week4_audit.py check --run-id week4_initial_20261103` (214/214 clean); exhaustive script; mutation script.

## 4. Limitations and open points for the independent reviewer

1. Monotone-enclosure soundness rests on the IEEE correct-rounding premise of this NumPy build; verified exhaustively here only. A different build/CPU/threading layout needs a re-run.
2. **Spec/code mismatch:** the task names INSIDE_VALIDATED/OUTSIDE/AMBIGUOUS/INVALID but `types.MembershipStatus` (W4-003, frozen) has different names. I mapped rather than edited `types.py` (outside write set). Reviewer to confirm the mapping or request a `types.py` change.
3. **Placement:** FILE_SPEC §2.3 puts `enclose_normalization` in `numerics.py`. W4-007's write set excludes it, so it lives in `reference.py`; W4-009 owns later wrapping.
4. The AMBIGUOUS definition (raw/normalized disagreement; missing bridge) is my design decision — reviewer to confirm it is the intended meaning.
5. Raw boxes whose endpoints exceed float32 range are rejected (INVALID), not clamped.
6. Only `CoordinateBox` exists. `FixedOverlapBox` anchor policy, selectors (W4-015), `AffineTailDomain`/PCA (W4-016) and `outer_box` are not implemented.
7. Scientific scope: this establishes the raw→normalized executable bridge and exact box membership only. It says nothing about the operator/float32 bridge (W4-008..010), coverage, or H-dispositions.
