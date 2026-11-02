# Week 1 source of truth

This is the canonical English scope record for Week 1. The editable LaTeX
authority is `../02_Week_1_Project_Brief.tex`, which is self-contained between
the `BEGIN CANONICAL WEEK1 BODY` and `END CANONICAL WEEK1 BODY` markers.
`brief_body.tex` is a generated compatibility extract for combined documents.
The reference is the frozen K=12 SmallCNN1D contract in the body. The historical
K=6 configuration and checkpoints are provenance; the shared parameterized
SmallCNN implementation is the current implementation used to instantiate K=12.

## Locked semantics

- HAPT: six raw accelerometer/gyroscope channels, 50 Hz, windows 6 x 128,
  stride 64, subject-disjoint 17/4/9 split, train-only z-score.
- Label snapshot: pure or dominant >=0.6, with full-stream and supervised
  denominators reported separately. The pipeline currently inherits labels
  from `labels[start:end+1]` on its zero-based raw array; the UCI README does
  not confirm the original author’s index origin, so this remains an explicit
  provenance check rather than a hidden assumption.
- Reference: same-padded Conv6->32 k5 p2 BN-ReLU Pool2; Conv32->64 k3 p1
  BN-ReLU Pool2; Conv64->64 k3 p1 BN-ReLU; Flatten2048 -> Linear128-ReLU ->
  Linear12. Executed trainable parameter count: 283,692.
- Safety: zero Event-Conditioned Certified Decision Reuse (ECDR) disagreements
  with the fixed reference decision. Accuracy
  against ground truth is a separate measure.
- Cost: total end-to-end cost, including preprocessing, event, membership,
  certificate work, fallback and I/O. Forward-only timing is insufficient.

## Evidence boundary

The K=12 run, metrics and selected checkpoint are now recorded. Certificate
runtime remains pending. The
parameterized shared SmallCNN1D implementation may instantiate K=12; the old
K=6 configuration and checkpoints must not be used as K=12 evidence. Transition
support gates remain failed because transition data are undersupported; the
exploratory K=12 feasibility decision does not pass G2 quality. This HOLD is
distinct from the later verifier and net-cost dependencies.

## Current run provenance

Run `week1_ref_1793620216_9d6c1d85` was created 2026-11-02T11:50:16Z
(18:50:16 Asia/Ho_Chi_Minh), with pipeline hash
`1ef93101f658f8a18ebf19f0c8743ec5311ffe94933aee3505be6ec29a0be4d6`. The audit
passed over 131 source/data hashes, 122 raw sensor files, 17,453 windows from
61 recordings, float32 shape 17,453 x 6 x 128, zero raw-slice mismatches, zero
label-builder mismatches, and all-pair aligned-overlap checks. A one-sample
offset changes 98 assignments and remains a stated provenance caveat.

Environment: Xeon W-11855M CPU, Python 3.11.15, PyTorch 2.14.1+cpu, NumPy
2.4.6, pandas 3.0.6, one thread. The audit and feasibility run are not a
quality-gate pass; ECDR certification and net-cost evaluation remain pending.

Frozen selected reference:
`output/week1_closeout_20261102/reference/selected_best.pt`, SHA-256
`8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
Train-only preprocessing ID:
`aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

The independent baseline artifact receipt
`output/week1_closeout_20261102/reference/independent_verification.json`
passes 65/65 checks with zero failures. It is a consistency verification of the
Week 1 baseline artifacts, not the formal ECDR certificate verifier. The formal
certificate verifier remains a Week 3 task; the G2 HOLD is caused by transition
undersupport, while net end-to-end cost remains unestablished.
