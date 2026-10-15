# 01 RQ GAP — Working Research Question + Gap Hypothesis (G0)

> Status: hypothesis with evidence, NOT a proven-new conclusion.
> Depends on: refs.bib + 02_LITERATURE.md. Next: 03_DATA_AUDIT.md after G0.

## 1. Working RQ

Can structured, event-conditioned class-pair bounds certify that a neural
backbone preserves its decision on temporally correlated inputs more often than
global Lipschitz bounds, while remaining cheap enough to produce net inference
savings?

Success = better Pareto frontier: tighter certificate + higher certified reuse
+ same backbone decision + lower total cost. No hard 50% latency target.

## 2. Scope

- Input: streaming sensor windows x_t (HAPT: 6 channels x 128 samples, 50 Hz).
- Backbone f: small verifier-friendly 1D-CNN (see roadmap suggestion.md Sec 3.3;
  24,428 params iff bias everywhere, no BN, K=12; K=6 -> 24,038; needs
  scripts/count_params.py in G2).
- Certificate proves reuse has the SAME PREDICTION as the backbone at a trusted
  anchor -- it does NOT prove correctness vs ground truth. Agreement and
  Accuracy are separate metrics.

## 3. Gap hypothesis (from 02 ledger)

Prior art covers each piece separately (all SUPPORTED in 02):
per-sample certified reuse exists (LipCache C1), spatio-temporal verification
exists (STBP C5), sparse temporal reuse without cert exists (DeltaCNN C7),
time-series smoothing cert exists (Franco C8).

Combination UNVERIFIED: event-conditioned structured sets + direct
backbone-decision cert + sequential anchor reuse + end-to-end cost WITH rebuild.
Do not claim first class-pair radius or first structured verifier.

## 4. Three killer risks (must survive G1/G3/G5)

1. Membership soundness: reused input must PROVABLY lie in the certified region,
   not just be classified as some event (G3).
2. Overlap != similarity: 50% window overlap does not imply small ||x_{t+1}-x_t||
   in CNN input space (G1 measures this).
3. Rebuild cost: E[T] = T_e+T_m+(1-r)*T_f+q*T_b; saving iff r*T_f > T_e+T_m+q*T_b
   (G5). Old formula hiding rebuild is invalid.

## 5. G0 exit for this file

RQ + gap stated as hypothesis with 02 ledger backing; no frozen novelty claim;
no implementation started. After G0 -> 03_DATA_AUDIT (deltas, coverage, label
balance) before freezing 04_THEORY.
