# 01 RQ GAP — Working Research Question + Gap Hypothesis (G0)

> Documentation sync: 2026-10-22. Status: hypothesis with evidence, NOT a proven-new conclusion.
> Source of truth: current source/config/JSON for implementation and saved measurements;
> refs.bib + 02_LITERATURE.md for the scope of literature claims.
> Gate state: G0 PASS; G1 COMPLETE_WITH_REVISIONS; G2 PASS; G3–G6 PENDING.
> G1/G2 measurements remain dated 2026-10-15; this documentation update reruns no experiment.

## 1. Working RQ

Can structured, event-conditioned class-pair bounds certify that a neural
backbone preserves its decision on temporally correlated inputs more often than
global Lipschitz bounds, while remaining cheap enough to produce net inference
savings?

Success = better Pareto frontier: tighter certificate + higher certified reuse
+ same backbone decision + lower total cost. No hard 50% latency target.

## 2. Scope

- Input: time-series sensor windows x_t (HAPT: 6 channels x 128 samples, 50 Hz;
  2.56 s/window, stride 64 = 1.28 s). A window is one model input; an event
  descriptor selects a candidate region, which is a set of possible inputs.
- Current backbone f: `SmallCNN1D` in `scripts/backbone.py`, with three
  Conv1d blocks (channels 6→32→64→64, kernels 5/3/3), BatchNorm + ReLU after
  every conv, MaxPool after the first two, then Flatten 2,048→Linear128→ReLU→Linear6.
  It has **282,918 trainable parameters**, per `data/g2_model_summary.json`.
  This trained architecture supersedes the old ~24k/no-BN proposal. Its
  verifier compatibility remains untested; the presence of a model is not G3 PASS.
- Reference pipeline F uses the frozen backbone and per-channel normalization
  fit on training windows only. Epoch and seed selection use validation
  macro-F1. The selected artifact is seed 44, epoch 25; G2 evaluation is
  restricted to pure K=6 windows (2,874 test windows).
- Data scope: 17,453 windows across the corpus, 5,053 in the full test stream;
  these are separate denominators from the pure K=6 classification test set.
- Certificate proves reuse has the SAME PREDICTION as the backbone at a trusted
  anchor -- it does NOT prove correctness vs ground truth. Agreement and
  Accuracy are separate metrics.

## 3. Gap hypothesis (from 02 ledger)

The ledger supports narrower claims about existing pieces; this is not a
complete reading record for all items in the reading plan:
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

## 5. Current evidence and interpretation

- G1 found exploratory membership coverage, not certified reuse. Adjacent
  window pairs are not equivalent to queries against a persistent trusted anchor.
- G2 provides a saved data/model baseline. Three-seed test macro-F1 is
  **0.9122 ± 0.0183**, mean ± population standard deviation on one fixed split;
  it is not a confidence interval. Seed mapping and metrics are recorded in
  [G2_CLASS_DECISION.md](G2_CLASS_DECISION.md).
- `class_margins` records `f_y − max_{j != y} f_j` at observed labeled points.
  Those correct-class point margins are not region lower bounds or certified radii.
- No verifier, online event selector or certified-anchor runtime is implemented
  in the current `scripts/` inventory. G3 remains PENDING, and C11 novelty / C12
  net saving remain UNVERIFIED.

## 6. Next evidence: four staged G3 outputs

These are planned outputs, not results created by the documentation sync:

| Stage | Output | Acceptance evidence |
|---|---|---|
| G3.1 | [Domain contract](07_G3_DOMAIN_CONTRACT.md) | One coordinate system, frozen model/preprocessing identity, explicit diagonal region, sound membership and tie/numerical policy |
| G3.2 | Minimal diagonal verifier | Valid class-pair lower bounds over the entire region on the actual BN/MaxPool backbone; unsupported/failed cases return fallback |
| G3.3 | Reuse gate and trusted-anchor lifecycle | Certificate identity + membership + positive lower margins; reset at recording boundaries; a reused input does not become a new trusted anchor |
| G3.4 | Component instrumentation | Separate event, lookup/membership, full forward and build/rebuild measurements; traces and denominators are explicit |

Numerical smoke checks or zero observed disagreement alone do not prove
soundness. Correlated/PCA regions follow only after their domain and membership
are reconciled with the audit. Component instrumentation does not establish
net saving: that claim belongs to the later **G5 end-to-end comparison**.

The current synthesis is in [PROJECT_BRIEF.md](PROJECT_BRIEF.md); the scientific
report is [report.md](report.md), with editable LaTeX in
[latex/dossier.tex](latex/dossier.tex). The documentation reconciliation record
is [DOCS_SYNC_2026-10-22.md](DOCS_SYNC_2026-10-22.md).
