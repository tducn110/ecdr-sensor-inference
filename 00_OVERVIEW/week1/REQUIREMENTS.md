# Week 1 requirements

1. Ground the brief in HAPT and targeted primary work on Lipschitz-margin,
   IBP, CROWN/auto-LiRPA, StreamiNNC, TinyDejaVu, LipCache and GuardNet.
2. Freeze the exact K=12 SmallCNN1D graph and expected 283,692 trainable
   parameters; keep BatchNorm in evaluation mode with frozen statistics.
3. Declare windowing, raw annotation coordinates, label policy, denominators,
   subject split and train-only normalization.
4. Separate classification accuracy, reference-decision agreement, and net
   end-to-end cost. The strict agreement target is zero disagreements.
5. Define candidate regions, anchors, events, membership, certificates and
   lifecycle identity separately.
6. Require 12-class confusion, per-class recall, margins and timing scopes;
   record run ID, command, environment and provenance hashes.
7. Keep transition support limitations explicit. K=12 is exploratory feasibility;
   historical G2 quality gates are not passed.
8. Keep old K=6 and valid-convolution/GAP models historical. Do not transfer
   metrics or GAP-specific theorems to the new graph.
9. Mark evidence VERIFIED, DERIVED, PROPOSED or NOT_ESTABLISHED.
10. Roadmap Week 1 brief/baseline, Week 2 data and deep-model rationale, Week 3
    certificate mathematics and executable actual-graph checks.

## Requirement-to-output map

| Audit ID | Requirement | Output / evidence |
|---|---|---|
| W1-01 | Research grounding | Canonical self-contained `.tex` grounding section; bibliography |
| W1-02 | K=12 lock | class order, checkpoint and preprocessing identity |
| W1-03 | Exact architecture | architecture audit table; executed 283,692 count and shape checks |
| W1-04 | Full pipeline execution | verified run record, receipts, exit 0 and hashes |
| W1-05 | Transition label policy/support blocker | support snapshot and policy section |
| W1-06 | Main RQ | RQ/objectives section |
| W1-07 | Evidence register | `EVIDENCE_REGISTER.md` and native PDF table |
| W1-08 | Sensor/data full contract | HAPT, split, window, labels and denominators |
| W1-09 | Hypothesis ledger | `HYPOTHESES.md` and native PDF table |
| W1-10 | Stable/transition stratification | measured descriptive H1 snapshot: 5,034 valid pairs |
| W1-11 | Week roadmap | roadmap section and `CLOSEOUT.md` |
| W1-12 | English presentation | canonical self-contained editable wrapper and companion records |
