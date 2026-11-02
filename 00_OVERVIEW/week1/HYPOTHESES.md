# Week 1 hypothesis ledger

| ID | Hypothesis | Status | Falsifier / required measurement |
|---|---|---|---|
| H1 | Stable windows have more stable adjacent reference predictions than transition windows. | MEASURED DESCRIPTIVE | 2,560/2,617 stable versus 23/44 transition; this is not ECDR/reference agreement. |
| H2 | Event-conditioned certified coverage can improve at a fixed matched cost. | PROPOSED | Future coverage, fallback rate, certificate-failure rate and total cost, stratified by pure-basic, pure-transition and mixed windows under a matched-cost policy; exclude cross-recording/gap pairs. |
| H3 | Positive lower bounds for all 11 competing margins imply constant reference decision on a region. | DERIVED | Check sound bound, membership, arithmetic, and lifecycle obligations. |
| H4 | ECDR yields net savings in at least one declared workload regime. | PROPOSED | Report total cost separately for pure-basic, pure-transition and mixed windows, including preprocessing, event, membership, rebuild, fallback and I/O. |

H1 is supported descriptively for this snapshot only; H2 and H4 remain
unmeasured and H3 remains a derived implication pending executable premises.
Point margins are diagnostic values, not certified regional bounds.
