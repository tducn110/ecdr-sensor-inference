# Research questions and gap — current English statement

This page is a concise current navigation record. The canonical Week 1 definitions and requirements are in [00_OVERVIEW/02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex) and [00_OVERVIEW/week1/](00_OVERVIEW/week1/).

## Research questions

**RQ1 — reference quality:** What classification behavior does the declared reference pipeline produce under the train-fit normalization, subject split, and K=12 evaluation protocol?

**RQ2 — decision preservation:** Under fixed preprocessing, a frozen reference model, a declared input domain, and a valid whole-domain bound, when does a reused class decision preserve the reference argmax with zero disagreements?

**RQ3 — coverage and cost:** At what event-conditioned coverage do membership checks, certificate construction, lookup, fallback inference, and rebuilds produce a measured net saving?

**RQ4 — soundness:** Which certificate bounds, membership predicates, and numerical assumptions are sufficient for the decision-preservation guarantee?

## Boundaries

Decision preservation is separate from classification accuracy. A strict method result requires zero disagreement with the frozen reference on the tested set and declared assumptions. The historical 5% empirical threshold is not a strict certificate criterion. Novelty and net saving remain hypotheses; no 50% latency target is assumed.

The SmallCNN1D K=12 path is exploratory but now has a retained executed baseline: run `week1_ref_1793620216_9d6c1d85`, selected seed 43 at epoch 8, 89.0538% supervised test accuracy and 77.6265% macro-F1 on 3,572 windows, with 17 internal checks passed. The legacy K=6 model/checkpoint and ECDRConv1DNet K=12 experiments remain separate historical contexts. Formal certification and net-saving evidence remain pending.

## Gap hypothesis

The candidate contribution is the combination of a direct-backbone class-pair certificate, event-conditioned domain geometry, anchor lifecycle, and an end-to-end cost model. This is a hypothesis, not a novelty claim. Prior-art collisions must be checked against primary sources in [02_LITERATURE.md](02_LITERATURE.md).
