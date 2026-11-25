# Start here — ECDR

The canonical current Week 1 source is the self-contained [00_OVERVIEW/02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex). Generated shared extracts and registers are in [00_OVERVIEW/week1/](00_OVERVIEW/week1/).

Week 1 defines the research problem, strict reference-decision guarantee, requirements, evidence contract, hypotheses, and the K=12 reference-pipeline training/evaluation and feasibility baseline. Week 2 owns deep-model and data analysis plus component rationale. Week 3 owns mathematical and verifier obligations. The combined report is a derivative and must not become a second Week 1 source.

Current boundaries:

- SmallCNN1D K=6 and its checkpoint are legacy baseline context.
- SmallCNN1D K=12 has an executed exploratory reference run: `week1_ref_1793620216_9d6c1d85`, selected seed 43 at epoch 8, with 17 internal verification checks passed. Formal certificate and net-saving results remain pending.
- ECDRConv1DNet K=12 is a separate historical model family.
- Strict decision preservation requires zero disagreements with the frozen reference; classification accuracy and the legacy 5% empirical threshold are different quantities.

Use [01_RQ_GAP.md](01_RQ_GAP.md), [02_LITERATURE.md](02_LITERATURE.md), [05_READING_PLAN.md](05_READING_PLAN.md), and [06_REVIEW_PROTOCOL.md](06_REVIEW_PROTOCOL.md) for the current English research route. Dated reports and Vietnamese documents remain historical snapshots.
