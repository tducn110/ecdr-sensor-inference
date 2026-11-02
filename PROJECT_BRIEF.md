# ECDR project brief — current navigation synopsis

The authoritative current Week 1 brief is the self-contained [00_OVERVIEW/02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex), which embeds the complete English body. [00_OVERVIEW/week1/](00_OVERVIEW/week1/) contains generated shared extracts and evidence registers. This root file is a concise navigation synopsis; it is not a second source of truth.

ECDR studies whether a frozen neural classifier's decision can be reused for a new sensor window only when a verifier establishes the required domain and class-pair margin conditions. The guarantee under study is strict preservation of the reference decision. It does not imply ground-truth accuracy, identical logits, or net speedup.

The current Week 1 contract keeps these branches separate:

- SmallCNN1D K=6 and its legacy checkpoint are historical baseline context.
- SmallCNN1D K=12 is the executed exploratory reference configuration. Run `week1_ref_1793620216_9d6c1d85` selected seed 43 at epoch 8; supervised test accuracy is 89.0538% and macro-F1 is 77.6265% on 3,572 windows. Formal certification and net-saving evidence remain pending.
- ECDRConv1DNet K=12 and its valid-GAP experiments are a separate historical model family and must not be merged into the SmallCNN1D claim.
- A strict decision-preservation test means zero disagreements with the frozen reference; it is not classification accuracy. The legacy 5% empirical threshold is not a strict method guarantee.

See [00_START_HERE.md](00_START_HERE.md) for the current route through the repository and [report_EN.md](report_EN.md) for the bounded research report.
