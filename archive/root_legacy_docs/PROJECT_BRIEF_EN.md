# ECDR project brief — English navigation copy

The canonical Week 1 document is the self-contained [00_OVERVIEW/02_Week_1_Project_Brief.tex](00_OVERVIEW/02_Week_1_Project_Brief.tex). The generated shared body extract and requirements, hypotheses, evidence, and closeout registers are in [00_OVERVIEW/week1/](00_OVERVIEW/week1/). This file is a root-level synopsis and link hub.

ECDR asks whether a frozen neural classifier can safely reuse a previously executed decision on a sequential sensor window. Reuse requires a sound domain membership check and a whole-domain class-pair margin proof. The intended guarantee is strict agreement with the frozen reference decision, not ground-truth correctness, equal logits, or an assumed performance gain.

The Week 1 source keeps the K=12 exploratory SmallCNN1D path separate from the legacy K=6 checkpoint and from ECDRConv1DNet K=12 historical experiments. Run `week1_ref_1793620216_9d6c1d85` selected seed 43 at epoch 8 and passed 17 internal verification checks; supervised test accuracy is 89.0538% and macro-F1 is 77.6265% on 3,572 windows. Formal certification and net-saving results remain pending.
