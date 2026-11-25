# Targeted prior work — current English ledger

This page records the current Week 1 comparison scope. Bibliographic metadata and locators remain in [refs.bib](refs.bib) and the paper notes under [notes/](notes/). A citation or metadata entry does not establish a full read, reproduced result, or accepted claim.

| Area | Primary reference | Boundary for ECDR |
|---|---|---|
| Certified caching | LipCache, [arXiv:2608.13144](https://arxiv.org/abs/2608.13144) | Related certified cache work; do not claim first temporal reuse or first certificate. |
| Incremental verification | Wei and Liu, [arXiv:2106.12732](https://arxiv.org/abs/2106.12732) | Reuses verification computation; distinct from skipping a direct backbone evaluation. |
| Conditional temporal inference | NoScope, [project page](https://dawn.cs.stanford.edu/publications/noscope/noscope-optimizing-neural-network-queries-over-video-scale) | Temporal difference gating is prior art; it is not a formal per-input decision certificate. |
| Sparse video inference | DeltaCNN, [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html) | Sparse computation is adjacent and has a different guarantee. |
| Structured verification | STBP, [arXiv:2606.09746](https://arxiv.org/abs/2606.09746) | Structured perturbation bounds are relevant; no claim of first structured verifier. |
| Bound machinery | CROWN, [NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html); auto-LiRPA, [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/0cbc5671ae26f67871cb914d81ef8fc1-Abstract.html) | Mathematical and implementation baselines; numerical soundness on this graph remains to verify. |

The Week 1 claim boundary is therefore: investigate direct reference-decision preservation for the declared SmallCNN1D K=12 pipeline, while keeping K=6 baseline history, ECDRConv1DNet K=12, heuristic reuse, and formal certificate claims separate.
