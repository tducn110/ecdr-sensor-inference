# ECDR context from primary sources

Research note | 2026-11-24 | Scope: ECDR only

This note collects primary-source context for the final-course report. It records what the cited work supports and what still has to be established by this project. It is not a verification of every statement in the manuscript or a replacement for the local experiment ledger.

## Project question and source boundary

The current project repository describes Event-Conditioned Certified Decision Reuse for sensor-stream inference, with a SmallCNN1D reference on the 12-class HAPT task. The report should frame its research question narrowly: under what declared input-domain, model, numerical, and membership conditions can a prior reference decision be reused, and does that policy reduce total measured cost on the specified stream? This phrasing is a synthesis of the project question and its source code, not a claim established by the external literature.

Primary project source: [ecdr-sensor-inference repository](https://github.com/tducn110/ecdr-sensor-inference). The local checkout inspected for this note is `main` at `c5c382f724009144e19c36e7241c2f8d0c2796c0`; it has extensive pre-existing dirty and deleted paths. No existing file was edited. Relevant local sources include `ecdr/replay.py`, `ecdr/certificate.py`, `ecdr/week3_verifier.py`, `PLAN.md`, `00_OVERVIEW/06_02_LITERATURE.md`, `00_OVERVIEW/10_P04_NoScope_Notes.md`, `04_SYNTHESIS/week4/design/PRIOR_WORK_COMPARATORS.md`, and the Week 5 report sources. A dirty worktree is not evidence that the checked-out report and repository page are the same revision; cite a commit only after matching it to the artifact being reported.

## Dataset source

The UCI HAPT record states that the dataset contains recordings from 30 volunteers, 6 basic activities and 6 postural transitions, waist-mounted smartphone accelerometer and gyroscope signals (3 axes each) sampled at 50 Hz, and 128 readings per window with 50% overlap. It also describes the original subject-level 70/30 participant split. [UCI HAPT dataset record](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)

For this project, distinguish the original UCI distribution and preprocessing description from the locally reconstructed raw stream and the project's selected subject split. Use the local data audit and provenance files for exact counts, split IDs, label mapping, normalization, and any annotation-coordinate caveat. Do not imply that the project split reproduces the official split unless the IDs establish that.

## Prior work and relevance

| Work | What its primary source supports | Relevance and limit for ECDR |
|---|---|---|
| NoScope (Kang et al., PVLDB 2017) | A video query system uses a difference detector, specialized model, and reference model in a cascade under target false-positive/false-negative constraints. The project has a scoped note at `00_OVERVIEW/10_P04_NoScope_Notes.md`; the paper is [PVLDB](https://www.vldb.org/pvldb/vol10/p1586-kang.pdf). | Prior art for conditional inference and temporal reuse. Its statistical/data-dependent error target is not a per-input formal certificate for this project's frozen classifier. Do not claim novelty merely for skipping inference when inputs look similar. |
| CROWN (Zhang et al., NeurIPS 2018) | Linear relaxations can provide certified bounds for neural-network robustness properties; see the [official proceedings paper](https://proceedings.neurips.cc/paper/2018/file/d04863f100d59b3eb688a11f95b0ae60-Paper.pdf) and [author repository](https://github.com/CROWN-Robustness/Crown). | Method family relevant to bounding class margins over a declared region. The paper's guarantees do not transfer automatically to ECDR's exact Conv1D/BatchNorm/MaxPool/Flatten graph, input domain, or floating-point runtime. |
| auto_LiRPA (Xu et al., NeurIPS 2020) | Generalizes LiRPA/CROWN-style perturbation analysis to computational graphs; the official [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2020/hash/0cbc5671ae26f67871cb914d81ef8fc1-Abstract.html) links the [authors' library](https://github.com/Verified-Intelligence/auto_LiRPA). | Relevant implementation machinery, not proof that the exact local graph is supported or that returned bounds enclose the deployed float32 computation. The local verifier/evidence owns those claims. |
| DeltaCNN (Parger et al., CVPR 2022) | Sparse frame-difference updates can accelerate CNN inference in practice on GPU with a specialized sparse implementation; paper and code are linked from the [CVPR Open Access entry](https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html) and [official repository](https://github.com/facebookresearch/DeltaCNN). | Prior work for exploiting temporal input redundancy. It changes computation through sparse propagation and reports empirical accuracy/performance behavior; it does not certify unchanged ECDR decisions over an input region. Its reported speedups are not HAPT or this repository's measurements. |
| Conformalized randomized smoothing for time-series classification (Franco et al., UAI/PMLR 2024) | Combines randomized smoothing and conformal prediction for robustness guarantees under transformations; see the [PMLR paper](https://proceedings.mlr.press/v244/franco24a.html). | Adjacent time-series certification literature. Its guarantee and prediction setup differ from deterministic reuse of a frozen classifier's cached class under explicit set membership. |
| Online verification under domain shift or network updates (Wei and Liu, 2021 preprint) | Prior work reuses/incrementally updates verification results when network/specification changes; cite the version inspected via [arXiv](https://arxiv.org/abs/2106.12732). | Prevents an overbroad “first reuse of verification results” claim. Its reuse target is verification computation, not necessarily skipping the deployed backbone's prediction for each new sensor window. |

The local bibliography also lists LipCache (arXiv:2608.13144) and a 2026 hybrid spatio-temporal verification preprint (arXiv:2606.09746). The current sources mark these as recent adjacent work, with applicability limits. Their metadata and versions must be rechecked against the exact text read before final IEEE citation; do not treat bibliography notes alone as a complete literature review.

## Theory the report may state, with conditions

For a fixed reference classifier with logits `f(x)`, anchor input `a`, and cached class `c`, a sufficient real-arithmetic condition for preserving that decision on a region `D` is that every competing margin `f_c(x)-f_j(x)` is strictly positive for all `x in D`, for every `j != c`. A sound verifier may prove this by returning strictly positive lower bounds for all `K-1` margins. Runtime reuse also needs a conservative membership check showing the query belongs to `D`, matching model/preprocessing/domain identity, and a valid certificate lifecycle. These are definitions/conditions to prove and implement, not a result supplied by CROWN/auto_LiRPA alone.

Separate these claims in the report:

- Mathematical soundness over real arithmetic.
- Sound enclosure for the executed floating-point graph.
- Query membership in the certified set.
- Reference-decision preservation on accepted queries.
- Useful certified reuse rate and total-cost savings on a named workload.

Evidence for one does not imply the next. In particular, full-stream agreement can be produced by falling back on every query; with zero accepted reuse, reused-query accuracy is undefined and no certified-reuse benefit has been shown.

## Local evidence to reconcile before drafting results

Use the current project sources and ledgers rather than rounded claims copied from narrative prose:

- Freeze the reported model branch and bind it to checkpoint, preprocessing, class decision, and source revision. Historical K=6, GAP models, synthetic cases, and later transition-model experiments are distinct evidence branches.
- Reconcile HAPT counts and split details against `01_DATA/` audit artifacts and the actual loader/build scripts.
- For theorem and verifier claims, use `03_MATH/`, `ecdr/week3_verifier.py`, and the current Week 4/5 evidence artifacts. Label unsupported or conditional results explicitly.
- For replay, latency, and break-even claims, use the paired replay evidence and cost accounting under `04_SYNTHESIS/week5/evidence/`; identify hardware, timing boundary, cache/rebuild cost, and fallback cost.
- The local Week 4 materials record that the formal reuse gate is fail-closed and that prior candidate regional methods had zero useful certified reuse on the stated stream. Reconfirm exact counts and artifact hashes before quoting them in the final report.
- Treat current `BLOG.md` and prose reports as leads, not sole evidence. They contain claims broader than the most conservative evidence notes; verify each number and interpretation from machine-readable artifacts before carrying it forward.

## Bibliography leads for IEEE references

The repository's `refs.bib`, `04_SYNTHESIS/07_refs.bib`, and the Week 4/5 `refs.bib` files contain overlapping but non-identical records. Select and deduplicate entries only after identifying the exact edition/version cited. At minimum, the literature section should cover NoScope, CROWN, auto_LiRPA, DeltaCNN, HAPT, and the adjacent time-series or online-verification papers actually used in the argument. Cite primary proceedings/dataset records and the project repository at a pinned commit in the completed report.

## Sources checked for this note

- UCI dataset record, read 2026-11-24: metadata, participant/task description, sensor setup, sampling, and windowing.
- NeurIPS official proceedings and author repositories for CROWN and auto_LiRPA, read 2026-11-24: method scope and software links.
- CVPR Open Access entry for DeltaCNN, read/search result 2026-11-24: paper identity and abstract-level method/claim. The HTML open returned HTTP 403 in this pass; inspect the linked PDF before quoting detailed sections or experimental setup.
- Repository-local NoScope note and bibliography, read 2026-11-24: source locators and evidence boundary recorded there; no new NoScope full-paper reread in this pass.
- Repository-local reports, code paths, and evidence inventory, read 2026-11-24: navigation leads only; result claims require individual artifact reconciliation.

## Outstanding literature checks

1. Reopen full primary texts and exact versions for LipCache, STBP, Franco et al., and Wei–Liu; capture page/section locators and compare their guarantees with ECDR's reuse target.
2. Inspect the complete DeltaCNN PDF before quoting details beyond the official abstract.
3. Pin the project repository SHA that matches the final report artifact and review any citations to current GitHub source at that SHA.
4. Reconcile each result sentence in the two final LaTeX files to a machine-readable source, with branch and revision recorded.
