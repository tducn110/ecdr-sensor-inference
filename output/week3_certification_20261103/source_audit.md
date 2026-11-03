# Week 3 source/dependency audit

Base: `main`, `8d4ebae576e612837dfd2f3cdcbaab737a705c4b` (the requested
starting revision itself). No branch, commit, push, deployment or retraining.
Pre-existing untracked Week 2 compile/QA outputs are preserved byte for byte.
`preservation_before.json` records 170 protected files including the complete
frozen reference, Week 2 receipts, and existing Week 1/2 source documents.

## Authority and dependency map

1. `00_OVERVIEW/week1/SOURCE_OF_TRUTH.md`, `REQUIREMENTS.md`, `HYPOTHESES.md`
   define the frozen K=12 contract and G2 HOLD. Week 1 H1-H4 are separate from
   Week 3 H01-H24; they are not renumbered or merged.
2. `scripts/week1_reference_pipeline.py` owns train-fit float32 preprocessing,
   checkpoint selection, class order and baseline evidence. `scripts/backbone.py`
   owns the parameterized SmallCNN1D. Its default K=6 loader/configuration is
   historical; current use explicitly instantiates K=12 from frozen metadata.
3. `ecdr/data.py` owns stable global row IDs, split validation and chronological
   full streams. `scripts/week2_data_audit.py` and `week2_model_audit.py`, canonical
   Week 2 reports and receipts document independently executed data/model checks.
4. New `ecdr/week3_verifier.py` depends on the frozen backbone and metadata, not
   on `ecdr/model.py`. New Week 3 scripts depend on this verifier and the existing
   full-stream loader. Labels are consumed only after runtime decisions for
   stratification and supervised metrics.
5. `ecdr/model.py` implements the alternative valid-convolution/ReLU/GAP graph.
   `bounds.py`, `incremental.py`, `replay.py`, `certified_adaptive.py` and their
   existing tests concern that graph/operational NumPy semantics. Their success
   cannot prove SmallCNN1D K=12, padded/BN/pooling/Flatten-MLP certification.
6. `03_MATH/10_script_11_gate_a_prototype.py` hard-codes a six-output 2D replica
   and `load_pure_k6`. `11_p0_verifier_probe.json` is a 1,450-window historical
   probe. Neither is run or relabeled as current K=12 evidence.
7. `03_MATH/03_Week_3_Mathematical_Foundations.tex` owns Report A and H01-H24.
   `04_ECDR_Mathematics_Explained.tex` owns pedagogical Report B. Their historical
   valid-convolution/GAP theorems remain conditional on original A1-A6; H14 is
   not transferred to Flatten-MLP. Norm, certification, wave and float supplements
   are supporting explanations, not evidence of implemented soundness.
8. Numbered `03_MATH` theory/contract documents are canonical Week 3 navigation.
   Root theory/contract copies are compatibility entrypoints and must resolve to
   that authority instead of maintaining contradictory K=6 current contracts.
   `me` report copies are compatibility sources; preserve history and clearly
   identify canonical ownership. Do not rewrite Week 1/2.

## Mismatches found at base revision

- Theory and G3 contracts call `R^768 -> R^6`, five margins and 282,918 parameters
  current. Actual frozen reference is `R^768 -> R^12`, eleven margins and 283,692.
- Gate summary's unconditional G2 PASS belongs to historical pure-only K=6.
  Current K=12 G2 quality is HOLD due to transition support. G3 implementation
  and formal numerical soundness require distinct statuses.
- Old H13/H14 feature/GAP identities have assumptions violated by padding,
  pooling and Flatten-MLP. An actual-graph interior feature theorem must trace
  every padding/pooling phase; an RF size alone does not determine endpoints.
- Real-valued IBP and strict margin arguments do not enclose PyTorch float32
  execution. No arbitrary epsilon or sampled agreement closes that obligation.
- Readiness checkmarks in the old math README indicate claimed understanding,
  not executed acceptance. They must be replaced with evidence/status links.

## Official/library investigation

Read on 2026-11-03:

- https://github.com/Verified-Intelligence/auto_LiRPA
- https://auto-lirpa.readthedocs.io/en/latest/
- https://auto-lirpa.readthedocs.io/en/latest/installation.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv1d.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool1d.html

Installed auto-LiRPA 0.8.0 metadata requires `torch>=2.0,<2.12`, conflicting with
frozen 2.14.1+cpu. Import nevertheless succeeds. The installation web page's
broad '2.0+' wording is less restrictive than package metadata/current README;
the explicit package constraint governs compatibility. Local official package
source `operators/convolution.py:BoundConv` includes 1D convolution branches;
`operators/pooling.py:BoundMaxPool` uses 2D padding indices and max_pool2d.
`week3_lirpa_probe.py` executes both native graph and singleton-height diagnostic
conversion; resulting errors/bounds are recorded without changing the environment.
`library_probe.json` captures installed requirements. Import is not operator
coverage, and an out-of-contract successful run would remain conditional.

## Verification boundary

The strict runtime is fail-closed while numerical proof is unresolved. Diagnose
conditional positive IBP bounds and candidate membership separately; do not call
them certified reuse. Test actual K=12 operators, stale identities, recording
resets and boundary/invalid cases; rerun historical tests as regression only.
Full test-stream reference agreement/reuse denominator is 5,053; supervised
ground-truth metrics use 3,572. Evaluate runtime without shadow overhead and
record normalization, selection, membership, builds, fallback and state costs.
