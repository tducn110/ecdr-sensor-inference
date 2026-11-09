# English delivery edition, 5 November 2026

`portal.js` owns the English slide content and speaker notes. `portal.css` owns
the shared layout. Each slide also appears as a section in the long-form portal.
The root and WEEK1–WEEK3 entry pages use this same source. Each week entry opens
its own week. The overview entry selects fifteen slides from the same content,
including the roadmap for each week.

The current edition contains 15 Week 1 slides, 20 Week 2 slides and 19 Week 3
slides. All 54 slides also exist as portal sections.

## Section navigation

Each week starts with an overview slide listing the sections, their purposes and
slide ranges. The requested Week 1 pipeline remains the first content slide,
immediately after its overview. The three-week route highlights the current week.

Every presentation slide has a persistent section bar under the week controls.
The bar highlights the current section, distinguishes earlier and upcoming
sections and reports the current slide's position within that section. Section
buttons open the beginning of the full section. The Overview button returns to
the current week's roadmap. Keyboard navigation, slide-number navigation and
week switching update the highlight from the same slide metadata.

| Week | Ordered sections |
| --- | --- |
| 1 | Problem & objective; Frozen reference; Research plan; Project organization; Code & handoff |
| 2 | Raw signals; Windows & labels; Data protocol; CNN architecture; Training & results; Code & handoff |
| 3 | Research map; Input assumptions; Decision & arithmetic; Geometry & features; Anchor lifecycle; Cost & next steps |

## Editorial decisions

- Week 1 introduces its section roadmap, then the requested data-to-reference flow and full
  inference, the research objective, the fixed reference, lifecycle and planning.
- Week 2 retains the data narrative: physical channels, waveform, transpose,
  window/stride, label policy, split denominators, normalization and class support.
  Architecture, operators, training and classification detail follow it.
- Week 3 groups all H01–H24 by function. The group pages give each item's purpose.
  H05, H13, H16, H22 and H24 have dedicated explanation slides. H12 provides the
  fixed-overlap dimension argument immediately before H13.
- The live pages omit audit badges, test counts, gate labels, hashes, receipts,
  DERIVED/FALSIFIED/CONDITIONAL verdicts, and validation-result charts. Their
  underlying research sources are unchanged.
- The data-selection term "validation" remains where it names a train/validation/
  test split or checkpoint selection. The Week 2 reference classification metrics
  remain part of its model explanation. No hypothesis validation verdict is shown.
- Mathematical decision conditions remain conceptual content. They are not shown
  as completed certifications or current-runtime acceptance claims.

## Coverage of the previous portal

| Previous material | Current location |
| --- | --- |
| Week 1 executive flow and synthesis flow | `w1-opening`, `w1-summary` |
| Standard inference and ECDR intervention | `w1-full-inference`, `w1-objective` |
| Scientific priority matrix | `w1-priorities` |
| Eight-stage lifecycle | `w1-lifecycle` |
| Model and freezing protocol | `w1-freeze`, `w1-contract`, Week 2 CNN/training |
| SWOT, communication, WBS and governance | `w1-swot` through `w1-governance` |
| Week 1 model code | `w1-code` |
| Week 1 audit/evidence panels | Removed from the presentation edition |
| Week 2 waveform and tensor extraction | `w2-waveform`, `w2-tensor` |
| Windowing, stride and label policy | `w2-window`, `w2-labels` |
| Subject split, leakage and normalization | `w2-counts`, `w2-normalization` |
| Class support and policy counts | `w2-support`, `w2-support-plot` |
| CNN graph, MACs and operators | `w2-cnn`, `w2-operators`, `w2-model-cost` |
| Training, metrics and class errors | `w2-training`, `w2-metrics`, `w2-errors`, `w2-confusion-plot` |
| Extraction and normalization examples | `w2-extraction-code`, `w2-normalization-code` |
| Week 2 synthesis | `w2-handoff` |
| Week 3 hypotheses table | `w3-map` and five group slides, with focus slides |
| Margin, IBP and Lipschitz explanations | `w3-h05`, `w3-bounds` |
| PCA, wave geometry and feature alignment | Geometry group, `w3-domains`, `w3-h13` |
| Anchor, membership and fallback | Lifecycle group, `w3-h16`, `w3-runtime` |
| Timing and radius outcome charts | Removed; conceptual roles remain in `w3-obstacles` and `w3-h24` |
| Week 3 handoff | `w3-next`, `w3-summary` |

## Hypothesis groups

| Function | Original IDs |
| --- | --- |
| Input structure and event assumptions | H01, H02, H03, H23 |
| Decision conditions and bound methods | H04, H05, H06, H07, H08, H09, H22 |
| Domain geometry and CNN structure | H10, H11, H12, H13, H14 |
| Anchor lifecycle and repeated reuse | H15, H16, H17 |
| Coverage, cost and practical feasibility | H18, H19, H20, H21, H24 |

## Source boundary

The delivery speech at `../../ECDR_01_Delivery_Speech_Week1_Week3.pdf`, the
pre-edit portal, `00_OVERVIEW/week1/SOURCE_OF_TRUTH.md` and the H01–H24 section
of `03_MATH/03_Week_3_Mathematical_Foundations.tex` informed this edition. The
source reports and their validation artifacts are retained. The previous five
HTML entries are recoverable under
`/tmp/ecdr-delivery-before-20261105-mRy2Q3/` for this working session.

Open an entry HTML in the browser and press P or its Present button. Left/right
changes slides, N toggles speaker notes, F requests fullscreen and Esc returns to
the portal. No CDN or translation service is required by the live pages.
