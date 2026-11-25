# Week 1 integration inventory

Audit date: 2026-11-02  
Scope: navigation, source links, combined-report integration, presentation derivatives, and Week 1 historical dependencies.  
Authority: [canonical Week 1 source](../../00_OVERVIEW/02_Week_1_Project_Brief.tex) and [Week 1 registers](../../00_OVERVIEW/week1/).

This inventory records paths, ownership, and evidence boundaries. It does not promote generated artifacts, static claims, or historical measurements into current execution or acceptance evidence.

## Current source and registers

| Path | Class | Status and reason |
|---|---|---|
| 00_OVERVIEW/02_Week_1_Project_Brief.tex | only editable canonical source | Preserve; Luna-owned self-contained standalone Week 1 entry with embedded English body markers. |
| 00_OVERVIEW/week1/brief_body.tex | generated shared extract | Preserve as generated dependency for combined source and compatibility wrappers; not a second authoring authority. |
| 00_OVERVIEW/week1/SOURCE_OF_TRUTH.md | current register | Preserve; current authority record. |
| 00_OVERVIEW/week1/EVIDENCE_REGISTER.md | current register | Preserve; claim-to-evidence status. |
| 00_OVERVIEW/week1/HYPOTHESES.md | current register | Preserve; hypothesis boundary. |
| 00_OVERVIEW/week1/CLOSEOUT.md | current register | Preserve; closeout state. |
| 00_OVERVIEW/week1/REQUIREMENTS.md | current register | Preserve; Week 1 requirements. |
| me/Week_1_Project_Brief.tex | generated compatibility wrapper | Regular wrapper includes generated `../00_OVERVIEW/week1/brief_body.tex`; only the canonical standalone source is editable authority. |
| 00_OVERVIEW/01_PROJECT_BRIEF.md | current canonical entry | Preserve; Luna-owned Markdown entry. |

## Current navigation and prose

| Path | Class | Status and reason |
|---|---|---|
| README.md | current root navigation | English synopsis and canonical links. |
| PROJECT_BRIEF.md | current synopsis | English root synopsis; no second authority. |
| PROJECT_BRIEF_EN.md | current synopsis | English link hub; no pending metrics. |
| 00_HOME.md and 00_HOME_EN.md | current navigation | English root/Obsidian routes. |
| 00_START_HERE.md | current navigation | Week 1 owns reference-pipeline plan and feasibility baseline. |
| 01_RQ_GAP.md | current RQ synopsis | Uses canonical RQ1 reference quality, RQ2 decision preservation, RQ3 coverage and cost, RQ4 soundness. |
| 02_LITERATURE.md | current targeted ledger | Primary-source collision boundaries. |
| 05_READING_PLAN.md | current plan | Plan is separate from full-reading evidence. |
| 06_REVIEW_PROTOCOL.md | current protocol | Source/runtime, literature, inference, and proposal remain distinct. |
| 00_OVERVIEW/README.md | redirect stub | Local relative links avoid symlink-relative-link-base errors; points to canonical source and root navigation. |
| 00_OVERVIEW/04_00_START_HERE.md through 08_06_REVIEW_PROTOCOL.md | redirect stubs | Local relative links avoid symlink-relative-link-base errors; each points to canonical source and corresponding root page. |
| me/README.md | current local navigation | Points to canonical Week 1 source; PDFs remain derivatives. |
| tiers/README.md and tiers/00_overview/README.md | current tier navigation | Point to canonical source; aliases do not supersede it. |
| 04_SYNTHESIS/README.md | current synthesis navigation | Combined report and presentation are derivatives. |
| PDF_LIBRARY/README.md | current PDF navigation | PDF shelf artifacts are generated derivatives. |

## Combined report and generator boundary

| Path | Class | Status and reason |
|---|---|---|
| me/ECDR_Weeks_1_2_3_Theory.tex | current combined source | Week 1 block includes generated shared body extract; exact BEGIN/END WEEK1, WEEK2, WEEK3 markers retained. |
| 04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.tex | generated physical copy | Root-owned combined delivery; Week1 body is canonical include, with tiny bibliography metadata synchronized. |
| me/build_sources.py | generator | Gemini-owned updated generator; exact canonical body markers are used to refresh generated extracts. |
| me/verify_artifacts.py | verifier | Gemini-owned updated verifier; canonical/body drift and artifact checks are maintained by that owner. |
| me/Week_1_Project_Brief.pdf, me/ECDR_Weeks_1_2_3_Theory.pdf | current generated PDF derivatives | Shell compilation exited 0; canonical standalone is 11 pages and the combined report is 36 pages. Builtin desktop canonical compilation also succeeded. |
| PDF_LIBRARY/01_Week_1_Project_Brief.pdf and PDF_LIBRARY/16_ECDR_Weeks_1_2_3_Theory_Full.pdf | current PDF delivery aliases | Aliases were verified against the canonical standalone and current combined PDF outputs. |
| tiers/00_overview/01_PROJECT_BRIEF.md and tiers/00_overview/04_00_START_HERE.md through 08_06_REVIEW_PROTOCOL.md | redirect stubs | Local relative links avoid symlink-relative-link-base errors and point to canonical Week 1/root pages. |
| tiers/00_overview/02_Week_1_Project_Brief.tex and tiers/latex_sources/Week_1_Project_Brief.tex | removed cleanup paths | Do not restore; canonical source is `00_OVERVIEW/02_Week_1_Project_Brief.tex`. |

## Presentation and static assets

| Path | Class | Status and reason |
|---|---|---|
| web/present/ and 04_SYNTHESIS/09_presentation_app/ | removed cleanup paths | Do not restore; former web source/assets were removed by accepted cleanup. Current delivery is canonical LaTeX/PDF. |

## QA and dated records

| Path | Class | Status and reason |
|---|---|---|
| me/qa/Week_1_Project_Brief.txt and me/qa/Week_1_Project_Brief_contact_1.jpg | current Week 1 QA derivatives | All 11 standalone pages were checked with contact/detail views; no overlap, clipping, or unresolved glyphs were found. |
| me/Week_1_Project_Brief.log | current compile provenance | Preserve; shell compilation exited 0 and the log remains provenance for the current standalone derivative. |
| output/week1_closeout_20261102/document_checks.json | current QA receipt | PASS; records document, alias, and body-drift checks for the current delivery. |
| output/week1_closeout_20261102/before/ | scoped source backup | Preserve retained source snapshots. Five historical backup PDFs were removed during accepted cleanup and are not claimed as retained. |
| output/week1_closeout_20261102/integration_backup/ | integration source backup | Preserve source files backed up before integration edits; this is not a second delivery authority. |
| ECDR_Project_Brief_2026-10-22.md and ECDR_Project_Brief_Full_EN_2026-10-22.md | historical dated briefs | Preserve with explicit historical banners. |

## Week 2/3 and mathematical dependencies

| Path family | Class | Status and reason |
|---|---|---|
| 01_DATA/, 02_MODEL/, 03_MATH/ | downstream tier sources | No change; Week 2/3 ownership. Authored text remains historical and is not newly certified or treated as fully English. |
| 04_THEORY.md and 07_G3_DOMAIN_CONTRACT.md | mathematical/domain contracts | No change; Week 3/G3 dependencies. Authored text remains historical and is not newly certified or treated as fully English. |
| me/Week_2_Data_Foundations.tex and me/Week_3_Mathematical_Foundations.tex | historical/generated weekly sources | No change; parent preserves Week 2/3 scope. These sections remain historical and are not newly certified or treated as fully English. |
| me/ECDR_Mathematics_Concepts_EN.tex and me/ECDR_Mathematics_Explained.tex | mathematical catalog | No change; Week 3 dependency. |
| me/ECDR_CNN_Model_Analysis.tex and 02_MODEL/ aliases | model catalog | No change; Week 2/component-rationale dependency. |
| 04_SYNTHESIS/01_ECDR_Weeks_1_2_3_Theory_Full.pdf and PDF_LIBRARY/16_ECDR_Weeks_1_2_3_Theory_Full.pdf | current generated derivatives | Combined 36-page output was checked with contact/detail views; no overlap, clipping, or unresolved glyphs were found, and the delivery aliases point to the current combined output. |

## Current evidence banners

G2_CLASS_DECISION.md, GATES.md, GATES_EN.md, and TMB.md now carry a 2026-11-02 current Week 1 supersession notice at the top. Their historical decision/gate bodies remain intact. The executed baseline is verified by the retained receipt and 65/65 independent checks; quality-gate promotion, formal certificate soundness, and net savings remain unestablished/HOLD.

## Executed reference baseline

The current Week1 baseline is now VERIFIED at the internal execution level:

- Run: `week1_ref_1793620216_9d6c1d85`
- Selected checkpoint: seed 43, epoch 8
- Supervised test: 3,572 windows; accuracy 89.0538%; macro-F1 77.6265%
- Full-stream predictions: 5,053 windows
- Verification receipt: `output/week1_closeout_20261102/reference/verification_receipt.json`, status PASS, 17 internal execution checks.
- Independent baseline verification: `output/week1_closeout_20261102/reference/independent_verification.json`, 65/65 checks passed.
- Checkpoint SHA-256: `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`
- Preprocessing ID/hash artifact: `output/week1_closeout_20261102/reference/normalization.json` and `normalization.npz`; prep ID/hash `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

This verifies the reference baseline execution and retained artifacts. Quality-gate promotion remains HOLD/unestablished; this does not verify strict ECDR reference-agreement, formal certificate soundness, or net savings. The descriptive pure-adjacent/transition rates remain in the run evidence and are not promoted here as ECDR agreement.

## Final compile and QA evidence

- Standalone canonical compile: SUCCESS; 11 pages.
- Combined compile: shell exit 0; 36 pages.
- Builtin desktop canonical compile: SUCCESS.
- `output/week1_closeout_20261102/document_checks.json`: PASS.
- Body-drift checks: PASS.
- All 11 Week 1 pages and all 36 combined pages were checked with contact/detail views; no overlap, clipping, or unresolved glyphs were found.
- Current PDF aliases were verified against the canonical standalone and current combined outputs.
- Strict decision-preservation results, formal certificate soundness, and net savings remain unestablished/HOLD despite the verified baseline execution.
