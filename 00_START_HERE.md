# 00 START HERE — Structured Event-Conditioned Certified Neural Inference

> SoT docs: 00_START_HERE.md + GATES.md + refs.bib
> LEGACY_REFERENCE: report.md (550 lines) -- DO NOT EDIT until audit done.
> Working plan: roadmap suggestion.md (258 lines, verified read 2026-10-15).
> Research Gates: 2/7 (G0 PASS, G1 COMPLETE_WITH_REVISIONS). Current focus: **G2-A done**
> (K=6 GO, K=12 HOLD — G2_CLASS_DECISION.md); G2-B backbone training next, in parallel
> with event-model REVISE + the reading plan (05_READING_PLAN.md).

## Reading order

1. This file (00_START_HERE.md)
2. GATES.md -- 7 acceptance gates G0-G6
3. refs.bib -- 7 verified entries (LipCache, STBP/SAIV26, DeltaCNN, Franco UAI24, HAPT, Wei & Liu, auto_LiRPA)
4. 01_RQ_GAP.md -- working RQ + gap hypothesis (DONE in G0)
5. 02_LITERATURE.md -- 4-axis prior-work table (DONE in G0; §5 = pending additions)
6. 03_DATA_AUDIT.md -- HAPT deltas + coverage (DONE in G1; 2026-10-15 correction note inside)
7. G2_CLASS_DECISION.md -- K=6 GO / K=12 HOLD with evidence (G2-A DONE)
8. 05_READING_PLAN.md -- 23-paper list, baselines B0-B7, metrics, 4-week plan
9. 04_THEORY.md and later -- FROZEN until G0+G1 pass and event model revised

## Working RQ (hypothesis, not proven novelty)

Can structured, event-conditioned class-pair bounds certify that a neural backbone
preserves its decision on temporally correlated inputs more often than global
Lipschitz bounds, while remaining cheap enough to produce net inference savings?

Success = better Pareto frontier (tighter cert + more reuse + same backbone
decision + lower total cost). NO hard 50% latency target.

## Non-goals right now

- NO implementation (no backbone/verifier code) until G0 passes.
- NO edits to report.md.
- NO novelty freeze: RQ stays a hypothesis with evidence, not a proven-new claim.
- NO Figma/Canva slides -- P3, irrelevant to core.

## Inventory command (skills on this machine)

```
find ~/.gemini/skills ~/.agents/skills -type f -name SKILL.md 2>/dev/null | sort
```

Result 2026-10-15: ~60 files, all frontend (pixijs/gsap/figma/design), 0 ML/verifier skills.

## G0 exit reminder

G0 passes when: every key claim has a checkable source; STBP/LipCache/auto_LiRPA
metadata+results verified separately; prior-work table on 4 axes exists; each
novelty claim labeled SUPPORTED / PARTIALLY_SUPPORTED / UNVERIFIED; 01+02 done.
Then move to 03_DATA_AUDIT.
