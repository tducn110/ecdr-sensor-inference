# GATES.md — Research Acceptance Gates (2/7 passed: G0, G1)

> Copy this checklist. A gate passes only with evidence, not confidence.
> report.md stays LEGACY_REFERENCE. Evidence lives in 01/02/refs.bib first.

- [x] G0 — Literature verified (passed 2026-10-15: refs.bib 7 entries, C4+C10 upgraded with PDF/API quotes, 01+02 done, new docs clean of turnX)
  - CHECK: every key claim in 02 has a refs.bib entry with URL + page/quote.
  - CHECK: LipCache metadata+speedup+GuardNet scope verified from arXiv/PDF.
  - CHECK: STBP existence verified from Springer; numbers only if full PDF quote.
  - CHECK: prior-work table on 4 axes (cert target, perturbation structure, temporal reuse, runtime cost) exists.
  - CHECK: novelty claims labeled SUPPORTED / PARTIALLY_SUPPORTED / UNVERIFIED.
  - EXPECT: 01_RQ_GAP.md + 02_LITERATURE.md done, gap stated as hypothesis.

- [x] G1 — Dataset feasible (passed 2026-10-15, verdict REVISE = COMPLETE_WITH_REVISIONS in paper wording: scripts 01-06b ran, integrity/splits/labels clean, 0 issues; event-model numbers RECOMPUTED after 05 index fix — p90 box test_same 52.5% vs test_change 2.3%; revise items per 03_DATA_AUDIT Sec 5)
  - CHECK: streaming order, subject split, event coverage, temporal deltas measured. — done (data/event_audit.json, data/corr_semantics.json)
  - EXPECT: 03_DATA_AUDIT.md answers 3 questions (delta distribution? coverage? label balance?). — done + verdict + post-review correction note
  - NOTE: pre-fix values preserved at commit cefd171; F1/F2/F3 restated 2026-10-15.

- [ ] G2 — Backbone reproducible
  - CHECK: config, seed, checkpoint, accuracy, macro-F1, param count via count_params.py.
  - EXPECT: shape test (B,6,128)->(B,K) passes.
  - STATUS 2026-10-15: **G2-A DONE** (scripts/06, 06b -> data/g2_class_support.json, data/corr_semantics.json): K=6 baseline GO, K=12 pure-only HOLD (transition pure support 5-65/class/split, 18 blockers). Decision + evaluation rules in G2_CLASS_DECISION.md. G2-B not started: torch not installed yet; re-export requirements-lock after install.

- [ ] G3 — Sound verifier
  - CHECK: bounds valid math + membership checker independent from event extractor.
  - CHECK: accepted inputs satisfy membership AND all class-pair margins > 0.
  - NOTE: zero observed violations = test evidence, NOT a soundness proof. Need both.
  - EXPECT: model version pin + anchor invalidation + trace log exist.

- [ ] G4 — Event structure useful
  - CHECK: structured beats Global Lipschitz + confidence gate on reuse or tightness.
  - EXPECT: ablation Global -> structured -> pair -> reuse.

- [ ] G5 — Net runtime saving
  - CHECK: parameter region with end-to-end latency better than full inference, INCLUDING rebuild (r*T_f > T_e+T_m+q*T_b).
  - EXPECT: lookup-only vs end-to-end tables, median+p95+mean on same CPU.

- [ ] G6 — Research report
  - CHECK: real numbers only, subject CIs, scripts, limitations.
  - EXPECT: 09_RESULTS.md + 10_PAPER_DRAFT.md.
