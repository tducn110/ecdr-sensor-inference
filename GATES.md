# GATES.md — Research Acceptance Gates (3/7 passed: G0, G1, G2)

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

- [x] G2 — Backbone reproducible (passed 2026-10-15: configs/backbone.yaml + training.yaml frozen; seeds 42/43/44; test acc 0.9349 / macro-F1 0.9354 best seed44 (val mF1 0.9504), 3-seed test macro-F1 0.9122 ± 0.0183; params 282,918; CPU p50/p95 = 0.236/0.331 ms batch1 = G5 baseline only)
  - CHECK: config, seed, checkpoint, accuracy, macro-F1, param count via count_params.py. — all done (data/g2_training.json, data/g2_metrics.json, data/g2_model_summary.json)
  - EXPECT: shape test (B,6,128)->(B,K) passes. — PASS (scripts/07)
  - SCOPE: K=6 pure-only per G2_CLASS_DECISION.md; denominator test=2,874 pure K=6 windows; weakest classes SITTING (F1 0.858) / STANDING (0.878).
  - NEXT: PredictionChangeRate on full stream now computable from frozen backbone (scripts/09 margins exist); G3 waits for event-model revision + 04_THEORY freeze.

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
