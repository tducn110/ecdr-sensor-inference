# GATES.md — Research Acceptance Gates (2/7 passed: G0, G1)

> Copy this checklist. A gate passes only with evidence, not confidence.
> report.md stays LEGACY_REFERENCE. Evidence lives in 01/02/refs.bib first.

- [x] G0 — Literature verified (passed 2026-10-15: refs.bib 6 entries, C4+C10 upgraded with PDF/API quotes, 01+02 done, new docs clean of turnX)
  - CHECK: every key claim in 02 has a refs.bib entry with URL + page/quote.
  - CHECK: LipCache metadata+speedup+GuardNet scope verified from arXiv/PDF.
  - CHECK: STBP existence verified from Springer; numbers only if full PDF quote.
  - CHECK: prior-work table on 4 axes (cert target, perturbation structure, temporal reuse, runtime cost) exists.
  - CHECK: novelty claims labeled SUPPORTED / PARTIALLY_SUPPORTED / UNVERIFIED.
  - EXPECT: 01_RQ_GAP.md + 02_LITERATURE.md done, gap stated as hypothesis.

- [x] G1 — Dataset feasible (passed 2026-10-15, verdict REVISE: 5 scripts ran, integrity/splits/labels clean, same-activity coverage 25-36%; raw-delta event model must be revised before G3 per 03_DATA_AUDIT Sec 5)
  - CHECK: streaming order, subject split, event coverage, temporal deltas measured. — done (data/event_audit.json)
  - EXPECT: 03_DATA_AUDIT.md answers 3 questions (delta distribution? coverage? label balance?). — done + verdict

- [ ] G2 — Backbone reproducible
  - CHECK: config, seed, checkpoint, accuracy, macro-F1, param count via count_params.py.
  - EXPECT: shape test (B,6,128)->(B,K) passes.

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
