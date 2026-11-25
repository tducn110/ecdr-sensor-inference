# GATES.md — Research Evidence Gates

> Reconciled 22/10/2026, documentation only. **G0 PASS; G1
> COMPLETE_WITH_REVISIONS; G2 PASS; G3–G6 PENDING.** The three completed baseline
> stages are not three unconditional scientific claims and do not establish
> novelty, certified reuse or net savings. New specifications/reports do not
> pass a gate without its required evidence. Saved measurements were not rerun.
> Current theory: [04_THEORY.md](04_THEORY.md); minimal next contract:
> [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).

- [x] **G0 — Initial literature/RQ baseline PASS** (historical 15/10/2026 gate)
  - EVIDENCE: initial 7-entry bibliography and scoped PDF/API checks; ledger,
    four-axis comparison and gap hypothesis in `01_RQ_GAP.md`/`02_LITERATURE.md`.
    The 21/10/2026 pilot added NoScope and refreshed Wei metadata; the bibliography
    gained CROWN and automatic-LiRPA paper metadata on 22/10/2026, making 10
    source records. This is not a complete review of the P01–P23 corpus.
  - KEEP: every promoted factual claim needs its source and locator/scope;
    GuardNet/backbone, verification/inference speed and certified-accuracy/runtime
    metrics remain distinct. `report.md` is the current scientific working
    report; its previous legacy content remains in Git history.
  - OPEN: C11 novelty and C12 net saving remain UNVERIFIED; the expanded
    literature review must test collisions rather than infer novelty from G0.

- [x] **G1 — Dataset feasible, COMPLETE_WITH_REVISIONS** (historically REVISE)
  - EVIDENCE: scripts 01–06b and saved integrity/split/label artifacts; chronology
    and subject/recording disjointness checked. Post-fix raw diagonal q90 test
    membership: same-activity 52.54%, label-change 2.26%, all pairs 31.94%.
  - CORRECTION: the script-05 row-index bug and recomputed F1/F2/F3 are documented
    in `03_DATA_AUDIT.md`; pre-fix values remain historical, not current evidence.
  - LIMIT: q50/q75/q90 names are budget levels, not implemented semantic events.
    PCA audit is centered score box plus orthogonal residual L2; its 35.42% test
    coverage is not a certificate, an anchor-reuse rate or a matched comparison.
  - OPEN: declare consistent domains/normalization; test anchor-based coverage
    and later matched structured comparisons. Empirical quantiles may define
    candidates; valid bounds and conservative membership must certify them.

- [x] **G2 — Frozen backbone baseline PASS** (saved 15/10/2026 results)
  - EVIDENCE: `configs/`, `scripts/backbone.py`, scripts 07–10, checkpoint and
    `data/g2_training.json`, `g2_metrics.json`, `g2_model_summary.json`.
  - SCOPE: K=6 pure-only, test denominator 2,874; seeds 42/43/44. Validation-selected
    seed44 has test accuracy 0.9349, macro-F1 0.9354; three-seed macro-F1
    0.9122 ± 0.0183 (population standard deviation on one fixed split, not CI).
  - MODEL: 282,918 parameters, BN/MaxPool present; frozen train-fit normalization.
    Weights, preprocessing and evaluation mode must match the verifier/reference.
  - LATENCY: saved forward-only CPU batch1 mean/p50/p95 0.2511/0.2357/0.3313 ms
    on already-normalized RAM tensors, 6 Torch threads. This is neither a fallback
    trace nor complete-path runtime evidence. Source/JSON inspection is not a rerun.

- [ ] **G3 — Sound minimal verifier, membership and reuse gate PENDING**
  - REQUIRED DOMAIN: implement the normalized diagonal contract in
    `07_G3_DOMAIN_CONTRACT.md`; raw `x`, fixed `N`, `F=f_theta o N`, normalized box
    and raw preimage must agree. Event selection only proposes a candidate.
  - REQUIRED VERIFIER: all five class-pair bounds valid on the entire declared
    domain or a proved enclosure; positive safe lower margins are sufficient.
    Ties, failed/unsupported bounds and inconclusive cases must fall back.
  - REQUIRED GATE: conservative membership independent of event heuristics;
    matched model/preprocessing/anchor/domain identities; trusted-anchor refresh
    only after actual reference inference; recording reset and stale invalidation.
  - REQUIRED SOUNDNESS: both mathematical argument and executable numerical
    assumptions/rounding or error-containment justification. An arbitrary tolerance
    or zero observed mismatches is not proof. Conditional arithmetic guarantees
    must be labeled and cannot close an unresolved numerical obligation.
  - REQUIRED EVIDENCE: independently checkable cases, negative/boundary/stale/
    non-finite/tie cases, shadow-reference mismatch counts and trace denominators.
    Zero reused queries do not give a measured 100% agreement result.
  - REQUIRED INSTRUMENTATION: descriptor/lookup/membership/forward/build timings,
    build count, anchor age, fallback reasons and identity trace. G3 component
    costs inform G5; a positive saving is not a G3 prerequisite.
  - CURRENT: contract/theory documented only; no verifier/runtime implemented.

- [ ] **G4 — Useful event structure PENDING**
  - CHECK: first distinguish bound-method quality (fixed domain) from region
    value (fixed bound method). Compare global/diagonal/correlated candidates
    with stated matching criterion and coverage/reuse denominators.
  - CHECK: include certified baselines and a clearly labeled non-certified
    confidence gate. A smaller region or PCA membership coverage alone cannot
    establish better certified reuse or novelty.
  - EXPECT: frozen-policy, anchor-based ablation with valid bounds; semantic
    event selection claims only if such a selector is actually implemented/tested.

- [ ] **G5 — Actual net runtime saving PENDING**
  - CHECK: compare complete full-inference and reuse paths on the same workload,
    preprocessing, device and declared accounting horizon. Include initialization,
    membership/lookup, fallback and build/rebuild costs; keep shadow-validation
    overhead identified separately from operational timing.
  - MODEL: `E[T]=T_e+T_m+(1-r)T_f+qT_b` is simplified accounting with roughly
    constant representative forward cost. Otherwise use conditional costs or
    measured query traces. Count initial build in build costs or add `T_init/N`.
  - EXPECT: lookup-only and complete-path tables, mean/p50/p95, throughput, memory,
    runtime environment and workload/split scope. Retain negative findings.
    Avoided forwards or the saved G2 latency alone do not pass G5.

- [ ] **G6 — Completed scientific report and reproducibility PENDING**
  - CHECK: source-grounded literature synthesis, formal assumptions, actual
    experiment results, uncertainty appropriate to design, scripts/artifacts,
    limitations and reproducibility metadata. Do not invent subject confidence
    intervals, benchmarks or a completed experiment to fill planned sections.
  - EXPECT: readable scientific LaTeX source and verified rendering, with
    existing results/proposed experiments clearly separated. A compiled current
    status report can be CREATED/VERIFIED as a document without closing the
    research gate or constituting user acceptance.
