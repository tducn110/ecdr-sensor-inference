# Week 3 — actual K=12 certification and feasibility, 2026-11-03

This run executes the requested Week 3 scope against base `main` at
`8d4ebae576e612837dfd2f3cdcbaab737a705c4b`; `git ls-remote` confirmed remote
HEAD/main is the same revision. Changes are local, uncommitted and unpublished.
The source and environment remain frozen; no training or label/split changes.

## Scientific result

- **DERIVED:** positive whole-domain bounds for all eleven margins preserve the
  winner in real arithmetic; actual-graph IBP transfer rules, fixed-anchor and
  finite-horizon implications retain their stated premises.
- **VERIFIED finite evidence:** actual padded Conv/BN-eval/Pool/Flatten-MLP graph,
  all 5,053 test windows, 3,572 supervised windows, 5,034 adjacent pairs and 19
  recordings. The current runtime is fail-closed and always runs the reference.
- **CONDITIONAL:** float64 IBP positives are estimates from real-valid formulas,
  not validated enclosures of the frozen executable float32 backend.
- **NOT ESTABLISHED:** executable numerical soundness, useful formal certified
  reuse, LipCache superiority and H24's combined useful/correct/cost-saving regime.
- **Negative measured cost:** two complete chronological repetitions for each of
  two domains yield zero reuse and negative savings. This does not prove all
  possible future policies infeasible. G2 remains HOLD; G3 is partial/not passed.

At the validation-selected normalized radius `1e-6`, full-stream diagnostic
builds have 5,034/5,053 positive diagonal estimates and 4,211/5,053 positive wave
estimates. Both have zero next-window members. Conditional build counts are
distinct from formal certificates (zero), runtime build attempts (49 or 69 per
trace), and accepted reuse (zero). Reused-query agreement is N/A.

The ordinary normalization+CNN core costs 0.215–0.224 ms/window; the frozen
wrapper with per-query identity rehash costs 1.589–1.675 ms/window. Complete ECDR
costs 3.107–3.198 ms/window. Query totals include identity, membership, state,
fallback and builds, including the current duplicate anchor forward on build.
CSV collection/stratification and shadow inference are outside query timers.
RAM loading/setup and whole-loop wall times are separately reported. These are
CPU observations on one host, not energy, GPU or production measurements.

Persistent verifier memory components: the additional float64 model snapshot
contains 2,272,120 bytes of tensor payload; one representative conditional
certificate occupies 54,839 reachable Python bytes, including its domain. The
reference float32 tensor payload is 1,136,072 bytes and already exists in the
baseline. These do not measure peak interval workspaces or allocator/RSS deltas;
see `memory_components.json`.

## Deliverables and evidence ownership

| Artifact | Responsibility |
|---|---|
| `../../03_MATH/03_Week_3_Mathematical_Foundations.tex` and matching PDF | Formal Report A, all H01–H24, proofs, counterexamples, execution and open obligations |
| `../../03_MATH/04_ECDR_Mathematics_Explained.tex` and matching PDF | Pedagogical Report B, definitions, units, worked substitutions and interpretation |
| `../../ecdr/week3_verifier.py` | Actual-graph IBP estimates, immutable domains, identity/lifecycle/membership and reference fallback |
| `../../tests/test_week3_verifier.py`, `../../tests/test_week3_math.py` | 27 actual-graph/runtime regressions and 6 explicit mathematical counterexamples |
| `source_audit.md` | Current/historical source ownership, mismatches and official-library investigation |
| `protocol.json` | Validation grid, fixed anchor subset, selection rule, final radius and test matching |
| `experiment_results.json` | Main executed experiment, environment, identities, metrics and scientific statuses |
| `domain_results.json`, `domain_trace.csv` | 10,106 build diagnostics, eleven margins, membership and failures |
| `runtime_results.json`, `runtime_trace.csv` | Four complete 5,053-query traces, component costs and fallback reasons |
| `geometry.json` | Training-fit PCA and 32 validation margin-gradient geometry observations; not certificates |
| `library_probe.json`, `lirpa_probe.json` | Installed metadata and actual native/singleton-height auto-LiRPA probes |
| `feature_alignment.json`, `independent_review.md` | Independent actual-feature checks, H03 witnesses, proof and methodology review |
| `week1_preservation.json` | Fresh 65-check baseline consistency verification, not an ECDR numerical proof |
| `regression_tests.log` | Actual 93-test execution: 60 historical regressions + 27 actual-graph + 6 counterexamples |
| `preservation_before.json` | Byte hashes of all 170 protected Week 1/2 artifacts and initial dirty state |
| `delivery_verification.json`, `pdf_review.json` | Final source/evidence consistency, frozen preservation, PDF build and visual QA receipts |

The historical K=6 probe and valid-convolution/GAP branch are preserved. They
are not current K=12 certification evidence. No user acceptance is claimed.

## Reproduce

From repository root, with the existing frozen environment/data present:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/week1_independent_verify.py --output /tmp/week1_preservation.json
.venv/bin/python scripts/week3_lirpa_probe.py
.venv/bin/python scripts/week3_experiments.py
.venv/bin/python scripts/week3_verify_delivery.py
```

The two research reports use XeLaTeX and the installed Noto fonts. Compile each
canonical source twice to a temporary build directory, copy only its final PDF
to the canonical `03_MATH` location, and render every final page for visual
review. Source hashes, PDF hashes, final logs and rendered-page hashes are in
the PDF receipt. Compiler success alone is not visual QA.

`me/build_sources.py --week3-only` synchronizes the two compatibility wrappers
without modifying Week 1/2. The old combined master remains historical.
