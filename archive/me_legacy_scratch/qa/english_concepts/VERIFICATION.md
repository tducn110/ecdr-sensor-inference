# English concept sources: coverage, mathematical review and pagination

Date: **28 October 2026**. Scope: a new English mathematics atlas and an
in-place English revision of the already-open CNN source. No new PDF export,
no replacement editor tab and no data/model implementation change.

| Source | Concept inventory | XDV typeset pages |
|---|---:|---:|
| `ECDR_Mathematics_Concepts_EN.tex` | 104: F01–F46, B01–B30, C01–C28 | 113 |
| `ECDR_CNN_Model_Analysis.tex` | 64: P01–P08, C01–C56 | 71 |

The total page counts include title/reading guide/contents/coverage/references.
**Every one of the 168 identified concepts occupies exactly one typeset page.**
The check compares start and end page markers for each concept and requires
consecutive concept pages, rather than assuming that `clearpage` alone proves
the requirement.

## Content acceptance

- Both delivered sources select English Babel and contain English explanatory
  prose. The earlier Vietnamese sources remain separate reading versions.
- Concept pages state definitions, local variables and dimensions, derivations
  or examples, purpose in ECDR, and assumptions/connections.
- Mathematics traces back to logic and quantifiers; real field, order and
  completeness; vector-space axioms; and probability axioms. Axioms, definitions,
  established theorems, proof sketches and modeling assumptions are distinguished.
- Linear span, annotation temporal span and receptive-field span have separate
  explanations. Related roots include basis, inner product, dual norms,
  compactness, induction, convexity, eigenpairs, PSD matrices, SVD, derivatives,
  Jacobians, chain rule, feasibility, complexity, exponential and logarithm.
- The mathematics crosswalk covers all 29 prior mathematical groups and all
  24 hypothesis-ledger entries. New foundation pages explain prerequisites that
  were previously only named inside a downstream concept such as PCA.
- The CNN source begins with eight prerequisite pages, then defines the input,
  physical/feature channels, kernels, weights, bias, sharing, cross-correlation,
  stride, padding, dilation, groups and output shape. Later concepts derive
  activation, pooling, head/decision, training, resources and reuse properties.
- The complete toy CNN, continuous-box proof, looser-versus-exact bounds,
  stale-anchor counterexample, horizon induction and numerical obligations
  remain present. Historical K6 results stay clearly separate from the
  theoretical valid/ReLU/GAP K12 architecture.

## Review and corrections

Independent read-only reviews covered all mathematics fragments and CNN
P01–P08/C01–C56. Corrections incorporated before the final check include:

- Vectorizing a convolution output on the left of a matrix equation, with
  matching matrix/vector dimensions.
- Keeping `F = f_theta ∘ P`, raw margins `g` and normalized margins `gamma`
  consistent; using raw coordinates for the raw-reference horizon theorem.
- Requiring time-homogeneous per-channel preprocessing for aligned overlap.
- Defining local waves, feature sums, indices, matrix shapes and class counts
  again on pages using them; distinguishing preactivation `V` from activation `H`.
- Stating temperature one in cross-entropy and showing the intermediate
  derivative of the log-sum-exp term before the `p - target` result.
- Distinguishing common-logit offset invariance from positive rescaling, which
  generally changes softmax concentration.
- Repairing the cost model to assume constant full-forward cost when using
  `(1-r)t_f`; variable cost requires a conditional miss mean or trace accounting.
- Clarifying covariance/PCA ranks, numerical error budgets, exact-box access
  premises, and the sufficient rather than necessary nature of positive gates.
- Correcting F27's nonclosed-interval example to `x_n = 1/(n+1)` for positive
  integers, so every term belongs to `(0,1)` while its limit does not.
- Correcting F40's prerequisites to F01–F07, F23 and F26; F05–F07 actually
  supply the field, order and completeness foundations.
- Restating `1 <= s <= W` on B10 itself, where `C(W-s)` counts aligned
  shared coordinates, so the page can be read independently.

The reviews verify mathematical reasoning under stated assumptions, not a
machine-checked formal proof of an executable neural verifier.

## Checks actually run

Final local passes used the desktop application's **Tectonic 0.17.0 with
`--outfmt xdv` and `--untrusted`**. Both exited
successfully. No final overfull boxes, missing characters, undefined citations
or undefined cross-references were reported. Source checks also validate
concept sequences, English language configuration, cross-reference targets
and bibliography keys. Nonfatal underfull-box spacing warnings remain in
some mathematics paragraphs; they do not indicate missing text or overflow.

```bash
env TECTONIC_CACHE_DIR=/home/pro/.config/ChatGPT/latex/tectonic \
  .venv/bin/python me/qa/english_concepts/verify_sources.py --compile \
  --tectonic /usr/lib/chatgpt/resources/tectonic/tectonic
.venv/bin/python me/qa/english_concepts/verify_sources.py
```

The final `verify_sources.py` check passed. `checks.json` records source
SHA256 values, the physical page of every concept, language/coverage checks,
and preservation checks. `typesetting_provenance.json` additionally binds each
source, XDV and log by SHA256 after a successful compilation. A source change
or replaced log/XDV invalidates that receipt and requires `--compile` again.

Arithmetic was rechecked without a HAPT inference run: 20,332 parameters,
2,343,936 MAC, lengths 124/122/120; exact rational toy logits and minimum
margins 35/4 and 11/8. Corner calculations check arithmetic only; the
continuous-box argument additionally proves stable ReLU branches and affine
margin expressions throughout the box.

## Built-in compiler repaired and checked successfully

The original failure occurred before source compilation: the desktop editor
uses its own cache at `/home/pro/.config/ChatGPT/latex/tectonic`, which initially
lacked the v33 bundle. Its sandbox cannot resolve the bundle server's hostname.
The working shell compiler's cache is separate and did not solve that failure.

The bundled Tectonic executable populated that specific cache with the required
TeX resources through an XDV-only pass. Missing entries from the same verified
v33 bundle digest were also copied from the existing local cache; existing
entries were checked for equality and left unchanged. The next app attempt
completed all 71 CNN pages in XDV, then exposed a missing PDF math font,
`cmmi6.pfb`. Classical font definitions were read from the two XDV files and
resolved through `pdftex.map`. All 13 required PDF font resources were fetched
with Tectonic's `bundle cat` and then checked with `--only-cached`.

After the repair, **`compile_latex_document` returned `kind: success` for both
saved final English sources**. Current tool results and their source hashes
are in `app_compiler_results.json`; earlier failures remain in
`app_compiler_results_before_cache_repair.json` and
`app_compiler_repair_history.json`. The resource hashes and offline-cache checks
are in `cache_repair_resources.json`.

No application binary, sandbox policy or global configuration was changed.
The current editor remained open, and no other tab was opened. Successful
desktop compilation is verified; the live preview was not separately inspected
visually. No separate PDF was exported or visually reviewed in this revision.
The CNN source remains at its original path.

## Preservation and remaining boundaries

SHA256 checks confirm that the original four Week1–3 report sources/PDFs,
the previous Vietnamese mathematics source/PDF, and the data report source/PDF
remain unchanged. The former CNN source and PDF are preserved exactly as
`ECDR_CNN_Model_Analysis_VI.tex/.pdf`.

The old `ECDR_CNN_Model_Analysis.pdf` is a retained pre-revision Vietnamese
export, not the English source's newly rendered output. Use the explicitly
named `_VI.pdf` for that historical edition. The English deliverables for
this request are the saved editable LaTeX sources.

No training, HAPT inference, new certified-coverage experiment, runtime benchmark,
executable numerical soundness validation or novelty claim is added by the
expanded explanatory documents. Those research obligations remain distinct
from this completed source-authoring and typesetting check.
