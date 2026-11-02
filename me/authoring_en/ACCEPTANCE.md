# English concept document acceptance criteria

Request: 28 October 2026. Latest language instruction: all new/revised
document content in English. Preserve the existing Vietnamese versions.

- New mathematics atlas: standalone `me/ECDR_Mathematics_Concepts_EN.tex`.
- Existing open CNN source: revise `me/ECDR_CNN_Model_Analysis.tex` in place;
  preserve the exact former source/PDF as explicitly named VI artifacts.
- No changes to the data analysis document or underlying source/data/model.
- One mathematical/CNN concept per physical page; front matter, crosswalks,
  and bibliography are labeled separately and are not concept pages.
- Each page: foundation/definition, locally defined symbols and dimensions,
  derivation/example, motivation in ECDR, conditions and connections.
- Foundation chain includes real field/order/completeness, vector-space and
  probability axioms. Distinguish axioms, definitions, assumptions and theorems.
- Distinguish linear span, annotation time span and receptive-field span.
- Retain the old mathematical coverage, complete toy example, conditional
  universal certificates, counterexamples, numerical obligations and cost.
- Use `compile_latex_document` against both saved sources. Keep current
  editor open. Do not export/compile a separate PDF or open another tab.
- If app compiler is unavailable, record it honestly; validate source,
  references and per-concept pagination with a local XDV-only TeX pass.
  No PDF generation or claims of rendered-preview success from that fallback.
