/* ECDR Presentation Page: Week 3
 * Defines slide contents and section mappings for Week 3.
 */
(function (global) {
  'use strict';
  global.ECDR_PAGES = global.ECDR_PAGES || {};
  const C = global.ECDR_COMPONENTS || {};

  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const lead = text => `<p class="lead">${text}</p>`;
  const points = items => `<ul class="points">${items.map(text => `<li>${text}</li>`).join('')}</ul>`;
  const table = (headers, rows) => C.DataTable ? C.DataTable(headers, rows) : `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const flow = (items, vertical = false) => C.PipelineFlow ? C.PipelineFlow(items, vertical) : `<ol class="flow ${vertical ? 'vertical' : ''}">${items.map(x => `<li>${x}</li>`).join('')}</ol>`;
  const pair = (left, right) => `<div class="columns"><div>${left}</div><div>${right}</div></div>`;
  const formula = (text, explanation = '') => `<div class="formula">${text}</div>${explanation ? `<p class="caption">${explanation}</p>` : ''}`;
  const code = text => `<pre><code>${escape(text)}</code></pre>`;
  const assetRoot = typeof document === 'undefined' ? '../' : new URL('../', document.currentScript ? document.currentScript.src : (window.location ? window.location.href : "http://localhost/")).href;
  const figure = (path, alt, caption) => `<figure><img src="${assetRoot}${path}" alt="${escape(alt)}" loading="lazy"><figcaption>${caption}</figcaption></figure>`;
  const slide = (id, title, subtitle, content, notes, extra = {}) => ({id, title, subtitle, content, notes, ...extra});
  const kpis = items => C.KPIBanner ? C.KPIBanner(items) : `<div class="kpi-row">${items.map(([val, lbl]) => `<div class="kpi-card"><div class="kpi-val">${val}</div><div class="kpi-lbl">${lbl}</div></div>`).join('')}</div>`;
  const verdict = (good, bad) => C.VerdictBox ? C.VerdictBox(good, bad) : `<div class="verdict-box"><div class="verdict-good"><h4>[+] ĐIỂM TỐT</h4>${points(good)}</div><div class="verdict-bad"><h4>[-] ĐIỂM XẤU</h4>${points(bad)}</div></div>`;
  const analysisQuad = args => C.QuadrantCard ? C.QuadrantCard(args) : `<div class="quadrant-grid"><div class="quadrant-card">${args.what}</div><div class="quadrant-card">${args.why}</div><div class="quadrant-card">${args.results}</div></div>`;

  const hypotheses = [
    ['H01', 'input', 'Aligned overlap', 'Identify which sensor values carry over after the window shifts.'],
    ['H02', 'input', 'Overlap and input distance', 'Ask whether shared samples also make the complete tensors close in the chosen coordinates.'],
    ['H03', 'input', 'Activity labels and predictions', 'Separate the activity annotation from the decision of the frozen CNN.'],
    ['H04', 'decision', 'Local decision region', 'Explain why a continuous classifier with a strict winner can have a local region with the same winner.'],
    ['H05', 'decision', 'Positive lower margins', 'State the central sufficient condition for preserving the winner throughout a candidate region.'],
    ['H06', 'decision', 'Lipschitz ball', 'Translate a valid sensitivity bound and an anchor margin into a candidate radius.'],
    ['H07', 'decision', 'IBP completeness', 'Ask what interval propagation can and cannot establish when its enclosures become loose.'],
    ['H08', 'decision', 'Direct class-pair margins', 'Bound the class difference directly to retain cancellation at the final affine layer.'],
    ['H09', 'decision', 'CROWN cost', 'Compare the cost and tightness of affine relaxations under a fixed model and domain.'],
    ['H10', 'geometry', 'PCA and margin directions', 'Ask whether directions of temporal variation align with directions that affect the class margin.'],
    ['H11', 'geometry', 'Structured regions', 'Compare useful query coverage across region shapes under a declared matching policy.'],
    ['H12', 'geometry', 'Fixed-overlap wave dimension', 'Use the known prefix to express uncertainty through the newly arriving sensor values.'],
    ['H13', 'geometry', 'Aligned intermediate features', 'Identify the internal CNN activations that can share the same aligned input support.'],
    ['H14', 'geometry', 'GAP incremental identity', 'Describe an update formula for a GAP head and identify the architecture it requires.'],
    ['H15', 'lifecycle', 'Recentering after a hit', 'Examine what would be needed before a reused query could become a new anchor.'],
    ['H16', 'lifecycle', 'Reuse from a fixed anchor', 'Keep each query tied to the same anchor, region and reference definition.'],
    ['H17', 'lifecycle', 'Multiple future steps', 'Describe how a horizon domain must include every admitted sequence of incoming waves.'],
    ['H18', 'cost', 'Membership complexity', 'Account for the coordinates that an exact box membership decision must inspect.'],
    ['H19', 'cost', 'Soundness and coverage', 'Separate the correctness of a region argument from how often future queries enter it.'],
    ['H20', 'cost', 'Coverage and speed', 'Include checking, construction and fallback work when asking whether reuse saves time.'],
    ['H21', 'cost', 'Comparison with other methods', 'Set up a fair comparison with the full CNN and relevant reuse approaches such as LipCache.'],
    ['H22', 'decision', 'Executable arithmetic', 'Connect the mathematical margin condition to the rounding behavior of the deployed numerical pipeline.'],
    ['H23', 'input', 'Event labels', 'Define the selector as a proposal mechanism and keep its event name separate from the reuse condition.'],
    ['H24', 'cost', 'Useful and economical reuse', 'Bring decision preservation, query coverage and complete runtime cost into one research question.']
  ].map(([id, group, title, role]) => ({id, group, title, role}));
  const groups = [
    {id:'input', name:'Input structure and event assumptions', question:'What can the incoming stream tell us?', key:'H01', scope:'Shared samples, distances, activity annotations and event labels.'},
    {id:'decision', name:'Decision conditions and bound methods', question:'What would keep the reference winner unchanged?', key:'H05, H22', scope:'Local margins, bounding methods and executable arithmetic.'},
    {id:'geometry', name:'Domain geometry and CNN structure', question:'Which structure can reduce uncertainty or repeated work?', key:'H12, H13', scope:'PCA directions, fixed overlap, aligned activations and head architecture.'},
    {id:'lifecycle', name:'Anchor lifecycle and repeated reuse', question:'Which information remains valid at the next query?', key:'H16', scope:'Fixed anchors, recentering and future horizons.'},
    {id:'cost', name:'Coverage, cost and practical feasibility', question:'When would the method actually help?', key:'H24', scope:'Membership work, useful hits, total cost and fair comparisons.'}
  ];
  const groupSlide = group => slide(`w3-group-${group.id}`, group.name, group.question,
    lead(group.scope) + table(['Hypothesis', 'Role in the research'], hypotheses.filter(h => h.group === group.id).map(h => [
      `<span class="hyp-id">${h.id}</span><strong>${h.title}</strong>`, h.role
    ])),
    `Introduce this as a group of questions with one function: ${group.question} Give the role of each item briefly. Spend the detailed explanation on ${group.key}. The IDs stay available for questions after the talk.`,
    {kind:'group', group:group.id, hypothesisIds:hypotheses.filter(h => h.group === group.id).map(h => h.id)});



  const weekData = {3: {
      title: 'Hypotheses and the reuse mechanism',
      subtitle: 'Twenty-four research items grouped by function, with detailed explanations of the central ideas.',
      slides: [
        slide('w3-objects', 'A query, an anchor and a region have different roles', 'The objects used in the reuse mechanism',
          table(['Object', 'Role'], [
            ['Query x', 'The new sensor window that needs a decision.'],
            ['Trusted anchor xʳ', 'An input that actually ran the frozen reference, together with its winning class c.'],
            ['Candidate region S', 'A declared set of possible inputs around or relative to the anchor.'],
            ['Event descriptor e', 'Information used to propose a region or a checking policy.']
          ]) +
          lead('A region describes possible inputs. Its mathematical decision condition and a query’s membership in it are separate questions.'),
          'Introduce these objects with one new window and one earlier reference evaluation. Avoid starting with all twenty-four hypothesis IDs. First make the mechanism concrete.'),
        slide('w3-map', 'The 24 hypotheses serve five research functions', 'Use the groups to follow the argument',
          table(['Function', 'Hypotheses', 'Main explanation'], groups.map(g => [g.name, hypotheses.filter(h => h.group === g.id).map(h => h.id).join(', '), g.key])) +
          lead('The detailed talk focuses on H05, H13, H16, H22 and H24. The other items explain the assumptions and supporting mechanisms.'),
          'Tell the audience that the IDs provide a map of the research questions. Briefly explain the function of each group. There is no need to read twenty-four items as twenty-four separate findings.'),
        groupSlide(groups[0]),
        groupSlide(groups[1]),
        slide('w3-h05', 'H05 supplies the central decision condition', 'Focus: keep the anchor class ahead of every competitor',
          formula('L<sub>cj</sub>(S) ≤ inf<sub>x ∈ S</sub> [F<sub>c</sub>(x) − F<sub>j</sub>(x)]') +
          formula('L<sub>cj</sub>(S) > 0 for all eleven j ≠ c, and query x ∈ S') +
          points(['c is the reference winner at the anchor.', 'Each L is a valid lower bound over the same complete region S.', 'All eleven competing classes matter because the runner-up can change across the region.']),
          'This is the most important mathematical slide. A positive point margin at the anchor is different from a positive lower bound across a region. Under the stated conditions, every competing logit stays below the anchor class throughout that region.', {focus:'H05'}),
        slide('w3-bounds', 'A bound method encloses possible CNN outputs over a region', 'The supporting roles of H06, H07, H08 and H09',
          table(['Method or idea', 'Function'], [
            ['Lipschitz bounds', 'Relate input distance to a maximum change in a class margin.'],
            ['Interval Bound Propagation', 'Propagate lower and upper activation limits through the actual operators.'],
            ['Direct pairwise margins', 'Combine final-layer class rows before bounding their difference.'],
            ['CROWN-style relaxations', 'Use affine constraints to retain more information than independent intervals.']
          ]) +
          lead('Bound tightness and construction cost both matter. A non-positive bound alone does not demonstrate a changed CNN decision.'),
          'Explain IBP as enclosing many possible inputs, not sampling a few points. Direct margins can retain cancellation at the last layer. The comparison between methods requires the same model and region.'),
        slide('w3-h22', 'H22 connects the mathematical condition to executable arithmetic', 'Focus: the reference runs with finite precision',
          flow(['Mathematical region', 'Stored normalization', 'Numerical bound computation', 'Float32 reference operations', 'Conservative acceptance condition'], true) +
          lead('Rounding, operation order and implementation details must be accounted for before a numerical result can authorize reuse.'),
          'Explain why a real-arithmetic argument cannot automatically be assigned to every float32 execution. Float64 calculations have their own rounding behavior. H22 names the connection that must be made between the mathematical argument and the reference implementation.', {focus:'H22'}),
        groupSlide(groups[2]),
        slide('w3-domains', 'Two candidate domains express different uncertainty', 'Full-coordinate boxes and fixed-overlap waves',
          pair(lead('<strong>Diagonal box</strong>') + points(['Allow variation in all 6 × 128 = 768 coordinates.', 'Choose a budget for each normalized coordinate.', 'Check membership using those same coordinates and budgets.']),
            lead('<strong>Fixed-overlap wave</strong>') + points(['Keep the known 6 × 64 overlap fixed at its aligned position.', 'Represent uncertainty in the 6 × 64 newly arriving values.', 'Use a declared center policy for the unknown suffix.'])) +
          formula('Full window: 768 scalars; new wave: 384 free scalars', 'H12 describes the dimension reduction under the fixed-prefix assumptions.'),
          'The domain center is a policy assumption about future values. The future query still needs to belong to the declared set. Reducing the number of free scalars does not establish a proportional runtime saving.'),
        slide('w3-h13', 'H13 identifies aligned intermediate features', 'Focus: input overlap can support a narrower structural reuse argument',
          flow(['Next-window feature position j', 'Aligned receptive-field inputs', 'Previous-window feature position j + 16']) +
          table(['Reference detail', 'Role in the alignment argument'], [
            ['Final feature positions j = 2…13', 'The interior positions covered by the actual padded and pooled graph argument.'],
            ['64 feature channels at each position', 'The activation channels to compare at those aligned positions.'],
            ['Fixed normalization and BatchNorm eval state', 'Preserve the same pointwise transform and inference state.'],
            ['Flatten–MLP head', 'Position-dependent head weights still require separate treatment.']
          ]),
          'State the structural relation before discussing implementation. H13 concerns selected activations under specific assumptions. It does not imply equal logits or unchanged predictions. H14 uses a GAP head assumption, so its update formula cannot simply be copied into this reference.', {focus:'H13'}),
        groupSlide(groups[3]),
        slide('w3-h16', 'H16 keeps repeated queries tied to one trusted anchor', 'Focus: reuse does not create a new reference evaluation',
          flow(['Run reference at anchor', 'Define its region and decision condition', 'Check each new query against that anchor', 'Reuse only when all conditions apply'], true) +
          points(['Keep the same model, preprocessing, anchor and region identity.', 'Invalidate stale state at recording boundaries, gaps or incompatible changes.', 'A miss runs the reference on a valid input and may establish the next anchor.']),
          'Use one anchor and two later queries as the example. Both queries refer back to the original anchor. Moving the region center after a hit is a separate proposal in H15, and a multi-step horizon is a separate construction in H17.', {focus:'H16'}),
        slide('w3-runtime', 'The proposed runtime routes each valid query through one decision path', 'Reference inference remains the fallback',
          table(['Step', 'Responsibility'], [
            ['Input boundary', 'Handle invalid shape or non-finite input as an input error.'],
            ['Lifecycle boundary', 'Identify the recording and invalidate stale anchor state.'],
            ['Reuse condition', 'Require a matching anchor and region with the necessary decision and numerical conditions.'],
            ['Membership', 'Check whether this query belongs to that same region.'],
            ['Reuse path', 'Return the stored anchor class when all requirements apply.'],
            ['Fallback path', 'Run the frozen reference on the valid query.']
          ]),
          'Describe the roles as the proposed mechanism. This is not a current-runtime acceptance badge. A valid input without a reuse argument can fall back. An invalid input requires error handling.'),
        groupSlide(groups[4]),
        slide('w3-h24', 'H24 asks whether safe reuse can also be useful and cheaper', 'Focus: the combined research objective',
          formula('Expected cost = T<sub>event</sub> + T<sub>membership</sub> + (1 − r)T<sub>reference</sub> + qT<sub>build</sub> + T<sub>other</sub>') +
          table(['Term', 'Meaning'], [
            ['r', 'Fraction of queries that reuse the reference decision.'],
            ['q', 'Average region-construction frequency per query.'],
            ['Treference', 'The cost of the chosen complete reference boundary.'],
            ['Tother', 'Additional preprocessing, identity, state or I/O work inside the declared comparison boundary.']
          ]) +
          lead('Decision preservation, useful query coverage and lower complete cost must hold together.'),
          'End the main hypothesis explanation here. H19 separates correctness from coverage. H20 separates coverage from speed. H18 accounts for membership work, and H21 places the comparison on the same workload and device. H24 combines these questions into the practical research objective.', {focus:'H24'}),
        slide('w3-obstacles', 'Region usefulness and checking overhead are separate design questions', 'What each part of the mechanism needs',
          pair(lead('<strong>Geometry and margins</strong>') + points(['A wider set may admit more inputs.', 'The margin bounds must remain useful on that set.', 'Directions of variation may matter more than a single scalar radius.']),
            lead('<strong>Complete processing cost</strong>') + points(['Construction work must be included.', 'A failed check still consumes time.', 'The benefit depends on reference evaluations actually avoided.'])),
          'Explain how the design pressures connect. A set must cover relevant future inputs, its decision condition must stay useful and its checking cost must fit within the computation that it avoids.'),
        slide('w3-next', 'The next tasks follow the hypotheses that they address', 'Proposed follow-up work',
          table(['Task', 'Related items', 'Purpose'], [
            ['Numerical arithmetic', 'H05, H22', 'Connect the decision condition to an executable acceptance rule.'],
            ['Structured domain design', 'H10, H11, H12', 'Align the admitted input set with actual temporal variation.'],
            ['Intermediate feature reuse', 'H13, H14', 'Study an implementation appropriate to the actual Flatten–MLP graph.'],
            ['Anchor and horizon policy', 'H15, H16, H17', 'Control which reference state remains valid over time.'],
            ['Complete cost accounting', 'H18, H19, H20, H21, H24', 'Connect useful hits and checking overhead with an appropriate comparator.']
          ]),
          'These are proposed research tasks. Introduce their purpose and dependency without converting them into completed outcomes or adding unconfirmed deadlines.'),
        slide('w3-quadrant', 'Tuần 3: Khung phân tích 4 câu hỏi', 'Cơ chế toán học, Điều kiện biên & Mô hình chi phí',
          analysisQuad({
            what: points([
              'Xây dựng hệ thống 24 giả thuyết (H01–H24) chia thành 5 nhóm chức năng nghiên cứu.',
              'Thiết lập điều kiện biên dương (H05), các phương pháp bao đóng miền (IBP, CROWN, Lipschitz) và mô hình chi phí runtime (H24).'
            ]),
            why: points([
              'Cung cấp chứng minh toán học nghiêm ngặt rằng nếu x<sub>t</sub> ∈ D(x<sub>0</sub>) thì nhãn mô hình chắc chắn không đổi.',
              'Định lượng điều kiện hòa vốn kinh tế để việc tái sử dụng thực sự mang lại lợi ích về tốc độ.'
            ]),
            results: points([
              'Chứng minh điều kiện đủ để bảo toàn quyết định thông qua biên logit trực tiếp Δ<sub>y*, j</sub> > 0.',
              'Thiết lập công thức chi phí đầy đủ: T<sub>ECDR</sub> = T<sub>v</sub> + R · 0 + (1-R) · T<sub>f</sub>.'
            ]),
            good: [
              'Lý thuyết toán học chặt chẽ, chứng minh bảo toàn 100% quyết định mà không làm sai lệch hành vi mạng tham chiếu.',
              'Xây dựng mô hình chi phí thực tế, dự báo trước rủi ro chi phí kiểm tra miền lấn át lợi ích.'
            ],
            bad: [
              'Chưa có đo đạc thực nghiệm trên stream thực tế; chưa tính đến sự lỏng lẻo của các chặn IBP trên dữ liệu IMU.',
              'Mô hình chi phí lý thuyết chưa tính đến overhead điều phối bộ nhớ trong môi trường Python/CPU.'
            ]
          }),
          'Tổng kết Tuần 3 theo 4 câu hỏi: Làm gì, Vì sao, Kết quả gì, Cái gì tốt cái gì xấu.'),
        slide('w3-summary', 'The three-week story leads to one concrete reuse question', 'Reference, data and mechanism',
          flow(['Week 1: which reference?', 'Week 2: which inputs and computation?', 'Week 3: which reuse conditions?']) +
          lead('Can this fixed classifier preserve its decision on later sensor windows while avoiding enough computation to reduce complete processing cost?'),
          'Close with the research question and return to H24. The functional groups provide the map for the follow-up discussion. The detailed outcomes and their placement can be chosen separately.')
      ]
    }}[3];
  weekData.sections = [
      {id:'map', label:'Research map', description:'The query, anchor, region and five hypothesis groups.', slides:['w3-objects','w3-map']},
      {id:'input', label:'Input assumptions', description:'The role of overlap, activity annotations and event labels.', slides:['w3-group-input']},
      {id:'decision', label:'Decision & arithmetic', description:'Bounds, the central margin condition and numerical execution.', slides:['w3-group-decision','w3-h05','w3-bounds','w3-h22']},
      {id:'geometry', label:'Geometry & features', description:'Candidate domains, fixed overlap and aligned CNN features.', slides:['w3-group-geometry','w3-domains','w3-h13']},
      {id:'lifecycle', label:'Anchor lifecycle', description:'Fixed-anchor reuse, repeated queries and runtime responsibilities.', slides:['w3-group-lifecycle','w3-h16','w3-runtime']},
      {id:'cost', label:'Cost & next steps', description:'Useful coverage, complete cost, 4-part summary and the follow-up research tasks.', slides:['w3-group-cost','w3-h24','w3-obstacles','w3-next','w3-quadrant','w3-summary']}
    ];

  global.ECDR_PAGES.week3 = weekData;
})(typeof window !== 'undefined' ? window : globalThis);
