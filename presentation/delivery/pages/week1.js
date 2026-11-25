/* ECDR Presentation Page: Week 1
 * Defines slide contents and section mappings for Week 1.
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



  const weekData = {1: {
      title: 'Problem and frozen reference',
      subtitle: 'The inference pipeline, the research objective and the reference used throughout the project.',
      slides: [
        slide('w1-opening', 'From sensor data to the Week 1 reference', 'The opening map for the full research story',
          flow(['HAPT raw signals + activity annotations', 'Window 128 / stride 64 + fixed label policy', 'Subject split 17 / 4 / 9 + train-only Z-score', 'Train SmallCNN1D K=12 with seeds 42, 43 and 44', 'Select checkpoint: seed 43, epoch 8', '<strong>Frozen reference: preprocessing + CNN + class decision</strong>'], true),
          'Start with this map. Explain the overall aim before introducing individual data operations. The first task is to define the classifier whose decision the research intends to preserve. Point to the transition from training and checkpoint selection to frozen inference. Week 2 expands the data and model steps. Week 3 expands the conditions for reusing an earlier decision.'),
        slide('w1-full-inference', 'Every new window runs through the full CNN', 'The standard streaming inference path',
          flow(['Six sensor channels', '6 × 128 window', 'Fixed normalization', 'SmallCNN1D', '12 logits', 'Argmax decision']) +
          lead('With stride 64 at 50 Hz, the next inference input arrives every 1.28 seconds.') +
          points(['Adjacent windows share part of the recording but occupy different positions in the input tensor.', 'The baseline processes each complete window as a separate input.', 'ECDR asks when an earlier decision could be reused at a lower complete processing cost.']),
          'Explain logits as the twelve class scores. The largest score determines the output class. Point out that overlap motivates a question about computation but does not establish decision invariance or a percentage of work saved.'),
        slide('w1-objective', 'ECDR aims to preserve a fixed reference decision', 'Event-Conditioned Certified Decision Reuse',
          pair(lead('A trusted anchor first runs the reference pipeline.') + points(['An event descriptor may propose a candidate region.', 'A later query may reuse the anchor decision only under the required conditions.']),
            lead('The complete path must also be economical.') + points(['Count preprocessing, event work, region work, membership, state and fallback.', 'Compare the resulting cost with processing the same stream through the reference.'])) +
          formula('h<sub>ECDR</sub>(x) = h<sub>reference</sub>(x)', 'The target concerns agreement with the reference classifier. Ground-truth classification quality is a separate quantity.'),
          'Introduce the research target in ordinary language: avoid a full evaluation when sufficient information already determines its decision. Explain that a wrong reference prediction can still be preserved, so reference accuracy and reuse agreement answer different questions.'),
        slide('w1-questions', 'Four research questions organize the work', 'The connection between the three weeks',
          table(['Question', 'Purpose'], [
            ['RQ1: Reference quality', 'Describe what the selected CNN classifies and how its quality is measured.'],
            ['RQ2: Decision preservation', 'Define conditions that keep the reference winner unchanged.'],
            ['RQ3: Coverage and cost', 'Ask whether incoming queries can use those conditions at a lower total cost.'],
            ['RQ4: Numerical soundness', 'Connect mathematical conditions with the executable preprocessing and CNN.']
          ]),
          'Use these questions to explain why model training, domain geometry, arithmetic and runtime cost are connected. The three-week story introduces the reference first, explains its data next and then examines reuse.'),
        slide('w1-freeze', 'The reference fixes more than the model weights', 'Training and selection end before reuse is studied',
          flow(['Train on training subjects', 'Select with the held-out selection split', 'Freeze the reference', 'Run inference']) +
          table(['Frozen component', 'Why it belongs to the reference'], [
            ['CNN graph and weights', 'They define the mapping from normalized input to logits.'],
            ['Input mean and standard deviation', 'They define the coordinates entering the CNN.'],
            ['Class order and decision rule', 'They define the meaning and selection of the output.'],
            ['BatchNorm state and evaluation mode', 'They define inference behavior for the saved network.']
          ]),
          'Explain that seed 43 is the random seed of the selected training run and epoch 8 is the selected checkpoint epoch. Training split, selection split and test split serve different purposes. Changing one frozen component creates a different reference function.'),
        slide('w1-contract', 'The reference is SmallCNN1D with twelve output classes', 'One input window contains six sensor channels and 128 time samples',
          table(['Property', 'Reference definition'], [
            ['Input', 'Float32 tensor (B, 6, 128), with B windows per batch.'],
            ['Time configuration', '50 Hz; 128 samples per window; stride 64.'],
            ['Feature extractor', 'Three same-padded Conv1D layers with BatchNorm and ReLU; MaxPool after the first two.'],
            ['Decision head', 'Flatten 2048, Linear 2048→128, ReLU, Linear 128→12.'],
            ['Trainable parameters', '283,692.'],
            ['Conv/Linear work', '1,172,992 multiply-accumulate operations per window.']
          ]),
          'The six input channels and the twelve output classes describe different things. This reference has a Flatten–MLP head. Results for a historical GAP network require their own architectural assumptions.'),
        slide('w1-lifecycle', 'The reference connects the research stages', 'An overview of the full lifecycle',
          table(['Stage', 'Role'], [
            ['1. Acquire raw signals', 'Load the accelerometer, gyroscope and annotations.'],
            ['2. Build the data pipeline', 'Create windows, assign labels, split subjects and fit normalization.'],
            ['3. Train the CNN', 'Fit the chosen architecture on the supervised training subset.'],
            ['4. Select a checkpoint', 'Use the held-out selection split.'],
            ['5. Freeze the reference', 'Keep preprocessing, graph, state and class order fixed.'],
            ['6. Run reference inference', 'Obtain the reference decision for each input.'],
            ['7. Study decision regions', 'Define the conditions under which an anchor decision can be reused.'],
            ['8. Study complete cost', 'Account for the full workload and all overhead.']
          ]),
          'Use the lifecycle after the opening flow. Stages seven and eight describe the research direction. A lifecycle diagram alone does not state that all stages have achieved their goals.'),
        slide('w1-priorities', 'The project separates the reference from the reuse mechanism', 'Scientific priorities',
          table(['Area', 'Priority'], [
            ['Reference integrity', 'Keep the input definition, preprocessing, architecture and decision rule consistent.'],
            ['Data and model understanding', 'Explain the raw measurements, labels, split, class support and CNN operations.'],
            ['Decision regions', 'Relate the anchor, candidate input set and all competing class margins.'],
            ['Practical reuse', 'Connect query membership and lifecycle with complete runtime cost.']
          ]),
          'This overview replaces a list of completion badges with the responsibilities of the research. Refer back to these responsibilities as the talk moves between weeks.'),
        slide('w1-swot', 'Project SWOT identifies the main trade-offs', 'A research planning view',
          table(['Area', 'Meaning for this project'], [
            ['Strengths', 'A finite reference graph and an explicit sensor-window contract make the target concrete.'],
            ['Weaknesses', 'Transition classes have limited support, while the position-sensitive head complicates reuse.'],
            ['Opportunities', 'Temporal overlap may support structured domains or reuse of selected intermediate activations.'],
            ['Threats', 'Loose bounds, floating-point effects and checking overhead can prevent useful savings.']
          ]),
          'Present this as project SWOT. It does not describe individual team members or claim that a target device has been benchmarked.'),
        slide('w1-communication', 'A proposed communication plan keeps decisions traceable', 'How changes to the reference and research claims should be discussed',
          points(['Use one shared issue and decision log for questions, changes and blockers.', 'Give each proposed task an owner, expected output and completion condition.', 'Review changes to the reference before using them in a comparison.', 'Bring the current plan and actual progress to each weekly review.']),
          'Describe this as a proposed workflow. The team still needs to confirm its channels, meeting cadence, people, effort and deadlines. Do not invent completed meetings or time records.'),
        slide('w1-wbs', 'The work breakdown follows the dependency order', 'From problem definition to a reuse study',
          table(['Work package', 'Output and dependency'], [
            ['Week 1: Problem and reference', 'Research questions, input contract, selected CNN and fixed reference definition.'],
            ['Week 2: Data and model', 'Signal structure, labels, split, normalization and explanation of the actual CNN.'],
            ['Week 3: Mathematical mechanism', 'Margin condition, candidate domains, anchor lifecycle and cost model.'],
            ['Follow-up work', 'Numerical arithmetic, useful geometry and implementation cost remain separate research tasks.']
          ]),
          'State the output of each package and what the next package needs from it. The labels organize the internal research weeks. They do not replace the tutor rubric or a confirmed schedule.'),
        slide('w1-governance', 'Eight questions describe each weekly handoff', 'A short structure for the tutor discussion',
          table(['Question', 'What to explain'], [
            ['What?', 'The reference pipeline and the research question for this stage.'],
            ['When?', 'The recorded work sequence and confirmed milestones.'],
            ['Why?', 'The dependency that makes the work necessary.'],
            ['How?', 'The chosen method and its assumptions.'],
            ['Who?', 'The confirmed owner of each task.'],
            ['Where?', 'The source or artifact location.'],
            ['How much?', 'The agreed effort and actual effort when available.'],
            ['What next?', 'The next output and the blocker it addresses.']
          ]),
          'Keep unconfirmed staffing, deadlines and effort out of the factual report. Use this structure briefly at a weekly handoff.'),
        slide('w1-code', 'The forward path has one clear responsibility', 'Implementation reference',
          code('def forward(self, x):\n    # x: (batch, 6, 128)\n    features = self.conv(x)\n    # features: (batch, 64, 32)\n    logits = self.head(features)\n    # logits: (batch, 12)\n    return logits') +
          lead('The CNN maps a normalized window to twelve logits. Preprocessing and the decision rule complete the reference pipeline.'),
          'Use this implementation slide only when the audience wants to connect the diagram to the code. The function returns logits. It does not select an event, build a domain or manage an anchor.', {support:true}),
        slide('w1-quadrant', 'Tuần 1: Khung phân tích 4 câu hỏi', 'Vấn đề nghiên cứu & Mô hình tham chiếu chuẩn',
          analysisQuad({
            what: points([
              'Định nghĩa bài toán suy luận streaming trên tập HAPT (6 kênh cảm biến, cửa sổ 128 mẫu, bước nhảy 64 mẫu).',
              'Huấn luyện, chọn lọc và đóng băng toàn bộ hàm tham chiếu SmallCNN1D (12 lớp, seed 43, epoch 8, 283,692 tham số).'
            ]),
            why: points([
              'Cần một hàm phân loại chuẩn, cố định và có thể tái lập 100% làm thước đo chân lý cho bài toán tái sử dụng.',
              'Tách biệt dứt khoát mục tiêu bảo toàn quyết định h<sub>ECDR</sub>(x) = h<sub>ref</sub>(x) với độ chính xác nhãn ground-truth.'
            ]),
            results: points([
              'Checkpoint tham chiếu `selected_best.pt` đạt độ chính xác kiểm thử ~89.85%, macro-F1 ~0.8018.',
              'Pipeline suy luận hoàn chỉnh được chuẩn hóa và xác thực toàn vẹn bằng mã băm SHA-256 cố định.'
            ]),
            good: [
              'Thiết lập pipeline tham chiếu minh bạch, có kiểm soát phiên bản và mã băm toàn vẹn tuyệt đối.',
              'Phân định ranh giới bài toán và câu hỏi nghiên cứu (RQ1–RQ4) rõ ràng ngay từ khởi điểm.'
            ],
            bad: [
              'Đầu phân loại dạng Flatten–MLP (2048→128→12) rất nhạy cảm với vị trí thời gian của tín hiệu.',
              '6 lớp chuyển thế (postural transitions) bị mất cân bằng dữ liệu nghiêm trọng, đe dọa cổng Gate G2.'
            ]
          }),
          'Tổng kết Tuần 1 theo đúng 4 câu hỏi: Làm gì, Vì sao, Kết quả gì, Cái gì tốt cái gì xấu.'),
        slide('w1-summary', 'Week 2 expands the data and CNN steps in the opening map', 'The handoff from the reference definition',
          flow(['Raw sensor values', 'Window tensor', 'Label and split', 'Fixed normalization', 'CNN features and logits']) +
          lead('The next section explains what one input means and how the reference turns it into a decision.'),
          'Return to the opening map. The audience should now understand why a fixed reference is needed. Continue with the physical sensor data before introducing mathematical regions.')
      ]
    }}[1];
  weekData.sections = [
      {id:'problem', label:'Problem & objective', description:'The opening pipeline, full inference and research questions.', slides:['w1-opening','w1-full-inference','w1-objective','w1-questions']},
      {id:'reference', label:'Frozen reference', description:'What is frozen and which CNN defines the reference.', slides:['w1-freeze','w1-contract']},
      {id:'plan', label:'Research plan', description:'The lifecycle and scientific priorities.', slides:['w1-lifecycle','w1-priorities']},
      {id:'organization', label:'Project organization', description:'SWOT, communication, work packages and the weekly handoff.', slides:['w1-swot','w1-communication','w1-wbs','w1-governance']},
      {id:'handoff', label:'Code & handoff', description:'The forward function, 4-part summary and the transition to Week 2.', slides:['w1-code','w1-quadrant','w1-summary']}
    ];

  global.ECDR_PAGES.week1 = weekData;
})(typeof window !== 'undefined' ? window : globalThis);
