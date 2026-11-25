/* ECDR Presentation Page: Week 2
 * Defines slide contents and section mappings for Week 2.
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



  const weekData = {2: {
      title: 'Data and the actual CNN',
      subtitle: 'Raw HAPT signals, tensor construction, labels, subject splits and the model operations.',
      slides: [
        slide('w2-data', 'HAPT supplies six raw channels and twelve activity classes', 'The physical meaning of the data',
          table(['Property', 'Meaning'], [
            ['Participants and recordings', '30 participants and 61 recordings.'],
            ['Accelerometer', 'Three axes measured in g.'],
            ['Gyroscope', 'Three axes of angular velocity measured in rad/s.'],
            ['Sampling rate', '50 samples per second.'],
            ['Classes', 'Six basic activities and six postural transitions.'],
            ['Input representation', 'Raw inertial signals, rather than the supplied 561 engineered features.']
          ]),
          'Explain one sensor value before explaining a window. Six channels describe what is measured at each time. Twelve classes describe the activity vocabulary. Those dimensions are independent.'),
        slide('w2-waveform', 'A raw signal records movement across time', 'Each row in a sensor file is one time sample',
          figure('WEEK2/assets/data_raw_first_recording.png', 'Accelerometer, gyroscope and inherited labels for the first ten seconds of recording exp01', 'Example from recording exp01, subject 01. Acceleration uses g, angular velocity uses rad/s. Label 0 denotes unannotated time samples.'),
          'Explain the signal shape using a movement example. Avoid assigning gravity permanently to one named accelerometer axis. File row alignment does not independently measure physical clock synchronization.'),
        slide('w2-tensor', 'The builder transposes the recording before cutting windows', 'From sensor files to a channel-first tensor',
          flow(['Accelerometer (T,3) + gyroscope (T,3)', 'Join channels: (T,6)', 'Transpose: (6,T)', 'Slice 128 columns: (6,128)', 'Batch windows: (B,6,128)'], true) +
          lead('T is the recording length. B is the number of windows in a batch.'),
          'Point to one matrix cell as a measurement and one column after transpose as a time sample. Transpose changes axis order. It does not create features or additional sensors.'),
        slide('w2-window', 'Window length and stride define the observation timeline', '128 samples per window and 64 samples per step',
          pair(formula('128 / 50 = 2.56 s', 'Window duration under the selected convention.') + formula('64 / 50 = 1.28 s', 'Time between successive windows.'),
            flow(['Current: [ A | B ]', 'Next: [ B | C ]'], true) + lead('Each block contains 64 time samples on all six channels.')) +
          points(['The known overlap changes local position in the tensor.', 'A window stays within a single recording.', 'The chosen window length is a fixed study configuration.']),
          'Use A, B and C to show what is dropped, retained and appended. The overlap is a property of the input construction. It does not imply that two tensors are close at the same local coordinate or have the same CNN decision.'),
        slide('w2-labels', 'The label policy distinguishes pure, majority and unresolved windows', 'The complete window is the denominator',
          table(['Window type', 'Rule and use'], [
            ['Pure', 'All 128 time samples belong to one accepted class.'],
            ['Majority', 'One class occupies at least 60% of 128 samples, requiring at least 77.'],
            ['Ambiguous', 'No class satisfies the accepted supervised policy.'],
            ['Unlabeled', 'No accepted activity target is available for the window.']
          ]) +
          lead('Ambiguous and unlabeled windows remain in the sequential stream. They are excluded from supervised training and accuracy calculations.'),
          'Explain why the stream and supervised subsets have different sizes. Unlabeled samples must not silently become a rest class. Annotation index origin is an inherited data-provenance question that should be checked separately.'),
        slide('w2-counts', 'Streaming and supervised questions use different denominators', 'Subject-disjoint train, selection and test splits',
          table(['Split', 'Subjects', 'All stream windows', 'Supervised windows'], [
            ['Train', '17', '9,978', '6,955'],
            ['Validation', '4', '2,422', '1,751'],
            ['Test', '9', '5,053', '3,572'],
            ['Total', '30', '17,453', '12,278']
          ]) +
          lead('The subject split keeps windows from one person in a single split.'),
          'State the denominator before reporting any quantity. Full-stream behavior uses every window. Supervised metrics use the accepted labeled subset. Overlapping windows are not independent observations.'),
        slide('w2-normalization', 'Each channel uses statistics fitted only on the training subset', 'The same transform serves selection, test and runtime inputs',
          formula('z[c,t] = (x[c,t] − μ[c]) / σ[c]', 'μ and σ are fixed per-channel statistics fitted on 6,955 supervised training windows.') +
          points(['Pool the training windows and their time samples when estimating each channel statistic.', 'Reuse the same stored statistics for the held-out splits.', 'Keep the transform fixed when defining a region or comparing reference decisions.']),
          'This is not normalization independently fitted to each new window. Explain why fitting the transform on held-out data would change the intended split boundary. Input normalization and BatchNorm inside the CNN are different operations.'),
        slide('w2-support', 'Postural transitions have less support than the basic activities', 'Data balance affects the interpretation of model quality',
          table(['Data view', 'Meaning'], [
            ['Pure basic windows', '10,017 windows across the complete corpus.'],
            ['Pure transition windows', '344 windows across the complete corpus.'],
            ['Ratio', 'About 29.1 basic windows per transition window in this pure-window view.'],
            ['Evaluation implication', 'Inspect per-class support and macro-F1 alongside aggregate accuracy.']
          ]),
          'These counts refer to the pure-window view across the corpus, not the complete supervised test denominator. Do not mix support from one subset with metrics from another. Limited support motivates a closer look at transitions without proving the cause of every error.'),
        slide('w2-support-plot', 'The label policy and class support describe different views of the data', 'The complete stream and the pure-window subset',
          figure('WEEK2/assets/data_policy_support.png', 'Window-policy counts and pure class support by subject split', 'Left: all 17,453 chronological windows by label policy. Right: pure-window support by activity ID and subject split.'),
          'Use the left plot to explain why not every stream window enters supervised learning. Use the right plot to compare basic and transition support. The right plot is restricted to pure windows.'),
        slide('w2-cnn', 'Three convolution blocks feed a position-sensitive head', 'The actual SmallCNN1D architecture',
          table(['Block', 'Output excluding batch', 'Role'], [
            ['Conv 6→32, k5, p2; BN; ReLU; Pool2', '32 × 64', 'Combine six channels over a local time neighborhood.'],
            ['Conv 32→64, k3, p1; BN; ReLU; Pool2', '64 × 32', 'Combine features and reduce the time dimension.'],
            ['Conv 64→64, k3, p1; BN; ReLU', '64 × 32', 'Create the final temporal features.'],
            ['Flatten; Linear 2048→128; ReLU', '128', 'Combine features with position-dependent weights.'],
            ['Linear 128→12', '12 logits', 'Produce class scores.']
          ]),
          'A CNN1D filter moves across time and combines channels. The 32 or 64 feature channels are learned representations, not new physical sensors. Argmax can select the output class directly from logits.'),
        slide('w2-operators', 'Each operator changes the representation in a specific way', 'How the window becomes features',
          table(['Operator', 'What it does'], [
            ['Conv1D', 'Apply learned local filters across time.'],
            ['BatchNorm in eval mode', 'Apply the saved running statistics and learned affine parameters.'],
            ['ReLU', 'Replace negative activations with zero.'],
            ['MaxPool', 'Keep a local maximum while reducing temporal resolution.'],
            ['Flatten', 'Keep feature positions in an ordered vector.'],
            ['Linear', 'Mix all connected features with learned weights.']
          ]),
          'Explain the operators before showing their interval counterparts in Week 3. The final head may depend on the full input window even though one convolutional output has a smaller receptive field.'),
        slide('w2-model-cost', 'Parameter count and arithmetic work describe different costs', 'The dense head dominates the parameter count',
          table(['Quantity', 'Reference value', 'Interpretation'], [
            ['Trainable parameters', '283,692', 'Learned scalar values in the network.'],
            ['Linear 2048→128 parameters', '262,272; about 92.45%', 'A large share of the stored trainable parameters.'],
            ['Conv/Linear MACs', '1,172,992 per input', 'Arithmetic in the counted convolution and linear layers.'],
            ['Final-feature receptive field', '20 samples, stride 4', 'The input support of one interior final feature position.']
          ]) +
          lead('A percentage of parameters is not a measured percentage of latency.'),
          'Distinguish model storage, arithmetic work and elapsed time. Pooling, normalization, memory access and runtime overhead have separate costs. A smaller model is not automatically faster in every environment.'),
        slide('w2-training', 'Training and checkpoint selection use separate data roles', 'The recorded training configuration',
          table(['Setting', 'Configuration'], [
            ['Objective and optimizer', 'CrossEntropyLoss with Adam.'],
            ['Learning rate', '0.001, cosine decay to 0.00001.'],
            ['Weight decay', '0.0001.'],
            ['Batch and training limit', '128 windows; at most 30 epochs.'],
            ['Early stopping', 'Patience 10.'],
            ['Independent seeds', '42, 43 and 44.'],
            ['Selection rule', 'Highest validation macro-F1; selected seed 43, epoch 8.'],
            ['Class weights', 'None in this recorded configuration.']
          ]),
          'Explain the data role of the validation split. This is the selected training configuration, not a new recommendation. Test data report the selected model rather than choose it.'),
        slide('w2-metrics', 'Accuracy and macro-F1 answer different classification questions', 'Metrics for the selected reference',
          pair(formula('Accuracy = correct predictions / labeled test windows', 'The selected reference reports 3,181 / 3,572 = 89.05%.'),
            formula('Macro-F1 = mean of the twelve class F1 scores', 'The selected reference reports 0.7763 on the same supervised test subset.')) +
          lead('Macro-F1 gives each class equal weight. The full-stream test count is 5,053, which is a different denominator.'),
          'Explain these as classification results for the existing reference. The metric denominator is the accepted labeled test subset. Neither number describes the fraction of queries that can reuse an anchor decision.'),
        slide('w2-errors', 'The confusion matrix identifies errors that need explanation', 'Rows are true classes and columns are predictions',
          table(['Observed confusion', 'Count'], [
            ['Sitting predicted as Standing', '57'],
            ['Standing predicted as Sitting', '106']
          ]) +
          points(['Basic-activity class F1 values range from about 0.82 to 0.99.', 'Transition-class F1 values range from about 0.55 to 0.74.', 'Use per-class support when interpreting these ranges.']),
          'Focus on two observed confusion pairs. The counts describe model errors. They do not establish a physical cause for each mistake. Changes to labels, data policy or weights need a separately identified reference version.'),
        slide('w2-confusion-plot', 'The complete confusion matrix retains all twelve classes', 'Reference detail for the classification discussion',
          figure('WEEK2/assets/model_confusion_matrix.png', 'Twelve-class supervised test confusion matrix', 'Ground-truth classes are on rows and predictions are on columns. This plot uses the supervised test subset.'),
          'Use this view when the audience asks about another class. In the main explanation, return to the two Sitting/Standing confusion counts rather than reading every cell.', {support:true}),
        slide('w2-extraction-code', 'Window construction follows the channel-first layout', 'Implementation reference',
          code('signals = concatenate([acc, gyro], axis=1)  # (T, 6)\nchannel_first = signals.T                    # (6, T)\nfor start in range(0, T - 128 + 1, 64):\n    window = channel_first[:, start:start + 128]\n    # window shape: (6, 128)') +
          lead('The example shows the data layout. Recording boundaries and label handling belong to the surrounding builder.'),
          'Show the transpose before the slice. This matches the construction order rather than suggesting that the CNN automatically transposes its input.', {support:true}),
        slide('w2-normalization-code', 'The stored transform is reused for every split', 'Implementation reference',
          code('mean = x_train.mean(axis=(0, 2))\nstd = maximum(x_train.std(axis=(0, 2)), 1e-6)\n\ndef normalize(x):\n    return (x - mean[None, :, None]) / std[None, :, None]') +
          lead('x_train contains the accepted supervised training windows. The statistics are stored with the reference.'),
          'Explain the axes: batch and time contribute to each channel statistic. This short example illustrates the transform and is not a replacement for the complete data pipeline.', {support:true}),
        slide('w2-quadrant', 'Tuần 2: Khung phân tích 4 câu hỏi', 'Dữ liệu HAPT & Khảo sát cấu trúc mạng SmallCNN1D',
          analysisQuad({
            what: points([
              'Kiểm toán 17,453 cửa sổ từ 30 đối tượng HAPT (chia tách Train 17, Val 4, Test 9 theo đối tượng).',
              'Phân tích chi tiết từng toán tử trong SmallCNN1D (Conv1D, BatchNorm, MaxPool, ReLU, Flatten, Linear) và đếm phép tính MACs.'
            ]),
            why: points([
              'Xác lập tính toàn vẹn của dữ liệu và hiểu sâu cấu trúc toán tử trước khi xây dựng lý thuyết bao đóng hình học.',
              'Tránh rò rỉ dữ liệu (data leakage) bằng cách chuẩn hóa Z-score độc lập chỉ fit trên tập huấn luyện.'
            ]),
            results: points([
              'Đo đạc chính xác 1,172,992 MACs/cửa sổ và độ trễ forward CPU ~0.14 ms.',
              'Phát hiện 6 lớp chuyển thế chỉ có 344 cửa sổ thuần (&lt; 3.5% tập dữ liệu), dẫn đến 18 rào cản cổng Gate G2.'
            ]),
            good: [
              'Quy trình chia tách đối tượng nghiêm ngặt loại trừ 100% rủi ro rò rỉ dữ liệu giữa train/val/test.',
              'Phân tích toán tử tường minh, xác định chính xác số lượng tham số và phép tính cần thiết.'
            ],
            bad: [
              'Dữ liệu mất cân bằng trầm trọng khiến macro-F1 của các lớp chuyển thế chỉ đạt mức trung bình yếu (0.68).',
              'Stride 64 (1.28 s) làm mất liên kết pha chuyển động giữa hai cửa sổ liên tiếp.'
            ]
          }),
          'Tổng kết Tuần 2 theo 4 câu hỏi: Làm gì, Vì sao, Kết quả gì, Cái gì tốt cái gì xấu.'),
        slide('w2-handoff', 'Week 3 studies the decision of this exact pipeline', 'From physical data to mathematical conditions',
          flow(['Raw input x', 'Stored transform N(x)', 'Frozen CNN f', 'Reference logits F(x)', 'Decision h(x)']) +
          formula('F(x) = f(N(x));  h(x) = argmax F(x)') +
          lead('The next question is what conditions would allow a later input to keep the anchor decision without another full evaluation.'),
          'The audience now knows the input shape, data roles, transform and actual head architecture. Use that concrete pipeline when introducing a domain and its margin conditions.')
      ]
    }}[2];
  weekData.sections = [
      {id:'signals', label:'Raw signals', description:'The physical channels and a recording example.', slides:['w2-data','w2-waveform']},
      {id:'windows', label:'Windows & labels', description:'Tensor construction, overlap and accepted labels.', slides:['w2-tensor','w2-window','w2-labels']},
      {id:'protocol', label:'Data protocol', description:'Subject splits, normalization and class support.', slides:['w2-counts','w2-normalization','w2-support','w2-support-plot']},
      {id:'cnn', label:'CNN architecture', description:'The actual graph, its operators and counted computation.', slides:['w2-cnn','w2-operators','w2-model-cost']},
      {id:'training', label:'Training & results', description:'Checkpoint selection, classification metrics and errors.', slides:['w2-training','w2-metrics','w2-errors','w2-confusion-plot']},
      {id:'handoff', label:'Code & handoff', description:'Builder examples, 4-part summary and the transition to Week 3.', slides:['w2-extraction-code','w2-normalization-code','w2-quadrant','w2-handoff']}
    ];

  global.ECDR_PAGES.week2 = weekData;
})(typeof window !== 'undefined' ? window : globalThis);
