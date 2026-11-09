/* English delivery content, shared by the portal and its presentation view. */
(function () {
  'use strict';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const lead = text => `<p class="lead">${text}</p>`;
  const points = items => `<ul class="points">${items.map(text => `<li>${text}</li>`).join('')}</ul>`;
  const table = (headers, rows) => `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const flow = (items, vertical = false) => `<ol class="flow ${vertical ? 'vertical' : ''}">${items.map(x => `<li>${x}</li>`).join('')}</ol>`;
  const pair = (left, right) => `<div class="columns"><div>${left}</div><div>${right}</div></div>`;
  const formula = (text, explanation = '') => `<div class="formula">${text}</div>${explanation ? `<p class="caption">${explanation}</p>` : ''}`;
  const code = text => `<pre><code>${escape(text)}</code></pre>`;
  const assetRoot = typeof document === 'undefined' ? '../' : new URL('../', document.currentScript.src).href;
  const figure = (path, alt, caption) => `<figure><img src="${assetRoot}${path}" alt="${escape(alt)}" loading="lazy"><figcaption>${caption}</figcaption></figure>`;
  const slide = (id, title, subtitle, content, notes, extra = {}) => ({id, title, subtitle, content, notes, ...extra});

  // Original IDs are retained. Group membership describes purpose, not a verdict.
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

  const weeks = {
    1: {
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
        slide('w1-summary', 'Week 2 expands the data and CNN steps in the opening map', 'The handoff from the reference definition',
          flow(['Raw sensor values', 'Window tensor', 'Label and split', 'Fixed normalization', 'CNN features and logits']) +
          lead('The next section explains what one input means and how the reference turns it into a decision.'),
          'Return to the opening map. The audience should now understand why a fixed reference is needed. Continue with the physical sensor data before introducing mathematical regions.')
      ]
    },
    2: {
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
        slide('w2-handoff', 'Week 3 studies the decision of this exact pipeline', 'From physical data to mathematical conditions',
          flow(['Raw input x', 'Stored transform N(x)', 'Frozen CNN f', 'Reference logits F(x)', 'Decision h(x)']) +
          formula('F(x) = f(N(x));  h(x) = argmax F(x)') +
          lead('The next question is what conditions would allow a later input to keep the anchor decision without another full evaluation.'),
          'The audience now knows the input shape, data roles, transform and actual head architecture. Use that concrete pipeline when introducing a domain and its margin conditions.')
      ]
    },
    3: {
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
        slide('w3-summary', 'The three-week story leads to one concrete reuse question', 'Reference, data and mechanism',
          flow(['Week 1: which reference?', 'Week 2: which inputs and computation?', 'Week 3: which reuse conditions?']) +
          lead('Can this fixed classifier preserve its decision on later sensor windows while avoiding enough computation to reduce complete processing cost?'),
          'Close with the research question and return to H24. The functional groups provide the map for the follow-up discussion. The detailed outcomes and their placement can be chosen separately.')
      ]
    }
  };

  const sectionPlans = {
    1: [
      {id:'problem', label:'Problem & objective', description:'The opening pipeline, full inference and research questions.', slides:['w1-opening','w1-full-inference','w1-objective','w1-questions']},
      {id:'reference', label:'Frozen reference', description:'What is frozen and which CNN defines the reference.', slides:['w1-freeze','w1-contract']},
      {id:'plan', label:'Research plan', description:'The lifecycle and scientific priorities.', slides:['w1-lifecycle','w1-priorities']},
      {id:'organization', label:'Project organization', description:'SWOT, communication, work packages and the weekly handoff.', slides:['w1-swot','w1-communication','w1-wbs','w1-governance']},
      {id:'handoff', label:'Code & handoff', description:'The forward function and the transition to Week 2.', slides:['w1-code','w1-summary']}
    ],
    2: [
      {id:'signals', label:'Raw signals', description:'The physical channels and a recording example.', slides:['w2-data','w2-waveform']},
      {id:'windows', label:'Windows & labels', description:'Tensor construction, overlap and accepted labels.', slides:['w2-tensor','w2-window','w2-labels']},
      {id:'protocol', label:'Data protocol', description:'Subject splits, normalization and class support.', slides:['w2-counts','w2-normalization','w2-support','w2-support-plot']},
      {id:'cnn', label:'CNN architecture', description:'The actual graph, its operators and counted computation.', slides:['w2-cnn','w2-operators','w2-model-cost']},
      {id:'training', label:'Training & results', description:'Checkpoint selection, classification metrics and errors.', slides:['w2-training','w2-metrics','w2-errors','w2-confusion-plot']},
      {id:'handoff', label:'Code & handoff', description:'Builder examples and the transition to Week 3.', slides:['w2-extraction-code','w2-normalization-code','w2-handoff']}
    ],
    3: [
      {id:'map', label:'Research map', description:'The query, anchor, region and five hypothesis groups.', slides:['w3-objects','w3-map']},
      {id:'input', label:'Input assumptions', description:'The role of overlap, activity annotations and event labels.', slides:['w3-group-input']},
      {id:'decision', label:'Decision & arithmetic', description:'Bounds, the central margin condition and numerical execution.', slides:['w3-group-decision','w3-h05','w3-bounds','w3-h22']},
      {id:'geometry', label:'Geometry & features', description:'Candidate domains, fixed overlap and aligned CNN features.', slides:['w3-group-geometry','w3-domains','w3-h13']},
      {id:'lifecycle', label:'Anchor lifecycle', description:'Fixed-anchor reuse, repeated queries and runtime responsibilities.', slides:['w3-group-lifecycle','w3-h16','w3-runtime']},
      {id:'cost', label:'Cost & next steps', description:'Useful coverage, complete cost and the follow-up research tasks.', slides:['w3-group-cost','w3-h24','w3-obstacles','w3-next','w3-summary']}
    ]
  };
  const journeyLabels = {1:'Problem and reference', 2:'Data and CNN', 3:'Hypotheses and reuse'};
  for (const [number, week] of Object.entries(weeks)) {
    const w = Number(number);
    week.sections = sectionPlans[w];
    const assigned = new Set();
    for (const section of week.sections) {
      for (const id of section.slides) {
        const item = week.slides.find(s => s.id === id);
        if (!item || assigned.has(id)) throw new Error(`Invalid section mapping: ${id}`);
        item.week = w;
        item.sectionId = section.id;
        assigned.add(id);
      }
    }
    if (assigned.size !== week.slides.length) throw new Error(`Unmapped Week ${w} slide`);
    const journey = `<ol class="journey-strip" aria-label="Three-week route">${Object.keys(weeks).map(n => `<li ${Number(n) === w ? 'aria-current="step"' : ''}><span>Week ${n}</span>${journeyLabels[n]}</li>`).join('')}</ol>`;
    const agenda = `<ol class="agenda-list">${week.sections.map((section, i) => {
      const first = week.slides.findIndex(s => s.id === section.slides[0]) + 2;
      const last = first + section.slides.length - 1;
      return `<li><button type="button" data-section-week="${w}" data-section-id="${section.id}" title="Open ${escape(section.label)}"><span class="agenda-number">${String(i + 1).padStart(2, '0')}</span><span class="agenda-topic"><strong>${escape(section.label)}</strong><span>${escape(section.description)}</span></span><span class="agenda-range">${first === last ? `Slide ${first}` : `Slides ${first}–${last}`}</span></button></li>`;
    }).join('')}</ol>`;
    week.slides.unshift(slide(`w${w}-agenda`, `Week ${w} overview`, 'The sections we will cover', journey + agenda,
      `Introduce the route for Week ${w}. Name each section briefly, then start with ${week.sections[0].label}. The highlighted section bar follows the talk as the slides change.`, {kind:'agenda', week:w, sectionId:null}));
  }

  const overviewIds = ['w1-agenda','w1-opening','w1-full-inference','w1-freeze','w2-agenda','w2-data','w2-tensor','w2-counts','w2-cnn','w3-agenda','w3-map','w3-h05','w3-h13','w3-h16','w3-h24'];
  const model = {weeks, hypotheses, groups, overviewIds};
  globalThis.ECDR_DELIVERY = model;
  if (typeof document === 'undefined') return;

  const weekLabel = w => `Week ${w}`;
  const article = (s, index, w) => `<article class="portal-section" id="${s.id}"><header class="section-heading"><div class="section-index">${String(index + 1).padStart(2,'0')}</div><div><p class="eyebrow">${weekLabel(w)}${s.focus ? ` / Focus ${s.focus}` : s.support ? ' / Implementation detail' : ''}</p><h2>${s.title}</h2><p class="subtitle">${s.subtitle}</p></div><button class="quiet open-slide" type="button" data-week="${w}" data-index="${index}">Present</button></header><div class="content">${s.content}</div><details class="speaker-notes"><summary>Speaker notes</summary><p>${s.notes}</p></details></article>`;
  const root = document.getElementById('app');
  let activeWeek = Number(document.body.dataset.startWeek || 1);
  let slideWeek = activeWeek;
  let slideIndex = 0;
  let previousFocus;
  const overview = document.body.dataset.overview === 'true';
  let presentingOverview = overview;
  const overviewDeck = overviewIds.map(id => Object.values(weeks).flatMap(w => w.slides).find(s => s.id === id));
  root.innerHTML = `
    <header class="site-header"><a class="brand" href="#">ECDR <span>Research delivery</span></a><nav aria-label="Week selection">${Object.keys(weeks).map(w => `<button type="button" class="week-tab" data-week="${w}">${weekLabel(w)}</button>`).join('')}</nav><button id="open-deck" class="primary" type="button">Present Week ${activeWeek}</button></header>
    <div class="page-intro"><p class="eyebrow">Weeks 1–3 / English edition</p><h1 id="week-heading"></h1><p id="week-description"></p><div class="intro-actions"><span id="section-count"></span><span>← / → change slides</span><span>N speaker notes</span><span>Esc return to portal</span></div></div>
    <main>${Object.entries(weeks).map(([w,week]) => `<section id="content-week-${w}" class="week-content" aria-label="${weekLabel(w)}"><nav class="contents" aria-label="${weekLabel(w)} contents">${week.slides.map((s,i) => `<a href="#${s.id}">${String(i+1).padStart(2,'0')} ${s.title}</a>`).join('')}</nav><div class="portal-body">${week.slides.map((s,i) => article(s,i,w)).join('')}</div></section>`).join('')}</main>
    <footer class="site-footer">Event-Conditioned Certified Decision Reuse / Weeks 1–3</footer>
    <dialog id="full-slide-modal" aria-labelledby="modal-slide-title"><div class="presentation-shell"><header class="deck-toolbar"><span class="brand">ECDR</span><nav aria-label="Presentation week">${Object.keys(weeks).map(w => `<button type="button" class="deck-week" data-week="${w}" title="Open the complete ${weekLabel(w)} deck">${weekLabel(w)}</button>`).join('')}</nav><span id="modal-slide-counter" aria-live="polite"></span><button id="notes-toggle" type="button" aria-pressed="false">Notes (N)</button><button id="fullscreen-toggle" type="button">Fullscreen (F)</button><button id="close-deck" type="button">Close (Esc)</button></header><div class="section-tracker"><p id="modal-section-location" aria-live="polite"></p><nav id="modal-section-trail" class="section-trail" aria-label="Sections in the current week"></nav></div><div id="slide-scroll" class="slide-scroll"><section class="slide-canvas"><p id="modal-slide-badge" class="eyebrow"></p><h2 id="modal-slide-title"></h2><p id="modal-slide-subtitle" class="subtitle"></p><div id="modal-slide-body" class="content"></div><aside id="modal-notes" class="notes-panel" hidden><h3>Speaker notes</h3><p id="modal-notes-text"></p></aside></section></div><footer class="deck-footer"><button id="prev-slide" type="button">← Previous</button><div id="modal-slide-pills" class="slide-pills" aria-label="Slide selection"></div><button id="next-slide" class="primary" type="button">Next →</button></footer></div></dialog>`;
  const modal = document.getElementById('full-slide-modal');
  const getDeck = () => presentingOverview ? overviewDeck : weeks[slideWeek].slides;
  function switchMasterWeek(w) {
    activeWeek = Number(w);
    if (!weeks[activeWeek]) activeWeek = 1;
    document.querySelectorAll('.week-content').forEach(el => { el.hidden = el.id !== `content-week-${activeWeek}`; });
    document.querySelectorAll('.week-tab').forEach(el => { el.setAttribute('aria-pressed', String(Number(el.dataset.week) === activeWeek)); });
    document.getElementById('week-heading').textContent = `${weekLabel(activeWeek)}: ${weeks[activeWeek].title}`;
    document.getElementById('week-description').textContent = weeks[activeWeek].subtitle;
    document.getElementById('section-count').textContent = `${weeks[activeWeek].sections.length} sections / ${weeks[activeWeek].slides.length} slides`;
    document.getElementById('open-deck').textContent = overview ? `Present ${overviewDeck.length}-slide overview` : `Present ${weekLabel(activeWeek)}`;
  }
  function renderSectionTracker(s, deck) {
    const sections = weeks[s.week].sections;
    const sectionIndex = sections.findIndex(section => section.id === s.sectionId);
    const section = sections[sectionIndex];
    const inSection = deck.filter(item => item.week === s.week && item.sectionId === s.sectionId);
    const location = section
      ? `Week ${s.week} / Section ${sectionIndex + 1} of ${sections.length}: ${section.label} / Slide ${inSection.findIndex(item => item.id === s.id) + 1} of ${inSection.length} in this section`
      : `Week ${s.week} overview / ${sections.length} sections ahead`;
    document.getElementById('modal-section-location').textContent = location;
    document.getElementById('modal-section-trail').innerHTML = `<button type="button" data-agenda-week="${s.week}" ${s.kind === 'agenda' ? 'aria-current="step"' : ''}>Overview</button>${sections.map((item, i) => {
      const state = i === sectionIndex ? 'current' : sectionIndex >= 0 && i < sectionIndex ? 'earlier' : 'upcoming';
      return `<button type="button" data-section-week="${s.week}" data-section-id="${item.id}" data-state="${state}" ${state === 'current' ? 'aria-current="step"' : ''}><span>${String(i + 1).padStart(2, '0')}</span>${escape(item.label)}</button>`;
    }).join('')}`;
    document.querySelector('#modal-section-trail [aria-current]')?.scrollIntoView?.({block:'nearest', inline:'nearest'});
  }
  function renderFullSlide() {
    const deck = getDeck();
    slideIndex = Math.max(0, Math.min(slideIndex, deck.length - 1));
    const s = deck[slideIndex];
    slideWeek = s.week;
    document.getElementById('modal-slide-counter').textContent = `${String(slideIndex+1).padStart(2,'0')} / ${String(deck.length).padStart(2,'0')}`;
    document.getElementById('modal-slide-badge').textContent = `${weekLabel(slideWeek)}${presentingOverview ? ' / Overview selection' : ''}${s.focus ? ` / Focus ${s.focus}` : ''}`;
    document.getElementById('modal-slide-title').textContent = s.title;
    document.getElementById('modal-slide-subtitle').textContent = s.subtitle;
    document.getElementById('modal-slide-body').innerHTML = s.content;
    document.getElementById('modal-notes-text').textContent = s.notes;
    document.getElementById('slide-scroll').scrollTop = 0;
    document.getElementById('prev-slide').disabled = slideIndex === 0;
    document.getElementById('next-slide').disabled = slideIndex === deck.length-1;
    document.querySelectorAll('.deck-week').forEach(el => {el.setAttribute('aria-pressed', String(Number(el.dataset.week) === slideWeek));});
    renderSectionTracker(s, deck);
    document.getElementById('modal-slide-pills').innerHTML = deck.map((s,i) => `<button type="button" data-index="${i}" title="${escape(s.title)}" ${i===slideIndex ? 'aria-current="step"' : ''}>${i+1}</button>`).join('');
    document.querySelector('#modal-slide-pills [aria-current]')?.scrollIntoView?.({block:'nearest',inline:'nearest'});
  }
  function openFullSlideModal(w = activeWeek, n = 1, useOverview = false) {
    presentingOverview = useOverview;
    slideWeek = Number(w);
    if (!weeks[slideWeek]) slideWeek = 1;
    slideIndex = Number(n)-1;
    previousFocus = document.activeElement;
    renderFullSlide();
    if (!modal.open) modal.showModal();
    document.body.classList.add('presenting');
    document.getElementById('close-deck').focus();
  }
  function closeFullSlideModal() {
    if (document.fullscreenElement === modal && document.exitFullscreen) document.exitFullscreen().catch(() => {});
    modal.close();
    document.body.classList.remove('presenting');
    previousFocus?.focus?.();
  }
  function moveSlide(delta) { slideIndex += delta; renderFullSlide(); }
  function jumpToSection(w, sectionId) {
    const section = weeks[w].sections.find(item => item.id === sectionId);
    if (!section) return;
    const index = weeks[w].slides.findIndex(item => item.id === section.slides[0]);
    if (!modal.open) openFullSlideModal(w, index + 1);
    else {presentingOverview=false;slideWeek=w;slideIndex=index;renderFullSlide();}
  }
  function showWeekAgenda(w) {
    const deck = getDeck();
    const index = deck.findIndex(item => item.id === `w${w}-agenda`);
    if (index >= 0) {slideIndex=index;renderFullSlide();}
  }
  function toggleNotes() {
    const panel = document.getElementById('modal-notes');
    panel.hidden = !panel.hidden;
    document.getElementById('notes-toggle').setAttribute('aria-pressed', String(!panel.hidden));
  }
  async function toggleFullscreen() {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await modal.requestFullscreen(); } catch { /* Native fullscreen may be unavailable in an embedded view. */ }
  }
  document.querySelectorAll('.week-tab').forEach(btn => btn.addEventListener('click', () => switchMasterWeek(btn.dataset.week)));
  document.querySelectorAll('.deck-week').forEach(btn => btn.addEventListener('click', () => {presentingOverview=false;slideWeek=Number(btn.dataset.week);slideIndex=0;renderFullSlide();}));
  document.querySelectorAll('.open-slide').forEach(btn => btn.addEventListener('click', () => openFullSlideModal(Number(btn.dataset.week), Number(btn.dataset.index)+1)));
  document.getElementById('open-deck').addEventListener('click', () => openFullSlideModal(activeWeek, 1, overview));
  document.getElementById('close-deck').addEventListener('click', closeFullSlideModal);
  document.getElementById('prev-slide').addEventListener('click', () => moveSlide(-1));
  document.getElementById('next-slide').addEventListener('click', () => moveSlide(1));
  document.getElementById('notes-toggle').addEventListener('click', toggleNotes);
  document.getElementById('fullscreen-toggle').addEventListener('click', toggleFullscreen);
  document.getElementById('modal-slide-pills').addEventListener('click', event => {const button=event.target.closest('button[data-index]');if(button){slideIndex=Number(button.dataset.index);renderFullSlide();}});
  root.addEventListener('click', event => {
    const sectionButton = event.target.closest('button[data-section-id]');
    if (sectionButton) jumpToSection(Number(sectionButton.dataset.sectionWeek), sectionButton.dataset.sectionId);
    const agendaButton = event.target.closest('button[data-agenda-week]');
    if (agendaButton) showWeekAgenda(Number(agendaButton.dataset.agendaWeek));
  });
  modal.addEventListener('cancel', event => {event.preventDefault();closeFullSlideModal();});
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if (!modal.open) {if(event.key.toLowerCase()==='p'){event.preventDefault();openFullSlideModal(activeWeek, 1, overview);}return;}
    if(['ArrowRight','PageDown',' '].includes(event.key)){event.preventDefault();moveSlide(1);}
    else if(['ArrowLeft','PageUp'].includes(event.key)){event.preventDefault();moveSlide(-1);}
    else if(event.key==='Escape'){event.preventDefault();closeFullSlideModal();}
    else if(event.key.toLowerCase()==='n'){event.preventDefault();toggleNotes();}
    else if(event.key.toLowerCase()==='f'){event.preventDefault();toggleFullscreen();}
    else if(['1','2','3'].includes(event.key)){event.preventDefault();presentingOverview=false;slideWeek=Number(event.key);slideIndex=0;renderFullSlide();}
  });
  Object.assign(globalThis, {switchMasterWeek, openFullSlideModal, closeFullSlideModal, renderFullSlide});
  switchMasterWeek(activeWeek);
})();
