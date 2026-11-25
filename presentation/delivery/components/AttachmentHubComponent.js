/* ECDR Component: AttachmentHubComponent.js
 * Dedicated Interactive File Attachment & Reproducibility Hub
 * (Direct local file links, hashes, terminal commands, and one-shot execution script)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  function render() {
    const files = [
      {
        category: 'scripts',
        badge: 'EXECUTABLE',
        title: 'scripts/week4_audit.py',
        path: '/home/pro/Downloads/research/scripts/week4_audit.py',
        desc: 'Bước 0 Preflight: Kiểm tra tính nguyên vẹn của 214 tệp baseline bảo vệ từ Tuần 1–3 và xác thực mô hình SmallCNN1D checkpoint.',
        command: '.venv/bin/python scripts/week4_audit.py check'
      },
      {
        category: 'scripts',
        badge: 'PIPELINE',
        title: 'scripts/week4_run.py',
        path: '/home/pro/Downloads/research/scripts/week4_run.py',
        desc: 'Thực thi đầy đủ luồng thực nghiệm: geometry sweep, policy selection, verify seal, stream replay (5,053 cửa sổ), CPU benchmark, memory analysis.',
        command: '.venv/bin/python scripts/week4_run.py replay --run-id week4_initial_20261103 --split test --sealed-policy output/week4/week4_initial_20261103/selected_policy.json'
      },
      {
        category: 'scripts',
        badge: 'TABLES & LEDGER',
        title: 'scripts/week4_tables.py',
        path: '/home/pro/Downloads/research/scripts/week4_tables.py',
        desc: 'Sinh ma trận đối sánh 17 phương pháp (M00–M16), sổ cái 24 giả thuyết (H01–H24), tam đề H24 và báo cáo tóm tắt Markdown.',
        command: '.venv/bin/python scripts/week4_tables.py --run-id week4_initial_20261103'
      },
      {
        category: 'scripts',
        badge: 'SCALING STUDY',
        title: 'scripts/post_w4_feasibility_study.py',
        path: '/home/pro/Downloads/research/scripts/post_w4_feasibility_study.py',
        desc: 'Nghiên cứu điểm hòa vốn kiến trúc (>26.1 MMACs), phân tích khe hở độ lệch pha và vi chuẩn tối ưu bộ đệm C++.',
        command: '.venv/bin/python scripts/post_w4_feasibility_study.py'
      },
      {
        category: 'scripts',
        badge: 'GATE G2 CLOSURE',
        title: 'scripts/post_w4_g2_full_closure.py',
        path: '/home/pro/Downloads/research/scripts/post_w4_g2_full_closure.py',
        desc: 'Khép lại 18 rào cản Gate G2 trên 12 lớp: Span-Anchored Windowing + Focal Loss, đạt Macro-F1 0.8537 (+10.95% chuyển thế).',
        command: '.venv/bin/python scripts/post_w4_g2_full_closure.py'
      },
      {
        category: 'scripts',
        badge: 'VERIFICATION',
        title: 'scripts/week4_verify_delivery.py',
        path: '/home/pro/Downloads/research/scripts/week4_verify_delivery.py',
        desc: 'Kiểm tra toàn diện tính toàn vẹn và tiêu chí nghiệm thu bàn giao của toàn bộ Tuần 4.',
        command: '.venv/bin/python scripts/week4_verify_delivery.py'
      },
      {
        category: 'reports',
        badge: 'PDF BÁO CÁO (VI)',
        title: '04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_VI.pdf',
        path: '/home/pro/Downloads/research/04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_VI.pdf',
        desc: 'Báo cáo tổng hợp khoa học toàn diện Tuần 4 (Tiếng Việt, chuẩn IEEE LaTeX, 220 KB, 14 trang).',
        command: 'xdg-open 04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_VI.pdf'
      },
      {
        category: 'reports',
        badge: 'PDF REPORT (EN)',
        title: '04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_EN.pdf',
        path: '/home/pro/Downloads/research/04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_EN.pdf',
        desc: 'Comprehensive English Scientific Synthesis Report (IEEE Trans. Mobile Computing style, 179 KB).',
        command: 'xdg-open 04_SYNTHESIS/week4/reports/ECDR_Full_Synthesis_Report_EN.pdf'
      },
      {
        category: 'reports',
        badge: 'WHY & DIRECTIONS',
        title: '04_SYNTHESIS/week4/reports/ECDR_Mechanism_Why_and_New_Directions_VI.pdf',
        path: '/home/pro/Downloads/research/04_SYNTHESIS/week4/reports/ECDR_Mechanism_Why_and_New_Directions_VI.pdf',
        desc: 'Chuyên khảo giải trình cơ chế vật lý "Vì sao kết quả như vậy" và 5 hướng nghiên cứu chiến lược (198 KB).',
        command: 'xdg-open 04_SYNTHESIS/week4/reports/ECDR_Mechanism_Why_and_New_Directions_VI.pdf'
      },
      {
        category: 'reports',
        badge: 'WALKTHROUGH GUIDE',
        title: '04_SYNTHESIS/week4/reports/REPRODUCE.md',
        path: '/home/pro/Downloads/research/04_SYNTHESIS/week4/reports/REPRODUCE.md',
        desc: 'Tài liệu hướng dẫn tái lập chính xác 100% từng lệnh, checksum SHA256 và bằng chứng preflight audit.',
        command: 'cat 04_SYNTHESIS/week4/reports/REPRODUCE.md | head -n 45'
      },
      {
        category: 'data',
        badge: 'CSV MATRIX',
        title: 'output/week4/week4_initial_20261103/tables/comparison_matrix.csv',
        path: '/home/pro/Downloads/research/output/week4/week4_initial_20261103/tables/comparison_matrix.csv',
        desc: 'Ma trận đối sánh 17 phương pháp: latency, accuracy, macro-F1, reuse count, speed ratio, net savings.',
        command: 'column -s, -t < output/week4/week4_initial_20261103/tables/comparison_matrix.csv | head -n 18'
      },
      {
        category: 'data',
        badge: 'CSV LEDGER',
        title: 'output/week4/week4_initial_20261103/tables/hypotheses.csv',
        path: '/home/pro/Downloads/research/output/week4/week4_initial_20261103/tables/hypotheses.csv',
        desc: 'Sổ cái trạng thái kiểm định của 24 giả thuyết (H01–H24) với nhân chứng và bằng chứng cụ thể.',
        command: 'column -s, -t < output/week4/week4_initial_20261103/tables/hypotheses.csv | head -n 25'
      },
      {
        category: 'data',
        badge: 'H24 TRILEMMA',
        title: 'output/week4/week4_initial_20261103/tables/h24_conjunction.csv',
        path: '/home/pro/Downloads/research/output/week4/week4_initial_20261103/tables/h24_conjunction.csv',
        desc: 'Bảng đánh giá hội tam đề H24: Soundness ∧ Coverage ∧ Speedup cho toàn bộ 17 phương pháp.',
        command: 'column -s, -t < output/week4/week4_initial_20261103/tables/h24_conjunction.csv | head -n 18'
      },
      {
        category: 'tests',
        badge: 'TEST SUITE',
        title: 'tests/test_week4_*.py',
        path: '/home/pro/Downloads/research/tests/test_week4_accounting.py',
        desc: 'Bộ 280+ bài kiểm thử tự động xác thực tính đúng đắn số học, bảo toàn nhãn và nghịch lý độ trễ.',
        command: '.venv/bin/python -m unittest discover -s tests -p "test_week4_*.py" -v'
      }
    ];

    const oneShotScript = `#!/usr/bin/env bash
# ==============================================================================
# ECDR WEEK 4: ONE-SHOT REPRODUCIBILITY & VERIFICATION PIPELINE
# Chạy chuỗi lệnh này để tái lập 100% mọi đo đạc, kiểm thử và báo cáo Tuần 4.
# ==============================================================================
set -e
echo "=== Bước 0: Kiểm tra tính nguyên vẹn baseline (214 tệp bảo vệ) ==="
.venv/bin/python scripts/week4_audit.py check

echo "=== Bước 1: Replay test stream (5,053 cửa sổ qua 17 phương pháp) ==="
.venv/bin/python scripts/week4_run.py replay \\
  --run-id week4_initial_20261103 \\
  --split test \\
  --sealed-policy output/week4/week4_initial_20261103/selected_policy.json

echo "=== Bước 2: Sinh ma trận so sánh, sổ cái 24 giả thuyết & tổng kết ==="
.venv/bin/python scripts/week4_tables.py --run-id week4_initial_20261103

echo "=== Bước 3: Nghiên cứu mở rộng Post-W4 (Điểm hòa vốn 26.1 MMACs & Stride 1) ==="
.venv/bin/python scripts/post_w4_feasibility_study.py

echo "=== Bước 4: Khép lại Gate G2 K=12 (Span-Anchored + Focal Loss) ==="
.venv/bin/python scripts/post_w4_g2_full_closure.py

echo "=== Bước 5: Chạy bộ kiểm thử tự động (280+ assertions) ==="
.venv/bin/python -m unittest discover -s tests -p 'test_week4_*.py' -v

echo "=== Bước 6: Xác thực nghiệm thu bàn giao chính thức ==="
.venv/bin/python scripts/week4_verify_delivery.py

echo "=== HOÀN TẤT THÀNH CÔNG: Toàn bộ kết quả Tuần 4 đã được tái lập 100%! ==="
`;

    return `
      <div class="component-card attachment-hub-component">
        <div class="component-header">
          <span class="component-badge">TRUNG TÂM TÁI LẬP (REPRODUCIBILITY HUB)</span>
          <h3 class="component-title">Đính kèm toàn bộ tệp mã nguồn, dữ liệu thực nghiệm và báo cáo PDF</h3>
          <p class="component-desc">Toàn bộ tài nguyên được đính kèm trực tiếp để bạn có thể chạy, kiểm tra và tái lập 100% kết quả Tuần 4.</p>
        </div>

        <div class="hub-tabs">
          <button class="hub-tab-btn active" type="button" data-hub-tab="all">Tất cả (${files.length})</button>
          <button class="hub-tab-btn" type="button" data-hub-tab="scripts">Scripts Python (${files.filter(f => f.category==='scripts').length})</button>
          <button class="hub-tab-btn" type="button" data-hub-tab="reports">Báo cáo PDF (${files.filter(f => f.category==='reports').length})</button>
          <button class="hub-tab-btn" type="button" data-hub-tab="data">Dữ liệu CSV/JSON (${files.filter(f => f.category==='data').length})</button>
          <button class="hub-tab-btn" type="button" data-hub-tab="tests">Kiểm thử Unittest (${files.filter(f => f.category==='tests').length})</button>
        </div>

        <div class="hub-grid">
          ${files.map((file, i) => `
            <div class="hub-card" data-category="${file.category}">
              <div class="hub-card-top">
                <span class="hub-badge badge-${file.category}">${file.badge}</span>
                <span class="hub-index">#${String(i+1).padStart(2, '0')}</span>
              </div>
              <h4 class="hub-card-title">
                <a href="file://${file.path}" title="Mở tệp cục bộ">${file.title}</a>
              </h4>
              <p class="hub-card-desc">${file.desc}</p>
              <div class="hub-command-box">
                <code>${escape(file.command)}</code>
                <button class="copy-pill-btn" type="button" onclick="navigator.clipboard.writeText('${escape(file.command)}'); this.textContent='Đã chép!'; setTimeout(()=>this.textContent='Sao chép lệnh', 2000);">Sao chép lệnh</button>
              </div>
            </div>
          `).join('')}
        </div>

        <div class="hub-oneshot-box">
          <div class="oneshot-header">
            <div>
              <strong>Kịch bản One-Shot Bash Script: Thực thi tái lập toàn bộ chỉ bằng một lệnh</strong>
              <p>Chạy chuỗi lệnh này trong môi trường <code>.venv</code> để tự động hóa toàn bộ quy trình kiểm định Tuần 4.</p>
            </div>
            <button class="primary copy-script-btn" type="button" onclick="navigator.clipboard.writeText(document.getElementById('oneshot-script-code').innerText); this.textContent='Đã chép toàn bộ kịch bản!'; setTimeout(()=>this.textContent='Sao chép toàn bộ kịch bản', 2500);">Sao chép toàn bộ kịch bản</button>
          </div>
          <pre><code id="oneshot-script-code" class="language-bash">${escape(oneShotScript)}</code></pre>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.AttachmentHubComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
