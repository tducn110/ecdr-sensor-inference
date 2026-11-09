# -*- coding: utf-8 -*-
"""
Part C: CNN Foundations (Slides 08 - 09) - Visual-Rich, No Speech Scripts
"""

SLIDE_08_HTML = r"""
<section class="slide-frame" id="slide-08" data-slide-idx="8" data-part="Part C. CNN Foundations">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part C. CNN Foundations &bull; Slide 08</span>
      <h2 class="slide-title">Tại sao chọn CNN1D? Kiến trúc &amp; Cấu trúc Tính toán</h2>
      <p class="slide-subtitle">Cơ chế xử lý 6 kênh cảm biến qua Conv1D, BatchNorm, MaxPool và tính nhạy vị trí của Flatten Head</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">08 / 13</span>
      <span class="gate-badge gate-pass">Exact Computation</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual Architecture Flow Pipeline -->
    <div class="flow-diagram" style="padding: 0.5rem 0.8rem; font-size: 0.74rem;">
      <div class="flow-step">
        <div class="flow-step-title" style="color: var(--accent-cyan);">Input Tensor</div>
        <div class="mono" style="font-size: 0.72rem;">(B, 6, 128)</div>
        <div class="flow-step-desc">6 kênh &bull; 128 mẫu</div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-title" style="color: var(--accent-indigo);">Conv Block 1</div>
        <div class="mono" style="font-size: 0.72rem;">Conv1D(6&rarr;32, k=5)</div>
        <div class="flow-step-desc">BN + ReLU + Pool(2) &rarr; <span class="mono">(B, 32, 64)</span></div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-title" style="color: var(--accent-indigo);">Conv Block 2</div>
        <div class="mono" style="font-size: 0.72rem;">Conv1D(32&rarr;64, k=3)</div>
        <div class="flow-step-desc">BN + ReLU + Pool(2) &rarr; <span class="mono">(B, 64, 32)</span></div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-title" style="color: var(--accent-indigo);">Conv Block 3</div>
        <div class="mono" style="font-size: 0.72rem;">Conv1D(64&rarr;64, k=3)</div>
        <div class="flow-step-desc">BN + ReLU &rarr; <span class="mono">(B, 64, 32)</span></div>
      </div>
      <div class="flow-arrow">&rarr;</div>
      <div class="flow-step">
        <div class="flow-step-title" style="color: var(--accent-emerald);">Flatten + MLP Head</div>
        <div class="mono" style="font-size: 0.72rem;">Flatten(2.048) &rarr; 128</div>
        <div class="flow-step-desc">Linear &rarr; 12 Logits</div>
      </div>
    </div>

    <!-- Side-by-side: Layer MACs Audit Chart + Receptive Field Table -->
    <div class="grid-2" style="align-items: center; margin-top: 0.2rem;">
      <!-- Layer MACs Image Card -->
      <div class="card" style="padding: 0.5rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/model_layer_macs.png')">
          <img src="assets/model_layer_macs.png" alt="Layer MACs and Parameter Breakdown" style="max-height: 180px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Phân bổ 1.172.992 MACs và 283.692 Tham số theo Từng Tầng (Click phóng to)</div>
        </div>
      </div>

      <!-- Receptive Field Table & Flatten Mechanism -->
      <div class="card" style="padding: 0.6rem;">
        <h4 style="color: var(--accent-cyan); font-size: 0.88rem; margin-bottom: 0.3rem;">
          🔍 Quá trình Mở rộng Receptive Field
        </h4>
        <table class="custom-table" style="font-size: 0.72rem;">
          <thead>
            <tr>
              <th>Toán tử</th>
              <th>Receptive Field (\(r\))</th>
              <th>Thời gian</th>
              <th>Bước nhảy (\(j\))</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><strong>Conv1D (\(k=5\))</strong></td><td>\(r = 5\) mẫu</td><td>\(0,10\text{ s}\)</td><td>\(j = 1\)</td></tr>
            <tr><td><strong>MaxPool1D (\(k=2, s=2\))</strong></td><td>\(r = 6\) mẫu</td><td>\(0,12\text{ s}\)</td><td>\(j = 2\)</td></tr>
            <tr><td><strong>Conv1D (\(k=3\))</strong></td><td>\(r = 10\) mẫu</td><td>\(0,20\text{ s}\)</td><td>\(j = 2\)</td></tr>
            <tr><td><strong>MaxPool1D (\(k=2, s=2\))</strong></td><td>\(r = 12\) mẫu</td><td>\(0,24\text{ s}\)</td><td>\(j = 4\)</td></tr>
            <tr><td><strong>Conv1D (\(k=3\))</strong></td><td>\(r = 20\) mẫu</td><td>\(0,40\text{ s}\)</td><td>\(j = 4\)</td></tr>
          </tbody>
        </table>

        <div class="callout callout-info" style="margin-top: 0.4rem; padding: 0.4rem 0.6rem; font-size: 0.74rem;">
          <strong>Tính nhạy vị trí (Position-Sensitive Head):</strong><br>
          Vì head sử dụng <code>Flatten(2.048)</code> thay vì Global Average Pooling (GAP), <em>mô hình nhạy cảm với vị trí của đặc trưng</em> (không bất biến tuyệt đối với phép dịch thời gian).
        </div>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Nguồn tài liệu: <code>02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex</code> &bull; 1.172.992 MACs audited</span>
    <span class="mono" style="color: var(--accent-indigo);">283.692 Trainable Params</span>
  </div>
</section>
"""

SLIDE_09_HTML = r"""
<section class="slide-frame" id="slide-09" data-slide-idx="9" data-part="Part C. CNN Foundations">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part C. CNN Foundations &bull; Slide 09</span>
      <h2 class="slide-title">Frozen CNN: Hành vi Baseline &amp; Đánh giá Hiệu năng</h2>
      <p class="slide-subtitle">Phân tích ma trận nhầm lẫn 12x12, sự sụt giảm hiệu năng ở các lớp chuyển tiếp và trạng thái Gate G2</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">09 / 13</span>
      <span class="gate-badge gate-hold">Gate G2: HOLD</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Top 3 Metrics Banner -->
    <div class="grid-3" style="margin-bottom: 0.3rem;">
      <div class="card card-accent-cyan" style="text-align: center; padding: 0.55rem;">
        <div class="stat-value" style="font-size: 1.55rem;">89,05%</div>
        <div class="stat-label">Test Accuracy</div>
        <div class="stat-desc" style="font-size: 0.72rem;">3.181 / 3.572 cửa sổ kiểm tra dự đoán chính xác nhãn.</div>
      </div>

      <div class="card card-accent-indigo" style="text-align: center; padding: 0.55rem;">
        <div class="stat-value" style="font-size: 1.55rem;">0,7763</div>
        <div class="stat-label">Macro-F1 Score</div>
        <div class="stat-desc" style="font-size: 0.72rem;">Trung bình không trọng số F1 trên toàn bộ 12 lớp hoạt động.</div>
      </div>

      <div class="card card-accent-emerald" style="text-align: center; padding: 0.55rem;">
        <div class="stat-value" style="font-size: 1.55rem;">100% Repro</div>
        <div class="stat-label">Reference Reproducibility</div>
        <div class="stat-desc" style="font-size: 0.72rem;">0/5.053 mismatches so với checkpoint frozen Week 1 (Max error = 0.0).</div>
      </div>
    </div>

    <!-- Side-by-side: Confusion Matrix Image + Critical Analysis -->
    <div class="grid-2" style="align-items: center;">
      <!-- Confusion Matrix Image Card -->
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/model_confusion_matrix.png')">
          <img src="assets/model_confusion_matrix.png" alt="Confusion Matrix 12x12" style="max-height: 200px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Ma trận Nhầm lẫn 12x12 trên Tập Kiểm tra HAPT (3.572 Supervised Windows &bull; Click phóng to)</div>
        </div>
      </div>

      <!-- Critical Analysis & Gate Status -->
      <div class="card card-accent-amber" style="padding: 0.65rem;">
        <h4 style="color: var(--accent-amber); font-size: 0.88rem; margin-bottom: 0.25rem;">
          ⚠️ Lý do Gate G2 (Model Quality) tiếp tục ở HOLD:
        </h4>
        <ul style="font-size: 0.76rem; color: #cbd5e1; line-height: 1.5; padding-left: 1.1rem;">
          <li><strong>Phân cực hiệu năng rất rõ:</strong>
            <ul style="margin-top: 0.1rem; padding-left: 0.8rem;">
              <li>6 Hoạt động cơ bản (Đi bộ, Lên lầu, Nằm...): <strong>F1 rất cao (0,8196 &ndash; 0,9890)</strong>.</li>
              <li>6 Chuyển tiếp tư thế (Đứng sang Ngồi, Ngồi sang Nằm...): <strong>F1 sụt giảm mạnh (0,5496 &ndash; 0,7356)</strong>.</li>
            </ul>
          </li>
          <li><strong>Nhầm lẫn tư thế tĩnh cốt lõi:</strong> SITTING và STANDING gây ra <strong>163 ca dự đoán sai</strong> (57 Sitting đoán thành Standing; 106 Standing đoán thành Sitting) do hướng vector trọng trường gần tương đương.</li>
          <li><strong>Quản trị chất lượng:</strong> Gate G2 tiếp tục giữ ở <strong>HOLD</strong> để kiểm soát rủi ro ở các lớp chuyển tiếp khi triển khai cơ chế ECDR.</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Part C: Mô hình CNN tham chiếu đã được khóa cứng, định lượng MACs và đo đạc chi tiết</span>
    <span class="mono" style="color: var(--accent-amber);">Gate G2: HOLD (Quality Gate)</span>
  </div>
</section>
"""
