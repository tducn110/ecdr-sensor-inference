# -*- coding: utf-8 -*-
"""
Part A: Research Direction (Slides 01 - 03) - Visual-Rich, No Speech Scripts
"""

SLIDE_01_HTML = r"""
<section class="slide-frame" id="slide-01" data-slide-idx="1" data-part="Part A. Research Direction">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part A. Research Direction &bull; Slide 01</span>
      <h2 class="slide-title">Tổng quan Dự án &amp; Kế thừa Kết quả Tuần 1</h2>
      <p class="slide-subtitle">Thiết lập bài toán ECDR, hiện trạng reference model và phạm vi bàn giao của Week 2</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">01 / 13</span>
      <span class="gate-badge gate-pass">G1: PASS (Baseline Feasibility)</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual Architecture Banner Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide01_overview_pipeline.png')" style="margin-bottom: 0.5rem;">
      <img src="assets/slide01_overview_pipeline.png" alt="ECDR System Pipeline Overview" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">🔍 Sơ đồ Kiến trúc Tổng thể: Từ Dòng Cảm biến HAPT &rarr; Cửa sổ Trượt &rarr; SmallCNN1D &rarr; Động cơ Certified Decision Reuse (Click để phóng to)</div>
    </div>

    <!-- 3 Pillars Grid -->
    <div class="grid-3" style="margin-top: 0.3rem;">
      <div class="card card-accent-cyan">
        <div class="stat-value">6 &times; 128</div>
        <div class="stat-label">HAPT Sensor Windows</div>
        <div class="stat-desc">
          Chuỗi thời gian đa kênh: 3 trục gia tốc \((a_x, a_y, a_z)\) và 3 trục con quay hồi chuyển \((\omega_x, \omega_y, \omega_z)\) tại 50 Hz (2,56 s / window, stride 1,28 s).
        </div>
      </div>

      <div class="card card-accent-indigo">
        <div class="stat-value">SmallCNN1D</div>
        <div class="stat-label">Frozen Reference (K=12)</div>
        <div class="stat-desc">
          Mô hình chuẩn 1D-CNN phân loại 12 lớp hoạt động (6 cơ bản + 6 chuyển tiếp). Checkpoint Seed 43, Epoch 8 được cố định làm reference logic bất biến cho toàn bộ dự án.
        </div>
      </div>

      <div class="card card-accent-emerald">
        <div class="stat-value">ECDR Goal</div>
        <div class="stat-label">Certified Decision Reuse</div>
        <div class="stat-desc">
          Tái sử dụng quyết định suy luận có chứng nhận toán học trên chuỗi cửa sổ tuần tự nhằm giảm chi phí tính toán mà vẫn bảo toàn 100% quyết định của CNN gốc.
        </div>
      </div>
    </div>

    <!-- 2 Sub-blocks: Baseline numbers vs Week 2 Handover -->
    <div class="grid-2" style="margin-top: 0.3rem;">
      <div class="card">
        <h4 style="color: var(--accent-indigo); margin-bottom: 0.3rem; font-size: 0.88rem;">📊 Kết quả Kế thừa từ Week 1 (Baseline Feasibility)</h4>
        <ul style="font-size: 0.78rem; color: #cbd5e1; line-height: 1.5; padding-left: 1.2rem;">
          <li><strong>Mô hình:</strong> SmallCNN1D với <strong>283.692</strong> tham số trainable; <strong>1.172.992 MACs</strong> / window.</li>
          <li><strong>Hiệu năng baseline:</strong> Test Accuracy đạt <strong>89,05%</strong> (3.181/3.572), Macro-F1 đạt <strong>0,7763</strong>.</li>
          <li><strong>Ranh giới nhận thức:</strong> Đây là bằng chứng khả thi của mô hình phân loại (Baseline Feasibility), <em>không phải bằng chứng ECDR đã tối ưu hóa inference thành công</em>.</li>
        </ul>
      </div>

      <div class="card">
        <h4 style="color: var(--accent-cyan); margin-bottom: 0.3rem; font-size: 0.88rem;">📦 Bàn giao Độc lập của Week 2 (Foundations Deliverables)</h4>
        <ul style="font-size: 0.78rem; color: #cbd5e1; line-height: 1.5; padding-left: 1.2rem;">
          <li><strong>Hai báo cáo chuyên sâu độc lập:</strong>
            <ul style="margin-top: 0.15rem; padding-left: 0.8rem;">
              <li><code>Report A (13 trang):</code> HAPT Data Foundations (tín hiệu, windowing, split, chuẩn hóa, provenance).</li>
              <li><code>Report B (16 trang):</code> SmallCNN1D Model Foundations (toán tử, receptive field, resource accounting, margins).</li>
            </ul>
          </li>
          <li><strong>Bộ kiểm toán độc lập:</strong> <code>week2_data_audit.py</code> và <code>week2_model_audit.py</code> tái lập 100% bằng chứng SHA-256.</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Nguồn xác minh: <code>README.md</code>, <code>output/week2_foundations_20261103/delivery_manifest.json</code></span>
    <span class="mono" style="color: var(--accent-cyan);">ECDR Foundation Deck</span>
  </div>
</section>
"""

SLIDE_02_HTML = r"""
<section class="slide-frame" id="slide-02" data-slide-idx="2" data-part="Part A. Research Direction">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part A. Research Direction &bull; Slide 02</span>
      <h2 class="slide-title">Làm rõ Bài toán Nghiên cứu (Refined Problem Statement)</h2>
      <p class="slide-subtitle">Chuyển đổi từ ý tưởng tiết kiệm tính toán trực giác sang bài toán khoa học có thể kiểm chứng toán học</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">02 / 13</span>
      <span class="gate-badge gate-target">Problem Refinement</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual Concept Comparison Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide02_problem_statement.png')" style="margin-bottom: 0.4rem;">
      <img src="assets/slide02_problem_statement.png" alt="Classification Correctness vs Decision Preservation" style="max-height: 150px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">🔍 Phân biệt Hai Khái niệm Tính Đúng đắn: Classification Correctness vs Decision Preservation (Click để phóng to)</div>
    </div>

    <!-- Refined Problem Statement Box -->
    <div class="problem-box" style="margin: 0.3rem 0; padding: 0.75rem 1.25rem; font-size: 0.95rem;">
      <span style="color: var(--accent-cyan); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.2rem;">Câu hỏi Nghiên cứu Trọng tâm của Đề tài (Refined Problem Statement)</span>
      &ldquo;Liệu phương pháp Event-Conditioned Certified Decision Reuse có thể bảo toàn chính xác các quyết định của một frozen CNN trên chuỗi cửa sổ cảm biến HAPT tuần tự, đồng thời giảm tổng chi phí suy luận end-to-end hay không?&rdquo;
    </div>

    <!-- The Triad Logic Flow -->
    <div class="grid-3" style="margin-top: 0.3rem;">
      <div class="card card-accent-cyan" style="padding: 0.6rem;">
        <div style="font-size: 0.78rem; font-weight: 700; color: var(--accent-cyan);">1. Quan sát (Observation)</div>
        <p style="font-size: 0.74rem; color: #cbd5e1; margin-top: 0.2rem;">
          Các cửa sổ cảm biến trượt chồng lấn 50% thời gian (\(X_t\) và \(X_{t+1}\) chia sẻ 64 mẫu tín hiệu giống hệt nhau).
        </p>
      </div>

      <div class="card card-accent-amber" style="padding: 0.6rem;">
        <div style="font-size: 0.78rem; font-weight: 700; color: var(--accent-amber);">2. Vấn đề (Problem)</div>
        <p style="font-size: 0.74rem; color: #cbd5e1; margin-top: 0.2rem;">
          Chồng lấn một phần dữ liệu <strong>không bảo đảm</strong> CNN sẽ trả về cùng quyết định nhãn hoạt động: \(h(X_t) \neq h(X_{t+1})\).
        </p>
      </div>

      <div class="card card-accent-emerald" style="padding: 0.6rem;">
        <div style="font-size: 0.78rem; font-weight: 700; color: var(--accent-emerald);">3. Mục tiêu (Research Goal)</div>
        <p style="font-size: 0.74rem; color: #cbd5e1; margin-top: 0.2rem;">
          Chỉ tái sử dụng quyết định khi có chứng nhận toán học bảo toàn 100% quyết định so với mô hình frozen CNN gốc.
        </p>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Nguồn xác minh: <code>01_RQ_GAP.md</code>, <code>PROJECT_BRIEF.md</code></span>
    <span class="mono" style="color: var(--accent-cyan);">Zero-Divergence Target</span>
  </div>
</section>
"""

SLIDE_03_HTML = r"""
<section class="slide-frame" id="slide-03" data-slide-idx="3" data-part="Part A. Research Direction">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part A. Research Direction &bull; Slide 03</span>
      <h2 class="slide-title">Ba Câu hỏi Nghiên cứu Cốt lõi (Three Research Questions)</h2>
      <p class="slide-subtitle">Cấu trúc hóa 3 RQs cấp độ dự án theo chuẩn đánh giá của Tutor từ chất lượng reference đến hiệu quả thực thi</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">03 / 13</span>
      <span class="gate-badge gate-pass">Rubric Aligned</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual RQ Roadmap Flowchart Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide03_rq_roadmap.png')" style="margin-bottom: 0.4rem;">
      <img src="assets/slide03_rq_roadmap.png" alt="Three Research Questions Roadmap" style="max-height: 135px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">🔍 Sơ đồ Chuỗi Nhân quả 3 Research Questions: RQ1 (Chất lượng) &rarr; RQ2 (Chứng nhận) &rarr; RQ3 (Hiệu quả chi phí) (Click phóng to)</div>
    </div>

    <!-- Table of 3 RQs -->
    <table class="custom-table" style="font-size: 0.76rem; margin-top: 0.2rem;">
      <thead>
        <tr>
          <th style="width: 10%;">ID</th>
          <th style="width: 25%;">Trọng tâm Nghiên cứu</th>
          <th style="width: 40%;">Câu hỏi Nghiên cứu Chi tiết (Research Question)</th>
          <th style="width: 25%;">Tiêu chí Đánh giá &amp; Đo lường</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="mono" style="color: var(--accent-cyan); font-weight: 700;">RQ1</span></td>
          <td><strong>Reference Quality</strong><br><span style="font-size: 0.7rem; color: var(--text-dim);">Chất lượng mô hình cơ sở</span></td>
          <td>How reliably does the frozen SmallCNN1D classify basic activities and postural transitions on HAPT?</td>
          <td>Test Accuracy (89,05%), Macro-F1 (0,7763), Classwise Recall, Confusion Matrix, Per-class Support.</td>
        </tr>
        <tr>
          <td><span class="mono" style="color: var(--accent-indigo); font-weight: 700;">RQ2</span></td>
          <td><strong>Certified Preservation</strong><br><span style="font-size: 0.7rem; color: var(--text-dim);">Điều kiện toán học bảo toàn</span></td>
          <td>Under what certified conditions can ECDR reuse a previous decision without changing the frozen CNN's prediction?</td>
          <td>Whole-region margins \(\inf_{X\in S}[F_c-F_j]>0\), Membership check, Reference identity, Numerical soundness.</td>
        </tr>
        <tr>
          <td><span class="mono" style="color: var(--accent-emerald); font-weight: 700;">RQ3</span></td>
          <td><strong>Inference Efficiency</strong><br><span style="font-size: 0.7rem; color: var(--text-dim);">Hiệu quả chi phí thực thi</span></td>
          <td>When does certified reuse reduce total inference cost compared with running the CNN on every window?</td>
          <td>Certified reuse rate, Fallback frequency, Verifier build cost, End-to-end execution latency.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Part A: Định vị đề tài là nghiên cứu tính khả thi của tái sử dụng quyết định có chứng nhận toán học</span>
    <span class="mono" style="color: var(--accent-cyan);">Part A Complete</span>
  </div>
</section>
"""
