# -*- coding: utf-8 -*-
"""
Part A: Research Direction (Slides 01 - 03)
Style: Pirsch Analytics (Sunlit Paper Notebook, Cream Paper, Hairline Borders)
"""

SLIDE_01_HTML = r"""
<section class="slide-frame" id="slide-01" data-slide-idx="1" data-part="Part A. Research Direction">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">01</span>
        <span class="step-name-cell">OVERVIEW &bull; WEEK 1 RECAP</span>
      </div>
      <h2 class="slide-title">Tổng quan Dự án &amp; Kế thừa Tuần 1</h2>
      <p class="slide-subtitle">Thiết lập bài toán ECDR, hiện trạng reference model và phạm vi bàn giao của Week 2</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">01 / 13</span>
      <span class="gate-badge gate-pass">&check; G1: BASELINE PASS</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual Architecture Banner Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide01_overview_pipeline.png')">
      <img src="assets/slide01_overview_pipeline.png" alt="ECDR System Pipeline Overview" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Sơ đồ Kiến trúc Tổng thể: Dòng Cảm biến HAPT &rarr; Cửa sổ Trượt &rarr; SmallCNN1D &rarr; Động cơ Certified Decision Reuse</div>
    </div>

    <!-- 3 Pillars Grid -->
    <div class="grid-3" style="margin-top: 0.5rem;">
      <div class="card">
        <div class="stat-value">6 &times; 128</div>
        <div class="stat-label">HAPT SENSOR WINDOWS</div>
        <div class="stat-desc">
          Chuỗi thời gian đa kênh: 3 trục gia tốc \((a_x, a_y, a_z)\) và 3 trục con quay hồi chuyển \((\omega_x, \omega_y, \omega_z)\) tại 50 Hz (2,56 s / window, stride 1,28 s).
        </div>
      </div>

      <div class="card">
        <div class="stat-value">SmallCNN1D</div>
        <div class="stat-label">FROZEN REFERENCE (K=12)</div>
        <div class="stat-desc">
          Mô hình chuẩn 1D-CNN phân loại 12 lớp hoạt động (6 cơ bản + 6 chuyển tiếp). Checkpoint Seed 43, Epoch 8 được cố định làm reference logic bất biến cho toàn bộ dự án.
        </div>
      </div>

      <div class="card" style="background: var(--color-sunshine-highlight);">
        <div class="stat-value">ECDR Goal</div>
        <div class="stat-label" style="color: var(--color-ink);">CERTIFIED DECISION REUSE</div>
        <div class="stat-desc" style="color: var(--color-ink);">
          Tái sử dụng quyết định suy luận có chứng nhận toán học trên chuỗi cửa sổ tuần tự nhằm giảm chi phí tính toán mà vẫn bảo toàn 100% quyết định của CNN gốc.
        </div>
      </div>
    </div>

    <!-- 2 Sub-blocks: Baseline numbers vs Week 2 Handover -->
    <div class="grid-2" style="margin-top: 0.5rem;">
      <div class="card">
        <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">📊 Kết quả Kế thừa từ Week 1 (Baseline Feasibility)</h4>
        <ul style="font-size: 0.82rem; color: var(--color-graphite); line-height: 1.6; padding-left: 1.2rem;">
          <li><strong>Mô hình:</strong> SmallCNN1D với <strong>283.692</strong> tham số trainable; <strong>1.172.992 MACs</strong> / window.</li>
          <li><strong>Hiệu năng baseline:</strong> Test Accuracy đạt <strong>89,05%</strong> (3.181/3.572), Macro-F1 đạt <strong>0,7763</strong>.</li>
          <li><strong>Ranh giới nhận thức:</strong> Đây là bằng chứng khả thi của mô hình phân loại (Baseline Feasibility), <em>không phải bằng chứng ECDR đã tối ưu hóa inference thành công</em>.</li>
        </ul>
      </div>

      <div class="card">
        <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">📦 Bàn giao Độc lập của Week 2 (Foundations Deliverables)</h4>
        <ul style="font-size: 0.82rem; color: var(--color-graphite); line-height: 1.6; padding-left: 1.2rem;">
          <li><strong>Hai báo cáo chuyên sâu độc lập:</strong>
            <ul style="margin-top: 0.2rem; padding-left: 0.8rem;">
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
    <span class="gate-badge">ECDR Foundation Deck</span>
  </div>
</section>
"""

SLIDE_02_HTML = r"""
<section class="slide-frame" id="slide-02" data-slide-idx="2" data-part="Part A. Research Direction">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">02</span>
        <span class="step-name-cell">PROBLEM STATEMENT</span>
      </div>
      <h2 class="slide-title">Làm rõ Bài toán Nghiên cứu (Problem Refinement)</h2>
      <p class="slide-subtitle">Chuyển đổi từ ý tưởng tiết kiệm tính toán trực giác sang bài toán khoa học có thể kiểm chứng toán học</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">02 / 13</span>
      <span class="gate-badge gate-hold">! REFINED OBJECTIVE</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual Concept Comparison Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide02_problem_statement.png')">
      <img src="assets/slide02_problem_statement.png" alt="Classification Correctness vs Decision Preservation" style="max-height: 155px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Phân biệt Hai Khái niệm Tính Đúng đắn: Classification Correctness vs Decision Preservation</div>
    </div>

    <!-- Refined Problem Statement Box in Sunshine Yellow -->
    <div class="callout-sunshine" style="text-align: center;">
      <span style="font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.286em; font-weight: 500; display: block; margin-bottom: 0.25rem;">
        CÂU HỎI NGHIÊN CỨU TRỌNG TÂM CỦA ĐỀ TÀI (REFINED PROBLEM STATEMENT)
      </span>
      <strong style="font-size: 1.05rem; font-weight: 500;">
        &ldquo;Liệu phương pháp Event-Conditioned Certified Decision Reuse có thể bảo toàn chính xác các quyết định của một frozen CNN trên chuỗi cửa sổ cảm biến HAPT tuần tự, đồng thời giảm tổng chi phí suy luận end-to-end hay không?&rdquo;
      </strong>
    </div>

    <!-- The Triad Logic Flow -->
    <div class="grid-3" style="margin-top: 0.5rem;">
      <div class="card">
        <div style="font-size: 0.85rem; font-weight: 500; color: var(--color-ink);">1. QUAN SÁT (OBSERVATION)</div>
        <p style="font-size: 0.8rem; color: var(--color-graphite); margin-top: 0.3rem;">
          Các cửa sổ cảm biến trượt chồng lấn 50% thời gian (\(X_t\) và \(X_{t+1}\) chia sẻ 64 mẫu tín hiệu giống hệt nhau).
        </p>
      </div>

      <div class="card" style="background: rgba(254, 226, 226, 0.5);">
        <div style="font-size: 0.85rem; font-weight: 500; color: #b91c1c;">2. VẤN ĐỀ (PROBLEM)</div>
        <p style="font-size: 0.8rem; color: var(--color-graphite); margin-top: 0.3rem;">
          Chồng lấn một phần dữ liệu <strong>không bảo đảm</strong> CNN sẽ trả về cùng quyết định nhãn hoạt động: \(h(X_t) \neq h(X_{t+1})\).
        </p>
      </div>

      <div class="card" style="background: var(--color-mint-signal);">
        <div style="font-size: 0.85rem; font-weight: 500; color: var(--color-ink);">3. MỤC TIÊU (GOAL)</div>
        <p style="font-size: 0.8rem; color: var(--color-ink); margin-top: 0.3rem;">
          Chỉ tái sử dụng quyết định khi có chứng nhận toán học bảo toàn 100% quyết định so với mô hình frozen CNN gốc.
        </p>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Nguồn xác minh: <code>01_RQ_GAP.md</code>, <code>PROJECT_BRIEF.md</code></span>
    <span class="gate-badge">Zero-Divergence Target</span>
  </div>
</section>
"""

SLIDE_03_HTML = r"""
<section class="slide-frame" id="slide-03" data-slide-idx="3" data-part="Part A. Research Direction">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">03</span>
        <span class="step-name-cell">RESEARCH QUESTIONS</span>
      </div>
      <h2 class="slide-title">Ba Câu hỏi Nghiên cứu Cốt lõi (Three Research Questions)</h2>
      <p class="slide-subtitle">Cấu trúc hóa 3 RQs cấp độ dự án theo chuẩn đánh giá của Tutor từ chất lượng reference đến hiệu quả thực thi</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">03 / 13</span>
      <span class="gate-badge gate-pass">&check; RUBRIC ALIGNED</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual RQ Roadmap Flowchart Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide03_rq_roadmap.png')">
      <img src="assets/slide03_rq_roadmap.png" alt="Three Research Questions Roadmap" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Sơ đồ Chuỗi Nhân quả 3 Research Questions: RQ1 (Chất lượng) &rarr; RQ2 (Chứng nhận) &rarr; RQ3 (Hiệu quả chi phí)</div>
    </div>

    <!-- Table of 3 RQs in Pirsch style -->
    <table class="custom-table" style="margin-top: 0.4rem;">
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
          <td><span class="mono" style="font-weight: 700;">RQ1</span></td>
          <td><strong>Reference Quality</strong><br><span style="font-size: 0.72rem; color: var(--color-graphite);">Chất lượng mô hình cơ sở</span></td>
          <td>How reliably does the frozen SmallCNN1D classify basic activities and postural transitions on HAPT?</td>
          <td>Test Accuracy (89,05%), Macro-F1 (0,7763), Classwise Recall, Confusion Matrix, Support.</td>
        </tr>
        <tr>
          <td><span class="mono" style="font-weight: 700; background: var(--color-sunshine-highlight);">RQ2</span></td>
          <td><strong>Certified Preservation</strong><br><span style="font-size: 0.72rem; color: var(--color-graphite);">Điều kiện toán học bảo toàn</span></td>
          <td>Under what certified conditions can ECDR reuse a previous decision without changing the frozen CNN's prediction?</td>
          <td>Whole-region margins \(\inf_{X\in S}[F_c-F_j]>0\), Membership check, Reference identity, Soundness.</td>
        </tr>
        <tr>
          <td><span class="mono" style="font-weight: 700; background: var(--color-mint-signal);">RQ3</span></td>
          <td><strong>Inference Efficiency</strong><br><span style="font-size: 0.72rem; color: var(--color-graphite);">Hiệu quả chi phí thực thi</span></td>
          <td>When does certified reuse reduce total inference cost compared with running the CNN on every window?</td>
          <td>Certified reuse rate, Fallback frequency, Verifier build cost, End-to-end execution latency.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Part A: Định vị đề tài là nghiên cứu tính khả thi của tái sử dụng quyết định có chứng nhận toán học</span>
    <span class="gate-badge">Part A Complete</span>
  </div>
</section>
"""
