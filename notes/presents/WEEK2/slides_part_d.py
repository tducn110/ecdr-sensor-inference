# -*- coding: utf-8 -*-
"""
Part D: From Data & CNN to Certified Reuse (Slides 10 - 11) - Visual-Rich, No Speech Scripts
"""

SLIDE_10_HTML = r"""
<section class="slide-frame" id="slide-10" data-slide-idx="10" data-part="Part D. Certified Reuse Concept">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part D. Certified Reuse Concept &bull; Slide 10</span>
      <h2 class="slide-title">Chồng lấn Thời gian &amp; Độ Ổn định Quyết định Thực nghiệm</h2>
      <p class="slide-subtitle">Tại sao sự trùng lấn 50% dữ liệu tạo ra cơ hội tái sử dụng, nhưng không đủ để bảo đảm an toàn?</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">10 / 13</span>
      <span class="gate-badge gate-pass">17.392 Pairs Verified</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Math formula box of overlap -->
    <div class="formula-box" style="padding: 0.5rem 1rem; margin: 0.2rem 0; font-size: 0.85rem;">
      Hai cửa sổ liên tiếp \(X_t, X_{t+1} \in \mathbb{R}^{6\times128}\) có quan hệ căn chỉnh thời gian:
      <strong style="color: var(--accent-cyan);">\(X_t[:, 64:128] \equiv X_{t+1}[:, 0:64]\)</strong> (Đã kiểm toán 17.392 cặp, \(\text{MAE} = 0,00000\)).
      <br><span style="color: var(--accent-rose); font-size: 0.78rem;">Nhưng sự trùng lấn đầu vào KHÔNG bảo đảm quyết định mô hình bất biến: \(h(X_t) \stackrel{?}{=} h(X_{t+1})\)</span>
    </div>

    <!-- Side-by-side: Stability Bar Chart + Top-Two Margins Plot -->
    <div class="grid-2" style="align-items: center; margin-top: 0.2rem;">
      <!-- Chart 1: Decision Stability -->
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/slide10_decision_stability.png')">
          <img src="assets/slide10_decision_stability.png" alt="Decision Stability Comparison" style="max-height: 180px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Độ Ổn định Quyết định: Cặp Ổn định (97,82%) vs Cặp Chuyển tiếp (52,27% &bull; Click phóng to)</div>
        </div>
      </div>

      <!-- Chart 2: Top-Two Margins Distribution -->
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/model_top_two_margins.png')">
          <img src="assets/model_top_two_margins.png" alt="Top-Two Logit Margins" style="max-height: 180px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Phân phối Top-Two Logit Margins (5.053 windows, min=0,00011 &bull; Click phóng to)</div>
        </div>
      </div>
    </div>

    <!-- Nuance Warning -->
    <div class="callout callout-warning" style="margin-top: 0.3rem; padding: 0.4rem 0.8rem; font-size: 0.76rem;">
      <strong>Ý nghĩa Khoa học Cốt tử:</strong> Độ ổn định quyết định thực nghiệm (Empirical Stability) là <em>động lực thôi thúc nghiên cứu bộ chọn lọc theo sự kiện (Event-Conditioned Selection)</em>, chứ tuyệt đối không phải là chứng nhận an toàn để tái sử dụng!
    </div>
  </div>

  <div class="slide-footer">
    <span>Xác thực căn chỉnh: <code>data_aligned_overlap_pairs.csv</code> (17.392 pairs audited)</span>
    <span class="mono" style="color: var(--accent-amber);">Heuristic Caching Banned</span>
  </div>
</section>
"""

SLIDE_11_HTML = r"""
<section class="slide-frame" id="slide-11" data-slide-idx="11" data-part="Part D. Certified Reuse Concept">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part D. Certified Reuse Concept &bull; Slide 11</span>
      <h2 class="slide-title">Cơ chế Khái niệm của Certified Decision Reuse</h2>
      <p class="slide-subtitle">Thiết lập khế ước chứng nhận: Anchor, Candidate Region, Điều kiện Cận dưới 11-Margin và Fallback an toàn</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">11 / 13</span>
      <span class="gate-badge gate-target">Certification Contract</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Python-rendered Central Certification Formula Banner Image -->
    <div class="slide-img-container" onclick="openModal('assets/math_certification_condition.png')" style="margin-bottom: 0.4rem;">
      <img src="assets/math_certification_condition.png" alt="Central Region Invariance Certification Condition" style="max-height: 60px; object-fit: contain; margin: 0 auto;">
    </div>

    <!-- Python-rendered 5-Stage Certification Workflow Diagram -->
    <div class="slide-img-container" onclick="openModal('assets/slide11_certification_workflow.png')" style="margin-bottom: 0.4rem;">
      <img src="assets/slide11_certification_workflow.png" alt="5-Stage Certification Lifecycle" style="max-height: 145px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">🔍 Quy trình Vận hành Khế ước Chứng nhận: Từ Anchor \(X^r\) &rarr; Miền Ứng viên \(S\) &rarr; 11 Biên Logit &rarr; Kiểm tra &rarr; Reuse / Fallback (Click phóng to)</div>
    </div>

    <!-- Boundary Distinction: Week 2 vs Week 3 Targets -->
    <div class="grid-2" style="margin-top: 0.2rem;">
      <div class="card card-accent-cyan" style="padding: 0.6rem;">
        <h4 style="color: var(--accent-cyan); font-size: 0.85rem; margin-bottom: 0.2rem;">
          📦 Bàn giao của Tuần 2 (Week 2 Boundary)
        </h4>
        <ul style="font-size: 0.74rem; color: #cbd5e1; line-height: 1.5; padding-left: 1.1rem;">
          <li>Định nghĩa cấu trúc hình học của Trusted Anchor, Candidate Region \(S\) và khế ước kiểm tra.</li>
          <li>Xác định 11 ràng buộc biên (11-margin condition) cần phải thỏa mãn trên toàn miền.</li>
          <li>Đo lường phân phối khoảng cách logit điểm (Pointwise Margin): Min = \(0,00011\), Median = \(5,978\).</li>
        </ul>
      </div>

      <div class="card card-accent-emerald" style="padding: 0.6rem;">
        <h4 style="color: var(--accent-emerald); font-size: 0.85rem; margin-bottom: 0.2rem;">
          🎯 Đầu việc Cam kết cho Tuần 3 (Week 3 Objectives)
        </h4>
        <ul style="font-size: 0.74rem; color: #cbd5e1; line-height: 1.5; padding-left: 1.1rem;">
          <li>Nghiên cứu thuật toán lan truyền cận (Bound Propagation) trên <strong>computation graph K=12 thực tế</strong>.</li>
          <li>Kiểm soát tính đúng đắn số học của số thực (Float32 Numerical Soundness, Interval Arithmetic).</li>
          <li>Xây dựng verifier nguyên mẫu và cơ chế Fail-Closed an toàn tuyệt đối.</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Part D: Thiết lập công thức chứng nhận trung tâm và chuyển giao đầu việc sang Week 3</span>
    <span class="mono" style="color: var(--accent-emerald);">All-11 Margins Contract</span>
  </div>
</section>
"""
