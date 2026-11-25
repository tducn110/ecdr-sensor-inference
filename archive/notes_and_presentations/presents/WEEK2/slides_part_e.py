# -*- coding: utf-8 -*-
"""
Part E: Planning, WBS & Weekly Management (Slides 12 - 13) - Visual-Rich, No Speech Scripts
"""

SLIDE_12_HTML = r"""
<section class="slide-frame" id="slide-12" data-slide-idx="12" data-part="Part E. Planning & Management">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part E. Planning &amp; Management &bull; Slide 12</span>
      <h2 class="slide-title">WBS Cập nhật &amp; Kế hoạch Triển khai Tuần 3</h2>
      <p class="slide-subtitle">Bản đồ phân rã công việc (Work Breakdown Structure), chuỗi phụ thuộc và điều kiện nghiệm thu khoa học</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">12 / 13</span>
      <span class="gate-badge gate-pass">Structured WBS</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual WBS Roadmap Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide12_wbs_roadmap.png')" style="margin-bottom: 0.4rem;">
      <img src="assets/slide12_wbs_roadmap.png" alt="WBS Progress Roadmap" style="max-height: 155px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">🔍 Bản đồ Phân rã Công việc (WBS) &amp; Trạng thái Tiến độ Từng Gói Công việc (Click phóng to)</div>
    </div>

    <!-- WBS Status Table -->
    <table class="custom-table" style="font-size: 0.72rem; margin-top: 0.2rem;">
      <thead>
        <tr>
          <th style="width: 10%;">WBS ID</th>
          <th style="width: 28%;">Gói Công việc (Work Package)</th>
          <th style="width: 22%;">Trạng thái tại mốc Week 2</th>
          <th style="width: 40%;">Sản phẩm bàn giao cụ thể (Deliverable)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="mono" style="color: #34d399;">WP1</span></td>
          <td><strong>Research brief &amp; frozen baseline</strong></td>
          <td><span class="gate-badge gate-pass">ĐÃ CÓ (W1)</span></td>
          <td>Week 1 Brief, mô hình CNN checkpoint, cấu hình huấn luyện.</td>
        </tr>
        <tr>
          <td><span class="mono" style="color: #34d399;">WP2</span></td>
          <td><strong>HAPT data foundations</strong></td>
          <td><span class="gate-badge gate-pass">ĐÃ KIỂM CHỨNG</span></td>
          <td>Data Report (13 trang), Data Audit JSON, Overlap Pairs CSV.</td>
        </tr>
        <tr>
          <td><span class="mono" style="color: #34d399;">WP3</span></td>
          <td><strong>CNN model foundations</strong></td>
          <td><span class="gate-badge gate-pass">ĐÃ KIỂM CHỨNG</span></td>
          <td>CNN Report (16 trang), Model Audit JSON, Confusion Matrix, MACs.</td>
        </tr>
        <tr style="background: rgba(99, 102, 241, 0.08);">
          <td><span class="mono" style="color: var(--accent-indigo); font-weight: 700;">WP4</span></td>
          <td><strong>Certified-region mathematics</strong></td>
          <td><span class="gate-badge gate-target">MỤC TIÊU TUẦN 3</span></td>
          <td>Định lý cận biên, giả định toán học, thuật toán lan truyền cận.</td>
        </tr>
        <tr style="background: rgba(99, 102, 241, 0.08);">
          <td><span class="mono" style="color: var(--accent-indigo); font-weight: 700;">WP5</span></td>
          <td><strong>Actual-graph verifier prototype</strong></td>
          <td><span class="gate-badge gate-target">MỤC TIÊU TUẦN 3</span></td>
          <td>Mã nguồn verifier, kiểm tra membership, bộ test Fail-closed.</td>
        </tr>
        <tr>
          <td><span class="mono" style="color: var(--text-dim);">WP6</span></td>
          <td><strong>Certified coverage &amp; efficiency</strong></td>
          <td><span class="gate-badge" style="background: rgba(255,255,255,0.05); color: var(--text-dim);">CÁC TUẦN SAU</span></td>
          <td>Ablation study, thực nghiệm stream hoàn chỉnh, đo đạc latency.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="slide-footer">
    <span>Kế hoạch phân công: Tham khảo <code>PLAN.md</code> và <code>output/week2_foundations_20261103/README.md</code></span>
    <span class="mono" style="color: var(--accent-indigo);">WP1-WP3 Closed &bull; WP4-WP5 Target</span>
  </div>
</section>
"""

SLIDE_13_HTML = r"""
<section class="slide-frame" id="slide-13" data-slide-idx="13" data-part="Part E. Planning & Management">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part E. Planning &amp; Management &bull; Slide 13</span>
      <h2 class="slide-title">Weekly Project Dashboard &amp; Quản trị Rủi ro</h2>
      <p class="slide-subtitle">Đáp ứng trọn vẹn rubric đánh giá hàng tuần của Tutor: WHAT, WHEN, WHY, WHO, Planned vs Actual, Risks &amp; Issues</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">13 / 13</span>
      <span class="gate-badge gate-pass">Closeout Ready</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Side-by-side: Planned vs Actual Hours Image + Tutor 5W Grid -->
    <div class="grid-2" style="align-items: center; margin-bottom: 0.3rem;">
      <!-- Chart Card -->
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/slide13_planned_vs_actual.png')">
          <img src="assets/slide13_planned_vs_actual.png" alt="Planned vs Actual Hours" style="max-height: 175px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 HOW MUCH: Đối soát Giờ Công Kế hoạch vs Thực tế (+4h Gap do đối soát 17.392 cặp &bull; Click phóng to)</div>
        </div>
      </div>

      <!-- Tutor 5W Summary -->
      <div class="grid-2" style="gap: 0.5rem;">
        <div class="card" style="padding: 0.5rem;">
          <div style="font-size: 0.7rem; font-weight: 700; color: var(--accent-cyan);">WHAT (Sản phẩm)</div>
          <div style="font-size: 0.76rem; color: #fff; font-weight: 600;">2 Báo cáo Độc lập</div>
          <div style="font-size: 0.7rem; color: var(--text-muted);">Data (13p) + CNN (16p) + Bộ kiểm toán SHA-256.</div>
        </div>

        <div class="card" style="padding: 0.5rem;">
          <div style="font-size: 0.7rem; font-weight: 700; color: var(--accent-indigo);">WHEN (Thời mốc)</div>
          <div style="font-size: 0.76rem; color: #fff; font-weight: 600;">03/11/2026</div>
          <div style="font-size: 0.7rem; color: var(--text-muted);">Hoàn tất kiểm toán kỹ thuật mốc Week 2.</div>
        </div>

        <div class="card" style="padding: 0.5rem;">
          <div style="font-size: 0.7rem; font-weight: 700; color: var(--accent-emerald);">WHY (Mục đích)</div>
          <div style="font-size: 0.76rem; color: #fff; font-weight: 600;">Xác lập Ground Truth</div>
          <div style="font-size: 0.7rem; color: var(--text-muted);">Tránh xây dựng chứng nhận trên dữ liệu rò rỉ.</div>
        </div>

        <div class="card" style="padding: 0.5rem;">
          <div style="font-size: 0.7rem; font-weight: 700; color: var(--accent-amber);">WHO (Nhân sự)</div>
          <div style="font-size: 0.76rem; color: #fff; font-weight: 600;">Chờ phê duyệt</div>
          <div style="font-size: 0.7rem; color: var(--text-muted);">Phân công Data, Model, Verifier, QA.</div>
        </div>
      </div>
    </div>

    <!-- Risk & Issue Management (2 Columns) -->
    <div class="grid-2" style="font-size: 0.74rem;">
      <div class="card card-accent-amber" style="padding: 0.5rem 0.75rem;">
        <h5 style="color: var(--accent-amber); font-size: 0.8rem; margin-bottom: 0.2rem;">⚠️ Top 3 Rủi ro Kỹ thuật &amp; Hành động:</h5>
        <ul style="padding-left: 1rem; color: #cbd5e1; line-height: 1.45;">
          <li><strong>Risk 1 (G2 HOLD):</strong> Lớp chuyển tiếp ít dữ liệu &rarr; <em>Duy trì Gate G2 HOLD, kiểm soát per-class recall.</em></li>
          <li><strong>Risk 2 (Annotation Origin):</strong> Dịch 1 mẫu đổi 98 nhãn &rarr; <em>Cố định quy ước tiền xử lý, kiểm tra upstream source.</em></li>
          <li><strong>Risk 3 (Float32 Backend):</strong> Sai số số học làm vỡ chứng nhận &rarr; <em>Thiết lập Interval Arithmetic và Fallback bảo thủ.</em></li>
        </ul>
      </div>

      <div class="card card-accent-rose" style="padding: 0.5rem 0.75rem;">
        <h5 style="color: var(--accent-rose); font-size: 0.8rem; margin-bottom: 0.2rem;">🔍 Key Issues &amp; Giải pháp Thực tế:</h5>
        <ul style="padding-left: 1rem; color: #cbd5e1; line-height: 1.45;">
          <li><strong>Issue 1 (Support Mismatch):</strong> Số lịch sử 13.866 vs 12.278 &rarr; <em>ĐÃ GIẢI MÃ: Do 1.588 Ambiguous windows bị gộp trong snapshot cũ!</em></li>
          <li><strong>Issue 2 (Duplicate Audit):</strong> Chưa có audit trùng lặp toàn cửa sổ &rarr; <em>Thêm bài test kiểm tra độc lập ở pha sau.</em></li>
          <li><strong>Issue 3 (Chưa có Speedup):</strong> Chưa đo đạc tiết kiệm ở W2 &rarr; <em>Đây là mốc nền tảng; chuyển thành KPI thực nghiệm Tuần 3+.</em></li>
        </ul>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Bài Thuyết trình Week 2 &bull; Tiếp theo: Phụ lục Chi tiết &amp; Q&amp;A Bằng chứng</span>
    <span class="mono" style="color: var(--accent-emerald);">Week 2 Delivery Complete</span>
  </div>
</section>
"""
