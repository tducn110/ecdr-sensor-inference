# -*- coding: utf-8 -*-
"""
Part E: Planning, WBS & Weekly Management (Slides 12 - 13)
Style: Pirsch Analytics (Sunlit Paper Notebook, Cream Paper, Hairline Borders)
"""

SLIDE_12_HTML = r"""
<section class="slide-frame" id="slide-12" data-slide-idx="12" data-part="Part E. Planning & Management">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">12</span>
        <span class="step-name-cell">WBS &bull; WEEK 3 ROADMAP</span>
      </div>
      <h2 class="slide-title">WBS Cập nhật &amp; Kế hoạch Triển khai Tuần 3</h2>
      <p class="slide-subtitle">Bản đồ phân rã công việc (Work Breakdown Structure), chuỗi phụ thuộc và điều kiện nghiệm thu khoa học</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">12 / 13</span>
      <span class="gate-badge gate-pass">&check; STRUCTURED WBS</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Visual WBS Roadmap Image -->
    <div class="slide-img-container" onclick="openModal('assets/slide12_wbs_roadmap.png')" style="margin-bottom: 0.3rem;">
      <img src="assets/slide12_wbs_roadmap.png" alt="WBS Progress Roadmap" style="max-height: 145px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Bản đồ Phân rã Công việc (WBS) &amp; Trạng thái Tiến độ Từng Gói Công việc (Click phóng to)</div>
    </div>

    <!-- WBS Status Table in Pirsch Table Style -->
    <table class="custom-table" style="font-size: 0.74rem; margin-top: 0.2rem;">
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
          <td><span class="mono" style="font-weight: 700;">WP1</span></td>
          <td><strong>Research brief &amp; frozen baseline</strong></td>
          <td><span class="gate-badge gate-pass">&check; ĐÃ CÓ (W1)</span></td>
          <td>Week 1 Brief, mô hình CNN checkpoint, cấu hình huấn luyện.</td>
        </tr>
        <tr>
          <td><span class="mono" style="font-weight: 700;">WP2</span></td>
          <td><strong>HAPT data foundations</strong></td>
          <td><span class="gate-badge gate-pass">&check; ĐÃ KIỂM CHỨNG</span></td>
          <td>Data Report (13 trang), Data Audit JSON, Overlap Pairs CSV.</td>
        </tr>
        <tr>
          <td><span class="mono" style="font-weight: 700;">WP3</span></td>
          <td><strong>CNN model foundations</strong></td>
          <td><span class="gate-badge gate-pass">&check; ĐÃ KIỂM CHỨNG</span></td>
          <td>CNN Report (16 trang), Model Audit JSON, Confusion Matrix, MACs.</td>
        </tr>
        <tr style="background: var(--color-sunshine-highlight);">
          <td><span class="mono" style="color: var(--color-ink); font-weight: 700;">WP4</span></td>
          <td><strong>Certified-region mathematics</strong></td>
          <td><span class="gate-badge gate-hold" style="border-color: var(--color-ink);">! MỤC TIÊU TUẦN 3</span></td>
          <td>Định lý cận biên, giả định toán học, thuật toán lan truyền cận.</td>
        </tr>
        <tr style="background: var(--color-sunshine-highlight);">
          <td><span class="mono" style="color: var(--color-ink); font-weight: 700;">WP5</span></td>
          <td><strong>Actual-graph verifier prototype</strong></td>
          <td><span class="gate-badge gate-hold" style="border-color: var(--color-ink);">! MỤC TIÊU TUẦN 3</span></td>
          <td>Mã nguồn verifier, kiểm tra membership, bộ test Fail-closed.</td>
        </tr>
        <tr>
          <td><span class="mono" style="color: var(--color-graphite);">WP6</span></td>
          <td><strong>Certified coverage &amp; efficiency</strong></td>
          <td><span class="gate-badge" style="background: #ffffff; color: var(--color-graphite); border-color: var(--color-border-muted);">CÁC TUẦN SAU</span></td>
          <td>Ablation study, thực nghiệm stream hoàn chỉnh, đo đạc latency.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="slide-footer">
    <span>Kế hoạch phân công: Tham khảo <code>PLAN.md</code> và <code>output/week2_foundations_20261103/delivery_manifest.json</code></span>
    <span class="gate-badge">WP1-WP3 Closed &bull; WP4-WP5 Target</span>
  </div>
</section>
"""

SLIDE_13_HTML = r"""
<section class="slide-frame" id="slide-13" data-slide-idx="13" data-part="Part E. Planning & Management">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">13</span>
        <span class="step-name-cell">PROJECT DASHBOARD &bull; SYNTHESIS</span>
      </div>
      <h2 class="slide-title">Weekly Project Dashboard &amp; Quản trị Rủi ro</h2>
      <p class="slide-subtitle">Đáp ứng trọn vẹn rubric đánh giá hàng tuần của Tutor: WHAT, WHEN, WHY, WHO, Planned vs Actual, Risks &amp; Issues</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">13 / 13</span>
      <span class="gate-badge gate-pass">&check; CLOSEOUT READY</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Side-by-side: Planned vs Actual Hours Image + Tutor 5W Grid -->
    <div class="grid-2" style="align-items: center; margin-bottom: 0.3rem;">
      <!-- Chart Card -->
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/slide13_planned_vs_actual.png')">
          <img src="assets/slide13_planned_vs_actual.png" alt="Planned vs Actual Hours" style="max-height: 165px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">HOW MUCH: Đối soát Giờ Công Kế hoạch vs Thực tế (+4h Gap do đối soát 17.392 cặp &bull; Click phóng to)</div>
        </div>
      </div>

      <!-- Tutor 5W Summary -->
      <div class="grid-2" style="gap: 0.4rem;">
        <div class="card" style="padding: 0.5rem;">
          <div class="stat-label">WHAT (SẢN PHẨM)</div>
          <div style="font-size: 0.85rem; color: var(--color-ink); font-weight: 500;">2 Báo cáo Độc lập</div>
          <div style="font-size: 0.72rem; color: var(--color-graphite); margin-top: 0.2rem;">Data (13p) + CNN (16p) + Bộ kiểm toán SHA-256.</div>
        </div>

        <div class="card" style="padding: 0.5rem;">
          <div class="stat-label">WHEN (THỜI MỐC)</div>
          <div style="font-size: 0.85rem; color: var(--color-ink); font-weight: 500;">03/11/2026</div>
          <div style="font-size: 0.72rem; color: var(--color-graphite); margin-top: 0.2rem;">Hoàn tất kiểm toán kỹ thuật mốc Week 2 đúng hạn.</div>
        </div>

        <div class="card" style="padding: 0.5rem; background: var(--color-sunshine-highlight);">
          <div class="stat-label" style="color: var(--color-ink);">WHY (MỤC ĐÍCH)</div>
          <div style="font-size: 0.85rem; color: var(--color-ink); font-weight: 500;">Xác lập Ground Truth</div>
          <div style="font-size: 0.72rem; color: var(--color-ink); margin-top: 0.2rem;">Tránh xây dựng chứng nhận trên dữ liệu rò rỉ.</div>
        </div>

        <div class="card" style="padding: 0.5rem; background: var(--color-mint-signal);">
          <div class="stat-label" style="color: var(--color-ink);">WHO (NHÂN SỰ)</div>
          <div style="font-size: 0.85rem; color: var(--color-ink); font-weight: 500;">Chờ phê duyệt</div>
          <div style="font-size: 0.72rem; color: var(--color-ink); margin-top: 0.2rem;">Phân công Data, Model, Verifier, QA.</div>
        </div>
      </div>
    </div>

    <!-- Risk & Issue Management (2 Columns) -->
    <div class="grid-2" style="font-size: 0.74rem;">
      <div class="card" style="padding: 0.5rem 0.75rem; border-left: 4px solid var(--color-sunshine-highlight);">
        <h5 style="color: var(--color-ink); font-size: 0.82rem; font-weight: 500; margin-bottom: 0.25rem;">⚠️ Top 3 Rủi ro Kỹ thuật &amp; Phương án Giảm thiểu:</h5>
        <ul style="padding-left: 1.1rem; color: var(--color-graphite); line-height: 1.5;">
          <li><strong>Risk 1 (G2 HOLD):</strong> Lớp chuyển tiếp ít dữ liệu &rarr; <em>Duy trì Gate G2 HOLD, kiểm soát per-class recall.</em></li>
          <li><strong>Risk 2 (Annotation Origin):</strong> Dịch 1 mẫu đổi 98 nhãn &rarr; <em>Cố định quy ước tiền xử lý, kiểm tra upstream source.</em></li>
          <li><strong>Risk 3 (Float32 Backend):</strong> Sai số số học làm vỡ chứng nhận &rarr; <em>Thiết lập Interval Arithmetic và Fallback bảo thủ.</em></li>
        </ul>
      </div>

      <div class="card" style="padding: 0.5rem 0.75rem; border-left: 4px solid var(--color-mint-signal);">
        <h5 style="color: var(--color-ink); font-size: 0.82rem; font-weight: 500; margin-bottom: 0.25rem;">🔍 Key Issues &amp; Giải pháp Thực tế:</h5>
        <ul style="padding-left: 1.1rem; color: var(--color-graphite); line-height: 1.5;">
          <li><strong>Issue 1 (Support Mismatch):</strong> Số lịch sử 13.866 vs 12.278 &rarr; <em>ĐÃ GIẢI MÃ: Do 1.588 Ambiguous windows bị gộp trong snapshot cũ!</em></li>
          <li><strong>Issue 2 (Duplicate Audit):</strong> Chưa có audit trùng lặp toàn cửa sổ &rarr; <em>Thêm bài test kiểm tra độc lập ở pha sau.</em></li>
          <li><strong>Issue 3 (Chưa có Speedup):</strong> Chưa đo đạc tiết kiệm ở W2 &rarr; <em>Đây là mốc nền tảng; chuyển thành KPI thực nghiệm Tuần 3+.</em></li>
        </ul>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Bài Thuyết trình Week 2 &bull; Tiếp theo: Phụ lục Chi tiết &amp; Q&amp;A Bằng chứng</span>
    <span class="gate-badge gate-pass">&check; Week 2 Delivery Complete</span>
  </div>
</section>
"""
