# -*- coding: utf-8 -*-
"""
Part B: Data Foundations (Slides 04 - 07) - Visual-Rich, No Speech Scripts
"""

SLIDE_04_HTML = r"""
<section class="slide-frame" id="slide-04" data-slide-idx="4" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part B. Data Foundations &bull; Slide 04</span>
      <h2 class="slide-title">HAPT Dataset: Nguồn gốc, Bản chất Vật lý &amp; Ý nghĩa</h2>
      <p class="slide-subtitle">Dữ liệu cảm biến đến từ đâu, mô tả hiện tượng vật lý gì và tại sao lại phù hợp với ECDR?</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">04 / 13</span>
      <span class="gate-badge gate-pass">Physical Grounding</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Top 4 Metrics Cards -->
    <div class="grid-4" style="margin-bottom: 0.3rem;">
      <div class="card card-accent-cyan" style="text-align: center; padding: 0.6rem;">
        <div class="stat-value" style="font-size: 1.5rem;">30</div>
        <div class="stat-label">Participants</div>
        <div class="stat-desc" style="font-size: 0.72rem;">Người tham gia (19–48 tuổi) gắn điện thoại Samsung Galaxy S II ở thắt lưng.</div>
      </div>

      <div class="card card-accent-indigo" style="text-align: center; padding: 0.6rem;">
        <div class="stat-value" style="font-size: 1.5rem;">61</div>
        <div class="stat-label">Recordings</div>
        <div class="stat-desc" style="font-size: 0.72rem;">122 file cảm biến thô (mỗi bản ghi gồm 1 file gia tốc + 1 file con quay hồi chuyển).</div>
      </div>

      <div class="card card-accent-emerald" style="text-align: center; padding: 0.6rem;">
        <div class="stat-value" style="font-size: 1.5rem;">50 Hz</div>
        <div class="stat-label">Sampling Rate</div>
        <div class="stat-desc" style="font-size: 0.72rem;">50 mẫu/giây (\(\Delta t = 20\text{ ms}\)), thu nhận chuyển động động học liên tục.</div>
      </div>

      <div class="card card-accent-amber" style="text-align: center; padding: 0.6rem;">
        <div class="stat-value" style="font-size: 1.5rem;">12 Classes</div>
        <div class="stat-label">Activity Taxonomy</div>
        <div class="stat-desc" style="font-size: 0.72rem;">6 hoạt động cơ bản (Basic) + 6 chuyển tiếp tư thế (Postural Transitions).</div>
      </div>
    </div>

    <!-- Side-by-side: Raw Waveforms Image + Physical Meaning -->
    <div class="grid-2" style="align-items: center; margin-top: 0.2rem;">
      <div class="card" style="padding: 0.5rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/data_raw_first_recording.png')">
          <img src="assets/data_raw_first_recording.png" alt="Raw 50Hz Sensor Waveforms" style="max-height: 190px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Dạng sóng Cảm biến Thô 50 Hz của Bản ghi Đầu tiên: 3 trục Gia tốc (g) và 3 trục Con quay hồi chuyển (rad/s) (Click phóng to)</div>
        </div>
      </div>

      <div class="card" style="padding: 0.75rem;">
        <h4 style="color: var(--accent-cyan); font-size: 0.88rem; margin-bottom: 0.3rem;">
          📡 Phân biệt Kênh Cảm biến vs 12 Lớp Hoạt động
        </h4>
        <ul style="font-size: 0.78rem; color: #cbd5e1; line-height: 1.55; padding-left: 1.1rem;">
          <li><strong>6 Kênh Cảm biến Vật lý:</strong>
            <ul style="padding-left: 0.8rem; margin-top: 0.1rem;">
              <li><em>Accelerometer (\(a_x, a_y, a_z\)):</em> Gia tốc trọng trường + vận động (đơn vị \(g\)).</li>
              <li><em>Gyroscope (\(\omega_x, \omega_y, \omega_z\)):</em> Vận tốc góc xoay quanh 3 trục (đơn vị \(\text{rad/s}\)).</li>
            </ul>
          </li>
          <li style="margin-top: 0.25rem;"><strong>12 Nhãn Hoạt động (Activity Classes):</strong>
            <ul style="padding-left: 0.8rem; margin-top: 0.1rem;">
              <li><em>1–6 (Cơ bản):</em> WALKING, UPSTAIRS, DOWNSTAIRS, SITTING, STANDING, LAYING.</li>
              <li><em>7–12 (Chuyển tiếp):</em> STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, LIE_TO_STAND.</li>
            </ul>
          </li>
        </ul>
        <div class="callout callout-warning" style="margin-top: 0.4rem; padding: 0.4rem 0.6rem; font-size: 0.74rem;">
          <strong>Lưu ý:</strong> Tuyệt đối không nhầm lẫn <strong>6 kênh cảm biến vật lý</strong> với <strong>12 lớp hoạt động</strong> hoặc mô hình K=6 lịch sử!
        </div>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Nguồn dữ liệu: Reyes-Ortiz et al. (2016), <code>01_DATA/03_Week_2_Data_Foundations.tex</code></span>
    <span class="mono" style="color: var(--accent-cyan);">6 Channels &ne; 12 Classes</span>
  </div>
</section>
"""

SLIDE_05_HTML = r"""
<section class="slide-frame" id="slide-05" data-slide-idx="5" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part B. Data Foundations &bull; Slide 05</span>
      <h2 class="slide-title">Từ Tín hiệu Thô &rarr; Cửa sổ &rarr; Chính sách Gán nhãn</h2>
      <p class="slide-subtitle">Quy trình cắt cửa sổ trượt 50% chồng lấn và phân bổ 17.453 windows theo chính sách nhãn nghiêm ngặt</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">05 / 13</span>
      <span class="gate-badge gate-pass">Exact Windowing</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Side-by-side: Policy Support Audit Chart + Table -->
    <div class="grid-2" style="align-items: center;">
      <!-- Image Card -->
      <div class="card" style="padding: 0.5rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/data_policy_support.png')">
          <img src="assets/data_policy_support.png" alt="Data Policy Support Distribution" style="max-height: 220px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Phân bổ Số lượng Cửa sổ theo Chính sách Gán nhãn (17.453 Windows &bull; Click phóng to)</div>
        </div>
      </div>

      <!-- Policy Breakdown Table -->
      <div class="card" style="padding: 0.6rem;">
        <h4 style="color: var(--accent-indigo); font-size: 0.88rem; margin-bottom: 0.3rem;">
          📊 Thống kê Phân bổ 17.453 Cửa sổ Cảm biến
        </h4>
        <table class="custom-table" style="font-size: 0.74rem;">
          <thead>
            <tr>
              <th>Nhóm Chính sách</th>
              <th>Số lượng</th>
              <th>Tỷ lệ</th>
              <th>Tiêu chí chấp nhận</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong style="color: #34d399;">Pure</strong></td>
              <td><span class="mono">10.361</span></td>
              <td>59,37%</td>
              <td>100% mẫu thuộc duy nhất 1 lớp hoạt động.</td>
            </tr>
            <tr>
              <td><strong style="color: #38bdf8;">Majority</strong></td>
              <td><span class="mono">1.917</span></td>
              <td>10,98%</td>
              <td>Lớp chiếm ưu thế bao phủ \(\ge 60\%\) (\(\ge 77/128\) mẫu).</td>
            </tr>
            <tr>
              <td><strong style="color: #fbbf24;">Ambiguous</strong></td>
              <td><span class="mono">1.588</span></td>
              <td>9,10%</td>
              <td>Hỗn hợp chuyển tiếp, lớp nhiều nhất &lt; 60%.</td>
            </tr>
            <tr>
              <td><strong style="color: #94a3b8;">Unlabeled</strong></td>
              <td><span class="mono">3.587</span></td>
              <td>20,55%</td>
              <td>Giai đoạn nghỉ, không có chú thích hoạt động.</td>
            </tr>
            <tr style="background: rgba(255,255,255,0.04); font-weight: 700;">
              <td>Tổng toàn bộ stream</td>
              <td><span class="mono">17.453</span></td>
              <td>100%</td>
              <td>Toàn bộ chuỗi dòng cảm biến theo thời gian.</td>
            </tr>
          </tbody>
        </table>

        <div class="callout callout-info" style="margin-top: 0.4rem; padding: 0.5rem 0.7rem; font-size: 0.74rem;">
          <strong>Supervised Subsets vs Chronological Stream:</strong><br>
          Tập huấn luyện &amp; kiểm tra có giám sát chỉ gồm <strong>12.278 cửa sổ</strong> (Pure + Majority). <strong>1.588 Ambiguous</strong> và <strong>3.587 Unlabeled</strong> vẫn được giữ trong luồng để đánh giá suy luận thời gian thực liên tục.
        </div>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Xác thực: <code>output/week2_foundations_20261103/data_audit.json</code> (17.453 rows verified)</span>
    <span class="mono" style="color: var(--accent-cyan);">Pure: 10.361 &bull; Majority: 1.917</span>
  </div>
</section>
"""

SLIDE_06_HTML = r"""
<section class="slide-frame" id="slide-06" data-slide-idx="6" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part B. Data Foundations &bull; Slide 06</span>
      <h2 class="slide-title">Làm sạch Dữ liệu, Phân chia Tập &amp; Chuẩn hóa Tránh Rò rỉ</h2>
      <p class="slide-subtitle">Kiểm soát rủi ro Data Leakage qua phân chia theo Subject và quy chuẩn hóa z-score cố định Train-only</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">06 / 13</span>
      <span class="gate-badge gate-pass">Leakage-Free</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Python-rendered Math Formula Banner Image -->
    <div class="slide-img-container" onclick="openModal('assets/math_normalization.png')" style="margin-bottom: 0.4rem;">
      <img src="assets/math_normalization.png" alt="Train-Only Normalization Formula" style="max-height: 60px; object-fit: contain; margin: 0 auto;">
    </div>

    <!-- Side-by-side: Overlap Sample Multiplicity Image + Split Table -->
    <div class="grid-2" style="align-items: center;">
      <!-- Image Card -->
      <div class="card" style="padding: 0.5rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/data_split_multiplicity.png')">
          <img src="assets/data_split_multiplicity.png" alt="Split Multiplicity and Overlap Counting" style="max-height: 180px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Bội số Mẫu Thời gian Trùng lặp (Sample Multiplicity) trong các Phân vùng Train/Val/Test (Click phóng to)</div>
        </div>
      </div>

      <!-- Split Table & Leakage Rules -->
      <div class="card card-accent-emerald" style="padding: 0.6rem;">
        <h4 style="color: #fff; font-size: 0.88rem; margin-bottom: 0.3rem;">
          🛡️ Phân chia Tập dữ liệu theo Người dùng (Subject-Disjoint)
        </h4>
        <table class="custom-table" style="font-size: 0.74rem;">
          <thead>
            <tr>
              <th>Phân vùng (Split)</th>
              <th>Số Subjects</th>
              <th>All Windows</th>
              <th>Supervised Windows</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Train</strong></td>
              <td>17 người</td>
              <td><span class="mono">9.978</span></td>
              <td><span class="mono" style="color: #34d399; font-weight: 700;">6.955</span></td>
            </tr>
            <tr>
              <td><strong>Validation</strong></td>
              <td>4 người</td>
              <td><span class="mono">2.422</span></td>
              <td><span class="mono" style="color: #38bdf8; font-weight: 700;">1.751</span></td>
            </tr>
            <tr>
              <td><strong>Test</strong></td>
              <td>9 người</td>
              <td><span class="mono">5.053</span></td>
              <td><span class="mono" style="color: #fbbf24; font-weight: 700;">3.572</span></td>
            </tr>
            <tr style="background: rgba(255,255,255,0.04); font-weight: 700;">
              <td>Tổng cộng</td>
              <td>30 người</td>
              <td><span class="mono">17.453</span></td>
              <td><span class="mono">12.278</span></td>
            </tr>
          </tbody>
        </table>

        <div class="callout callout-warning" style="margin-top: 0.4rem; padding: 0.4rem 0.6rem; font-size: 0.74rem;">
          <strong>Chống Data Leakage Tuyệt đối:</strong> Thống kê chuẩn hóa \(\mu_c, \sigma_c\) được fit <strong>chỉ trên 6.955 supervised training windows</strong> và khóa cứng. Việc mẫu thời gian trùng lặp được tính lặp lại (window-weighted) không gây subject leakage.
        </div>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Mã băm chuẩn hóa: <code>aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e</code></span>
    <span class="mono" style="color: var(--accent-emerald);">Train-Only Fit</span>
  </div>
</section>
"""

SLIDE_07_HTML = r"""
<section class="slide-frame" id="slide-07" data-slide-idx="7" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <span class="slide-part-tag">Part B. Data Foundations &bull; Slide 07</span>
      <h2 class="slide-title">Chất lượng Dữ liệu, Sự phụ thuộc Thời gian &amp; Rủi ro Thiên lệch</h2>
      <p class="slide-subtitle">Bảng đối soát rubric bắt buộc: từ sự mất cân bằng lớp chuyển tiếp đến độ nhạy nguồn gốc chú thích</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">07 / 13</span>
      <span class="gate-badge gate-hold">Audit Findings</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Grid: 2 Audit Plots side-by-side -->
    <div class="grid-2" style="margin-bottom: 0.3rem;">
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/data_normalized_distances.png')">
          <img src="assets/data_normalized_distances.png" alt="Normalized Distances Distribution" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Khoảng cách Chuẩn hóa giữa các Cặp Overlap (Click phóng to)</div>
        </div>
      </div>

      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/delta_distributions.png')">
          <img src="assets/delta_distributions.png" alt="Temporal Delta Distributions" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">🔍 Phân phối Delta Biến thiên Tín hiệu theo Thời gian (Click phóng to)</div>
        </div>
      </div>
    </div>

    <!-- Audit Rubric Table -->
    <table class="custom-table" style="font-size: 0.72rem;">
      <thead>
        <tr>
          <th style="width: 20%;">Chiều Đánh giá</th>
          <th style="width: 38%;">Phát hiện Kiểm toán Thực tế (Audit Finding)</th>
          <th style="width: 42%;">Ý nghĩa Khoa học &amp; Ranh giới</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Missing values</strong></td>
          <td><span class="mono" style="color: #34d399;">0 NaN / 0 Inf</span> trên toàn bộ 122 sensor files.</td>
          <td>Độ toàn vẹn số học hoàn hảo; khác biệt với việc thiếu nhãn.</td>
        </tr>
        <tr>
          <td><strong>Missing labels</strong></td>
          <td><span class="mono" style="color: #fbbf24;">3.587 Unlabeled windows</span> (20,55% stream).</td>
          <td>Không có Ground Truth để tính Supervised Accuracy; giữ lại cho stream.</td>
        </tr>
        <tr>
          <td><strong>Duplicates &amp; Overlap</strong></td>
          <td><span class="mono">17.392 cặp liền kề</span> overlap 64 mẫu chính xác (\(\text{MAE} = 0\)).</td>
          <td>Chồng lấn hợp lệ (stride 64) không đồng nghĩa trùng lặp phi mong muốn.</td>
        </tr>
        <tr>
          <td><strong>Class imbalance</strong></td>
          <td>Các lớp chuyển tiếp có support cực thấp (Test: <strong>32–68</strong> windows).</td>
          <td>Accuracy tổng thể cao dễ che giấu hiệu năng kém ở các lớp chuyển tiếp.</td>
        </tr>
        <tr>
          <td><strong>Annotation provenance</strong></td>
          <td>Dịch tọa độ nhãn thô đi <strong>1 mẫu</strong> làm đổi <strong>98 gán nhãn</strong>!</td>
          <td>Nguồn gốc chú thích thượng nguồn (upstream convention) vẫn còn mở.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Part B: Hoàn thành khế ước dữ liệu (Data Contract) và nhận diện các rủi ro thiên lệch</span>
    <span class="mono" style="color: var(--accent-amber);">Gate G2: Transition Alert</span>
  </div>
</section>
"""
