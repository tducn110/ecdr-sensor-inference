# -*- coding: utf-8 -*-
"""
Part B: Data Foundations (Slides 04 - 07)
Style: Pirsch Analytics (Sunlit Paper Notebook, Cream Paper, Hairline Borders)
"""

SLIDE_04_HTML = r"""
<section class="slide-frame" id="slide-04" data-slide-idx="4" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">04</span>
        <span class="step-name-cell">HAPT DATASET &bull; SENSOR PHYSICS</span>
      </div>
      <h2 class="slide-title">HAPT Dataset: Tín hiệu Thô &amp; Cấu tạo Dạng sóng</h2>
      <p class="slide-subtitle">Bản chất vật lý của dữ liệu cảm biến thô 50 Hz, phân tích dao động bước đi và sự chuyển tiếp tư thế</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">04 / 13</span>
      <span class="gate-badge gate-pass">&check; PHYSICAL GROUNDING</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Top 4 Stats Cards -->
    <div class="grid-4" style="margin-bottom: 0.3rem;">
      <div class="card" style="padding: 0.75rem; text-align: center;">
        <div class="stat-value" style="font-size: 1.6rem;">30</div>
        <div class="stat-label">PARTICIPANTS</div>
        <div class="stat-desc" style="font-size: 0.75rem;">Người tham gia (19–48 tuổi) gắn điện thoại Samsung Galaxy S II ở thắt lưng.</div>
      </div>

      <div class="card" style="padding: 0.75rem; text-align: center;">
        <div class="stat-value" style="font-size: 1.6rem;">61</div>
        <div class="stat-label">RECORDINGS</div>
        <div class="stat-desc" style="font-size: 0.75rem;">122 tệp thô (mỗi bản ghi gồm 1 file gia tốc + 1 file con quay hồi chuyển).</div>
      </div>

      <div class="card" style="padding: 0.75rem; text-align: center;">
        <div class="stat-value" style="font-size: 1.6rem;">50 Hz</div>
        <div class="stat-label">SAMPLING RATE</div>
        <div class="stat-desc" style="font-size: 0.75rem;">50 mẫu/giây (\(\Delta t = 20\text{ ms}\)), thu nhận chuyển động liên tục.</div>
      </div>

      <div class="card" style="padding: 0.75rem; text-align: center; background: var(--color-sunshine-highlight);">
        <div class="stat-value" style="font-size: 1.6rem;">12 Classes</div>
        <div class="stat-label" style="color: var(--color-ink);">TAXONOMY</div>
        <div class="stat-desc" style="font-size: 0.75rem; color: var(--color-ink);">6 hoạt động cơ bản + 6 chuyển tiếp tư thế (Postural Transitions).</div>
      </div>
    </div>

    <!-- High-res Real Waveform Snippet Image -->
    <div class="slide-img-container" onclick="openModal('assets/pirsch_raw_waveform_extracted.png')">
      <img src="assets/pirsch_raw_waveform_extracted.png" alt="Raw 50Hz Sensor Waveforms and Text Snippet" style="max-height: 230px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Cấu tạo Dạng sóng Trích xuất Thực tế: 3 trục Gia tốc (g) + 3 trục Con quay (rad/s) + Snippet Số liệu Thô trong Tệp (Click phóng to)</div>
    </div>

    <!-- 2 Sub-cards: Sensor Physics & Channels vs Classes -->
    <div class="grid-2" style="margin-top: 0.2rem;">
      <div class="card" style="padding: 0.85rem;">
        <h4 style="font-size: 0.88rem; font-weight: 500; margin-bottom: 0.25rem;">📡 Bản chất Vật lý 6 Kênh Cảm biến:</h4>
        <ul style="font-size: 0.78rem; color: var(--color-graphite); line-height: 1.5; padding-left: 1.1rem;">
          <li><strong>Accelerometer (\(a_x, a_y, a_z\)):</strong> Gia tốc trọng trường + vận động (đơn vị \(g = 9,81\text{ m/s}^2\)). Khi đứng yên, trục \(X\) luôn xấp xỉ \(1,0\,g\) do trọng lực Trái Đất.</li>
          <li><strong>Gyroscope (\(\omega_x, \omega_y, \omega_z\)):</strong> Vận tốc góc xoay quanh 3 trục (đơn vị \(\text{rad/s}\)). Khi đổi tư thế (gập người, xoay người), con quay hồi chuyển tạo đỉnh xung tới \(\pm 0,8\text{ rad/s}\).</li>
        </ul>
      </div>

      <div class="card" style="padding: 0.85rem; background: rgba(255, 218, 110, 0.35);">
        <h4 style="font-size: 0.88rem; font-weight: 500; margin-bottom: 0.25rem; color: var(--color-ink);">⚠️ Phân biệt Cốt lõi: Kênh đo vs Nhãn lớp:</h4>
        <p style="font-size: 0.78rem; color: var(--color-ink); line-height: 1.5;">
          <strong>6 Sensor Channels</strong> là số chiều vật lý của không gian tín hiệu đầu vào. <strong>12 Activity Classes</strong> là số giả thuyết phân loại đầu ra. Tuyệt đối không nhầm lẫn 6 kênh cảm biến với mô hình phân loại 6 lớp lịch sử (\(K=6\))!
        </p>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Nguồn dữ liệu: Reyes-Ortiz et al. (2016), <code>01_DATA/03_Week_2_Data_Foundations.tex</code></span>
    <span class="gate-badge">6 Channels &ne; 12 Classes</span>
  </div>
</section>
"""

SLIDE_05_HTML = r"""
<section class="slide-frame" id="slide-05" data-slide-idx="5" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">05</span>
        <span class="step-name-cell">WINDOWING &bull; MATRIX EXTRACTION &bull; LABELS</span>
      </div>
      <h2 class="slide-title">Trích xuất Ma trận Đặc trưng &amp; Cơ chế Gán nhãn</h2>
      <p class="slide-subtitle">Biến đổi từ chuỗi liên tục thành ma trận 6×128 và phân tích vì sao cửa sổ được gán nhãn hoặc bị từ chối</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">05 / 13</span>
      <span class="gate-badge gate-pass">&check; EXACT POLICY</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Top Image: Matrix Extraction Diagram -->
    <div class="slide-img-container" onclick="openModal('assets/pirsch_window_matrix_extraction.png')">
      <img src="assets/pirsch_window_matrix_extraction.png" alt="Window Matrix Extraction Process" style="max-height: 135px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Quy trình Trích xuất: Tín hiệu Thô &rarr; Cửa sổ Trượt 128 mẫu &rarr; Ma trận Đầu vào CNN \(\mathbf{X} \in \mathbb{R}^{6 \times 128}\)</div>
    </div>

    <!-- Bottom Image: Label Policy Timeline -->
    <div class="slide-img-container" onclick="openModal('assets/pirsch_label_policy_timeline.png')" style="margin-top: 0.3rem;">
      <img src="assets/pirsch_label_policy_timeline.png" alt="Label Policy Timeline Analysis" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Phân tích Timeline: Tại sao Cửa sổ được gán nhãn (Pure, Majority) hoặc bị từ chối (Ambiguous, Unlabeled)?</div>
    </div>

    <!-- Summary Numbers Callout -->
    <div class="grid-2" style="margin-top: 0.3rem;">
      <div class="card" style="padding: 0.75rem;">
        <h4 style="font-size: 0.85rem; font-weight: 500; margin-bottom: 0.2rem;">📊 Thống kê Phân bổ 17.453 Cửa sổ:</h4>
        <ul style="font-size: 0.76rem; color: var(--color-graphite); line-height: 1.5; padding-left: 1.1rem;">
          <li><strong style="color: var(--color-ink);">Pure (10.361):</strong> 100% mẫu cùng 1 nhãn &rarr; ĐƯỢC GÁN NHÃN.</li>
          <li><strong style="color: var(--color-ink);">Majority (1.917):</strong> Nhãn chiếm ưu thế \(\ge 60\%\) &rarr; ĐƯỢC GÁN NHÃN.</li>
          <li><strong>Tổng Supervised Windows:</strong> <span class="mono" style="background: var(--color-mint-signal); font-weight: 700;">12.278 cửa sổ</span> dùng cho huấn luyện và đánh giá.</li>
        </ul>
      </div>

      <div class="card" style="padding: 0.75rem; background: var(--color-sunshine-highlight);">
        <h4 style="font-size: 0.85rem; font-weight: 500; margin-bottom: 0.2rem; color: var(--color-ink);">⚖️ Cửa sổ Không Được Gán Nhãn:</h4>
        <ul style="font-size: 0.76rem; color: var(--color-ink); line-height: 1.5; padding-left: 1.1rem;">
          <li><strong>Ambiguous (1.588):</strong> Điểm giao thoa hoạt động, lớp nhiều nhất &lt; 60% &rarr; Loại khỏi supervised, giữ trong stream.</li>
          <li><strong>Unlabeled (3.587):</strong> Khoảng nghỉ giữa các bài test &rarr; Không có Ground Truth, giữ trong sequential stream.</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Xác thực: <code>output/week2_foundations_20261103/data_audit.json</code> (17.453 rows verified)</span>
    <span class="gate-badge">Pure: 10.361 &bull; Majority: 1.917</span>
  </div>
</section>
"""

SLIDE_06_HTML = r"""
<section class="slide-frame" id="slide-06" data-slide-idx="6" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">06</span>
        <span class="step-name-cell">DATA CLEANING &bull; SPLIT &bull; NORMALIZATION</span>
      </div>
      <h2 class="slide-title">Làm sạch, Phân chia Tập &amp; Chuẩn hóa Tránh Rò rỉ</h2>
      <p class="slide-subtitle">Kiểm soát rủi ro Data Leakage qua phân chia theo Subject và quy chuẩn hóa z-score cố định Train-only</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">06 / 13</span>
      <span class="gate-badge gate-pass">&check; LEAKAGE-FREE</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Python-rendered Math Formula Banner Image -->
    <div class="slide-img-container" onclick="openModal('assets/pirsch_math_normalization.png')">
      <img src="assets/pirsch_math_normalization.png" alt="Train-Only Normalization Formula" style="max-height: 55px; object-fit: contain; margin: 0 auto;">
    </div>

    <!-- Side-by-side: Overlap Sample Multiplicity Image + Split Table -->
    <div class="grid-2" style="align-items: center; margin-top: 0.3rem;">
      <div class="card" style="padding: 0.5rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/data_split_multiplicity.png')">
          <img src="assets/data_split_multiplicity.png" alt="Split Multiplicity and Overlap Counting" style="max-height: 180px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">Bội số Mẫu Thời gian Trùng lặp (Sample Multiplicity) trong các Phân vùng Train/Val/Test</div>
        </div>
      </div>

      <div class="card" style="padding: 0.75rem;">
        <h4 style="font-size: 0.88rem; font-weight: 500; margin-bottom: 0.3rem;">
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
              <td><span class="mono" style="background: var(--color-mint-signal); font-weight: 700;">6.955</span></td>
            </tr>
            <tr>
              <td><strong>Validation</strong></td>
              <td>4 người</td>
              <td><span class="mono">2.422</span></td>
              <td><span class="mono" style="background: var(--color-sunshine-highlight); font-weight: 700;">1.751</span></td>
            </tr>
            <tr>
              <td><strong>Test</strong></td>
              <td>9 người</td>
              <td><span class="mono">5.053</span></td>
              <td><span class="mono" style="font-weight: 700;">3.572</span></td>
            </tr>
            <tr style="background: rgba(0,0,0,0.03); font-weight: 700;">
              <td>Tổng cộng</td>
              <td>30 người</td>
              <td><span class="mono">17.453</span></td>
              <td><span class="mono">12.278</span></td>
            </tr>
          </tbody>
        </table>

        <div class="callout-sunshine" style="margin-top: 0.4rem; padding: 0.4rem 0.75rem; font-size: 0.75rem;">
          <strong>Chống Data Leakage:</strong> Thống kê chuẩn hóa \(\mu_c, \sigma_c\) được fit <strong>chỉ trên 6.955 supervised training windows</strong> và khóa cứng. Việc mẫu thời gian trùng lặp được tính lặp lại (window-weighted) không gây subject leakage.
        </div>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Mã băm chuẩn hóa: <code>aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e</code></span>
    <span class="gate-badge">Train-Only Fit</span>
  </div>
</section>
"""

SLIDE_07_HTML = r"""
<section class="slide-frame" id="slide-07" data-slide-idx="7" data-part="Part B. Data Foundations">
  <div class="slide-header">
    <div>
      <div class="step-badge">
        <span class="step-num-cell">07</span>
        <span class="step-name-cell">STRIDE &bull; TEMPORAL DEPENDENCE &bull; BIAS</span>
      </div>
      <h2 class="slide-title">Giải thích Stride, Sự phụ thuộc &amp; Rủi ro Thiên lệch</h2>
      <p class="slide-subtitle">Lý do cốt tử của Stride 64 mẫu (50% overlap), kiểm toán tương quan chuỗi và độ nhạy nhãn</p>
    </div>
    <div class="slide-meta-right">
      <span class="slide-num-pill">07 / 13</span>
      <span class="gate-badge gate-hold">! AUDIT FINDINGS</span>
    </div>
  </div>

  <div class="slide-body">
    <!-- Top Image: Stride Rationale Diagram -->
    <div class="slide-img-container" onclick="openModal('assets/pirsch_stride_rationale.png')">
      <img src="assets/pirsch_stride_rationale.png" alt="Stride 64 and 50% Overlap Rationale" style="max-height: 145px; object-fit: contain; margin: 0 auto;">
      <div class="img-caption">Giải thích Stride 64 (50% Overlap): Tránh Điểm Mù Ranh giới, Tăng Tốc độ Phản hồi &amp; Tạo Tiền đề Cho Decision Reuse</div>
    </div>

    <!-- Side-by-side: Normalized Distances + Audit Rubric Table -->
    <div class="grid-2" style="align-items: center; margin-top: 0.2rem;">
      <div class="card" style="padding: 0.4rem; text-align: center;">
        <div class="slide-img-container" onclick="openModal('assets/data_normalized_distances.png')">
          <img src="assets/data_normalized_distances.png" alt="Normalized Distances Distribution" style="max-height: 140px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">Khoảng cách Chuẩn hóa giữa các Cặp Overlap (Click phóng to)</div>
        </div>
      </div>

      <div class="card" style="padding: 0.6rem;">
        <table class="custom-table" style="font-size: 0.72rem;">
          <thead>
            <tr>
              <th>Chiều Đánh giá</th>
              <th>Phát hiện Kiểm toán Thực tế</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Missing values</strong></td>
              <td><span class="mono" style="background: var(--color-mint-signal); font-weight: 700;">0 NaN / 0 Inf</span> trên toàn bộ 122 sensor files.</td>
            </tr>
            <tr>
              <td><strong>Duplicates &amp; Overlap</strong></td>
              <td><span class="mono">17.392 cặp liền kề</span> overlap 64 mẫu chính xác (\(\text{MAE} = 0\)).</td>
            </tr>
            <tr>
              <td><strong>Class imbalance</strong></td>
              <td>Các lớp chuyển tiếp có support cực thấp (Test: <strong>32–68</strong> windows).</td>
            </tr>
            <tr>
              <td><strong>Annotation provenance</strong></td>
              <td>Dịch tọa độ nhãn thô đi <strong>1 mẫu</strong> làm đổi <strong>98 gán nhãn</strong>!</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="slide-footer">
    <span>Kết thúc Part B: Hoàn thành khế ước dữ liệu (Data Contract) và nhận diện các rủi ro thiên lệch</span>
    <span class="gate-badge">Gate G2: Transition Alert</span>
  </div>
</section>
"""
