# -*- coding: utf-8 -*-
"""
Appendix & Deep-Dive Q&A Section with Full Empirical Evidence & Tables.
Style: Pirsch Analytics (Sunlit Paper Notebook, Cream Paper, Hairline Borders)
"""

APPENDIX_HTML = r"""
<section class="appendix-section" id="appendix-qa" style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--color-border-hairline);">
  <div class="card" style="margin-bottom: 2rem; background: var(--color-sunshine-highlight);">
    <div class="step-badge" style="background: var(--color-cream-paper); margin-bottom: 0.5rem;">
      <span class="step-num-cell">APPX</span>
      <span class="step-name-cell">TECHNICAL APPENDIX &bull; TUTOR Q&amp;A DEFENSE</span>
    </div>
    <h2 style="font-size: 1.5rem; color: var(--color-ink); font-weight: 500; display: flex; align-items: center; gap: 0.6rem;">
      Phụ lục Kỹ thuật &amp; Bộ Câu hỏi Đối thoại Chuyên sâu
    </h2>
    <p style="font-size: 0.88rem; color: var(--color-ink); margin-top: 0.4rem; line-height: 1.6;">
      Các bảng dữ liệu đầy đủ, dẫn xuất toán học chi tiết và bằng chứng kiểm toán được bảo lưu độc lập nhằm phục vụ phiên chất vấn sâu với Tutor và Hội đồng đánh giá khoa học.
    </p>
  </div>

  <!-- 8 Deep-Dive Accordions -->
  <div style="margin-bottom: 2.5rem;">
    <h3 style="font-size: 1.15rem; font-weight: 500; color: var(--color-ink); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
      <span>🎯</span> 8 Câu hỏi Trọng tâm Được Tutor Hỏi Sâu &amp; Lời Giải Khoa học
    </h3>

    <!-- Q1 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q1">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q1')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q1: Vì sao chọn kích thước cửa sổ 128 samples và bước trượt stride 64 samples?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p><strong>Dẫn xuất cơ sinh học &amp; Tần số lấy mẫu:</strong></p>
        <ul style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li>Tần số lấy mẫu của cảm biến là \(50\text{ Hz}\) (chu kỳ \(T = 20\text{ ms}\)). Do đó, cửa sổ 128 mẫu có độ dài thời gian chính xác là:
            \[ \Delta t = \frac{128}{50} = 2,56\text{ giây} \]
          </li>
          <li>Theo nghiên cứu cơ sinh học dáng đi người (Human Gait Analysis), một chu kỳ bước đi (gait cycle - gồm 2 bước chân trái và phải) của người trưởng thành thường dao động từ \(0,8\) đến \(1,3\) giây. Cửa sổ \(2,56\text{ giây}\) bảo đảm chứa trọn vẹn từ <strong>2 đến 3 chu kỳ bước đi đầy đủ</strong>, cung cấp đủ thông tin tần số và biên độ để phân biệt đi bộ thường, đi lên lầu và đi xuống lầu.</li>
          <li>Bước trượt \(64\text{ mẫu}\) (\(1,28\text{ giây}\)) tạo ra độ chồng lấn chính xác \(50\%\). Khoảng trượt \(1,28\text{ giây}\) là mốc thời gian phản hồi hợp lý trong các hệ thống giám sát sức khỏe người già (vừa đủ nhanh để phát hiện ngã hoặc đổi tư thế, vừa tiết kiệm năng lượng).</li>
          <li><strong>Ranh giới bản ghi:</strong> Toàn bộ 61 bản ghi đều được ngắt cửa sổ độc lập (reset chỉ số). Không có bất kỳ cửa sổ nào vắt qua 2 người hoặc 2 bản ghi khác nhau. Bộ kiểm toán đã kiểm tra chính xác <strong>17.392 cặp liền kề</strong> và xác nhận độ lệch tuyệt đối của 64 mẫu trùng lấn là \(0,00000\).</li>
        </ul>
      </div>
    </div>

    <!-- Q2 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q2">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q2')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q2: Quy trình Data Cleaning đã thực sự kiểm tra những gì để đảm bảo tính toàn vẹn?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p>Quy trình kiểm toán thực thi của Week 2 (chạy qua <code>scripts/week2_data_audit.py</code>) thực hiện 2 tầng kiểm soát số học khắt khe:</p>
        <ol style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>Tầng 1 - Kiểm tra Dữ liệu thô (Raw Data Validation):</strong>
            <ul style="padding-left: 1rem; margin-top: 0.2rem;">
              <li>Ghép cặp chính xác 61 file gia tốc với 61 file con quay hồi chuyển theo đúng định danh thực nghiệm và người tham gia.</li>
              <li>Kiểm tra tính hữu hạn số học: Quét qua toàn bộ \(17.453 \times 6 \times 128 = 13.403.904\) giá trị số thực float32 &rarr; <strong>Không phát hiện bất kỳ giá trị NaN hoặc Inf nào</strong>.</li>
              <li>Kiểm tra ranh giới chú thích: Xác nhận toàn bộ các khoảng thời gian bắt đầu và kết thúc của nhãn đều nằm trong phạm vi chỉ số của bản ghi.</li>
            </ul>
          </li>
          <li><strong>Tầng 2 - Xây dựng Bộ dữ liệu &amp; Chống Rò rỉ (Dataset Construction &amp; Anti-Leakage):</strong>
            <ul style="padding-left: 1rem; margin-top: 0.2rem;">
              <li>Chia tập theo người dùng độc lập (Subject-disjoint): 17 người cho Train, 4 người cho Validation, 9 người cho Test. Ba tập này rời nhau từng đôi một (pairwise disjoint).</li>
              <li>Phân loại chính sách gán nhãn nghiêm ngặt: Pure (10.361), Majority (1.917), Ambiguous (1.588), Unlabeled (3.587). Chỉ 12.278 cửa sổ Pure + Majority được đưa vào tập Supervised.</li>
              <li>Chuẩn hóa z-score cố định: Fit duy nhất trên 6.955 cửa sổ Train và lưu hash SHA-256 cố định, bảo đảm dữ liệu Test không bao giờ bị rò rỉ vào phân phối chuẩn hóa.</li>
            </ul>
          </li>
        </ol>
      </div>
    </div>

    <!-- Q3 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q3">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q3')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q3: Vì sao mạng phân loại 12 lớp (K=12) mà đầu vào lại là ma trận kích thước 6×128?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p>Đây là sự phân biệt rạch ròi giữa <strong>không gian tín hiệu cảm biến vật lý</strong> và <strong>không gian giả thuyết phân loại</strong>:</p>
        <ul style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>Đầu vào \([6, 128]\):</strong> 6 là số kênh đo đạc vật lý độc lập (3 trục gia tốc tuyến tính \(a_x, a_y, a_z\) từ Accelerometer và 3 trục vận tốc góc \(\omega_x, \omega_y, \omega_z\) từ Gyroscope). 128 là số điểm lấy mẫu theo trục thời gian trong 2,56 giây. Đây là biểu diễn động học của cơ thể.</li>
          <li><strong>Đầu ra 12 Logits (\(K=12\)):</strong> 12 là số lượng lớp hoạt động trong phân loại HAPT:
            <ul style="padding-left: 1rem; margin-top: 0.2rem;">
              <li>6 hoạt động cơ bản lặp lại: WALKING (1), WALKING_UPSTAIRS (2), WALKING_DOWNSTAIRS (3), SITTING (4), STANDING (5), LAYING (6).</li>
              <li>6 hoạt động chuyển tiếp tư thế: STAND_TO_SIT (7), SIT_TO_STAND (8), SIT_TO_LIE (9), LIE_TO_SIT (10), STAND_TO_LIE (11), LIE_TO_STAND (12).</li>
            </ul>
          </li>
          <li><em>Lưu ý lịch sử:</em> Trong các nghiên cứu lý thuyết ban đầu (ví dụ UCI HAR gốc), người ta chỉ dùng 6 lớp cơ bản (K=6). Nhóm Week 2 đã đính chính hoàn toàn các tài liệu cũ để đồng bộ 100% với mô hình thực tế SmallCNN1D K=12.</li>
        </ul>
      </div>
    </div>

    <!-- Q4 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q4">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q4')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q4: Vì sao lựa chọn CNN1D thay vì các kiến trúc như LSTM hay Transformer?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p><strong>Lý do lựa chọn CNN1D:</strong></p>
        <ul style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>Tính cục bộ thời gian (Temporal Locality):</strong> Kernel tích chập trượt trên trục thời gian học rất tốt các mẫu hình cơ sinh học cục bộ (ví dụ: cú dập gót chân, lực xoay hông). Kernel đầu tiên có độ rộng 5 mẫu (\(0,10\text{ s}\)), phù hợp với các xung lực ngắn hạn.</li>
          <li><strong>Chia sẻ trọng số &amp; Tiết kiệm tham số:</strong> So với MLP kết nối dày đặc, CNN1D giảm số lượng trọng số xuống chỉ còn 283.692 tham số, tiêu tốn 1,17 triệu MACs, rất lý tưởng để chạy trên các vi điều khiển nhúng hoặc vi xử lý cạnh (Edge Computing).</li>
          <li><strong>Khả năng đóng khung toán học (Mathematical Tractability):</strong> Các toán tử Conv1D, BatchNorm và ReLU là các hàm tuyến tính từng đoạn (Piecewise Linear). Việc tính toán cận dưới biên (Bound Propagation) trên các hàm tuyến tính từng đoạn dễ dàng và chặt chẽ hơn nhiều so với các cổng phi tuyến tính phức tạp như Sigmoid/Tanh trong LSTM hay cơ chế Softmax Attention trong Transformer.</li>
          <li><em>Tính trung thực khoa học:</em> Nhóm ghi nhận rõ trong báo cáo rằng Week 2 chưa làm ablation study thực nghiệm để so sánh định lượng tốc độ/độ chính xác giữa CNN1D và LSTM. Đây là kiến trúc được chọn làm frozen reference cho đề tài.</li>
        </ul>
      </div>
    </div>

    <!-- Q5 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q5">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q5')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q5: Hiệu năng Baseline thực sự tốt đến đâu và vì sao Gate G2 vẫn ở trạng thái HOLD?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p>Báo cáo kiểm toán <code>model_findings.md</code> chỉ rõ hai mặt của baseline:</p>
        <ul style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>Mặt mạnh:</strong> Độ chính xác tổng thể đạt <strong>89,05%</strong> trên 3.572 cửa sổ kiểm tra có giám sát. Macro-F1 đạt <strong>0,7763</strong>. Các hoạt động cơ bản như Đi bộ (F1 = 0,958) hay Nằm (F1 = 0,989) có độ tin cậy vượt trội.</li>
          <li><strong>Mặt yếu:</strong> Các lớp chuyển tiếp tư thế có hiệu năng rất thấp: STAND_TO_LIE chỉ đạt F1 = 0,5496, LIE_TO_STAND đạt F1 = 0,5510, LIE_TO_SIT đạt F1 = 0,6667. Thêm vào đó, kích thước mẫu thử nghiệm của các lớp này rất nhỏ (từ 32 đến 68 cửa sổ), dẫn đến độ biến động thống kê lớn.</li>
          <li><strong>Lỗi nhầm lẫn tư thế tĩnh:</strong> SITTING và STANDING gây ra 163 ca dự đoán sai (57 mẫu Ngồi đoán thành Đứng; 106 mẫu Đứng đoán thành Ngồi).</li>
          <li><strong>Quyết định quản trị:</strong> Gate G2 (Model Quality Gate) đánh giá chất lượng mô hình cơ sở. Vì mô hình còn điểm yếu rõ rệt ở các lớp chuyển tiếp, nhóm quyết định giữ Gate G2 ở trạng thái <strong>HOLD</strong>, không vội vàng đóng PASS. Điều này thể hiện tính liêm chính và sự thận trọng khoa học cao nhất.</li>
        </ul>
      </div>
    </div>

    <!-- Q6 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q6">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q6')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q6: Vì sao sự trùng lấn 50% thời gian lại KHÔNG ĐỦ để tái sử dụng quyết định an toàn?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p>Có 3 nguyên nhân cốt tử giải thích vì sao trùng lấn 50% không đồng nghĩa với giữ nguyên nhãn:</p>
        <ol style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>Tín hiệu mới mang biến thiên lớn:</strong> Dù 64 mẫu cũ trùng khớp hoàn toàn, nhưng 64 mẫu mới (\(1,28\text{ s}\) mới) mang thông tin vận động mới. Sai số tuyệt đối trung bình (MAE) cục bộ giữa 2 cửa sổ có thể lên tới \(0,43\text{ rad/s}\) trên con quay hồi chuyển (tương đương biến thiên chuẩn hóa dimensionless lên tới \(0,96\sigma\)).</li>
          <li><strong>Hiện tượng lật nhãn thực nghiệm:</strong> Bằng chứng Week 1 cho thấy ở các cửa sổ chuyển tiếp, có tới <strong>47,73% số cặp bị lật nhãn</strong>! Ngay cả ở các cửa sổ hoạt động ổn định, vẫn có <strong>2,18% số cặp bị lật nhãn</strong> do biến động bước chân hoặc rung động cảm biến.</li>
          <li><strong>Khoảng cách biên quyết định điểm (Pointwise Margin) có thể rất hẹp:</strong> Phân tích 5.053 cửa sổ test cho thấy giá trị khoảng cách giữa logit lớn nhất và logit lớn nhì có giá trị cực tiểu chỉ là:
            \[ \text{Margin}_{\min} = 0,0001135 \]
            Khi khoảng cách biên quá nhỏ, chỉ cần một biến động cực nhỏ ở 64 mẫu mới cũng đủ làm đảo chiều vị trí cực đại (\(\arg\max\)) của hàm Softmax!</li>
        </ol>
      </div>
    </div>

    <!-- Q7 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q7">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q7')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q7: Chứng nhận (Certificate) thực sự chứng minh điều gì và có hợp đồng kiểm tra ra sao?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p>Chứng chỉ ECDR không phải là một xác suất hay một độ tin cậy ước lượng, mà là một <strong>bảo đảm toán học bất biến toàn miền (Deterministic Region Invariance)</strong>:</p>
        <div style="background: var(--color-white); border: 1px solid var(--color-border-hairline); border-radius: 12px; padding: 0.6rem 1rem; margin: 0.5rem 0;">
          \[ \forall X \in S, \quad F_c(X) - F_j(X) > 0 \quad (\forall j \neq c) \implies \arg\max_{k} F_k(X) \equiv c \]
        </div>
        <p>Để hệ thống runtime được phép tái sử dụng quyết định, phải vượt qua Khế ước Kiểm tra 4 điều kiện:</p>
        <ul style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>1. Membership Check:</strong> Kiểm tra xem cửa sổ mới \(X_{new}\) có nằm trọn vẹn trong miền ứng viên \(S\) đã được chứng nhận hay không.</li>
          <li><strong>2. Reference Identity Check:</strong> Mã băm SHA-256 của trọng số mạng và ID chuẩn hóa dữ liệu phải trùng khớp 100% với Anchor đã lưu.</li>
          <li><strong>3. Numerical Soundness:</strong> Cận dưới của hiệu số logit phải được tính toán bằng số học khoảng (Interval Arithmetic) có kiểm soát sai số làm tròn của số thực float32.</li>
          <li><strong>4. Fail-Closed Fallback:</strong> Nếu bất kỳ điều kiện nào không thỏa mãn hoặc tính toán bị nghi ngờ, hệ thống lập tức Fallback chạy full CNN forward pass. Tuyệt đối không bao giờ trả về kết quả phỏng đoán!</li>
        </ul>
      </div>
    </div>

    <!-- Q8 -->
    <div class="card" style="margin-bottom: 0.75rem; padding: 1rem 1.25rem;" id="q8">
      <div style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;" onclick="toggleAccordion('q8')">
        <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink);">
          Q8: Tuần tiếp theo (Week 3) nhóm sẽ kiểm chứng toán học và thực thi verifier như thế nào?
        </h4>
        <span style="font-size: 1.1rem; font-weight: 700;">&darr;</span>
      </div>
      <div class="accordion-content" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--color-border-muted);">
        <p>Kế hoạch triển khai kỹ thuật của Tuần 3 tập trung giải quyết 3 nghĩa vụ cốt lõi:</p>
        <ul style="padding-left: 1.2rem; margin: 0.4rem 0; font-size: 0.85rem; color: var(--color-graphite); line-height: 1.6;">
          <li><strong>Thuật toán tính cận trên Computation Graph thực tế:</strong> Không sử dụng các mô hình đồ chơi (toy model) hay giả định Global Average Pooling (GAP). Nhóm sẽ xây dựng thuật toán Bound Propagation đi qua đúng 3 tầng Conv1D, 3 tầng BatchNorm, 3 tầng ReLU, 2 tầng MaxPool, tầng Flatten (2.048 chiều) và 2 tầng Linear của SmallCNN1D K=12.</li>
          <li><strong>Kiểm soát số học Float32 (Numerical Soundness):</strong> Xác lập bằng chứng toán học chứng minh cận dưới tính ra không bị vi phạm bởi sai số làm tròn điểm động (floating-point rounding errors) của GPU/CPU backend.</li>
          <li><strong>Hạch toán chi phí thực thi (Latency &amp; Net Cost Accounting):</strong> Đo đạc thời gian thực hiện của bộ kiểm tra Membership và Verifier. Thời gian chạy của Verifier bắt buộc phải nhỏ hơn thời gian chạy forward pass của CNN (hiện đo được trên CPU là \(p_{50} = 0,1557\text{ ms}\)), nếu không cơ chế Certified Reuse sẽ không mang lại lợi ích tăng tốc thực tế (No net speedup).</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- Visual Evidence Gallery (3 Core Audit Figures) -->
  <div class="card" style="margin-bottom: 2rem;">
    <h4 style="font-size: 1.05rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
      <span>🖼️</span> Thư viện Đồ thị Bằng chứng Kiểm toán Thực nghiệm (Visual Evidence Gallery)
    </h4>
    <div class="grid-3">
      <div class="card" style="padding: 0.5rem; text-align: center; background: var(--color-white);">
        <div class="slide-img-container" onclick="openModal('assets/model_confusion_matrix.png')">
          <img src="assets/model_confusion_matrix.png" alt="Confusion Matrix 12x12" style="max-height: 160px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">Hình 1: Ma trận Nhầm lẫn 12x12 (HAPT Test Set, 3.572 mẫu)</div>
        </div>
        <p style="font-size: 0.74rem; color: var(--color-graphite); margin-top: 0.4rem;">
          Làm rõ 163 ca nhầm lẫn giữa Sitting và Standing, cùng sự suy giảm recall ở các lớp chuyển tiếp tư thế.
        </p>
      </div>

      <div class="card" style="padding: 0.5rem; text-align: center; background: var(--color-white);">
        <div class="slide-img-container" onclick="openModal('assets/model_layer_macs.png')">
          <img src="assets/model_layer_macs.png" alt="Layer MACs Breakdown" style="max-height: 160px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">Hình 2: Phân bổ MACs theo Tầng (1.172.992 MACs)</div>
        </div>
        <p style="font-size: 0.74rem; color: var(--color-graphite); margin-top: 0.4rem;">
          Khối tích chập Conv chiếm 77,5% chi phí (909.312 MACs), phần Flatten + Linear chiếm 22,5% (263.680 MACs).
        </p>
      </div>

      <div class="card" style="padding: 0.5rem; text-align: center; background: var(--color-white);">
        <div class="slide-img-container" onclick="openModal('assets/model_top_two_margins.png')">
          <img src="assets/model_top_two_margins.png" alt="Top-Two Logit Margins" style="max-height: 160px; object-fit: contain; margin: 0 auto;">
          <div class="img-caption">Hình 3: Phân phối Khoảng cách Biên (Top-Two Margins)</div>
        </div>
        <p style="font-size: 0.74rem; color: var(--color-graphite); margin-top: 0.4rem;">
          Đo trên 5.053 stream windows; trung vị là 5,98 nhưng giá trị tối thiểu chỉ 0,00011, cho thấy nguy cơ lật nhãn tại biên.
        </p>
      </div>
    </div>
  </div>

  <!-- Detailed Data Tables Grid -->
  <div class="grid-2" style="margin-bottom: 2rem;">
    <!-- 12-Class Metrics Table -->
    <div class="card">
      <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.5rem;">
        📊 Bảng Chỉ số Đánh giá Chi tiết 12 Lớp (HAPT Test Set &bull; 3.572 Windows)
      </h4>
      <div style="max-height: 380px; overflow-y: auto;">
        <table class="custom-table" style="font-size: 0.74rem;">
          <thead>
            <tr>
              <th>ID</th>
              <th>Tên lớp hoạt động</th>
              <th>Support</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>1</td><td>WALKING</td><td>531</td><td>0,9603</td><td>0,9567</td><td style="font-weight: 700; color: #047857;">0,9585</td></tr>
            <tr><td>2</td><td>WALKING_UPSTAIRS</td><td>537</td><td>0,9331</td><td>0,9088</td><td style="font-weight: 700; color: #047857;">0,9208</td></tr>
            <tr><td>3</td><td>WALKING_DOWNSTAIRS</td><td>492</td><td>0,9237</td><td>0,9837</td><td style="font-weight: 700; color: #047857;">0,9528</td></tr>
            <tr><td>4</td><td>SITTING</td><td>546</td><td>0,7997</td><td>0,8407</td><td style="font-weight: 700;">0,8196</td></tr>
            <tr><td>5</td><td>STANDING</td><td>597</td><td>0,8798</td><td>0,8090</td><td style="font-weight: 700;">0,8429</td></tr>
            <tr><td>6</td><td>LAYING</td><td>586</td><td>0,9831</td><td>0,9949</td><td style="font-weight: 700; color: #047857;">0,9890</td></tr>
            <tr><td>7</td><td>STAND_TO_SIT</td><td>45</td><td>0,7619</td><td>0,7111</td><td style="font-weight: 700; color: #b45309;">0,7356</td></tr>
            <tr><td>8</td><td>SIT_TO_STAND</td><td>32</td><td>0,7778</td><td>0,6563</td><td style="font-weight: 700; color: #b45309;">0,7119</td></tr>
            <tr><td>9</td><td>SIT_TO_LIE</td><td>51</td><td>0,5893</td><td>0,6471</td><td style="font-weight: 700; color: #b91c1c;">0,6168</td></tr>
            <tr><td>10</td><td>LIE_TO_SIT</td><td>46</td><td>0,7714</td><td>0,5870</td><td style="font-weight: 700; color: #b91c1c;">0,6667</td></tr>
            <tr><td>11</td><td>STAND_TO_LIE</td><td>68</td><td>0,5714</td><td>0,5294</td><td style="font-weight: 700; color: #b91c1c;">0,5496</td></tr>
            <tr><td>12</td><td>LIE_TO_STAND</td><td>41</td><td>0,4737</td><td>0,6585</td><td style="font-weight: 700; color: #b91c1c;">0,5510</td></tr>
            <tr style="background: var(--color-sunshine-highlight); font-weight: 700;">
              <td colspan="2">Trung bình Toàn mạng (Macro Avg)</td>
              <td>3.572</td>
              <td>0,7888</td>
              <td>0,7741</td>
              <td style="color: var(--color-ink);">0,7763</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Model Layers & Arithmetic Footprint -->
    <div class="card">
      <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.5rem;">
        ⚡ Bảng Cấu trúc Tầng &amp; Đếm Số học MACs (15 Modules)
      </h4>
      <div style="max-height: 380px; overflow-y: auto;">
        <table class="custom-table" style="font-size: 0.74rem;">
          <thead>
            <tr>
              <th>Module</th>
              <th>Toán tử</th>
              <th>Kích thước Đầu ra</th>
              <th>Trainable Params</th>
              <th>MACs / Sample</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><code>conv.0</code></td><td>Conv1d (6&rarr;32, k=5)</td><td><code>[1, 32, 128]</code></td><td>992</td><td>122.880</td></tr>
            <tr><td><code>conv.1</code></td><td>BatchNorm1d</td><td><code>[1, 32, 128]</code></td><td>64</td><td>&mdash;</td></tr>
            <tr><td><code>conv.2</code></td><td>ReLU</td><td><code>[1, 32, 128]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>conv.3</code></td><td>MaxPool1d (k=2, s=2)</td><td><code>[1, 32, 64]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>conv.4</code></td><td>Conv1d (32&rarr;64, k=3)</td><td><code>[1, 64, 64]</code></td><td>6.208</td><td>393.216</td></tr>
            <tr><td><code>conv.5</code></td><td>BatchNorm1d</td><td><code>[1, 64, 64]</code></td><td>128</td><td>&mdash;</td></tr>
            <tr><td><code>conv.6</code></td><td>ReLU</td><td><code>[1, 64, 64]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>conv.7</code></td><td>MaxPool1d (k=2, s=2)</td><td><code>[1, 64, 32]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>conv.8</code></td><td>Conv1d (64&rarr;64, k=3)</td><td><code>[1, 64, 32]</code></td><td>12.352</td><td>393.216</td></tr>
            <tr><td><code>conv.9</code></td><td>BatchNorm1d</td><td><code>[1, 64, 32]</code></td><td>128</td><td>&mdash;</td></tr>
            <tr><td><code>conv.10</code></td><td>ReLU</td><td><code>[1, 64, 32]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>head.0</code></td><td>Flatten</td><td><code>[1, 2048]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>head.1</code></td><td>Linear (2048&rarr;128)</td><td><code>[1, 128]</code></td><td>262.272</td><td>262.144</td></tr>
            <tr><td><code>head.2</code></td><td>ReLU</td><td><code>[1, 128]</code></td><td>0</td><td>&mdash;</td></tr>
            <tr><td><code>head.3</code></td><td>Linear (128&rarr;12)</td><td><code>[1, 12]</code></td><td>1.548</td><td>1.536</td></tr>
            <tr style="background: var(--color-mint-signal); font-weight: 700;">
              <td colspan="3">Tổng cộng toàn mạng</td>
              <td>283.692</td>
              <td>1.172.992 MACs</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Normalization & Timing Tables Grid -->
  <div class="grid-2" style="margin-bottom: 2rem;">
    <!-- Normalization Vectors -->
    <div class="card">
      <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.5rem;">
        📐 Tham số Chuẩn hóa Khóa Cố định (ID: <code>aa8e1cc...</code>)
      </h4>
      <table class="custom-table" style="font-size: 0.74rem;">
        <thead>
          <tr>
            <th>Kênh Cảm biến</th>
            <th>Thứ nguyên gốc</th>
            <th>Mean (\(\mu_c\))</th>
            <th>Std (\(\sigma_c\))</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><strong>Acc X</strong></td><td>\(g\)</td><td>\(0,8043609262\)</td><td>\(0,4057796597\)</td></tr>
          <tr><td><strong>Acc Y</strong></td><td>\(g\)</td><td>\(0,0411366001\)</td><td>\(0,4071546793\)</td></tr>
          <tr><td><strong>Acc Z</strong></td><td>\(g\)</td><td>\(0,0817700475\)</td><td>\(0,3549118936\)</td></tr>
          <tr><td><strong>Gyro X</strong></td><td>\(\text{rad/s}\)</td><td>\(0,0091355676\)</td><td>\(0,4476351142\)</td></tr>
          <tr><td><strong>Gyro Y</strong></td><td>\(\text{rad/s}\)</td><td>\(-0,0043975562\)</td><td>\(0,4212706089\)</td></tr>
          <tr><td><strong>Gyro Z</strong></td><td>\(\text{rad/s}\)</td><td>\(-0,0071832035\)</td><td>\(0,2964415252\)</td></tr>
        </tbody>
      </table>
      <p style="font-size: 0.72rem; color: var(--color-graphite); margin-top: 0.4rem;">
        Quy chuẩn: Tích lũy float64, lưu trữ float32, ngưỡng sàn \(\sigma_{\min} = 10^{-6}\).
      </p>
    </div>

    <!-- CPU Timing Benchmark -->
    <div class="card">
      <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.5rem;">
        ⏱️ Benchmark Độ trễ Forward Pass trên CPU (Reference Baseline)
      </h4>
      <table class="custom-table" style="font-size: 0.74rem;">
        <thead>
          <tr>
            <th>Chỉ số Đo đạc</th>
            <th>Giá trị Thực nghiệm</th>
            <th>Quy ước Đo &amp; Thiết lập Môi trường</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><strong>Trung vị (p50 Latency)</strong></td><td><span class="mono" style="background: var(--color-mint-signal); font-weight: 700;">0,1557 ms</span></td><td>100 warmups &bull; 1.000 lần đo bằng <code>perf_counter_ns</code></td></tr>
          <tr><td><strong>Phân vị 95 (p95 Latency)</strong></td><td><span class="mono" style="font-weight: 700;">0,3111 ms</span></td><td>Batch size = 1, Resident memory tensor</td></tr>
          <tr><td><strong>Trung bình (Mean Latency)</strong></td><td><span class="mono">0,1892 ms</span></td><td>1 PyTorch Thread &bull; Thiết bị CPU</td></tr>
          <tr><td><strong>Phạm vi đo (Scope)</strong></td><td colspan="2" style="color: var(--color-graphite);">Chỉ tính forward pass mạng CNN; loại trừ I/O, tiền xử lý và fallback.</td></tr>
        </tbody>
      </table>
      <div class="callout-sunshine" style="margin-top: 0.4rem; padding: 0.4rem 0.6rem; font-size: 0.74rem;">
        <strong>Hạn ngạch cho Verifier Tuần 3:</strong> Tổng thời gian chạy của bộ Verifier ECDR bắt buộc phải dưới <strong>0,15 ms</strong> mới có thể mang lại tốc độ thực tế (Net Speedup).
      </div>
    </div>
  </div>

  <!-- Delivery Manifest & SHA-256 Hashes Table -->
  <div class="card" style="margin-bottom: 2rem;">
    <h4 style="font-size: 0.95rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.5rem;">
      🔒 Danh mục Sản phẩm Bàn giao &amp; Mã băm Toàn vẹn (Delivery Manifest SHA-256)
    </h4>
    <div style="max-height: 320px; overflow-y: auto;">
      <table class="custom-table" style="font-size: 0.72rem;">
        <thead>
          <tr>
            <th>Đường dẫn Tệp (Relative Path)</th>
            <th>Kích thước</th>
            <th>Mã băm SHA-256 (Tái lập kiểm toán)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td><code>01_DATA/03_Week_2_Data_Foundations.tex</code></td><td>39.842 B</td><td><span class="mono">9e3abf03da4c5d9125646b598691f355802974153bf732ea29c6244a6ef8275b</span></td></tr>
          <tr><td><code>01_DATA/03_Week_2_Data_Foundations.pdf</code></td><td>108.744 B</td><td><span class="mono">90ebf2e324bbcde3d87b9be8c03fd37dd8f5003f6c4051418acd4f7701bbca9d</span></td></tr>
          <tr><td><code>02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex</code></td><td>45.428 B</td><td><span class="mono">53ba8581063f4dd0ed1c1c047b78fca8ddf17cf71e5ba4aab55e1b997c5dae89</span></td></tr>
          <tr><td><code>02_MODEL/01_ECDR_CNN_Model_Analysis_EN.pdf</code></td><td>283.737 B</td><td><span class="mono">02de3fb4c6ca0a07eb1407a6017ed540c0dd45b8292cbc0c1e822c42d4e57188</span></td></tr>
          <tr><td><code>output/week2_foundations_20261103/data_findings.md</code></td><td>9.993 B</td><td><span class="mono">4884d74fefc2273bae4d62eb5265cddd9b9129a6382cd25097a9a15e7bfadc58</span></td></tr>
          <tr><td><code>output/week2_foundations_20261103/model_findings.md</code></td><td>8.838 B</td><td><span class="mono">319d8bccdaab14953fd22689e9d867af65f52e4612c576a53e4fd06daf77fb29</span></td></tr>
          <tr><td><code>output/week2_foundations_20261103/data_audit.json</code></td><td>25.421 B</td><td><span class="mono">e50d8fc069ef6930f126d8c1b3ec7d32fc9ac320aba62b67daafdea3ab107f36</span></td></tr>
          <tr><td><code>output/week2_foundations_20261103/model_audit.json</code></td><td>18.493 B</td><td><span class="mono">40fb586b1231207fcccea5d0656f1ccf1e5b75759bc232f82eedd4d26bc9fbce</span></td></tr>
          <tr><td><code>scripts/week2_data_audit.py</code></td><td>16.019 B</td><td><span class="mono">12110e9b70745fcad13b85c8dc4ab38fed3fe8433ac2375a821c0718c2c94ac2</span></td></tr>
          <tr><td><code>scripts/week2_model_audit.py</code></td><td>15.490 B</td><td><span class="mono">66eb8d8c206d2c72079893a0abf2e66110e6f5a6df4ea1c9cce5f44312d0c81e</span></td></tr>
          <tr><td><code>scripts/week2_verify_delivery.py</code></td><td>32.187 B</td><td><span class="mono">aea037d67e6edb4173038c468cdaa6be1635f98b7251ea5ca5203b4c8c8b56ed</span></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>
"""
