/* ECDR Component: ConclusionComponent.js
 * Group 7 / Conclusion: 3 Key Research Findings & Next Agenda
 * Authentic Boundary Conditions & The Open Architectural Question
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card conclusion-component">
        <div class="component-header">
          <span class="component-badge">TỔNG KẾT &amp; CHƯƠNG TRÌNH TIẾP THEO</span>
          <h3 class="component-title">Group 7 / Tổng kết: 3 Phát hiện then chốt &amp; Câu hỏi Nghiên cứu Mở</h3>
          <p class="component-desc">Định hình ranh giới chân thực của ECDR và xác lập chương trình kiểm chứng cho các kiến trúc chuỗi thời gian quy mô lớn.</p>
        </div>

        <div class="columns">
          <div>
            <div style="background:var(--card-bg); border:1px solid var(--line); border-radius:8px; padding:18px; margin-bottom:16px;">
              <h4 style="margin:0 0 12px; font-size:14px; text-transform:uppercase; color:var(--brand-navy); letter-spacing:0.5px;">
                3 PHÁT HIỆN THỰC NGHIỆM THEN CHỐT:
              </h4>
              <ol style="margin:0; padding-left:18px; font-size:13px; line-height:1.7; color:var(--ink-secondary);">
                <li style="margin-bottom:10px;">
                  <strong>Tái sử dụng tính toán gia tăng (Incremental Compute):</strong> Có thể tái sử dụng đặc trưng trung gian và bảo toàn 100% quyết định của mô hình tham chiếu (khớp bit tuyệt đối), nhưng chi phí quản lý bộ nhớ, cấp phát và cắt lát tensor khiến nó <strong>chậm hơn 5.6×</strong> so với suy luận toàn phần trên CPU. Cắt giảm phép tính lý thuyết không đồng nghĩa với giảm độ trễ thực tế.
                </li>
                <li style="margin-bottom:10px;">
                  <strong>Tái sử dụng theo chứng nhận quyết định (Certified Decision Reuse):</strong> Tuyệt đối an toàn nhờ cơ chế fail-closed fallback về suy luận gốc, nhưng toàn bộ các phương pháp bao đóng hình học tĩnh đều thất bại trong việc tạo ra tỷ lệ tái sử dụng hữu ích trên luồng cảm biến thực tế (<strong>R = 0%</strong>).
                </li>
                <li>
                  <strong>Tính kinh tế và Mở rộng quy mô (Economic Scaling &amp; Crossover):</strong> Các kết quả âm trên SmallCNN1D không phải là thất bại của nguyên lý ECDR, mà đã xác lập ranh giới kinh tế chân thực: Tái sử dụng chỉ có hiệu quả khi mô hình đủ lớn và chi phí kiểm chứng tương đối rẻ. Ngưỡng hòa vốn lý thuyết bắt đầu từ <strong>≈ 26.1 MMACs</strong>.
                </li>
              </ol>
            </div>

            <div class="formula-box">
              <span class="formula-tag">CÂU HỎI NGHIÊN CỨU MỞ CHO CÁC MÔ HÌNH LỚN</span>
              $$\\textbf{Soundness} \\;\\wedge\\; \\textbf{Useful Coverage } (R > 34.49\\% ) \\;\\wedge\\; \\textbf{Net Runtime Speedup}?$$
              <span class="formula-note">Liệu một kiến trúc chuỗi thời gian lớn có thể đồng thời đạt được cả 3 tiêu chuẩn cốt lõi?</span>
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[+] Ý NGHĨA KHOA HỌC CỦA CÁC KẾT QUẢ ÂM:</h4>
              <ul class="points">
                <li>Các kết quả âm định lượng trong Tuần 4 mang lại giá trị học thuật cao: Bác bỏ những giả định ngây thơ về việc áp dụng bao đóng hình học tĩnh lên tín hiệu cảm biến động học và chứng minh thực nghiệm nghịch lý độ trễ bộ nhớ CPU.</li>
                <li>Hệ thống vạch rõ ranh giới điều kiện biên chân thực (authentic boundaries) để công nghệ ECDR có thể vận hành khả thi trong thực tiễn.</li>
              </ul>
            </div>

            <div class="verdict-bad">
              <h4>[-] 3 BƯỚC HÀNH ĐỘNG TRONG CHƯƠNG TRÌNH TIẾP THEO:</h4>
              <ul class="points">
                <li><strong>1. Đo đạc trực tiếp trên Transformer:</strong> Chấm dứt việc dùng proxy chi phí của CNN; đo đạc trực tiếp cơ chế kiểm chứng (attention verifier) và KV-cache trên phần cứng mục tiêu.</li>
                <li><strong>2. Đổi mới phương pháp chứng nhận miền:</strong> Chuyển từ bao đóng tĩnh không gian ℓ<sub>p</sub> sang mô hình hóa dịch pha tần số và đa tạp động học chuyển động.</li>
                <li><strong>3. Kiểm chứng tăng tốc thực tế (Result B):</strong> Replay toàn bộ luồng HAPT trên mô hình lớn để chứng minh ECDR đạt tỷ lệ tái sử dụng R &gt; 34.5% kèm tăng tốc end-to-end.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.ConclusionComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
