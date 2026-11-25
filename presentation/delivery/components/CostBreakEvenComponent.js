/* ECDR Component: CostBreakEvenComponent.js
 * Group 5: Runtime Economics & The Two Bottlenecks (Coverage & Runtime Bottlenecks)
 * Full Cost Decomposition Model & Evaluation of the H24 Trilemma
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card cost-breakeven-component">
        <div class="component-header">
          <span class="component-badge">NHÓM 5: KINH TẾ HỌC THỜI GIAN CHẠY &amp; HAI ĐIỂM NGHẼN</span>
          <h3 class="component-title">Group 5: Kinh tế học thời gian chạy &amp; Hai điểm nghẽn hệ thống (The Two Bottlenecks)</h3>
          <p class="component-desc">Mô hình hóa chi phí toàn phần, phân biệt độ trễ luồng M00 (0.342 ms) vs forward thô (0.140 ms), và xác định hai điểm nghẽn của ECDR.</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Thành phần chi phí</th>
                    <th scope="col">SmallCNN1D (Mạng nhỏ)</th>
                    <th scope="col">TemporalTransformer (Mạng lớn)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Thời gian forward tham chiếu (T<sub>full</sub>)</td>
                    <td>0.140 ms (Raw CPU) / 0.342 ms (Pipeline)</td>
                    <td>3.870 ms (Raw forward CPU loop)</td>
                  </tr>
                  <tr>
                    <td>Chi phí kiểm tra verifier (T<sub>check</sub>)</td>
                    <td>1.335 ms (Hiệu số M01 - M00)</td>
                    <td>1.335 ms (Giả định proxy từ CNN)</td>
                  </tr>
                  <tr>
                    <td>Ngưỡng hòa vốn tối thiểu (r<sub>min</sub>)</td>
                    <td>&gt; 100% (Toán học không thể hòa vốn)</td>
                    <td><strong>r &gt; 34.49%</strong> (Giả định T<sub>build</sub>=0, T<sub>hit</sub>=0)</td>
                  </tr>
                  <tr style="background:#fef2f2;">
                    <td>Chi phí khi R = 0% (HAPT Stride 64 thực tế)</td>
                    <td>1.475 ms (Chậm hơn 10.5×)</td>
                    <td><strong>5.205 ms (Chậm hơn 34.5%)</strong></td>
                  </tr>
                  <tr style="background:#f0fdf4;">
                    <td>Chi phí giả định minh họa (nếu r = 80%)</td>
                    <td>0.367 ms (Chưa hòa vốn raw)</td>
                    <td><strong>0.774 ms (Tỷ số lý thuyết 5.0×)</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="formula-box">
              <span class="formula-tag">MÔ HÌNH HÒA VỐN TOÀN PHẦN ECDR</span>
              $$T_{\\text{ECDR}} = T_{\\text{check}} + (1-r)T_{\\text{full}} + r T_{\\text{hit}} + T_{\\text{build}} \\implies r > r_{\\min} = \\frac{T_{\\text{check}} + T_{\\text{build}}}{T_{\\text{full}} - T_{\\text{hit}}} \\approx 34.49\\% \\quad$$
              <span class="formula-note">Ngưỡng tỷ lệ tái sử dụng tối thiểu để ECDR có lãi thời gian chạy</span>
            </div>

            <div style="font-size:12px; color:var(--text-muted); margin-top:8px; line-height:1.5;">
              * <em>Phân biệt hai chỉ số quan trọng:</em><br>
              &bull; <strong>Reference Agreement (100.0%):</strong> Quyết định của ECDR khớp hoàn toàn với SmallCNN1D nhờ fail-closed fallback.<br>
              &bull; <strong>Classification Accuracy (89.05%):</strong> Tỷ lệ SmallCNN1D đoán đúng nhãn thực tế hoạt động người dùng trên tập HAPT.
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[!] HAI ĐIỂM NGHẼN HỆ THỐNG CỐT LÕI (THE TWO BOTTLENECKS):</h4>
              <ul class="points">
                <li><strong>1. Điểm nghẽn Tỷ lệ Tái sử dụng (Coverage Bottleneck):</strong> Trên luồng cảm biến vật lý ở bước trượt thô S=64, độ dịch pha động học khiến các phương pháp bao đóng hình học đều cho R = 0%. Hệ thống không thể tạo ra lần skip nào.</li>
                <li><strong>2. Điểm nghẽn Thời gian chạy (Runtime Bottleneck):</strong> Trên mô hình nhỏ, thời gian chạy forward quá ngắn (0.14–0.34 ms). Chi phí biên của verifier (1.33 ms) và quản lý bộ nhớ (1.93 ms) áp đảo toàn bộ ngân sách tính toán, khiến hệ thống luôn lỗ thời gian.</li>
              </ul>
            </div>

            <div class="verdict-bad">
              <h4>[-] KẾT LUẬN VỀ TAM ĐỀ H24 TRÊN SMALLCNN1D:</h4>
              <ul class="points">
                <li><strong>Không thể thỏa mãn đồng thời:</strong> Trên SmallCNN1D, 0/17 cấu hình đạt được Tam đề H24 (Safety + Coverage + Cost Reduction).</li>
                <li>Khi giữ Safety = 100%, Coverage sụp đổ về 0%. Và ngay cả khi gượng ép ép có Coverage, chi phí verifier vẫn khiến tổng thời gian lớn hơn full inference.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.CostBreakEvenComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
