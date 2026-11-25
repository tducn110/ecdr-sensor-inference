/* ECDR Component: ScalingComponent.js
 * Group 6: Break-Even Crossover Analysis & Architectural Scaling Study
 * Rigorous Separation of Result A (Theoretical 26.1 MMACs Crossover) vs Result B (Stream Speedup: Unverified)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card scaling-component">
        <div class="component-header">
          <span class="component-badge">NHÓM 6: PHÂN TÍCH CROSSOVER HÒA VỐN</span>
          <h3 class="component-title">Group 6: Phân tích Crossover Hòa vốn &amp; Mở rộng Quy mô Kiến trúc</h3>
          <p class="component-desc">Khảo sát dải quy mô từ 0.28M đến 5.0M tham số trên CPU đơn luồng, phân định rành mạch giữa Kết quả A (Ngưỡng cắt lý thuyết) &amp; Kết quả B (Tăng tốc thực tế trên luồng).</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Kiến trúc mạng</th>
                    <th scope="col">Tham số</th>
                    <th scope="col">MACs</th>
                    <th scope="col">Forward (T<sub>f</sub>)</th>
                    <th scope="col">Trần lý thuyết (R=1)</th>
                    <th scope="col">Ngưỡng r<sub>min</sub></th>
                    <th scope="col">Thực tế nếu R=0 (S=64)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>SmallCNN1D (Baseline)</td>
                    <td>0.28 M</td>
                    <td>1.17 M</td>
                    <td>0.14 ms</td>
                    <td>0.11× (Lỗ)</td>
                    <td>&gt; 100% (Bất khả thi)</td>
                    <td><span class="badge badge-danger">1.48 ms (Chậm hơn 10.5×)</span></td>
                  </tr>
                  <tr>
                    <td>MediumCNN1D</td>
                    <td>2.23 M</td>
                    <td>8.50 M</td>
                    <td>0.38 ms</td>
                    <td>0.28× (Lỗ)</td>
                    <td>&gt; 100% (Bất khả thi)</td>
                    <td><span class="badge badge-danger">1.71 ms (Chậm hơn 4.5×)</span></td>
                  </tr>
                  <tr style="background:#f0fdf4;">
                    <td>TemporalResNet1D</td>
                    <td>2.70 M</td>
                    <td>38.2 M</td>
                    <td>1.99 ms</td>
                    <td><strong>1.49× (Lãi lý thuyết)</strong></td>
                    <td>r &gt; 67.1%</td>
                    <td><span class="badge badge-danger">3.33 ms (Chậm hơn 1.67×)</span></td>
                  </tr>
                  <tr style="background:#ecfdf5; font-weight:700;">
                    <td>TemporalTransformer1D</td>
                    <td>4.99 M</td>
                    <td>115 M</td>
                    <td>3.87 ms</td>
                    <td><strong>2.90× (Trần lý thuyết)</strong></td>
                    <td><strong>r &gt; 34.5%</strong></td>
                    <td><span class="badge badge-danger">5.20 ms (Chậm hơn 34.5%)</span></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="formula-box">
              <span class="formula-tag">CROSSOVER TÍNH TOÁN LÝ THUYẾT (KẾT QUẢ A)</span>
              $$T_{\\text{ECDR}} = T_v + (1-r)T_{\\text{full}} + r T_{\\text{hit}} \\implies r > r_{\\min} = \\frac{T_v}{T_{\\text{full}}} \\approx 34.49\\% \\quad$$
              <span class="formula-note">Điểm giao nhau lý thuyết ≈ 26.1 MMACs; Tốc độ trần 2.90× trên TemporalTransformer1D</span>
            </div>

            <div style="font-size:12px; color:var(--text-muted); margin-top:8px; line-height:1.5;">
              * <em>Lưu ý phương pháp luận:</em><br>
              &bull; T<sub>f</sub> là thời gian chạy thuần túy <code>model.forward()</code> CPU đơn luồng batch=1 (0.14 ms cho SmallCNN) trên tensor ngẫu nhiên <code>torch.randn(1, 6, 128)</code>. Trong Bảng VI, thời gian M00 là 0.342 ms bao gồm trọn gói luồng streaming query (chuẩn hóa, wrappers, telemetry).<br>
              &bull; Số MACs là giá trị ước lượng giải tích trong <code>estimate_macs()</code>, chưa phải số đếm từ profiler đồ thị tính toán.<br>
              &bull; Điểm hòa vốn 26.1 MMACs là kết quả nội suy theo mô hình chi phí hiện tại, không phải bảo đảm phổ quát rằng mọi mô hình lớn đều tự động có speedup.
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[A] KẾT QUẢ A: COMPUTATIONAL CROSSOVER (ĐÃ CHỨNG MINH LÝ THUYẾT):</h4>
              <ul class="points">
                <li>Quy mô tính toán của mô hình chuỗi thời gian càng lớn thì chi phí suy luận càng áp đảo chi phí kiểm chứng giả định (T<sub>v</sub> = 1.33 ms).</li>
                <li>Ngưỡng hòa vốn tính toán lý thuyết nằm tại <strong>≈ 26.1 MMACs (2.51M tham số)</strong> cho verifier caching và <strong>≈ 37.0 MMACs (2.68M tham số)</strong> cho incremental recomputation.</li>
                <li>Tại 115 MMACs (TemporalTransformer1D 4.99M tham số), tỷ số tốc độ trần đạt <strong>2.90×</strong> nếu giả định tỷ lệ tái sử dụng hoàn hảo (r = 1).</li>
              </ul>
            </div>

            <div class="verdict-bad">
              <h4>[B] KẾT QUẢ B: ACTUAL STREAM SPEEDUP (CHƯA XÁC MINH - NOT ESTABLISHED):</h4>
              <ul class="points">
                <li><strong>Chi phí verifier chưa đo trực tiếp:</strong> T<sub>v</sub> = 1.33 ms mượn từ SmallCNN1D; cơ chế attention verifier và KV-cache của Transformer chưa có phép đo thực tế trên phần cứng.</li>
                <li><strong>Rào cản tỷ lệ tái sử dụng:</strong> Trên luồng HAPT Stride 64, các phương pháp hình học đều có R = 0%. Khi đó T<sub>ECDR</sub> = 1.33 + 3.87 = <strong>5.20 ms</strong> (làm chậm hơn 34.5% so với suy luận thẳng).</li>
                <li><strong>Điều kiện tiên quyết:</strong> Tăng tốc thực tế chỉ xảy ra khi chứng minh được tỷ lệ tái sử dụng trên luồng stream đạt <strong>R &gt; 34.5%</strong>.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.ScalingComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
