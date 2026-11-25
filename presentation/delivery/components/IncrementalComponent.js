/* ECDR Component: IncrementalComponent.js
 * Incremental Compute (M03): Bit-Exact Verification vs Latency Paradox on CPU
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card incremental-component">
        <div class="component-header">
          <span class="component-badge">TÍNH TOÁN GIA TĂNG (M03)</span>
          <h3 class="component-title">Tái sử dụng tính toán từng phần (M03): Khớp bit 100% vs Nghịch lý độ trễ</h3>
          <p class="component-desc">Kiểm chứng thực nghiệm tính bất biến của trường tiếp nhận tích chập và đo đạc chi phí quản lý bộ nhớ trên CPU.</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Thành phần thực thi M03</th>
                    <th scope="col">Độ trễ đo được</th>
                    <th scope="col">Tỷ trọng %</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td>Cắt lát tensor (slicing & indexing)</td><td>0.482 ms</td><td>25.0%</td></tr>
                  <tr><td>Cấp phát & dịch chuyển đệm (buffer shift)</td><td>0.854 ms</td><td>44.3%</td></tr>
                  <tr><td>Tích chập 64 mẫu mới (conv forward 64)</td><td>0.591 ms</td><td>30.7%</td></tr>
                  <tr style="background:#f8fafc; font-weight:700;">
                    <td>Tổng thời gian thực thi M03</td>
                    <td>1.927 ms</td>
                    <td>100.0%</td>
                  </tr>
                  <tr style="background:#eff6ff;">
                    <td>Baseline Forward toàn phần (M00)</td>
                    <td>0.342 ms</td>
                    <td>1.00× (Nhanh hơn 5.63×)</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div class="formula-box">
              <span class="formula-tag">NGHỊCH LÝ ĐỘ TRỄ TÁI SỬ DỤNG M03</span>
              $$T_{\\text{inc}} = T_{\\text{slice}} + T_{\\text{shift}} + T_{\\text{conv}} = 1.927\\text{ ms} \\gg T_{\\text{full}} (0.342\\text{ ms})$$
              <span class="formula-note">Chi phí cắt lát tensor và quản lý bộ nhớ khiến M03 chậm hơn 5.63×</span>
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[+] KHẲNG ĐỊNH KHOA HỌC:</h4>
              <ul class="points">
                <li><strong>Khớp bit 100% (Bit-Exact):</strong> Feature map tầng Conv3 trùng khớp tuyệt đối trên toàn bộ 5,034 cặp kề cận (sai số max = 0.0, D_all = 0).</li>
                <li>Xác thực trọn vẹn giả thuyết H13 về trường tiếp nhận bất biến mà không cần huấn luyện lại mạng.</li>
              </ul>
            </div>
            <div class="verdict-bad">
              <h4>[-] NGHỊCH LÝ ĐỘ TRỄ CPU:</h4>
              <ul class="points">
                <li>PyTorch C++ thực thi forward một khối tensor liền mạch trong L1/L2 cache cực nhanh.</li>
                <li>Thao tác chia cắt tensor trong Python sinh ra chi phí cấp phát đối tượng và copy bộ nhớ lấn át hoàn toàn lượng FLOPs tiết kiệm được.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.IncrementalComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
