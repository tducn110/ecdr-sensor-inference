/* ECDR Component: DomainCollapseComponent.js
 * Domain Certification Collapse (M05–M13) & The 1,562.7x Paired Gap Ratio
 * Detailed Analysis of 4 Collapse Mechanisms (Lipschitz, IBP, Wavelet, PCA)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card domain-collapse-component">
        <div class="component-header">
          <span class="component-badge">NHÓM 4: NGUYÊN NHÂN SỤP ĐỔ MIỀN CHỨNG NHẬN</span>
          <h3 class="component-title">Group 4: Vì sao các phương pháp chứng nhận thất bại? (Root Cause Analysis)</h3>
          <p class="component-desc">Phân tích khe hở vật lý 1,562.7× và 4 cơ chế sụp đổ độc lập của các phương pháp bao đóng hình học tĩnh ở bước trượt S=64.</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Đại lượng đo đạc (N = 1,000 cặp)</th>
                    <th scope="col">Giá trị thực nghiệm</th>
                    <th scope="col">Định nghĩa toán học &amp; Ý nghĩa vật lý</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Dịch chuyển khoảng cách ℓ<sub>2</sub> (d<sub>i</sub>)</td>
                    <td><strong>median 25.49</strong> (mean 24.26)</td>
                    <td>Độ lệch vector cảm biến giữa hai cửa sổ kề cận</td>
                  </tr>
                  <tr>
                    <td>Biên độ logit mô hình (margin)</td>
                    <td>median 6.10 (mean 5.74)</td>
                    <td>$\\Delta_j(x) = f_c(x) - f_j(x)$ (khoảng cách lớp thắng vs lớp khác)</td>
                  </tr>
                  <tr style="background:#fef2f2;">
                    <td>Bán kính Lipschitz an toàn (r<sub>i</sub>)</td>
                    <td><strong>median 0.01487</strong> (mean 0.01448)</td>
                    <td>$r_i = \\min_{j \\neq c} \\frac{f_c(x) - f_j(x)}{L_{cj}}$</td>
                  </tr>
                  <tr style="background:#fee2e2; font-weight:700;">
                    <td>Trung vị tỷ số khe hở: median(d<sub>i</sub> / r<sub>i</sub>)</td>
                    <td><strong>1,562.7×</strong> (mean 50,990.3×)</td>
                    <td><strong>Trung vị tỷ số tính trên từng cặp</strong> (không phải tỷ số 2 trung vị)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="formula-box">
              <span class="formula-tag">KHE HỞ ĐỘ DỊCH PHA VẬN ĐỘNG TRÊN TỪNG CẶP</span>
              $$\\operatorname{median}\\left(\\frac{d_i}{r_i}\\right) = 1,562.7\\times \\quad \\left[\\frac{\\operatorname{median}(d_i)}{\\operatorname{median}(r_i)} = \\frac{25.4923}{0.01487} \\approx 1,714.3\\times \\neq 1,562.7\\times\\right]$$
              <span class="formula-note">Trung vị tỷ số tính trên từng cặp kề cận (không phải tỷ số giữa 2 trung vị)</span>
            </div>

            <div style="font-size:12px; color:var(--text-muted); margin-top:8px; line-height:1.5;">
              * <em>Giải thích toán học:</em> Bán kính an toàn r_i bị co hẹp cực mạnh do tích chuẩn phổ của mạng tích chập sâu. Đồng thời, bước trượt 64 mẫu (1.28s) tạo ra dịch pha tuần hoàn bước chân người, khiến d<sub>i</sub> vượt xa r_i ở mọi cặp kiểm thử, dẫn đến R = 0.
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[!] 4 CƠ CHẾ SỤP ĐỔ ĐỘC LẬP THEO TỪNG PHƯƠNG PHÁP:</h4>
              <ul class="points">
                <li><strong>M05 (Lipschitz Ball):</strong> Khoảng cách dịch chuyển thực tế d<sub>i</sub> lớn gấp <strong>1,562.7×</strong> bán kính an toàn r<sub>i</sub> do cận phổ tích lũy qua 3 tầng Conv1D.</li>
                <li><strong>M06/M07 (IBP &amp; CROWN):</strong> Lan truyền khoảng biến thiên qua các hàm kích hoạt ReLU liên tiếp gây bùng nổ cận (interval bound explosion), làm hộp bao quá rộng để khẳng định bất biến logit.</li>
                <li><strong>M09 (Wavelet DWT):</strong> Cửa sổ trượt thiếu phần đuôi tương lai chưa quan sát (unobserved future tail), làm sai lệch hệ số năng lượng tần số.</li>
                <li><strong>M11 (PCA Subspace):</strong> Trục phương sai lớn nhất của tín hiệu cảm biến không trùng khớp với pháp vector của mặt phân chia lớp (margin normal misalignment).</li>
              </ul>
            </div>

            <div class="verdict-bad">
              <h4>[-] BÀI HỌC CỐT LÕI VỀ MIỀN DỮ LIỆU:</h4>
              <ul class="points">
                <li>Bao đóng hình học tĩnh giả định các biến thiên nhỏ xung quanh một điểm dừng tĩnh (stationary perturbation).</li>
                <li>Ngược lại, tín hiệu cảm biến con người là quá trình động học tuần hoàn có dịch pha thời gian liên tục; bước trượt S=64 là quá thô so với bán kính bất biến của mô hình.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.DomainCollapseComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
