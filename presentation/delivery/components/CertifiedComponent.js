/* ECDR Component: CertifiedComponent.js
 * Group 3: Certified Decision Reuse (M05–M13)
 * Evaluation of Lipschitz, IBP, CROWN: 100% Reference Agreement vs Zero Useful Coverage (R=0)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card certified-component">
        <div class="component-header">
          <span class="component-badge">NHÓM 3: CHỨNG NHẬN MIỀN QUYẾT ĐỊNH</span>
          <h3 class="component-title">Group 3: Tái sử dụng theo chứng nhận quyết định (Certified Decision Reuse)</h3>
          <p class="component-desc">Đánh giá các kỹ thuật bao đóng hình học (Lipschitz, IBP, CROWN) và phân định ranh giới giữa Reference Agreement (Độ khớp tham chiếu) vs Soundness (Tính an toàn trên luồng).</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Phương pháp chứng nhận</th>
                    <th scope="col">Cơ chế hình học</th>
                    <th scope="col">Tỷ lệ tái sử dụng (R)</th>
                    <th scope="col">Độ khớp (A<sub>full</sub>)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>M05 (Lipschitz Ball)</td>
                    <td>Quả cầu chuẩn ℓ<sub>2</sub> với hằng số phổ L<sub>cj</sub></td>
                    <td><strong>0.00% (0 / 5,034)</strong></td>
                    <td>100.00% (Fallback)</td>
                  </tr>
                  <tr>
                    <td>M06 (IBP Box)</td>
                    <td>Hộp siêu chữ nhật truyền khoảng biến thiên</td>
                    <td><strong>0.00% (0 / 5,034)</strong></td>
                    <td>100.00% (Fallback)</td>
                  </tr>
                  <tr>
                    <td>M07 (CROWN Polytope)</td>
                    <td>Đa diện xấp xỉ tuyến tính trên/dưới ReLU</td>
                    <td><strong>0.00% (0 / 5,034)</strong></td>
                    <td>100.00% (Fallback)</td>
                  </tr>
                  <tr>
                    <td>M09 (Wavelet DWT)</td>
                    <td>Bao đóng năng lượng dải tần số sóng con</td>
                    <td><strong>0.00% (0 / 5,034)</strong></td>
                    <td>100.00% (Fallback)</td>
                  </tr>
                  <tr>
                    <td>M11 (PCA Subspace)</td>
                    <td>Chiếu khoảng cách lên k thành phần chính</td>
                    <td><strong>0.00% (0 / 5,034)</strong></td>
                    <td>100.00% (Fallback)</td>
                  </tr>
                  <tr style="background:#fef2f2; font-weight:700;">
                    <td>Tổng kết Nhóm Chứng nhận</td>
                    <td>Fail-Closed Arbiter (Toàn bộ M05–M13)</td>
                    <td><strong>R = 0.00%</strong></td>
                    <td><strong>100.00% (D_all = 0)</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="formula-box">
              <span class="formula-tag">RANH GIỚI THAM CHIẾU VS CHỨNG NHẬN TRÊN LUỒNG</span>
              $$A_{\\text{full}} = 1.0000 \\quad \\wedge \\quad R_{\\text{cert}} = 0.00\\% \\implies \\text{Fallback } M00$$
              <span class="formula-note">100% quyết định khớp tham chiếu đến từ fail-closed fallback về suy luận toàn phần</span>
            </div>

            <div style="font-size:12px; color:var(--text-muted); margin-top:8px; line-height:1.5;">
              * <em>Lưu ý phương pháp luận cốt lõi:</em><br>
              &bull; <strong>Reference Agreement = 100%:</strong> Đạt được hoàn toàn là nhờ cơ chế fail-closed fallback về M00 khi verifier không chứng minh được tính an toàn.<br>
              &bull; <strong>Useful Reuse Coverage = 0%:</strong> Không có bất kỳ cửa sổ nào được skip tính toán thông qua chứng nhận miền trong toàn bộ chuỗi kiểm thử HAPT.
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[+] KIẾN TRÚC FAIL-CLOSED BẢO VỆ TUYỆT ĐỐI:</h4>
              <ul class="points">
                <li>Bộ trọng tài (arbiter) vận hành chính xác theo nguyên tắc an toàn: Khi cận chứng nhận không đủ hẹp để bảo đảm bất biến quyết định, hệ thống từ chối tái sử dụng và chuyển ngay về suy luận toàn phần M00.</li>
                <li>Bảo toàn 100% quyết định của mô hình tham chiếu (D<sub>all</sub> = 0 trên 5,034 cặp), không tạo ra bất kỳ sai lệch nào đối với hệ thống bên ngoài.</li>
              </ul>
            </div>

            <div class="verdict-bad">
              <h4>[-] KẾT QUẢ ÂM: KHÔNG CÓ LẦN TÁI SỬ DỤNG HỮU ÍCH:</h4>
              <ul class="points">
                <li>Tất cả các phương pháp bao đóng hình học tĩnh (Lipschitz, IBP, CROWN) đều thất bại trong việc tạo ra tỷ lệ tái sử dụng hữu ích (R = 0%).</li>
                <li>Không tiết kiệm được phép tính nào, trong khi pipeline phải gánh thêm chi phí chạy verifier (T<sub>v</sub> = 1.33 ms), khiến hệ thống chạy chậm hơn suy luận gốc.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.CertifiedComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
