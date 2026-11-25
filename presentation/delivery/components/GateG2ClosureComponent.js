/* ECDR Component: GateG2ClosureComponent.js
 * Post-W4 Gate G2 K=12 Full Closure Study: Span-Anchored Extraction & Class-Balanced Focal Loss
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card gate-g2-component">
        <div class="component-header">
          <span class="component-badge">ĐỘT PHÁ POST-W4 (3)</span>
          <h3 class="component-title">Đột phá Cổng Gate G2: Khép lại 18 rào cản K=12, Macro-F1 đạt 0.8537</h3>
          <p class="component-desc">Giải quyết triệt để vấn đề lệch lưới chuyển thế tư thế bằng chính sách neo vùng và Focal Loss cân bằng lớp.</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Chỉ số đánh giá</th>
                    <th scope="col">Tham chiếu Tuần 1 (Baseline)</th>
                    <th scope="col">Sau nâng cấp Gate G2 (Post-W4)</th>
                    <th scope="col">Chuyển biến chất lượng</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style="background:#f0fdf4;">
                    <td>Số rào cản dữ liệu &amp; mô hình</td>
                    <td>18 rào cản bị giữ (HOLD)</td>
                    <td><strong>0 rào cản (Zero Blockers)</strong></td>
                    <td><span class="badge badge-success">HOÀN TOÀN ĐẠT</span></td>
                  </tr>
                  <tr>
                    <td>Macro-F1 tập kiểm thử (K=12)</td>
                    <td>0.8018 ± 0.0045</td>
                    <td><strong>0.8537 ± 0.0020</strong></td>
                    <td>+5.19% (Vượt xa mốc 0.75)</td>
                  </tr>
                  <tr style="background:#ecfdf5; font-weight:700;">
                    <td>Macro-F1 6 lớp chuyển thế</td>
                    <td>0.6822 ± 0.0091</td>
                    <td><strong>0.7917 ± 0.0024</strong></td>
                    <td><strong>+10.95% (Cực kỳ rõ rệt)</strong></td>
                  </tr>
                  <tr>
                    <td>Độ chính xác toàn phần (Accuracy)</td>
                    <td>89.85% ± 0.22%</td>
                    <td>87.92% ± 0.19%</td>
                    <td>Cân bằng tối ưu giữa các lớp</td>
                  </tr>
                  <tr>
                    <td>Hỗ trợ cửa sổ kiểm thử</td>
                    <td>3,572 cửa sổ</td>
                    <td>4,544 cửa sổ (+27.2%)</td>
                    <td>Bao phủ trọn vẹn chuyển thế</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[+] KHÉP LẠI TOÀN BỘ 18 RÀO CẢN:</h4>
              <ul class="points">
                <li>Chính sách Span-Anchored trích xuất 9 cửa sổ trên mỗi vùng chuyển thế, đảm bảo &ge; 54 cửa sổ/lớp ở val và &ge; 162 ở test across cả 12 lớp.</li>
                <li>Hàm Focal Loss (&gamma; = 1.5, &beta; = 0.35) giải quyết triệt để mất cân bằng dữ liệu, bảo toàn nguyên vẹn 100% kiến trúc SmallCNN1D gốc.</li>
              </ul>
            </div>
            <div class="verdict-bad">
              <h4>[-] LƯU Ý BẢO TRÌ:</h4>
              <ul class="points">
                <li>Cần lưu trữ metadata phân đoạn thời gian chuyển thế trong quá trình gán nhãn tập huấn luyện.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.GateG2ClosureComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
