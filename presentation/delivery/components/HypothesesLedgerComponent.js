/* ECDR Component: HypothesesLedgerComponent.js
 * 24-Hypothesis Final Outcomes Ledger (H01–H24)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    const hypotheses = [
      ['H01', 'Hình học thời gian', 'VERIFIED_FINITE', 'Chồng lấn căn chỉnh pha là chính xác giữa 2 cửa sổ kề nhau', '5,034 cặp kề cận khớp bit 100% trên 12 kênh Conv1D.'],
      ['H02', 'Hình học thời gian', 'FALSIFIED_UNCONDITIONAL', '50% trùng lặp suy ra khoảng cách tọa độ nhỏ', 'Nhiễu cảm biến tạo delta L2 rất lớn dù trùng lặp 50% thời gian.'],
      ['H03', 'Ngữ nghĩa nhãn', 'FALSIFIED_UNCONDITIONAL', 'Cùng nhãn hành vi suy ra cùng dự đoán CNN', 'Hàng 682->683 nhãn 11 (DOWNSTAIRS) nhưng model đổi dự đoán từ 10->8.'],
      ['H04', 'Nền tảng toán học', 'DERIVED', 'Biên dương nghiêm ngặt tạo ra vùng quyết định bất biến cục bộ', 'Định lý tô-pô đảm bảo lân cận mở Delta > 0, nhưng bán kính rất bé.'],
      ['H05', 'Nền tảng toán học', 'DERIVED_REAL_ARITHMETIC', 'Biên dương trên toàn miền là điều kiện đủ bảo toàn quyết định', 'Định lý soundness đúng dưới số thực; float32 cần cầu nối số học.'],
      ['H06', 'Chứng thực miền', 'CONDITIONAL_THEOREM', 'Chặn Lipschitz toàn cục xác lập quả cầu chứng thực', 'Bán kính r = Delta / (2 * L_global). Đạt 0% trúng trên stream liên tục.'],
      ['H07', 'Chặn khoảng IBP', 'FALSIFIED_COMPLETENESS', 'IBP chứng thực được mọi miền bất biến', 'IBP bị nới lỏng nghiêm trọng qua 3 tầng Conv1D và lớp Flatten.'],
      ['H08', 'Công thức biên', 'DERIVED', 'Biên trực tiếp 11 cặp lớp triệt tiêu mất mát phụ thuộc', 'Công thức trực tiếp c_i = e_y - e_i chặt chẽ hơn chặn rời rạc từng logit.'],
      ['H09', 'Phương pháp nới lỏng', 'NOT_ESTABLISHED', 'CROWN luôn rẻ hơn và chặt hơn các phương pháp khác', 'CROWN chặt hơn nhưng tốn 12-14ms build cost vs 0.65ms của IBP.'],
      ['H10', 'Hình học miền', 'NOT_ESTABLISHED', 'Các hướng PCA/thời gian trùng với pháp tuyến biên quyết định', 'Thành phần chính PCA không trùng với siêu phẳng phi tuyến của CNN.'],
      ['H11', 'Hình học miền', 'NOT_ESTABLISHED', 'Miền cấu trúc cho độ phủ tốt hơn hộp đường chéo', 'Cả đa diện PCA và hộp đường chéo đều đạt 0% tái sử dụng trên test stream.'],
      ['H12', 'Nhúng sóng', 'DERIVED', 'Chồng lấn cố định giảm chiều tự do từ 768 xuống 384', 'Stride 64 cố định chính xác 64 mẫu (384 tọa độ) giữa 2 cửa sổ cùng file.'],
      ['H13', 'Tính toán gia tăng', 'VERIFIED_FINITE', 'Tái sử dụng đặc trưng trung gian bảo toàn kết quả bit-for-bit', 'M03 đạt 100% khớp bit (5,034 cặp kề cận, D_all = 0).'],
      ['H14', 'Tính toán gia tăng', 'NOT_APPLICABLE', 'Đẳng thức bù trừ trượt GAP (new-minus-dropped) áp dụng cho đầu phân lớp hiện tại', 'Đẳng thức O(1) chỉ áp dụng cho tầng GAP; SmallCNN1D dùng Flatten(2048)->Linear(128) nhạy vị trí nên không áp dụng được.'],
      ['H15', 'Vòng đời & Trạng thái', 'FALSIFIED_UNCONDITIONAL', 'Tái định tâm sau khi trúng tái sử dụng không cần chứng minh mới', 'Tái định tâm không có chặn mới dẫn đến trôi dạt và mất tính an toàn.'],
      ['H16', 'Vòng đời & Trạng thái', 'CONDITIONAL_THEOREM', 'Cố định anchor bảo toàn tính đúng đắn lặp lại', 'Tính bất biến của anchor đảm bảo nhãn đúng trong suốt vòng đời.'],
      ['H17', 'Vòng đời & Trạng thái', 'NOT_ESTABLISHED', 'Tập đến được đa bước chân trời thời gian bảo toàn tính bất biến', 'Tập đến được ở H >= 2 giãn nở quá nhanh, đạt 0% tái sử dụng.'],
      ['H18', 'Độ phức tạp & Chi phí', 'DERIVED', 'Độ phức tạp kiểm tra hộp trong trường hợp xấu nhất là Theta(m)', 'Cần đọc toàn bộ m tọa độ để khẳng định nằm trong hộp; xác nhận qua đo đạc.'],
      ['H19', 'Mở rộng thực nghiệm', 'FALSIFIED_UNCONDITIONAL', 'Tính đúng đắn toán học suy ra độ phủ hữu ích trên stream', 'M05-M13 có thuật toán toán học đúng nhưng đạt R = 0% trên test stream.'],
      ['H20', 'Mở rộng thực nghiệm', 'FALSIFIED_UNCONDITIONAL', 'Độ phủ hữu ích suy ra tăng tốc thời gian chạy ròng', 'M03 đạt 100% độ phủ nhưng chậm hơn 5.63x do overhead thao tác tensor.'],
      ['H21', 'So sánh hệ thống', 'NOT_ESTABLISHED', 'ECDR vượt trội hơn CNN baseline hoặc LipCache trên HAPT stream', 'M00 là nhanh nhất (0.34ms). Các phương pháp vùng chậm hơn do T_v.'],
      ['H22', 'Cầu nối số học', 'FALSIFIED_UNCONDITIONAL', 'Chứng minh số thực tự động chuyển giao an toàn sang Float32', 'Cần outward rounding và Float64; chạy Float32 thuần bị vi phạm 1-ULP.'],
      ['H23', 'Ngữ nghĩa biến cố', 'FALSIFIED_UNCONDITIONAL', 'Tên biến cố thời gian đủ để tái sử dụng quyết định an toàn', 'Các cửa sổ cùng tên WALKING vẫn có đặc trưng khác nhau và nhãn khác nhau.'],
      ['H24', 'Tam đề cốt lõi', 'NOT_ESTABLISHED', 'Tồn tại chế độ vận hành đồng thời đúng đắn, hữu ích và tốc độ cao', '0 trong 17 phương pháp thỏa mãn đồng thời cả 3 điều kiện ở Stride 64.']
    ];

    const getBadge = status => {
      if (status === 'VERIFIED_FINITE' || status === 'DERIVED' || status === 'DERIVED_REAL_ARITHMETIC') {
        return `<span class="badge badge-success">${status}</span>`;
      }
      if (status === 'FALSIFIED_UNCONDITIONAL' || status === 'FALSIFIED_COMPLETENESS') {
        return `<span class="badge badge-danger">${status}</span>`;
      }
      if (status === 'CONDITIONAL_THEOREM') {
        return `<span class="badge badge-warning">${status}</span>`;
      }
      return `<span class="badge badge-neutral">${status}</span>`;
    };

    return `
      <div class="component-card hypotheses-ledger-component">
        <div class="component-header">
          <span class="component-badge">SỔ CÁI KHOA HỌC</span>
          <h3 class="component-title">Sổ cái toàn diện 24 giả thuyết khoa học (H01–H24)</h3>
          <p class="component-desc">Tổng hợp phán quyết kiểm định độc lập giữa lý thuyết Tuần 3 và thực nghiệm Tuần 4.</p>
        </div>
        <div class="table-wrap">
          <table class="data-table compact">
            <thead>
              <tr>
                <th scope="col">Mã</th>
                <th scope="col">Chuyên mục</th>
                <th scope="col">Phán quyết Tuần 4</th>
                <th scope="col">Tuyên bố cốt lõi</th>
                <th scope="col">Bằng chứng &amp; Nhân chứng thực nghiệm</th>
              </tr>
            </thead>
            <tbody>
              ${hypotheses.map(h => `
                <tr>
                  <td><strong>${h[0]}</strong></td>
                  <td><span class="badge badge-neutral">${h[1]}</span></td>
                  <td>${getBadge(h[2])}</td>
                  <td>${h[3]}</td>
                  <td>${h[4]}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.HypothesesLedgerComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
