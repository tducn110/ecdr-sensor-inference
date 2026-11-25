/* ECDR Component: BaselineComponent.js
 * Group 1: Reference & Streaming Baseline (M00)
 * Motivation: 50% Overlap on HAPT, 3 Core Criteria (Soundness, Coverage, Speedup)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    return `
      <div class="component-card baseline-component">
        <div class="component-header">
          <span class="component-badge">NHÓM 1: ĐƯỜNG CƠ SỞ &amp; BÀI TOÁN NGHIÊN CỨU</span>
          <h3 class="component-title">Group 1: Bài toán nghiên cứu &amp; Đường cơ sở Streaming (Baseline M00)</h3>
          <p class="component-desc">Động lực từ 50% cửa sổ chồng lấn trên tập dữ liệu cảm biến HAPT và 3 tiêu chuẩn cốt lõi: Soundness, Useful Coverage, Net Runtime Speedup.</p>
        </div>

        <div class="columns">
          <div>
            <div class="table-wrap">
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">Đặc tính hệ thống &amp; Thực nghiệm</th>
                    <th scope="col">Giá trị đo đạc</th>
                    <th scope="col">Ý nghĩa phương pháp luận</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Tập dữ liệu kiểm thử (Dataset)</td>
                    <td><strong>HAPT Streaming</strong></td>
                    <td>6 kênh cảm biến (Acc/Gyro), 50 Hz</td>
                  </tr>
                  <tr>
                    <td>Kích thước cửa sổ &amp; Bước trượt</td>
                    <td><strong>W = 128, S = 64</strong></td>
                    <td>Cửa sổ 2.56s, <strong>50% chồng lấn (overlap)</strong></td>
                  </tr>
                  <tr>
                    <td>Khối lượng kiểm thử liên tục</td>
                    <td><strong>5,053 cửa sổ</strong></td>
                    <td>5,034 cặp cửa sổ kề cận liên tiếp</td>
                  </tr>
                  <tr>
                    <td>Mô hình tham chiếu (Frozen Model)</td>
                    <td><strong>SmallCNN1D</strong></td>
                    <td>0.28M tham số, 1.17 MMACs (3 tầng Conv1D)</td>
                  </tr>
                  <tr style="background:#eff6ff;">
                    <td>Độ chính xác phân loại gốc (Accuracy)</td>
                    <td><strong>89.05%</strong></td>
                    <td>Dự đoán đúng nhãn hoạt động con người (Ground Truth)</td>
                  </tr>
                  <tr>
                    <td>Độ trễ luồng stream M00 (Pipeline)</td>
                    <td><strong>0.3420 ms</strong></td>
                    <td>End-to-end (chuẩn hóa tensor, buffer, telemetry)</td>
                  </tr>
                  <tr>
                    <td>Thời gian forward CPU thuần (Raw)</td>
                    <td><strong>0.1402 ms</strong></td>
                    <td>torch.randn(1, 6, 128) CPU đơn luồng batch=1</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="formula-box">
              <span class="formula-tag">MỤC TIÊU TỐI ƯU HÓA HỆ THỐNG</span>
              $$\\min T_{\\text{infer}} \\quad \\text{s.t.} \\quad \\operatorname{Agreement}(f_{\\text{ECDR}}, f_{\\text{frozen}}) = 1.0000$$
              <span class="formula-note">Tối thiểu hóa độ trễ với ràng buộc bảo toàn quyết định tuyệt đối (0 vi phạm nhãn)</span>
            </div>

            <div style="font-size:12px; color:var(--text-muted); margin-top:8px; line-height:1.5;">
              * <em>Lưu ý:</em> Phân biệt rõ giữa <strong>Độ chính xác phân loại (89.05%)</strong> của mô hình CNN trên nhãn thực tế và <strong>Độ khớp tham chiếu (100.0%)</strong> của ECDR đối với các quyết định của chính mô hình CNN đó.
            </div>
          </div>

          <div class="verdict-box">
            <div class="verdict-good">
              <h4>[+] CƠ HỘI TÁI SỬ DỤNG TỪ 50% CHỒNG LẤN:</h4>
              <ul class="points">
                <li><strong>Trực giác vật lý:</strong> Do hai cửa sổ kề cận chia sẻ 64 mẫu cảm biến trùng lặp (1.28 giây), câu hỏi nghiên cứu là liệu có thể tái sử dụng tính toán để giảm tải suy luận mà vẫn bảo toàn 100% quyết định của mô hình CNN đóng băng.</li>
                <li><strong>3 Tiêu chuẩn kiểm chứng khắt khe:</strong>
                  <br>&bull; <em>Soundness:</em> Đảm bảo không có bất kỳ sai lệch nào so với mô hình gốc.
                  <br>&bull; <em>Useful Coverage:</em> Tỷ lệ tái sử dụng hữu ích đạt mức đáng kể (R &gt; 0).
                  <br>&bull; <em>Net Runtime Speedup:</em> Thời gian thực thi thực tế phải nhanh hơn suy luận đầy đủ.
                </li>
              </ul>
            </div>

            <div class="verdict-bad">
              <h4>[-] THÁCH THỨC ĐẶC THÙ CỦA MẠNG NHỎ (SMALLCNN1D):</h4>
              <ul class="points">
                <li><strong>Ngân sách thời gian cực thấp:</strong> SmallCNN1D suy luận cực nhanh (0.14 ms raw forward, 0.34 ms stream query), tạo ra một rào cản rất lớn cho bất kỳ cơ chế quản lý nào.</li>
                <li><strong>Rủi ro chi phí biên (Overhead Risk):</strong> Mọi thao tác kiểm chứng toán học hoặc dịch chuyển tensor trong bộ nhớ nếu vượt quá 0.34 ms sẽ ngay lập tức khiến hệ thống bị trễ hơn (slowdown) thay vì tăng tốc.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.BaselineComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
