# Protocol Design Review & Pre-Fit Peer Assessment (W4-002)

- **Task ID**: `W4-002`
- **Run ID**: `week4_initial_20261103`
- **Review Date**: 2026-11-03
- **Reviewer**: `orchestrator_independent_review`
- **Subject Artifacts**:
  - `configs/week4/protocol.json`
  - `configs/week4/methods.json`
  - `04_SYNTHESIS/week4/design/EVALUATION_PROTOCOL.md`

---

## 1. Mục tiêu & Tuân thủ Yêu cầu

| Yêu cầu | Nội dung đối chiếu | Trạng thái | Đánh giá độc lập |
|---|---|---|---|
| **W4-002-C01** | Công bố `KNOWN_TEST_EXPLORATORY` và phân định 3 giai đoạn giao thức | **VERIFIED** | Đã khai báo minh bạch lịch sử tiếp xúc của tập test 5.053 windows; tách rõ Initial Protocol $\to$ Validation Selection $\to$ Sealed Replay. |
| **W4-002-C02** | Hệ thống mẫu số độc lập (5.053, 3.572, 5.034, 19) và phân tách lỗi reuse vs fallback | **VERIFIED** | Mẫu số khớp chính xác với kiểm toán Week 3; $A_{reuse}$ được định nghĩa nullable khi $R=0$, loại trừ nguy cơ claim 100% agreement ảo khi zero reuse. |
| **W4-002-C03** | Lưới ứng viên, ngân sách build, timeout và hàm mục tiêu validation | **VERIFIED** | Đã chốt trước lưới bán kính $\epsilon$, hạng PCA $k$, chân trời $H$, timeout 500ms, và tiêu chí chọn lọc trên validation trước khi chạy test. |
| **W4-002-C04** | Tiêu chí so sánh hình học công bằng & hull-loss control | **VERIFIED** | Đã phân biệt thể tích 768D ambient vs 384D wave; thiết lập cặp đối chứng M08 (box hull) và M09 (native zonotope). |
| **W4-002-C05** | Môi trường phần cứng CPU, batch 1, single thread và ranh giới đo | **VERIFIED** | Ranh giới cold setup, inference, membership check và certificate build được phân định rõ ràng. Dữ liệu RAM-resident loại trừ độ trễ disk I/O. |
| **W4-002-C06** | Bảng đăng ký đầy đủ 17 phương pháp M00–M16 | **VERIFIED** | Đầy đủ 17 phương pháp; các phương pháp policy-axis (M12, M13) và applicability-only (M15) được gắn nhãn đúng bản chất, không gộp thành thuật toán lan truyền độc lập. |
| **W4-002-C07** | Đơn vị lấy mẫu thống kê và xử lý bất định | **VERIFIED** | Lấy 19 phiên ghi (recordings) làm đơn vị lấy mẫu chính cho các kiểm định paired t-test và bootstrap CI 95%, triệt tiêu lạm phát bậc tự do do tự tương quan. |

---

## 2. Kết luận Thẩm định Độc lập

Bộ tài liệu giao thức và tệp cấu hình máy đọc được đã hoàn thành toàn diện, đáp ứng nghiêm ngặt các nguyên tắc phương pháp luận khoa học:
1. Không có hiện tượng rò rỉ dữ liệu kiểm thử (test leakage) trong giai đoạn thiết kế.
2. Các mẫu số và tiêu chuẩn nghiệm thu loại bỏ hoàn toàn khả năng báo cáo kết quả dương giả mạo.
3. Cấu hình sẵn sàng để chuyển sang các task tiếp theo thuộc Milestone M0.

**Quyết định thẩm định**: **ACCEPT** (Đạt chuẩn để đóng task W4-002).
