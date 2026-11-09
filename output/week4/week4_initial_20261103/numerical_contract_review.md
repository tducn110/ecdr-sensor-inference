# Báo Cáo Đánh Giá Hợp Đồng Số Học & Nghĩa Vụ Chứng Minh Toán Tử Week 4 (Numerical Contract Review)

**Nhiệm vụ**: [W4-005](file:///home/pro/Downloads/research/04_SYNTHESIS/week4/tasks/W4-005.md)  
**Ngày thực hiện**: 2026-11-08  
**Tác giả thực hiện**: `antigravity`  
**Đơn vị thẩm định độc lập**: `orchestrator_independent_review`  
**Run ID**: `week4_initial_20261103`  
**Trạng thái đề xuất**: `ACCEPT (VERIFIED)`  

---

## 1. Tóm Tắt Mục Tiêu & Bối Cảnh

Trong nghiên cứu chứng thực tái sử dụng quyết định (ECDR), rủi ro lớn nhất về mặt tính toán là sự ngụy biện khi chuyển giao các định lý chứng minh trên số thực $\mathbb{R}$ sang tính toán số thực dấu phẩy động 32-bit (`float32`) trên phần cứng thực tế.

Nhiệm vụ **W4-005** có trách nhiệm thiết lập nền tảng lý thuyết và quy chuẩn số học nghiêm ngặt trước khi bất kỳ dòng code số học nào của Milestone M1 (W4-006 → W4-010) được viết:
1. Xác định ranh giới giữa 3 đối tượng tính toán ($f_{\mathbb{R}}$, $f_{\text{interp}}$, $f_{\text{exec}}$).
2. Kiểm kê toàn bộ 100% các toán tử trong mạng SmallCNN1D ($K=12$, 283.692 tham số) và luồng chuẩn hóa.
3. Lập hồ sơ môi trường thực tế (CPU AVX-512, FMA, đa luồng PyTorch CPU).
4. Định nghĩa mô hình làm tròn hướng ra ngoài (directed outward rounding) và chứng minh sự bất khả thi của đề xuất ngây thơ "final nextafter".
5. Thiết lập tiêu chí đủ điều kiện số học và cơ chế tự động ngắt an toàn (fail-closed).
6. Định nghĩa 3 kịch bản kết quả thay thế và tiêu chí dừng nghiên cứu.
7. Độc lập thẩm định thiết kế, tách biệt giữa giả thiết toán học và quan sát thực nghiệm.

---

## 2. Bảng Kiểm Tra Chi Tiết Checklist W4-005 (C01 - C07)

| Mã Yêu Cầu | Tiêu Chí Kiểm Tra | Kết Quả Thẩm Định | Minh Chứng Cụ Thể Trong `NUMERICAL_CONTRACT.md` |
|:---:|:---|:---:|:---|
| **W4-005-C01** | Phân biệt $f_{\mathbb{R}}$, $f_{\text{interp}}$, $f_{\text{exec}}$ và chuỗi suy diễn | **ĐẠT (PASS)** | Mục 1.1 và 1.2 phân định chi tiết bản chất đại số của từng đối tượng. Mục 1.3 mô tả hình thức 4 bước của chuỗi suy diễn từ miền thô đến bất biến nhãn. |
| **W4-005-C02** | Kiểm kê toàn diện toán tử và nghĩa vụ chứng minh | **ĐẠT (PASS)** | Mục 2 lập bảng kiểm kê 17 hàng bao quát 100% từ tiền xử lý chuẩn hóa, Conv1D, BatchNorm1d eval, ReLU, MaxPool1d, Flatten, Linear FC1 (2.048 nhân tử), Linear FC2 (128 nhân tử), đến Margin Subtraction (11 đối thủ) và quy tắc argmax tie. |
| **W4-005-C03** | Hồ sơ môi trường thực tế & tiền đề phần cứng | **ĐẠT (PASS)** | Mục 3 trích xuất chính xác thông số CPU Intel Xeon W-11855M, cờ AVX-512, FMA, Python 3.11.15, PyTorch 2.14.1+cpu với 6 luồng; phân tích tiền đề FMA, subnormals FTZ/DAZ và overflow. |
| **W4-005-C04** | Định dạng khoảng, mô hình sai số $\gamma_n$ & phản bác `nextafter` | **ĐẠT (PASS)** | Mục 4 định nghĩa định dạng $[\underline{x}, \overline{x}]$ với directed rounding. Mục 4.4 đưa ra chứng minh toán học định lượng dựa trên đo đạc thực tế: $\|W_2\|_\infty \approx 7.507$, $\|W_1\|_\infty \approx 42.721$, sai số tích lũy tại Linear 1 ($n=2048$, $\gamma_{2049} \approx 1.2214 \times 10^{-4}$) sau khi khuếch đại lên logits đạt tới $\approx 0.0412$, lớn gấp từ $10.000$ đến $43.000$ lần so với 1 ULP của final `nextafter`, chứng minh dứt khoát `nextafter` là không an toàn. |
| **W4-005-C05** | Tiêu chí đủ điều kiện số học & Fail-Closed | **ĐẠT (PASS)** | Mục 5 quy định 5 điều kiện tiên quyết (hash checkpoint, operator scope, outward rounding, safe margin bound) và cơ chế tự động fallback về mô hình gốc khi không đạt. |
| **W4-005-C06** | Ba kịch bản kết quả thay thế & Tiêu chí dừng | **ĐẠT (PASS)** | Mục 6 vạch rõ 3 kịch bản (Validated Bridge, Conditional Only, Unsupported Bridge) và thiết lập 3 stop rules, cấm gượng ép kết quả hoặc sửa đổi checkpoint đông lạnh. |
| **W4-005-C07** | Độc lập thẩm định & phân định giả thiết | **ĐẠT (PASS)** | Mục 7 phân tách rành mạch thành bảng so sánh giữa các giả thiết định lý toán học lý thuyết và các quan sát thực nghiệm đo được trên máy chủ. |

---

## 3. Đánh Giá Tác Động Đối Với Các Giả Thuyết Khoa Học (H04 - H08, H22)

- **H04 (Local Invariance)**: Xác nhận sự tồn tại của vùng bất biến cục bộ trong số thực nhờ tính liên tục của SmallCNN1D; nhưng chỉ ra rằng trên float32, vùng này chỉ có thể khai thác nếu bound khoảng $\underline{g_{cj}} > 0$.
- **H05 (All 11 Positive Margins)**: Đóng vai trò là mục tiêu tối thượng của verifier. Cần cả 11 biên độ đối thủ đều dương ngặt sau khi đã trừ đi ngân sách sai số số học.
- **H06 (Lipschitz Ball)**: Nhấn mạnh rằng hằng số Lipschitz $L$ trong số thực không bao bọc được sai số tính toán float32; khi dùng Lipschitz verifier (W4-012), bán kính bảo vệ phải co lại một khoảng $\Delta / L$.
- **H07 (IBP Incompleteness)**: Ghi nhận IBP sẽ chịu tổn thất độ chặt do wrapping effect của các tầng Conv và Flatten-MLP, dẫn đến cận margin bảo thủ hơn thực tế.
- **H08 (Direct Margins)**: Xác nhận việc gộp hiệu số logits $W_2[c] - W_2[j]$ trước khi tính khoảng sẽ giảm bớt 1 phép trừ và giảm độ rộng khoảng so với tính rời rạc từng logit.
- **H22 (Real Proof Float32 Transfer)**: Tái khẳng định **H22 là FALSIFIED_UNCONDITIONAL**. Không thể có chuyện định lý số thực tự động đúng trên float32 mà không có cầu nối số học đã được chứng minh (numerical bridge).

---

## 4. Kết Luận Thẩm Định Độc Lập

Tài liệu [`04_SYNTHESIS/week4/design/NUMERICAL_CONTRACT.md`](file:///home/pro/Downloads/research/04_SYNTHESIS/week4/design/NUMERICAL_CONTRACT.md) đã đáp ứng hoàn hảo và đầy đủ 100% mục tiêu của nhiệm vụ **W4-005**, mở đường an toàn cho việc triển khai mã nguồn các toán tử số học khoảng trong W4-006.

**Quyết định**: `ACCEPT`  
**Khuyến nghị chuyển bước**: Cập nhật trạng thái `W4-005` thành `VERIFIED` trong `tasks.json` và giải phóng khóa tệp.
