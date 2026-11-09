# 04_SYNTHESIS/week4/design/PRIOR_WORK_COMPARATORS.md — Prior-Work Comparators & Applicability Audit

**Task:** W4-029 (Milestone M4)  
**Tác giả:** Antigravity (Pair Programming Agent)  
**Trạng thái:** AUTHORITATIVE AUDIT REPORT  
**Mục tiêu:** Thẩm định nguồn chính (primary sources) của LipCache, StreamiNNC, và TinyDejaVu để xác định khả năng áp dụng trung thực (faithful applicability) và tính khả thi trên chuỗi dữ liệu HAPT đóng băng (frozen reference K=12 SmallCNN1D).

---

## 1. Tóm tắt điều hành & Kết luận bảo lãnh (Executive Summary)

Dựa trên việc đọc và phân tích trực tiếp toàn văn các tài liệu gốc và mã nguồn công bố:

1. **LipCache (Xiang et al., 2026, arXiv:2608.13144v1):**
   - **Đối tượng chứng nhận:** Giấy chứng nhận toán học (certificate) của LipCache **hoàn toàn nằm ở phía proxy (GuardNet)**; bảo đảm toán học trên mạng chính (MainNet) bị **LOẠI TRỪ RÕ RÀNG** (Remark 5.2: *"the theoretical certificate of LipCache is on the GuardNet side, not on the MainNet side"*).
   - **Miền đánh giá:** Chỉ thử nghiệm trên ảnh i.i.d. (CIFAR, Tiny-ImageNet, SVHN). Tác giả xác nhận tính dư thừa thời gian liên tục (streaming consecutive frames) và hiện tượng trôi khái niệm (concept drift) là *nghiên cứu tương lai* (Sec 7.1).
   - **Xung đột phạm vi đóng băng:** Yêu cầu huấn luyện thêm GuardNet chuyên biệt. Trong khi đó, Week 4 cấm huấn luyện lại mạng ngoài luồng hoặc thay đổi checkpoint frozen `SmallCNN1D`.
   - **Kết luận:** **`M14` được xếp loại `NOT_APPLICABLE_IN_FROZEN_SCOPE` / `DIFFERENT_GUARANTEE`**. Giả thuyết **H21 ("ECDR beats LipCache") giữ nguyên trạng thái `NOT_ESTABLISHED`** vì không tồn tại một comparator trung thực, cùng đối tượng bảo đảm trên tập HAPT. Tuyệt đối không chuyển giao kết quả gia tốc ảnh sang HAPT.

2. **StreamiNNC (Kechris et al., 2024, arXiv:2408.03223):**
   - **Bản chất:** Tái sử dụng tính toán intermediate feature (exact incremental computation) cho mạng CNN streaming qua các cửa sổ trượt.
   - **Ràng buộc:** Nhạy cảm với zero-padding tại ranh giới thời gian (receptive field padding leakage). Dự án ECDR đã trừu tượng hóa và kiểm chứng nguyên lý này thành **`M03` (Exact incremental computation, W4-019..W4-021)** trên chính đồ thị `SmallCNN1D`.
   - **Kết luận:** **`M15` xếp loại `AUDITED_APPLICABILITY_ONLY`**; mã nguồn gốc của bài báo không chạy trực tiếp trên checkpoint HAPT và bị loại khỏi bảng xếp hạng hiệu năng thực nghiệm.

3. **TinyDejaVu (Huang & Baccelli, 2026, arXiv:2512.09786v2):**
   - **Bản chất:** Tối ưu hóa bộ nhớ RAM và độ trễ trên vi điều khiển (MCU ARM Cortex-M) thông qua circular activation buffer.
   - **Ràng buộc:** Phụ thuộc vào kiến trúc phần cứng nhúng bare-metal / CMSIS-NN, không tương thích với CPU runtime float32 PyTorch hiện tại.
   - **Kết luận:** **`M15` xếp loại `AUDITED_APPLICABILITY_ONLY`**; bị loại khỏi bảng xếp hạng hiệu năng thực nghiệm.

---

## 2. Thẩm định chi tiết LipCache (Xiang et al., 2026)

### 2.1. Định vị tài liệu & Đặc tả kỹ thuật
- **Primary Source:** arXiv:2608.13144v1 [cs.DC], 21 trang, công bố 2026.
- **Tiêu đề:** *LipCache: A Local Inference Proxy with Certified Caching for Edge Image Classification Service*.
- **Tác giả:** Zhengzhe Xiang, Yinlin Chen, Fuli Ying, Binbin Zhou, Hailiang Zhao, Schahram Dustdar.
- **Mã nguồn / Artifact:** Local proxy edge-server prototype trên nền PyTorch cho image classification.

### 2.2. Phân tích đối tượng chứng nhận & Kiến trúc GuardNet/MainNet
LipCache phân tách hệ thống thành 2 thành phần:
1. **MainNet ($f_{\text{main}}$):** Mạng nơ-ron sâu phức tạp, đặt tại Cloud hoặc Server mạnh.
2. **GuardNet ($f_{\text{guard}}$):** Mạng nơ-ron nhẹ hơn, có hằng số Lipschitz được kiểm soát (thông qua spectral normalization hoặc Lipschitz training), đặt tại Edge device.

Khi có query $x$, GuardNet tính bán kính chứng nhận cục bộ $r(x) = \frac{\Delta(x)}{2 L_{\text{guard}}}$ dựa trên hằng số Lipschitz $L_{\text{guard}}$ của chính nó. Với query mới $x'$, nếu $\|x' - x\| \le r(x)$, LipCache tái sử dụng nhãn đã lưu.

**Mâu thuẫn toán học cốt lõi (Semantic Mismatch):**
- Theo **Remark 5.2** của bài báo:
  > *"The theoretical certificate of LipCache is on the GuardNet side, not on the MainNet side."*
- Nghĩa là: Khi cache hit, LipCache chỉ chứng minh rằng **GuardNet** không đổi nhãn. Hoàn toàn KHÔNG có chứng chỉ nào bảo đảm rằng **MainNet** sẽ không đổi nhãn! Việc gán nhãn cache cho MainNet là một **ước lượng heuristic có rủi ro sai số**, không phải một chứng chỉ bất biến (certified decision invariance).
- Trong khi đó, bài toán ECDR Week 4 đặt ra yêu cầu: Giấy chứng nhận phải bảo toàn quyết định của **chính frozen reference model `SmallCNN1D`** (đóng vai trò là MainNet).

### 2.3. Dữ liệu & Động lực học thời gian
- LipCache đánh giá trên các tập dữ liệu ảnh tĩnh i.i.d.: CIFAR-10, SVHN, Tiny-ImageNet.
- Trong **Section 7.1**, tác giả thừa nhận rõ:
  > *"All experiments are conducted on i.i.d. test splits; temporal redundancy, concept drift, and consecutive-frame reuse in streaming scenarios are left as future work."*
- Chuỗi cảm biến HAPT là streaming liên tục với 50% fixed overlap giữa các cửa sổ $t$ và $t+1$. Tốc độ thay đổi tín hiệu gia tốc/con quay hồi chuyển tạo ra bước dịch chuyển không gian lớn hơn rất nhiều so với các biến thiên nhỏ giữa các ảnh tĩnh.

### 2.4. Điều kiện tiên quyết & Khả năng thích ứng (Adaptation Prerequisites)
Để triển khai một phiên bản LipCache trung thực trên HAPT, cần:
1. Thiết kế và huấn luyện một mạng `GuardNet_1D` chuyên biệt cho HAPT 12 lớp với ràng buộc Lipschitz.
2. Hiệu chuẩn ngưỡng tự tin giữa GuardNet và MainNet.
3. Chấp nhận threat model nới lỏng (heuristic agreement thay vì strict reference equivalence).

**Vi phạm quy tắc dự án:** Task W4-001/002 cấm huấn luyện mạng mới hoặc sửa đổi frozen checkpoint. Do đó, không thể xây dựng một faithful LipCache comparator mà không vi phạm protocol đóng băng của Week 4.

---

## 3. Thẩm định chi tiết StreamiNNC (Kechris et al., 2024)

### 3.1. Định vị tài liệu & Đặc tả kỹ thuật
- **Primary Source:** arXiv:2408.03223 [cs.LG], công bố 2024.
- **Tiêu đề:** *Don't Think It Twice: Efficient Inference on Streaming Data via Incremental CNN Computation*.
- **Tác giả:** Christos Kechris, et al.

### 3.2. Phân tích toán học & Kỹ thuật tái sử dụng
- **Cơ chế:** Khai thác độ trượt (stride) nhỏ hơn kích thước cửa sổ để tái sử dụng các cột feature map đã tính từ cửa sổ trước.
- **Vấn đề Receptive Field & Padding:**
  - Trong mạng Conv1d có zero-padding, các vị trí biên nhận giá trị 0. Khi cửa sổ trượt dịch chuyển sang phải, các giá trị trước đây nằm gần biên bây giờ nằm sâu hơn bên trong, dẫn đến tích chập với các giá trị khác nhau (receptive field distortion).
  - StreamiNNC giải quyết bằng cách áp dụng unpadded convs hoặc quản lý bộ nhớ đệm lịch sử nghiêm ngặt.
- **Quan hệ với ECDR Week 4:**
  - Dự án Week 4 đã tự phát triển độc lập và kiểm chứng chặt chẽ cơ chế này dưới tên gọi **`M03` (Exact incremental computation, W4-019..W4-021)** trên đồ thị `SmallCNN1D`.
  - Kết quả Week 4 cho thấy: Dù số phép tính FLOPs của tầng Conv giảm đáng kể, chi phí tái tính toán tầng Flatten + Linear (chiếm phần lớn thời gian) và overhead quản lý mảng khiến thời gian thực tế không giảm nhiều so với full-forward vectorized trên CPU modern.

---

## 4. Thẩm định chi tiết TinyDejaVu (Huang & Baccelli, 2026)

### 4.1. Định vị tài liệu & Đặc tả kỹ thuật
- **Primary Source:** arXiv:2512.09786v2, công bố 2026.
- **Tiêu đề:** *TinyDejaVu: Smaller RAM and Faster Inference with Neural Networks on MCUs for Sensor Data Streams*.
- **Tác giả:** Huang và Baccelli.

### 4.2. Phân tích kiến trúc & Ràng buộc phần cứng
- **Mục tiêu:** Tối ưu hóa bộ nhớ tĩnh (SRAM < 256KB) trên các vi điều khiển cấp thấp (như STM32, Nordic nRF).
- **Cơ chế:** Chuyển đổi đồ thị tính toán thành circular ring buffers, chỉ tính toán vi sai (differential computing).
- **Tính bất tương thích:** Thiết kế nhúng cấp thanh ghi (bare-metal C / CMSIS-NN instructions), không thể ánh xạ một-một lên môi trường máy chủ Linux x86_64 CPU đa luồng với bộ nhớ RAM tính bằng Gigabytes và PyTorch runtime.

---

## 5. Ma trận so sánh & Phân định phương pháp (Method Dispositions)

| Thuộc tính | ECDR (Đề xuất Week 4) | LipCache (Xiang et al. 2026) | StreamiNNC (Kechris et al. 2024) | TinyDejaVu (Huang et al. 2026) |
|---|---|---|---|---|
| **Mã đăng ký** | `M05–M13` | `M14` | `M15 (streaminnc)` | `M15 (tinydejavu)` |
| **Mục tiêu bảo đảm** | Quyết định của Frozen `SmallCNN1D` K=12 | Quyết định của proxy `GuardNet` | Quyết định đầu ra mạng streaming | Tối ưu hóa SRAM vi điều khiển |
| **Bảo đảm MainNet** | **Toán học nghiêm ngặt (Strict Proof)** | **KHÔNG CÓ (Explicitly excluded)** | **Toán học tương đương (Algebraic)** | **Toán học tương đương (Algebraic)** |
| **Loại dữ liệu** | Multivariate Sensor Time-series (HAPT) | Tĩnh (i.i.d. Image classification) | Generic streaming audio/sensor | Cảm biến nhúng trên MCU |
| **Độ trượt cửa sổ** | 50% temporal overlap (64 steps) | Không có (i.i.d. queries) | Arbitrary stride | Fixed stride |
| **Yêu cầu huấn luyện** | Không (Zero retraining, frozen ref) | **Cần huấn luyện GuardNet riêng** | Không (dựa trên graph hiện có) | Không (yêu cầu layout bộ nhớ nhúng) |
| **Trạng thái Week 4** | **ACTIVE EXECUTION** | **NOT_APPLICABLE_IN_FROZEN_SCOPE** | **AUDITED_APPLICABILITY_ONLY** | **AUDITED_APPLICABILITY_ONLY** |

---

## 6. Phân định giả thuyết H21 (Hypothesis H21 Disposition)

- **Giả thuyết H21:** *"ECDR beats LipCache in practice on streaming HAPT."*
- **Phán quyết khoa học:** **`NOT_ESTABLISHED` (Vẫn chưa xác lập)**.
- **Lý do khoa học:**
  1. Thiếu một đối thủ cạnh tranh trung thực (faithful comparator): Do LipCache yêu cầu proxy training và chỉ bảo đảm phía GuardNet, không thể tạo một phiên bản đối đầu hợp lệ trên frozen HAPT mà không bẻ gãy giao thức đóng băng.
  2. Bất khả thi trong việc so sánh tốc độ: Không thể lấy con số tăng tốc 1.65x trên CIFAR-10 của LipCache để so sánh với latency đo bằng microsecond của Conv1D trên HAPT.
  3. Mọi tuyên bố cho rằng ECDR "vượt trội hơn" LipCache trên HAPT là ngụy tạo (unfounded) khi chưa có benchmark chung ranh giới đo lường.
