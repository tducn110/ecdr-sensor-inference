# Báo Cáo Thiết Kế và Đánh Giá Hợp Đồng Dữ Liệu Week 4 (Contracts Review)

**Mã Run:** `week4_initial_20261103`  
**Task:** `W4-003` — Tạo immutable public types và frozen reference adapter  
**Trạng thái kiểm tra:** `VERIFIED`  
**Ngày thực hiện:** 2026-11-03  

---

## 1. Tổng quan mục tiêu thiết kế

Trong kiến trúc chứng thực hình thức và tái sử dụng suy luận tại runtime (ECDR Week 4), mục tiêu cốt lõi của tầng Hợp đồng Dữ liệu (`ecdr/week4/types.py`) và Bộ chuyển tiếp Tham chiếu (`ecdr/week4/reference.py`) là:
1. **Thiết lập ranh giới tin cậy (Trust Boundary)**: Ngăn chặn triệt để hiện tượng rò rỉ nhãn giám sát (`labels`, `purity`, `activity_id`) vào luồng suy luận runtime.
2. **Bảo tồn tính toàn vẹn của mô hình tham chiếu**: Khóa chặt mô hình $K=12$ SmallCNN1D (283.692 tham số, SHA-256 `8ffc5dd8...`, tiền xử lý `aa8e1cc2...`) trên CPU float32 với chế độ đánh giá (`eval()`, `requires_grad=False`).
3. **Chống giả mạo chứng chỉ và suy luận (Non-forgeable Receipts)**: Đảm bảo rằng chỉ có các suy luận thực sự được thực hiện bởi `ReferenceOwner` mới có thể tạo ra `TrustedAnchorReceipt`, ngăn ngừa việc bịa đặt logits hoặc giả mạo trạng thái anchor.
4. **Bảo đảm bất biến bộ nhớ (Memory Immutability & Anti-Aliasing)**: Dữ liệu cửa sổ cảm biến và miền biên hộp (`CandidateBox`) được lưu trữ dạng tuple bất biến, không trả về con trỏ khả biến cho các module bên ngoài.

---

## 2. Chi tiết các thành phần hợp đồng (`ecdr/week4/types.py`)

### 2.1. `QueryView` — Cửa sổ truy vấn thuần túy
- **Cấu trúc trường**:
  - `global_row_id: int` ($\ge 0$): Định danh vị trí toàn cục trong chuỗi stream.
  - `recording_id: str`: Định danh phiên ghi cảm biến (19 recordings độc lập).
  - `step_idx: int` ($\ge 0$): Chỉ số bước thời gian của cửa sổ hiện tại.
  - `raw_window: Tuple[float, ...]`: Đúng 768 giá trị số thực hữu hạn (kích thước $6 \times 128$).
  - `schema_version: str`: Cố định `"w4.v1"`.
- **Ranh giới an toàn**:
  - Tuyệt đối không chứa bất kỳ trường nào về nhãn (`label`, `activity`, `purity`, `split`, `future_input`).
  - Evaluator giữ tập nhãn độc lập và chỉ ghép cặp sau khi runtime đã ban hành quyết định cuối cùng.
  - Phương thức `array()` trả về bản sao NumPy float32 với cờ `writeable = False`.

### 2.2. Các bảng trạng thái rời nhau (Disjoint Status Vocabularies)
Hệ thống loại bỏ hoàn toàn việc gộp chung các trạng thái vào một biến boolean `valid` mơ hồ:
- **`MembershipStatus`**:
  - `EXACT_MEMBER`: Thuộc miền biên hộp chính xác lưu trữ.
  - `NON_MEMBER`: Nằm ngoài miền biên hộp.
  - `VALIDATED_OUTSIDE`: Đã chứng minh bằng bộ giải toán học là nằm ngoài.
  - `CONDITIONAL_MEMBER`: Thuộc miền biên theo giả định số thực lý thuyết.
  - `UNSUPPORTED_DOMAIN`: Dạng miền không được hỗ trợ.
  - `ERROR`: Lỗi giải thuật / dữ liệu không hợp lệ.
- **`BoundStatus`**:
  - `EXACT_ENCLOSURE`: Bao đóng khoảng chính xác.
  - `CONDITIONAL_POSITIVE`: Biên ước lượng $>0$ theo số thực float64, chưa có chứng minh làm tròn float32.
  - `CONDITIONAL_INCONCLUSIVE`: Biên không đủ dương hoặc cận kề biên âm.
  - `VALIDATED_POSITIVE`: Đã chứng thực dương qua cầu nối số học float32.
  - `VALIDATED_INCONCLUSIVE`: Biên bao đóng không đủ dương sau làm tròn.
  - `FAILED`: Lỗi tính toán hoặc lan truyền biên thất bại.
  - `UNSUPPORTED_METHOD`: Phương pháp không hỗ trợ toán tử trên đồ thị.
- **`DecisionAction`**:
  - `REUSE_CERTIFIED`: Chấp thuận tái sử dụng dự đoán được bảo vệ bởi chứng chỉ.
  - `FALLBACK_REFERENCE`: Chuyển tiếp về mô hình tham chiếu gốc.
  - `FALLBACK_EXPIRED`: Chứng chỉ đã vượt quá tuổi thọ tối đa (`max_age`).
  - `FALLBACK_IDENTITY_MISMATCH`: Nhận diện mô hình hoặc môi trường bị lệch.
  - `FALLBACK_RECORDING_RESET`: Bước sang phiên ghi mới, anchor bị hủy.
  - `FALLBACK_NON_MEMBER`: Truy vấn nằm ngoài miền bao đóng.
  - `FALLBACK_NUMERICAL_GATE_CLOSED`: Cổng số học chưa mở (chưa có nghiệm làm tròn float32).

### 2.3. Quy chuẩn tuần tự hóa nghiêm ngặt (Strict Serialization)
- Toàn bộ các dataclass đều tuân thủ `schema_version = "w4.v1"`. Bất kỳ chuỗi JSON nào mang phiên bản khác đều bị từ chối qua `SerializationError`.
- Áp dụng `json.dumps(..., sort_keys=True, allow_nan=False)`: Cấm triệt để việc serialize các giá trị `NaN` hoặc `Inf`.
- Hash nhận diện (`domain_id`, `identity_id`, `receipt_id`, `certificate_id`) được sinh một cách tất định từ chuỗi byte JSON chuẩn hóa.

---

## 3. Kiến trúc bộ chuyển tiếp tham chiếu (`ecdr/week4/reference.py`)

### 3.1. `ReferenceOwner`
- Khởi tạo trực tiếp `SmallCNN1D(num_classes=12, cfg=...)`.
- Kiểm tra tính toàn vẹn trước khi nạp:
  - Checkpoint SHA-256: `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
  - Normalization Preprocessing ID: `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.
  - Thứ tự 12 nhãn: Khớp chính xác với `ecdr.schema.CLASS_NAMES_12`.
  - Số lượng tham số mô hình: Đúng $283.692$ tham số.
- Thiết lập trạng thái mô hình:
  - `model.eval()`, `requires_grad_(False)`.
  - Toàn bộ trọng số và buffer nằm trên CPU với kiểu dữ liệu `torch.float32`.

### 3.2. Cơ chế kiểm tra danh tính sâu (`assert_identity()`)
Tại mỗi ranh giới suy luận (`infer`) hoặc kiểm tra chứng chỉ, hệ thống thực hiện kiểm tra sâu:
1. Đảm bảo toàn bộ module giữ nguyên chế độ `training == False`.
2. Kiểm tra không có forward hook hoặc pre-forward hook nào được gắn trên mô hình hoặc trên `torch.nn.modules.module`.
3. So khớp mã băm trạng thái trọng số (`_state_digest`) và running statistics của BatchNorm (`_bn_digest`).
4. Kiểm tra mã băm bytecode của hàm forward (`_graph_signature`).
5. Nếu phát hiện bất kỳ sự biến đổi nào (kể cả thay đổi 1 bit ULP trong trọng số hay buffer), hệ thống lập tức ném ra ngoại lệ `IdentityError`.

### 3.3. Cơ chế biên lai tin cậy không thể giả mạo (`TrustedAnchorReceipt`)
- `ReferenceOwner` sở hữu một khóa bí mật phiên ngẫu nhiên 256-bit (`self._secret_salt`) và chỉ số thế hệ (`_generation_id`).
- Khi thực thi suy luận, biên lai được ký bằng HMAC-SHA256 trên tổ hợp:
  $$\text{HMAC}(\text{salt}, \text{identity\_id} \parallel \text{generation\_id} \parallel \text{row\_id} \parallel \text{rec\_id} \parallel \text{step} \parallel \text{raw\_hash} \parallel \text{norm\_hash} \parallel \text{logits\_hash} \parallel \text{winner})$$
- Caller bên ngoài không thể tự tạo ra một `TrustedAnchorReceipt` hợp lệ.
- Khi gọi `invalidate()`, muối bí mật được quay vòng và thế hệ tăng lên, làm vô hiệu hóa tức thì toàn bộ các biên lai cũ.

---

## 4. Bằng chứng kiểm thử độc lập (`tests/test_week4_contracts.py`)

Bộ kiểm thử gồm 18 bài kiểm tra chuyên sâu, đạt kết quả **18/18 PASS (0.102s)**:
- `TestQueryViewContracts`:
  - `test_query_view_fields_contract`: Xác nhận không rò rỉ bất kỳ trường nhãn nào (PASS).
  - `test_query_view_immutability`: Xác nhận không thể sửa đổi thuộc tính sau khi tạo (PASS).
  - `test_query_view_aliasing_protection`: Xác nhận buffer NumPy đầu ra là read-only, việc sửa đổi mảng ngoài không tác động vào QueryView (PASS).
  - `test_query_view_validation_failures`: Bắt chính xác lỗi hình dạng ma trận, giá trị NaN/Inf, chỉ số âm, boolean giả mạo (PASS).
- `TestTypeEnumsAndContracts`:
  - `test_membership_statuses_disjoint`: Kiểm tra tính rời nhau của tập trạng thái miền (PASS).
  - `test_bound_statuses_disjoint`: Kiểm tra tính rời nhau của tập trạng thái cận (PASS).
  - `test_decision_actions_disjoint`: Kiểm tra tính rời nhau của các hành động quyết định (PASS).
  - `test_bound_result_contract`: Kiểm tra số lượng 11 đối thủ cạnh tranh, loại trừ nhãn mục tiêu, phát hiện trùng lặp (PASS).
  - `test_strict_serialization_round_trip`: Kiểm tra vòng tròn tuần tự hóa JSON và từ chối phiên bản schema không hỗ trợ (PASS).
- `TestCandidateBoxContracts`:
  - `test_candidate_box_valid_endpoints`: Kiểm tra tính hợp lệ của biên hộp và phép toán thành viên `contains()` (PASS).
  - `test_candidate_box_inverted_endpoints_fail`: Bắt lỗi khi cận dưới vượt cận trên (PASS).
- `TestFrozenReferenceAdapterContracts`:
  - `test_reference_loading_spec`: Khớp chính xác 12 lớp, 283.692 tham số trên CPU float32 (PASS).
  - `test_normalization_deterministic`: Chuẩn hóa dữ liệu tất định và bất biến (PASS).
  - `test_infer_and_receipt_authenticity`: Phát hành biên lai có chữ ký số hợp lệ (PASS).
  - `test_forged_receipt_rejected`: Từ chối biên lai giả mạo chữ ký hoặc sửa đổi logits (PASS).
  - `test_cross_query_receipt_rejected`: Từ chối việc dùng biên lai của truy vấn này cho truy vấn khác (PASS).
  - `test_receipt_invalidation_lifecycle`: Hủy toàn bộ biên lai khi vòng đời bị reset (PASS).
  - `test_strong_identity_assertion_catches_mutations`: Phát hiện ngay lập tức đột biến trọng số, đột biến BatchNorm running stats, bật chế độ training, hoặc chèn forward hook (PASS).

---

## 5. Kết luận khoa học và bàn giao

Hợp đồng dữ liệu `W4-003` đã thiết lập nền tảng toán học và kỹ thuật an toàn tuyệt đối cho Week 4:
- Trạng thái hoàn thành: **VERIFIED**.
- Sẵn sàng bàn giao cho Task `W4-004` (Công cụ quản lý task claim, locks và receipts CLI).
