# W4-031 — Independent Pre-Test Integration and Readiness Review

**Trạng thái:** HOÀN TẤT ĐÁNH GIÁ (VERIFIED). Cổng kiểm soát Milestone M4.
**Tác giả rà soát độc lập:** `independent_reviewer` (điều phối qua `antigravity`).
**Ngày rà soát:** 2026-11-08 (UTC). Run ID: `week4_initial_20261103`.
**Phạm vi:** Toàn bộ hệ thống trước khi khởi chạy Milestone M5 (Benchmarking & Replay).

---

## 1. Mục đích và Tuyên bố Độc lập

Tài liệu này ghi nhận kết quả rà soát độc lập toàn diện và nghiệm thu cổng kỹ thuật (Gate Review) W4-031 trước khi mở khóa Milestone M5. Mọi nhận định dựa trên:
1. `[A] Source/Runtime đã đọc và thực thi`: kiểm tra trực tiếp mã nguồn, bảng điều khiển, bộ nhớ và log chạy thực tế.
2. `[B] Official docs & Mathematical Proofs`: đối chiếu các định lý hình thức, hợp đồng số học và đặc tả giao thức.
3. `[C] Inference`: suy luận logic được kiểm chứng bằng thực nghiệm độc lập.
4. `[D] Recommendation`: khuyến nghị điều phối và phân định phạm vi khoa học.

Tuyệt đối không sử dụng các giả định chưa được kiểm chứng; giữ vững nguyên vẹn 214 protected files của dự án.

---

## 2. Kiểm toán Trạng thái Hệ thống & Danh mục Nhiệm vụ (C01)

### 2.1. Tiến độ Tổng thể
Tính đến trước cổng W4-031, hệ thống đã hoàn thành nghiệm thu độc lập **30/40 nhiệm vụ (75.0%)**:
- **Milestone M0 (Governance & Protocol Sealing):** W4-001..W4-004 (**4/4 tasks VERIFIED, 100%**).
- **Milestone M1 (Mathematical Foundations & Proofs):** W4-005..W4-010 (**6/6 tasks VERIFIED, 100%**).
- **Milestone M2 (Interval Bounds & Coordinate Geometries):** W4-011..W4-018 (**8/8 tasks VERIFIED, 100%**).
- **Milestone M3 (Incremental Engine, State & Runtime):** W4-019..W4-027 (**9/9 tasks VERIFIED, 100%**).
- **Milestone M4 (Fair Comparisons, Method Registry & Protocol Seal):**
  - W4-028 (Method Registry M00–M16): **VERIFIED**.
  - W4-029 (Prior-Work Applicability Matrix): **VERIFIED**.
  - W4-030 (Validation Selection & Protocol Seal): **VERIFIED**.
  - W4-031 (Pre-Test Integration & Readiness Review): **ĐANG NGIỆM THU (Gate M4)**.

### 2.2. Kiểm toán Định danh Tham số & Trọng số Bất biến
- **Checkpoint SHA-256:** `6e6113bdae5e7bb96f424ecab2c40c3ff4b58e723cc58ad44a04bf5a478b0f71` (Khớp 100% với Week 1–3 frozen checkpoint `models/smallcnn1d_model.pt`).
- **Normalization SHA-256:** `f5bc3a4e92a43b23b3e24deeb2b774a3f4e15de5b61033a382aa0639d6b9d628` (Khớp 100% với `data/normalization_constants.json`).
- **Graph Topology:** `SmallCNN1D(K=12)` với 3 tầng Conv1D (kernel=3, padding=1), BatchNorm1d, ReLU, MaxPool1d(stride=2), Flatten(2048), Linear(128), ReLU, Linear(12). Tuyệt đối loại bỏ cấu trúc GAP (Global Average Pooling) cũ.
- **Phân chia Dữ liệu HAPT:**
  - Train: 9,978 windows
  - Validation: 2,422 windows
  - Test: 5,053 windows
  - Tổng cộng: 17,453 windows (Đã xác minh độc lập, khớp 100%).
- **Bảo vệ Toàn vẹn:** Toàn bộ 214 file thuộc phạm vi Week 1–3 hoàn toàn nguyên vẹn, SHA-256 khớp 100% qua công cụ `scripts/week4_audit.py check`.

---

## 3. Kiểm toán Kết quả Kiểm thử Toàn diện (C02)

Toàn bộ test suite của repository được phát hiện và thực thi tự động qua Python 3.11 trong môi trường `.venv`:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```
- **Tổng số Unit Tests:** **345 tests**.
- **Kết quả:** **345 / 345 PASSED (100% PASS RATE)**.
- **Thời gian thực thi:** 46.195 giây.
- **Số lỗi (Failures/Errors):** **0 failures, 0 errors**.
- **Không có hồi quy (Zero Regressions):** Toàn bộ các kiểm thử từ Week 1–3 và các bài kiểm tra chuyên biệt Week 4 (adversarial fault-injection, numerical bridge, LiRPA, incremental engine, horizon drift, affine domains) đều đạt chuẩn.

---

## 4. Kiểm toán Khả năng Vận hành của 17 Method Adapters M00–M16 (C03)

Quy trình mini-replay trên tập validation split khẳng định 17 phương pháp đã được đăng ký và phân loại minh bạch trong [`ecdr/week4/methods.py`](file:///home/pro/Downloads/research/ecdr/week4/methods.py):

| ID | Tên Phương Pháp | Phân Loại Semantics | Threat Model | Numerical Status | Kết Quả Mini-Replay |
|---|---|---|---|---|---|
| **M00** | Frozen full core | Baseline | `NO_INTEGRITY_CHECK` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; tham chiếu operational reference, 100% logits parity. |
| **M01** | Full equivalent wrapper | Baseline Wrapper | `PER_QUERY_INTEGRITY_CHECK` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; đo lường overhead băm SHA-256 (1.39 ms). |
| **M02** | Exact input replay | Singleton Reuse | `EXACT_INPUT_EQUALITY` | `EXACT_DISCRETE_EQUALITY` | Sẵn sàng; so khớp bitwise byte mảng chuẩn hóa; 0% false regional claim. |
| **M03** | Exact incremental | Feature Cache | `INCREMENTAL_STAGE_EQUIVALENCE` | `STRICT_ELIGIBLE_FOR_PROFILE` | Sẵn sàng; tái sử dụng Conv3 đặc trưng, 100% quyết định và logits khớp M00. |
| **M04** | Heuristic reuse | Heuristic Trade-off | `EMPIRICAL_HEURISTIC_NO_GUARANTEE` | `EMPIRICAL_DISTANCE_HEURISTIC` | Sẵn sàng; tái sử dụng theo ngưỡng khoảng cách; không có badge hình thức. |
| **M05** | Global Lipschitz cache | Regional Certification | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `CONDITIONAL_REAL_ARITHMETIC` | Sẵn sàng; bán kính $L_2$ xác thực từ hằng số phổ ma trận. |
| **M06** | IBP diagonal box | Regional Certification | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; lan truyền hộp 768 chiều với 11 ràng buộc margin trực tiếp. |
| **M07** | IBP fixed-overlap wave | Regional Certification | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; miền wave 384 fixed + 384 free dimensions. |
| **M08** | Correlated coordinate hull | Control Ablation | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `CORRELATED_COORDINATE_HULL` | Sẵn sàng; đo lường tổn thất hình học (hull loss) khi bọc hộp affine. |
| **M09** | Native correlated bounds | Regional Certification | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `NATIVE_CORRELATED_BOUNDS` | Sẵn sàng; lan truyền trực tiếp affine zonotope factors. |
| **M10** | CROWN diagonal | Regional Certification | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `SURROGATE_BACKWARD_RELAXATION` | Sẵn sàng; giải phóng tuyến tính CROWN trên hộp tọa độ. |
| **M11** | CROWN wave/correlated | Regional Certification | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `SURROGATE_BACKWARD_RELAXATION` | Sẵn sàng; giải phóng tuyến tính CROWN trên miền wave và affine. |
| **M12** | Fixed-anchor policy | Policy Axis | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; trục chính sách neo cứng anchor duy nhất. |
| **M13** | Finite-horizon policy | Policy Axis | `REGIONAL_MATHEMATICAL_CERTIFICATION` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; trục chính sách giới hạn horizon rời rạc $H \in \{1, 2\}$. |
| **M14** | Faithful LipCache | Prior-Work Comparator | `AUDITED_UNSUPPORTED` | `NOT_APPLICABLE` | Không thi hành; đã audit trong W4-029 (cần đào tạo GuardNet, ngoài scope). |
| **M15** | Prior-work audit | Applicability Audit | `AUDITED_UNSUPPORTED` | `NOT_APPLICABLE` | Không thi hành; đã audit trong W4-029 (StreamiNNC/TinyDejaVu). |
| **M16** | OWNED_REFERENCE | Secondary Baseline | `OWNED_PROCESS_BOUNDARY` | `VALIDATED_FLOAT32_ENCLOSURE` | Sẵn sàng; baseline tối ưu hóa bộ nhớ in-process (0.439 ms). |

---

## 5. Kiểm toán Causal Boundaries & Zero-Leakage (C04, C05)

Hệ thống tuân thủ 100% nguyên tắc không rò rỉ thông tin tương lai và nhãn ground-truth:
1. **Giao tiếp Đầu vào Causal:** Tất cả các method adapters chỉ nhận đối tượng `QueryView` hoặc mảng `(6, 128)` float32. `QueryView` hoàn toàn không có thuộc tính `label` hoặc `ground_truth`.
2. **Tách biệt Proposal Khỏi Nhãn Sự kiện:** Miền ứng viên $\mathcal{B}(x_0, \epsilon)$ và $\mathcal{D}_{\text{wave}}$ được xây dựng hoàn toàn từ vector tín hiệu chuẩn hóa $z_0$, không phụ thuộc vào nhãn của query tiếp theo.
3. **Đánh giá Độc lập Ngoại vi:** Nhãn ground-truth và forward tham chiếu chỉ được gọi ở tầng evaluator độc lập ngoài runtime selection nhằm tính toán ma trận nhầm lẫn và độ chính xác ngoại tuyến.
4. **Không Lookahead:** Chỉ số chuỗi $t$ được giám sát nghiêm ngặt; bất kỳ bước nhảy lùi ($t_{i+1} \le t_i$) hoặc bước nhảy vọt không liên tục đều lập tức kích hoạt cơ chế xóa cache fail-closed.

---

## 6. Kiểm toán Khóa Mật mã Giao thức (Protocol Seal - C06)

Giao thức thực nghiệm và cấu hình đã được niêm phong mật mã tại [`output/week4/week4_initial_20261103/protocol_seal.json`](file:///home/pro/Downloads/research/output/week4/week4_initial_20261103/protocol_seal.json):
- **Master Seal Hash:** `5680676232ddb0103b69368a3200353ef1a3ac31ef80fc24876c0e14628b811b`
- **Tình trạng:** Khóa 100% hợp lệ (`verify-seal` PASSED).
- **Mức độ Tiết lộ Dữ liệu:** `KNOWN_TEST_EXPLORATORY` (minh bạch toàn bộ bối cảnh dữ liệu test đã tiếp xúc trong các giai đoạn trước).
- **Cơ chế Chống Giả mạo:** Bất kỳ sự thay đổi nào đối với file cấu hình (`protocol.json`, `methods.json`) hoặc mã nguồn thi hành (`methods.py`, `bounds.py`, `runtime.py`, `scripts/week4_run.py`) sẽ làm lệch hash niêm phong và từ chối chạy thực nghiệm test replay.

---

## 7. Kết luận và Phê duyệt Cổng Milestone M4 (C07)

Rà soát độc lập kết luận:
1. **Tính hoàn thiện:** Cả 30 nhiệm vụ tiền đề thuộc Milestones M0, M1, M2, M3 và M4 đều đã đạt trạng thái `VERIFIED`.
2. **Tính an toàn:** 214 protected files nguyên vẹn 100%; toàn bộ 345 tests vượt qua không có ngoại lệ.
3. **Tính khoa học:** Đầy đủ bằng chứng phản chứng và giới hạn thực nghiệm được lưu giữ trung thực (H02, H03, H06, H07, H11, H14, H15, H19, H20, H21, H22, H23).
4. **Phán quyết:** **CHÍNH THỨC PHÊ DUYỆT CỔNG W4-031 — SẴN SÀNG TIẾN HÀNH BENCHMARK MILESTONE M5**.
