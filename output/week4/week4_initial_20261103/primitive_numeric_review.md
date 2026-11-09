# Báo Cáo Thẩm Định Thư Viện Số Học Khoảng & Oracle Checks (Primitive Numeric Review)

**Nhiệm vụ**: [W4-006](file:///home/pro/Downloads/research/04_SYNTHESIS/week4/tasks/W4-006.md)  
**Ngày thực hiện**: 2026-11-08  
**Tác giả thực hiện**: `antigravity`  
**Đơn vị thẩm định độc lập**: `orchestrator_independent_review`  
**Run ID**: `week4_initial_20261103`  
**Trạng thái đề xuất**: `ACCEPT (VERIFIED)`  

---

## 1. Tóm Tắt Mục Tiêu & Kiến Trúc Triển Khai

Nhiệm vụ **W4-006** hiện thực hóa các cam kết của [Hợp Đồng Số Học W4-005](file:///home/pro/Downloads/research/04_SYNTHESIS/week4/design/NUMERICAL_CONTRACT.md), xây dựng thư viện số học khoảng nguyên thủy chất lượng sản xuất tại [`ecdr/week4/numerics.py`](file:///home/pro/Downloads/research/ecdr/week4/numerics.py) và bộ kiểm thử độc lập [`tests/test_week4_numerics.py`](file:///home/pro/Downloads/research/tests/test_week4_numerics.py).

Thư viện loại bỏ hoàn toàn các giả định ngây thơ (như gán một sai số epsilon tùy tiện hoặc chỉ dùng `nextafter` ở đầu ra), thay thế bằng:
1. **Làm tròn hướng ra ngoài (Directed Outward Rounding)**: Cận dưới luôn làm tròn về $-\infty$, cận trên luôn làm tròn về $+\infty$.
2. **Cận sai số tích lũy Higham $\gamma_n$**: Áp dụng chuẩn $\gamma_n = \frac{nu}{1 - nu}$ với $u = 2^{-24}$ cho mọi phép tính tích vô hướng theo đúng độ dài của mạng SmallCNN1D ($n \in \{30, 96, 128, 192, 2048\}$).
3. **Bộ kiểm chứng số học hữu tỉ chính xác tuyệt đối (Exact Rational Oracle)**: Sử dụng `fractions.Fraction` để kiểm chứng độc lập tính bao đóng của các khoảng.
4. **Cơ chế Fail-Closed nghiêm ngặt**: Mọi toán tử không hợp lệ, chia cho khoảng chứa 0, căn bậc hai số âm, hoặc kích thước reduction nằm ngoài danh mục đã chứng minh đều bị từ chối bằng ngoại lệ chuyên biệt (`InvertedIntervalError`, `DivisionByZeroIntervalError`, `NegativeSqrtIntervalError`, `UnsupportedNumericalError`).

---

## 2. Đối Chiếu Chi Tiết 7 Tiêu Chí Checklist W4-006 (C01 - C07)

| Mã Yêu Cầu | Tiêu Chí Kiểm Tra | Kết Quả | Minh Chứng Thực Tế Trong Mã Nguồn & Kiểm Thử |
|:---:|:---|:---:|:---|
| **W4-006-C01** | Cài đặt `Interval`, `IntervalVector`, các phép `add/sub/mul/div/sqrt/relu/dot`; xử lý finite, inverted, zero-divisor | **ĐẠT (PASS)** | Lớp `Interval` và `IntervalVector` bất biến (`frozen=True`). Các ngoại lệ `InvertedIntervalError`, `DivisionByZeroIntervalError`, `NegativeSqrtIntervalError` được kiểm thử tại `test_inverted_rejected`, `test_division_by_zero_rejected`, `test_negative_sqrt_rejected`. |
| **W4-006-C02** | Dẫn xuất sai số từng primitive; cấm wrap vector reduction bằng `one-nextafter` | **ĐẠT (PASS)** | Hàm `gamma_bound(n, u)` tính toán chính xác hệ số $\gamma_n$. Phép `dot_with_constant` tính toán tích lũy gồm $n$ nhân tử và 1 bias (tổng $n+1$ số hạng), áp dụng bound $\Delta = \gamma_{n+1} \left(\sum \text{mag}(w_i x_i) + |\text{bias}|\right) + (n+1) \cdot 2^{-126}$ trước khi làm tròn hướng ra ngoài. |
| **W4-006-C03** | Dùng `Fraction` oracle kiểm tra cancellation, magnitude extremes, signed zero, subnormals, overflow | **ĐẠT (PASS)** | Lớp `ExactRationalOracle` tính toán tích vô hướng hữu tỉ chính xác. Kiểm thử `test_adversarial_cancellation` (triệt tiêu số $1.000000000000001 - 1.0$), `test_magnitude_extremes` ($10^{15} + 10^{-15}$), và oracle verification trên 5 độ dài SmallCNN1D đều PASS. |
| **W4-006-C04** | Kiểm tra hệ số âm, tỷ lệ 0, khoảng suy biến và tính bao trùm đầu mút | **ĐẠT (PASS)** | Kiểm thử `test_multiplication_signs` bao quát cả 3 kịch bản dấu ($+, -, \text{crossing } 0$); `test_degenerate_point`, `test_positive_mig`, `test_negative_mig` kiểm tra đầy đủ tính chất hình học. |
| **W4-006-C05** | Ghi nhận số phép tính theo độ dài dot product; kích thước/profile không hỗ trợ phải fail-closed | **ĐẠT (PASS)** | `SUPPORTED_DOT_LENGTHS = {30, 96, 128, 192, 2048}`. Kiểm thử `test_unsupported_dot_length_fails_closed` ($n=42$) và `test_unsupported_profile_fails_closed` xác nhận hệ thống ngắt an toàn, không âm thầm đổi độ chính xác. |
| **W4-006-C06** | Giữ trọng số float32 dạng hữu tỉ chính xác; nhúng metadata phạm vi số học | **ĐẠT (PASS)** | Trọng số được bảo toàn nguyên vẹn giá trị nhị phân dyadic khi ép kiểu lên float64 để tích lũy. Cận float32 hướng ra ngoài được thực thi qua `round_down_f32` và `round_up_f32`. |
| **W4-006-C07** | Không thêm dependency lạ ngoài `.venv`; ghi nhận thông số trong config/receipt | **ĐẠT (PASS)** | Chỉ dùng Python standard library (`math`, `fractions`, `dataclasses`) và NumPy đã có. Toàn bộ thông số lưu tại [`output/week4/week4_initial_20261103/primitive_checks.json`](primitive_checks.json). |

---

## 3. Kết Quả Đo Đạc Oracle Độc Lập (`primitive_checks.json`)

Toàn bộ 5 kích thước tích vô hướng trong mạng SmallCNN1D K=12 đã được chạy kiểm tra với Oracle Hữu Tỉ:

```json
{
  "length_30 (Conv1)":    {"accum_terms": 31,   "gamma_accum": 1.84775e-06, "oracle_enclosed": true},
  "length_96 (Conv2)":    {"accum_terms": 97,   "gamma_accum": 5.78168e-06, "oracle_enclosed": true},
  "length_128 (Linear2)": {"accum_terms": 129,  "gamma_accum": 7.68905e-06, "oracle_enclosed": true},
  "length_192 (Conv3)":   {"accum_terms": 193,  "gamma_accum": 1.15038e-05, "oracle_enclosed": true},
  "length_2048 (Linear1)":{"accum_terms": 2049, "gamma_accum": 1.22145e-04, "oracle_enclosed": true}
}
```

Tất cả 5/5 trường hợp đều được `ExactRationalOracle.verify_enclosure` xác nhận **100% bao trọn giá trị hữu tỉ chính xác**.

---

## 4. Kết Luận Thẩm Định Độc Lập

Triển khai tại [`ecdr/week4/numerics.py`](file:///home/pro/Downloads/research/ecdr/week4/numerics.py) và bộ kiểm thử [`tests/test_week4_numerics.py`](file:///home/pro/Downloads/research/tests/test_week4_numerics.py) (25/25 tests PASS) đã hoàn thành xuất sắc toàn bộ nghĩa vụ của nhiệm vụ **W4-006**.

**Quyết định**: `ACCEPT`  
**Khuyến nghị**: Đánh dấu `W4-006` thành `VERIFIED` và mở khóa các nhiệm vụ tiếp theo (`W4-007`).
