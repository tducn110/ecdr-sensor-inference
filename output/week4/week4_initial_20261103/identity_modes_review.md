# W4-024 Identity Overhead & Threat-Model Trade-Off Review

**Run ID:** week4_initial_20261103  
**Milestone:** M3  
**Hypotheses Addressed:** H18, H20, H21, H22  

---

## 1. Executive Summary

Trong thiết kế zero-trust ban đầu (M01 strong wrapper), `assert_identity()` thực hiện băm SHA-256 trên toàn bộ 283,692 trọng số mạng nơ-ron, tất cả các running statistics của BatchNorm, và AST graph bytecode trên mỗi query inference.

Khảo sát thực nghiệm đo đạc định lượng xác nhận:
- **Pure PyTorch CNN Forward:** `0.160 ms`
- **Full `assert_identity` Digest:** `1.186 ms`
- **M01 Total `infer()` Latency:** `1.623 ms`
- **M16 Total `infer()` Latency:** `0.422 ms`
- **M00 Total `infer()` Latency:** `0.325 ms`

**Tỷ lệ chi phí:** Trong M01, `assert_identity()` chiếm **73.1%** tổng thời gian inference, khiến inference bị chậm gấp **4.99x** so với core baseline M00.

Chế độ **M16 (`OWNED_GENERATION`)** giảm thời gian inference xuống còn `0.422 ms` (tăng tốc **3.85x** so với M01), đồng thời duy trì bảo vệ toàn vẹn đầy đủ nhờ quyền sở hữu độc quyền của `ReferenceOwner` trong tiến trình.

---

## 2. Component Latency Breakdown

| Thành phần | Thời gian (ms) | Mô tả & Trách nhiệm |
| :--- | :--- | :--- |
| **Pure PyTorch Forward** | `0.160` | Đánh giá mạng SmallCNN1D (K=12) trên CPU float32 |
| **NumPy Normalization** | `0.015` | Chuẩn hóa phần tử bằng hằng số float32 bất biến |
| **Fast Shallow Check** | `0.0948` | Kiểm tra `eval` mode, `requires_grad=False`, CPU device, không hooks |
| **Full SHA-256 Digest** | `1.186` | Băm SHA-256 toàn bộ tensor weights, BN buffers, AST bytecode |

---

## 3. Comparison Across Standard Baselines (C06)

| Phương pháp (Baseline) | Identity Mode | Latency (ms/query) | Relative to M00 | Giả định bảo mật / Threat Model |
| :--- | :--- | :--- | :--- | :--- |
| **M00 Core Inference** | `CORE_INFERENCE` | `0.325` | 1.00x | Tin cậy hoàn toàn runtime PyTorch không bị can thiệp. Không có chữ ký receipt. |
| **M01 Strong Wrapper** | `STRONG_PER_QUERY` | `1.623` | `4.99x` | **Zero-Trust tuyệt đối:** Giả định bộ nhớ tiến trình có thể bị can thiệp bất kỳ lúc nào giữa các query. Băm lại toàn bộ 283k params mỗi query. |
| **M16 Owned Reference** | `OWNED_GENERATION` | `0.422` | `1.30x` | **In-process Encapsulation:** Tin cậy mô hình đóng gói trong `ReferenceOwner`. Fast shallow check mỗi query; deep assertion tại init & generation boundaries; HMAC receipt authentication. |

---

## 4. Elimination of Duplicate Work (C01 & C02)

1. **Zero Redundant Forwards on Miss+Build (C01):**
   - Khi xảy ra cache miss, `StreamingRuntimeExecutor` chỉ gọi `reference.infer()` đúng 1 lần duy nhất.
   - Quá trình xây dựng `Certificate` tái sử dụng trực tiếp `TrustedAnchorReceipt` đã sinh, không thực hiện thêm bất kỳ forward call nào.
2. **Zero Forward on Certified Reuse (C01):**
   - Khi một query nằm trong miền đã chứng nhận (`REUSE_CERTIFIED`), `StreamingRuntimeExecutor` trả về trực tiếp nhãn mà không gọi `reference.infer()`.
3. **Immutable Constants Caching (C02):**
   - `NormalizationConstants` được cache an toàn trong `ReferenceOwner`, ngăn chặn chi phí re-hash preprocessing ID và tạo mảng lặp lại.
   - Các mảng `mean` và `std` được bảo vệ bằng `writeable = False`.

---

## 5. Security & Verification Status

- Toàn bộ 5 mutation tests trong `test_c04_exhaustive_mutation_detection` đã pass (weights, BN running stats, training mode, grad, forward hooks, mean/std arrays).
- Không sửa đổi bất kỳ điều kiện số học, rounding, hay numerical gate nào (C07).
