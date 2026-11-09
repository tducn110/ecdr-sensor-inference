# Rà soát Báo cáo Giải thích và Hướng dẫn Tái lập (Explained Source Review)

**Mã Task:** W4-038  
**Run ID:** `week4_initial_20261103`  
**Ngày rà soát:** 2026-11-09T01:28:00Z  
**Tài liệu rà soát:**  
- `04_SYNTHESIS/week4/reports/Week_4_Explained.tex` (Report B)  
- `04_SYNTHESIS/week4/reports/REPRODUCE.md` (Hướng dẫn Tái lập)  

---

## 1. Đối chiếu và Rà soát Checklist C01–C07

- **W4-038-C01 (Định nghĩa hình thức):** Section 2 của `Week_4_Explained.tex` định nghĩa tường minh tensor shapes ($X \in \mathbb{R}^{6 \times 128}$, vector hoá $m=768$), không gian toạ độ Raw vs Normalized, 11 output margins, các trạng thái membership và ranh giới quyền sở hữu state/reference.
- **W4-038-C02 (6 Worked Toy Examples):** Section 3 trình bày chi tiết 6 ví dụ mô hình đồ chơi (Toy Examples 1 đến 6) được đánh dấu rõ ràng, giải thích cặn kẽ:
  1. *Interval Cancellation Defect* ($[0,1] - [0,1] = [-1,1]$) trong IBP (H07).
  2. *Coordinate Hull Bloat* khi chuyển đổi miền tương quan sang bounding box (H11).
  3. *Receptive Field Mapping* và ranh giới 12 shared positions trong 1D CNN (H13).
  4. *Catastrophic Drift* khi căn lại tâm certificate mà không tính toán proof mới (H15).
  5. *1-ULP Floating-Point Boundary Violation* trong Float32 và nhu cầu outward rounding (H22).
  6. *Slicing Overhead Paradox* giải thích vì sao chi phí thao tác tensor vượt quá forward pass trực tiếp trên CPU batch=1 (H14).
- **W4-038-C03 (Mapping 24 Giả thuyết):** Section 4 cung cấp bảng đối chiếu Table I kết nối toàn bộ 24 giả thuyết H01..H24 với cơ chế lý thuyết và biểu hiện thực nghiệm, giữ vững ranh giới giữa Flatten-MLP thực tế và GAP lịch sử.
- **W4-038-C04 (Exact Reproducibility Guide):** `REPRODUCE.md` ghi nhận đầy đủ môi trường hermetic (Python 3.11.10, PyTorch 2.14.1+cpu), các lệnh CLI cụ thể từ Step 0 đến Step 7, kiểm tra `--help`, các artifact đầu ra và SHA-256 checksums tương ứng.
- **W4-038-C05 (Traceability Claim $\to$ Artifact):** Mọi công bố đều được liên kết trực tiếp với mã nguồn kiểm thử và các file kết quả thực tế trong `output/week4/<run_id>/`.
- **W4-038-C06 (Tái lập Kết quả Âm):** Mục 5 của `REPRODUCE.md` hướng dẫn cụ thể cách tái lập kết quả âm của H14 (tỷ lệ tốc độ 0.1775x), H19 (tỷ lệ reuse bằng 0 trên continuous stream) và H24 (0/17 phương pháp đạt đồng thời 3 tính chất).
- **W4-038-C07 (Ngôn ngữ Báo cáo):** Hai báo cáo chính (Report A và Report B) cùng hướng dẫn tái lập được viết bằng tiếng Anh học thuật chuẩn mực; các tài liệu kiểm toán, điều phối và receipt sử dụng tiếng Việt/Anh chuẩn.

---

## 2. Phán quyết Rà soát

Các tài liệu nguồn `Week_4_Explained.tex` và `REPRODUCE.md` đã hoàn thiện đầy đủ, trực quan, chính xác về mặt toán học và sẵn sàng nghiệm thu W4-038 để chuyển sang bước biên dịch PDF và rà soát visual ở W4-039.
