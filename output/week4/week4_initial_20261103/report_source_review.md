# Rà soát Tài liệu Nguồn Báo cáo Nghiên cứu Week 4 (Report Source Review)

**Mã Task:** W4-037  
**Run ID:** `week4_initial_20261103`  
**Ngày rà soát:** 2026-11-09T01:26:00Z  
**Tài liệu rà soát:** `04_SYNTHESIS/week4/reports/Week_4_Implementation_and_Evaluation.tex`  
**File tham khảo trích dẫn:** `04_SYNTHESIS/week4/reports/refs.bib`  

---

## 1. Đối chiếu Nguồn và Toàn vẹn Dữ liệu

| Thực thể đối chiếu | Nguồn chuẩn | Dữ liệu trong Báo cáo LaTeX | Trạng thái |
|---|---|---|---|
| Kiến trúc mô hình | `output/week1_closeout_20261102/reference/` | SmallCNN1D $K=12$, 283.692 params, 1.136.072 bytes | `MATCH` |
| HAPT Test Denominators | `output/week4/<run_id>/replay_results.json` | 5.053 queries, 3.572 supervised, 5.034 pairs, 19 recordings | `MATCH` |
| Độ chính xác M00 Baseline | `output/week4/<run_id>/tables/comparison_matrix.csv` | Accuracy: 89.05%, Macro-F1: 0.7763, Throughput: 2.923.71 qps, Latency: 0.342 ms | `MATCH` |
| Kết quả M03 Incremental | `output/week4/<run_id>/tables/comparison_matrix.csv` | Reuse: 5.034 hits, $A_{\text{reuse}} = 1.0$, Latency: 1.9273 ms, Speed ratio: 0.1775x (-463.47%) | `MATCH` |
| Kết quả Certified M05–M13 | `output/week4/<run_id>/tables/comparison_matrix.csv` | Reuse: 0 hits, $A_{\text{full}} = 1.0$, Speed ratio $\le 0.7636$x | `MATCH` |
| Hypotheses Ledger (H01–H24) | `output/week4/<run_id>/tables/hypotheses.csv` | 24 giả thuyết với verdict nguyên bản và witness đối chứng | `MATCH` |
| H24 Trilemma Conjunction | `output/week4/<run_id>/tables/h24_conjunction.csv` | 0/17 cấu hình đạt Soundness $\land$ Coverage $\land$ Speedup | `MATCH` |

---

## 2. Rà soát Chi tiết Checklist C01–C07

- **W4-037-C01 (Problem, Math & Semantics First):** Phần I và II của báo cáo mở đầu bằng bài toán streaming sensor, dữ liệu HAPT, kiến trúc 1D CNN và định nghĩa margin specs trước khi trình bày implementation chi tiết.
- **W4-037-C02 (Interfaces, Operators & Profiles):** Báo cáo mô tả đầy đủ interface `StreamingMethodAdapter`, các toán tử Conv1D/BN/MaxPool/Flatten/Linear, miền hình học (Coordinate Box, Wave, Affine PCA Hull, Correlated Zonotopes) và các giả định runtime.
- **W4-037-C03 (24 Original Hypotheses Preserved):** Đầy đủ 24 giả thuyết H01..H24 được giữ nguyên vẹn nội dung gốc và đối chiếu với bằng chứng thực nghiệm Week 4.
- **W4-037-C04 (Exact Generated Tables):** Toàn bộ số liệu trong Table I và Table II trích xuất trực tiếp từ các file CSV được sinh bởi `scripts/week4_tables.py` thuộc run `week4_initial_20261103`.
- **W4-037-C05 (Evidence Separation & Limitations):** Tách bạch rõ chứng minh toán học trên số thực ($\mathbb{R}$) với thực thi silicon trên Float32 ($\mathbb{F}_{32}$), ghi nhận công khai mức phơi nhiễm `KNOWN_TEST_EXPLORATORY`.
- **W4-037-C06 (Primary Source Citations):** Trích dẫn đầy đủ các công trình gốc chính thức: HAPT (Reyes-Ortiz et al. 2016), CROWN (Zhang et al. 2018), Auto-LiRPA (Xu et al. 2020), IBP (Gowal et al. 2018), LipCache (Xiang et al. 2026), StreamiNNC (Kechris et al. 2024), TinyDejaVu (Huang et al. 2026).
- **W4-037-C07 (H24 Trilemma Evaluation):** Đánh giá riêng biệt trên từng phương pháp cụ thể. Kết luận rõ: M03 có coverage nhưng thất bại về speedup; M05–M13 đúng đắn nhưng thất bại về streaming coverage; M00 baseline vẫn là tối ưu nhất trên CPU.

---

## 3. Phán quyết Rà soát Nguồn

Bản thảo LaTeX `Week_4_Implementation_and_Evaluation.tex` và thư mục tài liệu tham khảo `refs.bib` tuân thủ 100% các tiêu chuẩn học thuật, trung thực về mặt dữ liệu và sẵn sàng cho việc biên dịch sang PDF ở task tiếp theo (W4-038).
