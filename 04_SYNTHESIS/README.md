# TẦNG 4: TỔNG HỢP, NGHIỆM THU & BÁO CÁO NGHIÊN CỨU TỔNG HỢP (TIER 4 - SYNTHESIS & FINAL REPORT)

> **Tình trạng nghiệm thu toàn trình (G0–G6 & Post-Week 4):**
> - **Cổng G0–G2:** Đã hoàn tất và đóng cổng toàn diện (`G2-Full Closure`, $K=12$, 0/18 blockers).
> - **Cổng G3:** Đã nghiệm thu và kiểm chứng hình thức (`VERIFIED`, bộ xác minh CROWN 64-bit + làm tròn hướng ra ngoài $\tau_{\text{fp}}$).
> - **Cổng G4–G5:** Đã nghiệm thu và đánh giá đầu-cuối trên 5.053 cửa sổ luồng (`VERIFIED`, xác nhận tính bảo toàn quyết định $100\%$, bác bỏ trung thực việc tái sử dụng trên lưới trượt thô $S=64$ do hiện tượng dịch pha biên độ lớn; mở rộng kiểm tra $S=1$ ghi nhận tỷ lệ đồng thuận liền kề đạt $97{,}97\%$ trên $322.856$ cặp nhưng hộp IBP vẫn cho $0/49.677$ ước lượng biên dương).
> - **Cổng G6:** Đã xuất bản báo cáo nghiên cứu tổng hợp cuối kỳ (1 cột, tài liệu tham khảo chuẩn **IEEEtran**) tại [`ECDR_Final_Report_EN.pdf`](../ECDR_Final_Report_EN.pdf) và vượt qua toàn bộ bộ kiểm thử đơn vị hồi quy.

---

## 1. Hệ Thống Báo Cáo Nghiên Cứu Tổng Hợp

1. [`week5/reports/ECDR_Final_Report_EN.tex`](week5/reports/ECDR_Final_Report_EN.tex) $\rightarrow$ [Bản PDF Báo Cáo Tổng Hợp (51 trang)](../ECDR_Final_Report_EN.pdf):
   - **Nội dung:** Báo cáo tổng hợp cuối kỳ 1 cột (tiếng Anh, trích dẫn chuẩn IEEE), trình bày đầy đủ Tuần 1–4, nền tảng toán học, hợp đồng số học `binary32` theo Higham, đối chứng 17 phương pháp (M00–M16), 24 giả thuyết (H01–H24), đóng cổng G2 toàn diện và kiểm chứng độ nhạy bước trượt / quy mô mô hình Post-Week 4.
2. [`week4/reports/Week_4_Implementation_and_Evaluation.tex`](week4/reports/Week_4_Implementation_and_Evaluation.tex) $\rightarrow$ [Bản PDF Tuần 4 (5 trang)](week4/reports/Week_4_Implementation_and_Evaluation.pdf):
   - **Nội dung:** Báo cáo kỹ thuật Tuần 4 trình bày triển khai, đánh giá 17 phương pháp trên 5.053 cửa sổ kiểm thử và phân tích cổng G0–G6.
3. [`week4/reports/Week_4_Explained.tex`](week4/reports/Week_4_Explained.tex) $\rightarrow$ [Bản PDF Giải Thích Tuần 4 (6 trang)](week4/reports/Week_4_Explained.pdf):
   - **Nội dung:** Tài liệu giải thích trực quan, ví dụ tính tay với mô hình $K=12$ và hướng dẫn cơ chế ECDR.
4. [`week5/README.md`](week5/README.md):
   - **Nội dung:** Hướng dẫn tái lập bằng chứng Tuần 5 (`stride_audit.py`, `reconcile_replay.py`, `paired_latency.py`) và biên dịch báo cáo tổng hợp.

---

## 2. Bằng Chứng Thực Nghiệm & Tài Liệu Nghiệm Thu Tuần 4

- [Báo cáo nghiệm thu Week 4 Closeout](week4/CLOSEOUT.md)
- [Hướng dẫn tái lập REPRODUCE.md](week4/reports/REPRODUCE.md)
- [Bảng so sánh 17 phương pháp](../output/week4/week4_initial_20261103/tables/comparison_matrix.csv)
- [Kết luận kiểm định 24 giả thuyết](../output/week4/week4_initial_20261103/tables/hypotheses.csv)
- [Báo cáo nghiệm thu độc lập Tuần 4](../output/week4/week4_initial_20261103/independent_audit.json)
- [Kết quả nghiên cứu mở rộng khả thi Post-Week 4](../output/post_week4/feasibility_results.json)
- [Hồ sơ đóng cổng G2 Full Closure ($K=12$)](../output/post_week4/g2_full_closure.json)

---

## 3. Tài Liệu Lịch Sử Đã Lưu Trữ

Các bản nháp cũ từ giai đoạn khởi động (tháng 10/2026) ghi nhận trạng thái sơ khai chưa qua thực nghiệm đã được chuyển vào lưu trữ an toàn tại:
- [`../archive/synthesis_stale_oct22/`](../archive/synthesis_stale_oct22/): Lưu trữ các tệp Markdown cũ (`03_report*.md`, `04_GATES_EVIDENCE*.md`, `05_TMB_BOARD.md`, `06_ROADMAP_SUGGESTION.md`).
- [`../archive/README.md`](../archive/README.md): Bản đồ chỉ mục lưu trữ toàn bộ dự án.
