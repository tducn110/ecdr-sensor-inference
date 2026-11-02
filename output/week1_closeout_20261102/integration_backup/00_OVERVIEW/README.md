# TẦNG 0: TỔNG QUAN, BÀI TOÁN & KHỞI ĐẦU (TIER 0 - OVERVIEW & FOUNDATIONS)

> **Mục tiêu của Tầng 0**:  
> Cung cấp góc nhìn toàn cảnh về đề tài **ECDR (Event-Conditioned Certified Decision Reuse)**: Hiểu được vì sao chạy mô hình học sâu liên tục trên thiết bị đeo gây cạn pin, tại sao phương pháp caching thông thường dựa trên độ tương đồng (similarity) không đảm bảo an toàn, và nắm bắt 2 câu hỏi nghiên cứu cốt lõi (C11 Novelty & C12 Net Savings).

---

## 1. Trình Tự Đọc Tuyến Tính (Step-by-Step Reading Path)

1. [`01_PROJECT_BRIEF.md`](01_PROJECT_BRIEF.md):  
   - **Nói cái gì**: Giới thiệu bài toán, định nghĩa các thực thể cơ sở (cửa sổ tín hiệu, mỏ neo, vùng sự kiện).  
   - **Đọc xong hiểu được**: Lý do cần tái sử dụng quyết định và ranh giới bài toán.
2. [`02_Week_1_Project_Brief.pdf`](02_Week_1_Project_Brief.pdf) ([Mã nguồn TeX](02_Week_1_Project_Brief.tex)):  
   - **Nói cái gì**: Báo cáo học thuật Tuần 1, đặt bài toán hình thức: $F = f_\theta \circ \mathcal{P}$, $h(X) = \arg\max F_k(X)$.  
   - **Đọc xong hiểu được**: Khái niệm $\mathcal{P}$ là preprocessing chuẩn hóa cố định và mục tiêu bảo toàn quyết định reference.
3. [`03_SUPP01_ECDR_Primer.pdf`](03_SUPP01_ECDR_Primer.pdf) ([Mã nguồn TeX](03_SUPP01_ECDR_Primer.tex)):  
   - **Nói cái gì**: Chuyên đề sơ nhập ECDR (SUPP-01): Vòng đời Anchor 4 trạng thái (`FRESH` $\rightarrow$ `ACTIVE` $\rightarrow$ `STALE` $\rightarrow$ `REFRESH`).  
   - **Đọc xong hiểu được**: Sự khác biệt bản chất giữa "Cache mềm" (heuristic) và "Chứng chỉ an toàn" (Certified).
4. [`04_00_START_HERE.md`](04_00_START_HERE.md):  
   - **Nói cái gì**: Hiện trạng dự án, số liệu checkpoint K=6 thực tế (282.918 tham số).  
   - **Đọc xong hiểu được**: Tránh nhầm lẫn với các mô hình 24k tham số cũ, xác định các ranh giới an toàn.
5. [`05_01_RQ_GAP.md`](05_01_RQ_GAP.md) & [`06_02_LITERATURE.md`](06_02_LITERATURE.md):  
   - **Nói cái gì**: Phân tích khoảng trống khoa học và so sánh 4 trục với các phương pháp quốc tế (NoScope, LipCache, Cavigelli).  
   - **Đọc xong hiểu được**: Đóng góp tiềm năng của ECDR trên bản đồ nghiên cứu thế giới.
6. [`09_TMB_BOARD.md`](09_TMB_BOARD.md):  
   - **Nói cái gì**: Bảng quản lý tiến độ thực chứng **TrustMeBro**.  
   - **Đọc xong hiểu được**: Trạng thái hiện tại của từng Gate G0–G6.

---

## 2. Tiêu Chí Nghiệm Thu Tầng 0 (Exit Criteria)
- [x] Nắm rõ định nghĩa $X, \mathcal{P}, f_\theta, F, h(X), X^r, \mathcal{S}_r$.
- [x] Hiểu tại sao $\mathcal{P}$ bắt buộc phải đóng băng cố định.
- [x] Sẵn sàng chuyển tiếp sang **[Tầng 1: Nền Tảng Dữ Liệu](../01_data_foundations/README.md)**.
