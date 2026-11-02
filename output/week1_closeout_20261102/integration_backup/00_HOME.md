---
title: ECDR Research Vault Home
tags:
  - ecdr
  - research
  - deep-learning
  - formal-verification
  - time-series
  - trustmebro
created: 2026-10-29
---

# 🌌 ECDR RESEARCH VAULT — TRANG CHỦ & BẢN ĐỒ ĐIỀU HƯỚNG
## Event-Conditioned Certified Decision Reuse in Sensor Time-Series Streams

> [!abstract] Chào Mừng Đến Với Kho Lưu Trữ Nghiên Cứu ECDR Trong Obsidian!
> Toàn bộ dự án nghiên cứu **ECDR** đã được cấu trúc hóa thành một **Obsidian Vault hoàn chỉnh**, kết nối đa chiều bằng Wikilinks, sơ đồ tư duy Canvas tương tác, và hệ thống phân tầng tri thức tuyến tính:
> **"Đọc xong [X] (DONE) $\rightarrow$ Nắm vững nền tảng [Y] $\rightarrow$ Mở khóa hiểu sâu [Z]"**.

---

> [!warning] ĐIỀU KIỆN TIÊN QUYẾT: Đọc Ký Hiệu Trước Khi Vào Week 1!
> Nếu bạn đọc ngay vào các tệp lý thuyết trong thư mục `me/` hay bản in Tuần 1, hãy ghi nhớ các quy ước toán học cốt lõi:
> - **$X \in \mathbb{R}^{6 \times 128} \cong \mathbb{R}^{768}$**: Cửa sổ tín hiệu cảm biến thô (3 trục gia tốc $g$, 3 trục con quay $\text{rad/s}$ trong $2{,}56\text{s}$ ở $50\text{Hz}$).
> - **$\mathcal{P}$ (hoặc $N$)**: Phép tiền xử lý chuẩn hóa z-score **cố định** từng kênh: $Z = \mathcal{P}(X) = (X - \mu)/\sigma$, với $\mu, \sigma$ đóng băng từ tập huấn luyện. *(Bắt buộc cố định để không làm trôi dạt chứng chỉ an toàn!)*
> - **$Z$**: Tensor đầu vào chuẩn hóa không thứ nguyên đưa vào mạng nơ-ron $f_\theta$.
> - **$f_\theta$**: Mạng nơ-ron sâu nền tảng [[scripts/backbone.py|SmallCNN1D]] (282.918 tham số, đóng băng ở Gate G2).
> - **$F = f_\theta \circ \mathcal{P}$**: Chuỗi suy luận tham chiếu end-to-end từ tín hiệu thô $X$ sang logits $\mathbb{R}^K$.
> - **$h(X) = \arg\max_{k} F_k(X)$**: Nhãn quyết định lớp cuối cùng.
> - **$X^r$**: Cửa sổ mỏ neo tin cậy (trusted anchor) đã được chạy suy luận thật bằng $F$.
> - **$\mathcal{S}_r$**: Miền quyết định an toàn xung quanh mỏ neo thỏa mãn: $X \in \mathcal{S}_r \implies h(X) = h(X^r)$.
> - Chi tiết xem tại: [[README.md#⚠️-instruction-0-bảng-ký-hiệu-tiền-quyết--bản-đồ-tiền-giả-định|Instruction 0 trong README]].

---

> [!tip] 3 Cách Điều Hướng Nhanh Trong Obsidian
> 1. 🗺️ **Mở Bản Đồ Tư Duy Tương Tác**: Nhấp vào [[ECDR_Knowledge_Map.canvas|ECDR Knowledge Map Canvas]] để xem toàn cảnh 5 tầng tri thức trực quan.
> 2. 🕸️ **Mở Graph View**: Nhấn `Ctrl + G` (hoặc `Cmd + G`) để khám phá mạng lưới liên kết giữa các tài liệu khoa học được gán màu theo tầng.
> 3. 🔖 **Dùng Bookmarks**: Mở thanh dấu trang ở cột bên trái để nhảy tức thì đến các tệp trọng tâm.

---

## 📚 Hệ Thống 5 Tầng Tri Thức (Tiered Architecture)

Mỗi tầng tương ứng với một giai đoạn tiếp thu kiến thức. Hãy đi tuần tự từ Tầng 0 đến Tầng 4:

| Tầng | Thư mục & Tài liệu điều hướng | Nội dung trọng tâm | Trạng thái nghiệm thu |
| :---: | :--- | :--- | :---: |
| **Tầng 0** | [[00_OVERVIEW/README.md\|Tầng 0: Tổng Quan & Khởi Đầu]] | Bài toán ECDR, RQ1 (Decision Reuse), RQ2 (Net Savings), Vòng đời Anchor 4 trạng thái | ✅ Đã ban hành |
| **Tầng 1** | [[01_DATA/README.md\|Tầng 1: Dữ Liệu UCI HAPT & Tín Hiệu]] | 61 recordings, 12 classes vs $K=6$ pure-only, Tự tương quan $r \approx 0.985$, Thất bại của i.i.d. | ✅ **G1 REVISED** |
| **Tầng 2** | [[02_MODEL/README.md\|Tầng 2: Mô Hình Học Sâu 1D-CNN]] | SmallCNN1D (282k params), Checkpoint seed 44 (F1 0.9354), FLOPs, Trễ forward CPU ~0.24ms | ✅ **G2 PASS** |
| **Tầng 3** | [[03_MATH/README.md\|Tầng 3: Lý Thuyết Toán Học & Chứng Nhận]] | Khế ước miền G3, Lan truyền khoảng IBP, CROWN relaxation, Hình học chuẩn $\mathbb{R}^{768}$, Sóng $T_S$ | ⏳ **G3 PENDING** |
| **Tầng 4** | [[04_SYNTHESIS/README.md\|Tầng 4: Tổng Hợp, Gates & Công Bố]] | Báo cáo lý thuyết 31 trang, Dossier tổng hợp, Tiêu chí nghiệm thu Gate G0–G6, Papers quốc tế | ⏳ **G4–G6 PENDING** |

---

## 📑 Thư Viện 21 Tệp PDF Đọc Tuyến Tính (`PDF_LIBRARY/`)

Toàn bộ tài liệu PDF đã được đánh số thứ tự từ `01` đến `21`. Nhấp vào từng liên kết để mở trình đọc PDF tích hợp trong Obsidian:

1. [[PDF_LIBRARY/01_Week_1_Project_Brief.pdf|01. Week 1: Project Brief & Bối Cảnh Nghiên Cứu]]
2. [[PDF_LIBRARY/02_SUPP01_ECDR_Primer.pdf|02. SUPP-01: Chuyên Đề Sơ Nhập ECDR & Vòng Đời Anchor]]
3. [[PDF_LIBRARY/03_Week_2_Data_Foundations.pdf|03. Week 2: Cơ Sở Dữ Liệu Cảm Biến UCI HAPT]]
4. [[PDF_LIBRARY/04_ECDR_Data_Analysis_EN.pdf|04. HAPT Data Analysis & Methodology (28 Trang EN)]]
5. [[PDF_LIBRARY/05_ECDR_Data_Analysis_VI.pdf|05. Phân Tích Dữ Liệu HAPT Bản Tiếng Việt (Lưu Trữ)]]
6. [[PDF_LIBRARY/06_SUPP04_Autocorrelation.pdf|06. SUPP-04: Tự Tương Quan Chuỗi Thời Gian & Phản Ví Dụ Dốc]]
7. [[PDF_LIBRARY/07_SUPP07_Statistical_Dependence.pdf|07. SUPP-07: Sự Sụp Đổ Của Giả Định i.i.d. & Block Bootstrap]]
8. [[PDF_LIBRARY/08_ECDR_CNN_Model_Analysis_EN.pdf|08. Bóc Tách Mô Hình 1D-CNN Từ Nguyên Lý Đầu (64 Concepts EN)]]
9. [[PDF_LIBRARY/09_ECDR_CNN_Model_Analysis_VI.pdf|09. Bóc Tách Mô Hình CNN Bản Tiếng Việt (Lưu Trữ)]]
10. [[PDF_LIBRARY/10_Week_3_Mathematical_Foundations.pdf|10. Week 3: Nền Tảng Toán Học & Cận Biên An Toàn]]
11. [[PDF_LIBRARY/11_SUPP02_Norm_Geometry.pdf|11. SUPP-02: Hình Học Chuẩn L2 vs L-inf Trong Không Gian 768 Chiều]]
12. [[PDF_LIBRARY/12_SUPP03_Certification_Methods.pdf|12. SUPP-03: Phương Pháp Chứng Nhận IBP, CROWN & LiRPA]]
13. [[PDF_LIBRARY/13_SUPP05_Wave_Domain.pdf|13. SUPP-05: Tham Số Hóa Miền Sóng & Toán Tử Dịch Pha TS]]
14. [[PDF_LIBRARY/14_SUPP06_Float_Precision.pdf|14. SUPP-06: Độ Chính Xác Số Học IEEE 754 & Lượng Tử Hóa int8]]
15. [[PDF_LIBRARY/15_ECDR_Mathematics_Explained.pdf|15. Cẩm Nang Giải Thích Toàn Bộ Toán Học (29 Trang, H01–H24)]]
16. [[PDF_LIBRARY/16_ECDR_Weeks_1_2_3_Theory_Full.pdf|16. Toàn Văn Báo Cáo Lý Thuyết 3 Tuần (31 Trang Hoàn Chỉnh)]]
17. [[PDF_LIBRARY/17_Dossier_Scientific_Report.pdf|17. Hồ Sơ Báo Cáo Nghiên Cứu Khoa Học Dossier LaTeX]]
18. [[PDF_LIBRARY/18_LipCache_Xiang2026_Paper.pdf|18. Bài Báo Nền Tảng LipCache (Xiang et al., arXiv 2026)]]
19. [[PDF_LIBRARY/19_STBP_Varghese2026_Paper.pdf|19. Bài Báo Tham Chiếu Chuỗi Thời Gian STBP (Varghese 2026)]]
20. [[PDF_LIBRARY/20_2605.21724v1_Paper.pdf|20. Tài Liệu Tham Khảo Mở Rộng 1]]
21. [[PDF_LIBRARY/21_got_Paper.pdf|21. Tài Liệu Tham Khảo Mở Rộng 2]]

---

## 🎯 Bảng Quản Lý Tiến Độ TrustMeBro: [[TMB.md]]

Tiến độ thực chứng hiện tại theo chuẩn **TrustMeBro**:
- ✅ **Gate G0 (PASS)**: Xác lập RQ và phân loại 4 trục đối chiếu với LipCache.
- ✅ **Gate G1 (COMPLETE_WITH_REVISIONS)**: Kiểm toán tập UCI HAPT, đính chính lỗi Script 05.
- ✅ **Gate G2 (PASS)**: Huấn luyện và đóng băng SmallCNN1D (K=6 pure-only, Macro-F1 0.9354).
- ⏳ **Gate G3 (PENDING)**: Khế ước miền chuẩn hóa và verifier tối thiểu.
- ⏳ **Gate G4 (PENDING)**: Đánh giá đóng góp của miền Event-Conditioned.
- ⏳ **Gate G5 (PENDING)**: Đo đạc độ trễ toàn chuỗi end-to-end trên phần cứng thực.
- ⏳ **Gate G6 (PENDING)**: Hoàn thiện bài báo và gói mã nguồn tái lập.

---
*Vault được tối ưu hóa cho Obsidian với KaTeX LaTeX rendering, Mermaid diagrams, Canvas maps và Wikilinks 2 chiều.*
