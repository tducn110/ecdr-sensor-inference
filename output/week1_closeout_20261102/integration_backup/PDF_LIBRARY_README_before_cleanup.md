# TỦ SÁCH PDF TOÀN DIỆN (CHRONOLOGICAL PDF READING SHELF)

> **Current Week 1 authority:** [canonical English source](../00_OVERVIEW/02_Week_1_Project_Brief.tex). `01_Week_1_Project_Brief.pdf` is a generated derivative and dated/Vietnamese PDFs are historical; rebuild status belongs to the owning pipeline.

> Thư mục này tập hợp **toàn bộ các tệp tài liệu PDF và bài báo nghiên cứu** của dự án ECDR, được đánh số từ `01` đến `21` theo đúng trình tự đọc tuyến tính khuyến nghị.  
> Bạn chỉ cần mở các tệp PDF theo thứ tự số tăng dần để tiếp cận toàn bộ dự án từ tổng quan bài toán, cơ sở dữ liệu, kiến trúc học sâu đến lý thuyết chứng nhận hình thức.

---

## Danh Mục Đọc Theo Thứ Tự Tuyến Tính

| STT | Tên tệp PDF | Nội dung truyền tải | Đọc xong hiểu được gì? |
| :---: | :--- | :--- | :--- |
| **01** | [`01_Week_1_Project_Brief.pdf`](01_Week_1_Project_Brief.pdf) | Tóm tắt đề tài ECDR: Bối cảnh, bài toán inference trên thiết bị cảm biến | Hiểu vì sao chạy mạng liên tục gây tốn pin và cơ hội tái sử dụng quyết định |
| **02** | [`02_SUPP01_ECDR_Primer.pdf`](02_SUPP01_ECDR_Primer.pdf) | Chuyên đề sơ nhập ECDR: Vòng đời Anchor 4 trạng thái, cơ chế an toàn | Hiểu vòng đời mỏ neo và sự khác biệt giữa "Cache mềm" vs "Chứng chỉ an toàn" |
| **03** | [`03_Week_2_Data_Foundations.pdf`](03_Week_2_Data_Foundations.pdf) | Cơ sở dữ liệu: Phân tích tập dữ liệu UCI HAPT, cắt cửa sổ và chia tập | Nắm được cấu trúc cửa sổ tín hiệu cảm biến 6 kênh, 128 điểm mẫu ở 50Hz |
| **04** | [`04_ECDR_Data_Analysis_EN.pdf`](04_ECDR_Data_Analysis_EN.pdf) | Báo cáo phân tích dữ liệu chuyên sâu 28 trang (Bản chuẩn tiếng Anh) | Hiểu sâu 4 độ đo vật lý, hình học chuẩn hóa z-score và độ phủ PCA |
| **05** | [`05_ECDR_Data_Analysis_VI.pdf`](05_ECDR_Data_Analysis_VI.pdf) | Báo cáo phân tích dữ liệu bản tiếng Việt (Lưu trữ lịch sử) | Đối chiếu thuật ngữ và phân tích dữ liệu song ngữ |
| **06** | [`06_SUPP04_Autocorrelation.pdf`](06_SUPP04_Autocorrelation.pdf) | Chuyên đề tự tương quan: Pearson ACF và phản ví dụ dốc tuyến tính | Hiểu vì sao tương quan cao ($r \approx 0.985$) nhưng overlap 50% vẫn đổi nhãn |
| **07** | [`07_SUPP07_Statistical_Dependence.pdf`](07_SUPP07_Statistical_Dependence.pdf) | Chuyên đề phụ thuộc thống kê: Sự sụp đổ i.i.d., lạm phát phương sai $\times 65$ | Hiểu cách dùng Block Bootstrap để ước lượng sai số chuẩn xác |
| **08** | [`08_ECDR_CNN_Model_Analysis_EN.pdf`](08_ECDR_CNN_Model_Analysis_EN.pdf) | Bóc tách mô hình 1D-CNN từ nguyên lý đầu (64 concepts tiếng Anh) | Hiểu chi tiết 282k tham số, MACs, FLOPs, bộ nhớ và receptive field của Conv1D |
| **09** | [`09_ECDR_CNN_Model_Analysis_VI.pdf`](09_ECDR_CNN_Model_Analysis_VI.pdf) | Bóc tách mô hình CNN bản tiếng Việt (Lưu trữ lịch sử) | Đối chiếu khái niệm kiến trúc mô hình học sâu |
| **10** | [`10_Week_3_Mathematical_Foundations.pdf`](10_Week_3_Mathematical_Foundations.pdf) | Cơ sở toán học ECDR: Hàm tham chiếu, cận biên an toàn, định lý Soundness | Hiểu điều kiện cần để tái sử dụng: Cận dưới margin logit $\min_{k \neq y} (z_y - z_k) > 0$ |
| **11** | [`11_SUPP02_Norm_Geometry.pdf`](11_SUPP02_Norm_Geometry.pdf) | Chuyên đề hình học chuẩn: $L_2$ vs $L_\infty$ trong $\mathbb{R}^{768}$, tỷ số $\sqrt{768} \approx 27.7$ | Hiểu tại sao việc chuyển đổi bán kính qua các chuẩn khác nhau là sai lầm chết người |
| **12** | [`12_SUPP03_Certification_Methods.pdf`](12_SUPP03_Certification_Methods.pdf) | Chuyên đề phương pháp chứng nhận: Quy tắc truyền khoảng IBP & CROWN | Hiểu cách tính cận an toàn qua tầng tuyến tính và hàm kích hoạt phi tuyến ReLU |
| **13** | [`13_SUPP05_Wave_Domain.pdf`](13_SUPP05_Wave_Domain.pdf) | Chuyên đề miền sóng: Toán tử dịch pha $T_S$ và cận sai số trôi dạt | Hiểu cách khai thác tính chất sóng tuần hoàn của bước chân để thắt chặt cận |
| **14** | [`14_SUPP06_Float_Precision.pdf`](14_SUPP06_Float_Precision.pdf) | Chuyên đề độ chính xác số học: IEEE 754 float32 vs float64, lượng tử hóa | Hiểu rủi ro làm tròn số trong verifier và cách cô lập sai số trên phần cứng nhúng |
| **15** | [`15_ECDR_Mathematics_Explained.pdf`](15_ECDR_Mathematics_Explained.pdf) | Cẩm nang giải thích toàn bộ toán học 29 trang: Giải nghĩa H01–H24, ví dụ tính tay | Nắm vững từng phép tính vi mô: Tự tay tính được GAP, logits và margins của CNN |
| **16** | [`16_ECDR_Weeks_1_2_3_Theory_Full.pdf`](16_ECDR_Weeks_1_2_3_Theory_Full.pdf) | Báo cáo lý thuyết tổng hợp toàn diện 3 tuần (31 trang) | Nắm vững toàn bộ bức tranh khoa học hoàn chỉnh |
| **17** | [`17_Dossier_Scientific_Report.pdf`](17_Dossier_Scientific_Report.pdf) | Hồ sơ báo cáo khoa học tổng hợp LaTeX | Tài liệu lưu trữ khoa học chính thức của đề tài |
| **18** | [`18_LipCache_Xiang2026_Paper.pdf`](18_LipCache_Xiang2026_Paper.pdf) | Bài báo LipCache (Xiang et al., arXiv 2026) | Đối chiếu phương pháp caching dựa trên hằng số Lipschitz toàn cục |
| **19** | [`19_STBP_Varghese2026_Paper.pdf`](19_STBP_Varghese2026_Paper.pdf) | Bài báo tham chiếu STBP (Varghese et al., arXiv 2026) | Đối chiếu phương pháp xử lý chuỗi thời gian cảm biến |
| **20** | [`20_2605.21724v1_Paper.pdf`](20_2605.21724v1_Paper.pdf) | Tài liệu nghiên cứu mở rộng 1 | Nghiên cứu đối chiếu độc lập |
| **21** | [`21_got_Paper.pdf`](21_got_Paper.pdf) | Tài liệu nghiên cứu mở rộng 2 | Nghiên cứu đối chiếu độc lập |
