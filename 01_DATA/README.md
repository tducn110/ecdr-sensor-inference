# TẦNG 1: NỀN TẢNG DỮ LIỆU & XỬ LÝ TÍN HIỆU (TIER 1 - DATA & SIGNALS)

> **Mục tiêu của Tầng 1**:  
> Nắm vững cấu trúc tập dữ liệu UCI HAPT (61 recordings, 12 classes, 30 subjects), hiểu rõ lý do cắt bỏ các lớp chuyển tiếp để chỉ tập trung vào 6 lớp cơ bản `pure-only`, phát hiện sự sụp đổ của giả định i.i.d. do hiện tượng tự tương quan cực cao ($r \approx 0.985$), và làm quen với các script audit dữ liệu.

---

## 1. Trình Tự Đọc Tuyến Tính (Step-by-Step Reading Path)

1. [`01_03_DATA_AUDIT.md`](01_03_DATA_AUDIT.md):  
   - **Nói cái gì**: Báo cáo kiểm toán dữ liệu UCI HAPT, lịch sử phát hiện và đính chính lỗi script 05 (lỗi index dòng khiến 98.2% cặp đọc nhầm exp01).  
   - **Đọc xong hiểu được**: Cấu trúc phân chia tập train/val/test theo người dùng (disjoint subjects) và số lượng cửa sổ chuẩn.
2. [`02_G2_CLASS_DECISION.md`](02_G2_CLASS_DECISION.md):  
   - **Nói cái gì**: Quyết định chiến lược: Đóng băng tập dữ liệu ở 6 lớp tĩnh/động cơ bản (`K=6 pure-only`).  
   - **Đọc xong hiểu được**: Lý do loại bỏ 6 lớp chuyển tiếp thế đứng (postural transitions) vì số lượng mẫu quá ít ($<2\%$), không đủ hỗ trợ chứng nhận an toàn.
3. [`03_Week_2_Data_Foundations.pdf`](03_Week_2_Data_Foundations.pdf) ([Mã nguồn TeX](03_Week_2_Data_Foundations.tex)):  
   - **Nói cái gì**: Báo cáo học thuật Tuần 2: Phân tích cửa sổ thời gian 128 điểm mẫu, bước trượt 64 điểm (overlap 50%) và hình học chuẩn hóa.  
   - **Đọc xong hiểu được**: Đẳng thức overlap $X'[:, 0:64] = X[:, 64:128]$ và cơ chế căn chỉnh tọa độ.
4. [`04_ECDR_Data_Analysis_EN.pdf`](04_ECDR_Data_Analysis_EN.pdf) ([Bản tiếng Việt](05_ECDR_Data_Analysis_VI.pdf)):  
   - **Nói cái gì**: Báo cáo phân tích dữ liệu chuyên sâu 28 trang: Phân tích 4 độ đo vật lý, hình học elip biến dạng dưới chuẩn hóa z-score.  
   - **Đọc xong hiểu được**: Độ phủ thực tế của PCA và tại sao Explained Variance không đồng nghĩa với Residual Bound.
5. [`06_SUPP04_Autocorrelation.pdf`](06_SUPP04_Autocorrelation.pdf) ([Mã nguồn TeX](06_SUPP04_Autocorrelation.tex)):  
   - **Nói cái gì**: Chuyên đề tự tương quan (SUPP-04): Công thức Pearson, ACF, và phản ví dụ dốc tuyến tính.  
   - **Đọc xong hiểu được**: Tại sao tín hiệu cảm biến liên tục có tương quan lag-1 cực cao ($r \approx 0.985$) nhưng hai cửa sổ overlap 50% vẫn có thể mang nhãn khác nhau.
6. [`07_SUPP07_Statistical_Dependence.pdf`](07_SUPP07_Statistical_Dependence.pdf) ([Mã nguồn TeX](07_SUPP07_Statistical_Dependence.tex)):  
   - **Nói cái gì**: Chuyên đề phụ thuộc thống kê (SUPP-07): Tại sao giả định i.i.d. thất bại, hiện tượng lạm phát phương sai $\times 65$.  
   - **Đọc xong hiểu được**: Cách áp dụng Block Bootstrap để ước lượng khoảng tin cậy chuẩn xác cho chuỗi thời gian phụ thuộc.

---

## 2. Các Script Kiểm Toán Dữ Liệu Có Thể Chạy Ngay
- [`script_01_fetch_hapt.py`](script_01_fetch_hapt.py): Kiểm tra dữ liệu raw gốc UCI HAPT trong [`data_dir`](data_dir).
- [`script_02_validate_raw.py`](script_02_validate_raw.py): Xác thực tính toàn vẹn các tệp raw signal.
- [`script_03_build_windows.py`](script_03_build_windows.py): Cắt cửa sổ 128 mẫu trượt 64 mẫu.
- [`script_04_audit_splits.py`](script_04_audit_splits.py): Kiểm toán phân chia tập train/val/test theo người dùng.
- [`script_05_audit_temporal_events.py`](script_05_audit_temporal_events.py): Đo đạc độ tương quan và overlap chuỗi thời gian.
- [`script_06_audit_class_support.py`](script_06_audit_class_support.py): Đo đạc số lượng mẫu cho 12 lớp.
- [`script_06b_corr_semantics.py`](script_06b_corr_semantics.py): Tính toán ma trận tự tương quan ngữ nghĩa.

---

## 3. Tiêu Chí Nghiệm Thu Tầng 1 (Exit Criteria)
- [x] Hiểu cấu trúc 6 kênh cảm biến và 2.874 pure test windows.
- [x] Hiểu lý do toán học và kỹ thuật của việc chọn $K=6$ thay vì $K=12$.
- [x] Nắm vững hiện tượng tự tương quan và thất bại của i.i.d.
- [x] Sẵn sàng chuyển tiếp sang **[Tầng 2: Kiến Trúc Học Sâu & Mô Hình CNN](../02_model_cnn/README.md)**.
