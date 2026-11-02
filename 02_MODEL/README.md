# TẦNG 2: KIẾN TRÚC MÔ HÌNH HỌC SÂU 1D-CNN (TIER 2 - DEEP LEARNING MODEL)

> **Mục tiêu của Tầng 2**:  
> Bóc tách kiến trúc mô hình học sâu `SmallCNN1D` từ nguyên lý đầu: Hiểu rõ từng tầng tích chập Conv1d, BatchNorm, MaxPool, Flatten, Linear; kiểm tra 282.918 tham số; đối chiếu số lượng phép nhân cộng (MACs) và độ trễ thực tế trên CPU (~0.24ms) để làm cơ sở cho bài toán tiết kiệm năng lượng.

---

## 1. Trình Tự Đọc Tuyến Tính (Step-by-Step Reading Path)

1. [`01_ECDR_CNN_Model_Analysis_EN.pdf`](01_ECDR_CNN_Model_Analysis_EN.pdf) ([Bản tiếng Việt](02_ECDR_CNN_Model_Analysis_VI.pdf)):  
   - **Nói cái gì**: Bóc tách chi tiết mô hình CNN qua 64 concepts: Receptive field, MACs, FLOPs, bộ nhớ và các đẳng thức toán học.  
   - **Đọc xong hiểu được**: Tại sao các tầng MaxPool gây khó khăn cho việc chứng nhận hình thức và cách tính toán chi phí inference.
2. [`03_backbone_model.py`](03_backbone_model.py):  
   - **Nói cái gì**: Mã nguồn PyTorch định nghĩa lớp `SmallCNN1D`: Conv1d `6→32→64→64`, BatchNorm, ReLU, 2 MaxPool, Flatten 2048, Linear 128, Linear 6.  
   - **Đọc xong hiểu được**: Dòng chảy tensor từ $(B, 6, 128)$ thành $(B, 6)$ logits.
3. [`04_backbone_config.yaml`](04_backbone_config.yaml):  
   - **Nói cái gì**: Tệp cấu hình siêu tham số huấn luyện (learning rate, batch size, optimizer, seeds).  
   - **Đọc xong hiểu được**: Thiết lập thí nghiệm chuẩn của Gate G2.
4. [`05_checkpoint_seed44_best.pt`](05_checkpoint_seed44_best.pt):  
   - **Nói cái gì**: Trọng số mô hình đã huấn luyện tốt nhất được chọn bằng validation (Seed 44).  
   - **Đọc xong hiểu được**: Điểm đóng băng của mô hình (Frozen Reference).
5. [`06_g2_model_summary.json`](06_g2_model_summary.json) & [`07_g2_cpu_latency.json`](07_g2_cpu_latency.json):  
   - **Nói cái gì**: Artifacts ghi nhận số lượng tham số (282.918) và phân phối độ trễ CPU batch=1 (mean 0.2511 ms, p50 0.2357 ms).  
   - **Đọc xong hiểu được**: Cột mốc thời gian cơ sở $\tau_{\text{backbone}}$ mà phương pháp reuse cần vượt qua.
6. [`08_g2_metrics.json`](08_g2_metrics.json) & [`10_confusion_matrix.png`](10_confusion_matrix.png):  
   - **Nói cái gì**: Độ chính xác và ma trận nhầm lẫn của mô hình: Test Macro-F1 $0.9354$ trên 2.874 pure windows.  
   - **Đọc xong hiểu được**: Khả năng phân loại thực tế của mô hình nền.

---

## 2. Các Script Huấn Luyện & Đo Độ Trễ Có Thể Thực Thi
- [`script_07_count_params.py`](script_07_count_params.py): Đếm chính xác từng tham số của từng tầng trong mô hình.
- [`script_08_train_backbone.py`](script_08_train_backbone.py): Script huấn luyện mô hình SmallCNN1D trên 3 hạt giống (42, 43, 44).
- [`script_09_evaluate_backbone.py`](script_09_evaluate_backbone.py): Đánh giá mô hình và vẽ ma trận nhầm lẫn.
- [`script_10_profile_inference.py`](script_10_profile_inference.py): Đo đạc độ trễ inference forward trên CPU.

---

## 3. Tiêu Chí Nghiệm Thu Tầng 2 (Exit Criteria)
- [x] Nắm rõ cấu trúc mạng SmallCNN1D và 282.918 tham số.
- [x] Hiểu mốc độ trễ cơ sở $\tau_{\text{backbone}} \approx 0.24\text{ms}$.
- [x] Xác định checkpoint `backbone_best.pt` là thực thể bất biến.
- [x] Sẵn sàng chuyển tiếp sang **[Tầng 3: Lý Thuyết Toán Học & Chứng Nhận An Toàn](../03_math_certification/README.md)**.
