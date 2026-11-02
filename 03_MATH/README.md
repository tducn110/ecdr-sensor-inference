# TẦNG 3: LÝ THUYẾT TOÁN HỌC & CHỨNG NHẬN AN TOÀN (TIER 3 - MATHEMATICAL THEORY)

> **Mục tiêu của Tầng 3**:  
> Làm chủ toàn bộ cơ sở lý thuyết toán học hình thức của ECDR: Hiểu cách chứng minh mọi mẫu trong miền $\mathcal{D}(X^r)$ giữ nguyên quyết định lớp $h(X) = c$, phương pháp lan truyền khoảng IBP, CROWN relaxation, giải quyết bẫy hình học chuẩn $L_2$ vs $L_\infty$ trong không gian 768 chiều, tham số hóa miền sóng $T_S$, và khế ước kỹ thuật Gate G3.

---

## 1. Trình Tự Đọc Tuyến Tính (Step-by-Step Reading Path)

1. [`01_04_THEORY.md`](01_04_THEORY.md):  
   - **Nói cái gì**: Khung lý thuyết toán học: Định nghĩa miền sự kiện $\mathcal{D}$, cận biên an toàn $g_{cj}(X)$, và định lý Soundness.  
   - **Đọc xong hiểu được**: Điều kiện toán học để tái sử dụng nhãn an toàn là $\min_{X \in \mathcal{D}} (F_c(X) - F_j(X)) > 0, \forall j \neq c$.
2. [`02_07_G3_DOMAIN_CONTRACT.md`](02_07_G3_DOMAIN_CONTRACT.md):  
   - **Nói cái gì**: Khế ước kỹ thuật Gate 3: Ràng buộc bất biến giữa Preprocessing, Normalization và Verifier.  
   - **Đọc xong hiểu được**: Khế ước sống còn: Mạng CNN và Verifier phải nhìn cùng một không gian đầu vào chuẩn hóa; không được nhầm lẫn giữa raw sensor và normalized tensor.
3. [`03_Week_3_Mathematical_Foundations.pdf`](03_Week_3_Mathematical_Foundations.pdf) ([Mã nguồn TeX](03_Week_3_Mathematical_Foundations.tex)):  
   - **Nói cái gì**: Báo cáo học thuật Tuần 3: Toàn văn các định lý, bất đẳng thức Lipschitz, IBP và cơ chế chống stale certificate.  
   - **Đọc xong hiểu được**: Cách thiết lập hệ thống chứng chỉ toán học có thể kiểm chứng độc lập.
4. [`04_ECDR_Mathematics_Explained.pdf`](04_ECDR_Mathematics_Explained.pdf) ([Mã nguồn TeX](04_ECDR_Mathematics_Explained.tex)):  
   - **Nói cái gì**: Cẩm nang giải thích toán học 29 trang: Giải nghĩa 24 giả thuyết H01–H24, ví dụ CNN nhỏ tính tay từng bước.  
   - **Đọc xong hiểu được**: Tự tay tính được GAP, logits, margins của một mạng CNN thu nhỏ và hiểu rõ phản ví dụ stale certificate.
5. [`05_ECDR_Mathematics_Concepts_104_Atlas.tex`](05_ECDR_Mathematics_Concepts_104_Atlas.tex):  
   - **Nói cái gì**: Bản đồ 104 khái niệm toán học tiếng Anh (mỗi trang 1 concept độc lập).  
   - **Đọc xong hiểu được**: Tra cứu nhanh bất kỳ khái niệm toán học nào trong đề tài.
6. [`06_SUPP02_Norm_Geometry.pdf`](06_SUPP02_Norm_Geometry.pdf) ([Mã nguồn TeX](06_SUPP02_Norm_Geometry.tex)):  
   - **Nói cái gì**: Chuyên đề hình học chuẩn (SUPP-02): $L_2$ vs $L_\infty$ trong $\mathbb{R}^{768}$, tỷ số $\sqrt{768} \approx 27.7$.  
   - **Đọc xong hiểu được**: Hiện tượng vỏ cầu cao chiều và lý do tại sao không thể dịch chuyển percentile ngây thơ qua các chuẩn khác nhau.
7. [`07_SUPP03_Certification_Methods.pdf`](07_SUPP03_Certification_Methods.pdf) ([Mã nguồn TeX](07_SUPP03_Certification_Methods.tex)):  
   - **Nói cái gì**: Chuyên đề phương pháp chứng nhận (SUPP-03): Quy tắc lan truyền khoảng IBP, CROWN relaxation và LiRPA framework.  
   - **Đọc xong hiểu được**: Sự đánh đổi giữa độ chặt (tightness) và chi phí tính toán giữa IBP và CROWN.
8. [`08_SUPP05_Wave_Domain.pdf`](08_SUPP05_Wave_Domain.pdf) ([Mã nguồn TeX](08_SUPP05_Wave_Domain.tex)):  
   - **Nói cái gì**: Chuyên đề miền sóng (SUPP-05): Toán tử dịch pha $T_S$, thành phần PCA bậc thấp và cận trôi dạt đa bước.  
   - **Đọc xong hiểu được**: Cách khai thác tính chất sóng tuần hoàn của bước chân để thắt chặt cận chứng nhận.
9. [`09_SUPP06_Float_Precision.pdf`](09_SUPP06_Float_Precision.pdf) ([Mã nguồn TeX](09_SUPP06_Float_Precision.tex)):  
   - **Nói cái gì**: Chuyên đề độ chính xác số học (SUPP-06): IEEE 754 float32 vs float64, machine epsilon và lượng tử hóa int8.  
   - **Đọc xong hiểu được**: Khi nào float32 làm hỏng chứng chỉ và cách containment lỗi số học.
10. [`10_script_11_gate_a_prototype.py`](10_script_11_gate_a_prototype.py):  
    - **Nói cái gì**: Prototype mã nguồn thực nghiệm cho Gate 3 verifier.

---

## 2. Tiêu Chí Nghiệm Thu Tầng 3 (Exit Criteria)
- [x] Hiểu công thức biên khoảng cách lớp $g_{cj}(X) = F_c(X) - F_j(X) > 0$.
- [x] Nắm vững cơ chế IBP và CROWN cho mạng Conv1d/ReLU.
- [x] Hiểu nghịch lý vỏ cầu $\mathbb{R}^{768}$ và toán tử dịch sóng $T_S$.
- [x] Sẵn sàng chuyển tiếp sang **[Tầng 4: Tổng Hợp & Đóng Gói Nghiên Cứu](../04_synthesis_and_gates/README.md)**.
