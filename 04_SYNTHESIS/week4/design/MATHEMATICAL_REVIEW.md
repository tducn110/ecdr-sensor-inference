# 04_SYNTHESIS/week4/design/MATHEMATICAL_REVIEW.md — Independent Mathematical Review & Comprehensive Hypothesis Ledger (H01–H24)

**Task:** W4-027 (Milestone M4)  
**Tác giả:** Antigravity (Pair Programming Agent / Independent Mathematical Reviewer)  
**Run ID:** week4_initial_20261103  
**Trạng thái:** AUTHORITATIVE COMPREHENSIVE REVIEW  
**Phạm vi:** 24 giả thuyết khoa học toàn bộ Week 4 (H01–H24)  

---

## 1. Mục Đích & Chuẩn Mực Đánh Giá Độc Lập

Báo cáo này thực hiện kiểm định toán học và thực nghiệm độc lập toàn bộ 24 giả thuyết khoa học (**H01–H24**) của đề tài ECDR (Event-triggered Certified Decision Reuse) trên mô hình mạng nơ-ron tích chập 1 chiều chuẩn hóa `SmallCNN1D` ($K=12$ lớp, $283,692$ tham số) và tập dữ liệu gia tốc kế / con quay hồi chuyển HAPT (Human Activities and Postural Transitions).

Tuân thủ nghiêm ngặt **`<RULE[user_global]>`** và các nguyên tắc khoa học:
- **Phân định rạch ròi 4 cấp độ nhận thức:**
  - `[A]` Kết quả thực nghiệm / runtime đã đo đạc (`Source / Runtime Verified`).
  - `[B]` Bằng chứng toán học hình thức (`Official Math / Formal Theorems`).
  - `[C]` Suy luận & Phản chứng (`Inference & Falsification Witnesses`).
  - `[D]` Khuyến nghị kỹ thuật (`Engineering Recommendations`).
- **Bảo toàn tính trung thực của các phản chứng (Falsifications):** Tuyệt đối không thay đổi câu chữ, không nới lỏng giả thiết, và không "gọt chân cho vừa giày" đối với các giả thuyết đã bị bác bỏ như H02, H03, H07, H14, H15, H19, H20, H22, H23.
- **Nguyên tắc Fail-Closed:** Khi thiếu bằng chứng toán học hoặc khi chưa đóng được khoảng cách số học (numerical bridge), cổng năng lực tái sử dụng phải giữ đóng (`Gate Closed / Fail-Closed`).

---

## 2. Bảng Tổng Hợp Toàn Bộ 24 Giả Thuyết (Ledger Table H01–H24)

| ID | Tên Giả Thuyết & Bản Chất | Loại Phát Biểu | Trạng Thái Week 3 | Trạng Thái Week 4 (Sau Review Độc Lập) | Bằng Chứng / Nhân Chứng Cốt Lõi | Task Chủ Chốt |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **H01** | Trùng khớp dịch chuỗi thời gian (Shift-Aligned Overlap) | Định lý cấu trúc | `DERIVED + VERIFIED` | **DERIVED_AND_VERIFIED** | 100% khớp bit-exact tiền tố 64 mẫu trên 5,034 cặp cửa sổ kế tiếp | W4-001, W4-007, W4-015 |
| **H02** | 50% overlap không chặn khoảng cách Euclid | Phản chứng | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_UNCONDITIONAL** | Nhân chứng dốc dốc (Ramp witness) & dao động điều hòa đạt delta tới $2A$ | W4-002, W4-015, W4-018 |
| **H03** | Cùng nhãn hoạt động không đảm bảo cùng output CNN | Phản chứng | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_UNCONDITIONAL** | 78 ca đổi nhãn dự đoán trên 2,661 cặp cùng nhãn; witness row 682 $\to$ 683 | W4-023, W4-026 |
| **H04** | Sự tồn tại lân cận bất biến từ winner nghiêm ngặt | Định lý tô pô | `DERIVED (REAL)` | **DERIVED_REAL_CONDITIONAL_FLOAT** | Tính liên tục của các toán tử tích chập / ReLU bảo đảm tồn tại quả cầu $B(x^*, r)$ | W4-005, W4-011, W4-012 |
| **H05** | Đủ 11 cận lề dưới dương bảo đảm bất biến vùng | Định lý đúng đắn | `DERIVED (REAL)` | **DERIVED_REAL_AND_FLOAT32_GATE** | $\inf_{x \in \mathcal{D}} (f_{c^*}(x) - f_j(x)) > 0, \forall j \ne c^*$; kiểm chứng 100% agreement | W4-005, W4-010, W4-023 |
| **H06** | Cận Lipschitz trên chặn bán kính hình cầu an toàn | Định lý chặn trên | `CONDITIONAL` | **DERIVED_WITH_STAGED_NORMS** | Tích chuẩn phổ ma trận trọng số $\prod \|W_l\|$; bán kính tính toán $\sim 10^{-8}$ | W4-006, W4-010, W4-012 |
| **H07** | IBP không đầy đủ (Incomplete do wrapping loss) | Phản chứng | `FALSIFIED_COMPLETENESS` | **FALSIFIED_COMPLETENESS** | Toy model $h_1 = h_2 = \operatorname{ReLU}(t+1)$ chứng minh khoảng trống triệt tiêu 2.0 | W4-006, W4-010, W4-017 |
| **H08** | Vector Direct Margin chặt hơn cận logit riêng rẽ | Định lý chặn chặt | `DERIVED` | **DERIVED_AND_VERIFIED** | Bất đẳng thức tam giác: gom $W_c - W_j$ trước khi interval hóa loại bỏ 149.47 hull loss | W4-011, W4-014, W4-017 |
| **H09** | CROWN chặt hơn IBP nhưng chi phí tính toán cao hơn | Độ phức tạp & Cận | `NOT_ESTABLISHED` | **EMPIRICALLY_VERIFIED** | auto-LiRPA cho cận chặt hơn IBP nhưng tốn gấp 5-10x thời gian tính toán | W4-013, W4-014, W4-028 |
| **H10** | Không gian con PCA nắm bắt tương quan cảm biến | Cấu trúc thực nghiệm | `VERIFIED_FINITE` | **VERIFIED_FINITE** | PCA $K=8$ giải thích 98.4% phương sai; cơ sở trực chuẩn chặt hơn cơ sở ngẫu nhiên ~57 units | W4-002, W4-016, W4-018 |
| **H11** | So sánh miền hình học phải cùng thể tích / budget | Phương pháp luận | `NOT_ESTABLISHED` | **ESTABLISHED** | Miền wave cố định tiền tố có thể tích suy biến ($0$ trong $\mathbb{R}^{768}$); cấm so sánh bán kính thô | W4-015, W4-017, W4-018 |
| **H12** | Tiền tố cố định giảm số chiều tự do từ 768 về 384 | Định lý cấu trúc | `DERIVED` | **DERIVED_AND_VERIFIED** | Cố định 64 mẫu tiền tố trên 6 kênh loại bỏ 384 chiều tự do tại horizon $h=1$ | W4-007, W4-015, W4-025 |
| **H13** | Khớp đặc trưng trung gian tích chập 1D trên mạng thực | Định lý tương đương | `DERIVED + VERIFIED` | **DERIVED_AND_BITWISE_VERIFIED** | Trường cảm nhận Conv3 `[4j-8, 4j+11]`, stride 4; 3.86 triệu số thực khớp bit-exact | W4-019, W4-020, W4-021 |
| **H14** | Công thức sai phân GAP không áp dụng cho Flatten-MLP | Giới hạn kiến trúc | `NOT_APPLICABLE` | **ESTABLISHED_NEGATIVE** | Đầu phân loại SmallCNN1D là Flatten-MLP nhạy cảm vị trí, không có lớp Global Avg Pool | W4-019, W4-020, W4-021 |
| **H15** | Cấm dời tâm (Re-centering) chứng chỉ sau khi hit | Phản chứng & Chính sách | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_RECENTERING_ESTABLISHED_FIXED** | Nhân chứng trôi dạt tích lũy (Cumulative drift); bắt buộc neo cứng anchor và hạn dùng xác định | W4-022, W4-023, W4-025 |
| **H16** | Chứng chỉ neo cố định bảo toàn tính đúng đắn đa bước | Định lý an toàn | `CONDITIONAL` | **DERIVED_AND_VERIFIED** | Không cập nhật anchor khi hit; chứng chỉ hết hạn xác định sau $\text{anchor\_step} + \text{max\_age}$ | W4-022, W4-023, W4-025 |
| **H17** | Tập reachable đa bước suy giảm độ chặt theo horizon | Định lý truy hồi | `CONDITIONAL` | **DERIVED_AND_VERIFIED** | Horizon $h=1$ có 384 free dims; horizon $h=2$ thay thế toàn bộ 128 mẫu (768 free dims) | W4-015, W4-025 |
| **H18** | Kiểm tra thành viên hộp tọa độ có độ phức tạp $\Theta(m)$ | Độ phức tạp tính toán | `DERIVED` | **DERIVED_AND_VERIFIED** | So sánh tọa độ độc lập trên $m=768$ chiều tốn $\sim 0.014$ ms trong NumPy | W4-007, W4-018, W4-024 |
| **H19** | Tính đúng đắn toán học không kéo theo độ bao phủ hữu ích | Phản chứng | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_IMPLICATION** | Quả cầu bán kính $10^{-4}$ đúng đắn 100% nhưng đạt 0.0% tỷ lệ tái sử dụng trên dữ liệu thực | W4-018, W4-023, W4-025 |
| **H20** | Chi phí assertion và verifier có thể triệt tiêu tốc độ | Phản chứng | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_IMPLICATION_EMPIRICALLY_VERIFIED** | Băm SHA-256 toàn bộ weights mỗi query tốn 1.394 ms (76.8% latency, chậm hơn 5.98x so với core) | W4-024, W4-026 |
| **H21** | Kiến trúc Owned-Reference giải phóng chi phí danh tính | Phát minh kiến trúc | `NOT_ESTABLISHED` | **DERIVED_AND_VERIFIED** | Đóng gói trong tiến trình đạt tốc độ 0.439 ms (tăng tốc 4.13x so với M01) | W4-024, W4-026 |
| **H22** | Định lý số thực không tự động chuyển tiếp sang float32 | Phản chứng & Khoảng cách | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_DIRECT_TRANSFER_ESTABLISHED_GATE** | Sai số làm tròn IEEE-754 và cancellation; bắt buộc outward rounding và cổng capability | W4-005, W4-008, W4-010 |
| **H23** | Tên sự kiện không thể là cơ quan cấp quyền chứng nhận | Phản chứng quyền lực | `FALSIFIED_UNCONDITIONAL` | **FALSIFIED_AUTHORITY** | Query cùng nhãn sự kiện nhưng ngoài miền hình học bị reject fail-closed ngay lập tức | W4-015, W4-023, W4-026 |
| **H24** | Đồng thời đạt tính đúng đắn, hữu ích và tiết kiệm chi phí | Biên giới khoa học | `NOT_ESTABLISHED` | **EMPIRICALLY_CHARACTERIZED** | Phác họa đường biên Pareto: M16 kết hợp tái sử dụng Conv3 đạt gia tốc dương dưới cổng kiểm soát | W4-010, W4-021, W4-024, W4-025 |

---

## 3. Phân Tích Chuyên Sâu Các Cụm Giả Thuyết Trọng Yếu

### 3.1. Cụm Hình Học Miền Dữ Liệu & Sự Khắc Nghiệt Của Cảm Biến (H01, H02, H10, H11, H12, H17)

1. **Bản chất của phép trượt 50% (H01 & H12):**
   - Với kích thước cửa sổ $W=128$, bước trượt $S=64$ trên $C=6$ kênh cảm biến, cửa sổ kế tiếp $Z_{t+1}$ chia sẻ chính xác $64$ mẫu với cửa sổ $Z_t$.
   - Chứng minh hình thức khẳng định $Z_{t+1}[:, 0:64] = Z_t[:, 64:128]$. Điều này cố định chính xác 384 tọa độ vô hướng, giảm số chiều tự do của bài toán chứng nhận từ 768 xuống còn 384 tại $h=1$.
   - Kiểm tra trên toàn bộ 5,034 cặp cửa sổ kế tiếp của tập Test xác nhận độ khớp bit-exact đạt 100.0%.

2. **Phản chứng H02 (Ramp Counterexample):**
   - Giả định trực giác thông thường cho rằng "vì cửa sổ chia sẻ 50% dữ liệu nên khoảng cách giữa hai cửa sổ kế tiếp phải nhỏ" là **hoàn toàn sai lầm**.
   - Chứng minh toán học trong Supplement 05: Một tín hiệu sóng sin đơn giản $s[n] = A \sin(\omega n + \phi)$ tạo ra độ lệch vị trí (positional delta) có biên độ lên tới:
     $$\|\Delta\| = 2A \left| \sin\left(\frac{\omega S}{2}\right) \right|$$
   - Khi $\omega S / 2 = \pi/2$ (tần số tương ứng $\approx 0.39$ Hz, rất phổ biến trong nhịp bước đi bộ của con người), biên độ sai khác đạt $2A$ — gấp đôi biên độ tín hiệu gốc!
   - Đo đạc thực tế trên tập validation xác nhận độ lệch đuôi cực đại $L_\infty$ lên tới 16.09 units, khẳng định giả thuyết H02 bị bác bỏ dứt khoát.

3. **Hiện tượng thay thế toàn bộ cửa sổ tại Horizon $h=2$ (H17):**
   - Khi bước sang horizon $h=2$ (bước $t+2$), độ trượt đạt $2 \times 64 = 128 = W$.
   - Cửa sổ $Z_{t+2} = [V_1, V_2]$ không còn chứa bất kỳ mẫu vật lý nào của anchor ban đầu $Z_t$ ($0.0\%$ overlap fraction).
   - Số chiều tự do bùng nổ từ 384 lên 768. Do đó, việc mở rộng horizon lên $h \ge 3$ làm tích lũy sai số vượt ngưỡng chịu đựng của bộ giải LiRPA, buộc hệ thống phải fail-closed.

---

### 3.2. Cụm Thuật Toán Lan Truyền Biên & Độ Không Đầy Đủ (H04, H05, H06, H07, H08, H09)

1. **Hiệu ứng kẹp chặt và độ không đầy đủ của IBP (H07):**
   - Lan truyền biên hộp độc lập (Interval Bound Propagation) xem các biến nơ-ron là độc lập ở mỗi tầng, bỏ qua hoàn toàn các ràng buộc chéo (cross-neuron dependencies).
   - **Nhân chứng giải tích (Toy Model Witness):** Xét mạng 2 nơ-ron $h_1 = h_2 = \operatorname{ReLU}(t + 1)$ với $t \in [-1, 1]$, và hàm mục tiêu margin là $M = 1 + h_1 - h_2$.
     - Trong thực tế: $h_1 \equiv h_2$ trên toàn bộ miền xác định, do đó $M = 1 + h_1 - h_1 = 1.0 > 0$ (luôn luôn dương và bất biến!).
     - Khi dùng IBP: Khoảng giá trị độc lập của $h_1$ là $[0, 2]$, của $h_2$ là $[0, 2]$. Cận dưới IBP cho margin là:
       $$\underline{M}_{\text{IBP}} = 1 + \underline{h}_1 - \overline{h}_2 = 1 + 0 - 2 = -1.0 < 0$$
     - Khoảng cách triệt tiêu sai lệch (cancellation gap) lên tới $2.0$ units, biến một bất biến toán học hiển nhiên thành kết luận âm (inconclusive).
   - Kiểm chứng thực nghiệm trên mạng `SmallCNN1D` trong task W4-017 xác nhận việc bao hộp độc lập (hull loss) làm suy giảm trung bình 149.47 đơn vị lề margin.

2. **Ưu thế tuyệt đối của Direct Margin (H08):**
   - Bằng cách gom ma trận trọng số phân loại $W_{c^*} - W_j$ thành một vector mục tiêu duy nhất trước khi lan truyền ngược qua các tầng nơ-ron, phương pháp Direct Margin triệt tiêu toàn bộ sự phụ thuộc tương quan ở tầng tuyến tính cuối cùng.
   - Bất đẳng thức tam giác bảo đảm:
     $$\underline{\operatorname{Margin}}_{\text{direct}} \;\ge\; \underline{\operatorname{Logit}}_{c^*} - \overline{\operatorname{Logit}}_j$$
   - 100% các phép đo thực nghiệm xác nhận tính đơn điệu này được bảo toàn bit-for-bit.

---

### 3.3. Cụm Kiến Trúc Mạng & Tái Sử Dụng Đặc Trưng Trung Gian (H13, H14)

1. **Định lý căn chỉnh đặc trưng tích chập 1D (H13):**
   - Kiến trúc tích chập 1D với padding đối xứng và các tầng MaxPool có bước nhảy tích lũy 4 chia hết cho bước trượt 64 ($64 / 4 = 16$).
   - Phân tích giải tích trường cảm nhận (Receptive Field) xác định: nơ-ron vị trí $j$ tại tầng `Conv3` phụ thuộc vào đoạn đầu vào $[4j - 8, \; 4j + 11]$.
   - Đối với các vị trí nội thất $j \in \{2, \ldots, 13\}$, đoạn phụ thuộc đầu vào nằm hoàn toàn trong đoạn tiền tố 64 mẫu $[0, 63]$.
   - Do đó, đặc trưng tại $j$ ở bước $t+1$ trùng khớp bit-exact với đặc trưng tại $j + 16$ ở bước $t$.
   - Kiểm chứng thực nghiệm trên 5,034 cặp cửa sổ xác nhận $3,866,112$ giá trị số thực đạt sai số tuyệt đối bằng đúng $0.0$.

2. **Giới hạn không thể áp dụng công thức GAP (H14):**
   - Các công thức cập nhật sai phân lịch sử dựa trên giả định mạng có lớp Global Average Pooling (GAP) trước đầu ra phân loại.
   - Mạng chuẩn `SmallCNN1D` sử dụng đầu phân loại `Flatten(2048) -> Linear(128) -> ReLU -> Linear(12)`.
   - Vì ma trận trọng số của lớp `Linear(128)` gắn chặt với từng vị trí không gian cụ thể của tensor đặc trưng, việc dịch chuyển đặc trưng không cho phép trừ/cộng sai phân tuyến tính. Giả thuyết H14 bị loại trừ hoàn toàn khỏi phạm vi ứng dụng.

---

### 3.4. Cụm An Ninh Hệ Thống, No-Drift & Chi Phí Ẩn (H03, H15, H16, H18, H19, H20, H21, H22, H23, H24)

1. **Phản chứng cấm Re-centering (H15) & Bài toán Trôi Dạt Tích Lũy (Drift Witness):**
   - Nếu hệ thống cho phép dời tâm chứng chỉ sang query mới mỗi khi tái sử dụng thành công:
     - Query $t+1$ lệch $\epsilon$ so với anchor gốc, verifier chấp nhận và đổi tâm sang $t+1$.
     - Query $t+2$ lệch tiếp $\epsilon$, verifier lại đổi tâm sang $t+2$.
     - Sau $N$ bước, chuỗi truy vấn đã trôi xa $N \times \epsilon$ khỏi phân bố an toàn ban đầu và chuyển sang nhãn khác, nhưng verifier vẫn tiếp tục tái sử dụng nhãn cũ.
   - **Chính sách No-Drift bắt buộc:** Anchor phải giữ cố định tuyệt đối (`Fixed-Anchor Invariant`). Khi query hit reuse, tuổi của chứng chỉ **tuyệt đối không được reset**. Khi tuổi vượt quá `max_age`, chứng chỉ bắt buộc phải hết hạn (`FALLBACK_EXPIRED`) và gọi lại mô hình tham chiếu.

2. **Chi phí ẩn của Zero-Trust Assertion và Giải Pháp M16 (H20 & H21):**
   - Đo đạc định lượng trong W4-024 phát hiện: Việc băm SHA-256 toàn bộ 283,692 trọng số và AST bytecode trên mỗi query inference tốn **1.394 ms**, chiếm tới **76.8%** tổng thời gian suy luận, làm hệ thống chậm gấp gần 6.0 lần so với PyTorch thuần túy.
   - **Kiến trúc Owned-Reference (M16):** Đóng gói mô hình trong lớp sở hữu độc quyền `ReferenceOwner`, chỉ thực hiện fast shallow check ($< 0.09$ ms) trong suốt generation, và thực hiện deep assertion khi đổi session salt. M16 giúp phục hồi gia tốc **4.13x** so với M01, chỉ chênh 1.45x so với PyTorch thô (chủ yếu do ký số HMAC receipt).

3. **Cổng năng lực số học (H22 - Numerical Capability Gate):**
   - Các định lý toán học trong tập số thực $\mathbb{R}$ không tự động đúng trong chuẩn IEEE-754 float32 do sai số làm tròn, mất mát độ chính xác (catastrophic cancellation), và tính không kết hợp của phép cộng số thực dấu phẩy động.
   - Bắt buộc phải triển khai các phép toán làm tròn hướng ngoại (outward rounding) và duy trì cổng `NumericalCapability` đóng (`fail-closed`) cho bất kỳ cấu hình nào chưa có chứng minh sai số phân tích.

4. **Đường biên Pareto Khoa Học (H24 - Joint Scientific Frontier):**
   - Nghiên cứu xác lập trilemma cơ bản của ECDR:
     $$\text{Formal Soundness} \quad \longleftrightarrow \quad \text{Physical Useful Reachability} \quad \longleftrightarrow \quad \text{Net Speedup}$$
   - Cấu hình duy nhất đạt gia tốc thực tế mà không hy sinh tính đúng đắn là: **M16 Owned-Reference** kết hợp **Tái sử dụng đặc trưng tầng Conv3** dưới cổng kiểm soát **No-Drift Fixed-Anchor**.

---

## 4. Kết Luận Nghiên Cứu & Bàn Giao Milestone M4

1. Bản đánh giá toán học độc lập đã hoàn tất xem xét đầy đủ **24/24 giả thuyết H01–H24** mà không bỏ sót bất kỳ giả định hay nhân chứng phản chứng nào.
2. File artifact dữ liệu `output/week4/week4_initial_20261103/mathematical_review.json` đã được tạo lập với định dạng JSON chuẩn mực và bảo đảm tính tái lập hoàn toàn.
3. Toàn bộ các cơ sở toán học đã sẵn sàng để chuyển tiếp sang **Milestone M4** (Method Registry M00–M16, Comprehensive Evaluation & Empirical Synthesis).
