# Evaluation Protocol, Metrics, Budgets & Experimental Disclosure (Week 4)

**Task**: [W4-002](../tasks/W4-002.md)  
**Status**: SPECIFIED & SEALED  
**Date**: 2026-11-03  
**Scope**: Experimental methodology, mathematical metrics, fair comparison boundaries, and stopping criteria for ECDR Week 4.

---

## 1. Known-Test Exploratory Disclosure (C01)

### 1.1 Trạng thái dữ liệu Test
Theo nguyên tắc trung thực khoa học, nghiên cứu Week 4 được định danh là:
$$\text{Status} = \mathbf{KNOWN\_TEST\_EXPLORATORY}$$

**Lịch sử tiếp xúc**: Toàn bộ $5.053$ cửa sổ kiểm thử (full-stream test windows) từ 19 phiên ghi HAPT (đối tượng 17, 4, 9) đã được quan sát và đánh giá trong các tuần 1–3 (Report A & B, nghiệm thu K=12). Do đó, các kết quả thực nghiệm trên tập test ở Week 4 **không được coi là một cuộc kiểm chứng mù hoàn toàn (wholly untouched held-out confirmation)**.

### 1.2 Ba giai đoạn vòng đời Protocol
Để loại bỏ nguy cơ over-fitting hoặc threshold tuning trên tập test:
1. **Initial Protocol (`configs/week4/protocol.json`)**: Định nghĩa toàn bộ không gian tham số, lưới ứng viên, tiêu chí dừng và hàm mục tiêu trước khi chạy bất kỳ bước fitting nào.
2. **Validation-Selected Policy**: Mọi siêu tham số (bán kính $\epsilon$, số chiều PCA $k$, ngưỡng cooldown, horizon $H$) phải được chọn lọc tuyệt đối trên tập Validation ($2.422$ cửa sổ, stride 38). Nghiêm cấm sử dụng bất kỳ nhãn hoặc dữ liệu test nào trong bước này.
3. **Final Sealed Configuration**: Sau khi chọn được cấu hình tối ưu trên validation, cấu hình đó được "đóng dấu" (seal) và ghi mã băm SHA-256 vào run manifest trước khi mở luồng test stream.

---

## 2. Hệ thống Mẫu số & Thuật ngữ Đo lường (C02)

### 2.1 Các Mẫu số Độc lập (Exact Denominators)
Không sử dụng một mẫu số chung mơ hồ. Tất cả báo cáo phải gắn rõ mẫu số tương ứng:
- **Toàn luồng kiểm thử (Full-stream queries)**: $N_{full} = 5.053$ cửa sổ trượt (gồm cả các đoạn chuyển tiếp và không nhãn).
- **Tập giám sát chuẩn (Supervised queries)**: $N_{sup} = 3.572$ cửa sổ trượt (thỏa mãn `label_policy` $\in \{\text{"pure"}, \text{"majority"}\}$ và `label_majority` $\in [1, 12]$).
- **Cặp cửa sổ liền kề trong cùng phiên ghi (Within-recording adjacent pairs)**: $N_{adj} = 5.034$ cặp.
- **Số phiên ghi độc lập (Test recordings / sessions)**: $N_{rec} = 19$ (tương ứng 19 chuỗi cảm biến liên tục).

### 2.2 Phân tách Lỗi Tái sử dụng và Lỗi Fallback
Để tránh tình trạng ngụy biện khi tỷ lệ reuse thấp, các biến đếm được định nghĩa riêng biệt:
- $R$: Số truy vấn được trả lời qua cơ chế reuse.
- $D_{reuse}$: Số truy vấn reuse có quyết định sai khác so với mô hình cơ sở SmallCNN1D ($c_{reuse} \neq c_{ref}$).
- $D_{fallback}$: Số truy vấn fallback có quyết định sai khác so với $c_{ref}$ (nếu có lỗi trong wrapper).
- **Tỷ lệ nhất quán trên truy vấn reuse ($A_{reuse}$)**:
  $$A_{reuse} = \begin{cases} 1.0 - \frac{D_{reuse}}{R}, & \text{nếu } R > 0 \\ \text{null (N/A)}, & \text{nếu } R = 0 \end{cases}$$
  *Quy tắc*: Nếu $R = 0$, $A_{reuse}$ bắt buộc để giá trị `null` kèm lý do `ZERO_REUSE_INELIGIBLE`; nghiêm cấm ghi $A_{reuse} = 100\%$.
- **Tỷ lệ nhất quán toàn luồng ($A_{full}$)**:
  $$A_{full} = 1.0 - \frac{D_{reuse} + D_{fallback}}{N_{full}}$$

### 2.3 Phân định 3 Nhánh Tái sử dụng
1. **Mode 1 — Exact Input Replay ($R_{exact}$)**: Tái sử dụng khi đầu vào trùng khớp bitwise hoặc hash canonical trong môi trường xác định.
2. **Mode 2 — Exact Incremental Computation ($R_{incr}$)**: Tái sử dụng các tầng Conv1D/BN/Pool không đổi, tính toán lại toàn bộ đầu Flatten–MLP để cho ra logits tương đương bitwise.
3. **Mode 3 — Regional Certified Decision Reuse ($R_{cert}$)**: Tái sử dụng nhãn khi đầu vào nằm trong miền đã được chứng minh margin dương qua IBP/CROWN/Lipschitz.

*Lưu ý*: Heuristic reuse (`M04`) được đo đạc như một đường cong đánh đổi sai số–tốc độ (Pareto trade-off), không được cộng gộp vào các chỉ số kiểm chứng chính thức.

---

## 3. Không gian Tìm kiếm, Ngân sách & Tiêu chí Dừng (C03)

### 3.1 Lưới Tham số Ứng viên (Candidate Grids)
- **Bán kính chuẩn hóa (Normalized radius $\epsilon$)**:
  $$\mathcal{G}_\epsilon = \{0.0, 10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0\}$$
- **Hạng PCA cho Affine Tail ($k$)**:
  $$\mathcal{G}_k = \{2, 4, 8, 16, 32\}$$
- **Giới hạn chân trời hữu hạn (Finite horizons $H$)**:
  $$\mathcal{G}_H = \{1, 2, 4, 8, 16, 32\}$$

### 3.2 Ngân sách và Ngưỡng Dừng (Budgets & Stopping Criteria)
- **Số lần thử dựng certificate tối đa**: $B_{max} = 128$ lần / recording session.
- **Thời gian timeout cho mỗi lần dựng certificate**: $T_{timeout} = 500.0\text{ ms}$.
- **Kích thước bộ nhớ đệm certificate tối đa**: $C_{max} = 64$ certificates hoạt động đồng thời.
- **Cooldown sau thất bại**: Bắt buộc bỏ qua tối thiểu $4$ bước trước khi thử dựng lại certificate mới nếu lần trước thất bại hoặc timeout.
- **Xử lý Timeout**: Ghi nhận `TIMEOUT_ABORT`, lập tức fallback về mô hình cơ sở, không để rò rỉ trạng thái dở dang.

### 3.3 Hàm Mục tiêu trên Validation
Trên tập validation, chọn cấu hình thỏa mãn:
$$\max_{\theta \in \Theta} R_{val\_cert}(\theta) \quad \text{s.t.} \quad D_{val\_reuse} = 0 \quad \text{và} \quad \text{NumericalGate} = \text{PASS}$$

**Quy tắc phá vỡ thế hòa (Tie-breaking)**:
1. Ưu tiên số lượt reuse có chứng nhận cao nhất ($R_{val\_cert}$).
2. Ưu tiên bán kính bao phủ trung bình lớn hơn.
3. Ưu tiên thời gian dựng certificate trung bình ($T_{build}$) nhỏ hơn.
4. Cấu hình đối chứng $\epsilon = 0.0$ (zero radius) không đủ tư cách được chọn làm certified configuration.

---

## 4. Tiêu chí So sánh Công bằng (Fair Matching Criteria) (C04)

1. **Kiểm soát Hình học (Geometry Control)**:
   - Các phương pháp so sánh phải sử dụng cùng một tâm neo (anchor center $x_0$), cùng hệ tọa độ chuẩn hóa, và cùng bộ trọng số SmallCNN1D K=12.
2. **Phân biệt Thể tích Không gian (Dimension & Volume Invariance)**:
   - Không được so sánh thể tích hộp tọa độ 768 chiều (ambient diagonal box) với thể tích phần đuôi sóng 384 chiều (wave tail box) như hai đại lượng cùng thứ nguyên. Báo cáo phải phân định rõ thể tích Hausdorff trong không gian tương ứng.
3. **Kiểm soát Mất mát Bao lồi (Hull-loss Control)**:
   - Cặp đối chứng giữa `M08` (chiếu miền affine ra hộp bao ngoài) và `M09` (lan truyền trực tiếp trên miền affine) được thiết lập để cô lập chính xác độ lỏng gây ra bởi việc xấp xỉ hình học.

---

## 5. Cấu hình Phần cứng & Ranh giới Đo lường (C05)

- **Môi trường thực thi**: CPU đơn luồng (`torch.set_num_threads(1)`), kích thước batch $B = 1$.
- **Lưu trữ dữ liệu**: Dữ liệu cảm biến float32 cư trú hoàn toàn trong RAM (in-memory numpy arrays). Thời gian đọc đĩa I/O bị loại trừ khỏi phép đo latency.
- **Quy trình Benchmark**:
  - Chạy 50 truy vấn khởi động (warmup queries) trước khi bật đồng hồ đo.
  - Lặp lại 3 lượt đo độc lập (3 complete stream replays) cho mỗi cấu hình.
  - Ranh giới đo lường tách bạch:
    * `cold_setup`: Nạp trọng số, khởi tạo bộ nhớ.
    * `query_inference`: Thời gian forward mô hình cơ sở.
    * `membership_check`: Kiểm tra điểm rơi vào miền bao ($x \in \mathcal{S}$).
    * `certificate_build`: Thời gian tính toán cận biên IBP/CROWN.
    * `state_update`: Đồng bộ hóa bộ nhớ cache và cập nhật trạng thái recording.

---

## 6. Danh mục 17 Phương pháp M00–M16 (C06)

Hệ thống Week 4 quản lý chặt chẽ 17 cấu hình phương pháp theo bảng đăng ký [`configs/week4/methods.json`](../configs/week4/methods.json):
- **Baselines**: `M00` (Core CNN), `M01` (Full wrapper), `M16` (Owned identity baseline).
- **Exact Alternatives**: `M02` (Exact input replay), `M03` (Exact incremental engine).
- **Heuristic Baseline**: `M04` (Heuristic thresholding).
- **Bound Propagation**: `M05` (Lipschitz cache), `M06` (IBP box), `M07` (IBP wave), `M08` (Hull control), `M09` (Correlated zonotope), `M10` (CROWN box), `M11` (CROWN wave/affine).
- **Policy Axes**: `M12` (Fixed anchor), `M13` (Finite horizon) — *chỉ là trục chính sách, không phải thuật toán lan truyền độc lập*.
- **Comparators & Audits**: `M14` (Faithful LipCache comparator), `M15` (StreamiNNC/TinyDejaVu applicability audit).

---

## 7. Xử lý Bất định & Đơn vị Lấy mẫu Thống kê (C07)

1. **Đơn vị lấy mẫu chính (Primary Sampling Unit)**:
   - Các phép kiểm định thống kê (Paired t-test, Wilcoxon signed-rank test) **phải lấy mẫu theo 19 Phiên ghi (Recordings, $N_{rec}=19$)**, tuyệt đối không lấy mẫu theo từng cửa sổ trượt riêng lẻ (nhằm triệt tiêu hiện tượng lạm phát bậc tự do do tự tương quan chuỗi thời gian).
2. **Khoảng tin cậy (Confidence Intervals)**:
   - Khoảng tin cậy 95% được tính toán bằng phương pháp Studentized Cluster Bootstrap trên 19 cụm phiên ghi.
3. **Độ biến thiên thời gian thực thi (Latency Variance)**:
   - Được báo cáo qua giá trị trung bình kèm độ lệch chuẩn ($\mu \pm \sigma$) qua 3 lượt lặp thực nghiệm.
