# Lịch Trình Tái Sử Dụng Đặc Trưng Trễ Tầng (Stagewise Feature-Cache Schedule)

**Nhiệm vụ sở hữu**: [W4-019](../tasks/W4-019.md)  
**Trạng thái**: SPECIFIED & REVIEW READY  
**Phiên bản**: 1.0 (Week 4 Implementation Phase)  
**Tệp mục tiêu**: `04_SYNTHESIS/week4/design/INCREMENTAL_SCHEDULE.md`  
**Giả thuyết liên quan**: H01, H12, H13, H14  

---

## 1. Dẫn Xuất Vùng Cảm Thụ (Receptive Field Derivation) Từng Tầng (C01)

Mô hình mục tiêu là **SmallCNN1D** ($K=12$, 283.692 tham số) với cấu trúc:
$$\text{Input (6, 128)} \to \text{Conv1} \to \text{BN1} \to \text{ReLU} \to \text{Pool1} \to \text{Conv2} \to \text{BN2} \to \text{ReLU} \to \text{Pool2} \to \text{Conv3} \to \text{BN3} \to \text{ReLU} \to \text{Head}$$

Vùng cảm thụ đầu vào (Receptive Field - RF) của một nơ-ron tại vị trí thời gian $i$ được định nghĩa bằng đoạn chỉ số bao đóng $[L(i), R(i)] \subseteq \mathbb{Z}$ trên trục thời gian $128$ bước của cửa sổ đầu vào $X \in \mathbb{R}^{6 \times 128}$.

### 1.1 Bảng Dẫn Xuất Chi Tiết Từng Tầng

| Tầng / Toán tử | Tham số Kiến trúc | Quan hệ Chỉ số với Tầng Trước | Vùng Cảm Thụ Đầu Vào $[L(i), R(i)]$ | Độ Rộng RF | Bước Nhảy (Jump / Stride) |
|:---|:---|:---|:---|:---:|:---:|
| **Input** | Shape: $(6, 128)$ | Gốc tọa độ thời gian | $[i, i]$ | 1 | 1 |
| **Conv1D Stage 1** | $k_1 = 5, p_1 = 2, s_1 = 1$ | $t \in [i - 2, i + 2]$ | $[i - 2, i + 2]$ | 5 | 1 |
| **BatchNorm1D (eval) & ReLU** | Pointwise channel | $t = i$ | $[i - 2, i + 2]$ | 5 | 1 |
| **MaxPool1D Stage 1** | $k_{p1} = 2, s_{p1} = 2, p_{p1} = 0$ | $t \in \{2i, 2i + 1\}$ | $[2i - 2, 2i + 3]$ | 6 | 2 |
| **Conv1D Stage 2** | $k_2 = 3, p_2 = 1, s_2 = 1$ | $t \in [i - 1, i + 1]$ | $[2i - 4, 2i + 5]$ | 10 | 2 |
| **BatchNorm1D (eval) & ReLU** | Pointwise channel | $t = i$ | $[2i - 4, 2i + 5]$ | 10 | 2 |
| **MaxPool1D Stage 2** | $k_{p2} = 2, s_{p2} = 2, p_{p2} = 0$ | $t \in \{2i, 2i + 1\}$ | $[4i - 4, 4i + 7]$ | 12 | 4 |
| **Conv1D Stage 3** | $k_3 = 3, p_3 = 1, s_3 = 1$ | $t \in [j - 1, j + 1]$ | $[4j - 8, 4j + 11]$ | 20 | 4 |
| **BatchNorm1D (eval) & ReLU** | Pointwise channel | $t = j$ | **$[4j - 8, 4j + 11]$** | **20** | **4** |

### 1.2 Chứng Minh Dẫn Xuất Cụ Thể Từng Bước
1. **MaxPool1D 1**: Nơ-ron $i$ lấy giá trị cực đại từ hai vị trí Conv1 tại $2i$ và $2i + 1$:
   - Support Conv1 tại $2i$: $[2i - 2, 2i + 2]$.
   - Support Conv1 tại $2i + 1$: $[(2i + 1) - 2, (2i + 1) + 2] = [2i - 1, 2i + 3]$.
   - Hợp hai đoạn: $[2i - 2, 2i + 3]$. Độ rộng: $2i + 3 - (2i - 2) + 1 = 6$. Bước nhảy: 2.
2. **Conv1D 2**: Nơ-ron $i$ kết hợp 3 vị trí Pool1 tại $i - 1, i, i + 1$:
   - Pool1 tại $i - 1$: $[2(i - 1) - 2, 2(i - 1) + 3] = [2i - 4, 2i + 1]$.
   - Pool1 tại $i + 1$: $[2(i + 1) - 2, 2(i + 1) + 3] = [2i, 2i + 5]$.
   - Hợp: $[2i - 4, 2i + 5]$. Độ rộng: 10. Bước nhảy: 2.
3. **MaxPool1D 2**: Nơ-ron $i$ lấy cực đại từ hai vị trí Conv2 tại $2i$ và $2i + 1$:
   - Conv2 tại $2i$: $[2(2i) - 4, 2(2i) + 5] = [4i - 4, 4i + 5]$.
   - Conv2 tại $2i + 1$: $[2(2i + 1) - 4, 2(2i + 1) + 5] = [4i - 2, 4i + 7]$.
   - Hợp: $[4i - 4, 4i + 7]$. Độ rộng: 12. Bước nhảy: 4.
4. **Conv1D 3**: Nơ-ron $j$ kết hợp 3 vị trí Pool2 tại $j - 1, j, j + 1$:
   - Pool2 tại $j - 1$: $[4(j - 1) - 4, 4(j - 1) + 7] = [4j - 8, 4j + 3]$.
   - Pool2 tại $j + 1$: $[4(j + 1) - 4, 4(j + 1) + 7] = [4j, 4j + 11]$.
   - Hợp: **$[4j - 8, 4j + 11]$**. Độ rộng: 20. Bước nhảy: 4.

*Lưu ý*: Độ rộng RF là 20 bước, nhưng điểm mút bắt đầu tại $4j - 8$ (không phải $4j - 9$ hay $4j - 10$). Việc xác định chính xác điểm mút cực kỳ quan trọng để xác định chính xác các chỉ số biên bị ảnh hưởng bởi zero-padding.

---

## 2. Bản Đồ Ánh Xạ Tái Sử Dụng (Reuse Mappings) Giữa Hai Cửa Sổ Kế Tiếp (C02)

Xét hai cửa sổ thời gian liên tiếp trong cùng một phiên ghi cảm biến (streaming mode):
- Cửa sổ cũ: $X_{\text{old}} \in \mathbb{R}^{6 \times 128}$.
- Cửa sổ mới: $X_{\text{new}} \in \mathbb{R}^{6 \times 128}$.
- Độ trượt (stride): $s = 64$ mẫu (chồng lấn $50\%$).
- Quan hệ dữ liệu thô và chuẩn hóa:
  $$X_{\text{new}}[:, t] = X_{\text{old}}[:, t + 64], \quad \forall t \in [0, 63]$$

Một vị trí đặc trưng $j$ trong $X_{\text{new}}$ được bảo đảm trùng khớp hoàn toàn với vị trí $j_{\text{old}}$ trong $X_{\text{old}}$ khi và chỉ khi:
1. Toàn bộ vùng cảm thụ $[L(j), R(j)]$ của nó nằm trọn vẹn trong vùng chồng lấn:
   $$0 \le L(j) \quad \text{và} \quad R(j) \le 63$$
2. Cả $X_{\text{new}}$ tại vị trí đó và $X_{\text{old}}$ tại vị trí tương ứng đều **không chạm vào phần đệm zero (zero-padding)** ở hai đầu biên.

### 2.1 Bảng Ánh Xạ Cụ Thể Qua Từng Tầng

| Tầng Biểu Diễn | Kích Thước Tensor | Điều Kiện Chồng Lấn $[L(i), R(i)] \subseteq [0, 63]$ | Dải Chỉ Số Mới Tái Sử Dụng Được | Dải Chỉ Số Cũ Tương Ứng | Công Thức Ánh Xạ | Số Lượng Vị Trí Trùng Khớp |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conv1** | $(32, 128)$ | $i - 2 \ge 0 \land i + 2 \le 63 \implies 2 \le i \le 61$ | $i \in [2, 61]$ | $i_{\text{old}} \in [66, 125]$ | $i_{\text{old}} = i + 64$ | **60 / 128** ($46.88\%$) |
| **Pool1** | $(32, 64)$ | $2i - 2 \ge 0 \land 2i + 3 \le 63 \implies 1 \le i \le 30$ | $i \in [1, 30]$ | $i_{\text{old}} \in [33, 62]$ | $i_{\text{old}} = i + 32$ | **30 / 64** ($46.88\%$) |
| **Conv2** | $(64, 64)$ | $2i - 4 \ge 0 \land 2i + 5 \le 63 \implies 2 \le i \le 29$ | $i \in [2, 29]$ | $i_{\text{old}} \in [34, 61]$ | $i_{\text{old}} = i + 32$ | **28 / 64** ($43.75\%$) |
| **Pool2** | $(64, 32)$ | $4i - 4 \ge 0 \land 4i + 7 \le 63 \implies 1 \le i \le 14$ | $i \in [1, 14]$ | $i_{\text{old}} \in [17, 30]$ | $i_{\text{old}} = i + 16$ | **14 / 32** ($43.75\%$) |
| **Conv3 (Cuối)**| $(64, 32)$ | $4j - 8 \ge 0 \land 4j + 11 \le 63 \implies 2 \le j \le 13$ | **$j \in [2, 13]$** | **$j_{\text{old}} \in [18, 29]$** | **$j_{\text{old}} = j + 16$** | **12 / 32** ($37.50\%$) |

---

## 3. Phân Loại Vị Trí Cần Tính Lại & Khoảng Phụ Thuộc (C03)

Tại tầng đặc trưng không gian cuối cùng (Conv3 sau ReLU, shape $64 \times 32$), ta phân chia 32 vị trí thời gian của cửa sổ mới thành hai nhóm rõ rệt:

### 3.1 Nhóm Tái Sử Dụng (Cached Interior Features: 12 vị trí)
- **Dải chỉ số**: $j \in [2, 13]$.
- **Hành vi**: Đọc trực tiếp từ bộ đệm của cửa sổ trước:
  $$H_{\text{new}}[:, j] = H_{\text{old}}[:, j + 16], \quad \forall j \in [2, 13]$$
- **Chi phí tính toán**: $0 \text{ FLOPs}$ (chỉ là thao tác copy mảng bộ nhớ / memory pointer).

### 3.2 Nhóm Cần Tính Lại (Recomputed Features: 20 vị trí)
Nhóm này gồm hai vùng:

1. **Vùng Biên Trái (Left Boundary: 2 vị trí $j \in \{0, 1\}$)**:
   - $j = 0$: $RF = [-8, 11]$. Bị ảnh hưởng bởi padding zero phía trước $t < 0$.
   - $j = 1$: $RF = [-4, 15]$. Bị ảnh hưởng bởi padding zero phía trước $t < 0$.
   - *Lưu ý*: Dù $X_{\text{new}}$ tại $t < 0$ là zero-padding, nhưng trong chuỗi gốc đó lại là dữ liệu của $X_{\text{old}}$, do đó giá trị Conv3 tại $j \in \{0, 1\}$ của $X_{\text{new}}$ **không bằng** bất kỳ điểm nào trong $X_{\text{old}}$. Bắt buộc phải tính lại.

2. **Vùng Dữ Liệu Mới (New Timesteps: 18 vị trí $j \in [14, 31]$)**:
   - $j = 14$: $RF = [48, 67]$. Chạm vào vùng dữ liệu mới $t \ge 64$.
   - $j \in [15, 31]$: $RF$ phụ thuộc hoàn toàn vào nửa sau của cửa sổ $X_{\text{new}}[:, 64:128]$ (chưa từng xuất hiện ở $X_{\text{old}}$).

### 3.3 Lịch Trình Tính Lại Từng Tầng Có Lưu Cache Trung Gian (Stagewise Partial Recomputation Schedule)

Khi duy trì bộ đệm đặc trưng cho tất cả 5 tầng trung gian (`h1`, `p1`, `h2`, `p2`, `h3`), mỗi tầng chỉ cần tính lại phần **Biên Trái (Left Boundary)** bị ảnh hưởng bởi zero-padding mới và phần **Đuôi Phải (Right New Region)** bị ảnh hưởng bởi 64 mẫu cảm biến mới $t \in [64, 127]$:

| Tầng | Vị Trí Tái Sử Dụng (Từ Cache Cũ) | Vị Trí Cần Tính Lại: Biên Trái | Vị Trí Cần Tính Lại: Đuôi Mới | Lát Đầu Vào Trực Tiếp & Đệm Một Phía (One-Sided Padding) | Số Cột Tính Lại / Tổng | MACs Tính Lại / MACs Đầy Đủ |
|:---|:---|:---|:---|:---|:---:|:---:|
| **Conv1 + BN1 + ReLU1** | $i \in [2, 61] \leftarrow i_{\text{old}} \in [66, 125]$ (60 cột) | $i \in \{0, 1\}$ (2 cột) | $i \in [62, 127]$ (66 cột) | Trái: $\text{pad}_{(2,0)}(X_{\text{norm}}[:, 0:4])$ (dài 6)<br>Phải: $\text{pad}_{(0,2)}(X_{\text{norm}}[:, 60:128])$ (dài 70) | **68 / 128** ($53.13\%$) | **65.280 / 122.880** (tiết kiệm $46.88\%$) |
| **MaxPool1D 1** | $i \in [1, 30] \leftarrow i_{\text{old}} \in [33, 62]$ (30 cột) | $i = 0$ (1 cột) | $i \in [31, 63]$ (33 cột) | Trái: $H_1^{\text{new}}[:, 0:2]$ (dài 2)<br>Phải: $H_1^{\text{new}}[:, 62:128]$ (dài 66) | **34 / 64** ($53.13\%$) | $0$ (chỉ so sánh max) |
| **Conv2 + BN2 + ReLU2** | $i \in [2, 29] \leftarrow i_{\text{old}} \in [34, 61]$ (28 cột) | $i \in \{0, 1\}$ (2 cột) | $i \in [30, 63]$ (34 cột) | Trái: $\text{pad}_{(1,0)}(P_1^{\text{new}}[:, 0:3])$ (dài 4)<br>Phải: $\text{pad}_{(0,1)}(P_1^{\text{new}}[:, 29:64])$ (dài 36) | **36 / 64** ($56.25\%$) | **221.184 / 393.216** (tiết kiệm $43.75\%$) |
| **MaxPool1D 2** | $i \in [1, 14] \leftarrow i_{\text{old}} \in [17, 30]$ (14 cột) | $i = 0$ (1 cột) | $i \in [15, 31]$ (17 cột) | Trái: $H_2^{\text{new}}[:, 0:2]$ (dài 2)<br>Phải: $H_2^{\text{new}}[:, 30:64]$ (dài 34) | **18 / 32** ($56.25\%$) | $0$ (chỉ so sánh max) |
| **Conv3 + BN3 + ReLU3** | $j \in [2, 13] \leftarrow j_{\text{old}} \in [18, 29]$ (12 cột) | $j \in \{0, 1\}$ (2 cột) | $j \in [14, 31]$ (18 cột) | Trái: $\text{pad}_{(1,0)}(P_2^{\text{new}}[:, 0:3])$ (dài 4)<br>Phải: $\text{pad}_{(0,1)}(P_2^{\text{new}}[:, 13:32])$ (dài 20) | **20 / 32** ($62.50\%$) | **245.760 / 393.216** (tiết kiệm $37.50\%$) |
| **Head (Flatten–MLP)** | Không tái sử dụng ($0$ phần tử) | Toàn bộ $2048 \to 128 \to 12$ | Toàn bộ $2048 \to 128 \to 12$ | Toàn bộ $\text{vec}(H_3^{\text{new}}) \in \mathbb{R}^{2048}$ | **100%** | **263.680 / 263.680** (tiết kiệm $0\%$) |
| **Tổng Toàn Mạng** | **134 cột đặc trưng qua 5 tầng** | **8 cột biên trái** | **168 cột dữ liệu mới** | — | — | **795.904 / 1.172.992** (**tiết kiệm 32.15% tổng MACs; 41.47% Conv MACs**) |

*Ghi chú về phụ thuộc ngược khi KHÔNG lưu cache các tầng trung gian*: Nếu chỉ lưu cache duy nhất ở tầng Conv3 mà không lưu `h1, p1, h2, p2`, để tính 20 vị trí $j \in \{0, 1\} \cup [14, 31]$ của Conv3 sẽ phải truy ngược:
- Pool2: $i \in \{0, 1, 2\} \cup [13, 31]$ (22 vị trí, trong đó $i \in \{1, 2, 13, 14\}$ lẽ ra đã có trong cache `p2`).
- Conv2: $i \in [0, 5] \cup [26, 63]$ (44 vị trí, trong đó $i \in [2, 5] \cup [26, 29]$ lẽ ra đã có trong cache `h2`).
- Pool1: $i \in [0, 6] \cup [25, 63]$ (46 vị trí, trong đó $i \in [1, 6] \cup [25, 30]$ lẽ ra đã có trong cache `p1`).
- Conv1: $i \in [0, 13] \cup [50, 127]$ (92 vị trí, trong đó $i \in [2, 13] \cup [50, 61]$ lẽ ra đã có trong cache `h1`).
Do đó, việc lưu cache đầy đủ từng tầng (`h1, p1, h2, p2, h3`) theo đúng **W4-020-C01 / C02** giúp giảm số cột phải tính ở Conv1 từ $92 \to 68$ và ở Conv2 từ $44 \to 36$, tối đa hóa mức tiết kiệm MACs của khối tích chập lên **41.47%** ($532.224$ so với $909.312$ MACs).
Đặc biệt, ở mọi tầng tích chập (Conv1, Conv2, Conv3), cả lát biên trái ($L_{\text{out}} = 2$) và lát đuôi mới ($L_{\text{out}} \in \{66, 34, 18\}$) đều có độ dài đầu ra $L_{\text{out}} \ge 2$, bảo đảm PyTorch ATen (`_slow_conv2d_forward`) luôn gọi nhánh `cblas_sgemm` với $m = L_{\text{out}} \ge 2$ thay vì nhánh `sgemv` ($m = 1$), giữ nguyên tuyệt đối thứ tự cộng dồn nội tích theo kênh/kernel và đạt **100% tương đương bitwise (0.0 ULP)** trên kiến trúc thực thi đã khóa.

---

## 4. Các Tiền Đề Bắt Buộc Để Đảm Bảo Tính Đúng Đắn (C04)

Thuật toán tái sử dụng trễ tầng chỉ đúng đắn tuyệt đối khi và chỉ khi thỏa mãn đầy đủ 4 tiền đề hệ thống:

1. **Tính Bất Biến Của BatchNorm Ở Chế Độ Eval (B1)**:
   - Các tầng BatchNorm1D phải ở chế độ `eval()`.
   - Các tham số $\mu_{\text{run}}, \sigma^2_{\text{run}}, \gamma, \beta$ là cố định hoàn toàn (đóng băng). Không được tính toán mean/var động theo batch dữ liệu đầu vào.
2. **Khớp Pha Tuyệt Đối Của MaxPool1D (B2)**:
   - Phép gom cụm MaxPool1D ($k=2, s=2$) có điểm xuất phát gốc pha tại $t=0$ (`origin=0`).
   - Khoảng trượt cửa sổ $s = 64$ là một bội số của cả bước nhảy tầng 1 ($s_{p1} = 2$) và bước nhảy tầng 2 ($s_{p2} = 4$). Cụ thể:
     $$\frac{64}{2} = 32 \in \mathbb{Z}, \quad \frac{64}{4} = 16 \in \mathbb{Z}$$
     Do đó, không xảy ra hiện tượng lệch pha (phase misalignment) giữa các cặp max.
3. **Tính Chuẩn Hóa Theo Từng Điểm Độc Lập (B3)**:
   - Phép chuẩn hóa $X_{\text{norm}} = (X - \mu)/\sigma$ là phép toán tác động cục bộ theo từng kênh và độc lập giữa các bước thời gian.
   - Do đó:
     $$X_{\text{new\_norm}}[:, t] = X_{\text{old\_norm}}[:, t + 64], \quad \forall t \in [0, 63]$$
4. **Tính Liên Tục Của Phiên Ghi (B4 - Recording Continuity)**:
   - Hai cửa sổ phải thuộc cùng một phiên ghi (`recording_id` giống nhau) và có chỉ số bước liên tiếp ($step_{\text{new}} = step_{\text{old}} + 1$).
   - Nếu có ngắt quãng (gap, reset phiên ghi, hoặc mất gói cảm biến), bộ đệm cache bắt buộc phải bị xóa sạch (reset fail-closed).

---

## 5. Tại Sao Giả Thuyết H14 (GAP Delta) Không Áp Dụng Cho Mạng Hiện Tại (C05)

### 5.1 Bản Chất Của Giả Thuyết Lịch Sử H14 (GAP Delta)
Trong kiến trúc nghiên cứu ban đầu (GAP Model), đầu ra của các tầng Conv được đi qua tầng **Global Average Pooling**:
$$v_c = \frac{1}{T} \sum_{t=0}^{T-1} H[c, t]$$
Khi cửa sổ trượt dịch chuyển, tổng toàn cục có thể được cập nhật gia số bằng công thức:
$$v_c^{\text{new}} = v_c^{\text{old}} - \frac{1}{T} \sum_{t \in \text{dropped}} H_{\text{old}}[c, t] + \frac{1}{T} \sum_{t \in \text{new}} H_{\text{new}}[c, t]$$
Sau đó, logits được tính bằng phép nhân tuyến tính $z = W v + b$.

### 5.2 Sự Phá Vỡ Của Kiến Trúc SmallCNN1D Hiện Tại
Mô hình đông lạnh chuẩn **SmallCNN1D K=12** **KHÔNG CÓ TẦNG GAP**.
Thay vào đó, đầu ra Conv3 ($64 \times 32$) được làm phẳng bằng `Flatten()` thành vector $2048$ phần tử:
$$v = \text{vec}(H) = [H[0, 0], \dots, H[0, 31], H[1, 0], \dots, H[63, 31]]^\top \in \mathbb{R}^{2048}$$
Sau đó đi qua mạng MLP phi tuyến:
$$h_1 = \text{ReLU}(W_1 v + b_1), \quad W_1 \in \mathbb{R}^{128 \times 2048}$$
$$z = W_2 h_1 + b_2, \quad W_2 \in \mathbb{R}^{12 \times 128}$$

### 5.3 Phân Tích Tính Nhạy Cảm Theo Vị Trí (Position-Sensitivity)
- Kích thước tensor đặc trưng Conv3 là $(B, 64, 32)$ với 64 kênh và 32 bước thời gian. Khi đi qua `Flatten()`, phần tử tại kênh $c \in [0, 63]$ và vị trí thời gian $j \in [0, 31]$ được trải phẳng thành chỉ số $k = c \cdot 32 + j$ trong vector 2048 chiều:
  $$v[c \cdot 32 + j] = H[c, j]$$
- Mỗi vị trí thời gian $j$ đóng góp vào toàn bộ các nút ẩn $h \in [0, 127]$ qua cột trọng số $W_1[:, c \cdot 32 + j]$.
- Dù đặc trưng bảo tồn tính đồng nhất $H_{\text{new}}[c, j] = H_{\text{old}}[c, j + 16]$ với mọi $c$, ở cửa sổ mới nó xuất hiện tại vị trí $j$, nên nó được nhân với cột trọng số $W_1[:, c \cdot 32 + j]$. Trong khi ở cửa sổ cũ, nó nằm ở vị trí $j + 16$, nên nó được nhân với cột trọng số $W_1[:, c \cdot 32 + j + 16]$.
- Đo đạc thực nghiệm trên trọng số đông lạnh của checkpoint cho thấy:
  $$\|W_1[:, c \cdot 32 + j] - W_1[:, c \cdot 32 + j + 16]\|_2 \ge 0.1709 \quad (\text{trung bình } \approx 0.2296)$$
  Hai tập trọng số này hoàn toàn khác nhau. Thêm vào đó, hàm phi tuyến $\text{ReLU}$ tại tầng ẩn thứ nhất ngăn cản mọi phép phân rã tuyến tính.
- Vì vậy, **không tồn tại bất kỳ công thức delta nào để cập nhật logits** từ logits cũ khi cửa sổ trượt. Giả thuyết lịch sử H14 (vốn giả định tầng GAP bình quân kênh) bị bãi bỏ cho mô hình SmallCNN1D này.
- **Giải pháp khả thi và đúng đắn nhất**: Tái sử dụng trọn vẹn 12 cột đặc trưng Conv3 ($j \in [2, 13]$), ghép với 20 cột tính lại ($j \in \{0, 1\} \cup [14, 31]$), sau đó chuyển tiếp toàn bộ tensor tái cấu trúc qua đầu `Flatten() -> Linear1 -> ReLU -> Linear2`. Đầu MLP chỉ tiêu tốn $\approx 263.000$ FLOPs, giúp duy trì cả tốc độ lẫn tính chính xác tuyệt đối $100\%$ bitwise.

---

## 6. Chiến Lược Kiểm Thử & Xác Nhận Căn Chỉnh (C06, C07)

Để nhiệm vụ W4-019 đạt chuẩn nghiệm thu và sẵn sàng cho W4-020:
1. **Kiểm thử trên tín hiệu xung (Impulse Response) & Đường dốc (Ramp)**:
   - Đưa xung Dirac đơn độc tại các vị trí thời gian khác nhau để kiểm chứng chính xác biên độ hỗ trợ $RF = [4j-8, 4j+11]$.
2. **Kiểm thử trên toàn bộ các cặp cửa sổ thực tế (Validation Pairs)**:
   - Sử dụng PyTorch forward hook tạm thời (được gỡ bỏ ngay sau kiểm thử) để trích xuất tensor Conv1, Pool1, Conv2, Pool2, Conv3.
   - So sánh bitwise tensor giữa cửa sổ mới và cửa sổ cũ trên toàn bộ các cặp cửa sổ liên tiếp trong tập Validation.
   - Xác nhận sai số cực đại trên vùng bảo đảm $j \in [2, 13]$ bằng **chính xác 0.0**.
   - Xác nhận vùng biên $j \in \{0, 1\}$ và $j \in [14, 15]$ có sai số khác 0 (chứng minh ranh giới lý thuyết là chặt chẽ).
3. **Xuất tệp biên nhận kiểm chứng**:
   - Ghi lại toàn bộ kết quả vào `output/week4/<run_id>/alignment_validation.json`.
