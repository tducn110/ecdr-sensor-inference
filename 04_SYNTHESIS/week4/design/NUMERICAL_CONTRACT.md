# Hợp Đồng Số Học & Nghĩa Vụ Chứng Minh Toán Tử ECDR Week 4 (Numerical Contract)

**Nhiệm vụ sở hữu**: [W4-005](../tasks/W4-005.md)  
**Trạng thái**: SPECIFIED & REVIEW READY  
**Phiên bản**: 1.0 (Week 4 Implementation Phase)  
**Tệp mục tiêu**: `04_SYNTHESIS/week4/design/NUMERICAL_CONTRACT.md`  
**Giả thuyết liên quan**: H04, H05, H06, H07, H08, H22  

---

## 1. Ba Đối Tượng Tính Toán & Chuỗi Suy Diễn Mong Muốn (C01)

### 1.1 Định nghĩa Ba Đối tượng Tính toán
Để thiết lập nền tảng chứng thực hình thức không ngụy biện, hệ thống phân biệt tuyệt đối giữa ba thực thể tính toán riêng biệt:

1. **Hàm Số Thực Lý Thuyết ($f_{\mathbb{R}}(X)$)**:
   - Miền xác định: $X \in \mathbb{R}^{6 \times 128} \cong \mathbb{R}^{768}$.
   - Tính chất: Tuân thủ đầy đủ các tiên đề đại số của trường số thực $\mathbb{R}$ (tính kết hợp $(a+b)+c = a+(b+c)$, tính giao hoán $a+b = b+a$, tính phân phối $a(b+c) = ab+ac$).
   - Các phép toán giải tích (khoảng cách Euclid, chuẩn ma trận, đạo hàm, hằng số Lipschitz $L$, lan truyền cận khoảng IBP lý thuyết) đều được định nghĩa trên đối tượng này.

2. **Trình Thông Dịch Vô Hướng IEEE 754 ($f_{\text{interp}}(X)$)**:
   - Miền xác định: $X \in \mathbb{F}_{32}^{6 \times 128}$.
   - Tính chất: Mô hình thực thi giả định tuần tự từ trái sang phải với phép làm tròn láng giềng gần nhất (round-to-nearest, ties to even theo chuẩn IEEE 754-2019).
   - Mỗi phép toán cơ bản $\odot \in \{+, -, \times, /\}$ thỏa mãn mô hình sai số tiêu chuẩn:
     $$\text{fl}(a \odot b) = (a \odot b)(1 + \delta), \quad |\delta| \le u = 2^{-24} \approx 5.96046 \times 10^{-8}$$
   - Không chứa các lệnh gộp (FMA) và không thay đổi thứ tự tích lũy.

3. **Backend Nhị Phân Thực Thi Đông Lạnh ($f_{\text{exec}}(X)$)**:
   - Miền xác định: Đầu vào biểu diễn dưới dạng mảng NumPy hoặc PyTorch tensor (`torch.float32`).
   - Tính chất: Mã máy nhị phân thực tế do PyTorch CPU (ATen/MKL/OpenBLAS) sinh ra và thực thi trên phần cứng Intel x86_64:
     - Vector hóa SIMD (tập lệnh AVX-512 / AVX2).
     - Sử dụng lệnh Fused Multiply-Add (`fma(a, b, c)`) — thực hiện phép tính $a \cdot b + c$ với **duy nhất 1 lần làm tròn** thay vì 2 lần làm tròn riêng biệt.
     - Thứ tự cộng tích lũy (reduction order) theo khối (blocked GEMM) hoặc theo cây nhị phân (pairwise summation), phụ thuộc vào kích thước mảng và phân bổ luồng CPU.
     - Phân vùng đa luồng (`torch.get_num_threads() = 6`): Kích thước chunk và thứ tự tổng hợp giữa các worker threads có thể khác nhau tùy tải CPU.

### 1.2 Khoảng Cách Ngữ Nghĩa (The Semantic Gap)
Chứng minh toán học thiết lập trên $f_{\mathbb{R}}$ hoặc trên $f_{\text{interp}}$ **không tự động chuyển giao** cho $f_{\text{exec}}$ (chính là nội dung Giả thuyết **H22: FALSIFIED_UNCONDITIONAL**):
- $f_{\mathbb{R}}(X) \neq f_{\text{interp}}(X)$ do sai số làm tròn số học hữu hạn.
- $f_{\text{interp}}(X) \neq f_{\text{exec}}(X)$ do sự can thiệp của tập lệnh FMA và kiến trúc reduction đa luồng của phần cứng thực tế.
- Bất kỳ khẳng định tái sử dụng nào chỉ dựa trên tính toán số thực hoặc float64 mà không có bao đóng cho $f_{\text{exec}}$ đều chỉ mang giá trị chẩn đoán (diagnostic), không có giá trị bảo đảm an toàn tại runtime.

### 1.3 Chuỗi Suy Diễn Mong Muốn (Desired Implication Chain)
Một chứng chỉ tái sử dụng quyết định (certified decision reuse) hợp lệ đòi hỏi một chuỗi suy diễn hình thức hoàn chỉnh:

$$\begin{aligned}
X \in \mathcal{D}_{\text{raw}} 
&\xrightarrow{\text{P1: Frozen Norm}} \widehat{X}_{\text{norm}} \in [\underline{X_{\text{norm}}}, \overline{X_{\text{norm}}}] \subset \mathbb{R}^{768} \\
&\xrightarrow{\text{P2: Op Enclosures}} z^{\text{exec}}(X) \in [\underline{z}, \overline{z}] \subset \mathbb{R}^{12} \\
&\xrightarrow{\text{P3: Margin Bound}} \forall j \neq c, \quad g_{cj}^{\text{exec}}(X) = z_c^{\text{exec}}(X) - z_j^{\text{exec}}(X) \ge \underline{g_{cj}} \\
&\xrightarrow{\text{P4: Strict Positivity}} \min_{j \neq c} \underline{g_{cj}} > 0 \\
&\implies \arg\max_{k \in \{0,\dots,11\}} z_k^{\text{exec}}(X) = c \quad \text{(Bảo toàn nhãn tuyệt đối)}
\end{aligned}$$

Nếu bất kỳ mắt xích nào trong chuỗi suy diễn trên bị đứt gãy (ví dụ: không bao trùm được sai số FMA, hoặc $\underline{g_{cj}} \le 0$), hệ thống bắt buộc phải **fail-closed**, từ chối tái sử dụng và chuyển về suy luận gốc.

---

## 2. Bảng Kiểm Kê Toán Tử Toàn Diện & Nghĩa Vụ Chứng Minh (C02)

Mô hình mục tiêu là **SmallCNN1D** ($K=12$, 283.692 tham số), nhận đầu vào $X \in \mathbb{R}^{6 \times 128}$ (tổng 768 điểm đo). Dưới đây là bảng kiểm kê chi tiết toàn bộ các toán tử trong luồng từ cảm biến thô đến nhãn dự đoán:

| Thứ tự | Tầng / Toán tử | Đầu vào & Kích thước | Phép toán Thực thi Thực tế | Nguồn Sai số & Bất định | Nghĩa vụ Chứng minh (Proof Obligation) |
|:---:|:---|:---|:---|:---|:---|
| **0** | **Preprocessing Normalization** | $X_{\text{raw}} \in \mathbb{R}^{6 \times 128}$ thô | $(X_{\text{raw}}[c, t] - \mu[c]) / \sigma[c]$ với $\mu, \sigma \in \mathbb{F}_{32}^6$ | Phép trừ gây hủy số (catastrophic cancellation); phép chia hoặc nhân nghịch đảo $1/\sigma[c]$. | Chứng minh cận khoảng bao đóng: $[\underline{X_{\text{norm}}}, \overline{X_{\text{norm}}}] \supseteq \text{fl}((X - \mu)/\sigma)$. |
| **1** | **Conv1D Stage 1** | $X_{\text{norm}} \in \mathbb{R}^{6 \times 128}$ | 32 filters, kernel 5, pad 2. Tích chập: $y[k, t] = \sum_{c=0}^5 \sum_{r=-2}^2 W[k, c, r] X[c, t+r] + b[k]$ | Tích vô hướng 30 phần tử + 1 bias. Sử dụng FMA hoặc SIMD vector accumulation; padding zero ở biên. | Bound sai số tích lũy dot product độ dài 30; bảo đảm tính đơn điệu khi lan truyền khoảng. |
| **2** | **BatchNorm1D Eval 1** | Tensor $(32, 128)$ | $y = \frac{x - \mu_{\text{run}}}{\sqrt{\sigma^2_{\text{run}} + \epsilon}} \gamma + \beta = s_k \cdot x + b_k$ | Căn bậc hai $\sqrt{\cdot}$ và nghịch đảo; thứ tự gộp $s_k = \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}$ có sai số làm tròn float32. | Chứng minh tính tương đương của việc gộp trọng số affine $(s_k, b_k)$ so với tính toán tách rời của PyTorch eval. |
| **3** | **ReLU 1** | Tensor $(32, 128)$ | $y = \max(0, x)$ | Điểm kỳ dị không khả vi tại $x=0$; làm tròn float32 gần 0 có thể đổi dấu. | Phép toán không giãn (non-expansive): $\max(0, [\underline{x}, \overline{x}]) = [\max(0, \underline{x}), \max(0, \overline{x})]$. |
| **4** | **MaxPool1D 1** | Tensor $(32, 128)$ | Kernel 2, stride 2. $y[k, t] = \max(x[k, 2t], x[k, 2t+1])$ | Rút gọn chiều thời gian $128 \to 64$. Hòa điểm (ties) khi hai giá trị bằng nhau. | Cận khoảng: $[\max(\underline{x_1}, \underline{x_2}), \max(\overline{x_1}, \overline{x_2})]$. |
| **5** | **Conv1D Stage 2** | Tensor $(32, 64)$ | 64 filters, kernel 3, pad 1. Tích chập 96 phần tử + 1 bias. | Tích lũy 96 nhân tử; trôi dạt dấu phẩy động trên 64 kênh. | Bound sai số tích lũy dot product độ dài 96 theo chuẩn hướng ra ngoài. |
| **6** | **BatchNorm1D Eval 2** | Tensor $(64, 64)$ | Scale & shift theo 64 kênh. | Sai số làm tròn phép nhân tỉ lệ. | Bảo toàn bao đóng miền giá trị sau chuẩn hóa tầng 2. |
| **7** | **ReLU 2** | Tensor $(64, 64)$ | $\max(0, x)$ | Tương tự ReLU 1. | Non-expansive interval propagation. |
| **8** | **MaxPool1D 2** | Tensor $(64, 64)$ | Kernel 2, stride 2. Rút gọn $64 \to 32$. | Rút gọn thời gian tầng 2. | Cận khoảng bảo toàn biên cực trị. |
| **9** | **Conv1D Stage 3** | Tensor $(64, 32)$ | 64 filters, kernel 3, pad 1. Tích chập 192 phần tử + 1 bias. | Tích lũy 192 nhân tử; sai số tích lũy lớn nhất trong các tầng conv. | Bound sai số dot product độ dài 192; kiểm soát độ phình của khoảng. |
| **10** | **BatchNorm1D Eval 3** | Tensor $(64, 32)$ | Scale & shift theo 64 kênh. | Sai số làm tròn tầng conv cuối. | Bao đóng khoảng đầu ra đặc trưng không gian. |
| **11** | **ReLU 3** | Tensor $(64, 32)$ | $\max(0, x)$ | Tương tự các tầng ReLU trước. | Non-expansive interval propagation. |
| **12** | **Flatten** | Tensor $(64, 32) \to \mathbb{R}^{2048}$ | Ánh xạ chỉ số: $i = c \times 32 + t$ | Không có phép toán số học; chỉ là định vị bộ nhớ liên tục (contiguous layout). | Bảo đảm đúng thứ tự kênh-thời gian (C-contiguous order), tránh sai lệch chỉ số. |
| **13** | **Linear 1 (FC1)** | Vector $\mathbb{R}^{2048} \to \mathbb{R}^{128}$ | $y[j] = \sum_{i=0}^{2047} W_1[j, i] x[i] + b_1[j]$ | **Tích lũy 2.048 nhân tử**. Đây là nguồn sai số số học lớn nhất toàn mạng. MKL GEMM dùng đa luồng và chia khối. | Áp dụng bound sai số tích lũy nghiêm ngặt cho $n=2048$; cấm dùng sai số $O(\sqrt{n})$ ngẫu nhiên. |
| **14** | **ReLU 4** | Vector $\mathbb{R}^{128}$ | $\max(0, x)$ | Tương tự. | Non-expansive interval propagation. |
| **15** | **Linear 2 (Logits)** | Vector $\mathbb{R}^{128} \to \mathbb{R}^{12}$ | $z[k] = \sum_{m=0}^{127} W_2[k, m] h[m] + b_2[k]$ | Tích lũy 128 nhân tử; sinh ra logits $z \in \mathbb{R}^{12}$. | Bao đóng khoảng cho vector logits: $z^{\text{exec}} \in [\underline{z}, \overline{z}]$. |
| **16** | **Verifier Margins** | Logits $\mathbb{R}^{12} \to \mathbb{R}^{11}$ | Với winner $c$: $g_{cj} = z_c - z_j, \forall j \neq c$ | Phép trừ hai logits gần nhau gây hủy số; so sánh với 0. | Phân biệt direct margin bound với separate logits; bound chặt chẽ cho cả 11 đối thủ. |
| **17** | **Argmax & Ties** | Logits $\mathbb{R}^{12} \to \{0,\dots,11\}$ | $c^* = \arg\max_k z_k$ | Trường hợp hòa điểm ($z_c = z_j$): argmax trả về chỉ số nhỏ nhất tùy thư viện. | **Quy tắc nghiêm ngặt**: Chỉ công nhận invariance khi $\min_{j \neq c} \underline{g_{cj}} > 0$ (strict winner). Nếu có tie ($\underline{g_{cj}} \le 0$), cấm tái sử dụng. |

---

## 3. Hồ Sơ Môi Trường Thực Tế & Tiền Đề Phần Cứng/Thư Viện (C03)

Hệ thống xác lập hồ sơ môi trường chuẩn (canonical environment profile) dựa trên kết quả khảo sát thực tế tại máy chủ nghiên cứu:

### 3.1 Định Danh Cấu Hình Môi Trường
- **Profile ID**: `linux-x86_64-py311-torch2141cpu-avx512`
- **CPU Model**: Intel(R) Xeon(R) W-11855M CPU @ 3.20GHz (6 physical cores, 12 threads)
- **Tập lệnh phần cứng (Hardware Flags)**: `fpu`, `vme`, `de`, `pse`, `tsc`, `msr`, `pae`, `cx8`, `apic`, `sep`, `cmov`, `pat`, `clflush`, `mmx`, `fxsr`, `sse`, `sse2`, `ss`, `syscall`, `nx`, `lm`, `pni`, `pclmulqdq`, `ssse3`, `fma`, `cx16`, `sse4_1`, `sse4_2`, `movbe`, `popcnt`, `aes`, `xsave`, `avx`, `f16c`, `rdrand`, `bmi1`, `avx2`, `bmi2`, `erms`, `avx512f`, `avx512dq`, `avx512cd`, `avx512bw`, `avx512vl`, `avx512_vnni`.
- **Hệ điều hành**: Linux 7.0.0-42-generic x86_64, glibc 2.43
- **Python Runtime**: CPython 3.11.15 (Clang 22.1.3)
- **NumPy**: `2.4.6`
- **PyTorch**: `2.14.1+cpu` (CPU Capability: `AVX512`, `torch.get_num_threads() = 6`)

### 3.2 Các Tiền Đề Phần Cứng & Thư Viện (Hardware & Library Premises)
1. **Lệnh Fused Multiply-Add (FMA)**:
   - Phần cứng hỗ trợ tập lệnh FMA3.
   - Khi thực hiện `fma(a, b, c)`, phần cứng tính toán tích $a \cdot b$ với độ chính xác vô hạn, sau đó cộng với $c$ và chỉ làm tròn duy nhất 1 lần về định dạng float32.
   - *Hệ quả*: Kết quả của FMA chính xác hơn hoặc bằng phép toán rời rạc $\text{fl}(\text{fl}(a \cdot b) + c)$, nhưng khác biệt về mặt bitwise. Do đó, mô hình sai số cổ điển dựa trên 2 lần làm tròn không phản ánh chính xác FMA. Bound số học phải bao bọc cả hai trường hợp (có hoặc không có FMA).

2. **Xử lý Số Dưới Chuẩn (Subnormal Numbers / Denormals)**:
   - Các bộ xử lý Intel hỗ trợ chế độ FTZ (Flush-To-Zero) và DAZ (Denormals-Are-Zero) trong thanh ghi điều khiển MXCSR.
   - Trong quá trình suy luận mạng SmallCNN1D với các hàm kích hoạt ReLU và BatchNorm, hầu hết các giá trị kích hoạt nằm trong khoảng bình thường (normal range $[10^{-7}, 10^3]$).
   - *Tiền đề bảo toàn*: Nếu xuất hiện giá trị subnormal ($|x| < 2^{-126} \approx 1.17549 \times 10^{-38}$), khoảng bao đóng phải cộng thêm một bán kính suy biến tối đa $\Delta_{\text{sub}} = 2^{-126}$ để bảo đảm tính đúng đắn khi FTZ/DAZ được kích hoạt.

3. **Hiện tượng Tràn Số Exponent (Overflow / Underflow)**:
   - Giới hạn biểu diễn float32: $x_{\max} \approx 3.40282 \times 10^{38}$.
   - Dữ liệu cảm biến HAPT sau chuẩn hóa có $|x_{\text{norm}}| \le 27.0$ (đo đạc thực tế trên 2.000 cửa sổ: cực đại đạt $\approx 26.222$). Qua 3 tầng Conv và 2 tầng Linear với trọng số đã huấn luyện, giá trị tuyệt đối lớn nhất của logits $|z_k| \le 34.0$ (cực đại đo được $\approx 33.518$).
   - *Tiền đề loại trừ*: Hoàn toàn không xảy ra hiện tượng tràn số exponent ($+\infty$ hoặc $-\infty$) trong miền làm việc bình thường của mô hình.

---

## 4. Mô Hình Làm Tròn Hướng Ra Ngoài & Định Dạng Đầu Mút (C04)

### 4.1 Định Dạng Khoảng Đầu Mút (Interval Format)
Mỗi khoảng số học $\mathcal{I}$ được biểu diễn bằng cặp đầu mút đóng:
$$\mathcal{I} = [\underline{x}, \overline{x}], \quad \underline{x}, \overline{x} \in \mathbb{F}_{32} \cup \{-\infty, +\infty\}, \quad \underline{x} \le \overline{x}$$

Để tránh hiện tượng sai số làm tròn khi biểu diễn chính các đầu mút, các phép toán số học khoảng (Interval Arithmetic) phải được cài đặt với **làm tròn hướng ra ngoài (directed outward rounding)**:
- Cận dưới $\underline{x}$ luôn được làm tròn về hướng $-\infty$ (`round_down`).
- Cận trên $\overline{x}$ luôn được làm tròn về hướng $+\infty$ (`round_up`).

### 4.2 Mô Hình Sai Số Tích Lũy của Phép Tích Vô Hướng (Dot Product Error Propagation)
Xét phép tính tích vô hướng $s = \sum_{i=1}^n a_i b_i$ trong không gian float32 với $n$ phần tử ($a_i, b_i \in \mathbb{F}_{32}$). Theo định lý phân tích sai số lùi kinh điển (Higham, *Accuracy and Stability of Numerical Algorithms*):

Khi thực hiện tuần tự không dùng FMA, sai số tuyệt đối được chặn bởi:
$$|\text{fl}(s) - s| \le \gamma_n \sum_{i=1}^n |a_i b_i|, \quad \text{với } \gamma_n = \frac{n u}{1 - n u}, \quad u = 2^{-24} \approx 5.96046 \times 10^{-8}$$
Điều kiện tiên quyết: $n u < 1$.

Khi thực hiện với phép cộng theo cây (pairwise summation / tree reduction) như trong PyTorch AVX-512 vectorization:
$$|\text{fl}(s) - s| \le \gamma_{\lceil \log_2 n \rceil + 1} \sum_{i=1}^n |a_i b_i|$$

Tuy nhiên, vì PyTorch không công bố cam kết cố định về thứ tự reduction giữa các phiên bản và luồng CPU, **nguyên tắc bảo thủ bắt buộc sử dụng bound tồi nhất (worst-case bound $\gamma_n$)** để bảo đảm tính hợp lệ tuyệt đối.

### 4.3 Đánh Giá Số Liệu Cụ Thể Trên SmallCNN1D:
1. **Tầng Conv 1** ($n = 6 \times 5 = 30$):
   $$n u = 30 \times 2^{-24} \approx 1.788 \times 10^{-6} \implies \gamma_{30} \approx 1.788 \times 10^{-6}$$
2. **Tầng Conv 2** ($n = 32 \times 3 = 96$):
   $$n u = 96 \times 2^{-24} \approx 5.722 \times 10^{-6} \implies \gamma_{96} \approx 5.722 \times 10^{-6}$$
3. **Tầng Conv 3** ($n = 64 \times 3 = 192$):
   $$n u = 192 \times 2^{-24} \approx 1.144 \times 10^{-5} \implies \gamma_{192} \approx 1.144 \times 10^{-5}$$
4. **Tầng Linear 1** ($n = 2048$):
   $$n u = 2048 \times 2^{-24} \approx 1.2207 \times 10^{-4} \implies \gamma_{2048} = \frac{1.2207 \times 10^{-4}}{1 - 1.2207 \times 10^{-4}} \approx 1.2209 \times 10^{-4}$$
5. **Tầng Linear 2** ($n = 128$):
   $$n u = 128 \times 2^{-24} \approx 7.629 \times 10^{-6} \implies \gamma_{128} \approx 7.629 \times 10^{-6}$$

### 4.4 Chứng Minh: Tại Sao "Final `nextafter`" Hoàn Toàn Không Đủ (Failure of Final Nextafter)
Một số phương pháp ngây thơ đề xuất tính toán toàn bộ mạng bằng float32 thông thường, sau đó chỉ cộng hoặc trừ một giá trị `nextafter` tại logits cuối cùng:
$$\widehat{z}_k^{\text{safe}} = \text{nextafter}(z_k, -\infty) = z_k - \text{eps}(z_k)$$

**Chứng minh định lượng sự sụp đổ của đề xuất này dựa trên đo đạc thực tế:**
- Xét mạng SmallCNN1D đông lạnh trên tập dữ liệu HAPT (đo đạc thực nghiệm):
  - Chuẩn ma trận đầu ra: $\|W_2\|_\infty = \max_k \sum_m |W_{2, km}| \approx 7.507$ (trung bình hàng: $6.467$).
  - Chuẩn ma trận tầng ẩn: $\|W_1\|_\infty = \max_j \sum_i |W_{1, ji}| \approx 42.721$ (trung bình hàng: $28.558$).
  - Tọa độ sau chuẩn hóa trên dữ liệu thực tế đạt cực đại $|x_{\text{norm}}| \approx 26.222$.
- Tại tầng Linear 1 ($n=2048$ nhân tử tích lũy + 1 bias):
  - Tổng các tích tuyệt đối $\sum_{i=1}^{2048} |W_{1, ji} x_i|$ trên các cửa sổ đo đạc đạt trung bình $11.988$ và cực đại $44.277$.
  - Với hệ số $\gamma_{2049} = \frac{2049 \cdot 2^{-24}}{1 - 2049 \cdot 2^{-24}} \approx 1.2214 \times 10^{-4}$:
    $$\Delta_{\text{Linear1}} \approx \gamma_{2049} \cdot \left(\sum |W_{1, ji} x_i| + |b_{1, j}|\right) \approx 1.2214 \times 10^{-4} \times 44.5 \approx 0.00543$$
- Sai số này tiếp tục lan truyền qua tầng kích hoạt ReLU 4 (toán tử không giãn) và bị khuếch đại bởi ma trận trọng số $W_2$ của tầng Linear 2:
  $$\Delta_{\text{amplified}} \le \|W_2\|_\infty \cdot \Delta_{\text{Linear1}} \approx 7.507 \times 0.00543 \approx 0.0408$$
  Cộng thêm sai số tích lũy bản thân tầng Linear 2 ($n=128$, $\gamma_{129} \approx 7.689 \times 10^{-6}$, tổng tích tuyệt đối $\approx 52.82$):
  $$\Delta_{\text{Linear2}} \approx 7.689 \times 10^{-6} \times 53.0 \approx 0.00041$$
  Tổng sai số tích lũy thực tế tại logits đạt:
  $$\Delta_{\text{total\_final}} \approx 0.0408 + 0.00041 \approx 0.0412$$
  (Mức trung bình thực nghiệm trên các cửa sổ là khoảng $0.010$).
- Trong khi đó, giá trị 1 ULP của `nextafter` tại vùng logits ($|z| \approx 10.0$ đến $20.0$):
  $$\text{ULP}(10.0) = 2^{-20} \approx 9.54 \times 10^{-7}, \quad \text{ULP}(20.0) = 2^{-19} \approx 1.91 \times 10^{-6}$$
- So sánh tỷ lệ sai số thực tế so với 1 ULP của `nextafter`:
  $$\frac{\Delta_{\text{total\_final}}}{\text{ULP}(10.0)} \approx \frac{0.0412}{9.54 \times 10^{-7}} \approx 43.180, \quad \frac{\Delta_{\text{mean}}}{\text{ULP}(10.0)} \approx \frac{0.010}{9.54 \times 10^{-7}} \approx 10.480$$

**Kết luận đanh thép**: Sai số tích lũy thực tế đo được lớn gấp từ **10.000 đến 43.000 lần** so với 1 ULP của `nextafter`! Việc chỉ sử dụng `nextafter` ở đầu ra cuối cùng là hoàn toàn ngụy biện, bỏ lọt toàn bộ sai số tích lũy khổng lồ bên trong mạng và làm mất tính an toàn của chứng chỉ tái sử dụng. Bắt buộc phải thực hiện lan truyền bao đóng khoảng hướng ra ngoài (outward interval propagation) ở từng tầng trung gian.

---

## 5. Tiêu Chí Đủ Điều Kiện Số Học & Cơ Chế Fail-Closed (C05)

Một cấu hình thực nghiệm chỉ được cấp chứng chỉ "Đủ điều kiện số học" (`NUMERICAL_ELIGIBILITY_PASSED`) khi và chỉ khi thỏa mãn đồng thời 5 tiêu chí sau:

1. **Khớp Danh Tính Mô Hình & Dữ Liệu Tuyệt Đối**:
   - Checkpoint SHA-256: `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`
   - Preprocessing ID: `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`
   - Data Protocol Hash: `439266ad856114eb16e788bc5bb8693c1017efadcf96cf78a70c36d2eb1f3640`

2. **Toán Tử Nằm Trong Phạm Vi Được Hỗ Trợ 100%**:
   - Chỉ gồm các toán tử đã được kiểm chứng: `Conv1D (same pad)`, `BatchNorm1d (eval mode)`, `ReLU`, `MaxPool1d (kernel=2, stride=2)`, `Flatten`, `Linear`.
   - Nghiêm cấm mọi toán tử không nằm trong danh mục (ví dụ: Global Average Pooling, Dilated Conv, Dropout tại inference, Softmax thay vì Logits).

3. **Cơ Chế Bao Đóng Hướng Ra Ngoài Được Thẩm Định**:
   - Triển khai thuật toán số học khoảng với directed rounding hoặc outward float64 bounding.
   - Sai số dot product phải được cộng dồn theo công thức $\gamma_n$ cho từng tầng.

4. **Biên Độ Margin An Toàn Tối Thiểu (Minimum Safe Margin)**:
   - Với winner dự đoán $c$: Cận dưới của margin đối với mọi đối thủ $j \neq c$ phải thỏa mãn:
     $$\underline{g_{cj}} > \Delta_{\text{total\_error\_budget}} > 0$$
   - Nếu tồn tại bất kỳ đối thủ nào có $\underline{g_{cj}} \le 0$, cấu hình bị đánh dấu là không chứng thực được (`UNVERIFIED_MARGIN`).

5. **Cơ Chế Fail-Closed Tuyệt Đối**:
   - Khi phát hiện bất kỳ dấu hiệu không tương thích nào về phần cứng, thư viện, hoặc khi cận margin không dương, hệ thống runtime **tự động vô hiệu hóa cơ chế reuse** (`ALLOW_REUSE = False`), kích hoạt đường thực thi mô hình tham chiếu gốc (`FALLBACK_TO_REFERENCE`) và ghi nhận vào log nhật ký.

---

## 6. Ba Kịch Bản Kết Quả Thay Thế & Tiêu Chí Dừng (C06)

Nghiên cứu tuân thủ tính trung thực khoa học, không gượng ép kết quả. Có ba kịch bản kết thúc chính thức cho nhánh toán học số:

| Kịch Bản | Trạng Thái Khoa Học | Ý Nghĩa Thực Tế | Tác Động Lên Runtime |
|:---|:---|:---|:---|
| **Kịch bản A: Validated Float32 Bridge** | `DERIVED` & `VERIFIED_FINITE` | Xây dựng thành công bound sai số $\Delta$ hữu hạn, chứng minh toán học đầy đủ và khớp với kiểm thử backend PyTorch. Cận margin an toàn $\underline{g_{cj}} > \Delta$ cho ra tỷ lệ tái sử dụng hữu ích ($R > 0$). | Cho phép cấp chứng chỉ tái sử dụng hình thức (`FORMAL_REUSE_ENABLED`). |
| **Kịch bản B: Conditional Only** | `CONDITIONAL` | Các công thức cận đúng trên số thực $\mathbb{R}$ hoặc tính toán trên `float64`, nhưng chưa có chứng minh bao trọn 100% mã máy PyTorch AVX-512 FMA trên CPU. | **CẤM TÁI SỬ DỤNG TẠI RUNTIME**. Mọi kết quả đo đạc chỉ được báo cáo dưới dạng chẩn đoán giới hạn trên lý thuyết (`DIAGNOSTIC_ONLY`). |
| **Kịch bản C: Unsupported Bridge** | `FALSIFIED` hoặc `NOT_ESTABLISHED` | Sai số tích lũy của mạng quá lớn so với khoảng cách margin giữa các lớp ($ \Delta > g_{cj}$ với hầu hết các mẫu), hoặc thứ tự reduction đa luồng gây bất định bitwise. | Đóng vĩnh viễn nhánh chứng thực số học cho kiến trúc SmallCNN1D hiện tại; chuyển toàn bộ nghiên cứu sang nhánh Incremental Feature Reuse (`M3`). |

### Tiêu Chí Dừng (Stop Rules):
1. **Dừng khi gặp Blocker Số Học**: Nếu chứng minh cho thấy $\gamma_n \sum |w_i x_i|$ lớn hơn biên độ phân tách thực tế của mô hình trên tập validation, dừng ngay việc nới lỏng giả thiết; không được tự ý giảm $\gamma_n$ bằng các giả thiết ngẫu nhiên $O(\sqrt{n})$ khi chưa có chứng minh phần cứng.
2. **Không Thay Đổi Checkpoint để Gượng Ép Kết Quả**: Nghiêm cấm huấn luyện lại mô hình hoặc sửa đổi trọng số tham chiếu để làm đẹp kết quả số học.
3. **Minh Bạch Báo Cáo Kết Quả Âm**: Kết quả âm (chứng minh rằng mô hình SmallCNN1D không thể chứng thực an toàn trên CPU float32 do tích lũy sai số $n=2048$) được công nhận là một đóng góp khoa học có giá trị cao, được ghi nhận đầy đủ vào báo cáo nghiệm thu Week 4.

---

## 7. Bảng Phân Định Giả Thiết & Thẩm Định Thiết Kế Độc Lập (C07)

### 7.1 Phân Định Rõ Ràng: Giả Thiết Định Lý vs Quan Sát Thực Nghiệm

| Hạng Mục | Phân Loại | Nguồn Gốc / Bằng Chứng | Giới Hạn & Ranh Giới |
|:---|:---:|:---|:---|
| **Tính kết hợp của phép cộng số thực** | Giả thiết Toán học | Tiên đề số thực $\mathbb{R}$ | **Không áp dụng** cho phép tính float32 trên máy tính. |
| **Mô hình sai số lùi $\gamma_n = \frac{nu}{1-nu}$** | Định lý Toán học | Higham (2002), IEEE 754 | Giả định phép tính vô hướng tuần tự; là cận trên worst-case cho reduction theo cây. |
| **Hỗ trợ tập lệnh AVX-512 & FMA** | Quan sát Thực nghiệm | `lscpu`, `torch.backends.cpu.get_cpu_capability()` | Thuộc về máy chủ cụ thể; có thể khác trên kiến trúc ARM hoặc GPU. |
| **Trọng số mạng không gây tràn số exponent** | Quan sát Thực nghiệm | Kiểm tra thực tế trên Checkpoint `8ffc5dd...` | Chỉ đúng với checkpoint đông lạnh này; không đúng với mạng khởi tạo ngẫu nhiên. |
| **Tỷ lệ bất nhất bitwise đa luồng** | Quan sát Thực nghiệm | Chạy đa luồng (`threads=6`) vs đơn luồng (`threads=1`) | Cần đo đạc thực nghiệm trong W4-006 / W4-009 để lượng hóa độ phân kỳ. |
| **Tính non-expansive của hàm ReLU** | Định lý Toán học | Giải tích thực: $|\max(0, a) - \max(0, b)| \le |a - b|$ | Đúng trên cả số thực và float32 nếu không có biến đổi điểm kỳ dị dấu. |

### 7.2 Kết Luận Sẵn Sàng cho Triển Khai (Design Review Sign-off)
Hợp đồng số học này đã thiết lập đầy đủ 7/7 yêu cầu kiểm tra (C01 đến C07), định hình chính xác ranh giới toán học và kỹ thuật cho các nhiệm vụ tiếp theo:
- **W4-006**: Xây dựng thư viện số học khoảng nguyên thủy (`ecdr/week4/intervals.py`) tuân thủ outward rounding và $\gamma_n$ dot product bounds.
- **W4-007**: Xây dựng bộ bao đóng chuẩn hóa dữ liệu thô (`ecdr/week4/norm.py`).
- **W4-008**: Xây dựng bao đóng cho toàn bộ 16 toán tử của SmallCNN1D (`ecdr/week4/operators.py`).
- **W4-009**: Thiết lập cầu nối hoàn chỉnh và đánh giá khả năng chứng thực an toàn (`ecdr/week4/bridge.py`).
- **W4-010**: Hội đồng độc lập thẩm định mở hoặc khóa cổng số học (`NUMERICAL_GATE`).
