# 04_SYNTHESIS/week4/design/AFFINE_DOMAIN.md — PCA/Affine-Tail Domain Specification & Bounded Residual Contract

**Task:** W4-016 (Milestone M2)  
**Tác giả:** Antigravity (Pair Programming Agent)  
**Trạng thái:** AUTHORITATIVE DESIGN DOCUMENT  
**Hypotheses:** H10, H11, H12, H18, H22  

---

## 1. Cơ sở Toán học & Định nghĩa Miền Affine Đuôi (Mathematical Foundation)

Trong luồng dữ liệu chuỗi thời gian liên tục với 50% độ trượt (stride = 64 bước, window = 128 bước):
- **Phần tiền tố (Prefix - 64 bước đầu, 384 chiều):** Đã được quan sát ở cửa sổ trước $t$, do đó hoàn toàn cố định bit-exact theo giả thuyết **H01** ($x_{t+1}[:, :64] = x_t[:, 64:]$).
- **Phần đuôi (Tail - 64 bước sau, 384 chiều):** Là phần dữ liệu mới cập bến, chứa động lực học biến thiên của cảm biến.

Miền hộp độc lập theo tọa độ (coordinate-aligned box) bỏ qua hoàn toàn tương quan không gian - thời gian giữa 6 kênh cảm biến (gia tốc kế và con quay hồi chuyển 3 trục). Để nắm bắt tương quan này mà không làm bùng nổ thể tích hộp bao ngoài, ta định nghĩa **Miền Affine Đuôi (Affine-Tail Domain)**:

$$\mathcal{D}_{\text{tail}} = \left\{ y \in \mathbb{R}^{384} \;\middle|\; y = c + B \operatorname{diag}(a) u + \operatorname{diag}(e) v, \quad \|u\|_\infty \le 1, \; \|v\|_\infty \le 1 \right\}$$

Trong đó:
1. $c \in \mathbb{R}^{384}$: Tâm đề xuất (Center), mặc định dùng *constant continuation* từ bước cuối cùng đã quan sát: $c = x_t[:, -1:] \otimes \mathbf{1}_{64}$.
2. $B \in \mathbb{R}^{384 \times K}$: Ma trận cơ sở trực chuẩn (Orthonormal PCA basis) gồm $K$ hướng phương sai lớn nhất của biến thiên đuôi, thỏa mãn $B^T B = I_K$.
3. $a \in \mathbb{R}_+^K$: Vector bán kính không gian ẩn (Latent singular values/radii) dọc theo các thành phần chính.
4. $e \in \mathbb{R}_+^{384}$: Vector cận phần dư (Residual coordinate bound), bảo đảm miền bao phủ cả những biến thiên nằm ngoài không gian con rank-$K$.
5. $u \in [-1, 1]^K, v \in [-1, 1]^{384}$: Các biến nhiễu chuẩn hóa độc lập.

---

## 2. Quy trình Huấn luyện Nghiêm ngặt (Strict Train-Only Fitting)

Để tuyệt đối tránh rò rỉ dữ liệu (data leakage) và tuân thủ checklist **W4-016-C02**:
- Chỉ sử dụng các cặp cửa sổ liên tiếp thuộc phân chia **Train split** (9,978 cửa sổ, tương ứng 9,944 cặp kế tiếp trong cùng phiên ghi âm `exp`).
- Tuyệt đối KHÔNG sử dụng tập Validation hoặc tập Test để fit basis $B$, mean $\mu$ hay residual $e$.
- Quy trình fit:
  1. Tính các vector sai phân đuôi trên tập Train: $\Delta y^{(i)} = x_{t+1}^{(i)}[:, 64:] - x_t^{(i)}[:, 64:] \in \mathbb{R}^{384}$.
  2. Tính trung bình mẫu $\mu = \frac{1}{N} \sum_{i=1}^N \Delta y^{(i)}$ và ma trận hiệp phương sai mẫu.
  3. Phân rã giá trị kỳ dị (SVD) hoặc trị riêng để rút ra $K$ vector trực chuẩn $B \in \mathbb{R}^{384 \times K}$.
  4. Xác định độ co giãn $a_k = \max_i |b_k^T (\Delta y^{(i)} - \mu)|$ hoặc quantile $99.9\%$.
  5. Tính phần dư trực giao: $r^{(i)} = (I - B B^T)(\Delta y^{(i)} - \mu)$ và gán vector chặn trên phần dư: $e_j = \max_i |r^{(i)}_j|$.
  6. Băm mật mã SHA-256 toàn bộ các mảng $B, a, \mu, e$ và lưu trữ vào `domain_fit.json`.

---

## 3. Thuật toán Kiểm tra Thành viên (Sufficient Membership Check)

Cho một query tail thực tế $y \in \mathbb{R}^{384}$:
1. Tính độ lệch tâm: $\delta = y - c$.
2. Chiếu trực giao lên không gian con PCA:
   $$u^* = \operatorname{diag}(a)^{-1} B^T \delta \in \mathbb{R}^K$$
3. Tính phần bù trực giao (Orthogonal residual):
   $$r^* = \delta - B \operatorname{diag}(a) u^* = (I - B B^T) \delta \in \mathbb{R}^{384}$$
4. Điều kiện thành viên đầy đủ (Sufficient Feasibility Condition):
   - $\|u^*\|_\infty \le 1.0$ (Tọa độ ẩn nằm trong siêu lập phương đơn vị).
   - $|r^*_j| \le e_j, \; \forall j \in \{1..384\}$ (Phần dư nằm trong hộp chặn sai số).
5. Nếu cả hai điều kiện thỏa mãn, $y \in \mathcal{D}_{\text{tail}}$ được xác nhận toán học chặt chẽ.
   - **Lưu ý (W4-016-C04):** Nghiêm cấm sử dụng dung sai tùy tiện ($\epsilon$-tolerance) trong bình phương tối thiểu (least squares) để ép nhận thành viên.

---

## 4. Hộp Bao Ngoài Minh Bạch (Outer Bounding Box Relaxation)

Do các công cụ IBP và CROWN tiêu chuẩn xử lý trên các hộp song song trục tọa độ (axis-aligned boxes), miền affine $\mathcal{D}_{\text{tail}}$ được chuyển đổi sang hộp bao nhỏ nhất (minimal enclosing axis-aligned bounding box) thông qua bất đẳng thức Cauchy-Schwarz:

$$y_j^L = c_j - \sum_{k=1}^K |B_{j, k}| a_k - e_j$$
$$y_j^U = c_j + \sum_{k=1}^K |B_{j, k}| a_k + e_j$$

- **Ý nghĩa:** Hộp này đảm bảo $\mathcal{D}_{\text{tail}} \subseteq [y^L, y^U]$.
- **Đo lường Hull-loss:** Thể tích của hộp bao ngoài $[y^L, y^U]$ lớn hơn thể tích của ellipsoidal affine domain. Tỷ số thể tích hoặc độ rộng cạnh phản ánh chi phí giãn nở (hull loss) khi nới lỏng từ miền tương quan sang miền hộp tọa độ (kiểm chứng giả thuyết **H10**).
