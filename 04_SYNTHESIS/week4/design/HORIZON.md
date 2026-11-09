# 04_SYNTHESIS/week4/design/HORIZON.md — Finite-Horizon Reachable Domains & No-Drift Reuse Policy

**Task:** W4-025 (Milestone M3)  
**Tác giả:** Antigravity (Pair Programming Agent)  
**Trạng thái:** AUTHORITATIVE DESIGN DOCUMENT  
**Hypotheses:** H12, H15, H16, H17, H19, H20  

---

## 1. Cơ sở Toán học: Hệ Thức Truy Hồi Dịch-Ghép (Shift-Append Recurrence)

Trong bài toán phân loại chuỗi thời gian trực tuyến (streaming HAR) trên tập dữ liệu HAPT:
- Cửa sổ phân tích có độ dài $W = 128$ mẫu (tương đương 2.56 giây ở tần số 50 Hz, gồm 6 kênh cảm biến gia tốc và con quay hồi chuyển).
- Bước trượt liên tục có độ dài $S = 64$ mẫu (50% overlap, tương đương 1.28 giây).

Cho điểm neo (anchor window) đã được mô hình tham chiếu suy luận tin cậy tại bước $t$:
$$Z_t \in \mathbb{R}^{6 \times 128}$$

### Horizon $h = 1$ (Bước $t+1$):
Cửa sổ $Z_{t+1}$ bắt đầu tại mẫu $(t+1)S = tS + 64$.
Theo vật lý dòng chảy cảm biến liên tục (Supplement 05 & giả thuyết **H01**):
$$Z_{t+1} = \left[ Z_t[:, 64:128], \; V_1 \right]$$
Trong đó:
- **Tiền tố $P_1 = Z_t[:, 64:128] \in \mathbb{R}^{6 \times 64}$:** Trùng khớp bit-exact 100% với 64 mẫu đuôi của cửa sổ $Z_t$. Đây là **384 tọa độ hoàn toàn cố định** (fixed, 0 chiều tự do).
- **Phần đuôi $V_1 \in \mathbb{R}^{6 \times 64}$:** Là 64 mẫu mới thu thập từ thế giới thực, thuộc miền dao động $\mathcal{T}_1 \subset \mathbb{R}^{6 \times 64}$.
- **Số chiều tự do (Free Dimensions):** $\dim_{\text{free}}(\mathcal{Z}^{(1)}) = 384$.

### Horizon $h = 2$ (Bước $t+2$):
Cửa sổ $Z_{t+2}$ bắt đầu tại mẫu $(t+2)S = tS + 128$.
Độ trượt tích lũy là $2 \times S = 128 = W$.
**Kết quả quan trọng:** Sau đúng 2 bước trượt, toàn bộ 128 mẫu của cửa sổ ban đầu $Z_t$ đã trôi qua hoàn toàn!
$$Z_{t+2} = \left[ Z_{t+1}[:, 64:128], \; V_2 \right] = \left[ V_1, \; V_2 \right]$$
Trong đó:
- **Tiền tố $P_2 = V_1 \in \mathbb{R}^{6 \times 64}$:** Chính là phần đuôi mới xuất hiện ở bước $t+1$.
- **Phần đuôi $V_2 \in \mathbb{R}^{6 \times 64}$:** Là 64 mẫu mới tiếp theo của bước $t+2$.
- **Tỷ lệ trùng lặp với anchor ban đầu $Z_t$:** $0.0\%$ (hoàn toàn bị thay thế).
- **Số chiều tự do (Free Dimensions):** $\dim_{\text{free}}(\mathcal{Z}^{(2)}) = 768$ (toàn bộ cửa sổ đều biến thiên).

---

## 2. Bao Đóng Tập Tích Ngoài (Product-Set Outer Box Enclosure)

Để chứng minh tính bất biến nhãn (label invariance) tại từng horizon mà vẫn tương thích với các bộ giải truyền truyền biên hình học (LiRPA / CROWN / IBP):

### Enclosure cho $h = 1$:
Với đuôi $V_1$ bị chặn bởi $[L_1, U_1] \subset \mathbb{R}^{6 \times 64}$:
$$\mathcal{B}^{(1)} = \left[ P_1, \; L_1 \right] \;\le\; Z_{t+1} \;\le\; \left[ P_1, \; U_1 \right]$$
- Cận dưới: $\underline{Z}_1 = \operatorname{concat}(P_1, L_1)$
- Cận trên: $\overline{Z}_1 = \operatorname{concat}(P_1, U_1)$
- Độ chặt biên được bảo toàn cao nhờ 50% tọa độ có độ rộng bằng 0 ($\overline{Z}_1[c, i] - \underline{Z}_1[c, i] = 0$ với mọi $i < 64$).

### Enclosure cho $h = 2$:
Vì $V_1 \in [L_1, U_1]$ và $V_2 \in [L_2, U_2]$, miền reachable ở $h=2$ là tích Descartes $\mathcal{T}_1 \times \mathcal{T}_2$:
$$\mathcal{B}^{(2)} = \left[ L_1, \; L_2 \right] \;\le\; Z_{t+2} \;\le\; \left[ U_1, \; U_2 \right]$$
- Cận dưới: $\underline{Z}_2 = \operatorname{concat}(L_1, L_2)$
- Cận trên: $\overline{Z}_2 = \operatorname{concat}(U_1, U_2)$
- **Outer Relaxation:** Dù $V_1$ ở bước 2 phụ thuộc thời gian vào $V_1$ ở bước 1, việc bao bọc bằng tích Descartes $\mathcal{T}_1 \times \mathcal{T}_2$ là một bao bọc ngoài (sound outer over-approximation) hợp lệ theo lý thuyết tập hợp, bảo đảm không bỏ sót bất kỳ quỹ đạo vật lý khả dĩ nào.

---

## 3. Chính Sách Không Trôi Dạt (No-Drift Policy) & Giả Thuyết H15

### Nguy cơ trôi dạt tiềm tàng (Drift Witness):
Nếu một hệ thống runtime cho phép **re-centering** (dời tâm chứng chỉ sang query mới mỗi khi tái sử dụng thành công) hoặc kéo dài hạn dùng vô hạn (**floating anchor**):
- Query tại $t+1$ sai khác nhỏ $\epsilon$ so với $Z_t$. Verifier chấp nhận và đổi tâm sang $Z_{t+1}$.
- Query tại $t+2$ tiếp tục sai khác $\epsilon$ so với $Z_{t+1}$, dời tâm sang $Z_{t+2}$.
- Sau $N$ bước, chuỗi truy vấn đã trôi xa $N \times \epsilon$ khỏi anchor gốc và chuyển hẳn sang hoạt động khác (ví dụ: từ WALKING sang STANDING), nhưng verifier vẫn liên tục tái sử dụng nhãn cũ mà không hề hay biết!

### Quy tắc Bất biến Cố định Điểm Neo (Fixed-Anchor Invariant):
Để loại bỏ 100% rủi ro trôi dạt vô hạn, `ecdr/week4` áp dụng các nguyên tắc fail-closed bắt buộc:
1. **Anchor Cố định Tuyệt đối:** Chứng chỉ $\mathcal{C}$ chỉ gắn với duy nhất một `anchor_step` $t_0$ và chữ ký mật mã `TrustedAnchorReceipt` của $Z_{t_0}$.
2. **Tuổi thực tăng đơn điệu:** Tuổi chứng chỉ tại bước $t$ luôn là $t - t_0$. Một lần tái sử dụng thành công (HIT) **TUYỆT ĐỐI KHÔNG RESET TUỔI** và không sinh lại anchor.
3. **Hạn dùng xác định (Deterministic Expiration):**
   Khi $t - t_0 > \text{max\_age}$ (mặc định $\text{max\_age} \in \{1, 2\}$):
   - Chứng chỉ lập tức hết hạn (`StoreFailureReason.EXPIRED_CERTIFICATE`).
   - Runtime bắt buộc phải fallback về suy luận chính xác của mô hình tham chiếu (`reference.infer()`), sinh anchor mới có provenance kiểm chứng được.
4. **Bảo vệ ngắt phiên (Session Boundary Reset):** Bất kỳ sự thay đổi mã ghi âm (`recording_id`) hoặc nhảy bước không liên tục (`step_idx != last_step + 1`) đều lập tức vô hiệu hóa toàn bộ chứng chỉ đang lưu trữ.

---

## 4. Bùng Nổ Chiều Tự Do & Giới Hạn $h \ge 3$

| Bước Horizon ($h$) | Mẫu bắt đầu | Mẫu kết thúc | Trùng lặp với $Z_0$ | Số chiều tự do | Tình trạng hỗ trợ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 1$** | $64$ | $191$ | **50%** (64 mẫu) | **384** | **Đầy đủ** (Exact Prefix + Bounded Tail) |
| **$h = 2$** | $128$ | $255$ | **0%** (0 mẫu) | **768** | **Hỗ trợ Outer Box** ($\mathcal{T}_1 \times \mathcal{T}_2$) |
| **$h \ge 3$** | $\ge 192$ | $\ge 319$ | **0%** (0 mẫu) | **768** | **Unsupported (Fail-Closed)** |

Khi $h \ge 3$:
- Sự tích lũy sai số bao bọc ngoài qua nhiều bước liên tiếp khiến biên chặn LiRPA bị nới rộng đáng kể.
- Giá trị lề dưới $\gamma_{\min}$ thường xuyên rơi xuống dưới 0 (không thể chứng nhận an toàn).
- Chi phí bộ nhớ và xác thực kiểm tra điều kiện không đem lại hiệu quả tái sử dụng.
- Vì vậy, theo checklist **W4-025-C04**, hệ thống từ chối dứt khoát $h \ge 3$ dưới dạng fail-closed (`FALLBACK_EXPIRED` hoặc `FALLBACK_UNSUPPORTED`), không bao giờ cố chấp suy rộng thiếu căn cứ.

---

## 5. Kết Luận & Bàn Giao Kiểm Chứng

1. `FiniteHorizonDomain` trong `ecdr/week4/domains.py` hiện thực hóa đầy đủ hệ thức truy hồi dịch-ghép cho $h \in \{1, 2\}$.
2. Thuật toán kiểm tra thành viên `check_membership()` kiểm tra bit-exact trên prefix ở $h=1$ và hộp tích ở $h=2$.
3. Chính sách No-Drift bảo đảm 100% độ chính xác so với mô hình tham chiếu trên mọi chuỗi thời gian streaming.
