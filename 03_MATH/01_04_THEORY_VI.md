> **Bản dịch lịch sử trước khi khóa reference K=12 (giữ để truy nguồn).**
> Các phát biểu K=6/G2 PASS hoặc “chưa triển khai” trong bản này mô tả snapshot cũ.
> Contract/trạng thái hiện tại ở [01_04_THEORY.md](01_04_THEORY.md), được cập nhật 03/11/2026.
> Không dùng bản lịch sử này làm evidence cho verifier hoặc metric K=12.

# 04 THEORY — Tái Sử Dụng Quyết Định Có Chứng Nhận Theo Điều Kiện Sự Kiện (ECDR)

> Cập nhật 22/10/2026. **Đặc tả hoạt động nghiên cứu; verifier và cơ chế certified reuse chưa triển khai, Gate G3 PENDING.** Hồ sơ nghiệm thu: G0 PASS; G1 COMPLETE_WITH_REVISIONS; G2 PASS. Lượt soát xét này đối chiếu hệ thống ký hiệu và các miền ứng viên với kết quả kiểm toán hiện hữu; không báo cáo thêm thực nghiệm mới và chưa đóng băng bản triển khai đã xác thực. Phạm vi tối thiểu của G3 xem tại: [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md). Kết quả đo lường đã lưu giữ tại `data/` và [03_DATA_AUDIT.md](03_DATA_AUDIT.md).

---

## 1. Đối tượng, hệ tọa độ và quyết định tham chiếu

### 1.1 Cửa sổ, bộ mô tả sự kiện và miền nhiễu động

- **Cửa sổ $x_t \in \mathbb{R}^{6 \times 128}$:** một quan sát cảm biến thô, được trải phẳng theo thứ tự kênh/thời gian gốc thành $\mathbb{R}^{768}$ khi cần thiết. Một cửa sổ kéo dài $2{,}56\text{s}$ ở tần số $50\text{Hz}$; bước nhảy (stride) $64$ tịnh tiến $1{,}28\text{s}$. Cửa sổ không phải là một sự kiện hay một miền.
- **Bộ mô tả sự kiện (Event descriptor) $e_t$:** thông tin đề xuất về sự biến thiên trong luồng tín hiệu, được bộ chọn (selector) dùng để xác định mẫu hình (template) hoặc ngân sách (budgets) ứng viên. Các đặc trưng cụ thể và chính sách chọn hiện chưa triển khai. Nó không nhất thiết phải bằng nhãn hoạt động thực tế (ground-truth). Các danh xưng như stable/locomotion/transition chỉ mang tính minh họa, chưa phải danh mục sự kiện ngữ nghĩa đã hiện thực hóa.
- **Miền ứng viên (Candidate region) $\mathcal{S}_e(x^r)$:** một tập hợp các cửa sổ thô tiềm năng được định nghĩa tương đối so với một mỏ neo tin cậy (trusted anchor), không phải một khoảng thời gian. Các ứng viên q75/q90/PCA hiện có chưa thiết lập một bộ chọn sự kiện trực tuyến (online event selector).
- **Miền được chứng nhận (Certified region):** một miền ứng viên đi kèm các cận hợp lệ chứng minh được tính chất bảo toàn quyết định yêu cầu. Chỉ riêng điều kiện thuộc miền (membership) không cấu thành chứng chỉ. Quá trình chứng nhận không nhất thiết làm thay đổi hình học của miền ứng viên.

### 1.2 Tiền xử lý đóng băng và mô hình nền tảng (Backbone)

Quy ước dùng $x$ cho đầu vào thô và $z$ cho đầu vào đã chuẩn hóa. Gọi $m$ và vector dương $\sigma$ là các tham số thống kê chuẩn hóa từng kênh khớp từ tập huấn luyện (train-fit), được lặp lại theo thời gian:

$$A = \text{diag}(1 / \sigma), \quad N(x) = A(x - m), \quad z = N(x)$$
$$f_\theta: \mathbb{R}^{768} \to \mathbb{R}^6, \quad F = f_\theta \circ N$$

$F$ là chuỗi xử lý tham chiếu (reference pipeline). Tệp `scripts/backbone.py` áp dụng chuẩn hóa bên ngoài mạng `SmallCNN1D`; mô hình hiện tại chứa BatchNorm, ReLU và MaxPool với 282.918 tham số. Việc xây dựng chứng chỉ bắt buộc phải sử dụng checkpoint đóng băng ở chế độ đánh giá (evaluation mode), bao gồm trạng thái động running của BatchNorm, thứ tự lớp và phép chuẩn hóa. Số lớp $K=6$ là cố định; phương án $K=12$ tiếp tục ở trạng thái TẠM HOÃN (HOLD) do thiếu dữ liệu hỗ trợ, không phải do Mệnh đề 1 bắt buộc 6 lớp.

Các phương trình afin này mô tả số học trên trường số thực. Chuỗi tính toán số thực dấu phẩy động float32 làm tròn các giá trị trung gian; Mục §3.4 nêu rõ nghĩa vụ chứng minh số học bổ sung.

Một **mỏ neo tin cậy (trusted anchor) $x^r$** là cửa sổ đã thực sự chạy qua chuỗi tham chiếu. Hệ thống lưu $z^r = N(x^r)$ và nhãn $a = \arg\max_k F_k(x^r)$. Bản khế ước tối thiểu bác bỏ các trường hợp hòa điểm (ties) và giá trị không hữu hạn (NaN/Inf). Một đầu vào được tái sử dụng không thể tự động trở thành mỏ neo tin cậy mới nếu không chạy lại suy luận tham chiếu hoặc không có một quy tắc suy diễn riêng biệt đã được chứng minh.

**Mục tiêu:** chỉ bỏ qua suy luận tham chiếu nếu tồn tại chứng chỉ hợp lệ và điều kiện thuộc miền của truy vấn xác lập cùng một quyết định. Cơ chế này bảo toàn quyết định của mô hình nền tảng, không bảo đảm tính đúng đắn so với nhãn thực tế. Một dự đoán sai của mỏ neo vẫn có thể được bảo toàn.

---

## 2. Các miền ứng viên và kiểm tra thuộc miền

### 2.1 Hộp đường chéo chuẩn hóa; mục tiêu G3 tối thiểu

Trong hệ tọa độ chuẩn hóa, miền ứng viên tối thiểu là:

$$\mathcal{Z}_e(z^r) = \{z^r + B_e u : \|u\|_\infty \le 1\}$$
$$B_e = \text{diag}(b_e), \quad b_e \ge 0, \quad \mathcal{S}_e(x^r) = N^{-1}(\mathcal{Z}_e(z^r))$$

Điều kiện thuộc miền toán học là $|z_i - z_i^r| \le b_{e,i}$ trên từng tọa độ. Miền tiền ảnh thô tương ứng có ngân sách $b_e^\text{raw} = \sigma \odot b_e$. Ngược lại, ngân sách kiểm toán thô tương ứng với $b_e = b_e^\text{raw} / \sigma$ trong tọa độ chuẩn hóa. Thứ tự tọa độ và tỷ lệ thuộc về danh tính của miền. Đây là các tương đương trên số thực; việc kiểm tra thuộc miền thực thi trên máy vẫn cần luận chứng số học.

Khi $B_e = \rho I$, miền là hộp $L_\infty$ bán kính $\rho$; khi $B_e = I$, bán kính là 1. Việc kiểm tra thuộc miền tốn $O(d)$ phép tính số học; thời gian chạy thực tế chưa được đo lường.

Kiểm toán khớp các phân vị delta thô từng tọa độ trên các cặp cùng hoạt động của tập train. Mức q50 có độ phủ đồng thời rất thấp; q75/q90 là các ứng viên cần đánh giá. Phân vị xác định hình dạng miền. Chúng không mang lại chứng chỉ quyết định lẫn bảo đảm thống kê cho độ phủ tương lai, nhưng không ngăn cản việc chứng nhận chặt chẽ miền đã chọn.

### 2.2 Miền chiếu PCA được đo lường trong kiểm toán

Tệp `scripts/05_audit_temporal_events.py` sử dụng hiệu số cửa sổ thô kề nhau $\delta = x_{t+1} - x_t$. Kịch bản khớp PCA tập trung (centered PCA) trên tập con train, với vector trung bình $\mu$ từ tất cả các cặp train và ngưỡng điểm/phần dư từ các cặp cùng hoạt động. Viết cơ sở trực chuẩn lý tưởng là $U \in \mathbb{R}^{d \times q}$, $U^T U = I$, và toán tử bù trực giao $P_\perp = I - U U^T$. Ứng viên đo lường là:

$$w = x - x^r - \mu$$
$$\mathcal{S}_e^\text{PCA}(x^r) = \{x : |U^T w| \le b, \; \|P_\perp w\|_2 \le \epsilon\}$$

Tương đương, trong số học thực:

$$x = x^r + \mu + U \text{diag}(b) u + v, \quad \|u\|_\infty \le 1, \; U^T v = 0, \; \|v\|_2 \le \epsilon$$

Các phép kiểm tra phép chiếu là điều kiện thuộc miền chính xác cho **miền định nghĩa bằng phép chiếu này**, giả định cơ sở trực chuẩn và số học chính xác. Phần dư sử dụng ngân sách chuẩn $L_2$, không phải hộp tọa độ. Việc dịch tâm bằng $\mu$ là cốt yếu. Miền tịnh tiến này không nhất thiết chứa $x^r$; chứng chỉ vẫn có thể so sánh các dự đoán trong miền với lớp nhận được tại $x^r$, nhưng không được giả định bao hàm mỏ neo.

Bản kiểm toán đã lưu báo cáo $q=108$, tỷ lệ phương sai giải thích $0{,}9008$ và độ phủ thuộc miền trên tập test là $0{,}3542$ so với hộp đường chéo q90 là $0{,}3194$. Đây là kết quả độ phủ trên cặp kề nhau, không phải tỷ lệ tái sử dụng có chứng nhận; chúng chưa được khớp đồng mức ngân sách/độ phủ. Phép phân rã float32 của script chưa được kiểm chứng là thủ tục runtime sound. Chưa có bản cài đặt chứng chỉ PCA nào được xác nhận.

Chuẩn hóa miền PCA thô này sẽ biến đổi các phương hướng và hình học phần dư; $AU$ nhìn chung không còn trực chuẩn. Không thay thế $AU$ vào công thức chiếu trực giao mà không hiệu chỉnh. Hãy kiểm chứng $F$ trên miền thô hoặc chuyển đổi cận/miền một cách nhất quán như Mục §3.3.

### 2.3 Biến thể lý thuyết riêng biệt: Hộp phần dư tọa độ

Miền trong bản thảo trước vẫn là một biến thể khả dĩ, phân biệt với kiểm toán PCA:

$$\mathcal{S}_e^\text{box} = \{c + D_e u + v : \|u\|_\infty \le 1, \; |v_i| \le \epsilon_i\}$$

Ở đây $D_e$ có thể không chéo và phần dư $v$ không nhất thiết trực giao. Việc kiểm tra thuộc miền tổng quát là bài toán tối ưu khả thi tồn tại. Một phân rã phép chiếu cụ thể thỏa mãn ràng buộc chỉ là nhân chứng đủ, nhìn chung không phải điều kiện cần và đủ cho miền tổng Minkowski này. Dự án chưa có số đo độ phủ hay verifier triển khai cho biến thể này.

### 2.4 Bộ chọn đối chiếu với Kiểm tra thuộc miền

Bộ chọn (selector) chỉ đề xuất miền/mẫu hình. Việc kiểm tra thuộc miền phải được thiết lập độc lập với thuật toán heuristic đó. Một tên sự kiện bị phân loại sai không làm mất hiệu lực của một chứng chỉ và phép kiểm tra thuộc miền hợp lệ. Một nhãn hoạt động đúng không thể thay thế cho cả hai điều kiện trên. Phép kiểm tra bảo thủ có thể từ chối các truy vấn hợp lệ và làm giảm độ phủ; chấp nhận sai (false acceptance) đe dọa tính đúng đắn (soundness) của toàn hệ thống.

---

## 3. Chứng chỉ và nghĩa vụ chứng minh

### 3.1 Điều kiện cặp lớp và Mệnh đề 1

Với lớp mỏ neo $a$, định nghĩa độ chênh lệch quyết định (margin) của chuỗi thô:

$$g_{aj}(x) = F_a(x) - F_j(x), \quad j \neq a$$

**Mệnh đề 1 — Điều kiện đủ bảo toàn quyết định.** Giả sử $a$ là lớp chiến thắng duy nhất của mô hình tham chiếu tại mỏ neo tin cậy. Với miền ứng viên $\mathcal{S}$, giả sử các cận dưới hợp lệ $\ell_{aj}$ thỏa mãn $g_{aj}(x) \ge \ell_{aj}$ với mọi $x \in \mathcal{S}$ và $j \neq a$. Nếu $\min_j \ell_{aj} > 0$, thì mọi $x \in \mathcal{S}$ đều có lớp chiến thắng duy nhất là $a$.

*Chứng minh.* Với bất kỳ $x \in \mathcal{S}$ và $j \neq a$, ta có $F_a(x) - F_j(x) \ge \ell_{aj} > 0$; do đó $F_a(x)$ vượt trội hơn mọi logit cạnh tranh. Phép kiểm tra truy vấn thuộc miền xác lập kết quả này áp dụng cho truy vấn đang xét. $\blacksquare$

Tính dương ngặt ($\min_j \ell_{aj} > 0$) là quy tắc chấp nhận đủ. Thuật toán cận dưới không nhất thiết chứng nhận mọi miền thực sự bất biến; cơ chế xử lý hòa điểm tất định (deterministic tie-breaking) cũng có thể bảo toàn quyết định tại margin bằng 0. Khế ước tối thiểu bác bỏ các trường hợp hòa điểm/không kết luận này thay vì xây dựng quy tắc hòa điểm riêng. Việc không chứng nhận được không đồng nghĩa với tính chất quyết định bị sai.

### 3.2 Cận dưới afin và hàm hỗ trợ (Support functions)

Giả sử phép nới lỏng afin hợp lệ cho $g_{aj}(x) \ge \alpha^T x + \beta$ trên toàn bộ miền thô. Cực tiểu hóa hàm afin đó trên các miền:

| Miền hình học | Cực hạ (Infimum) của cận dưới afin |
|---|---|
| Đường chéo thô $x^r + \text{diag}(b^\text{raw}) u$, $\|u\|_\infty \le 1$ | $\alpha^T x^r + \beta - \|\text{diag}(b^\text{raw}) \alpha\|_1$ |
| Hộp phần dư tọa độ $c + D u + v$, $|v| \le \epsilon$ | $\alpha^T c + \beta - \|D^T \alpha\|_1 - \sum_i \epsilon_i |\alpha_i|$ |
| Miền chiếu PCA kiểm toán Mục §2.2 | $\alpha^T (x^r + \mu) + \beta - \|\text{diag}(b) U^T \alpha\|_1 - \epsilon \|P_\perp \alpha\|_2$ |

$\|\cdot\|_1, \|\cdot\|_2, \|\cdot\|_\infty$ lần lượt là các chuẩn $L_1, L_2$ và $L_\infty$.

Số hạng phần dư của PCA là hàm hỗ trợ của quả cầu $L_2$ trong không gian bù trực giao của $U$. Nó khác với số hạng hộp tọa độ. Khi $D = \rho I$ và phần dư bằng 0, biểu thức thứ hai suy biến thành $\alpha^T c + \beta - \rho \|\alpha\|_1$.

Trong hệ tọa độ chuẩn hóa, nếu $f_a(z) - f_j(z) \ge h^T z + k$ hợp lệ trên toàn bộ hộp đường chéo chuẩn hóa, cận dưới an toàn là:

$$\ell_{aj} = h^T z^r + k - \|\text{diag}(b_e) h\|_1$$

Đây là các đẳng thức hàm hỗ trợ trên số thực, không phải bằng chứng cho thấy một lượt chạy CROWN đã sinh ra các hệ số hợp lệ cho mô hình đóng băng.

### 3.3 Tính hợp lệ của miền và biến đổi tọa độ

CROWN/LiRPA là công cụ đề xuất; mục tiêu G3 đầu tiên là hộp đường chéo chuẩn hóa. Một cận trên một tập bao (enclosure) cũng hợp lệ trên miền ứng viên nếu chứng minh được tính bao hàm. Các hệ số tính trên miền nhỏ hơn hoặc không liên quan sẽ không có bảo đảm hợp lệ ngoài miền đó. Kiểm tra thuộc miền và phép nới lỏng afin bắt buộc phải quy chiếu về cùng miền/phiên bản hoặc miền nới lỏng bao hàm đã được ghi nhận.

Nếu $h^T z + k$ hợp lệ với mọi $z \in N(\mathcal{S})$, biểu thức thô tương ứng là:

$$\alpha = A^T h, \quad \beta = k - h^T A m, \quad h^T N(x) + k = \alpha^T x + \beta$$

Khi đó có thể áp dụng các hàm hỗ trợ thô Mục §3.2. Biến đổi này không tự động xác lập tính hợp lệ của phép nới lỏng trên $N(\mathcal{S})$. Công thức hỗ trợ PCA không cung cấp hỗ trợ miền tương quan trong `auto_LiRPA`. Quá trình tích hợp phải kiểm chứng các toán tử và cận trên đồ thị thực tế chứa BN/MaxPool; hiện chưa có kết quả thực nghiệm.

### 3.4 Tính đúng đắn toán học và số học (Soundness)

Mệnh đề 1 chứng minh một hệ quả toán học lý tưởng. Sự nhất quán với mô hình thực thi float32 đòi hỏi mối liên kết có căn cứ giữa mã thực thi và các cận tính toán, đầu vào chuẩn hóa và phép kiểm tra thuộc miền. G3 phải văn bản hóa việc hỗ trợ toán tử, chế độ evaluation, giả định số học/làm tròn, xử lý số không hữu hạn và kiểm tra biên bảo thủ. Một dung sai dương tùy ý không phải bằng chứng kiểm soát sai số dấu phẩy động. Nếu tính đúng đắn số học chưa giải quyết trọn vẹn, phải dán nhãn bảo đảm là có điều kiện dựa trên giả định số học nêu rõ, thay vì tuyên bố formal soundness toàn diện.

Zero sai khác quan sát được trong thử nghiệm là bằng chứng triển khai, không phải bằng chứng cho mọi đầu vào được chấp nhận. Danh tính chứng chỉ gồm mô hình, chuẩn hóa, thứ tự lớp, mỏ neo, miền và ngữ nghĩa số học liên quan. Một chứng minh hợp lệ trên sai checkpoint/sai miền không có giá trị chứng nhận truy vấn hiện tại.

### 3.5 Ví dụ minh họa: Cấu trúc có thể mang lại lợi ích

Xét $g(x_1, x_2) = 1 + x_1 - x_2$, mỏ neo $(0, 0)$. Trên miền $|x_1|, |x_2| \le 0{,}8$, margin cực tiểu là $-0{,}6$. Khi $x_1 = x_2 = s$ với $|s| \le 0{,}8$, margin giữ nguyên bằng 1. Ngân sách phần dư tọa độ $|v_1|, |v_2| \le 0{,}1$ cho margin dưới là $0{,}8$. Ví dụ này minh họa lợi ích tiềm năng khi các phương hướng tự triệt tiêu trong margin; nó chưa xác lập độ phủ HAPT, chất lượng cận mô hình hay tiết kiệm runtime.

---

## 4. Cơ chế G3 đề xuất và đánh giá chi phí

Đường đi tối thiểu: mỏ neo tin cậy $\to$ chọn ứng viên đường chéo chuẩn hóa $\to$ xây dựng cận cặp lớp hợp lệ $\to$ lưu chứng chỉ thành công $\to$ kiểm tra danh tính trực tuyến và kiểm tra thuộc miền bảo thủ $\to$ tái sử dụng hoặc fallback chạy full. Xây dựng ứng viên, kiểm tra thuộc miền, verifier và reuse gate là các trách nhiệm tách biệt. Chi tiết xem tại [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md). Các thành phần này hiện chưa được hiện thực hóa.

Chỉ suy luận tham chiếu thực sự mới làm mới mỏ neo trong chính sách tối thiểu. Thiết lập lại (reset) giữa các bản ghi; bác bỏ phiên bản mô hình/tiền xử lý/miền đã lỗi thời. Xây dựng thất bại hoặc không kết luận, không thuộc miền, hòa điểm hoặc toán tử chưa hỗ trợ đều kích hoạt fallback. Cần đo lường chi phí từng phần và tuổi thọ mỏ neo trong G3; **Gate G5 quyết định net saving**, còn Gate G4 kiểm nghiệm giá trị cấu trúc.

Theo mô hình chi phí forward đại diện gần như không đổi:

$$\mathbb{E}[T] = T_e + T_m + (1 - r) T_f + q T_b$$
$$\text{Tiết kiệm (saving)} \iff r T_f > T_e + T_m + q T_b$$

Trong đó $T_e, T_m$ là chi phí trung bình xử lý sự kiện và tra cứu/thuộc miền; $r$ là tỷ lệ tái sử dụng; $q$ là số lượt dựng chứng chỉ trung bình mỗi truy vấn; $T_b$ là chi phí mỗi lượt dựng. Với chi phí forward phụ thuộc đường đi, hãy đo chi phí có điều kiện hoặc tổng trên từng truy vấn thay vì tách kỳ vọng mà không có căn cứ. Không giả định $q = 1 - r$. Bắt buộc phải tính chi phí khởi tạo ban đầu vào tổng chi phí dựng hoặc cộng thêm $T_\text{init} / N$ cho chu kỳ đánh giá đã công bố; tuyệt đối không bỏ qua.

Baseline đo trễ forward CPU batch 1 đã lưu: trung bình $0{,}2511\text{ms}$, p50 $0{,}2357\text{ms}$, p95 $0{,}3313\text{ms}$ (`data/g2_cpu_latency.json`). Đây là thời gian đo trên tensor RAM đã chuẩn hóa sẵn, không gồm tiền xử lý hay vết chạy fallback. G5 đòi hỏi đường chạy hoàn chỉnh có thể so sánh trên cùng khối lượng công việc/phần cứng, bao gồm quá trình dựng/dựng lại và tiền xử lý thực tế.

---

## 5. Chỉ số đo lường và thực nghiệm bắt buộc

- **Mức độ đồng thuận quyết định (Decision agreement):** số lượng/tỷ lệ sai khác giữa quyết định tái sử dụng và suy luận tham chiếu chạy ngầm (shadow reference). Zero sai khác quan sát được là bằng chứng cần thiết nhưng chưa đủ chứng minh. Chỉ báo cáo bảo đảm hình thức dưới các giả định toán học/số học đã xác thực.
- **Tỷ lệ tái sử dụng có chứng nhận (Certified reuse):** số truy vấn được tái sử dụng có chứng chỉ hợp lệ chia cho mẫu số luồng công bố rõ ràng. Báo cáo riêng biệt tỷ lệ dựng thành công, tỷ lệ trúng thuộc miền, nguyên nhân fallback, tuổi thọ mỏ neo, biên margin của cận và số lần dựng lại. Không có tái sử dụng đồng nghĩa với việc mức đồng thuận thực nghiệm là không xác định, không phải đo được 100%.
- **Độ chính xác tác vụ / Macro-F1:** chỉ tính trên phạm vi nhãn chuẩn $K=6$ công bố. Macro-F1 test tốt nhất hiện tại là $0{,}9354$; trung bình 3 seed là $0{,}9122 \pm 0{,}0183$ trên $2.874$ cửa sổ test thuần $K=6$, không phải độ chính xác toàn luồng.
- **Thời gian chạy (Runtime):** trung bình/p50/p95, thông lượng, chi phí dựng/thuộc miền/forward, bộ nhớ và môi trường phần cứng. Riêng tỷ lệ tái sử dụng không chứng minh có tiết kiệm điện toán.

Mẫu số chuẩn hóa: toàn bộ kho dữ liệu gồm $17.453$ cửa sổ, trong đó có $5.053$ cửa sổ test; kho thuần $K=6$ gồm $10.017$ cửa sổ, trong đó có $2.874$ cửa sổ test. $13.866$ cửa sổ có nhãn không đồng nhất với phạm vi tác vụ thuần $K=6$. Chọn chính sách trên tập validation, sau đó đánh giá trên các bản ghi test độc lập đã đóng băng; không gộp chung mọi phân chia làm test. Sự đồng thuận quyết định tham chiếu được xác định cả trên cửa sổ chuyển tiếp/chưa gán nhãn dù tính đúng đắn tác vụ $K=6$ không xác lập tại đó.

Độ phủ kiểm toán và chỉ số G2 hiện có đã từng được quan sát trên các đối tượng test. Không được mô tả tập test này là hoàn toàn chưa từng chạm tới trong toàn bộ quy trình; phải công khai việc khảo sát thăm dò và bất kỳ ảnh hưởng nào đến việc lựa chọn phương pháp sau đó.

Tách bạch hai phép so sánh: cố định mô hình + miền và thay đổi phương pháp cận; cố định mô hình + phương pháp cận và thay đổi miền toàn cục / đường chéo / tương quan. Xác định rõ tiêu chí khớp (độ phủ, ngân sách hoặc thước đo kích thước miền hợp lý) và báo cáo minh bạch. Các miền siêu nhỏ có thể được chứng nhận nhưng không bao phủ được truy vấn hữu ích nào. Riêng ưu thế độ phủ cặp kề của PCA không đủ để nghiệm thu G3 lẫn G4/G5.

---

## 6. Biên giới bằng chứng hiện tại

Bằng chứng hiện hữu xác lập tính khả thi của dữ liệu với các điểm hiệu chỉnh và mô hình nền tảng đã đóng băng. Các miền, đẳng thức hàm hỗ trợ và mệnh đề điều kiện đủ chỉ đặc tả một verifier đề xuất; chúng chưa khép lại Gate G3. Bộ verifier tối thiểu, kiểm tra thuộc miền số học bảo thủ, cổng tái sử dụng, vòng đời và hệ thống đo kiểm vẫn đang ở trạng thái cần triển khai và kiểm chứng. Bộ chọn sự kiện, lợi ích từ cấu trúc và net saving vẫn là bài toán mở. Tính mới mở rộng và net saving tiếp tục ở trạng thái CHƯA XÁC NHẬN trong [02_LITERATURE.md](02_LITERATURE.md).
