# 07 Khế Ước Miền G3 — Bản Thử Nghiệm Tái Sử Dụng Quyết Định Hộp Đường Chéo Tối Thiểu

> 22/10/2026 · **Khế ước đề xuất, chỉ mang tính tài liệu; chưa triển khai verifier hay runtime. Gate G3 PENDING.** Tệp này đặc tả bản thử nghiệm tiếp theo. Nó không làm thay đổi trọng số mô hình, chính sách nhãn/dữ liệu, kết quả đã lưu hay trạng thái các Gate G4/G5. Công thức toán học xem tại: [04_THEORY.md](04_THEORY.md).

---

## 1. Mục tiêu, phạm vi và ranh giới hoàn thành

Gate G3 bắt buộc phải kết nối các cận cặp lớp (class-pair bounds) hợp lệ với phép kiểm tra thuộc miền truy vấn bảo thủ và cổng tái sử dụng mỏ neo tin cậy cho mô hình đóng băng $K=6$. Mục tiêu đầu tiên là **hộp ứng viên đường chéo chuẩn hóa**; các hộp toàn cục (global boxes) là trường hợp đẳng hướng đặc biệt của nó. PCA và bộ chọn sự kiện ngữ nghĩa thuộc phạm vi sau này, không phải điều kiện tiên quyết cho chứng chỉ đường chéo đầu tiên.

Các đầu ra bắt buộc: bản ghi miền, verifier tối thiểu, cổng kiểm tra thuộc miền/tái sử dụng, vòng đời mỏ neo, hệ thống đo kiểm (instrumentation) và bằng chứng xác thực. Kết quả runtime dương không phải yêu cầu của G3: Gate G4 đánh giá cấu trúc và Gate G5 đánh giá net saving bao gồm cả quá trình dựng/dựng lại chứng chỉ. Việc tạo văn bản khế ước này chưa hiện thực hóa hay xác thực các đầu ra trên.

---

## 2. Đầu vào chuẩn tắc và tham chiếu đóng băng

| Đối tượng | Quy chuẩn khế ước |
|---|---|
| Cửa sổ thô $x$ | Kích thước $6 \times 128$, thứ tự kênh/thời gian gốc, không cắt qua các bản ghi; ghi nhận thứ tự trải phẳng |
| Chuẩn hóa | $z = N(x) = A(x - m)$, $A = \text{diag}(1 / \sigma)$, các tỷ lệ dương train-fit lặp lại theo thời gian |
| Hàm tham chiếu $F$ | Mạng `SmallCNN1D` đóng băng hợp thành với phép chuẩn hóa đóng băng: $F = f_\theta \circ N$ |
| Tham chiếu thực thi | Khớp trọng số, trạng thái running của BatchNorm, chế độ evaluation, tiền xử lý, thứ tự lớp, dtype/thiết bị và ngữ nghĩa toán tử |
| Mỏ neo tin cậy | Cửa sổ thô $x^r$ đã thực sự chạy qua tham chiếu; lưu $z^r$, logits và nhãn chiến thắng duy nhất $a$ |
| Bộ mô tả sự kiện | Thông tin tùy chọn dùng để chọn miền/mẫu hình; không phải tiên tri hoạt động hay bằng chứng toán học |

Thẩm quyền nguồn: `scripts/backbone.py` (`SmallCNN1D`, `apply_norm`), thư mục cấu hình `configs/`, trọng số checkpoint và các thông số thống kê chuẩn hóa. Mô hình hiện tại gồm 282.918 tham số chứa BatchNorm/MaxPool; việc tích hợp đồ thị thực tế bắt buộc phải được kiểm chứng. Một tên thư viện không tự động xác lập sự hỗ trợ toán tử hay tính hợp lệ của cận.

Các phương trình trên trường số thực không tự mô hình hóa việc làm tròn float32. Ngữ nghĩa số học và dung sai bao hàm sai số thuộc về phạm vi bảo đảm (Mục §5).

---

## 3. Bản ghi miền và kiểm tra thuộc miền

Với vector gióng theo tọa độ $b_e \in \mathbb{R}^{768}$, mọi phần tử không âm, định nghĩa:

$$\mathcal{Z}_e(z^r) = \{z^r + \text{diag}(b_e) u : \|u\|_\infty \le 1\}$$
$$\mathcal{S}_e(x^r) = \{x : N(x) \in \mathcal{Z}_e(z^r)\}$$

Điều kiện thuộc miền toán học là $|N(x) - z^r| \le b_e$ trên từng tọa độ. Ngân sách thô tương đương là $\sigma \odot b_e$; các phân vị thô train-fit chuyển đổi sang ngân sách chuẩn hóa bằng cách chia cho $\sigma$. Đây là các tương đương trên số thực; bộ kiểm tra thực thi phải xác lập tính bao hàm bảo thủ trong số học của nó.

$b_e = \rho \mathbf{1}$ cho bán kính chuẩn $L_\infty$ chuẩn hóa $\rho$; $b_e = \mathbf{1}$ cho bán kính 1. Cho phép ngân sách bằng 0 như các tọa độ suy biến với cơ chế xử lý biên bảo thủ. Bác bỏ các ngân sách âm/không hữu hạn và các truy vấn không hữu hạn (NaN/Inf).

Một bản ghi miền bất biến bắt buộc phải ràng buộc:
- ID miền/mẫu hình và phiên bản, không gian/thứ tự tọa độ, ID mỏ neo, toàn bộ ngân sách và kiểu dữ liệu (dtype), phương pháp/phiên bản xây dựng.
- Danh tính checkpoint/đồ thị, thứ tự lớp và danh tính phép chuẩn hóa ($m, \sigma$).
- Bất kỳ miền nới lỏng bao hàm nào, căn cứ bao hàm, giả định số học và chính sách làm tròn/sai số thuộc miền.
- Nguồn gốc ngân sách và phạm vi khớp: khớp trên train, chọn mức/chính sách trên val, đóng băng trước khi đánh giá trên test giữ lại.

Phân vị thực nghiệm có thể chọn ứng viên. Chúng không chứng nhận quyết định hay độ phủ tổng thể tương lai. Các cận hợp lệ trên các ứng viên đó vẫn có thể đem lại chứng chỉ quyết định sound. Bộ chọn không được âm thầm mở rộng hoặc đổi tâm một miền đã chứng nhận; mọi thay đổi đều cần danh tính cận/miền tương ứng.

---

## 4. Verifier tối thiểu và bản ghi chứng chỉ

Với lớp chiến thắng của mỏ neo $a$, thiết lập tất cả 5 đặc tả cặp lớp:

$$h_{aj}(z) = f_{\theta, a}(z) - f_{\theta, j}(z), \quad j \neq a$$

Sinh các cận dưới hợp lệ trên **toàn bộ $\mathcal{Z}_e$** hoặc một miền bao hàm đã được ghi nhận. Nếu phép nới lỏng afin hợp lệ cho $h_{aj}(z) \ge p_{aj}^T z + k_{aj}$, thì:

$$\ell_{aj} = p_{aj}^T z^r + k_{aj} - \sum_i b_{e, i} |p_{aj, i}|$$

Có thể sử dụng trực tiếp đầu ra cận hợp lệ mà không cần để lộ các hệ số afin. Nêu rõ thuật toán, toán tử hỗ trợ và các giả định chứng minh tính hợp lệ. Công cụ IBP/CROWN được hỗ trợ là đề xuất tích hợp đầu tiên; không giả định trước ưu thế tốc độ hay độ chặt trên đồ thị này trước khi thử nghiệm.

Một ứng viên chỉ nhận được chứng chỉ nếu mọi margin dưới an toàn đã kiểm chứng đều dương ngặt ($\ell_{aj} > 0$). Quy tắc này là điều kiện đủ, không phải điều kiện cần cho tính bất biến; việc không chứng nhận được không chứng minh quyết định bị thay đổi. Margin bằng 0, hòa điểm, toán tử chưa hỗ trợ, cận thất bại và các trường hợp số học bất định đều dẫn tới thất bại xây dựng kèm lý do, không phải chứng nhận thành công.

Lưu trữ danh tính miền/mỏ neo/mô hình/tiền xử lý, lớp chiến thắng $a$, toàn bộ 5 margin dưới an toàn, cấu hình/phiên bản verifier, trạng thái/thời gian dựng và bảo đảm số học. Tuyệt đối không áp dụng các cận từ một miền nhỏ hơn cho một ứng viên lớn hơn.

---

## 5. Nghĩa vụ chứng minh số học và bằng chứng xác thực

Phân biệt rõ ba cấp độ trong mọi kết quả:

1. **Hệ quả toán học:** các cận hợp lệ trên một miền đã công bố cộng với kiểm tra thuộc miền và margin dương ngặt sẽ suy ra quyết định của mô hình lý tưởng.
2. **Tính đúng đắn số học thực thi (Executable numerical soundness):** mã tính cận/thuộc miền dấu phẩy động thực tế xác lập các điều kiện đó một cách bảo thủ, và tính chất áp dụng cho tham chiếu thực thi đã chọn. Phải giải trình các bảo đảm được hỗ trợ hoặc các tập bao sai số/làm tròn đã kiểm chứng. Nếu sai số chuẩn hóa được xử lý bằng tính bao hàm đầu vào, tập bao của nó phải nằm vừa vặn bên trong miền chuẩn hóa được chứng nhận.
3. **Tính nhất quán thực nghiệm:** một tham chiếu chạy ngầm (shadow reference) đồng thuận trên các truy vấn được chấp nhận thử nghiệm. Báo cáo số lượng và phạm vi; zero sai khác quan sát được không thể thay thế cho nghĩa vụ chứng minh.

Một dung sai so sánh epsilon tùy ý hay một margin dương nhỏ không phải chứng minh số học. Nếu cấp độ 2 chưa được giải quyết, phải dán nhãn tính đúng đắn là có điều kiện dựa trên các giả định số học công bố; không tuyên bố formal soundness toàn diện hay coi như đã khép lại nghĩa vụ số học của G3. Các giá trị trung gian không hữu hạn và biên nhập nhằng phải kích hoạt thất bại/fallback một cách bảo thủ.

---

## 6. Cổng tái sử dụng và vòng đời mỏ neo tin cậy

Cổng kiểm tra đề xuất cho từng truy vấn:
1. Lấy mô tả sự kiện / lựa chọn mẫu hình nếu chính sách yêu cầu.
2. Đòi hỏi một chứng chỉ thành công với danh tính mỏ neo/mô hình/tiền xử lý/miền/số học tương thích và các margin dương hợp lệ.
3. Xác lập điều kiện thuộc miền bảo thủ của truy vấn trong miền đó.
4. Chỉ tái sử dụng lớp $a$ đã lưu khi mọi kiểm tra đều đạt. Ngược lại, chạy suy luận tham chiếu và trả về kết quả của nó.
5. Chỉ suy luận tham chiếu hoàn tất mới tạo ra mỏ neo tin cậy trong chính sách tối thiểu này. Một truy vấn được tái sử dụng không thể làm mới mỏ neo. Việc dựng chứng chỉ là một quyết định chính sách và chi phí riêng.

Thiết lập lại (reset) tại ranh giới bản ghi; tuyệt đối không so sánh qua các bản ghi khác nhau. Thay đổi mô hình, chuẩn hóa, thứ tự lớp, tham số miền hoặc ngữ nghĩa số học liên quan sẽ làm mất hiệu lực các chứng chỉ không tương thích. Không hợp thành các miền cục bộ thông qua các mỏ neo tái sử dụng chưa được kiểm chứng.

Sai sót của bộ mô tả sự kiện có thể ảnh hưởng đến độ phủ/chi phí mà không phá vỡ bảo đảm an toàn nếu danh tính, cận và kiểm tra thuộc miền vẫn hợp lệ. Nhãn hoạt động test ground-truth chỉ dùng cho phân tích ngoại tuyến, không dùng cho cổng trực tuyến.

---

## 7. Hệ thống đo kiểm và đánh giá thực nghiệm

Vết chạy truy vấn (trace) đề xuất phải ghi nhận: phân chia tập, bản ghi/cửa sổ, phiên bản mỏ neo và chứng chỉ/miền, mẫu hình đã chọn, lý do chấp nhận/fallback, tuổi thọ mỏ neo, kết quả thuộc miền, margin an toàn nhỏ nhất, cờ gọi tham chiếu, cờ dựng/dựng lại, lớp trả về và lớp tham chiếu chạy ngầm tùy chọn. Đo đạc riêng biệt thời lượng mô tả sự kiện, tra cứu/thuộc miền, forward, dựng/dựng lại chứng chỉ và tổng thời gian, cộng với chi phí bộ nhớ cache và chi tiết môi trường.

Báo cáo mẫu số toàn bộ truy vấn / truy vấn tái sử dụng, số lượt dựng mỗi truy vấn và số ca sai khác. Khi không có truy vấn tái sử dụng nào, ước lượng đồng thuận thực nghiệm là không xác định; không báo cáo 100% đồng thuận từ một mẫu số rỗng.

Chọn chính sách trên các bản ghi validation, sau đó đánh giá tuần tự trên luồng test giữ lại cố định kèm lệnh reset từng bản ghi. Luồng test đầy đủ: 5.053 cửa sổ. Sự đồng thuận tham chiếu có thể định nghĩa mà không cần ground-truth hoạt động. Độ chính xác/F1 phải giữ đúng phạm vi ground-truth $K=6$ công bố (số lượng test thuần $K=6$ hiện có là 2.874). Không gộp chung train/val/test làm test.

**Gate G3 ghi nhận chi phí từng thành phần; Gate G5 đánh giá mức tiết kiệm end-to-end.** Đối với G5, hãy so sánh suy luận đầy đủ và tái sử dụng trên cùng điều kiện tiền xử lý / thiết bị / tải công việc và chu kỳ thời gian. Tính cả chi phí khởi tạo ban đầu vào chi phí dựng hoặc cộng thêm $T_\text{init} / N$ trên chu kỳ đó. Bao gồm cả dựng lại, báo cáo trung bình/p50/p95, và lưu giữ đầy đủ cả các kết quả âm. Mô hình đơn giản hóa $T_e + T_m + (1-r)T_f + qT_b$ giả định chi phí forward gần như không đổi; các phép đo có điều kiện / trên từng truy vấn sẽ quyết định khi giả định đó không thỏa mãn.

---

## 8. Các biến thể PCA và hộp phần dư tọa độ thuộc phạm vi sau này

Bản kiểm toán thô dùng $w = x - x^r - \mu$, hộp điểm số $|U^T w| \le b$ và chuẩn $L_2$ phần dư trực giao $\|(I - UU^T) w\|_2 \le \epsilon$. Nó phân biệt với hộp phần dư tọa độ không ràng buộc của bản thảo trước. Xem lý thuyết Mục §2.2–2.3 và §3.2 để nắm định nghĩa và các hàm hỗ trợ khác nhau của chúng.

Không lấy độ phủ thuộc miền của PCA làm bằng chứng verifier. Bất kỳ triển khai tương quan nào cũng phải gắn chặt việc dịch tâm, hệ cơ sở, chuẩn/trực giao phần dư, biến đổi tọa độ và kiểm tra thuộc miền sound với đúng miền nới lỏng. Chứng chỉ đường chéo đầu tiên không chứng nhận cho các ứng viên khác này.

---

## 9. Danh mục kiểm tra hoàn thành hiện tại

- [x] Đã văn bản hóa miền đề xuất, vai trò, bản ghi và yêu cầu bằng chứng.
- [ ] Danh tính mô hình/miền/chuẩn hóa được nạp và xác thực trong một bản thử nghiệm.
- [ ] Verifier cặp lớp tối thiểu được tích hợp và các cận được giải trình chặt chẽ.
- [ ] Kiểm tra thuộc miền số học bảo thủ và tiêu chí chấp nhận được triển khai/xác thực.
- [ ] Cổng tái sử dụng, cơ chế hủy hiệu lực và vòng đời mỏ neo tin cậy được hiện thực hóa.
- [ ] Ghi nhận hệ thống đo kiểm, các trường hợp âm và bằng chứng tham chiếu chạy ngầm.
- [ ] Giải quyết triệt để nghĩa vụ tính đúng đắn toán học và số học thực thi.

Gate G3 tiếp tục ở trạng thái **PENDING**. Không có việc huấn luyện lại, kiểm toán mới, xây dựng chứng chỉ, đo đạc benchmark runtime hay thực nghiệm mới nào được thực hiện trong đợt cập nhật tài liệu này.
