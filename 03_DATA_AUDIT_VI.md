# 03 DATA AUDIT — Tính Khả Thi Của HAPT & Bằng Chứng Miền Ứng Viên (G1)

> Tài liệu đối chiếu ngày 22/10/2026; các số liệu đo lường bên dưới là kết quả sau khi sửa lỗi ngày 15/10/2026, không phải một lượt chạy mới. Kết luận G1 hiện tại: **COMPLETE_WITH_REVISIONS** (trước đây là REVISE). Gate G2 PASS; Gate G3 PENDING. Bản kiểm toán này đo đạc các khoảng chênh lệch cửa sổ thô và tỷ lệ thuộc miền ứng viên, không phải tái sử dụng có chứng nhận. Định nghĩa miền xem tại: [04_THEORY.md](04_THEORY.md); khế ước tối thiểu tiếp theo xem tại: [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).

---

## 1. Cấu trúc nguồn thô (Được xác minh từ trang UCI id=341)

- `RawData/acc_expXX_userYY.txt`: các hàng gia tốc 3 trục ở tần số $50\text{Hz}$.
- `RawData/gyro_expXX_userYY.txt`: các hàng vận tốc góc 3 trục ở tần số $50\text{Hz}$.
- Các cột trong `RawData/labels.txt`: ID thử nghiệm, ID người dùng, ID hoạt động, mẫu bắt đầu, mẫu kết thúc (ở $50\text{Hz}$).
- `activity_labels.txt`: 12 ID = 6 hoạt động cơ bản (đứng, ngồi, nằm, đi bộ, lên dốc, xuống dốc) + 6 chuyển tiếp (đứng sang ngồi, ngồi sang đứng, ngồi sang nằm, nằm sang ngồi, đứng sang nằm, nằm sang đứng).
- `Train/X_train.txt` (81MB) + `Test/X_test.txt` (33MB): các cửa sổ đặc trưng trích thủ công 561 chiều + nhãn $y$ + subject_id (1–30), đặc trưng chuẩn hóa trong khoảng $[-1, 1]$. Tín hiệu thô dùng ở đây có gia tốc theo đơn vị $g$ và con quay theo $\text{rad/s}$; các tệp đặc trưng thủ công không phải đầu vào của bài toán luồng trực tuyến.
- Tổng dung lượng tải về 75.9MB, 133 tệp. Giấy phép CC BY 4.0, DOI 10.24432/C54G7M.
- **ĐIỂM MẤU CHỐT cho luồng trực tuyến:** UCI công bố tín hiệu thô thay vì bản đã cắt cửa sổ sẵn nhằm phục vụ các phép thử trực tuyến (online tests). Bắt buộc dùng `RawData` + `labels.txt` để dựng lại các cửa sổ theo đúng trình tự thời gian; TUYỆT ĐỐI KHÔNG dùng phép xáo trộn (shuffle) của Train/Test cho đánh giá luồng trực tuyến.

---

## 2. Câu hỏi kiểm toán và phạm vi bằng chứng

Các câu hỏi này giữ nguyên động lực ban đầu của Gate G1. Kịch bản thực tế triển khai các hộp ứng viên theo mức phân vị và PCA tập trung; kịch bản chưa triển khai bộ phát hiện sự kiện ngữ nghĩa, verifier, chính sách fallback có chứng nhận hay vòng đời mỏ neo. Một cửa sổ là mảng quan sát $6 \times 128$; một bộ mô tả sự kiện sẽ chọn ứng viên; một miền là tập hợp các cửa sổ khả dĩ. Đây là các đối tượng tách biệt hoàn toàn.

### Q1 — Phân phối khoảng chênh lệch thời gian (Gối nhau $\neq$ Tương đồng)
- Dựng các cửa sổ $6 \times 128$ bước nhảy (stride) 64 theo từng thử nghiệm theo thứ tự thời gian.
- Đo $\|x_{t+1} - x_t\|_2$ và $\|\cdot\|_\infty$ giữa các cửa sổ liên tiếp, phân tách theo cùng hoạt động so với chuyển tiếp/ranh giới nhãn.
- Báo cáo các phân vị chênh lệch p50/p90/p99. Gate G2 hiện đã có, nhưng chưa có tỷ lệ thay đổi dự đoán (PredictionChangeRate) toàn luồng hay biên margin chứng nhận miền nào được tạo ra từ bản kiểm toán này.
- Tiêu chí ĐẠT (PASS): khối lượng đáng kể các khoảng chênh lệch liên tiếp đủ nhỏ để MỘT VÀI tập sự kiện có thể bao phủ; các cửa sổ ranh giới được gắn cờ riêng.

### Q2 — Độ phủ của miền ứng viên
- Khớp các ngân sách đường chéo thô trên khoảng chênh lệch cùng hoạt động của tập train ở các mức q50/q75/q90; đánh giá điều kiện thuộc miền của các cửa sổ liên tiếp trên từng tập phân chia.
- Tên gọi stable/moderate/wide của script biểu thị các mức ngân sách, không phải các loại sự kiện ngữ nghĩa đã phân loại. Các miền chuyên biệt cho chu kỳ/chuyển tiếp/đột biến vẫn là các đề xuất.
- Đo lường độ phủ thuộc miền và các tập con cùng nhãn so với đổi nhãn. Không thuộc miền chưa phải là fallback thời gian chạy, và chỉ riêng thuộc miền chưa cấu thành tái sử dụng có chứng nhận.
- Khảo sát hộp điểm số PCA tập trung cộng chuẩn $L_2$ phần dư trực giao như một ứng viên riêng biệt; không đồng nhất nó với công thức hộp phần dư tọa độ.

### Q3 — Cân bằng nhãn + Quyết định $K$
- Đếm số lượng cửa sổ cho từng ID trong 12 ID từ nhãn thời gian start/end; báo cáo thời lượng chuyển tiếp (dự kiến ngắn và mất cân bằng nghiêm trọng).
- Quyết định phương án $K=12$ so với $K=6$ (+ quy tắc ranh giới: bầu đa số? bỏ qua? lớp chuyển tiếp riêng biệt?) TRƯỚC KHI huấn luyện backbone.
- Phân chia theo đối tượng (ví dụ 70/30 theo tình nguyện viên như UCI, hoặc LOSO), giữ nguyên thứ tự thời gian bên trong mỗi bản ghi. Tuyệt đối không ngẫu nhiên hóa các cửa sổ gối nhau giữa train/test.
- Khớp các phân phối/ngân sách thực nghiệm trên tập train; lựa chọn các mức/chính sách trên đối tượng validation và đóng băng trước khi đánh giá trên tập test giữ lại (held-out test). Kiểm toán hiện hữu báo cáo toàn bộ các mức ứng viên; nó chưa thiết lập một chính sách tái sử dụng cuối cùng.

---

## 3. Các kịch bản kiểm toán hiện hữu (Chưa có verifier)

- `scripts/01_fetch_hapt.py`: tải về và kiểm tra checksum nguồn gốc.
- `scripts/02_validate_raw.py`: kiểm tra tính thẳng hàng và toàn vẹn của dữ liệu thô.
- `scripts/03_build_windows.py`: dựng cửa sổ theo trình tự thời gian và quy tắc gán nhãn.
- `scripts/04_audit_splits.py`: kiểm tra tính rời nhau giữa các đối tượng/bản ghi và thứ tự.
- `scripts/05_audit_temporal_events.py`: khoảng chênh lệch thô, tương quan và tỷ lệ thuộc miền ứng viên.
- `scripts/06_audit_class_support.py`: số lượng hỗ trợ lớp/chuyển tiếp cho quyết định $K$.
- `scripts/06b_corr_semantics.py`: phân biệt các thang đo tương quan khác nhau.

---

## 4. Kết quả G1 (Đo lường ngày 15/10/2026, scripts 01–06b)

> **HIỆU CHÍNH 2026-10-15 (Kiểm chứng lại sau G1):** lượt chạy gốc của kịch bản `05` đã lập chỉ mục tệp `windows.npz` bằng chỉ số cửa sổ nội bộ `win_idx` thay vì chỉ số hàng toàn cục `row_id` — dẫn tới **98.2% các cặp liên tiếp** lấy tín hiệu từ các cửa sổ của `exp01` trong khi nhãn lấy từ bản ghi hiện tại. Các con số Q1/tương quan/độ phủ bên dưới đã được tính toán lại bằng kịch bản đã sửa lỗi (`05` + `row_id`; giá trị trước khi sửa được lưu trong lịch sử git tại commit `cefd171`). Các phát hiện F1–F3 thay đổi căn bản; kết luận REVISE mang tính lịch sử hiện được ghi nhận là COMPLETE_WITH_REVISIONS, không phải một phê duyệt vô điều kiện cho mô hình sự kiện.

### Tính toàn vẹn + Nguồn gốc (01, 02)
- Tệp Zip: 79.596.192 Bytes, SHA-256 `4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09`, 133 tệp con, DOI 10.24432/C54G7M, CC BY 4.0 (`data/manifest.json`).
- 61 bản ghi, 1.214 hàng nhãn, acc/gyro được ghép cặp **thẳng hàng hoàn hảo**, 0 lỗi, từ 9.898 đến 32.089 hàng/bản ghi (`data/raw_validation.json`).
- Độ phủ gán nhãn tối thiểu 0.3921 — tồn tại các khoảng trống không được chú thích, tuyệt đối không tự động gán nhãn.

### Cửa sổ + Nhãn (03)
- 17.453 cửa sổ ($6 \times 128$, stride 64), kích thước $(17453, 6, 128)$, 30 đối tượng, 61 bản ghi.
- Chính sách nhãn: thuần nhất (pure) 10.361 (59.4%) / không nhãn 3.587 (20.55%) / đa số (majority) 1.917 (11.0%) / nhập nhằng (ambiguous) 1.588 (9.10%); cắt qua ranh giới 7.29%. Chính sách đã ĐÓNG BĂNG trước G2.

### Phân chia tập dữ liệu (04)
- Train 17 / Val 4 / Test 9 đối tượng (test = tập Test của UCI), số cửa sổ tương ứng: 9.978 / 2.422 / 5.053.
- `subject_disjoint`: TRUE, `recording_disjoint`: TRUE, `chronological_kept`: TRUE, 0 lỗi phát sinh (`data/split_audit.json`).

### Q1 Khoảng cách thời gian (05, 17.392 cặp — Tính lại sau khi sửa lỗi chỉ mục)

Các chuẩn này sử dụng các kênh thô với các đơn vị vật lý khác nhau, không phải tọa độ mô hình chuẩn hóa. Chúng là các khoảng cách mang tính mô tả cho kiểm toán; không áp dụng trực tiếp làm bán kính verifier chuẩn hóa.

| Loại cặp cửa sổ | Số lượng $n$ | $L_2$ p50 / p90 | $L_\infty$ p50 / p90 |
|---|---|---|---|
| Cùng hoạt động (`same_activity`) | 9.278 | 1.84 / 17.35 | 0.39 / 2.80 |
| Đổi nhãn hoạt động (`label_change`) | 1.651 | 13.27 / 22.98 | 2.14 / 3.82 |

**Phát hiện F1 (Đã đính chính): Các khoảng chênh lệch thô TÁCH BIỆT RÕ RÀNG giữa cùng nhãn và đổi nhãn tại trung vị** — $L_2$ p50 cùng hoạt động là 1.84 so với đổi nhãn p50 là 13.27 (chênh lệch gấp ~7 lần); $L_\infty$ p50 là 0.39 so với 2.14. Sự trùng lặp chỉ còn ở phần đuôi phân phối (p90 là 17.35 so với 22.98), do đó nhận định "gối nhau $\neq$ tương đồng" vẫn đúng với một thiểu số cặp cùng hoạt động, nhưng tuyên bố trước khi sửa lỗi cho rằng các phân phối "hầu như không tách biệt" là hệ quả của lỗi lập chỉ mục.

### Tương quan thời gian (05, tính lại) + Ngữ nghĩa (06b)
Hệ số Pearson trễ 1 (lag-1) từng kênh sau khi sửa (các cửa sổ liên tiếp trải phẳng): acc_z 0.050, acc_y 0.025, acc_x 0.016, gyro_x -0.029, gyro_y -0.013, gyro_z -0.007; các độ trễ 2/4/8 $\approx 0$.
Ngữ nghĩa được đo bởi `06b_corr_semantics.py` (`data/corr_semantics.json`): đại lượng trong script 05 ghép cặp mẫu thô thời điểm $t$ với $t+64$ (1.28 giây) vì các cửa sổ gối nhau 50%.
Ba thang đo riêng biệt: **tín hiệu mẫu thô lag-1 ở 50Hz = 0.82–0.98**, **đặc trưng giá trị trung bình kênh cấp cửa sổ lag-1 = 0.56–0.97**, ghép cặp vị trí trải phẳng liên cửa sổ = -0.03…0.05.
**Phát hiện F2 (Khẳng định lại): Phụ thuộc thời gian là rất mạnh ở thang đo tín hiệu và đặc trưng cửa sổ, chỉ yếu ở thang đo ghép cặp vị trí trải phẳng** — mọi khẳng định bắt buộc phải nêu rõ thang đo; nhận định "dữ liệu có tương quan thời gian yếu" là không có cơ sở.

### Q2 Độ phủ miền sự kiện (Ngân sách chỉ khớp trên tập TRAIN; tính lại sau sửa lỗi)
| Hộp (Phân vị train) | Train | Val | Test | Test cùng nhãn | Test đổi nhãn |
|---|---|---|---|---|---|
| Stable p50 | 0.002 | 0.004 | 0.001 | 0.003 | 0.000 |
| Moderate p75 | 0.280 | 0.302 | 0.255 | 0.458 | 0.002 |
| Wide p90 | 0.330 | 0.353 | 0.319 | 0.525 | 0.023 |

- **Hộp p50 bị suy biến**: các ngưỡng p50 từng chiều trên các chênh lệch cùng hoạt động là siêu nhỏ (trung vị 0.023, cực đại 0.056) vì $\ge 50\%$ các cặp cùng hoạt động ở trạng thái gần như tĩnh, dẫn tới hộp kết hợp 768 chiều mỏng như lưỡi dao (độ phủ cùng nhãn trên train chỉ đạt 0.27%). Cần chọn lại mức ngân sách (q75/q90 hoặc theo hoạt động) — TUYỆT ĐỐI KHÔNG dùng mức q50 làm hộp "stable".
- Tỷ lệ rơi vào hộp stable nhưng bị đổi nhãn = **0.00%** tổng số lần trúng stable (trước khi sửa lỗi là 7.68%).
- **Hộp rộng p90 phân tách rất mạnh**: bao phủ 52.5% các cặp cùng hoạt động trên tập test trong khi chỉ để lọt 2.3% các cặp đổi nhãn.
- **Ứng viên phép chiếu PCA ($k=108$, phương sai giải thích $0{,}9008$): độ phủ 0.3663 / 0.3886 / 0.3542 so với hộp đường chéo q90 là 0.3296 / 0.3525 / 0.3194** — tăng +3.48 điểm phần trăm trên tập test (con số +8 điểm phần trăm trước sửa lỗi bị bãi bỏ). Kịch bản dịch tâm $\delta$ bằng trung bình train $\mu$, dùng $\text{scores} = (\delta - \mu) U$, và chấp nhận $|\text{scores}| \le b$ cùng $\|\delta - \mu - U \cdot \text{scores}\|_2 \le \epsilon$. Phần dư dùng chuẩn $L_2$, không phải ngưỡng từng tọa độ. Cơ sở PCA khớp trên tập con train, $\mu$ khớp trên mọi cặp train, và q90 điểm số/phần dư khớp trên các cặp cùng hoạt động tập train. Đây là các tham số ứng viên thực nghiệm, không phải chứng chỉ quyết định.

**Phát hiện F3 (Đã đính chính): Các hộp sự kiện chênh lệch thô hoạt động tốt hơn nhiều so với kiểm toán trước khi sửa; ưu thế độ phủ của PCA nhỏ hơn và CHƯA ĐƯỢC KHỚP NGÂN SÁCH ĐỒNG MỨC.** Phép so sánh cấu trúc thuộc về Gate G4 sau khi có verifier tối thiểu ở G3, với độ phủ/ngân sách được khớp công bằng. Tỷ lệ đổi dự đoán toàn luồng (PredictionChangeRate) hiện có thể tính từ backbone G2 đóng băng nhưng chưa có trong các đầu ra kiểm toán này. Đổi nhãn là siêu dữ liệu ground-truth, không phải vi phạm chứng chỉ.

---

## 5. Kết luận hiện tại: **COMPLETE_WITH_REVISIONS**

Nhãn lịch sử: REVISE. Tính khả thi của tập dữ liệu đã có bằng chứng; sự hữu ích của miền ứng viên cho suy luận có chứng nhận tiếp tục là một giả thuyết khoa học. Không có phép đo nào được chạy lại cho lượt đối chiếu tài liệu ngày 22/10/2026.

- **Tiêu chuẩn GO đã đạt:** tính toàn vẹn/thẳng hàng/phân chia/nhãn đều chuẩn xác; độ phủ cùng hoạt động của hộp p90 sau đính chính đạt 52.5% so với 2.3% cặp đổi nhãn trên test là tín hiệu đầy đủ để thử nghiệm tái sử dụng có điều kiện sự kiện. Không thuộc diện NO-GO.
- **Nghĩa vụ soát xét và đánh giá còn lại:**
  1. Khớp ngân sách phải tránh mức suy biến q50; sử dụng q75/q90 làm ứng viên, và báo cáo mức độ tách biệt cùng nhãn vs đổi nhãn thay vì chỉ đưa độ phủ thô. Ngân sách theo hoạt động đòi hỏi một bộ chọn trực tuyến độc lập; nhãn test ground-truth không thể dùng để kích hoạt cổng online.
  2. Các tuyên bố tương quan phải chỉ rõ thang đo (06b: tín hiệu 0.82–0.98 / đặc trưng cửa sổ 0.56–0.97 / ghép vị trí trải phẳng ~0.05). PCA và đường chéo cần một phép so sánh khớp công bằng trước khi khẳng định lợi ích từ cấu trúc (G4); đây không phải điều kiện tiên quyết cho bản thử nghiệm đường chéo G3 đầu tiên. Các phân vị có thể định hình các ứng viên kiểm chứng hình thức; G3 đòi hỏi các cận hợp lệ và kiểm tra thuộc miền bảo thủ trên đúng miền đã tuyên bố.
  3. Tỷ lệ không nhãn 20.55% + nhập nhằng 9.10% tiếp tục bị loại khỏi mục tiêu huấn luyện theo chính sách ĐÃ ĐÓNG BĂNG; báo cáo minh bạch mẫu số có nhãn (13.866) so với toàn luồng (17.453) trong mọi đánh giá liên quan. Kho thuần $K=6$ là 10.017 cửa sổ; luồng test giữ lại là 5.053 và test thuần $K=6$ là 2.874. Sự đồng thuận quyết định có thể kiểm tra trên mọi cửa sổ test mà không cần ground-truth; độ chính xác/F1 đòi hỏi phạm vi nhãn $K=6$ tường minh. Không gộp train/val/test làm một tập đánh giá.
- Quyết định $K$ phục vụ huấn luyện ghi nhận trong `G2_CLASS_DECISION.md` ($K=6$ baseline GO; $K=12$ thuần nhất TẠM DỪNG).
- Gate G2-B đã hoàn tất theo `G2_CLASS_DECISION.md` và các artifact đã lưu. Gate G3 tiếp tục ở trạng thái PENDING; `07_G3_DOMAIN_CONTRACT.md` đặc tả bản thử nghiệm đường chéo chuẩn hóa đề xuất. Bản khế ước tài liệu không thay thế việc triển khai verifier.
- Các điều kiện tuyệt đối đã xác minh trong code: script 03 không bao giờ cắt qua các bản ghi; phần không nhãn không bao giờ tự động gán nhãn.

---

## 6. Giới hạn miền và suy luận cho bản thử nghiệm tiếp theo

Sử dụng đầu vào thô $x$, phép chuẩn hóa train-fit cố định $N$, vector chuẩn hóa $z=N(x)$ và hàm tham chiếu $F=f_\theta \circ N$. Phải biến đổi ngân sách đường chéo thô bằng $\sigma$ hoặc kiểm chứng một đồ thị tính toán chứa cả phép chuẩn hóa; việc trộn lẫn ngân sách thô với đầu vào chuẩn hóa mà không biến đổi sẽ không có bảo đảm nào trên miền đã tuyên bố. Các đẳng thức tọa độ số thực vẫn cần một bản cài đặt dấu phẩy động bảo thủ.

Kiểm toán PCA là miền hộp điểm số/chuẩn $L_2$ phần dư có dịch tâm định nghĩa bằng phép chiếu; nó không phải hộp phần dư tọa độ không ràng buộc của lý thuyết trước đó. Quy tắc thuộc miền và hàm hỗ trợ của chúng khác nhau; xem lý thuyết Mục §2–3. Phép chuẩn hóa cũng làm thay đổi hình học trực giao của PCA thô và không thể bảo toàn quy tắc chiếu chỉ bằng cách thay $U$ bằng cơ sở tỷ lệ lại.

Mọi độ phủ báo cáo đều dành cho $x_{t+1}$ tương đối với cửa sổ ngay trước đó $x_t$. Nó không phải độ phủ tương đối với một mỏ neo tin cậy được giữ qua nhiều bước tái sử dụng. Việc xây dựng chứng chỉ và vòng đời mỏ neo như vậy hoàn toàn vắng mặt trong bản kiểm toán này, cũng như chưa có bằng chứng về tính đúng đắn số học hay sự tiết kiệm thời gian chạy thực tế.
