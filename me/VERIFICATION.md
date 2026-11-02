# Kiểm tra bộ báo cáo ECDR lý thuyết

Ngày: **28/10/2026**. Trạng thái artifact: **CREATED và VERIFIED** trong phạm vi
biên soạn, kiểm công thức/số liệu và kiểm PDF dưới đây. Chưa gắn trạng thái
ACCEPTED; chưa xác nhận novelty, usefulness hoặc runtime benefit của ECDR.

## Đầu ra

| Báo cáo | Số trang PDF |
|---|---:|
| ECDR_Weeks_1_2_3_Theory | 31 |
| Week_1_Project_Brief | 6 |
| Week_2_Data_Foundations | 10 |
| Week_3_Mathematical_Foundations | 17 |

Cả bốn nguồn `.tex` là standalone, có bibliography nhúng và hình TikZ, không
cần file hình ngoài. Nguồn tổng hợp là canonical; ba nguồn tuần được tạo bằng
`build_sources.py`. Có 24 dòng hypothesis ledger H01–H24 và 11 nguồn bibliography
trong bản tổng hợp.

## Kiểm nội dung và toán học

- Đọc cả hai attachment, áp dụng correction mới: Project brief → data →
  mathematical foundations. Implementation cũ không xác định phương pháp mới.
- Tách schema HAPT 12 nhãn và tham số `K` tổng quát khỏi checkpoint K=6 cũ.
  CNN valid–ReLU–GAP là mô hình phân tích có giả thiết riêng, không có accuracy
  hay latency mới được gán cho nó.
- Có review toán độc lập cho các kết quả về window/overlap, continuity,
  Lipschitz, IBP, support/PCA, GAP, lifecycle và cost. Đã bổ sung các điều kiện
  được phát hiện: certificate hypothesis trong fixed-anchor theorem,
  preprocessing đóng dưới dịch, domain/nonnegative budgets của support/PCA,
  miền không rỗng của compactness và cách chọn Lipschitz upper bound.
- Kiểm lại bằng lập luận đại số phần thêm sau review: union bound cho joint
  coverage, ví dụ independence `q^768`, và selector composition. Không dùng một
  theorem prover, không claim máy đã xác minh các proof hình thức.
- Đối chiếu các số của mô hình lý thuyết: 20.332 parameters, 2.343.936 MAC,
  4.687.872 FLOP xấp xỉ cho Conv/head, RF 9, 120 feature positions, 56 shared
  positions; ví dụ tail lower margin 0,1; sensitivity/break-even đúng công thức.
- Những khẳng định không đúng khi bỏ giả thiết có phản ví dụ; coverage hữu ích,
  superiority và speedup thực tế được giữ là câu hỏi mở.

## Kiểm dữ liệu và phép tính

Lệnh đã chạy thành công:

```bash
.venv/bin/python me/recompute.py
```

- Đọc toàn bộ tensor lưu: shape `(17453, 6, 128)`, 13.403.904 scalar finite.
- 17.392 cặp cùng recording, `win_idx` liên tiếp và start tăng 64; so chính xác
  6.678.528 tọa độ overlap, mọi cặp bằng nhau, max absolute difference bằng 0.
- Count pure12 train/val/test 5.878/1.490/2.993 và pure6 5.693/1.450/2.874
  được đếm lại từ Parquet, khớp artifact audit.
- Chạy lại script tạo JSON giống hệt byte. SHA-256 của `calculations.json`:
  `e5a3243a4b4b1941b1666c9364e1fe3b2ee8704b65cda064409db1717d23007e`.

Check này xác nhận cấu trúc của corpus lưu, không xác nhận lại annotation
index convention trong dataset gốc, ý nghĩa vật lý bằng phép đo độc lập, hoặc
decision invariance của CNN. Các phần legacy của JSON chỉ là phép tính trên
artifact cũ, được tách khỏi lý thuyết và data audit mới.

## Biên dịch và PDF QA

Compiler trong app được thử với bản tổng hợp nhưng lỗi trước khi biên dịch vì
không tải được TeX bundle v33. Đã dùng **Tectonic 0.15.0 cục bộ** để biên dịch
thành công cả bốn báo cáo:

```bash
bash me/build.sh
.venv/bin/python me/verify_artifacts.py
```

- Cả bốn compile trả exit code 0.
- Không có missing character, overfull box, undefined citation hoặc undefined
  cross-reference trong final logs. Các underfull warning do cột bảng ngắn
  còn tồn tại; đã kiểm render, không thấy text đè hoặc tràn biên.
- `pdfinfo` đọc được cả bốn PDF; `pdftotext` trích được tiếng Việt, công thức,
  mục lục và bibliography; không có dấu `??` unresolved.
- Đã render toàn bộ trang của bốn bản bằng Poppler. Đã xem contact sheets của
  mọi trang, kiểm bảng, hình overlap, công thức, hypothesis ledger và nguồn;
  đã sửa lỗi font, tràn đường dẫn và một trang nội dung bị dồn ít dòng.
- File `qa/artifact_checks.json` ghi page counts, hashes, references/citations
  và kết quả kiểm tự động. Contact sheets/PNG dưới `qa/` là artifact kiểm tra.
- Font của bản PDF là Noto Serif/Sans/Mono; build trên máy khác cần các font
  tương ứng hoặc thay khai báo `fontspec`. PDF đã nhúng font để đọc độc lập.

## Ranh giới hoàn thành

Đã hoàn thành bộ báo cáo lý thuyết được yêu cầu và PDF đọc được. Chưa chạy
training, inference, verifier hoặc benchmark ECDR mới; chưa chứng minh
executable floating-point soundness, useful HAPT certified coverage hay lợi
ích runtime. Những giới hạn này được ghi trong nội dung báo cáo, không bị
thay bằng kết quả giả định. Các file mới của nhiệm vụ nằm dưới `me/`.

## Bổ sung hai chuyên đề data và CNN

Ngày 28/10/2026 bổ sung `ECDR_Data_Analysis` (23 trang) và
`ECDR_CNN_Model_Analysis` (24 trang), đều có `.tex` và `.pdf`. Các phép tính
raw và model, review độc lập, compile/render checks và giới hạn được ghi tại
[qa/data_cnn/VERIFICATION.md](qa/data_cnn/VERIFICATION.md); kết quả máy đọc tại
[qa/data_cnn/checks.json](qa/data_cnn/checks.json). SHA256 xác nhận năm nguồn
và PDF trước đó vẫn giữ nguyên. Phần data mới làm rõ khác biệt giữa raw
correlations tính trực tiếp và artifact cũ flatten overlapping windows.
