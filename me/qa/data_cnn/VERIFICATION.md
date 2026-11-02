# Kiểm chứng hai chuyên đề data và CNN

Ngày: **28/10/2026**. Trạng thái: **CREATED và VERIFIED** trong phạm vi
biên soạn, đọc nguồn, phép tính, review toán và kiểm artifact dưới đây.

| Tài liệu | PDF | Trọng tâm |
|---|---:|---|
| `ECDR_Data_Analysis` | 23 trang | Raw/annotation/window/split, thống kê, temporal dependence, hệ quả nghiên cứu |
| `ECDR_CNN_Model_Analysis` | 24 trang | Conv/ReLU/GAP/head, counts, proof conditions, toy certificate và lifecycle |

Nguồn/PDF của năm báo cáo trước được so với hashes đã lưu trong
`qa/artifact_checks.json` và `qa/math_explained/checks.json`: **không thay đổi**.
Chỉ thêm tài liệu và assets dưới `me/`, cập nhật README và danh sách build.

## Kiểm dữ liệu thực chạy

```bash
.venv/bin/python me/qa/data_cnn/prepare_data_analysis.py
.venv/bin/python me/qa/data_cnn/verify_documents.py
```

- Đọc toàn bộ 61 cặp raw acc/gyro và 1.214 annotation spans của 30 người:
  1.122.772 raw time samples, 6.736.632 sensor coordinates; finite và shape
  checks không phát hiện sai khác đã được định nghĩa.
- Đối chiếu **tất cả 13.403.904 tọa độ window** với raw sau cùng phép ép
  float32 của builder: **0 mismatch**. Policy/majority được tái tạo cho đủ
  17.453 windows: **0 mismatch**.
- Đối chiếu **6.678.528 tọa độ overlap** trên 17.392 pairs: **0 mismatch**.
  Không nối pair qua recording boundary; không dùng `win_idx` riêng làm
  tensor row index.
- Kiểm split ownership theo người và recording; các subject sets rời nhau.
  Kiểm lại pure K12/K6 support và denominator của từng split.
- Đếm multiplicity: 1.876 mẫu không được grid bao phủ, 7.808 mẫu xuất hiện
  một lần, 1.113.088 mẫu xuất hiện hai lần. Tổng occurrences bằng
  17.453 × 128 = 2.233.984; không coi overlap là dữ liệu thu độc lập mới.
- Kiểm độc lập công thức span-grid bằng enumerate starts trên đủ 1.214
  spans: **40 short + 91 đủ dài nhưng lệch grid + 1.083 có aligned window**;
  tổng pure-grid windows là **10.361**.
- Tính channel statistics trên raw samples duy nhất; normalized distances
  dùng đúng train-pure-K6 float32 mean/std convention được khai báo. Pooled
  và per-class/stratum results là exploratory descriptive audit.
- Tính mới lag1/lag64 trực tiếp trên raw của từng recording; reviewer cũng
  đọc lại raw độc lập, mean/std/correlations khớp sai số dưới 1e-12.
  `scripts/06b_corr_semantics.py` cũ flatten overlapping windows rồi gọi là
  raw series, nên các số cũ không cùng định nghĩa. Giữ nguyên source cũ và
  giải thích khác biệt trong chuyên đề mới.
- JSON lưu SHA256 của nguồn chính và từng cặp raw files; verification script
  đối chiếu hash nguồn chính để phát hiện drift.

## Review CNN và chứng minh

- Review độc lập toàn bộ hai nguồn. Đã sửa floor delimiter, vector-column
  convention của head, temperature τ=1 trong loss, ba tỷ lệ làm tròn và
  chuẩn hóa mapping `F = f_theta ∘ P` giữa raw/normalized coordinates.
- Tính độc lập lengths **124/122/120**, RF **5/7/9**, parameters **20.332**,
  MAC **2.343.936**, quy ước FLOP **4.687.872**, ReLU **19.456 scalars**,
  GAP **7.616 additions + 64 scales**.
- Weights float32 payload **81.328 bytes**, tổng activation tensors được
  liệt kê **81.200 bytes**, cặp buffers lớn nhất **61.952 bytes**. Đây là
  accounting có giả thiết, không phải peak RAM đo bằng framework.
- Kiểm toy CNN bằng `Fraction`: old/new logits, 4 corner values của affine
  toy objective, minima **35/4 và 11/8**, stale logits **17/2, -4, 9**.
  Universal continuous-box proof dựa trên stable ReLU branches và affine
  expressions đã chứng minh trong tài liệu; sampling corners chỉ kiểm số học.
- Reviewer không phát hiện sai logic còn lại trong aligned shift, GAP/margin
  delta, sign-aware tail bound, whole-domain positivity, finite horizon và
  numerical gate. Không dùng theorem prover hoặc neural verifier executable.
- Đối chiếu operator contracts với official PyTorch 2.14 docs; citations nằm
  trong LaTeX. Model SmallCNN1D K6 cũ chỉ đối chiếu ở phụ lục.

## Compile, readback và layout

Compiler trong app được thử cho cả hai nguồn nhưng lỗi môi trường khi tải
TeX bundle v33, trước khi đọc lỗi source. Dùng **Tectonic 0.15.0 cục bộ**,
hai compile final thành công, exit code 0.

- Final logs không có overfull box, missing character, undefined reference
  hoặc undefined citation. Một số underfull warnings ở bảng là layout warnings.
- `pdfinfo`, `pdftotext`, `pdffonts` đọc được cả hai PDF: text Việt, TOC,
  bibliography và số trang; mọi font nhúng, không trang trắng hoặc `??`.
- Kiểm vị trí word boxes: không có từ tràn sát ra ngoài trang theo ngưỡng
  kiểm trong script. Check này không thay visual review.
- Render đủ **47 trang** bằng Poppler, xem toàn bộ sáu contact sheets; kiểm
  phóng lớn convolution formula, continuous-box proof, span-grid proof và
  correlation table. Bảng/hình/công thức không bị cắt hoặc đè lên nhau.
- Bbox/text extraction của CNN có ba control-code mappings từ glyph toán
  TeX; script giữ tọa độ word và bỏ control codes chỉ khi parse XML. Các ký
  hiệu hiển thị được kiểm bằng render. Không claim plain-text extraction
  bảo toàn tuyệt đối mọi ký hiệu toán.

`checks.json` ghi hashes final, page counts, citations/references, số học và
trạng thái kiểm tự động. PNG/contact sheets là evidence layout; visual review
được ghi ở đây riêng, không suy ra chỉ từ script trả exit code 0.

## Giới hạn

Không chạy training, HAPT checkpoint inference, verifier mới hoặc runtime
benchmark. Không claim accuracy K12, certified HAPT coverage hay speedup.
Annotation index convention gốc vẫn cần xác minh; metadata frequency không
được kiểm bằng timestamps; license README/UCI có xung đột provenance được
ghi nhận. Những vấn đề này không được dùng làm lý do bỏ phần tài liệu đã
được yêu cầu; chúng được phát biểu đúng ranh giới trong kết quả.

PDF data đọc độc lập; source LaTeX cần giữ các bảng/hình dưới `qa/data_cnn`
khi rebuild. PDF CNN có nguồn standalone và bibliography nhúng.
