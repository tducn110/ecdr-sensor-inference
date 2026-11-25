# Kiểm tra tài liệu giải thích toàn bộ mathematics ECDR

Ngày: **28/10/2026**. Đầu ra: `me/ECDR_Mathematics_Explained.tex` và bản PDF
**29 trang**, standalone, có bibliography nhúng và hình TikZ.

Trạng thái: **CREATED và VERIFIED** trong phạm vi nội dung/định dạng được
kiểm dưới đây; chưa có trạng thái ACCEPTED từ người nghiên cứu.

## Scope và coverage

- Đã đối chiếu toàn bộ phần toán của báo cáo Week 1–3, gồm data/window/labels,
  normalization, probability, CNN và metrics, ngoài phần chứng nhận Week 3.
- Tài liệu có 24 mục chính, bảng map 29 nhóm M01–M29 và giải nghĩa đủ H01–H24.
- Có review độc lập toàn nội dung về coverage và mathematics. Đã sửa các điểm
  phát hiện: loại target khỏi competitor min/max, row/column shape của margin
  coefficient, assumptions của exact membership lower bound, cả hai biên của
  numerical membership domain, nghĩa H10/H04 và khai báo lại C1–C10.
- Mỗi phần có nghĩa ký hiệu, cơ chế suy công thức, ví dụ và điều kiện/giới hạn.
  Không suy từ soundness thành usefulness hoặc từ hit rate thành speedup.

## Các ví dụ số đã kiểm độc lập

- Window count T1000/W128/s64: 14; span [100,400): starts128/192/256.
- Normalization/raw-affine equivalence; confusion matrix accuracy17/20,
  Macro-F1 113/133; sample/population SD của (2,4,6).
- IBP row có dấu âm: interval [-6,8]; CROWN toy có exact minimum1/2 và
  independent-IBP lower -1/2; support norm/PCA/coordinate-residual examples.
- CNN nhỏ W6/C1/s2: old features ((0,1),(1,1),(2,1),(3,1)); new features
  ((2,1),(3,1),(4,1),(5,1)); GAP delta(2,0).
- Exact wave-box lower margins(35/4,11/8); conservative lower margins ở hai
  mức interval(8,7/8) và (17/2,9/8). Mọi cận đều dương và đúng chiều.
- Stale one-step counterexample: hai waves(7,8) đều thuộc original box nhưng
  bước hai logits(17/2,-4,9), winner đổi sang lớp3 vì overlap identity đổi.
- CNN illustrative complexity20.332 parameters,2.343.936 MAC; cost N100/H80/
  B20: baseline400ms, ECDR118ms; variable-cost counterexample20ms→22ms.

Lệnh đã chạy thành công:

```bash
.venv/bin/python me/qa/verify_math_explained.py
```

Script dùng Fraction cho ví dụ hữu tỉ, tính trực tiếp CNN và enumerate corners
của miền affine sau khi xác lập ReLU ổn định. Finite checks chỉ kiểm arithmetic;
universal claims vẫn dựa trên proof/assumptions trong tài liệu, không phải
sampled points hay corner enumeration cho arbitrary neural network.

## LaTeX và PDF

- Compiler trong app lỗi môi trường tải bundle v33 trước khi xử lý source.
  Tectonic0.15.0 cục bộ biên dịch thành công, exit0.
- Không có missing glyph, overfull box, undefined citation/reference hoặc
  duplicate page-anchor trong final diagnostics. Underfull warnings ở cột bảng
  còn tồn tại; đã kiểm render thấy không đè chữ hoặc tràn nội dung.
- `pdfinfo`, `pdftotext`, `pdffonts` xác nhận PDF29trang, trích được tiếng Việt,
  mục lục/công thức và font nhúng. Các citation và cross-reference đều có đích.
- Render toàn bộ29trang bằng Poppler; đã xem cả bốn contact sheets của bản cuối,
  gồm cover, mục lục, hình norms, các công thức, ví dụ CNN và hai bảng phụ lục.
- Source/PDF hashes cùng kết quả kiểm được lưu trong `checks.json`.

## Ranh giới

Hash của bốn nguồn LaTeX và bốn PDF trước đó vẫn khớp bản đã giao; không sửa
nội dung báo cáo chính. README/build helper được cập nhật để tìm và biên dịch
thêm tài liệu đồng hành. Không chạy lại training, raw-data audit, verifier hoặc
runtime benchmark. Số minh họa không là kết quả thực nghiệm ECDR và các budget
numerical error trong ví dụ là giả thiết, không phải error bounds đã đo.
