# ECDR — bộ báo cáo lý thuyết Week 1–3

> **Current Week 1 authority:** [canonical English source](../00_OVERVIEW/02_Week_1_Project_Brief.tex) and [Week 1 registers](../00_OVERVIEW/week1/). The `me/` PDFs are generated derivatives; dated Vietnamese material is historical.

> 📌 **Điều hướng**: 
> - Bản đồ lộ trình đọc tuyến tính cho toàn bộ dự án: xem [`../README.md`](../README.md).
> - Bảng tra cứu ký hiệu tiền quyết ($\mathcal{P}$ cố định, $X$, $Z$, $f_\theta$): xem [Instruction 0 trong README.md](../README.md#️-instruction-0-bảng-ký-hiệu-tiền-quyết--bản-đồ-tiền-giả-định).
> - Bảng quản lý tiến độ thực chứng (TrustMeBro): xem [`../TMB.md`](../TMB.md).

Ngày biên soạn: **28/10/2026**. Đích theo yêu cầu: `/home/pro/Downloads/research/me`.

Nội dung đi theo thứ tự **Project brief → Data foundations → Mathematical
foundations** trong bản chuẩn hóa mới do người nghiên cứu cung cấp. Đây là báo
cáo lý thuyết; implementation, P0 và gate cũ không định nghĩa phương pháp mới.

## Các bản đọc và chỉnh sửa

| Báo cáo | LaTeX | PDF |
|---|---|---|
| Tổng hợp đủ ba tuần | [Nguồn](ECDR_Weeks_1_2_3_Theory.tex) | [Bản đọc](ECDR_Weeks_1_2_3_Theory.pdf) |
| Week 1: Project Brief | [Nguồn](Week_1_Project_Brief.tex) | [Bản đọc](Week_1_Project_Brief.pdf) |
| Week 2: Data Foundations | [Nguồn](Week_2_Data_Foundations.tex) | [Bản đọc](Week_2_Data_Foundations.pdf) |
| Week 3: Mathematical Foundations | [Nguồn](Week_3_Mathematical_Foundations.tex) | [Bản đọc](Week_3_Mathematical_Foundations.pdf) |
| Giải thích toàn bộ mathematics — 29 trang | [Nguồn](ECDR_Mathematics_Explained.tex) | [Bản đọc](ECDR_Mathematics_Explained.pdf) |
| Phân tích dữ liệu HAPT — bản Việt cũ, 23 trang | [Nguồn VI](ECDR_Data_Analysis_VI.tex) | [Bản đọc VI](ECDR_Data_Analysis_VI.pdf) |
| HAPT Data Analysis & Methodology — English, clarified definitions, 28 pages | [Nguồn](ECDR_Data_Analysis.tex) | [Bản đọc](ECDR_Data_Analysis.pdf) |
| Bóc tách mô hình CNN — bản Việt cũ, 24 trang | [Nguồn VI](ECDR_CNN_Model_Analysis_VI.tex) | [Bản đọc VI](ECDR_CNN_Model_Analysis_VI.pdf) |
| Mathematics concept atlas — English, 104 concepts | [English source](ECDR_Mathematics_Concepts_EN.tex) | Source/editor edition; no new PDF export in this revision |
| CNN from first principles — English, 64 concepts | [English source in the existing editor path](ECDR_CNN_Model_Analysis.tex) | Source/editor edition; no new PDF export in this revision |

Nguồn chuẩn để sửa là `ECDR_Weeks_1_2_3_Theory.tex`. Ba bản tuần là bản standalone
được trích từ các marker `BEGIN WEEK1/2/3` và có bibliography nhúng, không cần
BibTeX hay các file hình ngoài. Chạy lại `build_sources.py` sau khi sửa bản tổng
hợp sẽ cập nhật ba bản tuần. Không sửa bản tuần rồi chạy lại script nếu chưa
chuyển sửa đổi về bản tổng hợp.

`ECDR_Mathematics_Explained.tex` là tài liệu đồng hành viết độc lập: 24 mục
chính giải thích từ ký hiệu đến từng bước suy ra, 29 nhóm đối chiếu toán của
báo cáo chính và phần giải nghĩa H01–H24. Có ví dụ CNN nhỏ xuyên suốt, tính
tay features/GAP/logits/margins, ba mức cận, và phản ví dụ stale certificate.
Chỉnh trực tiếp file này; `build_sources.py` không tạo lại hoặc ghi đè nó.

Hai chuyên đề mới cũng được viết độc lập. `ECDR_Data_Analysis.tex` đi từ
raw/annotation/window/split tới thống kê, temporal dependence, normalization
và các hệ quả đối với ECDR. Có bốn biểu đồ, bảng đủ 12 lớp, inventory đủ 61
recordings và phân tích vì sao pure-window policy làm giảm transition support.
`ECDR_CNN_Model_Analysis.tex` bóc tách Conv1D/ReLU/GAP/head, shape, receptive
field, parameters, MAC/FLOP, memory, training/loss, shift identities, margin
bounds và lifecycle. Mô hình valid/ReLU/GAP K12 là ví dụ lý thuyết; model K6
cũ chỉ được đối chiếu trong phụ lục.

File data cần giữ cùng các assets `.tex`/`.pdf` dưới `qa/data_cnn/` khi chuyển
sang máy khác. File CNN có bibliography nhúng và không cần hình ngoài. Không
chạy `build_sources.py` để tạo lại các tài liệu đồng hành: sửa trực tiếp nguồn
của từng tài liệu.

```bash
cd /home/pro/Downloads/research
bash me/build.sh
.venv/bin/python me/recompute.py
.venv/bin/python me/qa/verify_math_explained.py
.venv/bin/python me/qa/data_cnn/prepare_data_analysis.py
.venv/bin/python me/qa/data_cnn/verify_documents.py
```

`build.sh` dùng Tectonic cục bộ. Lần đầu có thể cần mạng để tải package TeX.
Ở đợt xuất các bản Việt cũ, compiler trong app lỗi khi tải bundle v33; lỗi này
đã được sửa trong đợt kiểm hai nguồn English ở cuối trang. Cả
bốn bản đã biên dịch thành công bằng Tectonic cục bộ; bản tổng hợp 31 trang,
Week 1/2/3 lần lượt 6/10/17 trang. Checks và giới hạn được ghi trong
[VERIFICATION.md](VERIFICATION.md); metadata/hashes trong
[qa/artifact_checks.json](qa/artifact_checks.json).
Tài liệu giải thích mới đã biên dịch và kiểm riêng; chi tiết tại
[qa/math_explained/VERIFICATION.md](qa/math_explained/VERIFICATION.md) và
[qa/math_explained/checks.json](qa/math_explained/checks.json).
Hai chuyên đề data/CNN được kiểm riêng tại
[qa/data_cnn/VERIFICATION.md](qa/data_cnn/VERIFICATION.md) và
[qa/data_cnn/checks.json](qa/data_cnn/checks.json). Các hash của năm tài liệu
đã có được đối chiếu để xác nhận nguồn/PDF trước đó không bị thay đổi.

Audit mới đọc raw rows thật để tính lag1/lag64. Các số này khác một số
correlations trong `data/corr_semantics.json` cũ, vì script cũ flatten các
windows có overlap rồi gọi đó là raw series. Tài liệu data giải thích khác
biệt định nghĩa và dùng số tính mới; source/artifact cũ được giữ nguyên.

## Nội dung được chuẩn hóa

- Định nghĩa dữ liệu vật lý, units, raw data/window/vector/region; không gộp raw
  sensor signals với nhánh 561 features của HAPT.
- Schema HAPT 12 nhãn, tham số toán học `K` tổng quát; không chuyển kết quả mô
  hình 6 lớp cũ thành kết quả của CNN 12 lớp.
- Chứng minh đếm window, overlap sau căn chỉnh, normalization transform, và
  counterexample cho suy luận overlap ⇒ ordinary similarity ⇒ same decision.
- Margin, local invariance, Lipschitz, IBP, CROWN/LiRPA, support functions,
  PCA/residual, wave-domain, valid-CNN/GAP identities, membership và lifecycle.
- Fixed-anchor và finite-horizon soundness có điều kiện; counterexample cho
  việc tự recenter sau một hit; numerical obligations.
- Break-even và sensitivity được tính có đơn vị; hypothesis ledger phân biệt
  proved/conditional/false/open. Không có claim speedup đã đo.

## Nguồn và giới hạn

Brief là hai attachment của người nghiên cứu, đặc biệt attachment mới
`2833ac03-2753-4230-b460-d427cd76826c/Pasted text.txt`. Định nghĩa/metadata data
được đối chiếu UCI HAPT và README local. Các count local lấy từ JSON audit, có
caveat convention annotation. Nguồn toán học/prior work được dẫn trong LaTeX.

`recompute.py` và `calculations.json` lưu phép kiểm số; các checks dữ liệu là
read-only. Không chạy lại training, không benchmark ECDR và không thay đổi
dataset/model/source cũ. Proof trong báo cáo dùng số thực trừ phần numerical
soundness đã phát biểu thêm giả thiết. Useful coverage, runtime benefit và
superiority so với LipCache vẫn là giả thuyết nghiên cứu.

## English concept revision — 28 October 2026

The latest instruction requests English throughout the new/revised documents.
The mathematics atlas contains **104 concepts**: 46 foundational concepts,
30 data/model bridges and 28 certification/systems concepts. The existing
CNN editor source now contains **64 concepts**: eight foundational pages and
56 CNN/ECDR pages. Each concept occupies exactly one typeset page, verified
from TeX page-start/page-end markers. Front matter and references are separate.

- Edit `ECDR_Mathematics_Concepts_EN.tex` directly for the new mathematics atlas.
- Edit `ECDR_CNN_Model_Analysis.tex` directly for the English CNN edition; this
  is the same source path that was already open in the editor.
- `ECDR_Mathematics_Explained.tex/.pdf` remains the previous Vietnamese
  mathematics explanation. `ECDR_CNN_Model_Analysis_VI.tex/.pdf` preserves the
  previous Vietnamese CNN edition byte for byte.
- The data report and original Week1–3 documents are unchanged.

No new PDF was exported during this editor revision. The old on-disk
`ECDR_CNN_Model_Analysis.pdf` is still the preceding Vietnamese export;
it must not be described as the revised English document. The `_VI.pdf`
link above identifies that historical edition explicitly.

The built-in compiler now returns **success for both saved English sources**.
Its private Tectonic cache was populated with the missing v33 resources and
13 PDF math fonts, resolving the earlier download/DNS failures. The application
binary and sandbox configuration were not changed.

Local **XDV-only** passes with the application's Tectonic 0.17.0 also verify
source syntax, references and pagination. The source/log/XDV hashes are bound
by a compilation receipt, preventing an old log from validating a new source.
Successful desktop compilation is verified; the live preview was not separately
inspected visually. No separate English PDF was exported or another tab opened.

Current revision QA: [English source verification](qa/english_concepts/VERIFICATION.md)
and [machine-readable results](qa/english_concepts/checks.json).

```bash
cd /home/pro/Downloads/research
.venv/bin/python me/qa/english_concepts/verify_sources.py --compile
```

Use `--compile` after any source change; a later run without it checks the
saved source/log/XDV receipt. The `--tectonic` option selects another installed
Tectonic executable. Exact verified commands and the repaired app-cache path
are recorded in the current verification document. Authoring fragments
and the mathematics assembler are under `authoring_en/`; they are not runtime
dependencies of either standalone source. Do not regenerate the atlas over
subsequent direct edits without reconciling those edits first. Older QA files
record the earlier exported editions and should not be interpreted as rendered
PDF verification for this English revision.


## Supplementary Explanation Documents — October 2026

The following **seven standalone LaTeX/PDF documents** were added as supplementary
reading companions to `ECDR_Data_Analysis.tex`. They add deeper conceptual
explanation and worked examples for concepts that appear in the main data analysis
but are not fully developed there.

> **Rule**: These supplements are **additive only**. The main document
> `ECDR_Data_Analysis.tex` (SHA-256: `98689dc6...`) is **not modified**.

| Supplement | Topic | Source | PDF |
|---|---|---|---|
| SUPP-01 | **ECDR Primer** — what ECDR is, how it differs from a cache, anchor lifecycle (FRESH→ACTIVE→STALE→REFRESH), the full safety argument, 561-feature vs. raw-6ch branch distinction | [Tex](ECDR_Data_Analysis_Supplement_ECDR_Primer.tex) | [PDF](ECDR_Data_Analysis_Supplement_ECDR_Primer.pdf) |
| SUPP-02 | **Norm Geometry** — $L_2$ vs $L_\infty$ in $\mathbb{R}^{768}$, the $\sqrt{768} \approx 27.7$ relationship, ellipsoid preimage under normalization, high-dimensional volume shell, why percentile translation across norms is invalid | [Tex](ECDR_Data_Analysis_Supplement_Norm_Geometry.tex) | [PDF](ECDR_Data_Analysis_Supplement_Norm_Geometry.pdf) |
| SUPP-03 | **Certification Methods** — IBP interval propagation (affine + ReLU rules), CROWN linear relaxation, LiRPA framework, soundness vs. completeness, domain size vs. certifiability tradeoff | [Tex](ECDR_Data_Analysis_Supplement_Certification_Methods.tex) | [PDF](ECDR_Data_Analysis_Supplement_Certification_Methods.pdf) |
| SUPP-04 | **Autocorrelation** — Pearson formula, ACF definition, stationarity, physical interpretation of the four metrics (lag-1, lag-64, window-positional, channel-mean), full linear-ramp counterexample derivation | [Tex](ECDR_Data_Analysis_Supplement_Autocorrelation.tex) | [PDF](ECDR_Data_Analysis_Supplement_Autocorrelation.pdf) |
| SUPP-05 | **Wave-Domain Parameterization** — shift operator $T_S$ derivation, PCA low-rank component $Vu$, why explained variance ≠ residual bound, full sinusoidal worked example with phase sensitivity formula, multi-step drift bound | [Tex](ECDR_Data_Analysis_Supplement_Wave_Domain.tex) | [PDF](ECDR_Data_Analysis_Supplement_Wave_Domain.pdf) |
| SUPP-06 | **Float Precision** — IEEE 754 float32/float64 formats, machine epsilon, cast semantics, which pipeline stages require float64 vs. float32, quantization (int8) certification obligations | [Tex](ECDR_Data_Analysis_Supplement_Float_Precision.tex) | [PDF](ECDR_Data_Analysis_Supplement_Float_Precision.pdf) |
| SUPP-07 | **Statistical Dependence** — why i.i.d. fails (overlap + clustering), true variance formula with off-diagonal covariances, AR(1) inflation factor worked example, block bootstrap procedure, why effective-N cannot be computed from geometric overlap, clustered standard errors, normalization population bias | [Tex](ECDR_Data_Analysis_Supplement_Statistical_Dependence.tex) | [PDF](ECDR_Data_Analysis_Supplement_Statistical_Dependence.pdf) |

All seven supplements compile cleanly with Tectonic 0.15.0 and use Noto fonts
(matching the main document's visual style). Each is self-contained with its own
bibliography.

To recompile any supplement:
```bash
tectonic --keep-logs --outdir me/ me/ECDR_Data_Analysis_Supplement_<NAME>.tex
```
