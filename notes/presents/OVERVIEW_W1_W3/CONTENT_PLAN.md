# ECDR — Kế hoạch slide overview Week 1–3
**Ngày đối chiếu:** 2026-11-04. **Khán giả:** tutor/reviewer có nền tảng kỹ thuật, nhưng chưa đọc toàn bộ GitHub.
**Đầu ra:** 12 slide HTML trình chiếu độc lập tại `OVERVIEW_W1_W3/index.html`; tài liệu này khóa wording, visual, caption, talking points và evidence.
**Phạm vi:** tổng kết ba tuần, không thay thế bốn báo cáo gốc, không chứng minh lại định lý, không viết lại code, không chỉnh sửa deck/tài liệu cũ. Slide bằng tiếng Việt; tên API/thuật ngữ kỹ thuật giữ tiếng Anh khi cần.

## A. Narrative và nguyên tắc biên tập
**Một câu chuyện:** Week 1 xác định vấn đề và khóa tham chiếu → Week 2 kiểm chứng data/model hiện có → Week 3 kiểm tra toán học và runtime → kết luận trung thực: *có baseline và verifier fail-closed, nhưng chưa có certified reuse hữu ích hay net saving*.

- Mỗi slide chỉ có **một thông điệp có thể nói hết trong một câu**; headline dùng kết luận thay vì tên chương, tối đa 2–3 dòng text hoặc 2–3 KPI.
- **Hình/sơ đồ:** dùng khi cần mô tả thời gian, pipeline, architecture, điều kiện ra quyết định, hoặc so sánh số lượng/latency. Hình phải có scale, nhãn, đơn vị, phạm vi đo và chú thích một câu. Ưu tiên sơ đồ vector vẽ lại từ nguồn hiện tại.
- **Text:** dùng cho objective, constraint, research status, câu hỏi nghiên cứu, limitation và những phân biệt dễ bị hiểu nhầm. Không chụp trang PDF, bảng 12 lớp, code, hash hay theorem proof lên slide chính.
- **Lời nói:** giải thích *vì sao* hình quan trọng, giới hạn phép đo và chuyển ý sang tuần tiếp theo. Ghi chú không phải lời cần đọc trên màn hình; có thể nói linh hoạt.
- **Footer:** ghi ngắn tên nguồn hiện hành; speaker notes và file này giữ đường dẫn đầy đủ. Chi tiết kỹ thuật và reproducibility xem GitHub.
- **Từ ngữ bắt buộc:** “đã kiểm chứng baseline/trace”, “cận dương *diagnostic/conditional*”, “membership”, “fallback”, “chưa chứng minh numerical soundness”, “net saving âm ở cấu hình đo này”.
- **Không dùng:** “đã chứng nhận 99.6%”, “0% lỗi nên 100% đúng”, “đã tối ưu tốc độ”, “G2 PASS”, kết quả K=6/GAP lịch sử như evidence K=12, hoặc kết luận CPU đo được tổng quát hóa cho GPU/production.
- **Định dạng:** 16:9; khuyến nghị 12–15 phút tổng cộng, không áp đặt thời lượng. Slide chính có thể mở bằng trình duyệt, điều hướng phím mũi tên, Space; `N` hiện ghi chú, `F` fullscreen.

## B. Storyboard chính thức (12 slide)
| # | Tuần/chủ đề | Trên slide: headline + text tối thiểu | Hình/biểu đồ cần show | Chú thích ảnh (caption) | Cần nói (không viết đầy lên slide) | Người nghe phải hiểu |
|---|---|---|---|---|---|---|
| 01 | Toàn cảnh | **Ba tuần: từ baseline đến kiểm tra khả năng reuse**. W1 / W2 / W3 | Timeline ba mốc | “Bằng chứng tăng dần: định nghĩa → kiểm chứng → thử tính khả thi.” | ECDR hướng tới giảm phép tính lặp khi suy luận cảm biến liên tiếp, không đổi đầu ra CNN tham chiếu. | Đây là tiến trình nghiên cứu, không phải demo tăng tốc đã thành công. |
| 02 | W1 — Problem | **Chỉ được bỏ qua CNN khi có căn cứ giữ nguyên quyết định.** Một luồng chuẩn và một luồng có gate | Sơ đồ hai nhánh: query → kiểm certificate và membership → reuse / fallback CNN | “Reuse là đường có điều kiện; không đạt điều kiện thì chạy CNN như thường.” | Đối tượng bảo toàn là *quyết định của reference*, khác accuracy so với ground truth. | Vì sao bài toán khó hơn caching dự đoán gần nhau. |
| 03 | W1 — Contract | **Một baseline K=12 cố định là điểm tựa cho mọi phép so sánh.** 6×128 → normalize → CNN → 12 logits; 283.692 params | Pipeline 4 node có shape và “frozen” | “Weights, graph và normalization phải đồng nhất khi đối chiếu.” | HAPT 6 sensor channels ở 50 Hz; 128 sample/window, stride 64; chọn seed 43 epoch 8. | Chỉ có một reference K=12 đang được nghiên cứu. |
| 04 | W1 — Outcome | **Baseline chạy và được kiểm tra; gate chất lượng vẫn HOLD.** 65/65 checks; G2 HOLD; certificate chưa có | Thang trạng thái 3 ô verified / hold / not established | “65 checks xác minh artifact baseline, không xác nhận chất lượng mô hình hay certificate.” | Transition classes thiếu support; label indexing gốc vẫn có caveat. | Không được đồng nhất tái lập baseline với research gate pass. |
| 05 | W2 — Data | **17.453 cửa sổ phục vụ streaming, 12.278 có nhãn supervised hợp lệ.** Test: 5.053 vs 3.572 | Hai window chồng 50% trên timeline + hai mẫu số cạnh nhau | “128 mẫu = 2,56 s; stride 64 = 1,28 s; full stream ≠ supervised subset.” | Raw 61 recordings/30 participants; split 17/4/9 subjects; tránh leakage theo subject và tránh coi cửa sổ overlap là i.i.d. | Mẫu số nào dùng cho prediction-trace và accuracy. |
| 06 | W2 — Model | **CNN thật có ba Conv1D và head Flatten–MLP.** K=12, 283.692 params | Sơ đồ khối Conv/BN/Pool → Conv/BN/Pool → Conv/BN → Flatten/MLP | “Không dùng lý thuyết GAP của kiến trúc lịch sử cho head hiện tại.” | Sơ đồ lớp chỉ đủ để hiểu lý do verifier phải theo actual graph và vì sao identity rất quan trọng. | Model đã khóa có architecture cụ thể, không phải một toy CNN. |
| 07 | W2 — Evidence | **Accuracy 89,05%; macro-F1 0,776 nhưng lớp transition còn yếu.** 3.572 test supervised; G2 HOLD | 2 KPI lớn + thanh range F1 cho basic vs transition | “Basic F1: 0,820–0,989; transition F1: 0,550–0,736; các giá trị là phạm vi theo lớp, không phải mean.” | Reproduce 0/5.053 prediction differences với frozen trace; đây *không* phải certified reuse. | Baseline hoạt động, nhưng macro-F1/transition support là hạn chế cần nêu. |
| 08 | W3 — Theorem | **Reuse cần cận dưới dương cho cả 11 lớp cạnh tranh trên toàn miền.** Formula ngắn: min lower margin > 0 | Gate 3 điều kiện: bound/identity → query membership → numerical soundness → reuse; fail → fallback | “Point margin dương tại một sample không chứng minh an toàn trên cả miền.” | Giải thích anchor, candidate domain và fail-closed chỉ bằng một ví dụ lời nói. Hệ số thực toán học không tự chứng nhận float32 backend. | Quy tắc hợp lệ để reuse là gì và vì sao cần nhiều bằng chứng. |
| 09 | W3 — Coverage | **5.034/5.053 cận diagonal dương, nhưng 0/5.053 next-window thuộc miền.** 0 formal certificate | Funnel: conditional-positive → membership zero → accepted zero; wave 4.211/5.053 chú thích nhỏ | “Cận dương là diagnostic float64 ở ρ=10⁻⁶; KHÔNG phải certified reuse.” | Miền đủ nhỏ để cận dương nhưng không chứa query tiếp theo; numerical bridge chưa có. | Soundness, coverage và reuse là ba điều khác nhau. |
| 10 | W3 — Cost | **Chi phí ECDR hiện cao hơn baseline wrapper khi không có reuse.** 1,589–1,675 vs 3,107–3,198 ms/query | Cặp thanh dải đo trên cùng scale ms/query | “Cùng full-query CPU policy; ECDR ≈0,50–0,53× tốc độ baseline wrapper (thấp hơn, không phải tăng tốc).” | Đừng so với core CNN 0,215–0,224 ms: core khác phạm vi đo, không có cost wrapper. Kết quả workload cụ thể, không kết luận mọi policy. | Không có hiệu quả ròng ở lần thực nghiệm được báo cáo. |
| 11 | Tổng hợp | **Kết quả quan trọng là biết chính xác phần nào đã kiểm chứng, phần nào còn mở.** W1 verified baseline; W2 audited data/model; W3 safe fallback nhưng 0 reuse | Matrix 3 tuần × Deliverable / Limitation | “G2 HOLD; G3 PARTIAL / NOT PASSED; G4–G6 chưa đạt.” | Các next research obligations: float32 enclosure, miền có membership hữu ích, giảm chi phí build/membership, sau đó matched end-to-end benchmark. | Hướng tiếp theo dựa trên blockers, không phải lời hứa thành công. |
| 12 | Tài liệu | **Slide để nắm ý chính; báo cáo và GitHub để kiểm chứng chi tiết.** | 4 thẻ đường dẫn: W1 / W2 data+model / W3 math / evidence | “Mỗi claim chính có report và receipt kiểm tra tương ứng.” | Chỉ dẫn đến nguồn chính; không đẩy người nghe đọc bảng chứng minh lúc thuyết trình. | Biết đọc sâu ở đâu và hiểu phạm vi evidence. |

## C. Thứ tự thực hiện và quyết định thiết kế
1. Khóa claim/đơn vị theo 4 report hiện hành và 3 receipts: Week 1 `00_OVERVIEW/week1/SOURCE_OF_TRUTH.md`; Week 2 `01_DATA/README.md`, `02_MODEL/README.md`; Week 3 `output/week3_certification_20261103/README.md`.
2. Xây slide *từ claim* rồi vẽ diagram/chart diễn giải claim. Không chọn hình có sẵn trước rồi ép wording theo hình.
3. Hình cũ trong `notes/presents/assets/` và `WEEK2/assets/`, `WEEK3/assets/` là **reference**, không tự động được coi là chính xác. Một số có chữ chồng/nhãn không cùng measurement scope (đặc biệt figure latency W3). Bản overview sẽ dùng vector HTML tự dựng, đọc được ở fullscreen và tương thích trình duyệt.
4. Caption ngay bên dưới hình: “Đây là gì”, “phạm vi/đơn vị”, và một cảnh báo ngắn nếu dễ hiểu sai. Phần *tại sao* nằm trong presenter notes.
5. Giữ slide theo thứ tự W1 (3) → W2 (3) → W3 (3), kèm cover, synthesis và sources; **không đưa 21 slide W3 chi tiết vào overview**.
6. Kiểm tra: đúng 12 slide; lời nói/nhãn có nguồn; điều hướng/notes; không lỗi thiếu hình; không overclaim certificate/accuracy/speedup; không đụng source hay evidence khác.

## D. Mapping chi tiết từ claim sang source
- **W1 reference và seed:** `00_OVERVIEW/week1/SOURCE_OF_TRUTH.md`, `output/week1_closeout_20261102/README.md`.
- **W1 quality gate:** `GATES.md`, `00_OVERVIEW/week1/REQUIREMENTS.md`; 65/65 là baseline artifact verification.
- **W2 data counts/split/windowing:** `output/week2_foundations_20261103/data_findings.md`, `01_DATA/03_Week_2_Data_Foundations.tex`.
- **W2 architecture/evaluation:** `output/week2_foundations_20261103/model_findings.md`, `02_MODEL/01_ECDR_CNN_Model_Analysis_EN.tex`.
- **W3 theorem/condition:** `03_MATH/03_Week_3_Mathematical_Foundations.tex`, `03_MATH/04_ECDR_Mathematics_Explained.tex`, `03_MATH/README.md`.
- **W3 reproduced metrics/cost:** `output/week3_certification_20261103/README.md`, `experiment_results.json`, `domain_results.json`, `runtime_results.json`; four runtime traces are repeats of the same 5.053-window stream, not 20.212 independent subjects or samples.
- **Status:** `GATES.md`, top block labeled 2026-11-03 supersedes historical K=6 status below.

## E. Verification ledger (status của artifact slide, không phải nghiên cứu)
- CREATED: `index.html`, file plan này.
- VERIFIED: chỉ ghi PASS khi đã kiểm thực tế bằng trình duyệt/DOM và kiểm claim mapping. Khi chưa chạy browser/visual QA, đánh dấu chưa kiểm.
- ACCEPTED: reviewer/user quyết định; không tự nhận.
