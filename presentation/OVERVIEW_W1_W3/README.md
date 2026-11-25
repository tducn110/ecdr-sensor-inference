# ECDR Week 1–3 Overview Slide Deck

**Mục tiêu:** Slide thuyết trình overview cho tutor/reviewer; mỗi tuần trả lời *What → How → Outcome*. Các báo cáo LaTeX là nguồn chuẩn cho nội dung khoa học, không thay thế bằng slide.

## Mở trên máy `tducn`
Mở file `index.html` trực tiếp trong Chrome/Firefox (hoạt động offline, không CDN/server).
- `←`, `→`, `Space`, `PageUp`, `PageDown`: di chuyển giữa slide.
- `N`: hiện/ẩn lời nói dành cho người thuyết trình.
- `F`: bật/tắt fullscreen; có các nút tương đương dưới màn hình.
- Link đến slide cụ thể: `index.html#slide-9`. In bằng Print của trình duyệt nếu cần.

## Các file
- [index.html](index.html): **bộ 12 slide trình chiếu**, sơ đồ và biểu đồ được dựng dưới dạng HTML/CSS/SVG-like native elements (không tải hình ngoài).
- [CONTENT_PLAN.md](CONTENT_PLAN.md): nội dung chi tiết của cả 12 slide: visible text, visual, caption, speaker notes, audience takeaway, bằng chứng và những claim không được dùng.
- [QA_REPORT.md](QA_REPORT.md): kiểm chứng đã thực hiện và giới hạn.

## Phân bổ
- Slide 01: câu chuyện Week 1–3.
- Slide 02–04: Week 1 (problem, reference, result).
- Slide 05–07: Week 2 (data, CNN, measured baseline).
- Slide 08–10: Week 3 (certification condition, coverage, cost).
- Slide 11: kết luận và blockers tiếp theo.
- Slide 12: nguồn để đọc sâu trên GitHub.

## Nguyên tắc an toàn scientific claims
- G2 hiện tại: **HOLD** (không dùng K=6 lịch sử để ghi PASS).
- G3: **PARTIAL / NOT PASSED**. `5.034/5.053` diagnostic-positive không được gọi là formal certificate.
- Zero accepted reuse ≠ 100% reused-query agreement (N/A).
- Latency baseline **wrapper** và ECDR complete path là cùng phạm vi đo; `CNN core-only` là phạm vi khác.
- Worktree cũ và 4 index.html trong các thư mục PRESENTS/WEEK1/WEEK2/WEEK3 không bị ghi đè.

**Trạng thái:** CREATED; kiểm tra static và render được ghi trong QA_REPORT; không tự nhận ACCEPTED.
