# QA Report — ECDR W1–W3 Overview
**Thời điểm:** 2026-11-04 (Asia/Ho_Chi_Minh)

## Kết quả đã kiểm chứng
| Hạng mục | Kết quả | Bằng chứng/phạm vi |
|---|---|---|
| 12 slide hiện hữu | PASS | `index.html` chứa đúng 12 `section.slide`. |
| 12 presenter notes | PASS | Có đúng 12 `presenter-note`; nội dung nối theo slide. |
| Truy cập offline | PASS cho nội dung trình chiếu | Render `file://.../index.html#slide-N` bằng Chrome headless, không CDN; việc host web không bắt buộc. |
| Rendering 16:9 | PASS | Headless Chrome 1440×900, tất cả slide 01–12 được chụp vào `/tmp/ecdr_overview_qa/slide_01.png` … `slide_12.png`. |
| Visual QA | PASS tại viewport trên | Đã xem đủ các trang render, điều chỉnh lại slide 10 để caption không sát/đè footer. Re-render slide 10 lưu `/tmp/ecdr_overview_qa/slide_10_final.png`. |
| Source/evidence mapping | PASS cho nội dung slide chính | Từng slide có path về nguồn hiện tại; `CONTENT_PLAN.md` có source ledger. |
| Claim separation | PASS review | Không gộp accuracy với reference agreement; không gọi conditional float64 bounds là certificates; so latency theo cùng wrapper scope; G2 HOLD/G3 PARTIAL. |
| Previous reports and decks | UNCHANGED | Chỉ tạo `OVERVIEW_W1_W3/` mới; không chỉnh 4 bản index.html cũ. |

## Giới hạn kiểm chứng
- **Chưa kiểm manual GUI:** thao tác click/keyboard fullscreen trên phiên Chrome người dùng thật, mobile device, projector.
- **Chưa kiểm nghiệm thu của người nghe:** chưa có acceptance từ tutor/reviewer.
- **Không chạy lại training, verifier experiments hay benchmark:** chỉ dùng execution receipts hiện hành, không tạo thêm kết quả nghiên cứu.
- Bản PowerPoint chỉnh sửa được tạo riêng trong môi trường xuất artifact; `HTML index.html` trên máy `tducn` mới là bản lưu tại workspace, không tự động đồng bộ PPTX vào máy.
- HTML render dùng text/figure vector, không dùng các PNG cũ có chữ chồng và nhãn latency dễ gây hiểu nhầm.

## Checklist kỹ thuật để tái kiểm
1. Mở `index.html#slide-1` trong Chrome trên máy và xem toàn bộ 12 slide.
2. Thử `← → Space N F` trước buổi thuyết trình.
3. Đối chiếu claims Week 1/2/3 với `CONTENT_PLAN.md` và 3 execution receipts.
4. Chỉ gắn `ACCEPTED` khi reviewer/tutor xác nhận.

**Status:** CREATED / VERIFIED (đúng các phạm vi trên) / NOT ACCEPTED.
