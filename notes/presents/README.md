# notes/ — reading notes và provenance nghiên cứu

**Đồng bộ hướng dẫn:** 22/10/2026. Literature workflow hiện ở pilot; chưa có
screening/full-text synthesis cho toàn inventory P01–P23.

| File | Vai trò và giới hạn |
|---|---|
| [P04_NoScope.md](P04_NoScope.md) | SOURCE_CHECKED_SCOPED ngày 21/10; claim hẹp có locator, chưa đọc sâu/reproduce toàn bài |
| [2026-10-21_REVIEW_KICKOFF.md](2026-10-21_REVIEW_KICKOFF.md) | Snapshot lịch sử của kickoff, tool/skill audit và môi trường ngày 21/10; không phải live install state ngày 22/10 |
| [2026-10-21_SEARCH_LOG.json](2026-10-21_SEARCH_LOG.json) | Query/provenance/giới hạn của search pilot; không ghi đè để giả thành lượt search mới |

Đặt tên note `P<ID>_<ShortName>.md` theo
[05_READING_PLAN.md](../05_READING_PLAN.md), dùng đúng 10 phần template §7.
Không biến abstract thành reading note đầy đủ. Mỗi note ghi trạng thái đọc,
ngày kiểm, source URL/version và printed page/section; phân biệt trang in với
PDF page. Mục 6 “Verified Claims” giữ các claim hẹp có evidence/locator;
diễn giải, inference và reproduction status ghi riêng. Có locator không đồng
nghĩa đã kiểm proof, code hay toàn bộ benchmark.

Khi note thay đổi novelty/gap, cập nhật [claim ledger](../02_LITERATURE.md) và
[refs.bib](../refs.bib) khi cần; mọi claim SUPPORTED phải truy lại được nguồn.
Bibliography entry chỉ ghi metadata không tự đóng reading item. Dùng log mới
cho một lượt search mới; giữ log cũ đúng ngày và provenance.

CREATED là đã tạo artifact; VERIFIED chỉ áp dụng với check và phạm vi được nêu;
ACCEPTED cần người có quyền chấp nhận. Không coi AI/source inspection là human
acceptance. Note kiểm nguồn 21/10 giữ ngày kiểm đó nếu 22/10 chỉ đồng bộ tài liệu.

Điều hướng hiện tại: [review protocol](../06_REVIEW_PROTOCOL.md),
[project brief](../PROJECT_BRIEF.md), [report](../report.md),
[G3 domain contract](../07_G3_DOMAIN_CONTRACT.md),
[sync record 22/10](../DOCS_SYNC_2026-10-22.md).
Reading notes được quản lý cùng source docs; `papers/` PDFs đang được gitignore.
