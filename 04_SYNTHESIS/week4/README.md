# Week 4 — bộ goal, implementation tasks và orchestration cho agy

Trạng thái lúc khởi tạo ngày 2026-11-03: **SPECIFIED; 40/40 tasks TODO**.
Trong lúc review, tiến trình khác đã tạo implementation M0 và registry ghi4tasks
VERIFIED; xem [readiness review](REVIEW.md) trước khi dùng các status đó để dispatch.
Current state đọc từ registry và receipts, không từ số đếm snapshot trong Markdown.
Mục đích của gói này là để bạn giao từng task cho `agy` với input, file cần viết,
checklist, dependencies, tests và evidence đủ cụ thể. Không có benchmark mới,
formal certificate mới hoặc gate PASS mới được tạo bởi việc viết plan.

## Đọc và dùng theo thứ tự

1. [GOAL.md](GOAL.md): goal tổng, research questions, frozen reference, success
   criteria và các kết luận hợp lệ khi nghiên cứu ra kết quả âm.
2. [ORCHESTRATOR.md](ORCHESTRATOR.md): quy tắc claim, file locks, review,
   state transitions và prompt dùng trực tiếp khi dispatch `agy`.
3. [TASKS.md](TASKS.md): danh sách 40 tasks theo dependencies, link từng task card.
4. [FILE_SPEC.md](FILE_SPEC.md): từng file dự kiến cần viết gì, API/data fields,
   điều kiện lỗi và test owner. Đây là spec, chưa là code đã chạy.
5. [HYPOTHESES.md](HYPOTHESES.md): mapping đúng 24 H01–H24, current evidence,
   task xử lý, proof/experiment và điều kiện thay đổi disposition.
6. [COMPARISON.md](COMPARISON.md): method matrix, fair comparisons và templates
   bảng coverage/bounds/runtime/memory/hypotheses; ô chưa đo luôn để pending/N/A.
7. [tasks.json](tasks.json): nguồn chuẩn cho task ID, deps, assignee, status và
   review outcome. [Receipt template](templates/TASK_RECEIPT.json) là schema mẫu,
   không phải evidence execution.
8. [REVIEW.md](REVIEW.md): review actual plan và implementation interface, các
   rework acceptance R01–R06 trước autonomous dispatch.

[PLAN.md](../../PLAN.md) là entry point root cho toàn bộ milestones.

## Bắt đầu bằng một task cụ thể

Mở [W4-001](tasks/W4-001.md), dùng prompt trong ORCHESTRATOR.md và thay
`<TASK_ID>` bằng `W4-001`. Agent đọc deps/state trước khi sửa file. Khi một task
nộp REVIEW, orchestrator kiểm receipt/diff/tests rồi mới đánh dấu VERIFIED.
W4-004 xây claim CLI; trước khi có CLI dùng cùng giao thức dưới dạng manual.
Lệnh CLI và run commands ghi trong spec là **PROPOSED**, chỉ chạy sau khi task
tương ứng đã implement và `--help`/tests xác nhận interface.

Đường critical path: 001 → 003/005 → 006–009 → 010 → 022/023 → 031 → 032–040.
Nhánh geometry và incremental có thể tiến hành độc lập khi deps/file locks cho
phép. Numerical failure không cho bypass gate; formal-reuse experiments báo zero
eligible methods, diagnostic/incremental studies giữ scope riêng.

## Nguồn chuẩn và preservation

Base đọc khi lập plan: `eede68e418713e4aed331705ce82834a3b4bfa94`, branch `main`,
working tree sạch trước khi tạo gói Week 4. Re-snapshot HEAD/dirty state ở W4-001;
không coi hash này là execution HEAD của task tương lai.

- [Week 1 source of truth](../../00_OVERVIEW/week1/SOURCE_OF_TRUTH.md).
- [Week 3 index/evidence](../../output/week3_certification_20261103/README.md).
- [Week 3 mathematical authority](../../03_MATH/README.md).
- [Current gates](../../GATES.md).

Frozen checkpoint SHA-256:
`8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
Preprocessing ID:
`aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.
Week 3 delivery receipt SHA-256 khi lập plan:
`5d3802b5f97732bc836b522ed36ba623c04cb5b1e083aaa8d302a71397eaf388`.

Giữ Week 1–3 sources/evidence, historical GAP/K=6 code, checkpoint, labels/splits,
normalization và `.venv` frozen. Code Week 4 nằm dưới `ecdr/week4/`, configs
`configs/week4/`, scripts/tests trong thư mục đã có; reports/design dưới gói này,
execution receipts dưới `output/week4/<run_id>/`. Không cần tạo một top-level
`docs/` hoặc đổi architecture để lấy speedup.

## Hai loại hoàn thành phải báo riêng

**Engineering closure:** các task khả thi được implement/review; task bị hạn chế
có blocker/evidence và explicit disposition. `VERIFIED_LIMITED` không tính như
đã implement feature còn thiếu; final delivery nêu rõ phần partial.

**Scientific success:** numerical validity, valid nonzero certified reuse và
positive net savings cùng xuất hiện ở cùng configuration/protocol. Không được
gom proof của method A với coverage của B và speedup của C để claim H24. G2 HOLD
vẫn là giới hạn dataset/model quality riêng; ACCEPTED chỉ user/người có quyền gắn.
