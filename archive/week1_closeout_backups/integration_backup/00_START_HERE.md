# 00 START HERE — Event-Conditioned Certified Decision Reuse

**Đồng bộ tài liệu:** 22/10/2026 (Asia/Ho_Chi_Minh). Đây là cập nhật hồ sơ từ
source/artifact hiện có; không phải lần chạy lại training, data audit hay benchmark.

Nghiên cứu kiểm tra liệu miền đầu vào được mô hình hóa theo biến thiên sự kiện
có thể chứng nhận **cùng quyết định với frozen backbone** và giúp giảm tổng chi
phí inference trên chuỗi cảm biến hay không. Novelty, certified reuse và net
saving vẫn là giả thuyết; không có mục tiêu cứng giảm 50% latency.

## Hiện trạng cần nhớ

| Gate | Trạng thái đang ghi nhận | Giới hạn |
|---|---|---|
| G0 | Repo-recorded PASS | Đủ hồ sơ để tiếp tục prototype; chưa phải systematic review hay chứng minh novelty |
| G1 | COMPLETE_WITH_REVISIONS | Data audit có kết quả và correction; event-domain revision còn mở |
| G2 | Repo-recorded PASS | Có backbone K=6 và artifact training/evaluation; lượt này không chạy lại |
| G3–G6 | PENDING | Chưa có verifier soundness validation, certified ablation, end-to-end saving hay paper kết quả hoàn chỉnh |

Backbone hiện tại là [SmallCNN1D](scripts/backbone.py): Conv1d `6→32→64→64`,
kernels `5/3/3`, BatchNorm + ReLU, hai MaxPool, Flatten → Linear128 → Linear6;
**282.918 tham số**, input `(B,6,128)`. K=6 pure-only, K=12 HOLD theo
[class decision](G2_CLASS_DECISION.md). Checkpoint `checkpoints/backbone_best.pt`
chọn bằng validation ở seed 44. Artifact ghi test accuracy `0,9349`, macro-F1
`0,9354` trên **2.874 test pure K=6 windows**; không đại diện toàn test stream.

CPU mean/p50/p95 lưu sẵn `0,2511/0,2357/0,3313 ms` chỉ đo forward trên tensor
đã normalize trong RAM. Đây không phải measured certified reuse/net saving,
hiệu năng thiết bị khác hoặc benchmark mới chạy ngày 22/10.

## Đọc theo nhu cầu

| Tài liệu | Vai trò |
|---|---|
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Problem, thuật ngữ window/event/region, scope và evidence hiện tại |
| [report.md](report.md) | Hồ sơ nghiên cứu tổng hợp hiện tại; observed/proposed/open có nguồn |
| [latex/dossier.tex](latex/dossier.tex) | Nguồn LaTeX của báo cáo; compile/render status theo sync record |
| [DOCS_SYNC_2026-10-22.md](DOCS_SYNC_2026-10-22.md) | Manifest đồng bộ, checks và phần chưa verify |
| [GATES.md](GATES.md) | Tiêu chí G0–G6; recorded status tách khỏi kiểm chứng mới |
| [01_RQ_GAP.md](01_RQ_GAP.md), [02_LITERATURE.md](02_LITERATURE.md), [refs.bib](refs.bib) | RQ, claim ledger, prior work và bibliography; entry không đồng nghĩa full reading |
| [03_DATA_AUDIT.md](03_DATA_AUDIT.md), [G2_CLASS_DECISION.md](G2_CLASS_DECISION.md) | Data/split/coverage, correction, label policy và backbone evidence |
| [04_THEORY.md](04_THEORY.md) | Formulation/assumptions; không tự chứng minh implementation |
| [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md) | Miền đầu vào, normalization và membership cần thống nhất trước verifier |
| [05_READING_PLAN.md](05_READING_PLAN.md), [06_REVIEW_PROTOCOL.md](06_REVIEW_PROTOCOL.md), [notes/README.md](notes/README.md) | Inventory P01–P23, protocol pilot, notes và search provenance |
| [roadmap suggestion.md](<roadmap suggestion.md>) | Dependency, milestone và điều kiện go/no-go G3–G6 |

Source/config/JSON/checkpoint là evidence cho behavior và số đo; narrative không
ghi đè chúng. `GPT-sources/` là tổng hợp nội bộ. Bản ghi 16/10 và
[kickoff 21/10](notes/2026-10-21_REVIEW_KICKOFF.md) giữ như snapshot lịch sử.
Bản root `report.md` cũ còn trong Git history; `report.md` hiện tại là hồ sơ
đã cập nhật, không còn bị khóa với nhãn LEGACY_REFERENCE.

## Việc tiếp theo

Thống nhất G3 domain contract và numerical/model/preprocessing assumptions;
đọc P01/P03/P08/P09/P10 theo nguy cơ collision; kiểm verifier trên backbone thực
tế rồi mới nối membership, certificate store và fallback. Không suy certified
reuse từ membership coverage, không coi overlap 50% là input similarity và
không nâng reused window thành trusted anchor chỉ từ prediction đã reuse.

Skill/tool audit gần nhất là snapshot [21/10](notes/2026-10-21_REVIEW_KICKOFF.md):
TMB generic package đã import dùng để định tuyến; scientific skill bên ngoài
chỉ được tìm/đánh giá trong lượt đó. Đồng bộ 22/10 không xác nhận cài đặt mới
hoặc thay đổi account/runtime state.
