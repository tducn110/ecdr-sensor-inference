# Kế hoạch hiện tại — Event-Conditioned Certified Decision Reuse

**Đồng bộ:** 22/10/2026. File này thay đề xuất 15/10 bị lỗi thời; bản trước còn
trong Git history. Phạm vi là milestone nghiên cứu G3–G6, không phải TMB
`PLAN.md`/`ROADMAP.md`. Việc viết lại docs không chạy training hay đóng research gate.

## 1. Điểm xuất phát

RQ: event-conditioned class-pair bounds có thể chứng nhận frozen backbone giữ
quyết định thường xuyên hơn baseline phù hợp và đủ rẻ để giảm tổng chi phí
inference hay không? C11 novelty và C12 net saving vẫn UNVERIFIED.

| Gate | Hiện trạng từ hồ sơ | Việc còn mở |
|---|---|---|
| G0 | Repo-recorded PASS | Collision search và novelty tiếp tục; literature review còn ở pilot |
| G1 | COMPLETE_WITH_REVISIONS | Domain, coordinates, centering và residual phải khớp trước G3 |
| G2 | Repo-recorded PASS | Giữ checkpoint và protocol K=6 hiện tại |
| G3 | PENDING | Miền, membership, valid bounds và model/numerical identity |
| G4 | PENDING | Certified reuse/tightness và ablation công bằng |
| G5 | PENDING | Runtime thật gồm lookup/membership/fallback/rebuild |
| G6 | PENDING | Paper kết quả và gói tái lập đủ evidence |

Nguồn: [start here](00_START_HERE.md), [brief](PROJECT_BRIEF.md),
[gates](GATES.md), [report](report.md), [sync record](DOCS_SYNC_2026-10-22.md).
Số dưới đây là artifact đã lưu, không phải phép đo mới ngày 22/10.

Backbone thực tế là [SmallCNN1D](scripts/backbone.py), config
[backbone.yaml](configs/backbone.yaml): **282.918 tham số**, Conv1d kernels
`5/3/3`, BatchNorm + ReLU, hai MaxPool, Flatten 2.048 → Linear128 → Linear6.
Kiến trúc no-BN/GAP khoảng 24 nghìn tham số trong đề xuất cũ không phải model
đang dùng. Shape `(B,6,128)→(B,6)`; seed 44 chọn bằng validation; test macro-F1
`0,9354` trên 2.874 pure K=6 windows. 3-seed mean ± population std
`0,9122 ± 0,0183` không phải confidence interval.

CPU forward mean/p50/p95 `0,2511/0,2357/0,3313 ms` trong
[g2_cpu_latency.json](data/g2_cpu_latency.json) chỉ là baseline trên tensor
normalized trong RAM. Chưa có speedup của phương pháp; không dùng latency
simulation trong đề xuất cũ làm kết quả hay mốc hiệu năng.

## 2. G3 — thống nhất miền và kiểm verifier

Đầu vào: [data audit](03_DATA_AUDIT.md), [theory](04_THEORY.md),
[G3 domain contract](07_G3_DOMAIN_CONTRACT.md), backbone/checkpoint và P08/P09/P10.
Thứ tự dependency:

1. Chốt raw-input graph có normalization hoặc biến đổi center/directions/budgets
   sang normalized input. Membership và certificate phải nói về cùng tập.
2. Chọn candidate region; ghi centering, residual norm, budget fit trên
   train/validation, anchor và model identity. PCA coverage audit chưa chứng minh
   miền ở theory đã được triển khai.
3. Tách exact membership khỏi check đủ bảo thủ; event selector chỉ đề xuất region.
4. Kiểm auto_LiRPA/IBP/CROWN trên backbone có **BatchNorm và MaxPool**; ghi version,
   eval mode, unsupported ops và numerical assumptions. Proposed library không
   đồng nghĩa đã cài hay đã hỗ trợ custom correlated domain.
5. Kiểm class-pair specification, toy cases có đối chiếu độc lập và graph thực.
   Chỉ reuse khi bounds hợp lệ trên toàn miền, mọi lower margin dương, membership
   và anchor/model/preprocessing/domain identity hợp lệ.

Đầu ra phải có: domain contract trạng thái rõ; proof/assumptions; compatibility
record; implementation và independent checks; fallback/trace evidence. Đây là
đầu ra cần làm, không phải artifacts đã hoàn thành.

Exit: cùng miền xuyên math → membership → verifier; bounds hợp lệ dưới giả định
đã ghi; certificates gắn đúng identity; accepted-hit mismatches được kiểm.
Zero observed violations là test evidence, không thay soundness proof. Chưa đạt
thì G3 mở.

## 3. G4 — giá trị của event structure

Dependency: G3 và baseline selection từ [reading plan](05_READING_PLAN.md),
claim ledger. Tách full inference, heuristic và certified baselines. So sánh
global Lipschitz, global-ball/box và event-conditioned domains với conditions và
budget/coverage minh bạch; tách domain geometry, bounds và reuse policy bằng ablation.

Báo dataset/split, denominator, membership coverage, certified reuse, reference
agreement, ground-truth accuracy, tightness và uncertainty theo strata phù hợp.
17.453 corpus windows, 5.053 test stream windows và 2.874 pure K=6 test windows
là ba denominator khác nhau. Coverage không phải certified reuse; PCA hơn box
trên budget chưa matched không chứng minh correlated domain tốt hơn.

Exit: có certified comparison tái kiểm được, gain hoặc negative result. Novelty
vẫn cần broader literature evidence; không đóng G4 chỉ từ data coverage audit.

## 4. G5 — net runtime saving

Dependency: G3/G4 và trusted-anchor lifecycle. Anchor phải thực sự chạy model;
không tự nâng reused window thành anchor. Trace mỗi hit về anchor, weights,
preprocessing, domain và certificate. Mô hình accounting cho phần computation
khác nhau giữa hai nhánh:

    E[T] = Tₑ + Tₘ + (1 − r)T𝒻 + qTᵦ
    Net saving khi rT𝒻 > Tₑ + Tₘ + qTᵦ.

`Tₑ`: mean event processing; `Tₘ`: mean lookup/membership/validation; `T𝒻`: mean
forward; `r`: reuse fraction; `q`: rebuilds/window; `Tᵦ`: mean cost/rebuild.
Dùng cùng đơn vị/denominator; không thay mean bằng p50 trong kỳ vọng. Đây là
cost model, chưa phải kết quả saving.

Đo full pipeline và đối chứng cùng môi trường, gồm preprocessing chung,
lookup/membership, fallback, certificate build/refresh, state maintenance và
memory. Lưu hardware/OS, versions, threads, warmup, repetitions, model identity.
Tách lookup-only với end-to-end; báo mean/p50/p95 và rebuild distribution.

Exit: measured end-to-end saving trong miền hoạt động được công bố, hoặc negative
result có trace giải thích. Không ép target 50%; không suy edge/energy/device
performance từ CPU forward benchmark hiện có.

## 5. G6 — paper và tái lập

[report.md](report.md) và [latex/dossier.tex](latex/dossier.tex) là hồ sơ khoa học
đang phát triển: observed evidence, formulation và protocol. File LaTeX hay
compile pass không đóng G6 khi kết quả G3–G5 chưa có.

Lắp paper từ ledger/notes, methods/assumptions, measured tables, ablations,
uncertainty, limitations và reproduction instructions. Mỗi kết luận trace tới
nguồn hoặc experiment artifact. Giữ CREATED/VERIFIED/ACCEPTED riêng; yêu cầu
đồng bộ docs không đồng nghĩa chấp nhận novelty, soundness hay kết quả nghiên cứu.

Exit: paper và package tái lập đủ evidence để người đọc hiểu/chạy lại kiểm cốt
lõi trong phạm vi công bố. Negative result hợp lệ; narrative đồng bộ chưa phải
G6 hoàn thành.

## 6. Literature và skill workflow

Theo [review protocol](06_REVIEW_PROTOCOL.md): P01/P03 → P08/P09/P10 → nhóm
structured/time-series/systems; lưu query, version, locator, screening và giới
hạn coverage. Review còn pilot, không gọi systematic review từ inventory hay
vài lượt web search.

[kickoff 21/10](notes/2026-10-21_REVIEW_KICKOFF.md) ghi TMB generic snapshot đã
import và K-Dense candidates chỉ được đánh giá. Đây không phải xác nhận live
installation state ngày 22/10. Protocol project độc lập; không cần game bootstrap
hay TMB game contracts cho research repo.

Việc tiếp theo: làm rõ G3 domain contract và verifier compatibility, đồng thời
đọc prior work có nguy cơ đổi gap/baseline. C11/C12 tiếp tục UNVERIFIED.
