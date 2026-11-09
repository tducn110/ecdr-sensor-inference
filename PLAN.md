<!-- trustmebro -->
# Plan: ECDR Week 4 — implementation, numerical validity and measured feasibility

Ngày lập: 2026-11-03. Trạng thái: **SPECIFIED — implementation NOT_STARTED**.
Đây là plan và bộ giao việc cho `agy`, theo yêu cầu của user. Những mô tả
NOT_STARTED là initial snapshot; execution từ tiến trình khác đã xuất hiện trong
workspace. Current readiness và rework findings ở
[REVIEW.md](04_SYNTHESIS/week4/REVIEW.md); task statuses không tự đặt gate PASS.

Nguồn chuẩn của Week 4 nằm trong [bộ điều phối](04_SYNTHESIS/week4/README.md).
Plan này là entry point; không duy trì backlog thứ hai. Các phase là milestone
trong Week 4, không phải cam kết tất cả nghiên cứu sẽ thành công trong bảy ngày.

## 0. Decisions

| Chủ đề | Evidence hoặc xung đột | Quyết định |
|---|---|---|
| Frozen reference | HEAD khi lập plan `eede68e418713e4aed331705ce82834a3b4bfa94`; Week 3 triển khai từ `8d4ebae` | Giữ SmallCNN1D K=12, checkpoint/preprocessing/class order; snapshot lại khi bắt đầu task |
| Older roadmap | `04_SYNTHESIS/06_ROADMAP_SUGGESTION.md` còn K=6/G2 PASS lịch sử | Week 4 dùng plan này và bộ điều phối; không kế thừa trạng thái lịch sử |
| Numerical evidence | Week 3 ordinary float64 IBP; formal reuse đóng | Numerical gate phải có proof/review cho đúng executable; kết quả conditional vẫn được đo riêng |
| Wave geometry | Fixed-prefix wave là box suy biến, không giữ PCA correlation | Domain affine/residual và correlated propagation là hai việc riêng, không đổi tên box để claim H11 |
| Incremental path | H13 có interior alignment; H14 GAP không áp Flatten–MLP | Một nhánh tính lại đầy đủ output từ cached features; tách với nhánh bỏ forward nhờ decision certificate |
| Known test | Test 5.053 windows đã được xem ở Week 1–3 | Week 4 là KNOWN_TEST_EXPLORATORY; freeze policy trên train/val trước replay; không gọi untouched confirmation |
| Scope | User yêu cầu orchestrator trước khi chạy agy | Tạo đầy đủ goal/spec/task cards; implementation do user dispatch sau |

## 1. Objectives

**General objective.** Implement một pipeline ECDR tái lập được trên frozen K=12,
kiểm độc lập mathematical/executable premises, đánh giá đủ H01–H24 và trả lời
H24 bằng coverage/cost đo thật hoặc kết luận âm có giới hạn.

**Specific objectives.**

1. Tách reference, numerical engine, domain, bounds, certificate, runtime và evaluator.
2. Chứng minh hoặc chỉ rõ blocker của float32 enclosure; không bật reuse bằng flag.
3. Implement và so diagonal/wave/correlated domains, IBP/CROWN/Lipschitz,
   exact incremental candidate và các baseline đúng semantics.
4. Đo toàn trace, memory và build amortization; mọi comparison có matching rule.
5. Bảo toàn frozen evidence, ledger đủ 24 ID, reports/PDF và receipt độc lập.

**Deliverables.** [GOAL](04_SYNTHESIS/week4/GOAL.md),
[40 task cards](04_SYNTHESIS/week4/TASKS.md),
[file specifications](04_SYNTHESIS/week4/FILE_SPEC.md),
[hypothesis matrix](04_SYNTHESIS/week4/HYPOTHESES.md),
[comparison contract](04_SYNTHESIS/week4/COMPARISON.md), implementation/test/config
theo task cards, hai reports editable + PDF, và `output/week4/<run_id>/` receipts.
Các file implementation/report trong danh sách là **planned**, chưa được tạo.

## 2. Technical design

`raw QueryView → frozen normalization/reference → candidate selector → bound engine
→ numerical eligibility → certificate store → membership/lifecycle → reuse/fallback`.
Label chỉ được join sau decisions để stratify/compute metrics. Incremental executor
là adapter riêng, chạy đầy đủ Flatten–MLP head; không đổi nghĩa decision reuse.

Chi tiết interface/file ownership tại [FILE_SPEC](04_SYNTHESIS/week4/FILE_SPEC.md).
Task state canonical tại [tasks.json](04_SYNTHESIS/week4/tasks.json); task cards
chứa checklist và evidence contract, không tự chứng minh task đã hoàn thành.

## 3. Research framing

**RQ1:** Có enclosure cho actual executable float32 nào đủ chặt để đóng numerical
obligation mà không thay reference không? **RQ2:** Geometry/method nào cho
coverage tốt dưới matching rule đã freeze? **RQ3:** Incremental computation hoặc
certified decision reuse có giảm total cost trên workload đã khai báo không?
**RQ4:** H21/H24 được hỗ trợ đến mức nào; comparison nào chưa tương thích semantics?

Contribution dự kiến là một evidence chain và kết quả có thể bác bỏ; không claim
novelty, useful reuse hoặc speedup trước khi có dữ liệu. H01–H24 giữ nguyên nghĩa.

Metrics bắt buộc: attempted/failed/conditional/validated builds, member counts,
accepted reuse, reused/fallback disagreements riêng, full-stream agreement,
supervised accuracy/F1, mean/p50/p95 total latency, total-trace ratio, build
amortization, memory components. Definitions và templates ở COMPARISON.md.

Evaluation: frozen 17/4/9 subject split; test 5.053 full-stream/3.572 supervised,
5.034 adjacent pairs/19 recordings. PCA/proposal fit train; tuning val; protocol
hash frozen trước Week 4 test access. Không giả vờ test chưa từng được xem.
Timing repetitions không là training seeds hoặc independent subjects. Báo từng
recording và paired repetition; uncertainty có unit/resampling assumptions.

Mỗi execution tạo run ID mới, giữ source/config/environment hashes, command,
exit code và timestamps. Không overwrite output Week 1–3 hoặc run đã sealed.

## 4. Environments

| Profile | Trách nhiệm | Boundary |
|---|---|---|
| Frozen CPU | Baseline/reference, default one Torch thread | Snapshot actual version/settings; không thay `.venv` |
| Isolated verifier | Kiểm auto-LiRPA/version/operator support | Environment ngoài frozen `.venv`; chuyển bounds về reference chỉ khi có bridge |
| Optional device | Chỉ nếu user cấp thêm device/workload | Gate numerical và equality phải review lại; chưa là scope mặc định |

## 5. Schedule — 8 milestones trong Week 4

### M0: Contract và provenance — W4-001…W4-004

**Build:** snapshot, protocol/config schemas, public types, claim/receipt utility.
**Measure:** kiểm hashes, denominators, ownership và collision/claim concurrency.
**Write:** source map và frozen protocol.
**Exit:** có exact run identity, task ownership và schema chặn labels/runtime oracle.

### M1: Numerical feasibility — W4-005…W4-010

**Build:** arithmetic contract, outward primitives, normalization/membership,
actual operator enclosures và float32 bridge.
**Measure:** adversarial/high-precision cases; scope, tightness, failure cases.
**Write:** derivation và independent numerical decision.
**Exit:** ELIGIBLE_FOR_PROFILE hoặc INELIGIBLE_WITH_BLOCKERS; cả hai phải có evidence.
INELIGIBLE đóng formal reuse, vẫn cho diagnostic experiments chạy.

### M2: Bounds và domains — W4-011…W4-018

**Build:** 11 direct margins, Lipschitz, isolated LiRPA/CROWN, wave, PCA/affine,
correlated propagation và matched geometry driver.
**Measure:** tightness/build/membership trên fixed train/val anchors.
**Write:** method/domain assumptions, comparison scopes và unsupported matrix.
**Exit:** mọi method có implemented/unsupported disposition, không giả làm certified.

### M3: Actual graph execution và lifecycle — W4-019…W4-026

**Build:** feature derivation/incremental engine, executable equality gate,
certificate state, runtime, safe cost reductions, horizon và negative checks.
**Measure:** cache boundary/ties/reset/mutation, per-component cost và feature equality.
**Write:** exact-vs-empirical equivalence, lifecycle/horizon obligations.
**Exit:** valid queries fallback đúng; formal path chỉ mở nếu numerical review đủ.

### M4: Baselines và pre-test review — W4-027…W4-031

**Build:** all-hypothesis proof review, strict/heuristic baseline adapters,
LipCache applicability, validation selection, integration freeze.
**Measure:** train/val smoke, timeout budgets, protocol reproducibility.
**Write:** pre-test review, selected policy và remaining blockers.
**Exit:** hashes sealed; no label oracle; unsupported methods ghi rõ lý do.

### M5: Full experiments — W4-032…W4-034

**Build:** full replay, paired benchmark và memory/scaling harness.
**Measure:** toàn 5.053 queries, recording-aware; failed builds, cold/warm/setup,
query and loop totals, memory; không có shadow forward trong timed production path.
**Write:** immutable run receipts và failure logs.
**Exit:** đủ declared configurations hoặc explicit timeout/unsupported disposition;
không bỏ run chậm để làm đẹp bảng.

### M6: Synthesis và independent review — W4-035…W4-036

**Build:** table generator, metric audit, complete H01–H24 ledger.
**Measure:** independent recount từ traces, paired summaries, frozen preservation.
**Write:** result matrices, scope/limitations, review findings và resolutions.
**Exit:** số liệu tái sinh được; mọi claim có artifact/hash, blocker không bị ẩn.

### M7: Reports và delivery — W4-037…W4-040

**Build:** Report A, Report B, reproducibility guide và final validator.
**Measure:** execute relevant checks; compile/render/read mọi trang PDF cuối.
**Write:** closeout và evidence-based gate recommendation.
**Exit:** implementation/delivery status tách scientific success; user ACCEPTED để trống.

## 6. Report outline

| Section | Tasks cung cấp | Tables/figures |
|---|---|---|
| Problem, data, frozen graph | 001–005 | Identity, data/denominator table |
| Proofs and numerical contract | 006–014, 019, 025, 027 | Operator/bridge/assumption matrix |
| Domains/runtime/methods | 015–026, 028–031 | Architecture, lifecycle, methods |
| Results and uncertainty | 032–036 | Bounds, coverage, cost, memory,24-H ledger |
| Conclusion and reproduce | 037–040 | H24 conjunction, gates and unresolved items |

## 7. Risks and fallbacks

| Risk | Tín hiệu | Hành động |
|---|---|---|
| Backend arithmetic không có error contract | 009/010 không đủ justification | Formal reuse đóng; conditional method vẫn có diagnostics; không đổi backend để giả equivalent |
| Certified region không useful | Bound-positive nhưng không member | Giữ kết quả âm; validation-only policy revision tạo protocol version mới |
| CROWN không support/cost quá cao | Operator failure/timeout | Ghi unsupported/timeout; không sửa frozen environment |
| Incremental output khác backend | Near-tie/kernel reduction witness | Chặn strict adapter; giữ experimental measurement riêng |
| LipCache khác semantics/cần train proxy | Faithful HAPT adaptation chưa có | Applicability audit + NOT_ESTABLISHED; không tự train model khác |
| Overhead dominates | Full trace ratio≤1 | Báo negative result và break-even, không inflate baseline |
| Agent overlap | Shared file/lock hoặc deps chưa review | Chờ owner/release; không overwrite, không tự claim DONE |

## 8. Deviations

- 2026-11-03: Khởi tạo plan Week 4 từ Week 3 đã publish; chưa có implementation
  execution hoặc scientific result mới. Mọi scope change phải ghi reason và
  user direction nếu đổi frozen reference/workload/quyền external write.
