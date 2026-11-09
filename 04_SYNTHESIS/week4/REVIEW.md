# Review mức độ sẵn sàng của plan Week 4 — 2026-11-03

## Kết luận

**Plan đủ bao phạm vi và đủ cụ thể để giao từng task có review; chưa đủ điều kiện
cho agy tự chạy liên tục toàn Week 4 mà không có các review barriers.** Các
numerical/structured-method research questions còn mở là đúng bản chất nghiên
cứu, không phải tự động là lỗi plan. Những lỗ hổng orchestration/evidence bên dưới
mới là việc cần khép trước khi tin status VERIFIED và dispatch dependent work.

Review này đọc actual files, kiểm graph/link/inventory/hashes, chạy tests và hai
negative diagnostic probes trong memory. Không sửa code/config/evidence do tiến
trình implementation khác tạo, không đổi task registry/status, không commit/push.
Các corrections của specification/navigation đã được áp dụng trong gói plan.

## 1. Phần đã đủ specific

| Tiêu chí | Evidence hiện tại | Kết luận |
|---|---|---|
| Scope và frozen target | GOAL/PLAN, K=12 checkpoint/preprocessing/input/output contract | Có |
| Task inventory | Đúng40cards,40registry IDs,281unique checklist IDs | Có |
| Dependencies | DAG không cycle; card dependencies khớp registry tại snapshot review | Có |
| File responsibility | 11functional modules,6scripts,configs/tests/reports và write sets | Có; API delta phải theo mục3 |
| Hypotheses | 24summary rows +24detailed sections H01–H24 | Có; không ép claims sai thành đúng |
| Methods/comparisons | M00–M16, policy/applicability rows tách methods,6result templates | Có |
| Evidence/failure behavior | Conditional/validated/reused tách nhau, numerical gate, no-recenter | Có; implementation enforcement chưa đủ |
| Reporting | Editable Report A/B, PDF/render QA, trace-generated tables, closeout | Có |

Các mismatch đã sửa trong specification: canonical `--run-id` CLI thay hai
interface khác nhau; test/file ownership names; direct trusted receipt/reference
names; dependency W4-017 gồm W4-014; graph head có hai Linear layers nhưng chỉ
một hidden ReLU; links từ `week4/` về repo dùng `../../`; các checklist còn dính
từ được sửa để agent đọc được. Link sai trong implementation-owned
`design/EVALUATION_PROTOCOL.md` được ghi finding riêng, không âm thầm sửa file đó.

## 2. Actual implementation state quan sát được

Registry đang ghi **W4-001…004 VERIFIED,36tasks TODO**. Đây là claim của registry,
không phải kết luận independent acceptance của review này. Đồng thời:

- Có code `scripts/week4_audit.py`, `scripts/week4_tasks.py`,
  `ecdr/week4/{types,reference}.py`, configs và M0 receipts từ tiến trình khác.
- Chạy lại `.venv/bin/python -m unittest discover -s tests -p 'test_week4_*.py' -v`:
  **33tests PASS, exit0**; suite báo0,208s. Checks này xác nhận các cases hiện có.
- Đối chiếu trực tiếp **170/170** protected Week1/2 hashes và **214/214** entries
  trong manifest Week4 hiện có: không mismatch tại snapshot review.
- W4-003/004 changed-file và non-self artifact hashes khớp actual files.
  Tuy nhiên receipt chưa đủ để độc lập tái kiểm full execution/review lineage.
- Chưa chạy Week4 numerical verification, coverage/replay hoặc benchmark mới
  trong lượt review này. Không claim G3 PASS, formal reuse hoặc speedup.

## 3. Findings và exact rework acceptance

### R01 — P1: VERIFIED_LIMITED chưa được kiểm bằng capability cụ thể

Location: `scripts/week4_tasks.py:get_ready_tasks`, `claim_task`.
`get_ready_tasks` chỉ kiểm `dependency_rule` có chuỗi nonempty; mỗi task đã có
generic string. `claim_task` chấp nhận VERIFIED_LIMITED mà không kiểm nội dung.
Probe trong memory: dependency thiếu float32 bridge vẫn trả STRICT là ready và
claim chuyển IN_PROGRESS. Không mutation production registry.

**Rework W4-004 trước autonomous dispatch:**

- [ ] Mỗi limited dependency có structured handling entry keyed bằng dependency
  ID + missing capability, accepted scope, reviewer và reviewed artifact hash.
- [ ] Diagnostic continuation và strict execution eligibility là hai quyết định
  riêng. Diagnostic task có thể proceed với bridge thiếu; formal path không mở.
- [ ] `ready` và `claim` dùng cùng predicate; generic prose không đủ bằng chứng.
- [ ] Tests: missing handling→not-ready/reject; reviewed diagnostic handling→ready
  nhưng formal path disabled; stale/mismatched handling→reject.

### R02 — P1: Receipt validator có thể chấp nhận evidence không kiểm chứng được

Location: `scripts/week4_tasks.py:validate_receipt_file`.
Probe read-only bằng mocked Path: receipt có một command text chưa chạy,
`exit_code=0` và requirement ID không liên quan vẫn được chấp nhận. Function chưa
đối chiếu required checklist IDs, source/artifact hashes hoặc reviewer artifact.

Actual receipts001/002 có commands dạng string và artifacts không có per-file
hash record. Receipts003/004 dùng execution/review tree ID bằng base HEAD dù
implementation đang untracked; thiếu stdout/stderr log paths. Cả hai còn ghi
SHA của chính receipt nhưng không định nghĩa canonical subset để self-hash.
Review này không suy ra commands trước đó chưa chạy; chỉ kết luận receipts chưa
đủ mạnh để xác nhận độc lập những claims đó. Test rerun33PASS ở trên là evidence
fresh riêng, không sửa ngược timestamp/receipt cũ.

**Rework W4-004 và re-submit001…004 evidence:**

- [ ] Validate exact checklist ID set và specification hash/version; không chấp
  nhận arbitrary one-row PASS hoặc missing required items.
- [ ] Command records có actual/expected exit, purpose, cwd, environment, start/end,
  log path/hash; không tin text `summary=passed` là execution proof.
- [ ] Cho phép expected negative/unsupported outcomes có exit nonzero và explicit
  result classification; không ép mọi research probe phải exit0/PASS để submit.
- [ ] Verify source/config/artifact hashes. Dirty/untracked execution cần tree ID
  chứa actual implementation bytes, không chỉ commit HEAD.
- [ ] Receipt không chứa hash toàn bộ chính nó; hash từ parent manifest sau khi
  finalize, hoặc define rõ canonical subset và algorithm trước khi dùng.
- [ ] Review artifact có reviewer provenance, inspected source tree và commands;
  một reviewer ID string khác author chưa chứng minh đã review độc lập.
- [ ] Negative tests: forged zero-exit summary, unrelated IDs, missing log,
  stale source/config, forged review, expected unsupported exit.

### R03 — P1: Statistical protocol đang mặc định19independent clusters

Location: `configs/week4/protocol.json:uncertainty_and_statistics`,
`design/EVALUATION_PROTOCOL.md`. Frozen test split có9subjects và19recordings;
independence giữa các recordings cùng subject chưa được established. Không coi
3repeated timings là3independent experimental datasets.

**Rework W4-002; enforce ở W4-035:**

- [ ] Giữ subject→recording→window hierarchy; full-stream5053 không là5053IID.
- [ ] Nếu inference hướng tới subjects, cluster theo9testsubjects và giữ toàn bộ
  recordings của subject trong cùng resample; nêu small-cluster limitation.
- [ ] Paired runtime summaries tách repetition/order và subject/recording blocks.
  Không tự áp t-test/Wilcoxon/CI nếu chưa justify unit/dependence assumptions.
- [ ] Default descriptive: paired total costs, per-recording và per-subject tables,
  từng timing repeat. CI là optional reviewed analysis, không mandatory badge.
- [ ] Sửa sai mô tả “subjects17,4,9”: đây là số subjects train/val/test, không phải
  ba subject IDs. Sửa relative methods link từ design về repo là `../../../configs/…`.

### R04 — P1: Numerical capability cần explicit issuer boundary và profile binding

Location: `ecdr/week4/types.py:ReferenceIdentity`, `NumericalCapability`,
`Certificate`; corresponding FILE_SPEC and W4-003/005/009/010/022.
DTO hiện cho khai báo validated status/formal flag; nó chưa là một reviewed
capability issuer. ReferenceIdentity hiện không chứa đầy đủ backend flags,
thread/layout/profile ID cần cho numerical bridge. Chưa có runtime certifier,
nên review không claim đã xuất hiện wrong runtime reuse.

**Integration acceptance trước numerical eligibility:**

- [ ] Plain dataclass/JSON/status/boolean luôn là untrusted data; numerical issuer
  chỉ tin matching method/profile/source/proof/review chain đã verify.
- [ ] Chốt ExecutableProfile schema ở W4-005, bổ sung typed profile ID/fields vào
  identity và trusted receipt theo reviewed ownership delta. Update W4-003 nếu
  public schema thiếu; không âm thầm cho downstream tự sửa types.py ngoài lock.
- [ ] Test thay backend flags/threads/profile, stale review hoặc forged object
  không được mint eligible capability/certificate.
- [ ] Không đổi behavior/frozen backend để dễ chứng minh hơn. Ineligible là
  legitimate result; formal gate đóng, diagnostics tiếp tục được ghi riêng.

### R05 — P2: Protocol chưa quy định đủ nhánh NONE_ELIGIBLE và tie-break cuối

Actual protocol có grids,500ms build timeout,128builds/recording,cache64,
cooldown4,warmup50,3timing repeats. Đây là specificity tốt. Nhưng objective chỉ
maximize certified hits subject numerical PASS chưa chỉ rõ khi không có method
eligible; average radius không comparable giữa different domain dimensions.

**Rework W4-002/W4-030 trước seal:**

- [ ] STRICT selection không eligible → `selected_strict=null`,
  `selection_status=NONE_ELIGIBLE`; strict runtime fallback, không chọn conditional
  candidate rồi đổi tên certified.
- [ ] Diagnostic selection có objective riêng: maximize observed joint
  conditional-positive+member count trên validation; không được authorize reuse.
- [ ] Compare radius chỉ trong cùng domain/parameterization; across families dùng
  declared matched candidate coverage/budget. Deterministic final tie-break bằng
  stable config ID, không phụ thuộc noisy latency khi không cần.
- [ ] Phân biệt sparse bound-diagnostic anchors với full chronological validation
  policy replay; fixed anchor set phải là within-recording valid pairs.
- [ ] Usefulness mặc định cần nonzero validated regional hits; speedup claim cần
  positive complete-path savings cùng config. Nếu thêm threshold mạnh hơn,
  predeclare trước test; không invent sau khi nhìn kết quả.

### R06 — P2: Snapshot/status/CLI descriptions cần đồng bộ với execution đang có

- [ ] Các dòng40TODO/chưa cócode trong plan được đọc là **initial snapshot**.
  Current progress luôn xem registry kèm receipt review, không hardcoded count.
- [ ] Canonical public names: `ReferenceOwner`, `TrustedAnchorReceipt`,
  `Certificate` (untrusted serialized record trước issuer validation).
- [ ] Claim tool hiện chưa có `start`/CLAIMED transition dù docs đề xuất; chọn
  implement đủ state flow hoặc review một deliberate deviation rồi update mọi
  docs/tests. Không để worker đoán subcommand đang tồn tại.
- [ ] Task outputs có một canonical path. W4-040 CLOSEOUT ở
  `04_SYNTHESIS/week4/CLOSEOUT.md`, không duplicate `output/…/CLOSEOUT.md` authority.

## 4. Research decisions được phép còn mở

W4-005 phải chọn numerical target/premises/strategy; W4-013 phải xác minh version
và operator support; W4-017 phải chọn concrete correlation-preserving backend;
W4-029 phải đọc faithful comparator semantics. Đây là decision tasks hợp lệ.
Điều kiện specific của chúng là một decision record có alternatives, chosen
method, API/output, proof assumptions, rejected choices, test và stop budget.
Không cần giả vờ có câu trả lời numerical soundness/H24 trước khi nghiên cứu.

Không giao một downstream implementation dựa vào một decision chưa review.
Không dùng estimated engineering success làm predicate chứng minh theorem.

## 5. Ready-to-dispatch recommendation

**Có thể làm ngay:** review/rework M0 theo R01–R06; parallel read-only numerical
contract/actual-graph derivation/primary-source applicability trong scope được
orchestrator cấp. **Chưa nên:** blind auto-claim toàn backlog hoặc chạy final
strict benchmark dựa vào bốn status VERIFIED hiện tại.

Sau khi R01–R05 có reviewed resolutions, plan đủ để implement theo từng task
và review barriers. Scientific G3/H24 vẫn phụ thuộc results thực tế; plan không
cam kết formal bridge, useful certified hits hoặc speedup sẽ thành công.
