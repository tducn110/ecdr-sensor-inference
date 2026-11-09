# Orchestrator contract — Week 4 / agy

## 1. Vai trò và authority

User điều khiển scope và dispatch. Orchestrator chọn task ready, cấp ownership,
kiểm dependencies, review evidence và cập nhật registry. Implementer hoàn thành
đúng task card. Independent reviewer kiểm proof/implementation/experiment phù
hợp chuyên môn; tác giả không tự review chính thay đổi của mình.

Đây là một orchestration specification. Không giả định `agy` đã có claim API,
headless mode hoặc syntax đặc biệt. User mở `agy` theo cách đang dùng và paste
prompt bên dưới. Trước automation phải kiểm `agy --help`/version trong task liên
quan; không bịa `agy claim ...` hoặc tự launch một interactive session lúc lập plan.

Đọc sources theo thứ tự: user direction hiện tại → applicable AGENTS →
GOAL/PLAN → task card → FILE_SPEC/HYPOTHESES/COMPARISON → actual source/evidence.
Task card có detail thực thi; khi conflict với frozen contract, dừng dependent
mutation, ghi conflict và đề nghị hướng xử lý. Không lấy old roadmap K=6 làm current.

## 2. Canonical state và statuses

`tasks.json` là task registry. `tasks/W4-NNN.md` là yêu cầu/checklist; receipt
execution là bằng chứng. Ba loại file có trách nhiệm khác nhau.

| State | Ý nghĩa | Ai có thể chuyển vào |
|---|---|---|
| TODO | Chưa claim | Orchestrator/registry initialization |
| CLAIMED | Đã cấp owner và file locks | Orchestrator hoặc claim utility sau khi kiểm deps |
| IN_PROGRESS | Implementer đang làm | Current owner |
| REVIEW | Có diff + receipt, chờ reviewer | Current owner |
| REWORK | Review phát hiện yêu cầu thiếu/sai | Reviewer/orchestrator |
| VERIFIED | Đạt đúng task purpose và required outputs | Independent reviewer/orchestrator |
| VERIFIED_LIMITED | Có reviewed limitation; required feature còn thiếu | Reviewer/orchestrator, với explicit missing outputs |
| BLOCKED | Không tiến hành được do dependency/environment/authority | Owner ghi blocker, orchestrator review |

`VERIFIED` của capability-decision task có thể là kết luận unsupported nếu task
purpose là ra quyết định có evidence; nó không cho numerical method nhận status
validated. Nếu task purpose là implement feature mà feature chưa có, dùng
VERIFIED_LIMITED hoặc BLOCKED và overall implementation vẫn partial. Không có
auto-DONE từ exit0, file tồn tại hoặc một reviewer đọc chuỗi `PASS` trong JSON.

Scientific disposition nằm trong result/proof ledger riêng: DERIVED, VERIFIED
(finite scope), CONDITIONAL, FALSIFIED, NOT_ESTABLISHED, DEFERRED. Task status
không tự đổi scientific disposition, G3/G4/G5/G6 hoặc user ACCEPTED.

## 3. Claim protocol

1. Đọc registry và exact task card. Ready nghĩa là mọi dependency đã review
   VERIFIED/VERIFIED_LIMITED, và missing capability có explicit handling trong
   task hiện tại. BLOCKED, REVIEW hoặc chỉ file đã tồn tại chưa là deps satisfied.
2. Snapshot current HEAD/branch/dirty state. Xác định file user/agent khác đang
   sửa. Không stage/commit thay đổi không thuộc task.
3. Acquire task claim + exclusive lock cho từng planned write path. Shared
   `ecdr/week4/*.py`, configs, aggregate tables và registry chỉ một writer/path.
   Hai task không chỉ khác ID mà còn phải không có conflicting write set.
4. Ghi owner/session/claim time/base SHA/planned paths. Không sửa code trước claim.
5. Owner cập nhật IN_PROGRESS và làm checklist. Nếu phát hiện cần thêm file,
   request lock qua orchestrator trước khi mở scope; đừng tiện sửa shared runtime.
6. Nộp REVIEW kèm task receipt, exact diff/files, commands/exit codes/hash artifacts,
   limitations và mapping từng checklist đến evidence. Giữ lock đến khi review
   chấp thuận hoặc orchestrator release để tránh review stale diff.
7. Reviewer rerun check hẹp có ý nghĩa, đọc proofs/negative cases/source, kiểm
   evidence version và scope; trả VERIFIED/VERIFIED_LIMITED/REWORK.
8. Release locks, cập nhật next-ready set. Không tự chuyển toàn bộ downstream
   tasks DONE hoặc overwrite registry từ một snapshot cũ.

Trước W4-004: orchestrator làm read–claim–edit thủ công, một writer registry tại
một thời điểm. Sau W4-004: utility đề xuất dùng OS exclusive create/lock và atomic
write, so base state/version trước mutation. Race claim loser phải nhận conflict;
không overwrite owner. Stale claim chỉ release sau khi xác minh session đã dừng
hoặc owner/user yêu cầu; timeout/elapsed time một mình không đủ để steal task.

Proposed CLI sau khi implement W4-004 (chưa tồn tại tại thời điểm lập plan):

```bash
.venv/bin/python scripts/week4_tasks.py ready
.venv/bin/python scripts/week4_tasks.py claim W4-001 --agent AGENT_ID
.venv/bin/python scripts/week4_tasks.py submit W4-001 --receipt RECEIPT_PATH
.venv/bin/python scripts/week4_tasks.py review W4-001 --reviewer REVIEWER_ID --decision VERIFIED
```

`review` là reviewer action với authorization phù hợp, không implementer tự gọi
để tăng status. CLI không là cơ chế bảo mật chống hostile same-user Python;
claim ownership và numerical capability là hai trust boundaries riêng.

## 4. File ownership và concurrency

Code modules chia theo FILE_SPEC; không sửa Week3 monolith để tránh làm stale
Week3 source/hash receipt. New package `ecdr/week4/` import frozen reference
contract nhưng không dựa vào historical GAP bounds/incremental engine.

Sau W4-001 có thể song song protocol, isolated dependency audit, feature
derivation và prior-work applicability. Sau core types có thể chạy numerics,
domain và incremental branches khi lock sets rời nhau. Mọi task viết cùng file
phải serialize dù dependency graph cho phép độc lập về ý tưởng.

Các bottleneck cần một owner tại một thời điểm: `types.py`, `reference.py`,
`domains.py`, `bounds.py`, `runtime.py`, protocol freeze, method registry,
aggregate table generation và final delivery receipt. Numerical/runtime reviewers
đọc shared sources nhưng chỉ ghi review/design paths được cấp.

Không benchmark song song với experiments/PCA/PDF compiler trên cùng host. Cold
and warm repeats dùng declared order; chạy nhiều agents cùng CPU rồi báo timing
là independent repeats sẽ làm sai experimental contract.

## 5. Run identity và evidence receipt

Mỗi task execution ghi `output/week4/<run_id>/tasks/W4-NNN/receipt.json` theo
[template](templates/TASK_RECEIPT.json). `<run_id>` phải resolve thành ID cụ thể
trước khi gọi tool; không dùng placeholder như output path thực tế.

Run ID gồm timestamp UTC và unique suffix; manifest giữ user-facing timezone
Asia/Ho_Chi_Minh, UTC timestamps cho machine logs. Không overwrite run đã sealed.
Artifact list chứa relative path, size, SHA256, source/config identity và scope.
Command record gồm literal argv hoặc exact command, cwd, env profile, start/end,
exit code và stdout/stderr log paths. Secrets không đưa vào receipts.

Checklist evidence map có `requirement_id`, implementation symbol hoặc proof
section, test/command, output/hash, observed outcome, limitation. Không điền
`exit_code:0` cho command chưa chạy. Receipt chỉ consistency-check receipt cũ
phải ghi rõ `command_rerun=false`; không tự claim mới chạy93tests chỉ vì đọc log.

## 6. Numerical và equivalence gates

W4-010 là một independent decision gate. Nó bind method implementation hash,
reference/normalization/domain/membership versions, device/backend/profile,
numeric assumptions và proof review. H22 không được close bằng boolean trong
config, untrusted JSON/string `numerical_status=validated`, arbitrary epsilon,
finite sample agreement hoặc cuối reduction mới gọi nextafter.

Nếu gate INELIGIBLE: runtime strict không cấp certificate/reuse; mọi query hợp
lệ fallback. Diagnostic bounds/geometry/cost vẫn được ghi riêng. Không cho agent
đổi `.venv`/backend rồi ngầm coi đó là reference frozen cũ.

W4-021 quyết định incremental executable equivalence riêng: real feature theorem
và bitwise equality trên finite data không là universal equality proof. Chỉ có
reviewed sufficient contract mới đăng ký strict incremental method. Nếu chưa đủ,
adapter báo experimental scope và full reference fallback cho strict invocation.
Không dùng `allclose` để che changed winner/near-tie risk.

TrustedAnchorReceipt chỉ reference owner cấp sau actual inference. Caller đưa
arbitrary logits hoặc reused-query prediction không được mint trusted anchor.
Tối ưu identity phải khai báo owned model threat model khác per-query full hash;
không bỏ checks rồi giữ claim chống arbitrary in-process mutation.

## 7. Experimental freeze và fair reporting

Labels không có trong QueryView. Dataset/evaluator giữ labels riêng và join sau
decisions. Domain center/budgets chỉ dùng current/past observable inputs; không
peek next window/next prediction. Fit train, tune val, seal policy ở W4-030,
review W4-031 rồi mới chạy test replay. Test đã biết → KNOWN_TEST_EXPLORATORY.

Metric counters: `R=reused_queries`, `D_reuse=reused_disagreements`,
`D_fallback=fallback_disagreements`, `D_all=D_reuse+D_fallback`.
`A_reuse=1-D_reuse/R` nếu R>0, nếu R=0 dùng null/N/A. Không trừ D_all vào R.
Report full5053 và supervised3572 denominators riêng; attempted builds, conditional
positive, validated builds, member/joint/reuse không gộp hoặc suy từ marginal rates.

Mỗi method row luôn tồn tại: EXECUTED, UNSUPPORTED, TIMED_OUT, BLOCKED,
NOT_APPLICABLE hoặc PLANNED. Không nhập published image speedup thành HAPT result.
Strict methods so cùng reference prediction output; heuristic/proxy semantics
tách scope, không xếp chung formal-certified ranking. M00 full core và M01 full
wrapper luôn hiện, owned-reference M16 ghi threat model riêng.

## 8. Prompt dùng trực tiếp cho một worker agy

Thay đúng TASK_ID/AGENT_ID trước khi paste. Không yêu cầu worker làm toàn backlog
trong một session; task tiếp theo chỉ claim sau review task hiện tại.

```text
Bạn là implementer cho ECDR Week 4 tại /home/pro/Downloads/research.
TASK_ID=<TASK_ID>; AGENT_ID=<AGENT_ID>.

Đọc applicable AGENTS, PLAN.md, 04_SYNTHESIS/week4/README.md, GOAL.md,
ORCHESTRATOR.md, tasks.json và tasks/<TASK_ID>.md. Đọc FILE_SPEC.md,
HYPOTHESES.md và COMPARISON.md cho phần task này. Reopen actual source/evidence.

Thực hiện duy nhất task đã claim. Kiểm deps/review và HEAD/dirty state trước
mutation; acquire file locks/owner, bảo toàn thay đổi không liên quan. Nếu chưa
có claim utility W4-004, dùng manual protocol và báo orchestrator claim trước.
Không sửa frozen checkpoint, labels, splits, preprocessing, Week1–3 evidence
hoặc frozen .venv. Không branch/commit/push/deploy/chạy task khác tự động.

Implement tất cả checklist trong task card với ownership FILE_SPEC. Không bịa
code/API hiện hữu từ đường dẫn PLANNED. Numerical soundness, correlation,
actual padded/BN/Pool graph và reference equality phải giữ đúng assumptions.
Mọi unsupported/inconclusive/tie/invalid case fail closed theo contract.
Nếu dependency capability chưa established, ghi blocker/diagnostic fallback;
không bypass bằng epsilon, heuristic hoặc đổi model/backend/semantics.

Chạy checks đúng task và regressions liên quan; giữ command/exit/log/hash.
Không claim test chạy nếu chỉ đọc receipt cũ. Tạo task receipt từ template,
mapping từng requirement→file/symbol/proof→check→evidence→limitation.
Update task sang REVIEW, gửi exact changed paths, checks thực chạy, kết quả,
missing outputs và next review action cho orchestrator. Không tự mark VERIFIED
hoặc ACCEPTED. Dừng sau handoff task này.
```

## 9. Prompt cho independent reviewer

```text
Review TASK_ID của ECDR Week4, không tự implement task khác.
Đọc goal/task/file specs, current diff, exact execution receipts và dependencies.
Kiểm từng requirement và evidence scope; rerun meaningful narrow checks.
Đối với numerical/feature proof, kiểm premises và counterexample độc lập;
tests/sample agreement không thay proof. Đối với benchmark, recount raw traces,
denominators, identity/profile, timers, failed builds và no-label-oracle.
Ghi findings với path/symbol/witness, severity và check phân biệt nguyên nhân.
Kết luận VERIFIED / VERIFIED_LIMITED / REWORK, missing outputs và scientific
claims được phép. Không tự ACCEPTED, không nâng gate do status task.
```

## 10. Quy tắc escalation và closeout

Tool error phải phân biệt syntax/capability/permission/environment và thử safe
in-scope alternatives. Không lặp cùng lệnh hỏng hoặc declare success khi tool
chưa xác nhận. Blocker không đồng nghĩa đổi reference hoặc mở scope training.
User direction cần thiết khi đổi frozen behavior/workload/external state; agent
vẫn làm phần độc lập đã có quyền, không dừng toàn task chỉ để hỏi lại thủ tục.

W4-040 kiểm toàn bộ registry, task outputs, all24 hypotheses, method rows,
source/receipt/PDF hashes và frozen preservation. Engineering delivery có thể
VERIFIED_PARTIAL nếu feature bắt buộc chưa có. Scientific result có thể negative;
G2 HOLD được giữ. Không đồng nhất CREATED, VERIFIED và user ACCEPTED.
