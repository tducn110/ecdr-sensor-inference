# Week 4 — method registry, fair comparisons và result templates

Trạng thái: **SPECIFIED; chưa có Week 4 execution metrics**. Các method IDs dưới
đây là registry để W4-028…035 triển khai adapters, replay và tables. Source of
truth cho task state là [tasks.json](tasks.json). TODO là initial snapshot;
current implementation/readiness phải đối chiếu [REVIEW.md](REVIEW.md).
Không coi bảng pending như evidence đã chạy; không điền0 thay measurement thiếu.

Frozen reference, protocol và hypotheses ở [GOAL.md](GOAL.md),
[HYPOTHESES.md](HYPOTHESES.md) và
[Week 3 evidence](../../output/week3_certification_20261103/README.md).
Week 1–3 artifacts là immutable inputs của Week 4; output mới nằm dưới
`output/week4/<run_id>/`. Các tên output trong tài liệu này là **PROPOSED schema
keys/paths** cần W4-003/W4-035 chốt; chưa là tools hoặc files đã tạo.

## Guarantee target trước performance ranking

`Owned-reference` là classifier/checkpoint/preprocessing actual K=12 do project
đóng băng. Method muốn claim same-reference decision phải trả cùng classifier
đó trong proof scope. Method có một proxy khác, heuristic agreement hoặc different
architecture được ghi target riêng; không đặt chung guaranteed-equivalence badge.

Ba nhánh có acceptance nghĩa khác nhau:

1. **Exact input replay:** chỉ reuse khi canonical input identity/equality và
   frozen reference/runtime semantics khớp. Đây là singleton/equality reuse;
   không cần một nontrivial regional certificate và không chứng minh neighboring
   windows invariant. Hash trùng cần collision-safe equality policy, không chấp
   nhận hash alone như mathematical equality. Input key phải ghi raw hay normalized
   bytes, shape/dtype/layout/canonicalization, model/preprocess/runtime identity.
2. **Exact incremental computation:** reuse intermediate values nhưng vẫn tính
   current owned-reference decision. Real algebra identity, bitwise executable
   equality, decision-equivalence proof và finite zero-disagreement tests là
   bốn mức evidence khác nhau. W4-021 định nghĩa level đủ để enable method.
3. **Regional decision reuse:** nhận nhiều different inputs trong certified
   domain mà không chạy full owned-reference. Cần whole-domain11margin bounds,
   numerical bridge, sound membership và lifecycle. W4-010 chưa pass đúng scope
   thì formal regional reuse bị disable, dù conditional diagnostics vẫn chạy.

Exact input replay còn phụ thuộc determinism/runtime state scope. Không cộng
exact replay hits vào regional coverage; exact incremental không phải skipped
classification. Heuristic reuse report error/latency trade-off riêng.

## Stable method registry M00–M16

| ID | Method/control | Guarantee target và implementation obligation | Tasks chính | Eligibility/result hiện tại |
|---|---|---|---|---|
| M00 | Frozen full core | Raw input→frozen normalization→actualfloat32 fullforward; operational reference |028,031–034|PENDING; chưa chạy Week4|
| M01 | Full equivalent wrapper | Same query identity/lifecycle/state wrapper, luôn fullforward; isolate overhead |022–024,028,031–034|PENDING|
| M02 | Exact input replay | Owned-reference same-input result; collision-safe equality, bounded cache, recording/runtime key và deterministic contract |003,022,023,026,028,032–034|PENDING; singleton branch, không regionalcert|
| M03 | Exact incremental | Actual paddedConv/BN/pool intermediate reuse + correct current Flatten–MLP; W4-021 equivalence scope |019–021,026,028,032–034|PENDING; chưa có executableequivalencegate|
| M04 | Heuristic reuse | Explicit input/state selector + fallback; empirical reference-disagreement trade-off |015,022,023,026,028,030,032–035|PENDING; không formalcert|
| M05 | Global Lipschitz cache | Actual11margin bound và correct norm/ball membership; numerical gate |005–010,012,022,028,030,032–035|PENDING; historicalGAPimplementation không tự eligible|
| M06 | IBP diagonal | Actualgraph11directmargin bounds fullbox; separate-logit ablation |005–011,018,022,030,032–035|PENDING; gateclosed until numerical review|
| M07 | IBP fixed-overlap wave | Immutable verified prefix + tailbox; selector/center and reachable embedding explicit |007,011,015,018,022,030,032–035|PENDING; reducedcoordinates không tự cheaper|
| M08 | Correlated candidate via coordinate hull | Same affine/PCA candidate converted to enclosingbox; control for correlationloss |011,016–018,030,032–035|PENDING; không gọi nativecorrelatedverifier|
| M09 | Native correlated bounds | Sound affine/zonotope/factor-domain propagation và sound residualmembership cho actualgraph |005–010,016–018,030,032–035|PENDING; unsupported propagation phải expose|
| M10 | CROWN diagonal | Specified compatible CROWN variant/budget, matched fullbox và11objectives |013,014,018,030,032–035|PENDING; isolatedenv và conversionscope required|
| M11 | CROWN wave/correlated | Declared transformed wave/affinedomain và corresponding sound objective/propagation |013–018,030,032–035|PENDING; không silently hull correlation|
| M12 | Fixed-anchor policy control | Same selected boundbackend, immutable anchor/domain, per-querymembership; policy axis |010,022,023,025,026,030,032–035|PENDING; report parentmethod/config|
| M13 | Finite-horizon policy control | Same backend với independently bounded reachable sets tại everydeclaredhorizon |010,015,017,022,025,026,030,032–035|PENDING; unsupported horizon disabled|
| M14 | Faithful LipCache comparator | Source-defined guarantee; adaptation to HAPT chỉ nếu prerequisites/semantics rõ |029–036|PENDING_APPLICABILITY_AUDIT; chưa implement faithful comparator|
| M15 | Prior-work applicability only | StreamiNNC/TinyDejaVu architecture/runtime/training applicability audit; không fabricatedadapter |029,035–038|PENDING_APPLICABILITY_AUDIT; excluded from performanceranking|
| M16 | OWNED_REFERENCE baseline | Luôn chạy frozen reference forward trong ownership/identity profile tối ưu đã review; setup và invalidation costs explicit |005,022–024,026,028,031–035|PENDING; optional secondary baseline, không thay M00/M01|

M12/M13 là policy controls, không độc lập bound algorithm. Config ID phải chứa
`parent_method_id`, backend, domain và horizon. M11 cần subconfiguration ghi
`domain_kind=wave|affine`, không gom khác geometry thành một latency row. M15 có
hai separate audit records theo paper/artifact; không gộp chúng thành một method.

Mỗi method config chỉ được mang nhãn guarantee mạnh nhất có đủ evidence. Khi
numerical gate còn thiếu, M05…M13 có thể có bound diagnostics nhưng
`formal_reuse_enabled=false`; actual acceptedregionalreuse luôn tách diagnostics.
M02 có singleton equality contract và deterministic profile gate riêng; M03 có
equivalence gate riêng; không để một global numeric boolean tự chặn hoặc tự
authorize sai nhánh. Exact-input replay chỉ claim same admitted input trong
deterministic profile đã chứng minh, không gọi đây là nontrivial certified domain.

M16 có alias `M16_OWNED_REFERENCE`: secondary full-forward baseline cho threat
model ownership đã review ở W4-005/022/024/026. Profile phải khai báo ai giữ
model/preprocessing/runtime state, ai có thể mutate/hook, cách enforce mutation
boundary, khi nào identity invalid và setup/revalidation costs. Nếu không enforce
được immutability assumptions thì optimized identity profile không eligible.
Không bỏ hook/state mutation checks khỏi adversarial tests vì muốn giảm timing.
Methods so với M16 phải dùng cùng profile và report differences; M00/M01 vẫn là
hai primary comparisons bắt buộc. Kết quả under owned profile không tự áp cho
unrestricted Python mutation threat model.

## Primary-source applicability audits

W4-029 đọc primary paper và official artifact/version của LipCache,
StreamiNNC và TinyDejaVu trước khi kết luận. Những tên này trong plan chỉ tạo
audit obligation, chưa chứng minh capability hiện tại. Ghi source URL/version,
ngày đọc, exact guarantee target, architecture/operators, inference semantics,
hardware/runtime, training/data/calibration needs, code/license/reproducibility,
numeric scope và điều kiện adaptation.

| Prior work | Câu hỏi bắt buộc | Kết quả có thể hợp lệ | Không được kết luận |
|---|---|---|---|
| LipCache | Certified object là GuardNet/proxy hay ownedMainNet? Có chuyển guarantee sang frozenSmallCNN1D mà không retrain/đổi target được không? |FAITHFUL_COMPARABLE; DIFFERENT_GUARANTEE; NEEDS_NEW_STUDY; BLOCKED_ARTIFACT; NOT_APPLICABLE|Image-paper speedup là HAPTmeasurement; proxydecision tự equal MainNet|
| StreamiNNC | Operator/padding/pooling/stream state của method có khớp actualgraph? Exactness target và hardware support là gì? |Architecture/operator mapping và applicability-only disposition|Đã implement/benchmark chỉ vì tên liên quan streamingCNN|
| TinyDejaVu | Reuse target, layer transformations, runtime/architecture constraints và training requirements là gì? |Applicability-only disposition với exact blocker/possible adaptation|Transferred accuracy/speedup hoặc unknown semantics thành strictreferenceproof|

Không sửa architecture/checkpoint/preprocessing hoặc train proxy ngoài scope để
làm row comparable. Nếu cần authority mới, ghi bounded blocker và phần thực thi
chưa làm. Citation/link paper không thay faithful implementation.

## Thiết kế comparisons công bằng

**Geometry ablation H10/H11:** same anchor/winner/reference, cùng coordinate system,
same boundbackend/objectives và declared budget. Chốt matching criterion trên
validation: same center/constraints, candidatecoverage, radius, dimension hoặc
budget là các study khác nhau. Include M08/M09 paired control để đo correlation
thực sự. Fixed-prefix wave subset versus fullbox khác volume phải report rõ.

**Bound ablation H07–H09:** giữ inputdomain và11objectives, đổi direct/separate
margin, IBP/Lip/CROWN. Ghi slopes/optimization/time budget và operator support.
Nếu phương pháp fail/timeout, giữ attempt và reason; không chỉ chọn successes.

**Lifecycle H15–H17:** fixed trusted anchor versus finitehorizon policy trên same
backend và buildbudget; count resets/gaps/expiry/invalidations, age và hits/cert.
Groundtruthlabels không tham gia event selection hoặc build timing decisions.

**Computation H13/H14:** fullcore versus exactincremental fulloutput trên same
inputstream. Trace reused/recomputed layers/positions và fullhead cost. Có thể
giảm arithmetic nhưng không giảm latency; report measured outcome.

**Operational H20/H21/H24:** same raw RAM input order, CPU/device/thread scope,
warmup/repetition policy, batch1 và included phases. Report speedratio against
both M00 và M01. Không chỉ inflate baseline identitywrapper; không bỏ guard cost
khỏi method. Fullchronological trace includes builds, failures và nohitperiods.

**Known test:** dữ liệu test đã có Week3observations; W4-002/030 freeze mới phải
disclose prior exposure. No test-fitting, repeated winning-configuration selection
hoặc post-hoc success threshold. Validationselection và testconclusion tách nhau.

## Counters và denominators

| Counter/metric | Định nghĩa | Missing/zero rule |
|---|---|---|
| N_stream | Full chronological queries |FrozenHAPTtest expected5053; verify W4-001/032|
| N_supervised | Query đủ frozen labelpolicy cho activitymetrics |Expected3572; không thay operationaldenominator|
| N_pairs | Within-recording adjacent pairs |Expected5034; khác queries/certificate lifetime|
| B_attempt / B_failed | Mọi certificate build attempted / failed |Failedbounds/unsupported/timeouts giữ reason|
| B_positive_numeric | Có11positive reported numeric estimates |Chưa tự là executablecertificate|
| B_executable_valid | Build đủ numerical/identity/domainproof trong declaredscope |Gateclosed⇒0actualcertificates, không đoán positivity|
| Q_candidate_member | Query thuộc candidate theo checker |Không imply validcertificate hoặc reuse|
| Q_positive_and_member | Joint count trên same trace/domain/anchor |Không lấy product của marginalrates|
| R_regional | Query actually accepted regional certificate |Tách exactinputreplay và incremental queries|
| R_exact_input | Query accepted singleton/exactinputreplay contract |Không cộng thành nontrivialregionalcoverage|
| Q_incremental | Current outputs tính bằng incremental executor |Không là unchangeddecisioncachehits|
| A_regional | Regionalreused queries equal reference / R_regional |R_regional=0⇒null/N/A|
| A_exact_input | Exactinputreplay outputs equal reference / R_exact_input |Zero denominator⇒null/N/A|
| D_reference | Queries different from frozenreference |Wholetrace count và acceptedpath count riêng|
| Accuracy / macroF1 | Against supervised activitylabels |Target khác referenceagreement; no unlabeled oracle|
| Speedratio_core | Paired totalM00 / totalmethod |Ratio>1 là faster cho declaredscope|
| Speedratio_wrapper | Paired totalM01 / totalmethod |Report alongside core, không substitute|
| Net_savings_core | (totalM00−totalmethod)/totalM00 |Count completeoperationalpath|

`PENDING` nghĩa chưa đo; `null` với reason khi undefined; `NOT_APPLICABLE` khi metric
không đúng semantic target.0chỉ dùng actual measuredcount hoặc logically disabled
path có receipt. Không điền100%agreement khi no acceptedreuse.

## Mandatory comparison record schema

W4-003/028/035 chốt typed fields, validation và serialization. Minimum fields:

- Identity: `run_id`, `config_id`, `method_id`, `parent_method_id`, `task_id`,
  `git_head`, dirty/source hashes, timestamp, config/protocol hash.
- Frozen reference: checkpoint/preprocess hash, graph/split/manifest identity,
  K,C,W,stride, normalized/raw coordinates, dtype, backend/device/version/threads.
- Guarantee: `target_semantics`, `mathematical_status`, `numerical_status`,
  `equivalence_status`, `formal_reuse_enabled`, proofscope/receipt IDs.
- Geometry/bound: domainkind/center rule/free dimensions, basis+residual identity,
  radius/norm/matching rule,11objectives, backend/slopes/budget, horizon/buildpolicy.
- Protocol: train-fit hashes, validationselection hash, known-test disclosure,
  order/reset rules, warmup/repetitions/timeouts and resourcecaps.
- Counts: all counters above; per-stratum and fallbackreason counts; acceptedpath
  referenceagreement and activitymetrics with explicitdenominators.
- Cost: paired totaloperationalns, per-query mean/p50/p95, setup/load/shadowcost,
  selector/identity/normalize/build/member/fallback/state/transfercomponents.
- Memory: baselinepersistent/referencebytes, additionalpersistentmodelcachebytes,
  certificate/domainstatebytes, workspaces, allocator/RSS/GPUmeasurementmethod,
  baseline/delta and units; unmeasuredfields carry reason.
- Status: `execution_status`, errors/unsupported/timeouts, evidencepaths, blockers,
  permittedclaim, missedacceptancecriteria và nexttask.

Giá trị `execution_status` tối thiểu:
`PENDING | EXECUTED | BLOCKED | NOT_APPLICABLE | DEFERRED`.
Taskstate và scientificstatus là fields riêng. `EXECUTED` không đồng nghĩa
guarantee đã proved hoặc task VERIFIED; reviewreceipt quyết định tasktransition.

## Template A — applicability và correctness

Tạo một row mỗi configuration; trong plan chỉ để PENDING. Có thể mở rộng
M12/M13 bằng parentmethod/horizon; M15 tách StreamiNNC/TinyDejaVu.

| Method/config | Frozen target match | Real proof | Executable numerical/equivalence gate | Enabled path | Evidence / blocker |
|---|---|---|---|---|---|
| M00 / pending |PENDING|N/A fullforward|PENDING referencecontract|PENDING|PENDING|
| M01 / pending |PENDING|N/A fullforward|PENDING wrappercontract|PENDING|PENDING|
| M02 / pending |PENDING|PENDING singletoncontract|PENDING equality/runtime|PENDING|PENDING|
| M03 / pending |PENDING|PENDING actualincremental|PENDING W4-021|PENDING|PENDING|
| M04 / pending |PENDING|N/A heuristic|N/A formalguarantee|PENDING|PENDING|
| M05 / pending |PENDING|PENDING|PENDING W4-010|PENDING|PENDING|
| M06 / pending |PENDING|PENDING|PENDING W4-010|PENDING|PENDING|
| M07 / pending |PENDING|PENDING|PENDING W4-010|PENDING|PENDING|
| M08 / pending |PENDING|PENDING hull|PENDING W4-010|PENDING|PENDING|
| M09 / pending |PENDING|PENDING correlated|PENDING W4-010|PENDING|PENDING|
| M10 / pending |PENDING|PENDING|PENDING W4-010 + env|PENDING|PENDING|
| M11 / pending |PENDING|PENDING transform|PENDING W4-010 + env|PENDING|PENDING|
| M12 / parentpending |PENDING|PENDING fixedanchor|PENDING parentgate|PENDING|PENDING|
| M13 / parentpending |PENDING|PENDING reachablehorizons|PENDING parentgate|PENDING|PENDING|
| M14 / pending |PENDING sourceaudit|PENDING sourcetarget|PENDING applicability|PENDING|PENDING|
| M15 / StreamiNNC |PENDING audit|PENDING source audit|N/A applicability only|N/A no execution planned|PENDING|
| M15 / TinyDejaVu |PENDING audit|PENDING source audit|N/A applicability only|N/A no execution planned|PENDING|
| M16 / ownedprofilepending |PENDING|N/A fullforward|PENDING ownership/profilegate|PENDING|PENDING|

## Template B — domain/bound ablations

| Run/config pair | Same anchor/domain/method? | Geometry/matching criterion | Attempts | Positive numeric | Executable valid | Candidate members | Joint positives+members | Evidence |
|---|---|---|---|---|---|---|---|---|
| H08 direct vs separate |PENDING|Same hiddeninterval|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
| H09 IBP vs CROWN |PENDING|Same domain/objectives/budgetdeclared|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
| H10 rank/random/residual |PENDING|Trainfit + validationsel|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
| H11 diagonal vs wave |PENDING|Declared matching; dimensionsexplicit|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
| H11 native correlated vs hull |PENDING|Same original candidate|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|
| H16/17 fixed vs horizon |PENDING|Same parentbackend/buildbudget|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|

Boundtable kèm per-competitor margins, buildtime và rejected/errorreason; bảng
aggregate không đủ substitute raw trace. Zero-width/singleton, boundary và
unsupportedoperator controls phải identifiable.

## Template C — full operational trace và quality

| Config | N_stream | R_regional | R_exact_input | Q_incremental | Fallbacks | Reference disagreements | A_regional | Supervised acc/F1 | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| pending_config |PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|

Mỗi row có same-recording chronologicalorder và per-recording resets. Separate
strata: purebasic, puretransition, majoritymixed, ambiguous, unlabeled theo frozen
policy; không ép groundtruthmetric cho unknown/unlabeled. Pair-level distribution
và multi-step anchorcoverage là studies riêng.

## Template D — paired runtime, complete path

| Config/repeat | Total operational ms | Query mean/p50/p95 ms | Builds total ms incl failures | Identity/member/state ms | Fallback ms | Core ratio | Wrapper ratio | Setup/load/shadow separately |
|---|---|---|---|---|---|---|---|---|
| pending_config / pending_repeat |PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|

Timing spec phải khai báo included/excluded phases. CPU/device transfers và
candidatepreparation trên proposed operationalpath được tính; outputCSV/analysis
và shadowreference có thể ngoài timer nhưng phải report riêng. Failedbuilds và
duplicateforward có thực thì tính, không hậu xử lý xóa. Setup-only identity check
chỉ cho phép nếu ownership/immutabilitycontract thực sự hỗ trợ replacement của
per-querycheck; W4-024 không được bỏ guards vô điều kiện.

W4-002/033 chốt repetitionorder/warmup/resources và statistic trước đo. Paired
totalratio không thay bằng ratioofmedians. Confidenceinterval nếu dùng phải ghi
samplingunit và dependence; successivewindows không độc lập tự nhiên. Không
cherry-pick bestrepeat hoặc bỏ timeout vào samplemissing.

## Template E — memory và scaling

| Config/dimension/rank/horizon/cachecap | Baseline reference bytes | Additional persistent bytes | Domain/certificate/state bytes | Peak workspace measurement | RSS/allocator delta | Unmeasured / method |
|---|---|---|---|---|---|---|
| pending_config |PENDING|PENDING|PENDING|PENDING|PENDING|PENDING|

Parameterpayload, Pythonreachableobjects, allocatorpeak, RSSprocess và GPUmemory
là measurements khác nhau. Không cộng nestedobjectbytes hai lần; không gọi
wholeprocessRSS cacheoverhead. RAMresidentdataset/setup PCA không tự thuộc runtime
cache. Measuredscaling chỉ áp configuredrange; memorybound theo certificatecount,
rank/horizon và cap phải có policy hếtbộnhớ/fallback.

## Template F — hypothesis outcome và final claim

W4-035 sinh đúng24rows từ [HYPOTHESES.md](HYPOTHESES.md), không copy số bằng tay.

| H_ID | Original claim | Week3 disposition | Week4 configuration/evidence | Result / missing premise | Allowed conclusion | Reviewer |
|---|---|---|---|---|---|---|
| H01…H24, one row each |FROM_LEDGER|FROM_LEDGER|PENDING|PENDING|PENDING|PENDING|

H24 chỉ join **cùng configuration**: whole-domain validity + executable soundness
+ actual useful nonzero regionalreuse + netcostsaving. Singletonexactreplay có thể
là engineeringbenefit nhưng không tự establish nontrivialregionalH24; incremental
computation cũng báo contribution riêng. Heuristicaccuracy không thay formalproof.

## Checklist W4-035/036/040 trước công bố

- [ ] M00…M16 inventory có explicit status; unavailable rows có blocker.
- [ ] M12/M13 parentconfiguration rõ; M15 hai audits riêng, excludedperformance.
- [ ] Every result links rawtrace, run/config/source/environment/protocol identity.
- [ ] Numericalclosed methods không xuất actualformalreuse; diagnostics labeled.
- [ ] Exactreplay, incremental và regional counts/guarantees không bị trộn.
- [ ] Groundtruthmetrics, referenceagreement và no-hitN/A đúng nghĩa.
- [ ] Fullpathcost includes failedbuilds và overhead; baselinecore lẫn wrapper.
- [ ] M16 secondary baseline có reviewed ownership profile; không thay M00/M01.
- [ ] Matchingcriteria và known-testexposure được disclose trước conclusions.
- [ ] Memoryunits/methods và missingmeasurements rõ; no inventedbenchmark.
- [ ] Final hypothesisledger đủ24rows; G2HOLD không bị researchpackage đổi thànhPASS.
