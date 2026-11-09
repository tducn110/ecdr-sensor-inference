# Week 4 — ledger đầy đủ H01–H24 và nghĩa vụ thực thi

Trạng thái gói này: **SPECIFIED; mọi task W4-001…W4-040 còn TODO**. Tài liệu
này giao việc nghiên cứu và implementation; không tạo numerical proof, benchmark
hay certificate mới. `Week 3 disposition` là kết quả đã có, còn `Week 4 target`
là công việc dự kiến. Agent không được sửa nghĩa giả thuyết để làm kết quả đẹp hơn.

Nguồn chuẩn là ledger tại
[Report A Week 3](../../03_MATH/03_Week_3_Mathematical_Foundations.tex),
[independent review](../../output/week3_certification_20261103/independent_review.md),
[execution index](../../output/week3_certification_20261103/README.md) và
[protocol](../../output/week3_certification_20261103/protocol.json).
Task IDs, dependencies và state tương lai lấy từ [tasks.json](tasks.json);
task cards dưới `tasks/W4-<ID>.md` mô tả file ownership và checklist.

## Cách đọc trạng thái và evidence

- `DERIVED`: theorem có proof trong phạm vi premises đã ghi.
- `VERIFIED_FINITE`: đã kiểm một tập input/backend hữu hạn; không thay theorem.
- `CONDITIONAL`: premise quan trọng chưa khép, đặc biệt numerical soundness.
- `FALSIFIED_UNCONDITIONAL`: có counterexample cho universal claim gốc; không
  đồng nghĩa mọi phiên bản có thêm giả thiết đều sai.
- `NOT_ESTABLISHED`: chưa đủ evidence cho claim; không tự kết luận sai.
- `NOT_APPLICABLE`: theorem/method không áp cho actual reference đã đóng băng.
- `ACCEPTED`: chỉ user hoặc reviewer có thẩm quyền gắn; agent không tự gắn.

Research disposition độc lập task state: task có thể nộp một counterexample hoặc
kết quả âm đầy đủ; feature còn thiếu không được tính như đã implement. Không
gom proof của method A, hits của B và timing của C để kết luận H24. Gate numerical
ở W4-010 chặn formal reuse; unresolved numerical premise phải giữ gate đóng.

## Snapshot Week 3 làm điểm xuất phát

Actual reference là `scripts/backbone.py:SmallCNN1D(num_classes=12)`, C=6,
W=128, stride64, sampling50Hz, padded Conv/BN-eval/ReLU/MaxPool và Flatten–MLP.
Checkpoint SHA-256:
`8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
Preprocessing ID:
`aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.

Week 3 có 5,053 full-stream test windows, 3,572 supervised test windows, 5,034
within-recording adjacent pairs và 19 recordings. Radius selected1e-6 cho
5,034/5,053 diagonal và 4,211/5,053 wave **conditional-positive estimates**,
nhưng cả hai có zero next-window membership. Four runtime replays có zero formal
reuse; reused-query agreement là N/A. Float64 IBP estimates chưa là validated
enclosures của executable float32. G2 HOLD, G3 PARTIAL/not passed.

Các mẫu và test set đã được xem trong Week 3. W4-002/W4-030 phải ghi rõ đây là
exploratory follow-up, train-fit/validation-select/test-evaluate; không gọi test
set này wholly untouched confirmation. Mọi số mới phải có receipt Week 4 riêng.

## Inventory và routing đầy đủ

| ID | Nghĩa gốc | Week 3 disposition | Week 4 task chính |
|---|---|---|---|
| H01 | Shift-aligned overlap bằng nhau | DERIVED + VERIFIED_FINITE |001,002,007,015,019,027|
| H02 | 50% overlap làm ordinary distance nhỏ | FALSIFIED_UNCONDITIONAL |002,015,018,027,035|
| H03 | Cùng activity cho cùng CNN prediction | FALSIFIED_UNCONDITIONAL; actual witness |023,027,032,035|
| H04 | Strict winner + continuity cho local invariant region | DERIVED; useful size NOT_ESTABLISHED |005,011,012,027|
| H05 | Positive valid whole-domain margins đủ reuse | DERIVED real; executable CONDITIONAL |005–011,022,023,026,036|
| H06 | Sound Lipschitz bound cho certified ball | Conditional theorem |005–010,012,028,033|
| H07 | IBP certifies mọi invariant domain | FALSIFIED completeness; numerical CONDITIONAL |006,008,010,011,027|
| H08 | Direct margins tránh một số dependency loss | DERIVED với matched hidden box |011,014,018,032,035|
| H09 | CROWN luôn rẻ hơn alternatives | NOT_ESTABLISHED |013,014,028,033,035|
| H10 | Temporal/PCA directions phù hợp margin normals | Descriptive local VERIFIED_FINITE; regional NOT_ESTABLISHED |002,016–018,030,032,035|
| H11 | Structured domain hơn matched diagonal coverage | Measured policy zero useful hits; superiority NOT_ESTABLISHED |015–018,030,032,035|
| H12 | Fixed overlap giảm free coordinates CW→Cs | DERIVED;768→384 |007,015,018,025,034|
| H13 | Exact aligned feature reuse | Historical theorem + separate actual theorem/finite check |019–021,026,032–034|
| H14 | GAP new-minus-dropped identity | DERIVED GAP; actual head NOT_APPLICABLE |019–021,027,035|
| H15 | Recenter after hit không cần proof mới | FALSIFIED_UNCONDITIONAL |022,023,025,026,027|
| H16 | Fixed-anchor membership giữ repeated soundness | Conditional theorem |010,022,023,025,026,032|
| H17 | Product-wave/reachable-set multi-step certificate | Conditional theorem; practical NOT_ESTABLISHED |015,017,022,025,032–034|
| H18 | Exact box membership worst-case Theta(m) | DERIVED trong coordinate-access model |007,016,018,026,033,034|
| H19 | Soundness kéo theo useful coverage | FALSIFIED_UNCONDITIONAL |027,030,032,035,036|
| H20 | Useful coverage kéo theo net speedup | FALSIFIED_UNCONDITIONAL |024,028,030,032–035|
| H21 | ECDR hơn CNN hoặc LipCache trên HAPT | NOT_ESTABLISHED |028–036|
| H22 | Real proof tự transfer sang float32 | FALSIFIED_UNCONDITIONAL; bridge open |005–010,021,026,036|
| H23 | Correct event name đủ reuse | FALSIFIED_UNCONDITIONAL |015,022,023,026,027|
| H24 | Có regime đồng thời sound, useful, low-cost | NOT_ESTABLISHED |010,021,030–040|

Trong cột task, `005–010` là shorthand cho các task IDs W4-005 đến W4-010,
không là lệnh chạy. W4-027 kiểm đủ24record; W4-035 tổng hợp bằng dữ liệu;
W4-036 review độc lập; W4-037…040 chịu trách nhiệm report/delivery.

## H01 — overlap sau dịch chỉ số

**Premises:** cùng recording, sample grid, stride64 và preprocessing pointwise
đóng băng. Phần giữ lại bằng nhau sau dịch index, không tại cùng local index.
Week 2 kiểm17,392pairs toàn corpus; test có5,034pairs.

- [ ] W4-001/002 freeze recording/grid/order contract; ghi xử lý gap và reset.
- [ ] W4-007/015 kiểm raw overlap và normalized overlap, kể cả boundary/NaN.
- [ ] W4-019 chuyển independent actual-feature check thành reproducible source.
- [ ] W4-027 liên kết theorem assumptions với đúng source/evidence.

**Kết quả hợp lệ:** DERIVED theo contract, VERIFIED_FINITE cho traces đã kiểm.
Future prefix không đảm bảo phải reject/rebuild; không suy raw-distance nhỏ.

## H02 — overlap không đảm bảo ordinary distance nhỏ

Không có smoothness/amplitude bound trong window contract. Ramp shift tạo
ordinary-coordinate delta lớn dù overlap50%; universal claim đã bị bác.

- [ ] W4-027 giữ counterexample runnable và kiểm claim bị bác đúng nghĩa gốc.
- [ ] W4-015/018 đo ordinary delta và wave-tail/residual delta trong cùng normalized
  coordinate system; định nghĩa norm, đơn vị, denominator và quantiles.
- [ ] W4-002/035 phân biệt empirical distribution với một universal bound.

**Không giao việc:** tìm proof cho universal claim sai. Thêm smoothness premise
là theorem mới, phải có ID extension riêng và nguồn bảo đảm premise đó.

## H03 — activity annotation không là decision oracle

Actual Week 3 có78prediction changes trên2,661same-pure-label adjacent pairs;
witness global682→683, label11, zero-based predictions10→8 trong
[feature_alignment.json](../../output/week3_certification_20261103/feature_alignment.json).

- [ ] W4-027 preserve/reproduce witness với đúng class-index convention.
- [ ] W4-023 selector chỉ đọc observable input/state; không đọc ground-truth label.
- [ ] W4-032/035 report agreement theo annotation strata như offline analysis.

**Boundary:** same-activity agreement hữu hạn không chứng minh decision invariance
toàn miền. Activity accuracy và reference agreement có target khác nhau.

## H04 — local existence nhờ continuity và strict winner

Finite logits continuous và strict positive point margin đủ cho invariant
neighborhood trong số thực. Graph actual có các operator continuous được khai báo.

- [ ] W4-005/027 ghi tie/strictness và real-valued operator semantics.
- [ ] W4-011/012 radius search báo cả anchor margin và whole-domain bounds.
- [ ] Include tie, near-tie, tiny-radius và nonfinite controls.

**Boundary:** theorem không cho useful radius hoặc hit rate; float32 cần H22.

## H05 — toàn bộ11positive valid margins

Với winner c và mọi j khác c, `inf(x in D)(F_c(x)-F_j(x)) > 0` đủ giữ winner
trên D nếu bounds, membership, model identity và executable semantics hợp lệ.

- [ ] W4-005…010 đóng chain normalization→operator enclosures→float32 error→safe
  acceptance predicate; independent numerical review ở W4-010.
- [ ] W4-011 cung cấp đúng11objectives, bounds và domain/model/preprocess identity.
- [ ] W4-022/023/026 reject mismatched/stale/mutated domains và invalid queries.
- [ ] W4-036 kiểm proof premises có khớp actual accepted runtime trace.

**Acceptance:** gate chỉ mở cho configuration có đủ proof scope. Float64 estimate,
sample tests và point margin không đủ; unresolved bridge giữ CONDITIONAL.

## H06 — Lipschitz-margin ball cho actual graph

Phải có valid per-margin Lipschitz upper bound trên miền chứa ball và đúng norm.
Anchor gradient không phải upper bound trên miền.

- [ ] W4-012 derive actual Conv/BN-eval/pool/Flatten–MLP composition và11L_cj.
- [ ] W4-005…010 bao numerical errors của bounds và ball membership.
- [ ] W4-028/033 đo radius, build time, membership, conditional positives và reuse.

**Boundary:** historical `ecdr/certified_adaptive.py` thuộc model khác; không lấy
constants hoặc K=6/GAP timing làm actualK12comparator.

## H07 — IBP incomplete

Counterexample: `h1=h2=ReLU(t+1)`, t∈[-1,1], margin=`1+h1-h2=1`; independent
hidden intervals[0,2] cho direct final-head IBP lowerbound−1.

- [ ] W4-027 preserve counterexample; không đổi tên inconclusive thành violated.
- [ ] W4-006/008 kiểm signed padding, BN negative gamma, ReLU, pool, flatten, linear.
- [ ] W4-010/011 tách real transfer validity, numerical enclosure và completeness.

**Boundary:** fail-to-certify không là counterexample to classifier invariance;
direct final-head objectives không phục hồi hidden dependencies đã mất.

## H08 — direct margin versus separate logits

Với cùng final hidden interval, combine `W2[c]-W2[j]` trước intervalization
không kém tightness của `lower(logit_c)-upper(logit_j)` trong real affine setting.

- [ ] W4-011 paired outputs hai cách trên cùng hidden box, domain, winner và dtype.
- [ ] W4-014/018 giữ method/domain/budget matching khi đổi backend.
- [ ] W4-032/035 report per-margin gain, positives, time và memory; giữ zero-gain cases.

**Boundary:** strict improvement mọi input không được theorem bảo đảm; executable
ordering/rounding phải bao riêng. Không so unmatched domains rồi gán gain cho H08.

## H09 — CROWN cost/tightness phải đo

Week 3 native1D probe gặp MaxPool conversion failure. Singleton-height2D probe
chạy được ở2anchors; finite conversion agreement không là universal proof.
Installed auto-LiRPA0.8.0 metadata không bao frozen torch2.14.1.

- [ ] W4-013 dùng isolated supported environment; không mutate frozen `.venv`.
- [ ] W4-014 khai báo conversion proof, operators, slopes, optimization/time budgets.
- [ ] W4-028/033 paired timing/domain/objectives; setup và query costs tách rõ.

**Boundary:** không mặc định CROWN rẻ hơn hoặc arbitrary CROWN relaxation luôn
tighter IBP. Timeout/unsupported/error là result cần giữ, không bỏ khỏi denominator.

## H10 — temporal/PCA geometry

Week 3 train-fit PCA9944deltas và32validation gradient observations là descriptive
local geometry. Chúng không chứng minh whole-region alignment hoặc future reuse.

- [ ] W4-002/016 freeze centering, basis/rank, normalized units và bounded residual.
- [ ] W4-017 backend giữ correlation hoặc ghi explicit enclosing-box limitation.
- [ ] W4-018 random-basis/rank/residual ablations; winner/competitor rule explicit.
- [ ] W4-030 train-fit/validation-select freeze trước Week4test replay.
- [ ] W4-032/035 kết nối local projections, nonlinear margins, memberships và cost.

**Boundary:** residual bị bỏ làm candidate không còn bao admitted input. PCA
projection membership alone không kiểm full-space domain.

## H11 — geometry comparison phải matched

Week 3 measured policies không có useful candidate hits; không chứng minh hoặc
bác structured superiority tổng quát. Fixed-prefix wave là degenerate diagonal box.

- [ ] W4-015…018 predeclare matching: same center/constraints, candidate coverage,
  radius, budget hoặc dimension; không coi các tiêu chí này tương đương.
- [ ] Include correlated-native versus its coordinate-hull control.
- [ ] W4-030 freeze matching/selection; W4-032/035 report all joint counts.

**Boundary:** IBP chỉ nhận coordinate ranges không khai thác correlation nội tại.
Nhỏ volume/dimension có thể dễ bound hơn mà không chứng minh intrinsically better geometry.

## H12 — 768full versus384tail variables

C=6,W=128,s=64 cho CW=768,Cs=384 nếu prefix fixed exactly và wave embedding
injective. K=12 là số output classes, không đổi C.

- [ ] W4-007/015 immutable prefix/tail embedding; full membership contract.
- [ ] W4-018/025 noisy-prefix, gap và horizon boundary checks.
- [ ] W4-034 report dimension, coordinates checked, workspaces và actual scaling.

**Boundary:** trusted prefix phải có mechanism; nếu vẫn đọc/check prefix thì cost
được tính. Dimensional reduction không suy50%runtime saving.

## H13 — actual exact intermediate reuse

Historical A1–A6 valid–ReLU theorem giữ nguyên. Actual padded/BN/pool theorem có
RF`[4j−8,4j+11]`, width20,jump4; new j=2…13 bằng old j+16. Week 3 finite check
5,034pairs ×12positions ×64channels =3,866,112scalars bitwise equal.

- [ ] W4-019 derive every layer index/padding/pool phase; fixed BN eval state.
- [ ] W4-020 recompute boundary/new dependencies và assemble đúng feature order.
- [ ] W4-021 test layer tensors, reset/gap/pool alignment/backend; define executable
  equivalence target: bitwise logits, proved decision equivalence hoặc finite-only.
- [ ] W4-032…034 measure complete incremental computation, state và memory.

**Boundary:**12shared positions không cho unchanged classifier output; backend-specific
finite equality không là universal arithmetic proof.

## H14 — current Flatten–MLP cần derivation mới

GAP new-minus-dropped sum chỉ áp GAP rồi affine logits. Reference dùng
Flatten2048→Linear128→ReLU→Linear12, position-sensitive.

- [ ] W4-019 derive `a=W1*vec(H)+b1`, rồi current nonlinear head đúng thứ tự.
- [ ] W4-020 không dùng previous-position weight cho feature chuyển position.
- [ ] W4-021 kiểm changed reduction-order errors; W4-027 preserve GAP premises.
- [ ] W4-035 ghi actual extension riêng, không rename theorem gốc.

**Boundary:** real algebra delta equivalence không tự bitwise float equivalence.
Full head recomputation có thể là phương án đúng khi delta arithmetic chưa sound.

## H15 — cấm certificate drift

Counterexample affine F1=1−x,F2=0, radius0.6: hit0.6 rồi tự recenter đến1.2 có
thể trả prediction sai. Reused query không tự là trusted anchor mới.

- [ ] W4-022/023 immutable original anchor và certificate domain.
- [ ] W4-025 explicit rebuild/transfer receipt nếu đổi anchor/horizon.
- [ ] W4-026/027 negative tests auto-recenter, expiry refresh, domain mutation.

**Boundary:** hit không refresh age, radius hoặc numerical proof.

## H16 — repeated fixed-anchor membership

Mỗi accepted query cần same valid certificate và sound membership. Domain/model/
preprocess/anchor identity và numerical bridge phải còn đúng tại lần accept đó.

- [ ] W4-022 lifecycle invalidate recording/gap/expiry/model/state changes.
- [ ] W4-023 check every query; W4-026 adversarial transition tests.
- [ ] W4-032 report lifetime, consecutive hits, hits/certificate và invalidation reasons.

**Boundary:** theorem không bảo đảm useful lifetime; zero-hit trace không verify
nontrivial accepted branch. Unresolved W4-010 giữ formal gate closed.

## H17 — finite horizon reachable sets

Certificate phải bao mọi admitted wave sequence và reachable state tại mọi
horizon. Sau hai stride64 có thể không còn raw sample của original window.

- [ ] W4-025 explicit finite horizons, reachable-set construction và dependencies.
- [ ] W4-015/017 bound transformed domains và every required margin/horizon.
- [ ] W4-022 reject outside horizon; W4-032…034 report cost/tightness/memory by horizon.

**Boundary:** không extrapolate one-step certificate thành multi-step proof.
Nếu unsupported, disable horizon method và nộp blocker thay fabricated hits.

## H18 — exact membership complexity

Worst-case Theta(m) dùng coordinate-access model và nondegenerate box, exact
decisions cho mọi input. Average early rejection và SIMD là câu hỏi đo riêng.

- [ ] W4-007 inclusive/exclusive boundary, errors, NaN/overflow và prefix checks.
- [ ] W4-016/018 define sound correlated-domain membership, residual constraints.
- [ ] W4-026 adversarial unread/near-boundary coordinates.
- [ ] W4-033/034 measure latency, coordinates touched, memory/scaling.

**Boundary:** all-reject filter không là exact membership; projection-only filter
có thể reject soundly nhưng acceptance cần full sufficient containment proof.

## H19 — sound region vẫn có thể zero useful hits

Tiny box/singleton chỉ chứa anchor là counterexample cho soundness⇒coverage.
Week 3 có positive numeric estimates nhưng zero next-membership.

- [ ] W4-027 preserve no-hit control.
- [ ] W4-030/032/035 report attempts, positives, members, positive-and-member joint,
  executable certificates, accepted reuse và all-stream denominator riêng.
- [ ] W4-036 zero accepted hits ⇒ agreementN/A; không nhân independent rates.

**Boundary:** positive bound counts không là formal certificate counts hoặc reuse.

## H20 — coverage không đủ speedup

Construction, selection, membership, identity, normalization, state và fallback
đều có cost. Build cost lớn là counterexample cho coverage⇒speedup.

- [ ] W4-024 tối ưu có trace ownership; tránh duplicate anchor forward nếu contract
  giữ được; không bỏ guards chỉ để lấy timing đẹp.
- [ ] W4-028 baseline controls; W4-030 freeze build policy.
- [ ] W4-032/033 time all paths, failed builds, no-hit intervals và amortization.
- [ ] W4-035 derive ratios từ paired totals; mean/p50/p95 không hoán đổi.

**Boundary:** offline shadow inference đo riêng; oracle hindsight không là runtime
policy. Report core baseline và equivalent-wrapper baseline cùng nhau.

## H21 — HAPT superiority cần comparable semantics

Week 3 chưa có faithful HAPT LipCache comparator. GuardNet own-decision guarantee
không mặc nhiên equal MainNet reference. CNN comparator phải actualK12.

- [ ] W4-028 adapter kiểm frozen identities, full input stream và target guarantee.
- [ ] W4-029 đọc primary source/version/artifact; ghi guarantee, training/data needs,
  architecture assumptions, numerical scope và faithful adaptation decision.
- [ ] W4-029 chỉ audit applicability StreamiNNC/TinyDejaVu; không tự gán results
  hoặc benchmark transferred từ paper sang HAPT.
- [ ] W4-030…036 compare executed compatible configurations; expose blockers.

**Boundary:** different-model/guarantee methods cần separate study; không đổi frozen
model hoặc retrain ngoài scope để tạo advantage. See [COMPARISON.md](COMPARISON.md).

## H22 — numerical bridge là critical dependency

Float32 không tự kế thừa real proof. Float64 diagnostic và final`nextafter`
không bao accumulated normalization/operator/reduction/membership errors.

- [ ] W4-005 define stored constants, backend/runtime contract, exact target.
- [ ] W4-006 outward primitives; W4-007 normalization/membership containment.
- [ ] W4-008 every actual operator, BN evaluation, pooling, errors and nonfinite paths.
- [ ] W4-009 bridge real margin to float32 decision with safe acceptance predicate.
- [ ] W4-010 independent derivation review; W4-021 incremental arithmetic scope;
  W4-026 adversarial controls; W4-036 trace-to-proof consistency.

**Boundary:** random/high-precision tests hỗ trợ implementation nhưng không thay
analytic error enclosure. Arbitrary epsilon hoặc passing tests không mở formal gate.

## H23 — selector chỉ propose

Event label không là whole-domain bound hay membership. Ground truth dùng offline
stratification, không dùng runtime accept oracle.

- [ ] W4-015 selector output candidate metadata, chưa accepted certificate.
- [ ] W4-022/023 mọi candidate đi qua same numerical/identity/margin/membership gate.
- [ ] W4-026 wrong event/right domain; right event/wrong domain; corrupted radius,
  center, recording và missing state tests.
- [ ] W4-027 preserve selector proposition và counterexample.

**Boundary:** tên event “stable” không có certification authority.

## H24 — joint scientific outcome

Một cùng configuration phải đồng thời có valid real proof, executable numerical
soundness, nonzero useful formal reuse và positive net savings theo protocol.
Week 3 measured policies chưa đạt; kết quả âm không bác mọi possible regime.

- [ ] W4-030 predeclare useful/cost criteria trước Week4test evaluation.
- [ ] W4-010/021/031 numerical/equivalence/integration gate scopes explicit.
- [ ] W4-032…036 join proof receipt, actual accepted trace và paired operational timing.
- [ ] W4-037…040 reports, reproducibility và gate recommendation có exact scope.

**Kết luận cho phép:** ESTABLISHED_IN_MEASURED_SCOPE nếu đủ mọi premise cùng config;
hoặc NOT_ESTABLISHED với blocker/kết quả âm cụ thể. Không ghi universal impossibility,
production ready, G2PASS hoặc ACCEPTED từ scoped Week4experiments.

## Checklist review ledger cuối tuần

- [ ] Có đúng24original IDs, không rename/drop counterexamples.
- [ ] Mỗi ID có meaning, premises, Week3baseline, Week4task/evidence, disposition.
- [ ] Mọi extension actual graph/finite horizon có nhãn riêng và giữ theorem gốc.
- [ ] Không claim numerical soundness từ zero disagreement/no-hit execution.
- [ ] Mọi metric truy ngược configuration/stream/revision/backend và denominator.
- [ ] Falsified, inconclusive, unsupported, unimplemented và not applicable khác nhau.
- [ ] Chưa hoàn thành obligation phải ghi missing evidence và dependency chặn.
- [ ] Agent/reviewer không tự gắn ACCEPTED; final report giữ G2quality limitation.
