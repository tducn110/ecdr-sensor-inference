# Week 4 — đặc tả từng file cho implementation

Trạng thái: **SPECIFICATION ONLY**, ngày 2026-11-03. Các API, CLI và artifact
dưới đây là đầu ra dự kiến của W4-001…W4-040; việc xuất hiện trong tài liệu này
không có nghĩa file đã implement, test đã chạy hoặc numerical gate đã mở.
Task ID, dependencies và trạng thái do [tasks.json](tasks.json) quản lý. Đọc
[GOAL.md](GOAL.md), [ORCHESTRATOR.md](ORCHESTRATOR.md),
[HYPOTHESES.md](HYPOTHESES.md) và [COMPARISON.md](COMPARISON.md) cùng đặc tả này.

## 1. Boundary và quy tắc dùng file

- Code mới nằm trong `ecdr/week4/`; các hàm selector đặt trong `domains.py`,
  policy đặt trong `runtime.py`. Giữ đúng 11 module trách nhiệm ở phần 2;
  không tạo thêm một implementation song song ngoài package để né review.
- Scripts mới dùng các thư mục `scripts/`, `tests/`, `configs/` đang có.
  Design/proofs nằm ở `04_SYNTHESIS/week4/design/`; reports ở `reports/`;
  execution evidence nằm ở `output/week4/<run_id>/`.
- W4-001 chụp Git HEAD/branch/dirty state, frozen file hashes và environment
  thực tế trước mutation. SHA ghi trong plan là planning base, không thay
  execution snapshot của agent tương lai.
- Giữ nguyên checkpoint, train-fit normalization, K=12 class order,
  labels/splits, Week 1–3 sources và receipts, historical K=6/GAP modules và
  `.venv` hiện tại. Không sửa `ecdr/week3_verifier.py` hoặc regenerate output
  Week 3 để làm cho kết quả Week 4 đẹp hơn.
- Một file có nhiều task owner theo section vẫn chỉ có một người giữ write
  lock tại một thời điểm. Agent ghi rõ symbols/sections mình sửa trong receipt;
  task sau đọc implementation đã review của task trước.
- Mỗi run có manifest định danh source/config/data/reference/environment.
  Artifact cũ không được chép sang run mới rồi coi là vừa thực thi.
- `VERIFIED_LIMITED` phải chỉ rõ phần đã verify và feature chưa implement.
  Numerical bridge không hỗ trợ, zero reuse, negative savings hoặc comparator
  thiếu capability là kết quả nghiên cứu có thể báo; chúng không là formal PASS.

## 2. Catalog code và ownership

| File dự kiến | Responsibility duy nhất | Task chính | Test owner |
|---|---|---|---|
| `ecdr/week4/__init__.py` | Package marker, version metadata và public exports | W4-003 | `tests/test_week4_contracts.py` |
| `ecdr/week4/types.py` | Immutable exchange types, status vocabularies, serialization contracts | W4-003 | `tests/test_week4_contracts.py` |
| `ecdr/week4/reference.py` | Frozen K=12 reference owner, environment identity và trusted inference receipts | W4-003, W4-007, W4-024 | `tests/test_week4_contracts.py`, `tests/test_week4_membership.py`, `tests/test_week4_runtime_cost.py` |
| `ecdr/week4/numerics.py` | Outward arithmetic, normalization enclosure và executable capability profiles | W4-005…W4-010 | `tests/test_week4_numerics.py` |
| `ecdr/week4/domains.py` | Stored domains, causal selectors, train-fit geometry, membership | W4-007, W4-015, W4-016, W4-018 | `tests/test_week4_domains.py` |
| `ecdr/week4/bounds.py` | Actual-graph IBP/Lipschitz bounds và shared bound result validation | W4-011, W4-012, W4-017 | `tests/test_week4_bounds.py`, `tests/test_week4_lipschitz.py`, `tests/test_week4_correlated_bounds.py` |
| `ecdr/week4/lirpa_adapter.py` | Supported graph conversion, isolated library capability và CROWN | W4-014, W4-017 | `tests/test_week4_lirpa.py` |
| `ecdr/week4/incremental.py` | Actual padded Conv/BN/Pool feature cache, full Flatten–MLP head | W4-019…W4-021 | `tests/test_week4_incremental.py` |
| `ecdr/week4/certificates.py` | Trusted certificate issuance, domain/reference/capability binding | W4-010, W4-022 | `tests/test_week4_certificates.py` |
| `ecdr/week4/runtime.py` | Recording lifecycle, causal build policy, fallback, cost instrumentation | W4-022…W4-026 | `tests/test_week4_runtime.py` |
| `ecdr/week4/methods.py` | Method registry, baselines và comparator semantic capability | W4-028, W4-029 | `tests/test_week4_methods.py` |
| `ecdr/week4/metrics.py` | Trace accounting, agreement/quality metrics và paired statistics | W4-032…W4-035 | `tests/test_week4_metrics.py` |

### 2.0 Test suite catalog

Mọi unit/integration/adversarial test phải có target file, runner và task sở hữu rõ ràng:

| Test file | Phạm vi và assertions chính | Task | Code module được test |
|---|---|---|---|
| `tests/test_week4_provenance.py` | Kiểm tra Git tree hash, class order, checkpoint SHA, protected manifest | W4-001 | `scripts/week4_audit.py` |
| `tests/test_week4_contracts.py` | Immutable exchange types, serialize/deserialize, QueryView | W4-003 | `ecdr/week4/types.py`, `reference.py` |
| `tests/test_week4_task_claims.py` | Lock tranh chấp, task state transition, atomic claim | W4-004 | `scripts/week4_tasks.py` |
| `tests/test_week4_numerics.py` | Outward rounding arithmetic, floating point error bounds | W4-006 | `ecdr/week4/numerics.py` |
| `tests/test_week4_membership.py` | Normalized domain membership, inclusion soundness | W4-007 | `ecdr/week4/domains.py`, `reference.py` |
| `tests/test_week4_operators.py` | Layer-wise interval propagation: Conv1D, BN, ReLU, Pool, Flatten, Linear | W4-008 | `ecdr/week4/numerics.py` |
| `tests/test_week4_numerical_bridge.py` | Float32 e2e sound margin enclosure, floating error compensation | W4-009 | `ecdr/week4/numerics.py` |
| `tests/test_week4_numerical_gate.py` | Capability check, non-bypass gate, fail-closed khi thiếu bridge | W4-010 | `ecdr/week4/certificates.py`, `numerics.py` |
| `tests/test_week4_bounds.py` | IBP actual-graph all-11 direct vs separate logit margins | W4-011 | `ecdr/week4/bounds.py` |
| `tests/test_week4_lipschitz.py` | Global Lipschitz constant computation, ball-domain margin bound | W4-012 | `ecdr/week4/bounds.py` |
| `tests/test_week4_lirpa.py` | auto_LiRPA graph conversion, CROWN bounding trên actual graph | W4-014 | `ecdr/week4/lirpa_adapter.py` |
| `tests/test_week4_domains.py` | Diagonal và wave proposal domain geometry, prefix contract | W4-015 | `ecdr/week4/domains.py` |
| `tests/test_week4_affine_domains.py` | PCA / affine-tail domain projection, bounded residual check | W4-016 | `ecdr/week4/domains.py` |
| `tests/test_week4_correlated_bounds.py` | Correlated bounds propagation, comparison với coordinate box hull | W4-017 | `ecdr/week4/bounds.py`, `lirpa_adapter.py` |
| `tests/test_week4_protocol.py` | Runner phases, synthetic stream replay, split sealing, protocol freeze | W4-018, W4-030 | `scripts/week4_run.py` |
| `tests/test_week4_alignment.py` | Receptive field [4j-8, 4j+11], exact bitwise feature alignment | W4-019 | `ecdr/week4/incremental.py` |
| `tests/test_week4_incremental.py` | Feature cache maintenance, full Flatten-MLP forward equivalence | W4-020 | `ecdr/week4/incremental.py` |
| `tests/test_week4_incremental_gate.py` | Executable equivalence gate, negative-path fallback check | W4-021 | `ecdr/week4/incremental.py` |
| `tests/test_week4_certificates.py` | Certificate store, immutable anchor, expiration / invalidation | W4-022 | `ecdr/week4/certificates.py` |
| `tests/test_week4_runtime.py` | Recording lifecycle, single forward anchor, trusted fallback path | W4-023 | `ecdr/week4/runtime.py` |
| `tests/test_week4_runtime_cost.py` | Timers, zero redundant forward, identity check amortization | W4-024 | `ecdr/week4/runtime.py`, `reference.py` |
| `tests/test_week4_horizon.py` | Finite-horizon reachable set propagation, no recenter policy | W4-025 | `ecdr/week4/domains.py`, `runtime.py` |
| `tests/test_week4_adversarial.py` | Chained adversarial traces: mutated model, wrong split, NaN input, invalid cert | W4-026 | `ecdr/week4/runtime.py` |
| `tests/test_week4_methods.py` | MethodRegistry discovery, unsupported capability graceful fallback | W4-028 | `ecdr/week4/methods.py` |
| `tests/test_week4_accounting.py` | Denominator accuracy, separate reuse vs fallback error counting | W4-032, W4-033 | `ecdr/week4/metrics.py`, `scripts/week4_run.py` |
| `tests/test_week4_memory.py` | Memory tracker, persistent domain bytes vs transient workspace bytes | W4-034 | `ecdr/week4/metrics.py` |
| `tests/test_week4_metrics.py` | Statistics aggregation, macro-F1, paired timing t-test | W4-035 | `ecdr/week4/metrics.py` |
| `tests/test_week4_delivery.py` | Manifest check, PDF compilation QA, evidence receipt integrity | W4-040 | `scripts/week4_verify_delivery.py` |

### 2.1 `ecdr/week4/types.py`

Viết shared types trước implementation, có `schema_version` và serializer
deterministic. Không import `runtime.py` hoặc script runner từ module này.

Các symbol cần định nghĩa:

- `QueryView`: `global_row_id`, `recording_id`, `step_idx`, raw window owned hoặc
  read-only theo contract. Shape `(6,128)`, finite real input; không có label,
  purity, activity name, next window, future prediction hoặc split-wide logits.
  Runtime nhận query từng bước; metadata label chỉ join sau decisions.
- `ReferenceIdentity`: checkpoint SHA, preprocessing ID, 12 class names theo
  thứ tự, graph/source/state ID, dtype/device, executable profile, dependency
  versions và identity-policy version. Stored artifact identity tách với
  process/generation identity.
- `TrustedAnchorReceipt`: query/raw/normalized hashes, reference identity, issuer
  generation, 12 finite logits, winner/tie policy và timing. Receipt do
  reference owner phát hành; tự tạo một dataclass cùng fields chưa đủ quyền
  làm trusted anchor.
- `MembershipResult`: domain ID, `member`, `status`, rejection reason, method
  version, elapsed time; phân biệt exact stored-endpoint check, validated solve,
  conditional solve và unsupported.
- `BoundResult`: target semantics/profile, reference/domain/method IDs, target
  class, đúng 11 competitor IDs, direct lower margins, optional diagnostics,
  numerical capability ID, finite/enclosure/positive statuses, failure reason,
  component timing. Non-finite result là failure, không tự clip thành số hữu hạn.
- `Certificate`, `RuntimeDecision`, `NumericalCapability` theo contract của các
  module owner bên dưới. `MethodSpec`, `RunIdentity`, `CapabilityResult` có thể
  được định nghĩa tại module owner; nếu đưa vào shared types, phải claim/review
  explicit public-schema delta, không tự sửa shared file ngoài write set.

Không gộp attempted build, finite diagnostic, conditional positive, validated
numerical enclosure, candidate membership, validated membership, potential hit,
accepted certified reuse, exact replay và incremental feature reuse vào một
boolean `valid`. Dùng status vocabularies được kiểm tra, reject unknown enum.
Thêm explicit semantics `REAL_GRAPH`, `DECLARED_IEEE_INTERPRETER`,
`FROZEN_PYTORCH_EXECUTABLE`; không suy ra chúng tương đương từ cùng weights.

Test: malformed shape/type, boolean thay integer, negative step, NaN/Inf,
competitor duplication/omission, wrong class count, serialization round trip,
stable hashes, alias mutation, forged status và unknown schema version.

### 2.2 `ecdr/week4/reference.py`

Owner duy nhất của executable model và preprocessing. Có thể reuse read-only
loading logic Week 3 nhưng không edit historical module hoặc import model GAP.

Các symbol dự kiến:

- `ReferenceOwner.load(reference_dir, execution_profile, identity_policy)`:
  verify frozen checkpoint/preprocessing metadata trước deserialize/instantiate;
  explicit `SmallCNN1D(num_classes=12)`, 283,692 parameters; eval BN running
  statistics, CPU float32 theo profile. Không dùng default K=6 loader.
- `snapshot_identity()` và `assert_identity()`: graph, weights, BN buffers,
  classes, mean/std, eval mode, forward/pre/global hooks, forward overrides,
  actual dtype/device và backend profile.
- `normalize(raw)`: tái hiện chính xác frozen NumPy float32 subtraction/division
  sequence; shape/conversion/normalization overflow reject trước state mutation.
- `infer(query) -> TrustedAnchorReceipt`: một complete trusted reference evaluation.
- `validate_receipt(receipt, query, generation)`: reference owner kiểm provenance,
  query binding và generation; không tin caller-provided logits/timing/status.
- `invalidate(reason)` và owned reference lifecycle nếu W4-024 tối ưu identity.

Ghi execution profile: Python/NumPy/PyTorch/build, CPU/device, torch threads,
batch=1/layout, deterministic flags, MKLDNN/backend selection, applicable
precision/denormal flags và các biến ảnh hưởng semantics có thể thu được.
Không claim đã ràng buộc yếu tố mà environment không thể inspect hoặc verify.

Giữ hai mode benchmark rõ ràng:

1. Strong identity mode rehash state/graph tại declared boundary, chống mutation
   mà checks thực tế bao phủ.
2. Owned-reference mode startup audit + generation checks, chỉ hợp lệ trong
   threat model owner không xuất mutable model/parameter aliases và mọi
   authorized mutation phải invalidate generation. Cheap generation check
   không tự chứng minh chống arbitrary shared-process mutation.

Test: từng identity field mutation; training BN; hooks; graph override;
preprocessing bytes; receipt replay wrong query/generation; caller aliases;
inference failure; restored/changed backend profile. Test raw inference call
count trên build-miss path thuộc runtime integration để ngăn duplicate forward.

### 2.3 `ecdr/week4/numerics.py`

W4-005 phải viết numerical contract trước. Hỗ trợ op nào phải nói rõ arithmetic,
rounding model, input range, overflow/subnormal/FMA/reduction assumptions và
target semantics. Không hardcode epsilon hoặc một `nextafter` cuối làm proof.

Các phần cần viết theo thứ tự:

1. `Interval`/`Enclosure` với finite ordered endpoints và status.
2. Outward primitives `add`, `sub`, `mul`, `div`, `sqrt` hoặc equivalent exact
   arithmetic backend; target precision/rounding explicit. Reject unsupported
   operations/ranges. Preserve enclosure ở từng operation, không chỉ output.
3. `enclose_normalization(raw_domain, frozen_mean, frozen_std, profile)`:
   enclose input conversion và frozen float32 subtract/divide sequence. Trường
   hợp domain định nghĩa trực tiếp bằng stored normalized float32 endpoints
   phải nói rõ membership dùng đúng normalized executable input nào.
4. `enclose_operator(...)` cho signed padded Conv1d, BN eval, ReLU, MaxPool1d,
   Flatten, Linear theo từng supported target. Không thay BN kernel bằng folded
   affine constants rồi quên error của folding/executable BN evaluation.
5. `check_float32_bridge(profile, proof_artifacts)`: bridge riêng từ real/scalar
   interpreter đến frozen backend. Bound cho interpreter mới không làm nó trở
   thành bound của PyTorch chỉ vì sampled logits khớp.
6. `NumericalCapabilityRegistry` và guarded loader dùng bởi certificate issuer.

Numerical gate không được là `is_sound=True`, environment variable, method name,
JSON field hoặc receipt string do implementation author tự đặt. Capability
phải ràng buộc exact supported profile, target, source hashes, proof/validation
artifact hashes và kết quả independent review W4-010. Loader kiểm integrity,
schema, profile match và review provenance theo orchestrator policy. Constructor
không cho user receipt tự nâng conditional thành validated. Hash chỉ xác định
artifact; nó không phải numerical proof hoặc quyền tự cấp `VALIDATED`.
Python privacy không phải security boundary chống arbitrary process tampering;
threat model và reviewer/issuer ownership phải ghi rõ.

Nếu không chứng minh được bridge cho frozen backend: trả
`UNSUPPORTED_NUMERICAL_BRIDGE`, giữ formal issuance/reuse tắt, vẫn chạy conditional
geometry và cost studies. Không đổi frozen backend hoặc checkpoint âm thầm để
làm bridge dễ hơn. Interpreter mới nếu có là secondary target có identity riêng.

Tests dùng tiny dyadic/rational/exact cases, cancellation, negative gamma,
large finite operands, signed zero, near underflow/overflow, tie và tiny margin.
Exhaustive tiny-domain checks và sampled containment phát hiện bug; tài liệu
không gọi chúng là universal proof. Tests forged review receipt, stale source
hash, wrong profile/version, absent bridge và partial operator coverage phải
chặn issuer. W4-010 review độc lập cả proof premise lẫn code path.

### 2.4 `ecdr/week4/domains.py`

Domain là stored mathematical/executable set có identity; selector là thuật
toán causal tạo set. Không coi predictor của future center là certificate.

Symbols/trách nhiệm:

- `CoordinateBox`: immutable finite lower/upper `(6,128)` endpoints; exact
  stored-endpoint comparison không tolerance; include boundary; radius=0 hợp lệ.
- `FixedOverlapBox`: prefix mới bằng trusted anchor suffix, tail 384 variables;
  constant continuation baseline `[old suffix, old suffix]`; normalized-coordinate
  contract phải giữ đúng frozen preprocessing. Đây vẫn là coordinate box.
- `AffineTailDomain`: prefix cố định, tail `c + B diag(a) u + diag(e) v`, latent
  bounds stored, basis/center/residual dimensions validated. Ghi basis rank,
  coordinate units và free-subspace dimension; không claim PCA axes là certified
  directions chỉ vì local gradient projection nhỏ.
- `DomainSelector`, `fit_geometry(train_queries, config)`,
  `select_candidate(trusted_history, fitted_artifact, config)`; các selector nằm
  trong module này để giữ một source of truth. Train-fit PCA có centering và
  normalized coordinate convention rõ; validation choose budgets/predictor.
- `contains(domain, normalized_query) -> MembershipResult` và
  `outer_box(domain)`. Affine membership cần exact/validated latent feasibility
  hoặc safe rejection. Least-squares + arbitrary tolerance chỉ conditional;
  ambiguous result reject trên strict path.

`outer_box()` phải đánh dấu correlation đã bị relax. CROWN/affine backend cần
nhận affine map/latent constraints để claim retained correlation; không chạy
IBP trên outer box rồi rename thành correlated verifier.

Fit artifacts chứa training global row IDs/split/source/config hashes, mean,
scales, basis và residual model; không chứa test-fit transform. Policy không
nhận label hoặc actual next suffix. Oracle-next domain chỉ được là offline
upper-bound diagnostic có tên và cột riêng, không vào runtime method registry.

Tests: shape/broadcast/rank/finiteness, prefix equality, radius0/boundaries,
alias immutability, serialization/hash, shifted overlap, rotated toy domains,
point inside outer box nhưng ngoài affine set, ambiguous solve, no future/label
access. So sánh controls thuộc W4-018: same-center wave slice, affine vs its
outer box, matched validation coverage; không so ambient 768D volume với
384D wave volume rồi kết luận hình học tốt hơn.

### 2.5 `ecdr/week4/bounds.py`

Implement actual K=12 graph operator order. `ActualGraphIBP` là Week4 class mới
hoặc equivalent symbol, không sửa class Week3 và không use valid/GAP formulas.

- `build_from_receipt(reference_owner, trusted_receipt, domain, method_profile)`:
  validate receipt/domain/reference identity trước build; không reference-forward
  lại anchor nếu owner đã cung cấp receipt hợp lệ cùng query/generation.
- `propagate_intervals(...)`: supported op registry, sign-aware weights/BN,
  padding/pool/flatten semantics, full head, exact 11 competitors.
- `direct_margin_bounds(...)`: final class-difference specifications; giữ separate
  logit bound làm diagnostic comparator. Tính exact-real direct weights nếu target
  yêu cầu, hoặc include arithmetic error của subtraction/propagation.
- `regional_lipschitz_bound(...)`: norm/domain và sound upper-bound method explicit;
  complete graph including BN, pool, Flatten and head. Local gradient không
  được dùng làm regional Lipschitz certificate.
- `validate_bound_result(...)`: all11 finite lower margins, matching target/domain,
  capability/profile, method and source hashes; nonpositive nghĩa inconclusive.

Conditional float64 backend vẫn hữu ích nhưng status phải `CONDITIONAL` và không
đi vào formal issuer. Bound cho box không tự chứng minh affine-set tightness;
correlation-aware delegation phải đi qua adapter/specified representation.

Tests: signed/padded Conv, negative BN gamma, tracked stats, pooling edge/tie,
Flatten order, all classes, direct-vs-separate trên matched hidden intervals,
dependency-cancellation counterexample, unsupported op/profile, snapshot mutation,
non-finite result, radius0 và strict positive-vs-tie branch. Tightness/performance
comparison không được thay universal CROWN>=IBP theorem.

### 2.6 `ecdr/week4/lirpa_adapter.py`

Chỉ chạy trong environment mới do W4-013 khóa. Frozen `.venv` không bị resolve
lại. `LibraryCapability` phải kiểm actual import, graph conversion và bound
execution; package metadata/import success không thay op coverage.

- `probe_library(profile)` ghi package/version/requirements/torch/platform,
  operator support, conversion status, error/traceback và timing.
- `convert_supported_graph(model, transform_profile)` hỗ trợ actual SmallCNN1D
  hoặc reject. Native 1D MaxPool capability kiểm bằng execution.
- Singleton-height transform nếu dùng phải map kernel/stride/padding/dilation/
  groups/bias và pool options đúng, hoặc reject những thuộc tính chưa implement.
  BN running state/eps/affine và Flatten order giữ đúng. Constructor Week3 chỉ
  cho actual profile; không quảng bá thành generic converter.
- `compute_margin_bounds(domain, receipt, method_config)` nhận all11 direct
  specifications và retained latent affine map nếu method hỗ trợ; result gắn
  surrogate/reference identities và equivalence status riêng.

Real algebra equivalence, finite sampled logits equality, declared-backend
equivalence và numerical containment là bốn obligations khác nhau. Chưa đóng
bridge nào thì ghi đúng bridge đó thiếu; không copy `VALIDATED` từ library name.

Tests: actual graph conversion success/failure path, explicit unsupported pool,
wrong groups/dilation/padding rejection, negative BN gamma, 2D flatten mapping,
version out-of-contract, method unavailable, timeout/build cap, corrupt result,
all11 competitors và outer-box vs retained correlation status. W4-014 so CROWN
với IBP trên cùng domain/winner/method configuration và ghi build latency/memory.

### 2.7 `ecdr/week4/incremental.py`

Đây là actual padded Conv/BN(eval)/Pool/Flatten–MLP branch mới. Không import
`ecdr.incremental.IncrementalEngine` như một proof của current reference.

Symbols dự kiến: `IncrementalState`, `IncrementalResult`,
`ActualK12IncrementalEngine.reset()`, `forward_full(query)`,
`forward_incremental(query)`, `equivalence_capability(profile)`.
State thuộc engine, immutable/owned cache copies, recording/step/reference/
preprocessing/executable identity; caller không mutate cache qua return values.

W4-019 viết stage derivation, zero-based indices:

| Stage mới | Vị trí reuse | Vị trí stage cũ | Raw dependency interval |
|---|---|---|---|
| Conv1 + BN + ReLU | `j=2..61` | `j+64` | `[j-2,j+2]` |
| Pool1 | `j=1..30` | `j+32` | `[2j-2,2j+3]` |
| Conv2 + BN + ReLU | `j=2..29` | `j+32` | `[2j-4,2j+5]` |
| Pool2 | `j=1..14` | `j+16` | `[4j-4,4j+7]` |
| Conv3 + BN + ReLU | `j=2..13` | `j+16` | `[4j-8,4j+11]` |

Recompute tất cả positions còn lại với đúng padding/pool phase; assemble full
`(64,32)` tensor theo same Flatten order; chạy toàn bộ Linear2048→128, ReLU,
Linear128→12. H14 GAP new-minus-dropped identity không áp dụng cho head này.

Milestones W4-020: full-stage correctness scaffold → thay partial recomputation
từng stage → full finite replay → benchmark. Nếu partial input shape chọn kernel/
reduction khác, allclose không đủ để gọi bitwise exact. W4-021 gắn capability
với locked executable profile, report full logits errors/disagreements; universal
claim cần executable equivalence argument riêng. Chưa có argument thì mode là
`EMPIRICALLY_MATCHED` hoặc `DIAGNOSTIC`; strict runtime không tự nâng cấp nó thành
certified output. Khi overlap/continuity/identity invalid: full reference fallback.

Tests: stage maps và padding boundaries; first query; one-bit overlap mismatch;
raw/normalized overlap; recording reset; repeated/backward/skipped step; caller
alias; mutated state; invalid input unchanged state; full head order; all12 logits;
full5034 adjacent-pair finite evidence. Boundary features ngoài reusable ranges
phải recompute, không “fix” bằng chép old features.

### 2.8 `ecdr/week4/certificates.py`

`CertificateIssuer` là owner của việc chuyển validated bound+membership contract
thành executable certificate. `Certificate` là canonical serialized record;
caller-created record vẫn untrusted. Diagnostic records giữ path riêng và
positive conditional result không mặc nhiên executable certificate.

Certificate chứa schema/version, reference/executable/preprocessing/class-order
IDs, trusted anchor receipt/raw/normalized IDs, recording/step, target winner,
all11 competitors/lower bounds, domain identity/full serialized definition,
bound method/numerical capability/membership method IDs, source/review artifact
hashes, creation/max-age/horizon policy và lifecycle state.

`issue(...)` kiểm trusted owner receipt, strict anchor winner, all11 positive
validated enclosures, matching numerical target, supported membership capability,
reviewed profile và finite lifecycle. `check(...)` kiểm exact same identities,
age, recording, optional trusted anchor linkage và membership. Có private
constructor không thay thế những checks này. Không có public force-enable,
`allow_unsafe=True` hoặc string override cho strict issuer.

Test: forged dataclass/JSON status, source/profile/review mismatch, absent bridge,
positive conditional bound, missing competitor, class reorder, stale/expired/wrong
recording, changed domain, anchor mismatch, serialization tampering và tied anchor.
H15 recenter counterexample phải tiếp tục fail: reused query không thành trusted
anchor nếu chưa có independent reference inference/transfer proof.

### 2.9 `ecdr/week4/runtime.py`

`RuntimeEngine.predict(QueryView) -> RuntimeDecision` quản recording-aware state.
Policy symbols đặt ở đây: `BuildPolicy`, `FrozenPolicyConfig`,
`choose_action(trusted_state, query_metadata)`; không module selector/policy phụ.

Flow bắt buộc: validate query → inspect continuity/identity → validate existing
certificate/member → accepted reuse nếu mọi gate pass → nếu không, một trusted
reference inference → optional build từ receipt đó → atomic commit state.
Nếu reference identity đã sai thì không dùng chính mutated model làm “frozen”
fallback; reject/invalidate theo contract. Invalid input/overflow reject trước
state mutation. Build failure không làm mất usable trusted state ngoài policy
invalidation đã chỉ định. Exceptions giữa build/state commit có behavior rõ.

`RuntimeDecision` gồm prediction, optional logits và output semantics, event/reason,
reference/member/build/policy/state timings, total, anchor age/certificate IDs,
fallback/build counters. Certified class reuse không được giả vờ trả reference
logits khi chỉ chứng minh class. Comparator contract cần biết mỗi method trả gì.

W4-024 loại duplicate anchor forward qua trusted receipt; expose counts/timers
để test không “tối ưu” bằng bỏ identity nghĩa vụ. Identity policy và baseline mode
phải matched hoặc report khác biệt. W4-025 policy có build cap, cooldown, max-age,
time/memory budget chọn trên validation; no future labels, no oracle build rate.
Finite-horizon policy cần independent reachable-set premise; elapsed time hoặc
đúng event name không tự kéo dài certificate.

Tests: missing/conditional/expired certificates, fixed anchor/no recenter, recording
reset, step discontinuity, member ambiguity, forced unsupported numerical profile,
build errors, invalid input atomicity, one reference call per miss+build, denied
caller receipt, policy budget exhaustion, exactly one event per query và no label
access. W4-026 tập adversarial traces ghép các failure thay vì chỉ isolated unit cases.

### 2.10 `ecdr/week4/methods.py`

`MethodRegistry`, `MethodSpec`, `CapabilityDecision`, `make_method(...)` giữ stable
method IDs, target/output semantics, guarantee status, config/version, supported
domains, numerical requirements, identity policy và baseline timing boundary.
Method không đủ capability có row reason `UNSUPPORTED`/`NOT_IMPLEMENTED`;
runner không tự bỏ row hoặc substitute method khác dưới cùng tên.

Các rows tối thiểu cần có trong registry/comparison:

1. Frozen CNN core: validation+normalization+forward, không identity rehash.
2. Frozen strong-wrapper CNN: cùng per-query identity policy với strict runtime.
3. Owned-reference CNN nếu W4-024 thực thi threat model riêng.
4. Exact-input replay cache: input bytes/identity match; gắn deterministic backend
   premise; đây là replay control, không regional certificate.
5. Actual K=12 incremental path với equivalence status đúng W4-021.
6. Week3 coordinate/fixed-overlap diagnostic reproduction.
7. Coordinate IBP, regional Lipschitz, CROWN, wave và affine/PCA variants theo
   capability; configurations là rows khác nhau, không gộp tên để cherry-pick.
8. Strict certified ECDR: có thể hợp lệ với zero eligible certificates/reuse.
9. Heuristic ECDR nếu scope triển khai: approximation branch, measure errors riêng.
10. LipCache capability/faithful comparator; external streaming methods nếu đã audit.

W4-029 ghi exact paper/version/source/license/API, required training/GuardNet,
target consistency vs MainNet agreement, dataset/hardware/workload differences.
Nếu faithful LipCache cần GuardNet mới và nằm ngoài frozen core, đóng capability
task bằng evidence/blocker; không gọi local inspired cache là LipCache đã reproduce.
Không đổi current reference, training hoặc labels để tạo superiority claim.

Tests: duplicate method IDs, unknown config, wrong guarantee/output target,
unsupported method remains visible, deterministic method construction, label-free
predict interface và errors của external capability paths.

### 2.11 `ecdr/week4/metrics.py`

Pure functions cho `summarize_trace`, `agreement_metrics`, `classification_metrics`,
`cost_metrics`, `paired_statistics`, `validate_trace_accounting`. Không fit/tune
policy hoặc đọc ground truth trong runtime selection.

Denominators: operational stream 5,053 windows; supervised 3,572; adjacent pairs
5,034; recording starts 19 theo frozen Week3 snapshot. W4-001 xác nhận lại; nếu
khác, fail preflight và giải thích data identity thay đổi thay vì sửa assert.

Counters tách riêng: build attempts/failed builds/conditional positives/validated
certificates/candidate members/validated members/potential hits/accepted reuse/
fallback, reused disagreements và fallback disagreements. Đúng công thức:

`A_reuse = 1 - reused_disagreements / accepted_reuse` khi denominator >0;
denominator=0 thì `null`/N/A. Không dùng full-stream disagreement trong tử số
của reused-query agreement. Week3 reuse=0 nên nhánh formula cũ chưa ảnh hưởng
reported result; sửa công thức trong Week4, không rewrite frozen Week3 source.

Supervised accuracy/Macro-F1 tách với reference agreement. Offline strata join
bằng stable global row ID sau decision; full-stream runtime vẫn gồm ambiguous/
unlabeled windows. Empty strata, no reuse và unsupported comparator không thành
100%/0ms/0B tùy ý. Dùng null + reason khi không đo hoặc không áp dụng.

Cost gồm failed builds, candidate generation, membership, normalization/conversion,
identity/state, reference fallback, real transfers theo workload. Shadow inference,
CSV write, offline labels và analysis nằm ngoài operational query timer và báo
riêng; whole-loop wall report thêm. Speedup từ paired total times cùng scope,
không ratio of independent medians hoặc suy luận từ hit rate/MAC count.

Test toy traces tính tay, zero denominators, missing/duplicate/out-of-order rows,
fallback mismatch không làm sai A_reuse, sum reuse+fallback=N, all reasons total,
missing failed build cost, timing overlap/double-count, strata join cardinality,
quantile/statistic definitions và paired grouping theo recording/repetition.

## 3. Scripts: interface, kiểm tra và evidence

CLI trong bảng là proposed contract; triển khai task tương ứng và kiểm `--help`
trước khi dùng. Runner không overwrite run đã có mà không có explicit resumable
manifest/phase matching. Giữ artifacts của failed phase để diagnosis.

| Script | Nội dung phải viết | Task | Output/check bắt buộc |
|---|---|---|---|
| `scripts/week4_audit.py` | Read-only snapshot/preflight: Git/local state, protected hashes, graph/data/reference/env/Week3 receipts | W4-001 | `provenance.json`, `protected_manifest.json`, `source_map.md`, `environment.json`; nonzero exit khi identity bắt buộc sai; không sửa môi trường |
| `scripts/week4_tasks.py` | List/ready/claim/heartbeat/submit/review/release task theo orchestrator schema | W4-004 | Atomic state update, shared-file locks, dependency checks, receipt validation; không tự claim VERIFIED hoặc execute shell từ task text |
| `scripts/week4_probe.py` | Isolated library/operator và fixed-domain bound-method probes theo explicit profile | W4-013 | Capability JSON, exact commands/env/tracebacks, source hashes; unsupported gate vẫn có artifact/reason; no pip mutation |
| `scripts/week4_run.py` | Train-fit → validation sweep/freeze → pretest integration → full replay/benchmark/memory phases | W4-018, W4-030, W4-032…W4-034 | Run manifest, fit/validation/frozen-policy hashes, per-query traces, build/domain traces, phase receipts; test-known disclosure; no implicit tuning during replay |
| `scripts/week4_tables.py` | Aggregate validated receipts thành comparison/hypothesis/stats tables và plot inputs | W4-035 | CSV/JSON/Markdown tables, denominator/method/source IDs; unknown=N/A; không re-fit hoặc fabricate pending numbers |
| `scripts/week4_verify_delivery.py` | Independent consistency/preservation/artifact/requirements validator | W4-040 | Per-check PASS/FAIL/NOT_RUN, stale-hash detection, missing task/evidence table; success không tự claim numerical proof hoặc ACCEPTED |

CLI canonical dự kiến dùng `--run-id RUN_ID` và resolve duy nhất thành
`output/week4/RUN_ID/`. Không dùng đồng thời một documented `--run-dir` interface
khác. Subcommands và owners phải đồng bộ task cards/`--help`/tests:

- Audit: `preflight`, `snapshot`, `check` — W4-001.
- Probe trong isolated Python: `lirpa`, `bound-methods` — W4-013/014.
- Runner: `fit`, `geometry` — W4-018; `select`, `smoke` — W4-030;
  `replay` — W4-032; `benchmark` — W4-033; `memory` — W4-034.
- `fit`, `geometry`, `select`, `smoke` có `--split train|val` phù hợp từng phase;
  `replay` chỉ `--split test --sealed-policy POLICY_PATH`. Benchmark/memory cũng
  nhận `--sealed-policy`. Không silently tune hoặc pick config trong replay.
- Tables/delivery nhận `--run-id RUN_ID`. Claim CLI theo ORCHESTRATOR.md.

Run manifest ghi protocol/profile paths và hashes; optional overrides chỉ được
chấp nhận trước seal. W4-030 implement cả `smoke` command để W4-031 review gọi
mà không cần sửa code ngoài write set của reviewer. Nếu đổi interface, update
tất cả docs/tests trước dispatch dependent task.

`week4_tasks.py` phải chống hai agent cùng claim: validate dependencies/state,
atomic exclusive lock/transaction, record assignee/lease/changed paths, reject
conflicting file lock. Expired lease không xóa edits của agent; release/reassign
theo protocol có receipt. Review transition do reviewer/orchestrator role;
`submit` chỉ tới REVIEW, không tự VERIFIED. Template/sample receipt phân biệt
với actual execution receipt. Unit tests `tests/test_week4_task_claims.py` chạy trên
temporary task board để không mutate live board của dự án.

Runner tests hẹp ở `tests/test_week4_protocol.py` và `tests/test_week4_accounting.py`
dùng tiny synthetic streams/mocks để kiểm label access, protocol hash freeze,
phase ordering và denominator schema;
không thay full execution evidence bằng mock test. Delivery validator tests ở
`tests/test_week4_delivery.py` inject stale hash, missing PDF/review/capability,
wrong counts, unsupported method bị giấu và forged completion.

## 4. Config files: contents trước khi chạy

| File | Phải chứa | Owner/negative checks |
|---|---|---|
| `configs/week4/protocol.json` | Schema/version, known-test exploratory label, frozen identities, split usage, methods/grids, train/val fit rules, validation objective/tie-break, budgets, random seeds, chronology, timing/hardware scope, strata, stopping criteria, numerical capability requirements | W4-002; không test-optimize, không implicit defaults đổi giữa phases |
| `configs/week4/methods.json` | Stable method IDs, config hashes, guarantees/output target, identity mode, dependency profile, available/unsupported state và comparison-group IDs | W4-002 (tạo ban đầu), W4-029 (cập nhật comparator); W4-028 đọc để build method registry; không gộp strict/heuristic/conditional rows |
| `configs/week4/requirements-verifier.in` | Minimal top-level packages và intended compatibility của isolated verifier env; comment frozen `.venv` protected | W4-013; không broad unbounded upgrade |
| `configs/week4/requirements-verifier.lock` | Resolved exact versions/hashes/platform/interpreter hoặc reproducible lock format, package source provenance | W4-013; resolver/install/import/op-execution khác nhau phải có receipts riêng |

Chọn grid và budgets trước test replay: candidate count/radii/rank/method/build
time/memory limits, validation selection objective và tie-break. Nếu không có
feasible validated configuration, frozen selection ghi `NONE_ELIGIBLE` với
reasons; không lấy conditional winner rồi gọi strict result.

## 5. Design/proof artifacts phải viết gì

Đặt dưới `04_SYNTHESIS/week4/design/`, task card ghi exact target basename trước
claim. Không cần mỗi theorem một file; các documents cùng responsibility có
thể dùng headings/anchors để tránh duplicate authority.

| File design/proof | Nội dung và trách nhiệm chính | Task sở hữu | Điều kiện hoàn thành / Review |
|---|---|---|---|
| `04_SYNTHESIS/week4/design/EVALUATION_PROTOCOL.md` | Protocol đánh giá, metrics, budgets, known-test exploratory disclosure | W4-002 | Protocol sealed; frozen data identities khớp Week 1–3 |
| `04_SYNTHESIS/week4/design/NUMERICAL_CONTRACT.md` | Executable numerical target, operator proof obligations, rounding model | W4-005 | Proof obligations exhaustively mapped; stop rules định nghĩa |
| `04_SYNTHESIS/week4/design/FLOAT32_BRIDGE.md` | Derivation end-to-end frozen float32 bridge, conservative margin bound | W4-009 | Sound enclosure chứng minh hoặc chỉ rõ blocker không thể vượt |
| `04_SYNTHESIS/week4/design/NUMERICAL_REVIEW.md` | Independent review cho numerical gate, formal eligibility decision | W4-010 | Review độc lập; gate fail-closed nếu bridge chưa sound |
| `04_SYNTHESIS/week4/design/LIPSCHITZ.md` | K=12 global Lipschitz derivation, spectral norm bounds, ball radius | W4-012 | Hằng số L sound; margin test pass; baseline method M05 ready |
| `04_SYNTHESIS/week4/design/AFFINE_DOMAIN.md` | PCA/affine-tail domain geometry, bounded residual, correlation retention | W4-016 | Domain support proof; residual bounds không bị tùy tiện bỏ qua |
| `04_SYNTHESIS/week4/design/INCREMENTAL_SCHEDULE.md` | Receptive field mapping, Conv/Pool/BN stage dependency intervals | W4-019 | H13 alignment proof [4j-8, 4j+11]; index mapping formal |
| `04_SYNTHESIS/week4/design/INCREMENTAL_REVIEW.md` | Independent equivalence review cho incremental candidate, Flatten-MLP | W4-021 | Executable equivalence scope chốt; fallback behavior xác thực |
| `04_SYNTHESIS/week4/design/HORIZON.md` | Finite-horizon reachable sets, no-drift reuse policy, anchor expiry | W4-025 | Horizon proof; bounded reachable domain; no-recenter contract |
| `04_SYNTHESIS/week4/design/MATHEMATICAL_REVIEW.md` | Independent review toàn bộ 24 giả thuyết H01–H24 sau implementation | W4-027 | Mọi H có disposition rành mạch (DERIVED/FALSIFIED/CONDITIONAL) |
| `04_SYNTHESIS/week4/design/PRIOR_WORK_COMPARATORS.md` | Audit tính ứng dụng của LipCache, semantic comparison boundaries | W4-029 | Không giả mạo kết quả; khoảng cách kiến trúc được chứng minh |
| `04_SYNTHESIS/week4/design/PRETEST_REVIEW.md` | Pre-test integration review, readiness confirmation trước full replay | W4-031 | Protocol và frozen policy đã seal; test set chưa bị ô nhiễm |
| `04_SYNTHESIS/week4/design/INDEPENDENT_AUDIT.md` | Independent audit toàn diện mã nguồn, thực nghiệm và số liệu đo đạc | W4-036 | Audit độc lập; hash/counts tái lập khớp 100% |

- W4-005 numerical target: real graph vs interpreter vs backend, stored constants,
  normalization/order/rounding, supported profile, proof obligations và stop rules.
- W4-006…W4-009 arithmetic/operator/bridge: assumptions, lemma, derivation,
  implementation-symbol mapping, invalid ranges và what remains unproved.
- W4-010 independent numerical gate review: read source/proofs independently,
  counterexample search, profile/source/proof hashes, explicit eligible/unsupported
  result. Reviewer không author numerical claim mình đang duyệt.
- W4-015…W4-018 domain geometry: stored set definition, PCA fit space/centering,
  membership feasibility, correlation retention, matched-control justification;
  local gradient không thành regional theorem.
- W4-019 incremental schedule: stage dependency intervals/index maps ở phần2.7,
  padding/pool phase proof, prefix overlap and fixed preprocessing premises.
- W4-021 executable equivalence gate: exact target/backend, finite-evidence scope,
  universal premise nếu có, negative result/fallback và independent disposition.
- W4-022…W4-025 lifecycle/policy: anchor ownership, reset/expiry/discontinuity,
  no recenter, finite horizon, reference receipt trust, budget/cost definitions.
- W4-027 mathematical review: đúng toàn bộ H01–H24 meanings, original theorem
  vs actual-graph extension, proof/conditional/empirical/falsified/open labels.
- W4-029 comparator audit: faithful method semantics, source/version/training,
  replication gaps; unsupported rows vẫn có justification.

Proof text phải trỏ code symbol/test/evidence và ngược lại. Review signature/name
trong Markdown không phải automated proof; report rõ review method thực hiện.

## 6. `output/week4/<run_id>/`: file inventory cho một run đầy đủ

Tên dưới đây là manifest keys/logical artifacts; implementation chọn exact
basename một lần ở W4-003 và dùng đồng bộ runner/tables/validator/reports.
Không tạo receipts trước khi command/phase tương ứng thực sự chạy.

| Artifact | Nguồn tạo | Nội dung/verification |
|---|---|---|
| Run manifest | audit/runner | `run_id`, timestamps/timezone, Git/source/config/data/reference/env hashes, phase status, parent run nếu resume |
| Snapshot + preservation | W4-001 | Dirty state, protected inventory/hashes, 170-file Week3 chain và `.venv` dependency snapshot |
| Numerical capability/review | W4-005…W4-010 | Target/profile/op coverage, proof/review/source hashes, missing bridge, gate disposition |
| Library capability | W4-013/014 | Isolated lock/install/import/conversion/op-bound results và error/timing |
| Train-fit geometry | W4-015/016 | Train IDs, mean/scales/PCA/residual constants, coordinate convention, artifact hash |
| Validation sweep | W4-018/030 | Mọi attempted configuration, coverage/conditional/validated/build cost, selection objective/tie-break, unsupported/failures |
| Frozen selection | W4-030 | Exact chosen config hoặc none-eligible, fit+protocol hashes, known-test disclosure, timestamp trước replay |
| Pretest integration receipt | W4-031 | Relevant tests/exit codes, identity/preservation, gate state, replay eligibility |
| Domain/build trace | W4-032 | Anchor/query IDs, all11 margin vectors/statuses, domain IDs, membership, build attempts/failures/times |
| Runtime query trace | W4-032/033 | Exactly one row/query/repetition/method, event/reason/output/reuse/fallback, component timing và anchor lifecycle |
| Quality/agreement summary | W4-032/035 | 5053 operational,3572 supervised,5034 pairs,19 recordings; reference agreement và ground-truth quality riêng |
| Benchmark results | W4-033 | Hardware/profile/order/warmups/repetitions, paired total/p50/p95/wall, setup/load/shadow costs, uncertainty scope |
| Memory/scaling results | W4-034 | Persistent tensor/cache/domain bytes, baseline subtraction, peak workspace/RSS deltas nếu đo, loading/PCA excluded or disclosed |
| Tables/stats | W4-035 | Machine-readable inputs và final method/hypothesis/comparison tables, N/A/reasons, hashes |
| Independent audit | W4-036 | Read-only checks, changed paths, reproduced commands, mismatch/blockers, reviewer scope |
| Logs + tests | Relevant task | Exact command/cwd/env/source identity, exit code/timing; stdout/stderr, failures kept |
| PDF/render QA | W4-039 | Exact source/PDF hashes, compiler logs, physical pages, render hashes và visual inspection scope |
| Final delivery/closure | W4-040 | Requirement→artifact→evidence→status; missing/limited items; all40 tasks and H01–H24 counted; gates/scientific result separate |

Memory report không gọi process peak RSS gồm toàn dataset/PCA là cache overhead.
Tách reference tensor bytes vốn có, extra bound/incremental/cache/domain bytes,
transient workspace, allocator/RSS delta và scope/hardware. Chưa đo energy/GPU
thì N/A; không suy ra từ MACs hoặc CPU latency.

## 7. Reports và native deliverables

| Artifact giao nộp | Trách nhiệm và nội dung | Task | Deliverable format |
|---|---|---|---|
| `04_SYNTHESIS/week4/reports/Week_4_Implementation_and_Evaluation.tex` | Báo cáo chính thức: problem, math, implementation, experiments, gates | W4-037 | LaTeX source (XeLaTeX/PDFLaTeX) |
| `04_SYNTHESIS/week4/reports/Week_4_Implementation_and_Evaluation.pdf` | Bản PDF biên dịch và QA trực quan toàn bộ trang | W4-037, W4-039 | PDF native |
| `04_SYNTHESIS/week4/reports/refs.bib` | Thư mục tài liệu tham khảo đồng bộ Week 1–4 | W4-037 | BibTeX |
| `04_SYNTHESIS/week4/reports/Week_4_Explained.tex` | Báo cáo giải thích trực quan, worked actual K=12 example, reproduction | W4-038 | LaTeX source |
| `04_SYNTHESIS/week4/reports/Week_4_Explained.pdf` | Bản PDF giải thích đã visual QA từng trang | W4-038, W4-039 | PDF native |
| `04_SYNTHESIS/week4/reports/REPRODUCE.md` | Hướng dẫn tái lập thực nghiệm từng bước với CLI commands thực | W4-038 | Markdown |
| `04_SYNTHESIS/week4/CLOSEOUT.md` | Tổng kết nghiệm thu Week 4, gate recommendations (G3–G6) | W4-040 | Markdown |
| `README.md`, `GATES.md`, `GATES_EN.md`, `04_SYNTHESIS/README.md` | Cập nhật navigation, kết quả nghiệm thu Week 4 và liên kết báo cáo | W4-040 | Markdown |

W4-037 tạo formal report trong `reports/`: problem/frozen contract → actual
graph/domain definitions → assumptions/theorems/proofs → implementation mapping
→ all24 hypotheses → complete experiments/comparison/cost → limitations/gates.
Trình bày kết quả âm và unsupported comparators như kết quả có scope, không xóa
để làm dashboard xanh. H24 cần soundness, useful reuse và positive net savings
trong **cùng** configuration/protocol, không lấy proof A + coverage B + speed C.

W4-038 tạo explained/reproduction report trong `reports/`: định nghĩa ký hiệu/
đơn vị tại chỗ, worked actualK12 example, shape/index maps, candidate/member/
certificate/runtime distinction, exact commands từ CLI đã implement, required
data/environment và expected artifacts. Không ghi expected success output như
đã chạy. Hai report dùng cùng machine-readable results và hypothesis IDs.

W4-039 compile final LaTeX thành PDF, inspect logs, render **mọi physical page**,
visual QA tables/equations/links/fonts/page breaks. Source đổi sau QA làm receipt
cũ stale; compile pass không thay visual QA. Thực hiện theo PDF skill có sẵn
khi triển khai task, không coi plan này là đã đọc skill thay agent.

W4-040 chỉ cập nhật current navigation/gates trong explicit files task card cho
phép, sau khi read current state. Giữ historical Week3 receipts như run record.
Không auto commit/push/deploy từ closeout nếu task/user không cấp quyền. Trạng
thái CREATED, VERIFIED, ACCEPTED tách riêng; không tự ACCEPTED.

## 8. Integration checks và các lỗi phải chủ động chặn

- Formal gate mở bằng author boolean, arbitrary JSON hoặc conditional float64
  positivity: FAIL.
- Verified interpreter khác target backend nhưng report cùng identity: FAIL.
- Correlated domain bị convert thành outer box mà tên/result còn claim giữ
  correlation: FAIL.
- PCA fit/centering/radius/build policy dùng test hoặc next-window runtime oracle:
  FAIL; offline oracle diagnostic phải tách hoàn toàn.
- Không công bố test split đã known từ Week3 và gọi pristine held-out: FAIL.
- Incremental reuse nhầm padding boundaries, pool phase hoặc GAP delta: FAIL.
- Full-stream sampled logits match được gọi universal float proof: FAIL.
- Identity optimization giảm threat model nhưng dùng baseline claim cũ: FAIL.
- Miss+build reference forward hai lần nhưng giấu một call khỏi timing: FAIL.
- Fullstream disagreement thay reused disagreement trong A_reuse: FAIL.
- Zero reuse được ghi100%reuse-agreement, unsupported method ghi0ms, missing
  memory ghi0B: FAIL; dùng N/A+reason.
- Omit failed builds, policy/identity/member work hoặc shadow nằm trong một
  method nhưng baseline khác timing boundary: FAIL.
- G2 quality HOLD biến thành PASS nhờ formal caching/performance: FAIL.
- Full40 task checkbox hoặc report/PDF presence được gọi H24 proved: FAIL.

Hoàn tất engineering nghĩa có artifact/review/limitations chính xác cho từng
task. Hoàn tất scientific goal yêu cầu evidence đáp ứng đúng goal; nếu numerical
bridge hoặc useful/cost-saving regime còn thiếu, final phải ghi partial/not
established tương ứng dù toàn bộ khả thi implementation đã giao xong.
