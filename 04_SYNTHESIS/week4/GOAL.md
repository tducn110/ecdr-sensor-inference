# Full goal — ECDR Week 4

## 1. Goal tổng

Xây và kiểm độc lập pipeline implementation cho frozen SmallCNN1D K=12 từ raw
sensor window đến domain, bounds, numerical contract, certificate, runtime và
đánh giá full cost. Trả lời đủ H01–H24 bằng theorem/counterexample/measurement
đúng loại, so sánh các phương pháp theo cùng semantics và giao một bộ report
có thể tái lập. Goal là giải quyết research obligations bằng bằng chứng; không
ép tất cả hypotheses thành đúng hoặc ép G3–G6 thành PASS.

**Đầu ra đang được tạo:** plan/task specifications để user dispatch `agy`.
**Đầu ra của implementation tương lai:** modules/tests/receipts/reports tại các
path trong FILE_SPEC.md và task cards. Không nhầm hai mức này.

## 2. Frozen problem

| Thành phần | Contract giữ nguyên |
|---|---|
| Input | Raw float32 6×128, 50 Hz, stride64; C=6, W=128, s=64; K=12 là classes |
| Data | HAPT; subject split17/4/9; train-fit z-score và hiện trạng annotation origin |
| Graph | padded Conv6→32 k5 BN-ReLU-Pool2; Conv32→64 k3 BN-ReLU-Pool2; Conv64→64 k3 BN-ReLU; Flatten2048→Linear128-ReLU→Linear12 |
| State | 283.692 trainable parameters; eval mode, frozen BN buffers |
| Checkpoint | `output/week1_closeout_20261102/reference/selected_best.pt` |
| SHA | `8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1` |
| Preprocessing ID | `aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e` |
| Denominators | 5.053 test windows;3.572 supervised;5.034 valid adjacent pairs;19 recordings |
| Ground-truth reference quality | Accuracy0,8905375;macro-F1 0,7762645;G2 HOLD |
| Correctness target | Strict preservation of reference decision; ground-truth correctness khác metric |

Không retrain, đổi labels/splits/class order, sửa checkpoint/preprocessing, đổi
backbone sang GAP, hoặc cập nhật frozen environment. Một environment verifier
riêng không tự định nghĩa lại frozen executable.

## 3. Điểm xuất phát Week 3

Week 3 có all-eleven real-arithmetic IBP formulas, actual graph/lifecycle tests,
fail-closed runtime và diagnostic evidence. Formal reuse vẫn disabled vì chưa
bao sai số frozen float32 executable. Radius val-selected1e-6 cho diagonal/wave
lần lượt5.034/4.211 conditional-positive builds trên5.053 anchors, cả hai zero
next members; bốn replays fallback toàn bộ. CPU complete ECDR mean3,107–3,198ms
so core0,215–0,224ms; measured net saving âm. Đây là historical measurement có
scope, không phải mục tiêu số phải tái tạo bằng mọi giá.

H13 real alignment: final RF `[4j−8,4j+11]`, jump4; next feature j2…13 khớp
old j+16 theo giả thiết. H14 GAP delta không áp Flatten–MLP. Wave fixed prefix
có384 tail variables nhưng vẫn là coordinate box; PCA correlation cần một
representation và propagation thật. Ordinary float64, epsilon tùy ý và sampled
agreement đều không đóng H22.

## 4. Specific goals và bằng chứng hoàn thành

| Goal ID | Cần làm | Exit evidence | Tasks |
|---|---|---|---|
| W4-G01 | Freeze source/model/data/runtime identities; known-test protocol | Manifest, immutable protocol hash, protected hashes |001–005|
| W4-G02 | Numerical primitives→normalization→operators→float32 bridge | Derivation + high-precision/adversarial checks + independent scope review |005–010|
| W4-G03 | All11 margin specs, Lip/IBP/CROWN compatibility | Actual graph tests, method/operator status matrix, conditional vs eligible bounds |011–014|
| W4-G04 | Diagonal/wave/PCA-affine domains và fair matching | Membership/support proofs, basis identity, correlation-preserving backend hoặc explicit blocker |015–018|
| W4-G05 | Actual padded/BN/Pool incremental candidate | Layer-index proof, full head output, backend-equivalence scope |019–021|
| W4-G06 | Immutable lifecycle, selector/runtime/cost/horizon | Non-bypass gates, label-free queries, no recenter, negative tests |022–026|
| W4-G07 | Preserve24 hypotheses và comparable baselines | H ledger, comparator semantics, no invented LipCache result |027–031|
| W4-G08 | Execute full replay/cost/memory | Complete traces, correct denominators, paired timing, unsupported/timeouts counted |032–034|
| W4-G09 | Independent result synthesis | Regenerated tables/recounts, reviewed evidence per H01–H24 |035–036|
| W4-G10 | Native reports/PDFs + reproducibility | Editable TeX, reviewed rendered PDFs, delivery receipt, gate recommendation |037–040|

## 5. Điều kiện cho từng loại claim

### Mathematical validity

Với anchor winner c, mọi j≠c trong12classes phải có bound hợp lệ trên cùng miền:
`ell_cj ≤ inf_{X∈S}(F_c(X)−F_j(X))`. Tất cả11 lower bounds strict-positive là
điều kiện đủ cho winner constant trong exact real semantics. Bound failure chỉ
là inconclusive, không chứng minh input đối nghịch tồn tại.

### Executable validity

Phải nối raw conversion→normalization→operator backend→output comparison với
validated enclosure của đúng frozen function. BN eval division/sqrt, fused
backend/reduction ordering/subnormal/overflow và direct-head arithmetic đều nằm
trong nghĩa vụ. `nextafter` cuối phép Torch/NumPy tổng không tự bao rounding bên
trong. Nếu thiếu contract, trả numerical status chưa established và đóng reuse.

### Useful coverage

Phân biệt proposal membership, conditional-positive member, validated build và
accepted formal reuse; H19 không cho phép suy coverage từ soundness. Báo zero
hits đúng nghĩa và `A_reuse=null/N/A`. Không chọn event bằng ground-truth labels
hoặc biết prediction query trước khi quyết định reuse.

### Net saving

Đo cùng trace/device/thread/data-residency/preprocessing/output contract; tính
setup, event, identity, membership, state, failed/successful builds, fallback,
data movement và I/O theo scope. Tỷ số chính là total baseline/total method;
không lấy tỷ số medians thay total. Diagnostic-only method không là certified
speedup dù prediction agreement hữu hạn đạt100%.

### H24 conjunction

Cùng **method+domain+policy+profile+protocol** phải đủ mathematical validity,
executable validity, nonzero useful certified reuse và positive complete-path
savings. Nếu một điều thiếu thì H24 cho cấu hình đó NOT_ESTABLISHED hoặc measured
negative; không kết luận mọi operating regime đều không thể tồn tại.

## 6. Definition of done của gói implementation

- [ ] 40 tasks có implemented/reviewed hoặc limited/blocker disposition cụ thể;
  task mandatory còn thiếu feature khiến overall delivery PARTIAL.
- [ ] 24/24 H IDs giữ nguyên statement/type; không trộn Week1 H1–H4 với Week3 H01–H24.
- [ ] Mọi method/experiment dự kiến có result hoặc explicit unsupported/timeout/
  blocked entry; không bỏ hàng bất lợi.
- [ ] Formal reuse không thể bật bằng CLI/config/JSON label khi numerical gate thiếu.
- [ ] Exact incremental và decision reuse báo hai rates khác nhau, output contract rõ.
- [ ] Full5.053-stream và supervised3.572 denominators; recording resets đầy đủ.
- [ ] Tables được tái tạo từ traces; no recycled historical result as new run.
- [ ] Source/checkpoint/frozen evidence preserved; no external mutation tự động.
- [ ] Report A/B có source+PDF, review mọi final page và hash receipt current.
- [ ] Final nêu engineering status, scientific status, gate status và limitations riêng.

## 7. Ngoài scope mặc định

Training proxy/GuardNet hoặc model mới; thay nhãn/normalization; GPU/mobile/energy
claim chưa có môi trường/measurement; deploy/publish/commit/push tự động bởi agy;
thêm benchmark dataset để giả tạo untouched confirmation. Những việc này cần
user direction mới nếu ảnh hưởng architecture/reference/workload hoặc external state.
