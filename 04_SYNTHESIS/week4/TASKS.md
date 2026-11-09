# Week 4 — 40 tasks để claim từng việc

Trạng thái khởi tạo: **40 TODO, 0 claimed, 0 executed, 0 verified**. Bảng này là
index của specifications; trạng thái live đọc từ [tasks.json](tasks.json), không
đếm số checkbox để suy gate PASS. Mỗi task card chứa write set, input, checklist,
proposed commands, outputs, failure/stop rules và handoff contract.

Đọc [ORCHESTRATOR](ORCHESTRATOR.md) trước claim. W4-001 là task đầu tiên.
W4-004 xây claim CLI; trước đó dùng manual single-writer protocol. Task graph
cho phép parallel branches nhưng conflicting file locks vẫn phải serialize.

| Task | Việc cụ thể | Dependencies | Hypotheses |
|---|---|---|---|
| [W4-001](tasks/W4-001.md) | Khóa baseline, source tree và preservation manifest | — | H01, H03, H13, H21, H22 |
| [W4-002](tasks/W4-002.md) | Định nghĩa protocol, metrics, budgets và known-test disclosure | W4-001 | H10, H11, H18, H19, H20, H21, H24 |
| [W4-003](tasks/W4-003.md) | Tạo immutable public types và frozen reference adapter | W4-001, W4-002 | H05, H15, H16, H22, H23 |
| [W4-004](tasks/W4-004.md) | Implement task claim, file locks và execution receipts | W4-001, W4-003 | Infrastructure |
| [W4-005](tasks/W4-005.md) | Chốt executable numerical target và operator proof obligations | W4-001, W4-003 | H04, H05, H06, H07, H08, H22 |
| [W4-006](tasks/W4-006.md) | Validated outward arithmetic primitives và independent oracle checks | W4-005 | H05, H07, H22 |
| [W4-007](tasks/W4-007.md) | Frozen normalization enclosure và conservative membership | W4-003, W4-006 | H01, H05, H12, H18, H22 |
| [W4-008](tasks/W4-008.md) | Enclose actual Conv–BN–ReLU–Pool–Flatten–Linear operators | W4-007 | H05, H07, H08, H22 |
| [W4-009](tasks/W4-009.md) | Derive end-to-end frozen float32 bridge và safe margin acceptance | W4-005, W4-006, W4-007, W4-008 | H05, H06, H08, H22 |
| [W4-010](tasks/W4-010.md) | Independent numerical eligibility review và non-bypass capability gate | W4-009 | H05, H22, H24 |
| [W4-011](tasks/W4-011.md) | Actual-graph IBP và all11 direct/separate margin specifications | W4-003, W4-008 | H04, H05, H07, H08, H09 |
| [W4-012](tasks/W4-012.md) | Actual K12 global Lipschitz-margin baseline | W4-011 | H06, H19, H21, H22 |
| [W4-013](tasks/W4-013.md) | Isolated LiRPA environment và native operator capability probe | W4-001, W4-005 | H07, H09, H21, H22 |
| [W4-014](tasks/W4-014.md) | CROWN adapter và fixed-domain bound-method comparison | W4-011, W4-013 | H08, H09, H11, H22 |
| [W4-015](tasks/W4-015.md) | Causal diagonal/wave proposals và exact prefix contract | W4-003, W4-007 | H01, H02, H03, H11, H12, H18, H23 |
| [W4-016](tasks/W4-016.md) | Train-fit PCA/affine-tail domain với bounded residual | W4-002, W4-015 | H10, H11, H12, H18, H22 |
| [W4-017](tasks/W4-017.md) | Correlation-preserving bounds và hull-loss control | W4-012, W4-014, W4-016 | H07, H08, H10, H11, H22 |
| [W4-018](tasks/W4-018.md) | Matched geometry và local-margin diagnostics driver | W4-014, W4-015, W4-016, W4-017 | H02, H04, H08, H09, H10, H11, H12, H19 |
| [W4-019](tasks/W4-019.md) | Derive actual stagewise feature-cache schedule | W4-001, W4-003 | H01, H12, H13, H14 |
| [W4-020](tasks/W4-020.md) | Implement actual incremental engine và complete Flatten–MLP head | W4-003, W4-019 | H01, H13, H14, H20, H22 |
| [W4-021](tasks/W4-021.md) | Independent incremental executable-equivalence gate | W4-005, W4-020 | H13, H14, H21, H22 |
| [W4-022](tasks/W4-022.md) | Certificate issuer, immutable store và fixed-anchor lifecycle | W4-003, W4-007, W4-010, W4-011 | H05, H15, H16, H18, H22, H23 |
| [W4-023](tasks/W4-023.md) | Runtime executor, causal selector và trusted fallback | W4-015, W4-022 | H03, H05, H15, H16, H18, H19, H22, H23 |
| [W4-024](tasks/W4-024.md) | Giảm duplicate work và đo identity threat-model trade-off | W4-023, W4-021 | H18, H20, H21, H22 |
| [W4-025](tasks/W4-025.md) | Finite-horizon reachable domains và no-drift reuse policy | W4-017, W4-024 | H12, H15, H16, H17, H19, H20 |
| [W4-026](tasks/W4-026.md) | Adversarial integration: gate, selector, identity và state failure paths | W4-010, W4-021, W4-024, W4-025 | H04, H05, H15, H16, H17, H18, H22, H23 |
| [W4-027](tasks/W4-027.md) | Independent mathematical review toàn bộ H01–H24 | W4-010, W4-012, W4-014, W4-017, W4-019, W4-021, W4-025, W4-026 | H01–H24 |
| [W4-028](tasks/W4-028.md) | Method registry, strict/heuristic/exact-input baselines | W4-003, W4-010, W4-021, W4-023 | H06, H13, H18, H19, H20, H21, H22, H24 |
| [W4-029](tasks/W4-029.md) | LipCache faithful applicability và prior-work comparison audit | W4-001, W4-002 | H09, H13, H14, H21, H24 |
| [W4-030](tasks/W4-030.md) | Validation selection, build policy và final configuration seal | W4-002, W4-018, W4-024, W4-025, W4-026, W4-027, W4-028, W4-029 | H04, H08, H09, H10, H11, H16, H17, H19, H20, H21, H24 |
| [W4-031](tasks/W4-031.md) | Independent pre-test integration và readiness review | W4-004, W4-030 | H05, H11, H15, H16, H18, H20, H21, H22, H23, H24 |
| [W4-032](tasks/W4-032.md) | Full chronological replay, coverage và reference agreement | W4-031 | H01, H03, H04, H05, H08, H09, H10, H11, H12, H13, H15, H16, H17, H19, H21, H22, H23, H24 |
| [W4-033](tasks/W4-033.md) | Paired complete-path CPU benchmark và break-even accounting | W4-032 | H09, H18, H20, H21, H24 |
| [W4-034](tasks/W4-034.md) | Memory components, horizon/build scaling và membership cost | W4-033 | H10, H11, H12, H17, H18, H20, H24 |
| [W4-035](tasks/W4-035.md) | Generate all comparison tables, statistics và24-hypothesis result ledger | W4-032, W4-033, W4-034 | H01–H24 |
| [W4-036](tasks/W4-036.md) | Independent experiment, implementation và claim audit | W4-035 | H01–H24 |
| [W4-037](tasks/W4-037.md) | Write formal Week4 implementation/evaluation report | W4-027, W4-036 | H01–H24 |
| [W4-038](tasks/W4-038.md) | Write explained report và exact reproducibility guide | W4-037 | H01–H24 |
| [W4-039](tasks/W4-039.md) | Compile final PDFs, render mọi trang và bind provenance | W4-037, W4-038 | Infrastructure |
| [W4-040](tasks/W4-040.md) | Final delivery verifier, gate recommendation và handoff | W4-004, W4-036, W4-039 | H01–H24 |

## Điều phối theo milestone

| Milestone | Task IDs | Review barrier |
|---|---|---|
| M0 Contract/infrastructure | 001–004 | Frozen identity, no-label schema, claim ownership |
| M1 Numerics | 005–010 | Numerical eligibility quyết định độc lập; ineligible không bị bypass |
| M2 Methods/domains | 011–018 | Real/float scope, actual ops, correlation and fair matching |
| M3 Execution/lifecycle | 019–026 | Incremental gate, trusted owner, fixed/horizon state, fail closed |
| M4 Baselines/pre-test | 027–031 | All24 math review, comparator semantics, validation seal |
| M5 Measurement | 032–034 | Chronological full stream, total costs and memory boundaries |
| M6 Synthesis/audit | 035–036 | Trace-generated tables and independent review |
| M7 Reports/delivery | 037–040 | Editable reports, every-page PDF QA, final evidence check |

## Critical decisions không được tự đóng

- W4-010 numerical gate: formal regional reuse cần adequate profile-bound justification.
- W4-021 incremental gate: real equality/finite evidence chưa tự là executable proof.
- W4-029 faithful comparator: blocked adaptation giữ H21 NOT_ESTABLISHED.
- W4-031 pre-test review: sealed policy và nguồn đã review trước replay.
- W4-036 audit: source/protocol thay đổi phải rerun evidence affected.
- W4-040 delivery: mandatory feature còn thiếu → PARTIAL; negative research result
  không bị sửa thành positive result để đóng task.

Toàn bộ cards có **281 checklist requirements riêng**, thêm checklist nộp review.
Mỗi ID checklist là đơn vị requirement→output→evidence; không phải 281 tests đã chạy.
Design/API/file contents chi tiết ở [FILE_SPEC](FILE_SPEC.md); coverage H01–H24 và
so sánh methods ở [HYPOTHESES](HYPOTHESES.md), [COMPARISON](COMPARISON.md).

