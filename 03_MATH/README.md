# Tầng 3 — Cơ sở toán học, chứng nhận miền và tính khả thi K=12

Bản chuẩn Week 3 hiện dùng đúng frozen `SmallCNN1D(num_classes=12)` đã khóa
ở Week 1–2: input `[B,6,128]`, 283.692 trainable parameters, padded
Conv–BN(eval)–ReLU–MaxPool và Flatten–MLP, 12 logits, 11 competing margins.
`C=6` là số sensor channels; `K=12` là số activity classes. **G2 giữ HOLD.**

Mục tiêu toán học là lower bound hợp lệ cho từng margin trên **toàn miền**;
membership của một query chỉ nối query đó với miền đã chứng minh. Định lý
số thực, operator implementation, numerical soundness float32, zero observed
disagreements và net savings là các mức bằng chứng khác nhau.

## Trình tự đọc và trách nhiệm nguồn

1. [Khung toán học](01_04_THEORY.md): định nghĩa raw `X`, frozen normalization
   `N`, normalized `Z`, reference `F=fθ∘N`, anchor, box/wave domain và
   whole-domain margin. Có đối chiếu K=6/GAP lịch sử và actual K=12.
2. [G3 domain contract](02_07_G3_DOMAIN_CONTRACT.md): identity, all-eleven
   margins, conservative membership, numerical obligation, lifecycle và fallback.
3. [Report A — Mathematical Foundations](03_Week_3_Mathematical_Foundations.tex)
   ([PDF](03_Week_3_Mathematical_Foundations.pdf)): báo cáo chuẩn gồm mô hình,
   assumptions, theorem/proof/counterexample, IBP/CROWN, domain geometry,
   numerical/cost analysis và audit từng H01–H24. Mỗi ID giữ nghĩa gốc,
   loại, giả thiết, applicability K=12, evidence, proof obligation và disposition.
4. [Report B — Mathematics Explained](04_ECDR_Mathematics_Explained.tex)
   ([PDF](04_ECDR_Mathematics_Explained.pdf)): giải thích ký hiệu, đơn vị,
   dimensions, thay số, lỗi diễn giải và cùng mapping H01–H24. Ví dụ toy
   tính tay được phân biệt với output của checkpoint; phần đầu giải thích
   actual graph, BN/Pool, direct margin và RF alignment từng bước.
5. [English concept atlas](05_ECDR_Mathematics_Concepts_104_Atlas.tex): 104
   khái niệm nền tảng. Các mục valid-convolution/GAP là ví dụ chuyên biệt
   giữ giả thiết, không được dùng như graph hiện tại.
6. [Norm geometry](06_SUPP02_Norm_Geometry.tex): norm trong `R^768`, raw
   versus normalized units, joint versus marginal coverage; structured-domain
   benefit là hypothesis phải kiểm, không phải hệ quả của correlation.
7. [Certification methods](07_SUPP03_Certification_Methods.tex): real IBP,
   actual BN/Pool/Flatten rules, CROWN relaxations và giới hạn software/numerical.
   Không khẳng định mọi CROWN variant luôn chặt/rẻ hơn IBP.
8. [Wave domains](08_SUPP05_Wave_Domain.tex): fixed overlap, 384 free tail
   values, centered PCA, exact reachable-state recurrence và giả thiết residual.
9. [Floating-point precision](09_SUPP06_Float_Precision.tex): stored float32
   constants, rounding, normalization containment, conditional error bounds và
   nghĩa của executable soundness. `float64` hoặc epsilon tùy ý không tự là proof.
10. [Gate-A prototype lịch sử](10_script_11_gate_a_prototype.py) và
    [probe lịch sử](11_p0_verifier_probe.json): K=6/pure-only; giữ để truy nguồn,
    không tính là operator hoặc certificate evidence K=12.

## Các kết quả toán học cần giữ đúng scope

Whole-domain positive lower margins bảo toàn winner trong số thực với đúng
identity và membership. Real IBP sound khi mọi operator enclosure đúng nhưng
incomplete. Continuity chỉ bảo đảm tồn tại local region, không bảo đảm
coverage. Feature alignment actual graph có inclusive RF `[4j−8,4j+11]`,
jump 4, và `H_j(X_next)=H_(j+16)(X_anchor)` cho `j=2,…,13` với fixed
BN evaluation và pooling phase: 12 positions × 64 channels. Equality này
không suy ra invariance của Flatten–MLP. Theorem valid/GAP A1–A6 vẫn hợp lệ
trong architecture lịch sử của nó; GAP delta không áp vào actual head.

Giữ fixed anchor hoặc dùng transfer/horizon theorem có proof. Không recenter
certificate sau một hit. Soundness không suy ra useful coverage; coverage
không suy ra speedup. H21/LipCache phải so đúng semantics; H24 đòi đồng thời
mathematical validity, executable numerical validity, useful hits và net savings.

## Execution evidence và trạng thái khoa học

Nguồn thực thi mới, tests và receipt ở
[`output/week3_certification_20261103/`](../output/week3_certification_20261103/).
Run cuối có 5.053 diagonal và 5.053 wave builds: lần lượt 5.034 và 4.211
conditional-positive, nhưng zero next-member và zero formal runtime reuse.
Bốn runtime replays fallback toàn bộ; ECDR mean 3,1069–3,1975 ms/query,
core reference 0,2147–0,2240 ms/query; saving âm cho measured CPU policies.
93 tests pass gồm 60 historical, 27 actual K=12, 6 math; thêm 65 baseline
artifact checks pass. Đọc scope cụ thể trong hai report và receipt cuối; không nâng
candidate hits hoặc positive diagnostic bounds thành executable certified reuse.
Zero accepted reuse có reused-query agreement **N/A**. G3 còn conditional nếu
numerical bridge chưa được chứng minh; G4/G5 chỉ được cập nhật theo evidence
coverage/cost của đúng workload. Không mục nào được tự gắn ACCEPTED.

Checkpoint SHA-256:
`8ffc5dd8eb0e9cdf832160d074f3ae816ae6cba02adf7214a63a8b623303adb1`.
Preprocessing ID:
`aa8e1cc20829307187e0a78200f9957f7dd068b1dc26063c2720ad832bb2fd6e`.
Frozen Week 1/2 evidence không được sửa hoặc thay bằng probe mới.

Các bản `_VI.md` là bản dịch lịch sử, giữ nguyên để truy nguồn; contract
English cập nhật ở trên là authority hiện tại. Tổng hợp/gates ở
[Tầng 4](../04_SYNTHESIS/README.md) và [GATES.md](../GATES.md) có trách nhiệm
khác với mathematical report này. PDF phải được compile từ đúng source hiện
hành và kiểm render; compiler exit0 riêng lẻ không phải visual QA.
