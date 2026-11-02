# ECDR — Báo cáo nghiên cứu và hiện trạng

**Cập nhật:** 22/10/2026. **Loại:** scientific working report.
**Trạng thái:** G0 repo-record PASS; G1 COMPLETE_WITH_REVISIONS; G2 repo-record PASS;
G3–G6 PENDING. Novelty và net saving chưa xác nhận.

Bản này thay nội dung legacy của report.md bằng hiện trạng có truy vết.
Bản legacy trước cập nhật còn trong Git history; simulation và proposal
architecture cũ không phải kết quả hiện tại. Bản khoa học LaTeX độc lập:
[latex/dossier.tex](latex/dossier.tex). Tóm lược:
[PROJECT_BRIEF.md](PROJECT_BRIEF.md). Domain contract:
[07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md).

## Tóm tắt

ECDR nghiên cứu cách tránh chạy lại neural classifier trên từng cửa sổ cảm
biến bằng cách tái sử dụng quyết định của trusted anchor. Query phải thuộc
đúng miền được chứng nhận; lower bounds của mọi class-pair margin phải dương
trên toàn miền. Ba trụ cột cần kiểm riêng là soundness, giá trị event structure
và hiệu quả end-to-end. Project có data/model baseline HAPT và exploratory
region audit, chưa có verifier/runtime reuse hoặc net saving. Các kết quả
bên dưới là artifact được project ghi nhận đo ngày 15/10/2026; ngày cập nhật
tài liệu 22/10/2026 không phải ngày chạy lại. Chưa rerun model hay benchmark.

## 1. Dự án làm gì và giải quyết vấn đề gì?

Reference pipeline phân loại đoạn accelerometer/gyroscope vào sáu hoạt động
cơ bản. Chạy full inference cho mọi đoạn có thể lặp computation, nhưng
similarity hoặc overlap không chứng minh classifier giữ quyết định. ECDR
đề xuất certificate trước khi trả lại lớp đã tính.

Guarantee hướng tới **decision agreement với frozen backbone**, không phải
ground-truth correctness hoặc giữ toàn bộ logits. Project không nghiên cứu
context window/bộ nhớ LLM; chưa có deployment, energy saving hoặc device
performance evidence.

## 2. RQ và giả thuyết

**RQ1:** Event-conditioned class-pair bounds có chứng nhận giữ nguyên quyết
định thường xuyên hơn global Lipschitz bounds, trong khi tổng chi phí đủ thấp
để tạo net inference saving không?

- **RQ2:** Domain, membership, identity và numerical assumptions nào đủ cho sound reuse?
- **RQ3:** Event structure có cải thiện certified coverage/reuse trong so sánh công bằng?
- **RQ4:** Reuse/amortization nào bù được construction, event, membership, fallback và rebuild?

**H1:** Valid lower bounds trên toàn miền phải đều dương, cùng membership đúng,
là điều kiện đủ cho decision preservation. **H2:** Structured region hữu ích ở ít nhất
một operating regime. **H3:** Có regime với net saving. H2/H3 còn mở;
không đặt target cứng giảm 50% latency.

## 3. Window, event và region

| Đối tượng | Định nghĩa | Trạng thái |
|---|---|---|
| Time-series window | Một input 6×128, 128 thời điểm/kênh; 2,56 s ở 50 Hz | Source đã dựng |
| Event descriptor/selector | Mô tả biến thiên và chọn template/budget | Online selector còn đề xuất |
| Candidate region | Tập input gắn với trusted anchor theo constraints | Có candidates audit; chưa có certificate |
| Certified region | Cùng miền đó khi có valid positive lower margins và identity đúng | Chưa triển khai |

Stride 64 = 1,28 s, overlap 50%. Shared samples dịch vị trí trong tensor CNN;
overlap không tự suy ra same decision. Project dựng lại raw windows; số
17.453 khác 10.929 feature records trên trang
[UCI HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions).

## 4. Prior work và candidate gap

[CROWN](https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html)
và [automatic LiRPA](https://papers.nips.cc/paper/2020/hash/0cbc5671ae26f67871cb914d81ef8fc1-Abstract.html)
cung cấp bound machinery. [API auto_LiRPA](https://auto-lirpa.readthedocs.io/en/latest/api.html)
có specification matrix C ánh xạ logits thành margins; tính năng này chưa
xác nhận compatibility/numerical soundness trên graph ECDR.

Theo [ledger](02_LITERATURE.md), LipCache, Wei & Liu, NoScope, DeltaCNN, STBP
và time-series smoothing giải quyết các phần liên quan với guarantee khác.
Không nhận first temporal reuse, first structured verifier hoặc first
class-pair construction. Candidate contribution là tổ hợp direct-backbone
certificate + event region + anchor lifecycle + measured full cost.
C11 novelty/C12 net-saving vẫn UNVERIFIED. Reading list P01–P23 chưa được
đọc đầy đủ; metadata check không phải full review.

## 5. Formulation và phương pháp đề xuất

Raw input x∈R^768, normalization N(x)=A(x−m) train-fit cố định,
normalized z=N(x), reference F=f_θ∘N. Anchor x^r phải chạy thật qua F;
a=argmax F(x^r).

G3 đầu tiên đề xuất normalized diagonal domain:

    Z_e(z^r) = { z^r + B_e u : ||u||∞ ≤ 1 },
    B_e = diag(b), b_i ≥ 0; S_e = N^−1(Z_e).

Với mọi j≠a, nếu ℓ_aj là valid lower bound của
g_aj(z)=f_θ,a(z)−f_θ,j(z) trên toàn Z_e và min_j ℓ_aj>0, mọi điểm trong miền
có unique winner a. Proof: logit a lớn hơn từng logit đối thủ. Đây là điều
kiện đủ, không tự chứng minh floating-point implementation sound.

Nếu g_aj(z)≥αᵀz+β là valid affine relaxation trên đúng domain:

    ℓ_aj = αᵀz^r + β − ||B_eᵀα||₁.

Global radius ρ dùng B_e=ρI; I chỉ cho unit box. Không dùng coefficients
hợp lệ trên miền nhỏ cho miền lớn hơn khi chưa chứng minh validity.

PCA audit dùng centered raw delta δ−μ, score box và residual **L2**.
Draft cũ dùng uncentered residual coordinate-box; hai region khác nhau.
Theory hiện phân biệt chúng, không đổi source audit hoặc gán PCA coverage
cho verifier chưa có.

### Reuse gate và lifecycle

1. Chọn candidate từ thông tin runtime có sẵn.
2. Tìm certificate đúng anchor/checkpoint/preprocessing/domain/verifier.
3. Kiểm membership và mọi stored lower margin dương.
4. Đủ mọi điều kiện mới reuse; fail thì full inference.
5. Chỉ input đã full inference mới có thể thành trusted anchor mới.
6. Reset ở recording boundary; invalidate certificate khi identity thay đổi.

Numerical assumptions, tie/NaN handling và validity của computed bounds là
đầu ra G3. Zero mismatch chỉ là test evidence. Quantile chọn candidate không
tự tạo decision/statistical certificate.

## 6. Kết quả baseline đang có

Các bảng lấy từ JSON/source, chưa rerun các experiment. Split subject/
recording disjoint, chỉ một split cố định.

| Denominator | Train | Val | Test | Tổng |
|---|---:|---:|---:|---:|
| Stream windows | 9.978 | 2.422 | 5.053 | 17.453 |
| Pure K=6 | 5.693 | 1.450 | 2.874 | 10.017 |
| Subjects | 17 | 4 | 9 | 30 |

Backbone thực tế: Conv1d 6→32→64→64, kernels 5/3/3, BN/ReLU, MaxPool sau
hai conv đầu, Flatten 2048→Linear 128→ReLU→Linear 6; **282.918 params**.
Proposal no-BN khoảng 24k không phải model hiện tại.

| Seed | Epoch chọn | Val macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|---:|
| 42 | 4 | 0,9454 | 0,9105 | 0,9085 |
| 43 | 3 | 0,9418 | 0,8906 | 0,8897 |
| 44 | 25 | 0,9504 | 0,9354 | 0,9349 |

Chọn checkpoint bằng validation. Mean ± population SD qua 3 seeds cùng split:
macro-F1 0,9122±0,0183, accuracy 0,9110±0,0185; không phải CI.
Test metrics trên **2.874 pure K6 windows**. Full test stream **5.053 windows**
dùng được cho prediction agreement; accuracy cần ground truth/scope tương ứng.
Script09 tính lại val/test từ selected checkpoints, không gọi workflow là
test chỉ được truy cập một lần.

| Candidate membership | Mọi test pair | Same-activity | Label-change |
|---|---:|---:|---:|
| Diagonal q75 | 25,53% | 45,81% | 0,21% |
| Diagonal q90 | 31,94% | 52,54% | 2,26% |
| PCA 108 components | 35,42% | Chưa lưu riêng | Chưa lưu riêng |

Có 17.392 adjacent pairs, test 5.034. PCA hơn q90 **3,48pp**, chưa matched
budget/coverage. Đây không phải certified reuse với anchor giữ nhiều query.

CPU batch1 saved forward: mean 0,2511 ms, p50 0,2357 ms, p95 0,3313 ms;
6 Torch threads, 200 warmups, 1.000 runs. Chỉ timing model(x) trên tensor
normalized resident RAM. Chưa tính windowing/event/membership/construction/
cache/rebuild/I/O; thiếu CPU model/OS trong artifact.

## 7. Cost model và phân tích

Mô hình trung bình đơn giản, giả sử forward cost tương đối đồng nhất:

    E[T] ≈ T_e + T_m + (1−r)T_f + qT_b
    net saving khi rT_f > T_e + T_m + qT_b.

T_e: mean event; T_m: mean lookup/membership/validity; T_f: mean forward;
r: reuse fraction; q: rebuild count/window; T_b: mean cost trên rebuilds.
Initial construction phải được tính trong count hoặc thêm T_init/N.
Shared preprocessing vẫn timing trên cả hai pipeline.

Với mean 0,2511 ms và **giả sử** reuse 50%, ngân sách overhead 125,55 µs/window.
Đây là sensitivity calculation, chưa phải measured reuse. p95 cần complete
trace. Stride 1.280 ms cho ratio 0,2511/1280×100≈0,0196%; chưa có evidence
áp lực deadline single-stream. Giá trị đa stream/device/energy cần đo riêng.

## 8. Bốn đầu ra G3 và kế hoạch đánh giá

| Đầu ra | Điều kiện |
|---|---|
| Domain contract | Cùng coordinates, normalization, model/anchor/domain identity |
| Minimal diagonal verifier | Valid lower bounds toàn domain frozen CNN; numerical contract |
| Certified reuse gate | Membership + validity + identity; fail fallback; anchor lifecycle/trace |
| Component instrumentation | Construction, membership, fallback, inference, initial/rebuild costs |

G4 kiểm value của structure ở matched comparisons. G5 kiểm net saving từ
complete end-to-end traces; G3 instrumentation chưa đủ. G6 hoàn chỉnh report/
reproducibility. Baselines: B0 full, B2 confidence, B3 global Lipschitz,
B5 diagonal CROWN, B6 structured CROWN, tùy feasibility. Tune validation,
test policy cố định; báo reuse denominator, mismatches, mean/p50/p95,
rebuild/anchor age, memory và exclusions. Zero reused queries cho empirical
agreement **N/A**, không phải 100%.

## 9. Giới hạn và kết luận

Saved runs chưa reproduce trong lượt này; G3 chưa implementation/numerical
soundness; một dataset/fixed split/K6 model; CPU forward-only chưa đại diện
end-to-end/device; literature chưa đủ chứng minh novelty. Ground-truth point
margin không phải certified radius.

ECDR có cơ sở thử feasibility, chưa chứng minh phương pháp tăng tốc thành
công. Đi bằng domain contract và minimal diagonal verifier trước PCA expansion;
kết quả âm về coverage hoặc overhead vẫn phải báo.

Nguồn: [metrics](data/g2_metrics.json), [training](data/g2_training.json),
[model](data/g2_model_summary.json), [latency](data/g2_cpu_latency.json),
[event audit](data/event_audit.json), [split](data/split_audit.json),
[theory](04_THEORY.md), [G3 contract](07_G3_DOMAIN_CONTRACT.md),
[backbone source](scripts/backbone.py), [bibliography](refs.bib).
Tình trạng đồng bộ: [DOCS_SYNC_2026-10-22.md](DOCS_SYNC_2026-10-22.md).
