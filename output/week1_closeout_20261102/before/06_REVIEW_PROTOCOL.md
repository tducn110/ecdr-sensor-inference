# 06 — Protocol tổng quan tài liệu cho đề tài hiện tại

Ngày khởi động: 21/10/2026; đồng bộ tài liệu: 22/10/2026 (Asia/Ho_Chi_Minh).
Trạng thái: PROTOCOL_CREATED / PILOT; đã thử quy trình với P04 NoScope; chưa
hoàn thành review. Lượt đồng bộ 22/10 không tạo thêm full-paper reading,
screening toàn corpus, benchmark hay kết quả systematic review.

## 1. Đích và câu hỏi

Đầu ra đang xây dựng là phần tổng quan tài liệu cho **event-conditioned certified
decision reuse trong sequential neural inference**, theo RQ của
[01_RQ_GAP.md](01_RQ_GAP.md). Bản hiện tại là **tổng quan phê bình có ghi lại cách
tìm và chọn tài liệu**. Chưa gọi là systematic review: chưa có export kết quả từ
nhiều database, sàng lọc toàn bộ, hay số lượng dedup/exclusion có thể kiểm lại.

Câu hỏi dẫn dắt review:

1. Những phương pháp nào tránh chạy lại backbone trên đầu vào liên tiếp?
2. Mỗi phương pháp bảo đảm điều gì, cho model nào và trên miền đầu vào nào?
3. Temporal structure được biểu diễn và kiểm membership như thế nào?
4. Tiết kiệm được đo ở verification, inference, lookup hay toàn hệ thống?
5. Tổ hợp nào còn là giả thuyết cần chứng minh và thực nghiệm trong G3–G5?

Không dùng độ trôi chảy của bài viết, số paper, hoặc không tìm thấy một kết quả
search làm bằng chứng novelty.

## 2. Nguồn chuẩn và trách nhiệm

| Nguồn | Trách nhiệm |
|---|---|
| [00_START_HERE.md](00_START_HERE.md), [GATES.md](GATES.md) | Trạng thái và gate của nghiên cứu |
| [01_RQ_GAP.md](01_RQ_GAP.md) | RQ và gap hypothesis |
| [02_LITERATURE.md](02_LITERATURE.md) | Claim ledger và so sánh prior work |
| [refs.bib](refs.bib) | Metadata thư mục, URL, version và locator |
| [05_READING_PLAN.md](05_READING_PLAN.md) | Inventory P01–P23, baseline và note template |
| [notes/](notes/) | Evidence đọc từng paper và nhật ký search |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [report.md](report.md), [latex/dossier.tex](latex/dossier.tex) | Diễn giải/báo cáo hiện tại; mọi claim vẫn trace về ledger, nguồn và artifact |
| [07_G3_DOMAIN_CONTRACT.md](07_G3_DOMAIN_CONTRACT.md) | Định nghĩa domain/membership cần đối chiếu với prior work và G3 |
| [DOCS_SYNC_2026-10-22.md](DOCS_SYNC_2026-10-22.md) | Phạm vi và checks của lần đồng bộ tài liệu hiện tại |
| Protocol này | Cách tìm, chọn, đọc, tổng hợp và kiểm bài review |

`report.md` hiện là hồ sơ tổng hợp đã cập nhật; bản root report cũ còn trong Git
history. Các file trong `GPT-sources/` là tài liệu tổng hợp nội bộ, không thay
thế primary sources; bản ghi ngày 16/10 có banner snapshot/superseded. Nhật ký
[21/10](notes/2026-10-21_REVIEW_KICKOFF.md) và search log của ngày đó giữ nguyên
provenance; không dùng như xác nhận mới về skill installation hoặc environment.
Quy trình này không sửa checkpoint, config, training, verifier hoặc số liệu đo.

Trạng thái nghiên cứu từ artifact: G0/G2 repo-recorded PASS, G1
COMPLETE_WITH_REVISIONS, G3–G6 PENDING. Backbone hiện tại là SmallCNN1D K=6,
282.918 tham số, có BatchNorm và MaxPool; đây là context để chọn prior work và
baseline, không phải kết quả mới của literature pilot. G0 đủ hồ sơ để tiếp tục
prototype không đồng nghĩa review đã đầy đủ hay C11 novelty đã được chứng minh.

## 3. Phạm vi tìm và chọn

Corpus khởi đầu: P01–P23 trong reading plan, không tự coi 23 mục là đầy đủ.
Không đặt mốc năm bắt đầu để tránh loại prior art nền tảng; ghi ngày search và
version mỗi nguồn. Ngày 21/10/2026 là ngày thao tác, không bảo đảm index đã bao
phủ mọi công trình đến ngày đó.

**Đưa vào** paper có đóng góp trực tiếp cho ít nhất một nhóm: certified caching;
incremental/online verification; conditional temporal inference; deterministic
bound propagation; time-series certification; hoặc evaluation cần cho RQ.
Giữ preprint với nhãn/version riêng. Dataset/API docs là technical references,
không đếm như paper thực nghiệm.

**Loại khỏi synthesis chính** bài chỉ trùng từ khóa nhưng không liên quan tới
prediction preservation, perturbation sets hay chi phí liên quan. Blog, AI
summary và trang tổng hợp có thể giúp discovery; claim khoa học phải quay về
paper/proceedings hoặc tài liệu chính thức. Nguồn chỉ đọc abstract giữ nhãn
ABSTRACT_ONLY; không suy rộng sang proof hay benchmark.

Khi có preprint và proceedings cùng công trình, liên kết thành một study record,
giữ mọi version cần giải thích khác biệt. Dedup ưu tiên DOI, arXiv ID và title +
authors; không gộp chỉ vì title gần giống. Ghi lý do loại và metadata conflict.

## 4. Search protocol

Đợt khởi động 21/10 thực dùng **web search và mở nguồn gốc**. Chưa truy vấn trực tiếp
ACM DL, IEEE Xplore, Scopus hay Web of Science. Không biến domain filter của web
search thành bằng chứng đã search một database.

Các nhóm truy vấn đề xuất cho lượt mở rộng, cần dịch theo syntax từng nguồn:

| Nhóm | Truy vấn gợi ý |
|---|---|
| Certified reuse | `("certified caching" OR "certified reuse" OR "decision reuse") AND (neural OR inference)` |
| Verification reuse | `("online verification" OR "incremental verification") AND "neural network"` |
| Temporal skipping | `("temporal redundancy" OR "frame difference" OR "conditional inference") AND neural` |
| Structured bounds | `("bound propagation" OR "Lipschitz" OR "CROWN") AND (structured OR temporal OR margin)` |
| Time series | `("time series" OR "human activity recognition") AND (certification OR robustness OR overlap)` |

Sau broad search: tra exact title, đọc backward references và forward citations
của các seed quan trọng. Đây là bước còn phải làm, chưa tính vào coverage.

Mỗi lượt lưu: ngày, công cụ/platform, exact query, filter, nguồn trả về đã mở,
version, quyết định và lý do. Nếu không có export/count đầy đủ, ghi `null` và
`NOT_EXPORTED`, không điền số ước lượng. Dùng search log mới cho từng lượt để
giữ lịch sử thực hiện; không ghi đè log ngày 21/10.

PRISMA 2020 cung cấp checklist và flow diagram cho báo cáo systematic review;
PRISMA-S có 16 mục về báo cáo literature search. Tham khảo phần minh bạch phù
hợp, không tự gắn nhãn tuân thủ: [PRISMA 2020](https://www.prisma-statement.org/prisma-2020),
[PRISMA-S](https://www.prisma-statement.org/prisma-search).

## 5. Đọc, trích xuất và đánh giá

Dùng đúng 10 phần của note template trong reading plan. Mỗi note phải có source
URL + version + printed page/section; phân biệt trang in và số trang PDF.

Thêm vào bảng so sánh những trường cần cho RQ:

- Task/data/model và phạm vi được evaluate.
- Đối tượng được bảo đảm: backbone, proxy/guard hoặc label thống kê.
- Loại guarantee: formal deterministic, probabilistic, empirical hoặc chưa rõ.
- Miền perturbation và điều kiện membership; giả định distribution/lifecycle.
- Thứ được reuse; fallback và điều kiện invalidation.
- Metric/denominator, hardware, latency unit, setup/rebuild và chi phí bị loại.
- Artifact/code availability, phiên bản đã đọc, limitation và reproduction status.

Đánh giá từng trường bằng `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNVERIFIED`,
`NOT_SUPPORTED` hoặc `NOT_APPLICABLE`, kèm lý do. Không cộng điểm chất lượng tùy
ý giữa các study khác task. AI/source inspection và kiểm tra của người viết review là
hai bước riêng; chưa có human acceptance cho note mới.

## 6. Cách viết synthesis

Bố cục đề xuất cho bản review:

1. Problem, scope và định nghĩa prediction agreement/correctness/certification.
2. Cách tìm và chọn tài liệu, nguồn dữ liệu, giới hạn coverage.
3. Certified caching và model được bảo đảm.
4. Incremental verification và phần computation được tái sử dụng.
5. Conditional temporal inference và trade-off sai số/chi phí.
6. Structured/class-pair bounds và time-series perturbations.
7. So sánh bằng ma trận chung; open problems và gap hypothesis.
8. Threats to validity, kết luận có giới hạn và bibliography.

Viết theo **vấn đề và trục so sánh**, không nối 23 abstract. Mỗi đoạn cần có
claim, nguồn hỗ trợ, điều kiện/giới hạn và ý nghĩa đối với RQ. Tách phát hiện
của paper khỏi suy luận của người review. Không đặt throughput khác hardware,
speedup verification và latency inference cạnh nhau như cùng metric.

## 7. Điều kiện hoàn thành

| Check | Evidence cần có | Trạng thái tại lần đồng bộ 22/10 |
|---|---|---|
| Inventory P01–P23 | Reading plan + mapping note/bibliography | Có inventory; mới có note P04 |
| Search tái kiểm được | Query/platform/date + export hoặc giới hạn rõ | Có pilot log; chưa có corpus export |
| Screening và dedup | Quyết định + lý do cho mỗi record | Chưa thực hiện đầy đủ |
| Claim có evidence | URL/version/locator + bibliography | P04 có claim hẹp; corpus còn phải đọc |
| Synthesis đúng RQ | Ma trận guarantee/domain/reuse/cost | Khung đã có; chưa viết xong |
| Citation và số liệu | Metadata, units, denominator, scope khớp nguồn | Kiểm theo từng paper |
| Review hoàn chỉnh | Toàn văn + limitations + references | Chưa hoàn thành; không tự đánh dấu ACCEPTED |

Bước đọc ưu tiên tiếp theo theo nguy cơ ảnh hưởng gap: P01/P03 → P08/P09/P10 →
P02/P11/P17 → P16 và nhóm systems/evaluation. P04 đã có note claim hẹp; việc
đọc sâu/reproduce vẫn mở. C11 novelty và C12 net saving vẫn UNVERIFIED.
