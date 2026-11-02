<!-- trustmebro -->
# BẢNG QUẢN LÝ TIẾN ĐỘ & HỒ SƠ NGHIÊN CỨU (TASK MANAGEMENT BOARD - TMB)

> **Dự án**: Event-Conditioned Certified Decision Reuse (ECDR) in Sensor Time-Series Streams  
> **Thư mục gốc**: `/home/pro/Downloads/research`  
> **Quy chuẩn áp dụng**:
> - **TrustMeBro (TMB) Engine Standard**: Quản lý trạng thái theo Gate/Phase thực chứng (`[x]` kèm bằng chứng file/số đo, `[~]` đã viết chưa kiểm chứng, `[ ]` đang mở).
> - **Ponytail Minimalist / Senior Rigor**: Zero-bloat, YAGNI, không bịa benchmark, phân định rõ ràng giữa giả thuyết nghiên cứu (Research Hypothesis) và kết quả đã kiểm chứng (Verified Evidence).
> - **Evidence-First Architecture**: Báo cáo tổng hợp phải ánh xạ 1-1 với code thực thi, checkpoint thật, và artifact kiểm toán dữ liệu.
> **Khởi tạo**: 2026-10-29  
> **Trạng thái Gate**: ✅ G0 (PASS) | ✅ G1 (COMPLETE_WITH_REVISIONS) | ✅ G2 (PASS) | ⏳ G3–G6 (PENDING)

---

## 1. Trạng Thái Tổng Thể (Status Overview)

```
[████████░░░░░░░░░░░░] 42.8% Hoàn thành (3/7 Research Gates hoàn tất)
```

| Gate / Phase | Trọng tâm nghiên cứu | Trạng thái | Bằng chứng kiểm chứng (Evidence Artifacts) |
| :--- | :--- | :--- | :--- |
| **G0. RQ & Literature Baseline** | Xác lập câu hỏi nghiên cứu, phân tích khoảng trống (Gap), đối chiếu 4 trục | ✅ **PASS** | [`01_RQ_GAP.md`](01_RQ_GAP.md), [`02_LITERATURE.md`](02_LITERATURE.md), [`refs.bib`](refs.bib), [`notes/P04_NoScope.md`](notes/P04_NoScope.md) |
| **G1. Data Feasibility & Audit** | Kiểm toán tập dữ liệu UCI HAPT (61 recordings, 12 classes), sửa bug index dòng | ✅ **REVISED** | [`03_DATA_AUDIT.md`](03_DATA_AUDIT.md), [`scripts/01_load_hapt.py`](scripts/01_load_hapt.py) đến [`06b`](scripts/06b_temporal_clustering_audit.py) |
| **G2. Frozen Backbone Baseline** | Huấn luyện SmallCNN1D (K=6 pure-only, 282k tham số), đo trễ CPU forward | ✅ **PASS** | [`G2_CLASS_DECISION.md`](G2_CLASS_DECISION.md), [`scripts/backbone.py`](scripts/backbone.py), [`checkpoints/backbone_best.pt`](checkpoints/backbone_best.pt) |
| **G3. Minimal Verifier & Contract** | Thiết lập miền đầu vào chuẩn hóa, chứng minh cận biên lớp (class-pair margins) | ⏳ **PENDING** | [`07_G3_DOMAIN_CONTRACT.md`](07_G3_DOMAIN_CONTRACT.md), [`04_THEORY.md`](04_THEORY.md), [`scripts/11_gate_a_prototype.py`](scripts/11_gate_a_prototype.py) |
| **G4. Event Structure Ablation** | Kiểm chứng tính thắt chặt (tightness) và giá trị của miền event-conditioned vs ball/box | ⏳ **PENDING** | Kế hoạch trong [`roadmap suggestion.md`](roadmap%20suggestion.md), baseline B0–B7 |
| **G5. End-to-End Runtime System** | Tích hợp Anchor Cache, Descriptor Lookup, Fallback & Rebuild đo đạc thực tế | ⏳ **PENDING** | Yêu cầu đo trễ chuỗi hoàn chỉnh (Full stream inference latency) |
| **G6. Scientific Paper & Artifact** | Hoàn thiện bài báo khoa học, gói tái lập mã nguồn và chứng minh novelty | ⏳ **PENDING** | [`report.md`](report.md), [`latex/dossier.tex`](latex/dossier.tex), bộ chuyên đề lý thuyết `me/` |

---

## 2. Danh Bạ Cốt Lõi Hệ Thống (Zero-Bloat Asset Registry)

| Hạng mục | Tệp tin / Đường dẫn | Trách nhiệm duy nhất (Single Responsibility) | Trạng thái xác thực |
| :--- | :--- | :--- | :--- |
| **Master Reading Guide** | [`README.md`](README.md) | Bản đồ tri thức khách hàng, hành trình đọc tuyến tính & đồ thị phụ thuộc | ✅ Đã ban hành |
| **Start Point** | [`00_START_HERE.md`](00_START_HERE.md) | Tóm tắt hiện trạng, số liệu checkpoint K=6 và cảnh báo ranh giới | ✅ Đã đồng bộ 22/10 |
| **Problem & Gaps** | [`01_RQ_GAP.md`](01_RQ_GAP.md) | 2 câu hỏi nghiên cứu cốt lõi (C11 Novelty, C12 Net Savings) | ✅ Đã xác nhận |
| **Prior Art Taxonomy** | [`02_LITERATURE.md`](02_LITERATURE.md) | Bảng phân loại 4 trục (Context, Method, Bounds, Reuse) | ✅ Đã rà soát |
| **Data Integrity Audit** | [`03_DATA_AUDIT.md`](03_DATA_AUDIT.md) | Audit 12 classes HAPT, sửa lỗi F1-F3 và phân tích window overlap | ✅ Đã đối chiếu UCI |
| **Theoretical Bounds** | [`04_THEORY.md`](04_THEORY.md) | Định nghĩa toán học về Event Region, Safety Certificate, Soundness | ✅ Đã hoàn thiện draft |
| **Literature Roadmap** | [`05_READING_PLAN.md`](05_READING_PLAN.md) | Danh mục 23 bài báo khoa học (P01–P23) và 8 baselines (B0–B7) | ✅ Đã phân tầng ưu tiên |
| **Review Protocol** | [`06_REVIEW_PROTOCOL.md`](06_REVIEW_PROTOCOL.md) | Quy chuẩn trích dẫn, chống hallucination trong nghiên cứu | ✅ Đã chuẩn hóa |
| **Domain Contract G3** | [`07_G3_DOMAIN_CONTRACT.md`](07_G3_DOMAIN_CONTRACT.md) | Khế ước bất biến giữa Preprocessing, Normalization và Verifier | ✅ Đã chốt hợp đồng |
| **Class Decision K=6** | [`G2_CLASS_DECISION.md`](G2_CLASS_DECISION.md) | Quyết định đóng băng K=6 pure-only, hold K=12 do class support thấp | ✅ Đã phê duyệt |
| **Evidence Gates** | [`GATES.md`](GATES.md) | Tiêu chí nghiệm thu định lượng từ Gate G0 đến G6 | ✅ Đã cập nhật 22/10 |
| **Unified Dossier** | [`report.md`](report.md) | Báo cáo nghiên cứu tổng thể, bảng hypothesis ledger | ✅ Bản mới nhất |
| **Deep Theory Bundle** | [`me/README.md`](me/README.md) | Bộ 3 tuần lý thuyết + 7 chuyên đề toán/dữ liệu độc lập | ✅ Đã compile PDF/TeX |

---

## 3. Nhật Ký Tiến Độ Chi Tiết Theo Gate (Gate-by-Gate Audit Log)

### Gate G0: Baseline Tài Liệu & Xác Lập Bài Toán (Status: ✅ PASS)
- [x] **RQ Formulation**: Xác lập RQ1 (Event-conditioned bounds có certify được frozen backbone không?) và RQ2 (Net speedup có dương sau khi trừ chi phí verifier không?).
- [x] **Taxonomy 4 trục**: Phân loại đối chiếu với Caching thông thường, Global Lipschitz, IBP/LiRPA và Early-exit.
- [x] **NoScope Pilot Audit**: Phân tích bài báo NoScope ([`notes/P04_NoScope.md`](notes/P04_NoScope.md)) làm tiền đề cho reuse filter.
- [x] **Exit criteria**: Đủ hồ sơ RQ/Gap và bibliography 10 entry cốt lõi trong [`refs.bib`](refs.bib).

### Gate G1: Kiểm Toán Khả Thi Dữ Liệu UCI HAPT (Status: ✅ COMPLETE_WITH_REVISIONS)
- [x] **Data Pipeline**: Triển khai [`scripts/01_load_hapt.py`](scripts/01_load_hapt.py) đến [`06b`](scripts/06b_temporal_clustering_audit.py).
- [x] **Sửa lỗi Script 05 (Row Index Bug)**: Phát hiện và sửa lỗi lệch window index khiến 98.2% cặp đọc nhầm recording exp01.
- [x] **Kiểm toán tương quan chuỗi**: Đo đạc lag-1 ($r \approx 0.985$) và lag-64 ($r \approx 0.720$), chứng minh giả định i.i.d. thất bại.
- [x] **PCA Coverage vs Residual**: Phân tích không gian 768 chiều ($128 \times 6$), chỉ ra rằng độ biến thiên giải thích (explained variance) không đồng nghĩa với cận sai số (residual bound).
- [x] **Exit criteria**: Bộ dữ liệu chia train/val/test theo người dùng (Disjoint subjects: Train 21, Val 3, Test 6), xác định 2.874 pure test windows.

### Gate G2: Đóng Băng Mô Hình Nền SmallCNN1D (Status: ✅ PASS)
- [x] **Kiến trúc mô hình**: Triển khai `SmallCNN1D` trong [`scripts/backbone.py`](scripts/backbone.py): 3 Conv1d tầng (`6→32→64→64`), BatchNorm, ReLU, 2 MaxPool, Flatten 2048, Linear 128, Linear 6.
- [x] **Tổng số tham số**: 282.918 tham số (được kiểm chứng độc lập, bác bỏ con số 24k không BatchNorm/GAP ở đề xuất cũ).
- [x] **Huấn luyện đa hạt giống**: 3 seeds (42, 43, 44), macro-F1 test đạt $0.9122 \pm 0.0183$. Seed 44 đạt Validation Accuracy 0.9504, Test Accuracy 0.9349, Macro-F1 0.9354.
- [x] **Đo độ trễ cơ sở**: Đo forward CPU batch=1 trên RAM tensor: mean 0.2511 ms, p50 0.2357 ms, p95 0.3313 ms ([`data/g2_cpu_latency.json`](data/g2_cpu_latency.json)).
- [x] **Quyết định lớp K=6**: Phê duyệt chính sách K=6 pure-only do các lớp chuyển tiếp (postural transitions) chỉ có dưới 2% mẫu, không đủ hỗ trợ chứng nhận an toàn.
- [x] **Exit criteria**: Checkpoint tốt nhất được lưu tại [`checkpoints/backbone_best.pt`](checkpoints/backbone_best.pt), tham số và preprocessing được đóng băng.

### Gate G3: Khế Ước Miền & Trình Chứng Nhận Tối Thiểu (Status: ⏳ PENDING)
- [x] **Soạn thảo khế ước miền**: Ban hành [`07_G3_DOMAIN_CONTRACT.md`](07_G3_DOMAIN_CONTRACT.md) quy định rõ quan hệ giữa raw sensor, z-score normalization và perturbation domain.
- [x] **Bóc tách lý thuyết toán**: Hoàn thiện 7 tài liệu bổ trợ chuyên sâu trong `me/` giải thích IBP, CROWN, Norm Geometry, Wave-Domain $T_S$.
- [ ] **Triển khai Auto_LiRPA / IBP Engine**: Xử lý layer BatchNorm và MaxPool của mô hình K=6 mà không làm lỏng cận biên.
- [ ] **Chứng minh 5 cặp biên an toàn (Class-Pair Lower Margins)**: Đảm bảo $\min_{k \neq y} (z_y - z_k) > 0$ trên toàn bộ miền $\mathcal{D}$.
- [ ] **Cơ chế Anchor Invalidation**: Ngăn chặn hiện tượng stale certificate khi chuyển recording hoặc drift tín hiệu.
- [ ] **Exit criteria**: Trình chứng nhận chạy được trên CPU, không có vi phạm an toàn nào trên tập kiểm thử (Zero false positive certificates).

### Gate G4: Đánh Giá Đóng Góp Của Event-Conditioned Domain (Status: ⏳ PENDING)
- [ ] **Ablation Cấu trúc miền**: So sánh Event Region (PCA score box + residual ellipsoid) vs Global $L_\infty$ Box vs Global $L_2$ Ball.
- [ ] **Đo tỷ lệ Certified Reuse**: Đo lường % cửa sổ thực sự được tái sử dụng quyết định mà không cần chạy lại backbone.
- [ ] **So sánh với Baselines**: Đối chiếu với LipCache (Xiang et al., 2026) và Early-Exit/Quantization.
- [ ] **Exit criteria**: Chứng minh được Event Conditioned Domain đem lại thể tích chứng nhận hẹp hơn nhưng độ phủ hữu ích cao hơn so với $L_\infty$ ngây thơ.

### Gate G5: Hệ Thống Thực Thi & Tối Ưu Tốc Độ Thực Tế (Status: ⏳ PENDING)
- [ ] **Tích hợp Pipeline End-to-End**: Descriptor extraction $\rightarrow$ Anchor lookup $\rightarrow$ Membership check $\rightarrow$ Certificate validation $\rightarrow$ Fallback inference.
- [ ] **Đo trễ chuỗi hoàn chỉnh (Wall-clock Stream Latency)**: Tính toán điểm hòa vốn (Break-even condition):
  $$\tau_{\text{reuse}} = \tau_{\text{descriptor}} + \tau_{\text{lookup}} + \tau_{\text{membership}} < \tau_{\text{backbone}}$$
- [ ] **Phân tích độ nhạy (Sensitivity Analysis)**: Xác định tỷ lệ reuse tối thiểu để đạt tiết kiệm năng lượng ròng.
- [ ] **Exit criteria**: Thời gian thực thi trung bình toàn chuỗi thấp hơn chế độ chạy 100% backbone.

### Gate G6: Đóng Gói Nghiên Cứu & Bài Báo Khoa Học (Status: ⏳ PENDING)
- [ ] **Biên soạn bài báo chính**: Đồng bộ nội dung vào [`latex/dossier.tex`](latex/dossier.tex) và [`report.md`](report.md).
- [ ] **Gói tái lập (Reproducibility Package)**: Toàn bộ script 01–11 có thể chạy một lệnh từ đầu với kết quả khớp 100% hash.
- [ ] **Bảo vệ Novelty & Claim C11**: Khẳng định sự khác biệt với các phương pháp caching hình ảnh/video nhờ tính chất wave-domain của tín hiệu cảm biến đeo.
- [ ] **Exit criteria**: Hoàn tất bản thảo hội nghị/tạp chí với toàn bộ chứng minh toán học và số liệu thực nghiệm.

---

## 4. Hành Động Tiếp Theo Theo Thứ Tự Tuyến Tính (Linear Next Actions)

1. [ ] **Thực thi script mẫu Gate 3 ([`scripts/11_gate_a_prototype.py`](scripts/11_gate_a_prototype.py))**: Kiểm tra tính tương thích giữa tensor normalize và cận IBP trên mô hình K=6 thực tế.
2. [ ] **Đo đạc chi phí Membership Check**: Viết benchmark đo vi giây ($\mu\text{s}$) của hàm kiểm tra $x \in \mathcal{D}(x_0)$ trên CPU 1 nhân.
3. [ ] **Đối chiếu với 7 tài liệu Supplement trong `me/`**: Đảm bảo các công thức giải tích trong code khớp 100% với [`me/ECDR_Data_Analysis_Supplement_Certification_Methods.pdf`](me/ECDR_Data_Analysis_Supplement_Certification_Methods.pdf).

---

## 5. Nhật Ký Thay Đổi (Change Log)

- **2026-10-29**: Tối ưu dọn dẹp dung lượng (xóa tệp trùng lặp LipCache 9MB, gom papers và context data), xây dựng cấu trúc phân tầng tri thức 5 Tiers (`00_OVERVIEW` đến `04_SYNTHESIS`) và 3 Thư viện chuyên dụng (`PDF_LIBRARY`, `LATEX_LIBRARY`, `SCRIPTS_LIBRARY`) với 137 verified symlinks; bổ sung Instruction 0 giải nghĩa các tiền giả định và ký hiệu toán học cốt lõi.
- **2026-10-29**: Khởi tạo bảng quản lý tiến độ chuẩn **TrustMeBro** (`TMB.md`), đồng bộ hiện trạng Gate G0–G2 hoàn tất, định hình khế ước G3 và liên kết toàn bộ tài liệu nghiên cứu.
- **2026-10-28**: Hoàn thành biên soạn và compile thành công 7 tài liệu giải thích chuyên sâu (SUPP-01 đến SUPP-07) trong thư mục `me/`.
- **2026-10-25**: Tạo gói dữ liệu ngữ cảnh đầy đủ `ECDR_full_context_chunks_20261025.jsonl` và audit nội bộ.
- **2026-10-22**: Đồng bộ hóa toàn bộ hồ sơ dự án, đính chính lỗi Script 05 trong `03_DATA_AUDIT.md`, đóng băng quyết định lớp `G2_CLASS_DECISION.md`.
