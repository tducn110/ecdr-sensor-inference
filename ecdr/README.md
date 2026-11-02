# ECDR (Event-Conditioned Certified Decision Reuse) — Báo Cáo Triển Khai & Kiểm Chứng Kỹ Thuật

> **Tài liệu quy chuẩn gốc**: `02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex`, `07_G3_DOMAIN_CONTRACT_VI.md`, `04_THEORY_VI.md`, `03_DATA_AUDIT_VI.md`.  
> **Trạng thái nghiên cứu**: `EXPERIMENTAL_HOLD` (K=12 do thiếu hụt mẫu chuyển tiếp / transition support).  
> **Bảo tồn phạm vi di sản**: 90 tệp gốc (bao gồm $K=6$ backbone tại `scripts/`, `configs/backbone.yaml`, `checkpoints/backbone_best.pt`, `data/g2_*.json`) được bảo tồn nguyên vẹn 100%.

---

## 1. Tổng Quan Kiến Trúc & Đặc Tả Mô Hình

Hệ thống ECDR triển khai mô hình mạng nơ-ron tích chập 1D lý thuyết (Conv1D-ReLU-GAP-Affine) phân loại 12 lớp hành vi trên tập dữ liệu HAPT (Human Activities and Postural Transitions), tuân thủ nghiêm ngặt Mục 6 trong `02_ECDR_CNN_Model_Analysis_VI.tex`:
- **Đầu vào**: Tín hiệu cảm biến chuyển động thô đa kênh `(6, 128)` (3 trục gia tốc tổng hợp `acc_x, acc_y, acc_z` và 3 trục con quay hồi chuyển `gyro_x, gyro_y, gyro_z` tại tần số 50 Hz).
- **Conv1**: Valid convolution ($p=0, s=1, d=1$), kernel $k=5$, 6 kênh vào $\to$ 32 kênh ra $\to$ kích thước `(32, 124)`.
- **ReLU1**: Hàm kích hoạt không tuyến tính $\max(0, x)$.
- **Conv2**: Valid convolution ($p=0, s=1, d=1$), kernel $k=3$, 32 kênh vào $\to$ 64 kênh ra $\to$ kích thước `(64, 122)`.
- **ReLU2**: $\max(0, x)$.
- **Conv3**: Valid convolution ($p=0, s=1, d=1$), kernel $k=3$, 64 kênh vào $\to$ 64 kênh ra $\to$ kích thước `(64, 120)`.
- **ReLU3**: $\max(0, x)$.
- **GAP**: Global Average Pooling trên chiều thời gian $T=120 \to$ vector đặc trưng `(64,)`.
- **Affine Head**: Ánh xạ tuyến tính $64 \to K=12$ logit đầu ra.

### Tham số và Chi phí Tính toán Lý thuyết
- **Số tham số**: $P(K) = 19,552 + 65 \times K$. Với $K=12$: **$P(12) = 20,332$ tham số** ($992 + 6,208 + 12,352 + 780$).
- **Chi phí MACs toàn phần**: $M(K) = 2,343,168 + 64 \times K$. Với $K=12$: **$M(12) = 2,343,936$ MACs**.
- **Receptive Field**: Conv1 = 5, Conv2 = 7, Conv3 = 9, GAP/Head = 128 mẫu.
- **Chi phí tính toán tăng dần (Incremental Shift $s=64$)**:
  - Tái sử dụng $120 - 64 = 56$ vị trí đặc trưng $H_3$ đã lưu trong bộ đệm.
  - Chỉ tính thêm 64 vị trí mới qua 3 tầng Conv: $64 \times (32 \times 6 \times 5 + 64 \times 32 \times 3 + 64 \times 64 \times 3) = 1,241,088$ MACs.
  - Head: $12 \times 64 = 768$ MACs.
  - Tổng cộng chế độ Incremental: **1,241,856 MACs** (tiết kiệm lý thuyết **46.84%** MACs mỗi cửa sổ trượt).

---

## 2. Chuẩn Số Học & Giả Định Chứng Nhận (Numerical Semantics & Assumptions)

ECDR định nghĩa một mô hình tham chiếu thực thi chuẩn mực trên CPU với NumPy float64 (canonical CPU NumPy64 reference) dưới các giả định số học đã nêu rõ (không tuyên bố loại trừ hoàn toàn mọi sai lệch giữa các trình biên dịch hoặc phần cứng GPU):

1. **Thực thi tuần tự không BLAS (`OperationalSequentialNumpyCNN`)**:
   - Sử dụng kiểu dữ liệu `numpy.float64`.
   - Vector hóa đồng thời số kênh xuất và các vị trí thời gian trong khi **bảo tồn vòng lặp tường minh trên các tích chập kênh nhập và taps** (`c in range(C_in)`, `r in range(k)`).
   - Mỗi phép nhân (`np.multiply`) và cộng tích lũy (`np.add`) được thực hiện tách biệt, không dùng FMA. Độ trễ thực thi tham chiếu đạt mức xấp xỉ **~3.6 – 4.2 ms / cửa sổ** trên CPU thử nghiệm.
   - Trọng số mô hình được phong tỏa bất biến (`flags.writeable = False`) và liên kết định danh bằng mã băm mật mã `model_id` (ràng buộc cấu trúc mạng, phiên bản thuật toán số học và thứ tự lớp HAPT cố định).

2. **Lan truyền khoảng số học ngoại suy (`Interval` & Directed Outward Rounding)**:
   - Các hàm số học ufunc chuẩn của NumPy thực thi ở chế độ làm tròn mặc định IEEE-754 (round-to-nearest). Việc làm tròn hướng ngoại (outward rounding) được thực hiện tường minh ở cấp độ phần mềm bằng cách áp dụng `np.nextafter(..., -np.inf)` cho cận dưới và `np.nextafter(..., np.inf)` cho cận trên ngay sau **từng** phép toán con riêng biệt (không dựa vào chế độ làm tròn phần cứng 'hardware directed ufuncs').
   - Giả định số học IEEE-754 binary64: Phép toán ufunc làm tròn chuẩn xác (round-to-nearest), các phép toán con tách rời hoàn toàn (không dùng FMA), hỗ trợ số phi chuẩn tuần tiến (gradual underflow, không áp dụng Flush-To-Zero / FTZ), và đầu vào hữu hạn không có overflow trước khi kích hoạt ReLU.
   - Hợp đồng quy nạp (Proof-by-Induction Contract): Bắt đầu từ hộp đầu vào hữu hạn $X \in [\underline{X}, \overline{X}]$, từng phép toán tuyến tính và phi tuyến bảo toàn tính chất bao bọc toán học thông qua làm tròn hướng ngoại.
   - Kiểm tra overflow trước khi kích hoạt ReLU: ngăn ngừa trường hợp tích âm cực lớn $-\infty$ bị hàm $\max(0, x)$ che giấu thành 0.0.

3. **Bằng chứng kiểm thử qua Phân số Hữu tỉ (`fractions.Fraction` Test Evidence)**:
   - Bộ kiểm thử thực thi 216 bất đẳng thức bao bọc hữu tỉ tuyệt đối trên các điểm mẫu được chọn ngẫu nhiên có hạt giống:
     $$\text{Fraction}(\underline{pre}) \le \text{exact\_lo} \le \text{exact\_hi} \le \text{Fraction}(\overline{pre})$$
   - Đây là **bằng chứng thực nghiệm (test evidence)** chứng minh thuật toán bao bọc đúng trên các cấu hình kiểm tra, không phải là một chứng minh toán học hình thức cho mọi miền tham số bất kỳ.

4. **Tình trạng của signed-q GAP delta**:
   - Công thức signed-q delta được duy trì riêng biệt dưới nhãn **non-proof diagnostic** (`signed_q_diagnostic_margins`), không bao giờ được dùng để cấp chứng chỉ tái sử dụng nếu không có bao bọc khoảng số học đầy đủ.

---

## 3. Quy Chuẩn Dữ Liệu & Denominators

Dữ liệu được nạp từ `data/windows.npz` (17,453 cửa sổ trượt, kích thước `(6, 128)`):
- **Nguyên tắc định danh dòng toàn cục**: Gán `global_row_id` ổn định (0 .. 17,452) **trước** bất kỳ thao tác tách/lọc/ghép dữ liệu nào. Không bao giờ lập chỉ mục cửa sổ bằng `win_idx` cục bộ của bản ghi ghi âm.
- **Mẫu số đánh giá dòng chảy đầy đủ (Full-Stream Denominators)**:
  - **Train stream**: 9,978 cửa sổ
  - **Validation stream**: 2,422 cửa sổ
  - **Test stream**: 5,053 cửa sổ
  - **Tổng**: 17,453 cửa sổ
- **Tập giám sát (Supervised Subsets)**:
  - Loại bỏ các cửa sổ mơ hồ (`label_policy != 'pure' & label_policy != 'majority'` hoặc `label_majority` không thuộc 1..12).
  - Kết quả phân bổ mẫu giám sát: **Train: 6,955**, **Val: 1,751**, **Test: 3,572**.
- **Trạng thái Hỗ trợ 12 Lớp (Class Support & Hold Status)**:
  - Chính sách `pure` gặp **18 tiêu chí chặn (criteria blockers)** trên 17 ô lớp/phân đoạn bị thiếu mẫu nghiêm trọng.
  - Chính sách `majority` gặp **10 ô lớp/phân đoạn không đạt ngưỡng**.
  - Toàn bộ pipeline $K=12$ được gắn cờ `EXPERIMENTAL_HOLD`, ngăn chặn việc ghi đè lên baseline $K=6$ (Gate 2).

---

## 4. Vòng Đời Chứng Chỉ & Bảo Vệ Cổng Xác Thực (Certificate Lifecycle)

Một chứng chỉ tái sử dụng (`Certificate`) chỉ hợp lệ khi vượt qua tất cả các lớp phòng thủ tại cổng (`validate()`):
1. **Kiểm tra biên độ toàn bộ 11 đối thủ cạnh tranh**:
   $$\ell_{cj}^{\text{direct}} = \underline{F}_c - \overline{F}_j > 0 \quad \forall j \in \{0, \dots, K-1\} \setminus \{c\}$$
2. **Tính nhất quán giữa đại lượng vô hướng và vector**:
   $$\min_{j \ne c} \ell_{cj}^{\text{direct}} = \text{min\_lower\_margin} > 0$$
3. **Mã băm nội dung mỏ neo (`anchor_state_hash`)**:
   - Được tính ngay tại thời điểm build proof từ mảng `float64` chuẩn hóa của cửa sổ mỏ neo thông qua `compute_window_hash(norm_window)`.
   - Phòng vệ trước kịch bản **Stale Wave Rebind**: Cơ chế đối soát mã băm tại `CertificateManager.establish_trusted_anchor` và cổng thực thi `StreamingReplayEngine` đảm bảo từ chối các trường hợp trùng ID nhưng khác biệt về nội dung mỏ neo (rejecting same-ID different-anchor-content) theo đúng hợp đồng kiểm định mã băm đã thiết lập.
4. **Mã băm toàn vẹn (`integrity_hash`)**:
   - Băm mật mã SHA-256 đóng băng toàn bộ các trường `model_id`, `prep_id`, `recording_id`, `anchor_id`, `target_class`, `anchor_state_hash`, `domain_box`, và vector biên độ. Mọi hành vi sửa đổi thuộc tính in-place đều làm sai lệch mã băm và bị hủy bỏ ngay lập tức.
5. **Cơ chế thiết lập lại (Recording & Continuity Reset)**:
   - Khi chuyển sang bản ghi ghi âm mới (`exp_id != current_exp`): Lập tức hủy chứng chỉ đang hoạt động và xóa bộ đệm mỏ neo.
   - Khi phát hiện gián đoạn chỉ số mẫu (`win_idx != prev_win_idx + 1`): Lập tức reset anchor và cache incremental.

---

## 5. Kết Quả Kiểm Chứng Độc Lập & Hệ Thống Test

### 1. Bộ kiểm thử đơn vị nội bộ (`tests/run_all_tests.py`)
- **Số lượng**: **57 / 57 tests PASS** (thời gian chạy ~1.6 giây; ban đầu là bộ 55 bài kiểm thử, đã mở rộng thành 57 sau khi bổ sung 2 bài kiểm tra chốt chặn tham số CLI `test_train_args_rejected` và `test_replay_args_rejected`).
- Bao gồm:
  - `test_model_spec.py`: 6 tests kiểm tra đặc tả mô hình, tham số, MACs và tính tương đương forward.
  - `test_toy_paper_worked_example.py`: 5 tests kiểm tra ví dụ số học Section 10 paper ($\ell_{12}=8, \ell_{13}=7/8$).
  - `test_toy_stale_regression.py`: 3 tests kiểm tra từ chối mỏ neo cũ (Section 10.2 regression).
  - `test_incremental_equivalence.py`: 4 tests kiểm tra tính bitwise-exact của bộ đệm gia tăng trên các bước dịch $s \in \{1, 64, 120\}$.
  - `test_intervals_soundness.py`: 8 tests kiểm tra tính bao bọc, directed rounding và chống alias vùng nhớ.
  - `test_bounds_all_competitors.py`: 6 tests kiểm tra chứng nhận 11 đối thủ cạnh tranh và phát hiện phản ví dụ làm tròn đường chéo.
  - `test_lifecycle_invalidation.py`: 10 tests kiểm tra vòng đời chứng chỉ, từ chối vector âm, kiểm tra tính nhất quán và từ chối tráo đổi nội dung mỏ neo (`test_establish_trusted_anchor_rejects_window_content_mismatch`).
  - `test_data_loader_denominators.py`: 5 tests kiểm tra mẫu số dòng chảy và phân tách subject độc lập.
  - `test_preprocessing_boundary.py`: 10 tests kiểm tra ranh giới tiền xử lý, loại trừ NaN/Inf, cùng chốt từ chối tham số đào tạo (`epochs <= 0`, `lr <= 0`/NaN, `batch_size <= 0`) và phát lại (`max_windows <= 0`).

### 2. Bộ kiểm thử độc lập (`output/ecdr/independent_evidence/check_core_36cases.py`)
- **Số lượng**: **36 / 36 cases PASS** (0 failures, thời gian chạy ~1.51 giây, `source_snapshot_stable: true`).
- Biên nhận độc lập chính thức: `output/ecdr/independent_evidence/results-final36.json`.
- Mã băm SHA-256 biên nhận: `e51a991d8592cbf3ceec310bad49f4e61fad4f191e318f585008bdf205193644`.

### 3. Kiểm chứng thực nghiệm mẫu đối chứng dương (Root Positive Controls on Final Guards)
- **Cấu hình Diagonal (shadow-off)**: 5 truy vấn / 2 bản ghi cho kết quả chính xác **2 hits / 2 resets**, `agreement = None` (do không bật shadow), chuỗi trạng thái thực thi: `[full, hit, incremental, full, hit]`.
- **Cấu hình Wave (shadow-on)**: 5 truy vấn / 2 bản ghi cho kết quả chính xác **2 hits / 2 resets**, kiểm tra shadow-agreement đạt **2/2 (100%)**, chuỗi trạng thái thực thi: `[full, hit, incremental, full, hit]`.
- Cửa sổ hằng số 0 thỏa mãn chính xác phần gối sóng với bộ nhớ đệm cũ nên kích hoạt chế độ incremental hợp lệ; hai kịch bản đối chứng này kiểm tra nhánh tái sử dụng hợp lệ (positive reuse path) và hành vi reset mỏ neo/bản ghi trong các điều kiện xác định, không đại diện cho một chứng minh tính đúng đắn phổ quát trên mọi phân phối đầu vào.

### 4. Biên nhận Đối soát Độc lập Trực tiếp từ Codex (Direct Codex Verification Receipts)
Toàn bộ các biên nhận đối soát độc lập được lưu trữ có kiểm chứng tại thư mục [`output/ecdr/verification/`](../output/ecdr/verification/):
1. **Dữ liệu & Mô hình**: [`data_metrics_codex_direct.json`](../output/ecdr/verification/data_metrics_codex_direct.json) (SHA-256: `c2968c302c193c243606e860bf992a145562eb7fc8faf6f9da03c54786f2a5e0`):
   - Căn chỉnh dòng chảy tuyệt đối trên cả 3 split (9978 / 2422 / 5053).
   - Đếm mẫu pure (5878 / 1490 / 2993) và majority (6955 / 1751 / 3572).
   - Khớp 100% chuẩn hóa train-only, ma trận nhầm lẫn và độ chính xác test 3572 mẫu (77.02% Acc, 51.50% Macro-F1).
   - Xác nhận 5 chốt kiểm tra dữ liệu hỏng (`guard_rejections`) từ chối chính xác.
   - Khớp mã checkpoint SHA và `current_operational_model_id: ab99a5c313e9de08`.
2. **Kiểm thử Thuật toán Lõi (36 ca)**: [`results-codex-direct36.json`](../output/ecdr/verification/results-codex-direct36.json) (SHA-256: `6ae5a90798ce8f1cc46ebd5e212235ad9b6861e95819ecf4bc4b1cd923422127`):
   - 36/36 ca kiểm thử độc lập PASS với `source_snapshot_stable: true`.
3. **Chốt Tham số Đầu vào & CLI**: [`cli_codex_direct.json`](../output/ecdr/verification/cli_codex_direct.json) (SHA-256: `cb6461799498e970ff3f8589039b92c0eb78b41274bfa52d9eb774777aa4a096`):
   - Xác nhận 11 chốt tham số/provenance từ chối chính xác TRƯỚC KHI nạp dữ liệu hoặc ghi đĩa (`epochs <= 0`, `lr <= 0`/NaN/Inf, `batch_size <= 0`, `max_windows <= 0`, checkpoint thiếu/không tồn tại).
   - Xác nhận 4 lệnh CLI parser invocations chạy đạt chuẩn.
   - Xác nhận 10 bài kiểm thử lifecycle chạy lại trực tiếp đều PASS với nguyên nhân lỗi cụ thể.
4. **Bảo tồn Tệp Di sản**: [`preservation_codex_direct.json`](../output/ecdr/verification/preservation_codex_direct.json) (SHA-256: `c0d6c27c33207b6334f6c11d230d6f0fbe6cc25ae2a4a7827eecbec07addace6`):
   - Xác nhận toàn bộ 90 tệp di sản được bảo tồn nguyên vẹn (`all_preserved: true`, 0 tệp bị thay đổi, 0 tệp bị thiếu).
5. **Kiểm thử Toàn diện Hiện tại (57 tests)**: [`unit_tests_codex_direct.json`](../output/ecdr/verification/unit_tests_codex_direct.json) (SHA-256: `00a511b6ee4f0a28644c67d06e4328ff5e29238a71a3056150796fee66498504`) và nhật ký chi tiết [`unit_tests_codex_direct.log`](../output/ecdr/verification/unit_tests_codex_direct.log) (SHA-256: `78ace77634bdc1955d6f7d8c17f96b8a19948257b057db2630dde912fd1c7a0b`):
   - 57/57 tests PASS, 0 failures, 0 errors, 0 skips, `source_snapshot_stable: true`.
6. **Xác thực Cờ Đánh giá CLI (`--out-json`)**: [`eval_flag_codex_direct.json`](../output/ecdr/verification/eval_flag_codex_direct.json) (SHA-256: `4d0921e052f446a4a40e0d4c767d773edb030a93352d41c6bb4a689a2c472d64`) và kết quả xuất thực tế [`eval_output_flag_test.json`](../output/ecdr/verification/eval_output_flag_test.json) (SHA-256: `ca8bcf2349f098b216499567a7770dc79e0ff236f2384a340183a9e1bbe3bd96`):
   - Khớp 100% 3,572 mẫu đánh giá, accuracy: 0.77015677, macro_f1: 0.51497474.

---

## 6. Kết Quả Thực Nghiệm Full-Stream Replay & Đánh Giá Giới Hạn

Thực nghiệm phát lại toàn bộ dòng chảy kiểm thử (`test` split, **5,053 cửa sổ trượt**) với checkpoint mô hình huấn luyện `ecdr_k12_best.pt` (Accuracy: 77.02%, Macro-F1: 51.50% trên tập test 3,572 mẫu).

### So sánh giữa Bản Lịch sử (Pre-Final-Integrity) và Bản Chạy Cuối Cùng (Final Run):
- **Bản lịch sử (pre-final-integrity)**: Lưu tại `output/ecdr/replay_test_full5053_diagonal_pre_final_integrity.json` (5,053 queries, 0 hits, 4,038 builds, 19 resets, baseline 3.638 ms, operational 61.679 ms).
- **Bản chạy chính thức cuối cùng (Final Run với đầy đủ chốt bảo vệ `anchor_state_hash`)**: Lưu tại `output/ecdr/replay_test_full5053_diagonal.json`.

```json
{
  "query_counts": {
    "total_queries": 5053,
    "hits": 0,
    "misses": 5053,
    "hit_rate_pct": 0.0,
    "build_attempts": 5053,
    "build_successes": 4038,
    "build_failures": 1015,
    "recording_resets": 19,
    "sequence_gap_resets": 0
  },
  "shadow_agreement": {
    "total_hits": 0,
    "checked_accepted_queries": 0,
    "agreed_queries": 0,
    "agreement_rate": null,
    "discrepancies": 0
  },
  "latencies_ms": {
    "t_e_candidate_selection": { "count": 5053, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0 },
    "t_m_membership": { "count": 5053, "mean_ms": 0.024, "p50_ms": 0.025, "p95_ms": 0.043 },
    "t_v_gate_validation": { "count": 5053, "mean_ms": 0.064, "p50_ms": 0.069, "p95_ms": 0.114 },
    "t_f_full_forward": { "count": 5053, "mean_ms": 2.829, "p50_ms": 2.709, "p95_ms": 3.705 },
    "t_b_certificate_build": { "count": 5053, "mean_ms": 59.494, "p50_ms": 58.409, "p95_ms": 71.417 },
    "t_shadow_execution": { "count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0 },
    "operational_query_latency": { "count": 5053, "mean_ms": 62.817, "p50_ms": 61.659, "p95_ms": 75.327 }
  },
  "computation_mac": {
    "full_reference_mac_per_window": 2343936,
    "incremental_mac_per_window": 1241856,
    "average_mac_per_window": 1245999.98,
    "theoretical_mac_saving_pct": 46.84
  },
  "empirical_speedup": {
    "measured_baseline_mean_ms": 4.159,
    "measured_operational_mean_ms": 62.817,
    "speedup_factor": 0.066
  }
}
```

### Phân Tích Giới Hạn & Kết Luận Nghiên Cứu
1. **Độ Tin Cậy của Chứng Nhận (Soundness & False-Accept Status)**:
   - Trong quá trình phát lại luân phiên không điều kiện trên 5,053 cửa sổ của test stream, tỷ lệ tái sử dụng thực nghiệm đạt **0.00% (0 / 5,053 cửa sổ)**.
   - Do không có bất kỳ truy vấn nào được chấp nhận tái sử dụng trên dòng dữ liệu này, tỷ lệ sai sót thực nghiệm (empirical false-accept rate) và mức độ tương thích shadow-reference trên luồng thực tế là **N/A (không áp dụng / không có mẫu tái sử dụng)**.
   - Tính đúng đắn của cơ chế phòng thủ cổng chỉ được khẳng định trên các bộ fixture kiểm thử biên và fixture đối nghịch có kiểm chứng (36/36 ca kiểm thử độc lập và 2/2 hits trên mẫu đối chứng dương).
2. **Chi Phí Tính Toán Thực Nghiệm (Operational Latency vs Baseline)**:
   - Baseline tham chiếu thuần túy: **~4.16 ms / cửa sổ**.
   - Operational Latency khi tích hợp ECDR: **~62.82 ms / cửa sổ**.
   - Tốc độ thực nghiệm: **0.07x** (hoàn toàn không có net latency speedup). Chi phí này xuất phát từ việc khi gặp Miss, hệ thống phải thực hiện thuật toán Interval Bound Propagation (IBP) 3 tầng tích chập để cố gắng xây dựng chứng chỉ mới cho mỏ neo kế tiếp (trung bình ~59.49 ms mỗi lần build).
3. **Phạm Vi Triển Khai & Ranh Giới Giả Thuyết**:
   - Mức ngân sách $b = 0.001$ là tham số cụ thể được đánh giá trong thực nghiệm phát lại dòng chảy này. Cần phân biệt rõ ranh giới giao diện: tham số dòng lệnh `--budget` của CLI hiện chỉ chấp nhận giá trị vô hướng dạng float (`float`), trong khi API thư viện Python nội bộ (`build_diagonal_box_certificate`, `build_wave_domain_certificate`, `Interval`) hỗ trợ đầy đủ các mức ngân sách dạng mảng vector đa kênh linh hoạt trên cả hai miền nhiễu (`diagonal_box` và `wave_domain`).
   - Các cơ chế Event Selector dựa trên năng lượng/ngữ nghĩa hoặc bộ chọn thích ứng được hoãn lại (deferred), chưa được cài đặt. Không đưa ra kết luận quy kết nguyên nhân (causal claim) từ kết quả của một mức ngân sách đơn lẻ.
   - **Quyết định**: Duy trì trạng thái `EXPERIMENTAL_HOLD`, không đề xuất thăng cấp gate (no gate promotion).

---

## 7. Danh Mục Lệnh Thực Thi (Runnable Commands)

Tất cả các lệnh được xác thực cú pháp và đối soát trực tiếp qua `--help`:

### 1. Kiểm tra hỗ trợ dữ liệu 12 lớp
```bash
./.venv/bin/python scripts/ecdr_audit_support.py --out-json data/ecdr_12_class_support.json
```

### 2. Huấn luyện mô hình ECDR K=12 (Smoke / Experimental)
```bash
./.venv/bin/python scripts/ecdr_train_k12.py --epochs 2 --seed 20261031 --save-dir output/ecdr/smoke/checkpoints_seed20261031 --out-metrics output/ecdr/smoke/train_metrics.json
```

### 3. Đánh giá mô hình trên tập kiểm thử giám sát (3,572 cửa sổ)
```bash
./.venv/bin/python scripts/ecdr_eval_k12.py --checkpoint output/ecdr/smoke/checkpoints_seed20261031/ecdr_k12_best.pt --split test --out-json data/ecdr_k12_eval_metrics.json
```

### 4. Chạy mô phỏng phát lại toàn bộ dòng chảy (5,053 cửa sổ)
```bash
./.venv/bin/python scripts/ecdr_stream_replay.py --split test --domain diagonal_box --budget 0.001 --checkpoint output/ecdr/smoke/checkpoints_seed20261031/ecdr_k12_best.pt --out-json output/ecdr/replay_test_full5053_diagonal.json
```

### 5. Chạy toàn bộ 57 bài kiểm thử nội bộ
```bash
PYTHONPATH=. ./.venv/bin/python tests/run_all_tests.py
```

### 6. Chạy bộ kiểm thử độc lập 36 ca kiểm tra
```bash
./.venv/bin/python output/ecdr/independent_evidence/check_core_36cases.py /tmp/results_run.json
```

---

## 8. Bảng Mã Băm SHA-256 Các Tệp Nguồn Gói `ecdr/`

Đối chiếu khớp 100% với biên nhận `output/ecdr/independent_evidence/results-final36.json` (SHA-256: `e51a991d8592cbf3ceec310bad49f4e61fad4f191e318f585008bdf205193644`):

```text
a6070ce3a3af30b1f15ba49f06af7c717472c78acd5ab4a1ef7c07944fcc3fcc  ecdr/__init__.py
8c0f112de65f15e4c4e6d94dd679a74bc6b8ebb0e2f9dd6053009936817fcce4  ecdr/bounds.py
cb074516f07991ababcdd6b22b425748161e9387212cd5168457ab79ffd92428  ecdr/certificate.py
b3e004259517be0410f3bb03af7dd490bf2b738d12940cb14e760c345fccb278  ecdr/data.py
43bb0288e7214fe9529c9a4395ba0b1894f4e07e42fd89f955bb4934b634f70f  ecdr/incremental.py
c00253c76ff0f30fd6b84917dab4efc2ece2e679167e373007ceb0605cf3d87e  ecdr/instrumentation.py
3fe6e20a5c32d66e0fdd916120e1b33cb6d558fb179fdf2a39d2de5b214f9621  ecdr/intervals.py
bfd2522c14e2bc112f421aadf5afdd20c10d57f536cb6245282c6e2616d859c3  ecdr/membership.py
829e7e40a1c50776dba6e0421316c02b5d508f36431e1e40140398f3c3912bfd  ecdr/model.py
891e6b695057be3b219a27c2fcc3746748d7c0d33f8fed793e08f42c6f898d2e  ecdr/preprocessing.py
de364bfe3a7362d2601ff5ea4394185e16eee5e371e9905fc03bdfa17601c7d3  ecdr/replay.py
f92510227ceb3a5adc764fd78d5e2f86c2b5938af9dfa0afd0152776c680a756  ecdr/schema.py
```
