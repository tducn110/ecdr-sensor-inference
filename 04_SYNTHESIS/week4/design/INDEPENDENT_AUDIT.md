# Báo cáo Kiểm toán Độc lập Week 4: Implementation, Thực nghiệm và Kết quả Khoa học

**Mã Task:** W4-036  
**Milestone:** M6 — Synthesis & Auditing  
**Phán quyết kiểm toán:** `PASSED_WITH_AUTHENTIC_NEGATIVE_RESULTS`  
**Run ID:** `week4_initial_20261103`  
**Ngày kiểm toán:** 2026-11-09T01:24:00Z  
**Kiểm toán viên độc lập:** `independent_auditor`  

---

## 1. Mục tiêu và Phạm vi Kiểm toán

Kiểm toán độc lập toàn diện quy trình thực thi, mã nguồn, dữ liệu thực nghiệm, tính toán thống kê và 24 giả thuyết khoa học của Week 4.
Phạm vi kiểm tra bao gồm:
1. **Toàn vẹn quản trị (Governance Integrity):** 214 protected files của Week 1–3, cryptographic seal hash trong `protocol_seal.json`, checkpoint `SmallCNN1D` $K=12$, và constants tiền xử lý.
2. **Kiểm đếm độc lập (Independent Denominator Recount):** Đếm lại toàn bộ raw trace `replay_trace.csv` (75.795 dòng) và `timing_trace.csv` (227.385 dòng).
3. **Tuân thủ quy tắc toán học & kế toán thực nghiệm:** Quy tắc $A_{\text{reuse}}$ là null khi $R = 0$, $D_{\text{all}} = D_{\text{reuse}} + D_{\text{fallback}}$, no-label-oracle trong query selector.
4. **Trung thực khoa học đối với kết quả âm:** Bảo toàn các phát hiện thực nghiệm phủ định giả thuyết (H02, H03, H07, H14, H15, H19, H20, H22, H23, H24).
5. **Đánh giá tam giác H24:** Kiểm tra Soundness $\land$ Useful Streaming Coverage $\land$ Low-Cost Net Speedup trên cùng một cấu hình thực nghiệm.

---

## 2. Toàn vẹn Quản trị và Mã nguồn

- **Protected Files Manifest:** Đã chạy `scripts/week4_audit.py check`. Toàn bộ **214 files** đóng băng từ Week 1–3 đều giữ nguyên trạng thái `clean` (khớp 100% SHA-256).
- **Cryptographic Protocol Seal:** Master Seal Hash `56806762e308960f4eb789b53ae7b973f1d2fe428bb8c8e11a196fc92ef4336c` trong `protocol_seal.json` được kiểm chứng hợp lệ trước khi thực hiện benchmark và replay trên tập test.
- **Mức độ phơi nhiễm tập Test:** Đã công bố minh bạch mức `KNOWN_TEST_EXPLORATORY` do tập test 5.053 cửa sổ HAPT đã từng quan sát ở Week 1–3. Không có tình trạng test-fitting hay tinh chỉnh siêu tham số post-hoc trên tập test.

---

## 3. Kiểm đếm Thực nghiệm Độc lập (Independent Recount)

| Thực thể kiểm đếm | Giá trị khai báo | Giá trị Recount độc lập | Sai lệch | Phán quyết |
|---|---|---|---|---|
| Tổng số query test stream ($N_{\text{full}}$) | 5.053 | 5.053 | 0 | `EXACT_MATCH` |
| Số query supervised ($N_{\text{sup}}$) | 3.572 | 3.572 | 0 | `EXACT_MATCH` |
| Số cặp liền kề cùng recording ($N_{\text{pairs}}$) | 5.034 | 5.034 | 0 | `EXACT_MATCH` |
| Số cụm bản ghi test (Recordings) | 19 | 19 | 0 | `EXACT_MATCH` |
| Số dòng trace replay (`replay_trace.csv`) | 75.795 | 75.795 (15 methods x 5.053) | 0 | `EXACT_MATCH` |
| Số dòng trace timing (`timing_trace.csv`) | 227.385 | 227.385 (3 reps x 15 methods x 5.053) | 0 | `EXACT_MATCH` |
| Số phương pháp đã đăng ký | 17 | 17 (15 EXECUTED, 2 AUDITED_UNSUPPORTED) | 0 | `EXACT_MATCH` |

### Phân bố Strata trên Test Stream:
- `pure_basic`: **2.874** cửa sổ (56.88%)
- `pure_transition`: **119** cửa sổ (2.35%)
- `majority_mixed`: **579** cửa sổ (11.46%)
- `ambiguous`: **479** cửa sổ (9.48%)
- `unlabeled`: **1.002** cửa sổ (19.83%)
- **Tổng cộng:** **5.053** cửa sổ (100.0%).

---

## 4. Xác minh Kế toán Thực nghiệm và Thỏa thuận Quyết định

1. **Quy tắc $A_{\text{reuse}}$ khi $R = 0$:**
   - Đối với các phương pháp M00, M01, M02, M04, M05–M13, M16, số lần reuse thành công trên tập test continuous là $R = 0$.
   - Kế toán thực nghiệm ghi nhận strictly `reused_agreement_rate: null` (không gán 1.0 hoặc 100%), đáp ứng đúng W4-032-C05 và W4-035-C02.
2. **Bảo toàn thỏa thuận quyết định qua Fail-closed Fallback:**
   - Tất cả các phương pháp chứng thực vùng (M05–M13) khi không có chứng chỉ bao đóng hợp lệ đều kích hoạt fallback an toàn về full reference forward, bảo toàn thỏa thuận quyết định 100% ($D_{\text{all}} = 0$, $A_{\text{full}} = 1.0$).
3. **Phân lập Timer đo lường:**
   - Timer `time.perf_counter_ns()` chỉ bao quanh phương thức xử lý truy vấn `adapter.process_query(qv)`.
   - Các chi phí logging ra đĩa CSV, tra cứu DataFrame, và đối chiếu shadow reference đều được loại bỏ khỏi số đo độ trễ vận hành.

---

## 5. Kết luận về 24 Giả thuyết Khoa học (H01–H24)

Kiểm toán xác nhận toàn bộ 24 kết luận giả thuyết trong `output/week4/week4_initial_20261103/tables/hypotheses.csv` được phản ánh chính xác từ dữ liệu thực tế:

1. **H01 (VERIFIED_FINITE):** Trùng lặp dịch chuyển chỉ số khớp chính xác trên 5.034 cặp liền kề; 12 vị trí đặc trưng trung gian Conv1D bitwise equal.
2. **H02 (FALSIFIED_UNCONDITIONAL):** Trùng lặp 50% thời gian không đảm bảo khoảng cách toạ độ thông thường nhỏ.
3. **H03 (FALSIFIED_UNCONDITIONAL):** Nhãn hoạt động không phải oracle quyết định; witness 682->683 đổi class dự đoán dù cùng nhãn WALKING_DOWNSTAIRS.
4. **H04 (DERIVED):** Winner margin nghiêm ngặt và tính liên tục xác lập vùng bất biến địa phương; nhưng bán kính hữu ích trên streaming chưa đạt.
5. **H05 (DERIVED_REAL_ARITHMETIC):** Output margin dương toàn miền đủ bảo toàn quyết định trên số thực; trong float32 cần numerical bridge outward rounding.
6. **H06 (CONDITIONAL_THEOREM):** Bán kính Lipschitz hình cầu được chứng minh đúng mặt lý thuyết nhưng bán kính không bao được mẫu tiếp theo trên streaming ($R=0$).
7. **H07 (FALSIFIED_COMPLETENESS):** IBP không thể chứng thực mọi miền bất biến do mất mát tương quan qua nhiều lớp mạng.
8. **H08 (DERIVED):** Công thức 11-margin trực tiếp loại bỏ mất mát phụ thuộc giữa các logits.
9. **H09 (NOT_ESTABLISHED):** CROWN chặt hơn IBP nhưng chi phí backward relaxation cao (~13ms vs 0.65ms), không rẻ hơn alternatives.
10. **H10 (NOT_ESTABLISHED):** Hướng PCA cục bộ phản ánh phương sai dữ liệu nhưng không trùng với pháp tuyến margin phi tuyến của mô hình.
11. **H11 (NOT_ESTABLISHED):** Miền cấu trúc affine không đem lại hữu ích vượt trội hơn hộp đường chéo tương đương trên dữ liệu streaming thực tế.
12. **H12 (DERIVED):** Cố định prefix 64 mẫu giảm toạ độ tự do từ 768 xuống 384.
13. **H13 (VERIFIED_FINITE):** Tái sử dụng đặc trưng Conv1D/BN/Pool đệm bảo toàn quyết định bitwise classifier ($A_{\text{reuse}} = 1.0$).
14. **H14 (FALSIFIED_UNCONDITIONAL):** Tính toán incremental không đem lại speedup ròng trên SmallCNN1D CPU (độ trễ 1.93ms vs 0.34ms, speed ratio 0.1775x, net savings -463.47%).
15. **H15 (FALSIFIED_UNCONDITIONAL):** Căn lại tâm sau khi hit mà không tính toán chứng chỉ mới gây trôi dạt vô hạn và mất tính đúng đắn.
16. **H16 (CONDITIONAL_THEOREM):** Chứng chỉ neo cố định bảo toàn tính đúng đắn lặp lại theo lifecycle bất biến.
17. **H17 (NOT_ESTABLISHED):** Miền bao đạt được đa bước suy sụp nhanh chóng do sai số tích luỹ theo horizon.
18. **H18 (DERIVED):** Độ phức tạp trường hợp xấu nhất của kiểm tra thuộc hộp là $\Theta(m)$.
19. **H19 (FALSIFIED_UNCONDITIONAL):** Tính đúng đắn toán học không kéo theo độ bao phủ hữu ích trên streaming ($R=0$ trên 5.053 cửa sổ test).
20. **H20 (FALSIFIED_UNCONDITIONAL):** Độ bao phủ 100% trong bản ghi không kéo theo speedup ròng (M03 đạt 100% coverage nhưng speed ratio 0.1775x).
21. **H21 (NOT_ESTABLISHED):** Frozen core M00 vượt trội trên CPU; LipCache audited unsupported trên continuous HAPT.
22. **H22 (FALSIFIED_UNCONDITIONAL):** Chứng minh số thực không tự động chuyển thành float32 sound nếu không có cầu nối outward rounding.
23. **H23 (FALSIFIED_UNCONDITIONAL):** Tên sự kiện thời gian không đủ chứng thực tái sử dụng quyết định.
24. **H24 (NOT_ESTABLISHED):** Không tồn tại chế độ vận hành đồng thời đúng đắn, bao phủ hữu ích và tiết kiệm chi phí trên HAPT (0/17 phương pháp thoả mãn đồng thời 3 điều kiện).

---

## 6. Đánh giá Tam giác Tính chất H24 (Soundness Trilemma)

Kiểm toán viên đã rà soát ma trận `output/week4/week4_initial_20261103/tables/h24_conjunction.csv`:
- Không có bất kỳ phương pháp nào đạt được đồng thời:
  $$\text{Soundness Guarantee} \land \text{Useful Streaming Coverage} \land \text{Low-Cost Net Speedup}$$
- M00, M01, M16: Đúng đắn nhưng không tái sử dụng ($C = \text{False}$).
- M03: Đúng đắn, đạt $C = \text{True}$ (5.034 hits) nhưng thất bại về chi phí ($S = \text{False}$, speed ratio 0.1775x).
- M04: Tái sử dụng heuristic nhưng không có bảo đảm đúng đắn toán học ($S = \text{False}$).
- M05–M13: Đúng đắn toán học nhưng $R = 0$ ($C = \text{False}$) và chi phí kiểm tra bounds khiến speed ratio $\le 0.69x$ ($S = \text{False}$).
- M14, M15: Không áp dụng được trên streaming HAPT đóng băng.

**Phán quyết:** Giả thuyết trung tâm H24 được xác nhận là **`NOT_ESTABLISHED`** (với bằng chứng phủ định thực nghiệm trên toàn bộ 17 phương pháp).

---

## 7. Phán quyết Chung

Kiểm toán độc lập nghiệm thu thành công Milestone M6 (W4-035 và W4-036), xác nhận toàn bộ số liệu, bảng biểu, ledger và các kết quả khoa học tiêu cực đều trung thực, tái lập được và tuân thủ chặt chẽ protocol đã niêm phong.
Hệ thống sẵn sàng chuyển sang Milestone M7 (Final Reporting & Delivery Verification).
