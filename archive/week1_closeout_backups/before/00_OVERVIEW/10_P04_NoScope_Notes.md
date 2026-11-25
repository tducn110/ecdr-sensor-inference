# P04 — NoScope: Optimizing Neural Network Queries over Video at Scale

Trạng thái: SOURCE_CHECKED_SCOPED, 21/10/2026; không phải đọc sâu/reproduce toàn bài.
Người kiểm nguồn: Codex; human acceptance chưa có. BibTeX: `kang2017noscope`.
Nhãn: [A] nội dung nguồn đã mở; [C] suy luận của người review.

Metadata: Kang, Emmons, Abuzaid, Bailis, Zaharia; PVLDB 10(11), 1586–1597 (2017).
DOI: [10.14778/3137628.3137664](https://doi.org/10.14778/3137628.3137664).
[Stanford record](https://dawn.cs.stanford.edu/publications/noscope).

Nguồn nội dung: [proceedings PDF](https://www.vldb.org/pvldb/vol10/p1586-kang.pdf).
Trang in 1586–1597 tương ứng PDF pages 1–12. Không có kết quả chạy lại code.

## 1. Research Question

[A] Tăng throughput truy vấn video dưới target FP/FN so với reference NN (§3, p.1588).

## 2. Assumptions

[A] Prototype giới hạn camera cố định và binary task (§8, pp.1591–1592).

## 3. Mathematical Core

[A] §6.2, p.1590 dùng cost model cho cascade. Note này chưa kiểm từng phương
trình hay chuyển nó thành soundness proof.

## 4. Method

[A] Difference detector → specialized NN → reference NN khi còn bất định
(§5, pp.1589–1590; §7, p.1591). Detector cho phép reuse nhãn trước.

## 5. Experiments

Chưa trích số benchmark; chưa reproduce. Không dùng tốc độ báo cáo để dự đoán HAPT.

## 6. Verified Claims

| Claim | Evidence locator |
|---|---|
| N01: Có temporal difference + conditional inference | §5 và §7 như trên; Figure 1 p.1586 có nhãn “short-circuit evaluation” |
| N02: Giới hạn FP/FN phụ thuộc dữ liệu | §6.3, p.1591: “training data reflects the testing data” |

Hai trích đoạn ngắn hỗ trợ claim hẹp; nội dung còn lại là diễn giải có locator.

## 7. Limitations

[A] N02 hạn chế khả năng suy rộng. [C] Evidence đã đọc không đủ để coi cơ chế
này là formal per-input certificate bảo toàn quyết định backbone.

## 8. Relation to Our RQ

[C] OVERLAP về temporal skipping. Vì vậy đề tài không được nhận novelty chỉ từ
việc dùng temporal difference để tránh inference. DISTINCT còn phải chứng minh:
sound event membership, direct-backbone certificate, chi phí có rebuild.

## 9. Reproduction

NOT_RUN. Không xác nhận code, môi trường, kết quả hay performance hiện hành.

## 10. Open Questions

Thiết kế baseline HAPT tương ứng cần adaptation riêng; lựa chọn này chưa được
thực hiện. C11 novelty và C12 net saving của đề tài vẫn UNVERIFIED.
