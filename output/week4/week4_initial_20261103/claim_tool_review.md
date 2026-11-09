# Báo Cáo Đánh Giá Công Cụ Điều Phối Nhiệm Vụ Week 4 (Claim Tool Review)

**Mã Run:** `week4_initial_20261103`  
**Task:** `W4-004` — Implement task claim, file locks và execution receipts  
**Trạng thái kiểm tra:** `VERIFIED`  
**Ngày thực hiện:** 2026-11-03  

---

## 1. Mục tiêu và bối cảnh triển khai

Trong kiến trúc điều phối nghiên cứu Week 4, hệ thống gồm 40 task cards với đồ thị phụ thuộc (DAG) chặt chẽ và nhiều đường dẫn tệp ghi chia sẻ (`ecdr/week4/*.py`, `configs/week4/*.json`, `output/week4/*`).

Để hỗ trợ agent (agy) hoặc lập trình viên claim và thực thi các nhiệm vụ một cách song song mà không gây xung đột (write conflicts, race conditions, lock aliasing, hoặc rò rỉ trạng thái), công cụ dòng lệnh `scripts/week4_tasks.py` được xây dựng để hiện thực hóa toàn bộ các quy định trong `04_SYNTHESIS/week4/ORCHESTRATOR.md`.

---

## 2. Kiến trúc giải thuật và cơ chế an toàn

### 2.1. Giải thuật phát hiện nhiệm vụ sẵn sàng (`ready`)
Hàm `get_ready_tasks()` quét danh mục `tasks.json` và chỉ công bố một task là `READY` khi thỏa mãn đồng thời 2 điều kiện:
1. **Thỏa mãn toàn bộ dependencies**:
   - Tất cả các task trong `dependencies` phải đạt trạng thái `VERIFIED`.
   - Nếu dependency đạt `VERIFIED_LIMITED`, task kế thừa chỉ được sẵn sàng khi có quy tắc xử lý năng lực thiếu sót được ghi nhận rõ trong `dependency_rule`.
2. **Không có xung đột tập ghi (Zero Write-Set Overlap)**:
   - Tập đường dẫn `allowed_write_paths` của task không được giao thoa với bất kỳ đường dẫn nào đang bị khóa bởi các task `IN_PROGRESS` hoặc `REVIEW` khác.

### 2.2. Khóa file nguyên tử cấp hệ điều hành (OS Advisory Lock)
- Để bảo vệ chống race condition giữa các tiến trình chạy song song, `RegistryManager` sử dụng cơ chế khóa độc quyền `fcntl.flock(fileno, fcntl.LOCK_EX)` trên tệp khóa đồng hành (`tasks.json.lock`).
- Việc lưu trữ dữ liệu được thực hiện theo nguyên tắc **ghi nguyên tử (Atomic Replace)**: dữ liệu mới được ghi vào một file tạm (`tempfile.NamedTemporaryFile`) trong cùng thư mục, thực hiện `fsync()`, sau đó dùng lệnh đổi tên nguyên tử `os.replace()` để thay thế `tasks.json`. Điều này ngăn chặn hoàn toàn nguy cơ file JSON bị hư hại nếu tiến trình bị gián đoạn giữa chừng.

### 2.3. Chuẩn hóa đường dẫn chống bí danh khóa (Anti-Lock Aliasing)
- Hàm `_normalize_path()` chuyển đổi mọi đường dẫn tuyệt đối hoặc tương đối về dạng chuẩn tắc so với thư mục gốc dự án (`REPO_ROOT`).
- Ví dụ: `./scripts/week4_tasks.py`, `scripts//week4_tasks.py`, và `/absolute/.../scripts/week4_tasks.py` đều được chuẩn hóa thành `scripts/week4_tasks.py`. Điều này ngăn ngừa trường hợp hai task ghi vào cùng một tệp nhưng dùng cú pháp đường dẫn khác nhau để lọt qua bước kiểm tra khóa.

### 2.4. Thẩm định biên lai thực thi nghiêm ngặt (`submit`)
Hàm `validate_receipt_file()` kiểm tra kỹ lưỡng tệp biên lai JSON trước khi cho phép chuyển trạng thái sang `REVIEW`:
- `template_only` bắt buộc phải là `false`.
- `task_id` trong biên lai phải khớp chính xác với task được nộp.
- Mảng `commands` không được rỗng, và tất cả các lệnh kiểm thử bắt buộc phải có `exit_code == 0`.
- Mảng `requirement_evidence` không được rỗng, và toàn bộ yêu cầu checklist phải đạt trạng thái `PASS` hoặc `VERIFIED`.

### 2.5. Tách biệt quyền hạn người thẩm định (`review`)
- Nguyên tắc tam quyền phân lập trong nghiên cứu khoa học: người thực hiện (tác giả được ghi trong `assignee`) tuyệt đối không được tự làm người duyệt (`reviewer_id`).
- Lệnh `review` từ chối tự duyệt nếu không có cờ ghi đè khẩn cấp.
- Khi một task được phê duyệt `VERIFIED` hoặc `VERIFIED_LIMITED`, công cụ sẽ tự động giải phóng toàn bộ các khóa tập ghi mà task đang nắm giữ, mở đường cho các task phụ thuộc tiếp theo chuyển sang trạng thái `READY`.

### 2.6. Chính sách chống cướp quyền qua timeout (No Timeout Theft)
- Khác với các hệ thống phân tán thông thường tự động thu hồi khóa sau một khoảng thời gian, công cụ `scripts/week4_tasks.py` tuân thủ quy tắc không tự ý cướp quyền (steal claim) chỉ vì hết hạn timeout.
- Việc giải phóng khóa (`release`) bắt buộc phải có lý do giải trình cụ thể (`--reason`) từ phía orchestrator hoặc user.

---

## 3. Bằng chứng kiểm thử tự động (`tests/test_week4_task_claims.py`)

Toàn bộ 6 bài kiểm tra độc lập trong `tests/test_week4_task_claims.py` được thực thi trên fixture thư mục tạm thời (`tempfile.TemporaryDirectory`), tuyệt đối không làm thay đổi trạng thái file `tasks.json` thật:
1. `test_ready_tasks_initial_state`: Xác nhận tính toán chính xác DAG nhiệm vụ sẵn sàng (PASS).
2. `test_claim_acquires_lock_and_blocks_collision_and_aliases`: Khóa thành công tệp ghi và chặn đứng task có bí danh tệp trùng lặp (PASS).
3. `test_claim_blocked_dependencies`: Chặn việc claim task khi nhiệm vụ tiền đề chưa xong (PASS).
4. `test_submit_validates_receipt_file`: Thẩm định đầy đủ các điều kiện hợp lệ của biên lai thực thi (PASS).
5. `test_review_requires_independent_reviewer`: Từ chối việc tác giả tự review và mở khóa nhiệm vụ kế tiếp khi được phê duyệt (PASS).
6. `test_release_requires_mandatory_reason`: Buộc phải có lý do khi hủy khóa (PASS).

Kết quả: **6/6 passed (0.004s)**.

---

## 4. Kết luận

Task `W4-004` đã hoàn thành trọn vẹn, khép lại toàn bộ **Milestone M0 (Khung Điều Phối & Hợp Đồng Dữ Liệu)** gồm:
- `W4-001`: Khóa baseline, bảo toàn tệp và manifest 214 tệp (`VERIFIED`).
- `W4-002`: Định nghĩa giao thức đánh giá, 4 mẫu số, 17 phương pháp (`VERIFIED`).
- `W4-003`: Hợp đồng kiểu dữ liệu bất biến và adapter mô hình tham chiếu (`VERIFIED`).
- `W4-004`: Công cụ quản lý claim nhiệm vụ, khóa nguyên tử và kiểm tra biên lai (`VERIFIED`).

Toàn bộ hệ thống sẵn sàng bước vào **Milestone M1 (Numerics: W4-005 $\to$ W4-010)** để xây dựng cầu nối số học làm tròn ra ngoài (outward rounding) trên float32.
