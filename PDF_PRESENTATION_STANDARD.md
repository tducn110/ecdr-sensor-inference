# Quy Chuẩn Trình Chiếu & Trình Bày Tài Liệu PDF (PDF Presentation Standard)
**ECDR Research Platform • Phiên Bản Chuẩn Hóa Học Thuật v2.0**

---

## 1. Mục Đích & Phạm Vi
Tài liệu này xác lập quy chuẩn kỹ thuật và thị giác cho việc hiển thị, trình chiếu, và phân phối 21 tệp PDF nghiên cứu thuộc 5 Tầng Tri Thức (Tiers 0–4) của Đề tài ECDR trên môi trường Web, Desktop và Thiết bị Di Động.

Mọi giao diện hiển thị PDF trong hệ thống bắt buộc phải tuân thủ nghiêm ngặt **hệ thống lưới nhịp 4px (4px baseline grid)** và **bảng màu học thuật Solid (Zero Gradients)** từ mã nguồn LaTeX gốc của bài báo.

---

## 2. Quy Chuẩn Lưới Canh Lề & Nhịp 4px (4px Grid Cadence)

Toàn bộ kích thước, khoảng đệm (padding), khoảng cách (margin, gap), bo góc (radius) và chiều cao (height) của khung trình chiếu PDF tuân theo bội số nguyên của 4px:

| Thành Phần | Kích Thước | Giá Trị Pixel | Ghi Chú Kỹ Thuật |
|---|---|---|---|
| **Thanh Tiêu Đề PDF (Toolbar)** | `h-10` | **40px** ($10 \times 4$) | Chứa icon, tên tệp, nhãn Tầng, và các nút thao tác nhanh |
| **Khoảng Đệm Khung (Padding)** | `p-3` (mobile) / `p-4` (desktop) | **12px / 16px** ($3 \times 4$ / $4 \times 4$) | Khoảng cách an toàn giữa viền trình duyệt và khung PDF |
| **Khoảng Cách Nút (Gap)** | `gap-2` | **8px** ($2 \times 4$) | Giữa các nút hành động (Fullscreen, Tab mới, Tải về) |
| **Chiều Cao Nút Bấm (Button Height)** | `h-8` | **32px** ($8 \times 4$) | Đạt chuẩn vùng bấm an toàn tối thiểu (Touch Target) |
| **Bo Góc Khung (Border Radius)** | `rounded-xl` | **12px** ($3 \times 4$) | Đồng bộ bo góc thẻ card học thuật |
| **Độ Dày Viền (Border Width)** | `border` | **1px** | Màu viền `#E2E8F0` phẳng, không đổ bóng đậm |

---

## 3. Kiến Trúc Bộ Điều Khiển Trình Chiếu (PDF Presentation Controls)

Mỗi tài liệu PDF khi mở trong hệ thống bắt buộc phải đi kèm thanh công cụ chuẩn (PDF Action Bar) gồm 4 chức năng cốt lõi:

1. **Nhận Diện Học Thuật (Academic Identity)**:
   - Biểu tượng tệp PDF chuẩn màu đỏ học thuật (`#EF4444`).
   - Tên tệp đầy đủ kèm nhãn Tầng tri thức tương ứng (`Tầng 0: RQ`, `Tầng 1: Dữ Liệu`, `Tầng 2: Mô Hình`, `Tầng 3: Toán Học`, `Tầng 4: Tổng Hợp`).
2. **Chế Độ Toàn Màn Hình (Present / Fullscreen Mode)**:
   - Mở rộng PDF ra 100% diện tích màn hình thiết bị, loại bỏ sidebar và header để phục vụ báo cáo / thuyết trình trước hội đồng. Phím tắt: `F`.
3. **Mở Tab Trình Duyệt Độc Lập (External Tab View)**:
   - Mở trực tiếp bằng PDF Viewer gốc của trình duyệt (Chrome PDF Viewer / Firefox PDF.js) để người đọc sử dụng mục lục (Table of Contents), thumbnail trang và tính năng zoom nâng cao.
4. **Tải Về Trực Tiếp (Direct Download)**:
   - Header `Content-Disposition: inline` với nút download 1-click lưu trữ cục bộ.

---

## 4. Quy Chuẩn Thích Ứng Thiết Bị Di Động (Mobile Responsiveness)

Đối với màn hình điện thoại (độ rộng $\le 768\text{px}$, kiểm thử chuẩn ở $375\text{px}$):
- Trình duyệt di động (iOS Safari, Chrome Mobile) thường chặn hoặc thu nhỏ thẻ `<iframe>` PDF.
- Hệ thống áp dụng **Cơ Chế Phản Ứng Kép (Dual Strategy)**:
  - Nếu iframe tải được: Hiển thị với chiều cao tối thiểu $480\text{px}$ và thanh công cụ chạm lớn ($40\text{px}$ tap target).
  - Cung cấp sẵn nút bấm nổi bật **"Mở PDF Toàn Màn Hình"** và **"Tải Về"** ở đầu trang để người dùng di động đọc tài liệu mượt mà mà không bị kẹt cảm ứng cuộn trang.

---

## 5. Bảng Màu Học Thuật Chuẩn (Strict LaTeX Color Palette)

```css
:root {
  --pdf-bg: #FFFFFF;
  --pdf-surface: #F8FAFC;
  --pdf-border: #E2E8F0;
  --pdf-ink: #183449;
  --pdf-teal: #187C80;
  --pdf-red: #EF4444;
  --pdf-red-bg: #FEF2F2;
}
```
*Tuyệt đối không sử dụng dải màu gradient, hiệu ứng phát sáng (glow), hoặc bóng đổ nhiều tầng trên khung hiển thị tài liệu.*
