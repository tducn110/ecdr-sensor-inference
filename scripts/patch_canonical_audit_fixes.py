# -*- coding: utf-8 -*-
"""
Canonical Audit Fixes for ECDR Presentation Portal.
Addresses 100% of P0, P1, P2 findings from the comprehensive audit report:
- P0-01: Correct SmallCNN1D computation graph & MACs (1,172,992 Conv/Linear MACs, k=5, 3, 3).
- P0-02: Correct data split numbers (Train 6,955 sup / 9,978 full; Val 1,751 sup / 2,422 full; Test 3,572 sup / 5,053 full).
- P0-03: Tone down soundness claims (G3 partial, 0 executable certs, 0 disagreements due to 100% fail-closed fallback).
- P0-04: Qualitative architectural comparison for CNN vs LSTM/Transformer/RF (remove fake numbers).
- P0-05: Fix code snippets (undefined variables, conceptual disclaimer).
- P0-06: Project governance & SWOT framed as proposed framework / methodology risks.
- P1-07: Radius 1e-6 clarification (validation grid choice, not universal limit).
- P1-08: H13 interior Conv3 feature alignment scope (12 interior positions, boundary differs due to padding).
- P1-09: H10 / PCA geometry (tangent to margin level set, not orthogonal to decision boundary).
- P1-10: 3-row latency boundaries (core 0.19-0.22ms, wrapper 1.59-1.68ms, ECDR 3.11-3.20ms).
- P1-11: Exact 24 hypotheses tallies (7 Derived, 3 Conditional, 9 Falsified, 5 Not Established / Descriptive).
- P1-13: Exact 12-class per-class metrics from model audit.
"""

import re
import os

INDEX_PATH = "/home/pro/Downloads/research/notes/presents/index.html"

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# -----------------------------------------------------------------------------
# 1. FIX P0-01: ARCHITECTURE & MACs IN MONOGRAPH & INLINE SLIDES
# -----------------------------------------------------------------------------
# Replace Hero text in Week 2
html = html.replace(
    "giải phẫu 1.46 triệu MACs của mô hình tham chiếu",
    "giải phẫu 1.173 triệu MACs (1,172,992 Conv/Linear MACs) của mô hình tham chiếu"
)

# Replace Hero card in Week 2
html = html.replace(
    '<div class="text-2xl font-serif font-bold text-black">1.46M MACs</div>\n                        <span class="text-xs font-mono text-zinc-600">92.45% params tại Linear1</span>',
    '<div class="text-2xl font-serif font-bold text-black">1.173M MACs</div>\n                        <span class="text-xs font-mono text-zinc-600">1,172,992 Conv/Linear MACs • 283,692 params</span>'
)

# Replace Badge in Section #w2-macs
html = html.replace(
    '<span class="acad-badge acad-badge-black font-mono">1.46M MACs &bull; 283,692 PARAMS</span>',
    '<span class="acad-badge acad-badge-black font-mono">1.173M MACs &bull; 283,692 PARAMS</span>'
)

# Replace the entire Table in #w2-macs with canonical verified values
old_macs_tbody = '''                            <tbody class="divide-y divide-zinc-200 font-mono text-xs">
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Conv1D_1 + BN + ReLU</td>
                                    <td class="p-2 border-r border-black">in=6, out=64, k=5, pad=2</td>
                                    <td class="p-2 border-r border-black">[B, 64, 128]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">1,984</td>
                                    <td class="p-2 border-r border-black text-right">0.70%</td>
                                    <td class="p-2 text-right">20,480</td>
                                </tr>
                                <tr>
                                    <td class="p-2 border-r border-black text-zinc-500">MaxPool1D_1</td>
                                    <td class="p-2 border-r border-black text-zinc-500">k=2, stride=2</td>
                                    <td class="p-2 border-r border-black text-zinc-500">[B, 64, 64]</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0.00%</td>
                                    <td class="p-2 text-right text-zinc-500">4,096</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Conv1D_2 + BN + ReLU</td>
                                    <td class="p-2 border-r border-black">in=64, out=64, k=5, pad=2</td>
                                    <td class="p-2 border-r border-black">[B, 64, 64]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">20,544</td>
                                    <td class="p-2 border-r border-black text-right">7.24%</td>
                                    <td class="p-2 text-right">204,800</td>
                                </tr>
                                <tr>
                                    <td class="p-2 border-r border-black text-zinc-500">MaxPool1D_2</td>
                                    <td class="p-2 border-r border-black text-zinc-500">k=2, stride=2</td>
                                    <td class="p-2 border-r border-black text-zinc-500">[B, 64, 32]</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0.00%</td>
                                    <td class="p-2 text-right text-zinc-500">2,048</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Conv1D_3 + BN + ReLU</td>
                                    <td class="p-2 border-r border-black">in=64, out=64, k=3, pad=1</td>
                                    <td class="p-2 border-r border-black">[B, 64, 32]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">12,352</td>
                                    <td class="p-2 border-r border-black text-right">4.35%</td>
                                    <td class="p-2 text-right">122,880</td>
                                </tr>
                                <tr class="bg-zinc-100">
                                    <td class="p-2 font-bold border-r border-black">Linear_1 + ReLU (Điểm Nghẽn)</td>
                                    <td class="p-2 border-r border-black font-bold">in=2,048, out=128</td>
                                    <td class="p-2 border-r border-black">[B, 128]</td>
                                    <td class="p-2 border-r border-black text-right font-bold text-black underline">262,272</td>
                                    <td class="p-2 border-r border-black text-right font-bold text-black underline">92.45%</td>
                                    <td class="p-2 text-right font-bold">262,144</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Linear_2 (Output Logits)</td>
                                    <td class="p-2 border-r border-black">in=128, out=12</td>
                                    <td class="p-2 border-r border-black">[B, 12]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">1,548</td>
                                    <td class="p-2 border-r border-black text-right">0.55%</td>
                                    <td class="p-2 text-right">1,536</td>
                                </tr>
                                <tr class="border-t-2 border-black bg-black text-white font-bold">
                                    <td class="p-2 border-r border-white" colspan="3">TỔNG TOÀN BỘ MÔ HÌNH SMALLCNN1D</td>
                                    <td class="p-2 border-r border-white text-right">283,692</td>
                                    <td class="p-2 border-r border-white text-right">100.0%</td>
                                    <td class="p-2 text-right">~1.46M MACs</td>
                                </tr>
                            </tbody>'''

new_macs_tbody = '''                            <tbody class="divide-y divide-zinc-200 font-mono text-xs">
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Conv1D_1 + BN + ReLU</td>
                                    <td class="p-2 border-r border-black">in=6, out=32, k=5, pad=2</td>
                                    <td class="p-2 border-r border-black">[B, 32, 128]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">1,056</td>
                                    <td class="p-2 border-r border-black text-right">0.37%</td>
                                    <td class="p-2 text-right">122,880</td>
                                </tr>
                                <tr>
                                    <td class="p-2 border-r border-black text-zinc-500">MaxPool1D_1</td>
                                    <td class="p-2 border-r border-black text-zinc-500">k=2, stride=2</td>
                                    <td class="p-2 border-r border-black text-zinc-500">[B, 32, 64]</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0.00%</td>
                                    <td class="p-2 text-right text-zinc-500">0 (chưa tính MACs)</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Conv1D_2 + BN + ReLU</td>
                                    <td class="p-2 border-r border-black">in=32, out=64, k=3, pad=1</td>
                                    <td class="p-2 border-r border-black">[B, 64, 64]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">6,336</td>
                                    <td class="p-2 border-r border-black text-right">2.23%</td>
                                    <td class="p-2 text-right">393,216</td>
                                </tr>
                                <tr>
                                    <td class="p-2 border-r border-black text-zinc-500">MaxPool1D_2</td>
                                    <td class="p-2 border-r border-black text-zinc-500">k=2, stride=2</td>
                                    <td class="p-2 border-r border-black text-zinc-500">[B, 64, 32]</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0</td>
                                    <td class="p-2 border-r border-black text-right text-zinc-500">0.00%</td>
                                    <td class="p-2 text-right text-zinc-500">0 (chưa tính MACs)</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Conv1D_3 + BN + ReLU</td>
                                    <td class="p-2 border-r border-black">in=64, out=64, k=3, pad=1</td>
                                    <td class="p-2 border-r border-black">[B, 64, 32]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">12,480</td>
                                    <td class="p-2 border-r border-black text-right">4.40%</td>
                                    <td class="p-2 text-right">393,216</td>
                                </tr>
                                <tr class="bg-zinc-100">
                                    <td class="p-2 font-bold border-r border-black">Linear_1 + ReLU (Điểm Nghẽn Tham Số)</td>
                                    <td class="p-2 border-r border-black font-bold">in=2,048, out=128</td>
                                    <td class="p-2 border-r border-black">[B, 128]</td>
                                    <td class="p-2 border-r border-black text-right font-bold text-black underline">262,272</td>
                                    <td class="p-2 border-r border-black text-right font-bold text-black underline">92.45%</td>
                                    <td class="p-2 text-right font-bold">262,144</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Linear_2 (Output Logits)</td>
                                    <td class="p-2 border-r border-black">in=128, out=12</td>
                                    <td class="p-2 border-r border-black">[B, 12]</td>
                                    <td class="p-2 border-r border-black text-right font-bold">1,548</td>
                                    <td class="p-2 border-r border-black text-right">0.55%</td>
                                    <td class="p-2 text-right">1,536</td>
                                </tr>
                                <tr class="border-t-2 border-black bg-black text-white font-bold">
                                    <td class="p-2 border-r border-white" colspan="3">TỔNG TOÀN BỘ MÔ HÌNH SMALLCNN1D</td>
                                    <td class="p-2 border-r border-white text-right">283,692</td>
                                    <td class="p-2 border-r border-white text-right">100.0%</td>
                                    <td class="p-2 text-right">1,172,992 MACs</td>
                                </tr>
                            </tbody>'''

html = html.replace(old_macs_tbody, new_macs_tbody)

# Fix caption / footnote in #w2-macs
html = html.replace(
    'Mô hình gồm 3 khối Conv1D (k=5, 5, 3), 2 tầng MaxPool1D và tầng Linear 2048->128 chiếm tới 92.45% tham số (262,272 tham số). Tổng chi phí 1.46 triệu MACs tốn 0.201 ms CPU.',
    'Mô hình gồm 3 khối Conv1D (k=5, 3, 3), 2 tầng MaxPool1D và tầng Linear 2048->128 chiếm tới 92.45% tham số (262,272 tham số). Tổng chi phí 1,172,992 Conv/Linear MACs tốn ~0.19 ms forward CPU thuần.'
)

# -----------------------------------------------------------------------------
# 2. FIX P0-02: DATASET SPLIT & DENOMINATORS IN MONOGRAPH
# -----------------------------------------------------------------------------
html = html.replace(
    '<div class="text-2xl font-serif font-bold text-black">7,734 / 3,572</div>\n                        <span class="text-xs font-mono text-zinc-600">Train vs Test windows</span>',
    '<div class="text-2xl font-serif font-bold text-black">6,955 / 3,572</div>\n                        <span class="text-xs font-mono text-zinc-600">Supervised Train vs Test (9,978 / 5,053 full)</span>'
)

html = html.replace(
    'Phân chia 30 tình nguyện viên thành 17 người tập Train (7,734 windows), 4 người tập Val (1,792 windows) và 9 người tập Test (3,572 windows). Khóa chặt tính độc lập của người kiểm thử.',
    'Phân chia 30 tình nguyện viên thành 17 người tập Train (6,955 supervised / 9,978 full stream), 4 người tập Val (1,751 supervised / 2,422 full stream) và 9 người tập Test (3,572 supervised / 5,053 full stream). Tổng cộng 12,278 supervised windows (10,361 pure + 1,917 majority); 1,588 ambiguous và 3,587 unlabeled bị loại khỏi tập có giám sát nhưng vẫn giữ trong full stream 17,453 windows.'
)

# -----------------------------------------------------------------------------
# 3. FIX P0-03: SOUNDNESS & RESEARCH STATUS TUẦN 3 IN MONOGRAPH
# -----------------------------------------------------------------------------
html = html.replace(
    '<h3 class="text-base font-serif font-bold text-black">\n                        Hệ Thống Toán &amp; Verifier Hoàn Tất &ndash; Nhưng Chưa Có Reuse &amp; Tiết Kiệm Chi Phí Âm\n                    </h3>\n                    <p class="text-xs text-zinc-800 leading-relaxed font-serif">\n                        Tuần 3 đã thiết lập thành công bộ kiểm chứng hình thức (Formal Verifier) trên đúng computation graph của SmallCNN1D với 93/93 tests PASS. Tuy nhiên, <strong>tỷ lệ tái sử dụng thực nghiệm đạt đúng 0.0% trên 20,212 truy vấn</strong>, và độ trễ toàn trình là 3.15 ms so với 1.63 ms của CNN gốc (đạt tốc độ 0.52x, tức savings âm). Nghiên cứu không che giấu sự thật này mà đi sâu bóc tách hai nguyên nhân gốc rễ toán học và đề xuất 3 hướng giải quyết thực chất cho Tuần 4.\n                    </p>',
    '<h3 class="text-base font-serif font-bold text-black">\n                        Định Lý Số Thực &amp; Verifier Chẩn Đoán IBP &ndash; Chưa Đạt Chứng Nhận Số Học Float32 (G3 PARTIAL)\n                    </h3>\n                    <p class="text-xs text-zinc-800 leading-relaxed font-serif">\n                        Tuần 3 đã chứng minh định lý Margin Theorem trong số học thực và thiết lập bộ kiểm chứng chẩn đoán IBP trên đồ thị SmallCNN1D thật với 93/93 unit/regression tests PASS. Tuy nhiên, <strong>chứng nhận bao hàm số học (numerical enclosure) cho phép thực thi float32 chưa được chứng minh</strong>, do đó cổng G3 vẫn ở trạng thái PARTIAL / NOT PASSED. Trên 20,212 truy vấn replay, hệ thống tuân thủ nguyên tắc fail-closed và kích hoạt fallback 100% sang mô hình tham chiếu gốc (tái sử dụng = 0.0%, 0 sai lệch so với CNN tham chiếu). Độ trễ toàn trình ECDR là 3.11–3.20 ms so với baseline wrapper 1.59–1.68 ms (đạt 0.52x, tức savings âm so với wrapper). Nghiên cứu đi sâu bóc tách hai rào cản toán học (Blocker A &amp; B) và chuyển giao lộ trình sang Tuần 4.\n                    </p>'
)

# Table in Figure W3-1
old_w3_table = '''                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Tính Đúng Đắn Quyết Định (Soundness)</td>
                                    <td class="p-2 border-r border-black text-center font-bold">100% (0 sai lệch)</td>
                                    <td class="p-2 border-r border-black text-center font-bold text-black bg-zinc-100">100% (0 sai lệch)</td>
                                    <td class="p-2 font-serif text-zinc-700">ĐẠT: Nhờ cơ chế Fail-Closed, không có bất kỳ quyết định sai lệch nào so với CNN gốc.</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Căn Chỉnh Đặc Trưng Overlap (H13)</td>
                                    <td class="p-2 border-r border-black text-center font-bold">Khớp Tuyệt Đối</td>
                                    <td class="p-2 border-r border-black text-center font-bold text-black bg-zinc-100">3,866,112 / 3,866,112</td>
                                    <td class="p-2 font-serif text-zinc-700">ĐẠT XUẤT SẮC: Sai số tối đa = 0.000000 bitwise trên toàn bộ các tầng Conv.</td>
                                </tr>'''

new_w3_table = '''                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Sự Phù Hợp Quyết Định Replay (Observed Full-Stream Agreement)</td>
                                    <td class="p-2 border-r border-black text-center font-bold">0 Sai Lệch</td>
                                    <td class="p-2 border-r border-black text-center font-bold text-black bg-zinc-100">0 / 20,212 Sai Lệch</td>
                                    <td class="p-2 font-serif text-zinc-700">100% khớp quyết định tham chiếu do toàn bộ truy vấn chuyển sang fallback chạy CNN gốc; tỷ lệ tái sử dụng được chấp nhận = 0.0% (A_reuse = N/A).</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Chứng Nhận Số Học Thực Thi (Executable Numerical Soundness)</td>
                                    <td class="p-2 border-r border-black text-center font-bold">Float32 Enclosure</td>
                                    <td class="p-2 border-r border-black text-center font-bold text-amber-800 bg-amber-50">CHƯA ĐẠT (G3 PARTIAL)</td>
                                    <td class="p-2 font-serif text-zinc-700">Định lý margin đúng trong số học thực; float64 IBP là chẩn đoán; chưa có proof bao hàm số học cho executable float32 backend. Validated certificates = 0.</td>
                                </tr>
                                <tr>
                                    <td class="p-2 font-bold border-r border-black">Căn Chỉnh Đặc Trưng Nội Bộ (H13)</td>
                                    <td class="p-2 border-r border-black text-center font-bold">Khớp Tuyệt Đối Vùng Nội Bộ</td>
                                    <td class="p-2 border-r border-black text-center font-bold text-black bg-zinc-100">3,866,112 / 3,866,112</td>
                                    <td class="p-2 font-serif text-zinc-700">ĐẠT: Khớp bitwise tuyệt đối trên 12 vị trí nội bộ sau Conv3 (j=2..13) trên 5,034 cặp cửa sổ; vị trí biên khác nhau do padding. Tiềm năng feature reuse, chưa suy ra bảo toàn quyết định.</td>
                                </tr>'''

html = html.replace(old_w3_table, new_w3_table)

# -----------------------------------------------------------------------------
# 4. FIX P1-08 (H13) & P1-09 (PCA) & P1-10 (LATENCY) IN MONOGRAPH
# -----------------------------------------------------------------------------
# H13 Section #w3-geometry
html = html.replace(
    'Thực nghiệm kiểm chứng giả thuyết H13 trên 5,034 cặp cửa sổ: 3,866,112 điểm đặc trưng trung gian qua Conv1, Conv2, Conv3 đạt sai số tuyệt đối tối đa bằng <strong>0.000000</strong>. Khẳng định tính chất bất biến cục bộ của phép tích chập 1D không có đệm viền lệch.',
    'Thực nghiệm kiểm chứng giả thuyết H13 trên 5,034 cặp cửa sổ: 12 vị trí của feature map sau Conv3 trong vùng nội bộ (j=2..13) qua 64 kênh ($5,034 \\times 12 \\times 64 = 3,866,112$ phép so sánh) đạt sai số tuyệt đối bằng <strong>0.000000 bitwise</strong>. Các vị trí sát biên khác biệt do ảnh hưởng padding và receptive field. Kết quả này chứng minh tiềm năng tái sử dụng đặc trưng tăng dần (incremental feature reuse), chưa bảo đảm bảo toàn quyết định argmax ở đầu ra.'
)

# Blocker A: The Radius Dilemma
html = html.replace(
    'Để 11 cận biên $F_c - F_j > 0$ giữ được giá trị dương, bán kính hình hộp an toàn $\\rho$ bị ép về $\\leq 10^{-6}$. Nhưng trong thực tế, ngay cả khi người dùng ngồi yên, nhiễu cảm biến giữa 2 cửa sổ liên tiếp đã có độ lệch $L_\\infty \\approx 0.1$ đến $1.0$, khiến <strong>mọi cửa sổ kế tiếp đều nằm ngoài miền chứng nhận $\\mathcal{S}$</strong>!',
    r'Tại bán kính $\rho = 10^{-6}$ được chọn qua lưới thẩm định (validation grid) trên 64 mỏ neo tập Val, bộ verifier dựng được cận dương cho 99.6% mỏ neo nhưng không bao phủ cửa sổ kế tiếp nào trong tập test (độ dịch chuyển tự nhiên $\|\Delta\|_\infty \approx 0.1 - 1.0$, lớn gấp 100,000 lần bán kính). Độ lỏng của cận IBP là một nguyên nhân quan sát được trong thiết kế hiện tại; không suy rộng thành giới hạn của mọi miền hoặc kỹ thuật chứng nhận trong tương lai.'
)

# Blocker B: Latency Overhead
html = html.replace(
    'Mô hình tham chiếu SmallCNN1D được thiết kế cực kỳ gọn nhẹ (chỉ tốn 0.201 ms tính toán thuần hoặc 1.63 ms kèm wrapper). Trong khi đó, việc kiểm tra điều kiện miền $X \\in \\mathcal{S}$ và tính toán cận chặn dấu chấm động IEEE-754 tốn 1.48 ms. Khi tỷ lệ reuse bằng 0%, hệ thống phải gánh chịu cả chi phí kiểm chứng lẫn chi phí chạy lại mạng CNN, dẫn đến <strong>thời gian thực thi tăng gấp đôi</strong>!',
    'Phân định rõ 3 ranh giới độ trễ trên cùng môi trường CPU: (1) Forward SmallCNN1D thuần túy tốn 0.189–0.224 ms; (2) Baseline wrapper (kèm rehash mã băm nhận diện) tốn 1.589–1.675 ms; (3) ECDR toàn trình (kèm kiểm tra thành viên và fallback forward) tốn 3.107–3.198 ms. Tỷ lệ tăng tốc 0.52x (tiết kiệm âm) được đo so với baseline wrapper, xuất phát từ việc chi phí kiểm tra thành viên + fallback vượt quá chi phí chạy thẳng mô hình.'
)

# PCA Geometry #w3-pca
html = html.replace(
    'Phân tích 9,944 vector dịch chuyển $\\Delta = X_{t+1} - X_t$: 3 thành phần chính đầu tiên chỉ giải thích được 21.48% phương sai. Đặc biệt, độ tương đồng cosine giữa vector dịch chuyển và gradient của mặt biên quyết định chỉ đạt <strong>0.051 (góc ~87.1&deg;)</strong>. Điều này chứng minh toán học rằng các dao động cảm biến tự nhiên gần như trực giao với mặt biên phân loại, giải thích vì sao nhãn hoạt động không đổi nhưng hình hộp IBP trục tọa độ lại bị vỡ!',
    'Phân tích 9,944 vector dịch chuyển $\\Delta = X_{t+1} - X_t$ của tập Train dùng để fit PCA (3 thành phần chính đầu tiên giải thích 21.48% phương sai). Độ tương đồng cosine tuyệt đối trung bình đạt ~0.051 được đo trên 32 gradient biên quyết định của tập validation. Vì gradient là pháp tuyến của mặt mức margin, cosine nhỏ chỉ ra rằng các vector biến thiên tự nhiên có xu hướng tiếp tuyến cục bộ với mặt mức, chứ không phải trực giao với ranh giới quyết định. Đây là khảo sát hình học định tính, không phải định lý bất biến toàn cục.'
)

# -----------------------------------------------------------------------------
# 5. FIX P1-11: 24 HYPOTHESES COUNTERS & SCRIPT
# -----------------------------------------------------------------------------
old_buttons_hyp = '''                    <!-- Category Filter Buttons -->
                    <div class="flex flex-wrap gap-2 pt-2 border-b border-black pb-3">
                        <button onclick="filterHypotheses('all')" id="btn-hyp-all" class="px-3 py-1 border border-black bg-black text-white font-mono text-xs font-bold">Tất Cả (24)</button>
                        <button onclick="filterHypotheses('DERIVED')" id="btn-hyp-der" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Derived / Đã Chứng Minh (6)</button>
                        <button onclick="filterHypotheses('CONDITIONAL')" id="btn-hyp-cnd" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Conditional / Có Điều Kiện (5)</button>
                        <button onclick="filterHypotheses('FALSIFIED')" id="btn-hyp-fal" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Falsified / Bị Bác Bỏ (7)</button>
                        <button onclick="filterHypotheses('NOT_ESTAB')" id="btn-hyp-not" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Not Established / Chưa Rõ (6)</button>
                    </div>'''

new_buttons_hyp = '''                    <!-- Category Filter Buttons (Exact 24 Hypotheses: 7 / 3 / 9 / 5) -->
                    <div class="flex flex-wrap gap-2 pt-2 border-b border-black pb-3">
                        <button onclick="filterHypotheses('all')" id="btn-hyp-all" class="px-3 py-1 border border-black bg-black text-white font-mono text-xs font-bold">Tất Cả (24)</button>
                        <button onclick="filterHypotheses('DERIVED')" id="btn-hyp-der" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Derived / Đã Chứng Minh (7)</button>
                        <button onclick="filterHypotheses('CONDITIONAL')" id="btn-hyp-cnd" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Conditional / Có Điều Kiện (3)</button>
                        <button onclick="filterHypotheses('FALSIFIED')" id="btn-hyp-fal" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Falsified / Bị Bác Bỏ (9)</button>
                        <button onclick="filterHypotheses('NOT_ESTAB')" id="btn-hyp-not" class="px-3 py-1 border border-black bg-white text-zinc-700 hover:bg-zinc-100 font-mono text-xs">Not Established &amp; Descriptive (5)</button>
                    </div>'''

html = html.replace(old_buttons_hyp, new_buttons_hyp)

# -----------------------------------------------------------------------------
# 6. FIX P0-05: CODE SNIPPETS
# -----------------------------------------------------------------------------
old_code_snippet = '''                        <pre id="code-pane-3" class="code-pane hidden"><code># ecdr/margin_verifier.py - Điều Kiện Bảo Toàn Quyết Định Trên Toàn Bộ 11 Biên
import numpy as np

def verify_11_competing_margins(anchor_class: int, logits_lower_bounds: np.ndarray) -> bool:
    """
    Điều kiện toán học ECDR: Quyết định c được bảo toàn trên miền S nếu:
    inf_{X in S} (F_c(X) - F_j(X)) > 0 đối với MỌI j != c (11 competing classes).
    """
    c = anchor_class
    c_lower = logits_lower_bounds[c]
    for j in range(12):
        if j == c:
            continue
        # Bắt buộc cận dưới của lớp c phải nghiêm ngặt lớn hơn cận trên của lớp j
        if c_lower - logits_upper_bounds[j] <= 0:
            return False # Không đủ chứng nhận -> Fallback CNN
    return True # Hợp lệ -> Reuse nhãn c</code></pre>'''

new_code_snippet = '''                        <pre id="code-pane-3" class="code-pane hidden"><code># GHI CHÚ: MINH HỌA KHÁI NIỆM RÚT GỌN - KHÔNG PHẢI MÃ NGUỒN CHỨNG NHẬN SỐ HỌC THỰC THI
# ecdr/margin_verifier.py - Điều Kiện Bảo Toàn Quyết Định Trên Toàn Bộ 11 Biên
import numpy as np

def verify_11_competing_margins(anchor_class: int, logits_lower_bounds: np.ndarray, logits_upper_bounds: np.ndarray) -> bool:
    """
    Điều kiện toán học ECDR: Quyết định c được bảo toàn trên miền S nếu:
    inf_{X in S} (F_c(X) - F_j(X)) > 0 đối với MỌI j != c (11 competing classes).
    """
    c = anchor_class
    c_lower = logits_lower_bounds[c]
    for j in range(12):
        if j == c:
            continue
        # Bắt buộc cận dưới của lớp c phải nghiêm ngặt lớn hơn cận trên của lớp j
        if c_lower - logits_upper_bounds[j] <= 0:
            return False # Không đủ chứng nhận -> Fallback CNN
    return True # Hợp lệ -> Reuse nhãn c</code></pre>'''

html = html.replace(old_code_snippet, new_code_snippet)

old_bench_snippet = '''                        <pre id="code-pane-4" class="code-pane hidden"><code># scripts/benchmark_cpu.py - Giao Thức Đo Độ Trễ Chuẩn
import time
import torch

# Đo kiểm CPU Intel Xeon W-11855M @ 3.20GHz, 1 thread, FP32
model.eval()
warmup_runs = 100
timed_runs = 1000'''

new_bench_snippet = '''                        <pre id="code-pane-4" class="code-pane hidden"><code># GHI CHÚ: MINH HỌA KHÁI NIỆM RÚT GỌN - XEM scripts/week2_model_audit.py ĐỂ CHẠY CHÍNH XÁC
# scripts/benchmark_cpu.py - Giao Thức Đo Độ Trễ Chuẩn
import time
import torch
from scripts.backbone import SmallCNN1D

model = SmallCNN1D(num_classes=12)
model.eval()
sample_input = torch.randn(1, 6, 128)
warmup_runs = 100
timed_runs = 1000'''

html = html.replace(old_bench_snippet, new_bench_snippet)

# -----------------------------------------------------------------------------
# 7. FIX FULLSCREEN SLIDES DECK (FULL_DECKS)
# -----------------------------------------------------------------------------
# We rewrite FULL_DECKS in the JS section with 100% verified numbers from the audit.
new_full_decks_js = r'''
const FULL_DECKS = {
    1: [
        {
            id: 1,
            badge: "WEEK 1 • EXECUTIVE BRIEF",
            title: "Event-Conditioned Certified Decision Reuse (ECDR)",
            subtitle: "Tái sử dụng quyết định có chứng nhận trên luồng cảm biến tuần tự (50 Hz)",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-zinc-50 rounded">
                            <span class="text-xs font-mono font-bold uppercase text-zinc-500 block">BÀI TOÁN CỐT LÕI (CORE THESIS)</span>
                            <p class="text-sm font-serif text-zinc-800 leading-relaxed mt-1">
                                Khi xử lý luồng cảm biến liên tục, việc chạy mạng neural trên mọi cửa sổ trượt (50% overlap) có thể tạo ra các phép tính lặp lại. <strong>ECDR nghiên cứu cơ chế chứng nhận toán học</strong> để xác định khi nào có thể bỏ qua forward pass mà vẫn bảo toàn 100% quyết định của mô hình tham chiếu cố định mà không cần đoán mò!
                            </p>
                        </div>
                        <div class="grid grid-cols-2 gap-3 text-left font-mono">
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Mô hình tham chiếu</span>
                                <span class="text-xl font-bold text-black block">SmallCNN1D</span>
                                <span class="text-[10px] text-zinc-600">283,692 tham số • Frozen K=12</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Chi phí Conv/Linear</span>
                                <span class="text-xl font-bold text-black block">1.173M MACs</span>
                                <span class="text-[10px] text-zinc-600">1,172,992 phép tính MACs</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Mục tiêu bảo toàn</span>
                                <span class="text-xl font-bold text-black block">0 Disagreement</span>
                                <span class="text-[10px] text-zinc-600">Fail-closed: 0 tái sử dụng sai</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Kiểm định Tuần 1</span>
                                <span class="text-xl font-bold text-emerald-700 block">65/65 PASS</span>
                                <span class="text-[10px] text-zinc-600">Độc lập & nhất quán dữ liệu</span>
                            </div>
                        </div>
                    </div>
                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/pipeline_flowchart.png" alt="ECDR High Level Flow" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Sơ đồ tổng thể: Luồng Cảm biến → Cửa sổ [6×128] → ECDR Gateway → Reuse / Fallback</span>
                    </div>
                </div>
            `,
            takeaway: "Mục tiêu Tuần 1: Thiết lập nền tảng dự án, đóng băng mô hình tham chiếu chuẩn mực F(X), chuẩn hóa luồng dữ liệu không rò rỉ."
        },
        {
            id: 2,
            badge: "WEEK 1 • GOVERNANCE FRAMEWORK",
            title: "Khung Quản Trị Đề Xuất Tuần 1: Khung 8 Câu Hỏi Bắt Buộc",
            subtitle: "Theo dõi toàn diện sản phẩm bàn giao, tiến độ dự kiến, phân công vai trò và quản trị rủi ro",
            content: `
                <div class="overflow-x-auto">
                    <table class="w-full text-xs font-mono border-collapse border border-black">
                        <thead>
                            <tr class="bg-black text-white text-left">
                                <th class="p-2 border border-black w-28">Câu hỏi</th>
                                <th class="p-2 border border-black">Nội dung giải trình & Kế hoạch tuần 1</th>
                                <th class="p-2 border border-black w-36">Minh chứng (Evidence)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-200 bg-white">
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">1. What</td>
                                <td class="p-2 border border-black">Bàn giao pipeline trích xuất HAPT, kiến trúc SmallCNN1D K=12, checkpoint Seed 43 Epoch 8 và tài liệu Project Brief.</td>
                                <td class="p-2 border border-black text-zinc-600"><code>00_OVERVIEW/</code></td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">2. When</td>
                                <td class="p-2 border border-black">Khóa baseline ngày 22/10/2026; nghiệm thu đóng gói tài liệu ngày 03/11/2026. Tiến độ theo kế hoạch WBS.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">ON SCHEDULE</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">3. Why</td>
                                <td class="p-2 border border-black">Khóa mô hình tham chiếu F(X) làm hàm toán học chuẩn để xác định ranh giới tái sử dụng quyết định, tách biệt hoàn toàn với bài toán train tối ưu accuracy.</td>
                                <td class="p-2 border border-black text-zinc-600">Frozen Contract</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">4. Who</td>
                                <td class="p-2 border border-black">Phân công đề xuất: Data Owner (Tiền xử lý & Z-score); Model Owner (SmallCNN1D & checkpoint); Evaluation Owner (Test harness 65 tests).</td>
                                <td class="p-2 border border-black text-zinc-600">Role Allocation</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">5. How much</td>
                                <td class="p-2 border border-black">Kế hoạch dự kiến 80h; thực tế phát sinh thêm thời gian xử lý Z-score Train-Only để bảo đảm tuyệt đối không rò rỉ dữ liệu.</td>
                                <td class="p-2 border border-black text-zinc-700 font-bold">Projected Effort</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">6. Top risks</td>
                                <td class="p-2 border border-black">Mất cân bằng dữ liệu giữa hoạt động tĩnh (Sit, Laying) và hoạt động chuyển thế (Sit-to-Stand) làm giảm độ tự tin phân loại.</td>
                                <td class="p-2 border border-black text-zinc-600">G2 Risk Audit</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">7. Major changes</td>
                                <td class="p-2 border border-black">Khóa chặt cấu hình tham chiếu SmallCNN1D (K=12, Flatten-MLP); không kế thừa các giả định của mô hình Conv-GAP cũ.</td>
                                <td class="p-2 border border-black text-zinc-600">Scope Revision</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">8. Key issues</td>
                                <td class="p-2 border border-black">Xác lập ranh giới hình học của miền đầu vào quanh mỏ neo sao cho không bị rò rỉ dữ liệu giữa các subject.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">Subject Split 17/4/9</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `,
            takeaway: "Khung quản trị giúp tutor nắm bắt cấu trúc phân rã công việc, ranh giới phạm vi và kế hoạch kiểm soát rủi ro của nhóm."
        },
        {
            id: 3,
            badge: "WEEK 1 SPECIAL • SWOT ANALYSIS",
            title: "Phân Tích SWOT Nhóm Nghiên Cứu (Group SWOT)",
            subtitle: "Đánh giá khách quan năng lực phương pháp luận, hạ tầng kỹ thuật và các thách thức nghiên cứu",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="p-4 border-2 border-emerald-700 bg-emerald-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-emerald-700 pb-1">
                            <span class="font-mono font-bold text-emerald-900 text-sm">STRENGTHS (ĐIỂM MẠNH HỆ THỐNG)</span>
                            <span class="text-xs font-mono bg-emerald-700 text-white px-2 py-0.5 rounded">Phương pháp</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Thiết lập bộ kiểm thử tự động với 65/65 checks baseline PASS, bảo đảm tính tái lập độc lập.</li>
                            <li>Khóa bất biến checkpoint, tham số chuẩn hóa và cấu hình bằng mã băm SHA256.</li>
                            <li>Tách biệt tường minh giữa việc đánh giá độ chính xác (accuracy) và bài toán bảo toàn quyết định (decision preservation).</li>
                        </ul>
                    </div>

                    <div class="p-4 border-2 border-amber-700 bg-amber-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-amber-700 pb-1">
                            <span class="font-mono font-bold text-amber-900 text-sm">WEAKNESSES (GIỚI HẠN HIỆN TẠI)</span>
                            <span class="text-xs font-mono bg-amber-700 text-white px-2 py-0.5 rounded">Hạ tầng & Dữ liệu</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Dữ liệu chuyển thế có số lượng mẫu rất hạn chế (32–68 cửa sổ test), làm độ tự tin mô hình bị giảm.</li>
                            <li>Chưa có đo đạc thực nghiệm về điện năng tiêu thụ (energy/thermal) trực tiếp trên thiết bị smartwatch phần cứng.</li>
                            <li>Thư viện auto-LiRPA tiêu chuẩn không hỗ trợ trực tiếp toán tử MaxPool1D thời gian trên PyTorch hiện tại.</li>
                        </ul>
                    </div>

                    <div class="p-4 border-2 border-blue-700 bg-blue-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-blue-700 pb-1">
                            <span class="font-mono font-bold text-blue-900 text-sm">OPPORTUNITIES (CƠ HỘI KHOA HỌC)</span>
                            <span class="text-xs font-mono bg-blue-700 text-white px-2 py-0.5 rounded">Định hướng</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Bài toán tiết kiệm chi phí suy luận (Inference Efficiency) trên thiết bị biên có ý nghĩa học thuật và ứng dụng cao.</li>
                            <li>Tập dữ liệu UCI HAPT công khai chuẩn mực với 17,453 cửa sổ toàn luồng và 12 nhãn chi tiết.</li>
                            <li>Khả năng mở rộng từ chứng nhận toàn phần (decision reuse) sang tái sử dụng đặc trưng một phần (feature reuse).</li>
                        </ul>
                    </div>

                    <div class="p-4 border-2 border-rose-700 bg-rose-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-rose-700 pb-1">
                            <span class="font-mono font-bold text-rose-900 text-sm">THREATS (RỦI RO NGHIÊN CỨU)</span>
                            <span class="text-xs font-mono bg-rose-700 text-white px-2 py-0.5 rounded">Thách thức</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Rủi ro độ lỏng của cận IBP (Interval Bound Looseness) khiến bán kính chứng nhận nhỏ hơn biến thiên tự nhiên.</li>
                            <li>Overhead của bộ kiểm chứng verifier có thể lớn hơn thời gian forward của mô hình CNN nhỏ (tiết kiệm âm).</li>
                            <li>Khoảng cách số học giữa định lý số thực và phép thực thi float32 dấu chấm động cần được chứng minh chặt chẽ.</li>
                        </ul>
                    </div>
                </div>
            `,
            takeaway: "Phân tích SWOT định vị rõ các rủi ro cốt lõi về độ lỏng của cận IBP và bất đối xứng chi phí để xây dựng các bài test kiểm chứng sớm."
        },
        {
            id: 4,
            badge: "WEEK 1 SPECIAL • COMMUNICATION PLAN",
            title: "Kế Hoạch Giao Tiếp & Phối Hợp Nhóm (Communication Plan)",
            subtitle: "Quy chuẩn trao đổi thông tin, nhịp họp định kỳ và quy trình xử lý nút thắt (Blockers)",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="p-4 border border-black bg-zinc-50 rounded space-y-3">
                        <div class="flex items-center space-x-2">
                            <span class="w-6 h-6 rounded-full bg-black text-white flex items-center justify-center font-mono text-xs font-bold">1</span>
                            <h4 class="font-serif font-bold text-black text-sm">Kênh Trao Đổi Thông Tin</h4>
                        </div>
                        <p class="text-xs font-serif text-zinc-700">Tổ chức không gian làm việc số tập trung:</p>
                        <div class="text-xs font-mono space-y-1 bg-white p-2 border border-zinc-300 rounded">
                            <div>• <strong>Discord/Slack:</strong> Trao đổi hàng ngày</div>
                            <div>• <code>#general</code>: Thông báo chung</div>
                            <div>• <code>#code-review</code>: Review PR, CI logs</div>
                            <div>• <code>#data-math</code>: Thảo luận toán & cận</div>
                        </div>
                    </div>

                    <div class="p-4 border border-black bg-zinc-50 rounded space-y-3">
                        <div class="flex items-center space-x-2">
                            <span class="w-6 h-6 rounded-full bg-black text-white flex items-center justify-center font-mono text-xs font-bold">2</span>
                            <h4 class="font-serif font-bold text-black text-sm">Nhịp Họp & Đồng Bộ (Cadence)</h4>
                        </div>
                        <p class="text-xs font-serif text-zinc-700">Đảm bảo luồng thông tin không đứt gãy:</p>
                        <div class="text-xs font-mono space-y-1 bg-white p-2 border border-zinc-300 rounded">
                            <div>• <strong>Standup:</strong> 15 phút, 2 ngày/lần</div>
                            <div>• <strong>Nội dung:</strong> Đã làm gì? Sẽ làm gì? Vướng gì?</div>
                            <div>• <strong>Weekly Review:</strong> Chốt biên bản nghiệm thu tuần và đồng bộ dữ liệu slide</div>
                        </div>
                    </div>

                    <div class="p-4 border border-black bg-zinc-50 rounded space-y-3">
                        <div class="flex items-center space-x-2">
                            <span class="w-6 h-6 rounded-full bg-black text-white flex items-center justify-center font-mono text-xs font-bold">3</span>
                            <h4 class="font-serif font-bold text-black text-sm">Quy Trình Leo Thang (Escalation)</h4>
                        </div>
                        <p class="text-xs font-serif text-zinc-700">Giải phóng nút thắt kỹ thuật nhanh:</p>
                        <div class="text-xs font-mono space-y-1 bg-white p-2 border border-zinc-300 rounded">
                            <div>• <strong>Ngưỡng 4 giờ:</strong> Vấn đề kỹ thuật tắc quá 4h phải đưa lên nhóm</div>
                            <div>• <strong>Pairing:</strong> Bố trí 2 thành viên cùng debug</div>
                            <div>• <strong>Tutor Consultation:</strong> Đặt lịch hỏi Tutor trong giờ Question Time</div>
                        </div>
                    </div>
                </div>

                <div class="mt-4 p-3 border-l-4 border-black bg-zinc-100 rounded text-xs font-mono flex items-center justify-between">
                    <span><strong>Quy chuẩn Git:</strong> Mọi thay đổi code đều qua GitHub PR; bắt buộc chạy test harness kiểm tra toàn vẹn trước khi merge!</span>
                    <span class="bg-black text-white px-2 py-0.5 rounded text-[10px]">Branch: main</span>
                </div>
            `,
            takeaway: "Giao tiếp minh bạch, có kỷ luật nhịp họp và quy tắc leo thang rõ ràng giúp nhóm kiểm soát rủi ro tiến độ ngay từ đầu."
        },
        {
            id: 5,
            badge: "WEEK 1 SPECIAL • DOMAIN & DATASET",
            title: "Tổng Quan Dự Án & Luận Cứ Chọn Lĩnh Vực / Dữ Liệu",
            subtitle: "Nhận diện hoạt động con người trên cảm biến đeo (UCI HAPT) là môi trường phù hợp để kiểm chứng",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="space-y-4">
                        <div class="p-4 border border-black bg-zinc-50 rounded space-y-2">
                            <h4 class="font-serif font-bold text-black text-sm uppercase">1. Đề tài của chúng tôi là gì?</h4>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Dự án nghiên cứu <strong>Event-Conditioned Certified Decision Reuse (ECDR)</strong>: Khảo sát cơ chế chứng nhận toán học đặt trước mô hình mạng nơ-ron tích chập cố định (SmallCNN1D). Mục tiêu là chứng minh trực tiếp rằng quyết định của mô hình trên cửa sổ trước vẫn được bảo toàn cho cửa sổ mới, nhằm xác định điều kiện khi nào việc bỏ qua forward pass là an toàn và có lợi ích chi phí thực tế.
                            </p>
                        </div>

                        <div class="p-4 border border-black bg-zinc-50 rounded space-y-2">
                            <h4 class="font-serif font-bold text-black text-sm uppercase">2. Động lực từ thiết bị đeo (Wearables)</h4>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Cảm biến trên thiết bị đeo lấy mẫu liên tục (50Hz), tạo ra khối lượng công việc suy luận liên tục. Việc nghiên cứu giảm bớt các phép tính suy luận dư thừa mà không dựa trên phỏng đoán heuristic không an toàn là một hướng đi quan trọng cho các ứng dụng tiết kiệm năng lượng.
                            </p>
                        </div>
                    </div>

                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                            <h4 class="font-serif font-bold text-black text-sm uppercase">3. Lý do chọn tập dữ liệu UCI HAPT</h4>
                            <div class="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                                <div class="p-2 border border-zinc-300 rounded bg-zinc-50">
                                    <span class="text-zinc-500 text-[10px] block">Tần số</span>
                                    <span class="font-bold text-black text-sm">50 Hz</span>
                                    <span class="text-[9px] text-zinc-600 block">Thời gian thực</span>
                                </div>
                                <div class="p-2 border border-zinc-300 rounded bg-zinc-50">
                                    <span class="text-zinc-500 text-[10px] block">Số kênh</span>
                                    <span class="font-bold text-black text-sm">6 Kênh</span>
                                    <span class="text-[9px] text-zinc-600">3 Accel + 3 Gyro</span>
                                </div>
                                <div class="p-2 border border-zinc-300 rounded bg-zinc-50">
                                    <span class="text-zinc-500 text-[10px] block">Số lớp</span>
                                    <span class="font-bold text-black text-sm">12 Lớp</span>
                                    <span class="text-[9px] text-zinc-600">6 Cơ bản + 6 Chuyển thế</span>
                                </div>
                            </div>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1 list-disc list-inside">
                                <li><strong>Tính liên tục thời gian:</strong> Bước trượt Stride 64 (50% overlap) chia sẻ 384 tọa độ giữa 2 cửa sổ liên tiếp, tạo tiền đề kiểm tra tính bất biến cục bộ.</li>
                                <li><strong>Thách thức thực sự:</strong> 6 lớp chuyển thế (Postural Transitions) mang tính thách thức cao với ranh giới phi tuyến.</li>
                                <li><strong>Tập dữ liệu chuẩn mực:</strong> 30 tình nguyện viên, 61 phiên ghi âm có video đối chiếu xác thực nhãn.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            `,
            takeaway: "UCI HAPT là môi trường thực nghiệm chuẩn mực với chuỗi thời gian liên tục 50Hz để khảo sát tính khả thi của bài toán tái sử dụng quyết định."
        },
        {
            id: 6,
            badge: "WEEK 1 SPECIAL • WBS 1 SLIDE",
            title: "Cấu Trúc Phân Rã Công Việc Toàn Dự Án AI (Project WBS)",
            subtitle: "Phân bổ 4 giai đoạn nghiên cứu từ thiết lập nền tảng đến nghiệm thu hệ thống",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono">
                    <div class="p-3 border-2 border-black bg-zinc-100 rounded space-y-2">
                        <div class="bg-black text-white px-2 py-1 font-bold text-center rounded">TUẦN 1: FOUNDATION</div>
                        <div class="text-[11px] text-zinc-600 font-bold uppercase">Baseline & Ingestion</div>
                        <ul class="space-y-1 text-zinc-800 list-disc list-inside text-[11px]">
                            <li>Ingestion dữ liệu HAPT</li>
                            <li>Huấn luyện SmallCNN1D</li>
                            <li>Chọn Checkpoint Seed 43</li>
                            <li>Đóng băng tham số F(X)</li>
                            <li>Bộ kiểm thử 65/65 PASS</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-emerald-700 font-bold">✔ Gate G1 PASS</div>
                    </div>

                    <div class="p-3 border-2 border-black bg-zinc-50 rounded space-y-2">
                        <div class="bg-zinc-800 text-white px-2 py-1 font-bold text-center rounded">TUẦN 2: DATA DYNAMICS</div>
                        <div class="text-[11px] text-zinc-600 font-bold uppercase">Physics & Leakage</div>
                        <ul class="space-y-1 text-zinc-800 list-disc list-inside text-[11px]">
                            <li>Phân tích 1.21M mẫu thô</li>
                            <li>Cơ chế trích xuất [6×128]</li>
                            <li>Chuẩn hóa Train-Only Z-Score</li>
                            <li>Subject Split 17/4/9</li>
                            <li>Kiểm toán 1,172,992 MACs</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-amber-700 font-bold">! Gate G2 HOLD (Trans)</div>
                    </div>

                    <div class="p-3 border-2 border-black bg-white rounded space-y-2">
                        <div class="bg-zinc-700 text-white px-2 py-1 font-bold text-center rounded">TUẦN 3: CERTIFICATION</div>
                        <div class="text-[11px] text-zinc-600 font-bold uppercase">Math & Verifier</div>
                        <ul class="space-y-1 text-zinc-800 list-disc list-inside text-[11px]">
                            <li>Toán học Margin Theorem</li>
                            <li>Chẩn đoán IBP trên đồ thị thật</li>
                            <li>Chứng minh H13 Bitwise</li>
                            <li>Khảo sát Blocker A (Radius)</li>
                            <li>Đo Net Speedup (Blocker B)</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-zinc-700 font-bold">✔ 93/93 Tests PASS (G3 Part)</div>
                    </div>

                    <div class="p-3 border-2 border-black bg-white rounded space-y-2">
                        <div class="bg-zinc-600 text-white px-2 py-1 font-bold text-center rounded">TUẦN 4: BENCHMARK</div>
                        <div class="text-[11px] text-zinc-600 font-bold uppercase">Hypotheses & Tests</div>
                        <ul class="space-y-1 text-zinc-800 list-disc list-inside text-[11px]">
                            <li>Thử nghiệm Adaptive Bounds</li>
                            <li>Thử nghiệm Early Trigger O(1)</li>
                            <li>Thử nghiệm Incremental Reuse</li>
                            <li>Đo đạc benchmark toàn diện</li>
                            <li>Đóng gói Báo cáo cuối kỳ</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-blue-700 font-bold">★ PLANNED MILESTONES</div>
                    </div>
                </div>

                <div class="mt-4 p-3 border border-black bg-zinc-50 rounded flex items-center justify-between text-xs font-mono">
                    <span><strong>Quy tắc WBS:</strong> Mỗi tuần có sản phẩm bàn giao kiểm chứng được độc lập; phân định rõ giữa kết quả đã đo và kế hoạch phía trước.</span>
                    <span class="font-bold text-black">Total: 4 Phases • 16 Work Packages</span>
                </div>
            `,
            takeaway: "Bản đồ phân rã công việc WBS 1 slide định vị chính xác lộ trình 4 tuần, phân tách rõ các kết quả đã kiểm chứng và mục tiêu kế tiếp."
        },
        {
            id: 7,
            badge: "WEEK 1 SPECIAL • REPO STRUCTURE",
            title: "Cấu Trúc Kho Mã Nguồn GitHub & Quy Chuẩn Thực Thi",
            subtitle: "Tổ chức mô-đun hóa dựa trên cấu trúc repository tducn110/ecdr-sensor-inference",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                    <div class="font-mono text-xs bg-zinc-900 text-zinc-100 p-4 rounded-lg overflow-x-auto shadow-inner">
                        <pre>tducn110/ecdr-sensor-inference/
├── scripts/                 # Kịch bản thực thi & kiểm toán
│   ├── backbone.py          # Kiến trúc chuẩn SmallCNN1D
│   ├── 03_build_windows.py  # Trích xuất cửa sổ HAPT
│   ├── week2_model_audit.py # Kiểm toán MACs & per-class
│   └── week3_experiments.py # Replay & đo độ trễ
├── ecdr/                    # Gói lõi ECDR
│   ├── week3_verifier.py    # Bộ kiểm chứng IBP trên đồ thị thật
│   ├── data.py              # Luồng tải dữ liệu tuần tự
│   └── preprocessing.py     # Chuẩn hóa Z-Score train-only
├── data/                    # Tệp cảm biến HAPT & bảng chuẩn hóa
├── configs/                 # Cấu hình siêu tham số (YAML)
├── tests/                   # 93 bài kiểm thử tự động (pytest/unittest)
├── notes/presents/          # Cổng thuyết trình Web & Slide Decks
└── output/                  # Nhật ký kết quả & receipts SHA256</pre>
                    </div>

                    <div class="space-y-3 font-serif text-xs text-zinc-800">
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">1. Tính Bất Biến Của Dữ Liệu</span>
                            Dữ liệu thô và các tệp checkpoint chuẩn được bảo vệ bằng mã băm SHA256, không thể bị ghi đè ngẫu nhiên trong quá trình chạy thử nghiệm.
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">2. CI/CD & Automated Verification</span>
                            Mỗi lượt commit đều chạy qua test harness với 93 bài kiểm tra đơn vị và tích hợp, bảo đảm không có thay đổi nào phá vỡ hợp đồng toán học.
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">3. Khả Năng Tái Lập (Reproducibility)</span>
                            Mọi thử nghiệm đều có script khởi chạy độc lập kèm cố định hạt giống ngẫu nhiên (Random Seed 43), tạo ra kết quả bitwise nhất quán trên các môi trường.
                        </div>
                    </div>
                </div>
            `,
            takeaway: "Kho mã nguồn được cấu trúc chính xác theo repo thực tế, phân định rõ ràng giữa module toán học, mô hình nơ-ron và kịch bản thực nghiệm."
        },
        {
            id: 8,
            badge: "WEEK 1 • MODEL ARCHITECTURE",
            title: "Mô Hình Tham Chiếu SmallCNN1D & Giao Thức Đóng Băng",
            subtitle: "Hàm toán học F(X) cố định 283,692 tham số & 1,172,992 Conv/Linear MACs",
            content: `
                <div class="space-y-4">
                    <div class="p-3 border-l-4 border-black bg-zinc-100 rounded text-xs font-serif text-zinc-800">
                        <strong>Nguyên lý khoa học sống còn:</strong> SmallCNN1D được huấn luyện, chọn checkpoint tối ưu tại <em>Epoch 8, Seed 43</em> và đóng băng toàn bộ 283,692 tham số. Nhóm nghiên cứu không tiếp tục chỉnh sửa trọng số để tăng accuracy trong tuyến ECDR, mà dùng mô hình này làm <strong>hàm toán học tham chiếu bất biến F(X)</strong> cho bài toán chứng nhận tái sử dụng quyết định!
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                        <div class="p-3 border border-black rounded bg-zinc-50 flex flex-col items-center">
                            <img src="assets/cnn_architecture_diagram.png" alt="SmallCNN1D Architecture" class="max-h-60 object-contain rounded">
                            <span class="text-[11px] font-mono text-zinc-500 mt-1">Cấu trúc SmallCNN1D: 3 Khối Conv1D (k=5, 3, 3) + 2 MaxPool + 2 Tầng Linear</span>
                        </div>

                        <div class="space-y-2 text-xs font-mono">
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Đầu vào (Input Shape):</span>
                                <span class="font-bold text-black">[Batch, 6 channels, 128 samples]</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Conv 1:</span>
                                <span class="font-bold text-black">Conv1D(6→32, k=5, p=2) + BN + ReLU + MaxPool(2)</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Conv 2:</span>
                                <span class="font-bold text-black">Conv1D(32→64, k=3, p=1) + BN + ReLU + MaxPool(2)</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Conv 3:</span>
                                <span class="font-bold text-black">Conv1D(64→64, k=3, p=1) + BN + ReLU</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Linear Phân Loại:</span>
                                <span class="font-bold text-black">Flatten(2048) → Linear(128) → Linear(12)</span>
                            </div>
                            <div class="p-2 border border-black rounded bg-black text-white flex justify-between">
                                <span>Tổng tham số / Chi phí Conv/Linear MACs:</span>
                                <span class="font-bold">283,692 params • 1,172,992 MACs</span>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            takeaway: "Khóa chặt kiến trúc (k=5, 3, 3) và tham số là cơ sở để thực hiện kiểm chứng hình thức trên đúng computation graph."
        }
    ],

    2: [
        {
            id: 1,
            badge: "WEEK 2 • GOVERNANCE FRAMEWORK",
            title: "Khung Quản Trị Đề Xuất Tuần 2: Khung 8 Câu Hỏi Bắt Buộc",
            subtitle: "Định vị dữ liệu, giải phẫu kiến trúc SmallCNN1D và quản trị rủi ro mất cân bằng",
            content: `
                <div class="overflow-x-auto">
                    <table class="w-full text-xs font-mono border-collapse border border-black">
                        <thead>
                            <tr class="bg-black text-white text-left">
                                <th class="p-2 border border-black w-28">Câu hỏi</th>
                                <th class="p-2 border border-black">Nội dung giải trình & Trạng thái tuần 2</th>
                                <th class="p-2 border border-black w-36">Minh chứng (Evidence)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-200 bg-white">
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">1. What</td>
                                <td class="p-2 border border-black">Bàn giao phân tích 1,215,488 mẫu tín hiệu thô HAPT, cơ chế trích xuất ma trận [6×128], Z-score Train-Only và kiểm toán 1,172,992 MACs.</td>
                                <td class="p-2 border border-black text-zinc-600"><code>01_DATA/</code></td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">2. When</td>
                                <td class="p-2 border border-black">Nghiệm thu toàn bộ kiểm toán dữ liệu và mô hình ngày 03/11/2026. Khớp tiến độ WBS.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">ON SCHEDULE</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">3. Why</td>
                                <td class="p-2 border border-black">Hiểu rõ cơ chế vật lý cảm biến và cấu trúc chồng lấn 50% thời gian là tiền đề thiết kế candidate region cho bộ chứng nhận.</td>
                                <td class="p-2 border border-black text-zinc-600">Stride Physics</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">4. Who</td>
                                <td class="p-2 border border-black">Phân công đề xuất: Data Analyst (Kiểm toán Z-score); Signal Engineer (Cắt lát cửa sổ & gán nhãn); Model Engineer (Đo đạc MACs tầng Linear).</td>
                                <td class="p-2 border border-black text-zinc-600">Role Allocation</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">5. How much</td>
                                <td class="p-2 border border-black">Kế hoạch dự kiến 75h; thực tế tập trung cho việc tính toán profile MACs chi tiết từng layer và đối soát 12,278 cửa sổ có giám sát.</td>
                                <td class="p-2 border border-black text-zinc-700 font-bold">Projected Effort</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">6. Top risks</td>
                                <td class="p-2 border border-black">Chênh lệch mẫu tới 29.1x giữa hoạt động tĩnh và chuyển thế khiến mô hình kém tự tin ở các nhãn chuyển thế (G2 HOLD).</td>
                                <td class="p-2 border border-black text-amber-700 font-bold">Gate G2 on HOLD</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">7. Major changes</td>
                                <td class="p-2 border border-black">Thiết lập quy tắc Subject Split 17/4/9 bất biến; không gộp dữ liệu các đối tượng để tính tham số chuẩn hóa.</td>
                                <td class="p-2 border border-black text-zinc-600">Zero Leakage Policy</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">8. Key issues</td>
                                <td class="p-2 border border-black">Cửa sổ 128 mẫu trượt 64 mẫu chia sẻ 50% dữ liệu nhưng bị dịch chuyển vị trí tọa độ (shift), không phải cùng tọa độ tuyệt đối.</td>
                                <td class="p-2 border border-black text-zinc-600">Coordinate Shift</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `,
            takeaway: "Tuần 2 giải quyết trọn vẹn nền tảng dữ liệu, loại bỏ hoàn toàn rủi ro rò rỉ dữ liệu và làm rõ bản chất chồng lấn thời gian."
        },
        {
            id: 2,
            badge: "WEEK 2 SPECIAL • REFINED PROBLEM",
            title: "Làm Rõ Bài Toán Nghiên Cứu & Ranh Giới Phạm Vi",
            subtitle: "Phân biệt rành mạch giữa bài toán phân loại hoạt động và bài toán tái sử dụng quyết định",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-zinc-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-zinc-500 block">HIỆN TƯỢNG VÀ NGUY CƠ TỪ PHỎNG ĐOÁN HEURISTIC</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Trong suy luận thời gian thực, 2 cửa sổ cảm biến kế tiếp nhau chia sẻ chính xác 64 mẫu (50% dữ liệu). Một cách tiếp cận thông thường là dựa vào <em>độ tương đồng tín hiệu hoặc độ tự tin softmax</em> để phỏng đoán nhãn không đổi.
                            </p>
                            <p class="text-xs font-serif text-rose-800 font-bold leading-relaxed">
                                ⚠ RỦI RO: Phỏng đoán dựa trên độ tương đồng không bảo đảm tính bất biến của quyết định! Ở các thời điểm chuyển giao tư thế, tín hiệu có thể tương đồng nhưng mạng neural có thể lật nhãn quyết định.
                            </p>
                        </div>

                        <div class="p-4 border-2 border-black bg-black text-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-zinc-400 block">ĐỊNH NGHĨA BÀI TOÁN ECDR CHUẨN XÁC</span>
                            <p class="text-xs font-serif text-zinc-200 leading-relaxed">
                                ECDR không đoán! ECDR đặt ra bài toán: <strong>Tìm một điều kiện chứng nhận toán học đủ</strong> trên miền đầu vào chứa cửa sổ mới, chứng minh rằng giá trị logit của lớp mỏ neo luôn lớn hơn toàn bộ 11 lớp cạnh tranh còn lại:
                            </p>
                            <div class="text-center font-mono text-xs bg-zinc-900 p-2 rounded border border-zinc-700 text-amber-300">
                                min_{j != c} inf_{X in S} [ F_c(X) - F_j(X) ] > 0
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/slide02_problem_statement.png" alt="Problem Statement Illustration" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Ranh giới phạm vi: Bảo toàn argmax(F(X)) chứ không tối ưu lại F(X)</span>
                    </div>
                </div>
            `,
            takeaway: "Ranh giới nghiên cứu được xác lập dứt khoát: Chỉ tái sử dụng khi có chứng nhận toán học bảo đảm an toàn."
        },
        {
            id: 3,
            badge: "WEEK 2 SPECIAL • 3 RESEARCH QUESTIONS",
            title: "Ba Câu Hỏi Nghiên Cứu Cốt Lõi (RQ1, RQ2, RQ3)",
            subtitle: "Hệ thống câu hỏi dẫn dắt toàn bộ thiết kế thực nghiệm và đánh giá hệ thống",
            content: `
                <div class="space-y-4">
                    <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-black pb-1">
                            <span class="font-mono font-bold text-black text-sm">RQ1: Cấu Trúc Miền Biến Thiên Đầu Vào (Input Region Geometry)</span>
                            <span class="text-xs font-mono bg-black text-white px-2 py-0.5 rounded">Toán học & Tín hiệu</span>
                        </div>
                        <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                            Cấu trúc biến thiên của chuỗi cảm biến HAPT (quán tính vật lý, bước trượt 50%) cho phép xây dựng những dạng miền ứng viên (Candidate Region S) nào đủ chặt để bao bọc các cửa sổ kế tiếp mà vẫn chứng minh được tính bất biến của quyết định?
                        </p>
                    </div>

                    <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-black pb-1">
                            <span class="font-mono font-bold text-black text-sm">RQ2: Điều Kiện Biên Logit & Tính Đúng Đắn Toán Học (Sound Margin Bounds)</span>
                            <span class="text-xs font-mono bg-black text-white px-2 py-0.5 rounded">Chứng nhận hình thức</span>
                        </div>
                        <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                            Phương pháp lan truyền cận khoảng (Interval Bound Propagation - IBP) trên đồ thị tính toán thực tế của SmallCNN1D có cung cấp được chặn dưới hiệu logit strictly positive (min_{j!=c} g_cj > 0) trên miền dữ liệu thực nghiệm hay không?
                        </p>
                    </div>

                    <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-black pb-1">
                            <span class="font-mono font-bold text-black text-sm">RQ3: Tỷ Lệ Tái Sử Dụng & Tăng Tốc Độ Trễ Toàn Trình Thực Tế (Net Speedup)</span>
                            <span class="text-xs font-mono bg-black text-white px-2 py-0.5 rounded">Hiệu năng hệ thống</span>
                        </div>
                        <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                            Khi tính gộp toàn bộ chi phí kiểm tra thành viên và chi phí fallback, cơ chế ECDR có mang lại tỷ lệ tăng tốc thực tế (Net Latency Speedup > 1.0x) so với việc chạy mô hình tham chiếu gốc trên thiết bị biên hay không?
                        </p>
                    </div>
                </div>
            `,
            takeaway: "3 RQs liên kết chặt chẽ: RQ1 (Dữ liệu) → RQ2 (Toán học chứng nhận) → RQ3 (Hiệu năng hệ thống thực tế)."
        },
        {
            id: 4,
            badge: "WEEK 2 SPECIAL • DATASET EXPLORATION",
            title: "Khám Phá Dữ Liệu UCI HAPT: 1.21M Mẫu Tín Hiệu Thô",
            subtitle: "Giải mã cơ chế vật lý của cảm biến gia tốc và con quay hồi chuyển trên 30 đối tượng",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-3 font-serif text-xs text-zinc-800">
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">Nguồn gốc & Thiết bị đo</span>
                            Dữ liệu thu thập từ 30 tình nguyện viên (19-48 tuổi) đeo smartphone Samsung Galaxy S II ở thắt lưng. Cảm biến lấy mẫu tại <strong>50 Hz</strong> (50 giá trị/giây).
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">6 Trục Cảm Biến Vật Lý</span>
                            <ul class="list-disc list-inside space-y-1 font-mono text-[11px] mt-1">
                                <li><strong>3 Trục Gia Tốc (Acc X, Y, Z):</strong> Đo gia tốc trọng trường và động học cơ thể (đơn vị: g).</li>
                                <li><strong>3 Trục Vận Tốc Góc (Gyro X, Y, Z):</strong> Đo vận tốc góc quay của thắt lưng (đơn vị: rad/s).</li>
                            </ul>
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">Quy Mô Toàn Tập</span>
                            <strong>1,215,488 mẫu tín hiệu thô</strong> chia thành 61 phiên ghi âm (122 tệp cảm biến). 17,453 cửa sổ trượt toàn luồng.
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/pirsch_raw_waveform_extracted.png" alt="Raw Waveform Extracted" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Dạng sóng 6 kênh thực tế: 3 trục Accel (trên) & 3 trục Gyro (dưới)</span>
                    </div>
                </div>
            `,
            takeaway: "Hiểu sâu đặc tính vật lý của 6 kênh cảm biến 50Hz là cơ sở để xử lý lọc nhiễu và thiết kế cửa sổ trượt chính xác."
        },
        {
            id: 5,
            badge: "WEEK 2 SPECIAL • WINDOWING & LABELS",
            title: "Trích Xuất Cửa Sổ [6×128] & Chính Sách Gán Nhãn Đa Số",
            subtitle: "Xây dựng ma trận tensor đầu vào 2.56 giây với độ chồng lấn 50% (Stride 64)",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-3 text-xs font-mono">
                        <div class="p-3 border border-black bg-white rounded space-y-1">
                            <span class="text-zinc-500 uppercase text-[10px] block">Kích thước cửa sổ (Window Size)</span>
                            <span class="text-lg font-bold text-black block">128 Mẫu (2.56 Giây tại 50Hz)</span>
                            <span class="text-zinc-600 font-serif text-[11px]">Đủ dài để chứa ít nhất một chu kỳ bước đi hoặc thao tác chuyển thế.</span>
                        </div>

                        <div class="p-3 border border-black bg-white rounded space-y-1">
                            <span class="text-zinc-500 uppercase text-[10px] block">Bước nhảy trượt (Stride)</span>
                            <span class="text-lg font-bold text-black block">64 Mẫu (1.28 Giây, 50% Overlap)</span>
                            <span class="text-zinc-600 font-serif text-[11px]">Tạo ra 384 tọa độ dữ liệu dùng chung giữa 2 cửa sổ kế tiếp nhau.</span>
                        </div>

                        <div class="p-3 border border-black bg-zinc-100 rounded space-y-1">
                            <span class="text-zinc-500 uppercase text-[10px] block">Phân loại 17,453 cửa sổ toàn luồng</span>
                            <span class="text-sm font-bold text-black block">12,278 Có Giám Sát (Supervised)</span>
                            <div class="text-[11px] font-serif text-zinc-800 space-y-0.5">
                                <div>• <strong>10,361 pure:</strong> 100% nhãn duy nhất.</div>
                                <div>• <strong>1,917 majority:</strong> Đa số chiếm từ 60% đến 99%.</div>
                                <div>• <strong>1,588 ambiguous:</strong> Bị loại do tranh chấp nhãn (&lt; 60%).</div>
                                <div>• <strong>3,587 unlabeled:</strong> Khoảng đệm giữa các hoạt động.</div>
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/pirsch_window_matrix_extraction.png" alt="Window Matrix Extraction" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Cơ chế cắt lát ma trận: 6 kênh × 128 điểm thời gian = Tensor [6, 128]</span>
                    </div>
                </div>
            `,
            takeaway: "Phân định rõ ràng 12,278 cửa sổ có giám sát và 5,175 cửa sổ bị loại trừ (ambiguous + unlabeled) bảo đảm tính nghiêm ngặt của tập dữ liệu."
        },
        {
            id: 6,
            badge: "WEEK 2 SPECIAL • DATA QUALITY AUDIT",
            title: "Kiểm Định Chất Lượng Dữ Liệu & Thách Thức Mất Cân Bằng Lớp",
            subtitle: "Không khuyết thiếu, không trùng lặp, nhưng mất cân bằng lớp trầm trọng lên tới 29.1 lần",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="grid grid-cols-2 gap-3 text-left font-mono">
                            <div class="p-3 border border-black rounded bg-emerald-50">
                                <span class="text-[10px] text-emerald-800 uppercase block font-bold">Giá trị khuyết thiếu (NaN)</span>
                                <span class="text-2xl font-bold text-emerald-900 block">0 Mẫu</span>
                                <span class="text-[10px] text-emerald-700">Tín hiệu thô hoàn chỉnh 100%</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-emerald-50">
                                <span class="text-[10px] text-emerald-800 uppercase block font-bold">Trùng lặp Timestamp</span>
                                <span class="text-2xl font-bold text-emerald-900 block">0 Mẫu</span>
                                <span class="text-[10px] text-emerald-700">Chuỗi thời gian liên tục 50Hz</span>
                            </div>
                        </div>

                        <div class="p-4 border-2 border-amber-700 bg-amber-50 rounded space-y-2">
                            <div class="flex items-center justify-between border-b border-amber-700 pb-1">
                                <span class="text-xs font-mono font-bold text-amber-900 uppercase">THÁCH THỨC MẤT CÂN BẰNG LỚP (29.1× IMBALANCE)</span>
                                <span class="text-[10px] font-mono bg-amber-700 text-white px-2 py-0.5 rounded">Gate G2 HOLD</span>
                            </div>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Trong thực tế, các hoạt động tĩnh kéo dài hàng chục phút, trong khi chuyển thế chỉ diễn ra trong 2-3 giây. Do đó, các lớp tĩnh có hàng nghìn cửa sổ, trong khi 6 lớp chuyển thế chỉ có 32–68 cửa sổ trong tập test!
                            </p>
                            <div class="text-xs font-mono text-zinc-900 bg-white p-2 border border-amber-300 rounded">
                                Lớp nhiều nhất (Laying): 1,944 cửa sổ vs Lớp ít nhất (Sit-to-Lie): chỉ 67 cửa sổ toàn tập!
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/data_policy_support.png" alt="Class Support Imbalance" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Phân bố hỗ trợ mẫu giữa 6 hoạt động cơ bản và 6 lớp chuyển thế</span>
                    </div>
                </div>
            `,
            takeaway: "Phân tích chất lượng dữ liệu chỉ ra nguyên nhân gốc rễ khiến G2 bị giữ ở trạng thái HOLD: sự chênh lệch mẫu tự nhiên giữa các hoạt động."
        },
        {
            id: 7,
            badge: "WEEK 2 SPECIAL • BIAS & LEAKAGE",
            title: "Kiểm Soát Rò Rỉ Dữ Liệu: Subject Split 17/4/9 & Train-Only Z-Score",
            subtitle: "Nguyên tắc thiết kế dữ liệu sống còn: Triệt tiêu 100% rò rỉ thông tin cá nhân",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-zinc-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">SUBJECT SPLIT 17 / 4 / 9: KHÔNG TRỘN NGƯỜI DÙNG</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Dữ liệu được chia theo đối tượng người dùng độc lập: 17 người Train, 4 người Validation và 9 người Test. Ngăn chặn triệt để hiện tượng mô hình học thuộc đặc điểm cá nhân.
                            </p>
                            <div class="grid grid-cols-3 gap-2 text-center font-mono text-xs pt-1">
                                <div class="p-2 border border-black rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Train (17 người)</span>
                                    <span class="font-bold text-black text-sm">6,955</span>
                                    <span class="text-[9px] text-zinc-500">Supervised (9,978 full)</span>
                                </div>
                                <div class="p-2 border border-black rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Val (4 người)</span>
                                    <span class="font-bold text-black text-sm">1,751</span>
                                    <span class="text-[9px] text-zinc-500">Supervised (2,422 full)</span>
                                </div>
                                <div class="p-2 border border-black rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Test (9 người)</span>
                                    <span class="font-bold text-black text-sm">3,572</span>
                                    <span class="text-[9px] text-zinc-500">Supervised (5,053 full)</span>
                                </div>
                            </div>
                        </div>

                        <div class="p-3 border-l-4 border-black bg-zinc-100 rounded text-xs font-serif text-zinc-800 space-y-1">
                            <span class="font-mono font-bold text-black block text-[11px]">CHUẨN HÓA Z-SCORE TRAIN-ONLY:</span>
                            Giá trị Trung bình (Mean) và Độ lệch chuẩn (Std) của 6 kênh được tính <strong>độc quyền trên 17 người dùng Train</strong>. Tập Validation và Test chỉ áp dụng lại bộ tham số này, loại trừ rò rỉ phân phối.
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/data_split_multiplicity.png" alt="Data Split Multiplicity" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Bảo vệ ranh giới đánh giá: Zero Data Leakage giữa các tập dữ liệu</span>
                    </div>
                </div>
            `,
            takeaway: "Quy chuẩn Subject Split 17/4/9 và Train-Only Z-Score bảo đảm kết quả đánh giá mang tính tổng quát hóa thực sự."
        },
        {
            id: 8,
            badge: "WEEK 2 SPECIAL • ROADMAP W3",
            title: "Cập Nhật WBS & Kế Hoạch Chuyển Giao Tuần 3",
            subtitle: "Hoàn tất giai đoạn dữ liệu, chuẩn bị khởi động động cơ chứng nhận toán học hình thức",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border border-black bg-zinc-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-emerald-800 block">✔ TIẾN ĐỘ ĐÃ HOÀN THÀNH (TUẦN 2 CLOSEOUT)</span>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1 list-disc list-inside">
                                <li>Kiểm toán toàn bộ 1.21 triệu mẫu tín hiệu thô HAPT.</li>
                                <li>Chuẩn hóa Z-score Train-Only và cấu trúc tensor [6, 128].</li>
                                <li>Kiểm toán 1,172,992 Conv/Linear MACs của kiến trúc SmallCNN1D.</li>
                                <li>Lập bảng đối soát 12x12 Confusion Matrix và xác định rủi ro G2 HOLD.</li>
                            </ul>
                        </div>

                        <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">➔ KẾ HOẠCH TRIỂN KHAI TUẦN 3 (CHUYỂN GIAO)</span>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                                <li><strong>Toán học chứng nhận:</strong> Xây dựng định lý Margin Theorem và kỹ thuật Interval Bound Propagation (IBP).</li>
                                <li><strong>Bộ Verifier số học:</strong> Lập trình module <code>week3_verifier.py</code> chạy trực tiếp trên computation graph của SmallCNN1D.</li>
                                <li><strong>Benchmark thực nghiệm:</strong> Replay luồng dữ liệu 4 phiên, đo đạc Net Latency Speedup và phân tích Blocker A & B.</li>
                            </ul>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/slide12_wbs_roadmap.png" alt="WBS Roadmap" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Bản đồ lộ trình chuyển giao công việc giữa Tuần 2 và Tuần 3</span>
                    </div>
                </div>
            `,
            takeaway: "Kế hoạch Tuần 3 chuyển dịch dứt khoát từ nền tảng dữ liệu sang thẩm định toán học và kiểm chứng tốc độ thực thi."
        }
    ],

    3: [
        {
            id: 1,
            badge: "WEEK 3 • GOVERNANCE FRAMEWORK",
            title: "Khung Quản Trị Đề Xuất Tuần 3: Khung 8 Câu Hỏi Bắt Buộc",
            subtitle: "Bộ kiểm chứng hình thức 93/93 tests PASS và phát hiện khoa học khách quan về Blocker A & B",
            content: `
                <div class="overflow-x-auto">
                    <table class="w-full text-xs font-mono border-collapse border border-black">
                        <thead>
                            <tr class="bg-black text-white text-left">
                                <th class="p-2 border border-black w-28">Câu hỏi</th>
                                <th class="p-2 border border-black">Nội dung giải trình & Trạng thái tuần 3</th>
                                <th class="p-2 border border-black w-36">Minh chứng (Evidence)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-200 bg-white">
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">1. What</td>
                                <td class="p-2 border border-black">Bàn giao bộ Verifier IBP trên đồ thị thật, test suite 93 tests PASS, chứng minh H13 bitwise và 2 báo cáo toán học.</td>
                                <td class="p-2 border border-black text-zinc-600"><code>ecdr/week3_verifier.py</code></td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">2. When</td>
                                <td class="p-2 border border-black">Nghiệm thu đóng gói toàn bộ mã nguồn và báo cáo Tuần 3 ngày 03/11/2026. Khớp tiến độ WBS.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">ON SCHEDULE</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">3. Why</td>
                                <td class="p-2 border border-black">Khảo sát auto-LiRPA phát hiện không hỗ trợ trực tiếp MaxPool1D thời gian; bắt buộc phải lập trình custom verifier trực tiếp trên đồ thị PyTorch thật.</td>
                                <td class="p-2 border border-black text-zinc-600">Actual Graph Proof</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">4. Who</td>
                                <td class="p-2 border border-black">Phân công đề xuất: Math Lead (Định lý Margin); Systems Lead (Verifier IBP & stream replay); Benchmark Lead (Đo đạc độ trễ).</td>
                                <td class="p-2 border border-black text-zinc-600">Role Allocation</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">5. How much</td>
                                <td class="p-2 border border-black">Kế hoạch dự kiến 85h; thực tế tập trung cho việc kiểm tra đối soát 3.86 triệu điểm H13 và replay 20,212 truy vấn.</td>
                                <td class="p-2 border border-black text-zinc-700 font-bold">Projected Effort</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">6. Top risks</td>
                                <td class="p-2 border border-black">Nghịch lý bán kính (Blocker A): Cận IBP quá bảo thủ khiến bán kính an toàn rho=1e-6 quá nhỏ so với độ dịch chuyển tự nhiên.</td>
                                <td class="p-2 border border-black text-rose-700 font-bold">Blocker A Identified</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">7. Major changes</td>
                                <td class="p-2 border border-black">Xác nhận định lý margin trong số học thực; float32 numerical bridge chưa đạt nên giữ nhánh reuse ở trạng thái khóa fail-closed.</td>
                                <td class="p-2 border border-black text-zinc-600">Fail-Closed Security</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">8. Key issues</td>
                                <td class="p-2 border border-black">Độ trễ toàn trình ECDR là 3.11–3.20 ms vs baseline wrapper 1.59–1.68 ms (0.52x speedup, savings âm do verifier overhead > forward).</td>
                                <td class="p-2 border border-black text-rose-700 font-bold">Blocker B (Target W4)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `,
            takeaway: "Khoa học trung thực và khách quan: Tuần 3 hoàn thành trọn vẹn phần mềm và toán học, đồng thời chỉ rõ hai rào cản gốc rễ cần giải quyết ở Tuần 4."
        },
        {
            id: 2,
            badge: "WEEK 3 SPECIAL • PREPROCESSING PIPELINE",
            title: "Pipeline Tiền Xử Lý Dữ Liệu Hoàn Chỉnh (End-to-End Pipeline)",
            subtitle: "Quy trình 4 giai đoạn khép kín từ cảm biến thô đến tensor đầu vào cho mô hình",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-3">
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">1. Cleaning & Đồng Bộ Tín Hiệu</span>
                            Đồng bộ thời gian 50Hz giữa 3 trục Accel và 3 trục Gyro. Toàn bộ 17,453 cửa sổ được bảo tồn nguyên vẹn cho full-stream replay.
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">2. Chuẩn Hóa Z-Score Train-Only</span>
                            Áp dụng công thức: X_norm = (X - Mean_train) / Std_train cho từng kênh độc lập. Ngăn chặn 100% rò rỉ phân phối sang tập kiểm thử.
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">3. Cắt Cửa Sổ Trượt (Sliding Window)</span>
                            Trích xuất các đoạn 128 mẫu thời gian (2.56s) với bước nhảy 64 mẫu (1.28s, 50% overlap). Tạo tensor ma trận đầu vào [6, 128].
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">4. Lọc Dữ Liệu Có Giám Sát (Supervised)</span>
                            Áp dụng Majority Vote >= 60% lọc 12,278 cửa sổ supervised (10,361 pure + 1,917 majority) chia theo 17/4/9 subjects.
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/slide01_overview_pipeline.png" alt="End-to-End Preprocessing Pipeline" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Sơ đồ luồng tiền xử lý từ dữ liệu thô đến tensor sẵn sàng huấn luyện</span>
                    </div>
                </div>
            `,
            takeaway: "Pipeline tiền xử lý chuẩn mực bảo đảm tính toàn vẹn của tín hiệu và loại bỏ triệt để các rủi ro thiên lệch dữ liệu."
        },
        {
            id: 3,
            badge: "WEEK 3 SPECIAL • MODEL JUSTIFICATION",
            title: "Luận Cứ Lựa Chọn Mô Hình: Tại Sao Chọn 1D-CNN?",
            subtitle: "So sánh định tính về đặc tính kiến trúc và khả năng chứng nhận toán học hình thức",
            content: `
                <div class="overflow-x-auto">
                    <table class="w-full text-xs font-mono border-collapse border border-black">
                        <thead>
                            <tr class="bg-black text-white text-left">
                                <th class="p-2 border border-black">Tiêu chí kiến trúc</th>
                                <th class="p-2 border border-black bg-zinc-800 text-amber-300">SmallCNN1D (Lựa chọn)</th>
                                <th class="p-2 border border-black">LSTM / RNN</th>
                                <th class="p-2 border border-black">Transformer / Attention</th>
                                <th class="p-2 border border-black">Random Forest</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-200 bg-white">
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Cấu trúc đồ thị tính toán</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">Tĩnh, hữu hạn (Feed-forward)</td>
                                <td class="p-2 border border-black text-zinc-600">Tuần tự, phụ thuộc bước thời gian</td>
                                <td class="p-2 border border-black text-zinc-600">Đa đầu chú ý, tương quan bậc hai</td>
                                <td class="p-2 border border-black text-zinc-600">Rời rạc, tập hợp cây nhị phân</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Chi phí Conv/Linear MACs</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">1,172,992 MACs (Đã kiểm toán)</td>
                                <td class="p-2 border border-black text-zinc-500">Chưa đo kiểm toán đối chứng</td>
                                <td class="p-2 border border-black text-zinc-500">Chưa đo kiểm toán đối chứng</td>
                                <td class="p-2 border border-black text-zinc-500">Không áp dụng phép tính MACs</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Dung lượng tham số RAM</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">1.14 MB (283,692 tham số FP32)</td>
                                <td class="p-2 border border-black text-zinc-500">Phụ thuộc hidden size</td>
                                <td class="p-2 border border-black text-zinc-500">Thường đòi hỏi tham số lớn</td>
                                <td class="p-2 border border-black text-zinc-500">Tùy thuộc độ sâu và số cây</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Khả năng chứng nhận toán (Formal Verification)</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">Đóng khung được cận IBP/Lipschitz dạng đóng</td>
                                <td class="p-2 border border-black text-rose-700">Bùng nổ độ lỏng qua các vòng lặp ẩn</td>
                                <td class="p-2 border border-black text-rose-700">Toán tử Softmax đa chiều phi tuyến rất khó đóng cận</td>
                                <td class="p-2 border border-black text-amber-700">Không khả vi liên tục; không có logit margin</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="mt-4 p-3 border-l-4 border-black bg-zinc-50 rounded text-xs font-serif text-zinc-800">
                    <strong>Ghi chú phương pháp:</strong> So sánh mang tính định tính dựa trên lý thuyết kiến trúc và cấu trúc đồ thị tính toán; nhóm không tự bịa các số liệu benchmark thời gian phần cứng chưa đo kiểm chứng cho LSTM, Transformer hay Random Forest. SmallCNN1D được chọn vì đồ thị tĩnh hỗ trợ tốt nhất cho bài toán đóng khung chứng nhận toán học.
                </div>
            `,
            takeaway: "1D-CNN được chọn nhờ sự cân bằng tối ưu giữa cấu trúc đồ thị tĩnh dễ đóng khung chứng nhận và độ phức tạp vừa vặn cho thiết bị biên."
        },
        {
            id: 4,
            badge: "WEEK 3 SPECIAL • BASELINE METRICS",
            title: "Hiệu Năng Mô Hình Baseline Trên Tập Kiểm Thử",
            subtitle: "Đánh giá chi tiết trên 3,572 cửa sổ kiểm thử độc lập hoàn toàn (9 người dùng)",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="grid grid-cols-2 gap-3 text-left font-mono">
                            <div class="p-3 border-2 border-black rounded bg-zinc-50">
                                <span class="text-[11px] text-zinc-500 uppercase block font-bold">Accuracy Tham Chiếu</span>
                                <span class="text-3xl font-bold text-black block tracking-tight">89.05%</span>
                                <span class="text-[10px] text-zinc-600">3,181 / 3,572 cửa sổ đúng</span>
                            </div>
                            <div class="p-3 border-2 border-black rounded bg-zinc-50">
                                <span class="text-[11px] text-zinc-500 uppercase block font-bold">Macro-F1 Tham Chiếu</span>
                                <span class="text-3xl font-bold text-black block tracking-tight">77.63%</span>
                                <span class="text-[10px] text-zinc-600">Đánh giá công bằng 12 lớp</span>
                            </div>
                        </div>

                        <div class="space-y-2 font-serif text-xs text-zinc-800">
                            <div class="p-3 border border-emerald-700 bg-emerald-50 rounded">
                                <span class="font-mono font-bold text-emerald-900 block text-[11px]">Nhóm 6 Hoạt Động Cơ Bản (Macro-F1 Trung Bình ~91.4%):</span>
                                Walk: 95.85%, Walk_Up: 92.08%, Walk_Down: 95.28%, Sitting: 81.96%, Standing: 84.29%, Laying: 98.90%.
                            </div>
                            <div class="p-3 border border-amber-700 bg-amber-50 rounded">
                                <span class="font-mono font-bold text-amber-900 block text-[11px]">Nhóm 6 Hoạt Động Chuyển Thế (F1 Dao Động 55.0% &ndash; 73.6%):</span>
                                Stand-to-Sit: 73.56%, Sit-to-Stand: 71.19%, Sit-to-Lie: 61.68%, Lie-to-Sit: 66.67%, Stand-to-Lie: 54.96%, Lie-to-Stand: 55.10%.
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/model_confusion_matrix.png" alt="Confusion Matrix Heatmap" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Ma trận nhầm lẫn 12×12: Nhầm lẫn lớn giữa Sitting↔Standing (57 & 106 lần)</span>
                    </div>
                </div>
            `,
            takeaway: "Số liệu per-class được đồng bộ chính xác với model audit, phản ánh trung thực điểm nghẽn ở các lớp chuyển thế."
        },
        {
            id: 5,
            badge: "WEEK 3 • MATHEMATICAL FORMULATION",
            title: "Cơ Sở Toán Học Của Tái Sử Dụng Quyết Định Có Chứng Nhận",
            subtitle: "Định lý Margin Theorem (Số học thực) & Chẩn đoán Cận IBP",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-zinc-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">ĐỊNH LÝ LOGIT MARGIN (SỐ HỌC THỰC)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Cho mô hình tham chiếu F(X), mỏ neo X_anchor có dự đoán c = argmax F(X_anchor). Quyết định c được bảo toàn trên miền S nếu:
                            </p>
                            <div class="text-center font-mono text-xs bg-black text-white p-2 rounded">
                                min_{j != c} inf_{X in S} [ F_c(X) - F_j(X) ] > 0
                            </div>
                            <p class="text-[11px] font-serif text-zinc-600">
                                Đây là điều kiện đủ bảo đảm lớp c duy trì argmax độc tôn trong số học thực. Cần thêm chứng minh bao hàm số học (enclosure) để áp dụng cho backend float32.
                            </p>
                        </div>

                        <div class="p-4 border border-black bg-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">KỸ THUẬT IBP TRỰC TIẾP TẦNG CUỐI (H08)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Thay vì tính chặn trên và chặn dưới riêng lẻ của từng logit rồi trừ nhau, ta tính trực tiếp ma trận trọng số hiệu:
                            </p>
                            <div class="text-center font-mono text-xs bg-zinc-100 p-2 rounded border border-zinc-300 text-zinc-900">
                                W_{cj} = W_c - W_j  và  b_{cj} = b_c - b_j
                            </div>
                            <p class="text-[11px] font-serif text-zinc-600">
                                Kỹ thuật này triệt tiêu sự lỏng lẻo ở tầng Linear cuối cùng dưới cùng một khoảng ẩn trung gian.
                            </p>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig01_dual_track.png" alt="Dual Track Architecture" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Cơ chế Dual-Track: Nhánh Suy Luận Tham Chiếu (trên) song song Cổng Chứng Nhận (dưới)</span>
                    </div>
                </div>
            `,
            takeaway: "Nền tảng toán học được chứng minh trong số học thực, đồng thời chỉ rõ nghĩa vụ chứng minh số học float32 còn đang mở."
        },
        {
            id: 6,
            badge: "WEEK 3 • EXPERIMENTAL EVIDENCE",
            title: "Chứng Minh Căn Chỉnh Đặc Trưng Bitwise (H13)",
            subtitle: "3,866,112 điểm kiểm tra khớp chính xác 100% bitwise trên 12 vị trí nội bộ sau Conv3",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-emerald-700 bg-emerald-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-emerald-900 block">KẾT QUẢ ĐỐI SOÁT BITWISE TRÊN VÙNG NỘI BỘ</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Do tính chất bất biến tịnh tiến của phép tích chập 1D, 12 vị trí nội bộ của feature map sau Conv3 (j=2..13) qua 64 kênh khớp hoàn toàn giữa 2 cửa sổ liền kề:
                            </p>
                            <div class="grid grid-cols-2 gap-2 text-center font-mono text-xs pt-1">
                                <div class="p-2 border border-emerald-700 rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Số điểm kiểm tra</span>
                                    <span class="font-bold text-emerald-900 text-sm">3,866,112</span>
                                    <span class="text-[9px] text-zinc-500">5,034 cặp × 12 vị trí × 64 kênh</span>
                                </div>
                                <div class="p-2 border border-emerald-700 rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Sai số tuyệt đối (MAE)</span>
                                    <span class="font-bold text-emerald-900 text-sm">0.000000</span>
                                    <span class="text-[9px] text-emerald-700">Bitwise Exact trên CPU</span>
                                </div>
                            </div>
                        </div>

                        <div class="p-3 border border-black bg-zinc-50 rounded text-xs font-serif text-zinc-800">
                            <strong>Ranh giới kết luận:</strong> Các vị trí sát biên khác biệt do ảnh hưởng của padding và receptive field. Kết quả này chứng minh tiềm năng tái sử dụng đặc trưng tăng dần (incremental feature reuse), chưa bảo đảm bảo toàn quyết định argmax đầu ra.
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig03_h13_feature_alignment.png" alt="H13 Feature Alignment Proof" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Biểu đồ kiểm chứng H13: Sai số tuyệt đối bằng 0.0 trên 12 vị trí nội bộ</span>
                    </div>
                </div>
            `,
            takeaway: "H13 được chứng minh chính xác trên 12 vị trí nội bộ sau Conv3, đặt nền tảng cho việc nghiên cứu tái sử dụng đặc trưng một phần."
        },
        {
            id: 7,
            badge: "WEEK 3 SPECIAL • ROOT CAUSE ANALYSIS",
            title: "Nghiệm Thu Thực Nghiệm: Nghịch Lý Bán Kính (Blocker A)",
            subtitle: "99.6% mỏ neo có cận chẩn đoán dương nhưng 0 cửa sổ kế tiếp rơi vào miền tái sử dụng",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-rose-700 bg-rose-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-rose-900 block">NGHỊCH LÝ BÁN KÍNH (THE RADIUS DILEMMA)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Trên 5,053 cửa sổ kiểm thử, tại bán kính thẩm định rho = 10^-6, bộ verifier dựng được ước lượng chẩn đoán dương cho <strong>5,034 / 5,053 mỏ neo (99.6%)</strong>. Tuy nhiên, khi kiểm tra các cửa sổ kế tiếp trong luồng thực tế:
                            </p>
                            <div class="p-2 bg-white border border-rose-300 rounded font-mono text-center text-rose-800 font-bold text-sm">
                                Tỷ lệ thành viên kế tiếp: 0 / 5,053 (Đúng 0.0%)!
                            </div>
                        </div>

                        <div class="p-3 border border-black bg-zinc-50 rounded text-xs font-serif text-zinc-800 space-y-1">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">Nguyên nhân quan sát được:</span>
                            Khoảng cách biến thiên tự nhiên giữa 2 cửa sổ cảm biến kế tiếp là $\|\Delta\|_\infty \approx 0.1 - 1.0$. Bán kính chứng nhận được rho = 10^-6 nhỏ hơn thực tế tới 100,000 lần do sự bảo thủ tích lũy của phép lan truyền cận khoảng (IBP looseness).
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig04_radius_dilemma.png" alt="The Radius Dilemma" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Trực quan hóa Nghịch lý Bán kính: Miền chứng nhận S quá nhỏ so với quỹ đạo chuyển động</span>
                    </div>
                </div>
            `,
            takeaway: "Khoa học trung thực: Bán kính chẩn đoán IBP rho=1e-6 quá hẹp so với độ biến thiên thực tế, dẫn đến tỷ lệ reuse đạt 0.0%."
        },
        {
            id: 8,
            badge: "WEEK 3 SPECIAL • LATENCY BENCHMARK",
            title: "Đo Đạc Độ Trễ Toàn Trình: Lợi Ích Âm Do Chi Phí Verifier (Blocker B)",
            subtitle: "Phân định 3 ranh giới độ trễ và nguyên nhân tiết kiệm âm trên CPU",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="grid grid-cols-3 gap-2 text-left font-mono">
                            <div class="p-2 border border-black rounded bg-zinc-50">
                                <span class="text-[9px] text-zinc-500 uppercase block font-bold">1. CNN Core Forward</span>
                                <span class="text-base font-bold text-black block">0.19–0.22 ms</span>
                                <span class="text-[9px] text-zinc-500">Mô hình FP32 thuần</span>
                            </div>
                            <div class="p-2 border border-black rounded bg-zinc-50">
                                <span class="text-[9px] text-zinc-500 uppercase block font-bold">2. Baseline Wrapper</span>
                                <span class="text-base font-bold text-black block">1.59–1.68 ms</span>
                                <span class="text-[9px] text-zinc-500">Kèm rehash mã băm</span>
                            </div>
                            <div class="p-2 border-2 border-black rounded bg-rose-50">
                                <span class="text-[9px] text-rose-800 uppercase block font-bold">3. ECDR Replay</span>
                                <span class="text-base font-bold text-rose-900 block">3.11–3.20 ms</span>
                                <span class="text-[9px] text-rose-700">Verifier + Fallback</span>
                            </div>
                        </div>

                        <div class="p-4 border-2 border-black bg-zinc-100 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">TỐC ĐỘ ĐO ĐẠC: 0.52× SPEEDUP (TIẾT KIỆM ÂM SO VỚI WRAPPER)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Do tỷ lệ reuse bằng 0.0%, 100% các truy vấn đều phải kích hoạt Fallback. Hệ thống vừa tốn chi phí kiểm tra thành viên, vừa forward lại mô hình tham chiếu, khiến tổng thời gian xử lý chậm hơn baseline wrapper khoảng 1.9 lần.
                            </p>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig05_latency_comparison.png" alt="Latency Comparison" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">So sánh độ trễ: Baseline wrapper vs Luồng ECDR mang theo chi phí kiểm chứng</span>
                    </div>
                </div>
            `,
            takeaway: "Phân định rõ ranh giới đo lường: 0.52x là so sánh đối chuẩn với baseline wrapper, minh chứng cho hiện tượng bất đối xứng chi phí."
        },
        {
            id: 9,
            badge: "WEEK 3 SPECIAL • ERROR ANALYSIS",
            title: "Phân Tích Lỗi & Điểm Nghẽn Lớp Chuyển Thế (Postural Transitions)",
            subtitle: "Bóc tách nguyên nhân gây nhầm lẫn trên ma trận và lộ trình cải thiện",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border border-black bg-zinc-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">1. Nhầm Lẫn Thực Tế Quan Sát Được</span>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                                <li><strong>Sitting ↔ Standing:</strong> Xảy ra 57 lần Sitting bị đoán thành Standing, và 106 lần Standing bị đoán thành Sitting (do vị trí thắt lưng gần như thẳng đứng ở cả 2 tư thế).</li>
                                <li><strong>Stand-to-Lie → Sit-to-Lie:</strong> Xảy ra 20 lần nhầm lẫn do pha cuối của hai hành động nằm xuống đều có tín hiệu cảm biến hội tụ về trạng thái nằm.</li>
                            </ul>
                        </div>

                        <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">2. Hướng Đi Đang Khảo Sát Cho Tuần 4</span>
                            <div class="text-xs font-serif text-zinc-800 space-y-1">
                                <div>• <strong>Early Event Trigger O(1):</strong> Khảo sát bộ lọc chênh lệch năng lượng gia tốc để phát hiện chuyển thế sớm, kích hoạt fallback ngay mà không tốn chi phí kiểm chứng.</div>
                                <div>• <strong>Incremental Conv Reuse:</strong> Thử nghiệm tái sử dụng 12 vị trí đặc trưng nội bộ sau Conv3 (theo H13) để giảm tải tính toán khi fallback.</div>
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/per_class_performance.png" alt="Per-class Performance Breakdown" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Hiệu năng phân loại từng lớp: Ranh giới rõ rệt giữa lớp cơ bản và lớp chuyển thế</span>
                    </div>
                </div>
            `,
            takeaway: "Phân tích lỗi dựa trên các ca nhầm lẫn thực tế quan sát được trên ma trận, định hướng giải pháp kỹ thuật có cơ sở khoa học."
        },
        {
            id: 10,
            badge: "WEEK 3 SPECIAL • WEEK 4 TASKS",
            title: "Kế Hoạch Mô Hình Hóa Tuần 4 (Planned Tasks for Week 4)",
            subtitle: "3 Hướng thử nghiệm có điều kiện để giải quyết Blocker A & Blocker B",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                        <div class="bg-black text-white px-2 py-1 font-mono font-bold text-xs text-center rounded">THỬ NGHIỆM 1 (P0)</div>
                        <h4 class="font-serif font-bold text-black text-sm">Adaptive Dynamic Bounds</h4>
                        <p class="text-xs font-serif text-zinc-700 leading-relaxed">
                            Khảo sát thiết kế miền elip / affine bám theo các hướng chính của PCA (giải thích 21.5% phương sai) và gradient biên quyết định, kiểm tra liệu có thể mở rộng bán kính bao phủ mà vẫn giữ cận dương.
                        </p>
                    </div>

                    <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                        <div class="bg-black text-white px-2 py-1 font-mono font-bold text-xs text-center rounded">THỬ NGHIỆM 2 (P0)</div>
                        <h4 class="font-serif font-bold text-black text-sm">Early Event Trigger Filter</h4>
                        <p class="text-xs font-serif text-zinc-700 leading-relaxed">
                            Xây dựng bộ lọc kích hoạt sự kiện kiểm tra độ dao động năng lượng cảm biến để chuyển hướng fallback nhanh, loại bỏ chi phí chạy verifier không cần thiết khi tín hiệu biến động mạnh.
                        </p>
                    </div>

                    <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                        <div class="bg-black text-white px-2 py-1 font-mono font-bold text-xs text-center rounded">THỬ NGHIỆM 3 (P1)</div>
                        <h4 class="font-serif font-bold text-black text-sm">Incremental Feature Reuse</h4>
                        <p class="text-xs font-serif text-zinc-700 leading-relaxed">
                            Dựa trên chứng minh bitwise H13 trên 12 vị trí nội bộ sau Conv3, thiết kế cơ chế lưu cache đặc trưng để giảm số phép tính Conv1D khi fallback phải chạy lại mạng.
                        </p>
                    </div>
                </div>

                <div class="mt-4 p-4 border border-black bg-zinc-50 rounded flex items-center justify-between font-mono text-xs">
                    <span><strong>Mục tiêu Tuần 4:</strong> Kiểm chứng các giả thuyết cải tiến bằng số liệu thực nghiệm đo đạc thật; sẵn sàng chấp nhận kết luận âm nếu các rào cản vẫn tồn tại.</span>
                    <span class="bg-zinc-800 text-white px-2 py-0.5 rounded font-bold">WEEK 4 HYPOTHESES</span>
                </div>
            `,
            takeaway: "Lộ trình Tuần 4 được thiết lập dưới dạng các thử nghiệm khoa học có điều kiện và có tiêu chí bác bỏ rõ ràng."
        }
    ]
};
'''

# Replace FULL_DECKS block in HTML
start_marker = "const FULL_DECKS = {"
end_marker = "};\n\n// State variables"

idx_start = html.find(start_marker)
idx_end = html.find(end_marker)

if idx_start != -1 and idx_end != -1:
    html = html[:idx_start] + new_full_decks_js.strip() + html[idx_end + 1:]
    print("Replaced FULL_DECKS successfully via string slicing.")
else:
    print(f"WARNING: markers not found: start={idx_start}, end={idx_end}")

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved updated index.html with canonical audit fixes.")
