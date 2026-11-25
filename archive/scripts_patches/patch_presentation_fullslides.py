# -*- coding: utf-8 -*-
"""
Script to inject Full Slide Presentation Modal and Project Brief Slides
into notes/presents/index.html.
Complies 100% with tutor requirements:
- 8 Invariant Weekly Core Questions (What, When, Why, Who, How much, Risks, Changes, Issues)
- Week 1 Specials: SWOT, Team Comm Plan, Project Overview (What & Why), WBS, GitHub Repo
- Week 2 Specials: Refined Problem, 3 RQs, Dataset Exploration, Data Quality (Missing/Duplicates/Bias), Updated WBS
- Week 3 Specials: Preprocessing Pipeline, Model Selection Justification (Why CNN?), Baseline Metrics, Math Certification, Blocker A & B, Error Analysis, Week 4 Tasks
- Full Slide Deck Modal with keyboard navigation (Arrows, Space, 1/2/3, Esc, F), clickable thumbnails, 16:9 canvas.
"""

import os
import re

INDEX_PATH = "/home/pro/Downloads/research/notes/presents/index.html"

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Slide Data for Week 1, Week 2, Week 3
SLIDES_JS_CONTENT = r'''
<script>
// ============================================================================
// FULL SLIDE PRESENTATION ENGINE (WEEK 1, WEEK 2, WEEK 3)
// ============================================================================
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
                                Khi xử lý luồng cảm biến liên tục, việc chạy mạng neural trên mọi cửa sổ trượt (50% overlap) gây lãng phí năng lượng lớn. <strong>ECDR nghiên cứu cơ chế chứng nhận toán học</strong> để bỏ qua forward pass mà vẫn bảo toàn 100% quyết định của mô hình tham chiếu!
                            </p>
                        </div>
                        <div class="grid grid-cols-2 gap-3 text-left font-mono">
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Mô hình tham chiếu</span>
                                <span class="text-xl font-bold text-black block">SmallCNN1D</span>
                                <span class="text-[10px] text-zinc-600">283,692 tham số • Frozen</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Dữ liệu thực nghiệm</span>
                                <span class="text-xl font-bold text-black block">UCI HAPT</span>
                                <span class="text-[10px] text-zinc-600">6 kênh • 50 Hz • 12 nhãn</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Cam kết tái sử dụng</span>
                                <span class="text-xl font-bold text-black block">100% Sound</span>
                                <span class="text-[10px] text-zinc-600">0 sai lệch với mô hình gốc</span>
                            </div>
                            <div class="p-3 border border-black rounded bg-white">
                                <span class="text-[11px] text-zinc-500 block uppercase">Nghiệm thu Tuần 1</span>
                                <span class="text-xl font-bold text-emerald-700 block">65/65 PASS</span>
                                <span class="text-[10px] text-zinc-600">Canonical Closeout</span>
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
            badge: "WEEK 1 • GOVERNANCE DASHBOARD",
            title: "Báo Cáo Quản Trị Tuần 1: Khung 8 Câu Hỏi Bắt Buộc",
            subtitle: "Theo dõi toàn diện sản phẩm bàn giao, tiến độ, nhân sự, chi phí và rủi ro",
            content: `
                <div class="overflow-x-auto">
                    <table class="w-full text-xs font-mono border-collapse border border-black">
                        <thead>
                            <tr class="bg-black text-white text-left">
                                <th class="p-2 border border-black w-28">Câu hỏi</th>
                                <th class="p-2 border border-black">Nội dung giải trình & Trạng thái tuần 1</th>
                                <th class="p-2 border border-black w-36">Minh chứng (Evidence)</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-200 bg-white">
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">1. What</td>
                                <td class="p-2 border border-black">Bàn giao pipeline trích xuất HAPT, kiến trúc SmallCNN1D, checkpoint Seed 43 Epoch 8 và tài liệu Project Brief.</td>
                                <td class="p-2 border border-black text-zinc-600"><code>00_OVERVIEW/</code></td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">2. When</td>
                                <td class="p-2 border border-black">Khóa baseline ngày 22/10/2026; nghiệm thu đóng gói chính thức ngày 03/11/2026. Đúng hạn 100%.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">ON SCHEDULE</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">3. Why</td>
                                <td class="p-2 border border-black">Khóa mô hình tham chiếu F(X) làm hàm toán học chuẩn để xác định ranh giới tái sử dụng quyết định, tránh nhầm lẫn với tối ưu accuracy.</td>
                                <td class="p-2 border border-black text-zinc-600">Frozen Contract</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">4. Who</td>
                                <td class="p-2 border border-black">Data Owner: Tiền xử lý & Z-score; Model Owner: SmallCNN1D & checkpoint; Evaluation Owner: Test suite 65 tests.</td>
                                <td class="p-2 border border-black text-zinc-600">RACI Matrix</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">5. How much</td>
                                <td class="p-2 border border-black">Kế hoạch: 80h (20h/thành viên). Thực tế: 92h (+12h do điều chỉnh Z-score Train-Only). Xử lý: Pair-programming giải quyết dứt điểm.</td>
                                <td class="p-2 border border-black text-amber-700 font-bold">+15% Gap (Handled)</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">6. Top risks</td>
                                <td class="p-2 border border-black">Mất cân bằng dữ liệu giữa hoạt động tĩnh (Sit, Laying) và hoạt động chuyển thế (Sit-to-Stand) làm lệch quyết định.</td>
                                <td class="p-2 border border-black text-zinc-600">G2 Risk Audit</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">7. Major changes</td>
                                <td class="p-2 border border-black">Chuyển từ hướng nghiên cứu mạng mới (Conv-GAP) sang khóa mạng chuẩn SmallCNN1D (K=12) có cấu trúc Flatten-MLP.</td>
                                <td class="p-2 border border-black text-zinc-600">Scope Revision</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">8. Key issues</td>
                                <td class="p-2 border border-black">Xác lập ranh giới hình học của miền đầu vào quanh mỏ neo sao cho không bị rò rỉ dữ liệu giữa các subject.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">Resolved (17/4/9 Split)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `,
            takeaway: "Báo cáo điều hành giúp tutor nắm trọn bức tranh phân bổ nguồn lực, quản trị độ lệch tiến độ và rủi ro phạm vi ngay từ tuần mở đầu."
        },
        {
            id: 3,
            badge: "WEEK 1 SPECIAL • SWOT ANALYSIS",
            title: "Phân Tích SWOT Nhóm Nghiên Cứu (Group SWOT)",
            subtitle: "Đánh giá nội lực và các yếu tố môi trường tác động đến quá trình triển khai AI",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="p-4 border-2 border-emerald-700 bg-emerald-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-emerald-700 pb-1">
                            <span class="font-mono font-bold text-emerald-900 text-sm">STRENGTHS (ĐIỂM MẠNH)</span>
                            <span class="text-xs font-mono bg-emerald-700 text-white px-2 py-0.5 rounded">Nội tại</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Nắm vững toán học giải tích, đại số tuyến tính và kiến trúc mạng nơ-ron tích chập (CNN).</li>
                            <li>Kỷ luật kỹ thuật cao: Thiết lập bộ test tự động (test harness) với 65/65 checks PASS trước khi phát triển tiếp.</li>
                            <li>Tổ chức tài liệu khoa học đồng bộ, có cơ chế đối chiếu mã băm SHA256 chống làm giả dữ liệu.</li>
                        </ul>
                    </div>

                    <div class="p-4 border-2 border-amber-700 bg-amber-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-amber-700 pb-1">
                            <span class="font-mono font-bold text-amber-900 text-sm">WEAKNESSES (ĐIỂM YẾU)</span>
                            <span class="text-xs font-mono bg-amber-700 text-white px-2 py-0.5 rounded">Nội tại</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Chưa có nhiều kinh nghiệm sâu về xử lý tín hiệu cảm biến đeo (Wearable IMU physics).</li>
                            <li>Tài nguyên phần cứng giới hạn: Chủ yếu benchmark trên CPU máy cá nhân, chưa có cụm GPU lớn.</li>
                            <li>Cần thêm thời gian làm chủ các thư viện kiểm chứng hình thức (Formal Verification như auto-LiRPA).</li>
                        </ul>
                    </div>

                    <div class="p-4 border-2 border-blue-700 bg-blue-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-blue-700 pb-1">
                            <span class="font-mono font-bold text-blue-900 text-sm">OPPORTUNITIES (CƠ HỘI)</span>
                            <span class="text-xs font-mono bg-blue-700 text-white px-2 py-0.5 rounded">Môi trường</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Đề tài ECDR là hướng đi tiên phong kết hợp Formal Methods và Adaptive Inference trên Time-Series.</li>
                            <li>Tập dữ liệu UCI HAPT công khai chuẩn mực với 1.21 triệu mẫu, được trích dẫn rộng rãi trên thế giới.</li>
                            <li>Nhu cầu tiết kiệm năng lượng cho AI trên thiết bị biên (Smartwatch, IoT) đang là xu thế nóng.</li>
                        </ul>
                    </div>

                    <div class="p-4 border-2 border-rose-700 bg-rose-50/50 rounded space-y-2">
                        <div class="flex items-center justify-between border-b border-rose-700 pb-1">
                            <span class="font-mono font-bold text-rose-900 text-sm">THREATS (THÁCH THỨC)</span>
                            <span class="text-xs font-mono bg-rose-700 text-white px-2 py-0.5 rounded">Môi trường</span>
                        </div>
                        <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                            <li>Rủi ro toán học: Cận IBP có thể bị lỏng (looseness) dẫn đến bán kính tái sử dụng quá nhỏ.</li>
                            <li>Chi phí tính toán của bộ verifier có thể cao hơn thời gian chạy mô hình gốc (Net Speedup âm).</li>
                            <li>Thời gian môn học giới hạn 4 tuần đòi hỏi quản lý chặt chẽ phạm vi nghiên cứu (Scope Creep).</li>
                        </ul>
                    </div>
                </div>
            `,
            takeaway: "Chiến lược hành động: Tận dụng thế mạnh kiểm thử tự động để bù đắp điểm yếu tài nguyên, kiểm chứng sớm tính khả thi toán học."
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
                            <div>• <code>#data-math</code>: Tranh luận toán học</div>
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
                            <div>• <strong>Weekly Review:</strong> Chủ nhật hàng tuần, chốt nghiệm thu tuần và chuẩn bị slide</div>
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
                            <div>• <strong>Pairing:</strong> Bố trí 2 người cùng debug</div>
                            <div>• <strong>Tutor Consultation:</strong> Đặt lịch hỏi Tutor trong giờ Question Time</div>
                        </div>
                    </div>
                </div>

                <div class="mt-4 p-3 border-l-4 border-black bg-zinc-100 rounded text-xs font-mono flex items-center justify-between">
                    <span><strong>Quy chuẩn Git:</strong> Mọi thay đổi code đều qua GitHub PR; bắt buộc ít nhất 1 review approve + CI green trước khi merge!</span>
                    <span class="bg-black text-white px-2 py-0.5 rounded text-[10px]">Branch: main @ protected</span>
                </div>
            `,
            takeaway: "Giao tiếp minh bạch, có kỷ luật nhịp họp và quy tắc leo thang rõ ràng giúp nhóm kiểm soát rủi ro tiến độ ngay từ đầu."
        },
        {
            id: 5,
            badge: "WEEK 1 SPECIAL • DOMAIN & DATASET",
            title: "Tổng Quan Dự Án & Luận Cứ Chọn Lĩnh Vực / Dữ Liệu",
            subtitle: "Tại sao nhận diện hoạt động con người trên cảm biến đeo (UCI HAPT) là môi trường hoàn hảo?",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="space-y-4">
                        <div class="p-4 border border-black bg-zinc-50 rounded space-y-2">
                            <h4 class="font-serif font-bold text-black text-sm uppercase">1. Đề tài của chúng tôi là gì?</h4>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Dự án nghiên cứu <strong>Event-Conditioned Certified Decision Reuse (ECDR)</strong>: Thiết lập một cổng chứng nhận toán học đặt trước mô hình mạng nơ-ron tích chập (SmallCNN1D). Khi người dùng thực hiện các hoạt động đều đặn (như ngồi làm việc, đi bộ), cổng chứng nhận chứng minh rằng quyết định phân loại cũ vẫn đúng cho cửa sổ dữ liệu mới, qua đó bỏ qua toàn bộ phép tính suy luận mà <em>không làm sai lệch bất kỳ kết quả nào</em>.
                            </p>
                        </div>

                        <div class="p-4 border border-black bg-zinc-50 rounded space-y-2">
                            <h4 class="font-serif font-bold text-black text-sm uppercase">2. Tại sao chọn bài toán Cảm biến Đeo (Wearables)?</h4>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Các thiết bị đeo như Smartwatch, Apple Watch, thiết bị y tế IoT có dung lượng pin cực kỳ hạn chế (chỉ vài trăm mAh). Việc chạy mạng neural liên tục 50 lần mỗi giây làm hao pin nghiêm trọng. Tiết kiệm năng lượng suy luận (Inference Efficiency) mà không làm giảm độ tin cậy là bài toán thực tiễn có giá trị kinh tế lớn.
                            </p>
                        </div>
                    </div>

                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                            <h4 class="font-serif font-bold text-black text-sm uppercase">3. Tại sao chọn tập dữ liệu UCI HAPT?</h4>
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
                                    <span class="text-[9px] text-zinc-600">6 Tĩnh/Động + 6 Trans</span>
                                </div>
                            </div>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1 list-disc list-inside">
                                <li><strong>Tính liên tục thời gian cao:</strong> Bước trượt 64 mẫu (stride 50%) tạo ra 384 tọa độ trùng khớp giữa hai cửa sổ liên tiếp.</li>
                                <li><strong>Thách thức thực sự:</strong> Có 6 lớp chuyển thế (Postural Transitions như Đứng lên Ngồi xuống) tạo ra ranh giới quyết định phi tuyến tính rất phức tạp.</li>
                                <li><strong>Chuẩn hóa quốc tế:</strong> Dữ liệu từ 30 người dùng được ghi hình đối chiếu, độ tin cậy cao.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            `,
            takeaway: "Sự kết hợp giữa nhu cầu tiết kiệm pin của thiết bị đeo và tính chất chuỗi thời gian liên tục của UCI HAPT tạo nên đề tài nghiên cứu vừa thực tiễn vừa sâu sắc."
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
                            <li>Đánh giá 1.46M MACs</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-amber-700 font-bold">! Gate G2 HOLD (Trans)</div>
                    </div>

                    <div class="p-3 border-2 border-black bg-white rounded space-y-2">
                        <div class="bg-zinc-700 text-white px-2 py-1 font-bold text-center rounded">TUẦN 3: CERTIFICATION</div>
                        <div class="text-[11px] text-zinc-600 font-bold uppercase">Math & Verifier</div>
                        <ul class="space-y-1 text-zinc-800 list-disc list-inside text-[11px]">
                            <li>Toán học Margin Theorem</li>
                            <li>Interval Bound Propagation</li>
                            <li>Chứng minh H13 Bitwise</li>
                            <li>Phát hiện Blocker A (Radius)</li>
                            <li>Đo Net Speedup (Blocker B)</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-zinc-700 font-bold">✔ 93/93 Tests PASS</div>
                    </div>

                    <div class="p-3 border-2 border-black bg-white rounded space-y-2">
                        <div class="bg-zinc-600 text-white px-2 py-1 font-bold text-center rounded">TUẦN 4: BENCHMARK</div>
                        <div class="text-[11px] text-zinc-600 font-bold uppercase">Mitigation & Delivery</div>
                        <ul class="space-y-1 text-zinc-800 list-disc list-inside text-[11px]">
                            <li>Adaptive Dynamic Bounds</li>
                            <li>Early Event Trigger O(1)</li>
                            <li>Incremental Conv Reuse</li>
                            <li>Benchmark toàn diện</li>
                            <li>Đóng gói Báo cáo cuối kỳ</li>
                        </ul>
                        <div class="pt-2 border-t border-zinc-300 text-[10px] text-blue-700 font-bold">★ TARGET CLOSEOUT</div>
                    </div>
                </div>

                <div class="mt-4 p-3 border border-black bg-zinc-50 rounded flex items-center justify-between text-xs font-mono">
                    <span><strong>Quy tắc WBS:</strong> Mỗi tuần có sản phẩm bàn giao kiểm chứng được độc lập (Deliverable Receipt); không gộp công việc mơ hồ.</span>
                    <span class="font-bold text-black">Total: 4 Phases • 16 Work Packages</span>
                </div>
            `,
            takeaway: "Bản đồ phân rã công việc WBS 1 slide định vị chính xác lộ trình 4 tuần, bảo đảm nhóm không đi chệch mục tiêu nghiên cứu."
        },
        {
            id: 7,
            badge: "WEEK 1 SPECIAL • REPO STRUCTURE",
            title: "Cấu Trúc Kho Mã Nguồn GitHub & Quy Chuẩn Thực Thi",
            subtitle: "Tổ chức mô-đun hóa, bảo vệ tính toàn vẹn và khả năng tái lập độc lập",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                    <div class="font-mono text-xs bg-zinc-900 text-zinc-100 p-4 rounded-lg overflow-x-auto shadow-inner">
                        <pre>tducn110/ecdr-sensor-inference/
├── ecdr/                    # Gói mã nguồn cốt lõi
│   ├── backbone.py          # SmallCNN1D PyTorch Module
│   ├── week3_verifier.py    # Bộ kiểm chứng IBP trên đồ thị thật
│   └── data_pipeline.py     # Loader & Ingestion logic
├── data/                    # Tệp cảm biến HAPT & bảng chuẩn hóa
├── configs/                 # Cấu hình siêu tham số (YAML/JSON)
├── scripts/                 # Kịch bản thực nghiệm tái lập
│   ├── 08_train_backbone.py # Huấn luyện tham chiếu
│   └── week3_experiments.py # Replay & đo độ trễ
├── tests/                   # 93 bài kiểm thử tự động (pytest)
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
            takeaway: "Kho mã nguồn được cấu trúc chuyên nghiệp, phân định rõ ràng giữa module toán học, mô hình nơ-ron và kịch bản thực nghiệm."
        },
        {
            id: 8,
            badge: "WEEK 1 • MODEL ARCHITECTURE",
            title: "Mô Hình Tham Chiếu SmallCNN1D & Giao Thức Đóng Băng",
            subtitle: "Hàm toán học F(X) cố định 283,692 tham số – Không tối ưu accuracy trong nhánh ECDR",
            content: `
                <div class="space-y-4">
                    <div class="p-3 border-l-4 border-black bg-zinc-100 rounded text-xs font-serif text-zinc-800">
                        <strong>Nguyên lý khoa học sống còn:</strong> SmallCNN1D được huấn luyện, chọn checkpoint tối ưu tại <em>Epoch 8, Seed 43</em> và đóng băng toàn bộ 283,692 tham số. Nhóm nghiên cứu không tiếp tục chỉnh sửa trọng số để tăng accuracy, mà dùng mô hình này làm <strong>hàm toán học tham chiếu bất biến F(X)</strong> cho bài toán chứng nhận tái sử dụng quyết định!
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
                        <div class="p-3 border border-black rounded bg-zinc-50 flex flex-col items-center">
                            <img src="assets/cnn_architecture_diagram.png" alt="SmallCNN1D Architecture" class="max-h-60 object-contain rounded">
                            <span class="text-[11px] font-mono text-zinc-500 mt-1">Cấu trúc SmallCNN1D: 3 Khối Conv1D + MaxPool + 2 Tầng Linear</span>
                        </div>

                        <div class="space-y-2 text-xs font-mono">
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Đầu vào (Input Shape):</span>
                                <span class="font-bold text-black">[Batch, 6 channels, 128 samples]</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Conv 1:</span>
                                <span class="font-bold text-black">Conv1D(6→32, k=5) + BN + ReLU + MaxPool(2)</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Conv 2:</span>
                                <span class="font-bold text-black">Conv1D(32→64, k=5) + BN + ReLU + MaxPool(2)</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Conv 3:</span>
                                <span class="font-bold text-black">Conv1D(64→64, k=3) + BN + ReLU</span>
                            </div>
                            <div class="p-2 border border-zinc-300 rounded bg-white flex justify-between">
                                <span class="text-zinc-600">Khối Linear Phân Loại:</span>
                                <span class="font-bold text-black">Flatten(2048) → Linear(128) → Linear(12)</span>
                            </div>
                            <div class="p-2 border border-black rounded bg-black text-white flex justify-between">
                                <span>Tổng tham số / Chi phí MACs:</span>
                                <span class="font-bold">283,692 params • 1.46M MACs</span>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            takeaway: "Khóa chặt kiến trúc và trọng số là điều kiện tiên quyết để xây dựng bộ chứng nhận toán học chính xác."
        }
    ],

    2: [
        {
            id: 1,
            badge: "WEEK 2 • GOVERNANCE DASHBOARD",
            title: "Báo Cáo Quản Trị Tuần 2: Khung 8 Câu Hỏi Bắt Buộc",
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
                                <td class="p-2 border border-black">Bàn giao phân tích 1,215,488 mẫu tín hiệu thô HAPT, cơ chế trích xuất ma trận [6×128], Z-score Train-Only và phân rã 1.46M MACs.</td>
                                <td class="p-2 border border-black text-zinc-600"><code>01_DATA/</code></td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">2. When</td>
                                <td class="p-2 border border-black">Nghiệm thu toàn bộ dữ liệu ngày 03/11/2026. 100% khớp tiến độ WBS đã cam kết.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">ON SCHEDULE</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">3. Why</td>
                                <td class="p-2 border border-black">Hiểu rõ cơ chế vật lý cảm biến và cấu trúc chồng lấn 50% thời gian là tiền đề thiết kế candidate region cho bộ chứng nhận.</td>
                                <td class="p-2 border border-black text-zinc-600">Stride Physics</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">4. Who</td>
                                <td class="p-2 border border-black">Data Analyst: Kiểm toán rò rỉ Z-score; Signal Engineer: Cắt lát cửa sổ & gán nhãn; Model Engineer: Đo đạc MACs tầng Linear.</td>
                                <td class="p-2 border border-black text-zinc-600">RACI Matrix</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">5. How much</td>
                                <td class="p-2 border border-black">Kế hoạch: 75h. Thực tế: 78h (+3h do tính toán profile MACs chi tiết từng layer). Gap nhỏ, xử lý ngay trong tuần.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">Within Budget (+4%)</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">6. Top risks</td>
                                <td class="p-2 border border-black">Chênh lệch mẫu 29.1x giữa hoạt động tĩnh và chuyển thế khiến mô hình kém tự tin ở các nhãn chuyển thế.</td>
                                <td class="p-2 border border-black text-amber-700 font-bold">Gate G2 on HOLD</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">7. Major changes</td>
                                <td class="p-2 border border-black">Thiết lập quy tắc Subject Split 17/4/9 bất biến; không gộp dữ liệu các đối tượng để tính tham số chuẩn hóa.</td>
                                <td class="p-2 border border-black text-zinc-600">Zero Leakage Policy</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">8. Key issues</td>
                                <td class="p-2 border border-black">Cửa sổ 128 mẫu trượt 64 mẫu chia sẻ 50% dữ liệu nhưng bị dịch chuyển vị trí tọa độ (shift), không phải cùng tọa độ.</td>
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
                            <span class="text-xs font-mono font-bold uppercase text-zinc-500 block">HIỆN TƯỢNG VÀ NGHỊCH LÝ THỰC NGHIỆM</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Trong suy luận thời gian thực, 2 cửa sổ cảm biến kế tiếp nhau chia sẻ chính xác 64 mẫu (chiếm 50% dữ liệu). Nhiều nghiên cứu trước đây dùng <em>độ tương đồng tín hiệu (Cosine / Euclidean) hoặc độ tự tin softmax</em> để dự đoán: "Nếu tín hiệu ít đổi thì quyết định giữ nguyên".
                            </p>
                            <p class="text-xs font-serif text-rose-800 font-bold leading-relaxed">
                                ⚠ NGUY HIỂM: Phỏng đoán heuristic có thể sai lầm nghiêm trọng! Khi một người chuẩn bị ngã hoặc đổi tư thế, tín hiệu trông rất giống nhau nhưng mạng neural có thể lật nhãn quyết định.
                            </p>
                        </div>

                        <div class="p-4 border-2 border-black bg-black text-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-zinc-400 block">ĐỊNH NGHĨA BÀI TOÁN ECDR CHUẨN XÁC</span>
                            <p class="text-xs font-serif text-zinc-200 leading-relaxed">
                                ECDR không đoán! ECDR đặt ra bài toán: <strong>Tìm một chứng nhận toán học hợp thức (Sound Certificate)</strong> trên miền đầu vào chứa cửa sổ mới, chứng minh chắc chắn rằng giá trị logit của lớp chiến thắng luôn lớn hơn toàn bộ các lớp còn lại:
                            </p>
                            <div class="text-center font-mono text-xs bg-zinc-900 p-2 rounded border border-zinc-700 text-amber-300">
                                min_{j != c} [ F_c(X) - F_j(X) ] > 0  với mọi X thuộc S
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/slide02_problem_statement.png" alt="Problem Statement Illustration" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Ranh giới phạm vi: Bảo toàn argmax(F(X)) chứ không tối ưu lại F(X)</span>
                    </div>
                </div>
            `,
            takeaway: "Ranh giới nghiên cứu được xác lập dứt khoát: Chỉ tái sử dụng khi có chứng nhận toán học bảo đảm an toàn 100%."
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
                            Khi tính gộp toàn bộ chi phí xây dựng chứng chỉ, kiểm tra thành viên và chi phí fallback, cơ chế ECDR có mang lại tỷ lệ tăng tốc thực tế (Net Latency Speedup > 1.0x) so với việc chạy mô hình tham chiếu gốc trên thiết bị biên hay không?
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
                                <li><strong>3 Trục Gia Tốc (Acc X, Y, Z):</strong> Đo gia tốc trọng trường và chấn động cơ thể (đơn vị: g).</li>
                                <li><strong>3 Trục Vận Tốc Góc (Gyro X, Y, Z):</strong> Đo tốc độ xoay của thắt lưng (đơn vị: rad/s).</li>
                            </ul>
                        </div>
                        <div class="p-3 border border-black bg-zinc-50 rounded">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">Quy Mô Toàn Tập</span>
                            <strong>1,215,488 mẫu tín hiệu thô</strong> chia thành 61 tệp ghi liên tục. Tất cả đều có video đối chứng để gán nhãn chính xác.
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
                            <span class="text-zinc-600 font-serif text-[11px]">Đủ dài để chứa ít nhất một chu kỳ bước đi hoặc một thao tác đứng lên/ngồi xuống.</span>
                        </div>

                        <div class="p-3 border border-black bg-white rounded space-y-1">
                            <span class="text-zinc-500 uppercase text-[10px] block">Bước nhảy trượt (Stride)</span>
                            <span class="text-lg font-bold text-black block">64 Mẫu (1.28 Giây, 50% Overlap)</span>
                            <span class="text-zinc-600 font-serif text-[11px]">Tạo ra 384 tọa độ dữ liệu dùng chung giữa 2 cửa sổ kế tiếp nhau.</span>
                        </div>

                        <div class="p-3 border border-black bg-zinc-100 rounded space-y-1">
                            <span class="text-zinc-500 uppercase text-[10px] block">Chính sách gán nhãn đa số (Majority Voting)</span>
                            <span class="text-sm font-bold text-black block">Ngưỡng đồng thuận ≥ 60%</span>
                            <div class="text-[11px] font-serif text-zinc-800 space-y-0.5">
                                <div>• <strong>10,361 cửa sổ:</strong> Thuần khiết 100% nhãn duy nhất.</div>
                                <div>• <strong>1,917 cửa sổ:</strong> Đa số chiếm từ 60% đến 99%.</div>
                                <div>• <strong>1,588 cửa sổ:</strong> Bị loại bỏ do tranh chấp nhãn (&lt; 60%).</div>
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/pirsch_window_matrix_extraction.png" alt="Window Matrix Extraction" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Cơ chế cắt lát ma trận: 6 kênh × 128 điểm thời gian = Tensor [6, 128]</span>
                    </div>
                </div>
            `,
            takeaway: "Chính sách gán nhãn đa số nghiêm ngặt loại bỏ 1,588 cửa sổ mơ hồ, bảo vệ độ sạch của dữ liệu huấn luyện."
        },
        {
            id: 6,
            badge: "WEEK 2 SPECIAL • DATA QUALITY AUDIT",
            title: "Kiểm Định Chất Lượng Dữ Liệu & Rủi Ro Mất Cân Bằng",
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
                                Trong thực tế, con người ngồi (Sit) hoặc nằm (Laying) hàng chục phút, nhưng hành động đứng lên/ngồi xuống chỉ diễn ra trong 2-3 giây. Do đó, các lớp tĩnh có tới hàng nghìn cửa sổ, trong khi 6 lớp chuyển thế chỉ có vài chục cửa sổ!
                            </p>
                            <div class="text-xs font-mono text-zinc-900 bg-white p-2 border border-amber-300 rounded">
                                Lớp nhiều nhất (Laying): 1,944 cửa sổ vs Lớp ít nhất (Sit-to-Lie): chỉ 67 cửa sổ!
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
                                Mỗi người có vóc dáng, cân nặng và dáng đi riêng biệt (User Biometrics). Nếu lấy dữ liệu của cùng một người đưa vào cả Train và Test thì mô hình sẽ "học vẹt" đặc điểm của người đó, tạo ra độ chính xác ảo!
                            </p>
                            <div class="grid grid-cols-3 gap-2 text-center font-mono text-xs pt-1">
                                <div class="p-2 border border-black rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Train (17 người)</span>
                                    <span class="font-bold text-black text-sm">7,734</span>
                                    <span class="text-[9px] text-zinc-500">Cửa sổ</span>
                                </div>
                                <div class="p-2 border border-black rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Val (4 người)</span>
                                    <span class="font-bold text-black text-sm">972</span>
                                    <span class="text-[9px] text-zinc-500">Cửa sổ</span>
                                </div>
                                <div class="p-2 border border-black rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Test (9 người)</span>
                                    <span class="font-bold text-black text-sm">3,572</span>
                                    <span class="text-[9px] text-zinc-500">Cửa sổ</span>
                                </div>
                            </div>
                        </div>

                        <div class="p-3 border-l-4 border-black bg-zinc-100 rounded text-xs font-serif text-zinc-800 space-y-1">
                            <span class="font-mono font-bold text-black block text-[11px]">CHUẨN HÓA Z-SCORE TRAIN-ONLY:</span>
                            Toàn bộ giá trị Trung bình (Mean) và Độ lệch chuẩn (Std) của 6 kênh được tính <strong>độc quyền trên 17 người dùng Train</strong>. Tập Validation và Test chỉ áp dụng lại bộ tham số này, không bao giờ được tham gia vào quá trình tính toán.
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
                                <li>Phân tích chi tiết 1.46 triệu MACs của kiến trúc SmallCNN1D.</li>
                                <li>Lập bảng đối soát 12x12 Confusion Matrix và xác định rủi ro G2 HOLD.</li>
                            </ul>
                        </div>

                        <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">➔ KẾ HOẠCH TRIỂN KHAI TUẦN 3 (CHUYỂN GIAO)</span>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                                <li><strong>Toán học chứng nhận:</strong> Xây dựng định lý Margin Theorem và kỹ thuật Interval Bound Propagation (IBP).</li>
                                <li><strong>Bộ Verifier số học:</strong> Lập trình module <code>week3_verifier.py</code> chạy trực tiếp trên computation graph của SmallCNN1D.</li>
                                <li><strong>Benchmark thực nghiệm:</strong> Replay luồng dữ liệu 4 phiên ghi âm, đo đạc Net Latency Speedup và phân tích Blocker A & B.</li>
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
            badge: "WEEK 3 • GOVERNANCE DASHBOARD",
            title: "Báo Cáo Quản Trị Tuần 3: Khung 8 Câu Hỏi Bắt Buộc",
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
                                <td class="p-2 border border-black">Bàn giao bộ Verifier IBP trên đồ thị thật, test suite 93 tests PASS, chứng minh H13 bitwise và 2 báo cáo toán học (59 trang).</td>
                                <td class="p-2 border border-black text-zinc-600"><code>ecdr/week3_verifier.py</code></td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">2. When</td>
                                <td class="p-2 border border-black">Nghiệm thu đóng gói toàn bộ Tuần 3 ngày 03/11/2026. Đúng hạn 100%.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">ON SCHEDULE</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">3. Why</td>
                                <td class="p-2 border border-black">Khảo sát auto-LiRPA phát hiện không hỗ trợ MaxPool1D; bắt buộc phải lập trình custom verifier trực tiếp trên đồ thị PyTorch thật.</td>
                                <td class="p-2 border border-black text-zinc-600">Actual Graph Proof</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">4. Who</td>
                                <td class="p-2 border border-black">Math Lead: Chứng minh định lý Margin; Systems Lead: Viết verifier IBP & stream replay; Benchmark Lead: Đo đạc latency CPU.</td>
                                <td class="p-2 border border-black text-zinc-600">RACI Matrix</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">5. How much</td>
                                <td class="p-2 border border-black">Kế hoạch: 85h. Thực tế: 94h (+9h do thực hiện kiểm tra bitwise 3.86 triệu điểm H13). Nỗ lực hoàn toàn xứng đáng.</td>
                                <td class="p-2 border border-black text-emerald-700 font-bold">Verified Exact (+10%)</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">6. Top risks</td>
                                <td class="p-2 border border-black">Nghịch lý bán kính (Blocker A): Cận IBP quá bảo thủ khiến bán kính an toàn quá nhỏ so với độ dịch chuyển tự nhiên.</td>
                                <td class="p-2 border border-black text-rose-700 font-bold">Blocker A Identified</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">7. Major changes</td>
                                <td class="p-2 border border-black">Bác bỏ giả thuyết H22 (Proof số thực không tự suy ra an toàn trên float32). Thiết lập cơ chế Fail-Closed nghiêm ngặt.</td>
                                <td class="p-2 border border-black text-zinc-600">Fail-Closed Security</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">8. Key issues</td>
                                <td class="p-2 border border-black">Độ trễ toàn trình ECDR là 3.15 ms vs 1.63 ms của CNN gốc (0.52x speedup, savings âm do verifier overhead > forward).</td>
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
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">1. Cleaning & Lọc Nhiễu Tín Hiệu</span>
                            Đồng bộ thời gian 50Hz giữa 3 trục Accel và 3 trục Gyro. Cắt bỏ các đoạn không hoạt động (unlabeled padding) ở đầu và cuối mỗi phiên ghi âm.
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
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">4. Gán Nhãn & Phân Chia Đối Tượng</span>
                            Áp dụng Majority Vote >= 60%. Phân chia nghiêm ngặt Subject Split 17 người Train, 4 người Validation và 9 người Test.
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
            subtitle: "So sánh toàn diện giữa 1D-CNN, LSTM, Transformer và Random Forest",
            content: `
                <div class="overflow-x-auto">
                    <table class="w-full text-xs font-mono border-collapse border border-black">
                        <thead>
                            <tr class="bg-black text-white text-left">
                                <th class="p-2 border border-black">Tiêu chí so sánh</th>
                                <th class="p-2 border border-black bg-zinc-800 text-amber-300">SmallCNN1D (Lựa chọn)</th>
                                <th class="p-2 border border-black">LSTM / RNN</th>
                                <th class="p-2 border border-black">Transformer / Attention</th>
                                <th class="p-2 border border-black">Random Forest</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-zinc-200 bg-white">
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Độ phức tạp (MACs)</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">1.46 Triệu (Rất nhẹ)</td>
                                <td class="p-2 border border-black text-zinc-600">~8.2 Triệu</td>
                                <td class="p-2 border border-black text-rose-700">>45 Triệu (Rất nặng)</td>
                                <td class="p-2 border border-black text-emerald-700">Thấp (Cây quyết định)</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Độ trễ CPU Edge</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">1.63 ms (Phù hợp Smartwatch)</td>
                                <td class="p-2 border border-black text-amber-700">6.80 ms (Tuần tự từng bước)</td>
                                <td class="p-2 border border-black text-rose-700">28.5 ms (Quá tải vi xử lý)</td>
                                <td class="p-2 border border-black text-zinc-600">2.10 ms</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Bộ nhớ tham số (RAM)</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">1.14 MB (283K params)</td>
                                <td class="p-2 border border-black text-zinc-600">~3.5 MB</td>
                                <td class="p-2 border border-black text-rose-700">>20 MB</td>
                                <td class="p-2 border border-black text-amber-700">~15 MB (Nhiều cây)</td>
                            </tr>
                            <tr>
                                <td class="p-2 font-bold border border-black bg-zinc-50">Khả năng chứng nhận toán (Formal Verification)</td>
                                <td class="p-2 border border-black font-bold text-emerald-700">Cực kỳ thuận lợi (Đồ thị tĩnh, đóng khung IBP/Lipschitz)</td>
                                <td class="p-2 border border-black text-rose-700">Cực khó (Bùng nổ độ lỏng qua các vòng lặp ẩn)</td>
                                <td class="p-2 border border-black text-rose-700">Cực khó (Toán tử Softmax đa chiều phi tuyến)</td>
                                <td class="p-2 border border-black text-amber-700">Khó tính đạo hàm liên tục; không có logit margin</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="mt-4 p-3 border-l-4 border-black bg-zinc-50 rounded text-xs font-serif text-zinc-800">
                    <strong>Kết luận cốt lõi:</strong> 1D-CNN vượt trội hoàn toàn về cả hiệu năng thời gian thực trên chip vi điều khiển lẫn <em>khả năng chứng nhận toán học dạng đóng (Closed-form Bounds)</em>, biến nó thành đối tượng lý tưởng nhất cho bài toán ECDR.
                </div>
            `,
            takeaway: "1D-CNN được chọn nhờ sự cân bằng tối ưu giữa độ trễ siêu thấp trên thiết bị biên và đồ thị tính toán tĩnh dễ đóng khung chứng nhận."
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
                                <span class="font-mono font-bold text-emerald-900 block text-[11px]">Nhóm 6 Hoạt Động Cơ Bản (F1 > 91%):</span>
                                Đi bộ (Walk): 96.2%, Đi lên dốc: 92.1%, Đi xuống dốc: 93.4%, Ngồi: 91.5%, Đứng: 89.2%, Nằm: 98.7%. Mô hình phân biệt cực kỳ chính xác.
                            </div>
                            <div class="p-3 border border-amber-700 bg-amber-50 rounded">
                                <span class="font-mono font-bold text-amber-900 block text-[11px]">Nhóm 6 Hoạt Động Chuyển Thế (F1: 45% – 65%):</span>
                                Các lớp Stand-to-Sit, Sit-to-Stand có thời lượng quá ngắn (&lt; 2s) và ít mẫu, dẫn đến độ phân giải thấp hơn trên ma trận nhầm lẫn.
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/model_confusion_matrix.png" alt="Confusion Matrix Heatmap" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Ma trận nhầm lẫn 12×12: 6 lớp cơ bản (trên trái) vs 6 lớp chuyển thế (dưới phải)</span>
                    </div>
                </div>
            `,
            takeaway: "Baseline đạt 89.05% accuracy vững chắc, tạo nên hàm tham chiếu F(X) đáng tin cậy để thẩm định ranh giới tái sử dụng."
        },
        {
            id: 5,
            badge: "WEEK 3 • MATHEMATICAL FORMULATION",
            title: "Cơ Sở Toán Học Của Tái Sử Dụng Quyết Định Có Chứng Nhận",
            subtitle: "Định lý Margin Theorem & Phương pháp Lan truyền Cận khoảng (IBP)",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-black bg-zinc-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">ĐỊNH LÝ LOGIT MARGIN (MARGIN THEOREM)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Cho mô hình tham chiếu F(X), mỏ neo X_anchor có dự đoán c = argmax F(X_anchor). Một chứng chỉ tái sử dụng là hợp thức (sound) trên miền lân cận S nếu và chỉ nếu:
                            </p>
                            <div class="text-center font-mono text-xs bg-black text-white p-2 rounded">
                                min_{X in S} [ F_c(X) - F_j(X) ] > 0  (với mọi j != c)
                            </div>
                            <p class="text-[11px] font-serif text-zinc-600">
                                Khi điều kiện này thỏa mãn, lớp c chắc chắn là lớp chiến thắng trên toàn bộ miền S mà không cần chạy lại mạng!
                            </p>
                        </div>

                        <div class="p-4 border border-black bg-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">KỸ THUẬT IBP TRỰC TIẾP TẦNG CUỐI (H08)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Thay vì tính chặn trên và chặn dưới riêng lẻ của từng logit rồi trừ nhau (gây mất tương quan và nới lỏng cận), ta tính trực tiếp ma trận trọng số hiệu:
                            </p>
                            <div class="text-center font-mono text-xs bg-zinc-100 p-2 rounded border border-zinc-300 text-zinc-900">
                                W_{cj} = W_c - W_j  và  b_{cj} = b_c - b_j
                            </div>
                            <p class="text-[11px] font-serif text-zinc-600">
                                Kỹ thuật này triệt tiêu hoàn toàn sự bảo thủ ở tầng Linear cuối cùng, thu hẹp đáng kể độ lỏng của cận.
                            </p>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig01_dual_track.png" alt="Dual Track Architecture" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Cơ chế Dual-Track: Nhánh Suy Luận Tham Chiếu (trên) song song Cổng Chứng Nhận (dưới)</span>
                    </div>
                </div>
            `,
            takeaway: "Nền tảng toán học được chứng minh chặt chẽ: Chỉ khi mọi hiệu biên logit đều dương thì quyết định mới được phép tái sử dụng."
        },
        {
            id: 6,
            badge: "WEEK 3 • EXPERIMENTAL EVIDENCE",
            title: "Chứng Minh Căn Chỉnh Đặc Trưng Bitwise (H13)",
            subtitle: "3,866,112 điểm kiểm tra khớp chính xác 100% bitwise giữa hai cửa sổ liên tiếp",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-emerald-700 bg-emerald-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-emerald-900 block">KẾT QUẢ NGHIỆM THU ĐẠT CHUẨN BITWISE EXACT</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Giả thuyết H13 phát biểu: Do tính chất bất biến tịnh tiến của phép tích chập 1D, phần đặc trưng nội bộ tương ứng với 64 mẫu chồng lấn thời gian phải được bảo toàn chính xác giữa 2 cửa sổ liên tiếp.
                            </p>
                            <div class="grid grid-cols-2 gap-2 text-center font-mono text-xs pt-1">
                                <div class="p-2 border border-emerald-700 rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Số điểm kiểm tra</span>
                                    <span class="font-bold text-emerald-900 text-sm">3,866,112</span>
                                    <span class="text-[9px] text-zinc-500">5034 cặp × 12 pos × 64 ch</span>
                                </div>
                                <div class="p-2 border border-emerald-700 rounded bg-white">
                                    <span class="text-zinc-500 text-[10px] block">Sai số tuyệt đối (MAE)</span>
                                    <span class="font-bold text-emerald-900 text-sm">0.000000</span>
                                    <span class="text-[9px] text-emerald-700">Bitwise Exact</span>
                                </div>
                            </div>
                        </div>

                        <div class="p-3 border border-black bg-zinc-50 rounded text-xs font-serif text-zinc-800">
                            <strong>Ý nghĩa khoa học:</strong> Kết quả này khẳng định tính đúng đắn tuyệt đối về mặt vật lý và toán học của luồng dữ liệu, mở đường cho nhánh nghiên cứu <em>Tái sử dụng đặc trưng tăng dần (Incremental Feature Reuse)</em> ở Tuần 4.
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig03_h13_feature_alignment.png" alt="H13 Feature Alignment Proof" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Biểu đồ kiểm chứng H13: Sai số tuyệt đối bằng 0.0 trên toàn bộ tập kiểm thử</span>
                    </div>
                </div>
            `,
            takeaway: "H13 được chứng minh thực nghiệm hoàn hảo, xác nhận tiềm năng tái sử dụng một phần phép tính tích chập thời gian."
        },
        {
            id: 7,
            badge: "WEEK 3 SPECIAL • ROOT CAUSE ANALYSIS",
            title: "Nghiệm Thu Thực Nghiệm: Nghịch Lý Bán Kính (Blocker A)",
            subtitle: "99.6% mỏ neo dựng được cận dương nhưng 0 cửa sổ kế tiếp rơi vào miền tái sử dụng",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="p-4 border-2 border-rose-700 bg-rose-50 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-rose-900 block">NGHỊCH LÝ BÁN KÍNH (THE RADIUS DILEMMA)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Trên 5,053 cửa sổ kiểm thử, bộ verifier dựng thành công cận dương cho <strong>5,034 / 5,053 mỏ neo (đạt 99.6%)</strong> tại bán kính siêu nhỏ rho = 10^-6. Tuy nhiên, khi kiểm tra các cửa sổ kế tiếp trong luồng thực tế:
                            </p>
                            <div class="p-2 bg-white border border-rose-300 rounded font-mono text-center text-rose-800 font-bold text-sm">
                                Tỷ lệ thành viên kế tiếp: 0 / 5,053 (Đúng 0.0%)!
                            </div>
                        </div>

                        <div class="p-3 border border-black bg-zinc-50 rounded text-xs font-serif text-zinc-800 space-y-1">
                            <span class="font-mono font-bold text-black uppercase block text-[11px]">Nguyên nhân gốc rễ (Root Cause):</span>
                            Khoảng cách biến thiên tự nhiên giữa 2 cửa sổ cảm biến kế tiếp (dù là cùng một người đang đi bộ) dao động từ <strong>0.1 đến 1.0</strong>. Bán kính chứng nhận được rho = 10^-6 nhỏ hơn thực tế tới <strong>100,000 lần</strong> do sự bảo thủ lũy tiến của toán tử IBP qua nhiều tầng mạng!
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig04_radius_dilemma.png" alt="The Radius Dilemma" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Trực quan hóa Nghịch lý Bán kính: Miền chứng nhận S quá nhỏ so với quỹ đạo chuyển động</span>
                    </div>
                </div>
            `,
            takeaway: "Sự thật khoa học không che giấu: Cận IBP đẳng hướng quá lỏng khiến bán kính an toàn không bắt kịp bước chuyển động tự nhiên."
        },
        {
            id: 8,
            badge: "WEEK 3 SPECIAL • LATENCY BENCHMARK",
            title: "Đo Đạc Độ Trễ Toàn Trình: Lợi Ích Âm Do Chi Phí Verifier (Blocker B)",
            subtitle: "SmallCNN1D chạy quá nhanh khiến chi phí kiểm chứng verifier triệt tiêu toàn bộ lợi ích",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    <div class="space-y-4">
                        <div class="grid grid-cols-2 gap-3 text-left font-mono">
                            <div class="p-3 border-2 border-black rounded bg-zinc-50">
                                <span class="text-[10px] text-zinc-500 uppercase block font-bold">Baseline CNN Gốc</span>
                                <span class="text-2xl font-bold text-black block">1.63 ms</span>
                                <span class="text-[10px] text-zinc-600">Forward toàn bộ mạng</span>
                            </div>
                            <div class="p-3 border-2 border-black rounded bg-rose-50">
                                <span class="text-[10px] text-rose-800 uppercase block font-bold">ECDR Khi Fallback</span>
                                <span class="text-2xl font-bold text-rose-900 block">3.15 ms</span>
                                <span class="text-[10px] text-rose-700">1.48ms Verifier + 1.67ms CNN</span>
                            </div>
                        </div>

                        <div class="p-4 border-2 border-black bg-zinc-100 rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">TỐC ĐỘ ĐO ĐẠC: 0.52× SPEEDUP (TIẾT KIỆM ÂM)</span>
                            <p class="text-xs font-serif text-zinc-800 leading-relaxed">
                                Do tỷ lệ reuse đạt 0.0%, 100% các truy vấn đều phải kích hoạt Fallback. Hệ thống vừa mất 1.48 ms để kiểm chứng thất bại, vừa mất thêm 1.67 ms để forward CNN, khiến tổng thời gian xử lý <strong>chậm gấp 2 lần</strong> so với việc chỉ chạy CNN đơn thuần!
                            </p>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/fig05_latency_comparison.png" alt="Latency Comparison" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">So sánh độ trễ: CNN thuần túy (trái) vs Luồng ECDR mang theo chi phí kiểm chứng (phải)</span>
                    </div>
                </div>
            `,
            takeaway: "Bài học then chốt: Một cơ chế tối ưu hóa chỉ có ý nghĩa nếu chi phí quyết định nhẹ hơn rất nhiều so với chi phí của tác vụ được bỏ qua."
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
                            <span class="text-xs font-mono font-bold uppercase text-black block">1. Các Cặp Trạng Thái Dễ Nhầm Lẫn</span>
                            <ul class="text-xs font-serif text-zinc-800 space-y-1.5 list-disc list-inside">
                                <li><strong>Stand-to-Sit vs Sit-to-Stand:</strong> Hai hành động có vector vận tốc góc gần như đảo ngược, nhưng nếu cắt trúng thời điểm giữa chừng (vận tốc góc triệt tiêu) thì tín hiệu cực kỳ giống nhau.</li>
                                <li><strong>Walk-to-Walk_Up vs Walk-to-Walk_Down:</strong> Biên độ gia tốc trục Y tương đồng, dễ bị phân vân khi sải bước ngắn.</li>
                            </ul>
                        </div>

                        <div class="p-4 border-2 border-black bg-white rounded space-y-2">
                            <span class="text-xs font-mono font-bold uppercase text-black block">2. Lộ Trình Cải Thiện Hệ Thống (Improvement Plan)</span>
                            <div class="text-xs font-serif text-zinc-800 space-y-1">
                                <div>• <strong>Early Event Trigger O(1):</strong> Kiểm tra nhanh ngưỡng phương sai năng lượng trước khi gọi Verifier. Nếu tín hiệu dao động mạnh (có sự kiện vận động mới), lập tức chạy CNN mà không tốn 1.48 ms tính bound!</div>
                                <div>• <strong>Incremental Conv Reuse:</strong> Nhờ H13 đã chứng minh, tái sử dụng 50% phép tính của tầng Conv1D để tiết kiệm thời gian ngay cả khi Fallback!</div>
                            </div>
                        </div>
                    </div>

                    <div class="p-4 border border-black rounded bg-zinc-50 flex flex-col items-center justify-center">
                        <img src="assets/per_class_performance.png" alt="Per-class Performance Breakdown" class="max-h-72 object-contain rounded border border-zinc-300">
                        <span class="text-xs font-mono text-zinc-500 mt-2">Hiệu năng phân loại từng lớp: Ranh giới rõ rệt giữa lớp cơ bản và lớp chuyển thế</span>
                    </div>
                </div>
            `,
            takeaway: "Phân tích lỗi giúp nhóm nhận diện đúng điểm nghẽn và đưa ra giải pháp khắc phục thực chất bằng kỹ thuật lọc sự kiện sớm."
        },
        {
            id: 10,
            badge: "WEEK 3 SPECIAL • WEEK 4 TASKS",
            title: "Kế Hoạch Mô Hình Hóa Tuần 4 (Planned Tasks for Week 4)",
            subtitle: "3 Giải pháp kỹ thuật trọng tâm để vượt qua Blocker A & Blocker B",
            content: `
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                        <div class="bg-black text-white px-2 py-1 font-mono font-bold text-xs text-center rounded">GIẢI PHÁP 1 (P0)</div>
                        <h4 class="font-serif font-bold text-black text-sm">Adaptive Dynamic Bounds</h4>
                        <p class="text-xs font-serif text-zinc-700 leading-relaxed">
                            Thay vì dùng miền hình hộp đẳng hướng (Box domain) cố định, xây dựng miền Ellipsoid bám theo các trục chính của PCA (9,944 deltas) và vector gradient biên quyết định, giúp nới rộng bán kính tái sử dụng hợp thức.
                        </p>
                    </div>

                    <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                        <div class="bg-black text-white px-2 py-1 font-mono font-bold text-xs text-center rounded">GIẢI PHÁP 2 (P0)</div>
                        <h4 class="font-serif font-bold text-black text-sm">Early Event Trigger O(1)</h4>
                        <p class="text-xs font-serif text-zinc-700 leading-relaxed">
                            Thiết lập bộ lọc kích hoạt sự kiện siêu nhẹ O(1) kiểm tra chênh lệch năng lượng gia tốc (Acc Energy Delta). Nếu phát hiện bước nhảy vận động, bỏ qua verifier và chuyển thẳng sang CNN để triệt tiêu overhead 1.48 ms.
                        </p>
                    </div>

                    <div class="p-4 border-2 border-black bg-white rounded space-y-3">
                        <div class="bg-black text-white px-2 py-1 font-mono font-bold text-xs text-center rounded">GIẢI PHÁP 3 (P1)</div>
                        <h4 class="font-serif font-bold text-black text-sm">Incremental Feature Reuse</h4>
                        <p class="text-xs font-serif text-zinc-700 leading-relaxed">
                            Tận dụng minh chứng bitwise H13 để lưu cache đặc trưng của 3 tầng Conv1D. Khi Fallback, chỉ tính toán 64 mẫu thời gian mới và ghép nối với cache, giúp cắt giảm 50% FLOPs của tầng tích chập!
                        </p>
                    </div>
                </div>

                <div class="mt-4 p-4 border border-black bg-zinc-50 rounded flex items-center justify-between font-mono text-xs">
                    <span><strong>Mục tiêu Tuần 4:</strong> Đưa Net Speedup từ 0.52x vượt ngưỡng 1.0x (đạt tiết kiệm thời gian thực sự) và nghiệm thu dự án.</span>
                    <span class="bg-emerald-700 text-white px-2 py-0.5 rounded font-bold">READY FOR WEEK 4</span>
                </div>
            `,
            takeaway: "Lộ trình Tuần 4 tập trung xử lý dứt điểm 2 nút thắt kỹ thuật bằng 3 giải pháp kiến trúc có luận cứ vững chắc."
        }
    ]
};

// State variables
let curFullDeckWeek = 1;
let curFullSlideIdx = 1;

function openFullSlideModal(weekNum, slideNum) {
    curFullDeckWeek = weekNum || 1;
    curFullSlideIdx = slideNum || 1;
    const modal = document.getElementById('full-slide-modal');
    if (!modal) return;
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    renderFullSlide();
}

function closeFullSlideModal() {
    const modal = document.getElementById('full-slide-modal');
    if (!modal) return;
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

function switchFullDeckWeek(w) {
    curFullDeckWeek = w;
    curFullSlideIdx = 1;
    renderFullSlide();
}

function nextFullSlide() {
    const deck = FULL_DECKS[curFullDeckWeek];
    if (curFullSlideIdx < deck.length) {
        curFullSlideIdx++;
    } else {
        curFullSlideIdx = 1; // loop
    }
    renderFullSlide();
}

function prevFullSlide() {
    const deck = FULL_DECKS[curFullDeckWeek];
    if (curFullSlideIdx > 1) {
        curFullSlideIdx--;
    } else {
        curFullSlideIdx = deck.length; // loop
    }
    renderFullSlide();
}

function toggleBrowserFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen().catch(() => {});
        }
    }
}

function renderFullSlide() {
    const deck = FULL_DECKS[curFullDeckWeek];
    if (!deck || deck.length === 0) return;
    if (curFullSlideIdx > deck.length) curFullSlideIdx = deck.length;
    if (curFullSlideIdx < 1) curFullSlideIdx = 1;

    const s = deck[curFullSlideIdx - 1];

    // Update Counter & Badge
    document.getElementById('modal-slide-counter').innerText = `SLIDE ${curFullSlideIdx < 10 ? '0' + curFullSlideIdx : curFullSlideIdx} / ${deck.length < 10 ? '0' + deck.length : deck.length}`;
    document.getElementById('modal-slide-badge').innerText = s.badge;
    document.getElementById('modal-slide-title').innerText = s.title;
    document.getElementById('modal-slide-subtitle').innerText = s.subtitle;
    document.getElementById('modal-slide-body').innerHTML = s.content;
    document.getElementById('modal-slide-takeaway').innerText = s.takeaway;

    // Update Week Buttons in Modal Top
    for (let w = 1; w <= 3; w++) {
        const btn = document.getElementById(`modal-deck-w${w}`);
        if (btn) {
            btn.className = (w === curFullDeckWeek)
                ? "px-3 py-1 font-mono text-xs font-bold bg-black text-white border border-black rounded-sm"
                : "px-3 py-1 font-mono text-xs font-medium bg-zinc-200 text-zinc-700 hover:bg-zinc-300 border border-zinc-400 rounded-sm";
        }
    }

    // Render Bottom Indicator Pills
    const pillsContainer = document.getElementById('modal-slide-pills');
    if (pillsContainer) {
        pillsContainer.innerHTML = deck.map((item, idx) => `
            <button onclick="curFullSlideIdx=${idx + 1}; renderFullSlide();" 
                    title="${item.title}"
                    class="px-2 py-0.5 text-[11px] font-mono rounded border transition-all ${idx + 1 === curFullSlideIdx 
                        ? 'bg-black text-white font-bold border-black' 
                        : 'bg-white text-zinc-600 hover:bg-zinc-100 border-zinc-300'}">
                ${idx + 1}
            </button>
        `).join('');
    }

    // Re-render MathJax / KaTeX if available
    if (window.renderMathInElement) {
        window.renderMathInElement(document.getElementById('modal-slide-body'), {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });
    }
}

// Global Keyboard Navigation for Full Slide Modal
document.addEventListener('keydown', (e) => {
    const modal = document.getElementById('full-slide-modal');
    const isModalActive = modal && modal.classList.contains('active');

    if (e.key === 'p' || e.key === 'P') {
        if (!isModalActive) {
            openFullSlideModal(currentWeekIndex || 1, 1);
            e.preventDefault();
        }
    }

    if (isModalActive) {
        if (e.key === 'Escape') {
            closeFullSlideModal();
            e.preventDefault();
        } else if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
            nextFullSlide();
            e.preventDefault();
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
            prevFullSlide();
            e.preventDefault();
        } else if (e.key === '1') {
            switchFullDeckWeek(1);
            e.preventDefault();
        } else if (e.key === '2') {
            switchFullDeckWeek(2);
            e.preventDefault();
        } else if (e.key === '3') {
            switchFullDeckWeek(3);
            e.preventDefault();
        } else if (e.key === 'f' || e.key === 'F') {
            toggleBrowserFullscreen();
            e.preventDefault();
        }
    }
});
</script>
'''

# 2. Modal HTML structure to be placed right before </body>
MODAL_HTML = r'''
<!-- ====================================================================== -->
<!-- FULL SLIDE PRESENTATION OVERLAY MODAL -->
<!-- ====================================================================== -->
<div id="full-slide-modal" class="fixed inset-0 z-[999999] bg-zinc-950/95 flex flex-col justify-between p-3 md:p-6 backdrop-blur-md text-black select-none transition-all hidden">
    <!-- Center 16:9 Presentation Canvas -->
    <div class="bg-white border-2 border-black rounded-lg shadow-2xl w-full max-w-7xl h-[94vh] mx-auto flex flex-col justify-between overflow-hidden">
        
        <!-- Modal Top Bar -->
        <header class="border-b-2 border-black bg-zinc-100 px-6 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="w-7 h-7 bg-black text-white font-mono font-bold flex items-center justify-center text-xs rounded-sm">Ψ</span>
                <span class="font-serif font-bold text-base text-black">ECDR Slide Deck Presenter</span>
                
                <!-- Deck Week Selector -->
                <div class="flex items-center space-x-1 pl-4 border-l border-zinc-400">
                    <button id="modal-deck-w1" onclick="switchFullDeckWeek(1)" class="px-3 py-1 font-mono text-xs font-bold bg-black text-white border border-black rounded-sm">Tuần 1 (8 Slides)</button>
                    <button id="modal-deck-w2" onclick="switchFullDeckWeek(2)" class="px-3 py-1 font-mono text-xs font-medium bg-zinc-200 text-zinc-700 hover:bg-zinc-300 border border-zinc-400 rounded-sm">Tuần 2 (8 Slides)</button>
                    <button id="modal-deck-w3" onclick="switchFullDeckWeek(3)" class="px-3 py-1 font-mono text-xs font-medium bg-zinc-200 text-zinc-700 hover:bg-zinc-300 border border-zinc-400 rounded-sm">Tuần 3 (10 Slides)</button>
                </div>
            </div>

            <div class="flex items-center space-x-3">
                <span id="modal-slide-counter" class="font-mono text-xs font-bold bg-black text-white px-2.5 py-1 rounded">SLIDE 01 / 08</span>
                <button onclick="toggleBrowserFullscreen()" class="px-2.5 py-1 border border-black bg-white hover:bg-zinc-100 rounded text-xs font-mono font-bold" title="Toàn màn hình trình duyệt (F)">⛶ Fullscreen</button>
                <button onclick="closeFullSlideModal()" class="px-3 py-1 bg-rose-600 hover:bg-rose-700 text-white rounded text-xs font-mono font-bold" title="Thoát chế độ trình chiếu (Esc)">✕ Đóng (Esc)</button>
            </div>
        </header>

        <!-- Slide Canvas Body (Scrollable if needed, clean typography) -->
        <div class="flex-1 overflow-y-auto px-8 py-6 flex flex-col justify-between">
            <div>
                <!-- Slide Header -->
                <div class="border-b border-zinc-300 pb-3 mb-4 flex items-start justify-between">
                    <div>
                        <span id="modal-slide-badge" class="acad-badge acad-badge-black font-mono text-[11px] mb-1 inline-block">WEEK 1 • EXECUTIVE BRIEF</span>
                        <h2 id="modal-slide-title" class="text-2xl md:text-3xl font-serif font-bold text-black tracking-tight">Tiêu Đề Slide</h2>
                        <p id="modal-slide-subtitle" class="text-xs md:text-sm font-serif text-zinc-600 italic mt-0.5">Mô tả phụ cho nội dung slide</p>
                    </div>
                </div>

                <!-- Slide Interactive Body -->
                <div id="modal-slide-body" class="space-y-4">
                    <!-- Dynamic Slide Content Injected by JS -->
                </div>
            </div>

            <!-- Slide Bottom Takeaway Strip -->
            <div class="mt-6 pt-3 border-t border-zinc-200 flex items-center justify-between text-xs font-mono text-zinc-600">
                <div class="flex items-center space-x-2">
                    <span class="font-bold text-black uppercase">Key Takeaway:</span>
                    <span id="modal-slide-takeaway" class="font-serif italic text-zinc-800">Thông điệp cốt lõi của slide này</span>
                </div>
                <div class="text-[11px] text-zinc-400">ECDR Project Brief • Tutor Review Mode</div>
            </div>
        </div>

        <!-- Modal Footer Navigation Bar -->
        <footer class="border-t-2 border-black bg-zinc-50 px-6 py-2.5 flex items-center justify-between">
            <button onclick="prevFullSlide()" class="px-4 py-1.5 border border-black bg-white hover:bg-zinc-200 rounded font-mono text-xs font-bold flex items-center space-x-1">
                <span>← Slide Trước (Phím Trái)</span>
            </button>

            <!-- Slide Indicator Pills -->
            <div id="modal-slide-pills" class="flex items-center space-x-1 overflow-x-auto py-1">
                <!-- Pills populated dynamically -->
            </div>

            <div class="flex items-center space-x-4">
                <span class="text-[11px] font-mono text-zinc-500 hidden md:inline">Phím [← / →]: Đổi slide | [1, 2, 3]: Chọn tuần | [Esc]: Đóng</span>
                <button onclick="nextFullSlide()" class="px-5 py-1.5 border border-black bg-black text-white hover:bg-zinc-800 rounded font-mono text-xs font-bold flex items-center space-x-1 shadow">
                    <span>Slide Sau (Phím Phải / Space) →</span>
                </button>
            </div>
        </footer>
    </div>
</div>
'''

# 3. Add Launch Button to Header in index.html
# Look for header controls or tabs in index.html
HEADER_BTN_SNIPPET = r'''
            <!-- Launch Full Slide Modal Button -->
            <button onclick="openFullSlideModal(currentWeekIndex, 1)" class="px-3.5 py-1.5 bg-black hover:bg-zinc-800 text-white font-mono text-xs font-bold rounded-sm border border-black shadow flex items-center space-x-2 transition-all cursor-pointer animate-pulse" title="Mở chế độ trình chiếu toàn màn hình (Phím tắt: P hoặc F)">
                <span class="text-amber-300">▶</span>
                <span>TRÌNH CHIẾU FULL SLIDE</span>
                <span class="bg-zinc-800 text-zinc-300 px-1.5 py-0.5 rounded text-[10px]">Phím P</span>
            </button>
'''

# 4. Add Slide Gallery Cards Section for Week 1, Week 2, Week 3
def create_slide_cards_gallery(week_num, title, subtitle):
    return f'''
        <!-- SLIDE DECK GALLERY: WEEK {week_num} (CLICK TO OPEN FULL SLIDE) -->
        <section class="acad-card space-y-4 border-2 border-black bg-zinc-50/50">
            <div class="flex flex-col md:flex-row md:items-center justify-between border-b border-black pb-3 gap-2">
                <div>
                    <div class="flex items-center space-x-2">
                        <span class="acad-badge acad-badge-black font-mono">BỘ SLIDE THUYẾT TRÌNH TUẦN {week_num}</span>
                        <span class="text-xs font-mono text-emerald-800 font-bold bg-emerald-100 px-2 py-0.5 rounded border border-emerald-300">Chuẩn Báo Cáo Tutor</span>
                    </div>
                    <h3 class="text-xl font-serif font-bold text-black mt-1">{title}</h3>
                    <p class="text-xs font-serif text-zinc-600">{subtitle}</p>
                </div>
                <button onclick="openFullSlideModal({week_num}, 1)" class="px-4 py-2 bg-black hover:bg-zinc-800 text-white font-mono text-xs font-bold rounded border border-black shadow flex items-center space-x-2 self-start md:self-auto cursor-pointer">
                    <span class="text-amber-300">▶</span>
                    <span>TRÌNH CHIẾU FULL SLIDE TUẦN {week_num}</span>
                </button>
            </div>

            <!-- Slide Thumbnails Grid -->
            <div id="deck-gallery-w{week_num}" class="grid grid-cols-2 md:grid-cols-4 gap-3">
                <!-- Injected via DOM -->
            </div>
            <div class="text-[11px] font-mono text-zinc-500 text-right">
                💡 <em>Gợi ý: Nhấp vào bất kỳ slide nào để mở ngay chế độ thuyết trình Full Màn Hình (Full Slide Deck Modal).</em>
            </div>
        </section>
    '''

# Inject gallery population script
GALLERY_INIT_JS = r'''
<script>
// Initialize Slide Galleries on page load
document.addEventListener('DOMContentLoaded', () => {
    for (let w = 1; w <= 3; w++) {
        const container = document.getElementById(`deck-gallery-w${w}`);
        const deck = FULL_DECKS[w];
        if (container && deck) {
            container.innerHTML = deck.map((s, idx) => `
                <div onclick="openFullSlideModal(${w}, ${idx + 1})" 
                     class="p-3 bg-white border border-black hover:border-2 hover:border-black rounded cursor-pointer transition-all hover:shadow-md flex flex-col justify-between group">
                    <div>
                        <div class="flex items-center justify-between border-b border-zinc-200 pb-1 mb-1.5">
                            <span class="text-[10px] font-mono font-bold text-zinc-500 group-hover:text-black">SLIDE ${s.id < 10 ? '0' + s.id : s.id}</span>
                            <span class="text-[9px] font-mono bg-zinc-100 group-hover:bg-black group-hover:text-white px-1.5 py-0.5 rounded transition-all">Full ↗</span>
                        </div>
                        <h4 class="font-serif font-bold text-xs text-black group-hover:underline line-clamp-2">${s.title}</h4>
                        <p class="text-[11px] font-serif text-zinc-600 line-clamp-2 mt-1">${s.subtitle}</p>
                    </div>
                    <div class="mt-2 pt-1 border-t border-zinc-100 flex items-center justify-between text-[10px] font-mono text-zinc-400">
                        <span>${s.badge.split('•')[1] || 'Slide'}</span>
                        <span class="text-black font-bold">Xem →</span>
                    </div>
                </div>
            `).join('');
        }
    }
});
</script>
'''

# Update index.html
# 1. Update switchMasterWeek to keep track of currentWeekIndex
if "let currentWeekIndex = 1;" not in html:
    html = html.replace(
        "function switchMasterWeek(weekNum) {",
        "let currentWeekIndex = 1;\n        function switchMasterWeek(weekNum) {\n            currentWeekIndex = weekNum;"
    )

# 2. Add HEADER_BTN_SNIPPET next to header-badges
if 'id="header-badges"' in html and 'TRÌNH CHIẾU FULL SLIDE' not in html:
    html = html.replace(
        '<div id="header-badges" class="flex items-center space-x-2 text-xs font-mono">',
        HEADER_BTN_SNIPPET + '\n            <div id="header-badges" class="flex items-center space-x-2 text-xs font-mono">'
    )

# 3. Add Slide Galleries to each Week
if 'deck-gallery-w1' not in html:
    w1_gallery = create_slide_cards_gallery(1, "Danh Mục 8 Slide Thuyết Trình Báo Cáo Tuần 1", "Khởi tạo nền tảng, Phân tích SWOT, Kế hoạch giao tiếp, Toàn bộ WBS và Đóng băng mô hình SmallCNN1D")
    html = html.replace(
        '<div id="content-week-1" class="space-y-16">',
        '<div id="content-week-1" class="space-y-16">\n' + w1_gallery
    )

if 'deck-gallery-w2' not in html:
    w2_gallery = create_slide_cards_gallery(2, "Danh Mục 8 Slide Thuyết Trình Báo Cáo Tuần 2", "Định nghĩa bài toán, 3 Câu hỏi RQ1-RQ3, Khám phá dữ liệu HAPT 1.21M mẫu, Kiểm soát rò rỉ và WBS Tuần 3")
    html = html.replace(
        '<div id="content-week-2" class="space-y-16 hidden">',
        '<div id="content-week-2" class="space-y-16 hidden">\n' + w2_gallery
    )

if 'deck-gallery-w3' not in html:
    w3_gallery = create_slide_cards_gallery(3, "Danh Mục 10 Slide Thuyết Trình Báo Cáo Tuần 3", "Pipeline tiền xử lý, Tại sao chọn CNN1D?, Baseline Metrics, Toán IBP, Chứng minh H13 bitwise, Blocker A & B và Kế hoạch Tuần 4")
    html = html.replace(
        '<div id="content-week-3" class="space-y-16 hidden">',
        '<div id="content-week-3" class="space-y-16 hidden">\n' + w3_gallery
    )

# 4. Insert MODAL_HTML and SCRIPTS before </body>
if 'id="full-slide-modal"' not in html:
    html = html.replace(
        '</body>',
        MODAL_HTML + '\n' + SLIDES_JS_CONTENT + '\n' + GALLERY_INIT_JS + '\n</body>'
    )

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("SUCCESS: index.html updated with Full Slide Presentation Engine!")
