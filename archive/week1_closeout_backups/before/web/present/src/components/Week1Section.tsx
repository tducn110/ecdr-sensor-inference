import React, { useState } from "react"
import {
  BookOpen,
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  ArrowRight,
  Eye,
  Layers,
  Scale,
  Compass,
  FileText,
  ChevronDown,
  ChevronUp,
  Cpu,
  Zap,
  Clock
} from "lucide-react"
import { Latex, MathText } from "./Latex"
import { soundManager } from "../utils/audio"

export const Week1Section: React.FC<{ onOpenFigure: (id: string) => void }> = ({ onOpenFigure }) => {
  const [showContractDetails, setShowContractDetails] = useState(false)
  const [showPedagogicalGuide, setShowPedagogicalGuide] = useState(true)

  return (
    <section id="week-1" className="py-12 border-b border-gray-200">
      <div className="max-w-5xl mx-auto space-y-12">
        {/* Section Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 text-xs font-bold uppercase tracking-wider bg-gray-900 text-white rounded">
              Week 1
            </span>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Project Brief, Taxonomy & Literature Foundations
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-gray-900 tracking-tight">
            Tổng Quan Đề Tài: Tái Sử Dụng Quyết Định Có Chứng Nhận Toán Học (ECDR)
          </h2>
          <p className="text-sm sm:text-base text-gray-600 leading-relaxed">
            Khung lý thuyết cho phép bỏ qua inference mạng nơ-ron trên chuỗi tín hiệu cảm biến liên tục bằng cách chứng minh tính bất biến tuyệt đối của quyết định phân loại, bảo đảm toán học trên toàn miền biến thiên.
          </p>
        </div>

        {/* Core Purpose Callout */}
        <div className="p-5 rounded-2xl border border-blue-200 bg-blue-50/40 text-xs sm:text-sm text-blue-950 space-y-2.5 shadow-2xs">
          <div className="flex items-center gap-2 font-extrabold text-blue-900">
            <Compass className="w-5 h-5 text-blue-700" />
            <span className="text-sm uppercase tracking-wide">
              Sợi Chỉ Đỏ: Vấn Đề Tuần 1 Đang Giải Quyết Cho Toàn Bộ Đề Tài
            </span>
          </div>
          <div className="space-y-1.5 text-gray-700 text-xs leading-relaxed">
            <p>
              <strong>Vấn đề thực tế đang tồn tại:</strong> Khi hai cửa sổ cảm biến liên tiếp trùng nhau 50% dữ liệu, tại sao các kỹ sư không thể dùng các phương pháp caching truyền thống như đo khoảng cách Euclidean hay Cosine similarity để tái sử dụng nhãn?
            </p>
            <p>
              <strong>Lời giải của Tuần 1:</strong> Mạng nơ-ron là hàm phi tuyến cực kỳ nhạy cảm — khoảng cách đầu vào nhỏ không bảo đảm cùng nhãn (Zero Mathematical Guarantee). Tuần 1 thiết lập <strong>Khung chứng nhận 3 tầng</strong> (Candidate Geometry &rarr; Whole-Domain Bound &rarr; Query Membership) và <strong>Hợp đồng chuẩn hóa C1–C10</strong>. Định vị học thuật: ECDR không cạnh tranh "tái sử dụng activation" (như StreamiNNC/TinyDéjàVu) mà là <strong>tái sử dụng quyết định có bảo chứng toán học 100%</strong>.
            </p>
          </div>
        </div>

        {/* Presentation & Defense Guide for Week 1 */}
        <div className="border border-blue-200 rounded-xl bg-blue-50/40 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Compass className="w-5 h-5 text-blue-700" />
              <h3 className="text-base font-bold text-blue-950">
                Hướng Dẫn Thuyết Trình & Bảo Vệ Week 1 (Speaker Defense Script)
              </h3>
            </div>
            <button
              onClick={() => {
                soundManager.playClick()
                setShowPedagogicalGuide(!showPedagogicalGuide)
              }}
              className="text-xs font-semibold text-blue-700 hover:text-blue-900 flex items-center gap-1"
            >
              {showPedagogicalGuide ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              {showPedagogicalGuide ? "Thu gọn kịch bản" : "Mở kịch bản bảo vệ"}
            </button>
          </div>

          {showPedagogicalGuide && (
            <div className="space-y-3 text-xs sm:text-sm text-blue-950/90 leading-relaxed border-t border-blue-200/60 pt-3">
              <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                <div className="p-3 bg-white rounded-lg border border-blue-200 space-y-1">
                  <div className="font-bold text-blue-800 text-xs uppercase tracking-wider">Bước 1: Bối Cảnh</div>
                  <p className="text-xs text-gray-700">
                    Nêu vấn đề thiết bị đeo thông minh (Wearables) nhận chuỗi 50Hz, forward CNN 2,800 lần/giờ gây hao pin nghiêm trọng dù phần lớn thời gian người dùng giữ nguyên tư thế.
                  </p>
                </div>

                <div className="p-3 bg-white rounded-lg border border-blue-200 space-y-1">
                  <div className="font-bold text-blue-800 text-xs uppercase tracking-wider">Bước 2: Cảnh Báo Heuristics</div>
                  <p className="text-xs text-gray-700">
                    Chỉ rõ vì sao caching theo heuristic khoảng cách hay cosine similarity là <strong>nguy hiểm</strong>: không có bảo đảm toán học, dễ bị phân loại sai khi vượt ranh giới quyết định.
                  </p>
                </div>

                <div className="p-3 bg-white rounded-lg border border-blue-200 space-y-1">
                  <div className="font-bold text-blue-800 text-xs uppercase tracking-wider">Bước 3: Bản Chất ECDR</div>
                  <p className="text-xs text-gray-700">
                    ECDR chứng minh một miền an toàn <Latex math="\mathcal{S}(x)" /> quanh anchor <Latex math="x" /> mà ở đó lớp chiến thắng <Latex math="y^* = \arg\max_k F_k(x)" /> giữ nguyên 100%.
                  </p>
                </div>

                <div className="p-3 bg-white rounded-lg border border-blue-200 space-y-1">
                  <div className="font-bold text-blue-800 text-xs uppercase tracking-wider">Bước 4: Ranh Giới Đề Tài</div>
                  <p className="text-xs text-gray-700">
                    Khẳng định trung thực: ECDR <strong>chỉ bảo toàn quyết định của reference frozen</strong>, không thay thế ground-truth accuracy và không tái tạo softmax probabilities.
                  </p>
                </div>

                <div className="p-3 bg-white rounded-lg border border-blue-200 space-y-1">
                  <div className="font-bold text-blue-800 text-xs uppercase tracking-wider">Bước 5: Lợi Thế Chi Phí</div>
                  <p className="text-xs text-gray-700">
                    Kiểm tra membership chỉ tốn <Latex math="O(C s)" /> (khoảng <Latex math="0.02\,\text{ms}" />), rẻ hơn gấp hàng trăm lần so với full CNN forward (<Latex math="4.0\,\text{ms}" />).
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Bento Grid 1: Problem vs Solution */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Card A: Problem Context */}
          <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-4 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-red-700">
              <AlertTriangle className="w-4 h-4 text-red-600" />
              1. Bối Cảnh Thực Tiễn: Lãng Phí Suy Luận Cảm Biến
            </div>
            <h3 className="text-lg font-bold text-gray-900">
              Nghịch Lý Của Xử Lý Cửa Sổ Trượt (Continuous Streaming Inference)
            </h3>
            <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
              Thiết bị nhận chuỗi cảm biến liên tục từ gia tốc kế và con quay hồi chuyển, cắt thành các cửa sổ <Latex math="X_t \in \mathbb{R}^{6 \times 128}" /> rồi chạy classifier <Latex math="f_\theta" /> cho từng cửa sổ.
            </p>
            <div className="p-3.5 bg-gray-50 rounded-lg border border-gray-200 text-xs text-gray-700 space-y-2 font-medium">
              <div className="flex items-start gap-2">
                <span className="text-red-500 font-bold">•</span>
                <span><strong>Quá tải suy luận:</strong> Stride 64 mẫu (1.28s) buộc mô hình forward liên tục <Latex math="f_s/s = 0.78125\,\text{windows/s}" /> (<Latex math="2,812.5" /> lần/giờ) bất kể tín hiệu đứng yên hay biến động.</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-500 font-bold">•</span>
                <span><strong>Độ lặp thừa thãi:</strong> Hai cửa sổ liên tiếp chia sẻ 64 mẫu (50% dữ liệu gối nhau). Khi người dùng ngồi hoặc ngủ, hàng nghìn forward liên tiếp cho ra cùng một kết quả.</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-red-500 font-bold">•</span>
                <span><strong>Hạn chế Heuristics:</strong> Các bộ nhớ đệm (caching) dựa trên heuristic khoảng cách hay cosine similarity không có chứng nhận hình học; khi phân loại sai (False Accept) sẽ gây nguy hiểm trong các ứng dụng y tế hoặc nhận diện té ngã.</span>
              </div>
            </div>
          </div>

          {/* Card B: Core Proposed Method */}
          <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-4 shadow-xs">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-700">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              2. Đột Phá Của ECDR: Chứng Nhận Toán Học Bất Biến
            </div>
            <h3 className="text-lg font-bold text-gray-900">
              Event-Conditioned Certified Decision Reuse
            </h3>
            <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
              Thay vì đoán mò xác suất, ECDR thiết lập một <strong>hợp đồng toán học nghiêm ngặt (Sound Certification Contract)</strong> quanh cửa sổ mốc (anchor window <Latex math="X^r" />):
            </p>
            <div className="p-3.5 bg-emerald-50/50 rounded-lg border border-emerald-200 text-xs text-emerald-950 space-y-2">
              <Latex display math="X \in \mathcal{S}_r \implies h(X) = h(X^r) = c, \quad c = \arg\max_{1 \le k \le K} F_k(X^r)" />
              <p className="leading-relaxed">
                Miễn là cửa sổ mới <Latex math="X" /> rơi trọn vào miền <Latex math="\mathcal{S}_r" />, hệ thống <strong>bảo đảm 100%</strong> lớp quyết định không đổi. Ta tái sử dụng ngay kết quả mà không cần kích hoạt mạng nơ-ron!
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs text-gray-600 pt-1">
              <div className="p-2 rounded bg-gray-50 border border-gray-200">
                <span className="block text-[11px] font-bold text-gray-500 uppercase">Chi Phí Kiểm Tra</span>
                <strong className="text-gray-900 font-mono text-sm">O(Cs) ≈ 0.02 ms</strong>
              </div>
              <div className="p-2 rounded bg-gray-50 border border-gray-200">
                <span className="block text-[11px] font-bold text-gray-500 uppercase">Full CNN Forward</span>
                <strong className="text-gray-900 font-mono text-sm">tf ≈ 4.0 ms</strong>
              </div>
            </div>
          </div>
        </div>

        {/* The 3-Tier Certification Hierarchy */}
        <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-gray-100">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-purple-700 block">
                Nguyên Lý Cốt Lõi
              </span>
              <h3 className="text-lg font-bold text-gray-900">
                Ba Tầng Bắt Buộc Trong Cấu Trúc Chứng Nhận (The 3-Tier Hierarchy)
              </h3>
            </div>
            <span className="text-xs font-mono bg-purple-50 text-purple-800 px-2 py-0.5 rounded border border-purple-200 font-semibold">
              Strict Dependency Order
            </span>
          </div>

          <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
            Chứng nhận ECDR không thể nhảy cóc. Ba tầng phải được thực thi theo thứ tự logic chặt chẽ, tầng trước là tiền đề cho tầng sau:
          </p>

          <div className="p-4 bg-gray-50 rounded-xl border border-gray-200 text-center font-mono text-sm">
            <Latex display math="\underbrace{\text{Candidate Geometry}}_{\text{Tier 1: Domain Definition}} \longrightarrow \underbrace{\text{Whole-Domain Bound}}_{\text{Tier 2: Certified Margin Proof}} \longrightarrow \underbrace{\text{Query Membership}}_{\text{Tier 3: Runtime Test } O(Cs)}" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-700">
            <div className="p-3.5 bg-white rounded-lg border border-gray-200 space-y-1.5">
              <div className="flex items-center gap-1.5 font-bold text-gray-900">
                <span className="w-5 h-5 rounded-full bg-gray-100 text-gray-800 flex items-center justify-center text-xs font-mono">1</span>
                <span>Candidate Geometry</span>
              </div>
              <p className="text-gray-600 leading-relaxed">
                Đề xuất hình học miền <Latex math="\mathcal{S}_r" /> (Axis-aligned Box, Wave-domain Box, Directional Tube, hoặc PCA Ellipsoid). Nếu chọn hình học quá thô hoặc sai lệch phân bố dữ liệu, miền sẽ vô dụng.
              </p>
            </div>

            <div className="p-3.5 bg-white rounded-lg border border-gray-200 space-y-1.5">
              <div className="flex items-center gap-1.5 font-bold text-gray-900">
                <span className="w-5 h-5 rounded-full bg-gray-100 text-gray-800 flex items-center justify-center text-xs font-mono">2</span>
                <span>Whole-Domain Bound</span>
              </div>
              <p className="text-gray-600 leading-relaxed">
                Chứng minh toán học rằng <Latex math="\min_{x \in \mathcal{S}_r} \min_{j \neq c} (F_c(x) - F_j(x)) > 0" />. Một box kiểm tra rất nhanh nhưng không chứng minh được margin dương toàn miền thì không thể dùng để reuse!
              </p>
            </div>

            <div className="p-3.5 bg-white rounded-lg border border-gray-200 space-y-1.5">
              <div className="flex items-center gap-1.5 font-bold text-gray-900">
                <span className="w-5 h-5 rounded-full bg-gray-100 text-gray-800 flex items-center justify-center text-xs font-mono">3</span>
                <span>Query Membership</span>
              </div>
              <p className="text-gray-600 leading-relaxed">
                Khi cửa sổ mới <Latex math="X_t" /> xuất hiện ở runtime, kiểm tra xem <Latex math="X_t \in \mathcal{S}_r" /> trong thời gian <Latex math="O(Cs)" />. Nếu thuộc miền, ta dispatch ngay quyết định <Latex math="c" />.
              </p>
            </div>
          </div>
        </div>

        {/* Literature & Related Work Positioning Matrix (Table 2) */}
        <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-gray-100">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 block">
                Định Vị Học Thuật (Related Work)
              </span>
              <h3 className="text-lg font-bold text-gray-900">
                Ma Trận Định Vị Các Hướng Tiếp Cận & Đối Chiếu LipCache
              </h3>
            </div>
            <span className="text-xs font-mono bg-gray-100 text-gray-800 px-2 py-0.5 rounded border border-gray-200 font-semibold">
              Table 2 Monograph
            </span>
          </div>

          <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">
            Đề tài ECDR được kế thừa và phân định ranh giới rõ ràng với các công trình tiên phong trong cộng đồng khoa học quốc tế:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border border-gray-200 rounded-lg">
              <thead className="bg-gray-50 border-b border-gray-200 text-gray-700 font-bold uppercase text-[11px]">
                <tr>
                  <th className="p-3 border-r border-gray-200">Hướng Nghiên Cứu</th>
                  <th className="p-3 border-r border-gray-200">Đối Tượng Chính</th>
                  <th className="p-3 border-r border-gray-200">Điều Kiện Cung Cấp</th>
                  <th className="p-3">Vai Trò Đối Với ECDR</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 text-gray-700">
                <tr className="hover:bg-gray-50/50">
                  <td className="p-3 font-bold text-gray-900 border-r border-gray-200">Full CNN (Standard)</td>
                  <td className="p-3 border-r border-gray-200">Mỗi cửa sổ <Latex math="X_t" /></td>
                  <td className="p-3 border-r border-gray-200">Quyết định reference sau forward đầy đủ</td>
                  <td className="p-3 font-semibold text-gray-600">Baseline chuẩn về độ chính xác và chi phí</td>
                </tr>

                <tr className="hover:bg-gray-50/50">
                  <td className="p-3 font-bold text-red-700 border-r border-gray-200">Heuristic Reuse</td>
                  <td className="p-3 border-r border-gray-200">Similarity / Distance</td>
                  <td className="p-3 border-r border-gray-200">Quy tắc thực nghiệm; tự nó không là whole-domain proof</td>
                  <td className="p-3 text-red-700 font-medium">Negative control (nguy cơ sai nhãn không kiểm soát)</td>
                </tr>

                <tr className="hover:bg-gray-50/50">
                  <td className="p-3 font-bold text-gray-900 border-r border-gray-200">StreamiNNC (arXiv 2024)</td>
                  <td className="p-3 border-r border-gray-200">Streaming activations</td>
                  <td className="p-3 border-r border-gray-200">Cơ chế giảm tính trùng; cần xử lý biên và alignment</td>
                  <td className="p-3 text-gray-600">Prior art chứng minh việc khai thác overlap đã có tiền lệ</td>
                </tr>

                <tr className="hover:bg-gray-50/50">
                  <td className="p-3 font-bold text-gray-900 border-r border-gray-200">TinyDéjàVu (arXiv 2025)</td>
                  <td className="p-3 border-r border-gray-200">Incremental sensor MCU</td>
                  <td className="p-3 border-r border-gray-200">Cơ chế streaming và quản lý bộ nhớ trên vi điều khiển</td>
                  <td className="p-3 text-gray-600">Prior art về tối ưu bộ nhớ cache trên MCU nhúng</td>
                </tr>

                <tr className="hover:bg-gray-50/50 bg-amber-50/40">
                  <td className="p-3 font-bold text-amber-900 border-r border-gray-200">LipCache (arXiv 2026)</td>
                  <td className="p-3 border-r border-gray-200">GuardNet / Cache Proxy</td>
                  <td className="p-3 border-r border-gray-200">Certificate bảo toàn GuardNet cho phân loại ảnh 2D</td>
                  <td className="p-3 font-semibold text-amber-900">
                    Đối chiếu cache semantics. Lưu ý: LipCache chỉ chứng nhận GuardNet, không chứng nhận MainNet!
                  </td>
                </tr>

                <tr className="hover:bg-gray-50/50">
                  <td className="p-3 font-bold text-blue-900 border-r border-gray-200">Lipschitz / IBP / CROWN</td>
                  <td className="p-3 border-r border-gray-200">Mạng và perturbation</td>
                  <td className="p-3 border-r border-gray-200">Cận toán học có điều kiện trên miền được khai báo</td>
                  <td className="p-3 text-blue-900 font-medium">Bộ công cụ toán học nền tảng (Machinery)</td>
                </tr>

                <tr className="hover:bg-emerald-50/50 bg-emerald-50/30">
                  <td className="p-3 font-bold text-emerald-950 border-r border-gray-200">ECDR (Đề Xuất)</td>
                  <td className="p-3 border-r border-gray-200">Quyết định frozen reference</td>
                  <td className="p-3 border-r border-gray-200">Reuse khi bound và membership đồng thời thỏa mãn</td>
                  <td className="p-3 font-bold text-emerald-800">
                    Đóng góp lý thuyết: kết hợp cấu trúc cửa sổ trượt cảm biến 1D với chứng nhận toán học
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-xs text-gray-600 leading-relaxed">
            <strong>Ghi chú liêm chính học thuật:</strong> Không tự chuyển con số speedup trên ảnh 2D của LipCache thành kết quả trên dữ liệu cảm biến HAPT. ECDR định vị rõ ràng các nền tảng được kế thừa và các câu hỏi mở cần kiểm chứng thực nghiệm.
          </div>
        </div>

        {/* 5 Research Questions (RQ1 - RQ5) */}
        <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-gray-100">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-blue-700 block">
                Khung Khoa Học
              </span>
              <h3 className="text-lg font-bold text-gray-900">
                5 Câu Hỏi Nghiên Cứu Trọng Tâm (Research Questions RQ1 – RQ5)
              </h3>
            </div>
            <span className="text-xs font-mono bg-gray-100 text-gray-800 px-2.5 py-1 rounded border border-gray-200 font-semibold">
              Formal RQ Framework
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* RQ1 */}
            <div className="p-4 rounded-xl border border-gray-200 bg-gray-50/50 space-y-2 hover:border-blue-300 transition-colors">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-blue-700">RQ1: Cấu Trúc Dữ Liệu</span>
                <span className="text-[10px] bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded font-semibold">Toán Học</span>
              </div>
              <h4 className="text-xs font-bold text-gray-900">
                Cấu Trúc Xác Định Nào Có Thể Dùng Để Giảm Bài Toán?
              </h4>
              <p className="text-xs text-gray-600 leading-relaxed">
                Đâu là quan hệ xác định do dịch cửa sổ (<Latex math="X_{t+1}[:, 0:W-s] = X_t[:, s:W]" />) và đâu chỉ là giả thuyết thống kê (tính mượt, phân bố)?
              </p>
            </div>

            {/* RQ2 */}
            <div className="p-4 rounded-xl border border-gray-200 bg-gray-50/50 space-y-2 hover:border-blue-300 transition-colors">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-blue-700">RQ2: Điều Kiện Đủ</span>
                <span className="text-[10px] bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded font-semibold">Toán Học</span>
              </div>
              <h4 className="text-xs font-bold text-gray-900">
                Điều Kiện Nào Bảo Toàn Quyết Định Trên Toàn Miền?
              </h4>
              <p className="text-xs text-gray-600 leading-relaxed">
                Với reference <Latex math="F" /> và candidate domain <Latex math="\mathcal{S}" />, điều kiện đủ nào bảo đảm mọi pairwise lower margin strictly positive: <Latex math="\min_{j \neq c} \ell_{cj} > 0" />?
              </p>
            </div>

            {/* RQ3 */}
            <div className="p-4 rounded-xl border border-gray-200 bg-gray-50/50 space-y-2 hover:border-blue-300 transition-colors">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-purple-700">RQ3: Hình Học Miền</span>
                <span className="text-[10px] bg-purple-100 text-purple-800 px-1.5 py-0.5 rounded font-semibold">Thực Nghiệm</span>
              </div>
              <h4 className="text-xs font-bold text-gray-900">
                Miền Căn Chỉnh Có Tăng Xác Suất Bao Phủ (Coverage)?
              </h4>
              <p className="text-xs text-gray-600 leading-relaxed">
                Miền căn chỉnh theo thời gian (Wave-domain) hoặc theo event (Directional Tube) có tăng hit rate so với hộp đẳng hướng đồng nhất hay không?
              </p>
            </div>

            {/* RQ4 */}
            <div className="p-4 rounded-xl border border-gray-200 bg-gray-50/50 space-y-2 hover:border-blue-300 transition-colors">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-amber-700">RQ4: Vòng Đời & Trôi</span>
                <span className="text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.5 rounded font-semibold">Soundness</span>
              </div>
              <h4 className="text-xs font-bold text-gray-900">
                Certificate Tái Sử Dụng Đa Bước Có Bị Trôi Anchor?
              </h4>
              <p className="text-xs text-gray-600 leading-relaxed">
                Có thể dùng certificate nhiều bước mà không làm trôi sai số (Anchor Drift) hay không? Cần mang theo thông tin gì trong chứng chỉ?
              </p>
            </div>

            {/* RQ5 */}
            <div className="p-4 rounded-xl border border-gray-200 bg-gray-50/50 space-y-2 hover:border-blue-300 transition-colors md:col-span-2 lg:col-span-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-emerald-700">RQ5: Amortization & Hiệu Quả Chi Phí</span>
                <span className="text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded font-semibold">Break-Even</span>
              </div>
              <h4 className="text-xs font-bold text-gray-900">
                Điều Kiện Hòa Vốn Giữa Chi Phí Build và Thời Gian Tiết Kiệm?
              </h4>
              <p className="text-xs text-gray-600 leading-relaxed">
                Những mức hit rate <Latex math="r" />, overhead kiểm tra <Latex math="t_v" />, chi phí xây dựng <Latex math="t_b" /> và chân trời <Latex math="H" /> nào đảm bảo ECDR thực sự nhanh hơn full forward: <Latex math="r > \frac{t_v + t_m + t_b}{t_f + t_b}" />?
              </p>
            </div>
          </div>
        </div>

        {/* Unified Certification Contract C1 - C10 */}
        <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-blue-700 block">
                Hợp Đồng Toàn Diện (Unified Contract)
              </span>
              <h3 className="text-lg font-bold text-gray-900">
                10 Điều Kiện Hợp Đồng C1 – C10 Của Chứng Nhận ECDR
              </h3>
            </div>
            <button
              onClick={() => {
                soundManager.playClick()
                setShowContractDetails(!showContractDetails)
              }}
              className="text-xs font-semibold text-blue-600 hover:underline flex items-center gap-1"
            >
              {showContractDetails ? "Thu gọn bảng C1–C10" : "Xem chi tiết 10 điều khoản"}
            </button>
          </div>

          <p className="text-xs text-gray-600 leading-relaxed">
            Mọi tuyên bố về tính đúng đắn (Soundness) và hiệu quả (Efficiency) của ECDR đều phải tuân thủ nghiêm ngặt 10 điều khoản hợp đồng được chuẩn hóa:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <strong className="text-blue-900 block font-mono">C1: Reference Frozen</strong>
              <span className="text-gray-600">Mạng cố định trọng số, graph, batchnorm eval, thứ tự lớp và quy tắc tie-break.</span>
            </div>

            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <strong className="text-blue-900 block font-mono">C2: Preprocessing Đóng Gói</strong>
              <span className="text-gray-600">Thang đo, thứ tự kênh, mean/std normalization được tích hợp trực tiếp vào hàm <Latex math="F = f_\theta \circ \mathcal{P}" />.</span>
            </div>

            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <strong className="text-blue-900 block font-mono">C3: Valid Lower Bounds</strong>
              <span className="text-gray-600">Mọi cận dưới logit hoặc margin phải sound trên toàn miền <Latex math="\mathcal{S}" /> hoặc enclosure chứa miền.</span>
            </div>

            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <strong className="text-blue-900 block font-mono">C4: Strictly Positive Lower Margin</strong>
              <span className="text-gray-600">Mọi đối thủ <Latex math="j \neq c" /> đều phải có <Latex math="\ell_{cj} > 0" /> sau khi trừ sai số số học.</span>
            </div>

            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <strong className="text-blue-900 block font-mono">C5: Conservative Membership</strong>
              <span className="text-gray-600">Kiểm tra thuộc miền tại runtime không được false-accept bất kỳ điểm nào nằm ngoài miền chứng nhận.</span>
            </div>

            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
              <strong className="text-blue-900 block font-mono">C6: Safe Fallback On Miss</strong>
              <span className="text-gray-600">Khi miss, hết hạn horizon, hoặc gặp giá trị non-finite (NaN/Inf), phải chạy lại full reference.</span>
            </div>

            {showContractDetails && (
              <>
                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <strong className="text-blue-900 block font-mono">C7: Complete Accounting</strong>
                  <span className="text-gray-600">Chi phí đo lường phải tính đủ thời gian build, verify, lookup, state management, và fallback.</span>
                </div>

                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <strong className="text-blue-900 block font-mono">C8: Fair Experimental Comparison</strong>
                  <span className="text-gray-600">So sánh trên cùng dữ liệu, cùng mô hình, cùng phần cứng và cùng độ chính xác số học.</span>
                </div>

                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <strong className="text-blue-900 block font-mono">C9: Lifecycle Identity Retention</strong>
                  <span className="text-gray-600">Giữ nguyên anchor gốc trong chứng chỉ hoặc dùng định lý chuyển giao hợp lệ để chặn anchor drift.</span>
                </div>

                <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                  <strong className="text-blue-900 block font-mono">C10: Float32 Soundness Justification</strong>
                  <span className="text-gray-600">Phải tính đến sai số làm tròn số thực dấu phẩy động <Latex math="\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}" />.</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* 3-Week Research Plan Matrix (Table 3) */}
        <div className="border border-gray-200 rounded-xl p-6 bg-white space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-gray-100">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-gray-500 block">
                Kế Hoạch Nghiên Cứu Chuẩn Hóa
              </span>
              <h3 className="text-lg font-bold text-gray-900">
                Lộ Trình Nghiên Cứu 3 Tuần Lý Thuyết (Table 3)
              </h3>
            </div>
            <span className="text-xs font-mono bg-gray-100 text-gray-800 px-2 py-0.5 rounded border border-gray-200 font-semibold">
              Research Roadmap
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/20 space-y-2">
              <div className="font-bold text-blue-900 text-sm">Tuần 1: Nghiên cứu cái gì?</div>
              <div className="text-gray-600 space-y-1">
                <div><strong>Đầu ra:</strong> Brief, RQ1–RQ5, thuật ngữ, prior work matrix, constraints.</div>
                <div><strong>Tiêu chí hoàn thành:</strong> Phân biệt rạch ròi giữa decision preservation, task accuracy và cost.</div>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/20 space-y-2">
              <div className="font-bold text-emerald-900 text-sm">Tuần 2: Input thật sự là gì?</div>
              <div className="text-gray-600 space-y-1">
                <div><strong>Đầu ra:</strong> Data contract, window/label, normalization, shift identities, CNN audit.</div>
                <div><strong>Tiêu chí hoàn thành:</strong> Truy được mỗi biến toán về đúng đối tượng dữ liệu cảm biến thật.</div>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-purple-200 bg-purple-50/20 space-y-2">
              <div className="font-bold text-purple-900 text-sm">Tuần 3: Khi nào reuse đúng & rẻ?</div>
              <div className="text-gray-600 space-y-1">
                <div><strong>Đầu ra:</strong> Formal proofs, counterexamples, GAP cancellation, amortization, 24 hypotheses.</div>
                <div><strong>Tiêu chí hoàn thành:</strong> Mỗi claim có giả thiết, lập luận toán học và ranh giới rõ ràng.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
