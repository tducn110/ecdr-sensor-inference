/* ECDR Presentation Page: week4.js
 * Comprehensive Component-Based Architecture for Week 4
 * Structured into Core Delivery (7 Main Slides for 2-3 Min Speech) + Appendix Deep Dives
 * (Zero speaker notes / No meta annotations)
 */
(function (global) {
  'use strict';
  global.ECDR_PAGES = global.ECDR_PAGES || {};
  const C = global.ECDR_COMPONENTS || {};

  const slide = (id, title, subtitle, content, extra = {}) => ({
    id, title, subtitle, content, notes: '', ...extra
  });

  const week4 = {
    title: 'Empirical Evaluation, Mechanics & Scaling Breakthroughs',
    subtitle: 'Đánh giá thực nghiệm 40 nhiệm vụ, đo lường 17 phương pháp, giải trình cơ chế sụp đổ và các bước đột phá mở rộng quy mô.',
    sections: [
      {
        id: 'core-delivery',
        label: 'Core Delivery (2–3 Min Speech)',
        description: '7 slide chính truyền tải trọn vẹn bản delivery speech 6 nhóm và ngưỡng cắt hòa vốn (G1–G6 + Kết luận).',
        slides: [
          'w4-g1-baseline',
          'w4-g2-incremental',
          'w4-g3-certified',
          'w4-g4-domain',
          'w4-g5-cost',
          'w4-g6-crossover',
          'w4-conclusion'
        ]
      },
      {
        id: 'appendix-methods',
        label: 'Phụ lục 1: Ma trận 40 Nhiệm vụ & 17 Phương pháp',
        description: 'Không gian thực nghiệm toàn diện từ M0 đến M7 và các phương pháp M00 đến M16.',
        slides: ['w4-app-tasks-methods']
      },
      {
        id: 'appendix-ledger',
        label: 'Phụ lục 2: Sổ cái 24 Giả thuyết (H01–H24)',
        description: 'Bằng chứng thực nghiệm đối chứng chi tiết từng giả thuyết khoa học.',
        slides: ['w4-app-hypotheses-ledger']
      },
      {
        id: 'appendix-stride',
        label: 'Phụ lục 3: Đánh đổi Bước trượt dày (Dense Stride)',
        description: 'Đánh đổi giữa độ tương quan tín hiệu cao và nguy cơ bùng nổ workload 64×.',
        slides: ['w4-app-dense-stride']
      },
      {
        id: 'appendix-gate',
        label: 'Phụ lục 4: Đột phá Cổng Gate G2 (K=12)',
        description: 'Chính sách Span-Anchored và Focal Loss xử lý mất cân bằng lớp cực đoan.',
        slides: ['w4-app-g2-full-closure']
      },
      {
        id: 'appendix-hub',
        label: 'Phụ lục 5: Trung tâm Tái lập & Đính kèm File',
        description: 'Toàn bộ mã nguồn, dữ liệu, báo cáo PDF IEEE và kịch bản thực thi một lệnh.',
        slides: ['w4-app-reproducibility-hub']
      },
      {
        id: 'appendix-future',
        label: 'Phụ lục 6: 5 Hướng nghiên cứu Chiến lược',
        description: 'Lộ trình phát triển dài hạn và chuyển giao công nghệ ECDR.',
        slides: ['w4-app-future-directions']
      }
    ],
    slides: [
      /* =========================================================================
       * 7 MAIN DELIVERY SLIDES (ALIGNED WITH 2-3 MINUTE DELIVERY SPEECH)
       * ========================================================================= */
      slide(
        'w4-g1-baseline',
        'Group 1: Bài toán nghiên cứu & Đường cơ sở Streaming (Baseline M00)',
        'Động lực từ 50% cửa sổ chồng lấn trên HAPT & 3 tiêu chuẩn cốt lõi: Soundness, Useful Coverage, Net Runtime Speedup',
        C.BaselineComponent ? C.BaselineComponent.render() : ''
      ),

      slide(
        'w4-g2-incremental',
        'Group 2: Tái sử dụng tính toán gia tăng (M03)',
        'Khớp bit 100% trên 5,034 cặp cửa sổ vs Nghịch lý độ trễ trên CPU (chậm hơn 5.6× do overhead cắt lát tensor)',
        C.IncrementalComponent ? C.IncrementalComponent.render() : ''
      ),

      slide(
        'w4-g3-certified',
        'Group 3: Tái sử dụng theo chứng nhận quyết định (M05–M13)',
        'Đánh giá Lipschitz, IBP, CROWN: 100% Reference Agreement nhờ fallback an toàn vs Tỷ lệ tái sử dụng hữu ích R = 0%',
        C.CertifiedComponent ? C.CertifiedComponent.render() : ''
      ),

      slide(
        'w4-g4-domain',
        'Group 4: Vì sao các phương pháp chứng nhận thất bại? (Root Cause Analysis)',
        'Phân tích khe hở vật lý 1,562.7× và 4 cơ chế sụp đổ độc lập của các phương pháp bao đóng hình học tĩnh ở S=64',
        C.DomainCollapseComponent ? C.DomainCollapseComponent.render() : ''
      ),

      slide(
        'w4-g5-cost',
        'Group 5: Kinh tế học thời gian chạy & Hai điểm nghẽn hệ thống (The Two Bottlenecks)',
        'Mô hình hóa chi phí toàn phần, phân biệt độ trễ M00 luồng (0.342 ms) vs forward thô (0.140 ms), và đánh giá Tam đề H24',
        C.CostBreakEvenComponent ? C.CostBreakEvenComponent.render() : ''
      ),

      slide(
        'w4-g6-crossover',
        'Group 6: Phân tích Crossover Hòa vốn & Mở rộng Quy mô Kiến trúc',
        'Khảo sát dải quy mô 0.28M–5.0M tham số, phân tách Kết quả A (Ngưỡng cắt 26.1 MMACs) vs Kết quả B (Chưa xác minh trên luồng)',
        C.ScalingComponent ? C.ScalingComponent.render() : ''
      ),

      slide(
        'w4-conclusion',
        'Group 7 / Tổng kết: 3 Phát hiện then chốt & Câu hỏi Nghiên cứu Mở',
        'Định hình ranh giới chân thực của ECDR và xác lập chương trình kiểm chứng cho các kiến trúc quy mô lớn',
        C.ConclusionComponent ? C.ConclusionComponent.render() : ''
      ),

      /* =========================================================================
       * APPENDIX SLIDES (TECHNICAL DEEP DIVES & DETAILED AUDIT FOR Q&A)
       * ========================================================================= */
      slide(
        'w4-app-tasks-methods',
        'Phụ lục 1: Không gian thực nghiệm: 40 nhiệm vụ & 17 phương pháp',
        'Ma trận đánh giá có hệ thống từ M0 đến M7 và M00 đến M16 trên toàn bộ pipeline thực nghiệm',
        C.MethodMatrixComponent ? C.MethodMatrixComponent.render() : ''
      ),

      slide(
        'w4-app-hypotheses-ledger',
        'Phụ lục 2: Sổ cái 24 Giả thuyết khoa học (H01–H24)',
        'Phân loại chi tiết và chứng cứ thực nghiệm đối chứng từ Tuần 3 sang Tuần 4',
        C.HypothesesLedgerComponent ? C.HypothesesLedgerComponent.render() : ''
      ),

      slide(
        'w4-app-dense-stride',
        'Phụ lục 3: Hướng nghiên cứu tương lai: Phân tích đánh đổi Stride dày',
        'Đánh đổi giữa độ tương quan tín hiệu và nguy cơ bùng nổ workload 64× (Giả thuyết chưa kiểm chứng thực nghiệm)',
        C.DenseStrideComponent ? C.DenseStrideComponent.render() : ''
      ),

      slide(
        'w4-app-g2-full-closure',
        'Phụ lục 4: Đột phá Cổng Gate G2: Khép lại 18 rào cản với K=12',
        'Chính sách cửa sổ Span-Anchored và hàm mất mát Class-Balanced Focal Loss',
        C.GateG2ClosureComponent ? C.GateG2ClosureComponent.render() : ''
      ),

      slide(
        'w4-app-reproducibility-hub',
        'Phụ lục 5: Trung tâm Tái lập Thực nghiệm & Đính kèm Tài nguyên',
        'Toàn bộ mã nguồn, dữ liệu, bài báo PDF và kịch bản thực thi một lệnh',
        C.AttachmentHubComponent ? C.AttachmentHubComponent.render() : ''
      ),

      slide(
        'w4-app-future-directions',
        'Phụ lục 6: 5 Hướng nghiên cứu chiến lược & Lộ trình phát triển',
        'Lộ trình chuyển giao công nghệ và tầm nhìn dài hạn của ECDR',
        C.FutureDirectionsComponent ? C.FutureDirectionsComponent.render() : ''
      )
    ]
  };

  global.ECDR_PAGES.week4 = week4;
})(typeof window !== 'undefined' ? window : globalThis);
