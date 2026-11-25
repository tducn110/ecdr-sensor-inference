/* ECDR Presentation Page: overview.js
 * Executive cross-week selection covering Weeks 1–4 roadmaps and core breakthroughs.
 */
(function (global) {
  'use strict';
  global.ECDR_PAGES = global.ECDR_PAGES || {};

  const overviewIds = [
    'w1-agenda', 'w1-opening', 'w1-full-inference', 'w1-freeze', 'w1-quadrant',
    'w2-agenda', 'w2-data', 'w2-tensor', 'w2-counts', 'w2-cnn', 'w2-quadrant',
    'w3-agenda', 'w3-map', 'w3-h05', 'w3-h13', 'w3-h16', 'w3-h24', 'w3-quadrant',
    'w4-agenda', 'w4-g1-baseline', 'w4-g2-incremental', 'w4-g3-certified', 'w4-g4-domain',
    'w4-g5-cost', 'w4-g6-crossover', 'w4-conclusion', 'w4-app-reproducibility-hub'
  ];

  global.ECDR_PAGES.overview = {
    title: 'Weeks 1–4 Executive Synthesis & Strategic Roadmap',
    subtitle: 'Toàn cảnh 4 tuần nghiên cứu ECDR: Từ định nghĩa bài toán, kiến trúc dữ liệu, cơ chế toán học đến đánh giá thực nghiệm và mở rộng quy mô.',
    overviewIds: overviewIds
  };
})(typeof window !== 'undefined' ? window : globalThis);
