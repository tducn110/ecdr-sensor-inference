/* ECDR Component: KPIBannerComponent.js
 * Renders key performance indicator badges for Week 4 executive verdict.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render(items = []) {
    const defaultKPIs = [
      { val: '100.0%', lbl: 'Độ bảo toàn quyết định (A_full)', tag: 'SOUND ARBITER', tone: 'emerald' },
      { val: '0', lbl: 'Vi phạm nhãn (D_all)', tag: 'ZERO FAULT', tone: 'emerald' },
      { val: '0.0%', lbl: 'Tỷ lệ tái sử dụng Stride 64', tag: 'VANISHING (H19)', tone: 'crimson' },
      { val: '1,562×', lbl: 'Tỷ số khe hở động học / Bán kính', tag: 'PHYSICAL GAP', tone: 'crimson' },
      { val: '2.90×', lbl: 'Ngưỡng cắt lý thuyết (A)', tag: 'B: CHƯA XÁC MINH TRÊN LUỒNG', tone: 'violet' },
      { val: '0.8537', lbl: 'Macro-F1 K=12 sau Gate G2', tag: 'RESOLVED', tone: 'emerald' }
    ];

    const data = items.length ? items : defaultKPIs;

    return `
      <div class="kpi-component-wrapper" role="region" aria-label="Chỉ số hiệu năng cốt lõi Tuần 4">
        <div class="kpi-row">
          ${data.map(kpi => `
            <div class="kpi-card tone-${kpi.tone || 'emerald'}">
              <div class="kpi-val">${kpi.val}</div>
              <div class="kpi-lbl">${kpi.lbl}</div>
              ${kpi.tag ? `<span class="kpi-tag">${kpi.tag}</span>` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.KPIBannerComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
