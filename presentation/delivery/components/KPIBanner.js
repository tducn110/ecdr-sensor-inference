/* ECDR Presentation Component: KPIBanner.js
 * Renders prominent metric cards with large figures, unit labels, and status badges.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function KPIBanner(items) {
    return `
      <div class="kpi-row" role="region" aria-label="Key Performance Indicators">
        ${items.map(item => {
          let val, lbl, tag = '', tone = '';
          if (Array.isArray(item)) {
            val = item[0];
            lbl = item[1];
            tag = item[2] || '';
            tone = item[3] || '';
          } else {
            val = item.val;
            lbl = item.lbl;
            tag = item.tag || '';
            tone = item.tone || '';
          }
          const toneClass = tone ? `tone-${tone}` : '';
          return `
            <div class="kpi-card ${toneClass}">
              <div class="kpi-val">${val}</div>
              <div class="kpi-lbl">${lbl}</div>
              ${tag ? `<span class="kpi-tag">${tag}</span>` : ''}
            </div>
          `;
        }).join('')}
      </div>
    `;
  }

  global.ECDR_COMPONENTS.KPIBanner = KPIBanner;
  global.ECDR_COMPONENTS.kpis = KPIBanner;
})(typeof window !== 'undefined' ? window : globalThis);
