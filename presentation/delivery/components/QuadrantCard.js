/* ECDR Presentation Component: QuadrantCard.js
 * 4-Part Analytical Matrix: 1. Action, 2. Motivation, 3. Results, 4. Verdict (Good vs Bad)
 * Engineered with high-contrast color badges, micro-typography, and double-bezel cards.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  const points = items => `<ul class="points">${items.map(text => `<li>${text}</li>`).join('')}</ul>`;

  function QuadrantCard({ what, why, results, good, bad, title = '' }) {
    const renderList = val => Array.isArray(val) ? points(val) : val;

    return `
      <div class="quadrant-container">
        ${title ? `<div class="quadrant-title">${title}</div>` : ''}
        <div class="quadrant-grid">
          <div class="quadrant-card card-action">
            <div class="card-header-badge badge-what">1. MÌNH ĐANG LÀM GÌ? (ACTION)</div>
            <div class="card-body">${renderList(what)}</div>
          </div>
          <div class="quadrant-card card-motivation">
            <div class="card-header-badge badge-why">2. VÌ SAO MÌNH LÀM VẬY? (MOTIVATION)</div>
            <div class="card-body">${renderList(why)}</div>
          </div>
          <div class="quadrant-card card-results">
            <div class="card-header-badge badge-result">3. KẾT QUẢ TRẢ VỀ LÀ GÌ? (RESULTS)</div>
            <div class="card-body">${renderList(results)}</div>
          </div>
          <div class="quadrant-card card-verdict">
            <div class="card-header-badge badge-good">4. CÁI GÌ TỐT, CÁI GÌ XẤU? (VERDICT)</div>
            <div class="card-body">
              <div class="verdict-mini-section good">
                <strong class="verdict-label">[+] ĐIỂM TỐT / ƯU ĐIỂM:</strong>
                ${renderList(good)}
              </div>
              <div class="verdict-mini-section bad">
                <strong class="verdict-label">[-] ĐIỂM XẤU / HẠN CHẾ & THÁCH THỨC:</strong>
                ${renderList(bad)}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.QuadrantCard = QuadrantCard;
  global.ECDR_COMPONENTS.analysisQuad = QuadrantCard;
})(typeof window !== 'undefined' ? window : globalThis);
