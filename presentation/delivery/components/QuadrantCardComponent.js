/* ECDR Component: QuadrantCardComponent.js
 * 4-Part Analytical Matrix: Action, Motivation, Results, Verdict
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render({ title = 'Khung phân tích 4 câu hỏi cốt lõi', what = [], why = [], results = [], good = [], bad = [] }) {
    const list = items => `<ul class="points">${items.map(x => `<li>${x}</li>`).join('')}</ul>`;

    return `
      <div class="component-card quadrant-component">
        <div class="component-header">
          <span class="component-badge">PHƯƠNG PHÁP LUẬN</span>
          <h3 class="component-title">${title}</h3>
        </div>
        <div class="quadrant-grid">
          <div class="quadrant-card">
            <span class="card-header-badge badge-what">1. MÌNH ĐANG LÀM GÌ? (ACTION)</span>
            <div class="card-body">${Array.isArray(what) ? list(what) : what}</div>
          </div>
          <div class="quadrant-card">
            <span class="card-header-badge badge-why">2. VÌ SAO MÌNH LÀM VẬY? (MOTIVATION)</span>
            <div class="card-body">${Array.isArray(why) ? list(why) : why}</div>
          </div>
          <div class="quadrant-card">
            <span class="card-header-badge badge-result">3. KẾT QUẢ TRẢ VỀ LÀ GÌ? (RESULTS)</span>
            <div class="card-body">${Array.isArray(results) ? list(results) : results}</div>
          </div>
          <div class="quadrant-card">
            <span class="card-header-badge badge-good">4. CÁI GÌ TỐT, CÁI GÌ XẤU? (VERDICT)</span>
            <div class="card-body">
              <div class="verdict-mini-section good">
                <strong class="verdict-label">[+] ĐIỂM TỐT:</strong>
                ${Array.isArray(good) ? list(good) : good}
              </div>
              <div class="verdict-mini-section bad">
                <strong class="verdict-label">[-] ĐIỂM XẤU:</strong>
                ${Array.isArray(bad) ? list(bad) : bad}
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.QuadrantCardComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
