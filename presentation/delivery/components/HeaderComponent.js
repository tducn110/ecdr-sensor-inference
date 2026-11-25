/* ECDR Component: HeaderComponent.js
 * Sticky Site Header & Global Navigation
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render({ activeWeek = 4, weeks = {}, isOverview = false }) {
    const weekLabel = w => `Tuần ${w}`;
    return `
      <header class="site-header" role="banner">
        <div class="brand-group">
          <a class="brand" href="index.html">
            <span class="brand-badge">ECDR</span>
            <span class="brand-title">Research Delivery</span>
          </a>
          <span class="edition-pill">Component Architecture</span>
        </div>
        <nav class="week-nav" aria-label="Lựa chọn tuần nghiên cứu">
          ${Object.keys(weeks).map(w => {
            const isActive = !isOverview && Number(w) === Number(activeWeek);
            return `
              <button type="button" 
                      class="week-tab ${isActive ? 'active' : ''}" 
                      data-week="${w}"
                      aria-pressed="${isActive}">
                ${weekLabel(w)}
              </button>
            `;
          }).join('')}
          <button type="button" 
                  class="week-tab ${isOverview ? 'active' : ''}" 
                  id="tab-overview" 
                  data-overview="true"
                  aria-pressed="${isOverview}">
            Tổng quan (Overview)
          </button>
        </nav>
        <div class="header-actions">
          <button id="open-deck" class="primary present-btn" type="button">
            <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/></svg>
            ${isOverview ? 'Trình chiếu Tổng quan' : `Trình chiếu Tuần ${activeWeek}`}
          </button>
        </div>
      </header>
    `;
  }

  global.ECDR_COMPONENTS.HeaderComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);
