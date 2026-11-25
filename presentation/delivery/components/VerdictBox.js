/* ECDR Presentation Component: VerdictBox.js
 * Renders high-contrast side-by-side assessment containers for Good (+) and Bad (-) engineering findings.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  const points = items => `<ul class="points">${items.map(text => `<li>${text}</li>`).join('')}</ul>`;

  function VerdictBox(good, bad, options = {}) {
    const { goodTitle = '[+] ĐIỂM TỐT / ƯU ĐIỂM', badTitle = '[-] ĐIỂM XẤU / HẠN CHẾ & THÁCH THỨC' } = options;
    const renderList = val => Array.isArray(val) ? points(val) : val;

    return `
      <div class="verdict-box" role="region" aria-label="Đánh giá ưu và nhược điểm">
        <div class="verdict-good">
          <h4>
            <svg class="verdict-icon" viewBox="0 0 20 20" fill="currentColor" width="18" height="18"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>
            ${goodTitle}
          </h4>
          ${renderList(good)}
        </div>
        <div class="verdict-bad">
          <h4>
            <svg class="verdict-icon" viewBox="0 0 20 20" fill="currentColor" width="18" height="18"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>
            ${badTitle}
          </h4>
          ${renderList(bad)}
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.VerdictBox = VerdictBox;
  global.ECDR_COMPONENTS.verdict = VerdictBox;
})(typeof window !== 'undefined' ? window : globalThis);
