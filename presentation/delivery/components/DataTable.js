/* ECDR Presentation Component: DataTable.js
 * Renders high-end, responsive data tables with tabular numerals, sticky headers, and status badges.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  function formatCell(val) {
    if (typeof val !== 'string') return String(val);
    // Automatic badge rendering for standard statuses
    if (val === 'PASS' || val === 'PROFITABLE' || val === 'HIGHLY PROFITABLE' || val === 'VERIFIED_FINITE' || val === 'DERIVED' || val === 'DERIVED_REAL_ARITHMETIC') {
      return `<span class="badge badge-success">${val}</span>`;
    }
    if (val === 'FAIL' || val === 'UNPROFITABLE' || val === 'FALSIFIED_UNCONDITIONAL' || val === 'FALSIFIED_COMPLETENESS') {
      return `<span class="badge badge-danger">${val}</span>`;
    }
    if (val === 'CONDITIONAL_THEOREM' || val === 'HOLD' || val === 'HOLD_CONDITIONAL') {
      return `<span class="badge badge-warning">${val}</span>`;
    }
    if (val === 'NOT_ESTABLISHED' || val === 'AUDITED_UNSUPPORTED') {
      return `<span class="badge badge-neutral">${val}</span>`;
    }
    return val;
  }

  function DataTable(headers, rows, options = {}) {
    const { caption = '', isCompact = false } = options;
    const tableClass = isCompact ? 'data-table compact' : 'data-table';
    return `
      <div class="table-wrap">
        <table class="${tableClass}">
          ${caption ? `<caption>${escape(caption)}</caption>` : ''}
          <thead>
            <tr>
              ${headers.map(h => `<th scope="col">${h}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${rows.map(row => `
              <tr>
                ${row.map(cell => `<td>${formatCell(cell)}</td>`).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.DataTable = DataTable;
  global.ECDR_COMPONENTS.table = DataTable;
})(typeof window !== 'undefined' ? window : globalThis);
