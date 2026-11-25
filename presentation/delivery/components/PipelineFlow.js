/* ECDR Presentation Component: PipelineFlow.js
 * Renders step-by-step data, math, and decision pipelines horizontally or vertically.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function PipelineFlow(items, vertical = false, options = {}) {
    const { numbered = false } = options;
    return `
      <ol class="flow ${vertical ? 'vertical' : 'horizontal'}" role="list">
        ${items.map((x, idx) => `
          <li class="flow-step">
            ${numbered ? `<span class="flow-num">${idx + 1}</span>` : ''}
            <span class="flow-content">${x}</span>
          </li>
        `).join('')}
      </ol>
    `;
  }

  global.ECDR_COMPONENTS.PipelineFlow = PipelineFlow;
  global.ECDR_COMPONENTS.flow = PipelineFlow;
})(typeof window !== 'undefined' ? window : globalThis);
