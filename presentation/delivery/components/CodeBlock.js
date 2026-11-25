/* ECDR Presentation Component: CodeBlock.js
 * Renders syntax-styled terminal/code blocks with filename headers and instant copy buttons.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  function CodeBlock(text, options = {}) {
    const { title = '', language = 'bash', isTerminal = false } = options;
    const blockId = 'code-' + Math.random().toString(36).substring(2, 9);

    return `
      <div class="code-block-wrapper ${isTerminal ? 'terminal-style' : ''}">
        ${title ? `
          <div class="code-block-header">
            <span class="code-title">${escape(title)}</span>
            <span class="code-lang">${language}</span>
            <button class="copy-btn" type="button" onclick="navigator.clipboard.writeText(document.getElementById('${blockId}').innerText); this.textContent='Copied!'; setTimeout(() => this.textContent='Copy', 2000);">Copy</button>
          </div>
        ` : ''}
        <pre><code id="${blockId}" class="language-${language}">${escape(text)}</code></pre>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.CodeBlock = CodeBlock;
  global.ECDR_COMPONENTS.code = CodeBlock;
})(typeof window !== 'undefined' ? window : globalThis);
