/* ECDR Presentation Component: SectionTracker.js
 * Section Tracker & Agenda Navigation
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  function JourneyStrip(activeWeek = 4) {
    const journeyLabels = {
      1: 'Problem & Reference',
      2: 'Data & CNN Architecture',
      3: 'Hypotheses & Mechanics',
      4: 'Evaluation, Mechanics & Scaling'
    };

    return `
      <ol class="journey-strip" aria-label="Lộ trình 4 tuần nghiên cứu">
        ${[1, 2, 3, 4].map(n => `
          <li ${Number(n) === Number(activeWeek) ? 'aria-current="step"' : ''}>
            <span class="journey-step-num">Tuần ${n}</span>
            <span class="journey-step-label">${journeyLabels[n]}</span>
          </li>
        `).join('')}
      </ol>
    `;
  }

  function AgendaList(weekNumber, sections, slides) {
    return `
      <ol class="agenda-list" aria-label="Mục lục chuyên đề Tuần ${weekNumber}">
        ${sections.map((section, i) => {
          const first = slides.findIndex(s => s.id === section.slides[0]) + 2;
          const last = first + section.slides.length - 1;
          const range = first === last ? `Slide ${first}` : `Slide ${first}–${last}`;
          return `
            <li>
              <button type="button" class="agenda-btn" data-section-week="${weekNumber}" data-section-id="${section.id}" title="Mở chuyên đề ${escape(section.label)}">
                <span class="agenda-number">${String(i + 1).padStart(2, '0')}</span>
                <span class="agenda-topic">
                  <strong>${escape(section.label)}</strong>
                  <span>${escape(section.description)}</span>
                </span>
                <span class="agenda-range">${range}</span>
              </button>
            </li>
          `;
        }).join('')}
      </ol>
    `;
  }

  global.ECDR_COMPONENTS.JourneyStrip = JourneyStrip;
  global.ECDR_COMPONENTS.AgendaList = AgendaList;
})(typeof window !== 'undefined' ? window : globalThis);
