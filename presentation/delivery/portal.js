/* ==============================================================================
 * ECDR RESEARCH DELIVERY PORTAL — MASTER ORCHESTRATOR
 * Integrates modular components (Header, Hero, DataTable, QuadrantCard, AttachmentHub, SlideModal)
 * and pages (Week 1, Week 2, Week 3, Week 4, Overview).
 * (Completely removes all speaker notes / meta annotations)
 * ============================================================================== */

(function () {
  'use strict';

  const global = typeof window !== 'undefined' ? window : globalThis;
  const C = global.ECDR_COMPONENTS || {};
  const P = global.ECDR_PAGES || {};

  const escape = value => String(value).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));

  const slide = (id, title, subtitle, content, extra = {}) => ({
    id, title, subtitle, content, notes: '', ...extra
  });

  // Ensure all pages are loaded; provide fallback if running standalone
  const weeks = {};
  if (P.week1) weeks[1] = P.week1;
  if (P.week2) weeks[2] = P.week2;
  if (P.week3) weeks[3] = P.week3;
  if (P.week4) weeks[4] = P.week4;

  const journeyLabels = {
    1: 'Problem & Frozen Reference',
    2: 'Data & CNN Architecture',
    3: 'Theoretical Hypotheses',
    4: 'Evaluation, Mechanics & Scaling'
  };

  // Build agenda and validate slide mappings for each week
  for (const [number, week] of Object.entries(weeks)) {
    const w = Number(number);
    const assigned = new Set();
    for (const section of week.sections) {
      for (const id of section.slides) {
        const item = week.slides.find(s => s.id === id);
        if (!item || assigned.has(id)) {
          continue;
        }
        item.week = w;
        item.sectionId = section.id;
        assigned.add(id);
      }
    }

    // Insert Agenda slide at the beginning of each week if not already present
    if (!week.slides.some(s => s.id === `w${w}-agenda`)) {
      const journey = C.JourneyStrip ? C.JourneyStrip(w) : `<ol class="journey-strip">${Object.keys(weeks).map(n => `<li ${Number(n) === w ? 'aria-current="step"' : ''}>Tuần ${n}: ${journeyLabels[n]}</li>`).join('')}</ol>`;
      const agenda = C.AgendaList ? C.AgendaList(w, week.sections, week.slides) : `<ol class="agenda-list">${week.sections.map(s => `<li>${s.label}</li>`).join('')}</ol>`;
      
      week.slides.unshift(slide(
        `w${w}-agenda`,
        `Tổng quan Tuần ${w}`,
        `Các chuyên đề cốt lõi trong Tuần ${w}`,
        journey + agenda,
        { kind: 'agenda', week: w, sectionId: null }
      ));
    }
  }

  const overviewIds = (P.overview && P.overview.overviewIds) || [
    'w1-agenda', 'w1-opening', 'w1-full-inference', 'w1-freeze', 'w1-quadrant',
    'w2-agenda', 'w2-data', 'w2-tensor', 'w2-counts', 'w2-cnn', 'w2-quadrant',
    'w3-agenda', 'w3-map', 'w3-h05', 'w3-h13', 'w3-h16', 'w3-h24', 'w3-quadrant',
    'w4-agenda', 'w4-g1-baseline', 'w4-g2-incremental', 'w4-g3-certified', 'w4-g4-domain',
    'w4-g5-cost', 'w4-g6-crossover', 'w4-conclusion', 'w4-app-reproducibility-hub'
  ];

  const model = { weeks, overviewIds };
  global.ECDR_DELIVERY = model;
  if (typeof document === 'undefined') return;

  const root = document.getElementById('app');
  if (!root) return;

  let activeWeek = Number(document.body.dataset.startWeek || 4);
  let isOverview = document.body.dataset.overview === 'true';
  let slideWeek = activeWeek;
  let slideIndex = 0;
  let previousFocus;
  let presentingOverview = isOverview;

  const getOverviewDeck = () => overviewIds
    .map(id => Object.values(weeks).flatMap(w => w.slides).find(s => s.id === id))
    .filter(Boolean);

  const getDeck = () => presentingOverview ? getOverviewDeck() : (weeks[slideWeek]?.slides || []);

  const weekLabel = w => `Tuần ${w}`;

  function renderArticle(s, index, w) {
    return `
      <article class="portal-section" id="${s.id}">
        <header class="section-heading">
          <div class="section-index">${String(index + 1).padStart(2, '0')}</div>
          <div>
            <p class="eyebrow">${weekLabel(w)}${s.focus ? ` / Tiêu điểm ${s.focus}` : ''}</p>
            <h2>${s.title}</h2>
            <p class="subtitle">${s.subtitle}</p>
          </div>
          <button class="quiet open-slide" type="button" data-week="${w}" data-index="${index}">
            Trình chiếu
          </button>
        </header>
        <div class="content">${s.content}</div>
      </article>
    `;
  }

  function initApp() {
    const currentWeekData = weeks[activeWeek] || weeks[4] || { title: '', subtitle: '', sections: [], slides: [] };

    const headerHtml = C.HeaderComponent ? C.HeaderComponent.render({ activeWeek, weeks, isOverview }) :
      (C.Header ? C.Header({ activeWeek, weeks, isOverview }) : '');

    const heroHtml = C.HeroComponent ? C.HeroComponent.render({
      weekNumber: activeWeek,
      title: isOverview ? 'Tổng quan Nghiên cứu ECDR: Tuần 1–4' : `Tuần ${activeWeek}: ${currentWeekData.title}`,
      subtitle: isOverview ? 'Khung tổng hợp điều hành toàn bộ 4 tuần nghiên cứu ECDR.' : currentWeekData.subtitle,
      sectionCount: currentWeekData.sections.length,
      slideCount: currentWeekData.slides.length,
      isOverview
    }) : (C.Hero ? C.Hero({
      weekNumber: activeWeek,
      title: isOverview ? 'Tổng quan Nghiên cứu ECDR: Tuần 1–4' : `Tuần ${activeWeek}: ${currentWeekData.title}`,
      subtitle: isOverview ? 'Khung tổng hợp điều hành toàn bộ 4 tuần nghiên cứu ECDR.' : currentWeekData.subtitle,
      sectionCount: currentWeekData.sections.length,
      slideCount: currentWeekData.slides.length,
      isOverview
    }) : '');

    const modalHtml = C.SlideModalComponent ? C.SlideModalComponent.render(weeks) :
      (C.SlideModal ? C.SlideModal(weeks) : '');

    root.innerHTML = `
      ${headerHtml}
      <div id="hero-mount">
        ${heroHtml}
      </div>
      <main id="main-content">
        ${Object.entries(weeks).map(([w, week]) => `
          <section id="content-week-${w}" class="week-content" aria-label="${weekLabel(w)}" ${Number(w) !== activeWeek ? 'hidden' : ''}>
            <nav class="contents" aria-label="Mục lục ${weekLabel(w)}">
              ${week.slides.map((s, i) => `
                <a href="#${s.id}" title="${escape(s.title)}">
                  ${String(i + 1).padStart(2, '0')} ${escape(s.title)}
                </a>
              `).join('')}
            </nav>
            <div class="portal-body">
              ${week.slides.map((s, i) => renderArticle(s, i, w)).join('')}
            </div>
          </section>
        `).join('')}
      </main>
      <footer class="site-footer">
        Event-Conditioned Certified Decision Reuse (ECDR) · Component-Based Architecture · 2026
      </footer>
      ${modalHtml}
    `;

    bindEvents();
    renderMath(root);
    window.addEventListener('load', () => renderMath(document.body));
  }

  function switchWeek(w) {
    isOverview = false;
    presentingOverview = false;
    activeWeek = Number(w);
    if (!weeks[activeWeek]) activeWeek = 4;

    document.querySelectorAll('.week-content').forEach(el => {
      el.hidden = el.id !== `content-week-${activeWeek}`;
    });

    document.querySelectorAll('.week-tab').forEach(el => {
      el.classList.toggle('active', Number(el.dataset.week) === activeWeek);
      el.setAttribute('aria-pressed', String(Number(el.dataset.week) === activeWeek));
    });

    const ovTab = document.getElementById('tab-overview');
    if (ovTab) {
      ovTab.classList.remove('active');
      ovTab.setAttribute('aria-pressed', 'false');
    }

    const currentWeekData = weeks[activeWeek];
    const heroMount = document.getElementById('hero-mount');
    if (heroMount) {
      const renderHero = C.HeroComponent ? C.HeroComponent.render : C.Hero;
      if (renderHero) {
        heroMount.innerHTML = renderHero({
          weekNumber: activeWeek,
          title: `Tuần ${activeWeek}: ${currentWeekData.title}`,
          subtitle: currentWeekData.subtitle,
          sectionCount: currentWeekData.sections.length,
          slideCount: currentWeekData.slides.length,
          isOverview: false
        });
      }
    }

    const openBtn = document.getElementById('open-deck');
    if (openBtn) openBtn.textContent = `Trình chiếu Tuần ${activeWeek}`;
  }

  function switchOverview() {
    isOverview = true;
    presentingOverview = true;
    document.querySelectorAll('.week-tab').forEach(el => {
      el.classList.remove('active');
      el.setAttribute('aria-pressed', 'false');
    });

    const ovTab = document.getElementById('tab-overview');
    if (ovTab) {
      ovTab.classList.add('active');
      ovTab.setAttribute('aria-pressed', 'true');
    }

    const heroMount = document.getElementById('hero-mount');
    if (heroMount) {
      const renderHero = C.HeroComponent ? C.HeroComponent.render : C.Hero;
      if (renderHero) {
        heroMount.innerHTML = renderHero({
          weekNumber: 4,
          title: 'Tổng quan Nghiên cứu ECDR: Tuần 1–4',
          subtitle: 'Khung tổng hợp điều hành, lộ trình nghiên cứu và các bước đột phá mở rộng.',
          sectionCount: 4,
          slideCount: overviewIds.length,
          isOverview: true
        });
      }
    }

    const openBtn = document.getElementById('open-deck');
    if (openBtn) openBtn.textContent = `Trình chiếu Tổng quan (${overviewIds.length} slide)`;
  }

  function renderSectionTracker(s, deck) {
    const trackerLoc = document.getElementById('modal-section-location');
    const trackerTrail = document.getElementById('modal-section-trail');
    if (!trackerLoc || !trackerTrail) return;

    const sections = weeks[s.week]?.sections || [];
    const sectionIndex = sections.findIndex(sec => sec.id === s.sectionId);
    const section = sections[sectionIndex];
    const inSection = deck.filter(item => item.week === s.week && item.sectionId === s.sectionId);

    const location = section
      ? `Tuần ${s.week} / Chuyên đề ${sectionIndex + 1}/${sections.length}: ${section.label} / Slide ${inSection.findIndex(item => item.id === s.id) + 1}/${inSection.length} trong phần này`
      : `Tuần ${s.week} / Tổng quan lộ trình (${sections.length} chuyên đề)`;

    trackerLoc.textContent = location;
    trackerTrail.innerHTML = `
      <button type="button" data-agenda-week="${s.week}" ${s.kind === 'agenda' ? 'aria-current="step"' : ''}>Lộ trình tuần</button>
      ${sections.map((item, i) => {
        const state = i === sectionIndex ? 'current' : sectionIndex >= 0 && i < sectionIndex ? 'earlier' : 'upcoming';
        return `
          <button type="button" data-section-week="${s.week}" data-section-id="${item.id}" data-state="${state}" ${state === 'current' ? 'aria-current="step"' : ''}>
            <span>${String(i + 1).padStart(2, '0')}</span> ${escape(item.label)}
          </button>
        `;
      }).join('')}
    `;
  }

  function renderFullSlide() {
    const deck = getDeck();
    if (!deck.length) return;
    slideIndex = Math.max(0, Math.min(slideIndex, deck.length - 1));
    const s = deck[slideIndex];
    slideWeek = s.week;

    const counter = document.getElementById('modal-slide-counter');
    if (counter) counter.textContent = `${String(slideIndex + 1).padStart(2, '0')} / ${String(deck.length).padStart(2, '0')}`;

    const badge = document.getElementById('modal-slide-badge');
    if (badge) badge.textContent = `${weekLabel(slideWeek)}${presentingOverview ? ' / Tuyển chọn Tổng quan' : ''}${s.focus ? ` / Tiêu điểm ${s.focus}` : ''}`;

    const title = document.getElementById('modal-slide-title');
    if (title) title.textContent = s.title;

    const subtitle = document.getElementById('modal-slide-subtitle');
    if (subtitle) subtitle.textContent = s.subtitle;

    const body = document.getElementById('modal-slide-body');
    if (body) body.innerHTML = s.content;

    const scrollBox = document.getElementById('slide-scroll');
    if (scrollBox) scrollBox.scrollTop = 0;

    const prevBtn = document.getElementById('prev-slide');
    if (prevBtn) prevBtn.disabled = slideIndex === 0;

    const nextBtn = document.getElementById('next-slide');
    if (nextBtn) nextBtn.disabled = slideIndex === deck.length - 1;

    document.querySelectorAll('.deck-week').forEach(el => {
      el.setAttribute('aria-pressed', String(Number(el.dataset.week) === slideWeek));
    });

    renderSectionTracker(s, deck);

    const pills = document.getElementById('modal-slide-pills');
    if (pills) {
      pills.innerHTML = deck.map((item, i) => `
        <button type="button" data-index="${i}" title="${escape(item.title)}" ${i === slideIndex ? 'aria-current="step"' : ''}>
          ${i + 1}
        </button>
      `).join('');
      document.querySelector('#modal-slide-pills [aria-current="step"]')?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    }

    renderMath(document.getElementById('slide-scroll'));
  }

  function renderMath(target) {
    const el = target || document.body;
    const renderFn = global.renderMathInElement || (typeof window !== 'undefined' ? window.renderMathInElement : null);
    if (typeof renderFn === 'function' && el) {
      try {
        renderFn(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
            { left: '$', right: '$', display: false }
          ],
          throwOnError: false,
          strict: false
        });
      } catch (err) {
        console.warn('Math rendering skipped:', err);
      }
    }
  }

  function openModal(w = activeWeek, n = 1, useOverview = presentingOverview) {
    const modal = document.getElementById('full-slide-modal');
    if (!modal) return;
    presentingOverview = useOverview;
    slideWeek = Number(w);
    if (!weeks[slideWeek]) slideWeek = 4;
    slideIndex = Number(n) - 1;
    previousFocus = document.activeElement;
    renderFullSlide();
    if (!modal.open) modal.showModal();
    document.body.classList.add('presenting');
  }

  function closeModal() {
    const modal = document.getElementById('full-slide-modal');
    if (!modal) return;
    if (document.fullscreenElement && document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
    modal.close();
    document.body.classList.remove('presenting');
    previousFocus?.focus?.();
  }

  function moveSlide(delta) {
    slideIndex += delta;
    renderFullSlide();
  }

  function jumpToSection(w, sectionId) {
    const drawer = document.getElementById('deck-menu-drawer');
    if (drawer) drawer.hidden = true;
    const section = weeks[w]?.sections.find(item => item.id === sectionId);
    if (!section) return;
    const index = weeks[w].slides.findIndex(item => item.id === section.slides[0]);
    const modal = document.getElementById('full-slide-modal');
    if (!modal?.open) {
      openModal(w, index + 1, false);
    } else {
      presentingOverview = false;
      slideWeek = w;
      slideIndex = index;
      renderFullSlide();
    }
  }

  function bindEvents() {
    // Drawer toggling
    document.getElementById('toggle-deck-menu')?.addEventListener('click', () => {
      const drawer = document.getElementById('deck-menu-drawer');
      if (drawer) drawer.hidden = !drawer.hidden;
    });

    document.getElementById('close-deck-menu')?.addEventListener('click', () => {
      const drawer = document.getElementById('deck-menu-drawer');
      if (drawer) drawer.hidden = true;
    });

    // Week tab switching
    document.querySelectorAll('.week-tab[data-week]').forEach(btn => {
      btn.addEventListener('click', () => switchWeek(btn.dataset.week));
    });

    const ovTab = document.getElementById('tab-overview');
    if (ovTab) ovTab.addEventListener('click', switchOverview);

    // Presentation triggers
    document.getElementById('open-deck')?.addEventListener('click', () => {
      openModal(activeWeek, 1, isOverview);
    });

    document.querySelectorAll('.open-slide').forEach(btn => {
      btn.addEventListener('click', () => {
        openModal(Number(btn.dataset.week), Number(btn.dataset.index) + 1, false);
      });
    });

    // Modal navigation
    document.getElementById('close-deck')?.addEventListener('click', closeModal);
    document.getElementById('prev-slide')?.addEventListener('click', () => moveSlide(-1));
    document.getElementById('next-slide')?.addEventListener('click', () => moveSlide(1));

    document.getElementById('fullscreen-toggle')?.addEventListener('click', async () => {
      try {
        const modal = document.getElementById('full-slide-modal');
        if (document.fullscreenElement) {
          await document.exitFullscreen();
        } else if (modal) {
          await modal.requestFullscreen();
        }
      } catch (err) {
        console.warn('Fullscreen toggle failed:', err);
      }
    });

    document.querySelectorAll('.deck-week').forEach(btn => {
      btn.addEventListener('click', () => {
        presentingOverview = false;
        slideWeek = Number(btn.dataset.week);
        slideIndex = 0;
        const drawer = document.getElementById('deck-menu-drawer');
        if (drawer) drawer.hidden = true;
        renderFullSlide();
      });
    });

    document.getElementById('modal-slide-pills')?.addEventListener('click', e => {
      const btn = e.target.closest('button[data-index]');
      if (btn) {
        slideIndex = Number(btn.dataset.index);
        renderFullSlide();
      }
    });

    root.addEventListener('click', e => {
      const secBtn = e.target.closest('button[data-section-id]');
      if (secBtn) {
        jumpToSection(Number(secBtn.dataset.sectionWeek), secBtn.dataset.sectionId);
      }
      const agendaBtn = e.target.closest('button[data-agenda-week]');
      if (agendaBtn) {
        const w = Number(agendaBtn.dataset.agendaWeek);
        const deck = getDeck();
        const idx = deck.findIndex(item => item.id === `w${w}-agenda`);
        if (idx >= 0) {
          slideIndex = idx;
          renderFullSlide();
        }
      }

      // AttachmentHub filtering
      const hubBtn = e.target.closest('.hub-tab-btn');
      if (hubBtn) {
        document.querySelectorAll('.hub-tab-btn').forEach(b => b.classList.remove('active'));
        hubBtn.classList.add('active');
        const cat = hubBtn.dataset.hubTab;
        document.querySelectorAll('.hub-card').forEach(card => {
          if (cat === 'all' || card.dataset.category === cat) {
            card.style.display = 'flex';
          } else {
            card.style.display = 'none';
          }
        });
      }
    });

    // Keyboard Shortcuts (No speaker notes toggle)
    document.addEventListener('keydown', e => {
      if (e.altKey || e.ctrlKey || e.metaKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      const modal = document.getElementById('full-slide-modal');
      const isModalOpen = modal?.open;

      if (!isModalOpen) {
        if (e.key.toLowerCase() === 'p') {
          e.preventDefault();
          openModal(activeWeek, 1, isOverview);
        }
        return;
      }

      if (['ArrowRight', 'PageDown', ' '].includes(e.key)) {
        e.preventDefault();
        moveSlide(1);
      } else if (['ArrowLeft', 'PageUp'].includes(e.key)) {
        e.preventDefault();
        moveSlide(-1);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        closeModal();
      } else if (e.key.toLowerCase() === 'f') {
        e.preventDefault();
        document.getElementById('fullscreen-toggle')?.click();
      } else if (['1', '2', '3', '4'].includes(e.key)) {
        e.preventDefault();
        presentingOverview = false;
        slideWeek = Number(e.key);
        slideIndex = 0;
        renderFullSlide();
      }
    });
  }

  // Export functions to global scope
  Object.assign(global, {
    switchWeek,
    switchOverview,
    openModal,
    closeModal,
    renderFullSlide
  });

  // Auto-init on DOMContentLoaded or immediate
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();
