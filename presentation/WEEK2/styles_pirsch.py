# -*- coding: utf-8 -*-
"""
Pirsch Analytics Design System — Light Theme (Sunlit paper notebook with highlighter swatches)
Canvas: #f8f5ed | Ink: #000000 | Graphite: #707070 | Sunshine: #ffda6e | Mint: #6ece9d
Hairline borders: 1px #000000 | Card radius: 24px | Zero drop shadows
"""

CSS_PIRSCH = """
:root {
  /* Colors */
  --color-sunshine-highlight: #ffda6e;
  --color-mint-signal: #6ece9d;
  --color-cream-paper: #f8f5ed;
  --color-cream-card: #f8f5ed;
  --color-ink: #000000;
  --color-graphite: #707070;
  --color-border-hairline: #000000;
  --color-border-muted: #d8d3c5;
  --color-white: #ffffff;

  /* Typography */
  --font-dm-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;

  /* Type Scale */
  --text-caption: 14px;
  --text-body-sm: 15px;
  --text-body: 17px;
  --text-subheading: 20px;
  --text-heading-sm: 24px;
  --text-heading: 28px;
  --text-display: 48px;

  /* Spacing */
  --spacing-8: 8px;
  --spacing-16: 16px;
  --spacing-24: 24px;
  --spacing-32: 32px;
  --spacing-48: 48px;
  --spacing-64: 64px;

  /* Radii */
  --radius-tags: 9999px;
  --radius-cards: 24px;
  --radius-images: 12px;
  --radius-inputs: 6px;
  --radius-buttons: 24px;

  /* Dimensions */
  --slide-max-width: 1200px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-dm-sans);
  background-color: var(--color-cream-paper);
  color: var(--color-ink);
  line-height: 1.5;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
  font-feature-settings: "ss03", "ss04";
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 7px;
  height: 7px;
}
::-webkit-scrollbar-track {
  background: var(--color-cream-paper);
}
::-webkit-scrollbar-thumb {
  background: var(--color-border-muted);
  border-radius: 4px;
  border: 1px solid var(--color-ink);
}
::-webkit-scrollbar-thumb:hover {
  background: var(--color-sunshine-highlight);
}

/* ========================================================================= */
/* HEADER BAR                                                                */
/* ========================================================================= */
header.app-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--color-cream-paper);
  border-bottom: 1px solid var(--color-border-hairline);
  padding: 0.75rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.brand-section {
  display: flex;
  align-items: center;
  gap: 1.25rem;
}

/* Avatar Brand Tile */
.avatar-tile {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 1.1rem;
  color: var(--color-ink);
}

.header-title-wrap h1 {
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--color-ink);
  letter-spacing: -0.01em;
}

.header-subtitle {
  font-size: 0.76rem;
  color: var(--color-graphite);
}

.gates-bar {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.gate-badge {
  font-size: 0.72rem;
  font-weight: 500;
  padding: 0.25rem 0.65rem;
  border-radius: var(--radius-tags);
  border: 1px solid var(--color-border-hairline);
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  background: var(--color-cream-paper);
  color: var(--color-ink);
}

.gate-pass {
  background: var(--color-mint-signal);
  color: var(--color-ink);
}

.gate-hold {
  background: var(--color-sunshine-highlight);
  color: var(--color-ink);
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

/* Pirsch Buttons */
.btn-sunshine {
  background: var(--color-sunshine-highlight);
  color: var(--color-ink);
  border: 1px solid var(--color-border-hairline);
  padding: 0.5rem 1.2rem;
  border-radius: var(--radius-buttons);
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: transform 0.15s ease, background 0.15s ease;
  text-decoration: none;
}

.btn-sunshine:hover {
  transform: translateY(-1px);
  background: #fcd058;
}

.btn-ghost-ink {
  background: transparent;
  color: var(--color-ink);
  border: 1px solid var(--color-border-hairline);
  padding: 0.5rem 1.1rem;
  border-radius: var(--radius-buttons);
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: all 0.15s ease;
  text-decoration: none;
}

.btn-ghost-ink:hover {
  background: var(--color-white);
  transform: translateY(-1px);
}

.slide-progress-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 2px;
  background: var(--color-ink);
  transition: width 0.3s ease;
  z-index: 101;
}

/* ========================================================================= */
/* NUMBERED STEP BADGE                                                       */
/* ========================================================================= */
.step-badge {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-tags);
  background: var(--color-cream-paper);
  overflow: hidden;
  margin-bottom: 0.4rem;
}

.step-num-cell {
  width: 24px;
  height: 24px;
  background: var(--color-sunshine-highlight);
  border-right: 1px solid var(--color-border-hairline);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 500;
  color: var(--color-ink);
}

.step-name-cell {
  padding: 0.2rem 0.65rem;
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.286em;
  color: var(--color-ink);
}

/* ========================================================================= */
/* SLIDE FRAME & PRESENTATION DECK                                            */
/* ========================================================================= */
body.mode-presentation {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

body.mode-presentation .doc-layout {
  display: none !important;
}

.deck-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem 1.5rem 5rem 1.5rem;
  position: relative;
}

.slide-frame {
  width: 100%;
  max-width: var(--slide-max-width);
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  display: none;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  box-shadow: none; /* Pirsch: NO drop shadows! */
}

.slide-frame.active-slide {
  display: flex;
  animation: fadeInSlide 0.25s ease forwards;
}

@keyframes fadeInSlide {
  from { opacity: 0.8; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.slide-header {
  padding: 1.5rem 2rem 1rem 2rem;
  border-bottom: 1px solid var(--color-border-hairline);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  background: var(--color-cream-paper);
}

.slide-title {
  font-size: 1.85rem;
  font-weight: 500;
  color: var(--color-ink);
  letter-spacing: -0.016em;
  line-height: 1.25;
  margin-top: 0.2rem;
}

.slide-subtitle {
  font-size: 0.95rem;
  color: var(--color-graphite);
  margin-top: 0.35rem;
  max-width: 820px;
}

.slide-meta-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.5rem;
}

.slide-num-pill {
  font-family: var(--font-dm-sans);
  font-size: 0.85rem;
  font-weight: 500;
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  padding: 0.25rem 0.75rem;
  border-radius: var(--radius-tags);
  color: var(--color-ink);
}

.slide-body {
  flex: 1;
  padding: 1.5rem 2rem;
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  overflow-y: auto;
}

.slide-footer {
  padding: 0.85rem 2rem;
  border-top: 1px solid var(--color-border-hairline);
  background: var(--color-cream-paper);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.82rem;
  color: var(--color-graphite);
}

/* Floating Bottom Controls */
.deck-controls {
  position: fixed;
  bottom: 1.25rem;
  left: 50%;
  transform: translateX(-50%);
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  padding: 0.4rem 1rem;
  border-radius: var(--radius-tags);
  display: flex;
  align-items: center;
  gap: 0.75rem;
  z-index: 99;
}

.slide-selector {
  background: var(--color-cream-paper);
  color: var(--color-ink);
  border: 1px solid var(--color-border-hairline);
  padding: 0.35rem 0.75rem;
  border-radius: var(--radius-buttons);
  font-size: 0.82rem;
  font-family: var(--font-dm-sans);
  outline: none;
  cursor: pointer;
}

/* ========================================================================= */
/* CARDS, GRIDS, CALLOUTS & IMAGE FRAMES                                      */
/* ========================================================================= */
.grid-2 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.25rem;
}

.grid-3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
}

.grid-4 {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
}

/* Feature Card: Cream surface, 24px radius, 1px ink hairline border, no shadow */
.card {
  background: var(--color-cream-card);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  padding: 1.25rem;
  position: relative;
}

.card-white {
  background: var(--color-white);
}

.stat-value {
  font-size: 2rem;
  font-weight: 500;
  color: var(--color-ink);
  letter-spacing: -0.016em;
  line-height: 1.1;
}

.stat-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--color-graphite);
  text-transform: uppercase;
  letter-spacing: 0.286em;
  margin-top: 0.3rem;
}

.stat-desc {
  font-size: 0.85rem;
  color: var(--color-graphite);
  margin-top: 0.5rem;
  line-height: 1.45;
}

/* Sunshine Highlighter Callout */
.callout-sunshine {
  background: var(--color-sunshine-highlight);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  padding: 1rem 1.5rem;
  color: var(--color-ink);
  font-size: 0.9rem;
  line-height: 1.5;
}

.callout-mint {
  background: var(--color-mint-signal);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  padding: 0.85rem 1.25rem;
  color: var(--color-ink);
  font-size: 0.88rem;
  line-height: 1.5;
}

.callout-cream {
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  padding: 1rem 1.25rem;
  font-size: 0.88rem;
  line-height: 1.5;
}

/* Image Frame: 12px radius, 1px ink border, no shadow */
.slide-img-container {
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-images);
  overflow: hidden;
  background: var(--color-white);
  cursor: zoom-in;
  position: relative;
  transition: transform 0.15s ease;
}

.slide-img-container:hover {
  transform: translateY(-2px);
}

.slide-img-container img {
  width: 100%;
  height: auto;
  display: block;
}

.img-caption {
  padding: 0.4rem 0.8rem;
  font-size: 0.75rem;
  color: var(--color-graphite);
  background: var(--color-cream-paper);
  border-top: 1px solid var(--color-border-hairline);
  text-align: center;
}

/* Tables in Pirsch Style */
table.custom-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  text-align: left;
}

table.custom-table th {
  background: var(--color-cream-paper);
  color: var(--color-ink);
  font-weight: 500;
  padding: 0.6rem 0.8rem;
  border-bottom: 1px solid var(--color-border-hairline);
  text-transform: uppercase;
  font-size: 0.72rem;
  letter-spacing: 0.15em;
}

table.custom-table td {
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid var(--color-border-muted);
  color: var(--color-ink);
}

table.custom-table tr:hover td {
  background: rgba(255, 218, 110, 0.15); /* Sunshine highlight hover */
}

table.custom-table tr:last-child td {
  border-bottom: none;
}

.mono {
  font-family: var(--font-mono);
  font-size: 0.82em;
  background: var(--color-white);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  border: 1px solid var(--color-border-muted);
  color: var(--color-ink);
}

/* ========================================================================= */
/* DOCUMENT / READING MODE                                                    */
/* ========================================================================= */
body.mode-document .deck-container,
body.mode-document .deck-controls {
  display: none !important;
}

.doc-layout {
  display: flex;
  max-width: 1360px;
  margin: 0 auto;
  width: 100%;
}

.doc-sidebar {
  width: 300px;
  position: sticky;
  top: 60px;
  height: calc(100vh - 60px);
  overflow-y: auto;
  padding: 2rem 1.25rem;
  border-right: 1px solid var(--color-border-hairline);
  background: var(--color-cream-paper);
}

.doc-sidebar h3 {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.286em;
  color: var(--color-graphite);
  margin-bottom: 1rem;
}

.doc-nav-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.doc-nav-link {
  display: block;
  padding: 0.45rem 0.85rem;
  color: var(--color-graphite);
  text-decoration: none;
  border-radius: var(--radius-buttons);
  font-size: 0.85rem;
  font-weight: 500;
  transition: all 0.15s;
}

.doc-nav-link:hover, .doc-nav-link.active-nav {
  background: var(--color-sunshine-highlight);
  color: var(--color-ink);
  border: 1px solid var(--color-border-hairline);
}

.doc-container {
  flex: 1;
  padding: 3rem 4rem 6rem 4rem;
  max-width: 1040px;
}

.doc-slide-section {
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  margin-bottom: 3rem;
  overflow: hidden;
}

/* Lightbox Modal */
#img-modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(248, 245, 237, 0.92);
  backdrop-filter: blur(4px);
  z-index: 999;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

#img-modal.open {
  display: flex;
}

#img-modal img {
  max-width: 90vw;
  max-height: 85vh;
  border-radius: var(--radius-images);
  border: 1px solid var(--color-border-hairline);
  background: var(--color-white);
}

#modal-close-btn {
  position: absolute;
  top: 1.5rem;
  right: 2rem;
  background: var(--color-sunshine-highlight);
  color: var(--color-ink);
  border: 1px solid var(--color-border-hairline);
  width: 40px;
  height: 40px;
  border-radius: var(--radius-tags);
  font-size: 1.2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Accordion for Appendix */
.accordion-item {
  background: var(--color-cream-paper);
  border: 1px solid var(--color-border-hairline);
  border-radius: var(--radius-cards);
  margin-bottom: 0.85rem;
  overflow: hidden;
}

.accordion-header {
  padding: 1rem 1.5rem;
  background: var(--color-cream-paper);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 500;
  font-size: 0.95rem;
  color: var(--color-ink);
}

.accordion-header:hover {
  background: rgba(255, 218, 110, 0.25);
}

.accordion-content {
  padding: 1.5rem;
  display: none;
  font-size: 0.88rem;
  line-height: 1.6;
  border-top: 1px solid var(--color-border-hairline);
  background: var(--color-white);
}

.accordion-item.open .accordion-content {
  display: block;
}

@media (max-width: 1024px) {
  .grid-2, .grid-3, .grid-4 {
    grid-template-columns: 1fr;
  }
}
"""
