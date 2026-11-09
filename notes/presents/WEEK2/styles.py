# -*- coding: utf-8 -*-
"""
CSS styles for ECDR Week 2 Presentation & Report.
Supports both Slide Deck Mode and Document Mode.
"""

CSS_CONTENT = """
:root {
  --bg-primary: #0a0e17;
  --bg-secondary: #111827;
  --bg-card: #1e293b;
  --bg-card-hover: #283548;
  --bg-glass: rgba(30, 41, 59, 0.7);
  --border-color: #334155;
  --border-glow: rgba(99, 102, 241, 0.3);
  
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  
  --accent-indigo: #6366f1;
  --accent-cyan: #06b6d4;
  --accent-emerald: #10b981;
  --accent-amber: #f59e0b;
  --accent-rose: #f43f5e;
  --accent-purple: #a855f7;
  
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
  
  --slide-width: 1280px;
  --slide-height: 720px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  background-color: var(--bg-primary);
  color: var(--text-main);
  line-height: 1.5;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
  background: var(--border-color);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--accent-indigo);
}

/* Header & Navigation Bar */
header.app-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(11, 15, 25, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-color);
  padding: 0.75rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.brand-section {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.brand-badge {
  background: linear-gradient(135deg, var(--accent-indigo), var(--accent-cyan));
  color: #fff;
  padding: 0.25rem 0.6rem;
  border-radius: 6px;
  font-weight: 700;
  font-size: 0.85rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.header-title-wrap h1 {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-main);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.header-subtitle {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.gates-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.gate-badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  border: 1px solid;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.gate-pass {
  background: rgba(16, 185, 129, 0.15);
  border-color: rgba(16, 185, 129, 0.4);
  color: #34d399;
}

.gate-hold {
  background: rgba(245, 158, 11, 0.15);
  border-color: rgba(245, 158, 11, 0.4);
  color: #fbbf24;
}

.gate-target {
  background: rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.4);
  color: #818cf8;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.btn {
  background: var(--bg-card);
  color: var(--text-main);
  border: 1px solid var(--border-color);
  padding: 0.45rem 0.85rem;
  border-radius: 6px;
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  transition: all 0.2s ease;
}

.btn:hover {
  background: var(--bg-card-hover);
  border-color: var(--accent-indigo);
  color: #fff;
}

.btn-primary {
  background: var(--accent-indigo);
  border-color: var(--accent-indigo);
  color: #fff;
}

.btn-primary:hover {
  background: #4f46e5;
}

.btn-active {
  background: rgba(99, 102, 241, 0.25);
  border-color: var(--accent-indigo);
  color: #818cf8;
}

/* Progress bar */
.slide-progress-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent-indigo), var(--accent-cyan));
  transition: width 0.3s ease;
  z-index: 101;
}

/* ========================================================================= */
/* MODE: PRESENTATION DECK                                                    */
/* ========================================================================= */
body.mode-presentation {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

body.mode-presentation .doc-sidebar {
  display: none !important;
}

body.mode-presentation .doc-container {
  display: none !important;
}

.deck-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1.5rem 1rem 4rem 1rem;
  position: relative;
}

.slide-frame {
  width: 100%;
  max-width: var(--slide-width);
  min-height: var(--slide-height);
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 30px var(--border-glow);
  display: none;
  flex-direction: column;
  overflow: hidden;
  position: relative;
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.slide-frame.active-slide {
  display: flex;
  animation: fadeInSlide 0.35s ease forwards;
}

@keyframes fadeInSlide {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.slide-header {
  padding: 1.25rem 2rem 0.85rem 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, transparent 100%);
}

.slide-part-tag {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent-cyan);
  margin-bottom: 0.25rem;
  display: inline-block;
}

.slide-title {
  font-size: 1.55rem;
  font-weight: 800;
  color: #fff;
  line-height: 1.2;
}

.slide-subtitle {
  font-size: 0.88rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

.slide-meta-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.35rem;
}

.slide-num-pill {
  font-family: var(--font-mono);
  font-size: 0.82rem;
  font-weight: 600;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  padding: 0.2rem 0.6rem;
  border-radius: 20px;
  color: var(--accent-indigo);
}

.slide-body {
  flex: 1;
  padding: 1.5rem 2rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

/* Slide footer / Speaker notes toggle */
.slide-footer {
  padding: 0.75rem 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.07);
  background: rgba(17, 24, 39, 0.95);
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--text-muted);
}

/* Speaker Notes Drawer */
.speaker-notes-drawer {
  background: #0d131f;
  border-top: 2px solid var(--accent-indigo);
  padding: 1.2rem 2rem;
  display: none;
  animation: slideUpNotes 0.25s ease;
}

.speaker-notes-drawer.open-notes {
  display: block;
}

@keyframes slideUpNotes {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.notes-header {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--accent-indigo);
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.notes-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  font-size: 0.84rem;
  line-height: 1.55;
  color: #cbd5e1;
}

.notes-block h5 {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--accent-cyan);
  margin-bottom: 0.35rem;
}

/* Deck Controls at Bottom */
.deck-controls {
  position: fixed;
  bottom: 1rem;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(15, 23, 42, 0.92);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-color);
  padding: 0.45rem 1rem;
  border-radius: 40px;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  z-index: 99;
}

.slide-selector {
  background: var(--bg-secondary);
  color: var(--text-main);
  border: 1px solid var(--border-color);
  padding: 0.3rem 0.6rem;
  border-radius: 6px;
  font-size: 0.82rem;
  font-family: var(--font-sans);
  outline: none;
  cursor: pointer;
}

.slide-selector:focus {
  border-color: var(--accent-indigo);
}

/* ========================================================================= */
/* MODE: DOCUMENT / READING                                                   */
/* ========================================================================= */
body.mode-document {
  display: flex;
  flex-direction: column;
}

body.mode-document .deck-container {
  display: none !important;
}

body.mode-document .deck-controls {
  display: none !important;
}

.doc-layout {
  display: flex;
  max-width: 1440px;
  margin: 0 auto;
  width: 100%;
}

.doc-sidebar {
  width: 280px;
  position: sticky;
  top: 56px;
  height: calc(100vh - 56px);
  overflow-y: auto;
  padding: 1.5rem 1rem;
  border-right: 1px solid var(--border-color);
  background: var(--bg-primary);
  font-size: 0.84rem;
}

.doc-sidebar h3 {
  font-size: 0.75rem;
  text-transform: uppercase;
  color: var(--text-dim);
  letter-spacing: 0.08em;
  margin-bottom: 0.75rem;
  padding-left: 0.5rem;
}

.doc-nav-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.doc-nav-link {
  display: block;
  padding: 0.4rem 0.6rem;
  color: var(--text-muted);
  text-decoration: none;
  border-radius: 6px;
  transition: all 0.2s;
  font-weight: 500;
}

.doc-nav-link:hover, .doc-nav-link.active-nav {
  background: var(--bg-card);
  color: #fff;
  border-left: 3px solid var(--accent-indigo);
}

.doc-container {
  flex: 1;
  padding: 2.5rem 3rem 6rem 3rem;
  max-width: 1100px;
}

.doc-slide-section {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  margin-bottom: 2.5rem;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* ========================================================================= */
/* SHARED UI COMPONENTS (Cards, Grids, Callouts, Tables)                      */
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

.card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 1.1rem;
  position: relative;
}

.card-accent-indigo {
  border-top: 3px solid var(--accent-indigo);
}

.card-accent-cyan {
  border-top: 3px solid var(--accent-cyan);
}

.card-accent-emerald {
  border-top: 3px solid var(--accent-emerald);
}

.card-accent-amber {
  border-top: 3px solid var(--accent-amber);
}

.card-accent-rose {
  border-top: 3px solid var(--accent-rose);
}

.stat-value {
  font-size: 1.85rem;
  font-weight: 800;
  color: #fff;
  font-family: var(--font-mono);
  line-height: 1.1;
}

.stat-label {
  font-size: 0.78rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 0.25rem;
}

.stat-desc {
  font-size: 0.8rem;
  color: #cbd5e1;
  margin-top: 0.5rem;
  line-height: 1.4;
}

/* Callout block */
.callout {
  padding: 1rem 1.25rem;
  border-radius: 8px;
  border-left: 4px solid;
  background: rgba(30, 41, 59, 0.5);
  font-size: 0.88rem;
  line-height: 1.5;
}

.callout-info {
  border-color: var(--accent-indigo);
  background: rgba(99, 102, 241, 0.08);
  color: #e0e7ff;
}

.callout-warning {
  border-color: var(--accent-amber);
  background: rgba(245, 158, 11, 0.08);
  color: #fef3c7;
}

.callout-success {
  border-color: var(--accent-emerald);
  background: rgba(16, 185, 129, 0.08);
  color: #d1fae5;
}

.callout-danger {
  border-color: var(--accent-rose);
  background: rgba(244, 63, 94, 0.08);
  color: #ffe4e6;
}

.callout-title {
  font-weight: 700;
  margin-bottom: 0.25rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

/* Blockquote problem statement */
.problem-box {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(6, 182, 212, 0.12));
  border: 1px solid rgba(99, 102, 241, 0.35);
  border-radius: 10px;
  padding: 1.25rem 1.75rem;
  font-size: 1.05rem;
  font-weight: 600;
  color: #fff;
  text-align: center;
  margin: 0.75rem 0;
  box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
}

/* Tables */
table.custom-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
  text-align: left;
}

table.custom-table th {
  background: #172033;
  color: var(--accent-cyan);
  font-weight: 600;
  padding: 0.6rem 0.8rem;
  border-bottom: 2px solid var(--border-color);
  text-transform: uppercase;
  font-size: 0.74rem;
  letter-spacing: 0.05em;
}

table.custom-table td {
  padding: 0.55rem 0.8rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  color: #cbd5e1;
}

table.custom-table tr:hover td {
  background: rgba(255, 255, 255, 0.03);
}

table.custom-table tr:last-child td {
  border-bottom: none;
}

/* Code & Monospace Badges */
code, .mono {
  font-family: var(--font-mono);
  font-size: 0.82em;
  background: rgba(0, 0, 0, 0.35);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  color: #38bdf8;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Formula Box */
.formula-box {
  background: #0f172a;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 1rem 1.5rem;
  text-align: center;
  margin: 0.85rem 0;
}

/* Diagram Flowchart in CSS/HTML */
.flow-diagram {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  background: #0f172a;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 1rem;
  margin: 0.5rem 0;
}

.flow-step {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 0.75rem 0.5rem;
  text-align: center;
}

.flow-step-title {
  font-size: 0.78rem;
  font-weight: 700;
  color: #fff;
}

.flow-step-desc {
  font-size: 0.7rem;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

.flow-arrow {
  color: var(--accent-indigo);
  font-size: 1.2rem;
  font-weight: 700;
}

/* Images & Zoom Modal */
.slide-img-container {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
  background: #0f172a;
  cursor: zoom-in;
  transition: transform 0.2s;
  position: relative;
}

.slide-img-container:hover {
  border-color: var(--accent-indigo);
  transform: scale(1.01);
}

.slide-img-container img {
  width: 100%;
  height: auto;
  display: block;
}

.img-caption {
  padding: 0.4rem 0.8rem;
  font-size: 0.74rem;
  color: var(--text-muted);
  background: rgba(15, 23, 42, 0.8);
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  text-align: center;
}

/* Lightbox Modal */
#img-modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.88);
  backdrop-filter: blur(8px);
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
  border-radius: 8px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
  border: 1px solid var(--border-color);
}

#modal-close-btn {
  position: absolute;
  top: 1.5rem;
  right: 2rem;
  background: var(--bg-card);
  color: #fff;
  border: 1px solid var(--border-color);
  width: 40px;
  height: 40px;
  border-radius: 50%;
  font-size: 1.2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Accordion for Appendix */
.accordion-item {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  margin-bottom: 0.75rem;
  overflow: hidden;
}

.accordion-header {
  padding: 0.85rem 1.25rem;
  background: #172033;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 600;
  font-size: 0.88rem;
  color: #fff;
  user-select: none;
}

.accordion-header:hover {
  background: #1e293b;
  color: var(--accent-cyan);
}

.accordion-icon {
  transition: transform 0.2s;
  font-size: 0.8rem;
  color: var(--accent-indigo);
}

.accordion-item.open .accordion-icon {
  transform: rotate(180deg);
}

.accordion-content {
  padding: 1.25rem;
  display: none;
  font-size: 0.84rem;
  line-height: 1.6;
  border-top: 1px solid var(--border-color);
  background: var(--bg-secondary);
}

.accordion-item.open .accordion-content {
  display: block;
}

/* Media Queries */
@media (max-width: 1024px) {
  :root {
    --slide-width: 95vw;
    --slide-height: auto;
  }
  .grid-2, .grid-3, .grid-4 {
    grid-template-columns: 1fr;
  }
  .notes-content {
    grid-template-columns: 1fr;
  }
}
"""
