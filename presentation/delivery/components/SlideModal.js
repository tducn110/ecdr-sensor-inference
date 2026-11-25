/* ECDR Component: SlideModal.js
 * Full-screen Presentation Modal
 * Streamlined presentation shell: Zero clutter, viewport-fitting slides, on-demand menu
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function SlideModal(weeks = {}) {
    return `
      <dialog id="full-slide-modal" aria-labelledby="modal-slide-title">
        <div class="presentation-shell">
          <!-- Minimal Top HUD Bar -->
          <header class="deck-hud">
            <div class="hud-left">
              <button id="toggle-deck-menu" class="hud-btn menu-btn" type="button" title="Mở danh sách chuyên đề & tuần">
                <span class="hud-icon">☰</span> <span class="hud-btn-text">Chuyên đề &amp; Tuần</span>
              </button>
              <span id="modal-section-location" class="hud-location-pill" aria-live="polite"></span>
            </div>

            <div class="hud-right">
              <span id="modal-slide-counter" class="slide-counter-badge" aria-live="polite">01 / 00</span>
              <button id="fullscreen-toggle" class="hud-btn" type="button" title="Toàn màn hình (F)">
                ⛶ Toàn màn hình
              </button>
              <button id="close-deck" class="hud-btn close-btn" type="button" title="Đóng (Esc)">
                ✕ Đóng (Esc)
              </button>
            </div>
          </header>

          <!-- On-demand Navigation Drawer (Only shows when user clicks Menu button) -->
          <div id="deck-menu-drawer" class="deck-menu-drawer" hidden>
            <div class="drawer-header">
              <div class="drawer-title">
                <strong>Điều hướng Chuyên đề &amp; Tuần</strong>
                <span class="drawer-hint">Chọn tuần hoặc chuyên đề để chuyển slide nhanh</span>
              </div>
              <button id="close-deck-menu" class="hud-btn" type="button" title="Đóng menu">✕ Đóng</button>
            </div>
            <div class="drawer-body">
              <div class="drawer-group">
                <label class="drawer-label">Tuần nghiên cứu:</label>
                <nav class="deck-week-nav" aria-label="Tuần trình chiếu">
                  ${Object.keys(weeks).map(w => `
                    <button type="button" class="deck-week" data-week="${w}" title="Chuyển sang Tuần ${w}">
                      Tuần ${w}
                    </button>
                  `).join('')}
                </nav>
              </div>
              <div class="drawer-group">
                <label class="drawer-label">Chuyên đề trong tuần:</label>
                <nav id="modal-section-trail" class="section-trail" aria-label="Các phần trong tuần hiện tại"></nav>
              </div>
            </div>
          </div>

          <!-- Main Slide Canvas: Fits 100% of viewport -->
          <div id="slide-scroll" class="slide-scroll">
            <section class="slide-canvas">
              <p id="modal-slide-badge" class="eyebrow"></p>
              <h2 id="modal-slide-title"></h2>
              <p id="modal-slide-subtitle" class="subtitle"></p>
              <div id="modal-slide-body" class="content"></div>
            </section>
          </div>

          <!-- Minimal Presentation Footer -->
          <footer class="deck-footer">
            <button id="prev-slide" class="nav-slide-btn" type="button">← Trước</button>
            <div id="modal-slide-pills" class="slide-pills" aria-label="Điều hướng nhanh slide"></div>
            <button id="next-slide" class="nav-slide-btn primary" type="button">Tiếp theo →</button>
          </footer>
        </div>
      </dialog>
    `;
  }

  global.ECDR_COMPONENTS.SlideModal = SlideModal;
  global.ECDR_COMPONENTS.SlideModalComponent = { render: SlideModal };
})(typeof window !== 'undefined' ? window : globalThis);
