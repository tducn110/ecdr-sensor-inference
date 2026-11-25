/* ECDR Presentation Component: Hero.js
 * Executive Hero Banner with Live Metrics and Clean Navigation Shortcuts
 * (No speaker notes / meta annotations)
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function Hero({ weekNumber = 4, title = '', subtitle = '', sectionCount = 0, slideCount = 0, isOverview = false }) {
    const eyebrow = isOverview ? 'Khung tổng hợp điều hành & Lộ trình chiến lược' : `Tuần ${weekNumber} / Đánh giá thực nghiệm & Mở rộng quy mô`;

    return `
      <section class="page-intro" aria-label="Giới thiệu nội dung">
        <div class="intro-container">
          <p class="eyebrow">${eyebrow}</p>
          <h1 id="week-heading">${title}</h1>
          <p id="week-description" class="lead-description">${subtitle}</p>
          
          <div class="intro-stats-bar">
            <div class="stat-pill">
              <span class="stat-num">${sectionCount}</span>
              <span class="stat-text">Khối chuyên đề</span>
            </div>
            <div class="stat-pill">
              <span class="stat-num">5,053</span>
              <span class="stat-text">Cửa sổ kiểm thử</span>
            </div>
            <div class="stat-pill">
              <span class="stat-num">17</span>
              <span class="stat-text">Phương pháp so sánh</span>
            </div>
            <div class="stat-pill highlight-pill">
              <span class="stat-num">100%</span>
              <span class="stat-text">Bảo toàn quyết định (0 vi phạm)</span>
            </div>
          </div>

          <div class="intro-actions">
            <span class="shortcut-tag"><kbd>P</kbd> Trình chiếu toàn màn hình</span>
            <span class="shortcut-tag"><kbd>→</kbd> <kbd>←</kbd> Chuyển slide</span>
            <span class="shortcut-tag"><kbd>F</kbd> Toàn màn hình</span>
            <span class="shortcut-tag"><kbd>Esc</kbd> Đóng trình chiếu</span>
          </div>
        </div>
      </section>
    `;
  }

  global.ECDR_COMPONENTS.Hero = Hero;
  global.ECDR_COMPONENTS.HeroComponent = { render: Hero };
})(typeof window !== 'undefined' ? window : globalThis);
