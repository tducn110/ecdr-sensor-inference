# -*- coding: utf-8 -*-
"""
Main assembler to compile all parts into /home/pro/Downloads/research/notes/WEEK2/index.html.
Style: Pirsch Analytics (Sunlit Paper Notebook, Cream Paper, Hairline Borders)
Theme: Light, Zero drop shadows, Sunshine highlighter (#ffda6e), Mint signal (#6ece9d)
No speech scripts, rich waveform/process images, dual-layer math (KaTeX + Python rendered images).
"""
import sys
import os

# Add local directory to path
sys.path.insert(0, '/home/pro/Downloads/research/notes/WEEK2')

from styles_pirsch import CSS_PIRSCH
from slides_part_a_pirsch import SLIDE_01_HTML, SLIDE_02_HTML, SLIDE_03_HTML
from slides_part_b_pirsch import SLIDE_04_HTML, SLIDE_05_HTML, SLIDE_06_HTML, SLIDE_07_HTML
from slides_part_c_pirsch import SLIDE_08_HTML, SLIDE_09_HTML
from slides_part_d_pirsch import SLIDE_10_HTML, SLIDE_11_HTML
from slides_part_e_pirsch import SLIDE_12_HTML, SLIDE_13_HTML
from appendix_pirsch import APPENDIX_HTML
from scripts_js import JS_CONTENT

OUTPUT_PATH = "/home/pro/Downloads/research/notes/WEEK2/index.html"

def build():
    all_slides_html = "\n".join([
        SLIDE_01_HTML,
        SLIDE_02_HTML,
        SLIDE_03_HTML,
        SLIDE_04_HTML,
        SLIDE_05_HTML,
        SLIDE_06_HTML,
        SLIDE_07_HTML,
        SLIDE_08_HTML,
        SLIDE_09_HTML,
        SLIDE_10_HTML,
        SLIDE_11_HTML,
        SLIDE_12_HTML,
        SLIDE_13_HTML,
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ECDR &mdash; Báo cáo Toàn diện &amp; Bài Thuyết trình Tuần 2 (Pirsch Style)</title>
  <meta name="description" content="Cấu trúc đầy đủ bài thuyết trình Week 2 của ECDR: HAPT Data Foundations, SmallCNN1D Model Foundations, Certified Decision Reuse, WBS và Weekly Dashboard">
  
  <!-- Google Fonts: DM Sans & JetBrains Mono -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  
  <!-- KaTeX CSS & JS for Mathematical Typography -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

  <style>
{CSS_PIRSCH}
  </style>
</head>
<body class="mode-presentation">

  <!-- Top Global Header -->
  <header class="app-header">
    <div class="brand-section">
      <div class="step-badge" style="background: var(--color-sunshine-highlight); margin-right: 0.5rem;">
        <span class="step-num-cell" style="background: var(--color-ink); color: #fff;">W2</span>
        <span class="step-name-cell" style="color: var(--color-ink);">FOUNDATIONS REPORT</span>
      </div>
      <div class="header-title-wrap">
        <h1>
          <span>Báo cáo Toàn diện Tuần 2: Nền tảng Dữ liệu HAPT &amp; Mô hình CNN</span>
        </h1>
        <div class="header-subtitle">
          Event-Conditioned Certified Decision Reuse on Sequential Sensor Inference &bull; Base Revision <code>main@633dc05</code>
        </div>
      </div>
    </div>

    <!-- Quality Gates Status Indicators -->
    <div class="gates-bar">
      <span class="gate-badge gate-pass" title="Week 1 Baseline Feasibility & Data Ingestion Passed">
        <span>&check;</span> G1: PASS
      </span>
      <span class="gate-badge gate-hold" title="Model Quality Gate on HOLD due to transition classes performance">
        <span>!</span> G2: HOLD (Transitions)
      </span>
      <span class="gate-badge" style="background: #ffffff; color: var(--color-graphite); border-color: var(--color-border-hairline);" title="Certified Region Mathematics & Verifier Target for Week 3">
        <span>&odot;</span> G3: TARGET W3
      </span>
    </div>

    <!-- Global Controls -->
    <div class="header-controls">
      <button class="btn btn-ghost" id="toggleModeBtn" onclick="toggleViewMode()" title="Chuyển đổi giữa Slide Deck và Báo cáo Dạng Tài liệu (Phím tắt: M)">
        <span>📄</span> Chế độ Báo cáo
      </button>
      <button class="btn btn-ghost" id="fullscreenBtn" onclick="toggleFullscreen()" title="Toàn màn hình (Phím tắt: F)">
        <span>⛶</span> Toàn màn hình
      </button>
      <a href="#appendix-qa" class="btn btn-primary" onclick="if(document.body.classList.contains('mode-presentation')) toggleViewMode();">
        <span>📚</span> Phụ lục &amp; Q&amp;A
      </a>
    </div>

    <!-- Slide progress bar -->
    <div class="slide-progress-bar" id="slideProgressBar" style="width: 0%;"></div>
  </header>

  <!-- ===================================================================== -->
  <!-- 1. PRESENTATION MODE DECK CONTAINER                                   -->
  <!-- ===================================================================== -->
  <main class="deck-container">
{all_slides_html}
  </main>

  <!-- Deck Floating Navigation Controls (Bottom Center) -->
  <nav class="deck-controls" aria-label="Slide controls">
    <button class="btn btn-ghost" onclick="prevSlide()" title="Slide trước (Phím: Mũi tên trái / PageUp)">
      &larr; Trước
    </button>

    <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--color-ink); font-size: 0.85rem; font-family: var(--font-mono); font-weight: 500;">
      <span id="currentSlideNum">01</span> / 13
    </div>

    <select class="slide-selector" id="slideSelector" onchange="goToSlide(this.value)" title="Nhảy đến slide">
      <option value="1">Slide 01 &bull; Tổng quan &amp; Recap W1</option>
      <option value="2">Slide 02 &bull; Refined Problem Statement</option>
      <option value="3">Slide 03 &bull; 3 Research Questions</option>
      <option value="4">Slide 04 &bull; HAPT Dataset &amp; Dạng sóng thô</option>
      <option value="5">Slide 05 &bull; Cắt Cửa sổ &amp; Cơ chế Gán nhãn</option>
      <option value="6">Slide 06 &bull; Làm sạch, Split &amp; Chuẩn hóa</option>
      <option value="7">Slide 07 &bull; Stride 64, Phụ thuộc &amp; Rủi ro Thiên lệch</option>
      <option value="8">Slide 08 &bull; Tại sao chọn CNN1D? MACs &amp; Head</option>
      <option value="9">Slide 09 &bull; Frozen CNN &amp; Gate G2 HOLD</option>
      <option value="10">Slide 10 &bull; Trùng lấn &amp; Độ ổn định Quyết định</option>
      <option value="11">Slide 11 &bull; Khế ước Certified Reuse</option>
      <option value="12">Slide 12 &bull; WBS Cập nhật &amp; Kế hoạch Tuần 3</option>
      <option value="13">Slide 13 &bull; Project Dashboard &amp; Quản trị Rủi ro</option>
    </select>

    <button class="btn btn-ghost" onclick="nextSlide()" title="Slide tiếp (Phím: Mũi tên phải / Space / PageDown)">
      Sau &rarr;
    </button>
  </nav>

  <!-- ===================================================================== -->
  <!-- 2. DOCUMENT MODE CONTAINER (Full technical reading layout)            -->
  <!-- ===================================================================== -->
  <div class="doc-layout">
    <!-- Sticky Left Sidebar Table of Contents -->
    <aside class="doc-sidebar">
      <h3>MỤC LỤC BÁO CÁO</h3>
      <ul class="doc-nav-list">
        <li><strong style="color: var(--color-ink); font-size: 0.75rem; text-transform: uppercase;">Part A. Research Direction</strong></li>
        <li><a href="#doc-slide-01" class="doc-nav-link">01. Tổng quan &amp; Kế thừa W1</a></li>
        <li><a href="#doc-slide-02" class="doc-nav-link">02. Refined Problem Statement</a></li>
        <li><a href="#doc-slide-03" class="doc-nav-link">03. 3 Research Questions</a></li>

        <li style="margin-top: 0.6rem;"><strong style="color: var(--color-ink); font-size: 0.75rem; text-transform: uppercase;">Part B. Data Foundations</strong></li>
        <li><a href="#doc-slide-04" class="doc-nav-link">04. HAPT Dataset &amp; Sóng cảm biến</a></li>
        <li><a href="#doc-slide-05" class="doc-nav-link">05. Cửa sổ &amp; Cơ chế Gán nhãn</a></li>
        <li><a href="#doc-slide-06" class="doc-nav-link">06. Làm sạch, Split &amp; Chuẩn hóa</a></li>
        <li><a href="#doc-slide-07" class="doc-nav-link">07. Stride, Tương quan &amp; Thiên lệch</a></li>

        <li style="margin-top: 0.6rem;"><strong style="color: var(--color-ink); font-size: 0.75rem; text-transform: uppercase;">Part C. CNN Foundations</strong></li>
        <li><a href="#doc-slide-08" class="doc-nav-link">08. Kiến trúc CNN1D &amp; MACs</a></li>
        <li><a href="#doc-slide-09" class="doc-nav-link">09. Baseline Behavior &amp; G2 HOLD</a></li>

        <li style="margin-top: 0.6rem;"><strong style="color: var(--color-ink); font-size: 0.75rem; text-transform: uppercase;">Part D. Certified Reuse</strong></li>
        <li><a href="#doc-slide-10" class="doc-nav-link">10. Trùng lấn &amp; Ổn định Quyết định</a></li>
        <li><a href="#doc-slide-11" class="doc-nav-link">11. Khế ước Chứng nhận Toàn miền</a></li>

        <li style="margin-top: 0.6rem;"><strong style="color: var(--color-ink); font-size: 0.75rem; text-transform: uppercase;">Part E. Management</strong></li>
        <li><a href="#doc-slide-12" class="doc-nav-link">12. WBS &amp; Kế hoạch Tuần 3</a></li>
        <li><a href="#doc-slide-13" class="doc-nav-link">13. Dashboard &amp; Quản trị Rủi ro</a></li>

        <li style="margin-top: 0.6rem;"><strong style="color: var(--color-ink); font-size: 0.75rem; text-transform: uppercase; background: var(--color-sunshine-highlight); padding: 0.1rem 0.3rem;">Phụ lục &amp; Bằng chứng</strong></li>
        <li><a href="#appendix-qa" class="doc-nav-link">8 Câu hỏi Đào sâu (Q&amp;A)</a></li>
        <li><a href="#appendix-qa" class="doc-nav-link">Thư viện Đồ thị Kiểm toán</a></li>
        <li><a href="#appendix-qa" class="doc-nav-link">Ma trận Nhầm lẫn 12x12</a></li>
        <li><a href="#appendix-qa" class="doc-nav-link">Bảng 15 Modules &amp; MACs</a></li>
        <li><a href="#appendix-qa" class="doc-nav-link">Bảng Benchmark CPU</a></li>
        <li><a href="#appendix-qa" class="doc-nav-link">Mã băm SHA-256 Deliverables</a></li>
      </ul>
    </aside>

    <!-- Main Document Content Area -->
    <article class="doc-container">
      <div class="card" style="margin-bottom: 2rem; background: var(--color-sunshine-highlight);">
        <h3 style="font-size: 1.1rem; font-weight: 500; color: var(--color-ink); margin-bottom: 0.3rem;">
          📘 Bản Báo cáo Khoa học Toàn văn &mdash; Mốc Bàn giao Tuần 2:
        </h3>
        <p style="font-size: 0.85rem; color: var(--color-ink); line-height: 1.5;">
          Văn bản này tích hợp đầy đủ 13 slide thuyết trình trực quan cùng hệ thống phụ lục đối soát kỹ thuật số, đồ thị sóng cảm biến thô, ma trận nhầm lẫn 12x12, bảng đếm số học MACs và kiểm toán SHA-256. Mọi số liệu đều được tái lập bit-for-bit từ các script kiểm toán độc lập trong repository.
        </p>
      </div>

      <!-- Slides as stacked sections in Document Mode -->
      <div id="doc-slide-01" class="doc-slide-section">{SLIDE_01_HTML}</div>
      <div id="doc-slide-02" class="doc-slide-section">{SLIDE_02_HTML}</div>
      <div id="doc-slide-03" class="doc-slide-section">{SLIDE_03_HTML}</div>
      <div id="doc-slide-04" class="doc-slide-section">{SLIDE_04_HTML}</div>
      <div id="doc-slide-05" class="doc-slide-section">{SLIDE_05_HTML}</div>
      <div id="doc-slide-06" class="doc-slide-section">{SLIDE_06_HTML}</div>
      <div id="doc-slide-07" class="doc-slide-section">{SLIDE_07_HTML}</div>
      <div id="doc-slide-08" class="doc-slide-section">{SLIDE_08_HTML}</div>
      <div id="doc-slide-09" class="doc-slide-section">{SLIDE_09_HTML}</div>
      <div id="doc-slide-10" class="doc-slide-section">{SLIDE_10_HTML}</div>
      <div id="doc-slide-11" class="doc-slide-section">{SLIDE_11_HTML}</div>
      <div id="doc-slide-12" class="doc-slide-section">{SLIDE_12_HTML}</div>
      <div id="doc-slide-13" class="doc-slide-section">{SLIDE_13_HTML}</div>

      <!-- Appendix Section -->
{APPENDIX_HTML}
    </article>
  </div>

  <!-- Lightbox Modal for Full Resolution Images -->
  <div id="img-modal" onclick="closeModal()">
    <button id="modal-close-btn" onclick="closeModal()" title="Đóng ảnh (Phím: Esc)">&times;</button>
    <img id="modal-img" src="" alt="Full resolution audit figure" onclick="event.stopPropagation()">
  </div>

  <!-- Client-side Logic Script -->
  <script>
{JS_CONTENT}
  </script>
</body>
</html>
"""
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Successfully generated visual-rich Pirsch style report to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")

if __name__ == "__main__":
    build()
