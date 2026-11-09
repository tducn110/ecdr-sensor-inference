# -*- coding: utf-8 -*-
"""
JavaScript functionality for Slide Deck, Document Mode, Modals, and KaTeX rendering.
"""

JS_CONTENT = """
document.addEventListener('DOMContentLoaded', function() {
  // 1. Initial State
  let currentSlide = 1;
  const totalSlides = 13;
  let isPresentationMode = true;
  let notesVisible = false;

  const slides = document.querySelectorAll('.slide-frame');
  const slideSelector = document.getElementById('slideSelector');
  const progressBar = document.getElementById('slideProgressBar');
  const currentSlideDisplay = document.getElementById('currentSlideNum');
  const modeBtn = document.getElementById('toggleModeBtn');
  const notesGlobalBtn = document.getElementById('toggleNotesGlobalBtn');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const imgModal = document.getElementById('img-modal');
  const modalImg = document.getElementById('modal-img');

  // 2. Navigation Functions
  function updateSlide(idx) {
    if (idx < 1) idx = 1;
    if (idx > totalSlides) idx = totalSlides;
    currentSlide = idx;

    // In presentation mode, only active slide is flex
    if (isPresentationMode) {
      slides.forEach((s, i) => {
        if (i + 1 === currentSlide) {
          s.classList.add('active-slide');
          s.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          s.classList.remove('active-slide');
        }
      });
    }

    if (slideSelector) slideSelector.value = currentSlide;
    if (currentSlideDisplay) currentSlideDisplay.textContent = currentSlide < 10 ? '0' + currentSlide : currentSlide;
    if (progressBar) {
      const pct = ((currentSlide - 1) / (totalSlides - 1)) * 100;
      progressBar.style.width = pct + '%';
    }
  }

  window.goToSlide = function(idx) {
    updateSlide(parseInt(idx, 10));
  };

  window.nextSlide = function() {
    if (currentSlide < totalSlides) {
      updateSlide(currentSlide + 1);
    }
  };

  window.prevSlide = function() {
    if (currentSlide > 1) {
      updateSlide(currentSlide - 1);
    }
  };

  // 3. View Mode Switcher
  window.toggleViewMode = function() {
    isPresentationMode = !isPresentationMode;
    if (isPresentationMode) {
      document.body.classList.remove('mode-document');
      document.body.classList.add('mode-presentation');
      if (modeBtn) modeBtn.innerHTML = '<span>📄</span> Chế độ Báo cáo';
      updateSlide(currentSlide);
    } else {
      document.body.classList.remove('mode-presentation');
      document.body.classList.add('mode-document');
      if (modeBtn) modeBtn.innerHTML = '<span>📽️</span> Chế độ Slide Deck';
      // In document mode, make all slides visible
      slides.forEach(s => s.classList.add('active-slide'));
    }
  };

  // 4. Speaker Notes Toggle
  window.toggleNotes = function(drawerId) {
    const el = document.getElementById(drawerId);
    if (el) {
      el.classList.toggle('open-notes');
    }
  };

  window.toggleAllNotes = function() {
    notesVisible = !notesVisible;
    const allDrawers = document.querySelectorAll('.speaker-notes-drawer');
    allDrawers.forEach(d => {
      if (notesVisible) {
        d.classList.add('open-notes');
      } else {
        d.classList.remove('open-notes');
      }
    });
    if (notesGlobalBtn) {
      notesGlobalBtn.classList.toggle('btn-active', notesVisible);
    }
  };

  // 5. Fullscreen Toggle
  window.toggleFullscreen = function() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.warn('Error attempting to enable fullscreen:', err.message);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  // 6. Image Modal
  window.openModal = function(src) {
    if (modalImg && imgModal) {
      modalImg.src = src;
      imgModal.classList.add('open');
    }
  };

  window.closeModal = function() {
    if (imgModal) {
      imgModal.classList.remove('open');
    }
  };

  // 7. Accordions for Appendix
  window.toggleAccordion = function(id) {
    const item = document.getElementById(id);
    if (item) {
      item.classList.toggle('open');
    }
  };

  // 8. Keyboard Shortcuts
  document.addEventListener('keydown', function(e) {
    // If modal open, Escape closes it
    if (imgModal && imgModal.classList.contains('open') && e.key === 'Escape') {
      closeModal();
      return;
    }

    // Ignore keyboard shortcuts if focusing input/select
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      return;
    }

    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
      e.preventDefault();
      nextSlide();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      prevSlide();
    } else if (e.key === 'f' || e.key === 'F') {
      toggleFullscreen();
    } else if (e.key === 'n' || e.key === 'N') {
      toggleAllNotes();
    } else if (e.key === 'm' || e.key === 'M') {
      toggleViewMode();
    }
  });

  // 9. Initialize KaTeX Auto-Render
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\\\[', right: '\\\\]', display: true },
        { left: '\\\\(', right: '\\\\)', display: false },
        { left: '$', right: '$', display: false }
      ],
      throwOnError: false
    });
  } else {
    console.warn('KaTeX auto-render not loaded.');
  }

  // Initial display setup
  updateSlide(1);
});
"""
