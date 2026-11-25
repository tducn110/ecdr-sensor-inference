# -*- coding: utf-8 -*-
"""
Generate all visual assets in Pirsch Analytics Style:
Canvas: #f8f5ed (Cream Paper)
Ink: #000000 (Black)
Graphite: #707070 (Gray)
Sunshine: #ffda6e (Yellow highlighter)
Mint: #6ece9d (Green status)
Hairline borders: 1px #000000
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ASSETS_DIR = "/home/pro/Downloads/research/notes/WEEK2/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Pirsch Design Tokens
CREAM = '#f8f5ed'
INK = '#000000'
GRAPHITE = '#707070'
SUNSHINE = '#ffda6e'
MINT = '#6ece9d'
WHITE = '#ffffff'
LIGHT_GRAY = '#e8e4dc'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.edgecolor'] = INK
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['text.color'] = INK
plt.rcParams['axes.labelcolor'] = INK
plt.rcParams['xtick.color'] = INK
plt.rcParams['ytick.color'] = INK

# ==============================================================================
# 1. RAW DATA SNIPPET & WAVEFORM STRUCTURE (Slide 04 & 05)
# ==============================================================================
def make_raw_waveform_extracted():
    # Load actual raw data
    acc_path = "/home/pro/Downloads/research/data/hapt/RawData/acc_exp01_user01.txt"
    gyro_path = "/home/pro/Downloads/research/data/hapt/RawData/gyro_exp01_user01.txt"
    acc = np.loadtxt(acc_path)
    gyro = np.loadtxt(gyro_path)

    # Slice 128 samples of WALKING (sample 7500 to 7628 = 2.56 seconds)
    t0 = 7500
    samples = 128
    time_sec = np.arange(samples) / 50.0  # 50 Hz

    fig = plt.figure(figsize=(11, 6.2), facecolor=CREAM)
    
    # Subplot 1: Accelerometer (3 channels, in g)
    ax1 = fig.add_subplot(3, 1, 1, facecolor=CREAM)
    ax1.plot(time_sec, acc[t0:t0+samples, 0], color='#111827', lw=1.8, label=r'Acc X ($a_x$ - Dọc cơ thể / Trọng lực)')
    ax1.plot(time_sec, acc[t0:t0+samples, 1], color='#4f46e5', lw=1.5, linestyle='--', label=r'Acc Y ($a_y$ - Ngang hông)')
    ax1.plot(time_sec, acc[t0:t0+samples, 2], color='#059669', lw=1.5, linestyle=':', label=r'Acc Z ($a_z$ - Tiến / Lùi)')
    ax1.set_ylabel(r'Gia tốc ($g$)', fontsize=9.5, weight='bold')
    ax1.set_title('CẤU TẠO SÓNG CẢM BIẾN TRÍCH XUẤT (1 CỬA SỔ 128 MẪU = 2,56 GIÂY @ 50 Hz - HOẠT ĐỘNG ĐI BỘ)', 
                  fontsize=10.5, weight='bold', pad=8)
    ax1.grid(True, linestyle='-', alpha=0.3, color=GRAPHITE)
    ax1.legend(loc='upper right', framealpha=0.9, facecolor=WHITE, edgecolor=INK, fontsize=8)
    
    # Highlight periodic step cycle
    ax1.axvspan(0.4, 0.95, color=SUNSHINE, alpha=0.35, label='1 Gait Cycle (~0.55s)')
    ax1.text(0.675, 1.45, '1 Chu kỳ sải chân (Gait Cycle)', ha='center', fontsize=8, weight='bold', color=INK,
             bbox=dict(boxstyle='round,pad=0.2', facecolor=SUNSHINE, edgecolor=INK, lw=1))

    # Subplot 2: Gyroscope (3 channels, in rad/s)
    ax2 = fig.add_subplot(3, 1, 2, facecolor=CREAM)
    ax2.plot(time_sec, gyro[t0:t0+samples, 0], color='#b91c1c', lw=1.8, label=r'Gyro X ($\omega_x$ - Lật ngửa / cúi)')
    ax2.plot(time_sec, gyro[t0:t0+samples, 1], color='#7c3aed', lw=1.5, linestyle='--', label=r'Gyro Y ($\omega_y$ - Xoay vặn người)')
    ax2.plot(time_sec, gyro[t0:t0+samples, 2], color='#0284c7', lw=1.5, linestyle=':', label=r'Gyro Z ($\omega_z$ - Nghiêng trái/phải)')
    ax2.set_ylabel(r'Vận tốc góc (rad/s)', fontsize=9.5, weight='bold')
    ax2.grid(True, linestyle='-', alpha=0.3, color=GRAPHITE)
    ax2.legend(loc='upper right', framealpha=0.9, facecolor=WHITE, edgecolor=INK, fontsize=8)

    # Subplot 3: Raw numerical text representation snippet
    ax3 = fig.add_subplot(3, 1, 3, facecolor='#ffffff')
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.0)
    
    raw_text = (
        "DỮ LIỆU THÔ TRONG TỆP RAW (Snippet từ acc_exp01_user01.txt & gyro_exp01_user01.txt):\n"
        "  Sample 7500 (t=0.00s):  Acc [1.0208, -0.1250, 0.1056] g   |   Gyro [-0.0027, -0.0043,  0.0027] rad/s   --> Label: WALKING\n"
        "  Sample 7501 (t=0.02s):  Acc [1.0250, -0.1250, 0.1014] g   |   Gyro [-0.0003, -0.0021,  0.0061] rad/s   --> Label: WALKING\n"
        "  Sample 7502 (t=0.04s):  Acc [1.0208, -0.1250, 0.1042] g   |   Gyro [ 0.0122,  0.0009, -0.0073] rad/s   --> Label: WALKING\n"
        "  Sample 7503 (t=0.06s):  Acc [1.0167, -0.1250, 0.1083] g   |   Gyro [ 0.0113, -0.0018, -0.0064] rad/s   --> Label: WALKING\n"
        "  ... 128 mẫu liên tiếp ghép thành Tensor [6 x 128] tương ứng 2.56 giây thu nhận liên tục ở 50 Hz."
    )
    ax3.text(0.02, 0.5, raw_text, family='monospace', fontsize=8, va='center', color=INK)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "pirsch_raw_waveform_extracted.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print("Generated:", path)

# ==============================================================================
# 2. MATRIX FEATURE EXTRACTION (Slide 05 & 06)
# ==============================================================================
def make_window_matrix_extraction():
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis('off')

    ax.text(0.5, 0.94, "QUY TRÌNH TRÍCH XUẤT ĐẶC TRƯNG: TỪ DÒNG TÍN HIỆU THÔ RA MA TRẬN [6 x 128]",
            fontsize=11, weight='bold', ha='center', color=INK)

    # Box 1: Continuous Stream
    rect1 = patches.FancyBboxPatch((0.03, 0.2), 0.24, 0.62,
                                  boxstyle="round,pad=0.02", facecolor=WHITE, edgecolor=INK, linewidth=1.2)
    ax.add_patch(rect1)
    ax.text(0.15, 0.74, "1. Dòng Tín hiệu Thô", fontsize=9.5, weight='bold', ha='center', color=INK)
    ax.text(0.15, 0.50, "Bản ghi liên tục\n(T samples x 6 channels)\nTần số lấy mẫu: 50 Hz\n(dt = 0.02s / sample)",
            fontsize=8.5, ha='center', color=GRAPHITE)

    # Arrow 1
    ax.annotate('', xy=(0.33, 0.51), xytext=(0.28, 0.51),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))

    # Box 2: Sliding Windowing
    rect2 = patches.FancyBboxPatch((0.34, 0.2), 0.26, 0.62,
                                  boxstyle="round,pad=0.02", facecolor=SUNSHINE, edgecolor=INK, linewidth=1.2)
    ax.add_patch(rect2)
    ax.text(0.47, 0.74, "2. Cắt Cửa sổ Trượt", fontsize=9.5, weight='bold', ha='center', color=INK)
    ax.text(0.47, 0.48, "Cửa sổ: 128 samples (2.56 s)\nBước trượt: Stride 64 (1.28 s)\nChồng lấn: Đúng 50% thời gian\n(Khóa ranh giới recording)",
            fontsize=8.5, ha='center', color=INK)

    # Arrow 2
    ax.annotate('', xy=(0.66, 0.51), xytext=(0.61, 0.51),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))

    # Box 3: 2D Matrix Output
    rect3 = patches.FancyBboxPatch((0.67, 0.2), 0.30, 0.62,
                                  boxstyle="round,pad=0.02", facecolor=WHITE, edgecolor=INK, linewidth=1.2)
    ax.add_patch(rect3)
    ax.text(0.82, 0.74, "3. Ma trận Đầu vào CNN", fontsize=9.5, weight='bold', ha='center', color=INK)
    matrix_tex = (
        "Tensor Input: [B, 6, 128]\n"
        " ┌                           ┐\n"
        " │ a_x(t0)   ...   a_x(t127) │\n"
        " │ a_y(t0)   ...   a_y(t127) │\n"
        " │ a_z(t0)   ...   a_z(t127) │  6 Kênh\n"
        " │ w_x(t0)   ...   w_x(t127) │  Cảm biến\n"
        " │ w_y(t0)   ...   w_y(t127) │\n"
        " │ w_z(t0)   ...   w_z(t127) │\n"
        " └                           ┘\n"
        "      128 Mẫu Thời gian"
    )
    ax.text(0.82, 0.44, matrix_tex, family='monospace', fontsize=7.5, ha='center', color=INK)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "pirsch_window_matrix_extraction.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print("Generated:", path)

# ==============================================================================
# 3. LABEL POLICY TIMELINE: WHY WINDOWS ARE LABELED OR NOT (Slide 05)
# ==============================================================================
def make_label_policy_timeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis('off')

    ax.text(0.5, 0.94, "CƠ CHẾ GÁN NHÃN: VÌ SAO CỬA SỔ ĐƯỢC CHẤP NHẬN HOẶC BỊ TỪ CHỐI?",
            fontsize=11, weight='bold', ha='center', color=INK)

    scenarios = [
        ("A. PURE WINDOW (10.361 Cửa sổ)", 0.72,
         [(0, 128, MINT, "100% Mẫu là WALKING (Class 1)")],
         "KẾT QUẢ: ĐƯỢC GÁN NHÃN = 1 (SUPERVISED)", MINT),

        ("B. MAJORITY WINDOW (1.917 Cửa sổ)", 0.50,
         [(0, 85, MINT, "85 Mẫu WALKING (66.4%)"), (85, 128, '#fdba74', "43 Mẫu SITTING")],
         "KẾT QUẢ: ĐƯỢC GÁN NHÃN = 1 (>= 60% ĐA SỐ)", SUNSHINE),

        ("C. AMBIGUOUS WINDOW (1.588 Cửa sổ)", 0.28,
         [(0, 60, '#fdba74', "60 SITTING (46.9%)"), (60, 128, '#f472b6', "68 STANDING (53.1%)")],
         "KẾT QUẢ: BỊ LOẠI KHỎI SUPERVISED (GIỮ TRONG STREAM)", '#fca5a5'),

        ("D. UNLABELED WINDOW (3.587 Cửa sổ)", 0.06,
         [(0, 128, '#e5e7eb', "100% Khoảng nghỉ / Không có chú thích hoạt động")],
         "KẾT QUẢ: KHÔNG CÓ GROUND TRUTH (GIỮ TRONG STREAM)", '#e5e7eb'),
    ]

    for title, y, blocks, result_str, tag_color in scenarios:
        ax.text(0.04, y + 0.12, title, fontsize=8.5, weight='bold', color=INK)

        # Draw 128 sample bar
        bar_x = 0.04
        bar_w = 0.56
        bar_h = 0.07

        rect_bg = patches.Rectangle((bar_x, y), bar_w, bar_h, facecolor=WHITE, edgecolor=INK, lw=1)
        ax.add_patch(rect_bg)

        for b_start, b_end, b_color, b_label in blocks:
            seg_x = bar_x + (b_start / 128.0) * bar_w
            seg_w = ((b_end - b_start) / 128.0) * bar_w
            rect_seg = patches.Rectangle((seg_x, y), seg_w, bar_h, facecolor=b_color, edgecolor=INK, lw=0.8)
            ax.add_patch(rect_seg)
            ax.text(seg_x + seg_w / 2.0, y + bar_h / 2.0, b_label, ha='center', va='center', fontsize=7, weight='bold', color=INK)

        # Result badge
        res_box = patches.FancyBboxPatch((0.63, y), 0.33, bar_h, boxstyle="round,pad=0.015",
                                         facecolor=tag_color, edgecolor=INK, lw=1)
        ax.add_patch(res_box)
        ax.text(0.795, y + bar_h / 2.0, result_str, ha='center', va='center', fontsize=7.5, weight='bold', color=INK)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "pirsch_label_policy_timeline.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print("Generated:", path)

# ==============================================================================
# 4. STRIDE RATIONALE & 50% OVERLAP (Slide 10 & 05)
# ==============================================================================
def make_stride_rationale():
    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis('off')

    ax.text(0.5, 0.94, "VÌ SAO STRIDE 64 (50% OVERLAP) LẠI TỐI QUAN TRỌNG CHO ECDR?",
            fontsize=11, weight='bold', ha='center', color=INK)

    # Top: Visualizing Overlap
    ax.text(0.04, 0.82, "1. Cơ chế Cửa sổ Liền kề: Nửa sau cửa sổ cũ trùng 100% với nửa đầu cửa sổ mới",
            fontsize=8.5, weight='bold', color=INK)

    # Window t
    rect_w1 = patches.Rectangle((0.08, 0.62), 0.40, 0.12, facecolor=WHITE, edgecolor=INK, lw=1.2)
    ax.add_patch(rect_w1)
    ax.text(0.18, 0.68, "Xt [0 : 64] (Cũ)", ha='center', fontsize=8, color=GRAPHITE)
    # Overlap part of w1
    rect_ov1 = patches.Rectangle((0.28, 0.62), 0.20, 0.12, facecolor=SUNSHINE, edgecolor=INK, lw=1.2)
    ax.add_patch(rect_ov1)
    ax.text(0.38, 0.68, "Xt [64 : 128] (Trùng)", ha='center', fontsize=8, weight='bold', color=INK)

    # Window t+1
    rect_ov2 = patches.Rectangle((0.28, 0.44), 0.20, 0.12, facecolor=SUNSHINE, edgecolor=INK, lw=1.2)
    ax.add_patch(rect_ov2)
    ax.text(0.38, 0.50, "Xt+1 [0 : 64] (Trùng)", ha='center', fontsize=8, weight='bold', color=INK)
    rect_w2 = patches.Rectangle((0.48, 0.44), 0.20, 0.12, facecolor=WHITE, edgecolor=INK, lw=1.2)
    ax.add_patch(rect_w2)
    ax.text(0.58, 0.50, "Xt+1 [64 : 128] (Mới)", ha='center', fontsize=8, color=GRAPHITE)

    # Right side: 3 Key Scientific Reasons
    reasons_box = patches.FancyBboxPatch((0.52, 0.62), 0.44, 0.14, boxstyle="round,pad=0.02",
                                         facecolor=MINT, edgecolor=INK, lw=1)
    ax.add_patch(reasons_box)
    ax.text(0.74, 0.69, "TIỀN ĐỀ TOÁN HỌC CỦA ECDR:\n17.392 Cặp Trùng lấn Khớp Tuyệt đối (MAE = 0.00000)",
            ha='center', va='center', fontsize=8, weight='bold', color=INK)

    # Bottom comparisons
    reasons_text = (
        "2. VÌ SAO PHẢI CÓ STRIDE 64 (50% OVERLAP)?\n"
        "  • Tránh Điểm mù Ranh giới (No Boundary Blindspots): Chuyển tiếp tư thế diễn ra ngắn (~1.2s).\n"
        "    Nếu không overlap (stride 128), biến cố xảy ra ở biên sẽ bị cắt làm đôi và mô hình đoán sai cả 2 cửa sổ.\n"
        "  • Tăng Độ nhạy Thời gian Thực: Cung cấp quyết định mới mỗi 1.28 giây (thay vì 2.56 giây).\n"
        "  • Cơ hội Vàng cho Decision Reuse: Nếu nửa đầu cửa sổ mới đã biết, ECDR chỉ cần kiểm định xem\n"
        "    nửa sau có làm thay đổi biên quyết định của mô hình hay không!"
    )
    rect_desc = patches.FancyBboxPatch((0.04, 0.05), 0.92, 0.32, boxstyle="round,pad=0.02",
                                       facecolor=WHITE, edgecolor=INK, lw=1)
    ax.add_patch(rect_desc)
    ax.text(0.06, 0.21, reasons_text, fontsize=8, va='center', color=INK, family='sans-serif', linespacing=1.4)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "pirsch_stride_rationale.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print("Generated:", path)

# ==============================================================================
# 5. PIRSCH-STYLED MATH FORMULAS
# ==============================================================================
def make_pirsch_math():
    # Normalization formula
    fig, ax = plt.subplots(figsize=(7.5, 1.8), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis('off')
    # Background card with black border
    rect = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.03",
                                  facecolor=WHITE, edgecolor=INK, lw=1.2)
    ax.add_patch(rect)
    # Sunshine badge
    badge = patches.FancyBboxPatch((0.05, 0.65), 0.45, 0.22, boxstyle="round,pad=0.02",
                                   facecolor=SUNSHINE, edgecolor=INK, lw=0.8)
    ax.add_patch(badge)
    ax.text(0.275, 0.76, "TRAIN-ONLY CHANNEL NORMALIZATION", fontsize=7.5, weight='bold', ha='center', color=INK)
    ax.text(0.5, 0.35, r"$Z_{c,t} = \frac{X_{c,t} - \mu_c}{\max(\sigma_c,\, 10^{-6})}$",
            fontsize=16, ha='center', va='center', color=INK)
    path = os.path.join(ASSETS_DIR, "pirsch_math_normalization.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()

    # Central Certification Formula
    fig, ax = plt.subplots(figsize=(9.2, 2.0), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.axis('off')
    rect = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.03",
                                  facecolor=WHITE, edgecolor=INK, lw=1.2)
    ax.add_patch(rect)
    badge = patches.FancyBboxPatch((0.05, 0.68), 0.50, 0.20, boxstyle="round,pad=0.02",
                                   facecolor=SUNSHINE, edgecolor=INK, lw=0.8)
    ax.add_patch(badge)
    ax.text(0.30, 0.78, "WHOLE-REGION 11-MARGIN CERTIFICATION", fontsize=7.5, weight='bold', ha='center', color=INK)
    ax.text(0.5, 0.35, r"$\inf_{X \in S} \left[ F_c(X) - F_j(X) \right] > 0 \quad \forall j \neq c, \quad j \in \{1, \dots, 12\}$",
            fontsize=15, ha='center', va='center', color=INK)
    path = os.path.join(ASSETS_DIR, "pirsch_math_certification.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close()
    print("Generated Pirsch math formulas.")

# Execute all Pirsch visual generators
make_raw_waveform_extracted()
make_window_matrix_extraction()
make_label_policy_timeline()
make_stride_rationale()
make_pirsch_math()

print("All new Pirsch visual assets generated successfully!")
