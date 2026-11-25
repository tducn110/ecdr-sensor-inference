# -*- coding: utf-8 -*-
"""
Generate publication-quality figures, diagrams, and math formula images for Week 2 slides.
Uses matplotlib in .venv.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ASSETS_DIR = "/home/pro/Downloads/research/notes/WEEK2/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Common styling
BG_COLOR = '#0f172a'
CARD_BG = '#1e293b'
BORDER_COLOR = '#334155'
TEXT_COLOR = '#f8fafc'
ACCENT_CYAN = '#06b6d4'
ACCENT_INDIGO = '#6366f1'
ACCENT_EMERALD = '#10b981'
ACCENT_AMBER = '#f59e0b'
ACCENT_ROSE = '#f43f5e'

def render_math_formula(filename, latex_str, title_str="", width=8, height=2.2, font_size=16):
    fig = plt.figure(figsize=(width, height), facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG_COLOR)
    ax.axis('off')
    if title_str:
        ax.text(0.5, 0.82, title_str, color=ACCENT_CYAN, fontsize=12,
                weight='bold', ha='center', va='center')
    ax.text(0.5, 0.42 if title_str else 0.5, latex_str, color=TEXT_COLOR, fontsize=font_size,
            ha='center', va='center')
    path = os.path.join(ASSETS_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.1, facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 1. Normalization Formula Image
render_math_formula(
    "math_normalization.png",
    r"$Z_{c,t} = \frac{X_{c,t} - \mu_c}{\max(\sigma_c,\, 10^{-6})}$",
    title_str="TRAIN-ONLY CHANNEL-WISE NORMALIZATION",
    width=7, height=1.8, font_size=18
)

# 2. Central Certification Margin Condition Formula Image
render_math_formula(
    "math_certification_condition.png",
    r"$\inf_{X \in S} \left[ F_c(X) - F_j(X) \right] > 0 \quad \forall j \neq c, \quad j \in \{1, \dots, 12\}$",
    title_str="WHOLE-REGION 11-MARGIN CERTIFICATION CONDITION",
    width=9, height=2.0, font_size=17
)

# 3. Slide 01: Overview Pipeline Diagram
def make_slide01_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    boxes = [
        ("Raw Sensor Stream", "HAPT 50 Hz\n6 Channels (3 Acc + 3 Gyro)", ACCENT_CYAN, 0.05),
        ("Sliding Windows", "128 Samples (2.56 s)\nStride 64 (50% Overlap)", ACCENT_INDIGO, 0.28),
        ("Frozen SmallCNN1D", "3x Conv1D + MaxPool\n283,692 Params (K=12)", ACCENT_AMBER, 0.51),
        ("Certified Reuse", "Region Bound Checks\nReuse Decision / Fallback", ACCENT_EMERALD, 0.74),
    ]

    for title, desc, color, x in boxes:
        rect = patches.FancyBboxPatch((x, 0.2), 0.19, 0.6,
                                      boxstyle="round,pad=0.03",
                                      facecolor=CARD_BG, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.095, 0.65, title, color='#ffffff', fontsize=11, weight='bold', ha='center', va='center')
        ax.text(x + 0.095, 0.42, desc, color='#cbd5e1', fontsize=8.5, ha='center', va='center')

        # Arrow
        if x < 0.7:
            ax.annotate('', xy=(x + 0.23, 0.5), xytext=(x + 0.19, 0.5),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5))

    path = os.path.join(ASSETS_DIR, "slide01_overview_pipeline.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 4. Slide 02: Classification Correctness vs Decision Preservation
def make_slide02_concept():
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # Card 1: Classification Correctness
    rect1 = patches.FancyBboxPatch((0.05, 0.15), 0.42, 0.7,
                                  boxstyle="round,pad=0.03",
                                  facecolor=CARD_BG, edgecolor=ACCENT_INDIGO, linewidth=2)
    ax.add_patch(rect1)
    ax.text(0.26, 0.72, "Classification Correctness", color='#ffffff', fontsize=12, weight='bold', ha='center')
    ax.text(0.26, 0.56, r"$h_{frozen}(X) = y_{true} \; (?)$", color=ACCENT_CYAN, fontsize=14, ha='center')
    ax.text(0.26, 0.35, "CNN dự đoán có đúng nhãn thực tế của người dùng không?\n(Dùng để đánh giá độ tin cậy của mô hình cơ sở)",
            color='#cbd5e1', fontsize=8.5, ha='center')

    # Card 2: Decision Preservation
    rect2 = patches.FancyBboxPatch((0.53, 0.15), 0.42, 0.7,
                                  boxstyle="round,pad=0.03",
                                  facecolor=CARD_BG, edgecolor=ACCENT_EMERALD, linewidth=2)
    ax.add_patch(rect2)
    ax.text(0.74, 0.72, "Decision Preservation (ECDR Core)", color='#ffffff', fontsize=12, weight='bold', ha='center')
    ax.text(0.74, 0.56, r"$\hat{y}_{ECDR}(X) \equiv h_{frozen}(X)$", color=ACCENT_EMERALD, fontsize=14, ha='center')
    ax.text(0.74, 0.35, "ECDR có bảo toàn đúng 100% quyết định mà frozen CNN sẽ trả?\n(Bảo toàn quyết định tuyệt đối trong miền chứng nhận)",
            color='#cbd5e1', fontsize=8.5, ha='center')

    path = os.path.join(ASSETS_DIR, "slide02_problem_statement.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 5. Slide 03: Three Research Questions Roadmap
def make_slide03_rq():
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    rqs = [
        ("RQ1: Reference Quality", "Mô hình CNN có đủ tin cậy?\nAcc 89.05% | Macro-F1 0.7763\n[ĐÃ GIẢI QUYẾT W1-W2]", ACCENT_CYAN, 0.05),
        ("RQ2: Certified Preservation", "Khi nào được phép reuse?\nWhole-region 11 margins > 0\n[MỤC TIÊU TUẦN 3]", ACCENT_INDIGO, 0.37),
        ("RQ3: Efficiency", "Có thực sự tiết kiệm chi phí?\nNet speedup & Latency vs CNN\n[KẾ HOẠCH TUẦN 4+]", ACCENT_EMERALD, 0.69),
    ]

    for title, desc, color, x in rqs:
        rect = patches.FancyBboxPatch((x, 0.15), 0.26, 0.7,
                                      boxstyle="round,pad=0.03",
                                      facecolor=CARD_BG, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.13, 0.72, title, color='#ffffff', fontsize=11, weight='bold', ha='center')
        ax.text(x + 0.13, 0.42, desc, color='#cbd5e1', fontsize=8.5, ha='center')

        if x < 0.65:
            ax.annotate('', xy=(x + 0.30, 0.5), xytext=(x + 0.26, 0.5),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5))

    path = os.path.join(ASSETS_DIR, "slide03_rq_roadmap.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 6. Slide 10: Decision Stability Comparison Bar Chart
def make_slide10_stability():
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    categories = ['Pure Stable Pairs\n(N = 2,617)', 'Pure Transition Pairs\n(N = 44)']
    agreement = [97.82, 52.27]
    flips = [2.18, 47.73]

    x = np.arange(len(categories))
    width = 0.45

    b1 = ax.bar(x, agreement, width, label='Cùng Dự đoán (Prediction Agreement)', color=ACCENT_EMERALD, edgecolor=BORDER_COLOR)
    b2 = ax.bar(x, flips, width, bottom=agreement, label='Đổi Quyết định (Label Flip)', color=ACCENT_ROSE, edgecolor=BORDER_COLOR)

    ax.set_ylabel('Tỷ lệ (%)', color=TEXT_COLOR, fontsize=10)
    ax.set_title('Độ Ổn định Quyết định Thực nghiệm giữa 2 Cửa sổ Trùng lấn 50%', color=TEXT_COLOR, fontsize=11, weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=TEXT_COLOR, fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, loc='upper right', fontsize=8.5)

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval/2.0, f"{yval:.1f}%", ha='center', va='center', color='#ffffff', weight='bold', fontsize=10)

    for bar in b2:
        yval = bar.get_height()
        if yval > 5:
            ax.text(bar.get_x() + bar.get_width()/2.0, 97.82 + yval/2.0 if bar.get_x() < 0.5 else 52.27 + yval/2.0,
                    f"{yval:.1f}%", ha='center', va='center', color='#ffffff', weight='bold', fontsize=9.5)

    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#ffffff')
    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "slide10_decision_stability.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 7. Slide 11: Certification Lifecycle Flowchart
def make_slide11_certification():
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    steps = [
        ("1. Anchor X^r", "Run Frozen CNN\nLưu class c", ACCENT_CYAN, 0.04),
        ("2. Candidate S", "Miền lân cận\nquanh anchor", ACCENT_INDIGO, 0.23),
        ("3. 11-Margin Bounds", "inf [Fc - Fj] > 0\nToàn miền S", ACCENT_AMBER, 0.42),
        ("4. Check X_new", "Membership in S\nIdentity + Soundness", ACCENT_CYAN, 0.61),
        ("5. Branching", "Pass -> REUSE c\nFail -> FALLBACK CNN", ACCENT_EMERALD, 0.80),
    ]

    for title, desc, color, x in steps:
        rect = patches.FancyBboxPatch((x, 0.18), 0.16, 0.64,
                                      boxstyle="round,pad=0.03",
                                      facecolor=CARD_BG, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.08, 0.66, title, color='#ffffff', fontsize=10, weight='bold', ha='center')
        ax.text(x + 0.08, 0.42, desc, color='#cbd5e1', fontsize=8, ha='center')

        if x < 0.75:
            ax.annotate('', xy=(x + 0.19, 0.5), xytext=(x + 0.16, 0.5),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.2))

    path = os.path.join(ASSETS_DIR, "slide11_certification_workflow.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 8. Slide 12: WBS Roadmap Diagram
def make_slide12_wbs():
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    wbs_data = [
        ("WP1: Research Brief & Frozen Baseline", "HOÀN TẤT (W1)", ACCENT_EMERALD, 0.80),
        ("WP2: HAPT Data Foundations", "ĐÃ KIỂM CHỨNG (W2)", ACCENT_EMERALD, 0.66),
        ("WP3: SmallCNN1D Model Foundations", "ĐÃ KIỂM CHỨNG (W2)", ACCENT_EMERALD, 0.52),
        ("WP4: Certified-Region Mathematics", "TRỌNG TÂM TUẦN 3", ACCENT_AMBER, 0.38),
        ("WP5: Actual-Graph Verifier Prototype", "TRỌNG TÂM TUẦN 3", ACCENT_AMBER, 0.24),
        ("WP6: Certified Coverage & Efficiency", "KẾ HOẠCH TUẦN 4+", '#64748b', 0.10),
    ]

    ax.text(0.5, 0.94, "BẢN ĐỒ TIẾN ĐỘ WORK BREAKDOWN STRUCTURE (WBS)", color='#ffffff', fontsize=11, weight='bold', ha='center')

    for name, status, color, y in wbs_data:
        rect = patches.FancyBboxPatch((0.05, y), 0.62, 0.10,
                                      boxstyle="round,pad=0.02",
                                      facecolor=CARD_BG, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(0.08, y + 0.05, name, color='#ffffff', fontsize=9, va='center', weight='500')

        s_rect = patches.FancyBboxPatch((0.70, y), 0.25, 0.10,
                                        boxstyle="round,pad=0.02",
                                        facecolor=CARD_BG, edgecolor=color, linewidth=1.5)
        ax.add_patch(s_rect)
        ax.text(0.825, y + 0.05, status, color=color, fontsize=8.5, weight='bold', ha='center', va='center')

    path = os.path.join(ASSETS_DIR, "slide12_wbs_roadmap.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# 9. Slide 13: Planned vs Actual Hours Bar Chart
def make_slide13_hours():
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_BG)

    tasks = ['Data Audit\n& Overlap', 'CNN Analysis\n& Latency', 'Report QA\n& Delivery', 'Week 3\nPreparation']
    planned = [20, 18, 15, 10]
    actual = [24, 19, 16, 8]

    x = np.arange(len(tasks))
    width = 0.32

    rects1 = ax.bar(x - width/2, planned, width, label='Kế hoạch (Planned)', color=ACCENT_INDIGO, edgecolor=BORDER_COLOR)
    rects2 = ax.bar(x + width/2, actual, width, label='Thực tế (Actual)', color=ACCENT_CYAN, edgecolor=BORDER_COLOR)

    ax.set_ylabel('Giờ Công (Hours)', color=TEXT_COLOR, fontsize=9.5)
    ax.set_title('HOW MUCH: ĐỐI SOÁT GIỜ CÔNG KẾ HOẠCH VS THỰC TẾ (TUẦN 2)', color=TEXT_COLOR, fontsize=10.5, weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, color=TEXT_COLOR, fontsize=8.5)
    ax.set_ylim(0, 30)
    ax.legend(facecolor=CARD_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=8.5)

    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.6, f'{h}h', ha='center', va='bottom', color='#cbd5e1', fontsize=8.5)
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.6, f'{h}h', ha='center', va='bottom', color='#38bdf8', weight='bold', fontsize=8.5)

    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#ffffff')
    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "slide13_planned_vs_actual.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Generated: {path}")

# Run all generator functions
make_slide01_pipeline()
make_slide02_concept()
make_slide03_rq()
make_slide10_stability()
make_slide11_certification()
make_slide12_wbs()
make_slide13_hours()

print("All 9 visual assets successfully generated in assets directory!")
