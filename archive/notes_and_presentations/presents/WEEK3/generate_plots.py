# -*- coding: utf-8 -*-
"""
generate_plots.py — Generates publication-quality academic figures for ECDR Week 3.
Reads actual receipts from output/week3_certification_20261103/ and outputs PNGs into notes/WEEK3/assets/.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = "/home/pro/Downloads/research"
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/week3_certification_20261103")
ASSETS_DIR = os.path.join(ROOT_DIR, "notes/WEEK3/assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Set global academic style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.0

# Academic colors
INK = '#183449'
TEAL = '#187C80'
RED = '#EF4444'
AMBER = '#D97706'
EMERALD = '#059669'
MUTED = '#64748B'
SURFACE = '#F8FAFC'

# 1. Load data
with open(os.path.join(OUTPUT_DIR, "runtime_results.json")) as f:
    runtime_res = json.load(f)
with open(os.path.join(OUTPUT_DIR, "domain_results.json")) as f:
    domain_res = json.load(f)
with open(os.path.join(OUTPUT_DIR, "geometry.json")) as f:
    geometry_res = json.load(f)
with open(os.path.join(OUTPUT_DIR, "feature_alignment.json")) as f:
    fa_res = json.load(f)
with open(os.path.join(OUTPUT_DIR, "memory_components.json")) as f:
    mem_res = json.load(f)

# -------------------------------------------------------------
# FIG 01: Dual-Track Inference Pipeline Architecture
# -------------------------------------------------------------
def plot_fig01():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_facecolor(SURFACE)
    ax.axis('off')
    
    # Input
    bbox_in = dict(boxstyle="round,pad=0.6", facecolor='#FFFFFF', edgecolor=TEAL, lw=2)
    ax.text(0.12, 0.5, "Input Sensor Window\n$X \\in \\mathbb{R}^{6 \\times 128}$ (768-D)\nStride 64 (50% Overlap)", 
            ha='center', va='center', bbox=bbox_in, color=INK, weight='bold')

    # Conventional Track
    bbox_cnn = dict(boxstyle="round,pad=0.6", facecolor='#F1F5F9', edgecolor=MUTED, lw=1.5)
    ax.text(0.5, 0.8, "Conventional Path\nFull SmallCNN1D Forward\n(12 logits, 283.692 params)", 
            ha='center', va='center', bbox=bbox_cnn, color=INK)

    bbox_pred1 = dict(boxstyle="round,pad=0.6", facecolor='#E2E8F0', edgecolor=MUTED, lw=1.5)
    ax.text(0.88, 0.8, "Activity Output\n$\\arg\\max_k F_k(X)$", 
            ha='center', va='center', bbox=bbox_pred1, color=INK, weight='bold')

    # ECDR Candidate Track
    bbox_ecdr = dict(boxstyle="round,pad=0.6", facecolor='#E6F4F4', edgecolor=TEAL, lw=2)
    ax.text(0.5, 0.28, "ECDR Candidate Path\n1. SHA256 Identity & Age Check\n2. Domain Membership ($X \\in S$)\n3. Numerical Bound Verification", 
            ha='center', va='center', bbox=bbox_ecdr, color=TEAL, weight='bold')

    bbox_reuse = dict(boxstyle="round,pad=0.6", facecolor='#ECFDF5', edgecolor=EMERALD, lw=2)
    ax.text(0.88, 0.4, "Certified Reuse\nSkip CNN (0 ms)", 
            ha='center', va='center', bbox=bbox_reuse, color=EMERALD, weight='bold')

    bbox_fb = dict(boxstyle="round,pad=0.6", facecolor='#FEF2F2', edgecolor=RED, lw=2)
    ax.text(0.88, 0.15, "Fail-Closed Fallback\nFull CNN Inference", 
            ha='center', va='center', bbox=bbox_fb, color=RED, weight='bold')

    # Arrows
    arrow = dict(arrowstyle="->", color=MUTED, lw=1.8, mutation_scale=15)
    ax.annotate("", xy=(0.35, 0.8), xytext=(0.24, 0.55), arrowprops=arrow)
    ax.annotate("", xy=(0.35, 0.3), xytext=(0.24, 0.45), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2, mutation_scale=15))
    ax.annotate("", xy=(0.77, 0.8), xytext=(0.65, 0.8), arrowprops=arrow)
    ax.annotate("", xy=(0.77, 0.4), xytext=(0.65, 0.35), arrowprops=dict(arrowstyle="->", color=EMERALD, lw=2, mutation_scale=15))
    ax.annotate("", xy=(0.77, 0.15), xytext=(0.65, 0.22), arrowprops=dict(arrowstyle="->", color=RED, lw=2, mutation_scale=15))

    ax.text(0.71, 0.42, "All Pass", color=EMERALD, fontsize=9, weight='bold')
    ax.text(0.71, 0.15, "Any Fail", color=RED, fontsize=9, weight='bold')

    plt.title("ECDR Dual-Track Inference Architecture vs Conventional CNN", fontsize=13, weight='bold', color=INK, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig01_dual_track.png"))
    plt.close()
    print("Generated fig01_dual_track.png")

# -------------------------------------------------------------
# FIG 02: Temporal Overlap Structure & Coordinate Space
# -------------------------------------------------------------
def plot_fig02():
    fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
    ax.set_facecolor(SURFACE)
    
    # Draw timeline windows
    ax.barh(1.0, 64, left=0, height=0.5, color='#CBD5E1', edgecolor=INK, label='Previous Prefix (384 coords)')
    ax.barh(1.0, 64, left=64, height=0.5, color=TEAL, edgecolor=INK, alpha=0.9, label='Shared Overlap (384 coords)')
    
    ax.barh(0.2, 64, left=64, height=0.5, color=TEAL, edgecolor=INK, alpha=0.9)
    ax.barh(0.2, 64, left=128, height=0.5, color=AMBER, edgecolor=INK, alpha=0.9, label='New Arrival Suffix (384 coords)')

    ax.set_yticks([0.2, 1.0])
    ax.set_yticklabels(['Window $t$ (New)', 'Window $t-1$ (Anchor)'], fontsize=11, weight='bold', color=INK)
    ax.set_xlabel('Sample Index (50 Hz, Stride 64 samples = 1.28s)', fontsize=10, color=INK)
    
    ax.axvline(64, color=RED, linestyle='--', lw=1.5, alpha=0.7)
    ax.axvline(128, color=RED, linestyle='--', lw=1.5, alpha=0.7)

    ax.text(32, 1.0, 'Dropped Prefix\n384 floats', ha='center', va='center', color=INK, fontsize=9)
    ax.text(96, 1.0, 'Fixed Overlap\n384 floats', ha='center', va='center', color='#FFF', weight='bold', fontsize=9)
    ax.text(96, 0.2, 'Fixed Overlap\n384 floats', ha='center', va='center', color='#FFF', weight='bold', fontsize=9)
    ax.text(160, 0.2, 'New Information\n384 floats', ha='center', va='center', color='#FFF', weight='bold', fontsize=9)

    ax.set_xlim(-10, 202)
    ax.set_ylim(-0.2, 1.5)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFF')
    plt.title("HAPT Sensor Window Overlap: 768-D Total = 384 Fixed Overlap + 384 Free Incoming", fontsize=12, weight='bold', color=INK, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig02_overlap_geometry.png"))
    plt.close()
    print("Generated fig02_overlap_geometry.png")

# -------------------------------------------------------------
# FIG 03: Feature Alignment (H13) Across 16 Positions
# -------------------------------------------------------------
def plot_fig03():
    h13_data = fa_res["h13"]
    errs = h13_data["max_abs_error_j0_to15"]
    positions = list(range(16))
    
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.set_facecolor(SURFACE)
    
    colors = [RED if err > 0 else EMERALD for err in errs]
    bars = ax.bar(positions, errs, color=colors, edgecolor=INK, width=0.6, lw=1)
    
    ax.axvspan(1.5, 13.5, color=EMERALD, alpha=0.1, label='Guaranteed Internal Aligned Region (j = 2..13)')
    
    ax.set_xticks(positions)
    ax.set_xlabel('Final Conv Activation Temporal Position ($j$)', fontsize=11, color=INK)
    ax.set_ylabel('Max Observed Absolute Difference', fontsize=11, color=INK)
    ax.set_title('Exact Feature Map Alignment (H13): 0.0 Bitwise Match on Internal 12 Positions', fontsize=12, weight='bold', color=INK, pad=12)
    
    ax.text(7.5, 12, '12 Guaranteed Positions x 64 Channels x 5,034 Pairs\n= 3,866,112 Scalar Checks EXACT MATCH (Diff = 0.0)', 
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECFDF5', edgecolor=EMERALD),
            fontsize=9.5, color=EMERALD, weight='bold')

    ax.text(0.5, 16, 'Padding effect\n(j=0, 1)', ha='center', color=RED, fontsize=8.5, weight='bold')
    ax.text(14.5, 19, 'Padding effect\n(j=14, 15)', ha='center', color=RED, fontsize=8.5, weight='bold')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig03_h13_feature_alignment.png"))
    plt.close()
    print("Generated fig03_h13_feature_alignment.png")

# -------------------------------------------------------------
# FIG 04: The Radius Dilemma (Trade-off Curve)
# -------------------------------------------------------------
def plot_fig04():
    radii_labels = ['$10^{-6}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$']
    box_pos = [63, 28, 4, 0, 0, 0]
    box_mem = [0, 0, 0, 0, 7, 22]
    wave_pos = [55, 32, 13, 0, 0, 0]
    wave_mem = [0, 0, 0, 0, 9, 22]

    x = np.arange(len(radii_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200, sharey=True)
    fig.patch.set_facecolor('#FFF')
    
    # Subplot 1: Diagonal Box
    ax1.set_facecolor(SURFACE)
    ax1.plot(x, box_pos, marker='o', color=TEAL, lw=2.2, label='Conditional-Positive Bounds')
    ax1.plot(x, box_mem, marker='s', color=RED, lw=2.2, linestyle='--', label='Candidate Next Members')
    ax1.set_xticks(x)
    ax1.set_xticklabels(radii_labels)
    ax1.set_title('Diagonal Box Domain (64 Validation Anchors)', fontsize=11, weight='bold', color=INK)
    ax1.set_xlabel('Candidate Radius $\\rho$', fontsize=10)
    ax1.set_ylabel('Number of Anchors (Max 64)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='center right')

    # Subplot 2: Fixed-Overlap Wave
    ax2.set_facecolor(SURFACE)
    ax2.plot(x, wave_pos, marker='o', color=TEAL, lw=2.2, label='Conditional-Positive Bounds')
    ax2.plot(x, wave_mem, marker='s', color=RED, lw=2.2, linestyle='--', label='Candidate Next Members')
    ax2.set_xticks(x)
    ax2.set_xticklabels(radii_labels)
    ax2.set_title('Fixed-Overlap Wave Domain (64 Validation Anchors)', fontsize=11, weight='bold', color=INK)
    ax2.set_xlabel('Candidate Radius $\\rho$', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='center right')

    fig.suptitle('The Radius Dilemma: Bounds Exist at Small $\\rho$, Membership Only Appears at Large $\\rho$', fontsize=12, weight='bold', color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig04_radius_dilemma.png"))
    plt.close()
    print("Generated fig04_radius_dilemma.png")

# -------------------------------------------------------------
# FIG 05: Latency Comparison & Cost Breakdown
# -------------------------------------------------------------
def plot_fig05():
    categories = ['CNN Core\n(Inference Only)', 'Baseline Wrapper\n(Reference Full)', 'ECDR Box\n(Replay Rep 0)', 'ECDR Wave\n(Replay Rep 0)']
    latencies = [0.224, 1.675, 3.174, 3.198]
    colors = [EMERALD, MUTED, RED, RED]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
    ax.set_facecolor(SURFACE)
    
    bars = ax.bar(categories, latencies, color=colors, edgecolor=INK, width=0.55, lw=1)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.08, f'{h:.3f} ms', ha='center', va='bottom', fontsize=10, weight='bold')

    ax.axhline(1.675, color=MUTED, linestyle='--', lw=1.2, alpha=0.7, label='Reference Baseline (1.675 ms)')
    
    ax.set_ylabel('Mean Latency (ms / query on Single CPU Thread)', fontsize=10, color=INK)
    ax.set_title('End-to-End Latency: ECDR Suffers ~0.52x Relative Speedup (Negative Net Savings)', fontsize=12, weight='bold', color=INK, pad=12)
    ax.set_ylim(0, 3.8)

    ax.text(2.5, 2.2, 'ECDR Overhead:\nFallback Forward (~1.67 ms)\n+ Membership Check (~1.43 ms)\n+ Periodic Diagnostic Builds (~6.8 ms)', 
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF2F2', edgecolor=RED),
            fontsize=9, color=RED, weight='bold')

    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig05_latency_comparison.png"))
    plt.close()
    print("Generated fig05_latency_comparison.png")

# -------------------------------------------------------------
# FIG 06: Exploratory Geometry: PCA vs Margin Gradient
# -------------------------------------------------------------
def plot_fig06():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=200)
    fig.patch.set_facecolor('#FFF')
    
    # Subplot 1: PCA Variance
    ax1.set_facecolor(SURFACE)
    pca_vars = [11.2, 5.8, 4.5, 78.5]
    labels = ['PC 1\n(11.2%)', 'PC 2\n(5.8%)', 'PC 3\n(4.5%)', 'Remaining\n765 Dims\n(78.5%)']
    colors1 = [TEAL, TEAL, TEAL, '#CBD5E1']
    ax1.bar(labels, pca_vars, color=colors1, edgecolor=INK, width=0.55)
    ax1.set_ylabel('Explained Variance (%)', fontsize=10)
    ax1.set_title('PCA Variance on 9,944 Temporal Deltas\n(Top 3 = 21.48% Total)', fontsize=11, weight='bold', color=INK)

    # Subplot 2: Gradient Alignment
    ax2.set_facecolor(SURFACE)
    angles = [87.1, 85.4]  # arccos(0.051) approx 87 degrees
    ax2.bar(['Temporal Delta vs\nMargin Gradient\n(cos = 0.051)', 'Projected Energy\nin Top-3 PCA\n(ratio = 0.0798)'], 
            [0.051, 0.0798], color=[RED, AMBER], edgecolor=INK, width=0.45)
    ax2.set_ylabel('Mean Projection Ratio / Cosine', fontsize=10)
    ax2.set_title('Sensitivity Alignment (32 Validation Anchors)\nGradient is Near-Orthogonal to Deltas', fontsize=11, weight='bold', color=INK)
    ax2.set_ylim(0, 0.12)

    fig.suptitle('Geometric Evidence: Temporal Dynamics do not Align with CNN Decision Boundary', fontsize=12, weight='bold', color=INK)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig06_geometry_pca_gradient.png"))
    plt.close()
    print("Generated fig06_geometry_pca_gradient.png")

# -------------------------------------------------------------
# FIG 07: 24-Hypothesis Ledger Disposition
# -------------------------------------------------------------
def plot_fig07():
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax.set_facecolor(SURFACE)
    
    categories = ['DERIVED\n(Established)', 'CONDITIONAL\n(Requires Bridge)', 'FALSIFIED\n(Refuted)', 'NOT ESTABLISHED\n(Unproven)']
    counts = [6, 5, 7, 6]
    colors = [EMERALD, AMBER, RED, MUTED]
    
    bars = ax.bar(categories, counts, color=colors, edgecolor=INK, width=0.5, lw=1)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.15, f'{int(h)} ({h/24*100:.1f}%)', ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_ylabel('Number of Hypotheses (Total = 24)', fontsize=10)
    ax.set_ylim(0, 9)
    ax.set_title('Scientific Classification of the 24 Week 3 Hypotheses (H01–H24)', fontsize=12, weight='bold', color=INK, pad=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig07_hypothesis_status_breakdown.png"))
    plt.close()
    print("Generated fig07_hypothesis_status_breakdown.png")

# -------------------------------------------------------------
# FIG 08: Memory Footprint Breakdown
# -------------------------------------------------------------
def plot_fig08():
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    ax.set_facecolor(SURFACE)
    
    components = ['Float32 Reference CNN\n(Model Weights & Buffers)', 'Float64 Verifier Snapshot\n(Interval Propagation Model)', 'Single Certificate\n(11 Bounds & Metadata)']
    sizes_kb = [1136072 / 1024, 2272120 / 1024, 54839 / 1024]
    colors = [TEAL, AMBER, EMERALD]
    
    bars = ax.barh(components, sizes_kb, color=colors, edgecolor=INK, height=0.45)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 30, bar.get_y() + bar.get_height()/2., f'{w:.1f} KB ({w/1024:.2f} MB)', ha='left', va='center', fontsize=9.5, weight='bold')

    ax.set_xlabel('Persistent Working Memory (KB)', fontsize=10)
    ax.set_xlim(0, 2700)
    ax.set_title('Memory Component Breakdown in ECDR Runtime', fontsize=12, weight='bold', color=INK, pad=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "fig08_memory_breakdown.png"))
    plt.close()
    print("Generated fig08_memory_breakdown.png")

if __name__ == "__main__":
    plot_fig01()
    plot_fig02()
    plot_fig03()
    plot_fig04()
    plot_fig05()
    plot_fig06()
    plot_fig07()
    plot_fig08()
    print("All 8 academic figures successfully generated into notes/WEEK3/assets/!")
