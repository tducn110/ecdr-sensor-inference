# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Test Pirsch colors
CREAM = '#f8f5ed'
INK = '#000000'
GRAPHITE = '#707070'
SUNSHINE = '#ffda6e'
MINT = '#6ece9d'

fig, ax = plt.subplots(figsize=(6, 2), facecolor=CREAM)
ax.set_facecolor(CREAM)
ax.text(0.5, 0.5, "Pirsch Style Test", color=INK, fontsize=16, ha='center', va='center')
ax.axis('off')
plt.savefig('/home/pro/Downloads/research/notes/WEEK2/assets/test_pirsch.png', facecolor=CREAM)
plt.close()
print("Pirsch test image generated.")
