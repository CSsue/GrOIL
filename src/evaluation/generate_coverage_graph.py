"""
Paper-ready keyphrase coverage diagram — v4
- Compact figure, larger fonts
- Right panel header: "Mapped / Total Keyphrases"
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import os

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          10,
    "axes.labelsize":     10,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "text.usetex":        False,
})

# ── Data ──────────────────────────────────────────────────────────────────────
datasets = [
    {"name": "Sentinel",  "total": 167,  "exact": 78,   "fuzzy": 48,   "sem": 18,  "unmapped": 23},
    {"name": "Equivita",  "total": 177,  "exact": 87,   "fuzzy": 56,   "sem": 13,  "unmapped": 21},
    {"name": "10-contracts",      "total": 416,  "exact": 127,  "fuzzy": 188,  "sem": 54,  "unmapped": 47},
    {"name": "100-contracts",     "total": 6407, "exact": 1290, "fuzzy": 3467, "sem": 849, "unmapped": 801},
]

labels  = [d["name"]  for d in datasets]
totals  = np.array([d["total"]    for d in datasets], dtype=float)
exact_n = np.array([d["exact"]    for d in datasets], dtype=float)
fuzzy_n = np.array([d["fuzzy"]    for d in datasets], dtype=float)
sem_n   = np.array([d["sem"]      for d in datasets], dtype=float)
unmap_n = np.array([d["unmapped"] for d in datasets], dtype=float)
mapped_n = exact_n + fuzzy_n + sem_n

exact_p  = exact_n  / totals * 100
fuzzy_p  = fuzzy_n  / totals * 100
sem_p    = sem_n    / totals * 100
unmap_p  = unmap_n  / totals * 100
coverage = mapped_n / totals * 100

C_EXACT    = "#08519C"
C_FUZZY    = "#3182BD"
C_SEM      = "#9ECAE1"
C_UNMAPPED = "#EEEEEE"
C_EDGE     = "#BBBBBB"

# ── Figure: compact, wider right panel ───────────────────────────────────────
fig = plt.figure(figsize=(4.8, 2))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.2], wspace=0.1,
                       left=0.22, right=0.98, top=0.90, bottom=0.32)
ax  = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

y      = np.arange(len(labels))
height = 0.7

# ── Left: stacked bars ────────────────────────────────────────────────────────
ax.barh(y, exact_p,  height,                             color=C_EXACT)
ax.barh(y, fuzzy_p,  height, left=exact_p,               color=C_FUZZY)
ax.barh(y, sem_p,    height, left=exact_p+fuzzy_p,       color=C_SEM)
ax.barh(y, unmap_p,  height, left=exact_p+fuzzy_p+sem_p, color=C_UNMAPPED)

def bar_label(lefts, widths, counts, bg_color, min_w=7.0):
    lum = sum(int(bg_color.lstrip("#")[j*2:j*2+2], 16) * f
              for j, f in zip(range(3), [0.299, 0.587, 0.114]))
    tc = "white" if lum < 155 else "#555555"
    for i, (l, w, n) in enumerate(zip(lefts, widths, counts)):
        if w < min_w:
            continue
        ax.text(l + w/2, y[i], f"{int(n)}",
                ha="center", va="center", fontsize=7.5,
                color=tc, fontweight="bold")

bar_label(np.zeros(len(y)),      exact_p, exact_n, C_EXACT)
bar_label(exact_p,               fuzzy_p, fuzzy_n, C_FUZZY)
bar_label(exact_p+fuzzy_p,       sem_p,   sem_n,   C_SEM,    min_w=4)
bar_label(exact_p+fuzzy_p+sem_p, unmap_p, unmap_n, C_UNMAPPED, min_w=5)

ax.axvline(100, color="#AAAAAA", linewidth=0.6, linestyle="--", zorder=0)
ax.xaxis.grid(True, linestyle=":", linewidth=0.35, color="#DDDDDD", zorder=0)
ax.set_axisbelow(True)

ax.set_yticks(y)
ax.set_yticklabels(labels)
# ax.set_xlabel("Percentage of keyphrases (%)", labelpad=4)
ax.set_xlim(0, 105)
ax.set_xticks([0, 25, 50, 75, 100])
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.invert_yaxis()
ax.spines["left"].set_linewidth(0.5)
ax.spines["bottom"].set_linewidth(0.5)
ax.tick_params(axis="both", length=2.5, width=0.5, pad=3)

# ── Right panel ───────────────────────────────────────────────────────────────
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.55, len(y) - 0.45)
ax2.axis("off")
ax2.invert_yaxis()

# Two sub-column headers
HDR_Y = 4.2
ax2.text(0.26, HDR_Y, "Mapped/Total",
         ha="center", va="bottom", fontsize=7.5,
         fontstyle="italic", color="#1B1A1A", linespacing=1.3)
ax2.text(0.76, HDR_Y, "Coverage",
         ha="center", va="bottom", fontsize=7.5,
         fontstyle="italic", color="#1B1A1A")

ax2.axvline(0.52, color="#DDDDDD", linewidth=0.5)

for i in range(len(datasets)):
    mapped = int(mapped_n[i])
    total  = int(totals[i])
    cov    = coverage[i]

    ax2.text(0.26, y[i], f"{mapped}/{total}",
             ha="center", va="center", fontsize=8,
             color="#222222", fontfamily="monospace")

    ax2.text(0.76, y[i], f"{cov:.1f}%",
             ha="center", va="center", fontsize=9.5,
             fontweight="bold", color="#08519C",
             fontfamily="sans-serif")

for sep in np.arange(0.5, len(y) - 0.5, 1.0):
    ax2.axhline(sep, color="#EEEEEE", linewidth=0.5)

# Thin border around right panel
for spine_pos in [(0, 0, 0, 1), (1, 0, 1, 1), (0, 0, 1, 0)]:
    ax2.plot([spine_pos[0], spine_pos[2]], [spine_pos[1]*len(y)-0.55, spine_pos[3]*len(y)-0.55+0.1*(spine_pos[3]-spine_pos[1])],
             color="#CCCCCC", linewidth=0.4, transform=ax2.transAxes)

# Vertical line separating left from right panel
ax2.spines["left"].set_visible(True)
ax2.spines["left"].set_linewidth(0.5)
ax2.spines["left"].set_color("#CCCCCC")

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor=C_EXACT,    label="Exact / substring match"),
    mpatches.Patch(facecolor=C_FUZZY,    label="Fuzzy match"),
    mpatches.Patch(facecolor=C_SEM,      label="Semantic match"),
    mpatches.Patch(facecolor=C_UNMAPPED, edgecolor=C_EDGE, linewidth=0.5, label="Unmapped"),
]
fig.legend(
    handles=patches,
    loc="lower center",
    bbox_to_anchor=(0.50, -0.01),
    ncol=4,
    frameon=False,
    fontsize=8.5,
    handlelength=1.0,
    handletextpad=0.45,
    columnspacing=1.0,
)

os.makedirs("out/ontology_coverage", exist_ok=True)
fig.savefig("out/ontology_coverage/coverage_diagram.png")
print("Saved: out/ontology_coverage/coverage_diagram.png")