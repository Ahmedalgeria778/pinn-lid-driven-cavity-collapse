# -*- coding: utf-8 -*-
"""
Manuscript aspect-ratio figures (3 geometries: AR = 0.5, 1, 2), Re=500,
seed 42, E*=0.25, base 6x5 Fourier control space.

Regenerates the two public figures stored under manuscript/figures:
  - fig6_aspect_ratio.png : (a) A1,A2,A3 bars ; (b) f2 and ft bars
  - fig7_lid_profiles.png : three-panel lid profiles U_lid(s), s = x/Lx

White / black-and-white print-friendly style. 300 dpi. Input: the tracked
aspect_ratio_results.csv (results_reviewers/09_aspect_ratio).

Run: py -3.11 results_reviewers/09_aspect_ratio/make_manuscript_ar_figures.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
CSV = os.path.join(BASE, "aspect_ratio_results.csv")
sys.path.insert(0, os.path.join(ROOT, "lbm_mrt_validation"))
from figure_style import style_axes, legend, save_fig  # noqa: E402

FIG_DIR = os.path.join(ROOT, "manuscript", "figures")

GEO_LIST = ["short_rect", "square", "rectangular"]
LABEL = {"short_rect": "0.5", "square": "1", "rectangular": "2"}
XTICK = {"short_rect": "$AR=0.5$", "square": "$AR=1$", "rectangular": "$AR=2$"}
GEO_HATCH = {"short_rect": "///", "square": "", "rectangular": "..."}
GEO_FILL = {"short_rect": "#ffffff", "square": "#c8c8c8", "rectangular": "#ffffff"}


rows = {}
with open(CSV, newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        rows[r["geometry"]] = r

# ----------------------------------------------------------------------
# Figure 1: (a) A1, A2, A3  --  (b) f2 and ft for the three aspect ratios
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
xpos = np.arange(3)
w = 0.30
for i, geo in enumerate(GEO_LIST):
    r = rows[geo]
    amps = [float(r[k]) for k in ("A10", "A20", "A30")]
    ax1.bar(xpos + (i - 1) * w, amps, w, color=GEO_FILL[geo],
            edgecolor="k", hatch=GEO_HATCH[geo], label=XTICK[geo])
ax1.set_xticks(xpos)
ax1.set_xticklabels(["$A_1$", "$A_2$", "$A_3$"], color="k", fontsize=13)
style_axes(ax1, yl="Control amplitude (--)",
           title="(a)  Stationary spatial coefficients $A_i$")
legend(ax1, fs=11, ncol=3)

f2 = [float(rows[g]["mode2_fraction"]) * 100 for g in GEO_LIST]
ft = [float(rows[g]["temporal_fraction"]) * 100 for g in GEO_LIST]
x2 = np.arange(3)
ax2.bar(x2 - 0.19, ft, 0.38, color="#8a8a8a", edgecolor="k", hatch="...",
        label="$f_t$ (%)")
ax2.bar(x2 + 0.19, f2, 0.38, color="#ffffff", edgecolor="k", hatch="///",
        label="$f_2$ (%)")
for xi, (a, b) in enumerate(zip(f2, ft)):
    ax2.annotate("%.1f" % a, (xi + 0.19, a + 0.8), color="k", fontsize=12,
                 ha="center")
    ax2.annotate("%.1f" % b, (xi - 0.19, b + 0.8), color="k", fontsize=12,
                 ha="center")
ax2.set_xticks(x2)
ax2.set_xticklabels([XTICK[g] for g in GEO_LIST], color="k", fontsize=13)
style_axes(ax2, yl="Energy fraction (%)",
           title="(b)  Modal energy fractions $f_2$ and $f_t$")
legend(ax2, fs=11, ncol=2, loc="lower right")
fig.tight_layout()
save_fig(fig, os.path.join(FIG_DIR, "fig6_aspect_ratio.png"))
print("   -> manuscript/figures/fig6_aspect_ratio.png  (AR = 0.5/1/2)")

# ----------------------------------------------------------------------
# Figure 2: lid profiles U_lid(s), s = x/Lx, one panel per aspect ratio
# ----------------------------------------------------------------------
s = np.linspace(0, 1, 500)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
for ax, geo in zip(axes, GEO_LIST):
    r = rows[geo]
    A2 = float(r["A20"])
    cs = [float(r["c_%d_0" % i]) for i in range(6)]
    mode = lambda m: sum(cs[i] * np.sin((i + 1) * np.pi * s) for i in range(m))
    ax.plot(s, A2 * np.sin(2 * np.pi * s), color="0.45", ls="--", lw=1.7,
            marker="o", markevery=28, ms=5, label="Reference $A_2\\sin(2\\pi s)$")
    ax.plot(s, mode(6), color="k", lw=2.4, marker="*", markevery=28, ms=9,
            label="PINN control (all modes)")
    ax.plot(s, mode(3), color="k", ls=":", lw=2.2, marker="s", markevery=28,
            ms=6, label="Reduced $A_1$+$A_2$+$A_3$ (3 modes)")
    ax.axhline(0, color="0.5", lw=0.8)
    style_axes(ax, xl="$s=x/L_x$ (--)", yl="$U_{lid}(s)$ (--)",
               title="( %s )  $AR=%s$" % (geo[0], LABEL[geo]))
    legend(ax, fs=10, loc="upper right")
fig.tight_layout()
save_fig(fig, os.path.join(FIG_DIR, "fig7_lid_profiles.png"))
print("   -> manuscript/figures/fig7_lid_profiles.png  (AR = 0.5/1/2)")
print("Manuscript aspect-ratio figures OK.")