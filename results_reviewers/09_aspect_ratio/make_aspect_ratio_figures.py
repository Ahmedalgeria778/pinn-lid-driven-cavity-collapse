# -*- coding: utf-8 -*-
"""
Aspect-ratio analysis figures (R3.2, closed).
Cavity 1:1 vs 2:1, Re=500, seed 42, E*=0.25, base 6x5 Fourier.
Generates:
  - fig_AR1_modal_structure.png  : A1/A2/A3 + f_(2,0), f_temp, E_total (bars)
  - fig_AR2_lid_profiles.png     : identified controls U_lid(s), s = x/Lx
        comparison: A2.sin(2.pi.s) reference / PINN control (all modes) /
                    reduced A1+A2+A3 (3 modes) -- one panel per geometry

White / black-and-white print-friendly style (AE request): white background,
black curves + symbols, "(-)" units, 300 dpi.

Run: py -3.11 results_reviewers/09_aspect_ratio/make_aspect_ratio_figures.py
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(BASE)), "lbm_mrt_validation"))
from figure_style import style_axes, figure_title, legend, save_fig  # noqa: E402

CSV = os.path.join(BASE, "aspect_ratio_results.csv")

LABEL = {"square": "Square (1:1)", "rectangular": "Rectangular (2:1)"}
GEO_HATCH = {"square": "", "rectangular": "///"}
GEO_FILL = {"square": "#c8c8c8", "rectangular": "#ffffff"}


rows = {}
with open(CSV, newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        rows[r["geometry"]] = r

# ----------------------------------------------------------------------
# Figure AR1: modal structure (A1, A2, A3) + fractions + energy
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
xpos = np.arange(3)
w = 0.34
for i, geo in enumerate(["square", "rectangular"]):
    r = rows[geo]
    amps = [float(r[k]) for k in ("A10", "A20", "A30")]
    ax1.bar(xpos + (i - 0.5) * w, amps, w, color=GEO_FILL[geo], edgecolor="k",
            hatch=GEO_HATCH[geo], label=LABEL[geo])
ax1.set_xticks(xpos)
ax1.set_xticklabels(["$A_1$", "$A_2$", "$A_3$"], color="k", fontsize=12)
style_axes(ax1, yl="Control amplitude (–)", title="Control coefficients $A_i = c_{i-1,0}$ (–)")
legend(ax1, fs=10)

geo = ["square", "rectangular"]
f2 = [float(rows[g]["mode2_fraction"]) * 100 for g in geo]
ft = [float(rows[g]["temporal_fraction"]) * 100 for g in geo]
x2 = np.arange(2)
ax2.bar(x2 - 0.19, ft, 0.38, color="#8a8a8a", edgecolor="k", hatch="...",
        label="$f_{temp}$ (%)")
ax2.bar(x2 + 0.19, f2, 0.38, color="#ffffff", edgecolor="k", hatch="///",
        label="$f_{(2,0)}$ (%)")
for xi, (a, b) in enumerate(zip(f2, ft)):
    ax2.annotate(f"{a:.1f}%", (xi + 0.19, a + 0.8), color="k", fontsize=11, ha="center")
    ax2.annotate(f"{b:.1f}%", (xi - 0.19, b + 0.8), color="k", fontsize=11, ha="center")
ax2.set_xticks(x2)
ax2.set_xticklabels([LABEL[g] for g in geo], color="k", fontsize=12)
style_axes(ax2, yl="Energy fraction (%)",
           title="Mode-2 collapse ($E_{{tot}}$ = {:.3f} vs {:.3f})".format(
               float(rows["square"]["E_total"]), float(rows["rectangular"]["E_total"])))
legend(ax2, fs=10)
figure_title(fig, "Aspect-ratio robustness (R3.2) — 1:1 vs 2:1 cavities (Re=500, seed 42): "
             "Mode-2 collapse persists")
save_fig(fig, os.path.join(BASE, "fig_AR1_modal_structure.png"))
print("   -> fig_AR1_modal_structure.png")

# ----------------------------------------------------------------------
# Figure AR2: identified controls U_lid(s), s = x/Lx (one panel per geometry)
#    1. reference    A2.sin(2.pi.s)
#    2. PINN control (all modes, i = 0..5)
#    3. reduced      A1 + A2 + A3  (3 modes, i = 0..2)
# ----------------------------------------------------------------------
s = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, geo in zip(axes, ["square", "rectangular"]):
    r = rows[geo]
    A2 = float(r["A20"])
    cs = [float(r[f"c_{i}_0"]) for i in range(6)]
    mode = lambda m: sum(cs[i] * np.sin((i + 1) * np.pi * s) for i in range(m))
    ax.plot(s, A2 * np.sin(2 * np.pi * s), color="0.45", ls="--", lw=1.6,
            marker="o", markevery=22, ms=5, label="Reference $A_2\\sin(2\\pi s)$")
    ax.plot(s, mode(6), color="k", lw=2.4, marker="*", markevery=22, ms=9,
            label="PINN control (all modes)")
    ax.plot(s, mode(3), color="k", ls=":", lw=2.2, marker="s", markevery=22, ms=6,
            label="Reduced $A_1$+$A_2$+$A_3$ (3 modes)")
    ax.axhline(0, color="0.5", lw=0.8)
    style_axes(ax, xl="$s = x/L_x$ (–)", yl="$U_{lid}(s)$ (–)",
               title=f"{LABEL[geo]} ($A_2$ = {A2:.3f})")
    legend(ax, fs=10)
figure_title(fig, "Identified controls 1:1 vs 2:1 — quasi-pure mode-2, geometry-invariant "
             "(reconstructions of $U_{lid}$)")
save_fig(fig, os.path.join(BASE, "fig_AR2_lid_profiles.png"))
print("   -> fig_AR2_lid_profiles.png")
print("Aspect-ratio figures OK (white / B&W style).")