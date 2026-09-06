# -*- coding: utf-8 -*-
"""
Aspect-ratio analysis figures (R3.2, closed).
Cavity 1:1 vs 2:1, Re=500, seed 42, E*=0.25, base 6x5 Fourier.
Generates:
  - fig_AR1_modal_structure.png  : A1/A2/A3 + f_(2,0), f_temp, E_total (bars)
  - fig_AR2_lid_profiles.png     : identified controls U_lid(s), s = x/Lx
        comparison: A2.sin(2.pi.s) reference / PINN control (all modes) /
                    reduced A1+A2+A3 (3 modes) -- one panel per geometry
Run: py -3.11 results_reviewers/09_aspect_ratio/make_aspect_ratio_figures.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "aspect_ratio_results.csv")

DARK, PANEL = "#090909", "#111111"
C = {"square": "#00e5ff", "rectangular": "#ff6b35"}
LABEL = {"square": "Square (1:1)", "rectangular": "Rectangular (2:1)"}


def dark_axes(ax, xl="", yl="", title="", fs=13):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="w", labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    ax.grid(True, alpha=0.15, color="#333")
    if xl:
        ax.set_xlabel(xl, color="w", fontsize=fs)
    if yl:
        ax.set_ylabel(yl, color="w", fontsize=fs)
    if title:
        ax.set_title(title, color="w", fontsize=fs, fontweight="bold")


rows = {}
with open(CSV, newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        rows[r["geometry"]] = r

# ----------------------------------------------------------------------
# Figure AR1: modal structure (A1, A2, A3) + fractions + energy
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK)
xpos = np.arange(3)
w = 0.34
for i, geo in enumerate(["square", "rectangular"]):
    r = rows[geo]
    amps = [float(r[k]) for k in ("A10", "A20", "A30")]
    ax1.bar(xpos + (i - 0.5) * w, amps, w, color=C[geo], alpha=0.9, label=LABEL[geo])
ax1.set_xticks(xpos)
ax1.set_xticklabels(["$A_1$", "$A_2$", "$A_3$"], color="w", fontsize=12)
dark_axes(ax1, yl="Control amplitude (–)", title="Control coefficients $A_i = c_{i-1,0}$ (–)")
ax1.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=10)

geo = ["square", "rectangular"]
f2 = [float(rows[g]["mode2_fraction"]) * 100 for g in geo]
ft = [float(rows[g]["temporal_fraction"]) * 100 for g in geo]
x2 = np.arange(2)
ax2.bar(x2 - 0.19, ft, 0.38, color="#7dff6b", alpha=0.85, label="$f_{temp}$ (%)")
ax2.bar(x2 + 0.19, f2, 0.38, color="#ffcc00", alpha=0.9, label="$f_{(2,0)}$ (%)")
for xi, (a, b) in enumerate(zip(f2, ft)):
    ax2.annotate(f"{a:.1f}%", (xi + 0.19, a + 0.8), color="w", fontsize=11, ha="center")
    ax2.annotate(f"{b:.1f}%", (xi - 0.19, b + 0.8), color="#7dff6b", fontsize=11, ha="center")
ax2.set_xticks(x2)
ax2.set_xticklabels([LABEL[g] for g in geo], color="w", fontsize=12)
dark_axes(ax2, yl="Energy fraction (%)",
          title="Mode-2 collapse ($E_{{tot}}$ = {:.3f} vs {:.3f})".format(
              float(rows["square"]["E_total"]), float(rows["rectangular"]["E_total"])))
ax2.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=10)
fig.suptitle("Aspect-ratio robustness (R3.2) — 1:1 vs 2:1 cavities (Re=500, seed 42): "
             "Mode-2 collapse persists",
             color="w", fontsize=15, fontweight="bold")
fig.savefig(os.path.join(BASE, "fig_AR1_modal_structure.png"), dpi=300,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("   -> fig_AR1_modal_structure.png")

# ----------------------------------------------------------------------
# Figure AR2: identified controls U_lid(s), s = x/Lx (one panel per geometry)
# U(s) = sum_i c_{i,0} sin((i+1)*pi*s) -- same modal form normalized by length
# Comparison per panel:
#   1. reference    A2.sin(2.pi.s)
#   2. PINN control (all modes, i = 0..5)
#   3. reduced      A1 + A2 + A3  (3 modes, i = 0..2)
# ----------------------------------------------------------------------
s = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), facecolor=DARK)
for ax, geo in zip(axes, ["square", "rectangular"]):
    r = rows[geo]
    A2 = float(r["A20"])
    cs = [float(r[f"c_{i}_0"]) for i in range(6)]
    mode = lambda m: sum(cs[i] * np.sin((i + 1) * np.pi * s) for i in range(m))
    ax.plot(s, A2 * np.sin(2 * np.pi * s), color="white", ls="--", lw=1.6,
            alpha=0.8, label="Reference $A_2\\sin(2\\pi s)$")
    ax.plot(s, mode(6), color="#00e5ff", lw=2.4,
            label="PINN control (all modes)")
    ax.plot(s, mode(3), color="#ffcc00", ls=":", lw=2.2,
            label="Reduced $A_1$+$A_2$+$A_3$ (3 modes)")
    ax.axhline(0, color="#555", lw=0.8)
    dark_axes(ax, xl="$s = x/L_x$ (–)", yl="$U_{lid}(s)$ (–)",
              title=f"{LABEL[geo]} ($A_2$ = {A2:.3f})")
    ax.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=10)
fig.suptitle("Identified controls 1:1 vs 2:1 — quasi-pure mode-2, geometry-invariant "
             "(reconstructions of $U_{lid}$)",
             color="w", fontsize=15, fontweight="bold")
fig.savefig(os.path.join(BASE, "fig_AR2_lid_profiles.png"), dpi=300,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("   -> fig_AR2_lid_profiles.png")
print("Aspect-ratio figures OK.")