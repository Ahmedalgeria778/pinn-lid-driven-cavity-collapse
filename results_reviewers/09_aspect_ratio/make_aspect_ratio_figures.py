# -*- coding: utf-8 -*-
"""
Figures d'analyse du cas 09_aspect_ratio (R3.2, clos).
Cavité 1:1 vs 2:1, Re=500, seed 42, E*=0.25, base 6x5 Fourier.
Génère :
  - fig_AR1_modal_structure.png   : A1/A2/A3 + f(2,0), f_temp, E_total (barres)
  - fig_AR2_lid_profiles.png      : contrôles identifiés U_lid(s), s=x/Lx
Lancement : py -3.11 results_reviewers/09_aspect_ratio/make_aspect_ratio_figures.py
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
LABEL = {"square": "1:1 (carré)", "rectangular": "2:1 (rectangulaire)"}


def dark_axes(ax, xl="", yl="", title="", fs=9):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="w", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    ax.grid(True, alpha=0.15, color="#333")
    if xl:
        ax.set_xlabel(xl, color="w", fontsize=fs)
    if yl:
        ax.set_ylabel(yl, color="w", fontsize=fs)
    if title:
        ax.set_title(title, color="w", fontsize=fs, fontweight="bold")


# Lecture du CSV (les vraies variables utiles)
rows = {}
with open(CSV, newline="") as f:
    rd = csv.DictReader(f)
    for r in rd:
        rows[r["geometry"]] = r

# ----------------------------------------------------------------------
# Figure AR1 : structure modale (A1, A2, A3) + fractions + énergie
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=DARK)
xpos = np.arange(3)
w = 0.34
for i, geo in enumerate(["square", "rectangular"]):
    r = rows[geo]
    amps = [float(r[k]) for k in ("A10", "A20", "A30")]
    ax1.bar(xpos + (i - 0.5) * w, amps, w, color=C[geo], alpha=0.9, label=LABEL[geo])
ax1.set_xticks(xpos)
ax1.set_xticklabels(["$A_1$", "$A_2$", "$A_3$"], color="w")
dark_axes(ax1, yl="Amplitude mode", title="Coefficients de contrôle $A_i = c_{i-1,0}$")
ax1.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=8)

geo = ["square", "rectangular"]
f2 = [float(rows[g]["mode2_fraction"]) * 100 for g in geo]
ft = [float(rows[g]["temporal_fraction"]) * 100 for g in geo]
x2 = np.arange(2)
ax2.bar(x2 - 0.19, ft, 0.38, color="#7dff6b", alpha=0.85, label="$f_{temp}$ (%)")
ax2.bar(x2 + 0.19, f2, 0.38, color="#ffcc00", alpha=0.9, label="$f_{(2,0)}$ (%)")
for xi, (a, b) in enumerate(zip(f2, ft)):
    ax2.annotate(f"{a:.1f}%", (xi + 0.19, a + 0.8), color="w", fontsize=9, ha="center")
    ax2.annotate(f"{b:.1f}%", (xi - 0.19, b + 0.8), color="#7dff6b", fontsize=9, ha="center")
ax2.set_xticks(x2)
ax2.set_xticklabels([LABEL[g] for g in geo], color="w")
dark_axes(ax2, yl="Fraction d'énergie (%)", title="Collapse Mode-2 / branches ($E_{{total}}$={:.3f}//{:.3f})".format(
    float(rows["square"]["E_total"]), float(rows["rectangular"]["E_total"])))
ax2.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=8)
fig.suptitle("Aspect ratio (R3.2) — cavité 1:1 vs 2:1, Re=500, seed 42 : le collapse Mode-2 persiste",
             color="w", fontsize=12, fontweight="bold")
fig.savefig(os.path.join(BASE, "fig_AR1_modal_structure.png"), dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   -> fig_AR1_modal_structure.png")

# ----------------------------------------------------------------------
# Figure AR2 : contrôles identifiés U_lid(s), s = x/Lx (moyenne temporelle)
# U(s) = sum_i A_i sin((i+1)*pi*s) — même forme modale normalisée à la longueur
# ----------------------------------------------------------------------
s = np.linspace(0, 1, 400)
fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK)
for geo in ["square", "rectangular"]:
    r = rows[geo]
    U = float(r["A10"]) * np.sin(np.pi * s) + float(r["A20"]) * np.sin(2 * np.pi * s) \
        + float(r["A30"]) * np.sin(3 * np.pi * s)
    ax.plot(s, U, color=C[geo], lw=2.5, label=f'{LABEL[geo]}   ($A_2$={float(r["A20"]):.3f})')
ax.plot(s, np.sin(2 * np.pi * s), color="white", ls="--", lw=1.4, alpha=0.75, label="$\sin(2\pi s)$ (repère mode-2)")
ax.axhline(0, color="#555", lw=0.8)
dark_axes(ax, xl="$s = x/L_x$", yl="$U_{lid}(s)$ (moy. temp.)", title="Contrôles identifiés 1:1 vs 2:1 — mode-2 quasi pur, invariant géométrique")
ax.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=9)
fig.savefig(os.path.join(BASE, "fig_AR2_lid_profiles.png"), dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print("   -> fig_AR2_lid_profiles.png")
print("Figures aspect ratio OK.")