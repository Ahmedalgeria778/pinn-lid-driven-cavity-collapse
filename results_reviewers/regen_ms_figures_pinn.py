# -*- coding: utf-8 -*-
"""Regenerate the PINN-side manuscript figures in the shared AE publication
style (white background, black curves, marker-distinguished cases, "(-)" units
on every axis, axis labels 14 pt, ticks 12 pt).

Figures written directly into manuscript/figures/:
  fig1_modal_fractions.png      f_2 vs Re, Fourier vs modified-Chebyshev
  fig2_temporal_fractions.png   f_t vs Re, Fourier vs modified-Chebyshev
  fig5_energy_sweep.png         f_2 and A_2 vs actuation-energy target E*

Only regenerates the three figures that predate the AE style. Data come from
the published campaign CSVs, so the content reproduces the reported numbers.

Run:  py -3.11 results_reviewers/regen_ms_figures_pinn.py
"""
import os
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "lbm_mrt_validation"))

from figure_style import style_axes, legend, save_fig  # noqa: E402

OUTDIR = os.path.join(ROOT, "manuscript", "figures")
RES = os.path.join(ROOT, "results_reviewers")

FOUR = {"ls": "-", "mk": "o"}
CHEB = {"ls": "--", "mk": "s"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- Fourier vs modified-Chebyshev: f_2 and f_t vs Re -------------------
    par = read_csv(os.path.join(RES, "13_parametrization", "parametrization_results.csv"))
    p_fou = {int(r["Re"]): r for r in par if r["basis_type"] == "fourier"}
    p_cheb = {int(r["Re"]): r for r in par if r["basis_type"] == "chebyshev_mod"}
    re_list = sorted(p_fou)
    x = np.array(re_list, dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for src, st in ((p_fou, FOUR), (p_cheb, CHEB)):
        y = np.array([float(src[r]["fourier_mode2_fraction"]) * 100.0 for r in re_list])
        ax.plot(x, y, ls=st["ls"], marker=st["mk"], color="k", lw=2.0,
                ms=7, label="Fourier" if src is p_fou else "Modified-Chebyshev")
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), fontsize=10, ha="center", color="k")
    style_axes(ax, xl="Re (-)", yl="Second-mode fraction $f_2$ (%)",
               title="Second-mode energy fraction")
    ax.set_xscale("log")
    ax.set_xticks(re_list)
    ax.set_xticklabels([str(r) for r in re_list], fontsize=12)
    ax.set_ylim(0, 105)
    legend(ax, fs=12, loc="lower right")
    save_fig(fig, os.path.join(OUTDIR, "fig1_modal_fractions.png"))
    print("-> fig1_modal_fractions.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for src, st in ((p_fou, FOUR), (p_cheb, CHEB)):
        y = np.array([float(src[r]["temporal_fraction"]) * 100.0 for r in re_list])
        ax.plot(x, y, ls=st["ls"], marker=st["mk"], color="k", lw=2.0,
                ms=7, label="Fourier" if src is p_fou else "Modified-Chebyshev")
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), fontsize=10, ha="center", color="k")
    style_axes(ax, xl="Re (-)", yl="Time-dependent fraction $f_t$ (%)",
               title="Time-dependent energy fraction")
    ax.set_xscale("log")
    ax.set_xticks(re_list)
    ax.set_xticklabels([str(r) for r in re_list], fontsize=12)
    ax.set_ylim(0, 105)
    legend(ax, fs=12, loc="upper left")
    save_fig(fig, os.path.join(OUTDIR, "fig2_temporal_fractions.png"))
    print("-> fig2_temporal_fractions.png")

    # ---- Energy sweep: f_2 and A_2 vs E* ------------------------------------
    esw = [float(r["fourier_mode2_fraction"]) for r in
           read_csv(os.path.join(RES, "03_energy_sweep", "energy_sweep.csv"))]
    rows = read_csv(os.path.join(RES, "03_energy_sweep", "energy_sweep.csv"))
    estar = np.array([float(r["E_target"]) for r in rows])
    f2 = np.array([float(r["mode2_fraction"]) * 100.0 for r in rows])
    a2 = np.array([float(r["A20"]) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax1.plot(estar, f2, ls="-", marker="o", color="k", lw=2.0, ms=7)
    for xi, yi in zip(estar, f2):
        ax1.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), fontsize=10, ha="center", color="k")
    style_axes(ax1, xl="Target energy $E^*$ (-)", yl="Second-mode fraction $f_2$ (%)",
               title="Mode-2 fraction vs $E^*$")
    ax1.set_xscale("log")
    ax1.set_xticks(list(estar))
    ax1.set_xticklabels(["%.2f" % e for e in estar], fontsize=11, rotation=30, ha="right")
    ax1.set_ylim(60, 100)

    ax2.plot(estar, a2, ls="-", marker="s", color="k", lw=2.0, ms=7)
    for xi, yi in zip(estar, a2):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), fontsize=10, ha="center", color="k")
    style_axes(ax2, xl="Target energy $E^*$ (-)", yl="Coefficient $A_2$ (-)",
               title="Second-harmonic coefficient vs $E^*$")
    ax2.set_xscale("log")
    ax2.set_xticks(list(estar))
    ax2.set_xticklabels(["%.2f" % e for e in estar], fontsize=11, rotation=30, ha="right")
    fig.suptitle("Actuation-energy sweep, Re = 500", fontsize=16, color="k", fontweight="bold")
    save_fig(fig, os.path.join(OUTDIR, "fig5_energy_sweep.png"))
    print("-> fig5_energy_sweep.png")


if __name__ == "__main__":
    main()