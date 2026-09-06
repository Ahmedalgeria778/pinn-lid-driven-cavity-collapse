# -*- coding: utf-8 -*-
"""
Continuation of the Ghia Re=1000 (uniform lid, N=256) case.

The nominal campaign stopped at 200k iterations (MAX_ITER cap) with residual
~4.7e-6, i.e. the residual-based stopping criterion was never reached. The
solver has no restart capability, so this script cold-relaunches the case with
tol=1e-9 as the STOPPING CRITERION and max_iter used only as a safety cap.

It then:
  1. recomputes integrals and saves the fields (overwrites uniform_Re1000_N256.npz)
  2. writes ghia_re1000_final.csv (Re, L2, Linf, iters, final residual)
  3. regenerates fig1_ghia_validation.png from the converged fields
     (Re=100 from the already-converged saved field, Re=1000 from the new one)

Run (AFTER the main campaign has finished):  py -3.11 lbm_mrt_validation/resume_ghia_re1000.py
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTDIR = HERE
FIELDS_DIR = os.path.join(HERE, "fields")

from lbm_mrt_pinn_validation import LBM_MRT_Solver, GHIA  # noqa: E402

N = 256
tol = 1e-9
max_iter = 1_000_000

logfile = os.path.join(HERE, "convergence_ghia_Re1000.log")
print(f"Resuming/relaunching Ghia Re=1000, N={N}, tol={tol:.0e}, max_iter={max_iter} (safety cap)")
solver = LBM_MRT_Solver(1000, N, "uniform", max_iter=max_iter, tol=tol)
solver.run(verbose=True, logfile=logfile)
solver.compute_integrals()
solver.save_fields(FIELDS_DIR)

y, u_mid, _, _ = solver.centerline_profiles()
g = GHIA[1000]
ui = np.interp(g["y"], y, u_mid)
L2 = float(np.sqrt(np.mean((ui - np.array(g["u"])) ** 2)))
Linf = float(np.max(np.abs(ui - np.array(g["u"]))))
res_final = solver.last_residual
print(f"\nGHIA Re=1000 converged: iters={solver.iters}, final residual={res_final:.3e}")
print(f"  L2 = {L2:.5f}, Linf = {Linf:.5f}")

with open(os.path.join(HERE, "ghia_re1000_final.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Re", "N", "L2", "Linf", "iters", "residual"])
    w.writerow([1000, N, L2, Linf, solver.iters, res_final])

# --- regenerate fig1 from the saved converged fields ----------------------
DARK, PANEL = "#090909", "#111111"


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


def centerline_u(prof, re):
    fp = os.path.join(FIELDS_DIR, f"{prof}_Re{re}_N{N}.npz")
    with np.load(fp) as z:
        ux = z["ux"]
        return ux[:, ux.shape[1] // 2]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=DARK)
for ax, Re, col in zip([ax1, ax2], [100, 1000], ["#00e5ff", "#7dff6b"]):
    um = centerline_u("uniform", Re)
    yy = np.linspace(0, 1, um.size)
    gg = GHIA[Re]
    ax.plot(um, yy, color=col, lw=2.5, label=f"LBM-MRT (N={N})")
    ax.scatter(gg["u"], gg["y"], color="white", s=55, zorder=5, label="Ghia et al. (1982)")
    ui2 = np.interp(gg["y"], yy, um)
    l2 = np.sqrt(np.mean((ui2 - np.array(gg["u"])) ** 2))
    ax.text(0.04, 0.94, f"$L_2$ error (–) = {l2:.4f}", transform=ax.transAxes,
            color="yellow", fontsize=12, bbox=dict(boxstyle="round", fc="black", alpha=0.5))
    dark_axes(ax, xl="$u/U_{lid}$ (–)  at $x=0.5$", yl="$y$ (–)",
              title=f"Re = {Re}  (uniform lid)", fs=13)
    ax.legend(facecolor="#1a1a1a", edgecolor="#444", labelcolor="w", fontsize=11)
fig.suptitle("LBM-MRT validation of the lid-driven cavity: Ghia et al. (1982)",
             color="w", fontsize=16, fontweight="bold")
fig.savefig(os.path.join(OUTDIR, "fig1_ghia_validation.png"), dpi=300,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("   -> fig1_ghia_validation.png (regenerated with converged Re=1000 profile)")
print("Ghia Re=1000 continuation DONE.")