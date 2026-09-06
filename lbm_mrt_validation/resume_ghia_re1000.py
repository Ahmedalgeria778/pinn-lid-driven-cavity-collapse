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

from lbm_mrt_pinn_validation import LBM_MRT_Solver, GHIA, GHIA_V  # noqa: E402

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

# --- consolidated Re=100 & Re=1000 validation metrics from final fields -------
def centerline_metrics(Re):
    fp = os.path.join(FIELDS_DIR, f"uniform_Re{Re}_N{N}.npz")
    with np.load(fp) as z:
        ux, uy = z["ux"], z["uy"]
    yl = np.linspace(0, 1, N)
    um = ux[:, N // 2]
    vm = uy[N // 2, :]
    g, gv = GHIA[Re], GHIA_V[Re]
    ui = np.interp(g["y"], yl, um)
    vi = np.interp(gv["x"], yl, vm)
    g_u = np.array(g["u"])
    g_v = np.array(gv["v"])
    return (float(np.sqrt(np.mean((ui - g_u) ** 2))),
            float(np.max(np.abs(ui - g_u))),
            float(np.sqrt(np.mean((vi - g_v) ** 2))),
            float(np.max(np.abs(vi - g_v))))

print("\nConsolidated Ghia validation metrics (final fields):")
with open(os.path.join(HERE, "ghia_validation_final.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Re", "L2_u", "Linf_u", "L2_v", "Linf_v"])
    for Re in [100, 1000]:
        L2u, Linfu, L2v, Linfv = centerline_metrics(Re)
        w.writerow([Re, L2u, Linfu, L2v, Linfv])
        print(f"  Re={Re}: L2_u={L2u:.4f}, Linf_u={Linfu:.4f}, L2_v={L2v:.4f}, Linf_v={Linfv:.4f}")

# --- regenerate ALL figures from the saved fields (converged Re=1000) -------
print("Regenerating all figures 1-11 from saved fields...")
import subprocess
subprocess.run([sys.executable, os.path.join(HERE, "regen_all_figures.py")], check=True)

print("Ghia Re=1000 continuation DONE.")