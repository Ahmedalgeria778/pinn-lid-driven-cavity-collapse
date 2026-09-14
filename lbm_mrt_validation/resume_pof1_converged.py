# -*- coding: utf-8 -*-
"""
Resume / relaunch of the Paper 1 (POF1) stationary LBM runs that were stopped
at the 200 000-iteration cap WITHOUT reaching the nominal residual criterion.

Protocol POF1 (definitive):
    paper  = POF1
    solver = MRT-D2Q9
    stationary convergence:
        stopping criterion (residual) = 1e-9
        max_iter                     = 5e6   (pure safety cap)
    production grid = 256x256
    GCI grids      = 128, 256, 512

Rule: every value published in Paper 1 must come from a run that satisfies
this protocol. The relaunched runs write their protocol metadata alongside
their results so that they can never be confused with the Paper 2 branch runs
(B1/B2/B3, which used cap 5e6 under a DIFFERENT reduced/pde protocol).

This script cold-relaunches (the LBM_MRT_Solver has no restart capability, cf.
resume_ghia_re1000.py) each stationary case that did not reach tol=1e-9:

    baseline production runs (Re=500, 1000; uniform/sin_pi/sin_2pi/pinn, N=256)
    GCI grid      : Re=500 sin_2pi N=256 (production eps_256) and N=512 (eps_512)

For every case it:
   1. relaunches with tol=1e-9 as the STOPPING criterion, max_iter=5e6 (cap)
   2. recomputes the integrals (eps, K, Z) and saves the fields
   3. writes a per-case metadata file (paper=POF1, case, Re, Nx, Ny, tol,
      max_iter, converged, iteration_final, residual_final) next to the log
   4. appends the converged values to pof1_results_converged.csv

Run (sequential, one case at a time):  py -3.11 lbm_mrt_validation/resume_pof1_converged.py
Optional filter:                        py -3.11 .../resume_pof1_converged.py --case 500 sin_2pi 256
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTDIR = HERE
FIELDS_DIR = os.path.join(OUTDIR, "pof1_converged", "fields")
META_DIR = os.path.join(OUTDIR, "pof1_converged", "meta")

from lbm_mrt_pinn_validation import LBM_MRT_Solver  # noqa: E402

TOL = 1e-9
MAX_ITER = 5_000_000

# (Re, profile, N)  -- stationary cases to (re)launch under Protocol POF1
# N=256 sin_2pi Re=500 serves BOTH the production table and GCI eps_256.
CASES = [
    (500, "uniform", 256),
    (500, "sin_pi", 256),
    (500, "sin_2pi", 256),
    (500, "pinn", 256),
    (500, "sin_2pi", 512),   # GCI fine grid
    (1000, "uniform", 256),
    (1000, "sin_pi", 256),
    (1000, "sin_2pi", 256),
    (1000, "pinn", 256),
]

RESULTS_CSV = os.path.join(OUTDIR, "pof1_converged", "pof1_results_converged.csv")


def run_case(Re, profile, N):
    tag = f"{profile}_Re{Re}_N{N}"
    logfile = os.path.join(META_DIR, f"convergence_{tag}_POF1.log")
    print(f"\n=== POF1 relaunch: {tag}  tol={TOL:.0e} cap={MAX_ITER} ===", flush=True)

    solver = LBM_MRT_Solver(Re, N, lid_profile=profile, max_iter=MAX_ITER, tol=TOL)
    solver.run(verbose=True, logfile=logfile)
    eps, K, Z = solver.compute_integrals()
    fname = solver.save_fields(FIELDS_DIR)

    converged = (solver.iters < MAX_ITER)
    meta = {
        "paper": "POF1",
        "case": profile,
        "Re": Re,
        "Nx": N,
        "Ny": N,
        "tol": TOL,
        "max_iter": MAX_ITER,
        "converged": bool(converged),
        "iteration_final": int(solver.iters),
        "residual_final": float(solver.last_residual),
        "eps": float(eps),
        "K": float(K),
        "Z": float(Z),
        "fields_file": fname,
        "logfile": os.path.basename(logfile),
    }
    with open(os.path.join(META_DIR, tag + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_row = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_row:
            w.writerow(["paper", "Re", "profile", "N", "eps", "K", "Z", "iters",
                        "converged", "residual_final", "meta_file"])
        w.writerow(["POF1", Re, profile, N, f"{eps:.12g}", f"{K:.12g}", f"{Z:.12g}",
                    solver.iters, converged, f"{solver.last_residual:.6e}",
                    os.path.basename(tag + "_meta.json")])

    print(f"   [{tag}] converged={converged} iters={solver.iters} "
          f"residual={solver.last_residual:.3e}  eps={eps:.12g} K={K:.12g} Z={Z:.12g}",
          flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs=3, metavar=("Re", "profile", "N"), type=str, default=None)
    args = ap.parse_args()

    os.makedirs(FIELDS_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    if args.case is not None:
        Re, profile, N = int(args.case[0]), args.case[1], int(args.case[2])
        run_case(Re, profile, N)
        return

    for Re, profile, N in CASES:
        run_case(Re, profile, N)

    print("\nAll POF1 relaunch cases finished.")


if __name__ == "__main__":
    main()