# -*- coding: utf-8 -*-
"""
A4/A5 - Compare old (200000-cap) vs new (POF1, tol=1e-9) stationary results
and recompute the grid-convergence metrics from the converged solutions.

Sources:
  old : lbm_mrt_validation/results_all.csv              (Re, profile, N, eps, K, Z, iters)
  new : lbm_mrt_validation/pof1_converged/pof1_results_converged.csv

A4 prints, per stationary case in the POF1 relaunch list, the old and new
eps, K, Z with the relative difference, and the convergence metadata.

A5 recomputes p_obs, eps_exact and the grid-convergence index at Re=500,
sin_2pi, N={128,256,512} following grid_convergence_study():
    e1,e2,e3 = eps_N128, eps_N256, eps_N512     (r = 2)
    p   = ln(|e1-e2|/|e2-e3|) / ln(r)
    eps_exact = e3 + (e3-e2)/(r**p - 1)
    GCI_512   = 1.25 |e3-e2| / (|e3| (r**p-1)) * 100   (fine grid index)
    GCI_256   = r**p * GCI_512                           (coarse grid index at N=256)
Requires the N=512 run to be finished; prints a reminder otherwise.
"""
import os
import csv
import math

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_CSV = os.path.join(HERE, "results_all.csv")
NEW_CSV = os.path.join(HERE, "pof1_converged", "pof1_results_converged.csv")
META_DIR = os.path.join(HERE, "pof1_converged", "meta")

RELAUNCH_KEYS = [
    (500, "uniform", 256), (500, "sin_pi", 256), (500, "sin_2pi", 256),
    (500, "pinn", 256), (500, "sin_2pi", 512),
    (1000, "uniform", 256), (1000, "sin_pi", 256),
    (1000, "sin_2pi", 256), (1000, "pinn", 256),
]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rd(a, b):
    """Relative difference (skip if reference missing or zero)."""
    if a is None or b is None:
        return None
    if abs(b) < 1e-15:
        return None
    return abs(a - b) / abs(b)


def main():
    old_rows = read_csv(OLD_CSV)
    new_rows = read_csv(NEW_CSV)
    for r in new_rows:
        r["_case"] = (int(float(r["Re"])), r["profile"], int(r["N"]))

    print("=" * 78)
    print("A4  OLD (200000-cap) vs NEW (POF1, tol=1e-9) — stationary runs")
    print("=" * 78)
    hdr = f"{'case':28s} {'eps_old':>13s} {'eps_new':>13s} {'rd%':>8s} {'K_old':>11s} {'K_new':>12s} {'iters_new':>10s} {'conv':>5s}"
    print(hdr)
    print("-" * len(hdr))
    problems = []
    for key in RELAUNCH_KEYS:
        Re, prof, N = key
        tag = f"{prof} Re{Re} N{N}"
        old = next((x for x in old_rows
                    if int(x["Re"]) == Re and x["profile"] == prof and int(x["N"]) == N), None)
        new = next((x for x in new_rows if x["_case"] == key), None)
        if new is None:
            print(f"{tag:28s} '--- still running / not found in POF1 csv ---'")
            problems.append(tag)
            continue
        eo = float(old["eps"]) if old else None
        en = float(new["eps"])
        ko = float(old["K"]) if old else None
        kn = float(new["K"])
        rd_eps = rd(en, eo)
        rd_eps_s = f"{100 * rd_eps:.3f}" if rd_eps is not None else "  n/a "
        conv = new["converged"]
        if conv != "True":
            problems.append(tag)
        print(f"{tag:28s} {eo:13.10g} {en:13.10g} {rd_eps_s:>7s}% "
              f"{ko:11.6g} {kn:12.6g} {int(new['iters']):10d} {conv:>5s}")

    print()
    print("=" * 78)
    print("A5  Grid-convergence metrics from converged solutions (Re=500, sin_2pi)")
    print("=" * 78)
    gci_eps = {}
    for key in [(500, "sin_2pi", 128), (500, "sin_2pi", 256), (500, "sin_2pi", 512)]:
        new = next((x for x in new_rows if x["_case"] == key), None)
        if new is not None and new["converged"] == "True":
            gci_eps[key[2]] = float(new["eps"])
        else:
            old = next((x for x in old_rows
                        if int(x["Re"]) == 500 and x["profile"] == "sin_2pi" and int(x["N"]) == key[2]), None)
            if old is not None and key[2] == 128:  # N128 already converged (<1e-9) under old campaign
                gci_eps[key[2]] = float(old["eps"])
                print(f"  N=128 uses old campaign eps (converged <1e-9): {gci_eps[128]:.10g}")
    if 512 not in gci_eps:
        print("  N=512 not available yet — rerun this script once resume_pof1_converged.py finishes.")
        return

    for key in RELAUNCH_KEYS:
        if key == (500, "sin_2pi", 256):
            pass
    e1, e2, e3 = gci_eps[128], gci_eps[256], gci_eps[512]
    r = 2.0
    print(f"  eps_128={e1:.10g}  eps_256={e2:.10g}  eps_512={e3:.10g}")
    if abs(e2 - e3) > 1e-20 and abs(e1 - e2) > 1e-20:
        p = math.log(abs((e1 - e2) / (e2 - e3))) / math.log(r)
        eps_exact = e3 + (e3 - e2) / (r ** p - 1.0)
        gci_512 = 1.25 * abs(e3 - e2) / (abs(e3) * (r ** p - 1.0)) * 100
        gci_256 = (r ** p) * gci_512
        print(f"  p_obs      = {p:.3f}")
        print(f"  eps_exact  = {eps_exact:.6g}   (was 0.0114)")
        print(f"  GCI@256    = {gci_256:.1f}%    (was ~11%)")
        print(f"  GCI@512    = {gci_512:.1f}%    (was ~5%)")
        print("  -> update manuscript.tex lines 1293-1309 if these differ by > rounded digit.")
    else:
        print("  Degenerate case: eps differences near zero; GCI/p undefined.")

    print()
    if problems:
        print("WARNING: still missing / not converged:", ", ".join(problems))
    else:
        print("ALL stationary POF1 runs converged to tol=1e-9. OK.")


if __name__ == "__main__":
    main()