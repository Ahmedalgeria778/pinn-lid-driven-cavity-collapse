# -*- coding: utf-8 -*-
"""
lbm_analysis.py — analyse consolidée post-campagne LBM (à lancer APRÈS la fin de
lbm_mrt_pinn_validation.py).

Lit les artefacts du solveur indépendant (results_all.csv, results_temporal.csv,
logs Ghia/GCI, champs bruts dans fields/) et les fusionne avec la campagne PINN
(09_aspect_ratio) pour produire :
  - validation_ghia_gci.csv   : concordance Ghia (L2, Linf) + convergence GCI (p, GCI%)
  - lbm_summary.csv           : métriques de la réponse indépendante par (Re, profil)
  - lbm_temporal_summary.csv  : admissibilité des branches temporelles
  - fig12_consolidated.png    : PINN vs LBM (A2 du flux, ε, branches, modes)
  - rebuttal_summary.md       : synthèse consolidée R2/R3 (réponse aux reviewers)

Lancement : py -3.11 lbm_mrt_validation/lbm_analysis.py
"""
import os
import re
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = HERE  # mêmes dossiers que la campagne

from figure_style import (style_axes, figure_title, legend, save_fig,  # noqa: E402
    PROFILE_FILL, PROFILE_HATCH, TEMP_FILL, TEMP_HATCH)

PROFILE_LABEL = {"uniform": "Uniform U=1", "sin_pi": r"$\sin(\pi x)$",
                 "sin_2pi": r"$A_2 \sin(2\pi x)$", "pinn": "PINN (mean)",
                 "pinn_t": "PINN Fourier (t-d)", "cheb_t": "Chebyshev (t-d)"}
A2_PINN = {100: 0.68758, 500: 0.68355, 1000: 0.58628}
RE_LIST = [100, 500, 1000]
PROFILES = ["uniform", "sin_pi", "sin_2pi", "pinn"]
TEMPORAL = ["pinn_t", "cheb_t"]


def load_csv(fname):
    with open(fname, newline="") as f:
        return list(csv.DictReader(f))


def parse_ghia_gci(log_path):
    ghia, gci = {}, {}
    if os.path.exists(log_path):
        txt = open(log_path, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"Re=(\d+)\s*:\s*L2\(u\)\s*=\s*([\d.eE+-]+),\s*L∞\(u\)\s*=\s*([\d.eE+-]+)", txt):
            ghia[int(m.group(1))] = {"L2": float(m.group(2)), "Linf": float(m.group(3))}
        m = re.search(r"p\s*=\s*([\d.eE+-]+),\s*GCI\s*=\s*([\d.eE+-]+)\s*%", txt)
        if m:
            gci = {"p": float(m.group(1)), "GCI": float(m.group(2))}
        m2 = re.search(r"extrapolated eps\s*=\s*([\d.eE+-]+)", txt)
        if m2:
            gci["eps_exact"] = float(m2.group(1))
    # Override Re=1000 with the converge-driven continuation result if available.
    fin = os.path.join(HERE, "ghia_re1000_final.csv")
    if os.path.exists(fin):
        with open(fin, newline="") as f:
            for row in csv.DictReader(f):
                if row["Re"] == "1000":
                    ghia[1000] = {"L2": float(row["L2"]), "Linf": float(row["Linf"])}
    # Consolidated final-field metrics (Re=100 & 1000) supersede log parsing.
    fcons = os.path.join(HERE, "ghia_validation_final.csv")
    if os.path.exists(fcons):
        with open(fcons, newline="") as f:
            for row in csv.DictReader(f):
                ghia[int(row["Re"])] = {"L2": float(row["L2_u"]), "Linf": float(row["Linf_u"]),
                                        "L2_v": float(row["L2_v"]), "Linf_v": float(row["Linf_v"])}
    return ghia, gci


def main():
    res_all = {r: {} for r in RE_LIST}
    for row in load_csv(os.path.join(OUTDIR, "results_all.csv")):
        res_all[int(row["Re"])][row["profile"]] = {k: float(row[k]) if k != "profile" else row[k]
                                                   for k in row}
    res_t = []
    if os.path.exists(os.path.join(OUTDIR, "results_temporal.csv")):
        for row in load_csv(os.path.join(OUTDIR, "results_temporal.csv")):
            res_t.append({k: (float(row[k]) if k not in ("control",) else row[k]) for k in row})

    # 1) Validation indépendante : Ghia + GCI
    ghia, gci = parse_ghia_gci(os.path.join(ROOT, "logs", "lbm_validation.log"))
    with open(os.path.join(OUTDIR, "validation_ghia_gci.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for Re in ghia:
            w.writerow([f"ghia_L2_Re{Re}", ghia[Re]["L2"]])
            w.writerow([f"ghia_Linf_Re{Re}", ghia[Re]["Linf"]])
            if "L2_v" in ghia[Re]:
                w.writerow([f"ghia_L2_v_Re{Re}", ghia[Re]["L2_v"]])
                w.writerow([f"ghia_Linf_v_Re{Re}", ghia[Re]["Linf_v"]])
        for k, v in gci.items():
            w.writerow([f"gci_{k}", v])

    # 2) Résumé réponse indépendante (production)
    with open(os.path.join(OUTDIR, "lbm_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Re", "profile", "eps", "K", "Z", "delta_eps_unif_pct"])
        for Re in RE_LIST:
            eps_unif = res_all[Re]["uniform"]["eps"]
            for prof in PROFILES:
                r = res_all[Re][prof]
                w.writerow([Re, prof, r["eps"], r["K"], r["Z"],
                            (r["eps"] - eps_unif) / eps_unif * 100 if eps_unif else 0.0])

    # 3) Résumé branches temporelles
    if res_t:
        with open(os.path.join(OUTDIR, "lbm_temporal_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Re", "control", "fluct_ratio", "mode_dom", "mode_frac",
                        "mode_dom_fluct", "mode_frac_fluct", "A2_flow", "A2_flow_fluct"])
            for r in res_t:
                w.writerow([r["Re"], r["control"], r["fluct_ratio"], r["mode_dom"],
                            r["mode_frac"], r["mode_dom_fluct"], r["mode_frac_fluct"],
                            r["A2_flow"], r["A2_flow_fluct"]])

    # 4) Consolidated figure: PINN control injected into an independent solver
    fig, axs = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={'hspace': 0.52, 'wspace': 0.32,
                                          'top': 0.92, 'bottom': 0.08})
    # (a) eps — independent response vs uniform lid
    ax = axs[0, 0]
    ax.plot(RE_LIST, [res_all[Re]["sin_2pi"]["eps"] for Re in RE_LIST], "k^-", lw=2.2, ms=9,
            label=PROFILE_LABEL["sin_2pi"])
    ax.plot(RE_LIST, [res_all[Re]["pinn"]["eps"] for Re in RE_LIST], "k*:", lw=2.2, ms=11,
            label=PROFILE_LABEL["pinn"])
    ax.plot(RE_LIST, [res_all[Re]["uniform"]["eps"] for Re in RE_LIST], color="0.4", marker="d",
            ls="--", lw=1.8, ms=7, label="Uniform (reference)")
    ax.set_xscale("log")
    style_axes(ax, xl=r"$Re$ (–)", yl=r"$\epsilon_{LBM}$ (–)", title="Independent dissipation (LBM-MRT)")
    legend(ax, fs=10)
    # (b) A2 of the flow: PINN (a priori) vs LBM measured
    ax = axs[0, 1]
    ax.plot(RE_LIST, [A2_PINN[r] for r in RE_LIST], color="0.35", marker="x", ls="--", lw=1.6,
            ms=10, label="$A_2$ PINN control (a priori)")
    # A2 of the flow measured on the LBM mean field for the 'pinn' control
    a2_flow = {}
    for Re in RE_LIST:
        fp = os.path.join(OUTDIR, "fields", f"pinn_Re{Re}_N256.npz")
        if os.path.exists(fp):
            with np.load(fp) as z:
                u_mid = z["ux"][:, z["ux"].shape[1] // 2]
            y = np.linspace(0, 1, u_mid.size)
            amps = [2.0 * np.trapz(u_mid * np.sin(n * np.pi * y), y) for n in range(1, 11)]
            a2_flow[Re] = float(amps[1])
    if a2_flow:
        ax.plot(list(a2_flow.keys()), list(a2_flow.values()), "ko-", lw=2.4, ms=9,
                label="$A_2$ measured (LBM, mean)")
    # aspect ratio (1:1 vs 2:1) A2 measured by the PINN
    ar_csv = os.path.join(ROOT, "results_reviewers", "09_aspect_ratio", "aspect_ratio_results.csv")
    ar_vals = []
    if os.path.exists(ar_csv):
        for row in load_csv(ar_csv):
            ar_vals.append(float(row["A20"]))
    if ar_vals:
        ax.scatter([400, 420], ar_vals, s=120, marker="*", facecolor="white", edgecolor="k",
                   linewidths=1.2, label="Aspect ratio 1:1 / 2:1 (PINN)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    style_axes(ax, xl=r"$Re$ (–)", yl=r"$A_2^{flow}$ (–)", title=r"Flow $A_2$: PINN vs independent LBM")
    legend(ax, fs=9)
    # (c) Time-dependent branches: K_fluct/K
    ax = axs[1, 0]
    xp = np.arange(len(RE_LIST))
    w = 0.34
    for i, ctrl in enumerate(TEMPORAL):
        vals = [next((r["fluct_ratio"] for r in res_t if r["Re"] == Re and r["control"] == ctrl),
                     np.nan) for Re in RE_LIST]
        ax.bar(xp + (i - 0.5) * w, vals, w, color=TEMP_FILL[ctrl], edgecolor="k",
               hatch=TEMP_HATCH[ctrl], label=PROFILE_LABEL[ctrl])
    ax.set_xticks(xp); ax.set_xticklabels([f"Re={r}" for r in RE_LIST], color="k", fontsize=11)
    ax.axhline(0.01, color="0.35", ls=":", lw=1.2, label="$R_K$ = 1% threshold")
    style_axes(ax, xl="Re (–)", yl=r"$K_{fluct}/K$ (–)",
               title="Weight of time fluctuations (branches)", fs=13)
    legend(ax, fs=10)
    # (d) Dominant modes: mean field vs fluctuations
    ax = axs[1, 1]
    markers_mean = {"pinn_t": "o", "cheb_t": "s"}
    markers_fluct = {"pinn_t": "^", "cheb_t": "D"}
    ls_mean = {"pinn_t": "-", "cheb_t": "-"}
    ls_fluct = {"pinn_t": "--", "cheb_t": ":"}
    for i, ctrl in enumerate(TEMPORAL):
        mm = [next((r["mode_dom"] for r in res_t if r["Re"] == Re and r["control"] == ctrl),
                   np.nan) for Re in RE_LIST]
        mf = [next((r["mode_dom_fluct"] for r in res_t if r["Re"] == Re and r["control"] == ctrl),
                   np.nan) for Re in RE_LIST]
        ax.plot(xp + (i - 0.5) * 0.3, mm,
                marker=markers_mean[ctrl], color="k", ls=ls_mean[ctrl],
                lw=2, ms=8, label=PROFILE_LABEL[ctrl] + " (mean)")
        ax.plot(xp + (i - 0.5) * 0.3, mf,
                marker=markers_fluct[ctrl], color="0.45", ls=ls_fluct[ctrl],
                lw=1.8, ms=7, label=PROFILE_LABEL[ctrl] + " (fluct.)")
    ax.set_xticks(xp); ax.set_xticklabels([f"Re={r}" for r in RE_LIST], color="k", fontsize=11)
    ax.set_ylim(0, 11)
    style_axes(ax, xl="Re (–)", yl="Dominant mode index (–)",
               title="Flow modes (mean vs fluctuations)", fs=13)
    legend(ax, fs=9, ncol=2)
    figure_title(fig, "PINN control injected into an independent solver (LBM-MRT): cross-validation")
    save_fig(fig, os.path.join(OUTDIR, "fig12_consolidated.png"))
    plt.close()
    print("   -> fig12_consolidated.png")

    # 5) Synthèse rebuttal (texte)
    lines = [
        "# Rebuttal summary (consolidated, R2/R3)",
        "",
        "## 1. Independent-lid physical validation (LBM-MRT D2Q9, this repo)",
        "",
    ]
    if ghia:
        for Re in sorted(ghia):
            line = f"- Ghia (lid uniforme, N=256) : Re={Re} — L2(u)={ghia[Re]['L2']:.4f}, " \
                   f"Linf(u)={ghia[Re]['Linf']:.4f}"
            if "L2_v" in ghia[Re]:
                line += f" — L2(v)={ghia[Re]['L2_v']:.4f}, Linf(v)={ghia[Re]['Linf_v']:.4f}"
            lines.append(line)
    else:
        lines.append("- Ghia : métriques non encore disponibles (campagne en cours)")
    if gci:
        lines.append(f"- GCI (Re=500, sin_2pi, N 128-256-512) : p={gci['p']:.2f}, "
                     f"GCI={gci['GCI']:.2f}% ({gci.get('eps_exact', float('nan')):.4e})")
    if res_t:
        lines.append("")
        lines.append("## 2. Branch admissibility (time-dependent forcing)")
        lines.append("")
        lines.append("| Re | control | K_fluct/K | dom. mode (mean) | dom. mode (fluct) | A2_flow |")
        lines.append("|----|---------|-----------|------------------|-------------------|---------|")
        for r in res_t:
            lines.append(f"| {r['Re']} | {r['control']} | {r['fluct_ratio']:.1%} | "
                         f"{r['mode_dom']} ({r['mode_frac']:.2f}) | {r['mode_dom_fluct']} "
                         f"({r['mode_frac_fluct']:.2f}) | {r['A2_flow']:.4f} |")
    ar_csv = os.path.join(ROOT, "results_reviewers", "09_aspect_ratio", "aspect_ratio_results.csv")
    if os.path.exists(ar_csv):
        ar = load_csv(ar_csv)
        lines.append("")
        lines.append("## 3. Aspect ratio (R3.2, Re=500, seed 42, E*=0.25)")
        lines.append("")
        lines.append("| geometry | A2 | f_(2,0) | f_temp | E_total |")
        lines.append("|----------|----|---------|--------|---------|")
        for r in ar:
            lines.append(f"| {r['geometry']} | {float(r['A20']):.4f} | {float(r['mode2_fraction']):.3f} "
                         f"| {float(r['temporal_fraction']):.3f} | {float(r['E_total']):.4f} |")
        lines.append("")
        lines.append("> Mode-2 collapse persists in the 2:1 cavity (geometric robustness,"
                     " parametrization-boundary preserved).")
    with open(os.path.join(OUTDIR, "rebuttal_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("   -> rebuttal_summary.md")
    print("analyse consolidée OK.")


if __name__ == "__main__":
    main()