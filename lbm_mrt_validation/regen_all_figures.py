# -*- coding: utf-8 -*-
"""
Regenerate ALL figures 1-11 from the saved field files (npz) + scalar CSVs +
Ghia (1982) reference tables (ghiau.txt / ghiav.txt).

Does NOT re-run any simulation. Called by resume_ghia_re1000.py (which first
overwrites uniform_Re1000_N256.npz with the converged field) or standalone.

White / black-and-white print-friendly style (AE request): white background,
black curves, marker symbols to distinguish cases, "(-)" units on axes and
enlarged axis fonts.  300 dpi.

Run:  py -3.11 lbm_mrt_validation/regen_all_figures.py
"""
import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTDIR = HERE
FIELDS_DIR = os.path.join(HERE, "fields")

from lbm_mrt_pinn_validation import (  # noqa: E402
    GHIA, GHIA_V, RE_LIST, PROFILES, TEMPORAL_CONTROLS,
    PROFILE_LABEL, GHIA_N, PROD_N, GCI_N_list,
)
from figure_style import (style_axes, figure_title, legend, save_fig,  # noqa: E402
                           style_colorbar, field_cmap, phase_cmap,
                           PROFILE_LS, PROFILE_MK, TEMP_MK, TEMP_LS, RE_MK, RE_LS,
                           PROFILE_HATCH, PROFILE_FILL, TEMP_HATCH, TEMP_FILL)

ME = 22  # markevery for 256-node centerlines


def linekws(prof, lw=2.2, me=ME):
    return dict(color='k', lw=lw, ls=PROFILE_LS[prof], marker=PROFILE_MK[prof],
                markevery=me, ms=7)


def tempkws(prof, lw=2.2, me=ME):
    return dict(color='k', lw=lw, ls=TEMP_LS[prof], marker=TEMP_MK[prof],
                markevery=me, ms=8)


def load_npz(prof, Re):
    fp = os.path.join(FIELDS_DIR, f'{prof}_Re{Re}_N{PROD_N}.npz')
    with np.load(fp) as z:
        return {k: z[k] for k in z.files}


class FieldProxy:
    """Minimal proxy reproducing the plotting-relevant solver API."""
    def __init__(self, prof, Re, d):
        self.lid_profile, self.Re = prof, Re
        self.N = int(d['N'].item() if d['N'].ndim == 0 else d['N'])
        it = d['iters'].item()
        self.iters = int(it) if np.ndim(it) == 0 else int(it)
        self.period = int(d.get('period').item()) if d.get('period') is not None and np.size(d.get('period')) else None
        self.ux = d['ux'].astype(float)
        self.uy = d['uy'].astype(float)
        self.ux_fluct = d.get('ux_fluct', np.zeros_like(self.ux)).astype(float)
        self.uy_fluct = d.get('uy_fluct', np.zeros_like(self.uy)).astype(float)
        self.probe = d.get('probe')
        self.phase_ux = d.get('phase_ux')
        self.compute_integrals()

    def compute_integrals(self):
        N, dx, nu = self.N, 1.0 / (self.N - 1), 1.0 / self.Re
        U, V = self.ux, self.uy
        dUdx, dUdy = np.gradient(U, dx, axis=1), np.gradient(U, dx, axis=0)
        dVdx, dVdy = np.gradient(V, dx, axis=1), np.gradient(V, dx, axis=0)
        S11, S22, S12 = dUdx, dVdy, 0.5 * (dUdy + dVdx)
        phi = nu * (2 * S11 ** 2 + 2 * S22 ** 2 + 4 * S12 ** 2)
        self.dissipation = phi
        self.eps = np.trapz(np.trapz(phi, dx=dx, axis=1), dx=dx)
        K = 0.5 * (U ** 2 + V ** 2)
        self.K = np.trapz(np.trapz(K, dx=dx, axis=1), dx=dx)
        Kf = 0.5 * (self.ux_fluct ** 2 + self.uy_fluct ** 2)
        self.K_fluct = np.trapz(np.trapz(Kf, dx=dx, axis=1), dx=dx)
        vort = dVdx - dUdy
        self.Z = np.trapz(np.trapz(vort ** 2, dx=dx, axis=1), dx=dx)
        return self.eps, self.K, self.Z

    def centerline_profiles(self):
        mid = self.N // 2
        y = np.linspace(0, 1, self.N)
        x = np.linspace(0, 1, self.N)
        return y, self.ux[:, mid], x, self.uy[mid, :]

    def modal_analysis(self):
        y, u_mid, _, _ = self.centerline_profiles()
        modes = np.arange(1, 11)
        amps = np.array([2.0 * np.trapz(u_mid * np.sin(n * np.pi * y), y) for n in modes])
        E = amps ** 2
        return modes, amps, E / E.sum() if E.sum() > 0 else E

    def fluct_modal_analysis(self):
        mid = self.N // 2
        y = np.linspace(0, 1, self.N)
        u_mid_rms = self.ux_fluct[:, mid]
        modes = np.arange(1, 11)
        amps = np.array([2.0 * np.trapz(u_mid_rms * np.sin(n * np.pi * y), y) for n in modes])
        E = amps ** 2
        return modes, amps, E / E.sum() if E.sum() > 0 else E

    def dominant_mean_mode(self):
        return int(np.argmax(self.modal_analysis()[2]) + 1)


# --- Rebuild result containers from npz -------------------------------------
ghia_solvers = {}
for Re in [100, 1000]:
    ghia_solvers[Re] = FieldProxy('uniform', Re, load_npz('uniform', Re))

all_results = {Re: {} for Re in RE_LIST}
for Re in RE_LIST:
    for prof in PROFILES:
        all_results[Re][prof] = FieldProxy(prof, Re, load_npz(prof, Re))

temporal_results = {Re: {} for Re in RE_LIST}
for Re in RE_LIST:
    for prof in TEMPORAL_CONTROLS:
        temporal_results[Re][prof] = FieldProxy(prof, Re, load_npz(prof, Re))

# --- GCI scalars from gci_progress.csv (last occurrence per N) ---------------
gci_eps_by_N = {}
with open(os.path.join(HERE, 'gci_progress.csv')) as f:
    for row in csv.reader(f):
        if len(row) >= 4 and row[0].strip() == '500':
            gci_eps_by_N[int(row[2])] = float(row[3])
gci_eps = [gci_eps_by_N[Nv] for Nv in GCI_N_list]
e1, e2, e3 = gci_eps
r = 2.0
p = np.log(abs((e1 - e2) / (e2 - e3))) / np.log(r)
gci_exact = e3 + (e3 - e2) / (r ** p - 1.0)
gci_GCI = 1.25 * abs(e3 - e2) / (abs(e3) * (r ** p - 1.0)) * 100
print(f'GCI scalars: eps={gci_eps}, p={p:.2f}, eps*={gci_exact:.4e}, GCI={gci_GCI:.2f}%')

GHIA_MK = 's'  # open square marker for Ghia reference points

# ============================================================================
# Figure 1 : Ghia validation (correct references from ghiau.txt)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, Re in zip([ax1, ax2], [100, 1000]):
    sol = ghia_solvers[Re]
    y, u_mid, _, _ = sol.centerline_profiles()
    g = GHIA[Re]
    ax.plot(u_mid, y, color='k', lw=2.5, marker='o', markevery=24, ms=6,
            label='LBM-MRT (N={})'.format(GHIA_N))
    ax.scatter(g['u'], g['y'], marker=GHIA_MK, s=45, facecolor='white', edgecolor='k',
               linewidths=1.2, zorder=5, label='Ghia et al. (1982)')
    u_interp = np.interp(g['y'], y, u_mid)
    L2 = np.sqrt(np.mean((u_interp - np.array(g['u'])) ** 2))
    ax.text(0.03, 0.94, '$L_2$ error (–) = {:.4f}'.format(L2), transform=ax.transAxes,
            color='k', fontsize=13, bbox=dict(boxstyle='round', fc='white', ec='k'))
    style_axes(ax, xl='$u/U_{lid}$ (–)  at $x=0.5$', yl='$y$ (–)',
               title=f'Re = {Re}  (uniform lid)')
    legend(ax, fs=12)
figure_title(fig, 'LBM-MRT validation of the lid-driven cavity: Ghia et al. (1982)')
save_fig(fig, os.path.join(OUTDIR, 'fig1_ghia_validation.png'))
print("   -> fig1_ghia_validation.png")

# ============================================================================
# Figure 2 : GCI (black curves + markers)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
N_vals = GCI_N_list
ax1.plot(N_vals, gci_eps, 'ko-', lw=2.5, ms=10, label=r'$\epsilon$ (LBM)')
ax1.axhline(gci_exact, color='0.4', ls='--', lw=2, label=r'$\epsilon^*$ = {:.4e}'.format(gci_exact))
for Nv, ev in zip(N_vals, gci_eps):
    ax1.text(Nv + 5, ev, '{:.4e}'.format(ev), color='k', fontsize=11)
style_axes(ax1, xl='$N$ (–)', yl=r'$\epsilon$ (–)', title=f'GCI = {gci_GCI:.2f}%, p = {p:.2f}')
legend(ax1)
h = 1.0 / np.array(N_vals)
err = np.abs(np.array(gci_eps) - gci_exact)
ax2.loglog(h, err, 'ko-', lw=2.5, ms=10, label=r'$|\epsilon - \epsilon^*|$')
if not np.isnan(p):
    ax2.loglog(h, err[0] * (h / h[0]) ** p, '--', color='0.35', lw=1.8,
               label='slope {:.2f}'.format(p))
style_axes(ax2, xl='$h = 1/N$ (–)', yl=r'$|\epsilon - \epsilon^*|$ (–)', title='Spatial convergence')
legend(ax2)
figure_title(fig, 'Grid convergence (GCI) — Re=500, controlled lid $A_2\\sin(2\\pi x)$')
save_fig(fig, os.path.join(OUTDIR, 'fig2_grid_convergence.png'))
print("   -> fig2_grid_convergence.png")

# ============================================================================
# Figure 3 : Velocity magnitude fields (grayscale, B&W-friendly)
# ============================================================================
fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)
x_coord = np.linspace(0, 1, PROD_N)
y_coord = np.linspace(0, 1, PROD_N)
X, Y = np.meshgrid(x_coord, y_coord)
for ri, Re in enumerate(RE_LIST):
    for pi, prof in enumerate(PROFILES):
        ax = fig.add_subplot(gs[ri, pi])
        sol = all_results[Re][prof]
        spd = np.sqrt(sol.ux ** 2 + sol.uy ** 2)
        cf = ax.contourf(X, Y, spd, levels=40, cmap=field_cmap())
        ax.streamplot(X, Y, sol.ux, sol.uy, color='k', linewidth=0.5, density=1.2,
                      arrowsize=0.7, arrowstyle='->')
        cb = fig.colorbar(cf, ax=ax, pad=0.01)
        style_colorbar(cb, '$|\\mathbf{u}|$ (–)')
        ax.set_title(f'Re={Re} · {PROFILE_LABEL[prof]}', color='k', fontsize=12, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
figure_title(fig, 'Velocity magnitude $|\\mathbf{u}|$ (–) — MRT D2Q9 (N=256)')
save_fig(fig, os.path.join(OUTDIR, 'fig3_velocity_fields.png'))
print("   -> fig3_velocity_fields.png")

# ============================================================================
# Figure 4 : Dissipation fields (grayscale, B&W-friendly)
# ============================================================================
fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)
for ri, Re in enumerate(RE_LIST):
    for pi, prof in enumerate(PROFILES):
        ax = fig.add_subplot(gs[ri, pi])
        sol = all_results[Re][prof]
        d = sol.dissipation
        vmax = np.percentile(d, 99)
        cf = ax.contourf(X, Y, np.clip(d, 0, vmax), levels=40, cmap=field_cmap())
        cb = fig.colorbar(cf, ax=ax, pad=0.01)
        style_colorbar(cb, '$\\Phi$ (–)')
        ax.text(0.03, 0.04, f'$\\epsilon$ = {sol.eps:.2e} (–)', transform=ax.transAxes,
                color='k', fontsize=11, bbox=dict(boxstyle='round', fc='white', ec='k'))
        ax.set_title(f'Re={Re} · {PROFILE_LABEL[prof]}', color='k', fontsize=12, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
figure_title(fig, 'Viscous dissipation $\\Phi$ (–) — $\\epsilon = \\iint \\Phi\\,\\mathrm{d}\\Omega$')
save_fig(fig, os.path.join(OUTDIR, 'fig4_dissipation_fields.png'))
print("   -> fig4_dissipation_fields.png")

# ============================================================================
# Figure 5 : Total dissipation comparison (grayscale bars + hatches)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
xp = np.arange(len(PROFILES))
for ax, Re in zip(axes, RE_LIST):
    eps_unif = all_results[Re]['uniform'].eps
    eps_vals = [all_results[Re][p].eps for p in PROFILES]
    ax.bar(xp, eps_vals, color=[PROFILE_FILL[p] for p in PROFILES],
           edgecolor='k', hatch=[PROFILE_HATCH[p] for p in PROFILES],
           alpha=1.0, width=0.6)
    ax.axhline(eps_unif, color='k', ls=':', lw=1.5, label='$\\epsilon$ (uniform lid)')
    for xi, ev in zip(xp, eps_vals):
        drel = (ev - eps_unif) / eps_unif * 100 if eps_unif > 0 else 0
        ax.text(xi, ev * 1.04, f'{drel:+.1f}%', ha='center', color='k', fontsize=11, fontweight='bold')
    ax.set_xticks(xp)
    ax.set_xticklabels([PROFILE_LABEL[p] for p in PROFILES], color='k', fontsize=12, rotation=15, ha='right')
    style_axes(ax, yl=r'$\epsilon = \iint \Phi \, \mathrm{d}\Omega$ (–)', title=f'Re = {Re}')
    legend(ax, fs=11)
figure_title(fig, 'Total dissipation $\\epsilon$ (–) — lid-profile comparison')
save_fig(fig, os.path.join(OUTDIR, 'fig5_dissipation_comparison.png'))
print("   -> fig5_dissipation_comparison.png")

# ============================================================================
# Figure 6 : Flow modal analysis (grayscale bars, dominant darker)
# ============================================================================
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
modes = np.arange(1, 11)
for ri, Re in enumerate(RE_LIST):
    for pi, prof in enumerate(PROFILES):
        ax = fig.add_subplot(gs[ri, pi])
        sol = all_results[Re][prof]
        _, _, fracs = sol.modal_analysis()
        fracs_pct = fracs * 100
        bars = ax.bar(modes, fracs_pct, color=PROFILE_FILL[prof], edgecolor='k',
                      hatch=PROFILE_HATCH[prof], alpha=1.0)
        dom = np.argmax(fracs) + 1
        if dom <= 10:
            bars[dom - 1].set_facecolor('#444444')
            bars[dom - 1].set_edgecolor('k')
        style_axes(ax, xl='Mode index $n$ (–)', yl='Modal energy fraction (%)',
                   title=f'Re={Re} · {PROFILE_LABEL[prof]}\nMode {dom}: {fracs_pct[dom - 1]:.1f}%',
                   fs=12, tick=11)
figure_title(fig, 'Modal analysis of $u_{mid}(y)$ at $x=0.5$ — MRT D2Q9')
save_fig(fig, os.path.join(OUTDIR, 'fig6_modal_analysis.png'))
print("   -> fig6_modal_analysis.png")

# ============================================================================
# Figure 7 : Centerline profiles (both Ghia tables, black lines + markers)
# ============================================================================
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)
for ri, Re in enumerate(RE_LIST):
    ax = fig.add_subplot(gs[ri, 0])
    for prof in PROFILES:
        sol = all_results[Re][prof]
        y, u_mid, _, _ = sol.centerline_profiles()
        ax.plot(u_mid, y, **linekws(prof), label=PROFILE_LABEL[prof])
    if Re in GHIA:
        g = GHIA[Re]
        ax.scatter(g['u'], g['y'], marker=GHIA_MK, s=32, facecolor='white', edgecolor='k',
                   linewidths=1.0, zorder=5, label='Ghia et al. (1982)')
    style_axes(ax, xl='$u/U_{lid}$ (–)  at $x=0.5$', yl='$y$ (–)',
               title=f'Re={Re} — $u(y)$')
    legend(ax, fs=10, loc='lower right')

    ax = fig.add_subplot(gs[ri, 1])
    for prof in PROFILES:
        sol = all_results[Re][prof]
        x, v_mid = sol.centerline_profiles()[2:4]
        ax.plot(x, v_mid, **linekws(prof), label=PROFILE_LABEL[prof])
    if Re in GHIA_V:
        gv = GHIA_V[Re]
        ax.scatter(gv['x'], gv['v'], marker=GHIA_MK, s=32, facecolor='white', edgecolor='k',
                   linewidths=1.0, zorder=5, label='Ghia et al. (1982)')
    ax.axhline(0, color='0.5', lw=0.8)
    style_axes(ax, xl='$x$ (–)', yl='$v/U_{lid}$ (–)  at $y=0.5$',
               title=f'Re={Re} — $v(x)$')
    legend(ax, fs=10)
figure_title(fig, 'Centerline profiles — $u(0.5,y)$ and $v(x,0.5)$')
save_fig(fig, os.path.join(OUTDIR, 'fig7_centerline_profiles.png'))
print("   -> fig7_centerline_profiles.png")

# ============================================================================
# Figure 8 : Robustness (A2 PINN black; enstrophy bars grayscale)
# ============================================================================
Re_vals = RE_LIST
A2_pinn = {Re: all_results[Re]['pinn'].modal_analysis()[1][1] for Re in Re_vals}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ax1.plot(Re_vals, list(A2_pinn.values()), 'ko-', lw=2.5, ms=10, label='$A_2$ PINN (mean)')
ax1.axhline(np.mean(list(A2_pinn.values())), color='0.4', ls=':', lw=1.2,
            label='$A_2$ PINN (mean)')
style_axes(ax1, xl='$Re$ (–)', yl='$A_2$ (–)', title='Mode 2 — Reynolds robustness')
legend(ax1)

xp = np.arange(len(Re_vals))
width = 0.18
for i, prof in enumerate(PROFILES):
    Z_vals = [all_results[Re][prof].Z for Re in Re_vals]
    ax2.bar(xp + i * width, Z_vals, width, color=PROFILE_FILL[prof], edgecolor='k',
            hatch=PROFILE_HATCH[prof], label=PROFILE_LABEL[prof])
ax2.set_xticks(xp + 1.5 * width)
ax2.set_xticklabels([f'Re={r}' for r in Re_vals], color='k', fontsize=12)
style_axes(ax2, yl=r'$Z = \iint \omega^2 \, \mathrm{d}\Omega$ (–)', title='Enstrophy — lid profiles')
legend(ax2, fs=10, ncol=2)
figure_title(fig, 'Mode-2 amplitude and enstrophy robustness')
save_fig(fig, os.path.join(OUTDIR, 'fig8_robustness.png'))
print("   -> fig8_robustness.png")

# ============================================================================
# Figure 9 : Time-averaged branch response (black + markers + grayscale bars)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
xp = np.arange(len(RE_LIST))
width = 0.38
for pi, prof in enumerate(TEMPORAL_CONTROLS):
    K_fluct_ratio = [temporal_results[Re][prof].K_fluct / max(temporal_results[Re][prof].K, 1e-30) for Re in RE_LIST]
    ax1.bar(xp + (pi - 0.5) * width, K_fluct_ratio, width, color=TEMP_FILL[prof],
            edgecolor='k', hatch=TEMP_HATCH[prof], label=PROFILE_LABEL[prof])
ax1.set_xticks(xp)
ax1.set_xticklabels([f'Re={r}' for r in RE_LIST], color='k', fontsize=12)
style_axes(ax1, yl='$K_{fluct}/K$ (–)', title='Weight of time fluctuations (branches)')
legend(ax1, fs=10)

for pi, prof in enumerate(TEMPORAL_CONTROLS):
    dom_fracs = []
    for Re in RE_LIST:
        _, _, ff = temporal_results[Re][prof].fluct_modal_analysis()
        dom_fracs.append(ff.max())
    ax2.plot(xp, dom_fracs, **tempkws(prof, me=1), label=PROFILE_LABEL[prof])
ax2.set_xticks(xp)
ax2.set_xticklabels([f'Re={r}' for r in RE_LIST], color='k', fontsize=12)
style_axes(ax2, yl='Dominant-mode energy fraction (–)', title='Modal structure of $u_{mid}$ (mean)')
ax2.set_ylim(0, 1.02)
legend(ax2, fs=10)
figure_title(fig, 'Independent-LBM branch response (time-averaged field)')
save_fig(fig, os.path.join(OUTDIR, 'fig9_temporal_branches.png'))
print("   -> fig9_temporal_branches.png")

# ============================================================================
# Figure 10 : per-case temporal dynamics (grayscale fields, black curves)
# ============================================================================
for prof in TEMPORAL_CONTROLS:
    for Re in RE_LIST:
        sol = temporal_results[Re][prof]
        probe = sol.probe
        period = sol.period
        N = sol.N
        yy = np.linspace(0, 1, N)
        fig = plt.figure(figsize=(15.5, 8.2))
        gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.38)
        ax = fig.add_subplot(gs[0, 0])
        im = ax.pcolormesh(sol.ux, cmap=field_cmap(), vmin=-0.08, vmax=0.08)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        style_colorbar(cb, '$u$ (–)')
        style_axes(ax, xl='$x$ (–)', yl='$y$ (–)', title='$u$ — time mean')
        ax = fig.add_subplot(gs[0, 1])
        fluct = np.hypot(sol.ux_fluct, sol.uy_fluct)
        im = ax.pcolormesh(fluct, cmap=field_cmap())
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        style_colorbar(cb, "$(u'^2+v'^2)^{1/2}$ (–)")
        style_axes(ax, xl='$x$ (–)', yl='$y$ (–)', title='RMS fluctuations')
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(sol.ux[:, N // 2], yy, color='k', lw=2, label='$u_{mean}$')
        ax.plot(sol.ux_fluct[:, N // 2], yy, color='0.35', ls='--', lw=2, label="$u'_{rms}$")
        style_axes(ax, xl='$u$ (–)', yl='$y$ (–)', title='Centerline $x=0.5$ (mean + RMS)')
        legend(ax, fs=10)
        ax = fig.add_subplot(gs[1, 0])
        win = probe[-8 * period:]
        tax = np.arange(len(win)) / period
        ax.plot(tax, win, color='k', lw=1.2)
        style_axes(ax, xl='$t/P_{forced}$ (–)', yl='$u_{probe}$ (–)',
                   title='Central probe ($x$=$y$=0.5) — last 8 periods')
        ax = fig.add_subplot(gs[1, 1])
        seg = probe[-16384:]
        spec = np.abs(np.fft.rfft(seg - seg.mean()))
        freq = np.fft.rfftfreq(len(seg)) * period
        ax.semilogy(freq[1:], spec[1:], color='k', lw=1.3)
        ax.axvline(1.0, color='0.35', ls='--', lw=1.5, label='$f_{forced}$')
        ax.set_xlim(0, 4)
        style_axes(ax, xl='$f/f_{forced}$ (–)', yl='$|FFT|$ (–)', title='Spectrum — lock-in')
        legend(ax, fs=10)
        ax = fig.add_subplot(gs[1, 2])
        cols = phase_cmap(sol.phase_ux.shape[0])
        for k in range(sol.phase_ux.shape[0]):
            ax.plot(sol.phase_ux[k][:, N // 2], yy, color=cols[k], lw=1.6,
                    label=f'$\\phi_{k}$')
        style_axes(ax, xl='$u(y)$ per phase (–)', yl='$y$ (–)', title='Phase-averaged field (8 phases)')
        legend(ax, fs=9, ncol=2, loc='upper right')
        figure_title(fig, f'Time-dependent response — {PROFILE_LABEL[prof]} · Re={Re} '
                     f'(forced period P={period}, $K_{{fluct}}/K$={sol.K_fluct / max(sol.K, 1e-30):.1%})',
                     fs=15)
        save_fig(fig, os.path.join(OUTDIR, f'fig10_{prof}_Re{Re}_temporal.png'))
print("   -> fig10_<control>_Re<Re>_temporal.png  (fields + probe + phases)")

# ============================================================================
# Figure 11 : spectra lock-in — one figure per Re (1×2 each)
# ============================================================================
fig11_idx = 0
for ri, Re in enumerate(RE_LIST):
    fig11_idx += 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ci, prof in enumerate(TEMPORAL_CONTROLS):
        ax = axes[ci]
        sol = temporal_results[Re][prof]
        seg = sol.probe[-16384:]
        spec = np.abs(np.fft.rfft(seg - seg.mean()))
        freq = np.fft.rfftfreq(len(seg)) * sol.period
        ax.semilogy(freq[1:], spec[1:], color='k', lw=1.3)
        ax.axvline(1.0, color='0.35', ls='--', lw=1.2, label='$f_{forced}$')
        ax.set_xlim(0, 2.5)
        ax.axhline(0.05 * spec.max(), color='0.6', ls=':', lw=1)
        kk = sol.K_fluct / sol.K if sol.K > 0 else np.nan
        ax.set_title(f'{PROFILE_LABEL[prof]}\n$K_{{fluct}}/K$={kk:.1%}',
                     color='k', fontsize=11, fontweight='bold', pad=6)
        style_axes(ax, xl='$f/f_{forced}$ (–)', yl='$|FFT|$ (–)', fs=12, tick=11)
        legend(ax, fs=9)
    figure_title(fig, f'Spectra lock-in — Re={Re}')
    save_fig(fig, os.path.join(OUTDIR, f'fig11{fig11_idx}_spectra_Re{Re}.png'))
    print(f"   -> fig11{fig11_idx}_spectra_Re{Re}.png")

print("\nALL FIGURES REGENERATED (white / B&W style).")