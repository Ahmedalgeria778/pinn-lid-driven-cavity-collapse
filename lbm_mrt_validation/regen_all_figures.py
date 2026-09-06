# -*- coding: utf-8 -*-
"""
Regenerate ALL figures 1-11 from the saved field files (npz) + scalar CSVs +
Ghia (1982) reference tables loaded from ghiau.txt / ghiav.txt.

Does NOT re-run any simulation. Called by resume_ghia_re1000.py (which first
overwrites uniform_Re1000_N256.npz with the converged field) or standalone.

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

DARK, PANEL = "#090909", "#111111"
C_RE = {100: '#00e5ff', 500: '#ff6b35', 1000: '#7dff6b'}
C_PR = {'uniform': '#cccccc', 'sin_pi': '#ffcc00', 'sin_2pi': '#00e5ff', 'pinn': '#ff6b35',
        'pinn_t': '#e040fb', 'cheb_t': '#40c4ff'}
LS = {'uniform': '-', 'sin_pi': '--', 'sin_2pi': '-.', 'pinn': ':'}


def dark_axes(ax, xl='', yl='', title='', fs=13):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='w', labelsize=11)
    for sp in ax.spines.values():
        sp.set_edgecolor('#555')
    ax.grid(True, alpha=0.15, color='#333')
    if xl:
        ax.set_xlabel(xl, color='w', fontsize=fs)
    if yl:
        ax.set_ylabel(yl, color='w', fontsize=fs)
    if title:
        ax.set_title(title, color='w', fontsize=fs, fontweight='bold')


def load_npz(prof, Re):
    fp = os.path.join(FIELDS_DIR, f'{prof}_Re{Re}_N{PROD_N}.npz')
    with np.load(fp) as z:
        return {k: z[k] for k in z.files}


class FieldProxy:
    """Minimal proxy reproducing the plotting-relevant solver API."""
    def __init__(self, prof, Re, d):
        self.lid_profile, self.Re = prof, Re
        self.N = int(d['N'].item() if d['N'].ndim == 0 else d['N'])
        self.iters = int(d['iters'].item() if np.ndim(d['iters']) == 0 else d['iters'] if np.ndim(d['iters']) == 1 else d['iters'])
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

# ============================================================================
# Figure 1 : Ghia validation (correct references from ghiau.txt)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=DARK)
for ax, Re in zip([ax1, ax2], [100, 1000]):
    sol = ghia_solvers[Re]
    y, u_mid, _, _ = sol.centerline_profiles()
    g = GHIA[Re]
    ax.plot(u_mid, y, color=C_RE[Re], lw=2.5, label='LBM-MRT (N={})'.format(GHIA_N))
    ax.scatter(g['u'], g['y'], color='white', s=55, zorder=5, label='Ghia et al. (1982)')
    u_interp = np.interp(g['y'], y, u_mid)
    L2 = np.sqrt(np.mean((u_interp - np.array(g['u'])) ** 2))
    ax.text(0.04, 0.94, '$L_2$ error (–) = {:.4f}'.format(L2), transform=ax.transAxes,
            color='yellow', fontsize=12, bbox=dict(boxstyle='round', fc='black', alpha=0.5))
    dark_axes(ax, xl='$u/U_{lid}$ (–)  at $x=0.5$', yl='$y$ (–)',
              title=f'Re = {Re}  (uniform lid)', fs=13)
    ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=11)
fig.suptitle('LBM-MRT validation of the lid-driven cavity: Ghia et al. (1982)',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig1_ghia_validation.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig1_ghia_validation.png")

# ============================================================================
# Figure 2 : GCI
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
N_vals = GCI_N_list
ax1.plot(N_vals, gci_eps, 'o-', color='#00e5ff', lw=2.5, ms=10, label=r'$\epsilon$ (LBM)')
ax1.axhline(gci_exact, color='#ff6b35', ls='--', lw=2, label=r'$\epsilon^*$ = {:.4e}'.format(gci_exact))
for Nv, ev in zip(N_vals, gci_eps):
    ax1.text(Nv + 5, ev, '{:.4e}'.format(ev), color='#00e5ff', fontsize=11)
dark_axes(ax1, xl='$N$ (–)', yl=r'$\epsilon$ (–)', title=f'GCI = {gci_GCI:.2f}%, p = {p:.2f}')
ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w')
h = 1.0 / np.array(N_vals)
err = np.abs(np.array(gci_eps) - gci_exact)
ax2.loglog(h, err, 'o-', color='#7dff6b', lw=2.5, ms=10, label=r'$|\epsilon - \epsilon^*|$')
if not np.isnan(p):
    ax2.loglog(h, err[0] * (h / h[0]) ** p, '--', color='#ffaa00', lw=1.8, label='slope {:.2f}'.format(p))
dark_axes(ax2, xl='$h = 1/N$ (–)', yl=r'$|\epsilon - \epsilon^*|$ (–)', title='Spatial convergence')
ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w')
fig.suptitle('Grid convergence (GCI) — Re=500, controlled lid $A_2\\sin(2\\pi x)$',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig2_grid_convergence.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig2_grid_convergence.png")

# ============================================================================
# Figure 3 : Velocity magnitude fields
# ============================================================================
fig = plt.figure(figsize=(20, 13), facecolor=DARK)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)
x_coord = np.linspace(0, 1, PROD_N)
y_coord = np.linspace(0, 1, PROD_N)
X, Y = np.meshgrid(x_coord, y_coord)
for ri, Re in enumerate(RE_LIST):
    for pi, prof in enumerate(PROFILES):
        ax = fig.add_subplot(gs[ri, pi])
        sol = all_results[Re][prof]
        spd = np.sqrt(sol.ux ** 2 + sol.uy ** 2)
        cf = ax.contourf(X, Y, spd, levels=40, cmap='plasma')
        ax.streamplot(X, Y, sol.ux, sol.uy, color='w', linewidth=0.5, density=1.2, arrowsize=0.7)
        cb = plt.colorbar(cf, ax=ax, pad=0.01)
        cb.ax.yaxis.set_tick_params(color='w', labelcolor='w', labelsize=10)
        cb.set_label('$|\\mathbf{u}|$ (–)', color='w', fontsize=11)
        ax.set_title(f'Re={Re} · {PROFILE_LABEL[prof]}', color='w', fontsize=12, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle('Velocity magnitude $|\\mathbf{u}|$ (–) — MRT D2Q9 (N=256)',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig3_velocity_fields.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig3_velocity_fields.png")

# ============================================================================
# Figure 4 : Dissipation fields
# ============================================================================
fig = plt.figure(figsize=(20, 13), facecolor=DARK)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)
for ri, Re in enumerate(RE_LIST):
    for pi, prof in enumerate(PROFILES):
        ax = fig.add_subplot(gs[ri, pi])
        sol = all_results[Re][prof]
        d = sol.dissipation
        vmax = np.percentile(d, 99)
        cf = ax.contourf(X, Y, np.clip(d, 0, vmax), levels=40, cmap='inferno')
        cb = plt.colorbar(cf, ax=ax, pad=0.01)
        cb.ax.yaxis.set_tick_params(color='w', labelcolor='w', labelsize=10)
        cb.set_label('$\\Phi$ (–)', color='w', fontsize=11)
        ax.text(0.03, 0.04, f'$\\epsilon$ = {sol.eps:.2e} (–)', transform=ax.transAxes,
                color='yellow', fontsize=10, bbox=dict(boxstyle='round', fc='black', alpha=0.5))
        ax.set_title(f'Re={Re} · {PROFILE_LABEL[prof]}', color='w', fontsize=12, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle('Viscous dissipation $\\Phi$ (–) — $\\epsilon = \\iint \\Phi\\,\\mathrm{d}\\Omega$',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig4_dissipation_fields.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig4_dissipation_fields.png")

# ============================================================================
# Figure 5 : Total dissipation comparison
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=DARK)
xp = np.arange(len(PROFILES))
for ax, Re in zip(axes, RE_LIST):
    eps_unif = all_results[Re]['uniform'].eps
    eps_vals = [all_results[Re][p].eps for p in PROFILES]
    ax.bar(xp, eps_vals, color=[C_PR[p] for p in PROFILES], alpha=0.85, width=0.6)
    ax.axhline(eps_unif, color='w', ls=':', lw=1.5, alpha=0.6, label='$\\epsilon$ (uniform lid)')
    for xi, ev in zip(xp, eps_vals):
        drel = (ev - eps_unif) / eps_unif * 100 if eps_unif > 0 else 0
        ax.text(xi, ev * 1.04, f'{drel:+.1f}%', ha='center', color='yellow', fontsize=11, fontweight='bold')
    ax.set_xticks(xp)
    ax.set_xticklabels([PROFILE_LABEL[p] for p in PROFILES], color='w', fontsize=12, rotation=15, ha='right')
    dark_axes(ax, yl=r'$\epsilon = \iint \Phi \, \mathrm{d}\Omega$ (–)', title=f'Re = {Re}', fs=13)
    ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=11)
fig.suptitle('Total dissipation $\\epsilon$ (–) — lid-profile comparison',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig5_dissipation_comparison.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig5_dissipation_comparison.png")

# ============================================================================
# Figure 6 : Flow modal analysis
# ============================================================================
fig = plt.figure(figsize=(20, 12), facecolor=DARK)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
modes = np.arange(1, 11)
for ri, Re in enumerate(RE_LIST):
    for pi, prof in enumerate(PROFILES):
        ax = fig.add_subplot(gs[ri, pi])
        sol = all_results[Re][prof]
        _, _, fracs = sol.modal_analysis()
        fracs_pct = fracs * 100
        bars = ax.bar(modes, fracs_pct, color=C_PR[prof], alpha=0.85)
        dom = np.argmax(fracs) + 1
        if dom <= 10:
            bars[dom - 1].set_color('#ffd700')
        dark_axes(ax, xl='Mode index $n$ (–)', yl='Modal energy fraction (%)',
                  title=f'Re={Re} · {PROFILE_LABEL[prof]}\nMode {dom}: {fracs_pct[dom - 1]:.1f}%', fs=11)
fig.suptitle('Modal analysis of $u_{mid}(y)$ at $x=0.5$ — MRT D2Q9',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig6_modal_analysis.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig6_modal_analysis.png")

# ============================================================================
# Figure 7 : Centerline profiles (both Ghia tables)
# ============================================================================
fig = plt.figure(figsize=(20, 12), facecolor=DARK)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.32)
for ri, Re in enumerate(RE_LIST):
    ax = fig.add_subplot(gs[ri, 0])
    for prof in PROFILES:
        sol = all_results[Re][prof]
        y, u_mid, _, _ = sol.centerline_profiles()
        ax.plot(u_mid, y, color=C_PR[prof], lw=2, ls=LS[prof], label=PROFILE_LABEL[prof])
    if Re in GHIA:
        g = GHIA[Re]
        ax.scatter(g['u'], g['y'], color='w', s=22, zorder=5, label='Ghia et al. (1982)')
    dark_axes(ax, xl='$u/U_{lid}$ (–)  at $x=0.5$', yl='$y$ (–)',
              title=f'Re={Re} — $u(y)$', fs=13)
    ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10, loc='lower right')

    ax = fig.add_subplot(gs[ri, 1])
    for prof in PROFILES:
        sol = all_results[Re][prof]
        x, v_mid = sol.centerline_profiles()[2:4]
        ax.plot(x, v_mid, color=C_PR[prof], lw=2, ls=LS[prof], label=PROFILE_LABEL[prof])
    if Re in GHIA_V:
        gv = GHIA_V[Re]
        ax.scatter(gv['x'], gv['v'], color='w', s=22, zorder=5, label='Ghia et al. (1982)')
    ax.axhline(0, color='#333', lw=0.8)
    dark_axes(ax, xl='$x$ (–)', yl='$v/U_{lid}$ (–)  at $y=0.5$',
              title=f'Re={Re} — $v(x)$', fs=13)
    ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10)
fig.suptitle('Centerline profiles — $u(0.5,y)$ and $v(x,0.5)$',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig7_centerline_profiles.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig7_centerline_profiles.png")

# ============================================================================
# Figure 8 : Robustness
# ============================================================================
Re_vals = RE_LIST
A2_pinn = {Re: all_results[Re]['pinn'].modal_analysis()[1][1] for Re in Re_vals}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK)
ax1.plot(Re_vals, list(A2_pinn.values()), 'o-', color='#ff6b35', lw=2.5, ms=10,
         label='$A_2$ PINN (mean)')
ax1.axhline(np.mean(list(A2_pinn.values())), color='#ff6b35', ls=':', lw=1.2, alpha=0.6, label='$A_2$ PINN (mean)')
dark_axes(ax1, xl='$Re$ (–)', yl='$A_2$ (–)', title='Mode 2 — Reynolds robustness')
ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w')

xp = np.arange(len(Re_vals))
width = 0.18
for i, prof in enumerate(PROFILES):
    Z_vals = [all_results[Re][prof].Z for Re in Re_vals]
    ax2.bar(xp + i * width, Z_vals, width, color=C_PR[prof], alpha=0.85, label=PROFILE_LABEL[prof])
ax2.set_xticks(xp + 1.5 * width)
ax2.set_xticklabels([f'Re={r}' for r in Re_vals], color='w', fontsize=12)
dark_axes(ax2, yl=r'$Z = \iint \omega^2 \, \mathrm{d}\Omega$ (–)', title='Enstrophy — lid profiles')
ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10, ncol=2)
fig.suptitle('Mode-2 amplitude and enstrophy robustness', color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig8_robustness.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig8_robustness.png")

# ============================================================================
# Figure 9 : Time-averaged branch response
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK)
xp = np.arange(len(RE_LIST))
width = 0.38
for pi, prof in enumerate(TEMPORAL_CONTROLS):
    K_fluct_ratio = [temporal_results[Re][prof].K_fluct / max(temporal_results[Re][prof].K, 1e-30) for Re in RE_LIST]
    ax1.bar(xp + (pi - 0.5) * width, K_fluct_ratio, width, color=C_PR[prof], alpha=0.9, label=PROFILE_LABEL[prof])
ax1.set_xticks(xp)
ax1.set_xticklabels([f'Re={r}' for r in RE_LIST], color='w', fontsize=12)
dark_axes(ax1, yl='$K_{fluct}/K$ (–)', title='Weight of time fluctuations (branches)')
ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10)

for pi, prof in enumerate(TEMPORAL_CONTROLS):
    dom_fracs = []
    for Re in RE_LIST:
        _, _, ff = temporal_results[Re][prof].fluct_modal_analysis()
        dom_fracs.append(ff.max())
    ax2.plot(xp, dom_fracs, 'o-', lw=2.5, ms=9, color=C_PR[prof], label=PROFILE_LABEL[prof])
ax2.set_xticks(xp)
ax2.set_xticklabels([f'Re={r}' for r in RE_LIST], color='w', fontsize=12)
dark_axes(ax2, yl='Dominant-mode energy fraction (–)', title='Modal structure of $u_{mid}$ (mean)')
ax2.set_ylim(0, 1.02)
ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10)
fig.suptitle('Independent-LBM branch response (time-averaged field)',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig9_temporal_branches.png'), dpi=300, bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig9_temporal_branches.png")

# ============================================================================
# Figure 10 : per-case temporal dynamics
# ============================================================================
for prof in TEMPORAL_CONTROLS:
    for Re in RE_LIST:
        sol = temporal_results[Re][prof]
        probe = sol.probe
        period = sol.period
        N = sol.N
        yy = np.linspace(0, 1, N)
        fig = plt.figure(figsize=(15.5, 8.2), facecolor=DARK)
        gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.38)
        ax = fig.add_subplot(gs[0, 0])
        im = ax.pcolormesh(sol.ux, cmap='inferno', vmin=-0.08, vmax=0.08)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label('$u$ (–)', color='w', fontsize=11)
        dark_axes(ax, xl='$x$ (–)', yl='$y$ (–)', title='$u$ — time mean')
        ax = fig.add_subplot(gs[0, 1])
        fluct = np.hypot(sol.ux_fluct, sol.uy_fluct)
        im = ax.pcolormesh(fluct, cmap='viridis')
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("$(u'^2+v'^2)^{1/2}$ (–)", color='w', fontsize=11)
        dark_axes(ax, xl='$x$ (–)', yl='$y$ (–)', title='RMS fluctuations')
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(sol.ux[:, N // 2], yy, color='#00e5ff', lw=2, label='$u_{mean}$')
        ax.plot(sol.ux_fluct[:, N // 2], yy, color='#ff6b35', ls='--', lw=2, label="$u'_{rms}$")
        dark_axes(ax, xl='$u$ (–)', yl='$y$ (–)', title='Centerline $x=0.5$ (mean + RMS)')
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10)
        ax = fig.add_subplot(gs[1, 0])
        win = probe[-8 * period:]
        tax = np.arange(len(win)) / period
        ax.plot(tax, win, color='#7dff6b', lw=1.2)
        dark_axes(ax, xl='$t/P_{forced}$ (–)', yl='$u_{probe}$ (–)',
                  title='Central probe ($x$=$y$=0.5) — last 8 periods')
        ax = fig.add_subplot(gs[1, 1])
        seg = probe[-16384:]
        spec = np.abs(np.fft.rfft(seg - seg.mean()))
        freq = np.fft.rfftfreq(len(seg)) * period
        ax.semilogy(freq[1:], spec[1:], color='#ffcc00', lw=1.3)
        ax.axvline(1.0, color='white', ls='--', lw=1.5, label='$f_{forced}$')
        ax.set_xlim(0, 4)
        dark_axes(ax, xl='$f/f_{forced}$ (–)', yl='$|FFT|$ (–)', title='Spectrum — lock-in')
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=10)
        ax = fig.add_subplot(gs[1, 2])
        cmap = plt.get_cmap('cool')
        for k in range(sol.phase_ux.shape[0]):
            ax.plot(sol.phase_ux[k][:, N // 2], yy, color=cmap(k / (sol.phase_ux.shape[0] - 1)),
                    lw=1.6, label=f'$\\phi_{k}$')
        dark_axes(ax, xl='$u(y)$ per phase (–)', yl='$y$ (–)', title='Phase-averaged field (8 phases)')
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=9, ncol=2, loc='upper right')
        fig.suptitle(f'Time-dependent response — {PROFILE_LABEL[prof]} · Re={Re} '
                     f'(forced period P={period}, $K_{{fluct}}/K$={sol.K_fluct / max(sol.K, 1e-30):.1%})',
                     color='w', fontsize=15, fontweight='bold')
        fig.savefig(os.path.join(OUTDIR, f'fig10_{prof}_Re{Re}_temporal.png'),
                    dpi=300, bbox_inches='tight', facecolor=DARK)
        plt.close()
print("   -> fig10_<control>_Re<Re>_temporal.png  (fields + probe + phases)")

# ============================================================================
# Figure 11 : spectra lock-in (all temporal cases)
# ============================================================================
fig, axes = plt.subplots(len(RE_LIST), len(TEMPORAL_CONTROLS),
                         figsize=(12.5, 9), facecolor=DARK)
for ri, Re in enumerate(RE_LIST):
    for ci, prof in enumerate(TEMPORAL_CONTROLS):
        ax = axes[ri, ci]
        sol = temporal_results[Re][prof]
        seg = sol.probe[-16384:]
        spec = np.abs(np.fft.rfft(seg - seg.mean()))
        freq = np.fft.rfftfreq(len(seg)) * sol.period
        ax.semilogy(freq[1:], spec[1:], color=C_RE[Re], lw=1.2)
        ax.axvline(1.0, color='white', ls='--', lw=1.2)
        ax.set_xlim(0, 2.5)
        ax.axhline(0.05 * spec.max(), color='#333', ls=':', lw=1)
        kk = sol.K_fluct / sol.K if sol.K > 0 else np.nan
        ax.set_title(f'{PROFILE_LABEL[prof]} · Re={Re} — $K_{{fluct}}/K$={kk:.1%}',
                     color='w', fontsize=11, fontweight='bold')
        dark_axes(ax, xl='$f/f_{forced}$ (–)', yl='$|FFT|$ (–)')
fig.suptitle('All time-dependent branches lock on the forced frequency $f=1/P$',
             color='w', fontsize=16, fontweight='bold')
fig.savefig(os.path.join(OUTDIR, 'fig11_temporal_lockin_all.png'), dpi=300,
            bbox_inches='tight', facecolor=DARK)
plt.close()
print("   -> fig11_temporal_lockin_all.png (global lock-in of all 6 cases)")

print("\nALL FIGURES REGENERATED from saved fields + Ghia references.")