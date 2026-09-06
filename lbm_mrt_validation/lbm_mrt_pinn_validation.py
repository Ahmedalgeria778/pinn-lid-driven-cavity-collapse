# -*- coding: utf-8 -*-
"""
LBM-MRT D2Q9 – Validation CFD pour publication (version corrigée)
Beniaiche et al. (2026) – PINN optimal control validation
- MRT complet (Correction Algébrique de l'Espace des Moments)
- Half-Way Bounce-Back PULL (Ultra-stable, conservation de masse absolue)
- Étude de convergence de maillage (N=128,256,512)
- GCI / Richardson
- 4 profils de lid : uniform, sin(πx), A₂·sin(2πx), PINN (3 modes)
- Analyse modale sur u_mid(y) (réponse de l'écoulement)
- Grandeurs intégrales : dissipation, énergie, enstrophie
- Validation Ghia (lid uniforme)
- Parallélisation Numba (tous cœurs)
- Sauvegarde automatique des résultats (CSV)
"""

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numba import njit, prange
import time
import os
import sys
import warnings
import csv

warnings.filterwarnings('ignore')

# Sortie console robuste aux caractères Unicode (Δ, ∫, ω, é, …)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
#  Constantes D2Q9 et MRT
# ----------------------------------------------------------------------
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)

# Tableau des directions opposées (pour le rebond parfait)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# Matrice de transformation M (9 moments)
M = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
    [4, -2, -2, -2, -2, 1, 1, 1, 1],
    [0, 1, 0, -1, 0, 1, -1, -1, 1],
    [0, -2, 0, 2, 0, 1, -1, -1, 1],
    [0, 0, 1, 0, -1, 1, 1, -1, -1],
    [0, 0, -2, 0, 2, 1, 1, -1, -1],
    [0, 1, -1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 1, -1]
], dtype=np.float64)
M_INV = np.linalg.inv(M)


# ----------------------------------------------------------------------
#  Fonction LBM-MRT itérative (Numba parallélisée)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def mrt_step(f, M, M_INV, S_diag, EX, EY, W, U_lid2d, t_idx, N, OPP):
    Q = 9
    rho = np.zeros((N, N))
    ux = np.zeros((N, N))
    uy = np.zeros((N, N))
    f_post = np.empty_like(f)

    # 1 & 2) Macroscopiques & Collision MRT correcte (Espace des moments)
    for i in prange(N):
        for j in range(N):
            r = 0.0
            mx = 0.0
            my = 0.0
            for q in range(Q):
                r += f[q, i, j]
                mx += EX[q] * f[q, i, j]
                my += EY[q] * f[q, i, j]

            # Protection anti-divergence
            rho[i, j] = max(r, 1e-10)
            ux_ = mx / rho[i, j]
            uy_ = my / rho[i, j]
            ux[i, j] = ux_
            uy[i, j] = uy_

            u2 = ux_ * ux_ + uy_ * uy_

            # Équilibre dans l'espace des moments
            m_eq = np.zeros(Q)
            m_eq[0] = r
            m_eq[1] = -2.0 * r + 3.0 * r * u2
            m_eq[2] = r - 3.0 * r * u2
            m_eq[3] = r * ux_
            m_eq[4] = -r * ux_
            m_eq[5] = r * uy_
            m_eq[6] = -r * uy_
            m_eq[7] = r * (ux_ * ux_ - uy_ * uy_)
            m_eq[8] = r * ux_ * uy_

            # Projection vers l'espace des moments
            m = np.zeros(Q)
            for q in range(Q):
                for k in range(Q):
                    m[q] += M[q, k] * f[k, i, j]

            # Relaxation exacte sur la DIAGONALE
            m_star = np.zeros(Q)
            for q in range(Q):
                m_star[q] = m[q] - S_diag[q] * (m[q] - m_eq[q])

            # Retour à l'espace des populations post-collision
            for q in range(Q):
                val = 0.0
                for k in range(Q):
                    val += M_INV[q, k] * m_star[k]
                f_post[q, i, j] = val

    # 3 & 4) Streaming PULL + Half-Way Bounce-Back (Ultra-stable)
    f_new = np.empty_like(f)
    for i in prange(N):
        for j in range(N):
            for q in range(Q):
                ex = int(EX[q])
                ey = int(EY[q])
                opp_q = OPP[q]

                # Coordonnées du nœud de provenance
                prev_i = i - ey
                prev_j = j - ex

                if 0 <= prev_i < N and 0 <= prev_j < N:
                    # Intérieur du domaine
                    f_new[q, i, j] = f_post[q, prev_i, prev_j]
                else:
                    # Mur : Half-way bounce back
                    f_new[q, i, j] = f_post[opp_q, i, j]

                    # Injection de quantité de mouvement sur le Lid (Mur Nord),
                    # profil éventuellement dépendant du temps : U_lid2d[t_idx, j]
                    if prev_i == N:
                        uw = U_lid2d[t_idx, j]
                        rho_w = rho[i, j]
                        if q == 7:  # SW
                            f_new[7, i, j] -= 6.0 * W[5] * rho_w * uw
                        elif q == 8:  # SE
                            f_new[8, i, j] += 6.0 * W[6] * rho_w * uw

    return f_new, rho, ux, uy


# ----------------------------------------------------------------------
#  Classe solveur
# ----------------------------------------------------------------------
class LBM_MRT_Solver:
    def __init__(self, Re, N, lid_profile='uniform', max_iter=150000, tol=1e-9):
        self.Re = Re
        self.N = N
        self.lid_profile = lid_profile
        self.max_iter = max_iter
        self.tol = tol

        # Vitesse de référence fixe pour incompressibilité (Ma ~ 0.087)
        self.U_ref = 0.05
        self.nu_lbm = self.U_ref * N / Re
        self.tau = 3.0 * self.nu_lbm + 0.5
        self.omega = 1.0 / self.tau

        # Le véritable vecteur de relaxation MRT (Diagonale de S)
        # s_e = 1.4, s_eps = 1.4, s_q = 1.2
        self.S_diag = np.array([0., 1.4, 1.4, 0., 1.2, 0., 1.2, self.omega, self.omega], dtype=np.float64)

        # Profil de lid (physique, normalisé).
        # Les contrôles 'pinn', 'pinn_t', 'cheb_t' sont chargés depuis les NPZ
        # générés par make_pinn_lid_series.py (contrôle PINN FINAL campaign 13,
        # seed 42) :  'pinn'   = moyenne temporelle du contrôle Fourier (stationnaire)
        #              'pinn_t' = contrôle Fourier complet U_lid(x,t) (périodique)
        #              'cheb_t' = contrôle Chebyshev_mod complet U_lid(x,t) (périodique)
        # La période forcée = 2 (période naturelle des modes temporels cos(j*pi*t)).
        x_norm = np.linspace(0, 1, N)
        series_map = {"pinn": ("fourier", "mean"),
                      "pinn_t": ("fourier", "series"),
                      "cheb_t": ("chebyshev_mod", "series")}
        if lid_profile in series_map:
            basis, kind = series_map[lid_profile]
            fname = os.path.join(SCRIPT_DIR, "lid_series", f"{basis}_Re{Re}.npz")
            if not os.path.exists(fname):
                raise FileNotFoundError(f"{fname} manquant. Lancer make_pinn_lid_series.py d'abord.")
            with np.load(fname) as z:
                self.period = int(z["NT_PER"]) if kind == "series" else 1
                data = z["U_series"] if kind == "series" else np.ascontiguousarray(z["U_mean"])[None, :]
            self.U_lid_norm = np.ascontiguousarray(data, dtype=np.float64)
        else:
            A2_table = {100: 0.68758, 500: 0.68355, 1000: 0.58628}
            if lid_profile == 'uniform':
                self.U_lid_norm = np.ones(N)
            elif lid_profile == 'sin_pi':
                self.U_lid_norm = np.sin(np.pi * x_norm)
            elif lid_profile == 'sin_2pi':
                A2 = A2_table.get(Re, 0.68)
                self.U_lid_norm = A2 * np.sin(2.0 * np.pi * x_norm)
            else:
                raise ValueError("Unknown lid_profile")
            self.period = 1
            self.U_lid_norm = np.ascontiguousarray(self.U_lid_norm)[None, :]

        self.U_lid2d = self.U_ref * self.U_lid_norm          # (period, N), vitesse physique

        # Initialisation à l'équilibre statique (ρ=1, u=0)
        self.f = np.ascontiguousarray(W[:, None, None] * np.ones((N, N))[None], dtype=np.float64)

    def run(self, verbose=True, logfile=None):
        N = self.N
        f = self.f.copy()
        S_diag = self.S_diag
        period = self.period
        t0 = time.time()
        ux_prev = np.zeros((N, N))
        convergence_history = []
        n_cycles_total = max(1, self.max_iter // period)
        settle_at = (n_cycles_total // 2) * period          # début de la moyenne temporelle

        # Accumulateurs de moyenne temporelle (champ moyen + fluctuations)
        m_sum_ux = np.zeros((N, N))
        m_sum_uy = np.zeros((N, N))
        m2_sum_ux = np.zeros((N, N))
        m2_sum_uy = np.zeros((N, N))
        m_rho = np.zeros((N, N))
        m_count = 0
        probe_ring = np.zeros(period)                       # fluct. sonde pour périodicité
        period_ok = 0

        for it in range(1, self.max_iter + 1):
            t_idx = (it - 1) % period
            f, rho, ux, uy = mrt_step(f, M, M_INV, S_diag, EX, EY, W, self.U_lid2d, t_idx, N, OPP)
            if it > settle_at:
                m_sum_ux += ux; m_sum_uy += uy
                m2_sum_ux += ux * ux; m2_sum_uy += uy * uy
                m_rho += rho
                m_count += 1

            if it % 2000 == 0:
                if period == 1:
                    err = np.max(np.abs(ux - ux_prev))
                else:
                    # Erreur de périodicité : comparaison de la sonde avec le même
                    # phase un cycle plus tôt (slot réutilisé du ring buffer).
                    probe_now = ux[N // 2, N // 2]
                    err = abs(probe_now - probe_ring[t_idx])
                    probe_ring[t_idx] = probe_now
                if logfile:
                    convergence_history.append([it, err, time.time() - t0])
                if verbose and it % 20000 == 0:
                    print(f"   it={it:6d} Δu={err:.2e} time={time.time() - t0:.1f}s")
                if err < self.tol and it > (5000 if period == 1 else 4 * period):
                    period_ok += 1
                    if period_ok >= (2 if period == 1 else 3):
                        if verbose:
                            print(f"   Converged at it={it} (Δu={err:.2e}, périodique={period > 1})")
                        break
                else:
                    period_ok = 0
                ux_prev = ux.copy()

        if m_count > 0:
            mean_ux = m_sum_ux / m_count
            mean_uy = m_sum_uy / m_count
            fluct_ux = np.sqrt(np.clip(m2_sum_ux / m_count - mean_ux ** 2, 0, None))
            fluct_uy = np.sqrt(np.clip(m2_sum_uy / m_count - mean_uy ** 2, 0, None))
            mean_rho = m_rho / m_count
        else:
            mean_ux, mean_uy, fluct_ux, fluct_uy, mean_rho = ux, uy, np.zeros_like(ux), np.zeros_like(uy), rho

        self.ux = mean_ux / self.U_ref
        self.uy = mean_uy / self.U_ref
        self.rho = mean_rho
        self.ux_fluct = fluct_ux / self.U_ref
        self.uy_fluct = fluct_uy / self.U_ref
        self.mean_count = m_count
        self.periodic = (period > 1)
        self.iters = it
        if logfile and convergence_history:
            np.savetxt(logfile, np.array(convergence_history), header='iteration,error,time', delimiter=',')
        if verbose:
            print(f"   Re={self.Re} N={N} profile={self.lid_profile} "
                  f"(period={period}) finished in {time.time() - t0:.1f}s")
        return self

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

        # Énergie cinétique des fluctuations (réponse instationnaire du régime
        # périodique) : K_fluct = ∫ 0.5*(u'² + v'²) dΩ
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
        # Structure modale du champ de FLUCTUATIONS RMS : u_mid(y) construit à
        # partir de ux_fluct (significatif même quand la moyenne temporelle est ~nulle,
        # cf. branche temporelle Chebyshev).
        mid = self.N // 2
        y = np.linspace(0, 1, self.N)
        u_mid_rms = self.ux_fluct[:, mid]
        modes = np.arange(1, 11)
        amps = np.array([2.0 * np.trapz(u_mid_rms * np.sin(n * np.pi * y), y) for n in modes])
        E = amps ** 2
        return modes, amps, E / E.sum() if E.sum() > 0 else E

    def dominant_mean_mode(self):
        return int(np.argmax(self.modal_analysis()[2]) + 1)


# ----------------------------------------------------------------------
#  Données de référence Ghia
# ----------------------------------------------------------------------
GHIA = {
    100: {'y': [0., 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                0.9688, 0.9766, 1.],
          'u': [0., -0.03717, -0.04192, -0.04775, -0.06434, -0.1015, -0.15662, -0.2109, -0.20581, -0.13641, 0.00332,
                0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.]},
    1000: {
        'y': [0., 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
              0.9609, 0.9688, 1.],
        'u': [0., -0.18109, -0.20196, -0.2222, -0.2973, -0.38289, -0.27805, -0.10648, -0.0608, 0.05702, 0.18719,
              0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.]}
}


# ----------------------------------------------------------------------
#  Étude de convergence de maillage (GCI, Richardson)
# ----------------------------------------------------------------------
def grid_convergence_study(Re, lid_profile, N_list=[128, 256, 512], max_iter=2000000, outdir='.'):
    eps_vals, solvers = [], {}
    for N in N_list:
        print(f"\n   N={N} ...")
        logfile = os.path.join(outdir, f'convergence_Re{Re}_{lid_profile}_N{N}.log')
        solver = LBM_MRT_Solver(Re, N, lid_profile=lid_profile, max_iter=max_iter, tol=1e-9)
        solver.run(verbose=True, logfile=logfile)
        eps, _, _ = solver.compute_integrals()
        eps_vals.append(eps)
        solvers[N] = solver
        # Sauvegarde immédiate des résultats partiels
        with open(os.path.join(outdir, 'gci_progress.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([Re, lid_profile, N, eps, solver.iters])
    e1, e2, e3 = eps_vals
    r = 2.0
    if abs(e2 - e3) > 1e-20 and abs(e1 - e2) > 1e-20:
        p = np.log(abs((e1 - e2) / (e2 - e3))) / np.log(r)
        eps_exact = e3 + (e3 - e2) / (r ** p - 1.0)
        GCI = 1.25 * abs(e3 - e2) / (abs(e3) * (r ** p - 1.0)) * 100
    else:
        p, eps_exact, GCI = np.nan, e3, np.nan
    return solvers, eps_vals, p, eps_exact, GCI


# ----------------------------------------------------------------------
#  Sauvegarde des résultats dans un CSV global
# ----------------------------------------------------------------------
def save_results_to_csv(all_results, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Re', 'profile', 'N', 'eps', 'K', 'Z', 'iters'])
        for Re in all_results:
            for prof, sol in all_results[Re].items():
                writer.writerow([Re, prof, sol.N, sol.eps, sol.K, sol.Z, sol.iters])


def save_temporal_results_to_csv(temporal_results, filename):
    # Réponse du flux forcé périodiquement : métriques sur le CHAMP MOYENNÉ DANS LE
    # TEMPS + énergie cinétique des fluctuations (K_fluct) + analyses modales de
    # u_mid (moyenne) et de u_mid_rms (fluctuations).
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Re', 'control', 'N', 'period_iters', 'iters', 'eps_mean', 'K_mean',
                         'Z_mean', 'K_fluct', 'fluct_ratio', 'mode_dom', 'mode_frac', 'A2_flow',
                         'mode_dom_fluct', 'mode_frac_fluct', 'A2_flow_fluct'])
        for Re in temporal_results:
            for prof, sol in temporal_results[Re].items():
                modes, amps, fracs = sol.modal_analysis()
                dom = int(np.argmax(fracs) + 1)
                fm, fa, ff = sol.fluct_modal_analysis()
                dom_f = int(np.argmax(ff) + 1)
                fluct_ratio = (sol.K_fluct / sol.K) if sol.K > 0 else np.nan
                writer.writerow([Re, prof, sol.N, sol.period, sol.iters, sol.eps, sol.K,
                                 sol.Z, sol.K_fluct, fluct_ratio, dom, fracs[dom - 1], amps[1],
                                 dom_f, ff[dom_f - 1], fa[1]])


# ----------------------------------------------------------------------
#  Paramètres de l'étude
# ----------------------------------------------------------------------
RE_LIST = [100, 500, 1000]
PROFILES = ['uniform', 'sin_pi', 'sin_2pi', 'pinn']
PROFILE_LABEL = {'uniform': 'Uniform U=1', 'sin_pi': r'$\sin(\pi x)$', 'sin_2pi': r'$A_2 \sin(2\pi x)$',
                 'pinn': 'PINN (mean)', 'pinn_t': 'PINN Fourier (t-d)', 'cheb_t': 'Chebyshev (t-d)'}
TEMPORAL_CONTROLS = ['pinn_t', 'cheb_t']   # admissibilité physique des branches (forçage périodique)
GHIA_N, PROD_N, MAX_ITER = 256, 256, 200000
GCI_N_list = [128, 256, 512]  # Attention: N=512 prendra du temps à s'exécuter

# A20 du contrôle FINAL identifié (campagne 13, seed 42, branche Fourier) — figure 8
A2_PINN_TABLE = {100: 0.68758, 500: 0.68355, 1000: 0.58628}

OUTDIR = "./lbm_mrt_validation"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------
#  Exécution des simulations
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("LBM-MRT VALIDATION CFD – PUBLICATION GRADE (version corrigée)")
    print("=" * 70)

    # 1) Validation Ghia (lid uniforme)
    print("\n[1] Validation Ghia (lid uniforme)")
    ghia_solvers = {}
    for Re in [100, 1000]:
        print(f"\n Re={Re}, N={GHIA_N}")
        logfile = os.path.join(OUTDIR, f'convergence_ghia_Re{Re}.log')
        solver = LBM_MRT_Solver(Re, GHIA_N, 'uniform', MAX_ITER, 1e-9)
        solver.run(verbose=True, logfile=logfile)
        solver.compute_integrals()
        ghia_solvers[Re] = solver

    # 2) Étude de convergence GCI (Re=500, sin_2pi)
    print("\n[2] Grid Convergence Study (Re=500, sin_2pi)")
    gci_solvers, gci_eps, gci_p, gci_exact, gci_GCI = grid_convergence_study(
        500, 'sin_2pi', GCI_N_list, MAX_ITER, outdir=OUTDIR)

    # 3) Comparaison des 4 profils pour chaque Re
    print("\n[3] Comparaison 4 profils × 3 Re")
    all_results = {Re: {} for Re in RE_LIST}
    for Re in RE_LIST:
        for prof in PROFILES:
            print(f"\n   Re={Re}, profile={prof}")
            logfile = os.path.join(OUTDIR, f'convergence_Re{Re}_{prof}.log')
            solver = LBM_MRT_Solver(Re, PROD_N, prof, MAX_ITER, 1e-9)
            solver.run(verbose=True, logfile=logfile)
            solver.compute_integrals()
            all_results[Re][prof] = solver
            # Sauvegarde incrémentale
            save_results_to_csv(all_results, os.path.join(OUTDIR, 'results_all.csv'))

    # Sauvegarde finale
    save_results_to_csv(all_results, os.path.join(OUTDIR, 'results_all.csv'))

    # 3bis) Branches temporelles : injection des contrôles PINN complets U_lid(x,t)
    print("\n[3bis] Branches temporelles (forçage périodique) — admissibilité physique")
    print("       Contrôles : 'pinn_t' = Fourier/Mode-2 quasi-stationnaire + composantes temporelles")
    print("                   'cheb_t' = Chebyshev_mod dominé par la branche temporelle (0,1)")
    temporal_results = {Re: {} for Re in RE_LIST}
    for Re in RE_LIST:
        for prof in TEMPORAL_CONTROLS:
            print(f"\n   Re={Re}, control={prof}")
            logfile = os.path.join(OUTDIR, f'convergence_Re{Re}_{prof}.log')
            solver = LBM_MRT_Solver(Re, PROD_N, prof, MAX_ITER, 1e-9)
            solver.run(verbose=True, logfile=logfile)
            solver.compute_integrals()
            temporal_results[Re][prof] = solver
            save_temporal_results_to_csv(temporal_results,
                                         os.path.join(OUTDIR, 'results_temporal.csv'))

    print("\n[Tableau C] Réponse des branches contrôlées (champ moyenné dans le temps)")
    print(f"{'Re':>5} | {'Contrôle':<8} | {'ε':>10} | {'K':>10} | {'K_fluct':>10} | {'Kf/K':>7} | {'Mode mov':>8} | {'Mode fluct':>10}")
    print("-" * 85)
    for Re in RE_LIST:
        for prof in TEMPORAL_CONTROLS:
            sol = temporal_results[Re][prof]
            fm, fa, ff = sol.fluct_modal_analysis()
            dom_f = int(np.argmax(ff) + 1)
            kf_k = sol.K_fluct / sol.K if sol.K > 0 else np.nan
            print(f"{Re:>5} | {prof:<8} | {sol.eps:>10.4e} | {sol.K:>10.4e} | {sol.K_fluct:>10.4e} | "
                  f"{kf_k:>7.1%} | {sol.dominant_mean_mode():>8} | {dom_f:>6} ({ff[dom_f-1]:.0%})")
    print("       K_fluct/K : poids des fluctuations temporelles de la réponse (branche temporelle)")
    print("       Mode mov = dominant du champ moyen ; Mode fluct = dominant du champ RMS")

    # ----------------------------------------------------------------------
    #  Post-traitement et figures (correction des grilles)
    # ----------------------------------------------------------------------
    print("\n[4] Génération des figures...")
    DARK = '#090909'
    PANEL = '#111111'
    C_RE = {100: '#00e5ff', 500: '#ff6b35', 1000: '#7dff6b'}
    C_PR = {'uniform': '#cccccc', 'sin_pi': '#ffcc00', 'sin_2pi': '#00e5ff', 'pinn': '#ff6b35'}
    LS = {'uniform': '-', 'sin_pi': '--', 'sin_2pi': '-.', 'pinn': ':'}


    def dark_axes(ax, xl='', yl='', title='', fs=9):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors='w', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#555')
        ax.grid(True, alpha=0.15, color='#333')
        if xl:
            ax.set_xlabel(xl, color='w', fontsize=fs)
        if yl:
            ax.set_ylabel(yl, color='w', fontsize=fs)
        if title:
            ax.set_title(title, color='w', fontsize=fs, fontweight='bold')


    # Figure 1 : Validation Ghia
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=DARK)
    for ax, Re in zip([ax1, ax2], [100, 1000]):
        sol = ghia_solvers[Re]
        y, u_mid, _, _ = sol.centerline_profiles()
        g = GHIA[Re]
        ax.plot(u_mid, y, color=C_RE[Re], lw=2.5, label='LBM-MRT (N={})'.format(GHIA_N))
        ax.scatter(g['u'], g['y'], color='white', s=55, zorder=5, label='Ghia et al. (1982)')
        u_interp = np.interp(g['y'], y, u_mid)
        L2 = np.sqrt(np.mean((u_interp - np.array(g['u'])) ** 2))
        ax.text(0.04, 0.94, f'$L_2$ = {L2:.4f}', transform=ax.transAxes,
                color='yellow', fontsize=9, bbox=dict(boxstyle='round', fc='k', alpha=0.5))
        dark_axes(ax, xl='u / U_lid  (x=0.5)', yl='y', title=f'Re = {Re}  (lid uniforme)', fs=10)
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=9)
    fig.suptitle('Validation Ghia (1982) – lid uniforme', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig1_ghia_validation.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig1_ghia_validation.png")

    # Figure 2 : GCI (Re=500, sin_2pi)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK)
    N_vals = GCI_N_list
    ax1.plot(N_vals, gci_eps, 'o-', color='#00e5ff', lw=2.5, ms=10, label='ε LBM')
    ax1.axhline(gci_exact, color='#ff6b35', ls='--', lw=2, label=f'ε* = {gci_exact:.4e}')
    for Nv, ev in zip(N_vals, gci_eps):
        ax1.text(Nv + 5, ev, f'{ev:.4e}', color='#00e5ff', fontsize=8)
    dark_axes(ax1, xl='N', yl='ε = ∫∫Φ dΩ', title=f'GCI = {gci_GCI:.2f}%, p = {gci_p:.2f}')
    ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w')
    h = 1.0 / np.array(N_vals)
    err = np.abs(np.array(gci_eps) - gci_exact)
    ax2.loglog(h, err, 'o-', color='#7dff6b', lw=2.5, ms=10, label='|ε - ε*|')
    if not np.isnan(gci_p):
        ax2.loglog(h, err[0] * (h / h[0]) ** gci_p, '--', color='#ffaa00', lw=1.8, label=f'slope {gci_p:.2f}')
    dark_axes(ax2, xl='h = 1/N', yl='|ε - ε*|', title='Convergence spatiale')
    ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w')
    fig.suptitle('Grid Convergence Index (GCI) – Re=500, profil A₂·sin(2πx)', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig2_grid_convergence.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig2_grid_convergence.png")

    # Figure 3 : Champs de vitesse (avec grilles corrigées)
    fig = plt.figure(figsize=(20, 13), facecolor=DARK)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)
    # Création des coordonnées physiques
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
            plt.colorbar(cf, ax=ax, pad=0.01).ax.yaxis.set_tick_params(color='w', labelcolor='w', labelsize=6)
            ax.set_title(f'Re={Re}  {PROFILE_LABEL[prof]}', color='w', fontsize=9, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Champs de vitesse |U| – MRT D2Q9 (N=256)', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig3_velocity_fields.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig3_velocity_fields.png")

    # Figure 4 : Dissipation (grilles corrigées)
    fig = plt.figure(figsize=(20, 13), facecolor=DARK)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.30, wspace=0.28)
    for ri, Re in enumerate(RE_LIST):
        for pi, prof in enumerate(PROFILES):
            ax = fig.add_subplot(gs[ri, pi])
            sol = all_results[Re][prof]
            d = sol.dissipation
            vmax = np.percentile(d, 99)
            cf = ax.contourf(X, Y, np.clip(d, 0, vmax), levels=40, cmap='inferno')
            plt.colorbar(cf, ax=ax, pad=0.01).ax.yaxis.set_tick_params(color='w', labelcolor='w', labelsize=6)
            ax.text(0.03, 0.04, f'ε={sol.eps:.2e}', transform=ax.transAxes,
                    color='yellow', fontsize=7.5, bbox=dict(boxstyle='round', fc='k', alpha=0.5))
            ax.set_title(f'Re={Re}  {PROFILE_LABEL[prof]}', color='w', fontsize=9, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle('Dissipation visqueuse Φ – ε = ∫∫Φ dΩ', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig4_dissipation_fields.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig4_dissipation_fields.png")

    # Figure 5 : Comparatif de la dissipation (inchangée)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=DARK)
    xp = np.arange(len(PROFILES))
    for ax, Re in zip(axes, RE_LIST):
        eps_unif = all_results[Re]['uniform'].eps
        eps_vals = [all_results[Re][p].eps for p in PROFILES]
        bars = ax.bar(xp, eps_vals, color=[C_PR[p] for p in PROFILES], alpha=0.85, width=0.6)
        ax.axhline(eps_unif, color='w', ls=':', lw=1.5, alpha=0.6, label='ε uniforme')
        for xi, ev in zip(xp, eps_vals):
            drel = (ev - eps_unif) / eps_unif * 100 if eps_unif > 0 else 0
            ax.text(xi, ev * 1.04, f'{drel:+.1f}%', ha='center', color='yellow', fontsize=8, fontweight='bold')
        ax.set_xticks(xp)
        ax.set_xticklabels([PROFILE_LABEL[p] for p in PROFILES], color='w', fontsize=7.5, rotation=15, ha='right')
        dark_axes(ax, yl='ε = ∫∫Φ dΩ', title=f'Re = {Re}', fs=10)
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=8)
    fig.suptitle('Dissipation totale – Comparaison des profils', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig5_dissipation_comparison.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig5_dissipation_comparison.png")

    # Figure 6 : Analyse modale (inchangée)
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
            dark_axes(ax, xl='Mode n', yl='Énergie (%)',
                      title=f'Re={Re}  {PROFILE_LABEL[prof]}\nMode {dom}: {fracs_pct[dom - 1]:.1f}%', fs=8)
            ax.set_xlim(0.5, 10.5)
    fig.suptitle('Analyse modale – u(x=0.5, y) – Réponse de l’écoulement', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig6_modal_analysis.png'), dpi=130, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig6_modal_analysis.png")

    # Figure 7 : Profils centre-ligne (inchangée)
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
            ax.scatter(g['u'], g['y'], color='w', s=22, zorder=5, label='Ghia')
        dark_axes(ax, xl='u / U_lid  (x=0.5)', yl='y', title=f'Re={Re}  u(y)', fs=10)
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=7.5, loc='lower right')

        ax = fig.add_subplot(gs[ri, 1])
        for prof in PROFILES:
            sol = all_results[Re][prof]
            x, v_mid = sol.centerline_profiles()[2:4]
            ax.plot(x, v_mid, color=C_PR[prof], lw=2, ls=LS[prof], label=PROFILE_LABEL[prof])
        ax.axhline(0, color='#333', lw=0.8)
        dark_axes(ax, xl='x', yl='v / U_lid  (y=0.5)', title=f'Re={Re}  v(x)', fs=10)
        ax.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=7.5)
    fig.suptitle('Profils centre-ligne – u(0.5,y) et v(x,0.5)', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig7_centerline_profiles.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig7_centerline_profiles.png")

    # Figure 8 : A₂ et enstrophie (inchangée)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK)
    Re_vals = RE_LIST
    A2_pinn = {k: A2_PINN_TABLE[k] for k in [100, 500, 1000]}
    A2_lbm = [all_results[Re]['sin_2pi'].modal_analysis()[1][1] for Re in Re_vals]
    ax1.plot(Re_vals, A2_lbm, 's--', color='#00e5ff', lw=2, ms=9, label='A₂ LBM (réponse flow)')
    ax1.plot(Re_vals, [A2_pinn[r] for r in Re_vals], 'o-', color='#ff6b35', lw=2.5, ms=8, label='A₂ PINN')
    ax1.axhline(np.mean(list(A2_pinn.values())), color='#ff6b35', ls=':', lw=1.2, alpha=0.6, label='PINN mean')
    dark_axes(ax1, xl='Re', yl='A₂', title='Mode 2 – Robustesse Reynolds')
    ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w')

    xp = np.arange(len(Re_vals))
    width = 0.18
    for i, prof in enumerate(PROFILES):
        Z_vals = [all_results[Re][prof].Z for Re in Re_vals]
        ax2.bar(xp + i * width, Z_vals, width, color=C_PR[prof], alpha=0.85, label=PROFILE_LABEL[prof])
    ax2.set_xticks(xp + 1.5 * width)
    ax2.set_xticklabels([f'Re={r}' for r in Re_vals], color='w')
    dark_axes(ax2, yl='Z = ∫∫ ω² dΩ', title='Enstrophie – 4 profils')
    ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=8, ncol=2)
    fig.suptitle('Robustesse des modes et enstrophie', color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig8_robustness.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig8_robustness.png")

    # Figure 9 : Réponse des branches contrôlées (champ moyenné dans le temps)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK)
    xp = np.arange(len(RE_LIST))
    width = 0.38
    for pi, prof in enumerate(TEMPORAL_CONTROLS):
        K_fluct_ratio = [temporal_results[Re][prof].K_fluct / max(temporal_results[Re][prof].K, 1e-30) for Re in RE_LIST]
        ax1.bar(xp + (pi - 0.5) * width, K_fluct_ratio, width, color=C_PR[prof], alpha=0.9, label=PROFILE_LABEL[prof])
    ax1.set_xticks(xp)
    ax1.set_xticklabels([f'Re={r}' for r in RE_LIST], color='w')
    dark_axes(ax1, yl='K_fluct / K', title='Poids des fluctuations temporelles (branches)')
    ax1.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=8)

    for pi, prof in enumerate(TEMPORAL_CONTROLS):
        dom_fracs = []
        for Re in RE_LIST:
            _, _, ff = temporal_results[Re][prof].fluct_modal_analysis()
            dom_fracs.append(ff.max())
        ax2.plot(xp, dom_fracs, 'o-', lw=2.5, ms=9, color=C_PR[prof], label=PROFILE_LABEL[prof])
    ax2.set_xticks(xp)
    ax2.set_xticklabels([f'Re={r}' for r in RE_LIST], color='w')
    dark_axes(ax2, yl='Fraction d’énergie du mode dominant', title='Structure modale de u_mid (moyenne)')
    ax2.set_ylim(0, 1.02)
    ax2.legend(facecolor='#1a1a1a', edgecolor='#444', labelcolor='w', fontsize=8)
    fig.suptitle('Validation des branches par le LBM indépendant : réponse moyennée dans le temps',
                 color='w', fontsize=12, fontweight='bold')
    fig.savefig(os.path.join(OUTDIR, 'fig9_temporal_branches.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print("   → fig9_temporal_branches.png")

    # ----------------------------------------------------------------------
    #  Tableaux Récapitulatifs finaux
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RÉSULTATS QUANTITATIFS")
    print("=" * 70)
    print("\n[Tableau A] Dissipation totale ε")
    print(f"{'Re':>5} | {'Profil':<15} | {'ε':>12} | {'Δε/ε_unif (%)':>15} | {'K':>12} | {'Z':>12}")
    print("-" * 85)
    for Re in RE_LIST:
        eps_unif = all_results[Re]['uniform'].eps
        for prof in PROFILES:
            sol = all_results[Re][prof]
            drel = (sol.eps - eps_unif) / eps_unif * 100 if eps_unif > 0 else 0
            print(
                f"{Re:>5} | {PROFILE_LABEL[prof]:<15} | {sol.eps:>12.4e} | {drel:>+14.2f}% | {sol.K:>12.4e} | {sol.Z:>12.4e}")

    print("\n[Tableau B] Mode dominant de u_mid(y)")
    print(f"{'Re':>5} | {'Profil':<15} | {'Mode dom':>8} | {'Énergie (%)':>12} | {'A₂ flow':>8}")
    print("-" * 65)
    for Re in RE_LIST:
        for prof in PROFILES:
            modes, amps, fracs = all_results[Re][prof].modal_analysis()
            dom = np.argmax(fracs) + 1
            dom_frac = fracs[dom - 1] * 100
            A2_flow = amps[1] if len(amps) > 1 else 0.0
            print(f"{Re:>5} | {PROFILE_LABEL[prof]:<15} | {dom:>8} | {dom_frac:>11.2f}% | {A2_flow:>8.4f}")

    print("\n[Validation Ghia]")
    for Re in [100, 1000]:
        sol = ghia_solvers[Re]
        y, u_mid, _, _ = sol.centerline_profiles()
        g = GHIA[Re]
        u_interp = np.interp(g['y'], y, u_mid)
        L2 = np.sqrt(np.mean((u_interp - np.array(g['u'])) ** 2))
        L_inf = np.max(np.abs(u_interp - np.array(g['u'])))
        print(f"  Re={Re} : L2 = {L2:.4f}, L∞ = {L_inf:.4f}")

    print("\n[Convergence GCI]")
    print(f"  Re=500, profil sin_2pi : p = {gci_p:.2f}, GCI = {gci_GCI:.2f}%")
    print(f"  ε extrapolé = {gci_exact:.4e}")

    print("\n" + "=" * 70)
    print(f"  Toutes les figures et logs sont dans : {OUTDIR}")
    print("=" * 70)