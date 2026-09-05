# -*- coding: utf-8 -*-
"""
============================================================
NIVEAU 1 ULTRA — Version corrigée (stabilisation de l'énergie)
============================================================
Corrections majeures :
  1. Pénalité d'énergie symétrique (E - E_target)² au lieu de relu(E_target - E)²
  2. Ajustement de LAMBDA_CTRL à 200 (au lieu de 500)
  3. Augmentation de LAMBDA_DISS à 0.05 (pour limiter l'énergie excessive)
  4. Normalisation correcte de Re dans coeff_net ([-1, 1] au lieu de [0.1, 1.0])
  5. Tous les autres hyperparamètres et fonctionnalités conservés
  6. REMPLACEMENT DE gplearn PAR PySR (meilleure fiabilité et performance)
============================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pysr import PySRRegressor  # PySR remplace gplearn
import os, time, json, warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DTYPE  = torch.float64
torch.set_default_dtype(DTYPE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

# ─── Hyperparamètres (corrigés) ─────────────────────────────────
L       = 1.0
T_MAX   = 1.0
RE_LIST = [100, 500, 1000]
Re_min, Re_max = 100.0, 1000.0

N_EPOCHS_PRE  = 300    # pré-entraînement (coeffs gelés)
N_EPOCHS_MAIN = 3000   # entraînement principal
N_PHYS  = 4000
N_BC    = 600
LR      = 8e-4
N_MX    = 6   # modes spatiaux
N_MT    = 5   # modes temporels

LAMBDA_PDE   = 1.0
LAMBDA_BC    = 15.0
LAMBDA_DISS  = 0.05      # augmenté (était 0.008)
LAMBDA_CTRL  = 200.0     # réduit (était 500) car pénalité symétrique
LAMBDA_VAR   = 1.0       # pénalise profils plats
LAMBDA_RE    = 2.0       # cohérence inter-Re
E_TARGET     = 0.25      # énergie cinétique cible

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(SCRIPT_DIR, "results_deepseek_20260903"))
os.makedirs(OUT, exist_ok=True)

# Normalisation Re (pour entrée du réseau)
def normalize_Re(Re):
    return 2.0 * (Re - Re_min) / (Re_max - Re_min) - 1.0

# ─── Architecture ──────────────────────────────────────────────
class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, dtype=DTYPE))
    def forward(self, x):
        return torch.sin(self.a * x)


class UltraPINN(nn.Module):
    """
    PINN unifié (x,y,t,Re) avec loi de contrôle U_lid(x,t,Re)
    """
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Linear(i, o, dtype=DTYPE), SineAct())

        # Réseau principal NS : (x,y,t,Re_norm) → (u,v,p)
        # Les entrées sont normalisées dans [-1,1]
        self.net = nn.Sequential(
            blk(4, 128), blk(128, 128), blk(128, 96), blk(96, 64),
            nn.Linear(64, 3, dtype=DTYPE)
        )

        n = N_MX * N_MT
        # Réseau de contrôle : Re_norm (normalisé) → coefficients modaux
        self.coeff_net = nn.Sequential(
            blk(1, 64), blk(64, 64), blk(64, 48),
            nn.Linear(48, n, dtype=DTYPE)
        )

        # Initialisation des biais (modes dominants)
        with torch.no_grad():
            bias = self.coeff_net[-1].bias
            bias.zero_()
            bias[0]      = 0.50   # sin(πx)·cos(0·πt)
            bias[1]      = 0.18   # sin(πx)·cos(πt)
            bias[N_MT]   = 0.20   # sin(2πx)·cos(0)
            bias[N_MT+1] = 0.08   # sin(2πx)·cos(πt)

    def U_lid(self, x, t, Re_phys):
        """
        x, t : tenseurs physiques (non normalisés)
        Re_phys : tenseur physique (non normalisé)
        Retourne U_lid (physique, non normalisé)
        """
        xn = x / L
        tn = t / T_MAX
        # Normalisation de Re pour le sous‑réseau
        Re_n = normalize_Re(Re_phys).view(-1, 1)
        # Construction de la base de Fourier
        basis = []
        for i in range(N_MX):
            for j in range(N_MT):
                f = torch.sin((i+1) * np.pi * xn) * torch.cos(j * np.pi * tn)
                basis.append(f)
        basis = torch.cat(basis, dim=1)
        coeffs = self.coeff_net(Re_n)
        U_phys = (coeffs * basis).sum(dim=1, keepdim=True)
        return U_phys

    def forward(self, x_n, y_n, t_n, Re_n):
        """
        Entrées normalisées dans [-1,1] (x_n, y_n, t_n, Re_n)
        Sorties normalisées (u_n, v_n, p_n)
        """
        inp = torch.cat([x_n, y_n, t_n, Re_n], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def net_params(self):
        return list(self.net.parameters())

    def ctrl_params(self):
        return list(self.coeff_net.parameters())


# ─── Dérivées ──────────────────────────────────────────────────
def D(u, v, cg=True):
    return torch_grad(u, v, grad_outputs=torch.ones_like(u),
                      create_graph=cg, retain_graph=True)[0]

def lap(u, x, y):
    return D(D(u, x), x) + D(D(u, y), y)


# ─── Génération des points (avec Re physique) ──────────────────
def gen_points(Re_phys_val):
    xp = torch.rand(N_PHYS, 1, dtype=DTYPE) * L
    yp = torch.rand(N_PHYS, 1, dtype=DTYPE) * L
    tp = torch.rand(N_PHYS, 1, dtype=DTYPE) * T_MAX
    rp = torch.full((N_PHYS, 1), Re_phys_val, dtype=DTYPE)

    def wall(n, xv=None, yv=None):
        if xv is not None:
            X = torch.full((n, 1), xv, dtype=DTYPE)
            Y = torch.rand(n, 1, dtype=DTYPE) * L
        else:
            X = torch.rand(n, 1, dtype=DTYPE) * L
            Y = torch.full((n, 1), yv, dtype=DTYPE)
        T = torch.rand(n, 1, dtype=DTYPE) * T_MAX
        R = torch.full((n, 1), Re_phys_val, dtype=DTYPE)
        return X, Y, T, R

    bc = {
        'bottom': wall(N_BC, yv=0.),
        'top'   : wall(N_BC, yv=L),
        'left'  : wall(N_BC, xv=0.),
        'right' : wall(N_BC, xv=L),
    }
    for k in bc:
        bc[k] = tuple(v.to(device) for v in bc[k])

    xp = xp.to(device).requires_grad_(True)
    yp = yp.to(device).requires_grad_(True)
    tp = tp.to(device).requires_grad_(True)
    rp = rp.to(device).requires_grad_(False)
    return (xp, yp, tp, rp), bc


# ─── Fonction de coût (corrigée) ───────────────────────────────
def compute_loss(model, pts, bc, nu, Re_val, ctrl_active=True):
    xp, yp, tp, rp = pts
    # Normalisation des entrées pour le réseau principal
    x_n = 2.0 * (xp - 0.0) / L - 1.0
    y_n = 2.0 * (yp - 0.0) / L - 1.0
    t_n = 2.0 * (tp - 0.0) / T_MAX - 1.0
    r_n = normalize_Re(rp)

    u_n, v_n, p_n = model(x_n, y_n, t_n, r_n)
    u = u_n * 1.0
    v = v_n * 1.0
    p = p_n * 1.0

    # Résidus Navier-Stokes
    res_u = D(u, tp) + u*D(u, xp) + v*D(u, yp) + D(p, xp) - nu*lap(u, xp, yp)
    res_v = D(v, tp) + u*D(v, xp) + v*D(v, yp) + D(p, yp) - nu*lap(v, xp, yp)
    res_c = D(u, xp) + D(v, yp)
    l_pde = (res_u**2 + res_v**2 + res_c**2).mean()

    # Conditions aux limites
    l_bc = torch.tensor(0., dtype=DTYPE, device=device)
    for w in ['bottom', 'left', 'right']:
        xb, yb, tb, rb = bc[w]
        xb_n = 2.0 * (xb - 0.0) / L - 1.0
        yb_n = 2.0 * (yb - 0.0) / L - 1.0
        tb_n = 2.0 * (tb - 0.0) / T_MAX - 1.0
        rb_n = normalize_Re(rb)
        ub_n, vb_n, _ = model(xb_n, yb_n, tb_n, rb_n)
        ub = ub_n * 1.0
        vb = vb_n * 1.0
        l_bc += ub.pow(2).mean() + vb.pow(2).mean()
    # Paroi supérieure
    xb, yb, tb, rb = bc['top']
    xb_n = 2.0 * (xb - 0.0) / L - 1.0
    yb_n = 2.0 * (yb - 0.0) / L - 1.0
    tb_n = 2.0 * (tb - 0.0) / T_MAX - 1.0
    rb_n = normalize_Re(rb)
    ub_n, vb_n, _ = model(xb_n, yb_n, tb_n, rb_n)
    ub = ub_n * 1.0
    vb = vb_n * 1.0
    u_lid = model.U_lid(xb, tb, rb)
    l_bc += (ub - u_lid).pow(2).mean() + vb.pow(2).mean()

    # Dissipation
    ux = D(u, xp); uy = D(u, yp)
    vx = D(v, xp); vy = D(v, yp)
    l_diss = nu * (ux**2 + vy**2 + 2*(0.5*(uy+vx))**2).mean()

    if not ctrl_active:
        total = LAMBDA_PDE * l_pde + LAMBDA_BC * l_bc
        return total, l_pde.item(), l_bc.item(), 0., 0., 0., 0.

    # --- Termes de contrôle (anti-triviaux) ---
    energy = (u_lid**2).mean()
    l_ctrl = (energy - E_TARGET)**2

    u_mean = u_lid.mean()
    u_var  = ((u_lid - u_mean)**2).mean()
    l_var  = -torch.log(u_var + 1e-5)

    Re_low  = torch.tensor([[Re_min]], dtype=DTYPE, device=device)
    Re_high = torch.tensor([[Re_max]], dtype=DTYPE, device=device)
    x_mid = torch.linspace(0, 1, 20, dtype=DTYPE, device=device).view(-1,1)
    t_mid = torch.full_like(x_mid, 0.5)
    U_low  = model.U_lid(x_mid, t_mid, Re_low.expand(20,1))
    U_high = model.U_lid(x_mid, t_mid, Re_high.expand(20,1))
    diff_Re = ((U_low - U_high)**2).mean()
    l_re = torch.relu(0.01 - diff_Re)

    total = (LAMBDA_PDE  * l_pde  +
             LAMBDA_BC   * l_bc   +
             LAMBDA_DISS * l_diss +
             LAMBDA_CTRL * l_ctrl +
             LAMBDA_VAR  * l_var  +
             LAMBDA_RE   * l_re)

    return (total, l_pde.item(), l_bc.item(), l_diss.item(),
            l_ctrl.item(), energy.item(), u_var.item())


# ─── Entraînement ──────────────────────────────────────────────
def train_ultra(verbose=True):
    print(f"\n{'='*65}")
    print(f"ULTRA PINN (corrigé) — Entraînement unifié Re={RE_LIST}")
    print(f"Pré-entraînement : {N_EPOCHS_PRE} ép.  Principal : {N_EPOCHS_MAIN} ép.")
    print(f"λ_ctrl={LAMBDA_CTRL} (symétrique)  λ_var={LAMBDA_VAR}  λ_re={LAMBDA_RE}  E_target={E_TARGET}")
    print(f"{'='*65}")

    model = UltraPINN().to(device)
    t0    = time.time()

    datasets = {}
    for Re in RE_LIST:
        pts, bc = gen_points(Re)
        datasets[Re] = (pts, bc, 1.0/Re, Re)

    # ── Phase 1 : pré-entraînement (coeff_net gelé) ───────────
    print(f"\n[PHASE 1] Pré-entraînement {N_EPOCHS_PRE} ép. — coeff_net gelé")
    opt_pre = optim.Adam(model.net_params(), lr=LR)
    for ep in range(1, N_EPOCHS_PRE + 1):
        opt_pre.zero_grad()
        total = torch.tensor(0., dtype=DTYPE, device=device)
        for Re, (pts, bc, nu, re_val) in datasets.items():
            loss, *_ = compute_loss(model, pts, bc, nu, re_val, ctrl_active=False)
            total += loss
        if torch.isnan(total):
            print(f"  ⚠ NaN @ {ep}"); break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.net_params(), 1.0)
        opt_pre.step()
        if verbose and ep % 100 == 0:
            print(f"  Ep {ep:4d}/{N_EPOCHS_PRE} | Loss={total.item():.3e}")
    print(f"  ✓ Pré-entraînement {time.time()-t0:.0f}s")

    # ── Phase 2 : entraînement complet ───────────────────────
    print(f"\n[PHASE 2] Entraînement principal {N_EPOCHS_MAIN} ép. — tous paramètres")
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS_MAIN, eta_min=5e-6)

    hist = {k: [] for k in ['total', 'pde', 'bc', 'diss', 'ctrl', 'energy', 'var']}
    t1   = time.time()

    for ep in range(1, N_EPOCHS_MAIN + 1):
        opt.zero_grad()
        total = torch.tensor(0., dtype=DTYPE, device=device)
        sums  = [0.] * 7

        for Re, (pts, bc, nu, re_val) in datasets.items():
            out = compute_loss(model, pts, bc, nu, re_val, ctrl_active=True)
            loss = out[0]
            total += loss
            for i, v in enumerate(out[1:]):
                sums[i] += v

        if torch.isnan(total):
            print(f"  ⚠ NaN @ époque {ep}"); break
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

        hist['total'].append(total.item())
        hist['pde'].append(sums[0]);    hist['bc'].append(sums[1])
        hist['diss'].append(sums[2]);   hist['ctrl'].append(sums[3])
        hist['energy'].append(sums[4]); hist['var'].append(sums[5])

        if verbose and (ep % 300 == 0 or ep == 1):
            elapsed = time.time() - t1
            eta     = elapsed / ep * (N_EPOCHS_MAIN - ep)
            print(f"  Ep {ep:4d}/{N_EPOCHS_MAIN} | Loss={total.item():.3e} | "
                  f"PDE={sums[0]:.3e} | E={sums[4]/3:.4f} | "
                  f"Var={sums[5]/3:.4f} | ETA={eta/60:.1f}min")

    print(f"\n  ✓ Entraînement terminé {(time.time()-t0)/60:.1f} min | "
          f"Loss={hist['total'][-1]:.4e} | Énergie moy={hist['energy'][-1]/3:.4f}")
    return model, hist


# ─── Extraction U_lid ──────────────────────────────────────────
def extract_lid(model, nx=80, nt=80):
    rows = []
    for Re in RE_LIST:
        xs = torch.linspace(0, L, nx, dtype=DTYPE, device=device).view(-1, 1)
        ts = torch.linspace(0, T_MAX, nt, dtype=DTYPE, device=device).view(-1, 1)
        X, T = torch.meshgrid(xs[:,0], ts[:,0], indexing='ij')
        Xf, Tf = X.reshape(-1,1), T.reshape(-1,1)
        Re_tensor = torch.full_like(Xf, Re)
        with torch.no_grad():
            U = model.U_lid(Xf, Tf, Re_tensor).cpu().numpy()
        rows.append(np.column_stack([
            Xf.cpu().numpy(), Tf.cpu().numpy(),
            np.full((len(U),1), Re/1000.), U
        ]))
    return np.vstack(rows), {Re: rows[i] for i, Re in enumerate(RE_LIST)}


# ─── Régression symbolique avec PySR (remplace gplearn) ───────
def run_pysr(data, label="", niter=500):
    X = data[:, :3].astype(np.float32)  # x, t, Re_norm
    y = data[:, 3].astype(np.float32)

    # Normalisation des entrées (z-score) pour PySR
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    print(f"\n  [PySR] {label} — {len(X)} points, vars=(x,t,Re_norm)")
    model_sr = PySRRegressor(
        niterations=niter,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "square", "sqrt"],
        loss="L2DistLoss()",
        populations=50,
        maxsize=15,
        constraints={"/": (2, 1), "sin": 1, "cos": 1, "exp": 1, "log": 1, "sqrt": 1},
        parsimony=0.01,
        random_state=42,
        deterministic=True,
        parallelism="serial",
        verbosity=1
    )

    model_sr.fit(X_norm, y_norm)
    eq_norm = model_sr.sympy()

    # Rétablir l'équation en variables physiques
    import sympy as sp
    x_sym, t_sym, Re_sym = sp.symbols('x t Re')
    from sympy import symbols as syms
    x0, x1, x2 = syms('x0 x1 x2')
    expr = sp.sympify(str(eq_norm))
    expr = expr.subs(x0, (x_sym - float(X_mean[0])) / float(X_std[0]))
    expr = expr.subs(x1, (t_sym - float(X_mean[1])) / float(X_std[1]))
    expr = expr.subs(x2, (Re_sym - float(X_mean[2])) / float(X_std[2]))
    expr_final = sp.simplify(float(y_mean) + float(y_std) * expr)

    # Calcul des métriques sur les données originales
    f = sp.lambdify((x_sym, t_sym, Re_sym), expr_final, modules='numpy')
    y_pred = f(data[:, 0], data[:, 1], data[:, 2])
    mse = np.mean((data[:, 3] - y_pred)**2)
    r2 = 1 - mse / np.var(data[:, 3])
    print(f"  MSE (physique) = {mse:.4e}, R² = {r2:.4f}")
    print(f"  Équation : {expr_final}")
    return model_sr, str(expr_final), mse, r2


# ─── Validation LOCO ───────────────────────────────────────────
def loco_ultra(all_data_by_re):
    print(f"\n{'='*65}\nVALIDATION LOCO — ULTRA\n{'='*65}")
    results = {}
    for test_Re in RE_LIST:
        train_data = np.vstack([
            all_data_by_re[Re] for Re in RE_LIST if Re != test_Re
        ])
        _, expr, _, _ = run_pysr(train_data, f"train excl Re={test_Re}", niter=200)

        td = all_data_by_re[test_Re]
        # Évaluer l'équation sur les données de test
        import sympy as sp
        f = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('Re')), expr, modules='numpy')
        y_pred = f(td[:, 0], td[:, 1], td[:, 2])
        y_true = td[:, 3]
        mse = float(((y_true - y_pred)**2).mean())
        r2 = float(1 - mse / y_true.var())
        print(f"  LOCO Re={test_Re} → MSE={mse:.4e}  R²={r2:.4f}")
        results[test_Re] = {
            'expr': expr, 'mse_test': mse, 'r2_test': r2,
            'data': td
        }
    return results


# ─── Figures publication (inchangées) ─────────────────────────
def make_all_figures(model, hist, loco_results, all_data_by_re):
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 12,
        'legend.fontsize': 8,  'figure.dpi': 150,
        'lines.linewidth': 1.8
    })
    colors = {'100': '#1D9E75', '500': '#185FA5', '1000': '#D85A30'}

    # Fig 1 : Convergence
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    ax.semilogy(hist['total'], color='#2C2C2A', lw=2, label='Total')
    ax.semilogy(hist['pde'],   color='#185FA5', ls='--', label='NS PDE')
    ax.semilogy(hist['bc'],    color='#D85A30', ls=':', label='BC')
    ax.set_xlabel('Époque'); ax.set_ylabel('Loss (log)')
    ax.set_title('Convergence globale')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(hist['ctrl'], color='#D85A30', label='Ctrl énergie')
    ax.semilogy(hist['diss'], color='#888780', ls='--', label='Dissipation')
    ax2 = ax.twinx()
    e_arr = np.array(hist['energy']) / 3
    ax2.plot(e_arr, color='#BA7517', lw=1.3, alpha=0.85, label='Énergie moy.')
    ax2.axhline(E_TARGET, color='#BA7517', ls='--', lw=0.8, alpha=0.5)
    ax2.set_ylabel('Énergie', color='#BA7517')
    ax.set_xlabel('Époque'); ax.set_ylabel('Loss (log)')
    ax.set_title('Termes de contrôle')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    v_arr = np.array(hist['var']) / 3
    ax.plot(v_arr, color='#1D9E75', label='Variance U_lid')
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xlabel('Époque'); ax.set_ylabel('Variance spatiale')
    ax.set_title('Variance de U_lid (anti-trivialité)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle('Ultra PINN — Convergence complète', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_convergence.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig1_convergence")

    # Fig 2 : Profils U_lid(x,t)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x_grid = np.linspace(0, L, 300)
    Xp = torch.tensor(x_grid[:, None], dtype=DTYPE, device=device)
    for ax, Re in zip(axes, RE_LIST):
        for t_val, ls, alpha in zip(
            [0., 0.2, 0.4, 0.6, 0.8, 1.0],
            ['-', '--', '-.', ':', '-', '--'],
            [1.0, 0.85, 0.7, 0.7, 0.85, 1.0]
        ):
            Tp = torch.full_like(Xp, t_val)
            Re_t = torch.full_like(Xp, Re)
            with torch.no_grad():
                Up = model.U_lid(Xp, Tp, Re_t).cpu().numpy().ravel()
            ax.plot(x_grid, Up, ls=ls, alpha=alpha,
                    label=f't={t_val:.1f}', color=colors[str(Re)])
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_xlabel('x'); ax.set_ylabel('U_lid(x,t)')
        ax.set_title(f'Re = {Re}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Ultra PINN — Profils spatio-temporels U_lid(x,t)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_lid_profiles.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig2_lid_profiles")

    # Fig 3 : Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    xs = np.linspace(0, L, 100); ts = np.linspace(0, T_MAX, 100)
    Xg, Tg = np.meshgrid(xs, ts)
    Xf = torch.tensor(Xg.ravel()[:, None], dtype=DTYPE, device=device)
    Tf = torch.tensor(Tg.ravel()[:, None], dtype=DTYPE, device=device)
    vmax_global = 0.
    for Re in RE_LIST:
        Re_t = torch.full_like(Xf, Re)
        with torch.no_grad():
            Uf = model.U_lid(Xf, Tf, Re_t).cpu().numpy().reshape(100, 100)
        vmax_global = max(vmax_global, np.abs(Uf).max())
    for ax, Re in zip(axes, RE_LIST):
        Re_t = torch.full_like(Xf, Re)
        with torch.no_grad():
            Uf = model.U_lid(Xf, Tf, Re_t).cpu().numpy().reshape(100, 100)
        im = ax.contourf(xs, ts, Uf, levels=40, cmap='RdBu_r',
                          vmin=-vmax_global, vmax=vmax_global)
        plt.colorbar(im, ax=ax, label='U_lid')
        ax.set_xlabel('x'); ax.set_ylabel('t')
        ax.set_title(f'Re = {Re}')
    fig.suptitle('Ultra PINN — Heatmap U_lid(x,t) par Re', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_heatmap.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig3_heatmap")

    # Fig 4 : Loi en Re
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    Re_scan = np.array([50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000])
    x_mids = [0.25, 0.5, 0.75]
    ax = axes[0]
    for x_mid in x_mids:
        U_scan = []
        for Re_v in Re_scan:
            Xp_ = torch.tensor([[x_mid]], dtype=DTYPE, device=device)
            Tp_ = torch.tensor([[0.5]], dtype=DTYPE, device=device)
            Re_t = torch.tensor([[Re_v]], dtype=DTYPE, device=device)
            with torch.no_grad():
                U_scan.append(model.U_lid(Xp_, Tp_, Re_t).item())
        ax.plot(Re_scan, U_scan, 'o-', lw=1.8, ms=5, label=f'x={x_mid:.2f}')
    ax.set_xlabel('Re'); ax.set_ylabel('U_lid(x, t=0.5)')
    ax.set_title('Loi optimale vs Re (linéaire)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    U_mid = []
    for Re_v in Re_scan:
        Xp_ = torch.tensor([[0.5]], dtype=DTYPE, device=device)
        Tp_ = torch.tensor([[0.5]], dtype=DTYPE, device=device)
        Re_t = torch.tensor([[Re_v]], dtype=DTYPE, device=device)
        with torch.no_grad():
            U_mid.append(model.U_lid(Xp_, Tp_, Re_t).item())
    U_mid = np.array(U_mid)
    mask = U_mid > 0
    if mask.sum() > 2:
        ax.loglog(Re_scan[mask], U_mid[mask], 'o-', color='#185FA5',
                  lw=2, ms=6, label='U_lid (x=0.5, t=0.5)')
        log_re = np.log(Re_scan[mask].astype(float))
        log_u  = np.log(U_mid[mask])
        alpha_pow = np.polyfit(log_re, log_u, 1)[0]
        C = np.exp(np.polyfit(log_re, log_u, 1)[1])
        Re_fit = np.linspace(50, 2000, 200)
        ax.loglog(Re_fit, C * Re_fit**alpha_pow, '--', color='#D85A30',
                  lw=1.8, label=f'U ~ Re^{{{alpha_pow:.2f}}}')
        ax.legend()
    ax.set_xlabel('Re (log)'); ax.set_ylabel('U_lid (log)')
    ax.set_title('Loi de puissance U_lid ~ Re^α')
    ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Ultra PINN — Dépendance en Re', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_Re_law.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig4_Re_law")

    # Fig 5 : Coefficients modaux
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for Re in RE_LIST:
        Re_t = torch.tensor([[Re]], dtype=DTYPE, device=device)
        with torch.no_grad():
            coeffs = model.coeff_net(normalize_Re(Re_t).view(-1,1)).squeeze().cpu().numpy()
        ax = axes[0]
        ax.plot(range(len(coeffs)), coeffs, 'o-', lw=1.5, ms=4,
                label=f'Re={Re}', color=colors[str(Re)])
    axes[0].axhline(0, color='gray', lw=0.7, ls=':')
    axes[0].set_xlabel('Indice modal (i·N_MT + j)')
    axes[0].set_ylabel('Coefficient'); axes[0].set_title('Coefficients modaux par Re')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ax = axes[1]
    Re_scan_fine = np.linspace(50, 1500, 80)
    coeff_matrix = []
    for Re_v in Re_scan_fine:
        Re_t = torch.tensor([[Re_v]], dtype=DTYPE, device=device)
        with torch.no_grad():
            c = model.coeff_net(normalize_Re(Re_t).view(-1,1)).squeeze().cpu().numpy()
        coeff_matrix.append(c)
    coeff_matrix = np.array(coeff_matrix)
    im = ax.imshow(coeff_matrix.T, aspect='auto', cmap='RdBu_r',
                    origin='lower',
                    extent=[Re_scan_fine[0], Re_scan_fine[-1], 0, N_MX*N_MT])
    plt.colorbar(im, ax=ax, label='Coefficient')
    ax.set_xlabel('Re'); ax.set_ylabel('Indice modal')
    ax.set_title('Carte des coefficients vs Re')
    fig.suptitle('Ultra PINN — Coefficients modaux U_lid', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_coefficients.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig5_coefficients")

    # Fig 6 : LOCO scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, test_Re in zip(axes, RE_LIST):
        res = loco_results[test_Re]
        td  = res['data']
        # Évaluer l'équation sur les données
        import sympy as sp
        f = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('Re')), res['expr'], modules='numpy')
        y_pred = f(td[:, 0], td[:, 1], td[:, 2])
        y_true = td[:, 3]
        ax.scatter(y_true, y_pred, s=3, alpha=0.4, color=colors[str(test_Re)])
        m = min(y_true.min(), y_pred.min()); M = max(y_true.max(), y_pred.max())
        ax.plot([m, M], [m, M], 'r--', lw=1)
        ax.set_xlabel('U_lid PINN'); ax.set_ylabel('U_lid SR')
        ax.set_title(f'LOCO Re={test_Re}  R²={res["r2_test"]:.4f}')
        expr_s = res['expr'][:42] + '...' if len(res['expr']) > 42 else res['expr']
        ax.annotate(f'SR: {expr_s}', xy=(0.02, 0.03),
                    xycoords='axes fraction', fontsize=6.5, color='#333')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Ultra PINN — Validation LOCO (généralisation inter-Re)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_loco.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig6_loco")

    # Fig 7 : Champs de vitesse
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    n = 30
    xs = np.linspace(0, L, n); ys = np.linspace(0, L, n)
    Xg, Yg = np.meshgrid(xs, ys)
    Xf = torch.tensor(Xg.ravel()[:, None], dtype=DTYPE, device=device)
    Yf = torch.tensor(Yg.ravel()[:, None], dtype=DTYPE, device=device)
    Tf = torch.full_like(Xf, 0.5)
    for ax, Re in zip(axes, RE_LIST):
        Re_t = torch.full_like(Xf, Re)
        Xn = 2.0 * (Xf - 0.0) / L - 1.0
        Yn = 2.0 * (Yf - 0.0) / L - 1.0
        Tn = 2.0 * (Tf - 0.0) / T_MAX - 1.0
        Rn = normalize_Re(Re_t)
        with torch.no_grad():
            u_n, v_n, _ = model(Xn, Yn, Tn, Rn)
        u = u_n.cpu().numpy().reshape(n, n)
        v = v_n.cpu().numpy().reshape(n, n)
        spd = np.sqrt(u**2 + v**2)
        strm = ax.streamplot(xs, ys, u, v, color=spd,
                              cmap='viridis', density=1.4, linewidth=0.8)
        plt.colorbar(strm.lines, ax=ax, label='|u|')
        ax.set_title(f'Re={Re}  (t=0.5)')
        ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.suptitle('Ultra PINN — Champs de vitesse sous contrôle optimal',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_velocity.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig7_velocity")

    # Fig 8 : Tableau des expressions SR
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    rows = [['Re test', 'MSE LOCO', 'R² LOCO', 'Expression SR découverte']]
    for Re in RE_LIST:
        res = loco_results[Re]
        expr = res['expr']
        rows.append([
            str(Re),
            f"{res['mse_test']:.3e}",
            f"{res['r2_test']:.4f}",
            expr[:80] + ('...' if len(expr) > 80 else '')
        ])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc='left', loc='center',
                    bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#ccc')
        if r == 0:
            cell.set_facecolor('#e8f0fe')
            cell.set_text_props(fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f8f8f8')
        if c == 3:
            cell.set_width(0.5)
    ax.set_title('Ultra PINN — Expressions SR découvertes (LOCO)', fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig8_sr_table.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig8_sr_table")

    # Fig 9 : Synthèse
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35,
                             left=0.05, right=0.97, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(hist['total'], color='#2C2C2A', lw=1.5, label='Total')
    ax.semilogy(hist['pde'],   color='#185FA5', ls='--', lw=1, label='PDE')
    ax.set_xlabel('Époque'); ax.set_ylabel('Loss'); ax.set_title('Convergence')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.array(hist['energy'])/3, color='#BA7517', lw=1.5)
    ax.axhline(E_TARGET, color='#BA7517', ls='--', lw=1, alpha=0.6, label=f'E_target={E_TARGET}')
    ax.set_xlabel('Époque'); ax.set_ylabel('Énergie moy.'); ax.set_title('Convergence énergie')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

    x_grid_plot = np.linspace(0, L, 300)
    Xp_plot = torch.tensor(x_grid_plot[:, None], dtype=DTYPE, device=device)
    for col, Re in zip([2, 3], [100, 1000]):
        ax = fig.add_subplot(gs[0, col])
        for t_val, ls in zip([0., 0.25, 0.5, 0.75, 1.], ['-','--','-.',':','-']):
            Tp_plot = torch.full_like(Xp_plot, t_val)
            Re_t_plot = torch.full_like(Xp_plot, Re)
            with torch.no_grad():
                Up_plot = model.U_lid(Xp_plot, Tp_plot, Re_t_plot).cpu().numpy().ravel()
            ax.plot(x_grid_plot, Up_plot, ls=ls, alpha=0.8, label=f't={t_val:.2f}',
                    color=colors[str(Re)])
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_xlabel('x'); ax.set_ylabel('U_lid'); ax.set_title(f'Profils Re={Re}')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.25)

    for col, test_Re in enumerate(RE_LIST):
        ax = fig.add_subplot(gs[1, col])
        res = loco_results[test_Re]
        td  = res['data']
        import sympy as sp
        f = sp.lambdify((sp.Symbol('x'), sp.Symbol('t'), sp.Symbol('Re')), res['expr'], modules='numpy')
        yp_ = f(td[:, 0], td[:, 1], td[:, 2])
        yt_ = td[:, 3]
        ax.scatter(yt_, yp_, s=2, alpha=0.35, color=colors[str(test_Re)])
        m_ = min(yt_.min(), yp_.min()); M_ = max(yt_.max(), yp_.max())
        ax.plot([m_, M_], [m_, M_], 'r--', lw=0.8)
        ax.set_title(f'LOCO Re={test_Re}  R²={res["r2_test"]:.3f}')
        ax.set_xlabel('PINN'); ax.set_ylabel('SR')
        ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[1, 3])
    if mask.sum() > 2:
        ax.loglog(Re_scan[mask], U_mid[mask], 'o-', color='#185FA5', lw=1.8, ms=5)
        ax.loglog(Re_fit, C * Re_fit**alpha_pow, '--', color='#D85A30',
                  lw=1.5, label=f'Re^{{{alpha_pow:.2f}}}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Re'); ax.set_ylabel('U_lid(0.5,0.5)')
    ax.set_title('Loi de puissance en Re'); ax.grid(True, which='both', alpha=0.25)

    fig.suptitle(
        'Ultra PINN — Contrôle actif de cavité entraînée\n'
        'Loi de contrôle universelle U_lid(x,t,Re) + Régression Symbolique',
        fontsize=13, fontweight='bold'
    )
    plt.savefig(f"{OUT}/fig9_synthesis.png", dpi=180, bbox_inches='tight')
    plt.close(); print("  → fig9_synthesis (figure de publication)")


# ─── Programme principal ───────────────────────────────────────
def main():
    print("=" * 65)
    print("ULTRA PINN — Version corrigée (énergie stabilisée)")
    print(f"Device={device}  N_PHYS={N_PHYS}  N_BC={N_BC}")
    print(f"Modes : {N_MX}×{N_MT}={N_MX*N_MT}  Époques: PRE={N_EPOCHS_PRE}+MAIN={N_EPOCHS_MAIN}")
    print(f"λ_ctrl={LAMBDA_CTRL} (symétrique)  λ_var={LAMBDA_VAR}  λ_re={LAMBDA_RE}  E*={E_TARGET}")
    print("=" * 65)

    t_global = time.time()

    # Entraînement
    model, hist = train_ultra()

    # Extraction
    print(f"\n{'='*65}\nEXTRACTION U_lid\n{'='*65}")
    all_data_flat, all_data_by_re = extract_lid(model, nx=80, nt=80)
    np.savetxt(f"{OUT}/lid_all.txt", all_data_flat, header="x t Re_norm U_lid")
    for Re in RE_LIST:
        np.savetxt(f"{OUT}/lid_Re{Re}.txt", all_data_by_re[Re],
                   header="x t Re_norm U_lid")
    print(f"  {len(all_data_flat)} points extraits")

    # LOCO SR
    loco_results = loco_ultra(all_data_by_re)

    # Figures
    print(f"\n{'='*65}\nGÉNÉRATION DES FIGURES\n{'='*65}")
    make_all_figures(model, hist, loco_results, all_data_by_re)

    # Sauvegarder le modèle
    torch.save(model.state_dict(), f"{OUT}/model.pt")
    print(f"  → model.pt")

    # Métriques JSON
    metrics = {
        str(Re): {
            'loco_mse':   loco_results[Re]['mse_test'],
            'loco_r2':    loco_results[Re]['r2_test'],
            'loco_expr':  loco_results[Re]['expr'],
            'energy_final': hist['energy'][-1] / 3,
        } for Re in RE_LIST
    }
    metrics['_meta'] = {
        'N_EPOCHS_PRE': N_EPOCHS_PRE, 'N_EPOCHS_MAIN': N_EPOCHS_MAIN,
        'N_PHYS': N_PHYS, 'N_BC': N_BC,
        'LAMBDA_CTRL': LAMBDA_CTRL, 'LAMBDA_VAR': LAMBDA_VAR,
        'E_TARGET': E_TARGET,
        'total_time_min': (time.time() - t_global) / 60
    }
    with open(f"{OUT}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Résumé console
    print("\n" + "=" * 65)
    print("RÉSULTATS FINAUX")
    print(f"{'Re':>6} | {'Énergie':>8} | {'LOCO R²':>8} | Expression SR")
    print("-" * 65)
    for Re in RE_LIST:
        res = loco_results[Re]
        expr_s = res['expr'][:45] + '...' if len(res['expr']) > 45 else res['expr']
        print(f"{Re:>6} | {hist['energy'][-1]/3:>8.4f} | "
              f"{res['r2_test']:>8.4f} | {expr_s}")

    print(f"\n  Règle : R² LOCO > 0.90 → généralisation fiable")
    print(f"  Durée totale : {(time.time()-t_global)/60:.1f} min")
    print(f"  Fichiers dans : {OUT}/")
    print("=" * 65)

    return model, hist, loco_results


if __name__ == "__main__":
    main()