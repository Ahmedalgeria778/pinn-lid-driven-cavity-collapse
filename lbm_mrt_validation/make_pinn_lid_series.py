# -*- coding: utf-8 -*-
"""
make_pinn_lid_series.py — Pré-génération des séries U_lid(x,t,Re) issues du PINN
final (campagne 13 parametrization) pour injection dans le solveur LBM-MRT.

Principe (séparation stricte PINN / LBM) :
  1. Ce script tourne SOUS Python 3.14 + torch uniquement (env PINN).
  2. Il charge les modèles identifiés (fourier & chebyshev_mod, seed 42,
     Re ∈ {100, 500, 1000}) et échantillonne le contrôle sur la grille LBM :
       x ∈ [0, 1]  (N points)
       t ∈ [0, 2]  (nT points = UNE période complète des modes temporels
                    cos(j*pi*t) ; U(x, t+2) = U(x, t) exactement)
  3. Il écrit NPZ contenant la série (nT, N) + la moyenne temporelle U_mean
     (le mode stationnaire c_{i,0}, muettes j>=1 s'annulent sur [0,2]) + stats.
  4. Le LBM (Python 3.11 + numba) lit ces NPZ : profil 'pinn'/'pinn_t'/'cheb_t'.

Usage :  python make_pinn_lid_series.py
Sortie : lbm_mrt_validation/lid_series/{basis}_Re{Re}.npz
"""

import os
import sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import PINN_Lid_driven_reviewers as P

N_LBM = 256          # grille spatiale LBM (identique aux runs de production)
NT_PER = 256         # pas par période forcée dans le LBM
T_PERIOD = 2.0       # période naturelle des modes temporels cos(j*pi*t)
BASIS_SET = ["fourier", "chebyshev_mod"]
RE_SET = [100, 500, 1000]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lid_series")
os.makedirs(OUT, exist_ok=True)


def load_model(basis_type, Re):
    sub = os.path.join(P.OUT, "13_parametrization", basis_type, f"re_{Re}")
    model_path = os.path.join(sub, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = P.UltraPINN(arch_config=None, n_mx=P.N_MX, n_mt=P.N_MT,
                        lx=P.LX, ly=P.LY, basis_type=basis_type).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def make_series(model, Re, nx=N_LBM, nt=NT_PER, t_period=T_PERIOD):
    xs = torch.linspace(0, P.LX, nx, dtype=P.DTYPE).view(-1, 1)
    ts = torch.linspace(0, t_period, nt, dtype=P.DTYPE).view(-1, 1)
    X, T = torch.meshgrid(xs[:, 0], ts[:, 0], indexing="ij")
    Xf, Tf = X.reshape(-1, 1), T.reshape(-1, 1)
    Re_t = torch.full_like(Xf, float(Re))
    with torch.no_grad():
        U = model.U_lid(Xf, Tf, Re_t).cpu().numpy().reshape(nx, nt)   # colonnes = t
    U_series = U.T                                            # (nt, nx) ; lignes = t
    U_mean = U_series.mean(axis=0)                            # c_{i,0} * phi(x)
    dx = P.LX / (nx - 1); dt = t_period / (nt - 1)
    wx = np.ones(nx); wx[0] = wx[-1] = 0.5
    wt = np.ones(nt); wt[0] = wt[-1] = 0.5
    W = dx * dt * np.outer(wt, wx)
    E_period = np.sum(W * U_series ** 2) / (P.LX * t_period)
    cont = float(np.max(np.abs(U_series[0] - U_series[-1])))
    return U_series, U_mean, E_period, cont


def main():
    print("=" * 70)
    print("SERIES U_lid(x,t,Re) PINN -> LBM (campagne 13, seed 42)")
    print("=" * 70)
    rows = []
    for basis in BASIS_SET:
        for Re in RE_SET:
            model = load_model(basis, Re)
            U_series, U_mean, E_period, cont = make_series(model, Re)
            fname = os.path.join(OUT, f"{basis}_Re{Re}.npz")
            np.savez(fname, U_series=U_series, U_mean=U_mean,
                     Re=Re, basis=basis, N_LBM=N_LBM, NT_PER=NT_PER,
                     T_PERIOD=T_PERIOD,
                     E_period=E_period, continuity_err=cont,
                     U_min=U_series.min(), U_max=U_series.max(),
                     U_mean_max=U_mean.max())
            rows.append((basis, Re, E_period, cont, U_series.min(), U_series.max()))
            print(f"  {basis:14s} Re={Re:4d} | E_period={E_period:.4f} "
                  f"continuity={cont:.2e} [Umin,Umax]=[{U_series.min():.3f},{U_series.max():.3f}]")
    np.savetxt(os.path.join(OUT, "lid_series_summary.csv"), rows,
               fmt="%s", delimiter=",", header="basis,Re,E_period,continuity_err,U_min,U_max", comments="")
    print(f"  Series -> {OUT}")


if __name__ == "__main__":
    main()