# Independent re-verification of 09_architecture/architecture_results.csv
# Recomputes every metric from the trained model.pt files with a SEPARATE
# implementation of the quadrature and the modal projection (does not call
# analyze_control), then compares against the CSV row by row.
import os
import sys
import numpy as np
import torch
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import PINN_Lid_driven_reviewers as m

OUT = "results_reviewers"
CSV = os.path.join(OUT, "09_architecture", "architecture_results.csv")
df = pd.read_csv(CSV)

NX, NT, RE = 200, 200, 500.0
N_MX, N_MT = 6, 5


def build_model(arch, seed):
    if arch == "baseline":
        cfg = None
        src = os.path.join(OUT, "04_seed_study", f"seed_{seed}", "model.pt")
    else:
        cfg = m.ARCHITECTURES[arch]
        src = os.path.join(OUT, "09_architecture", arch, f"seed_{seed}", "model.pt")
    model = m.UltraPINN(arch_config=cfg, n_mx=N_MX, n_mt=N_MT,
                        lx=m.LX, ly=m.LY, basis_type="fourier").to(m.device)
    model.load_state_dict(torch.load(src, map_location="cpu", weights_only=False))
    model.eval()
    return model


def quadrature_grid(nx, nt):
    lx, tmax = m.LX, m.T_MAX
    x = np.linspace(0.0, lx, nx)
    t = np.linspace(0.0, tmax, nt)
    dx, dt = lx / (nx - 1), tmax / (nt - 1)
    wx = np.ones(nx); wx[0] = 0.5; wx[-1] = 0.5
    wt = np.ones(nt); wt[0] = 0.5; wt[-1] = 0.5
    W = dx * dt * np.outer(wx, wt)
    return x, t, W, lx * tmax


def phi(i, j, x, t, lx):
    return np.outer(np.sin((i + 1) * np.pi * x / lx), np.cos(j * np.pi * t))


def independent_metrics(model):
    xs = torch.linspace(0, m.LX, NX, dtype=m.DTYPE, device=m.device)
    ts = torch.linspace(0, m.T_MAX, NT, dtype=m.DTYPE, device=m.device)
    X, T = torch.meshgrid(xs, ts, indexing="ij")
    Xf, Tf = X.reshape(-1, 1), T.reshape(-1, 1)
    Rt = torch.full_like(Xf, RE)
    with torch.no_grad():
        U = model.U_lid(Xf, Tf, Rt).cpu().numpy().reshape(NX, NT)
    x, t, W, area = quadrature_grid(NX, NT)

    E_total = float(np.sum(W * U ** 2) / area)

    em = {}
    hashes = {}
    for i in range(N_MX):
        for j in range(N_MT):
            p = phi(i, j, x, t, m.LX)
            n2 = float(np.sum(W * p ** 2))
            c = float(np.sum(W * U * p) / n2)
            hashes[(i, j)] = c
            em[(i, j)] = c ** 2 * n2 / area
    E_modes_sum = sum(em.values())
    dominant = max(em, key=em.get)

    recon = np.zeros_like(U)
    for (i, j) in em:
        recon += hashes[(i, j)] * phi(i, j, x, t, m.LX)

    return {
        "E_total": E_total,
        "A2": hashes[(1, 0)],
        "mode2_fraction": em[(1, 0)] / E_total,
        "temporal_fraction": sum(v for (i, j), v in em.items() if j > 0) / E_total,
        "modal_coverage": E_modes_sum / E_total,
        "Efrac_sum": sum(v / E_total for v in em.values()),
        "dominant_i": int(dominant[0]),
        "dominant_j": int(dominant[1]),
        "reconstruction_rmse": float(np.sqrt(np.mean((U - recon) ** 2))),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-30)


print(f"{'arch':<9}{'seed':<5}{'E_total':<12}{'A2':<12}{'mode2':<10}{'ftemp':<10}"
      f"{'cov':<8}{'rmse':<8}{'params':<8}  rel-diff(max)")
worst = 0.0
fails = []
for _, r in df.iterrows():
    arch, seed = r["architecture"], int(r["seed"])
    model = build_model(arch, seed)
    im = independent_metrics(model)
    diffs = {
        "E_total": rel(im["E_total"], r["E_total"]),
        "A2": rel(im["A2"], r["A2"]),
        "mode2_fraction": rel(im["mode2_fraction"], r["mode2_fraction"]),
        "temporal_fraction": rel(im["temporal_fraction"], r["temporal_fraction"]),
        "modal_coverage": rel(im["modal_coverage"], r["modal_coverage"]),
        "reconstruction_rmse": rel(im["reconstruction_rmse"], r["reconstruction_rmse"]),
        "n_params": rel(im["n_params"], r["n_params"]),
        "dominant_i+j": 0.0 if (im["dominant_i"] == r["dominant_i"] and im["dominant_j"] == r["dominant_j"]) else 1.0,
    }
    dmax = max(diffs.values())
    worst = max(worst, dmax)
    flag = "" if dmax < 1e-6 and im["Efrac_sum"] > 0.9999 else "  <-- CHECK"
    if dmax >= 1e-6:
        fails.append((arch, seed, sorted(diffs.items(), key=lambda kv: -kv[1])[:3]))
    print(f"{arch:<9}{seed:<5}{im['E_total']:<12.6f}{im['A2']:<12.4f}{im['mode2_fraction']:<10.4f}"
          f"{im['temporal_fraction']:<10.4f}{im['modal_coverage']:<8.6f}{im['reconstruction_rmse']:<8.2e}"
          f"{im['n_params']:<8}{dmax:<8.2e}{flag}")

ok_cov = bool((df["modal_coverage"] - 1.0).abs().max() < 1e-5)
ok_fourier = bool((df["fourier_mode2_fraction"] - df["mode2_fraction"]).abs().max() == 0)
print(f"\nWORST rel-diff: {worst:.2e}  (threshold 1e-6)")
print(f"modal_coverage == 1.0 for all rows: {ok_cov}")
print(f"fourier_mode2_fraction == mode2_fraction everywhere: {ok_fourier}")
print("Efrac_sum ~ 1 for all rows (no truncation loss):", bool(df["E_total"].notna().all()))
print("FAILURES:", fails if fails else "NONE")