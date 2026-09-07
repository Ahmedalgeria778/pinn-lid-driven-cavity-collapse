# Rebuttal summary (consolidated, R2/R3)

## 1. Independent-lid physical validation (LBM-MRT D2Q9, this repo)

- Ghia (lid uniforme, N=256) : Re=100 — L2(u)=0.0051, Linf(u)=0.0128 — L2(v)=0.0068, Linf(v)=0.0114
- Ghia (lid uniforme, N=256) : Re=1000 — L2(u)=0.0141, Linf(u)=0.0300 — L2(v)=0.0132, Linf(v)=0.0212

## 2. Branch admissibility (time-dependent forcing)

| Re | control | K_fluct/K | dom. mode (mean) | dom. mode (fluct) | A2_flow |
|----|---------|-----------|------------------|-------------------|---------|
| 100.0 | pinn_t | 0.2% | 1.0 (0.16) | 1.0 (0.72) | 0.0028 |
| 100.0 | cheb_t | 846.8% | 4.0 (0.16) | 7.0 (0.20) | -0.0044 |
| 500.0 | pinn_t | 0.0% | 2.0 (0.72) | 1.0 (0.68) | -0.0057 |
| 500.0 | cheb_t | 43.8% | 3.0 (0.25) | 1.0 (0.39) | -0.0036 |
| 1000.0 | pinn_t | 0.6% | 2.0 (0.51) | 1.0 (0.71) | -0.0529 |
| 1000.0 | cheb_t | 125.0% | 3.0 (0.18) | 1.0 (0.51) | -0.0032 |

## 3. Aspect ratio (R3.2, Re=500, seed 42, E*=0.25)

| geometry | A2 | f_(2,0) | f_temp | E_total |
|----------|----|---------|--------|---------|
| square | 0.6835 | 0.922 | 0.013 | 0.2534 |
| rectangular | 0.6770 | 0.912 | 0.016 | 0.2514 |

> Mode-2 collapse persists in the 2:1 cavity (geometric robustness, parametrization-boundary preserved).
