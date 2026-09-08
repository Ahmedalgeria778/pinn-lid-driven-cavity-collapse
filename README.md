# PINN–ULTRa : Modal control of the lid-driven cavity

Reproducibility material for the manuscript submitted to *Physics of Fluids* (reviewer responses R2.1–R2.10 / R3.1–R3.3).

Active flow control in a square lid-driven cavity, implemented with a physics-informed neural network (PINN). The actuation `U_lid(x,t,Re)` is expanded on the Fourier basis

```
U_lid(x,t,Re) = Σ_{i,j} c_{i,j}(Re) · sin((i+1)πx/Lx) · cos(jπt)
```

(the "ULTRa" control basis: rich in spatial × temporal modes). The central scientific finding: the identified control **collapses onto the stationary second spatial mode (2,0)** = `sin(2πx/Lx)`, robustly across most modeling choices.

**Author**: Ahmed Beniaiche (Laboratory of Fluid Mechanics, École Militaire Polytechnique, Algiers, Algeria). Corresponding author: Ahmed Beniaiche.

---

## 1. Repository layout

```
PoF_R_lid_driven_paper/
├── README.md                         ← this file
├── INSTALL.md                        ← detailed installation (Windows / CPU / Julia)
├── requirements.txt                  ← pinned Python dependencies
├── PINN_Lid_driven_reviewers.py      ← main code (all campaigns, CLI selector)
├── REPORT_CODING.md                  ← detailed technical report (v5): code + results
├── historical_baseline/
│   └── code_find_U_PINN_Lid_driven.py  ← ORIGINAL manuscript code, untouched copy
└── results_reviewers/
    ├── 02_loss_ablation/             ← loss ablation (14 configs)
    ├── 03_energy_sweep/              ← E* sweep (7 targets) + figure
    ├── 04_seed_study/                ← seed sensitivity (10 seeds)
    ├── 07_mode_count/                ← basis dimension (6→72 coefficients)
    ├── 08_temporal/                  ← number of temporal modes (n_t = 1, 3, 5)
    ├── 09_architecture/              ← small / baseline / large (seeds {0,1,2})
    └── mode_count_analysis/          ← analysis scripts + figures
```

> Every run stores `model.pt` (trained PINN weights), `training_history.csv` (epoch, losses, online energy), `residual_stats.csv`, and `metadata.json` (full hyperparameter record).

## 2. Key results (from `results_reviewers/`)

### Mode (2,0) dominance — fraction `f_{(2,0)} = E_{(2,0)}/E_total`

| Experiment | Varied parameter | f_{(2,0)} |
|---|---|---|
| `07_mode_count` (6→72 coeffs) | Fourier basis richness | **87.3–92.3 %** |
| `08_temporal` (n_t = 1, 3, 5) | number of temporal modes | **89.2–92.3 %**, f_t ≤ 1.5 % |
| `09_architecture` (small/baseline/large) | network capacity (9k→140k params) | **86.7–93.6 %** (new runs; intra-seed Δf2 ≤ ±4 % on the mode-2 branch) |
| `13_parametrization` (3 Re × 2 bases) | basis family × Reynolds | Fourier: **0.92 / 0.92 / 0.64** at Re=100/500/1000; Chebyshev_mod: **≤0.06 (all Re, temporal (0,1) branch)** |
| `09_aspect_ratio` (1:1 vs 2:1, Re=500) | cavity aspect ratio | square **0.922** vs rectangular **0.912** → collapse persists (geometric robustness, R3.2 closed) |
| `lbm_mrt_validation` | LBM-MRT D2Q9 independent solver | Ghia-validated; injection of PINN controls; branch admissibility (Fourier quasi-static, Chebyshev fluctuation-dominated) — campaign running |

### Parametrization matrix — f(2,0) in the common Fourier projection (seed 42, E\*=0.25, 6×5)

| Re | fourier f₂ | fourier f_temp | chebyshev_mod f₂ | chebyshev_mod f_temp |
|---:|---:|---:|---:|---:|
| 100 | 0.916 | 0.007 | 0.003 | 0.993 |
| 500 | 0.922 | 0.013 | 0.062 | 0.933 |
| 1000 | 0.640 | 0.322 | 0.005 | 0.988 |

> **Headline for R2-2**: the (2,0) collapse is **parametrization-dependent and, within the Fourier family, Reynolds-conditional** — it holds at Re=100–500 (f₂≈0.92), weakens at Re=1000 (f₂=0.64, 32 % temporal), and disappears in the modulated-Chebyshev family at all tested Re (dominant sin(πx)·cos(πt), temporal fraction ≥ 93 %).

### Energy sweep `03_energy_sweep` (Re=500, basis 6×5)

| E* (target) | E_actual | A₂ | f_{(2,0)} | f_temp |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.0540 | 0.317 | 92.8 % | 0.4 % |
| 0.05 | 0.0790 | 0.382 | 92.2 % | 0.5 % |
| 0.10 | 0.1183 | 0.467 | 92.1 % | 0.7 % |
| 0.25 | 0.2534 | 0.685 | 92.5 % | 1.2 % |
| 0.50 | 0.5006 | 0.925 | 85.4 % | 6.6 % |
| 1.00 | 0.9756 | 1.298 | 86.3 % | 9.2 % |
| 2.00 | 1.9333 | 1.692 | 74.0 % | 21.8 % |

The collapse is robust for `E_actual ≲ 0.25` and degrades at high energy (temporal modes grow jointly).

### Loss ablation `02_loss_ablation` (f_{(2,0)})

| Modified loss | f_{(2,0)} | Interpretation |
|---|---:|---|
| BASELINE / NO_LRE / NO_LDISS / LRE_± / LDISS_± / LCTRL_± | 0.918–0.928 | robust |
| LVAR_HALF / LVAR_DOUBLE | 0.893 / 0.928 | robust |
| **NO_LVAR** (+ NO_LVAR_LRE, NO_LVAR_LDISS) | **0.002–0.003** | **L_var is critical**: its removal switches to a temporally dominated branch |

> Exact wording (do not write "U_lid ≈ 0 degenerate"): *"Removal of L_var causes a transition from the second-spatial-mode-dominated branch to a strongly temporally dominated branch."*

### Seeds `04_seed_study` (f_{(2,0)})

- seeds 1, 2, 6, 7: (2,0) branch — f₂ = 0.86–0.91;
- seeds 0, 4: mixed branch — f₂ ≈ 0.52;
- seeds 3, 5, 8, 9: temporally dominated branch — f₂ ≈ 0.001–0.02, f_t ≈ 0.98.

→ **No global uniqueness**: the branch reached depends on initialization (seed); mode (2,0) is not the only fixed point. The following campaigns quantify *within-branch* robustness.

## 3. Official definition of the energy (cite in Methods)

- **Published metric: `E_total`** = `⟨U_lid²⟩` computed by **consistent trapezoidal quadrature** on a uniform 200×200 grid (2D weights `W = dx·dt·outer(w_x,w_t)`, half-weight edges), normalized by `lx·T_MAX`. Deterministic; `modal_coverage = Σ E_modes / E_total = 1.0` at machine precision (orthonormality-consistent basis).
- **Online metric (training_history.csv)**: Monte-Carlo average over 600 fixed BC points (~±3–5 % noise). Optimization driver only, **not** the published value.

## 4. Installation

See `INSTALL.md`. In brief:

```bash
pip install -r requirements.txt
```

- PyTorch: CPU build is sufficient (`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu`).
- The `symbolic` campaign (PySR) requires a Julia installation beforehand — not needed for the main campaigns.

## 5. Usage

```bash
python PINN_Lid_driven_reviewers.py <mode>
```

Available modes: `baseline`, `loss_ablation`, `energy_sweep`, `seed_study`, `architecture`, `sampling`, `mode_count`, `temporal`, `aspect_ratio`, `unseen_re`, `symbolic`, `convergence`, `parametrization`, `all`.

Useful environment variables:
- `OUT_DIR` — output directory (default: `results_reviewers/`);
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` — set to 5–6 before launch; reduce when several campaigns run in parallel.

Reproducibility: `set_seed(seed)` locks all RNGs (random, numpy, torch, cuda). Every campaign uses the same seeds as in the manuscript. The `energy_sweep` and `mode_count` campaigns implement a **skip logic**: if `model.pt` already exists, the model is reloaded instead of retrained.

Indicative run times (CPU, 5 threads): baseline 6×5 ≈ 1 h; `architecture` large ≈ 1.5–2 h per run; full `all` campaign ≈ 2–3 days.

## 6. License

This software is distributed under the **Apache License, Version 2.0**.
See [LICENSE](LICENSE) for the full text and [NOTICE](NOTICE) for the
required attribution statement.

Copyright 2026 Ahmed Beniaiche, Laboratory of Fluid Mechanics,
Ecole Militaire Polytechnique, Algiers, Algeria.

## 7. Citation

Any publication, thesis, report, presentation, or derivative work that
uses, adapts, or redistributes this code, its outputs, or its figures
**must** cite:

> Ahmed Beniaiche,
> *"Physics-informed identification and independent numerical verification
> of conditional modal collapse in active control of lid-driven cavity
> flow"*, **Physics of Fluids** (2026).

**BibTeX**

```bibtex
@article{beniaiche2026pinn,
  author  = {Beniaiche, Ahmed},
  title   = {Physics-informed identification and independent numerical verification
             of conditional modal collapse in active control of lid-driven cavity
             flow},
  journal = {Physics of Fluids},
  year    = {2026}
}
```

A `CITATION.cff` file is included so that GitHub displays a **Cite this
repository** button automatically.

## 8. Statements for the manuscript

1. The historical code (`historical_baseline/`) has **never been modified**.
2. No external NS solver is used for the loss (direct autodiff); the Ghia benchmark and the independent optimization (LBM) are handled separately.
3. "unseen-Re" = inter-Re interpolation, not exhaustive validation.
4. No GCI: robustness is quantified by seeds, collocation density, architecture, basis dimension, geometry.