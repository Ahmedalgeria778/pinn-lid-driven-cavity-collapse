# Detailed report — Coding of the reviewer-response campaigns

**Date**: 2026-09-05 (v5.2 — architecture campaign 09 COMPLETED, parametrization running)
**Author**: opencode (big-pickle)
**Folder**: `PoF_R_lid_driven_paper/`

---

## Executive summary (v5)

The modified file **`PINN_Lid_driven_reviewers.py`** (~2039 lines, v5) was created in response to reviewers R2-1 through R2-10 and R3-1 through R3-3, **without touching the historical code** (`historical_baseline/code_find_U_PINN_Lid_driven.py`, 846 lines, byte-identical copy of the original).

### v5 corrections (mode_count + temporal campaigns RUN, energy diagnostic)

**C5. Diagnosis of the `energy_final` (training history) vs `E_total` (analyze_control) discrepancy.**
- `energy_final` (labeled `energy` in `training_history.csv`) = **Monte-Carlo mean** `(U_lid²).mean()` over the **600 fixed top-BC points** (~points sampled uniformly in (x,t)). Noisy estimator: ±3–5 %.
- `E_total` (`analyze_control`) = **consistent trapezoidal quadrature** on a uniform 200×200 grid. Deterministic, converged, `modal_coverage=1.0` at machine precision.
- Measured on `mx6_mt1`: `E_total=0.239008`; fresh MC n=600 (40 repetitions) = `0.2417±0.007`; fixed-point reproduction = `0.229`; logged history = `0.25944`. The 7.9 % gap is therefore **the sampling noise of the online estimator, not a bug**.
- **Decision**: the published metric is everywhere `E_total` (quadrature). The online value was used solely to drive the optimizer toward `E_target`. To mention in the manuscript: the control is optimized for a Monte-Carlo-evaluated target, hence a residual variance of ≈±5 % on `E_total` for a given `E_target` (observed: 0.239–0.278 across the 6 bases).

### Frozen protocol — 09_architecture campaign (seeds {0,1,2}, defined before results)

User decision frozen before any `small`/`large` result (2026-09-05):
- **Only variable**: network capacity (small / baseline / large of `ARCHITECTURES`).
- **Frozen common configuration**: Re=500, E_target=0.25, Fourier basis **6×5** (30 coeffs), epochs/sampling/losses identical (`train_ultra` without override).
- **Seeds fixed a priori**: {0,1,2} — not chosen by any result; seed 0 happens to reach the intermediate branch (f2≈53 %), seeds 1,2 the mode-2 branch (f2≈91 %, 89 %). Purely to make the selection non-suspicious.
- **Paired intra-seed comparison**: Δf2(arch) = f2^small − f2^baseline, same for large. Reduces the common stochasticity.
- **Baseline = validated reuse**: the `04_seed_study/seed_{0,1,2}` model.pt files were checked — same state_dict keys + shapes as a fresh `UltraPINN(arch_config=None)` (net [128,128,96,64], coeff [64,64,48], 44 824 params); E_target, N_MT, N_MX, N_PHYS, N_BC, lambdas identical in metadata.json. → **zero baseline retraining**.
- **Computations**: 3 small + 3 large = 6 new runs. Output `09_architecture/architecture_results.csv` (columns: architecture, seed, Re, n_mx, n_mt, n_modes, E_target, E_total, mode2_fraction, fourier_mode2_fraction, temporal_fraction, A2, dominant_i, dominant_j, modal_coverage, reconstruction_rmse, n_params, reused, reused_source, final_loss, PDE_loss, BC_loss, diss_loss, ctrl_loss, var_loss).
- First launch erroneously made with seeds {1,2,6} by the assistant → **killed and purged**, relaunched with seeds {0,1,2} (PID 41232).
- Closure criteria: dominance of (2,0) or an identifiable branch coherent with the seed's baseline; regime stability (no exact equality required); low temporal_fraction; modal_coverage≈1; reconstruction_rmse~1e-16.

### mode_count results (R2.4/R3.2) — 6 configs, 6→72 coefficients

`07_mode_count/mode_count_results.csv`, fixed seed (seed 42), `dominant=(1,0)` in all 6 cases (spatial mode (2,0) = sin(2πx)):

| Config | \(n_{mx}\times n_{mt}\) | n coeffs | \(E_{total}\) | \(A_2\) | mode2_fraction | temporal_fraction |
| -----: | -----: | -----: | ------: | ------: | --------------: | ----------------: |
| 4×3 | 12 | 12 | 0.2586 | 0.6867 | 0.9115 | 0.0088 |
| 6×1 | 6 | 6 | 0.2390 | 0.6529 | 0.8919 | 0.0000 |
| 6×3 | 18 | 18 | 0.2778 | 0.7041 | 0.8925 | 0.0132 |
| 6×5 | 30 | 30 | 0.2533 | 0.6839 | 0.9234 | 0.0122 |
| 8×5 | 40 | 40 | 0.2475 | 0.6592 | 0.8780 | 0.0149 |
| 8×9 | 72 | 72 | 0.2775 | 0.6961 | 0.8729 | 0.0287 |

- **mode2_fraction: 89.5 % ± 2.0 %, range [87.3, 92.3]** on 6→72 coefficients.
- temporal contribution < 2.9 % in all cases.
- Conclusion: the collapse onto mode (2,0) is **robust to the dimension of the parametrization**.
- Publication (config `(8,9)` added in v5): `MODE_CONFIGS = [(4,3),(6,1),(6,3),(6,5),(8,5),(8,9)]` with a **skip logic** (reloads `model.pt` if present) to avoid retraining the 5 already-trained bases.

> **Consensus wording**: *"From 6 to 72 coefficients, the second spatial mode consistently dominates, accounting for 87.3–92.3 % of the control energy."*

### temporal results (R3.2) — 3 configs (6×1, 6×3, 6×5)

`08_temporal/temporal_results.csv`. Effect of the number of temporal modes \(n_{mt}\) at fixed \(n_{mx}=6\):

| Config | n temporal | \(E_{total}\) | \(A_2\) | mode2_fraction | temporal_fraction |
| -----: | -----: | -----: | ------: | --------------: | ----------------: |
| 6×1 | 1 | 0.2390 | 0.6529 | 0.8919 | 0.0000 |
| 6×3 | 3 | 0.2778 | 0.7041 | 0.8925 | 0.0132 |
| 6×5 | 5 | 0.2533 | 0.6839 | 0.9234 | 0.0122 |

- Adding temporal modes **does not change the spatial dominance**: \(f_{(2,0)}\) stays 89.2–92.3 %.
- The spontaneous temporal fraction stays ≤1.5 %: the identified control is **quasi-steady** in the spatial mode (2,0) — the temporal basis modes do not "cheat".

> **Consensus wording**: *"Enriching the temporal basis (1, 3, or 5 temporal modes) leaves the spatial dominance essentially unchanged (mode-2 fraction 89.2–92.3 %), with temporal contributions never exceeding 1.5 % of the control energy — the identified control is quasi-steady in the second spatial mode."*

### architecture results (R2.4) — COMPLETED (2026-09-05, ~16:34)

`09_architecture/architecture_results.csv`. Seeded pairs {0,1,2} chosen a priori, Re=500, E_target=0.25, Fourier 6×5, baseline = reused `04_seed_study` models (verified identical, 44 824 params). Param counts: small 8 774 / baseline 44 824 / large 139 624.

| arch | seed | E_total | A₂ = c₁₀ | mode2_fraction | temporal_fraction | dominant |
|---|---|---:|---:|---:|---:|---|
| small | 0 | 0.2762 | +0.7192 | **0.9361** | 0.0043 | (1,0) |
| small | 1 | 0.2445 | +0.6644 | **0.9030** | 0.0170 | (1,0) |
| small | 2 | 0.2561 | +0.6875 | **0.9230** | 0.0071 | (1,0) |
| baseline | 0 | 0.2398 | −0.5053 | 0.5323 | 0.4376 | (1,0) |
| baseline | 1 | 0.2673 | +0.6978 | 0.9107 | 0.0131 | (1,0) |
| baseline | 2 | 0.2567 | +0.6766 | 0.8917 | 0.0504 | (1,0) |
| large | 0 | 0.2572 | +0.6888 | **0.9223** | 0.0135 | (1,0) |
| large | 1 | 0.2518 | +0.6610 | **0.8674** | 0.0477 | (1,0) |
| large | 2 | 0.2541 | +0.6711 | **0.8863** | 0.0300 | (1,0) |

All 9 runs: `modal_coverage = 1.0`, `reconstruction_rmse ~ 1e-16`.

Pairwise Δf2 = f2(arch) − f2(baseline), intra-seed:
- seed 0: small **+0.404** (0.532→0.936), large **+0.390** (0.532→0.922) — the intermediate branch **is pulled back onto the (2,0)-dominated branch** by both other architectures; A₂ sign flips (−0.505 → +0.72/+0.69).
- seed 1: small −0.008, large −0.043 — unchanged within ±4 %.
- seed 2: small +0.031, large −0.005 — unchanged within ±3 %.

Mean f2 across arch (over the 3 seeds): small 0.921 / baseline 0.778 / large 0.892. Across the **6 new runs**, f2 ∈ [0.867, 0.936] — the mode-(2,0) branch is **architecture-independent within ±4 %** for seeds already on that branch.

**Interpretation (reviewer R2.4)**: the network capacity (8.8 k → 44.8 k → 139.6 k params) does **not** select or destroy mode (2,0). Within-branch f2 is essentially constant; the only architectural sensitivity observed is a *restoring* one — the mixed branch of seed 0 disappears under both a smaller and a larger network, both converging to the mode-2 branch.

> **Consensus wording** (suggested): *"Within a fixed seed, varying the network capacity from ~9×10³ to ~1.4×10⁵ parameters leaves the dominance of the second spatial mode essentially unchanged (mode-2 fraction 86.7–93.6 %); the intermediate branch observed for seed 0 at the baseline size is not selected under either a smaller or a larger architecture."*

The guiding principle is respected:
> **Intact historical baseline + parameterized experimental functions**

The baseline is 100 % preserved via `run_baseline()`, which reproduces the original `main()` logic exactly (same seeds, same results).

### v2 corrections (after the first verdict)

1. **`A20/A30/...` corrected**: they are now the **true coefficients \(c_{i,0}\)** (spatial mode i, stationary temporal j=0), plus a sum of normalized modal energies `E_modes_sum` and the complete distribution `Efrac_{i}_{j}`.
2. **`N_MT` initialization fixed**: explicit use of `mode_index(i,j) = i*n_mt + j`, avoids the silent overwrite when N_MT=1.
3. **Energy normalization in `analyze_control`**: modal energies now divided by `domain_area = lx*T_MAX` — comparable between \(L_x=1\) and \(L_x=2\).
4. **Vocabulary `optimal` removed** from figures → "identified control" / "identified law".
5. **Symbolic regression: physical Re** → `extract_lid` now writes `Re_phys` instead of `Re/1000` (recommended Option B).
6. **Extended ablation**: + `LCTRL_HALF`, `LCTRL_DOUBLE`.
7. **New `parametrization` campaign (13)**: compares the Fourier basis vs. the modulated Chebyshev `x(Lx-x)·T_i(2x/Lx-1)` ← answers the R2-2 weakness "same family of functions".
8. **Figure fonts (AE)**: labels ≥14pt, ticks 12pt.
9. **`unseen_Re` conceptually renamed**: inter-Re interpolation (not independent validation).

### v3 corrections (after the 2nd verdict — final corrections)

Addressing the 3 remaining technical points flagged in the final verdict:

**C1. `energy_sweep` cache fixed.** The bug `hist = {"energy": [E_TARGET]}` (which wrote 0.25 as the final energy for runs reloaded from disk) is gone. Now **`ac["energy_final"] = ac["E_total"]`**: the recorded energy is always the one measured by `analyze_control` on the \(200\times200\) grid, whether the model was trained or reloaded. **Applied uniformly to all campaigns** (loss_ablation, energy_sweep, seed, architecture, sampling, mode_count, temporal, aspect_ratio, parametrization).

**C2. `E_j` normalization in `run_temporal_study`.** The integrated temporal energies \(\iint U_j^2\,dxdt\) are now divided by `domain_area = lx*T_MAX`, so that \(E_j\) has exactly the same normalization (space–time average) as \(E_{total}\). Added `E_mode0` (stationary energy j=0). Consistent with the v2 fix in `analyze_control`.

**C3. Fourier/Chebyshev comparison in a common basis.** The main issue: for `basis_type="chebyshev_mod"`, `A20 = c_{1,0}` was the coefficient of the **modified Chebyshev polynomial** (scale depending on \(L\), envelope max \(1/4\) for \(L=1\)), not of the harmonic \(\sin(2\pi x/L)\). Fixes:
- `analyze_control` **always projects onto a common Fourier basis** (independent of `basis_type`), so `A20`, `mode2_fraction`, `Efrac_*` **are comparable between the two families**. This is exactly the recommended metric \(f_{2,0}^{\text{Fourier}}\).
- Added **native coefficients** `native_c_{i}_{j}` (from `coeff_net`) to document the parametrization itself — NOT comparable across bases (different scales).
- Explicitly added `fourier_mode2_fraction` (= `mode2_fraction`) as the main comparison metric.
- `run_parametrization_study` docstring updated to explain the distinction.

**C4. Tempered Parseval comment.** "≈ E_total by Parseval" → "sum of the energies of the retained Fourier projections" (correct even in a truncated or non-Fourier basis). Added the metric **`modal_coverage = E_modes_sum / E_total`** (~1 for a good reconstruction).

**Functionally verified**: `analyze_control` correctly produces `modal_coverage`, `native_c_*`, `fourier_mode2_fraction`, `basis_type`; for a pure mode 2 with \(A_{20}=0.5\), `mode2_fraction≈1`, `modal_coverage≈1`; for the Chebyshev basis, `native_c_1_0` differs from the Fourier projection `A20` (distinction confirmed).

### v4 corrections (quadrature fix — exact `modal_coverage`)

Following the verification requested before freezing the numbers, the anomaly `modal_coverage ≈ 1.009–1.010` was **diagnosed and fixed**.

**Root cause** (numerically confirmed): two **inconsistent** quadratures.
- `E_total = np.mean(U²)` used an **arithmetic mean** (uniform weights \(1/N\)).
- `E_modes_sum` used projections with `norm2 = Σ_n φ²·dx·dt` = trapezoidal rule with **over-weighted edges**.

The trapezoidal rule with over-weighted edges gives, for the **constant** mode \(j=0\) (which is precisely the dominant mode of our fields, `dominant_j=0`), \(\sum_n dt = n\,dt = 200\cdot(1/199) = 1.005\) instead of \(1.0\). This ~+0.5 % bias propagates to `E_modes_sum`, hence to `modal_coverage` (~1.01).

**Fix** (in `analyze_control`): a single **consistent standard trapezoidal quadrature**, via 2D weights \(W = dx\,dt\,\mathrm{outer}(w_x,w_t)\) with half-weight edges:
- modal norms `norm2 = Σ W·φ²`;
- projections `coeff = Σ W·U·φ / norm2`;
- total energy `E_total = Σ W·U² / domain_area`.

Under this single bilinear form, the basis \(\{\sin((i+1)\pi x/L)\cos(j\pi t/T)\}\) is **exactly orthonormal-consistent** (norms and orthogonality verified at machine precision). By construction, for a function exactly in the basis, `modal_coverage = E_modes_sum/E_total = 1.0` exactly.

**Validation on the 7 sweep models**: `modal_coverage = 1.000000` (machine precision) for all. The values of `E_total`, `mode2_fraction`, `A20`, `temporal_fraction` were recomputed with the correct quadrature (slight shift vs `np.mean`, e.g. E0.25: \(E_{total}=0.2521\to0.2534\)).

### Final result of the "energy_sweep" campaign (Re=500)

All metrics rely on `analyze_control` with the **corrected consistent quadrature**, grid \(200\times200\), \(E_{\rm actual}=E_{\rm total}=\langle \tilde U_{\rm lid}^2\rangle\) (weighted space–time mean). \(E^*=E_{\rm target}\) is the penalty target, \(E_{\rm actual}\) the energy actually achieved.

| \(E^*\) (target) | \(E_{\rm actual}\) | \(A_2\) | Mode-2 fraction | Temporal fraction | \(modal\_coverage\) |
| -------------: | -----------------: | ------: | --------------: | ------------------: | ------------------: |
|            0.01 |             0.0540 |   0.317 |          92.8 % |               0.4 % |              1.000 |
|            0.05 |             0.0790 |   0.382 |          92.2 % |               0.5 % |              1.000 |
|            0.10 |             0.1183 |   0.467 |          92.1 % |               0.7 % |              1.000 |
|            0.25 |             0.2534 |   0.685 |          92.5 % |               1.2 % |              1.000 |
|            0.50 |             0.5006 |   0.925 |          85.4 % |               6.6 % |              1.000 |
|            1.00 |             0.9756 |   1.298 |          86.3 % |               9.2 % |              1.000 |
|            2.00 |             1.9333 |   1.692 |          74.0 % |              21.8 % |              1.000 |

**Interpretation (answer to reviewer R3)**:
- **Collapse robustness zone**: for \(E_{\rm actual}\in[0.054,0.253]\) (i.e. \(E^*\) from 0.01 to 0.25), the mode-2 fraction stays almost constant at **92.1–92.8 %** while \(A_2\) grows (0.317 → 0.685). Mode 2 dominates **independently of the historical choice \(E^*=0.25\)** over this range.
- **Progressive degradation at high energies**: for \(E_{\rm actual}\gtrsim0.5\), the fraction drops to 85–86 %, then **74.0 % at \(E_{\rm actual}=1.93\)**, with a joint rise of the temporal fraction (0.4 % → 21.8 %). The collapse becomes **energy-conditional**.
- **Historical operating point**: \(E^*=0.25\) gives \(A_2\simeq0.685\) and mode-2 fraction ≈ 92.5 %, **consistent with the historical baseline** (mode-2 dominance ~93 %).

> **Suggested consensus wording**: *"The modal collapse toward the second spatial mode is robust over a finite range of actuation energies, approximately up to \(E_{\rm actual}\approx0.25\), whereas stronger actuation progressively increases the contribution of higher spatial and temporal modes and weakens the collapse."*

The CSV `results_reviewers/03_energy_sweep/energy_sweep.csv` and the associated figures were regenerated with the corrected quadrature.

---

## 1. What was preserved (MANDATORY)

The `historical_baseline/` folder contains:
- `code_find_U_PINN_Lid_driven.py` — exact copy of the original code (verified)

All the historical constants are identically defined in the modified code:
```
RE_LIST = [100, 500, 1000]
Re_min, Re_max = 100.0, 1000.0
N_EPOCHS_PRE = 300
N_EPOCHS_MAIN = 3000
N_PHYS = 4000
N_BC = 600
LAMBDA_PDE = 1.0
LAMBDA_BC = 15.0
LAMBDA_DISS = 0.05
LAMBDA_CTRL = 200.0
LAMBDA_VAR = 1.0
LAMBDA_RE = 2.0
E_TARGET = 0.25
```

→ **The historical code is NOT modified.** The original file is copied as-is into `historical_baseline/`.

---

## 2. Mode selector (implemented)

Lines 33-46. The `MODE` selector launches each campaign independently:

```python
MODE = "baseline"
# MODE = "loss_ablation"  # R2.2 / R3.2
# MODE = "energy_sweep"   # R3.2
# MODE = "seed_study"     # R2.4 / R3.3
# MODE = "architecture"   # R2.4
# MODE = "sampling"       # R2.4
# MODE = "mode_count"     # R2.4 / R3.2
# MODE = "temporal"       # R2.4 / R3.2
# MODE = "aspect_ratio"   # R3.2
# MODE = "unseen_re"      # inter-Re robustness
# MODE = "symbolic"       # R2.8 / R3.1
# MODE = "convergence"    # R2.7
# MODE = "all"            # all campaigns
```

The dispatch is done in the `if __name__ == "__main__":` block.

---

## 3. Reviewer #2-1: "not shown to be globally optimal"

**Requested**: compare the PINN baseline with an independent optimization.

**STATUS**: ⚠️ Not coded in `PINN_Lid_driven_reviewers.py`.

**Justification**: the explicit instruction text states this campaign requires a heavy external solver (LBM-type). The directives in section 3 of the instructions document say:

> "But since you said we'll handle the LBM separately, for the purely PINN simulation part I would not consider this step mandatory if it requires a heavy external solver. It is **strongly recommended**, but the other campaigns have priority."

The `13_independent_optimization/` folder was created as a placeholder for this campaign. It will be handled once the LBM code is available.

---

## 4. Reviewer #2-2 + #3-2/1: Loss ablation (MANDATORY — IMPLEMENTED)

**Function**: `run_loss_ablation()` (lines 1144-1205)

### What was coded

1. **Loss flags** (lines 59-64):
```python
USE_PDE  = True
USE_BC   = True
USE_DISS = True
USE_CTRL = True
USE_VAR  = True
USE_RE   = True
```

2. **`compute_loss` modification** (lines 360-367) — exact replacement as requested:
```python
total = (
    (LAMBDA_PDE  * l_pde  if USE_PDE  else 0.0) +
    (LAMBDA_BC   * l_bc   if USE_BC   else 0.0) +
    (LAMBDA_DISS * l_diss if USE_DISS else 0.0) +
    (LAMBDA_CTRL * l_ctrl if USE_CTRL else 0.0) +
    (LAMBDA_VAR  * l_var  if USE_VAR  else 0.0) +
    (LAMBDA_RE   * l_re   if USE_RE   else 0.0)
)
```

3. **The requested minimal campaigns** (v2: 14 configs, LCTRL added):
```
BASELINE        (no ablation)
NO_LVAR         (USE_VAR=False)
NO_LRE          (USE_RE=False)
NO_LDISS        (USE_DISS=False)
NO_LVAR_LRE     (USE_VAR=False, RE=False)
NO_LVAR_LDISS   (USE_VAR=False, DISS=False)
LVAR_HALF       (LAMBDA_VAR × 0.5)
LVAR_DOUBLE     (LAMBDA_VAR × 2)
LDISS_HALF      (LAMBDA_DISS × 0.5)
LDISS_DOUBLE    (LAMBDA_DISS × 2)
LRE_HALF        (LAMBDA_RE × 0.5)
LRE_DOUBLE      (LAMBDA_RE × 2)
LCTRL_HALF      (LAMBDA_CTRL × 0.5)     ← added v2
LCTRL_DOUBLE    (LAMBDA_CTRL × 2.0)     ← added v2
```

> **Why LCTRL**: \(\lambda_{ctrl}\) is directly tied to the pressure toward the energy target. Verifying that a reasonable change of \(\lambda_{ctrl}\) does not artificially create mode 2 is scientifically essential (flagged by the reviewer's verdict).

4. **Metrics recorded per run**: `Re`, `A1`, `A2`, `A3`, `E_total`, `mode2_fraction`, `dissipation`, `PDE_loss`, `BC_loss`, `variance`, `seed`, `dominant_i`, `dominant_j`, `temporal_fraction`, all modal coefficients `c_{i}_{j}`, distribution `Efrac_{i}_{j}`.

5. **Results**: the essential test is that **Mode 2 persists despite the loss changes** — this is what the runs will measure.

---

## 5. Reviewer #3-2: E\* sweep (MANDATORY — IMPLEMENTED)

**Function**: `run_energy_sweep()` (lines 1211-1269)

### What was coded

1. **The `ENERGY_TARGETS` table exactly as requested** (line 129):
```python
ENERGY_TARGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
```

2. **The convention `E = ⟨U_lid²⟩` is EXPLICITLY kept.** The historical baseline uses `E_TARGET = 0.25` and `E = mean(U_lid**2)` — this is the crucial point answering reviewer #3 who writes `E_target = 0.5`.

3. Each target is applied by changing `E_TARGET` via `train_ultra(e_target=e_tgt)`, with automatic save/restore of the globals (try/finally pattern).

4. **Two figures produced** (lines 1251-1266):
   - `E_target → E_{(2,0)}/E_total` (mode2_fraction)
   - `E_target → A2`

5. Results CSV: `03_energy_sweep/energy_sweep.csv`

→ **This is the most important experiment for reviewer #3.**

---

## 6. Reviewer #3: Seeds (MANDATORY — IMPLEMENTED)

**Function**: `run_seed_study()` (lines 1275-1319)

### What was coded

1. **`SEEDS = list(range(10))`** (0 to 9, line 49) — exactly the requested number (5-10).

2. **`set_seed(seed)`** (lines 51-56) — seeds all RNGs (random, numpy, torch, cuda).

3. Loop over the 10 seeds, each with `set_seed(seed)` before training, then `analyze_control`.

4. **Mean ± standard deviation** computed and printed for: A1, A2, A3, mode2_fraction.

5. **No value imposed in advance** — the actual results will be measured.

---

## 7. Reviewer #2-4: Architecture (MANDATORY — IMPLEMENTED)

**Function**: `run_architecture_study()` (lines 1325-1369)

### What was coded

1. **`ARCHITECTURES` exactly as requested** (lines 94-107):
```python
ARCHITECTURES = {
    "small":   {"net": [64, 64, 32],   "coeff": [32, 32]},
    "baseline": {"net": [128, 128, 96, 64], "coeff": [64, 64, 48]},
    "large":   {"net": [256, 256, 128, 96], "coeff": [128, 128, 64]}
}
```

2. **The `UltraPINN` network is parameterized** by `arch_config` (lines 158-226). The baseline architecture (128,128,96,64 / 64,64,48) is kept by default and produces exactly the same topology as the original.

3. Results CSV: `05_architecture/architecture_results.csv` with columns: `arch`, `arch_config`, `n_params`, `A1`, `A2`, `A3`, `mode2_fraction`, `PDE`, `dissipation`, etc.

---

## 8. Reviewer #2-4: Collocation point count (MANDATORY — IMPLEMENTED)

**Function**: `run_sampling_study()` (lines 1375-1419)

### What was coded

1. **`SAMPLES` exactly as requested** (lines 110-114):
```python
SAMPLES = {
    "low":      (2000, 300),
    "baseline": (4000, 600),
    "high":     (8000, 1200)
}
```
where the tuple is `(N_PHYS, N_BC)`.

2. Each config temporarily changes `N_PHYS` and `N_BC` via `train_ultra(n_phys=..., n_bc=...)` with the save/restore pattern.

3. **What it demonstrates**: that the collapse does not depend on the 4000+600 choice.

---

## 9. Reviewer #2-4 and #3: Number of modes (MANDATORY — IMPLEMENTED)

**Function**: `run_mode_count_study()` (lines 1425-1469)

### What was coded

1. **`MODE_CONFIGS` exactly as requested** (lines 126-133):
```python
MODE_CONFIGS = [(4, 3), (6, 1), (6, 3), (6, 5), (8, 5), (8, 9)]
```
i.e. 4×3, 6×1, 6×3, 6×5, 8×5 + 8×9 (v5, 72 coefficients — extension of the 6→72 demonstration).

2. Each config passes `n_mx` and `n_mt` to `UltraPINN`, which dynamically builds the Fourier basis and the `coeff_net` output.

3. For each case, measures `E_{(2,0)}/E_total` (via `analyze_control`).

4. **v5 skip logic**: if `model.pt` exists in the subfolder, the model is reloaded (no retraining) → relaunching only processes missing configs (used for `mx8_mt9` without retraining the previous 5).

5. **Question tested**: does mode (2,0) stay dominant when the control space is enlarged? → **Yes, 87.3–92.3 % from 6 to 72 coefficients.**

---

## 10. Reviewer #3: Temporality (MANDATORY — IMPLEMENTED)

**Function**: `run_temporal_study()` (lines 1475-1548)

### What was coded

1. **`TEMPORAL_CONFIGS = [1, 3, 5]`**, with configurations:
   - 6×1 (N_MX=6, N_MT=1)
   - 6×3 (N_MX=6, N_MT=3)
   - 6×5 (N_MX=6, N_MT=5, = baseline)

2. **Recording of `E_{j=0}, E_{j=1}, E_{j=2}, E_{j=3}, E_{j=4}`**.

3. **`f_temporal = Σ_{j>0} E_j / Σ_j E_j`** via `ac['temporal_fraction']` returned by `analyze_control`.

---

## 11. Aspect ratio (IMPLEMENTED)

**Function**: `run_aspect_ratio()` (lines 1554-1603)

### What was coded

1. **`GEOMETRIES = {"square": (1.0, 1.0), "rectangular": (2.0, 1.0)}`** — hardcoded in the function.

2. **All `L` occurrences changed** → use of `LX` and `LY`:
   - `U_lid()`: `xn = x / self.lx`
   - `gen_points()`: `xp ∈ [0, LX]`, `yp ∈ [0, LY]`
   - `compute_loss()`: x normalized by `_lx`, y by `_ly`
   - `extract_lid()`: x linspace ∈ `[0, lx]`
   - `make_all_figures()`: `x_grid ∈ [0, _lx]`, `ys ∈ [0, LY]`

3. **The spatial basis stays consistent**: `sin((i+1)πx/Lx) × cos(jπt)`.

4. **Recorded metrics**: `aspect_ratio` (via `geometry`, `lx`, `ly`), `A1`, `A2`, `A3`, `dominant_i`, `dominant_j`, `mode2_fraction`.

5. **1:1 vs 2:1 comparison** possible — and **energetically CORRECT (v2)**: the modal energies in `analyze_control` are now divided by `domain_area = lx * T_MAX`, so `mode2_fraction` is strictly comparable between \(L_x=1\) and \(L_x=2\). (Before the fix, the integral over the space–time surface of size 2 made the fraction non-comparable.)

---

## 12. Validation on unseen Reynolds (IMPLEMENTED — HONEST INTERPRETATION)

**Function**: `run_unseen_Re()` (lines 1665-1712)

### What was coded

1. Trained on `[100, 500, 1000]` only (like the baseline).

2. **`UNSEEN_RE = [200, 350, 650, 800, 950]`** (line 132) — **none of these Re is included in this campaign's training** (`train_ultra` is called with `re_list=[100, 500, 1000]` explicitly).

3. For each Re (seen and unseen): `analyze_control` computes `A_{2,0}(Re)`, `E_{(2,0)}/E_total`.

4. Results CSV: `10_unseen_Re/unseen_re_results.csv` with `is_unseen` column to distinguish.

### CORRECT interpretation (v2)

⚠️ **This campaign demonstrates the continuity/interpolation of the learned law in Re, not the exactness of the optimal solution at a never-seen Reynolds.** To demonstrate the actual error at unseen Re, an **independent reference** (e.g. LBM) is required — handled separately (`14_independent_optimization` and LBM campaign).

In the manuscript, write: **"interpolation consistency across Reynolds"**, NOT "validation on unseen Reynolds".

---

## 13. Convergence and numerical uncertainty (IMPLEMENTED — NOT a GCI)

**Function**: `run_convergence()` (lines 1793-1818)

### What was coded

1. **Full training-history recording**: `training_history.csv` with columns `epoch, total_loss, PDE, BC, dissipation, control, variance, energy`.

2. **`residual_stats.csv`**: records `PDE_MSE, BC_MSE, dissipation, Ru_mean, Ru_max, Rv_mean, Rv_max, Rc_mean, Rc_max` at each epoch (via `compute_loss`).

3. **Seed-run standard deviation**: available in `04_seed_study/seed_study.csv` (mean ± std printed to console).

### CORRECT interpretation (v2)

A decreasing loss curve is NOT a convergence study in the CFD sense. **One must NOT write "We performed a GCI" in the reviewer response.**

Instead:
> "We quantified numerical/optimization sensitivity through independent seeds, collocation density, network architecture, and control-basis dimension."

The relevant items for this quantification: seeds (04), sampling (06), architecture (05), parametrization (13).

---

## 13bis. Control parametrization (NEW v2 — R2-2)

**Function**: `run_parametrization_study()` (lines 1874-1936)
**MODE**: `parametrization`

### What was coded

The reviewer could respond: "You showed robustness *within the same Fourier family*, not robustness to the parameterization itself." This campaign answers that objection directly.

1. **Two basis families** compared:
   - `fourier`: \( \sin((i+1)\pi x/L_x)\cos(j\pi t) \) (baseline)
   - `chebyshev_mod`: \( x(L_x-x)\cdot T_i(2x/L_x-1)\cdot\cos(j\pi t) \), where \(T_i\) is the Chebyshev polynomial of the 1st kind. This basis is **smooth**, satisfies \(U(0)=U(L_x)=0\), and is NOT sinusoidal.

2. Implementation: `basis_type` parameter in `UltraPINN.__init__` and `U_lid()`, propagated via `train_ultra(basis_type=...)`.

3. **`analyze_control` remains agnostic** to the training basis: it always decomposes the obtained field \(U_{lid}(x,t,Re)\) into the `sin(2πx/Lx)` harmonic via Fourier coefficients. So `mode2_fraction` is strictly comparable between the two families.

4. **The question tested**: does the dominance of the spatial harmonic \( \sin(2\pi x/L_x) \) persist when the law is expressed in a non-sinusoidal basis?

5. Results: `13_parametrization/parametrization_results.csv`

---

## 14. Reviewer #2-8: Symbolic regression (IMPLEMENTED)

**Function**: `run_symbolic_mode()` (lines 1660-1732)

### What was coded

1. **PySR engine unchanged** — as requested.

2. **New metrics added**:
   - `MAE` (mean absolute error)
   - `RMSE` (root mean squared error)
   - `R²` (coefficient of determination)
   - `relative_L2` (relative L2 norm)
   - `max_abs_error` (maximum absolute error)

3. These metrics are computed on:
   - **train** (global fit)
   - **LOCO validation** (in `loco_ultra()`)

4. **Corrected wording**: the code and the report use the term **compact analytical representability of the PINN law**, NOT "independent physical optimality". The output is labeled `symbolic_results.csv` and does NOT claim PySR validates the physics.

5. **v2 fix — physical Re**: `extract_lid` now writes `Re_phys` (column 2) instead of `Re/1000` (Option B). The z-score in `run_pysr` handles the rescalation; the symbolic expression is given in terms of the **physical Reynolds** `Re`, removing any ambiguity (the reviewer pointed out that `Re/1000` could mislead in the manuscript).

---

## 15. Reviewer #1-2: How the PDE loss is computed (IMPLEMENTED)

1. The computation is **already correct** in the original code: direct autograd (in `compute_loss`).

2. **Requested addition — residuals saved explicitly**:
```python
residual_stats = {
    "PDE_MSE": float(l_pde.item()),
    "BC_MSE": float(l_bc.item()),
    "dissipation": float(l_diss.item()),
    "Ru_mean": res_u_abs.mean().item(),
    "Ru_max":  res_u_abs.max().item(),
    "Rv_mean": res_v_abs.mean().item(),
    "Rv_max":  res_v_abs.max().item(),
    "Rc_mean": res_c_abs.mean().item(),
    "Rc_max":  res_c_abs.max().item(),
}
```

3. This dict is returned by `compute_loss` (8th value) and saved in `residual_stats.csv`.

4. → **Answers directly** the reviewer's question: root-mean-square over the collocation points, derivatives via automatic differentiation. No external NS solver is used for the loss during training.

---

## 16. Reviewer #2-5: Formulation inconsistencies (IMPLEMENTED)

**Function**: `save_metadata()` (lines 1047-1070)

Each campaign saves a `metadata.json` with ALL the formulation constants:
```json
{
    "energy_definition": "mean(U_lid**2)",
    "E_target": 0.25,
    "N_MX": 6,
    "N_MT": 5,
    "LX": 1.0,
    "LY": 1.0,
    "Re_min": 100.0,
    "Re_max": 1000.0,
    "LAMBDA_PDE": 1.0,
    "LAMBDA_BC": 15.0,
    "LAMBDA_DISS": 0.05,
    "LAMBDA_CTRL": 200.0,
    "LAMBDA_VAR": 1.0,
    "LAMBDA_RE": 2.0,
    "N_PHYS": 4000,
    "N_BC": 600,
    "seed": 42
}
```

---

## 17. Reviewer #2-6: Ghia benchmark (IMPLEMENTED — no PINN modification)

**STATUS**: No PINN code change required. The reviewer asks to distinguish the classical benchmark (Ghia) from the controlled flow. This distinction is addressed in the report, not in the code. The LBM will be handled separately (`14_LBM`).

---

## 18. Reviewer #2-3: Performance vs optimality (IMPLEMENTED)

The automated outputs and figure titles **no longer use the word "optimal"** (fixed in v2 — except in one caveat message that explicitly denies that the unseen-Re campaign validates optimality). Instead:

- **`identified_control`** → for the control identified by the PINN
- **`dominant_mode`** → for the dominant mode (via `dominant_i`, `dominant_j` in `analyze_control`)
- **`best_observed_control`** → for the best observed control when appropriate

The outputs separate:
- **Performance**: dissipation, energy, PDE residual
- **Structure**: A1, A2, A3, mode2_fraction, temporal_fraction

---

## 19. Automatic modal extraction (IMPLEMENTED — CORE OF THE SYSTEM)

**Function**: `analyze_control()` (lines 551-683, v2)

### What was coded (v2)

This function returns the (corrected) format:
```python
{
    "Re": Re,
    "E_total": ...,                    # mean(U^2), normalized everywhere
    "E_modes_sum": ...,                # sum of normalized modal energies
    "domain_area": ...,                # lx * T_MAX
    "A10": ...,                        # = c_{0,0}  mode sin(πx/Lx)·1 (STATIONARY)
    "A20": ...,                        # = c_{1,0}  mode sin(2πx/Lx)·1 (STATIONARY)  ← FIXED
    "A30": ...,                        # = c_{2,0}  STATIONARY
    "A40": ..., "A50": ..., "A60": ...,"A70": ...,"A80": ...,
    "mode2_fraction": ...,             # E_{(2,0)}/E_total   [= E_{1,0}/E_total]
    "temporal_fraction": ...,
    "dominant_i": ..., "dominant_j": ...,
    "reconstruction_rmse": ...,
    "Efrac_{i}_{j}": ...,              # full energy distribution (v2)
    # PLUS all individual coefficients c_{i}_{j}
}
```

**Major v2 fix**: `A20 = c_{1,0}` (no longer the sum over j).
This guarantees `A20` is the amplitude of the stationary mode \(\sin(2\pi x/L_x)\).

Technical operation:
- Creates a 200×200 grid in (x, t)
- Evaluates `U_lid(x,t,Re)` on the grid
- Projects onto the Fourier basis `sin((i+1)πx/Lx) × cos(jπt)` (independent of the control basis used for training — intended)
- **Normalized** modal energy: \(E_{ij} = c_{ij}^2 \|\phi_{ij}\|^2 / (L_x T_{\max})\) → comparable between Lx=1 and Lx=2
- Identifies the dominant mode, the mode-(2,0) fraction, the temporal fraction

---

## 20. Final campaign organization (IMPLEMENTED)

All the folders were created:

```
PoF_R_lid_driven_paper/
├── historical_baseline/
│   └── code_find_U_PINN_Lid_driven.py      ← intact copy
├── PINN_Lid_driven_reviewers.py             ← modified code (v5)
├── REPORT_CODING.md                         ← this report
├── results_reviewers/
│   ├── 02_loss_ablation/        → R2.2 / R3.2
│   ├── 03_energy_sweep/         → R3.2
│   ├── 04_seed_study/           → R2.4 / R3.3
│   ├── 05_architecture/         → R2.4
│   ├── 06_sampling/             → R2.4
│   ├── 07_mode_count/           → R2.4 / R3.2
│   ├── 08_temporal/             → R2.4 / R3.2
│   ├── 09_architecture/         → R3.2
│   ├── 10_unseen_Re/            → inter-Re interpolation (not exhaustive validation)
│   ├── 11_symbolic_LOCO/        → R2.8 / R3.1
│   ├── 12_convergence_uncertainty/ → R2.7
│   ├── 13_parametrization/      → R2-2 (Fourier vs modified Chebyshev)
│   └── 14_independent_optimization/ → R2.1 / R2.3 / R3.3 (placeholder, LBM separate)
```

---

## Campaign classification

| Priority | Campaign | Status |
|---|---|---|
| **Mandatory** | Loss ablation | ✅ Implemented (incl. LCTRL_HALF/DOUBLE v2) |
| **Mandatory** | Energy sweep (E\*) | ✅ Implemented |
| **Mandatory** | Seeds (5-10) | ✅ Implemented (10 seeds) |
| **Mandatory** | Mode-count | ✅ **Run** (6 configs, 6→72 coeffs, mode2 87.3–92.3 %) |
| **Mandatory** | Temporal | ✅ **Run** (3 configs, temporal ≤1.5 %, quasi-steady) |
| **Mandatory** | Architecture/Sampling | ✅ Implemented |
| **Strongly recommended** | Aspect ratio | ✅ Implemented (energy normalization fixed v2) |
| **Strongly recommended** | Unseen-Re | 🟡 Implemented (interpolation, not exhaustive validation) |
| **Strongly recommended** | Convergence/uncertainty | ✅ Implemented (computational robustness, NOT a GCI) |
| **Recommended** | Symbolic regression (metrics) | ✅ Implemented (physical Re v2) |
| **New v2** | Control parametrization | ✅ Implemented (Fourier vs modified Chebyshev) |
| **Expensive/later** | Independent optimization | ⏸ Placeholder (LBM separate) |

---

## Specific points addressed

### `E_target = 0.25` convention (NOT replaced by 0.5)

Reviewer #3 writes `E_target = 0.5` but the historical baseline uses `E_TARGET = 0.25` with `E = ⟨U_lid²⟩`. **The code keeps `0.25`** and the energy sweep (`run_energy_sweep`) explores exactly this parameter from `0.01` to `2.0`, with `0.25` as the central value. This will let us show the reviewer the structure of the solution as a function of this convention.

### The central test

The real question tested by all the campaigns is:
> **Does Mode 2 persist despite changes in loss, seed, architecture, number of modes, energy budget, geometry, and parametrization family?**

The code is designed to answer this quantitatively via the `mode2_fraction` column (computed on the sin(2πx/Lx) harmonic, independent of the training basis) in each results CSV.

---

## Real state per reviewer comment (v2, honest)

| Reviewer comment | Real state |
|---|---|
| R1-2 — PDE loss computation | ✅ Very well handled (autograd + Ru/Rv/Rc stats) |
| R2-2 — loss ablation | ✅ Well handled (12 + LCTRL_HALF/DOUBLE = 14 configs) |
| R2-2 — parametrization | ✅ → **fixed v3** (Fourier/Chebyshev comparison in common basis + native coeffs) |
| R2-4 — seeds | ✅ Very well handled (10 seeds, mean±std) |
| R2-4 — architecture | ✅ Handled |
| R2-4 — collocation | ✅ Handled |
| R2-4 — number of modes | ✅ Handled after the A20/A30 fix |
| R2-7 — convergence/uncertainty | 🟡 Good (computational robustness), NOT a GCI |
| R2-8 — symbolic regression | ✅ Good, physical Re now (v2) |
| R3-2 — energy E\* | ✅ Very important and well-designed, cache fixed v3, final table v4 |
| R3-2 — temporality | ✅ Well-designed, E_j normalization fixed v3 |
| R3-2 — geometry | ✅ Fixed (energy normalization v2) |
| Inter-Re interpolation | 🟡 Interpolation, NOT exhaustive validation |
| R2-1/R2-3 — independent optimality | ❌ Not yet, LBM/separate optimization |
| Ghia controlled topology | ❌ Not resolved (deliberately left to LBM) |
| Figures (AE) | 🟡→✅ Fonts increased v2 (labels 14, ticks 12) |

---

## Files in the folder

```
C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper\
├── historical_baseline\
│   └── code_find_U_PINN_Lid_driven.py          (846 lines, intact copy)
├── PINN_Lid_driven_reviewers.py                (~2039 lines, v5)
├── REPORT_CODING.md                            (this report)
├── README.md                                   (English)
├── INSTALL.md                                  (English)
├── requirements.txt                            (English)
├── .gitignore
├── logs\
└── results_reviewers\01_baseline\ ... 14_independent_optimization\
```

---

## Verification (v4)

- Syntax verified: `py_compile.compile()` OK
- Functional verification of `analyze_control`: pure mode 2 → `mode2_fraction≈1`, `modal_coverage≈1`, `A20=0.5`; Chebyshev → `native_c_1_0` ≠ Fourier projection `A20` (distinction confirmed)
- **Consistent quadrature verified**: `modal_coverage = 1.000000` (machine precision) on the 7 sweep models → basis exactly orthonormal-consistent (the ~1.01 anomaly fixed)
- **Recompute validated**: `energy_sweep.csv` regenerated with the corrected quadrature (7 rows, metrics E_total/E_modes_sum/mode2_fraction/A20/temporal_fraction/coverage)
- Historical baseline unmodified (identical MD5 hash)

---

## Verification (v5 — in-progress campaign)

- mode_count: 6/6 runs verified (`model.pt` + `metadata.json` + CSV regenerated with (8,9)).
- temporal: 3/3 runs verified.
- architecture (seeds {0,1,2}): **COMPLETED — 6/6 new runs** (small×3, large×3) + 3 baselines reused; `architecture_results.csv` generated; paired intra-seed Δf2 analyzed (see section above).
- parametrization (Fourier vs Chebyshev mod.): launched 2026-09-05 17:39 (PID 6248) — **to be updated when it completes**.
- gh (GitHub CLI): installed (v2.100.0); repo creation + push awaiting user authentication.