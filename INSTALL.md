# Installation (verified on Windows 11, Python 3.14)

## Verified environment

| Component | Tested version |
|---|---|
| OS | Windows 11 (x64) |
| Python | 3.14.7 |
| numpy | 2.5.2 |
| torch | 2.13.0+cpu |
| matplotlib | 3.11.1 |
| pandas | 3.0.5 |
| pysr | 2.1.0 |

## 1. Create an environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

## 2. Install dependencies

CPU build recommended (no GPU required — the calculations were calibrated on CPU):

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 3. Julia (only for the `symbolic` campaign)

The `symbolic` mode uses PySR, which runs Julia. The main campaigns
(`loss_ablation`, `energy_sweep`, `seed_study`, `architecture`, `mode_count`,
`temporal`, `parametrization`, `aspect_ratio`, …) do **not** need it.

```bash
curl -fsSL https://install.julialang.org | sh      # Windows: official .exe
python -c "from pysr import install; install()"      # installs PySR's Julia runtime
```

> Note: at first import, PySR/juliacall prints a `UserWarning`
> ("torch was imported before juliacall…"). It is **harmless** for the main
> campaigns: the computation remains correct and reproducible.

## 4. Windows console (encoding)

The script forces UTF-8 output itself (`sys.stdout.reconfigure`) — no action
required, including on a default `cp1252` console.

## 5. Launch

```bash
python PINN_Lid_driven_reviewers.py seed_study
```

Before launching a campaign, fix the number of threads (5–6 per campaign):

```bash
set OMP_NUM_THREADS=5
set MKL_NUM_THREADS=5
```

If several campaigns run in parallel (16 logical cores → 2 campaigns at 5–6 threads), reduce the threads of each (e.g., 5) to avoid contention.

Default output: `results_reviewers/`. Overridable:

```bash
set OUT_DIR=path\to\results   # Linux/macOS: OUT_DIR=... python ...
```