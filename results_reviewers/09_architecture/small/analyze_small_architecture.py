from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import sys

ROOT = Path(
    r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"
)

sys.path.insert(0, str(ROOT))

import PINN_Lid_driven_reviewers as main

DEVICE = torch.device("cpu")
DTYPE = torch.float64

BASE = (
    ROOT
    / "results_reviewers"
    / "09_architecture"
    / "large"
    / "seed_1"
)

MODEL_FILE = BASE / "model.pt"
META_FILE = BASE / "metadata.json"

# ------------------------------------------------------------
# metadata
# ------------------------------------------------------------

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("=" * 90)
print("LARGE / SEED 2 — MODAL POST-PROCESSING")
print("=" * 90)

print("Metadata:", metadata)

# ------------------------------------------------------------
# reconstruction exacte LARGE
# ------------------------------------------------------------

model = main.UltraPINN(
    arch_config=main.ARCHITECTURES["large"],
    n_mx=int(metadata["N_MX"]),
    n_mt=int(metadata["N_MT"]),
    lx=float(metadata["LX"]),
    ly=float(metadata["LY"]),
    basis_type="fourier"
).to(
    device=DEVICE,
    dtype=DTYPE
)

# ------------------------------------------------------------
# load checkpoint
# ------------------------------------------------------------

state_dict = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

print()
print("Checkpoint type:", type(state_dict))
print("Nombre tensors:", len(state_dict))

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print("STATE_DICT LARGE CHARGÉ — strict=True")

# ------------------------------------------------------------
# modal analysis
# ------------------------------------------------------------

result = main.analyze_control(
    model,
    500,
    lx=float(metadata["LX"]),
    ly=float(metadata["LY"]),
    nx=200,
    nt=200
)

print()
print("=" * 90)
print("RÉSULTATS LARGE / SEED 1")
print("=" * 90)

for key in [
    "E_total",
    "A10",
    "A20",
    "A30",
    "mode2_fraction",
    "fourier_mode2_fraction",
    "temporal_fraction",
    "dominant_i",
    "dominant_j",
    "modal_coverage",
    "reconstruction_rmse"
]:
    if key in result:
        print(
            f"{key:25s} = {result[key]}"
        )

# ------------------------------------------------------------
# sauvegarde
# ------------------------------------------------------------

out_json = BASE / "modal_results_seed_1.json"

with open(
    out_json,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        result,
        f,
        indent=2
    )

# CSV
out_csv = BASE / "modal_results_seed_1.csv"

pd.DataFrame([result]).to_csv(
    out_csv,
    index=False
)

print()
print("JSON :", out_json)
print("CSV  :", out_csv)
print("=" * 90)