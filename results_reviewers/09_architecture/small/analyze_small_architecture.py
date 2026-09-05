# =============================================================================
# analyze_small_architecture.py
# Analyse des checkpoints SMALL — seeds 0, 1, 2
# =============================================================================
#
# Aucun réentraînement.
#
# Les model.pt contiennent directement le state_dict.
# On reconstruit UltraPINN avec ARCHITECTURES["small"] du script principal,
# puis on applique exactement analyze_control().
# =============================================================================

from pathlib import Path
import json
import csv
import sys
import traceback

import numpy as np
import pandas as pd
import torch


# =============================================================================
# 1. CHEMINS
# =============================================================================

ROOT = Path(
    r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"
)

MAIN_SCRIPT = ROOT / "PINN_Lid_driven_reviewers.py"

RESULTS_DIR = (
    ROOT
    / "results_reviewers"
    / "09_architecture"
    / "small"
)

OUTPUT_DIR = RESULTS_DIR / "postprocess"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    RESULTS_DIR
    / "small_architecture_modal_summary.csv"
)

SEEDS = [0, 1, 2]

DEVICE = torch.device("cpu")

print("Device :", DEVICE)


# =============================================================================
# 2. IMPORT DU SCRIPT PRINCIPAL
# =============================================================================

if not MAIN_SCRIPT.exists():
    raise FileNotFoundError(
        f"Script principal introuvable :\n{MAIN_SCRIPT}"
    )

sys.path.insert(0, str(MAIN_SCRIPT.parent))

print("=" * 90)
print("IMPORT DU SCRIPT PRINCIPAL")
print("=" * 90)

try:
    main = __import__(MAIN_SCRIPT.stem)
except Exception:
    traceback.print_exc()
    raise


# =============================================================================
# 3. RÉCUPÉRER CLASSE + CONFIGURATION EXACTES
# =============================================================================

UltraPINN = main.UltraPINN
ARCHITECTURES = main.ARCHITECTURES
analyze_control = main.analyze_control

print()
print("UltraPINN :", UltraPINN)

print(
    "Signature UltraPINN :",
    UltraPINN.__init__.__signature__
    if hasattr(UltraPINN.__init__, "__signature__")
    else "voir définition du script"
)

print()
print("ARCHITECTURES disponibles :")
for name, cfg in ARCHITECTURES.items():
    print(f"  {name}: {cfg}")


# =============================================================================
# 4. VÉRIFICATION CONFIG SMALL
# =============================================================================

if "small" not in ARCHITECTURES:
    raise RuntimeError(
        "ARCHITECTURES['small'] n'existe pas."
    )

SMALL_CONFIG = ARCHITECTURES["small"]

EXPECTED_SMALL = {
    "net": [64, 64, 32],
    "coeff": [32, 32],
}

if SMALL_CONFIG != EXPECTED_SMALL:

    print()
    print("ATTENTION :")
    print("Configuration small dans le script :")
    print(SMALL_CONFIG)

    print("Configuration attendue d'après le checkpoint :")
    print(EXPECTED_SMALL)


print()
print("=" * 90)
print("CONFIGURATION SMALL")
print("=" * 90)
print(SMALL_CONFIG)


# =============================================================================
# 5. METADATA
# =============================================================================

def load_metadata(seed_dir):

    path = seed_dir / "metadata.json"

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =============================================================================
# 6. CHECKPOINT
# =============================================================================

def load_state_dict(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Checkpoint inattendu : {type(checkpoint)}"
        )

    return checkpoint


# =============================================================================
# 7. VÉRIFICATION DU CHECKPOINT
# =============================================================================

def inspect_checkpoint(state_dict):

    print()
    print("Checkpoint :")
    print("  type      =", type(state_dict))
    print("  n tensors =", len(state_dict))

    # Dimensions du réseau physique
    s0 = tuple(
        state_dict["net.0.0.weight"].shape
    )

    s1 = tuple(
        state_dict["net.1.0.weight"].shape
    )

    s2 = tuple(
        state_dict["net.2.0.weight"].shape
    )

    s3 = tuple(
        state_dict["net.3.weight"].shape
    )

    # Dimensions réseau coefficients
    c0 = tuple(
        state_dict["coeff_net.0.0.weight"].shape
    )

    c1 = tuple(
        state_dict["coeff_net.1.0.weight"].shape
    )

    c2 = tuple(
        state_dict["coeff_net.2.weight"].shape
    )

    print()
    print("State network:")
    print("  net.0.0.weight:", s0)
    print("  net.1.0.weight:", s1)
    print("  net.2.0.weight:", s2)
    print("  net.3.weight  :", s3)

    print()
    print("Coefficient network:")
    print("  coeff_net.0.0.weight:", c0)
    print("  coeff_net.1.0.weight:", c1)
    print("  coeff_net.2.weight  :", c2)

    # Vérification exacte
    expected_shapes = {
        "net.0.0.weight": (64, 4),
        "net.0.0.bias": (64,),

        "net.1.0.weight": (64, 64),
        "net.1.0.bias": (64,),

        "net.2.0.weight": (32, 64),
        "net.2.0.bias": (32,),

        "net.3.weight": (3, 32),
        "net.3.bias": (3,),

        "coeff_net.0.0.weight": (32, 1),
        "coeff_net.0.0.bias": (32,),

        "coeff_net.1.0.weight": (32, 32),
        "coeff_net.1.0.bias": (32,),

        "coeff_net.2.weight": (30, 32),
        "coeff_net.2.bias": (30,),
    }

    for key, expected_shape in expected_shapes.items():

        actual_shape = tuple(
            state_dict[key].shape
        )

        if actual_shape != expected_shape:

            raise RuntimeError(
                f"\nShape incorrecte pour {key} :\n"
                f"  attendu : {expected_shape}\n"
                f"  obtenu  : {actual_shape}"
            )

    print()
    print(
        "CHECKPOINT = ARCHITECTURE SMALL CONFIRMÉE"
    )


# =============================================================================
# 8. CONSTRUCTION EXACTE DU MODÈLE
# =============================================================================

def build_small_model(metadata, state_dict):

    inspect_checkpoint(
        state_dict
    )

    n_mx = int(
        metadata["N_MX"]
    )

    n_mt = int(
        metadata["N_MT"]
    )

    lx = float(
        metadata["LX"]
    )

    ly = float(
        metadata["LY"]
    )

    basis_type = "fourier"

    print()
    print("=" * 90)
    print("CONSTRUCTION DU MODÈLE SMALL")
    print("=" * 90)

    print("arch_config =", SMALL_CONFIG)
    print("n_mx        =", n_mx)
    print("n_mt        =", n_mt)
    print("lx          =", lx)
    print("ly          =", ly)
    print("basis_type  =", basis_type)

    # -------------------------------------------------------------------------
    # CONSTRUCTION EXACTE
    # -------------------------------------------------------------------------

    model = UltraPINN(
        arch_config=SMALL_CONFIG,
        n_mx=n_mx,
        n_mt=n_mt,
        lx=lx,
        ly=ly,
        basis_type=basis_type
    )

    model = model.to(
        device=DEVICE,
        dtype=torch.float64
    )

    # -------------------------------------------------------------------------
    # CHARGEMENT STRICT
    # -------------------------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print()
    print(
        "STATE_DICT CHARGÉ AVEC SUCCÈS"
    )

    print(
        "strict=True"
    )

    return model


# =============================================================================
# 9. TRAINING HISTORY
# =============================================================================

def read_training_history(path):

    df = pd.read_csv(
        path
    )

    if len(df) == 0:
        return {}

    last = df.iloc[-1]

    out = {}

    for col in df.columns:

        try:
            out[
                f"history_{col}"
            ] = float(last[col])

        except Exception:

            out[
                f"history_{col}"
            ] = str(last[col])

    return out


# =============================================================================
# 10. ANALYSE MODALE
# =============================================================================

def run_analysis(model, metadata):

    Re = 500

    lx = float(
        metadata["LX"]
    )

    ly = float(
        metadata["LY"]
    )

    print()
    print("=" * 90)
    print("ANALYZE_CONTROL")
    print("=" * 90)

    print(
        f"Re={Re}"
    )

    print(
        f"lx={lx}, ly={ly}"
    )

    result = analyze_control(
        model,
        Re,
        lx=lx,
        ly=ly,
        nx=200,
        nt=200
    )

    return result


# =============================================================================
# 11. RESULTAT -> DICT
# =============================================================================

def result_to_dict(result):

    if result is None:
        return {}

    if not isinstance(
        result,
        dict
    ):

        raise RuntimeError(
            "analyze_control() n'a pas retourné un dictionnaire."
        )

    out = {}

    for key, value in result.items():

        if torch.is_tensor(value):

            if value.numel() == 1:

                out[key] = float(
                    value.detach()
                    .cpu()
                    .item()
                )

            else:

                out[key] = str(
                    value.detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

        elif isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
                str,
                bool
            )
        ):

            if isinstance(
                value,
                (
                    np.integer,
                    np.floating
                )
            ):
                out[key] = float(value)
            else:
                out[key] = value

        else:

            try:
                out[key] = float(value)
            except Exception:
                out[key] = str(value)

    return out


# =============================================================================
# 12. TRAITEMENT DES SEEDS
# =============================================================================

all_results = []


for seed in SEEDS:

    print()
    print()
    print("#" * 100)
    print(
        f"# SEED {seed}"
    )
    print("#" * 100)

    seed_dir = (
        RESULTS_DIR
        / f"seed_{seed}"
    )

    metadata_file = (
        seed_dir
        / "metadata.json"
    )

    model_file = (
        seed_dir
        / "model.pt"
    )

    history_file = (
        seed_dir
        / "training_history.csv"
    )

    # -------------------------------------------------------------------------
    # fichiers
    # -------------------------------------------------------------------------

    for path in [
        metadata_file,
        model_file,
        history_file
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"\nFichier absent :\n{path}"
            )

    # -------------------------------------------------------------------------
    # metadata
    # -------------------------------------------------------------------------

    metadata = load_metadata(
        seed_dir
    )

    print()
    print("Metadata:")
    print(
        "  arch      =",
        metadata.get("arch")
    )
    print(
        "  seed      =",
        metadata.get("seed")
    )
    print(
        "  N_MX      =",
        metadata.get("N_MX")
    )
    print(
        "  N_MT      =",
        metadata.get("N_MT")
    )
    print(
        "  LX        =",
        metadata.get("LX")
    )
    print(
        "  LY        =",
        metadata.get("LY")
    )
    print(
        "  E_target  =",
        metadata.get("E_target")
    )

    # -------------------------------------------------------------------------
    # checkpoint
    # -------------------------------------------------------------------------

    state_dict = load_state_dict(
        model_file
    )

    # -------------------------------------------------------------------------
    # modèle
    # -------------------------------------------------------------------------

    model = build_small_model(
        metadata,
        state_dict
    )

    # -------------------------------------------------------------------------
    # historique
    # -------------------------------------------------------------------------

    history = read_training_history(
        history_file
    )

    # -------------------------------------------------------------------------
    # modal
    # -------------------------------------------------------------------------

    modal_result = run_analysis(
        model,
        metadata
    )

    modal_result = result_to_dict(
        modal_result
    )

    # -------------------------------------------------------------------------
    # assemblage
    # -------------------------------------------------------------------------

    row = {}

    row["architecture"] = "small"
    row["seed"] = seed
    row["Re"] = 500

    # architecture
    row["net_arch"] = "4-64-64-32-3"
    row["coeff_arch"] = "1-32-32-30"

    # training final
    row.update(
        history
    )

    # modal
    row.update(
        modal_result
    )

    # -------------------------------------------------------------------------
    # nombre de paramètres
    # -------------------------------------------------------------------------

    row["n_params"] = sum(
        p.numel()
        for p in model.parameters()
    )

    # -------------------------------------------------------------------------
    # fichiers
    # -------------------------------------------------------------------------

    row["model_file"] = str(
        model_file
    )

    row["metadata_file"] = str(
        metadata_file
    )

    row["history_file"] = str(
        history_file
    )

    # -------------------------------------------------------------------------
    # sauvegarde seed
    # -------------------------------------------------------------------------

    json_file = (
        OUTPUT_DIR
        / f"seed_{seed}_modal_results.json"
    )

    with open(
        json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            row,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "JSON sauvegardé :"
    )
    print(
        json_file
    )

    all_results.append(
        row
    )


# =============================================================================
# 13. CSV GLOBAL
# =============================================================================

print()
print("=" * 90)
print("CRÉATION DU CSV GLOBAL")
print("=" * 90)

df = pd.DataFrame(
    all_results
)

# Colonnes importantes au début
priority = [
    "architecture",
    "seed",
    "Re",
    "net_arch",
    "coeff_arch",
    "n_params",

    "history_epoch",
    "history_total",
    "history_pde",
    "history_bc",
    "history_diss",
    "history_ctrl",
    "history_energy",
    "history_var",

    "E_total",
    "mode2_fraction",
    "fourier_mode2_fraction",
    "temporal_fraction",

    "A10",
    "A20",
    "A30",
    "A40",
    "A50",
    "A60",

    "dominant_i",
    "dominant_j",

    "modal_coverage",
    "reconstruction_rmse",
]

ordered = []

for col in priority:

    if col in df.columns:
        ordered.append(col)

for col in df.columns:

    if col not in ordered:
        ordered.append(col)

df = df[
    ordered
]

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =============================================================================
# 14. RÉSUMÉ
# =============================================================================

print()
print()
print("=" * 100)
print("RÉSUMÉ SMALL — SEEDS 0,1,2")
print("=" * 100)

for _, row in df.iterrows():

    print()
    print(
        f"SEED {int(row['seed'])}"
    )

    keys = [
        "history_energy",
        "history_var",
        "E_total",
        "mode2_fraction",
        "temporal_fraction",
        "A10",
        "A20",
        "A30",
        "dominant_i",
        "dominant_j",
        "modal_coverage",
        "reconstruction_rmse",
    ]

    for key in keys:

        if key in row.index:

            print(
                f"  {key:25s} = {row[key]}"
            )


# =============================================================================
# 15. STATISTIQUES DES 3 SEEDS
# =============================================================================

print()
print("=" * 100)
print("STATISTIQUES SMALL")
print("=" * 100)

for key in [
    "E_total",
    "mode2_fraction",
    "temporal_fraction",
    "A20",
]:

    if key not in df.columns:
        continue

    values = pd.to_numeric(
        df[key],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    print(
        f"{key:25s} "
        f"mean={values.mean():.8f}  "
        f"std={values.std(ddof=1):.8f}  "
        f"min={values.min():.8f}  "
        f"max={values.max():.8f}"
    )


print()
print("=" * 100)
print("TERMINE")
print("=" * 100)

print()
print(
    "CSV final :"
)

print(
    OUTPUT_CSV
)

print()
print(
    "JSON individuels :"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Aucun modèle n'a été réentraîné."
)

print("=" * 100)