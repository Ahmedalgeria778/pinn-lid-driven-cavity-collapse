# -*- coding: utf-8 -*-
"""
================================================================================
COMPARATIVE POST-PROCESSING
Fourier vs Chebyshev_mod
Re = 100, 500, 1000

PoF reviewer campaign:
13_parametrization

Ce script :
    1) charge metadata.json de chaque run
    2) reconstruit exactement UltraPINN
    3) charge model.pt
    4) applique analyze_control()
    5) lit training_history.csv
    6) extrait les métriques finales
    7) compare Fourier vs Chebyshev_mod
    8) calcule des différences relatives
    9) produit :
         - parametrization_comparison.csv
         - parametrization_comparison.json
         - training_summary.csv
         - parametrization_comparison_table.csv
         - éventuellement des figures PNG

IMPORTANT
---------
Le script suppose que :
    PINN_Lid_driven_reviewers.py

se trouve dans BASE_DIR.
================================================================================
"""

import os
import sys
import json
import csv
import math
import traceback

import numpy as np
import torch

# =============================================================================
# 1. BASE DIRECTORY
# =============================================================================

BASE_DIR = r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results_reviewers",
    "13_parametrization"
)

# =============================================================================
# 2. IMPORT DU MODELE
# =============================================================================

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from PINN_Lid_driven_reviewers import UltraPINN, analyze_control
except Exception:
    print("\nERREUR : impossible d'importer PINN_Lid_driven_reviewers.py")
    print("Chemin recherché :")
    print(os.path.join(BASE_DIR, "PINN_Lid_driven_reviewers.py"))
    traceback.print_exc()
    raise SystemExit(1)

# =============================================================================
# 3. DEFINITIONS
# =============================================================================

RUNS = [
    {
        "parametrization": "fourier",
        "Re": 100,
        "directory": os.path.join(
            RESULTS_DIR, "fourier", "re_100"
        ),
    },
    {
        "parametrization": "fourier",
        "Re": 500,
        "directory": os.path.join(
            RESULTS_DIR, "fourier", "re_500"
        ),
    },
    {
        "parametrization": "fourier",
        "Re": 1000,
        "directory": os.path.join(
            RESULTS_DIR, "fourier", "re_1000"
        ),
    },
    {
        "parametrization": "chebyshev_mod",
        "Re": 100,
        "directory": os.path.join(
            RESULTS_DIR, "chebyshev_mod", "re_100"
        ),
    },
    {
        "parametrization": "chebyshev_mod",
        "Re": 500,
        "directory": os.path.join(
            RESULTS_DIR, "chebyshev_mod", "re_500"
        ),
    },
    {
        "parametrization": "chebyshev_mod",
        "Re": 1000,
        "directory": os.path.join(
            RESULTS_DIR, "chebyshev_mod", "re_1000"
        ),
    },
]

# =============================================================================
# 4. ARCHITECTURES
# =============================================================================

ARCHITECTURES = {
    "small": {
        "net": [64, 64, 32],
        "coeff": [32, 32],
    },

    "baseline": {
        "net": [128, 128, 96, 64],
        "coeff": [64, 64, 48],
    },

    "large": {
        "net": [256, 256, 128, 96],
        "coeff": [128, 128, 64],
    },
}

# =============================================================================
# 5. UTILITAIRES
# =============================================================================

def safe_float(value, default=np.nan):
    """Conversion robuste vers float."""
    try:
        return float(value)
    except Exception:
        return default


def rel_difference(a, b):
    """
    Différence relative en % :
        100 * |a-b| / |b|

    Si b=0, retourne NaN.
    """
    a = safe_float(a)
    b = safe_float(b)

    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan

    if abs(b) < 1e-15:
        return np.nan

    return 100.0 * abs(a - b) / abs(b)


def signed_difference(a, b):
    """a-b si les deux valeurs sont disponibles."""
    a = safe_float(a)
    b = safe_float(b)

    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan

    return a - b


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_last_numeric_row(csv_path):
    """
    Lit training_history.csv et retourne la dernière ligne numérique.

    Le format attendu historiquement est :

    L_total,
    L_PDE,
    L_BC,
    L_diss,
    L_ctrl,
    energy,
    var,
    epoch
    """
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            try:
                values = [float(x.strip()) for x in row]
                rows.append(values)
            except Exception:
                # Ignore header / texte
                continue

    if not rows:
        return None

    return rows[-1]


def count_parameters(model):
    """Nombre total de paramètres du modèle."""
    return sum(
        p.numel()
        for p in model.parameters()
    )


def build_model(metadata):
    """
    Reconstruit UltraPINN à partir des metadata.
    """

    architecture = metadata.get(
        "architecture",
        "baseline"
    )

    if architecture not in ARCHITECTURES:
        print(
            f"ATTENTION : architecture inconnue '{architecture}'. "
            f"Utilisation de baseline."
        )
        architecture = "baseline"

    arch_config = ARCHITECTURES[architecture]

    n_mx = int(
        metadata.get("n_mx", 6)
    )

    n_mt = int(
        metadata.get("n_mt", 5)
    )

    Lx = safe_float(
        metadata.get("Lx", 1.0),
        1.0
    )

    Ly = safe_float(
        metadata.get("Ly", 1.0),
        1.0
    )

    basis_type = metadata.get(
        "basis_type",
        metadata.get("basis", "fourier")
    )

    model = UltraPINN(
        arch_config=arch_config,
        n_mx=n_mx,
        n_mt=n_mt,
        lx=Lx,
        ly=Ly,
        basis_type=basis_type,
    )

    return (
        model,
        architecture,
        n_mx,
        n_mt,
        Lx,
        Ly,
        basis_type,
    )


def load_checkpoint(model, model_path):
    """
    Charge :
        - state_dict direct
        - checkpoint["state_dict"]
        - checkpoint["model_state_dict"]
    """

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise TypeError(
            f"Checkpoint non supporté : {type(checkpoint)}"
        )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    return missing, unexpected


# =============================================================================
# 6. ANALYSE D'UN RUN
# =============================================================================

def analyze_run(run_info):

    parametrization = run_info["parametrization"]
    requested_Re = float(run_info["Re"])
    run_dir = run_info["directory"]

    print("\n")
    print("=" * 100)
    print(
        f"RUN : {parametrization.upper()}   |   Re = {requested_Re:g}"
    )
    print("=" * 100)

    metadata_path = os.path.join(
        run_dir,
        "metadata.json"
    )

    model_path = os.path.join(
        run_dir,
        "model.pt"
    )

    history_path = os.path.join(
        run_dir,
        "training_history.csv"
    )

    # -------------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------------

    missing_files = []

    for path in [
        metadata_path,
        model_path,
        history_path,
    ]:
        if not os.path.isfile(path):
            missing_files.append(path)

    if missing_files:
        print("\nERREUR : fichiers manquants :")
        for path in missing_files:
            print("   ", path)

        return {
            "parametrization": parametrization,
            "Re": requested_Re,
            "status": "MISSING_FILES",
            "directory": run_dir,
        }

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = read_json(
        metadata_path
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    (
        model,
        architecture,
        n_mx,
        n_mt,
        Lx,
        Ly,
        basis_type,
    ) = build_model(metadata)

    missing, unexpected = load_checkpoint(
        model,
        model_path
    )

    model.eval()

    checkpoint_ok = (
        len(missing) == 0
        and len(unexpected) == 0
    )

    print("\nPARAMETERS")
    print("-" * 100)
    print(f"parametrization = {parametrization}")
    print(f"architecture    = {architecture}")
    print(f"n_mx            = {n_mx}")
    print(f"n_mt            = {n_mt}")
    print(f"Lx              = {Lx}")
    print(f"Ly              = {Ly}")
    print(f"Re              = {requested_Re}")
    print(f"basis_type      = {basis_type}")
    print(f"parameters      = {count_parameters(model):,}")
    print(f"checkpoint      = {'OK' if checkpoint_ok else 'WARNING'}")

    if missing:
        print("Missing keys:")
        for x in missing:
            print("   ", x)

    if unexpected:
        print("Unexpected keys:")
        for x in unexpected:
            print("   ", x)

    # -------------------------------------------------------------------------
    # Reynolds metadata
    # -------------------------------------------------------------------------

    metadata_Re = safe_float(
        metadata.get(
            "Re",
            metadata.get(
                "re",
                requested_Re
            )
        ),
        requested_Re
    )

    # -------------------------------------------------------------------------
    # MODAL ANALYSIS
    # -------------------------------------------------------------------------

    print("\nAnalyse modale...")

    modal = analyze_control(
        model,
        Re=metadata_Re,
        lx=Lx,
        ly=Ly,
        nx=200,
        nt=200,
    )

    # -------------------------------------------------------------------------
    # TRAINING HISTORY
    # -------------------------------------------------------------------------

    last_row = read_last_numeric_row(
        history_path
    )

    history = {}

    if last_row is not None:

        # Format historique :
        # L_total, PDE, BC, diss, ctrl, E, var, epoch

        if len(last_row) >= 8:

            history["L_total_final"] = last_row[0]
            history["L_PDE_final"] = last_row[1]
            history["L_BC_final"] = last_row[2]
            history["L_diss_final"] = last_row[3]
            history["L_ctrl_final"] = last_row[4]
            history["energy_MC_final"] = last_row[5]
            history["L_var_final"] = last_row[6]
            history["epoch_final"] = int(round(last_row[7]))

        else:

            print(
                "ATTENTION : training_history.csv "
                "ne possède pas au moins 8 colonnes numériques."
            )

    # -------------------------------------------------------------------------
    # Base record
    # -------------------------------------------------------------------------

    record = {
        "status": "OK",

        "parametrization": parametrization,
        "basis_type": basis_type,

        "Re": metadata_Re,

        "architecture": architecture,

        "n_mx": n_mx,
        "n_mt": n_mt,

        "Lx": Lx,
        "Ly": Ly,

        "n_parameters": count_parameters(model),

        "checkpoint_strict": checkpoint_ok,

        "energy_definition":
            metadata.get(
                "energy_definition",
                ""
            ),

        "E_target":
            safe_float(
                metadata.get(
                    "E_target",
                    np.nan
                )
            ),

        # ---------------------------------------------------------------------
        # Modal results
        # ---------------------------------------------------------------------

        "E_total":
            safe_float(
                modal.get("E_total")
            ),

        "E_modes_sum":
            safe_float(
                modal.get("E_modes_sum")
            ),

        "modal_coverage":
            safe_float(
                modal.get("modal_coverage")
            ),

        "reconstruction_rmse":
            safe_float(
                modal.get("reconstruction_rmse")
            ),

        "A10":
            safe_float(
                modal.get("A10")
            ),

        "A20":
            safe_float(
                modal.get("A20")
            ),

        "A30":
            safe_float(
                modal.get("A30")
            ),

        "A40":
            safe_float(
                modal.get("A40")
            ),

        "A50":
            safe_float(
                modal.get("A50")
            ),

        "A60":
            safe_float(
                modal.get("A60")
            ),

        "mode2_fraction":
            safe_float(
                modal.get("mode2_fraction")
            ),

        "fourier_mode2_fraction":
            safe_float(
                modal.get("fourier_mode2_fraction")
            ),

        "temporal_fraction":
            safe_float(
                modal.get("temporal_fraction")
            ),

        "dominant_i":
            modal.get("dominant_i", None),

        "dominant_j":
            modal.get("dominant_j", None),

        # ---------------------------------------------------------------------
        # File paths
        # ---------------------------------------------------------------------

        "metadata_path": metadata_path,
        "model_path": model_path,
        "history_path": history_path,
    }

    # Add training history
    record.update(history)

    # -------------------------------------------------------------------------
    # Branche dominante lisible
    # -------------------------------------------------------------------------

    di = record["dominant_i"]
    dj = record["dominant_j"]

    if di == 0 and dj == 0:
        record["dominant_branch"] = "spatial_mode_1"
    elif di == 1 and dj == 0:
        record["dominant_branch"] = "spatial_mode_2"
    elif di == 2 and dj == 0:
        record["dominant_branch"] = "spatial_mode_3"
    elif di == 0 and dj == 1:
        record["dominant_branch"] = "temporal_mode_1"
    elif di is not None and dj is not None:
        record["dominant_branch"] = (
            f"i={di},j={dj}"
        )
    else:
        record["dominant_branch"] = "unknown"

    # -------------------------------------------------------------------------
    # Affichage principal
    # -------------------------------------------------------------------------

    print("\nRESULTATS PRINCIPAUX")
    print("-" * 100)

    main_keys = [
        "E_total",
        "E_modes_sum",
        "A10",
        "A20",
        "A30",
        "A40",
        "A50",
        "A60",
        "mode2_fraction",
        "temporal_fraction",
        "dominant_branch",
        "modal_coverage",
        "reconstruction_rmse",
    ]

    for key in main_keys:
        print(
            f"{key:28s} = {record.get(key)}"
        )

    print("\nTRAINING FINAL")
    print("-" * 100)

    for key in [
        "epoch_final",
        "L_total_final",
        "L_PDE_final",
        "L_BC_final",
        "L_diss_final",
        "L_ctrl_final",
        "energy_MC_final",
        "L_var_final",
    ]:
        if key in record:
            print(
                f"{key:28s} = {record[key]}"
            )

    return record


# =============================================================================
# 7. EXECUTION DES 6 RUNS
# =============================================================================

all_results = []

for run in RUNS:
    result = analyze_run(run)
    all_results.append(result)


# =============================================================================
# 8. CONSOLIDATION
# =============================================================================

print("\n\n")
print("=" * 100)
print("CONSOLIDATION DES 6 CAS")
print("=" * 100)

valid = [
    r for r in all_results
    if r.get("status") == "OK"
]

print(
    f"\nRuns valides : {len(valid)} / {len(all_results)}"
)

# Trier
valid.sort(
    key=lambda r: (
        r["parametrization"],
        r["Re"]
    )
)


# =============================================================================
# 9. TABLEAU PRINCIPAL
# =============================================================================

columns_main = [
    "parametrization",
    "Re",
    "E_total",
    "energy_MC_final",
    "A20",
    "mode2_fraction",
    "temporal_fraction",
    "dominant_branch",
    "modal_coverage",
    "reconstruction_rmse",
    "L_total_final",
    "L_PDE_final",
    "epoch_final",
]

main_csv = os.path.join(
    RESULTS_DIR,
    "parametrization_comparison.csv"
)

with open(
    main_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns_main
    )

    writer.writeheader()

    for r in valid:
        writer.writerow(
            {
                k: r.get(k, "")
                for k in columns_main
            }
        )


# =============================================================================
# 10. TABLEAU FOURIER VS CHEBYSHEV A CHAQUE Re
# =============================================================================

comparison_rows = []

for Re in [100, 500, 1000]:

    fourier = next(
        (
            r for r in valid
            if r["parametrization"] == "fourier"
            and abs(r["Re"] - Re) < 1e-9
        ),
        None
    )

    cheb = next(
        (
            r for r in valid
            if r["parametrization"] == "chebyshev_mod"
            and abs(r["Re"] - Re) < 1e-9
        ),
        None
    )

    if fourier is None or cheb is None:
        continue

    row = {
        "Re": Re,

        # -------------------------------------------------------------
        # Energy
        # -------------------------------------------------------------

        "E_fourier":
            fourier["E_total"],

        "E_chebyshev":
            cheb["E_total"],

        "E_relative_difference_percent":
            rel_difference(
                fourier["E_total"],
                cheb["E_total"]
            ),

        # -------------------------------------------------------------
        # A20
        # -------------------------------------------------------------

        "A20_fourier":
            fourier["A20"],

        "A20_chebyshev":
            cheb["A20"],

        "A20_difference":
            signed_difference(
                fourier["A20"],
                cheb["A20"]
            ),

        # -------------------------------------------------------------
        # Mode 2
        # -------------------------------------------------------------

        "mode2_fourier":
            fourier["mode2_fraction"],

        "mode2_chebyshev":
            cheb["mode2_fraction"],

        "mode2_difference_percentage_points":
            100.0 * (
                fourier["mode2_fraction"]
                -
                cheb["mode2_fraction"]
            ),

        # -------------------------------------------------------------
        # Temporal
        # -------------------------------------------------------------

        "temporal_fourier":
            fourier["temporal_fraction"],

        "temporal_chebyshev":
            cheb["temporal_fraction"],

        "temporal_difference_percentage_points":
            100.0 * (
                cheb["temporal_fraction"]
                -
                fourier["temporal_fraction"]
            ),

        # -------------------------------------------------------------
        # Reconstruction
        # -------------------------------------------------------------

        "coverage_fourier":
            fourier["modal_coverage"],

        "coverage_chebyshev":
            cheb["modal_coverage"],

        "rmse_fourier":
            fourier["reconstruction_rmse"],

        "rmse_chebyshev":
            cheb["reconstruction_rmse"],

        # -------------------------------------------------------------
        # Dominant branch
        # -------------------------------------------------------------

        "dominant_fourier":
            fourier["dominant_branch"],

        "dominant_chebyshev":
            cheb["dominant_branch"],
    }

    comparison_rows.append(row)


# =============================================================================
# 11. SAUVEGARDE DU TABLEAU COMPARATIF
# =============================================================================

comparison_csv = os.path.join(
    RESULTS_DIR,
    "parametrization_comparison_table.csv"
)

if comparison_rows:

    fieldnames = list(
        comparison_rows[0].keys()
    )

    with open(
        comparison_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in comparison_rows:
            writer.writerow(row)


# =============================================================================
# 12. ANALYSE TRANSVERSALE PAR PARAMETRISATION
# =============================================================================

summary_by_param = {}

for parametrization in [
    "fourier",
    "chebyshev_mod",
]:

    subset = [
        r for r in valid
        if r["parametrization"] == parametrization
    ]

    if not subset:
        continue

    def mean_of(key):
        vals = [
            safe_float(r.get(key))
            for r in subset
        ]
        vals = [
            x for x in vals
            if np.isfinite(x)
        ]
        return float(np.mean(vals)) if vals else np.nan

    def std_of(key):
        vals = [
            safe_float(r.get(key))
            for r in subset
        ]
        vals = [
            x for x in vals
            if np.isfinite(x)
        ]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    summary_by_param[parametrization] = {

        "n_cases":
            len(subset),

        "mean_E":
            mean_of("E_total"),

        "std_E":
            std_of("E_total"),

        "mean_mode2_fraction":
            mean_of("mode2_fraction"),

        "std_mode2_fraction":
            std_of("mode2_fraction"),

        "mean_temporal_fraction":
            mean_of("temporal_fraction"),

        "std_temporal_fraction":
            std_of("temporal_fraction"),

        "mean_A20":
            mean_of("A20"),

        "std_A20":
            std_of("A20"),

        "mean_modal_coverage":
            mean_of("modal_coverage"),

        "mean_reconstruction_rmse":
            mean_of("reconstruction_rmse"),
    }


# =============================================================================
# 13. SAUVEGARDE COMPLETE JSON
# =============================================================================

json_output = {
    "study": {
        "title":
            "Fourier vs Chebyshev_mod parametrization study",

        "re_values": [
            100,
            500,
            1000
        ],

        "parametrizations": [
            "fourier",
            "chebyshev_mod"
        ],

        "number_of_runs":
            len(all_results),

        "number_of_valid_runs":
            len(valid),
    },

    "runs": valid,

    "pairwise_comparison": comparison_rows,

    "summary_by_parametrization":
        summary_by_param,
}

json_file = os.path.join(
    RESULTS_DIR,
    "parametrization_comparison.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        indent=2,
        ensure_ascii=False
    )


# =============================================================================
# 14. TABLEAU TRAINING
# =============================================================================

training_columns = [
    "parametrization",
    "Re",
    "epoch_final",
    "L_total_final",
    "L_PDE_final",
    "L_BC_final",
    "L_diss_final",
    "L_ctrl_final",
    "energy_MC_final",
    "L_var_final",
]

training_csv = os.path.join(
    RESULTS_DIR,
    "training_summary.csv"
)

with open(
    training_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=training_columns
    )

    writer.writeheader()

    for r in valid:
        writer.writerow(
            {
                k: r.get(k, "")
                for k in training_columns
            }
        )


# =============================================================================
# 15. AFFICHAGE DU TABLEAU FINAL
# =============================================================================

print("\n")
print("=" * 120)
print("TABLEAU FINAL : FOURIER vs CHEBYSHEV_MOD")
print("=" * 120)

header = (
    f"{'Re':>6} | "
    f"{'Param.':>14} | "
    f"{'E':>10} | "
    f"{'A20':>10} | "
    f"{'f2 (%)':>10} | "
    f"{'ft (%)':>10} | "
    f"{'Branche':>22} | "
    f"{'Coverage':>10} | "
    f"{'RMSE':>10}"
)

print(header)
print("-" * 120)

for r in valid:

    print(
        f"{r['Re']:6.0f} | "
        f"{r['parametrization']:>14} | "
        f"{r['E_total']:10.6f} | "
        f"{r['A20']:10.6f} | "
        f"{100*r['mode2_fraction']:10.3f} | "
        f"{100*r['temporal_fraction']:10.3f} | "
        f"{r['dominant_branch']:>22} | "
        f"{r['modal_coverage']:10.6f} | "
        f"{r['reconstruction_rmse']:10.6f}"
    )


# =============================================================================
# 16. TABLEAU DES DIFFERENCES FOURIER-CHEBYSHEV
# =============================================================================

print("\n")
print("=" * 120)
print("DIFFÉRENCE FOURIER / CHEBYSHEV_MOD")
print("=" * 120)

for row in comparison_rows:

    print(f"\nRe = {row['Re']}")

    print(
        f"  E Fourier      = "
        f"{row['E_fourier']:.8f}"
    )

    print(
        f"  E Chebyshev    = "
        f"{row['E_chebyshev']:.8f}"
    )

    print(
        f"  ΔE relatif     = "
        f"{row['E_relative_difference_percent']:.4f} %"
    )

    print(
        f"  f2 Fourier     = "
        f"{100*row['mode2_fourier']:.4f} %"
    )

    print(
        f"  f2 Chebyshev   = "
        f"{100*row['mode2_chebyshev']:.4f} %"
    )

    print(
        f"  Δf2            = "
        f"{row['mode2_difference_percentage_points']:.4f} points"
    )

    print(
        f"  ft Fourier     = "
        f"{100*row['temporal_fourier']:.4f} %"
    )

    print(
        f"  ft Chebyshev   = "
        f"{100*row['temporal_chebyshev']:.4f} %"
    )

    print(
        f"  Δft            = "
        f"{row['temporal_difference_percentage_points']:.4f} points"
    )

    print(
        f"  Branche Fourier   = "
        f"{row['dominant_fourier']}"
    )

    print(
        f"  Branche Chebyshev = "
        f"{row['dominant_chebyshev']}"
    )


# =============================================================================
# 17. RESUME STATISTIQUE
# =============================================================================

print("\n")
print("=" * 100)
print("RESUME PAR PARAMETRISATION")
print("=" * 100)

for parametrization, s in summary_by_param.items():

    print(
        f"\n{parametrization.upper()}"
    )

    print(
        f"  nombre de cas              = "
        f"{s['n_cases']}"
    )

    print(
        f"  E moyen                    = "
        f"{s['mean_E']:.8f}"
        f" +/- {s['std_E']:.8f}"
    )

    print(
        f"  f2 moyen                   = "
        f"{100*s['mean_mode2_fraction']:.4f} %"
        f" +/- {100*s['std_mode2_fraction']:.4f}"
    )

    print(
        f"  fraction temporelle moyenne = "
        f"{100*s['mean_temporal_fraction']:.4f} %"
        f" +/- {100*s['std_temporal_fraction']:.4f}"
    )

    print(
        f"  A20 moyen                  = "
        f"{s['mean_A20']:.8f}"
        f" +/- {s['std_A20']:.8f}"
    )

    print(
        f"  couverture modale moyenne = "
        f"{s['mean_modal_coverage']:.8f}"
    )

    print(
        f"  RMSE reconstruction moyen = "
        f"{s['mean_reconstruction_rmse']:.8f}"
    )


# =============================================================================
# 18. FIGURES OPTIONNELLES
# =============================================================================

MAKE_FIGURES = True

if MAKE_FIGURES:

    try:
        import matplotlib.pyplot as plt

        # ---------------------------------------------------------------------
        # Préparer les données
        # ---------------------------------------------------------------------

        Res = [100, 500, 1000]

        data = {
            "fourier": {
                "f2": [],
                "ft": [],
                "E": [],
            },

            "chebyshev_mod": {
                "f2": [],
                "ft": [],
                "E": [],
            },
        }

        for param in data:

            for Re in Res:

                r = next(
                    (
                        x for x in valid
                        if x["parametrization"] == param
                        and abs(x["Re"] - Re) < 1e-9
                    ),
                    None
                )

                if r is None:
                    data[param]["f2"].append(np.nan)
                    data[param]["ft"].append(np.nan)
                    data[param]["E"].append(np.nan)

                else:
                    data[param]["f2"].append(
                        100*r["mode2_fraction"]
                    )

                    data[param]["ft"].append(
                        100*r["temporal_fraction"]
                    )

                    data[param]["E"].append(
                        r["E_total"]
                    )

        # ---------------------------------------------------------------------
        # Figure 1 : Mode 2
        # ---------------------------------------------------------------------

        plt.figure(figsize=(7, 5))

        plt.plot(
            Res,
            data["fourier"]["f2"],
            marker="o",
            label="Fourier"
        )

        plt.plot(
            Res,
            data["chebyshev_mod"]["f2"],
            marker="s",
            label="Chebyshev mod."
        )

        plt.xlabel("Re")
        plt.ylabel("Mode-2 energy fraction (%)")
        plt.title("Mode-2 dominance vs Reynolds number")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig1 = os.path.join(
            RESULTS_DIR,
            "mode2_fraction_vs_Re.png"
        )

        plt.savefig(
            fig1,
            dpi=300
        )

        plt.close()

        # ---------------------------------------------------------------------
        # Figure 2 : Temporal
        # ---------------------------------------------------------------------

        plt.figure(figsize=(7, 5))

        plt.plot(
            Res,
            data["fourier"]["ft"],
            marker="o",
            label="Fourier"
        )

        plt.plot(
            Res,
            data["chebyshev_mod"]["ft"],
            marker="s",
            label="Chebyshev mod."
        )

        plt.xlabel("Re")
        plt.ylabel("Temporal energy fraction (%)")
        plt.title("Temporal contribution vs Reynolds number")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig2 = os.path.join(
            RESULTS_DIR,
            "temporal_fraction_vs_Re.png"
        )

        plt.savefig(
            fig2,
            dpi=300
        )

        plt.close()

        # ---------------------------------------------------------------------
        # Figure 3 : Energy
        # ---------------------------------------------------------------------

        plt.figure(figsize=(7, 5))

        plt.plot(
            Res,
            data["fourier"]["E"],
            marker="o",
            label="Fourier"
        )

        plt.plot(
            Res,
            data["chebyshev_mod"]["E"],
            marker="s",
            label="Chebyshev mod."
        )

        plt.axhline(
            0.25,
            linestyle="--",
            label="Target E = 0.25"
        )

        plt.xlabel("Re")
        plt.ylabel(r"$E_{\mathrm{total}}$")
        plt.title("Control energy vs Reynolds number")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig3 = os.path.join(
            RESULTS_DIR,
            "energy_vs_Re.png"
        )

        plt.savefig(
            fig3,
            dpi=300
        )

        plt.close()

        print("\nFigures saved:")
        print(fig1)
        print(fig2)
        print(fig3)

    except Exception as e:

        print(
            "\nATTENTION : les figures n'ont pas pu être générées."
        )

        print("Détail :", e)


# =============================================================================
# 19. SAUVEGARDE D'UN RAPPORT TEXTE SIMPLE
# =============================================================================

report_txt = os.path.join(
    RESULTS_DIR,
    "parametrization_report.txt"
)

with open(
    report_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COMPARATIVE PARAMETRIZATION STUDY\n"
    )

    f.write(
        "Fourier vs Chebyshev_mod\n"
    )

    f.write(
        "Re = 100, 500, 1000\n\n"
    )

    f.write("=" * 100)
    f.write("\n")

    for r in valid:

        f.write(
            f"\n{r['parametrization']} | Re={r['Re']}\n"
        )

        for key in [
            "E_total",
            "E_modes_sum",
            "A10",
            "A20",
            "A30",
            "A40",
            "A50",
            "A60",
            "mode2_fraction",
            "temporal_fraction",
            "dominant_branch",
            "modal_coverage",
            "reconstruction_rmse",
            "energy_MC_final",
            "L_total_final",
            "L_PDE_final",
            "L_BC_final",
            "L_diss_final",
            "L_ctrl_final",
            "L_var_final",
            "epoch_final",
        ]:

            if key in r:
                f.write(
                    f"{key:30s} = {r[key]}\n"
                )

        f.write("-" * 100)
        f.write("\n")


# =============================================================================
# 20. FIN
# =============================================================================

print("\n")
print("=" * 100)
print("FICHIERS PRODUITS")
print("=" * 100)

print(
    "\n1. Main results:"
)
print(main_csv)

print(
    "\n2. Fourier vs Chebyshev table:"
)
print(comparison_csv)

print(
    "\n3. Training summary:"
)
print(training_csv)

print(
    "\n4. Complete JSON:"
)
print(json_file)

print(
    "\n5. Text report:"
)
print(report_txt)

print("\n")
print("=" * 100)
print("FIN DU TRAITEMENT")
print("=" * 100)