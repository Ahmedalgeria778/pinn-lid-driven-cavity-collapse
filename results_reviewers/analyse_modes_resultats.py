# -*- coding: utf-8 -*-

"""
====================================================================
MODE COUNT — ANALYSE MODALE DES RESULTATS DEJA CALCULES
====================================================================

Objectif
--------
Analyser les quatre paramétrisations du contrôle :

    mx6_mt1  ->  6 coefficients
    mx4_mt3  -> 12 coefficients
    mx6_mt3  -> 18 coefficients
    mx6_mt5  -> 30 coefficients

A partir de :

    results_reviewers/07_mode_count/mode_count_results.csv

Le script calcule :

    1. Energie totale
    2. Coefficients stationnaires A1, A2, A3, ...
    3. Energies de chaque mode spatial
    4. Fractions E_i / E_total
    5. Energie du mode (2,0)
    6. Fraction du mode (2,0)
    7. Fraction stationnaire j=0
    8. Fraction temporelle j>=1
    9. Mode dominant
   10. Couverture modale
   11. Erreur énergétique de reconstruction
   12. Tableau final exploitable pour le manuscrit
   13. Fichiers CSV
   14. Fichier Excel si openpyxl est installé

Convention
----------
Le contrôle est :

    U_lid(x,t) =
        sum_{i=0}^{n_mx-1}
        sum_{j=0}^{n_mt-1}
        c_ij sin((i+1) pi x/L) cos(j pi t)

Pour L = 1 et T = 1 :

    ||phi_i,0||^2 = 1/2

    ||phi_i,j||^2 = 1/4   pour j > 0

Donc :

    E_ij = c_ij^2 / 2     pour j = 0

    E_ij = c_ij^2 / 4     pour j > 0

Attention
---------
A1, A2, A3 sont définis ici UNIQUEMENT par :

    A1 = c_0,0
    A2 = c_1,0
    A3 = c_2,0

On ne somme PAS les coefficients temporels pour construire A1,A2,A3.

Cela permet une comparaison cohérente des amplitudes
stationnaires entre les quatre paramétrisations.
====================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ====================================================================
# 1. CHEMIN DU FICHIER RESULTAT
# ====================================================================

CSV_PATH = Path(
    r"C:\Users\kings\OneDrive\Documents\Default Project"
    r"\PoF_R_lid_driven_paper\results_reviewers"
    r"\07_mode_count\mode_count_results.csv"
)


# ====================================================================
# 2. DOSSIER DE SORTIE
# ====================================================================

OUT_DIR = CSV_PATH.parent / "modal_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================
# 3. CONFIGURATION EXACTE DES QUATRE CAS
# ====================================================================

CASE_CONFIG = {
    "mx6_mt1": {
        "n_mx": 6,
        "n_mt": 1,
    },

    "mx4_mt3": {
        "n_mx": 4,
        "n_mt": 3,
    },

    "mx6_mt3": {
        "n_mx": 6,
        "n_mt": 3,
    },

    "mx6_mt5": {
        "n_mx": 6,
        "n_mt": 5,
    },
}


# Ordre souhaité pour les tableaux
CASE_ORDER = [
    "mx6_mt1",
    "mx4_mt3",
    "mx6_mt3",
    "mx6_mt5",
]


# ====================================================================
# 4. UTILITAIRES
# ====================================================================

def safe_float(x, default=np.nan):
    """
    Conversion robuste en float.
    """
    try:
        return float(x)
    except Exception:
        return default


def mode_label(i_zero, j):
    """
    Convertit l'indice interne i=0,1,2,...
    en numéro spatial 1,2,3,...
    """
    return f"({i_zero + 1},{j})"


# ====================================================================
# 5. VERIFICATION DU FICHIER SOURCE
# ====================================================================

print("=" * 100)
print("MODE COUNT — ANALYSE MODALE")
print("=" * 100)

print("\nFichier source :")
print(CSV_PATH)


if not CSV_PATH.exists():
    raise FileNotFoundError(
        "\nERREUR : le fichier suivant est introuvable :\n"
        f"{CSV_PATH}\n\n"
        "Verifiez que la campagne mode_count a bien produit "
        "mode_count_results.csv."
    )


# ====================================================================
# 6. LECTURE DU CSV
# ====================================================================

df = pd.read_csv(CSV_PATH)

df.columns = [str(c).strip() for c in df.columns]


print("\nColonnes du fichier :")
for c in df.columns:
    print(f"  - {c}")


print(f"\nNombre de lignes : {len(df)}")


# ====================================================================
# 7. VERIFICATION DES COLONNES ESSENTIELLES
# ====================================================================

required_basic = [
    "n_mx",
    "n_mt",
]

missing_basic = [
    c for c in required_basic
    if c not in df.columns
]


if missing_basic:
    raise ValueError(
        "\nERREUR : colonnes essentielles absentes : "
        f"{missing_basic}\n\n"
        "Le fichier doit provenir directement de la campagne "
        "07_mode_count du script PINN."
    )


# ====================================================================
# 8. CONVERSION NUMERIQUE
# ====================================================================

for c in ["n_mx", "n_mt"]:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


# ====================================================================
# 9. IDENTIFICATION AUTOMATIQUE DU CAS
# ====================================================================

df["case"] = (
    "mx"
    + df["n_mx"].astype("Int64").astype(str)
    + "_mt"
    + df["n_mt"].astype("Int64").astype(str)
)


# ====================================================================
# 10. VERIFICATION DES QUATRE CAS
# ====================================================================

print("\n")
print("=" * 100)
print("CAS DETECTES")
print("=" * 100)

for case in CASE_ORDER:

    n = int(
        (
            (df["case"] == case)
        ).sum()
    )

    expected_n_modes = (
        CASE_CONFIG[case]["n_mx"]
        *
        CASE_CONFIG[case]["n_mt"]
    )

    print(
        f"{case:10s} | "
        f"attendu = {expected_n_modes:2d} coeff. | "
        f"lignes trouvées = {n}"
    )


# ====================================================================
# 11. FONCTION PRINCIPALE D'ANALYSE
# ====================================================================

def analyse_row(row):
    """
    Analyse une ligne du mode_count_results.csv.
    """

    case = str(row["case"])

    if case not in CASE_CONFIG:
        raise ValueError(
            f"\nCas inconnu : {case}\n"
            f"Cas attendus : {list(CASE_CONFIG.keys())}"
        )

    n_mx = CASE_CONFIG[case]["n_mx"]
    n_mt = CASE_CONFIG[case]["n_mt"]

    n_modes = n_mx * n_mt


    # ================================================================
    # 11.1 EXTRACTION DES COEFFICIENTS
    # ================================================================

    coefficients = {}

    missing_coefficients = []

    for i in range(n_mx):

        for j in range(n_mt):

            key = f"c_{i}_{j}"

            if key not in row.index:
                missing_coefficients.append(key)
                continue

            coefficients[(i, j)] = safe_float(
                row[key]
            )


    if missing_coefficients:

        raise ValueError(
            f"\nERREUR pour {case} : "
            f"coefficients absents : {missing_coefficients}"
        )


    # ================================================================
    # 11.2 ENERGIES MODALES
    # ================================================================

    mode_energies = {}

    for (i, j), c in coefficients.items():

        if j == 0:

            # cos(0*pi*t)=1
            norm2 = 0.5

        else:

            # intégrales spatiale + temporelle
            norm2 = 0.25

        Eij = c * c * norm2

        mode_energies[(i, j)] = Eij


    # ================================================================
    # 11.3 ENERGIE TOTALE DU CSV
    # ================================================================

    if "E_total" in row.index:

        E_total_csv = safe_float(
            row["E_total"]
        )

    elif "energy_final" in row.index:

        E_total_csv = safe_float(
            row["energy_final"]
        )

    else:

        E_total_csv = np.nan


    # ================================================================
    # 11.4 SOMME DES ENERGIES MODALES
    # ================================================================

    E_modal_sum = sum(
        mode_energies.values()
    )


    # ================================================================
    # 11.5 AMPLITUDES STATIONNAIRES
    # ================================================================

    A = {}

    for i in range(n_mx):

        A[i + 1] = coefficients.get(
            (i, 0),
            0.0
        )


    # ================================================================
    # 11.6 ENERGIE SPATIALE PAR MODE
    # ================================================================

    E_spatial = {}

    for i in range(n_mx):

        E_spatial[i + 1] = sum(
            Eij
            for (ii, jj), Eij
            in mode_energies.items()
            if ii == i
        )


    # ================================================================
    # 11.7 FRACTIONS SPATIALES
    # ================================================================

    # Pour les comparaisons, on utilise la somme modale.
    # Cela évite une incohérence entre moyenne discrète et
    # intégration modale.

    E_reference = E_modal_sum

    if E_reference <= 0:
        raise ValueError(
            f"Energie modale nulle pour {case}."
        )


    E_spatial_frac = {}

    for i in range(n_mx):

        E_spatial_frac[i + 1] = (
            E_spatial[i + 1]
            /
            E_reference
        )


    # ================================================================
    # 11.8 MODE (2,0)
    # ================================================================

    # Convention interne :
    #
    # spatial mode 1 -> i=0
    # spatial mode 2 -> i=1
    # spatial mode 3 -> i=2
    #
    # Le mode physique (2,0) correspond donc à (i=1,j=0).

    E_20 = mode_energies.get(
        (1, 0),
        0.0
    )

    mode2_fraction = (
        E_20
        /
        E_reference
    )


    # ================================================================
    # 11.9 FRACTION STATIONNAIRE j=0
    # ================================================================

    E_j0 = sum(
        Eij
        for (i, j), Eij
        in mode_energies.items()
        if j == 0
    )


    # ================================================================
    # 11.10 FRACTION TEMPORELLE j>=1
    # ================================================================

    E_jpos = sum(
        Eij
        for (i, j), Eij
        in mode_energies.items()
        if j >= 1
    )


    frac_j0 = (
        E_j0
        /
        E_reference
    )


    frac_jpos = (
        E_jpos
        /
        E_reference
    )


    # ================================================================
    # 11.11 MODE DOMINANT
    # ================================================================

    dominant_mode = max(
        mode_energies,
        key=mode_energies.get
    )

    dominant_i = dominant_mode[0] + 1
    dominant_j = dominant_mode[1]

    dominant_energy = mode_energies[
        dominant_mode
    ]

    dominant_fraction = (
        dominant_energy
        /
        E_reference
    )


    # ================================================================
    # 11.12 COUVERTURE MODALE
    # ================================================================

    if np.isfinite(E_total_csv) and E_total_csv > 0:

        modal_coverage_vs_csv = (
            E_modal_sum
            /
            E_total_csv
        )

    else:

        modal_coverage_vs_csv = np.nan


    # ================================================================
    # 11.13 ERREUR DE RECONSTRUCTION ENERGETIQUE
    # ================================================================

    if np.isfinite(E_total_csv) and E_total_csv > 0:

        relative_energy_error = (
            abs(E_modal_sum - E_total_csv)
            /
            E_total_csv
        )

    else:

        relative_energy_error = np.nan


    # ================================================================
    # 11.14 RESULTAT
    # ================================================================

    result = {

        "case": case,

        "n_mx": n_mx,
        "n_mt": n_mt,
        "n_modes": n_modes,

        "E_total_csv": E_total_csv,
        "E_modal_sum": E_modal_sum,

        "A1": A.get(1, 0.0),
        "A2": A.get(2, 0.0),
        "A3": A.get(3, 0.0),
        "A4": A.get(4, 0.0),
        "A5": A.get(5, 0.0),
        "A6": A.get(6, 0.0),

        "mode2_fraction": mode2_fraction,

        "E_j0_fraction": frac_j0,
        "E_jpos_fraction": frac_jpos,

        "dominant_spatial_mode": dominant_i,
        "dominant_temporal_mode": dominant_j,

        "dominant_mode_fraction": dominant_fraction,

        "modal_coverage_vs_csv":
            modal_coverage_vs_csv,

        "relative_energy_error":
            relative_energy_error,
    }


    # ================================================================
    # 11.15 FRACTIONS SPATIALES
    # ================================================================

    for i in range(1, n_mx + 1):

        result[
            f"E{i}_fraction"
        ] = E_spatial_frac[i]


    # ================================================================
    # 11.16 ENERGIES DE CHAQUE MODE
    # ================================================================

    for (i, j), Eij in mode_energies.items():

        physical_i = i + 1

        result[
            f"E_{physical_i}_{j}"
        ] = Eij

        result[
            f"frac_{physical_i}_{j}"
        ] = (
            Eij
            /
            E_reference
        )


    # ================================================================
    # 11.17 COEFFICIENTS BRUTS
    # ================================================================

    for (i, j), c in coefficients.items():

        result[
            f"c_{i}_{j}"
        ] = c


    return result


# ====================================================================
# 12. ANALYSE DE CHAQUE LIGNE
# ====================================================================

results = []

for _, row in df.iterrows():

    results.append(
        analyse_row(row)
    )


results_df = pd.DataFrame(results)


# ====================================================================
# 13. ORDRE DES CAS
# ====================================================================

results_df["case"] = pd.Categorical(
    results_df["case"],
    categories=CASE_ORDER,
    ordered=True
)

results_df = (
    results_df
    .sort_values("case")
    .reset_index(drop=True)
)


# ====================================================================
# 14. TABLEAU PRINCIPAL
# ====================================================================

main_columns = [
    "case",
    "n_mx",
    "n_mt",
    "n_modes",

    "E_total_csv",
    "E_modal_sum",

    "A1",
    "A2",
    "A3",

    "E1_fraction",
    "E2_fraction",
    "E3_fraction",
    "E4_fraction",
    "E5_fraction",
    "E6_fraction",

    "mode2_fraction",

    "E_j0_fraction",
    "E_jpos_fraction",

    "dominant_spatial_mode",
    "dominant_temporal_mode",
    "dominant_mode_fraction",

    "modal_coverage_vs_csv",
    "relative_energy_error",
]

main_columns = [
    c for c in main_columns
    if c in results_df.columns
]

main_table = results_df[
    main_columns
].copy()


# ====================================================================
# 15. VERSION EN POURCENTAGES
# ====================================================================

paper_table = pd.DataFrame()

paper_table["Parameterization"] = (
    main_table["case"].astype(str)
)

paper_table["N_coeff"] = (
    main_table["n_modes"]
)

paper_table["A1"] = (
    main_table["A1"]
)

paper_table["A2"] = (
    main_table["A2"]
)

paper_table["A3"] = (
    main_table["A3"]
)

paper_table["E1/E (%)"] = (
    100.0 * main_table["E1_fraction"]
)

paper_table["E2/E (%)"] = (
    100.0 * main_table["E2_fraction"]
)

paper_table["E3/E (%)"] = (
    100.0 * main_table["E3_fraction"]
)

if "E4_fraction" in main_table:
    paper_table["E4/E (%)"] = (
        100.0 * main_table["E4_fraction"]
    )

paper_table["Mode-2 (2,0) (%)"] = (
    100.0 * main_table["mode2_fraction"]
)

paper_table["Stationary j=0 (%)"] = (
    100.0 * main_table["E_j0_fraction"]
)

paper_table["Temporal j>=1 (%)"] = (
    100.0 * main_table["E_jpos_fraction"]
)

paper_table["Dominant spatial mode"] = (
    main_table["dominant_spatial_mode"]
)

paper_table["Dominant temporal mode"] = (
    main_table["dominant_temporal_mode"]
)

paper_table["Dominant energy (%)"] = (
    100.0 * main_table["dominant_mode_fraction"]
)


# ====================================================================
# 16. TABLEAU TRES COMPACT POUR LE MANUSCRIT
# ====================================================================

paper_compact = paper_table[
    [
        "Parameterization",
        "N_coeff",
        "A1",
        "A2",
        "A3",
        "E1/E (%)",
        "E2/E (%)",
        "E3/E (%)",
        "Mode-2 (2,0) (%)",
        "Stationary j=0 (%)",
        "Temporal j>=1 (%)",
        "Dominant spatial mode",
        "Dominant temporal mode",
        "Dominant energy (%)",
    ]
].copy()


# ====================================================================
# 17. AFFICHAGE TABLEAU PRINCIPAL
# ====================================================================

print("\n")
print("=" * 150)
print("RESULTATS MODAUX — TABLEAU PRINCIPAL")
print("=" * 150)

display_main = paper_compact.copy()

numeric_columns = [
    "A1",
    "A2",
    "A3",
    "E1/E (%)",
    "E2/E (%)",
    "E3/E (%)",
    "Mode-2 (2,0) (%)",
    "Stationary j=0 (%)",
    "Temporal j>=1 (%)",
    "Dominant energy (%)",
]

for c in numeric_columns:

    if c in display_main.columns:

        display_main[c] = (
            pd.to_numeric(
                display_main[c],
                errors="coerce"
            )
            .round(5)
        )


print(
    display_main.to_string(
        index=False
    )
)


# ====================================================================
# 18. DIAGNOSTIC DE ROBUSTESSE DU MODE 2
# ====================================================================

m2 = (
    results_df["mode2_fraction"]
    .to_numpy(dtype=float)
)


print("\n")
print("=" * 100)
print("ROBUSTESSE DU MODE (2,0)")
print("=" * 100)


for _, row in results_df.iterrows():

    print(
        f"{str(row['case']):10s} | "
        f"N={int(row['n_modes']):2d} | "
        f"Mode(2,0) = "
        f"{100*row['mode2_fraction']:.6f}% | "
        f"dominant = "
        f"({int(row['dominant_spatial_mode'])},"
        f"{int(row['dominant_temporal_mode'])}) | "
        f"j=0 = "
        f"{100*row['E_j0_fraction']:.6f}% | "
        f"j>=1 = "
        f"{100*row['E_jpos_fraction']:.6f}%"
    )


# ====================================================================
# 19. STATISTIQUES ENTRE LES QUATRE PARAMETRISATIONS
# ====================================================================

print("\n")
print("=" * 100)
print("STATISTIQUES ENTRE PARAMETRISATIONS")
print("=" * 100)

m2_mean = np.mean(m2)
m2_std = np.std(m2, ddof=1)
m2_min = np.min(m2)
m2_max = np.max(m2)
m2_range = m2_max - m2_min

print(
    f"Mode (2,0) moyen      : "
    f"{100*m2_mean:.6f}%"
)

print(
    f"Ecart-type            : "
    f"{100*m2_std:.6f} points"
)

print(
    f"Minimum               : "
    f"{100*m2_min:.6f}%"
)

print(
    f"Maximum               : "
    f"{100*m2_max:.6f}%"
)

print(
    f"Etendue               : "
    f"{100*m2_range:.6f} points"
)

if abs(m2_mean) > 1e-15:

    m2_cv = (
        100.0
        *
        m2_std
        /
        abs(m2_mean)
    )

    print(
        f"CV                    : "
        f"{m2_cv:.6f}%"
    )


# ====================================================================
# 20. VARIATION DE L'ENERGIE
# ====================================================================

Evals = (
    results_df["E_total_csv"]
    .to_numpy(dtype=float)
)

print("\n")
print("=" * 100)
print("VARIATION DE L'ENERGIE")
print("=" * 100)

print(
    f"E min                 : "
    f"{np.min(Evals):.10f}"
)

print(
    f"E max                 : "
    f"{np.max(Evals):.10f}"
)

print(
    f"Delta E               : "
    f"{np.max(Evals)-np.min(Evals):.10e}"
)

print(
    f"Variation relative     : "
    f"{100*(np.max(Evals)-np.min(Evals))/np.mean(Evals):.6f}%"
)


# ====================================================================
# 21. VERIFICATION DE LA SOMME DES FRACTIONS SPATIALES
# ====================================================================

print("\n")
print("=" * 100)
print("VERIFICATION DES FRACTIONS")
print("=" * 100)

for _, row in results_df.iterrows():

    spatial_sum = 0.0

    for i in range(1, int(row["n_mx"]) + 1):

        key = f"E{i}_fraction"

        if key in row.index:

            spatial_sum += (
                float(row[key])
            )


    total_fraction = (
        spatial_sum
    )

    print(
        f"{str(row['case']):10s} | "
        f"Somme fractions spatiales = "
        f"{total_fraction:.10f} | "
        f"j=0 + j>=1 = "
        f"{row['E_j0_fraction'] + row['E_jpos_fraction']:.10f}"
    )


# ====================================================================
# 22. VERIFICATION DES COUVERTURES
# ====================================================================

print("\n")
print("=" * 100)
print("COUVERTURE MODALE")
print("=" * 100)

for _, row in results_df.iterrows():

    print(
        f"{str(row['case']):10s} | "
        f"E_modal/E_CSV = "
        f"{row['modal_coverage_vs_csv']:.12f} | "
        f"erreur relative = "
        f"{100*row['relative_energy_error']:.6e}%"
    )


# ====================================================================
# 23. COEFFICIENTS
# ====================================================================

coefficient_columns = [
    "case"
]

for i in range(6):

    for j in range(5):

        key = f"c_{i}_{j}"

        if key in results_df.columns:

            coefficient_columns.append(
                key
            )


coefficients_table = results_df[
    coefficient_columns
].copy()


# ====================================================================
# 24. ENERGIES MODALES
# ====================================================================

modal_energy_columns = [
    "case"
]

for i in range(6):

    for j in range(5):

        key = f"E_{i+1}_{j}"

        if key in results_df.columns:

            modal_energy_columns.append(
                key
            )

modal_energy_table = results_df[
    modal_energy_columns
].copy()


# ====================================================================
# 25. FRACTIONS MODALES DETAILLEES
# ====================================================================

modal_fraction_columns = [
    "case"
]

for i in range(6):

    for j in range(5):

        key = f"frac_{i+1}_{j}"

        if key in results_df.columns:

            modal_fraction_columns.append(
                key
            )

modal_fraction_table = results_df[
    modal_fraction_columns
].copy()


# ====================================================================
# 26. SAUVEGARDE CSV
# ====================================================================

main_csv = (
    OUT_DIR
    /
    "mode_count_modal_analysis.csv"
)

paper_csv = (
    OUT_DIR
    /
    "mode_count_paper_table.csv"
)

paper_compact_csv = (
    OUT_DIR
    /
    "mode_count_paper_compact.csv"
)

coeff_csv = (
    OUT_DIR
    /
    "mode_count_coefficients.csv"
)

energy_csv = (
    OUT_DIR
    /
    "mode_count_modal_energies.csv"
)

fraction_csv = (
    OUT_DIR
    /
    "mode_count_modal_fractions.csv"
)

full_csv = (
    OUT_DIR
    /
    "mode_count_modal_full.csv"
)


main_table.to_csv(
    main_csv,
    index=False
)

paper_table.to_csv(
    paper_csv,
    index=False
)

paper_compact.to_csv(
    paper_compact_csv,
    index=False
)

coefficients_table.to_csv(
    coeff_csv,
    index=False
)

modal_energy_table.to_csv(
    energy_csv,
    index=False
)

modal_fraction_table.to_csv(
    fraction_csv,
    index=False
)

results_df.to_csv(
    full_csv,
    index=False
)


# ====================================================================
# 27. EXCEL
# ====================================================================

excel_path = (
    OUT_DIR
    /
    "mode_count_modal_analysis.xlsx"
)

try:

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        paper_compact.to_excel(
            writer,
            sheet_name="Paper_compact",
            index=False
        )

        paper_table.to_excel(
            writer,
            sheet_name="Paper_table",
            index=False
        )

        main_table.to_excel(
            writer,
            sheet_name="Main_analysis",
            index=False
        )

        coefficients_table.to_excel(
            writer,
            sheet_name="Coefficients",
            index=False
        )

        modal_energy_table.to_excel(
            writer,
            sheet_name="Modal_energies",
            index=False
        )

        modal_fraction_table.to_excel(
            writer,
            sheet_name="Modal_fractions",
            index=False
        )

        results_df.to_excel(
            writer,
            sheet_name="Full_results",
            index=False
        )

    excel_ok = True

except ModuleNotFoundError:

    excel_ok = False

    print(
        "\nWARNING : openpyxl n'est pas installe."
    )

    print(
        "Les fichiers CSV ont ete crees."
    )

    print(
        "Pour creer le fichier Excel :"
    )

    print(
        "python -m pip install openpyxl"
    )


# ====================================================================
# 28. RESUME FINAL
# ====================================================================

print("\n")
print("=" * 100)
print("RESUME FINAL")
print("=" * 100)

for _, row in results_df.iterrows():

    print(
        f"\n{str(row['case'])}"
    )

    print(
        f"  N coefficients       = "
        f"{int(row['n_modes'])}"
    )

    print(
        f"  E_total              = "
        f"{row['E_total_csv']:.10f}"
    )

    print(
        f"  A1                   = "
        f"{row['A1']:.8f}"
    )

    print(
        f"  A2                   = "
        f"{row['A2']:.8f}"
    )

    print(
        f"  A3                   = "
        f"{row['A3']:.8f}"
    )

    print(
        f"  Mode (2,0)           = "
        f"{100*row['mode2_fraction']:.6f}%"
    )

    print(
        f"  Stationnaire j=0     = "
        f"{100*row['E_j0_fraction']:.6f}%"
    )

    print(
        f"  Temporel j>=1        = "
        f"{100*row['E_jpos_fraction']:.6f}%"
    )

    print(
        f"  Mode dominant        = "
        f"({int(row['dominant_spatial_mode'])},"
        f"{int(row['dominant_temporal_mode'])})"
    )

    print(
        f"  Fraction dominante   = "
        f"{100*row['dominant_mode_fraction']:.6f}%"
    )


# ====================================================================
# 29. FICHIERS
# ====================================================================

print("\n")
print("=" * 100)
print("FICHIERS GENERES")
print("=" * 100)

print(
    f"\n1. {main_csv}"
)

print(
    f"2. {paper_csv}"
)

print(
    f"3. {paper_compact_csv}"
)

print(
    f"4. {coeff_csv}"
)

print(
    f"5. {energy_csv}"
)

print(
    f"6. {fraction_csv}"
)

print(
    f"7. {full_csv}"
)

if excel_ok:

    print(
        f"8. {excel_path}"
    )


print("\n")
print("=" * 100)
print("ANALYSE TERMINEE")
print("=" * 100)