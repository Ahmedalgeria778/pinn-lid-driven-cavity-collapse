# -*- coding: utf-8 -*-

"""
====================================================================
MODE COUNT ANALYSIS — 3 CASES
====================================================================

Cases:
    1) mx6_mt1
    2) mx6_mt3
    3) mx4_mt3

Expected columns:
    total,pde,bc,diss,ctrl,energy,var,epoch

The three runs are assumed to be complete up to epoch 3000.

Outputs:
    mode_count_summary.csv
    mode_count_history_combined.csv
    mode_count_last500.csv
    mode_count_analysis.xlsx
    figures/
        total_comparison.png
        components_comparison.png
        energy_comparison.png
        var_comparison.png
        stabilization_last500.png
====================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ====================================================================
# 1. PATHS
# ====================================================================

FILES = {
    "mx6_mt1": Path(
        r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"
        r"\results_reviewers\07_mode_count\mx6_mt1\training_history.csv"
    ),

    "mx6_mt3": Path(
        r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"
        r"\results_reviewers\07_mode_count\mx6_mt3\training_history.csv"
    ),

    "mx4_mt3": Path(
        r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"
        r"\results_reviewers\07_mode_count\mx4_mt3\training_history.csv"
    ),
    "mx6_mt5": Path(
        r"C:\Users\kings\OneDrive\Documents\Default Project\PoF_R_lid_driven_paper"
        r"\results_reviewers\07_mode_count\mx6_mt5\training_history.csv"
    ),
}


# ====================================================================
# 2. OUTPUT DIRECTORY
# ====================================================================

BASE_DIR = FILES["mx6_mt1"].parent.parent.parent
OUT_DIR = BASE_DIR / "mode_count_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ====================================================================
# 3. EXPECTED COLUMNS
# ====================================================================

EXPECTED_COLUMNS = [
    "total",
    "pde",
    "bc",
    "diss",
    "ctrl",
    "energy",
    "var",
    "epoch",
]


# ====================================================================
# 4. READ AND CLEAN
# ====================================================================

def load_history(path: Path, case_name: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"\nFichier introuvable:\n{path}")

    df = pd.read_csv(path)

    # Normalise les noms de colonnes
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nColonnes manquantes dans {path.name}: {missing}\n"
            f"Colonnes trouvées: {list(df.columns)}"
        )

    # Garde exactement les variables demandées
    df = df[EXPECTED_COLUMNS].copy()

    # Conversion numérique
    for c in EXPECTED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Supprime lignes invalides
    df = df.dropna(subset=EXPECTED_COLUMNS)

    # Trie par epoch
    df = df.sort_values("epoch").reset_index(drop=True)

    # Supprime les epochs dupliqués en gardant la dernière valeur
    df = df.drop_duplicates(subset="epoch", keep="last").reset_index(drop=True)

    # Ajoute le nom du cas
    df["case"] = case_name

    return df


# ====================================================================
# 5. LOAD ALL CASES
# ====================================================================

histories = {}

for case, path in FILES.items():
    print("=" * 80)
    print(f"Lecture : {case}")
    print(path)

    df = load_history(path, case)

    histories[case] = df

    print(f"Nombre de lignes : {len(df)}")
    print(f"Epoch initial   : {df['epoch'].iloc[0]}")
    print(f"Epoch final     : {df['epoch'].iloc[-1]}")

    if int(df["epoch"].iloc[-1]) != 3000:
        print(
            f"WARNING : {case} ne se termine pas exactement à 3000 "
            f"(dernier epoch = {df['epoch'].iloc[-1]})"
        )


# ====================================================================
# 6. FUNCTION FOR STATISTICS
# ====================================================================

def compute_case_summary(df: pd.DataFrame) -> dict:

    metrics = [
        "total",
        "pde",
        "bc",
        "diss",
        "ctrl",
        "energy",
        "var",
    ]

    last_epoch = int(df["epoch"].iloc[-1])

    result = {
        "case": df["case"].iloc[0],
        "n_rows": len(df),
        "epoch_first": int(df["epoch"].iloc[0]),
        "epoch_final": last_epoch,
    }

    # ------------------------------------------------------------
    # Final values
    # ------------------------------------------------------------

    final = df.iloc[-1]

    for m in metrics:
        result[f"{m}_final"] = float(final[m])

    # ------------------------------------------------------------
    # Minimum total loss
    # ------------------------------------------------------------

    idx_best = df["total"].idxmin()
    best = df.loc[idx_best]

    result["total_min"] = float(best["total"])
    result["epoch_total_min"] = int(best["epoch"])

    # ------------------------------------------------------------
    # Energy target information
    # ------------------------------------------------------------

    E_TARGET = 0.25

    result["energy_target"] = E_TARGET
    result["energy_error_abs"] = abs(float(final["energy"]) - E_TARGET)
    result["energy_error_rel_percent"] = (
        100.0 * abs(float(final["energy"]) - E_TARGET) / E_TARGET
    )

    # ------------------------------------------------------------
    # Statistics over several final windows
    # ------------------------------------------------------------

    for window in [100, 200, 500]:

        n = min(window, len(df))
        tail = df.tail(n)

        result[f"n_last{window}"] = n

        for m in metrics:
            result[f"{m}_mean_last{window}"] = float(tail[m].mean())
            result[f"{m}_std_last{window}"] = float(tail[m].std(ddof=1))

            mean_val = abs(float(tail[m].mean()))

            if mean_val > 1e-15:
                result[f"{m}_cv_percent_last{window}"] = (
                    100.0 * float(tail[m].std(ddof=1)) / mean_val
                )
            else:
                result[f"{m}_cv_percent_last{window}"] = np.nan

        # --------------------------------------------------------
        # Linear slope on the final window
        # --------------------------------------------------------

        x = tail["epoch"].to_numpy(dtype=float)

        for m in metrics:

            y = tail[m].to_numpy(dtype=float)

            if len(x) >= 2:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = np.nan

            result[f"{m}_slope_last{window}"] = float(slope)

    # ------------------------------------------------------------
    # Overall reduction from first epoch to last
    # ------------------------------------------------------------

    first = df.iloc[0]

    for m in metrics:

        x0 = float(first[m])
        xf = float(final[m])

        result[f"{m}_delta"] = xf - x0

        if abs(x0) > 1e-15:
            result[f"{m}_change_percent"] = 100.0 * (xf - x0) / abs(x0)
        else:
            result[f"{m}_change_percent"] = np.nan

    # ------------------------------------------------------------
    # Loss composition at final epoch
    # ------------------------------------------------------------

    total_final = float(final["total"])

    for m in ["pde", "bc", "diss", "ctrl", "var"]:

        if abs(total_final) > 1e-15:
            result[f"{m}_fraction_of_total_percent"] = (
                100.0 * float(final[m]) / total_final
            )
        else:
            result[f"{m}_fraction_of_total_percent"] = np.nan

    return result


# ====================================================================
# 7. COMPUTE SUMMARY
# ====================================================================

summary_rows = []

for case, df in histories.items():
    summary_rows.append(compute_case_summary(df))

summary = pd.DataFrame(summary_rows)


# ====================================================================
# 8. COMBINED HISTORY
# ====================================================================

combined = pd.concat(histories.values(), ignore_index=True)

combined = combined[
    [
        "case",
        "epoch",
        "total",
        "pde",
        "bc",
        "diss",
        "ctrl",
        "energy",
        "var",
    ]
]

combined = combined.sort_values(["case", "epoch"]).reset_index(drop=True)


# ====================================================================
# 9. LAST 500 EPOCHS TABLE
# ====================================================================

last500 = combined.groupby("case", group_keys=False).tail(500).copy()


# ====================================================================
# 10. SIMPLE FINAL RESULTS TABLE
# ====================================================================

final_table = summary[
    [
        "case",
        "epoch_final",

        "total_final",
        "pde_final",
        "bc_final",
        "diss_final",
        "ctrl_final",
        "energy_final",
        "var_final",

        "energy_error_abs",
        "energy_error_rel_percent",

        "total_min",
        "epoch_total_min",

        "total_mean_last100",
        "total_std_last100",
        "total_cv_percent_last100",

        "energy_mean_last100",
        "energy_std_last100",
        "energy_cv_percent_last100",

        "var_mean_last100",
        "var_std_last100",
        "var_cv_percent_last100",
    ]
].copy()


# ====================================================================
# 11. ROUND FOR HUMAN-READABLE TABLE
# ====================================================================

final_table_display = final_table.copy()

numeric_cols = final_table_display.select_dtypes(include=[np.number]).columns

final_table_display[numeric_cols] = (
    final_table_display[numeric_cols].round(10)
)


# ====================================================================
# 12. PRINT MAIN TABLE
# ====================================================================

print("\n")
print("=" * 120)
print("TABLEAU PRINCIPAL — MODE COUNT")
print("=" * 120)

print(final_table_display.to_string(index=False))


# ====================================================================
# 13. MORE SCIENTIFIC COMPARISON TABLE
# ====================================================================

scientific_table = summary[
    [
        "case",

        "epoch_final",

        "total_final",
        "total_min",
        "epoch_total_min",

        "pde_final",
        "bc_final",
        "diss_final",
        "ctrl_final",

        "energy_final",
        "energy_error_rel_percent",

        "var_final",

        "total_mean_last500",
        "total_std_last500",
        "total_slope_last500",

        "energy_mean_last500",
        "energy_std_last500",
        "energy_slope_last500",

        "var_mean_last500",
        "var_std_last500",
        "var_slope_last500",
    ]
].copy()

scientific_table = scientific_table.round(12)


print("\n")
print("=" * 120)
print("TABLEAU SCIENTIFIQUE — STABILITE DES 500 DERNIERES EPOCHS")
print("=" * 120)

print(scientific_table.to_string(index=False))


# ====================================================================
# 14. SAVE CSV
# ====================================================================

summary_path = OUT_DIR / "mode_count_summary.csv"
combined_path = OUT_DIR / "mode_count_history_combined.csv"
last500_path = OUT_DIR / "mode_count_last500.csv"
final_path = OUT_DIR / "mode_count_final_table.csv"
scientific_path = OUT_DIR / "mode_count_scientific_table.csv"

summary.to_csv(summary_path, index=False)
combined.to_csv(combined_path, index=False)
last500.to_csv(last500_path, index=False)
final_table.to_csv(final_path, index=False)
scientific_table.to_csv(scientific_path, index=False)


# ====================================================================
# 15. EXCEL OUTPUT
# ====================================================================

excel_path = OUT_DIR / "mode_count_analysis.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    final_table_display.to_excel(
        writer,
        sheet_name="Final_results",
        index=False
    )

    scientific_table.to_excel(
        writer,
        sheet_name="Scientific_summary",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Full_summary",
        index=False
    )

    combined.to_excel(
        writer,
        sheet_name="All_history",
        index=False
    )

    last500.to_excel(
        writer,
        sheet_name="Last500_epochs",
        index=False
    )


# ====================================================================
# 16. FIGURE 1 — TOTAL LOSS
# ====================================================================

plt.figure(figsize=(10, 6))

for case, df in histories.items():
    plt.plot(
        df["epoch"],
        df["total"],
        label=case,
        linewidth=1.5
    )

plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("Total loss — comparison of mode-count parametrizations")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "total_comparison.png",
    dpi=300
)

plt.close()


# ====================================================================
# 17. FIGURE 2 — LOSS COMPONENTS
# ====================================================================

for metric in ["pde", "bc", "diss", "ctrl"]:

    plt.figure(figsize=(10, 6))

    for case, df in histories.items():
        plt.plot(
            df["epoch"],
            df[metric],
            label=case,
            linewidth=1.5
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} loss — comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"{metric}_comparison.png",
        dpi=300
    )

    plt.close()


# ====================================================================
# 18. FIGURE 3 — ENERGY
# ====================================================================

plt.figure(figsize=(10, 6))

for case, df in histories.items():
    plt.plot(
        df["epoch"],
        df["energy"],
        label=case,
        linewidth=1.5
    )

plt.axhline(
    0.25,
    linestyle="--",
    linewidth=1.2,
    label="E target = 0.25"
)

plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Control energy — comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "energy_comparison.png",
    dpi=300
)

plt.close()


# ====================================================================
# 19. FIGURE 4 — VAR
# ====================================================================

plt.figure(figsize=(10, 6))

for case, df in histories.items():
    plt.plot(
        df["epoch"],
        df["var"],
        label=case,
        linewidth=1.5
    )

plt.xlabel("Epoch")
plt.ylabel("Variance loss")
plt.title("Variance loss — comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "var_comparison.png",
    dpi=300
)

plt.close()


# ====================================================================
# 20. FIGURE 5 — LAST 500 EPOCHS
# ====================================================================

plt.figure(figsize=(10, 6))

for case, df in histories.items():

    tail = df.tail(500)

    plt.plot(
        tail["epoch"],
        tail["total"],
        label=case,
        linewidth=1.5
    )

plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("Final convergence — last 500 epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "stabilization_last500.png",
    dpi=300
)

plt.close()


# ====================================================================
# 21. AUTOMATIC INTERPRETATION
# ====================================================================

print("\n")
print("=" * 100)
print("DIAGNOSTIC AUTOMATIQUE")
print("=" * 100)

for _, row in scientific_table.iterrows():

    case = row["case"]

    print(f"\n--- {case} ---")

    print(f"Epoch final                 : {row['epoch_final']}")
    print(f"Total final                 : {row['total_final']:.10f}")
    print(f"PDE final                   : {row['pde_final']:.10f}")
    print(f"BC final                    : {row['bc_final']:.10f}")
    print(f"Diss final                  : {row['diss_final']:.10f}")
    print(f"Ctrl final                  : {row['ctrl_final']:.10f}")
    print(f"Energy final                : {row['energy_final']:.10f}")
    print(f"Erreur énergie (%)          : {row['energy_error_rel_percent']:.6f}")
    print(f"Var final                   : {row['var_final']:.10f}")

    print(
        f"Total moyen dernières 500  : "
        f"{row['total_mean_last500']:.10f}"
    )

    print(
        f"Std total dernières 500    : "
        f"{row['total_std_last500']:.10e}"
    )

    print(
        f"Pente total dernières 500  : "
        f"{row['total_slope_last500']:.10e}"
    )

    print(
        f"Energy moyen dernières 500 : "
        f"{row['energy_mean_last500']:.10f}"
    )

    print(
        f"Std energy dernières 500   : "
        f"{row['energy_std_last500']:.10e}"
    )


# ====================================================================
# 22. RANKING
# ====================================================================

ranking = summary[
    ["case", "total_final", "total_min", "energy_final"]
].sort_values("total_final")

print("\n")
print("=" * 80)
print("CLASSEMENT SELON LA TOTAL LOSS FINALE")
print("=" * 80)

for rank, (_, row) in enumerate(ranking.iterrows(), start=1):
    print(
        f"{rank}. {row['case']:10s} | "
        f"total_final = {row['total_final']:.10f}"
    )


# ====================================================================
# 23. FILES GENERATED
# ====================================================================

print("\n")
print("=" * 100)
print("FICHIERS GENERES")
print("=" * 100)

print(f"Summary CSV       : {summary_path}")
print(f"Final table CSV   : {final_path}")
print(f"Scientific CSV    : {scientific_path}")
print(f"Combined history  : {combined_path}")
print(f"Last 500 epochs   : {last500_path}")
print(f"Excel             : {excel_path}")
print(f"Figures directory : {FIG_DIR}")

print("\nAnalyse terminée.")