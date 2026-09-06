import os
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(base, "parametrization_results.csv"))

raw["dominant"] = raw["dominant_i"].astype(str) + "," + raw["dominant_j"].astype(str)

metrics = ["E_total", "mode2_fraction", "temporal_fraction", "A20", "native_c20"]
piv = pd.DataFrame({"Re": [100, 500, 1000]}).set_index("Re")

for m in metrics:
    wide = raw.pivot_table(index="Re", columns="basis_type", values=m)
    for basis in wide.columns:
        piv[f"{m}_{basis}"] = wide[basis]
    piv[f"{m}_delta_cheb_minus_fourier"] = wide["chebyshev_mod"] - wide["fourier"]

dom = raw.pivot_table(index="Re", columns="basis_type", values="dominant", aggfunc="first")
for basis in dom.columns:
    piv[f"dominant_{basis}"] = dom[basis]

piv["E_overshoot_vs_target_percent_fourier"] = (piv["E_total_fourier"] - 0.25) / 0.25 * 100
piv["E_overshoot_vs_target_percent_chebyshev"] = (piv["E_total_chebyshev_mod"] - 0.25) / 0.25 * 100

out = piv.reset_index()
out.to_csv(os.path.join(base, "parametrization_matrix_pivot.csv"), index=False,
           float_format="%.6f")
print(out.round(6).to_string(index=False))
print("\nWROTE:", os.path.join(base, "parametrization_matrix_pivot.csv"))