# -*- coding: utf-8 -*-
"""
Aspect-ratio AR=0.5:1 extra campaign (R3.2 geometry sensitivity).
Re=500, E*=0.25, seed 42, baseline architecture, Fourier 6x5 basis.
Same protocol as run_aspect_ratio(): train_ultra + analyze_control,
then append the row to aspect_ratio_results.csv (header preserved).
"""
import os
import sys
import csv
import json
import types

if "pysr" not in sys.modules:
    _pysr = types.ModuleType("pysr")
    class _PySR:
        def __init__(self, *a, **k):
            raise RuntimeError("pysr stubbed out (not needed for this campaign)")
    _pysr.PySRRegressor = _PySR
    sys.modules["pysr"] = _pysr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import PINN_Lid_driven_reviewers as P

LX_VAL, LY_VAL = 0.5, 1.0
GEO = "short_rect"

out_dir = os.path.join(P.OUT, "09_aspect_ratio", GEO)
os.makedirs(out_dir, exist_ok=True)

P.set_seed(42)
model, hist, _ = P.train_ultra(re_list=[500], lx=LX_VAL, ly=LY_VAL, out_dir=out_dir)

ac = P.analyze_control(model, Re=500, lx=LX_VAL, ly=LY_VAL)
ac["geometry"] = GEO
ac["lx"] = LX_VAL
ac["ly"] = LY_VAL
ac["energy_final"] = ac["E_total"]

P.save_metadata(out_dir, extra={"campaign": "aspect_ratio", "geometry": GEO,
                                "lx": LX_VAL, "ly": LY_VAL, "seed": 42})

csv_path = os.path.join(os.path.dirname(out_dir), "aspect_ratio_results.csv")
with open(csv_path, newline="") as f:
    rd = csv.DictReader(f)
    fieldnames = list(rd.fieldnames)
    old = [dict(r) for r in rd]

missing = [k for k in ac if k not in fieldnames]
if missing:
    fieldnames = fieldnames + missing

newrow = {k: (ac[k] if k in ac else "") for k in fieldnames}
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in old:
        w.writerow(r)
    w.writerow(newrow)

print("\n==== AR=0.5:1 RESULT ====")
summary = {k: ac[k] for k in
           ("Re", "E_total", "A10", "A20", "A30", "mode2_fraction",
            "temporal_fraction", "dominant_i", "dominant_j", "basis_type")}
print(json.dumps(summary, indent=1))
print(f"E_total={ac['E_total']:.4f}  A2={ac['A20']:.4f}  "
      f"f2={ac['mode2_fraction']*100:.2f}%  ft={ac['temporal_fraction']*100:.2f}%")
print("CSV updated:", csv_path)