#!/usr/bin/env python
"""Export exact in-vitro release curves for the MATLAB figure suite.

The dissolution table stores fitted rate constants, not the curves themselves,
so F10 would otherwise have to reconstruct approximations. This writes the true
Q(t) profiles straight from the digital twin.

Run from the project root:
    python scripts/13_nano_dds/export_release_profiles.py
Output -> tables/nano_dds_release_profiles.csv
"""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
import numpy as np, pandas as pd
from src.models.nano_dds import Formulation, POLYMERS, DRUGS, release_profile

T = np.array([0.083,0.25,0.5,0.75,1,1.5,2,3,4,6,8,10,12,16,20,24,36,48,60,72], float)
rows = []
for d in DRUGS:
    for p in POLYMERS:
        f = Formulation(drug=d, polymer=p, polymer_drug_ratio=0,
                        stabilizer_pct=0, homogenization=0)
        t, Q = release_profile(f, T)
        rows += [dict(drug=d, polymer=p, A_coded=0, B_coded=0, C_coded=0,
                      series="centre", time_h=ti, Q_pct=round(float(qi), 4))
                 for ti, qi in zip(t, Q)]
for lab, (a, b, c) in {"A_low": (-1,0,0), "A_high": (1,0,0), "B_low": (0,-1,0),
                       "B_high": (0,1,0), "C_low": (0,0,-1), "C_high": (0,0,1)}.items():
    f = Formulation(drug="lumefantrine", polymer="PLGA_50_50",
                    polymer_drug_ratio=a, stabilizer_pct=b, homogenization=c)
    t, Q = release_profile(f, T)
    rows += [dict(drug="lumefantrine", polymer="PLGA_50_50", A_coded=a, B_coded=b,
                  C_coded=c, series=lab, time_h=ti, Q_pct=round(float(qi), 4))
             for ti, qi in zip(t, Q)]

out = root / "tables" / "nano_dds_release_profiles.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out} ({len(rows)} rows)")
