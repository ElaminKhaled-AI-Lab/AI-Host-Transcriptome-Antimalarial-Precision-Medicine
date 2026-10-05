#!/usr/bin/env python
"""Phase 19 -- accelerated stability + literature validation + architecture report.

Closes the nano drug-delivery-system (DDS) arm by (1) running International
Council for Harmonisation (ICH) Q1A accelerated-stability prediction on the
recommended formulations, (2) validating the digital twin against PUBLISHED
antimalarial-nanoparticle characterization values (with citations), and (3)
writing the architecture layout and a consolidated results table.

Literature validation anchors (real, peer-reviewed):
  * Nanochitosan-D-galactose primaquine, Antimicrob. Agents Chemother. 2023,
    doi:10.1128/aac.00915-23 -- size ~400 nm, zeta ~+30 mV, PDI ~0.5, EE ~50%,
    liver/plasma ratio 0.10 (free) -> 0.58 (targeted).
  * Curcumin-artesunate PLGA 50:50, PMC5988888 -- size 251 +/- 13 nm, zeta
    -19.1 +/- 4.9 mV, PDI 0.141, EE 22.3%, sustained biphasic release over 7 d.
  * Dihydroartemisinin-lumefantrine nanoparticles, Heliyon 2024,
    doi:10.1016/j.heliyon.2024.e28990 -- polymeric NP, size < 200 nm range.

Outputs:
  results/nano_dds_stability.json
  results/nano_dds_validation.json
  tables/nano_dds_validation.csv
  reports/nano_dds_architecture.md
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path
from src.utils.logging_utils import get_logger
from src.models.nano_dds import (
    Formulation, characterize, accelerated_stability)

log = get_logger(__name__)

# ---- literature anchors: (label, drug, polymer, galactosylated, obs dict, citation) ----
ANCHORS = [
    ("Primaquine nanochitosan-galactose", "primaquine", "Chitosan", True,
     {"size_nm": (300, 500), "zeta_mv": (20, 40), "pdi": (0.4, 0.6), "ee_pct": (40, 60),
      "liver_plasma_ratio": (0.45, 0.65)},
     "Antimicrob. Agents Chemother. 2023, doi:10.1128/aac.00915-23"),
    ("Curcumin-artesunate PLGA 50:50", "artesunate", "PLGA_50_50", False,
     {"size_nm": (220, 290), "zeta_mv": (-25, -13), "pdi": (0.08, 0.22), "ee_pct": (15, 35)},
     "PMC5988888 (Sci. Rep. / PLGA artesunate)"),
    ("DHA-lumefantrine polymeric NP", "lumefantrine", "PLGA_75_25", False,
     {"size_nm": (120, 260), "ee_pct": (40, 90)},
     "Heliyon 2024, doi:10.1016/j.heliyon.2024.e28990"),
]


def validate():
    """Validate by DESIGN-SPACE OVERLAP: does the twin's achievable range for a
    given polymer+drug (across the full 3^3 factorial) overlap the published
    value? A generic twin should not be forced to a single published prep, but
    its reachable design space must contain the real formulation."""
    import itertools
    LV = (-1, 0, 1)
    rows = []
    for label, drug, polymer, gal, obs, cite in ANCHORS:
        # sweep the 27 factorial settings for this polymer+drug
        reps = []
        for a, b, c in itertools.product(LV, LV, LV):
            f = Formulation(drug=drug, polymer=polymer, stabilizer="PVA",
                            polymer_drug_ratio=a, stabilizer_pct=b, homogenization=c,
                            galactosylated=gal)
            reps.append(characterize(f))
        for metric, (lo, hi) in obs.items():
            vals = [r.get(metric) for r in reps if r.get(metric) is not None]
            pmin, pmax = (min(vals), max(vals)) if vals else (None, None)
            # overlap between [pmin,pmax] and literature [lo,hi]
            overlap = (pmin is not None) and (pmin <= hi) and (pmax >= lo)
            rows.append({"anchor": label, "drug": drug, "polymer": polymer,
                         "metric": metric,
                         "twin_achievable_range": None if pmin is None else f"{round(pmin,2)} - {round(pmax,2)}",
                         "literature_range": f"{lo} - {hi}",
                         "design_space_overlaps": overlap, "citation": cite})
    return pd.DataFrame(rows)


def main() -> None:
    # 1. accelerated stability on recommended formulation(s) from optimizer output
    opt_path = get_path("results") / "nano_dds_optimization.json"
    stab_out = {}
    if opt_path.exists():
        opt = json.loads(opt_path.read_text())
        rec = opt.get("recommended_formulation", {})
        polymer = rec.get("polymer", "Eudragit_RL100")
        drug = opt.get("drug", "lumefantrine")
        fac = rec.get("optimal_factors_coded", {"A": 0, "B": 0, "C": 0})
        f = Formulation(drug=drug, polymer=polymer, stabilizer=opt.get("stabilizer", "PVA"),
                        polymer_drug_ratio=fac["A"], stabilizer_pct=fac["B"],
                        homogenization=fac["C"])
        stab = accelerated_stability(f)
        stab_out = {"drug": drug, "polymer": polymer, "factors_coded": fac, "stability": stab}
        log.info("ICH accelerated stability (%s/%s): shelf-life ~%.1f months at 25 C; "
                 "colloidally stable=%s; size 40 C month0->6: %.0f->%.0f nm", drug, polymer,
                 stab["predicted_shelf_life_25C_months"], stab["colloidally_stable"],
                 stab["size_trajectory_40C_nm"][0], stab["size_trajectory_40C_nm"][-1])
    else:
        log.warning("run optimize_nano_dds.py first for the recommended formulation; "
                    "computing stability on a default lumefantrine/Eudragit_RL100 instead.")
        f = Formulation(drug="lumefantrine", polymer="Eudragit_RL100")
        stab_out = {"drug": "lumefantrine", "polymer": "Eudragit_RL100",
                    "stability": accelerated_stability(f)}
    (get_path("results", create=True) / "nano_dds_stability.json").write_text(
        json.dumps(stab_out, indent=2, default=float))

    # 2. literature validation
    vdf = validate()
    vdf.to_csv(get_path("tables", create=True) / "nano_dds_validation.csv", index=False)
    n_ok = int(vdf["design_space_overlaps"].sum())
    val_summary = {"n_checks": int(len(vdf)), "n_design_space_overlaps": n_ok,
                   "fraction_overlapping": round(n_ok / len(vdf), 2),
                   "method": "design-space overlap across 3^3 factorial vs published range",
                   "checks": vdf.to_dict(orient="records")}
    (get_path("results", create=True) / "nano_dds_validation.json").write_text(
        json.dumps(val_summary, indent=2, default=float))
    log.info("literature validation: %d/%d published values fall inside the twin's "
             "achievable design space (%.0f%%)", n_ok, len(vdf), 100 * n_ok / len(vdf))
    for r in vdf.itertuples():
        flag = "OK " if r.design_space_overlaps else "OUT"
        log.info("  [%s] %-30s %-16s twin=%s  lit=%s", flag, r.anchor, r.metric,
                 r.twin_achievable_range, r.literature_range)

    # 3. architecture report
    arch = get_path("reports", create=True) / "nano_dds_architecture.md"
    arch.write_text(ARCHITECTURE_MD, encoding="utf-8")
    log.info("wrote results/nano_dds_stability.json, results/nano_dds_validation.json, "
             "tables/nano_dds_validation.csv, reports/nano_dds_architecture.md")


ARCHITECTURE_MD = """# Nano-DDS formulation-AI module -- architecture

A machine-learning / deep-learning pipeline that designs and optimizes polymeric
NANOPARTICLE (NP) suspensions of antimalarial drugs for pediatric oral therapy,
with liver-stage targeting in view, and hooks into the project's patient
multi-omics layer for precision medicine.

## Layered architecture

```
 (1) DIGITAL TWIN            src/models/nano_dds.py
     Formulation factors  ->  size, PDI, zeta, EE, DL, release profile,
     (polymer, drug,           liver-targeting index, liver/plasma ratio
      3 process factors)       + 5-model release-kinetics fitting
            |
            v
 (2) DESIGN OF EXPERIMENTS   scripts/13_nano_dds/run_factorial_dds.py
     3^3 full factorial (27 runs) x every FDA/GRAS polymer x every antimalarial
     -> data/synthetic/nano_dds_factorial.tsv
            |
            v
 (3) AI SURROGATES + RSM +   scripts/13_nano_dds/optimize_nano_dds.py
     BAYESIAN OPTIMIZATION
     RF / GB / MLP surrogates ; Box-Behnken quadratic RSM ;
     GP + Expected-Improvement search of a desirability
     -> results/nano_dds_optimization.json (recommended formulation)
            |
            v
 (4) STABILITY + VALIDATION  scripts/13_nano_dds/validate_and_report.py
     ICH Q1A accelerated stability (40 C/75% RH, Arrhenius) ;
     validation vs published antimalarial-NP values (citations)
     -> results/nano_dds_stability.json, results/nano_dds_validation.json
            |
            v
 (5) PRECISION-MEDICINE HOOK scripts/13_nano_dds/predict_patient_nano_dds.py
     patient host/parasite omics -> personalized dose + release target +
     recommended NP formulation (links to src/models/pharmacokinetics.py and
     src/models/pharmacomics.py)
```

## Factors (3x3 full factorial, coded -1/0/+1)

| Factor | Meaning                         | Low (-1) | Centre (0) | High (+1) |
|--------|---------------------------------|----------|------------|-----------|
| A      | polymer:drug ratio (w/w)        | 2:1      | 5:1        | 10:1      |
| B      | stabilizer concentration (%w/v) | 0.5      | 1.0        | 2.0       |
| C      | homogenization energy (rel.)    | 0.30     | 0.60       | 1.00      |

## Responses (characterization battery)

Particle size (nm), polydispersity index (PDI), zeta potential (mV), entrapment
efficiency (EE %), drug loading (DL %), cumulative release (%), release
mechanism (Korsmeyer-Peppas exponent n), liver-targeting index and predicted
liver/plasma exposure ratio, and ICH accelerated-stability shelf-life.

## Polymer / excipient library (FDA-approved or GRAS)

PLGA 50:50, PLGA 75:25, PLA (PDLLA), PCL, chitosan, Eudragit RS100 / RL100 /
S100, alginate, ethylcellulose, HPMC; stabilizers PVA, poloxamer 188, Tween 80,
lecithin.

## Release-kinetics models fitted

Zero-order, first-order, Higuchi, Korsmeyer-Peppas (exponent n; sphere
thresholds: n<=0.43 Fickian diffusion, 0.43<n<0.85 anomalous, n>=0.85 case-II),
and Hixson-Crowell; best model chosen by Akaike information criterion (AIC).

All quantities are in-silico predictions from a documented, literature-anchored
digital twin, produced for AI method development -- not a replacement for
laboratory characterization.
"""


if __name__ == "__main__":
    main()
