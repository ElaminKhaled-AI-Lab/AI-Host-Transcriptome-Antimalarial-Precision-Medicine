#!/usr/bin/env python
"""Phase 19 -- Full-factorial 3^3 nanoparticle-DDS design + release-kinetics fitting.

Runs a three-factor, three-level full factorial (3^3 = 27 runs) over the nano
drug-delivery-system (DDS) digital twin (src/models/nano_dds.py), for every
FDA-approved / GRAS polymer in the library and every antimalarial drug requested.
Each run is fully characterized -- particle size, polydispersity index (PDI),
zeta potential (ZP), entrapment efficiency (EE), drug loading (DL), 24 h release,
liver-targeting index -- and its release profile is fitted to five classical
kinetic models (zero-order, first-order, Higuchi, Korsmeyer-Peppas, Hixson-
Crowell) to report the rate constant, the best model, the K-P exponent n and the
transport mechanism.

The three factors (coded -1/0/+1):
  A  polymer:drug ratio (w/w)        2:1 / 5:1 / 10:1
  B  stabilizer concentration (%w/v) 0.5 / 1.0 / 2.0
  C  homogenization energy (rel.)    0.30 / 0.60 / 1.00

Outputs (all in-silico, clearly labelled; for method development):
  data/synthetic/nano_dds_factorial.tsv   -- the full run x response dataset
  results/nano_dds_factorial_summary.json -- design metadata + response ranges
  tables/nano_dds_factorial.csv           -- human-readable table
  reports/nano_dds_factorial.md           -- narrative summary
"""
import _bootstrap  # noqa: F401
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path
from src.utils.logging_utils import get_logger
from src.models.nano_dds import (
    Formulation, POLYMERS, DRUGS, characterize, RATIO_MAP, STAB_MAP, ENERGY_MAP)

log = get_logger(__name__)

LEVELS = (-1, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drugs", nargs="*", default=["lumefantrine", "artemether",
                    "primaquine", "artesunate", "dihydroartemisinin", "atovaquone",
                    "piperaquine", "amodiaquine", "mefloquine"],
                    help="antimalarial drugs to sweep (default: all in library)")
    ap.add_argument("--polymers", nargs="*", default=list(POLYMERS),
                    help="polymers to sweep (default: full FDA/GRAS library)")
    ap.add_argument("--stabilizer", default="PVA")
    ap.add_argument("--galactosylated", action="store_true",
                    help="add liver-targeting galactose ligand to every run")
    args = ap.parse_args()

    rows = []
    # full factorial 3^3 over (A,B,C)
    design = list(itertools.product(LEVELS, LEVELS, LEVELS))
    assert len(design) == 27, "3^3 factorial must be 27 runs"

    for drug in args.drugs:
        for polymer in args.polymers:
            for run_id, (a, b, c) in enumerate(design, 1):
                f = Formulation(drug=drug, polymer=polymer, stabilizer=args.stabilizer,
                                polymer_drug_ratio=a, stabilizer_pct=b, homogenization=c,
                                galactosylated=args.galactosylated)
                rep = characterize(f)
                rows.append({
                    "run": run_id, "drug": drug, "polymer": polymer,
                    "stabilizer": args.stabilizer,
                    "A_polymer_drug_ratio_coded": a, "B_stabilizer_pct_coded": b,
                    "C_homogenization_coded": c,
                    "polymer_drug_ratio": RATIO_MAP[a], "stabilizer_pct": STAB_MAP[b],
                    "homogenization_energy": ENERGY_MAP[c],
                    "size_nm": rep["size_nm"], "pdi": rep["pdi"], "zeta_mv": rep["zeta_mv"],
                    "ee_pct": rep["ee_pct"], "dl_pct": rep["dl_pct"],
                    "release_24h_pct": rep["release_24h_pct"],
                    "release_best_model": rep["release_best_model"],
                    "peppas_n": rep["peppas_n"], "release_mechanism": rep["release_mechanism"],
                    "liver_targeting_index": rep["liver_targeting_index"],
                    "liver_plasma_ratio": rep["liver_plasma_ratio"],
                    "release_class": rep["release_class"],
                })

    df = pd.DataFrame(rows)
    syn = get_path("data_synthetic", create=True)
    out_tsv = syn / "nano_dds_factorial.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    tbl = get_path("tables", create=True) / "nano_dds_factorial.csv"
    df.to_csv(tbl, index=False)

    resp_cols = ["size_nm", "pdi", "zeta_mv", "ee_pct", "dl_pct",
                 "release_24h_pct", "liver_targeting_index", "liver_plasma_ratio"]
    summary = {
        "design": "3^3 full factorial (27 runs) per polymer x drug",
        "n_runs_per_cell": 27,
        "n_polymers": df["polymer"].nunique(),
        "n_drugs": df["drug"].nunique(),
        "n_total_rows": int(len(df)),
        "factors": {"A": "polymer:drug ratio (2/5/10)", "B": "stabilizer % (0.5/1/2)",
                    "C": "homogenization energy (0.3/0.6/1.0)"},
        "response_ranges": {c: {"min": float(df[c].min()), "median": float(df[c].median()),
                                "max": float(df[c].max())} for c in resp_cols},
        "release_mechanism_counts": df["release_mechanism"].value_counts().to_dict(),
        "best_model_counts": df["release_best_model"].value_counts().to_dict(),
    }
    (get_path("results", create=True) / "nano_dds_factorial_summary.json").write_text(
        json.dumps(summary, indent=2))

    # narrative report
    lines = ["# Nano-DDS full-factorial (3^3) design -- in-silico digital twin", "",
             f"- Design: 3x3x3 full factorial = 27 runs per polymer x drug",
             f"- Polymers swept: {df['polymer'].nunique()} ({', '.join(sorted(df['polymer'].unique()))})",
             f"- Drugs swept: {df['drug'].nunique()} ({', '.join(sorted(df['drug'].unique()))})",
             f"- Total formulations characterized: {len(df)}", "",
             "## Response ranges (all runs)", ""]
    for c in resp_cols:
        r = summary["response_ranges"][c]
        lines.append(f"- {c}: {r['min']:.2f} -- {r['max']:.2f} (median {r['median']:.2f})")
    lines += ["", "## Release mechanism distribution (Korsmeyer-Peppas)", ""]
    for k, v in summary["release_mechanism_counts"].items():
        lines.append(f"- {k}: {v} runs")
    lines += ["", "All values are in-silico predictions from a documented, literature-",
              "anchored digital twin, generated for AI method development -- not a",
              "substitute for laboratory characterization."]
    (get_path("reports", create=True) / "nano_dds_factorial.md").write_text(
        "\n".join(lines), encoding="utf-8")

    log.info("factorial: %d rows (%d polymers x %d drugs x 27 runs)",
             len(df), df["polymer"].nunique(), df["drug"].nunique())
    log.info("size %.0f-%.0f nm | zeta %.0f..%.0f mV | EE %.0f-%.0f%% | release24h %.0f-%.0f%%",
             df.size_nm.min(), df.size_nm.max(), df.zeta_mv.min(), df.zeta_mv.max(),
             df.ee_pct.min(), df.ee_pct.max(), df.release_24h_pct.min(), df.release_24h_pct.max())
    log.info("mechanisms: %s", summary["release_mechanism_counts"])
    log.info("wrote %s , %s , tables/nano_dds_factorial.csv , reports/nano_dds_factorial.md",
             out_tsv, "results/nano_dds_factorial_summary.json")


if __name__ == "__main__":
    main()
