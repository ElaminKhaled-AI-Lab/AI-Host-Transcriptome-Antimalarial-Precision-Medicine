#!/usr/bin/env python
"""Phase 20 -- personalized formulation for EVERY child in the cohort.

Assigns each child in the paediatric cohort the formulation that maximizes
predicted therapeutic utility (efficacy net of toxicity risk), and quantifies
what personalization buys over a single one-size-fits-all product.

Outputs:
  tables/personalized_formulation_per_patient.csv -- one row per child
  results/personalization_summary.json            -- cohort-level summary
  reports/personalization.md                      -- narrative summary
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def load_dataset():
    syn = get_path("data_synthetic")
    for p in (syn / "patient_formulation_dataset.parquet",
              syn / "patient_formulation_dataset.tsv"):
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, sep="\t")
    raise SystemExit("run build_patient_formulation_dataset.py first")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", default="utility",
                    choices=["utility", "cure_prob", "day7_ng_per_mL"])
    args = ap.parse_args()

    df = load_dataset()
    obj = args.objective

    # one-size-fits-all: the formulation with the best cohort-mean objective
    global_mean = df.groupby("formulation_id")[obj].mean()
    global_best_id = global_mean.idxmax()
    gb = df[df["formulation_id"] == global_best_id].set_index("patient_id")

    # personalized: per-child argmax
    idx = df.groupby("patient_id")[obj].idxmax()
    per = df.loc[idx].copy()
    per["global_best_formulation"] = global_best_id
    per["objective_global_best"] = gb[obj].reindex(per["patient_id"]).to_numpy()
    per["objective_gain"] = per[obj].to_numpy() - per["objective_global_best"].to_numpy()
    per["is_different_from_global"] = per["formulation_id"] != global_best_id

    cols = ["patient_id", "age_years", "weight_kg", "host_metabolic_z", "host_tox_z",
            "parasite_resistance", "formulation_id", "polymer", "run", "A", "B", "C",
            "size_nm", "pdi", "zeta_mv", "ee_pct", "DE_pct", "MDT_h", "t50_h",
            "n_peppas", "liver_idx", "day7_ng_per_mL", "cure_prob", "toxicity_risk",
            "utility", "global_best_formulation", "objective_global_best",
            "objective_gain", "is_different_from_global"]
    per = per[[c for c in cols if c in per.columns]]
    out_tbl = get_path("tables", create=True) / "personalized_formulation_per_patient.csv"
    per.to_csv(out_tbl, index=False)

    summary = {
        "objective": obj,
        "n_children": int(df["patient_id"].nunique()),
        "n_formulations": int(df["formulation_id"].nunique()),
        "one_size_fits_all_formulation": global_best_id,
        "n_distinct_personalized_formulations": int(per["formulation_id"].nunique()),
        "pct_children_differing_from_global": round(
            100 * float(per["is_different_from_global"].mean()), 1),
        "mean_objective_personalized": round(float(per[obj].mean()), 4),
        "mean_objective_one_size": round(float(per["objective_global_best"].mean()), 4),
        "mean_gain": round(float(per["objective_gain"].mean()), 4),
        "max_gain": round(float(per["objective_gain"].max()), 4),
        "polymer_counts": per["polymer"].value_counts().to_dict(),
    }
    (get_path("results", create=True) / "personalization_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    lines = ["# Personalized nano-DDS formulation across the paediatric cohort", "",
             f"- Children: {summary['n_children']}; candidate formulations: {summary['n_formulations']}",
             f"- Objective: {obj} (therapeutic utility = efficacy net of toxicity risk)",
             f"- One-size-fits-all best formulation: `{global_best_id}`",
             f"- Distinct personalized formulations chosen: "
             f"{summary['n_distinct_personalized_formulations']}",
             f"- Children whose optimum differs from the single best product: "
             f"{summary['pct_children_differing_from_global']}%",
             f"- Mean objective: personalized {summary['mean_objective_personalized']} vs "
             f"one-size {summary['mean_objective_one_size']} "
             f"(mean gain {summary['mean_gain']}, max {summary['max_gain']})", "",
             "## Polymer chosen, by number of children", ""]
    for k, v in summary["polymer_counts"].items():
        lines.append(f"- {k}: {v}")
    (get_path("reports", create=True) / "personalization.md").write_text(
        "\n".join(lines), encoding="utf-8")

    log.info("children=%d | one-size best=%s | distinct personalized=%d | differ=%.1f%%",
             summary["n_children"], global_best_id,
             summary["n_distinct_personalized_formulations"],
             summary["pct_children_differing_from_global"])
    log.info("mean %s: personalized %.4f vs one-size %.4f (gain %+.4f, max %+.4f)",
             obj, summary["mean_objective_personalized"], summary["mean_objective_one_size"],
             summary["mean_gain"], summary["max_gain"])
    log.info("polymers chosen: %s", summary["polymer_counts"])
    log.info("wrote %s + results/personalization_summary.json", out_tbl)


if __name__ == "__main__":
    main()
