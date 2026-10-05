#!/usr/bin/env python
"""Phase 20 -- complete dissolution / release characterization for every formulation.

Computes, for all (drug x polymer x 27-run) formulations, the full set of release
parameters a formulation scientist reports -- not only the winning model:

  Kinetic constants + goodness of fit for ALL five models
    zero-order k0 ; first-order k1 ; Higuchi kH ; Korsmeyer-Peppas kKP and n ;
    Hixson-Crowell kHC -- each with R^2, adjusted R^2, RMSE and AIC.
  Model-independent parameters
    dissolution efficiency (DE), mean dissolution time (MDT), t25/t50/t80,
    burst release at 1 h, Q at 24 h, final Q.
  Similarity to a reference carrier
    f1 (difference) and f2 (similarity) factors against the same drug formulated
    in PLGA 50:50 at the centre point -- the conventional polymeric reference --
    with the FDA/EMA equivalence call (f2 >= 50 and f1 <= 15).

Outputs:
  tables/nano_dds_dissolution_parameters.csv   -- one row per formulation, all parameters
  results/nano_dds_dissolution_summary.json    -- per-polymer aggregates
  reports/nano_dds_dissolution.md              -- narrative summary
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
    Formulation, POLYMERS, DRUGS, release_profile, predict_size, predict_pdi,
    predict_zeta, predict_ee, predict_dl, liver_targeting_index, liver_plasma_ratio,
    RATIO_MAP, STAB_MAP, ENERGY_MAP)
from src.models.dissolution_metrics import full_dissolution_report, f1_f2

log = get_logger(__name__)

# Dense sampling grid so DE / MDT / t50 are accurate (h)
TIMES = np.array([0.083, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12,
                  16, 20, 24, 36, 48, 60, 72], float)
LEVELS = (-1, 0, 1)
REFERENCE_POLYMER = "PLGA_50_50"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drugs", nargs="*", default=list(DRUGS))
    ap.add_argument("--polymers", nargs="*", default=list(POLYMERS))
    ap.add_argument("--stabilizer", default="PVA")
    ap.add_argument("--replicates", type=int, default=3,
                    help="replicate batches per run (a real DoE is run in replicate)")
    ap.add_argument("--noise-rsd", type=float, default=0.05,
                    help="relative standard deviation of batch-to-batch variability; "
                         "0 gives the noiseless deterministic twin")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # reference profile per drug: PLGA 50:50 at the centre point
    ref_profiles = {}
    for drug in args.drugs:
        fref = Formulation(drug=drug, polymer=REFERENCE_POLYMER, stabilizer=args.stabilizer,
                           polymer_drug_ratio=0, stabilizer_pct=0, homogenization=0)
        _, Qref = release_profile(fref, TIMES)
        ref_profiles[drug] = Qref

    rows = []
    for drug in args.drugs:
        for polymer in args.polymers:
            for run_id, (a, b, c) in enumerate(itertools.product(LEVELS, LEVELS, LEVELS), 1):
                f = Formulation(drug=drug, polymer=polymer, stabilizer=args.stabilizer,
                                polymer_drug_ratio=a, stabilizer_pct=b, homogenization=c)
                t, Q0 = release_profile(f, TIMES)
                for rep_i in range(1, args.replicates + 1):
                  # batch-to-batch variability: multiplicative lognormal noise on the
                  # characterization and on the release curve, as seen between real batches
                  if args.noise_rsd > 0:
                      jit = lambda v: float(v * rng.lognormal(0, args.noise_rsd))
                      Q = np.clip(Q0 * rng.lognormal(0, args.noise_rsd, size=Q0.shape), 0, 100)
                      Q = np.maximum.accumulate(Q)   # cumulative release must not decrease
                  else:
                      jit = lambda v: float(v)
                      Q = Q0
                  rep = full_dissolution_report(t, Q, Q_reference=ref_profiles[drug])
                  m = rep["models"]
                  sim = rep.get("similarity_vs_reference", {})
                  rows.append({
                    "drug": drug, "polymer": polymer, "run": run_id, "replicate": rep_i,
                    "A_ratio": RATIO_MAP[a], "B_stabilizer_pct": STAB_MAP[b],
                    "C_energy": ENERGY_MAP[c],
                    # characterization
                    "size_nm": round(jit(predict_size(f)), 1),
                    "pdi": round(jit(predict_pdi(f)), 3),
                    "zeta_mv": round(predict_zeta(f) * (1 + rng.normal(0, args.noise_rsd)), 1)
                                if args.noise_rsd > 0 else round(predict_zeta(f), 1),
                    "ee_pct": round(min(jit(predict_ee(f)), 99.0), 1),
                    "dl_pct": round(jit(predict_dl(f)), 2),
                    "liver_targeting_index": round(liver_targeting_index(f), 3),
                    "liver_plasma_ratio": liver_plasma_ratio(f),
                    # model-independent dissolution parameters
                    "DE_pct": rep["DE_pct"], "MDT_h": rep["MDT_h"],
                    "t25_h": rep["t25_h"], "t50_h": rep["t50_h"], "t80_h": rep["t80_h"],
                    "burst_1h_pct": rep["burst_1h_pct"], "Q_24h_pct": rep["Q_24h_pct"],
                    "Q_final_pct": rep["Q_final_pct"],
                    # kinetic constants + fits
                    "k0_zero_order": m["zero_order"]["k0_pct_per_h"],
                    "r2_zero_order": m["zero_order"]["r2"],
                    "k1_first_order": m["first_order"]["k1_per_h"],
                    "r2_first_order": m["first_order"]["r2"],
                    "kH_higuchi": m["higuchi"]["kH_pct_per_sqrt_h"],
                    "r2_higuchi": m["higuchi"]["r2"],
                    "kKP_peppas": m["korsmeyer_peppas"].get("kKP_per_h^n"),
                    "n_peppas": m["korsmeyer_peppas"].get("n"),
                    "r2_peppas": m["korsmeyer_peppas"].get("r2"),
                    "kHC_hixson": m["hixson_crowell"]["kHC_per_h"],
                    "r2_hixson": m["hixson_crowell"]["r2"],
                    "best_model_aic": rep["best_model_by_aic"],
                    "best_model_r2": rep["best_model_by_r2"],
                    "release_mechanism": rep["release_mechanism"],
                    # similarity to reference carrier
                    "f1_vs_PLGA_ref": sim.get("f1_difference"),
                    "f2_vs_PLGA_ref": sim.get("f2_similarity"),
                    "equivalent_to_ref": sim.get("equivalent"),
                    "release_class": POLYMERS[polymer]["release_class"],
                })

    df = pd.DataFrame(rows)
    tbl = get_path("tables", create=True) / "nano_dds_dissolution_parameters.csv"
    df.to_csv(tbl, index=False)

    # per-polymer aggregates
    agg = df.groupby("polymer").agg(
        DE_pct=("DE_pct", "mean"), MDT_h=("MDT_h", "mean"), t50_h=("t50_h", "mean"),
        t80_h=("t80_h", "mean"), burst_1h=("burst_1h_pct", "mean"),
        n_peppas=("n_peppas", "mean"), f2_vs_ref=("f2_vs_PLGA_ref", "mean")).round(3)
    summary = {
        "n_formulations": int(len(df)),
        "time_points_h": TIMES.tolist(),
        "reference_carrier": f"{REFERENCE_POLYMER} centre point (per drug)",
        "per_polymer": agg.to_dict(orient="index"),
        "mechanism_counts": df["release_mechanism"].value_counts().to_dict(),
        "best_model_counts_aic": df["best_model_aic"].value_counts().to_dict(),
        "n_equivalent_to_reference": int(df["equivalent_to_ref"].fillna(False).sum()),
    }
    (get_path("results", create=True) / "nano_dds_dissolution_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    lines = ["# Dissolution / release characterization -- all formulations", "",
             f"- Formulations analyzed: {len(df)}",
             f"- Sampling grid: {len(TIMES)} time points, {TIMES[0]}-{TIMES[-1]} h",
             f"- Reference carrier for f1/f2: {REFERENCE_POLYMER} centre point (per drug)",
             f"- Formulations equivalent to reference (f2>=50, f1<=15): "
             f"{summary['n_equivalent_to_reference']}", "",
             "## Mean dissolution parameters by polymer", "",
             "| Polymer | DE (%) | MDT (h) | t50 (h) | t80 (h) | burst 1h (%) | K-P n | f2 vs ref |",
             "|---|---|---|---|---|---|---|---|"]
    for p, r in agg.iterrows():
        lines.append(f"| {p} | {r.DE_pct:.1f} | {r.MDT_h:.2f} | {r.t50_h:.2f} | "
                     f"{r.t80_h:.2f} | {r.burst_1h:.1f} | {r.n_peppas:.3f} | {r.f2_vs_ref:.1f} |")
    lines += ["", "## Release mechanism distribution", ""]
    for k, v in summary["mechanism_counts"].items():
        lines.append(f"- {k}: {v}")
    (get_path("reports", create=True) / "nano_dds_dissolution.md").write_text(
        "\n".join(lines), encoding="utf-8")

    log.info("dissolution analysis: %d formulations, %d time points", len(df), len(TIMES))
    log.info("DE range %.1f-%.1f%% | MDT %.2f-%.2f h | t50 %.2f-%.2f h",
             df.DE_pct.min(), df.DE_pct.max(), df.MDT_h.min(), df.MDT_h.max(),
             df.t50_h.min(), df.t50_h.max())
    log.info("K-P n range %.2f-%.2f | mechanisms: %s",
             df.n_peppas.min(), df.n_peppas.max(), summary["mechanism_counts"])
    log.info("best model by AIC: %s", summary["best_model_counts_aic"])
    log.info("wrote %s + results/nano_dds_dissolution_summary.json + reports/nano_dds_dissolution.md", tbl)


if __name__ == "__main__":
    main()
