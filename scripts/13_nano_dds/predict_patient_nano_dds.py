#!/usr/bin/env python
"""Phase 19 -- precision-medicine hook: patient omics -> personalized nano-DDS.

Links the nano drug-delivery-system (DDS) arm to the patient multi-omics layer.
Given a patient's host/parasite molecular proxies (and age/weight), it adjusts the
formulation target and selects, from the 3^3 factorial design space, the
nanoparticle formulation whose predicted behavior best fits that patient, then
estimates exposure through the project pharmacokinetic (PK) model.

Personalization logic (documented):
  * A faster host metabolic phenotype (host_metabolic_z high -> faster clearance)
    shifts the target toward higher entrapment efficiency (EE) and a more
    sustained-release (SR) profile, to hold exposure above the day-7 target.
  * A higher parasite-resistance burden shifts the target toward stronger
    liver-stage targeting and adequate release, to raise local drug pressure.
  * Younger age favors smaller, palatable, well-dispersed particles (low PDI).

Output -> results/patient_nano_dds_recommendation.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.models.nano_dds import (
    Formulation, POLYMERS, characterize, predict_ee, predict_size, predict_zeta,
    liver_targeting_index, release_profile, RATIO_MAP)

log = get_logger(__name__)


def patient_adjusted_desirability(rep, host_metabolic_z, parasite_resistance, age_years):
    """Desirability re-weighted by the patient's omic phenotype."""
    size, ee = rep["size_nm"], rep["ee_pct"]
    rel24, liver = rep["release_24h_pct"], rep["liver_targeting_index"]
    zeta = rep["zeta_mv"]
    # base terms
    d_size = np.exp(-((size - 150) / 60) ** 2)
    d_zeta = 1.0 / (1.0 + np.exp(-(abs(zeta) - 20) / 4))
    # metabolism: faster metabolizers want higher EE + more sustained (lower 24h burst)
    w_ee = 1.0 + 0.5 * max(0.0, host_metabolic_z)
    d_ee = (ee / 100.0) ** (1.0 / w_ee if w_ee else 1.0)
    rel_target = 70 - 15 * max(0.0, host_metabolic_z)          # slower release if fast metabolizer
    d_rel = np.exp(-((rel24 - rel_target) / 25) ** 2)
    # resistance: weight liver targeting more heavily
    w_liver = 1.0 + 1.2 * parasite_resistance
    d_liver = liver ** (1.0 / w_liver if w_liver else 1.0)
    # young age: reward low PDI (uniform, palatable suspension)
    d_pdi = np.exp(-((rep["pdi"] - 0.15) / 0.2) ** 2) if age_years < 5 else 1.0
    return float((d_size * d_ee * d_rel * d_liver * d_zeta * d_pdi) ** (1 / 6))


def pk_for_formulation(drug, rep, weight_kg, age_years, host_metabolic_z):
    """Estimate exposure with the project PK model, using EE + 24 h release as an
    oral bioavailability proxy F, personalized by host metabolic clearance."""
    from src.models.pharmacokinetics import (
        DRUGS as PKDRUGS, scale_params, PKParams, exposure_metrics, cure_probability)
    cls = drug if drug in PKDRUGS else "lumefantrine"
    F = float(np.clip(0.3 + 0.5 * rep["ee_pct"] / 100 + 0.2 * rep["release_24h_pct"] / 100, 0.1, 1.0))
    CL, V, ka, dose = scale_params(cls, weight_kg, age_years, F=F)
    CL = CL * float(np.exp(0.25 * host_metabolic_z))
    pk = exposure_metrics(PKParams(CL=float(CL), V=float(V), ka=float(ka), F=F, dose_mg=float(dose)))
    cure = float(cure_probability(np.array([pk["day7"]]), cls)[0])
    return {"bioavailable_F": round(F, 3), "AUC_ng_h_per_mL": round(pk["auc"], 1),
            "day7_ng_per_mL": round(pk["day7"], 1), "predicted_cure_prob": round(cure, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug", default="lumefantrine")
    ap.add_argument("--age", type=float, default=3.0, help="patient age (years)")
    ap.add_argument("--weight", type=float, default=None, help="patient weight (kg)")
    ap.add_argument("--host-metabolic-z", type=float, default=0.0,
                    help="host metabolic/CYP proxy z-score (higher = faster clearance)")
    ap.add_argument("--parasite-resistance", type=float, default=0.0,
                    help="0-1 parasite resistance burden from parasite omics")
    ap.add_argument("--galactosylated", action="store_true",
                    help="allow galactose-targeted (liver) variants in the search")
    args = ap.parse_args()

    weight = args.weight if args.weight else float(np.clip(2 * args.age + 8, 4, 40))
    log.info("patient: %s, age=%.1f y, weight=%.1f kg, host_metabolic_z=%.2f, "
             "parasite_resistance=%.2f", args.drug, args.age, weight,
             args.host_metabolic_z, args.parasite_resistance)

    # search the full design space (all polymers x 27 runs) for this patient
    import itertools
    LV = (-1, 0, 1)
    best = None
    for polymer in POLYMERS:
        for a, b, c in itertools.product(LV, LV, LV):
            f = Formulation(drug=args.drug, polymer=polymer, stabilizer="PVA",
                            polymer_drug_ratio=a, stabilizer_pct=b, homogenization=c,
                            galactosylated=args.galactosylated)
            rep = characterize(f)
            d = patient_adjusted_desirability(rep, args.host_metabolic_z,
                                              args.parasite_resistance, args.age)
            if best is None or d > best["desirability"]:
                best = {"desirability": round(d, 3), "polymer": polymer,
                        "factors_coded": {"A": a, "B": b, "C": c},
                        "characterization": rep}

    pk = pk_for_formulation(args.drug, best["characterization"], weight, args.age,
                            args.host_metabolic_z)
    out = {"patient": {"drug": args.drug, "age_years": args.age, "weight_kg": weight,
                       "host_metabolic_z": args.host_metabolic_z,
                       "parasite_resistance": args.parasite_resistance,
                       "galactosylated_allowed": args.galactosylated},
           "recommended_formulation": best, "predicted_pharmacokinetics": pk}
    dest = get_path("results", create=True) / "patient_nano_dds_recommendation.json"
    dest.write_text(json.dumps(out, indent=2, default=float))

    rep = best["characterization"]
    log.info("RECOMMENDED for this patient: %s / %s (desirability %.3f)",
             args.drug, best["polymer"], best["desirability"])
    log.info("  size %.0f nm, PDI %.2f, zeta %.0f mV, EE %.0f%%, release24h %.0f%%, "
             "liver idx %.2f, mechanism=%s", rep["size_nm"], rep["pdi"], rep["zeta_mv"],
             rep["ee_pct"], rep["release_24h_pct"], rep["liver_targeting_index"],
             rep["release_mechanism"])
    log.info("  PK: F=%.2f, day7=%.0f ng/mL, cure prob=%.2f",
             pk["bioavailable_F"], pk["day7_ng_per_mL"], pk["predicted_cure_prob"])
    log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
