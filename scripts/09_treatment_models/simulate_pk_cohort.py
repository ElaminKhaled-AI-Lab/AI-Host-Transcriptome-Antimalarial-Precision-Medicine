#!/usr/bin/env python
"""Phase 16 -- simulate an antimalarial PK cohort (blueprint Sections 31, 35.9, 68).

Combines each synthetic patient (age, host molecular state) with a candidate
formulation (dissolution -> bioavailability) and a drug to simulate plasma
exposure and outcome. It produces the dataset the AI PK model learns from, and
it is the exposure bridge linking the host/parasite arm to the formulation arm.

Declared mechanistic assumptions (fixed here, before modelling):
  * weight is estimated from age with a paediatric weight-for-age relation;
  * bioavailable fraction F rises with in-vitro dissolution (formulation -> PK);
  * a host covariate (first transcriptome component, a proxy for metabolic /
    inflammatory state) modulates clearance, so host molecular state genuinely
    affects exposure -- the signal the AI PK model can exploit beyond allometry.

To give the AI formulation signal to learn, each patient is paired with several
sampled formulations (a patient x formulation design), i.e. "given this child
and this formulation, predict exposure".

Output -> data/synthetic/pk_cohort.tsv
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.models.pharmacokinetics import simulate_population, cure_probability, DRUGS

log = get_logger(__name__)


def weight_for_age(age_years):
    """Paediatric weight (kg) ~ 2*age + 8 (WHO-ish, ages 1-10), floored."""
    return np.clip(2.0 * np.asarray(age_years, float) + 8.0, 4.0, 40.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-seed", type=int, default=11)
    ap.add_argument("--drug", default="lumefantrine", choices=list(DRUGS))
    ap.add_argument("--forms-per-patient", type=int, default=3,
                    help="formulations sampled per patient (gives formulation signal)")
    ap.add_argument("--host-cl-beta", type=float, default=0.25,
                    help="strength of host (tx_PC1) effect on clearance")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]
    rng = np.random.default_rng(seed)

    syn = get_path("data_synthetic")
    p = syn / f"cohort_seed{args.cohort_seed}.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(
        syn / f"cohort_seed{args.cohort_seed}.tsv", sep="\t")
    age = pd.to_numeric(df.get("clin_age"), errors="coerce").fillna(3.0).to_numpy()
    age = np.clip(age, 0.3, 12.0)
    sexM = pd.to_numeric(df.get("clin_sex_M", 0), errors="coerce").fillna(0).to_numpy()
    txpc1 = df.get("tx_PC1")
    txpc1 = (pd.to_numeric(txpc1, errors="coerce").fillna(0).to_numpy()
             if txpc1 is not None else np.zeros(len(df)))
    txz = (txpc1 - txpc1.mean()) / (txpc1.std() + 1e-9)      # host covariate

    # formulations (dissolution, particle size) from the in-silico DoE
    doe = get_path("data_raw") / "formulation" / "in_silico_doe" / "doe_dataset.tsv"
    if doe.exists():
        forms = pd.read_csv(doe, sep="\t")
    else:
        log.warning("no in-silico DoE found; sampling generic formulations.")
        forms = pd.DataFrame({"dissolution_pct_30min": rng.uniform(30, 100, 200),
                              "particle_size_d50_um": rng.uniform(2, 50, 200)})

    rows = []
    for i in range(len(df)):
        fidx = rng.integers(0, len(forms), args.forms_per_patient)
        diss = forms["dissolution_pct_30min"].to_numpy()[fidx]
        d50 = forms.get("particle_size_d50_um", pd.Series(np.nan, index=forms.index)).to_numpy()[fidx]
        F = np.clip(0.35 + 0.6 * diss / 100.0, 0.1, 1.0)     # dissolution -> F
        wt = weight_for_age(np.full(args.forms_per_patient, age[i]))
        ag = np.full(args.forms_per_patient, age[i])
        exp = simulate_population(args.drug, wt, ag, F=F, seed=seed + i)
        # host effect on clearance -> re-scale exposures (AUC ~ 1/CL)
        host_factor = np.exp(args.host_cl_beta * txz[i])
        auc = exp["auc"] / host_factor
        day7 = exp["day7"] / host_factor
        cmax = exp["cmax"] / host_factor
        cure_p = cure_probability(day7, args.drug)
        cure = (rng.random(args.forms_per_patient) < cure_p).astype(int)
        for j in range(args.forms_per_patient):
            rows.append(dict(
                patient_id=f"P{args.cohort_seed}_{i:05d}", form_rep=j,
                drug=args.drug, age_years=float(ag[j]), weight_kg=float(wt[j]),
                sex_M=float(sexM[i]), host_txPC1=float(txz[i]),
                dissolution_pct_30min=float(diss[j]), particle_size_d50_um=float(d50[j]),
                bioavailable_F=float(F[j]), CL=float(exp["CL"][j]),
                auc=float(auc[j]), cmax=float(cmax[j]), day7=float(day7[j]),
                cure_prob=float(cure_p[j]), cure=int(cure[j])))
    out = pd.DataFrame(rows)
    dest = syn / "pk_cohort.tsv"
    out.to_csv(dest, sep="\t", index=False)
    log.info("WROTE PK cohort: %d patient x formulation records -> %s", len(out), dest)
    log.info("drug=%s  median day7=%.1f ng/mL (target %s)  cure rate=%.2f",
             args.drug, np.median(out["day7"]), DRUGS[args.drug]["day7_target"],
             out["cure"].mean())
    log.info("Next: pk_ai_pipeline.py -- learn exposure from host+drug+formulation.")


if __name__ == "__main__":
    main()
