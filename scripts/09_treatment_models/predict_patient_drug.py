#!/usr/bin/env python
"""Phase 16 -- patient x drug pharmaco-omics prediction (Sections 6, 60, 68).

Two modes:
  single  : one patient + one drug + dosage form -> full ADMET/PK/response report.
  batch   : cohort patients x a drug panel -> an in-silico response/ADMET/PK
            dataset, then train AI surrogates that predict each endpoint from
            patient omics + drug descriptors and report the driving features.

Drug descriptors are read from the PubChem antimalarial file
(data/raw/formulation/PubChem/antimalarial_physicochemical.json); pass --drug to
pick one by name. For an arbitrary preclinical candidate, supply descriptors on
the command line (--mw --logp --tpsa --hbd --hba).

Outputs:
  single -> results/patient_drug_report.json
  batch  -> results/pharmacomics_surrogate.json  (+ dataset in data/synthetic/)
"""
import _bootstrap  # noqa: F401
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_path, dataset_dest, load_config
from src.utils.logging_utils import get_logger
from src.models.pharmacomics import Patient, Drug, predict

log = get_logger(__name__)


def _num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_drugs():
    """Load antimalarial descriptors from PubChem; map to Drug objects."""
    p = dataset_dest("reference_pubchem", create=False) / "antimalarial_physicochemical.json"
    drugs = {}
    if p.exists():
        for e in json.loads(p.read_text()):
            name = str(e.get("query_name") or e.get("Title") or "drug").lower()
            drugs[name] = Drug(
                name=name, mw=_num(e.get("MolecularWeight"), 400),
                logp=_num(e.get("XLogP"), 3.0), tpsa=_num(e.get("TPSA"), 90),
                hbd=int(_num(e.get("HBondDonorCount"), 2)),
                hba=int(_num(e.get("HBondAcceptorCount"), 5)),
                complexity=_num(e.get("Complexity"), 400),
                rotatable=int(_num(e.get("RotatableBondCount"), 5)),
                pk_class=("piperaquine" if "piperaquine" in name else
                          "dihydroartemisinin" if ("artemisinin" in name or "artemether" in name
                                                   or "artesunate" in name or "dihydro" in name)
                          else "lumefantrine"))
    return drugs


def patient_from_cohort(df, i):
    age = _num(df.get("clin_age", pd.Series([3.0])).iloc[i], 3.0)
    age = float(np.clip(age, 0.3, 12))
    weight = float(np.clip(2 * age + 8, 4, 40))
    def z(col):
        s = pd.to_numeric(df.get(col), errors="coerce")
        if s is None or s.isna().all():
            return 0.0
        val = (s.iloc[i] - s.mean()) / (s.std() + 1e-9)
        return float(np.nan_to_num(val, nan=0.0))
    # parasite resistance burden: mean of parasite feature columns, squashed to 0..1
    para = df[[c for c in df.columns if c.startswith("para_")]]
    res = 0.0
    if para.shape[1]:
        pmat = para.apply(pd.to_numeric, errors="coerce")
        v = np.nan_to_num(pmat.iloc[i].mean(), nan=float(np.nanmean(pmat.values)))
        gmean = float(np.nanmean(pmat.values))
        res = float(np.nan_to_num(1 / (1 + np.exp(-(v - gmean))), nan=0.0))
    return Patient(age_years=age, weight_kg=weight, host_metabolic_z=z("tx_PC1"),
                   host_tox_z=z("tx_PC2"), parasite_resistance=res)


def flatten(report):
    out = {}
    for section, d in report.items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    out[f"{section}.{k}"] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--drug", default="lumefantrine")
    ap.add_argument("--patient", type=int, default=0)
    ap.add_argument("--cohort-seed", type=int, default=11)
    ap.add_argument("--dissolution", type=float, default=None, help="dosage-form dissolution %")
    ap.add_argument("--mw", type=float); ap.add_argument("--logp", type=float)
    ap.add_argument("--tpsa", type=float); ap.add_argument("--hbd", type=int)
    ap.add_argument("--hba", type=int)
    ap.add_argument("--n-patients", type=int, default=300, help="batch: patients to sample")
    args = ap.parse_args()

    syn = get_path("data_synthetic")
    p = syn / f"cohort_seed{args.cohort_seed}.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(
        syn / f"cohort_seed{args.cohort_seed}.tsv", sep="\t")
    drugs = load_drugs()

    if args.mw and args.logp:   # arbitrary preclinical candidate from CLI
        drugs[args.drug] = Drug(name=args.drug, mw=args.mw, logp=args.logp,
                                tpsa=args.tpsa or 90, hbd=args.hbd or 2, hba=args.hba or 5)

    if args.mode == "single":
        drug = drugs.get(args.drug.lower())
        if not drug:
            log.error("drug '%s' not found; available: %s (or pass --mw --logp ...)",
                      args.drug, list(drugs)[:8]); return
        pat = patient_from_cohort(df, args.patient)
        rep = predict(pat, drug, dissolution_pct=args.dissolution)
        out = get_path("results", create=True) / "patient_drug_report.json"
        out.write_text(json.dumps({"patient_index": args.patient,
                                   "patient": pat.__dict__, "prediction": rep}, indent=2))
        log.info("PATIENT %d x %s (dissolution=%s):", args.patient, drug.name, args.dissolution)
        for section, d in rep.items():
            if isinstance(d, dict):
                log.info("  %-16s %s", section, {k: v for k, v in d.items()})
        log.info("wrote %s", out)
        return

    # ---- batch: build patient x drug dataset + AI surrogate --------------
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    n = min(args.n_patients, len(df))
    diss = args.dissolution
    rows = []
    for i in range(n):
        pat = patient_from_cohort(df, i)
        for name, drug in drugs.items():
            rep = flatten(predict(pat, drug, dissolution_pct=diss))
            rep.update(drug=name, mw=drug.mw, logp=drug.logp, tpsa=drug.tpsa,
                       hbd=drug.hbd, hba=drug.hba,
                       pat_age=pat.age_years, pat_weight=pat.weight_kg,
                       pat_host_metabolic=pat.host_metabolic_z,
                       pat_host_tox=pat.host_tox_z, pat_resistance=pat.parasite_resistance)
            rows.append(rep)
    data = pd.DataFrame(rows)
    ds_path = syn / "pharmacomics_dataset.tsv"
    data.to_csv(ds_path, sep="\t", index=False)
    log.info("built pharmaco-omics dataset: %d patient x drug rows -> %s", len(data), ds_path)

    feats = ["mw", "logp", "tpsa", "hbd", "hba", "pat_age", "pat_weight",
             "pat_host_metabolic", "pat_host_tox", "pat_resistance"]
    targets = {"response": "response.resistance_adjusted_response",
               "bioavailability": "bioavailability.F_estimate",
               "day7_exposure": "pharmacokinetics.day7_ng_per_mL",
               "hepatotoxicity": "toxicity.hepatotoxicity_risk"}
    seed = load_config()["reproducibility"]["global_seed"]
    # NaN-safe feature matrix (parasite genotype NaNs, missing descriptors -> median/0)
    Xmat = data[feats].apply(pd.to_numeric, errors="coerce")
    Xmat = Xmat.fillna(Xmat.median(numeric_only=True)).fillna(0.0)
    surro = {}
    for label, col in targets.items():
        if col not in data:
            continue
        y = pd.to_numeric(data[col], errors="coerce").fillna(0).to_numpy()
        m = GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=seed)
        pred = cross_val_predict(m, Xmat, y, cv=4)
        m.fit(Xmat, y)
        imp = dict(sorted(zip(feats, m.feature_importances_), key=lambda kv: -kv[1]))
        surro[label] = {"cv_r2": float(r2_score(y, pred)),
                        "top_features": {k: round(float(v), 3) for k, v in list(imp.items())[:4]}}
        log.info("surrogate %-14s CV R2=%.3f  top: %s",
                 label, surro[label]["cv_r2"], list(surro[label]["top_features"]))
    out = get_path("results", create=True) / "pharmacomics_surrogate.json"
    out.write_text(json.dumps(surro, indent=2))
    log.info("wrote %s", out)
    log.info("The AI surrogate learns patient-omics x drug -> full-spectrum endpoints, "
             "for fast personalized screening of any candidate + dosage form.")


if __name__ == "__main__":
    main()
