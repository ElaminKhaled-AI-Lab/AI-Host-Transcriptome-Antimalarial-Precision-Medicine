#!/usr/bin/env python
"""Revision analysis for R2.6b / R2.8b-2: uncertainty and nulls for the personalization gain.

The personalization gain is an ORACLE quantity: each child is assigned the formulation
that maximizes the simulator's own utility, and the gain is measured against the single
formulation with the best cohort-mean utility. This script reproduces the simulator in
vectorized form (verified against the stored dataset), then reports:

  1. The gain under the outcome model as submitted (hepatic term on) and as revised
     for lumefantrine (hepatic term off), with a 95% bootstrap CI over children
     (B = 2000; the one-size product is re-selected inside every resample).
  2. A molecular permutation null: the three molecular covariates (host PC1, host PC2,
     parasite transcript load) are permuted jointly across children with age held
     fixed (1000 permutations). One-sided p = P(null gain >= observed).
  3. A random-covariate control: molecular covariates replaced by independent draws
     from their own marginal distributions (rank-normal / uniform), 1000 draws.
  4. A negative control with every patient-by-formulation term switched off.

Output -> revision/results/personalization_stats.json
"""
import _bootstrap  # noqa: F401
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "14_omics_formulation"))
from src.utils.patient_cohort import load_patients  # noqa: E402
from src.models.pharmacokinetics import DRUGS as PK, scale_params  # noqa: E402
from build_patient_formulation_dataset import formulation_library  # noqa: E402

OUT = ROOT / "revision" / "results"
OUT.mkdir(parents=True, exist_ok=True)
DRUG = "lumefantrine"
B, NPERM = 2000, 1000


def utility_matrix(pat, lib, hepatic=True, interactions=True):
    """Children x formulations utility, identical to build_patient_formulation_dataset.outcomes."""
    ee = lib["ee_pct"].to_numpy(); q24 = lib["Q24"].to_numpy()
    liver = lib["liver_idx"].to_numpy(); burst = lib["burst_1h"].to_numpy() / 100.0
    F = np.clip(0.30 + 0.50 * ee / 100 + 0.20 * q24 / 100, 0.10, 1.0)
    U = np.empty((len(pat), len(lib)))
    tgt = PK[DRUG]["day7_target"]
    for i, p in enumerate(pat.itertuples()):
        CL, V, ka, dose = scale_params(DRUG, p.weight_kg, p.age_years)
        if interactions:
            CL = CL * float(np.exp(0.25 * p.host_metabolic_z))
        ke = CL / V
        ka_ = ka if abs(ka - ke) >= 1e-9 else ke * 1.001
        c = dose * ka_ / (V * (ka_ - ke)) * (np.exp(-ke * 168.0) - np.exp(-ka_ * 168.0))
        day7 = np.clip(F * c, 0, None) * 1000.0
        cure = 1.0 / (1.0 + np.exp(-0.03 * (day7 - tgt)))
        if hepatic and interactions:
            cure = cure * (1.0 - 0.6 * p.parasite_resistance * (1.0 - liver))
        cure = np.clip(cure, 0, 1)
        tox = 1.0 / (1.0 + np.exp(-1.2 * (day7 / tgt - 8.0)))
        scale = float(np.exp(0.35 * p.host_tox_z)) if interactions else 1.0
        tox = np.clip(tox * scale * (1.0 + 0.3 * burst), 0, 1)
        U[i] = cure * (1.0 - tox)
    return U


def gain_stats(U, ids=None):
    """Oracle personalization summary from a children x formulations utility matrix."""
    j0 = int(np.argmax(U.mean(0)))
    best = U.argmax(1)
    g = U.max(1) - U[:, j0]
    return dict(mean_gain=float(g.mean()), max_gain=float(g.max()), one_size_index=j0,
                pct_differ=float(100 * np.mean(best != j0)), best=best, gains=g)


def boot_ci(U, rng, b=B):
    n = U.shape[0]
    vals = np.empty(b)
    for k in range(b):
        idx = rng.integers(0, n, n)
        vals[k] = gain_stats(U[idx])["mean_gain"]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(name, U, lib, rng, pat, hepatic, interactions=True, do_nulls=True):
    s = gain_stats(U)
    poly = lib["polymer"].to_numpy()
    fid = lib["formulation_id"].to_numpy()
    res = {
        "condition": name, "hepatic_term": hepatic, "interactions": interactions,
        "n_children": int(U.shape[0]), "n_formulations": int(U.shape[1]),
        "one_size_formulation": str(fid[s["one_size_index"]]),
        "pct_children_differing_formulation": round(s["pct_differ"], 1),
        "pct_children_differing_polymer": round(float(100 * np.mean(
            poly[s["best"]] != poly[s["one_size_index"]])), 1),
        "n_distinct_formulations": int(len(set(s["best"]))),
        "polymer_counts": pd.Series(poly[s["best"]]).value_counts().to_dict(),
        "mean_utility_one_size": round(float(U[:, s["one_size_index"]].mean()), 4),
        "mean_utility_personalized": round(float(U.max(1).mean()), 4),
        "mean_gain": round(s["mean_gain"], 4), "max_gain": round(s["max_gain"], 4),
        "mean_gain_95ci_bootstrap_children": [round(x, 4) for x in boot_ci(U, rng)],
    }
    if do_nulls:
        mol = ["host_metabolic_z", "host_tox_z", "parasite_resistance"]
        perm = np.empty(NPERM)
        for k in range(NPERM):
            q = pat.copy()
            idx = rng.permutation(len(q))
            q[mol] = pat[mol].to_numpy()[idx]
            perm[k] = gain_stats(utility_matrix(q, lib, hepatic, interactions))["mean_gain"]
        rnd = np.empty(NPERM)
        for k in range(NPERM):
            q = pat.copy()
            q["host_metabolic_z"] = rng.standard_normal(len(q))
            q["host_tox_z"] = rng.standard_normal(len(q))
            q["parasite_resistance"] = rng.uniform(0, 1, len(q))
            rnd[k] = gain_stats(utility_matrix(q, lib, hepatic, interactions))["mean_gain"]
        obs = s["mean_gain"]
        res["molecular_permutation_null"] = {
            "n_permutations": NPERM, "null_median": round(float(np.median(perm)), 4),
            "null_95pct_range": [round(float(np.percentile(perm, 2.5)), 4),
                                 round(float(np.percentile(perm, 97.5)), 4)],
            "p_one_sided": round(float((1 + np.sum(perm >= obs)) / (NPERM + 1)), 4)}
        res["random_covariate_control"] = {
            "n_draws": NPERM, "median_gain": round(float(np.median(rnd)), 4),
            "range_95pct": [round(float(np.percentile(rnd, 2.5)), 4),
                            round(float(np.percentile(rnd, 97.5)), 4)],
            "p_one_sided": round(float((1 + np.sum(rnd >= obs)) / (NPERM + 1)), 4)}
    return res


def main():
    rng = np.random.default_rng(20261105)
    pat = load_patients(prefer_site="mali", paediatric_only=True).reset_index(drop=True)
    lib = formulation_library(DRUG, "PVA", None or list(__import__(
        "src.models.nano_dds", fromlist=["POLYMERS"]).POLYMERS))
    # verification against the stored dataset (hepatic on, as submitted)
    stored = pd.read_parquet(ROOT / "revision" / "results" / "pfd_fixed_hepatic_on.parquet")
    piv = stored.pivot(index="patient_id", columns="formulation_id", values="utility")
    piv = piv.loc[pat["patient_id"], lib["formulation_id"]]
    U_on = utility_matrix(pat, lib, hepatic=True)
    max_abs = float(np.abs(piv.to_numpy() - U_on).max())
    print("vectorized simulator vs stored dataset, max |difference| =", max_abs)
    assert max_abs < 1e-9

    U_off = utility_matrix(pat, lib, hepatic=False)
    U_none = utility_matrix(pat, lib, hepatic=False, interactions=False)
    out = {"drug": DRUG, "verification_max_abs_diff": max_abs,
           "note": "Oracle gain: assignment uses the simulator's own utility; no fitted model.",
           "as_submitted_hepatic_on": summarize("as submitted", U_on, lib, rng, pat, True),
           "revised_hepatic_off": summarize("revised", U_off, lib, rng, pat, False),
           "negative_control_no_interactions": summarize("no interactions", U_none, lib, rng,
                                                         pat, False, False, do_nulls=False)}
    # describe the dominant assignment axis under each model
    for key, U in (("as_submitted_hepatic_on", U_on), ("revised_hepatic_off", U_off)):
        best = lib["polymer"].to_numpy()[U.argmax(1)]
        out[key]["mean_covariates_by_assigned_polymer"] = (
            pat.assign(polymer=best).groupby("polymer")[
                ["host_metabolic_z", "host_tox_z", "parasite_resistance", "age_years"]]
            .mean().round(3).to_dict(orient="index"))
    (OUT / "personalization_stats.json").write_text(json.dumps(out, indent=2, default=str))
    for k in ("as_submitted_hepatic_on", "revised_hepatic_off", "negative_control_no_interactions"):
        r = out[k]
        print(k, r["one_size_formulation"], r["pct_children_differing_formulation"],
              r["mean_gain"], r["mean_gain_95ci_bootstrap_children"], r["polymer_counts"],
              r.get("molecular_permutation_null", {}).get("p_one_sided"),
              r.get("random_covariate_control", {}).get("p_one_sided"))


if __name__ == "__main__":
    main()
