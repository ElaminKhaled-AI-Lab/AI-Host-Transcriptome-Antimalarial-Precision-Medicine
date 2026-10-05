#!/usr/bin/env python
"""Phase 19 -- AI surrogates + response-surface + Bayesian optimization for nano-DDS.

Learns the nano drug-delivery-system (DDS) factor -> response map from the 3^3
factorial dataset and finds the optimal pediatric antimalarial nanoparticle
formulation for a chosen drug.

Three complementary layers (all documented, in-silico):
  1. AI surrogates -- Random Forest (RF), Gradient Boosting (GB) and a small
     multilayer perceptron (MLP) predict each response (size, PDI, zeta, EE,
     24 h release, liver-targeting index) from the coded factors + polymer/drug
     descriptors; cross-validated R2 and factor importances are reported.
  2. Response-surface methodology (RSM) -- a Box-Behnken design (BBD, 3 factors)
     is fitted with a full quadratic model per response, giving the classical
     pharmaceutics view (main, interaction, quadratic terms).
  3. Bayesian optimization (BO) -- a Gaussian-process (GP) surrogate with an
     Expected-Improvement (EI) acquisition searches the continuous factor space
     to maximize a formulation desirability that rewards the ~100-200 nm liver
     window, high EE, sustained-but-adequate release, high liver-targeting index
     and colloidal stability.

Output -> results/nano_dds_optimization.json  (+ per-drug optimum + surrogate R2)
"""
import _bootstrap  # noqa: F401
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.models.nano_dds import (
    Formulation, POLYMERS, characterize, RATIO_MAP, STAB_MAP, ENERGY_MAP,
    predict_size, predict_ee, predict_zeta, liver_targeting_index, release_profile)

log = get_logger(__name__)

RESP = ["size_nm", "pdi", "zeta_mv", "ee_pct", "release_24h_pct", "liver_targeting_index"]


# ---------------------------------------------------------------- desirability
def desirability(size, ee, release24, liver_idx, zeta):
    """0-1 overall desirability for a liver-targeted pediatric antimalarial NP.

    Rewards: size in 100-200 nm (Kupffer window), high EE, moderate-to-high but
    controlled 24 h release (target ~70%), high liver-targeting index, and
    |zeta| >= 20 mV for colloidal stability.
    """
    d_size = np.exp(-((size - 150) / 60) ** 2)
    d_ee = ee / 100.0
    d_rel = np.exp(-((release24 - 70) / 25) ** 2)
    d_liver = liver_idx
    d_zeta = 1.0 / (1.0 + np.exp(-(abs(zeta) - 20) / 4))
    return float((d_size * d_ee * d_rel * d_liver * d_zeta) ** (1 / 5))  # geometric mean


# ---------------------------------------------------------------- AI surrogates
def train_surrogates(df, seed):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    feats = ["A_polymer_drug_ratio_coded", "B_stabilizer_pct_coded", "C_homogenization_coded"]
    # add polymer descriptors so one model spans the library
    df = df.copy()
    from src.models.nano_dds import DRUGS as _DRUGS
    df["poly_basesize"] = df["polymer"].map(lambda p: POLYMERS[p]["base_size_nm"])
    df["poly_zeta"] = df["polymer"].map(lambda p: POLYMERS[p]["zeta_mv"])
    df["poly_degrad"] = df["polymer"].map(lambda p: POLYMERS[p]["degrad"])
    df["drug_logp"] = df["drug"].map(lambda d: _DRUGS.get(d, {"logp": 3.0})["logp"])
    df["drug_mw"] = df["drug"].map(lambda d: _DRUGS.get(d, {"mw": 400.0})["mw"])
    feats += ["poly_basesize", "poly_zeta", "poly_degrad", "drug_logp", "drug_mw"]
    X = df[feats].to_numpy(float)

    models = {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed),
        "gradient_boost": GradientBoostingRegressor(n_estimators=250, max_depth=3, random_state=seed),
        "mlp_neural": make_pipeline(StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes=(64, 32),
                                                 max_iter=1200, random_state=seed)),
    }
    out = {}
    for resp in RESP:
        y = df[resp].to_numpy(float)
        out[resp] = {}
        for name, m in models.items():
            pred = cross_val_predict(m, X, y, cv=4)
            out[resp][name] = round(float(r2_score(y, pred)), 3)
        # feature importance from RF
        rf = RandomForestRegressor(n_estimators=300, random_state=seed).fit(X, y)
        imp = dict(sorted(zip(feats, rf.feature_importances_), key=lambda kv: -kv[1]))
        out[resp]["top_drivers"] = {k: round(float(v), 3) for k, v in list(imp.items())[:3]}
    return out


# ---------------------------------------------------------------- Box-Behnken RSM
def box_behnken_3():
    """13-run Box-Behnken design for 3 factors (12 edge + 1 centre), coded -1/0/+1."""
    edges = [(a, b, 0) for a in (-1, 1) for b in (-1, 1)]
    edges += [(a, 0, c) for a in (-1, 1) for c in (-1, 1)]
    edges += [(0, b, c) for b in (-1, 1) for c in (-1, 1)]
    return edges + [(0, 0, 0)]


def rsm_quadratic(df_bbd, resp):
    """Fit full quadratic RSM: y = b0 + bi Xi + bij XiXj + bii Xi^2 (OLS)."""
    X = df_bbd[["A", "B", "C"]].to_numpy(float)
    terms = np.column_stack([
        np.ones(len(X)), X[:, 0], X[:, 1], X[:, 2],
        X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2],
        X[:, 0] ** 2, X[:, 1] ** 2, X[:, 2] ** 2])
    y = df_bbd[resp].to_numpy(float)
    beta, *_ = np.linalg.lstsq(terms, y, rcond=None)
    names = ["b0", "A", "B", "C", "AB", "AC", "BC", "A2", "B2", "C2"]
    yhat = terms @ beta
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return {"coeffs": {n: round(float(b), 3) for n, b in zip(names, beta)},
            "r2": round(float(1 - ss_res / ss_tot), 3)}


# ---------------------------------------------------------------- Bayesian optimization
def bayes_optimize(drug, polymer, stabilizer, seed, n_init=8, n_iter=25):
    """GP + Expected-Improvement search over continuous coded factors in [-1,1]^3."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    from scipy.stats import norm

    rng = np.random.default_rng(seed)

    def evaluate(x):
        # map continuous coded x in [-1,1] to nearest usable characterization
        f = Formulation(drug=drug, polymer=polymer, stabilizer=stabilizer,
                        polymer_drug_ratio=int(np.round(x[0])),
                        stabilizer_pct=int(np.round(x[1])),
                        homogenization=int(np.round(x[2])))
        size, ee = predict_size(f), predict_ee(f)
        t, Q = release_profile(f)
        rel24 = float(np.interp(24, t, Q))
        return desirability(size, ee, rel24, liver_targeting_index(f), predict_zeta(f))

    Xs = rng.uniform(-1, 1, size=(n_init, 3))
    ys = np.array([evaluate(x) for x in Xs])
    kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5)
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=seed).fit(Xs, ys)
        cand = rng.uniform(-1, 1, size=(400, 3))
        mu, sd = gp.predict(cand, return_std=True)
        best = ys.max()
        imp = mu - best
        z = imp / (sd + 1e-9)
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)
        xnext = cand[int(np.argmax(ei))]
        Xs = np.vstack([Xs, xnext]); ys = np.append(ys, evaluate(xnext))
    ibest = int(np.argmax(ys))
    xb = np.round(Xs[ibest]).astype(int)
    fb = Formulation(drug=drug, polymer=polymer, stabilizer=stabilizer,
                     polymer_drug_ratio=int(xb[0]), stabilizer_pct=int(xb[1]),
                     homogenization=int(xb[2]))
    return {"best_desirability": round(float(ys[ibest]), 3),
            "optimal_factors_coded": {"A": int(xb[0]), "B": int(xb[1]), "C": int(xb[2])},
            "optimal_formulation": characterize(fb)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug", default="lumefantrine")
    ap.add_argument("--stabilizer", default="PVA")
    ap.add_argument("--top-polymers", type=int, default=3,
                    help="optimize the N polymers with best mean desirability for the drug")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]

    syn = get_path("data_synthetic")
    df = pd.read_csv(syn / "nano_dds_factorial.tsv", sep="\t")

    # 1. AI surrogates on the whole library
    log.info("training AI surrogates (RF / GB / MLP) ...")
    surro = train_surrogates(df, seed)
    for r in RESP:
        log.info("  %-22s R2  RF=%.2f GB=%.2f MLP=%.2f  drivers=%s", r,
                 surro[r]["random_forest"], surro[r]["gradient_boost"],
                 surro[r]["mlp_neural"], list(surro[r]["top_drivers"]))

    # 2. rank polymers for this drug by mean desirability, then RSM + BO on top ones
    dd = df[df["drug"] == args.drug].copy()
    dd["desir"] = [desirability(r.size_nm, r.ee_pct, r.release_24h_pct,
                                r.liver_targeting_index, r.zeta_mv) for r in dd.itertuples()]
    poly_rank = dd.groupby("polymer")["desir"].mean().sort_values(ascending=False)
    top_polymers = list(poly_rank.head(args.top_polymers).index)
    log.info("top polymers for %s: %s", args.drug,
             {p: round(float(poly_rank[p]), 3) for p in top_polymers})

    # Box-Behnken RSM (rebuild BBD responses from the twin for the best polymer)
    from src.models.nano_dds import release_profile
    best_poly = top_polymers[0]
    bbd_rows = []
    for (a, b, c) in box_behnken_3():
        f = Formulation(drug=args.drug, polymer=best_poly, stabilizer=args.stabilizer,
                        polymer_drug_ratio=a, stabilizer_pct=b, homogenization=c)
        t, Q = release_profile(f)
        bbd_rows.append({"A": a, "B": b, "C": c, "size_nm": predict_size(f),
                         "ee_pct": predict_ee(f), "zeta_mv": predict_zeta(f),
                         "release_24h_pct": float(np.interp(24, t, Q)),
                         "liver_targeting_index": liver_targeting_index(f)})
    bbd = pd.DataFrame(bbd_rows)
    rsm = {r: rsm_quadratic(bbd, r) for r in ["size_nm", "ee_pct", "release_24h_pct",
                                              "liver_targeting_index"]}
    log.info("Box-Behnken RSM on %s: size R2=%.2f, EE R2=%.2f, release R2=%.2f",
             best_poly, rsm["size_nm"]["r2"], rsm["ee_pct"]["r2"], rsm["release_24h_pct"]["r2"])

    # 3. Bayesian optimization per top polymer
    bo = {}
    for p in top_polymers:
        bo[p] = bayes_optimize(args.drug, p, args.stabilizer, seed)
        opt = bo[p]["optimal_formulation"]
        log.info("BO %-14s -> desirability %.3f | size %.0f nm, EE %.0f%%, zeta %.0f mV, "
                 "release24h %.0f%%, liver idx %.2f", p, bo[p]["best_desirability"],
                 opt["size_nm"], opt["ee_pct"], opt["zeta_mv"], opt["release_24h_pct"],
                 opt["liver_targeting_index"])

    overall = max(bo.items(), key=lambda kv: kv[1]["best_desirability"])
    out = {
        "drug": args.drug, "stabilizer": args.stabilizer,
        "ai_surrogate_cv_r2": surro,
        "polymer_ranking_by_desirability": {p: round(float(poly_rank[p]), 3)
                                            for p in poly_rank.index},
        "box_behnken_rsm": {"polymer": best_poly, "models": rsm},
        "bayesian_optimization": bo,
        "recommended_formulation": {"polymer": overall[0], **overall[1]},
    }
    dest = get_path("results", create=True) / "nano_dds_optimization.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    log.info("RECOMMENDED: %s / %s  desirability=%.3f", args.drug, overall[0],
             overall[1]["best_desirability"])
    log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
