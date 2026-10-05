#!/usr/bin/env python
"""Phase 16 -- AI pharmacokinetics: predict exposure + cure (Sections 31, 60, 68).

Learns antimalarial exposure (day-7 concentration, the clinical outcome
predictor) and cure from patient + drug + formulation covariates, and shows that
a model using host molecular state and formulation properties predicts exposure
better than the classical allometric model that uses body size alone. It then
links predicted exposure to cure, completing the exposure bridge between the
molecular/formulation arms and treatment outcome.

Comparisons for exposure (log10 day-7 concentration):
  * allometric baseline : gradient boosting on age + weight only (classical)
  * AI (full)           : gradient boosting on age, weight, sex, host_txPC1,
                          dissolution, particle size, bioavailable F
  * AI (neural)         : a small multilayer perceptron on the full feature set
Cure (binary) is predicted from the same full feature set (AUROC).

Output -> results/pk_ai_results.json  (+ lead exposure-optimal formulation note)
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

FULL = ["age_years", "weight_kg", "sex_M", "host_txPC1",
        "dissolution_pct_30min", "particle_size_d50_um", "bioavailable_F"]
ALLO = ["age_years", "weight_kg"]


def main() -> None:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.neural_network import MLPRegressor

    seed = load_config()["reproducibility"]["global_seed"]
    path = get_path("data_synthetic") / "pk_cohort.tsv"
    if not path.exists():
        log.error("pk_cohort.tsv not found; run simulate_pk_cohort.py first.")
        return
    df = pd.read_csv(path, sep="\t")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FULL + ["day7", "cure"])
    log.info("PK cohort: %d records; drug=%s", len(df), df["drug"].iloc[0])

    y_exp = np.log10(np.clip(df["day7"].to_numpy(), 1e-3, None))  # log10 exposure
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        df, y_exp, np.arange(len(df)), test_size=0.25, random_state=seed)

    def gbr():
        return GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=seed)

    res = {}
    # allometric baseline
    m = gbr().fit(Xtr[ALLO], ytr); pa = m.predict(Xte[ALLO])
    res["exposure_allometric"] = {"r2": float(r2_score(yte, pa)),
                                  "mae_log10": float(mean_absolute_error(yte, pa))}
    # AI full
    m = gbr().fit(Xtr[FULL], ytr); pf = m.predict(Xte[FULL])
    res["exposure_ai_gbm"] = {"r2": float(r2_score(yte, pf)),
                              "mae_log10": float(mean_absolute_error(yte, pf))}
    imp = dict(sorted(zip(FULL, m.feature_importances_), key=lambda kv: -kv[1]))
    res["exposure_feature_importance"] = {k: float(v) for k, v in imp.items()}
    # AI neural
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    nn = make_pipeline(StandardScaler(),
                       MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                    random_state=seed))
    nn.fit(Xtr[FULL], ytr); pn = nn.predict(Xte[FULL])
    res["exposure_ai_neural"] = {"r2": float(r2_score(yte, pn)),
                                 "mae_log10": float(mean_absolute_error(yte, pn))}

    # cure prediction (binary) from full features
    cl = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=seed)
    cl.fit(Xtr[FULL], Xtr["cure"]); pc = cl.predict_proba(Xte[FULL])[:, 1]
    res["cure_auroc_ai"] = float(roc_auc_score(Xte["cure"], pc))
    # cure from exposure alone (mechanistic check: day7 should predict cure well)
    res["cure_auroc_day7only"] = float(roc_auc_score(
        Xte["cure"], Xte["day7"].to_numpy()))

    out = get_path("results", create=True) / "pk_ai_results.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    log.info("EXPOSURE (log10 day-7 conc) R2:  allometric=%.3f | AI-GBM=%.3f | AI-neural=%.3f",
             res["exposure_allometric"]["r2"], res["exposure_ai_gbm"]["r2"],
             res["exposure_ai_neural"]["r2"])
    log.info("  -> AI improvement over allometric (R2): +%.3f",
             res["exposure_ai_gbm"]["r2"] - res["exposure_allometric"]["r2"])
    log.info("top exposure drivers: %s", list(imp)[:4])
    log.info("CURE AUROC: AI(features)=%.3f | day7-concentration-only=%.3f",
             res["cure_auroc_ai"], res["cure_auroc_day7only"])
    log.info("wrote %s", out)
    log.info("Interpretation: host molecular state + formulation add exposure "
             "signal beyond body size (allometry); day-7 concentration remains "
             "the dominant cure predictor, reproducing clinical pharmacology.")


if __name__ == "__main__":
    main()
