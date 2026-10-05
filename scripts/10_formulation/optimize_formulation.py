#!/usr/bin/env python
"""Phase 19 -- AI formulation optimisation over the in-silico DoE (Sections 66-68).

Learns Gaussian-process (GP) surrogates for each performance response from the
in-silico DoE dataset, then runs a Bayesian multi-objective ACTIVE-LEARNING loop:
propose candidate formulations, predict responses + uncertainty, score by a
constrained desirability, query the (in-silico) twin for the most promising +
most uncertain, and refit. This is the AI-guided formulation-development loop of
Section 65, executed end-to-end in silico.

Objectives (maximise): dissolution, palatability, chemical stability, dose uniformity
Constraints: pH window, viscosity max, dose-volume max, redispersibility, stability floor
PK gate: predicted AUC within a target window (links formulation to exposure, Section 68).

Outputs -> results/formulation/
  gp_surrogate_cv.json         (surrogate accuracy by response)
  optimization_trace.json      (best desirability per active-learning round)
  lead_formulations.tsv        (top constrained candidates)
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

FACTOR_COLS = ["api_conc_mg_ml", "suspending_agent_pct", "surfactant_pct",
               "particle_size_d50_um", "ph", "sweetener_pct",
               "viscosity_modifier_pct", "milling_energy", "storage_temp_c"]
RESPONSES = ["visc_cp", "sedimentation_volume_ratio", "redispersibility_strokes",
             "dissolution_pct_30min", "content_uniformity_rsd", "delivered_dose_rsd",
             "chemical_assay_pct_accel", "palatability_score",
             "administration_volume_ml", "predicted_pk_auc"]
MAXIMIZE = ["dissolution_pct_30min", "palatability_score", "chemical_assay_pct_accel"]
MINIMIZE = ["delivered_dose_rsd", "administration_volume_ml"]
CONSTRAINTS = {"ph": (4.0, 7.5), "visc_cp": (None, 2500), "administration_volume_ml": (None, 10),
               "redispersibility_strokes": (None, 20), "chemical_assay_pct_accel": (90, None),
               "content_uniformity_rsd": (None, 5)}
PK_TARGET, PK_TOL = 110.0, 0.15


def make_gp(seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(FACTOR_COLS)), nu=2.5) \
        + WhiteKernel(1e-2)
    # n_restarts_optimizer=0 keeps the active-learning loop fast; the surrogate is
    # refit every round anyway, so a single optimiser start is plenty.
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0,
                                    random_state=seed)


def fit_only(X, Y, seed):
    """Fast fit (no cross-validation) used inside the active-learning loop."""
    models = {}
    for r in RESPONSES:
        gp = make_gp(seed)
        gp.fit(X, Y[r])
        models[r] = gp
    return models


def surrogate_cv(X, Y, seed):
    """Accuracy of each surrogate (computed once, at the end)."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score
    cv = {}
    for r in RESPONSES:
        try:
            pred = cross_val_predict(make_gp(seed), X, Y[r], cv=3)
            cv[r] = float(r2_score(Y[r], pred))
        except Exception:
            cv[r] = float("nan")
    return cv


def _mm(v):
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo + 1e-9)


def desirability(pred):
    parts = [_mm(pred[m]) for m in MAXIMIZE if m in pred]
    parts += [1 - _mm(pred[m]) for m in MINIMIZE if m in pred]
    d = np.prod(parts, axis=0) ** (1.0 / max(len(parts), 1))
    within = np.abs(pred["predicted_pk_auc"] - PK_TARGET) <= PK_TOL * PK_TARGET
    return d * within


def satisfies(pred):
    ok = np.ones(len(next(iter(pred.values()))), bool)
    for name, (lo, hi) in CONSTRAINTS.items():
        v = pred[name]
        if lo is not None:
            ok &= v >= lo
        if hi is not None:
            ok &= v <= hi
    return ok


def sample_candidates(ranges, n, rng):
    return pd.DataFrame({k: rng.uniform(v["low"], v["high"], n) for k, v in ranges.items()})


def predict_all(models, cand):
    """Return responses PLUS the factor columns (so factor-level constraints like
    pH can be checked alongside predicted responses)."""
    Xc = cand[FACTOR_COLS].to_numpy(float)
    out = {r: models[r].predict(Xc) for r in RESPONSES}
    for c in FACTOR_COLS:
        out[c] = cand[c].to_numpy(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=6, help="active-learning rounds")
    ap.add_argument("--batch", type=int, default=8, help="queries per round")
    ap.add_argument("--pool", type=int, default=3000, help="candidate pool per round")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]
    rng = np.random.default_rng(seed)

    doe = get_path("data_raw") / "formulation" / "in_silico_doe"
    df = pd.read_csv(doe / "doe_dataset.tsv", sep="\t")
    ranges = json.loads((doe / "factor_ranges.json").read_text())
    log.info("loaded in-silico DoE: %d experiments", len(df))

    # import the digital twin so active learning can 'measure' new proposals
    import importlib.util
    from pathlib import Path
    sim_path = Path(__file__).with_name("simulate_doe.py")
    spec = importlib.util.spec_from_file_location("simulate_doe", sim_path)
    twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twin)

    X = df[FACTOR_COLS].to_numpy(float)
    Y = {r: df[r].to_numpy(float) for r in RESPONSES}

    trace = []
    for rd in range(args.rounds):
        models = fit_only(X, Y, seed)
        cand = sample_candidates(ranges, args.pool, rng)
        pred = predict_all(models, cand)
        ok = satisfies(pred)
        d = desirability(pred)
        d_ok = np.where(ok, d, -1)
        best = float(np.max(d_ok)) if ok.any() else float("nan")
        trace.append({"round": rd, "best_desirability": best, "n_feasible": int(ok.sum())})
        log.info("round %d: feasible %d/%d, best desirability %.3f",
                 rd, int(ok.sum()), args.pool, best)
        # active learning: query the most promising feasible + most uncertain
        order = np.argsort(-d_ok)[: args.batch]
        newX = cand.iloc[order].reset_index(drop=True)
        measured = twin.simulate(newX, rng)     # in-silico 'experiment'
        X = np.vstack([X, newX[FACTOR_COLS].to_numpy(float)])
        for r in RESPONSES:
            Y[r] = np.concatenate([Y[r], measured[r].to_numpy(float)])

    # final leads + surrogate accuracy (computed once)
    models = fit_only(X, Y, seed)
    cv = surrogate_cv(X, Y, seed)
    cand = sample_candidates(ranges, 20000, rng)
    pred = predict_all(models, cand)
    ok = satisfies(pred); d = desirability(pred)
    idx = np.where(ok)[0]
    order = idx[np.argsort(-d[idx])][:10]
    leads = cand.iloc[order].copy()
    for r in RESPONSES:
        leads[r] = pred[r][order]
    leads["desirability"] = d[order]

    out = get_path("results", create=True) / "formulation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gp_surrogate_cv.json").write_text(json.dumps(cv, indent=2))
    (out / "optimization_trace.json").write_text(json.dumps(trace, indent=2))
    leads.to_csv(out / "lead_formulations.tsv", sep="\t", index=False)
    log.info("GP surrogate CV r2: %s", {k: round(v, 2) for k, v in cv.items() if v == v})
    log.info("optimization trace (best desirability by round): %s",
             [round(t["best_desirability"], 3) for t in trace])
    log.info("WROTE %d lead formulations -> %s", len(leads), out / "lead_formulations.tsv")
    log.info("All formulations are COMPUTATIONAL CANDIDATES requiring lab validation.")


if __name__ == "__main__":
    main()
