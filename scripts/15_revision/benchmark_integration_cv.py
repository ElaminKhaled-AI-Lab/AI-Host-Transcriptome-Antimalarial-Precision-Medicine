#!/usr/bin/env python
"""Revision analysis for R2.6d / R2.8b-2 / R2.8b-7: repeated grouped cross-validation
of the four integration strategies, with child-level bootstrap CIs and Holm-adjusted
paired comparisons.

Design (pre-declared, no hyperparameter tuning, so no nesting is required):
  * unit of resampling = child; GroupKFold-style 5-fold split over the 56 children,
    repeated R times with different shuffles; every child is predicted once per repeat.
  * architectures, widths, epochs (80), learning rate and batch size fixed as in the
    submitted benchmark; network initialization seed varies with repeat and fold.
  * per-child metrics on pooled out-of-fold predictions:
      - interaction-residual Spearman (formulation main effect removed using all
        children's out-of-fold predictions in that repeat) [PRIMARY]
      - model-based gain: true utility of the model's top formulation minus true utility
        of the single formulation with the best mean TRUE utility in the TRAINING
        children of that fold (no test-set oracle), averaged over repeats
      - exposure and utility R^2 (pooled, descriptive)
  * 95% CIs: cluster bootstrap over children (B = 2000) of the repeat-averaged per-child
    metric; paired contrasts between strategies use the same bootstrap indices; two-sided
    Wilcoxon signed-rank on the 56 per-child values, Holm-adjusted over the 6 pairs.
  * oracle gain on the same children reported as the ceiling.

Output -> revision/results/benchmark_cv_<tag>.json
"""
import _bootstrap  # noqa: F401
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import r2_score

from src.models.profile_net import STRATEGIES, train_model, predict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision" / "results"
PAT = ["age_years", "weight_kg", "host_metabolic_z", "host_tox_z", "parasite_resistance"]
FORM = ["A", "B", "C", "ratio", "stab_pct", "energy", "size_nm", "pdi", "zeta_mv", "ee_pct",
        "dl_pct", "DE_pct", "MDT_h", "t50_h", "n_peppas", "liver_idx", "poly_basesize",
        "poly_zeta", "poly_degrad"]


def std(tr, te):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd == 0] = 1
    return (tr - mu) / sd, (te - mu) / sd


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    torch_threads = __import__("torch").set_num_threads(2)  # noqa: F841

    df = pd.read_parquet(a.data).sort_values(["patient_id", "formulation_id"]).reset_index(drop=True)
    kids = np.array(sorted(df.patient_id.unique()))
    fids = np.array(sorted(df.formulation_id.unique()))
    nK, nF = len(kids), len(fids)
    Ut = df.pivot(index="patient_id", columns="formulation_id", values="utility").loc[kids, fids].to_numpy()
    Et = df.pivot(index="patient_id", columns="formulation_id", values="log10_day7").loc[kids, fids].to_numpy()
    rng = np.random.default_rng(20261105)
    preds = {s: np.full((a.repeats, nK, nF, 2), np.nan) for s in STRATEGIES}
    train_onesize = np.full((a.repeats, nK), -1, int)
    t0 = time.time()
    for r in range(a.repeats):
        perm = rng.permutation(nK)
        folds = np.array_split(perm, a.folds)
        for k, te_idx in enumerate(folds):
            te_ids = set(kids[te_idx]); tr = df[~df.patient_id.isin(te_ids)]; te = df[df.patient_id.isin(te_ids)]
            tr_k = np.setdiff1d(np.arange(nK), te_idx)
            train_onesize[r, te_idx] = int(np.argmax(Ut[tr_k].mean(0)))
            Xp_tr, Xp_te = std(tr[PAT].to_numpy(float), te[PAT].to_numpy(float))
            Xf_tr, Xf_te = std(tr[FORM].to_numpy(float), te[FORM].to_numpy(float))
            Ytr = np.column_stack([tr.log10_day7, tr.utility]).astype(float)
            mu, sd = Ytr.mean(0), Ytr.std(0); sd[sd == 0] = 1
            rows_k = np.searchsorted(kids, te.patient_id.to_numpy())
            rows_f = np.searchsorted(fids, te.formulation_id.to_numpy())
            for s, cls in STRATEGIES.items():
                m = cls(d_pat=Xp_tr.shape[1], d_form=Xf_tr.shape[1], n_reg=2)
                m = train_model(m, Xp_tr, Xf_tr, (Ytr - mu) / sd, None, epochs=a.epochs,
                                seed=1000 * r + 10 * k + 11, device="cpu")
                P, _ = predict(m, Xp_te, Xf_te)
                P = P * sd + mu
                preds[s][r, rows_k, rows_f, :] = P
            print(f"repeat {r+1}/{a.repeats} fold {k+1}/{a.folds} done, {time.time()-t0:.0f}s", flush=True)

    # per-child metrics
    def center(M):  # remove formulation main effect across children
        return M - M.mean(0, keepdims=True)
    Utc = center(Ut)
    oracle_onesize = int(np.argmax(Ut.mean(0)))
    oracle_gain_child = Ut.max(1) - Ut[:, oracle_onesize]
    per = {}
    for s in STRATEGIES:
        inter = np.zeros((a.repeats, nK)); gain = np.zeros((a.repeats, nK))
        for r in range(a.repeats):
            Pu = preds[s][r, :, :, 1]
            Pc = center(Pu)
            for i in range(nK):
                if np.std(Pc[i]) < 1e-12 or np.std(Utc[i]) < 1e-12:
                    inter[r, i] = 0.0
                else:
                    inter[r, i] = spearmanr(Utc[i], Pc[i]).correlation
            choice = Pu.argmax(1)
            gain[r] = Ut[np.arange(nK), choice] - Ut[np.arange(nK), train_onesize[r]]
        er2 = np.mean([r2_score(Et.ravel(), preds[s][r, :, :, 0].ravel()) for r in range(a.repeats)])
        ur2 = np.mean([r2_score(Ut.ravel(), preds[s][r, :, :, 1].ravel()) for r in range(a.repeats)])
        per[s] = dict(inter=inter.mean(0), gain=gain.mean(0), exposure_r2=er2, utility_r2=ur2,
                      inter_by_repeat=inter.mean(1), gain_by_repeat=gain.mean(1))
    B = 2000
    bidx = rng.integers(0, nK, (B, nK))
    res = {"data": a.data, "tag": a.tag, "n_children": nK, "n_formulations": nF,
           "repeats": a.repeats, "folds": a.folds, "epochs": a.epochs,
           "oracle_gain_mean": round(float(oracle_gain_child.mean()), 4),
           "oracle_gain_95ci": [round(float(np.percentile(oracle_gain_child[bidx].mean(1), q)), 4)
                                for q in (2.5, 97.5)],
           "strategies": {}}
    for s, v in per.items():
        bi = v["inter"][bidx].mean(1); bg = v["gain"][bidx].mean(1)
        res["strategies"][s] = {
            "interaction_rank_mean": round(float(v["inter"].mean()), 4),
            "interaction_rank_95ci": [round(float(np.percentile(bi, q)), 4) for q in (2.5, 97.5)],
            "model_gain_mean": round(float(v["gain"].mean()), 5),
            "model_gain_95ci": [round(float(np.percentile(bg, q)), 5) for q in (2.5, 97.5)],
            "exposure_r2": round(float(v["exposure_r2"]), 4),
            "utility_r2": round(float(v["utility_r2"]), 4),
            "interaction_rank_by_repeat": [round(float(x), 4) for x in v["inter_by_repeat"]],
            "model_gain_by_repeat": [round(float(x), 5) for x in v["gain_by_repeat"]]}
    pairs = list(itertools.combinations(STRATEGIES, 2))
    for metric in ("inter", "gain"):
        rows, ps = [], []
        for s1, s2 in pairs:
            d = per[s1][metric] - per[s2][metric]
            bd = d[bidx].mean(1)
            try:
                p = wilcoxon(per[s1][metric], per[s2][metric]).pvalue
            except ValueError:
                p = 1.0
            ps.append(p)
            rows.append({"a": s1, "b": s2, "mean_difference": round(float(d.mean()), 5),
                         "diff_95ci": [round(float(np.percentile(bd, q)), 5) for q in (2.5, 97.5)],
                         "wilcoxon_p": float(p)})
        for row, pa in zip(rows, holm(ps)):
            row["holm_p"] = float(pa)
        res[f"paired_{'interaction' if metric == 'inter' else 'gain'}"] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"benchmark_cv_{a.tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({s: (v["interaction_rank_mean"], v["interaction_rank_95ci"], v["model_gain_mean"],
                          v["model_gain_95ci"]) for s, v in res["strategies"].items()}, indent=1))
    print("oracle", res["oracle_gain_mean"], res["oracle_gain_95ci"])


if __name__ == "__main__":
    main()
