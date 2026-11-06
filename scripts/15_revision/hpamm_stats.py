#!/usr/bin/env python
"""Revision analysis for R1.3 / R2.3 / R2.8b-2 / R2.8b-3: HPAMM-Net uncertainty.

HPAMM-Net is trained and tested on ten independent semi-synthetic cohorts (n = 2000
each; 1600 train / 400 test), not on the dual RNA-sequencing cohorts. Each cohort is an
independent replicate, so a t-interval over the ten paired cohort-level differences is a
valid uncertainty estimate (seed-only SDs are not).

Part 1 uses the stored per-cohort result files (the values plotted in the submitted
Figure 5). Part 2 re-trains, under ONE protocol (the main split of
train_hpamm_multitask.py, 200 epochs), the full model and the four ablations on all ten
cohorts, together with an L2 logistic / ridge baseline (correctly specified for the
linear-logistic generator) and sklearn gradient boosting (xgboost unavailable here).
Paired contrasts against the full model: mean, 95% t-CI over cohorts, Wilcoxon signed-
rank, Holm-adjusted within each family.

Output -> revision/results/hpamm_stats.json
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "08_hpamm_net"))
from train_hpamm_multitask import load_cohort, get_blocks, CLASS, REG  # noqa: E402
from run_ablations_multitask import CONFIGS  # noqa: E402
from src.utils.config import load_config  # noqa: E402

SEEDS = [11, 23, 47, 71, 101, 131, 167, 197, 233, 271]
OUT = ROOT / "revision" / "results"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    se = d.std(ddof=1) / np.sqrt(n); h = tdist.ppf(0.975, n - 1) * se
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = 1.0
    return dict(mean_diff=round(float(d.mean()), 4), ci95=[round(float(d.mean() - h), 4),
                round(float(d.mean() + h), 4)], wilcoxon_p=float(p))


def part1():
    rows = [json.loads((ROOT / "results" / f"hpamm_multitask_seed{s}.json").read_text()) for s in SEEDS]
    res = {}
    for t in CLASS + REG:
        key = "auroc" if t in CLASS else "mae"
        h = [r["hpamm_full"][t][key] for r in rows]; b = [r["baseline_xgb"][t][key] for r in rows]
        res[t] = dict(metric=key, hpamm_mean=round(float(np.mean(h)), 4),
                      xgb_mean=round(float(np.mean(b)), 4), **paired(h, b),
                      seed101_hpamm=round(rows[SEEDS.index(101)]["hpamm_full"][t][key], 4),
                      seed101_xgb=round(rows[SEEDS.index(101)]["baseline_xgb"][t][key], 4))
    for t, a in zip(res, holm([res[t]["wilcoxon_p"] for t in res])):
        res[t]["holm_p"] = float(a)
    return res


def train_eval(cfg, X, clin, yc, yr, tr, te, seed, epochs):
    import torch
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from src.models.hpamm_net import HPAMMNet, multitask_loss
    torch.manual_seed(seed)
    Xu = {m: v for m, v in X.items() if m in cfg["blocks"]}
    n = clin.shape[0]
    Xt = {m: torch.tensor(v, dtype=torch.float32) for m, v in Xu.items()}
    clt = torch.tensor(clin, dtype=torch.float32)
    mask = {m: torch.ones(n) for m in Xu}
    rmu = {t: float(yr[t][tr].mean()) for t in REG}; rsd = {t: float(yr[t][tr].std() + 1e-9) for t in REG}
    tgt = {**{t: torch.tensor(yc[t], dtype=torch.float32) for t in CLASS},
           **{t: torch.tensor((yr[t] - rmu[t]) / rsd[t], dtype=torch.float32) for t in REG}}
    lm = {t: torch.ones(n) for t in CLASS + REG}; w = {t: 1.0 for t in CLASS + REG}
    model = HPAMMNet({m: v.shape[1] for m, v in Xu.items()}, latent_dim=64, clinical_dim=clin.shape[1],
                     treatment_dim=0, tasks=CLASS, regression_tasks=REG,
                     modality_dropout_p=cfg["dropout"], gated_fusion=cfg["gating"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    trt = torch.tensor(tr); model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model({m: Xt[m][trt] for m in Xu}, {m: mask[m][trt] for m in Xu}, clinical=clt[trt])
        loss = multitask_loss(out, {t: tgt[t][trt] for t in CLASS + REG},
                              {t: lm[t][trt] for t in CLASS + REG}, w, CLASS, REG)
        loss.backward(); opt.step()
    tet = torch.tensor(te); model.eval()
    with torch.no_grad():
        out = model({m: Xt[m][tet] for m in Xu}, {m: mask[m][tet] for m in Xu}, clinical=clt[tet])
    r = {t: float(roc_auc_score(yc[t][te], torch.sigmoid(out[t]).numpy())) for t in CLASS}
    r.update({t: float(mean_absolute_error(yr[t][te], out[t].numpy() * rsd[t] + rmu[t])) for t in REG})
    return r


def classical(Xcat, yc, yr, tr, te):
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    lin, gb = {}, {}
    for t in CLASS:
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xcat[tr], yc[t][tr])
        lin[t] = float(roc_auc_score(yc[t][te], m.predict_proba(Xcat[te])[:, 1]))
        g = GradientBoostingClassifier(random_state=11).fit(Xcat[tr], yc[t][tr])
        gb[t] = float(roc_auc_score(yc[t][te], g.predict_proba(Xcat[te])[:, 1]))
    for t in REG:
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xcat[tr], yr[t][tr])
        lin[t] = float(mean_absolute_error(yr[t][te], m.predict(Xcat[te])))
        g = GradientBoostingRegressor(random_state=11).fit(Xcat[tr], yr[t][tr])
        gb[t] = float(mean_absolute_error(yr[t][te], g.predict(Xcat[te])))
    return lin, gb


def part2(epochs):
    import torch
    torch.set_num_threads(1)
    gseed = load_config()["reproducibility"]["global_seed"]
    per = {name: [] for name in list(CONFIGS) + ["linear_baseline", "gradient_boosting"]}
    for s in SEEDS:
        df = load_cohort(s)
        X, clin = get_blocks(df)
        yc = {t: df[f"out_{t}"].to_numpy(int) for t in CLASS}
        yr = {t: df[f"out_{t}"].to_numpy(float) for t in REG}
        n = len(df); perm = np.random.default_rng(gseed).permutation(n)
        te, tr = perm[: n // 5], perm[n // 5:]
        for name, cfg in CONFIGS.items():
            per[name].append(train_eval(cfg, X, clin, yc, yr, tr, te, gseed, epochs))
        Xcat = np.concatenate([X["host_rna"], X["host_proteome"], X["parasite_genome"], clin], 1)
        lin, gb = classical(Xcat, yc, yr, tr, te)
        per["linear_baseline"].append(lin); per["gradient_boosting"].append(gb)
        print("cohort", s, {k: round(v[-1]["resistance"], 3) for k, v in per.items()}, flush=True)
    summary = {name: {t: round(float(np.mean([r[t] for r in rows])), 4) for t in CLASS + REG}
               for name, rows in per.items()}
    contrasts = {}
    for other in [k for k in per if k != "full"]:
        c = {t: paired([r[t] for r in per["full"]], [r[t] for r in per[other]]) for t in CLASS + REG}
        for t, a in zip(c, holm([c[t]["wilcoxon_p"] for t in c])):
            c[t]["holm_p"] = float(a)
        contrasts[f"full_minus_{other}"] = c
    return dict(epochs=epochs, split="main split of train_hpamm_multitask.py (global seed), 1600/400",
                means_over_10_cohorts=summary, paired_contrasts=contrasts,
                per_cohort={k: v for k, v in per.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--skip-retrain", action="store_true"); a = ap.parse_args()
    res = {"cohorts": SEEDS, "part1_stored_results_hpamm_vs_xgboost": part1()}
    print(json.dumps(res["part1_stored_results_hpamm_vs_xgboost"], indent=1))
    if not a.skip_retrain:
        res["part2_unified_protocol"] = part2(a.epochs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hpamm_stats.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
