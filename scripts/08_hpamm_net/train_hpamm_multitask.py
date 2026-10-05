#!/usr/bin/env python
"""Phase 15 -- HPAMM-Net multi-task training + evaluation (blueprint Sections 49-56, 75).

Trains the novel Host-Parasite Adaptive Missingness-aware Multi-task Network on
the semi-synthetic cohort to jointly predict the six treatment outcomes, and
compares it to a strong per-task baseline (XGBoost / gradient boosting) both on
complete data and under MODALITY DROPOUT -- the scenario where a missingness-
aware fusion is supposed to earn its keep (Section 76).

Blocks: host_rna (tx PCs), host_proteome (prot PCs), parasite_genome (para feats).
Tasks : classification {resistance, cure, recrudescence, toxicity}
        regression     {pk_auc, clearance}

Outputs -> results/hpamm_multitask_seed<seed>.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

CLASS = ["resistance", "cure", "recrudescence", "toxicity"]
REG = ["pk_auc", "clearance"]


def load_cohort(seed):
    d = get_path("data_synthetic")
    p = d / f"cohort_seed{seed}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(d / f"cohort_seed{seed}.tsv", sep="\t")


def zscore(a):
    a = np.asarray(a, float)
    mu, sd = np.nanmean(a, 0), np.nanstd(a, 0)
    sd[sd == 0] = 1
    return np.nan_to_num((a - mu) / sd)


def get_blocks(df):
    tx = zscore(df[[c for c in df.columns if c.startswith("tx_PC")]].values)
    pr = zscore(df[[c for c in df.columns if c.startswith("prot_PC")]].values)
    pa = zscore(df[[c for c in df.columns if c.startswith("para_")]]
                .apply(pd.to_numeric, errors="coerce").values)
    cl = zscore(df[[c for c in df.columns if c.startswith("clin_")]].values)
    return {"host_rna": tx, "host_proteome": pr, "parasite_genome": pa}, cl


def baseline_scores(Xcat, y_class, y_reg, tr, te):
    """Per-task gradient boosting on the concatenated feature matrix."""
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    out = {}
    try:
        from xgboost import XGBClassifier, XGBRegressor
        mk_c = lambda: XGBClassifier(n_estimators=200, max_depth=4, random_state=11,
                                     eval_metric="logloss", n_jobs=-1)
        mk_r = lambda: XGBRegressor(n_estimators=200, max_depth=4, random_state=11, n_jobs=-1)
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        mk_c = lambda: GradientBoostingClassifier(random_state=11)
        mk_r = lambda: GradientBoostingRegressor(random_state=11)
    for t in CLASS:
        m = mk_c().fit(Xcat[tr], y_class[t][tr])
        out[t] = {"auroc": float(roc_auc_score(y_class[t][te], m.predict_proba(Xcat[te])[:, 1]))}
    for t in REG:
        m = mk_r().fit(Xcat[tr], y_reg[t][tr])
        out[t] = {"mae": float(mean_absolute_error(y_reg[t][te], m.predict(Xcat[te])))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    gseed = load_config()["reproducibility"]["global_seed"]

    try:
        import torch
        import torch.nn as nn  # noqa: F401
        from src.models.hpamm_net import HPAMMNet, multitask_loss
    except Exception as e:
        log.error("PyTorch required for HPAMM-Net: %s", e)
        return

    df = load_cohort(args.seed)
    X, clin = get_blocks(df)
    y_class = {t: df[f"out_{t}"].to_numpy(int) for t in CLASS}
    y_reg = {t: df[f"out_{t}"].to_numpy(float) for t in REG}
    n = len(df)
    rng = np.random.default_rng(gseed)
    perm = rng.permutation(n)
    te = perm[: n // 5]; tr = perm[n // 5:]
    log.info("cohort seed %d: n=%d (train %d / test %d); blocks %s clin %d",
             args.seed, n, len(tr), len(te), {k: v.shape[1] for k, v in X.items()},
             clin.shape[1])

    # ---- baseline ----------------------------------------------------------
    Xcat = np.concatenate([X["host_rna"], X["host_proteome"], X["parasite_genome"], clin], 1)
    base = baseline_scores(Xcat, y_class, y_reg, tr, te)
    log.info("baseline (XGBoost per-task): %s",
             {t: round(list(v.values())[0], 3) for t, v in base.items()})

    # ---- HPAMM-Net ---------------------------------------------------------
    dev = "cpu"
    torch.manual_seed(gseed)
    Xt = {m: torch.tensor(v, dtype=torch.float32, device=dev) for m, v in X.items()}
    clt = torch.tensor(clin, dtype=torch.float32, device=dev)
    maskt = {m: torch.ones(n, device=dev) for m in X}   # complete data
    # standardize regression targets so their MSE is comparable to the
    # classification BCE (otherwise PK/clearance MSE (~100s) drowns out the
    # classification heads -- the cause of below-chance cure AUROC).
    reg_mu = {t: float(y_reg[t][tr].mean()) for t in REG}
    reg_sd = {t: float(y_reg[t][tr].std() + 1e-9) for t in REG}
    tgt = {**{t: torch.tensor(y_class[t], dtype=torch.float32, device=dev) for t in CLASS},
           **{t: torch.tensor((y_reg[t] - reg_mu[t]) / reg_sd[t],
                              dtype=torch.float32, device=dev) for t in REG}}
    lm = {t: torch.ones(n, device=dev) for t in CLASS + REG}
    w = {t: 1.0 for t in CLASS + REG}

    model = HPAMMNet({m: v.shape[1] for m, v in X.items()}, latent_dim=64,
                     clinical_dim=clin.shape[1], treatment_dim=0,
                     tasks=CLASS, regression_tasks=REG,
                     modality_dropout_p=0.3, gated_fusion=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    trt = torch.tensor(tr, device=dev)
    model.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        xb = {m: Xt[m][trt] for m in X}
        mb = {m: maskt[m][trt] for m in X}
        out = model(xb, mb, clinical=clt[trt])
        loss = multitask_loss(out, {t: tgt[t][trt] for t in CLASS + REG},
                              {t: lm[t][trt] for t in CLASS + REG}, w, CLASS, REG)
        loss.backward(); opt.step()
    log.info("HPAMM-Net trained (%d epochs, final loss %.3f)", args.epochs, loss.item())

    # ---- evaluate HPAMM-Net (complete + parasite-dropped) ------------------
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    tet = torch.tensor(te, device=dev)

    def evaluate(mask_override=None):
        model.eval()
        mb = {m: maskt[m][tet].clone() for m in X}
        if mask_override:
            for m in mask_override:
                mb[m] = torch.zeros(len(te), device=dev)
        with torch.no_grad():
            out = model({m: Xt[m][tet] for m in X}, mb, clinical=clt[tet])
        r = {}
        for t in CLASS:
            r[t] = {"auroc": float(roc_auc_score(y_class[t][te], torch.sigmoid(out[t]).numpy()))}
        for t in REG:
            pred = out[t].numpy() * reg_sd[t] + reg_mu[t]   # back to raw units
            r[t] = {"mae": float(mean_absolute_error(y_reg[t][te], pred))}
        return r

    hp_full = evaluate()
    hp_nopar = evaluate(mask_override=["parasite_genome"])   # missingness-aware test
    log.info("HPAMM-Net full     : %s", {t: round(list(v.values())[0], 3) for t, v in hp_full.items()})
    log.info("HPAMM-Net -parasite: %s", {t: round(list(v.values())[0], 3) for t, v in hp_nopar.items()})

    results = {"baseline_xgb": base, "hpamm_full": hp_full,
               "hpamm_parasite_dropped": hp_nopar,
               "note": "resistance is parasite-driven; dropping the parasite block "
                       "should hurt resistance AUROC while host-driven PK/toxicity persist."}
    out = get_path("results", create=True) / f"hpamm_multitask_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    log.info("Read: HPAMM-Net vs baseline on complete data (multi-task in ONE model), "
             "and the graceful degradation when the parasite block is missing.")


if __name__ == "__main__":
    main()
