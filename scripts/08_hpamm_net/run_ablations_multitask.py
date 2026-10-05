#!/usr/bin/env python
"""Phase 15 -- HPAMM-Net ablation study (blueprint Section 75).

Removes one architectural component at a time and measures the effect, so each
novel piece has to justify itself. Ablations:
  full                 -- all blocks, gated fusion, modality dropout on
  no_gating            -- mean fusion instead of the missingness-aware gate
  no_modality_dropout  -- train without modality dropout (tests robustness source)
  host_only            -- drop the parasite block (host RNA + proteome only)
  parasite_only        -- drop the host blocks (parasite genome only)

Reports, averaged over a few cohort seeds: mean classification AUROC, resistance
AUROC (parasite-driven), PK-AUC MAE (host-driven), and -- for the dropout
comparison -- resistance AUROC when the parasite block is masked at test time
(the robustness that modality dropout is supposed to buy).

Output -> results/hpamm_ablations.json
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


def zscore(a):
    a = np.asarray(a, float); mu, sd = np.nanmean(a, 0), np.nanstd(a, 0)
    sd[sd == 0] = 1
    return np.nan_to_num((a - mu) / sd)


def load(seed):
    d = get_path("data_synthetic")
    p = d / f"cohort_seed{seed}.parquet"
    df = pd.read_parquet(p) if p.exists() else pd.read_csv(d / f"cohort_seed{seed}.tsv", sep="\t")
    X = {"host_rna": zscore(df[[c for c in df.columns if c.startswith("tx_PC")]].values),
         "host_proteome": zscore(df[[c for c in df.columns if c.startswith("prot_PC")]].values),
         "parasite_genome": zscore(df[[c for c in df.columns if c.startswith("para_")]]
                                   .apply(pd.to_numeric, errors="coerce").values)}
    clin = zscore(df[[c for c in df.columns if c.startswith("clin_")]].values)
    yc = {t: df[f"out_{t}"].to_numpy(int) for t in CLASS}
    yr = {t: df[f"out_{t}"].to_numpy(float) for t in REG}
    return X, clin, yc, yr


def train_eval(cfg, X, clin, yc, yr, seed, epochs=150):
    import torch
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from src.models.hpamm_net import HPAMMNet, multitask_loss

    torch.manual_seed(seed)
    Xu = {m: v for m, v in X.items() if m in cfg["blocks"]}
    n = clin.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); te = perm[: n // 5]; tr = perm[n // 5:]
    Xt = {m: torch.tensor(v, dtype=torch.float32) for m, v in Xu.items()}
    clt = torch.tensor(clin, dtype=torch.float32)
    mask = {m: torch.ones(n) for m in Xu}
    rmu = {t: float(yr[t][tr].mean()) for t in REG}
    rsd = {t: float(yr[t][tr].std() + 1e-9) for t in REG}
    tgt = {**{t: torch.tensor(yc[t], dtype=torch.float32) for t in CLASS},
           **{t: torch.tensor((yr[t] - rmu[t]) / rsd[t], dtype=torch.float32) for t in REG}}
    lm = {t: torch.ones(n) for t in CLASS + REG}; w = {t: 1.0 for t in CLASS + REG}

    model = HPAMMNet({m: v.shape[1] for m, v in Xu.items()}, latent_dim=64,
                     clinical_dim=clin.shape[1], treatment_dim=0,
                     tasks=CLASS, regression_tasks=REG,
                     modality_dropout_p=cfg["dropout"], gated_fusion=cfg["gating"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    trt = torch.tensor(tr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model({m: Xt[m][trt] for m in Xu}, {m: mask[m][trt] for m in Xu}, clinical=clt[trt])
        loss = multitask_loss(out, {t: tgt[t][trt] for t in CLASS + REG},
                              {t: lm[t][trt] for t in CLASS + REG}, w, CLASS, REG)
        loss.backward(); opt.step()

    tet = torch.tensor(te)
    model.eval()

    def ev(drop_par=False):
        mb = {m: mask[m][tet].clone() for m in Xu}
        if drop_par and "parasite_genome" in mb:
            mb["parasite_genome"] = torch.zeros(len(te))
        with torch.no_grad():
            out = model({m: Xt[m][tet] for m in Xu}, mb, clinical=clt[tet])
        auc = {t: roc_auc_score(yc[t][te], torch.sigmoid(out[t]).numpy()) for t in CLASS}
        pk = mean_absolute_error(yr["pk_auc"][te], out["pk_auc"].numpy() * rsd["pk_auc"] + rmu["pk_auc"])
        return auc, pk

    auc, pk = ev()
    _, _ = None, None
    res_drop = ev(drop_par=True)[0]["resistance"] if "parasite_genome" in Xu else float("nan")
    return {"mean_class_auroc": float(np.mean(list(auc.values()))),
            "resistance_auroc": float(auc["resistance"]),
            "pk_mae": float(pk),
            "resistance_auroc_parasite_dropped": float(res_drop)}


CONFIGS = {
    "full":                {"blocks": ["host_rna", "host_proteome", "parasite_genome"], "gating": True,  "dropout": 0.3},
    "no_gating":           {"blocks": ["host_rna", "host_proteome", "parasite_genome"], "gating": False, "dropout": 0.3},
    "no_modality_dropout": {"blocks": ["host_rna", "host_proteome", "parasite_genome"], "gating": True,  "dropout": 0.0},
    "host_only":           {"blocks": ["host_rna", "host_proteome"],                    "gating": True,  "dropout": 0.3},
    "parasite_only":       {"blocks": ["parasite_genome"],                              "gating": True,  "dropout": 0.3},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 23, 47])
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    try:
        import torch  # noqa: F401
    except Exception as e:
        log.error("PyTorch required: %s", e); return

    agg = {}
    for name, cfg in CONFIGS.items():
        rows = []
        for s in args.seeds:
            X, clin, yc, yr = load(s)
            rows.append(train_eval(cfg, X, clin, yc, yr, s, args.epochs))
        agg[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        log.info("%-20s meanAUROC=%.3f resistance=%.3f pk_MAE=%.2f res(-para)=%.3f",
                 name, agg[name]["mean_class_auroc"], agg[name]["resistance_auroc"],
                 agg[name]["pk_mae"], agg[name]["resistance_auroc_parasite_dropped"])

    out = get_path("results", create=True) / "hpamm_ablations.json"
    out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    log.info("Read: each row vs 'full' shows that component's contribution (Section 75).")


if __name__ == "__main__":
    main()
