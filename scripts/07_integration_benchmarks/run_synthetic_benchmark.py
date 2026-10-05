#!/usr/bin/env python
"""Phase 14 -- integration benchmark on the semi-synthetic cohort (Sections 39-47, 69).

Runs multi-omics integration methods on a generated cohort and scores them on
the BLISS-paper evaluation battery (ARI, silhouette, cluster balance, bootstrap
stability). The omics blocks are the transcriptome PCs, proteome PCs, and
parasite features; the ground-truth label is the synthetic severity group
(CM / SMA / CM+SMA), which the transcriptome block carries by construction.

Methods:
  * concat+kmeans   -- early-integration baseline (Section 39.1 / 40)
  * BLISS           -- the group's balanced latent integration (src/models/bliss.py)
  (MOFA+/SNF/DIABLO/MOGONET are heavier; add once this pipeline is confirmed.)

Outputs -> results/
  integration_benchmark_seed<seed>.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.metrics.integration_eval import full_report

log = get_logger(__name__)


def load_cohort(seed):
    d = get_path("data_synthetic")
    p = d / f"cohort_seed{seed}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(d / f"cohort_seed{seed}.tsv", sep="\t")
    return df


def blocks(df):
    tx = df[[c for c in df.columns if c.startswith("tx_PC")]].to_numpy(float)
    prot = df[[c for c in df.columns if c.startswith("prot_PC")]].to_numpy(float)
    para = df[[c for c in df.columns if c.startswith("para_")]]
    para = para.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    return {"host_rna": tx, "host_proteome": prot, "parasite_genome": para}


def ground_truth(df):
    """Prefer the planted molecular subtype (controlled benchmark); otherwise
    fall back to the severity group (which barely differs molecularly)."""
    if "subtype" in df.columns:
        return df["subtype"].to_numpy(int), "planted_subtype"
    cm = df.get("clin_grp_CM", pd.Series(0, index=df.index)).to_numpy() > 0.5
    sma = df.get("clin_grp_SMA", pd.Series(0, index=df.index)).to_numpy() > 0.5
    return np.where(cm, 0, np.where(sma, 1, 2)), "severity_group"


def zscore(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    return (X - mu) / sd


def concat_kmeans(X, k, seed):
    from sklearn.cluster import KMeans
    Z = np.concatenate([zscore(v) for v in X.values()], axis=1)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    return Z, lab


def run_bliss(X, k, seed):
    from src.models.bliss import train_bliss, BlissConfig
    Xs = {m: zscore(v) for m, v in X.items()}
    _, latent, labels = train_bliss(
        Xs, BlissConfig(k_clusters=k, latent_dim=16, epochs=150, seed=seed))
    return latent, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=11, help="which cohort seed to use")
    ap.add_argument("--k", type=int, default=4, help="clusters (default matches subtypes)")
    args = ap.parse_args()

    gseed = load_config()["reproducibility"]["global_seed"]
    df = load_cohort(args.seed)
    X = blocks(df)
    y, label_name = ground_truth(df)
    log.info("cohort seed %d: %d patients; blocks: %s; ground-truth=%s classes: %s",
             args.seed, len(df), {m: v.shape for m, v in X.items()},
             label_name, np.bincount(y).tolist())

    results = {}
    # baseline
    Z, lab = concat_kmeans(X, args.k, gseed)
    results["concat_kmeans"] = full_report(Z, lab, y, k=args.k, seed=gseed)
    log.info("[concat+kmeans] ARI=%.3f silhouette=%.3f balance=%.3f stability=%.3f",
             results["concat_kmeans"]["ari"], results["concat_kmeans"]["silhouette"],
             results["concat_kmeans"]["cluster_balance"],
             results["concat_kmeans"]["stability_mean_pairwise_ari"])

    # BLISS (needs torch)
    try:
        import torch  # noqa: F401
        latent, labels = run_bliss(X, args.k, gseed)
        results["BLISS"] = full_report(latent, labels, y, k=args.k, seed=gseed)
        log.info("[BLISS] ARI=%.3f silhouette=%.3f balance=%.3f stability=%.3f",
                 results["BLISS"]["ari"], results["BLISS"]["silhouette"],
                 results["BLISS"]["cluster_balance"],
                 results["BLISS"]["stability_mean_pairwise_ari"])
    except Exception as e:  # noqa: BLE001
        log.warning("BLISS skipped (%s). Install torch to enable.", e)

    out = get_path("results", create=True) / f"integration_benchmark_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log.info("wrote %s", out)
    if "BLISS" in results:
        d = results["BLISS"]["ari"] - results["concat_kmeans"]["ari"]
        log.info("BLISS vs concat baseline ARI delta: %+.3f", d)
    log.info("Next: HPAMM-Net (train_hpamm_net) + transportability (synthetic->real).")


if __name__ == "__main__":
    main()
