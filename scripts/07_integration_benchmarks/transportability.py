#!/usr/bin/env python
"""Synthetic-vs-real transportability assessment (blueprint Sections 38, 71, 77;
BLISS paper Sections 2.3, 2.8, 2.9).

THE CRUCIAL EXPERIMENT the project asks for: can an integration method learned
on the semi-synthetic cohort be applied to ANY OTHER real cohort? This runner
evaluates every integration method on the same battery across cohorts and
reports the synthetic->real->external generalization gap.

Cohort roles (malaria analog of the paper's TCGA/CPTAC arc):
  synthetic       -> data/synthetic/         (development)
  real_mali       -> data/processed/         (real anchor)
  external_gambia -> data/external_validation/ (independent transportability test)

For each method (BLISS reference + HPAMM-Net successor + established methods):
  1. fit on / embed each cohort
  2. compute ARI, silhouette, cluster balance, bootstrap stability,
     per-sample confidence
  3. layer ablation on at least one cohort
  4. summarize the transportability gaps

This is a RUNNABLE scaffold: it demonstrates the flow on a small synthetic demo
matrix when no real data are present, and marks where real matrices plug in.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.metrics.integration_eval import (
    full_report, layer_ablation, ari, transportability_report,
)

log = get_logger(__name__)


def _demo_cohort(n, seed, shift=0.0):
    """Synthetic 3-layer demo with 4 latent clusters (for scaffold self-test)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (4, 5))
    y = rng.integers(0, 4, n)
    Z = centers[y] + rng.normal(0, 1 + shift, (n, 5))
    X = {
        "host_rna": Z @ rng.normal(0, 1, (5, 40)) + rng.normal(0, 1, (n, 40)),
        "parasite_genome": Z @ rng.normal(0, 1, (5, 20)) + rng.normal(0, 1, (n, 20)),
        "clinical": Z @ rng.normal(0, 1, (5, 8)) + rng.normal(0, 1, (n, 8)),
    }
    return X, y


def _integrate_kmeans(X, k=4, seed=11):
    """Fallback integrator: concat -> PCA -> k-means (stands in for any method)."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    concat = np.concatenate([np.asarray(v) for v in X.values()], axis=1)
    d = min(10, concat.shape[1], max(2, concat.shape[0] - 1))
    Z = PCA(n_components=d, random_state=seed).fit_transform(concat)
    return Z, KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def evaluate_cohort(X, y_true, method="kmeans", k=4, seed=11):
    if method == "bliss":
        try:
            from src.models.bliss import train_bliss, BlissConfig
            _, latent, labels = train_bliss(X, BlissConfig(k_clusters=k, seed=seed, epochs=50))
        except Exception as e:
            log.warning("BLISS unavailable (%s); using PCA+kmeans fallback", e)
            latent, labels = _integrate_kmeans(X, k, seed)
    else:
        latent, labels = _integrate_kmeans(X, k, seed)
    rep = full_report(latent, labels, y_true, k=k, seed=seed)
    return rep, latent, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="kmeans",
                    choices=["kmeans", "bliss"],
                    help="integration method to evaluate (extend with hpamm)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--demo", action="store_true",
                    help="run on built-in synthetic demo (no real data needed)")
    args = ap.parse_args()

    seed = load_config()["reproducibility"]["global_seed"]
    results = {}

    if args.demo:
        cohorts = {
            "synthetic": _demo_cohort(400, seed, shift=0.0),
            "real_mali": _demo_cohort(180, seed + 1, shift=0.8),      # harder
            "external_gambia": _demo_cohort(100, seed + 2, shift=1.6),  # hardest
        }
    else:
        log.info("Non-demo mode: load real matrices per cohort here.")
        log.info("  synthetic  -> data/synthetic/  ; real_mali -> data/processed/ ;")
        log.info("  external_gambia -> data/external_validation/")
        log.info("Falling back to --demo flow so the scaffold is runnable.")
        cohorts = {
            "synthetic": _demo_cohort(400, seed, shift=0.0),
            "real_mali": _demo_cohort(180, seed + 1, shift=0.8),
            "external_gambia": _demo_cohort(100, seed + 2, shift=1.6),
        }

    for name, (X, y) in cohorts.items():
        rep, latent, labels = evaluate_cohort(X, y, method=args.method, k=args.k, seed=seed)
        results[name] = rep
        log.info("[%s] ARI=%.3f silhouette=%.3f balance=%.3f stability=%.3f conf>0.9=%.2f",
                 name, rep.get("ari", float("nan")), rep["silhouette"],
                 rep["cluster_balance"], rep["stability_mean_pairwise_ari"],
                 rep["stability_frac_confident_0.9"])

    # Layer ablation on the synthetic cohort (which layer is essential?).
    Xs, ys = cohorts["synthetic"]
    abl = layer_ablation(lambda XX: _integrate_kmeans(XX, args.k, seed)[1], Xs, ys, k=args.k)
    log.info("layer ablation (ARI drop per layer): %s",
             {k: round(v, 3) for k, v in abl.items()})

    report = transportability_report(results)
    report["layer_ablation_synthetic"] = abl
    out = get_path("results", create=True) / f"transportability_{args.method}.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log.info("wrote transportability report: %s", out)

    for key, val in report.items():
        if key.startswith("ari_gap") or key.startswith("silhouette_gap"):
            log.info("  %s = %.3f", key, val)
    log.info("Large synthetic->real gap = LOW transportability. This is the "
             "property to expose before claiming broad applicability (Section 77).")


if __name__ == "__main__":
    main()
