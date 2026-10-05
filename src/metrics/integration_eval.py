"""Integration-method evaluation battery + synthetic-vs-real transportability.

Mirrors the exact axes used in the group's BLISS paper (Elamin et al.,
Int. J. Mol. Sci. 2026, 27, 7511) so the malaria successor is judged on the
same ruler:

  * Adjusted Rand Index (ARI) vs known labels     (paper Section 2.3)
  * silhouette width                               (2.3)
  * cluster balance                                (2.3 -- caught SNF collapse)
  * bootstrap stability (15 x 80% subsamples)      (2.8, Figure 8A)
  * layer ablation (ARI drop per layer removed)    (2.8, Figure 8B)
  * per-sample bootstrap confidence                (2.8, Figure 8C)

The transportability question the project cares about ("can an integration
method learned on the synthetic cohort apply to any OTHER real cohort?") is the
paper's synthetic->real->external arc: ARI 0.90 (synthetic) -> 0.28 (real TCGA)
-> 0.13 (external CPTAC). ``transportability_report`` computes that generalization
gap for any (synthetic, real, external) triple.

Depends only on numpy + scikit-learn.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- core axes
def ari(labels_true, labels_pred) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(labels_true, labels_pred))


def silhouette(latent, labels) -> float:
    from sklearn.metrics import silhouette_score
    try:
        return float(silhouette_score(latent, labels))
    except Exception:
        return float("nan")


def cluster_balance(labels, k: int | None = None) -> float:
    """1 - normalized entropy deficit. 1.0 = perfectly balanced; ~0 = collapsed.

    Catches the BLISS-paper SNF failure mode (422/425 in one cluster) that ARI
    alone missed. Defined as the cluster-size entropy divided by log(k).
    """
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    k = k or len(counts)
    if k <= 1:
        return 0.0
    entropy = -(p * np.log(p + 1e-12)).sum()
    return float(entropy / np.log(k))


def bootstrap_stability(latent, k=4, n_subsamples=15, frac=0.8, seed=11) -> dict:
    """Mean pairwise ARI across independent subsamples (paper Section 2.8)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = latent.shape[0]
    assign = np.full((n_subsamples, n), -1)
    for i in range(n_subsamples):
        idx = rng.choice(n, int(frac * n), replace=False)
        assign[i, idx] = KMeans(n_clusters=k, n_init=10,
                                random_state=seed + i).fit_predict(latent[idx])
    aris = []
    for a in range(n_subsamples):
        for b in range(a + 1, n_subsamples):
            sh = (assign[a] >= 0) & (assign[b] >= 0)
            if sh.sum() > k:
                aris.append(adjusted_rand_score(assign[a][sh], assign[b][sh]))
    conf = np.zeros(n)
    for j in range(n):
        labs = assign[assign[:, j] >= 0, j]
        if len(labs):
            _, c = np.unique(labs, return_counts=True)
            conf[j] = c.max() / len(labs)
    return {"mean_pairwise_ari": float(np.mean(aris)) if aris else float("nan"),
            "frac_confident_0.9": float((conf > 0.9).mean()),
            "per_sample_confidence": conf}


def layer_ablation(integrate_fn, X: dict, labels_true, k=4, seed=11) -> dict:
    """ARI drop when each layer is removed (paper Section 2.8 / Figure 8B).

    ``integrate_fn(X_subset) -> labels_pred`` must cluster the given layers.
    Returns {layer: ari_drop}. Large drop => essential layer (mRNA in the paper).
    """
    full = ari(labels_true, integrate_fn(X))
    out = {"_full_ari": full}
    for layer in list(X):
        reduced = {m: v for m, v in X.items() if m != layer}
        if not reduced:
            continue
        out[layer] = full - ari(labels_true, integrate_fn(reduced))
    return out


def full_report(latent, labels_pred, labels_true=None, k=4, seed=11) -> dict:
    """One call -> all single-cohort axes."""
    rep = {
        "silhouette": silhouette(latent, labels_pred),
        "cluster_balance": cluster_balance(labels_pred, k),
        **{f"stability_{key}": val
           for key, val in bootstrap_stability(latent, k, seed=seed).items()
           if key != "per_sample_confidence"},
    }
    if labels_true is not None:
        rep["ari"] = ari(labels_true, labels_pred)
    return rep


# --------------------------------------------------- synthetic -> real gap
def transportability_report(results_by_cohort: dict[str, dict]) -> dict:
    """Summarize the synthetic->real->external generalization arc.

    ``results_by_cohort`` maps a cohort name (e.g. 'synthetic', 'real_tcga',
    'external_cptac' -> here 'synthetic', 'real_mali', 'external_gambia') to a
    dict containing at least 'ari' and 'silhouette'. Returns the drops that
    quantify whether an integration method TRANSPORTS to other cohorts.

    Interpretation (paper): a method that scores high on synthetic but collapses
    on real/external has LOW transportability -- exactly the property to expose
    before claiming a method is broadly applicable.
    """
    def g(c, key):
        return results_by_cohort.get(c, {}).get(key, float("nan"))

    out = {"per_cohort": results_by_cohort}
    if "synthetic" in results_by_cohort:
        for real in [c for c in results_by_cohort if c != "synthetic"]:
            out[f"ari_gap_synthetic_to_{real}"] = g("synthetic", "ari") - g(real, "ari")
            out[f"silhouette_gap_synthetic_to_{real}"] = (
                g("synthetic", "silhouette") - g(real, "silhouette"))
    return out
