#!/usr/bin/env python
"""Phase 17 -- REAL-data validation on ENA dual host-parasite RNA (Sections 71, 77).

This is the real-data arm. After preprocess_dual_rna.py has produced the real
host-RNA and parasite-RNA matrices, this runs the integration methods on the
REAL matched cohort (same patients carry both layers) and measures how the
synthetic-cohort performance transports to real data -- the synthetic -> real
Mali -> external Gambia arc that is the scientific spine of the paper.

For each cohort (real Mali, external Gambia if present):
  * build two real omics blocks: host_rna, parasite_rna (PCA-reduced)
  * cluster with BLISS + a concat baseline
  * score with the paper battery (silhouette, cluster balance, bootstrap stability;
    ARI too if a severity-label file is supplied)
Then compare to the synthetic benchmark (if results/integration_benchmark_seed*.json
exist) to quantify the synthetic->real transportability gap.

Labels (optional): metadata/sample_maps/dualrna_labels_<cohort>.tsv with columns
[sample_id, group]; if absent, evaluation is unsupervised (structure only).

Output -> results/real_validation.json
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.metrics.integration_eval import full_report, transportability_report

log = get_logger(__name__)


def _load_matrix(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df


def _pca(counts: pd.DataFrame, k, seed):
    """log1p -> standardize -> PCA to k components (real expression -> latent block)."""
    from sklearn.decomposition import PCA
    X = counts.fillna(0).to_numpy(float)
    lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    X = np.log1p(X / lib * 1e6)                 # CPM before log: see note above
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    X = (X - mu) / sd
    k = min(k, X.shape[0] - 1, X.shape[1])
    return PCA(n_components=k, random_state=seed).fit_transform(X)


def _block_scale(P):
    """Scale a block of principal components to unit total variance.

    The previous code z-scored every component, which gave PC1 (often the only
    informative axis) the same weight as PC30, so dozens of noise dimensions
    outvoted the signal. Scaling the BLOCK instead preserves the variance
    ordering inside each block while still preventing the host block from
    dominating the parasite block purely by scale. This is the standard
    multiple-factor-analysis block normalisation.
    """
    tot = float(np.sqrt((P.var(axis=0)).sum()))
    return P / (tot if tot > 0 else 1.0)


def evaluate_real(cohort_dir, cohort, k_clusters, seed):
    host = _load_matrix(cohort_dir / "host_rna_counts.tsv")
    para = _load_matrix(cohort_dir / "parasite_rna_counts.tsv")
    if host is None or para is None:
        log.warning("%s: processed dual-RNA matrices not found in %s "
                    "(run preprocess_dual_rna.py first)", cohort, cohort_dir)
        return None
    common = host.index.intersection(para.index)
    if len(common) < 3 * k_clusters:
        log.warning("%s: only %d matched samples (need >= %d) -- skipping. "
                    "Has salmon quant finished for this cohort? Re-run --assemble-only "
                    "and this script AFTER quantification completes.",
                    cohort, len(common), 3 * k_clusters)
        return None
    host, para = host.loc[common], para.loc[common]
    log.info("%s: %d matched samples; host %s, parasite %s",
             cohort, len(common), host.shape, para.shape)

    X = {"host_rna": _pca(host, 30, seed), "parasite_rna": _pca(para, 20, seed)}

    # ---- optional clinical labels ------------------------------------------
    # An earlier version coerced UNLABELLED samples to class 0 with
    #   y = np.where(y < 0, 0, y)
    # which silently merged every unlabelled sample into the first real group
    # (and, for a cohort with no labels at all, produced a single constant class
    # and therefore an ARI of exactly 0.0 that looked like a result). ARI is now
    # computed ONLY over samples that carry a real label, and a cohort without
    # usable labels is reported as unsupervised (ari = None).
    lbl_path = get_path("sample_maps") / f"dualrna_labels_{cohort}.tsv"
    y_true, lab_mask, group_counts = None, None, {}
    if lbl_path.exists():
        lab = pd.read_csv(lbl_path, sep="\t").set_index("sample_id")["group"]
        srs = lab.reindex(common)
        mask = srs.notna().to_numpy()
        n_lab = int(mask.sum())
        n_grp = int(srs[mask].nunique()) if n_lab else 0
        if n_lab >= 2 * k_clusters and n_grp >= 2:
            y_true = srs[mask].astype(str).to_numpy()
            lab_mask = mask
            group_counts = srs[mask].astype(str).value_counts().to_dict()
            log.info("%s: %d/%d samples carry a clinical label across %d groups %s",
                     cohort, n_lab, len(common), n_grp, group_counts)
        else:
            log.info("%s: label file present but unusable for ARI "
                     "(%d labelled, %d distinct groups) -- evaluating unsupervised",
                     cohort, n_lab, n_grp)
    else:
        log.info("%s: no label file -- evaluating unsupervised", cohort)
    y = None   # never hand partial labels to full_report; ARI added separately

    from sklearn.cluster import KMeans
    Z = np.concatenate([_block_scale(v) for v in X.values()], axis=1)
    lab_c = KMeans(n_clusters=k_clusters, n_init=10, random_state=seed).fit_predict(Z)
    res = {"concat_kmeans": full_report(Z, lab_c, y, k=k_clusters, seed=seed)}
    bliss_labels = None
    try:
        import torch  # noqa: F401
        from src.models.bliss import train_bliss, BlissConfig
        _, latent, bliss_labels = train_bliss({m: _block_scale(v) for m, v in X.items()},
                                              BlissConfig(k_clusters=k_clusters, latent_dim=16,
                                                          epochs=150, seed=seed))
        res["BLISS"] = full_report(latent, bliss_labels, y, k=k_clusters, seed=seed)
    except Exception as e:  # noqa: BLE001
        log.warning("BLISS skipped on %s (%s)", cohort, e)
        bliss_labels = None
    # ---- ARI on the labelled subset only -----------------------------------
    assignments = {"concat_kmeans": lab_c}
    if "BLISS" in res and bliss_labels is not None:
        assignments["BLISS"] = bliss_labels
    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score
        for key, cl in assignments.items():
            res[key]["ari"] = float(adjusted_rand_score(y_true, np.asarray(cl)[lab_mask]))
            res[key]["n_labelled"] = int(lab_mask.sum())
    else:
        for key in res:
            if isinstance(res[key], dict):
                res[key]["ari"] = None

    res["_meta"] = {"n_samples": int(len(common)),
                    "labelled": y_true is not None,
                    "n_labelled": int(lab_mask.sum()) if lab_mask is not None else 0,
                    "group_counts": group_counts}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=2, help="clusters (e.g. severe vs uncomplicated)")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]
    proc = get_path("data_processed")

    cohorts = {"real_mali": proc / "dual_rna_mali",
               "external_gambia": proc / "dual_rna_gambia"}
    out = {}
    for name, d in cohorts.items():
        r = evaluate_real(d, name, args.k, seed)
        if r:
            out[name] = r
            m = r.get("BLISS", r["concat_kmeans"])
            ari = m.get("ari")
            if ari is None:
                tail = " (unsupervised -- no usable labels)"
            else:
                tail = f" ARI={ari:.3f} on {m.get('n_labelled','?')} labelled samples"
            log.info("%s: silhouette=%.3f balance=%.3f stability=%.3f%s",
                     name, m["silhouette"], m["cluster_balance"],
                     m["stability_mean_pairwise_ari"], tail)

    # synthetic -> real transportability gap (uses the synthetic benchmark if present)
    syn = sorted(glob.glob(str(get_path("results") / "integration_benchmark_seed*.json")))
    if syn and out:
        s = json.loads(open(syn[0]).read())
        smetric = s.get("BLISS", s.get("concat_kmeans", {}))
        by_cohort = {"synthetic": {"ari": smetric.get("ari", float("nan")),
                                   "silhouette": smetric.get("silhouette", float("nan"))}}
        for name, r in out.items():
            m = r.get("BLISS", r["concat_kmeans"])
            # ari is legitimately None for an unsupervised cohort; carry it as NaN
            # so the gap arithmetic stays defined instead of raising TypeError.
            a = m.get("ari")
            by_cohort[name] = {"ari": float("nan") if a is None else float(a),
                               "silhouette": m.get("silhouette", float("nan"))}
        # compute BEFORE inserting the transportability entry, otherwise the
        # comprehension would iterate over it and look for cohort keys inside it
        ari_available = {
            n: (r.get("BLISS", r.get("concat_kmeans", {})).get("ari") is not None)
            for n, r in out.items()
            if isinstance(r, dict) and ("BLISS" in r or "concat_kmeans" in r)}
        out["transportability"] = transportability_report(by_cohort)
        out["transportability"]["ari_available"] = ari_available

    (get_path("results", create=True) / "real_validation.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    log.info("wrote results/real_validation.json")
    if not out:
        log.info("No processed dual-RNA yet. Pipeline: "
                 "preprocess_dual_rna.py --build-index; then re-run this.")


if __name__ == "__main__":
    main()
