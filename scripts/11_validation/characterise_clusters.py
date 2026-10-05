#!/usr/bin/env python
"""Phase 21 -- what do the dual-RNA clusters actually track?

The integration recovers clusters that are strongly separated and highly
reproducible (Gambia silhouette 0.737, bootstrap stability 1.000) yet show no
correspondence with clinical severity (adjusted Rand index, ARI, 0.022 over the
44 labelled Gambian children). A reader will immediately ask what the clusters
DO reflect. This script answers that question rather than leaving it open.

It reproduces exactly the clustering used in validate_on_real_dualrna.py (same
principal-component reduction, same seed, same model), then tests the cluster
assignment against every covariate available for the cohort:

  clinical    severity group (cerebral vs uncomplicated malaria), sex
  demographic age
  parasite    parasite-read fraction, parasite transcripts detected,
              parasite library size
  host        host transcripts detected, host library size
  technical   run-accession order, a proxy for sequencing run / batch

Statistics. A categorical covariate is tested with the chi-square test of
independence, or Fisher's exact test for a 2x2 table with small expected counts,
and reported with Cramer's V as the effect size. A continuous covariate is
tested with the Mann-Whitney U test for two clusters, or the Kruskal-Wallis test
for more, and reported with the rank-biserial correlation (two clusters) or
epsilon-squared (more). All p values are corrected across covariates with the
Benjamini-Hochberg false-discovery-rate (FDR) procedure.

The principal components are tested the same way, so the report also states
which axis of variation the clusters sit on and how much variance it carries.

Outputs:
  tables/cluster_characterisation_<cohort>.csv
  results/cluster_characterisation_<cohort>.json
  reports/cluster_characterisation_<cohort>.md
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

COHORT_DIR = {"real_mali": "dual_rna_mali", "external_gambia": "dual_rna_gambia"}


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def _load(path):
    return pd.read_csv(path, sep="\t", index_col=0) if path.exists() else None


def _pca(counts, k, seed):
    """Identical to validate_on_real_dualrna.py so the clusters match exactly."""
    from sklearn.decomposition import PCA
    X = np.log1p(counts.fillna(0).to_numpy(float))
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    X = (X - mu) / sd
    k = min(k, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=k, random_state=seed)
    return p.fit_transform(X), p.explained_variance_ratio_


def _zc(a):
    mu, sd = a.mean(0), a.std(0)
    sd[sd == 0] = 1
    return (a - mu) / sd


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p values (no statsmodels dependency)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        val = p[i] * m / (m - rank + 1)
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def test_categorical(cluster, values):
    d = pd.DataFrame({"c": cluster, "v": values}).dropna()
    if d.empty or d["v"].nunique() < 2 or d["c"].nunique() < 2:
        return None
    tab = pd.crosstab(d["c"], d["v"])
    if tab.shape == (2, 2) and (stats.chi2_contingency(tab.to_numpy(),
                                                       correction=False)[3] < 5).any():
        p = float(stats.fisher_exact(tab.to_numpy())[1])
        test = "Fisher exact"
    else:
        p = float(stats.chi2_contingency(tab.to_numpy(), correction=False)[1])
        test = "chi-square"
    return {"test": test, "p": p, "effect": cramers_v(tab.to_numpy()),
            "effect_name": "Cramer's V", "n": int(len(d)),
            "detail": tab.to_dict()}


def test_continuous(cluster, values):
    d = pd.DataFrame({"c": cluster, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if d.empty or d["c"].nunique() < 2:
        return None
    groups = [g["v"].to_numpy() for _, g in d.groupby("c")]
    if any(len(g) < 3 for g in groups):
        return None
    if len(groups) == 2:
        u, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        n1, n2 = len(groups[0]), len(groups[1])
        eff = float(2.0 * u / (n1 * n2) - 1.0)          # rank-biserial correlation
        test, ename = "Mann-Whitney U", "rank-biserial r"
    else:
        h, p = stats.kruskal(*groups)
        n = len(d)
        eff = float((h - len(groups) + 1) / (n - len(groups)))   # epsilon-squared
        test, ename = "Kruskal-Wallis", "epsilon-squared"
    med = {int(k): float(np.median(g["v"])) for k, g in d.groupby("c")}
    return {"test": test, "p": float(p), "effect": abs(eff), "effect_name": ename,
            "n": int(len(d)), "detail": {"median_per_cluster": med}}


# --------------------------------------------------------------------------- #
def build_covariates(cohort, common, host, para, seed):
    """Assemble every covariate available for this cohort."""
    cov, kind = {}, {}
    d = get_path("data_processed") / COHORT_DIR[cohort]

    # --- clinical / demographic, from the label file -------------------------
    lp = get_path("sample_maps") / f"dualrna_labels_{cohort}.tsv"
    if lp.exists():
        lab = pd.read_csv(lp, sep="\t")
        if "sample_id" in lab.columns:
            lab = lab.set_index("sample_id")
            if "group" in lab.columns:
                cov["severity_group"] = lab["group"].reindex(common)
                kind["severity_group"] = "categorical"
            for c, nm in (("age", "age_years"), ("sex", "sex")):
                if c in lab.columns:
                    s = lab[c].reindex(common)
                    if s.notna().sum() >= 6:
                        cov[nm] = s
                        kind[nm] = "continuous" if c == "age" else "categorical"

    # --- parasite signal ------------------------------------------------------
    frac = _load(d / "parasite_read_fraction.tsv")
    if frac is not None:
        cov["parasite_read_fraction"] = frac.reindex(common).iloc[:, 0]
    else:
        cov["parasite_read_fraction"] = (para.sum(1) /
                                         (host.sum(1) + para.sum(1) + 1e-9)).reindex(common)
    kind["parasite_read_fraction"] = "continuous"
    cov["parasite_transcripts_detected"] = (para > 0).sum(1).reindex(common)
    kind["parasite_transcripts_detected"] = "continuous"
    cov["parasite_library_size"] = para.sum(1).reindex(common)
    kind["parasite_library_size"] = "continuous"

    # --- host signal ----------------------------------------------------------
    cov["host_transcripts_detected"] = (host > 0).sum(1).reindex(common)
    kind["host_transcripts_detected"] = "continuous"
    cov["host_library_size"] = host.sum(1).reindex(common)
    kind["host_library_size"] = "continuous"

    # --- technical proxy ------------------------------------------------------
    # Run accessions are issued sequentially by the archive, so their numeric
    # order approximates submission / sequencing order. A cluster that tracks it
    # is a warning sign of a batch effect rather than biology.
    order = pd.Series(
        [int("".join(ch for ch in str(s) if ch.isdigit()) or 0) for s in common],
        index=common)
    cov["run_accession_order"] = order
    kind["run_accession_order"] = "continuous"

    return pd.DataFrame(cov, index=common), kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=list(COHORT_DIR), default="external_gambia")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]

    d = get_path("data_processed") / COHORT_DIR[args.cohort]
    host, para = _load(d / "host_rna_counts.tsv"), _load(d / "parasite_rna_counts.tsv")
    if host is None or para is None:
        log.error("processed dual-RNA not found under %s", d)
        return
    common = host.index.intersection(para.index)
    if len(common) < 3 * args.k:
        log.error("%s: only %d matched samples", args.cohort, len(common))
        return
    host, para = host.loc[common], para.loc[common]
    log.info("%s: %d samples", args.cohort, len(common))

    # ---- reproduce the published clustering exactly --------------------------
    Hpc, Hvar = _pca(host, 30, seed)
    Ppc, Pvar = _pca(para, 20, seed)
    Z = np.concatenate([_zc(Hpc), _zc(Ppc)], axis=1)

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, n_init=10, random_state=seed).fit_predict(Z)
    clusters, source = km, "concat_kmeans"
    try:
        import torch  # noqa: F401
        from src.models.bliss import train_bliss, BlissConfig
        _, _, bl = train_bliss({"host_rna": _zc(Hpc), "parasite_rna": _zc(Ppc)},
                               BlissConfig(k_clusters=args.k, latent_dim=16,
                                           epochs=150, seed=seed))
        clusters, source = np.asarray(bl), "BLISS"
    except Exception as e:  # noqa: BLE001
        log.warning("BLISS unavailable (%s); characterising the k-means clusters", e)
    sizes = pd.Series(clusters).value_counts().sort_index().to_dict()
    log.info("cluster source=%s sizes=%s", source, sizes)

    # ---- covariate association ----------------------------------------------
    cov, kind = build_covariates(args.cohort, common, host, para, seed)
    rows = []
    for name in cov.columns:
        r = (test_categorical(clusters, cov[name].to_numpy())
             if kind[name] == "categorical"
             else test_continuous(clusters, cov[name].to_numpy()))
        if r is None:
            log.info("  %-30s skipped (insufficient data)", name)
            continue
        r["covariate"] = name
        r["type"] = kind[name]
        rows.append(r)

    # ---- principal components ------------------------------------------------
    for i in range(min(5, Hpc.shape[1])):
        r = test_continuous(clusters, Hpc[:, i])
        if r:
            r.update(covariate=f"host_PC{i+1}", type="continuous",
                     detail={**r["detail"], "variance_explained": float(Hvar[i])})
            rows.append(r)
    for i in range(min(3, Ppc.shape[1])):
        r = test_continuous(clusters, Ppc[:, i])
        if r:
            r.update(covariate=f"parasite_PC{i+1}", type="continuous",
                     detail={**r["detail"], "variance_explained": float(Pvar[i])})
            rows.append(r)

    if not rows:
        log.error("no covariate could be tested")
        return

    T = pd.DataFrame(rows)
    T["p_fdr"] = bh_fdr(T["p"].to_numpy())
    T["significant"] = T["p_fdr"] < 0.05          # valid identifier: itertuples-safe
    T = T.sort_values(["significant", "effect"], ascending=[False, False])

    out_cols = ["covariate", "type", "test", "n", "effect_name", "effect",
                "p", "p_fdr", "significant"]
    tbl = get_path("tables", create=True) / f"cluster_characterisation_{args.cohort}.csv"
    T[out_cols].rename(columns={"significant": "significant_fdr_0.05"}).to_csv(tbl, index=False)

    summary = {
        "cohort": args.cohort, "n_samples": int(len(common)),
        "cluster_source": source, "k": args.k, "cluster_sizes": sizes,
        "n_covariates_tested": int(len(T)),
        "n_significant_fdr_0.05": int(T["significant"].sum()),
        "results": T.to_dict(orient="records"),
    }
    (get_path("results", create=True) /
     f"cluster_characterisation_{args.cohort}.json").write_text(
        json.dumps(summary, indent=2, default=float))

    # ---- narrative report ----------------------------------------------------
    sig = T[T["significant"]]
    lines = [f"# What the dual-RNA clusters track ({args.cohort})", "",
             f"- Samples: {len(common)}; clustering: {source}, k = {args.k}, sizes {sizes}",
             f"- Covariates and components tested: {len(T)}",
             f"- Significant after Benjamini-Hochberg FDR correction (q < 0.05): {len(sig)}",
             "",
             "| covariate | type | test | effect | value | p | q (FDR) | significant |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        lines.append(f"| {r['covariate']} | {r['type']} | {r['test']} | {r['effect_name']} | "
                     f"{r['effect']:.3f} | {r['p']:.3g} | {r['p_fdr']:.3g} | "
                     f"{'YES' if r['significant'] else 'no'} |")
    lines += ["", "## Interpretation", ""]
    if len(sig) == 0:
        lines.append("No covariate reached significance. The clusters are reproducible "
                     "but are not explained by any measured clinical, parasitological, "
                     "host or technical variable available here.")
    else:
        top = sig.iloc[0]
        lines.append(f"The strongest association is **{top['covariate']}** "
                     f"({top['effect_name']} = {top['effect']:.3f}, q = {top['p_fdr']:.3g}).")
        if "run_accession_order" in set(sig["covariate"]):
            lines.append("")
            lines.append("**Caution:** run-accession order is associated with cluster "
                         "membership, which is consistent with a sequencing batch effect. "
                         "State this openly and treat the clusters as partly technical.")
        if "severity_group" in set(sig["covariate"]):
            lines.append("")
            lines.append("Severity is associated with cluster membership on this test even "
                         "though the ARI is low, which indicates a real but weak "
                         "relationship rather than a clean partition.")
    (get_path("reports", create=True) /
     f"cluster_characterisation_{args.cohort}.md").write_text("\n".join(lines),
                                                              encoding="utf-8")

    log.info("tested %d covariates; %d significant after FDR correction",
             len(T), int(T["significant"].sum()))
    for _, r in T.iterrows():
        flag = "*" if r["significant"] else " "
        log.info("  %s %-30s %-16s %s=%.3f  p=%.3g  q=%.3g", flag, r["covariate"],
                 r["test"], r["effect_name"], r["effect"], r["p"], r["p_fdr"])
    log.info("wrote %s + results/ + reports/", tbl)


if __name__ == "__main__":
    main()
