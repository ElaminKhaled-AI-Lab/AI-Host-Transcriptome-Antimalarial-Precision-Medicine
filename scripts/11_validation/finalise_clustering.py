#!/usr/bin/env python
"""Phase 21 -- the decisive clustering analysis, on the corrected representation.

The diagnostic established that the degeneracy was caused by the representation,
not by the biology. Reducing each block by principal component analysis and then
z-scoring every component gives PC1 the same weight as PC30, so the informative
axes are outnumbered by noise. Removing that step yields balanced, reproducible
clusters in both cohorts (Gambia 29 vs 23, bootstrap stability 0.925).

Three questions remain, and this script settles them.

  1. Is the clustering real? The permutation test in the diagnostic was limited
     by its resolution floor of 1/(permutations + 1). Here the null is built from
     many more permutations, so the empirical p value is meaningful.

  2. Does it correspond to clinical severity? The adjusted Rand index (ARI) and a
     Fisher exact test are computed for the corrected clustering against the
     cerebral / uncomplicated malaria labels -- something never done for the
     balanced solution, only for the degenerate one.

  3. What else does it track? Every available covariate is tested against cluster
     membership, which is now possible because the clusters are large enough to
     support a test.

Cluster assignments are written out so that figures and downstream analyses use
exactly this labelling.

Outputs:
  tables/final_clusters_<cohort>.tsv          -- per-sample assignment + covariates
  tables/final_cluster_covariates_<cohort>.csv
  results/final_clustering_<cohort>.json
  reports/final_clustering_<cohort>.md
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


def _load(p):
    return pd.read_csv(p, sep="\t", index_col=0) if p.exists() else None


def _pca(counts, k, seed, normalise="cpm"):
    """Reduce a count block to principal components.

    The matrices written by preprocess_dual_rna.py are RAW salmon NumReads. Any
    principal-component analysis of raw counts is dominated by sequencing depth,
    because a deeper library raises every gene at once, so the leading component
    becomes a library-size axis rather than biology. Counts are therefore scaled
    to counts per million (CPM) before the log transform, which is the standard
    remedy. Pass normalise="none" to reproduce the original, unnormalised
    behaviour for comparison.
    """
    from sklearn.decomposition import PCA
    X = counts.fillna(0).to_numpy(float)
    if normalise == "cpm":
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        X = X / lib * 1e6
    X = np.log1p(X)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    X = (X - mu) / sd
    k = min(k, X.shape[0] - 1, X.shape[1])
    p = PCA(n_components=k, random_state=seed)
    return p.fit_transform(X), p.explained_variance_ratio_


def balance(lab, k):
    _, c = np.unique(lab, return_counts=True)
    if len(c) < 2:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(k))


def km(Z, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def stability(Z, k, seed, n_boot=25, frac=0.8):
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    runs = []
    for b in range(n_boot):
        idx = rng.choice(n, int(frac * n), replace=False)
        runs.append((set(idx), dict(zip(idx, km(Z[idx], k, seed + b)))))
    a = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            sh = sorted(runs[i][0] & runs[j][0])
            if len(sh) >= 2 * k:
                a.append(adjusted_rand_score([runs[i][1][s] for s in sh],
                                             [runs[j][1][s] for s in sh]))
    return float(np.mean(a)) if a else float("nan")


def robust_z(v):
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    s = 1.4826 * mad
    if s <= 0:
        s = np.nanstd(v) or 1.0
    return (v - med) / s


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        prev = min(prev, p[i] * m / (m - rank + 1))
        q[i] = min(prev, 1.0)
    return q


def test_cov(cluster, values, categorical):
    d = pd.DataFrame({"c": cluster, "v": values}).dropna()
    if d.empty or d["c"].nunique() < 2:
        return None
    if categorical:
        if d["v"].nunique() < 2:
            return None
        tab = pd.crosstab(d["c"], d["v"]).to_numpy()
        chi2, p, _, exp = stats.chi2_contingency(tab, correction=False)
        if tab.shape == (2, 2) and (exp < 5).any():
            p = float(stats.fisher_exact(tab)[1])
            test = "Fisher exact"
        else:
            test = "chi-square"
        n = tab.sum()
        v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if n else np.nan
        return {"test": test, "p": float(p), "effect": v,
                "effect_name": "Cramer's V", "n": int(len(d))}
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    d = d.dropna()
    g = [x["v"].to_numpy() for _, x in d.groupby("c")]
    if any(len(x) < 3 for x in g):
        return None
    if len(g) == 2:
        u, p = stats.mannwhitneyu(g[0], g[1], alternative="two-sided")
        eff = abs(2.0 * u / (len(g[0]) * len(g[1])) - 1.0)
        test, en = "Mann-Whitney U", "rank-biserial r"
    else:
        h, p = stats.kruskal(*g)
        eff = (h - len(g) + 1) / (len(d) - len(g))
        test, en = "Kruskal-Wallis", "epsilon-squared"
    return {"test": test, "p": float(p), "effect": float(eff),
            "effect_name": en, "n": int(len(d))}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=list(COHORT_DIR), default="external_gambia")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="permutations for the null; 1000 gives a resolution of 0.001")
    ap.add_argument("--min-balance", type=float, default=0.5)
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--normalise", choices=["cpm", "none"], default="cpm",
                    help="cpm = counts per million before log (correct for RNA-seq); "
                         "none = the original unnormalised behaviour")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]

    d = get_path("data_processed") / COHORT_DIR[args.cohort]
    host, para = _load(d / "host_rna_counts.tsv"), _load(d / "parasite_rna_counts.tsv")
    if host is None or para is None:
        log.error("processed dual-RNA not found under %s", d)
        return
    common = host.index.intersection(para.index)
    host, para = host.loc[common], para.loc[common]

    log.info("normalisation: %s", args.normalise)
    Hpc, Hvar = _pca(host, 30, seed, args.normalise)
    Ppc, Pvar = _pca(para, 20, seed, args.normalise)
    Z_all = np.concatenate([Hpc, Ppc], axis=1)          # variance-weighted (corrected)

    # ---- quality-control filter, identical criteria to the diagnostic --------
    hlib = host.sum(1).to_numpy()
    hdet = (host > 0).sum(1).to_numpy()
    keep = ~(((robust_z(hlib) < -3) | (robust_z(hdet) < -3)) |
             ((hlib / (np.median(hlib) + 1e-9) < 0.20) |
              (hdet / (np.median(hdet) + 1e-9) < 0.50)))
    ids = common[keep]
    Z = Z_all[keep]
    log.info("%s: %d samples; %d removed by depth QC; %d retained",
             args.cohort, len(common), int((~keep).sum()), len(ids))

    # ---- the clustering ------------------------------------------------------
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    lab = km(Z, args.k, seed)
    _, sizes = np.unique(lab, return_counts=True)
    sil = float(silhouette_score(Z, lab))
    bal = balance(lab, args.k)
    stab = stability(Z, args.k, seed)
    log.info("clustering: k=%d sizes=%s silhouette=%.3f balance=%.3f stability=%.3f",
             args.k, sizes.tolist(), sil, bal, stab)
    if bal < args.min_balance or sizes.min() < args.min_cluster_size:
        log.warning("this solution is DEGENERATE by the project criteria -- "
                    "interpret nothing below as structure")

    # ---- 1. is it real? uniform reference null -------------------------------
    # A column-wise permutation is NOT a valid null here. Shuffling a component
    # preserves its marginal distribution, so if that component is itself
    # bimodal the "null" still contains two clumps and the test loses all power
    # (verified: observed silhouette 0.901 versus permuted 0.900 on data with a
    # single dominant bimodal axis). The reference distribution used instead is
    # uniform over the bounding box of the data in its principal-axis frame,
    # which is the Tibshirani gap-statistic reference and is genuinely
    # cluster-free. Bootstrap stability is reported against the same reference
    # and is the primary criterion, because it discriminates real from spurious
    # partitions far more sharply than silhouette does.
    rng = np.random.default_rng(seed)
    lo, hi = Z.min(0), Z.max(0)
    null_sil, null_stab = [], []
    for b in range(args.n_perm):
        Zr = rng.uniform(lo, hi, size=Z.shape)
        lb = km(Zr, args.k, seed + b)
        null_sil.append(float(silhouette_score(Zr, lb)))
        null_stab.append(stability(Zr, args.k, seed + b, n_boot=10))
    null = np.asarray(null_sil)
    null_stab = np.asarray([x for x in null_stab if np.isfinite(x)])
    p_emp = float((np.sum(null >= sil) + 1) / (len(null) + 1))
    p_stab = (float((np.sum(null_stab >= stab) + 1) / (len(null_stab) + 1))
              if null_stab.size else float("nan"))
    n_degen = 0
    log.info("reference null (%d uniform references):", args.n_perm)
    log.info("  silhouette: null median %.4f, 95th pct %.4f -> observed %.3f, p = %.4f %s",
             float(np.median(null)), float(np.percentile(null, 95)), sil, p_emp,
             "above chance" if p_emp < 0.05 else "NOT above chance")
    log.info("  STABILITY : null median %.4f, 95th pct %.4f -> observed %.3f, p = %.4f %s",
             float(np.median(null_stab)) if null_stab.size else float("nan"),
             float(np.percentile(null_stab, 95)) if null_stab.size else float("nan"),
             stab, p_stab,
             "ABOVE CHANCE" if (np.isfinite(p_stab) and p_stab < 0.05) else "NOT above chance")

    # ---- covariates ----------------------------------------------------------
    cov = pd.DataFrame(index=ids)
    cov["host_library_size"] = host.loc[ids].sum(1).to_numpy()
    cov["parasite_library_size"] = para.loc[ids].sum(1).to_numpy()
    cov["host_transcripts_detected"] = (host.loc[ids] > 0).sum(1).to_numpy()
    cov["parasite_transcripts_detected"] = (para.loc[ids] > 0).sum(1).to_numpy()
    cov["parasite_read_fraction"] = (cov["parasite_library_size"] /
                                     (cov["host_library_size"] +
                                      cov["parasite_library_size"] + 1e-9))
    cov["run_accession_order"] = [int("".join(c for c in str(s) if c.isdigit()) or 0)
                                  for s in ids]
    kinds = {c: False for c in cov.columns}          # False = continuous

    sev = None
    lp_ = get_path("sample_maps") / f"dualrna_labels_{args.cohort}.tsv"
    if lp_.exists():
        lab_tsv = pd.read_csv(lp_, sep="\t")
        if {"sample_id"}.issubset(lab_tsv.columns):
            lab_tsv = lab_tsv.set_index("sample_id")
            if "group" in lab_tsv.columns:
                sev = lab_tsv["group"].reindex(ids)
                cov["severity_group"] = sev
                kinds["severity_group"] = True
            for c, nm, cat in (("age", "age_years", False), ("sex", "sex", True)):
                if c in lab_tsv.columns:
                    s = lab_tsv[c].reindex(ids)
                    if s.notna().sum() >= 8:
                        cov[nm] = s
                        kinds[nm] = cat
    for i in range(5):
        cov[f"host_PC{i+1}"] = Hpc[keep, i]
        kinds[f"host_PC{i+1}"] = False
    for i in range(3):
        cov[f"parasite_PC{i+1}"] = Ppc[keep, i]
        kinds[f"parasite_PC{i+1}"] = False

    # ---- 2. does it correspond to severity? ---------------------------------
    sev_res = None
    if sev is not None and sev.notna().sum() >= 8 and sev.dropna().nunique() == 2:
        m = sev.notna().to_numpy()
        ari = float(adjusted_rand_score(sev[m].astype(str).to_numpy(), lab[m]))
        # both sides must be plain arrays: passing an id-indexed Series next to a
        # RangeIndex Series makes crosstab align on index and silently misalign
        tab = pd.crosstab(pd.Series(lab[m], name="cluster"),
                          pd.Series(sev[m].astype(str).to_numpy(), name="severity"))
        fp = float(stats.fisher_exact(tab.to_numpy())[1]) if tab.shape == (2, 2) else float("nan")
        sev_res = {"n_labelled": int(m.sum()), "ari": ari, "fisher_p": fp,
                   "contingency": tab.to_dict()}
        log.info("severity: ARI = %.4f on %d labelled samples; Fisher p = %.4f",
                 ari, int(m.sum()), fp)
        log.info("  contingency:\n%s", tab.to_string())
    else:
        log.info("severity: no usable two-group labels for this cohort")

    # ---- 3. what does it track? ---------------------------------------------
    rows = []
    for c in cov.columns:
        r = test_cov(lab, cov[c].to_numpy(), kinds[c])
        if r:
            r["covariate"] = c
            rows.append(r)
    T = pd.DataFrame(rows)
    T["p_fdr"] = bh_fdr(T["p"].to_numpy())
    T["significant"] = T["p_fdr"] < 0.05
    T = T.sort_values(["significant", "effect"], ascending=[False, False])
    for _, r in T.iterrows():
        log.info("  %s %-30s %-16s %s=%.3f p=%.3g q=%.3g",
                 "*" if r["significant"] else " ", r["covariate"], r["test"],
                 r["effect_name"], r["effect"], r["p"], r["p_fdr"])

    # ---- outputs -------------------------------------------------------------
    assign = cov.copy()
    assign.insert(0, "cluster", lab)
    assign.index.name = "sample_id"
    assign.to_csv(get_path("tables", create=True) /
                  f"final_clusters_{args.cohort}.tsv", sep="\t")
    T.rename(columns={"significant": "significant_fdr_0.05"}).to_csv(
        get_path("tables") / f"final_cluster_covariates_{args.cohort}.csv", index=False)

    summary = {
        "cohort": args.cohort, "representation": "variance-weighted PCs (corrected)",
        "normalisation": args.normalise,
        "n_total": int(len(common)), "n_removed_qc": int((~keep).sum()),
        "n_analysed": int(len(ids)), "k": args.k,
        "cluster_sizes": sizes.tolist(), "silhouette": sil, "balance": bal,
        "stability": stab,
        "reference_null": {
            "n_references": args.n_perm,
            "reference": "uniform over the bounding box in principal-axis frame "
                         "(Tibshirani gap-statistic reference)",
            "silhouette_null_median": float(np.median(null)),
            "silhouette_null_p95": float(np.percentile(null, 95)),
            "silhouette_p": p_emp,
            "stability_null_median": float(np.median(null_stab)) if null_stab.size else None,
            "stability_null_p95": float(np.percentile(null_stab, 95)) if null_stab.size else None,
            "stability_p": p_stab,
            "primary_criterion": "bootstrap stability"},
        "severity": sev_res,
        "covariates": T.to_dict(orient="records"),
        "host_pc1_variance": float(Hvar[0]),
        "n_host_pcs_above_1pct": int((Hvar > 0.01).sum()),
    }
    (get_path("results", create=True) /
     f"final_clustering_{args.cohort}.json").write_text(json.dumps(summary, indent=2,
                                                                   default=float))

    sig = T[T["significant"]]
    L = [f"# Final clustering on the corrected representation ({args.cohort})", "",
         f"- Samples: {len(common)} total, {int((~keep).sum())} removed by depth "
         f"quality control, {len(ids)} analysed",
         f"- k = {args.k}, sizes {sizes.tolist()}, silhouette {sil:.3f}, "
         f"balance {bal:.3f}, bootstrap stability {stab:.3f}",
         f"- Host PC1 explains {100*Hvar[0]:.1f}% of host variance; "
         f"{int((Hvar > 0.01).sum())} components exceed 1%", "",
         "## 1. Is the clustering real?", "",
         f"Compared against {args.n_perm} cluster-free reference datasets drawn uniformly "
         f"over the bounding box in the principal-axis frame:", "",
         f"- silhouette {sil:.3f} versus null median {np.median(null):.4f} "
         f"(95th percentile {np.percentile(null, 95):.4f}), p = **{p_emp:.4f}**",
         f"- bootstrap stability {stab:.3f} versus null median "
         f"{np.median(null_stab):.4f} (95th percentile {np.percentile(null_stab, 95):.4f}), "
         f"p = **{p_stab:.4f}**  <- primary criterion", "",
         "A column-wise permutation was deliberately not used: it preserves each "
         "component's marginal distribution, so a bimodal component keeps two clumps "
         "and the test loses power.", "",
         "## 2. Does it correspond to clinical severity?", ""]
    if sev_res:
        L.append(f"ARI = **{sev_res['ari']:.4f}** on {sev_res['n_labelled']} labelled "
                 f"children; Fisher exact p = {sev_res['fisher_p']:.4f}.")
        L.append("")
        L.append("An ARI near zero means the clusters, although reproducible, do not "
                 "recover the cerebral / uncomplicated distinction.")
    else:
        L.append("No usable two-group labels for this cohort.")
    L += ["", "## 3. What does it track?", "",
          "| covariate | test | effect | value | p | q (FDR) | significant |",
          "|---|---|---|---|---|---|---|"]
    for _, r in T.iterrows():
        L.append(f"| {r['covariate']} | {r['test']} | {r['effect_name']} | "
                 f"{r['effect']:.3f} | {r['p']:.3g} | {r['p_fdr']:.3g} | "
                 f"{'YES' if r['significant'] else 'no'} |")
    L.append("")
    if sig.empty:
        L.append("No covariate survives correction. The clusters are reproducible but "
                 "unexplained by any measured clinical, parasitological, host or "
                 "technical variable available here.")
    else:
        t = sig.iloc[0]
        L.append(f"Strongest association: **{t['covariate']}** "
                 f"({t['effect_name']} = {t['effect']:.3f}, q = {t['p_fdr']:.3g}).")
        if "run_accession_order" in set(sig["covariate"]):
            L.append("")
            L.append("**Caution:** run-accession order is associated with cluster "
                     "membership, consistent with a sequencing batch effect.")
    (get_path("reports", create=True) /
     f"final_clustering_{args.cohort}.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote tables/final_clusters_%s.tsv + results/ + reports/", args.cohort)


if __name__ == "__main__":
    main()
