#!/usr/bin/env python
"""Phase 21 -- diagnose the degenerate clustering of the real dual-RNA cohorts.

The clustering reported for the real cohorts is DEGENERATE: it isolates one or
two samples from the whole remainder (Mali 1 vs 55, Gambia 2 vs 50). Silhouette
is high in that situation for a trivial reason -- a far-flung point is by
definition well separated -- so the reported silhouette advantage is an artefact
rather than evidence of subtypes. The project's own `cluster_balance`, which is
the normalised entropy of the cluster sizes, was reporting the problem all along
(0.129 corresponds exactly to a 1-vs-55 split).

This script establishes what is really there, in three stages.

STAGE 1 -- who are the outliers, and are they quality-control failures?
    The minority-cluster samples are identified and profiled against the cohort
    on host and parasite library size, transcripts detected and parasite-read
    fraction, using robust z scores (median and median absolute deviation, MAD),
    which are not themselves distorted by the outliers. If the isolated samples
    are simply shallow or failed libraries, the degeneracy is a preprocessing
    artefact with a clean fix.

STAGE 2 -- is there ANY non-degenerate structure?
    Flagged samples are removed and k is swept from 2 to 6. Every solution is
    scored for silhouette, balance, minimum cluster size and bootstrap
    stability, and solutions that are degenerate (balance below the threshold,
    or a cluster smaller than the minimum) are rejected outright. The surviving
    best solution is then tested against a PERMUTATION NULL: each feature is
    shuffled independently to destroy the covariance structure while preserving
    every marginal distribution, and the whole pipeline is rerun. A silhouette
    that does not exceed that null is not evidence of clustering, however large
    it looks.

STAGE 3 -- is the biology graded rather than discrete?
    Discrete clusters are only one hypothesis. Severity is regressed on the
    continuous host and parasite principal components with cross-validated
    logistic regression, and each component is tested univariately. A graded
    association can be real even when no discrete partition exists.

Outputs:
  tables/degeneracy_outliers_<cohort>.csv
  tables/degeneracy_ksweep_<cohort>.csv
  results/degeneracy_diagnosis_<cohort>.json
  reports/degeneracy_diagnosis_<cohort>.md
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
def _load(path):
    return pd.read_csv(path, sep="\t", index_col=0) if path.exists() else None


def _pca(counts, k, seed):
    """Identical reduction to validate_on_real_dualrna.py."""
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


def balance(labels, k=None):
    """Normalised entropy of the cluster sizes; 1.0 = perfectly balanced."""
    lab = np.asarray(labels)
    _, c = np.unique(lab, return_counts=True)
    if len(c) < 2:
        return 0.0
    p = c / c.sum()
    H = -(p * np.log2(p)).sum()
    k = k or len(c)
    return float(H / np.log2(k)) if k > 1 else 0.0


def robust_z(v):
    """(x - median) / (1.4826 * MAD); resistant to the outliers being measured."""
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = np.nanstd(v) or 1.0
    return (v - med) / scale


def cluster_once(Z, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def bootstrap_stability(Z, k, seed, n_boot=15, frac=0.8):
    """Mean pairwise ARI between clusterings of overlapping 80% subsamples."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    runs = []
    for b in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        runs.append((set(idx), dict(zip(idx, cluster_once(Z[idx], k, seed + b)))))
    aris = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            shared = sorted(runs[i][0] & runs[j][0])
            if len(shared) < 2 * k:
                continue
            a = [runs[i][1][s] for s in shared]
            bb = [runs[j][1][s] for s in shared]
            aris.append(adjusted_rand_score(a, bb))
    return float(np.mean(aris)) if aris else float("nan")


def score_solution(Z, lab, k, seed):
    from sklearn.metrics import silhouette_score
    _, counts = np.unique(lab, return_counts=True)
    out = {"k": k, "silhouette": float("nan"), "balance": balance(lab, k),
           "min_cluster_size": int(counts.min()), "sizes": counts.tolist()}
    if len(np.unique(lab)) >= 2:
        out["silhouette"] = float(silhouette_score(Z, lab))
    out["stability"] = bootstrap_stability(Z, k, seed)
    return out


# --------------------------------------------------------------------------- #
def stage1_outliers(common, host, para, Z, seed, k=2):
    """Identify the isolated samples and profile them against the cohort."""
    lab = cluster_once(Z, k, seed)
    _, counts = np.unique(lab, return_counts=True)
    minority = int(np.argmin(counts))
    is_min = lab == minority

    qc = pd.DataFrame({
        "host_library_size": host.sum(1).to_numpy(),
        "parasite_library_size": para.sum(1).to_numpy(),
        "host_transcripts_detected": (host > 0).sum(1).to_numpy(),
        "parasite_transcripts_detected": (para > 0).sum(1).to_numpy(),
    }, index=common)
    qc["parasite_read_fraction"] = (qc["parasite_library_size"] /
                                    (qc["host_library_size"] + qc["parasite_library_size"] + 1e-9))
    # distance from the centroid of the fused latent space
    qc["distance_from_centroid"] = np.linalg.norm(Z - Z.mean(0), axis=1)

    for c in list(qc.columns):
        qc[f"rz_{c}"] = robust_z(qc[c].to_numpy())
    qc["in_minority_cluster"] = is_min

    # A sample is a quality-control failure if it is far below the cohort on
    # sequencing depth. Two criteria are combined because neither alone is
    # sufficient: the robust z score fails when depth is bimodal (the median
    # absolute deviation is then inflated and hides a genuine 50-fold deficit),
    # while a plain ratio to the median is scale-free and catches exactly that.
    depth_rz = ["rz_host_library_size", "rz_host_transcripts_detected"]
    qc["qc_fail_robust_z"] = (qc[depth_rz] < -3.0).any(axis=1)
    med_lib = float(np.nanmedian(qc["host_library_size"]))
    med_det = float(np.nanmedian(qc["host_transcripts_detected"]))
    qc["depth_ratio_library"] = qc["host_library_size"] / (med_lib + 1e-9)
    qc["depth_ratio_detected"] = qc["host_transcripts_detected"] / (med_det + 1e-9)
    qc["qc_fail_ratio"] = ((qc["depth_ratio_library"] < 0.20) |
                           (qc["depth_ratio_detected"] < 0.50))
    qc["qc_fail"] = qc["qc_fail_robust_z"] | qc["qc_fail_ratio"]
    qc["extreme_distance"] = qc["rz_distance_from_centroid"] > 3.0
    qc["flagged"] = qc["qc_fail"] | qc["extreme_distance"]

    minority_ids = list(qc.index[is_min])
    overlap = int((qc["in_minority_cluster"] & qc["flagged"]).sum())
    explained = bool(len(minority_ids) > 0 and overlap == len(minority_ids))
    return qc, minority_ids, explained


def _sweep(Z, seed, ks, min_balance, min_size, tag=""):
    rows = []
    for k in ks:
        if Z.shape[0] <= k:
            continue
        lab = cluster_once(Z, k, seed)
        s = score_solution(Z, lab, k, seed)
        s["degenerate"] = (s["balance"] < min_balance) or (s["min_cluster_size"] < min_size)
        s["representation"] = tag
        rows.append(s)
    return pd.DataFrame(rows)


def peel_until_structure(Z, seed, ks, min_balance, min_size, max_frac=0.10):
    """Iteratively drop the minority cluster until a non-degenerate solution appears.

    Reports how many samples had to be discarded, which quantifies how fragile
    any structure is. Stops once more than `max_frac` of the cohort would be
    removed, because a partition that needs that much surgery is not a finding.
    """
    idx = np.arange(Z.shape[0])
    removed = []
    budget = int(max_frac * Z.shape[0])
    while True:
        sw = _sweep(Z[idx], seed, ks, min_balance, min_size, tag="peeled")
        ok = sw[~sw["degenerate"]] if not sw.empty else sw
        if not ok.empty:
            best = ok.loc[ok["silhouette"].idxmax()].to_dict()
            return best, removed, idx
        if len(removed) >= budget or len(idx) <= 3 * max(ks):
            return None, removed, idx
        lab = cluster_once(Z[idx], 2, seed)
        vals, counts = np.unique(lab, return_counts=True)
        drop = idx[lab == vals[int(np.argmin(counts))]]
        removed.extend(drop.tolist())
        idx = np.setdiff1d(idx, drop)


def stage2_ksweep(Z, seed, ks=(2, 3, 4, 5, 6), min_balance=0.5, min_size=5,
                  n_perm=20, Z_alt=None):
    """Sweep k on the pipeline representation and, if given, on an alternative."""
    sweep = _sweep(Z, seed, ks, min_balance, min_size, tag="z-scored PCs")
    if Z_alt is not None:
        sweep = pd.concat([sweep, _sweep(Z_alt, seed, ks, min_balance, min_size,
                                         tag="variance-weighted PCs")],
                          ignore_index=True)
    ok = sweep[~sweep["degenerate"]]
    best = None if ok.empty else ok.loc[ok["silhouette"].idxmax()].to_dict()

    # permutation null: destroy covariance, keep marginals
    rng = np.random.default_rng(seed)
    null = []
    for b in range(n_perm):
        Zp = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        best_null = -1.0
        for k in ks:
            if Zp.shape[0] <= k:
                continue
            lb = cluster_once(Zp, k, seed + b)
            _, c = np.unique(lb, return_counts=True)
            if balance(lb, k) < min_balance or c.min() < min_size:
                continue
            from sklearn.metrics import silhouette_score
            best_null = max(best_null, float(silhouette_score(Zp, lb)))
        if best_null > -1:
            null.append(best_null)

    p_emp = None
    if best is not None and null:
        p_emp = float((np.sum(np.asarray(null) >= best["silhouette"]) + 1) / (len(null) + 1))
    return sweep, best, (null or []), p_emp


def stage3_continuous(Hpc, Ppc, Hvar, Pvar, y, seed):
    """Is severity graded along the continuous latent axes?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    out = {"univariate": [], "multivariate": None}
    if y is None:
        return out
    mask = pd.notna(y)
    yy = pd.Series(y)[mask].astype(str).to_numpy()
    if len(np.unique(yy)) != 2 or mask.sum() < 12:
        out["note"] = "insufficient labelled samples or not a two-group comparison"
        return out
    yb = (yy == np.unique(yy)[1]).astype(int)

    blocks = [("host_PC", Hpc, Hvar, 5), ("parasite_PC", Ppc, Pvar, 3)]
    cols, names = [], []
    for pre, M, var, top in blocks:
        for i in range(min(top, M.shape[1])):
            v = M[mask.to_numpy(), i]
            u, p = stats.mannwhitneyu(v[yb == 0], v[yb == 1], alternative="two-sided")
            n1, n2 = int((yb == 0).sum()), int((yb == 1).sum())
            out["univariate"].append({
                "component": f"{pre}{i+1}",
                "variance_explained": float(var[i]),
                "test": "Mann-Whitney U", "p": float(p),
                "rank_biserial_r": float(abs(2.0 * u / (n1 * n2) - 1.0))})
            cols.append(v); names.append(f"{pre}{i+1}")
    if not cols:
        return out

    X = np.column_stack(cols)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    n_splits = min(5, int(min(np.bincount(yb))))
    if n_splits >= 3:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        prob = cross_val_predict(clf, X, yb, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(yb, prob))
    else:
        auc = float("nan")
    clf.fit(X, yb)
    coef = clf[-1].coef_.ravel()
    out["multivariate"] = {
        "model": "logistic regression on standardised components",
        "n": int(mask.sum()), "n_features": len(names),
        "cv_auroc": auc, "cv_folds": n_splits,
        "coefficients": {n: float(c) for n, c in zip(names, coef)}}
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=list(COHORT_DIR), default="external_gambia")
    ap.add_argument("--min-balance", type=float, default=0.5,
                    help="reject a solution whose normalised entropy is below this")
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=50,
                    help="permutations for the null; the smallest reachable "
                         "empirical p is 1/(n_perm+1), so use >=50 for p<0.02")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]

    d = get_path("data_processed") / COHORT_DIR[args.cohort]
    host, para = _load(d / "host_rna_counts.tsv"), _load(d / "parasite_rna_counts.tsv")
    if host is None or para is None:
        log.error("processed dual-RNA not found under %s", d)
        return
    common = host.index.intersection(para.index)
    host, para = host.loc[common], para.loc[common]
    log.info("%s: %d samples", args.cohort, len(common))

    Hpc, Hvar = _pca(host, 30, seed)
    Ppc, Pvar = _pca(para, 20, seed)
    # The pipeline z-scores every component, which gives PC1 (often >90% of the
    # variance) exactly the same weight as PC30 (often <0.1%). The informative
    # axes are then outnumbered by dozens of noise dimensions. Both
    # representations are therefore evaluated.
    Z = np.concatenate([_zc(Hpc), _zc(Ppc)], axis=1)            # pipeline
    Z_var = np.concatenate([Hpc, Ppc], axis=1)                  # variance-weighted
    log.info("representation: %d dims; host PC1 explains %.1f%% of host variance; "
             "%d host PCs exceed 1%%", Z.shape[1], 100 * Hvar[0], int((Hvar > 0.01).sum()))

    # ---------------- STAGE 1 ------------------------------------------------
    qc, minority_ids, explained = stage1_outliers(common, host, para, Z, seed)
    log.info("STAGE 1  minority cluster holds %d sample(s): %s",
             len(minority_ids), ", ".join(map(str, minority_ids[:6])))
    for sid in minority_ids[:6]:
        r = qc.loc[sid]
        log.info("   %-14s host_lib rz=%+.2f  host_genes rz=%+.2f  "
                 "para_frac rz=%+.2f  centroid_dist rz=%+.2f", sid,
                 r["rz_host_library_size"], r["rz_host_transcripts_detected"],
                 r["rz_parasite_read_fraction"], r["rz_distance_from_centroid"])
    n_qcfail = int(qc["qc_fail"].sum())
    log.info("STAGE 1  %d sample(s) fail depth QC; minority fully explained by "
             "flags: %s", n_qcfail, explained)
    qc.to_csv(get_path("tables", create=True) / f"degeneracy_outliers_{args.cohort}.csv")

    # ---------------- STAGE 2 ------------------------------------------------
    keep = ~qc["flagged"].to_numpy()
    log.info("STAGE 2  removing %d flagged sample(s); %d remain",
             int((~keep).sum()), int(keep.sum()))
    Zk, Zk_var = Z[keep], Z_var[keep]
    sweep, best, null, p_emp = stage2_ksweep(
        Zk, seed, min_balance=args.min_balance,
        min_size=args.min_cluster_size, n_perm=args.n_perm, Z_alt=Zk_var)
    for _, r in sweep.iterrows():
        log.info("   [%-22s] k=%d  sil=%.3f  balance=%.3f  min_size=%d  stab=%.3f  %s",
                 r["representation"], int(r["k"]), r["silhouette"], r["balance"],
                 int(r["min_cluster_size"]), r["stability"],
                 "DEGENERATE" if r["degenerate"] else "ok")
    sweep.to_csv(get_path("tables", create=True) /
                 f"degeneracy_ksweep_{args.cohort}.csv", index=False)
    peel_best, peel_removed = None, []
    if best is None:
        log.info("STAGE 2  NO non-degenerate solution in either representation")
        peel_best, peel_idx_removed, _ = peel_until_structure(
            Zk_var, seed, (2, 3, 4, 5, 6), args.min_balance, args.min_cluster_size)
        peel_removed = [str(common[keep][i]) for i in peel_idx_removed]
        if peel_best is None:
            log.info("STAGE 2  peeling: structure never emerges within the 10%% budget "
                     "(%d sample(s) discarded)", len(peel_removed))
        else:
            log.info("STAGE 2  peeling: a non-degenerate k=%d solution appears only after "
                     "discarding %d sample(s) (sil=%.3f balance=%.3f)",
                     int(peel_best["k"]), len(peel_removed),
                     peel_best["silhouette"], peel_best["balance"])
    else:
        log.info("STAGE 2  best non-degenerate: k=%d sil=%.3f balance=%.3f stab=%.3f",
                 int(best["k"]), best["silhouette"], best["balance"], best["stability"])
        if p_emp is not None:
            floor = 1.0 / (len(null) + 1)
            note = "(NOT above chance)" if p_emp > 0.05 else "(above chance)"
            if abs(p_emp - floor) < 1e-9:
                note += f" [at the resolution floor 1/{len(null)+1}; raise --n-perm]"
            log.info("STAGE 2  permutation null: median %.3f over %d permutations; "
                     "empirical p = %.3f %s", float(np.median(null)), len(null), p_emp, note)

    # ---------------- STAGE 3 ------------------------------------------------
    y = None
    lp = get_path("sample_maps") / f"dualrna_labels_{args.cohort}.tsv"
    if lp.exists():
        lab = pd.read_csv(lp, sep="\t")
        if {"sample_id", "group"}.issubset(lab.columns):
            y = lab.set_index("sample_id")["group"].reindex(common)
    cont = stage3_continuous(Hpc, Ppc, Hvar, Pvar, y, seed)
    if cont.get("multivariate"):
        m = cont["multivariate"]
        log.info("STAGE 3  severity ~ continuous components: cross-validated AUROC "
                 "= %.3f on %d samples (%d-fold)", m["cv_auroc"], m["n"], m["cv_folds"])
        for u in sorted(cont["univariate"], key=lambda r: r["p"])[:5]:
            log.info("   %-14s var=%.3f  r=%.3f  p=%.3g",
                     u["component"], u["variance_explained"],
                     u["rank_biserial_r"], u["p"])
    else:
        log.info("STAGE 3  skipped: %s", cont.get("note", "no usable labels"))

    # ---------------- outputs -------------------------------------------------
    summary = {
        "cohort": args.cohort, "n_samples": int(len(common)),
        "stage1": {"minority_cluster_ids": minority_ids,
                   "n_qc_fail": n_qcfail,
                   "minority_explained_by_flags": explained,
                   "n_flagged": int((~keep).sum())},
        "stage2": {"n_after_removal": int(keep.sum()),
                   "sweep": sweep.to_dict(orient="records"),
                   "best_non_degenerate": best,
                   "permutation_null_median": float(np.median(null)) if null else None,
                   "permutation_p": p_emp,
                   "peeling_best": peel_best,
                   "peeling_n_removed": len(peel_removed),
                   "peeling_removed_ids": peel_removed,
                   "host_pc1_variance": float(Hvar[0]),
                   "n_host_pcs_above_1pct": int((Hvar > 0.01).sum()),
                   "criteria": {"min_balance": args.min_balance,
                                "min_cluster_size": args.min_cluster_size}},
        "stage3": cont,
    }
    (get_path("results", create=True) /
     f"degeneracy_diagnosis_{args.cohort}.json").write_text(
        json.dumps(summary, indent=2, default=float))

    # narrative
    L = [f"# Cluster degeneracy diagnosis ({args.cohort})", "",
         f"- Samples: {len(common)}", "",
         "## Stage 1 - who is isolated, and why", "",
         f"- Minority cluster at k = 2: **{len(minority_ids)} sample(s)** "
         f"({', '.join(map(str, minority_ids[:8]))})",
         f"- Samples failing depth quality control: {n_qcfail}",
         f"- Minority fully explained by quality flags: **{explained}**", "",
         "## Stage 2 - is there any non-degenerate structure?", "",
         f"- Flagged samples removed: {int((~keep).sum())}; retained: {int(keep.sum())}",
         f"- Rejection criteria: balance < {args.min_balance} or a cluster "
         f"smaller than {args.min_cluster_size}", "",
         f"- Host PC1 explains {100*Hvar[0]:.1f}% of host variance; "
         f"{int((Hvar > 0.01).sum())} host components exceed 1%. The pipeline "
         f"z-scores all {Z.shape[1]} components, equalising informative and noise axes, "
         f"so both representations are reported below.", "",
         "| representation | k | silhouette | balance | min size | stability | verdict |",
         "|---|---|---|---|---|---|---|"]
    for _, r in sweep.iterrows():
        L.append(f"| {r['representation']} | {int(r['k'])} | {r['silhouette']:.3f} | "
                 f"{r['balance']:.3f} | {int(r['min_cluster_size'])} | "
                 f"{r['stability']:.3f} | "
                 f"{'DEGENERATE' if r['degenerate'] else 'acceptable'} |")
    L.append("")
    if best is None:
        L.append("**No non-degenerate clustering exists for k = 2 to 6 in either "
                 "representation.** The data do not support a discrete partition of "
                 "these children.")
        if peel_best is None:
            L.append("")
            L.append(f"Iteratively discarding the minority cluster did not produce "
                     f"structure either, within a budget of 10% of the cohort "
                     f"({len(peel_removed)} sample(s) discarded).")
        else:
            L.append("")
            L.append(f"A non-degenerate k = {int(peel_best['k'])} solution appears only "
                     f"after discarding {len(peel_removed)} sample(s) "
                     f"(silhouette {peel_best['silhouette']:.3f}, balance "
                     f"{peel_best['balance']:.3f}). Structure that requires removing "
                     f"that many samples should be treated as fragile.")
    else:
        L.append(f"Best acceptable solution: k = {int(best['k'])}, silhouette "
                 f"{best['silhouette']:.3f}, balance {best['balance']:.3f}, "
                 f"stability {best['stability']:.3f}.")
        if p_emp is not None:
            verdict = ("**does not exceed chance**" if p_emp > 0.05
                       else "**exceeds the permutation null**")
            L.append(f"Against a permutation null (median silhouette "
                     f"{np.median(null):.3f}), this {verdict} (empirical p = {p_emp:.3f}).")
    L += ["", "## Stage 3 - is severity graded rather than discrete?", ""]
    if cont.get("multivariate"):
        m = cont["multivariate"]
        L.append(f"Logistic regression of severity on the continuous components gives a "
                 f"cross-validated AUROC of **{m['cv_auroc']:.3f}** "
                 f"({m['n']} samples, {m['cv_folds']}-fold).")
        L += ["", "| component | variance explained | effect r | p |", "|---|---|---|---|"]
        for u in sorted(cont["univariate"], key=lambda r: r["p"]):
            L.append(f"| {u['component']} | {u['variance_explained']:.3f} | "
                     f"{u['rank_biserial_r']:.3f} | {u['p']:.3g} |")
    else:
        L.append(cont.get("note", "No usable labels for this cohort."))
    (get_path("reports", create=True) /
     f"degeneracy_diagnosis_{args.cohort}.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote tables/, results/ and reports/degeneracy_diagnosis_%s.md", args.cohort)


if __name__ == "__main__":
    main()
