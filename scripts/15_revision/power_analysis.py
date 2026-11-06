#!/usr/bin/env python
"""Revision analysis for R2.8b-1 / R2.6a: sensitivity (power) of the two null findings,
plus the continuous parasite-burden gradient statistics.

A. Clustering null. For each cohort the corrected representation Z (30 host + 20
   parasite variance-weighted PCs, after the same depth QC) is used only for its
   per-component standard deviations. Gaussian data with those SDs are simulated at the
   cohort's n; two groups are planted along PC1, separated by D x SD(PC1), for a 50/50
   and a 70/30 split, D in {0,1,2,3,4,5}. The test statistic is the pipeline's bootstrap
   stability (25 subsamples of 80%); a simulated data set is called "structured" when its
   stability exceeds the 95th percentile of the uniform bounding-box reference
   distribution computed on the real cohort (from finalise_clustering). D = 0 gives the
   false-positive rate under a unimodal Gaussian. 60 simulations per cell.
   Approximation: the threshold is the real cohort's box-null p95, not re-estimated per
   simulated data set (re-estimation would cost ~1000 x more compute).
B. Severity null (Gambia, deduplicated 17 CM vs 20 UM; and the submitted 20 vs 24 count).
   Exact power of the two-sided Fisher test by enumeration, for cluster-membership
   probabilities differing between groups; observed difference with Newcombe 95% CI.
C. Continuous gradient: Spearman correlation between host PC1 and parasite read fraction
   in each cohort (the quantity the submitted text described as r = 0.876, which was a
   between-cluster rank-biserial, not a correlation along a gradient).

Output -> revision/results/power_analysis.json
"""
import _bootstrap  # noqa: F401
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact, spearmanr, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "11_validation"))
import finalise_clustering as fc  # noqa: E402
from src.utils.config import get_path  # noqa: E402

OUT = ROOT / "revision" / "results"
NSIM = 60
DS = [0, 1, 2, 3, 4, 5]


def cohort_Z(cohort):
    d = get_path("data_processed") / fc.COHORT_DIR[cohort]
    host, para = fc._load(d / "host_rna_counts.tsv"), fc._load(d / "parasite_rna_counts.tsv")
    common = host.index.intersection(para.index)
    host, para = host.loc[common], para.loc[common]
    H, _ = fc._pca(host, 30, 11, "cpm"); P, _ = fc._pca(para, 20, 11, "cpm")
    Z = np.concatenate([H, P], 1)
    hlib = host.sum(1).to_numpy(); hdet = (host > 0).sum(1).to_numpy()
    keep = ~(((fc.robust_z(hlib) < -3) | (fc.robust_z(hdet) < -3)) |
             ((hlib / (np.median(hlib) + 1e-9) < 0.20) | (hdet / (np.median(hdet) + 1e-9) < 0.50)))
    plib = para.sum(1).to_numpy()
    frac = plib / (plib + hlib)
    return Z[keep], H[keep, 0], frac[keep], int(keep.sum())


def fisher_power(n1, n2, p1, p2, alpha=0.05):
    pw = 0.0
    for x in range(n1 + 1):
        px = binom.pmf(x, n1, p1)
        for y in range(n2 + 1):
            if fisher_exact([[x, n1 - x], [y, n2 - y]])[1] <= alpha:
                pw += px * binom.pmf(y, n2, p2)
    return pw


def newcombe(x1, n1, x2, n2, z=1.96):
    def wilson(x, n):
        p = x / n; d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1); l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d, d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def main():
    res = {"clustering_power": {}, "severity_power": {}, "gradient": {}}
    rng = np.random.default_rng(20261105)
    for cohort, null_file in (("real_mali", "final_clustering_real_mali.json"),
                              ("external_gambia_dedup", "final_clustering_external_gambia_dedup.json")):
        Z, hpc1, frac, n = cohort_Z(cohort)
        thr = json.loads((ROOT / "results" / null_file).read_text())["reference_null"]["stability_null_p95"]
        sd = Z.std(0)
        rho = spearmanr(hpc1, frac)
        res["gradient"][cohort] = {"n": n, "spearman_hostPC1_vs_parasite_read_fraction": round(float(rho.correlation), 3),
                                   "p": float(rho.pvalue)}
        cells = {}
        for split in (0.5, 0.7):
            for D in DS:
                hits = 0
                for s in range(NSIM):
                    X = rng.standard_normal((n, Z.shape[1])) * sd
                    g = rng.random(n) < split
                    X[g, 0] += D * sd[0]
                    st = fc.stability(X, 2, 1000 + s)
                    hits += int(st > thr)
                cells[f"split{int(split*100)}_D{D}"] = round(hits / NSIM, 3)
                print(cohort, split, D, hits / NSIM, flush=True)
        res["clustering_power"][cohort] = {"n": n, "threshold_box_null_p95": thr, "n_sim_per_cell": NSIM,
                                           "power_by_cell": cells}
    # severity
    for label, (n1, n2) in {"deduplicated_17CM_20UM": (17, 20), "as_submitted_20CM_24UM": (20, 24)}.items():
        grid = {}
        for p1, p2 in [(0.5, 0.5), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8)]:
            grid[f"{p1:.2f}_vs_{p2:.2f}"] = round(fisher_power(n1, n2, p1, p2), 3)
        res["severity_power"][label] = grid
    sev = json.loads((ROOT / "results" / "final_clustering_external_gambia_dedup.json").read_text())["severity"]
    ct = sev["contingency"]
    x1, n1 = ct["CM"]["1"], ct["CM"]["0"] + ct["CM"]["1"]
    x2, n2 = ct["UM"]["1"], ct["UM"]["0"] + ct["UM"]["1"]
    d, lo, hi = newcombe(x1, n1, x2, n2)
    res["severity_observed"] = {"CM_in_cluster1": f"{x1}/{n1}", "UM_in_cluster1": f"{x2}/{n2}",
                                "difference": round(float(d), 3), "newcombe_95ci": [round(float(lo), 3), round(float(hi), 3)],
                                "fisher_p": sev["fisher_p"], "ari": sev["ari"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "power_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
