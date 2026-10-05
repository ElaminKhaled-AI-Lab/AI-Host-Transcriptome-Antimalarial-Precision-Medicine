#!/usr/bin/env python
"""Phase 11 -- real-data anchored semi-synthetic cohort (blueprint Sections 33-36).

Builds the development cohort for integration-method benchmarking. It is
"semi-synthetic" because it preserves REAL structure and only synthesises new
records under explicit, documented assumptions:

  REAL and preserved:
    * per-omics internal covariance   (PCA fit to the actual transcriptome and
                                        proteome matrices -> gene-gene and
                                        protein-protein structure is real)
    * clinical phenotype distribution  (resampled from the 79 labelled
                                        transcriptome patients)
    * transcriptome <-> phenotype link (regression learned from labelled data)
    * parasite resistance structure    (real Pf8 rows resampled by geography)

  ASSUMED and documented (honest limitations):
    * proteome/metabolome are NOT phenotype-labelled in public data, so their
      synthetic values reproduce real INTERNAL covariance but are drawn
      phenotype-independently in v1 (cross-omic RNA<->protein covariance is NOT
      claimed). This is a benchmark cohort for METHOD development, not a source
      of biological claims (blueprint Section 7.1).

Outputs (per cohort seed) -> data/synthetic/
  cohort_seed<seed>.parquet   (clinical + transcriptome PCs + proteome PCs + parasite feats)
  provenance_seed<seed>.json
And once: reports/synthetic_validation.md  (real vs synthetic checks, Section 38)
"""
import _bootstrap  # noqa: F401
import argparse
import json
import re

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _norm(s):
    return re.sub(r"[-_ ]+", "-", str(s).strip()).upper()


def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"([-\d.]+)")[0], errors="coerce")


def load_real():
    proc = get_path("data_processed")
    sm = get_path("sample_maps")
    tx = pd.read_csv(proc / "host_transcriptomics/transcriptome_patient_by_gene.tsv",
                     sep="\t", index_col=0)
    clin = pd.read_csv(sm / "transcriptome_clinical.tsv", sep="\t")
    clin["key"] = clin["join_key"].map(_norm)
    tx.index = tx.index.map(_norm)
    common = tx.index.intersection(clin["key"])
    tx = tx.loc[common]
    clin = clin.set_index("key").loc[common]
    prot = pd.read_csv(proc / "host_proteomics/proteome_patient_by_protein.tsv",
                       sep="\t", index_col=0)
    para = pd.read_csv(proc / "parasite_genomics/parasite_feature_matrix.tsv",
                       sep="\t", index_col=0)
    log.info("real: transcriptome %s, clinical %d, proteome %s, parasite %s",
             tx.shape, len(clin), prot.shape, para.shape)
    return tx, clin, prot, para


def phenotype_design(clin: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix from clinical phenotype (age, sex, Hb, parasitemia, group)."""
    d = pd.DataFrame(index=clin.index)
    d["age"] = _num(clin.get("clin_age_years")).fillna(_num(clin.get("clin_age")))
    d["hb"] = _num(clin.get("clin_hemoglobin_g_dl"))
    d["log_parasitemia"] = np.log1p(_num(clin.get("clin_parasitemia")))
    d["coma"] = _num(clin.get("clin_blantyre_coma_scale"))
    sex = clin.get("clin_Sex", pd.Series(index=clin.index)).astype(str).str.upper()
    d["sex_M"] = (sex.str.startswith("M")).astype(float)
    grp = clin.get("clin_match_group", pd.Series(index=clin.index)).astype(str)
    for g in ["CM", "SMA"]:  # CM+SMA is the reference cell
        d[f"grp_{g}"] = (grp == g).astype(float)
    return d.fillna(d.mean(numeric_only=True))


def fit_generator(tx, clin, prot, kt=30, kp=20, seed=11):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    # transcriptome latent + phenotype regression (REAL linkage)
    kt = min(kt, tx.shape[0] - 1, tx.shape[1])
    pca_t = PCA(n_components=kt, random_state=seed).fit(tx.values)
    Zt = pca_t.transform(tx.values)
    design = phenotype_design(clin)
    reg = LinearRegression().fit(design.values, Zt)
    resid = Zt - reg.predict(design.values)
    Rt = np.cov(resid, rowvar=False)
    # proteome latent (REAL internal covariance; phenotype-independent in v1)
    kp = min(kp, prot.shape[0] - 1, prot.shape[1])
    prot_f = prot.fillna(prot.mean())
    pca_p = PCA(n_components=kp, random_state=seed).fit(prot_f.values)
    Zp = pca_p.transform(prot_f.values)
    mp, Cp = Zp.mean(0), np.cov(Zp, rowvar=False)
    return dict(pca_t=pca_t, reg=reg, Rt=Rt, design_cols=list(design.columns),
                design_real=design, kt=kt, kp=kp, mp=mp, Cp=Cp, rng=rng,
                pc_sd_t=Zt.std(0), pc_sd_p=Zp.std(0))


def subtype_centroids(G, n_subtypes, seed=999):
    """Fixed latent subtype directions (consistent across cohorts), per-PC scaled.

    This is the controlled-benchmark ground truth, the analogue of CMS in the
    TCGA/BLISS synthetic cohort: a molecular subtype that drives the omics
    latent blocks with a tunable effect size, so integration methods can be
    scored on how well they RECOVER it (ARI)."""
    r = np.random.default_rng(seed)
    cen_t = r.normal(0, 1, (n_subtypes, G["kt"])) * G["pc_sd_t"]
    cen_p = r.normal(0, 1, (n_subtypes, G["kp"])) * G["pc_sd_p"]
    return cen_t, cen_p


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def add_outcomes(out, par_mat, pheno_df, subtype, design_cols):
    """DECLARED outcome model (Section 35.10) -- fixed weights set here, BEFORE any
    modelling, so outcomes cannot be tuned to favour any method.

    Outcomes depend on BOTH organisms and clinical state, which is the scenario
    where a dual host-parasite, multi-task model can add value:
      resistance   <- parasite resistance features (+ subtype)
      pk_auc       <- host clinical (age, sex)
      clearance    <- resistance + pk
      cure         <- (not resistance) + adequate pk
      recrudescence<- resistance
      toxicity     <- host state + pk
    """
    w = np.random.default_rng(2024)                      # fixed declaration seed
    par_mat = np.nan_to_num(np.asarray(par_mat, float))  # missing genotypes -> 0
    P = par_mat.shape[1]
    mu, sd = par_mat.mean(0), par_mat.std(0)
    par_z = (par_mat - mu) / (sd + 1e-9)
    w_res = w.normal(0, 1, P)
    sub_res = w.normal(0, 0.5, int(subtype.max()) + 1) if subtype is not None else None
    res_score = par_z @ w_res
    if sub_res is not None:
        res_score = res_score + sub_res[subtype]
    res_score = (res_score - res_score.mean()) / (res_score.std() + 1e-9)
    p_res = _sigmoid(1.2 * res_score)
    resistance = (w.random(len(res_score)) < p_res).astype(int)

    def col(name):
        return pheno_df[f"clin_{name}"].to_numpy() if f"clin_{name}" in pheno_df else np.zeros(len(res_score))
    age = col("age"); sexM = col("sex_M")
    pk_auc = 100 - 4.0 * (age - np.nanmean(age)) - 8 * sexM + w.normal(0, 12, len(age))
    clearance = 36 + 20 * resistance - 0.12 * (pk_auc - 100) + w.normal(0, 6, len(age))
    p_cure = _sigmoid(2.2 - 3.0 * resistance + 0.02 * (pk_auc - 100))
    cure = (w.random(len(age)) < p_cure).astype(int)
    p_recr = _sigmoid(-2.0 + 3.0 * resistance)
    recrud = (w.random(len(age)) < p_recr).astype(int)
    p_tox = _sigmoid(-1.8 + 0.6 * sexM + 0.015 * (pk_auc - 100))
    tox = (w.random(len(age)) < p_tox).astype(int)

    out["out_resistance"] = resistance
    out["out_pk_auc"] = pk_auc
    out["out_clearance"] = clearance
    out["out_cure"] = cure
    out["out_recrudescence"] = recrud
    out["out_toxicity"] = tox
    return out


def sample_cohort(G, clin, para, n, seed, n_subtypes=0, effect=0.0, with_outcomes=True):
    rng = np.random.default_rng(seed)
    # 1. phenotype: bootstrap real clinical rows, jitter the numeric columns
    real = G["design_real"].values
    idx = rng.integers(0, real.shape[0], n)
    pheno = real[idx].copy().astype(float)
    numcols = [i for i, c in enumerate(G["design_cols"])
               if c in ("age", "hb", "log_parasitemia", "coma")]
    for j in numcols:
        sd = np.nanstd(real[:, j]) * 0.15
        pheno[:, j] += rng.normal(0, sd, n)
    # 2. transcriptome PCs: phenotype-conditional mean + real residual covariance
    Zt = G["reg"].predict(pheno) + rng.multivariate_normal(np.zeros(G["kt"]), G["Rt"], n)
    # 3. proteome PCs: real internal covariance (phenotype-independent, v1)
    Zp = rng.multivariate_normal(G["mp"], G["Cp"], n)
    # 4. parasite: resample real Pf8 rows (preserves resistance combinations)
    p_idx = rng.integers(0, para.shape[0], n)
    par = para.iloc[p_idx].reset_index(drop=True)

    # optional planted molecular subtype (controlled benchmark ground truth)
    subtype = None
    if n_subtypes and effect > 0:
        cen_t, cen_p = subtype_centroids(G, n_subtypes)
        subtype = rng.integers(0, n_subtypes, n)
        Zt = Zt + effect * cen_t[subtype]
        Zp = Zp + effect * cen_p[subtype]

    out = pd.DataFrame(pheno, columns=[f"clin_{c}" for c in G["design_cols"]])
    out.insert(0, "synthetic_patient_id", [f"SYN{seed}_{i:05d}" for i in range(n)])
    if subtype is not None:
        out["subtype"] = subtype
    for k in range(G["kt"]):
        out[f"tx_PC{k+1}"] = Zt[:, k]
    for k in range(G["kp"]):
        out[f"prot_PC{k+1}"] = Zp[:, k]
    par_mat = par.to_numpy(float)
    par.columns = [f"para_{c}" for c in par.columns]
    out = pd.concat([out, par], axis=1)
    if with_outcomes:
        out = add_outcomes(out, par_mat, out[[c for c in out.columns
                                              if c.startswith("clin_")]], subtype,
                           G["design_cols"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="patients per cohort")
    ap.add_argument("--n-cohorts", type=int, default=1,
                    help="number of independent cohorts (up to 10)")
    ap.add_argument("--subtypes", type=int, default=4,
                    help="planted molecular subtypes as controlled-benchmark ground "
                         "truth (0 = pure real-anchored, no planted structure)")
    ap.add_argument("--effect", type=float, default=1.5,
                    help="subtype effect size (latent-SD units); 0 disables planting")
    args = ap.parse_args()

    seeds = load_config()["seeds_cfg"]["synthetic_cohort_seeds"][: args.n_cohorts]
    tx, clin, prot, para = load_real()
    G = fit_generator(tx, clin, prot, seed=seeds[0])
    out_dir = get_path("data_synthetic", create=True)
    if args.subtypes and args.effect > 0:
        log.info("planting %d molecular subtypes at effect=%.1f (controlled benchmark)",
                 args.subtypes, args.effect)

    for seed in seeds:
        coh = sample_cohort(G, clin, para, args.n, seed,
                            n_subtypes=args.subtypes, effect=args.effect)
        try:
            coh.to_parquet(out_dir / f"cohort_seed{seed}.parquet")
            fmt = "parquet"
        except Exception:
            coh.to_csv(out_dir / f"cohort_seed{seed}.tsv", sep="\t", index=False)
            fmt = "tsv"
        prov = {"generator_version": "0.2.0", "seed": int(seed), "n": int(args.n),
                "planted_subtypes": int(args.subtypes), "subtype_effect": float(args.effect),
                "tx_pcs": G["kt"], "prot_pcs": G["kp"],
                "n_parasite_features": para.shape[1],
                "real_anchors": ["transcriptome internal covariance (PCA)",
                                 "clinical phenotype distribution",
                                 "transcriptome<->phenotype regression",
                                 "proteome internal covariance (PCA)",
                                 "parasite resistance rows (resampled)"],
                "documented_assumptions": [
                    "proteome/metabolome phenotype-independent in v1 (no public labels)",
                    "cross-omic RNA<->protein covariance NOT claimed",
                    "synthetic != clinical evidence (Section 7.1)"]}
        (out_dir / f"provenance_seed{seed}.json").write_text(json.dumps(prov, indent=2))
        log.info("cohort seed %d: %d patients x %d cols -> %s (%s)",
                 seed, coh.shape[0], coh.shape[1], out_dir, fmt)

    # ---- synthetic vs real validation (Section 38) -------------------------
    coh0 = sample_cohort(G, clin, para, args.n, seeds[0])
    tx_real_pc = G["pca_t"].transform(tx.values)
    lines = ["# Semi-synthetic validation (Section 38)", "",
             "Compares real vs synthetic on transcriptome PC structure.", ""]
    for k in range(min(5, G["kt"])):
        rmean, rsd = tx_real_pc[:, k].mean(), tx_real_pc[:, k].std()
        smean = coh0[f"tx_PC{k+1}"].mean(); ssd = coh0[f"tx_PC{k+1}"].std()
        lines.append(f"- tx_PC{k+1}: real mean={rmean:.2f} sd={rsd:.2f} | "
                     f"synthetic mean={smean:.2f} sd={ssd:.2f}")
    lines += ["", "Real transcriptome-phenotype linkage preserved via regression; "
              "proteome internal covariance preserved via PCA sampling.",
              "REJECT criterion (Section 38): if synthetic destroys real joint "
              "structure. v1 preserves marginals + internal covariance by construction."]
    (get_path("reports", create=True) / "synthetic_validation.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log.info("wrote reports/synthetic_validation.md")
    log.info("DECISION GATE 2 (Section 95): inspect synthetic_validation.md before benchmarking.")
    log.info("Next: run integration benchmarks + HPAMM-Net on data/synthetic/.")


if __name__ == "__main__":
    main()
