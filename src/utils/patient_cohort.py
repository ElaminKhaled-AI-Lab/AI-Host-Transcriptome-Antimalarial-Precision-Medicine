"""Paediatric patient cohort loader for the formulation-omics arm.

Supplies the patient side of the patient x formulation problem, preferring REAL
project data and degrading gracefully:

  1. REAL dual-RNA cohort  data/processed/dual_rna_<site>/  (host + parasite
     counts) -> principal-component latents. Host PC1 is used as the metabolic /
     cytochrome-P450 (CYP) capacity proxy and PC2 as the toxicity-susceptibility
     proxy, matching the convention already used in the pharmaco-omics engine;
     the parasite block gives a 0-1 resistance burden.
  2. Synthetic cohort parquet/TSV in data/synthetic/ (cohort_seed*.parquet).
  3. A generated paediatric cohort, used only when neither of the above exists,
     so the pipeline remains testable.

Ages are read from metadata/sample_maps/dualrna_labels_<cohort>.tsv when that
file carries an `age` column (the ENA BioSample records for the Mali cohort do
expose an age attribute); otherwise a paediatric age distribution is sampled.
Only children are retained -- the cohort of interest for this project.

Abbreviations: PC principal component; CYP cytochrome P450; RNA ribonucleic acid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_path

PAEDIATRIC_MAX_AGE = 12.0


def _zscore(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def _rint(v):
    """Rank-based inverse-normal transform of a latent covariate.

    A plain z score of a principal component is heavy-tailed: one outlying
    sample in a cohort of 56 can reach z = +7. That matters here because the
    day-7 concentration is EXPONENTIAL in clearance, so such a tail produces a
    16,000-fold spread in predicted exposure, far outside anything
    physiological, and drives the therapeutic utility to its bounds for most
    children (held-out R^2 then goes negative). The rank-based inverse-normal
    transform maps the covariate onto a well-behaved standard normal, bounded
    near +/-2.4 at this sample size, which corresponds to roughly a two-fold
    clearance range -- consistent with reported inter-individual variability in
    paediatric antimalarial population pharmacokinetics.
    """
    from scipy.stats import norm, rankdata
    v = np.asarray(v, float)
    n = len(v)
    if n < 3:
        return np.zeros(n)
    r = rankdata(v)
    return norm.ppf((r - 0.5) / n)


def _pca_first(counts: pd.DataFrame, k=5, seed=11):
    from sklearn.decomposition import PCA
    X = counts.fillna(0).to_numpy(float)
    lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    X = np.log1p(X / lib * 1e6)     # CPM: raw counts make PC1 a sequencing-depth axis
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    X = (X - mu) / sd
    k = max(1, min(k, X.shape[0] - 1, X.shape[1]))
    return PCA(n_components=k, random_state=seed).fit_transform(X)


def _ages_for(index, cohort, rng):
    """Read real ages when available; otherwise sample a paediatric distribution."""
    path = get_path("sample_maps") / f"dualrna_labels_{cohort}.tsv"
    if path.exists():
        try:
            lab = pd.read_csv(path, sep="\t")
            agecol = next((c for c in lab.columns if c.lower() in
                           ("age", "age_years", "host_age")), None)
            if agecol and "sample_id" in lab.columns:
                m = lab.set_index("sample_id")[agecol]
                ages = pd.to_numeric(m.reindex(index), errors="coerce")
                if ages.notna().sum() >= max(3, 0.3 * len(index)):
                    return ages.fillna(ages.median()).to_numpy(float), "real_biosample_age"
        except Exception:
            pass
    return rng.uniform(0.5, 10.0, len(index)), "sampled_paediatric"


def load_real_cohort(site="mali", seed=11):
    """Load the real dual-RNA cohort as patient latents; return None if absent."""
    d = get_path("data_processed") / f"dual_rna_{site}"
    hp, pp = d / "host_rna_counts.tsv", d / "parasite_rna_counts.tsv"
    if not hp.exists() or not pp.exists():
        return None
    host = pd.read_csv(hp, sep="\t", index_col=0)
    para = pd.read_csv(pp, sep="\t", index_col=0)
    common = host.index.intersection(para.index)
    if len(common) < 5:
        return None
    host, para = host.loc[common], para.loc[common]
    rng = np.random.default_rng(seed)

    Hpc = _pca_first(host, k=5, seed=seed)
    # rank-based inverse-normal, not a plain z score: see _rint for why
    host_metabolic_z = _rint(Hpc[:, 0])
    host_tox_z = _rint(Hpc[:, 1]) if Hpc.shape[1] > 1 else np.zeros(len(common))
    # parasite burden -> 0-1 resistance proxy via rank of total parasite signal
    pload = para.sum(1).to_numpy(float)
    resistance = (pd.Series(pload).rank(pct=True).to_numpy()
                  if np.ptp(pload) > 0 else np.full(len(common), 0.5))

    ages, age_src = _ages_for(common, f"real_{site}", rng)
    df = pd.DataFrame({
        "patient_id": list(common), "age_years": ages,
        "host_metabolic_z": host_metabolic_z, "host_tox_z": host_tox_z,
        "parasite_resistance": resistance,
    })
    df["weight_kg"] = np.clip(2 * df["age_years"] + 8, 4, 45)
    df["source"] = f"real_dual_rna_{site}"
    df["age_source"] = age_src
    return df


def load_synthetic_cohort(seed=11):
    syn = get_path("data_synthetic")
    for p in sorted(syn.glob("cohort_seed*.parquet")) + sorted(syn.glob("cohort_seed*.tsv")):
        try:
            d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, sep="\t")
        except Exception:
            continue
        rng = np.random.default_rng(seed)
        n = len(d)
        age = (pd.to_numeric(d["clin_age"], errors="coerce").to_numpy()
               if "clin_age" in d.columns else rng.uniform(0.5, 10, n))
        age = np.clip(np.nan_to_num(age, nan=3.0), 0.3, PAEDIATRIC_MAX_AGE)

        def zc(col):
            if col in d.columns:
                return _zscore(pd.to_numeric(d[col], errors="coerce").fillna(0).to_numpy())
            return np.zeros(n)

        para_cols = [c for c in d.columns if c.startswith("para_")]
        if para_cols:
            pm = d[para_cols].apply(pd.to_numeric, errors="coerce")
            res = pd.Series(np.nan_to_num(pm.mean(1))).rank(pct=True).to_numpy()
        else:
            res = rng.uniform(0, 1, n)
        out = pd.DataFrame({
            "patient_id": [f"SYN{i:04d}" for i in range(n)], "age_years": age,
            "host_metabolic_z": zc("tx_PC1"), "host_tox_z": zc("tx_PC2"),
            "parasite_resistance": res})
        out["weight_kg"] = np.clip(2 * out["age_years"] + 8, 4, 45)
        out["source"] = f"synthetic:{p.name}"
        out["age_source"] = "synthetic"
        return out
    return None


def generate_cohort(n=120, seed=11):
    """Last-resort generated paediatric cohort so the pipeline stays testable."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(0.5, 10.0, n)
    df = pd.DataFrame({
        "patient_id": [f"GEN{i:04d}" for i in range(n)], "age_years": age,
        "host_metabolic_z": rng.normal(0, 1, n), "host_tox_z": rng.normal(0, 1, n),
        "parasite_resistance": rng.uniform(0, 1, n)})
    df["weight_kg"] = np.clip(2 * df["age_years"] + 8, 4, 45)
    df["source"] = "generated"
    df["age_source"] = "generated"
    return df


def load_patients(prefer_site="mali", paediatric_only=True, seed=11, n_generated=120):
    """Load the best available cohort, restricted to children by default."""
    df = load_real_cohort(prefer_site, seed=seed)
    if df is None:
        df = load_synthetic_cohort(seed=seed)
    if df is None:
        df = generate_cohort(n=n_generated, seed=seed)
    if paediatric_only:
        df = df[df["age_years"] <= PAEDIATRIC_MAX_AGE].reset_index(drop=True)
    return df
