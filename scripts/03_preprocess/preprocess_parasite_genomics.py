#!/usr/bin/env python
"""Phase 3 -- Parasite genomic preprocessing, Pf8 (blueprint Section 28).

Builds a ``sample x parasite-feature`` matrix from the analysis-ready Pf8 tables
downloaded from Zenodo:
  Pf8_samples.txt                                  (metadata + QC pass)
  Pf8_drug_resistance_marker_genotypes.tsv         (resistance loci genotypes)
  Pf8_inferred_resistance_status_classification.tsv(inferred R/S per drug)
  Pf8_cnv_calls.tsv                                 (copy-number, incl. plasmepsin/mdr1)
  Pf8_fws.tsv                                       (Fws = within-host complexity)

The script is column-adaptive: it auto-detects the sample-id column, joins the
tables, keeps QC-pass samples, optionally filters to a study/country cohort, and
encodes features:
  - resistance-marker genotypes -> categorical codes (+ optional one-hot)
  - inferred resistance status   -> binary R=1 / S=0 per drug
  - CNV                          -> numeric copy number
  - Fws                          -> numeric complexity of infection

Geography (Country) is retained as a column because parasite allele frequencies
vary by region (Section 28) and must be modelled downstream.

Outputs -> data/processed/parasite_genomics/
  parasite_feature_matrix.tsv     (rows = samples, cols = features)
  parasite_feature_dictionary.tsv (feature -> type/source)
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import dataset_dest, get_path, load_config
from src.utils.provenance import result_manifest, write_checksum
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

SAMPLE_ID_CANDIDATES = ["Sample", "sample", "Sample ID", "sample_id", "ID"]


def _read_any(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("missing (skipping): %s", path.name)
        return None
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log.info("loaded %s: %d rows x %d cols", path.name, *df.shape)
    return df


def _sample_col(df: pd.DataFrame) -> str:
    for c in SAMPLE_ID_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]  # fall back to first column


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default=None, help="filter Study contains this, e.g. 1167")
    ap.add_argument("--country", default=None, help="filter Country == this, e.g. Tanzania")
    ap.add_argument("--qc-pass-only", action="store_true", default=True)
    ap.add_argument("--one-hot", action="store_true",
                    help="one-hot encode resistance genotypes instead of codes")
    args = ap.parse_args()

    raw = dataset_dest("parasite_genomics_pf8", create=False)
    samples = _read_any(raw / "Pf8_samples.txt")
    if samples is None:
        log.error("Pf8_samples.txt not found under %s", raw)
        return
    sid = _sample_col(samples)
    log.info("sample id column: '%s'", sid)

    # --- cohort filtering ---------------------------------------------------
    df = samples.copy()
    if args.qc_pass_only and "QC pass" in df.columns:
        before = len(df)
        df = df[df["QC pass"].astype(str).str.lower().isin(["true", "1", "pass", "yes"])]
        log.info("QC pass: kept %d/%d", len(df), before)
    if args.study and "Study" in df.columns:
        df = df[df["Study"].astype(str).str.contains(str(args.study), na=False)]
        log.info("study filter '%s': %d samples", args.study, len(df))
    if args.country and "Country" in df.columns:
        df = df[df["Country"].astype(str) == args.country]
        log.info("country filter '%s': %d samples", args.country, len(df))
    if df.empty:
        log.error("no samples after filtering; relax --study/--country.")
        return

    keep_ids = set(df[sid].astype(str))
    feat = df[[sid]].copy()
    feat[sid] = feat[sid].astype(str)
    if "Country" in df.columns:
        feat["geo_country"] = df["Country"].values
    if "Population" in df.columns:
        feat["geo_population"] = df["Population"].values
    dictionary = [("geo_country", "categorical", "samples"),
                  ("geo_population", "categorical", "samples")]

    def join(table: Path, kind: str):
        t = _read_any(table)
        if t is None:
            return
        tsid = _sample_col(t)
        t[tsid] = t[tsid].astype(str)
        t = t[t[tsid].isin(keep_ids)]
        cols = [c for c in t.columns if c != tsid]
        merged = feat.merge(t[[tsid] + cols], left_on=sid, right_on=tsid, how="left")
        if tsid != sid and tsid in merged.columns:
            merged = merged.drop(columns=[tsid])
        for c in cols:
            dictionary.append((c, kind, table.name))
        return merged

    # --- resistance marker genotypes (categorical) --------------------------
    m = join(raw / "Pf8_drug_resistance_marker_genotypes.tsv", "resistance_genotype")
    if m is not None:
        feat = m
    # --- inferred resistance status (binary per drug) -----------------------
    m = join(raw / "Pf8_inferred_resistance_status_classification.tsv", "inferred_resistance")
    if m is not None:
        feat = m
    # --- CNV (numeric) ------------------------------------------------------
    m = join(raw / "Pf8_cnv_calls.tsv", "cnv")
    if m is not None:
        feat = m
    # --- Fws complexity (numeric) ------------------------------------------
    m = join(raw / "Pf8_fws.tsv", "complexity_fws")
    if m is not None:
        feat = m

    feat = feat.rename(columns={sid: "sample_id"}).set_index("sample_id")

    # --- encode categoricals -----------------------------------------------
    if args.one_hot:
        cat_cols = feat.select_dtypes(include="object").columns
        feat_enc = pd.get_dummies(feat, columns=list(cat_cols), dummy_na=True)
    else:
        feat_enc = feat.copy()
        for c in feat_enc.select_dtypes(include="object").columns:
            feat_enc[c] = feat_enc[c].astype("category").cat.codes.replace(-1, np.nan)

    out_dir = get_path("data_processed", create=True) / "parasite_genomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_path = out_dir / "parasite_feature_matrix.tsv"
    feat_enc.to_csv(mat_path, sep="\t")
    write_checksum(mat_path)
    pd.DataFrame(dictionary, columns=["feature", "type", "source"]).to_csv(
        out_dir / "parasite_feature_dictionary.tsv", sep="\t", index=False)

    result_manifest(
        "preprocess_parasite_genomics",
        seed=load_config()["reproducibility"]["global_seed"],
        extra={"n_samples": int(feat_enc.shape[0]),
               "n_features": int(feat_enc.shape[1]),
               "study": args.study, "country": args.country,
               "encoding": "one_hot" if args.one_hot else "category_codes",
               "output_matrix": str(mat_path)},
    )
    log.info("WROTE %s  (%d samples x %d features)",
             mat_path, feat_enc.shape[0], feat_enc.shape[1])
    log.info("Feature columns present: %s", list(feat.columns)[:20])
    log.info("Inspect parasite_feature_dictionary.tsv to see feature sources. "
             "If resistance columns need specific encoding, tell me the names.")


if __name__ == "__main__":
    main()
