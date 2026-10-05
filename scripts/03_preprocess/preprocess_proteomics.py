#!/usr/bin/env python
"""Phase 3 -- Host proteomics preprocessing (blueprint Section 26, Track B).

Builds the ``patient x protein`` integration matrix from the DIA-NN precursor
matrix shipped with PXD058621 (SEARCH/REC2202.pr_matrix.tsv).

Input layout (DIA-NN pr_matrix):
  annotation columns : Protein.Group, Protein.Ids, Protein.Names, Genes,
                       First.Protein.Description, Proteotypic, Stripped.Sequence,
                       Modified.Sequence, Precursor.Charge, Precursor.Id
  sample columns     : one per run, header like
                       'Z:\\data\\...\\Sample_01.raw' (linear intensities)

Steps (Section 26 Track B):
  1. read the precursor matrix
  2. detect sample columns and clean them to sample ids (Sample_01 ...)
  3. roll precursors up to protein groups (sum of precursor intensities/sample)
  4. filter proteins present in >= a minimum fraction of samples
  5. log2 transform
  6. transpose to patient x protein
  7. write processed matrix + a protein annotation table + a manifest

This is a solid, transparent first pass. A MaxLFQ roll-up (via the `iq` or
`diann` package) is a recommended upgrade once the pipeline is validated; the
sum-of-precursors approach here is monotone with abundance and dependency-light.

Outputs -> data/processed/host_proteomics/
  proteome_patient_by_protein.tsv   (rows = samples, cols = protein groups)
  protein_annotation.tsv            (Protein.Group -> Genes, names)
"""
import _bootstrap  # noqa: F401
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import dataset_dest, get_path, load_config
from src.utils.provenance import result_manifest, write_checksum
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

ANNOT_COLS = [
    "Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
    "First.Protein.Description", "Proteotypic", "Stripped.Sequence",
    "Modified.Sequence", "Precursor.Charge", "Precursor.Id",
]
SAMPLE_RE = re.compile(r"(Sample[_-]?\d+)", re.I)


def _clean_sample_id(col: str) -> str | None:
    """Extract 'Sample_NN' from a DIA-NN run-path column header."""
    base = col.replace("\\", "/").split("/")[-1]
    base = re.sub(r"\.(raw|mzml|d)$", "", base, flags=re.I)
    m = SAMPLE_RE.search(base)
    if not m:
        return None
    # normalise to Sample_0NN
    num = re.search(r"\d+", m.group(1)).group()
    return f"Sample_{int(num):02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=None,
                    help="path to the DIA-NN pr_matrix.tsv (auto-detected if omitted)")
    ap.add_argument("--min-sample-frac", type=float, default=0.5,
                    help="keep proteins seen in >= this fraction of samples")
    args = ap.parse_args()

    raw = dataset_dest("host_proteomics", create=False)
    matrix = Path(args.matrix) if args.matrix else next(
        iter(sorted(raw.glob("**/*pr_matrix*.tsv"))), None)
    if not matrix or not matrix.exists():
        log.error("could not find pr_matrix.tsv under %s (pass --matrix)", raw)
        return
    log.info("reading precursor matrix: %s", matrix)
    df = pd.read_csv(matrix, sep="\t", low_memory=False)
    log.info("precursor matrix: %d precursors x %d columns", *df.shape)

    # --- identify sample columns --------------------------------------------
    annot = [c for c in df.columns if c in ANNOT_COLS]
    sample_cols = [c for c in df.columns if c not in ANNOT_COLS]
    id_map = {c: _clean_sample_id(c) for c in sample_cols}
    id_map = {c: s for c, s in id_map.items() if s}
    log.info("detected %d sample columns", len(id_map))
    if not id_map:
        log.error("no sample columns detected; check the header format.")
        return

    # numeric intensities
    inten = df[list(id_map)].apply(pd.to_numeric, errors="coerce")
    inten.columns = [id_map[c] for c in id_map]
    group = df["Protein.Group"].astype(str)

    # --- roll precursors up to protein groups (sum per sample) --------------
    log.info("rolling %d precursors up to protein groups...", len(df))
    prot = inten.groupby(group).sum(min_count=1)   # protein_group x sample
    log.info("protein groups: %d", prot.shape[0])

    # --- filter, log2, transpose -------------------------------------------
    present_frac = prot.notna().mean(axis=1)
    keep = present_frac >= args.min_sample_frac
    log.info("keeping %d/%d proteins present in >= %.0f%% of samples",
             int(keep.sum()), len(keep), 100 * args.min_sample_frac)
    prot = prot[keep]
    prot_log = np.log2(prot.replace(0, np.nan))
    patient_by_protein = prot_log.T                # samples x proteins
    patient_by_protein.index.name = "sample_id"

    # --- write outputs ------------------------------------------------------
    out_dir = get_path("data_processed", create=True) / "host_proteomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_path = out_dir / "proteome_patient_by_protein.tsv"
    patient_by_protein.to_csv(mat_path, sep="\t")
    write_checksum(mat_path)

    annot_tbl = (df[["Protein.Group", "Protein.Ids", "Protein.Names", "Genes",
                     "First.Protein.Description"]]
                 .drop_duplicates("Protein.Group")
                 .set_index("Protein.Group"))
    annot_tbl = annot_tbl.loc[annot_tbl.index.intersection(prot.index)]
    ann_path = out_dir / "protein_annotation.tsv"
    annot_tbl.to_csv(ann_path, sep="\t")

    result_manifest(
        "preprocess_proteomics",
        seed=load_config()["reproducibility"]["global_seed"],
        inputs=[str(matrix)],
        extra={"n_samples": int(patient_by_protein.shape[0]),
               "n_proteins": int(patient_by_protein.shape[1]),
               "min_sample_frac": args.min_sample_frac,
               "rollup": "sum-of-precursors (log2)",
               "output_matrix": str(mat_path)},
    )
    log.info("WROTE %s  (%d samples x %d proteins)",
             mat_path, patient_by_protein.shape[0], patient_by_protein.shape[1])
    log.info("annotation -> %s", ann_path)
    log.info("Next: map Sample_NN to biological patient ids in the sample map, "
             "then intersect with transcriptome + metabolome (Gate 1).")


if __name__ == "__main__":
    main()
