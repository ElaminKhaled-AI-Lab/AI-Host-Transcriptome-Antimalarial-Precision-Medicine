#!/usr/bin/env python
"""Phase 3 -- Host transcriptomic preprocessing, GSE289197 (blueprint Section 25).

Starts from count-level data (GSE289197 does not release patient FASTQ). Builds
a ``patient x gene`` matrix suitable for integration.

Steps (Section 25):
  1. import raw counts (genes x samples)
  2. drop extremely low-expression genes (expressed in too few samples)
  3. library-size normalise -> counts per million (CPM)
  4. log2(CPM + 1) (variance-stabilising-ish, dependency-light)
  5. (optional) keep the top-N most variable genes for a compact ML matrix
  6. transpose -> patient x gene

For small cohorts, pathway-level features are often more stable than thousands
of genes (Section 25); a pathway-scoring step (e.g. ssGSEA / AUCell) is a
recommended follow-up once gene sets are chosen. This script emits the gene
matrix; pathway scoring is a later module.

Outputs -> data/processed/host_transcriptomics/
  transcriptome_patient_by_gene.tsv          (rows = samples, cols = genes; log2 CPM)
  transcriptome_gene_variance.tsv            (per-gene variance ranking)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default=None,
                    help="path to raw counts (auto-detected if omitted)")
    ap.add_argument("--min-count", type=int, default=10,
                    help="a gene must reach this count in >= min-frac samples")
    ap.add_argument("--min-frac", type=float, default=0.2,
                    help="fraction of samples in which a gene must be expressed")
    ap.add_argument("--top-var", type=int, default=0,
                    help="keep only the top-N most variable genes (0 = keep all)")
    args = ap.parse_args()

    raw = dataset_dest("host_transcriptomics", create=False)
    counts_path = Path(args.counts) if args.counts else next(
        iter(sorted(raw.glob("**/*raw_counts*.txt.gz"))
             or sorted(raw.glob("**/*counts*.tsv*"))
             or sorted(raw.glob("**/*counts*.txt*"))), None)
    if not counts_path or not counts_path.exists():
        log.error("count matrix not found under %s. Download it first:\n"
                  "  curl -o %s/GSE289197/GSE289197_raw_counts.txt.gz \\\n"
                  "    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289197/"
                  "suppl/GSE289197_raw_counts.txt.gz", raw, raw)
        return
    log.info("reading counts: %s", counts_path)
    df = pd.read_csv(counts_path, sep=None, engine="python", index_col=0)
    # keep only numeric sample columns (drop any gene-annotation columns)
    num = df.apply(pd.to_numeric, errors="coerce")
    sample_cols = [c for c in num.columns if num[c].notna().mean() > 0.9]
    counts = num[sample_cols]
    log.info("counts: %d genes x %d samples", *counts.shape)

    # --- filter low-expression genes ---------------------------------------
    expressed = (counts >= args.min_count).mean(axis=1) >= args.min_frac
    counts = counts[expressed]
    log.info("kept %d genes expressed (>=%d) in >=%.0f%% of samples",
             int(expressed.sum()), args.min_count, 100 * args.min_frac)

    # --- CPM + log2 ---------------------------------------------------------
    lib = counts.sum(axis=0).replace(0, np.nan)
    cpm = counts.div(lib, axis=1) * 1e6
    logcpm = np.log2(cpm + 1.0)

    # --- variance ranking (+ optional top-N) --------------------------------
    var = logcpm.var(axis=1).sort_values(ascending=False)
    if args.top_var and args.top_var < len(var):
        keep = var.head(args.top_var).index
        logcpm = logcpm.loc[keep]
        log.info("kept top %d most-variable genes", args.top_var)

    patient_by_gene = logcpm.T
    patient_by_gene.index.name = "sample_id"

    out_dir = get_path("data_processed", create=True) / "host_transcriptomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_path = out_dir / "transcriptome_patient_by_gene.tsv"
    patient_by_gene.to_csv(mat_path, sep="\t")
    write_checksum(mat_path)
    var.to_frame("log2cpm_variance").to_csv(
        out_dir / "transcriptome_gene_variance.tsv", sep="\t")

    result_manifest(
        "preprocess_transcriptomics",
        seed=load_config()["reproducibility"]["global_seed"],
        inputs=[str(counts_path)],
        extra={"n_samples": int(patient_by_gene.shape[0]),
               "n_genes": int(patient_by_gene.shape[1]),
               "min_count": args.min_count, "min_frac": args.min_frac,
               "normalisation": "log2(CPM+1)", "top_var": args.top_var,
               "output_matrix": str(mat_path)},
    )
    log.info("WROTE %s  (%d samples x %d genes)",
             mat_path, patient_by_gene.shape[0], patient_by_gene.shape[1])
    log.info("sample ids (first 5): %s", list(patient_by_gene.index[:5]))
    log.info("Next: sample-map intersection transcriptome ∩ proteome ∩ metabolome (Gate 1).")


if __name__ == "__main__":
    main()
