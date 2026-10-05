#!/usr/bin/env python
"""Stage 2 -- Mali dual host-parasite transcriptomics, PRJNA962942 (Section 13).

Dual RNA sequencing of infected pediatric blood -> reads assignable to BOTH
Homo sapiens and Plasmodium falciparum. Enables HostRNA_i <-> ParasiteRNA_i
for the same biological sample.

FASTQ files are large. Default behaviour lists the run report FIRST so you can
inspect run count and total bytes before committing disk. Use --download to
fetch, --max-runs to cap for a pilot, or use `prefetch`/`fasterq-dump`
(sra-tools) if you prefer the SRA route.

References:
  BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA962942
  Article:    https://pmc.ncbi.nlm.nih.gov/articles/PMC10918175/
"""
import _bootstrap  # noqa: F401
import argparse

import _ena
from src.utils.config import get_dataset, dataset_dest
from src.utils.provenance import dataset_record, inventory_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
ACCESSION = "PRJNA962942"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="download FASTQ files")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="cap number of runs (pilot)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = get_dataset("dual_rna_mali")
    dest = dataset_dest("dual_rna_mali")
    rows = _ena.fetch_filereport(ACCESSION)
    _ena.save_report(rows, dest, ACCESSION)

    total_bytes = sum(int(b) for r in rows for b in r.get("fastq_bytes", "").split(";")
                      if b.isdigit())
    log.info("total FASTQ size ~ %.1f GB across %d runs",
             total_bytes / 1e9, len(rows))

    n = 0
    if args.download:
        n = _ena.download_fastqs(rows, dest, max_runs=args.max_runs,
                                 overwrite=args.overwrite)
    else:
        log.info("run report only. Re-run with --download (optionally --max-runs N).")

    dataset_record(
        accession=ACCESSION, repository="SRA/ENA",
        download_url=ds["url_landing"], dest_dir=dest,
        publication=ds["article"], raw_or_processed_status="raw_fastq",
        organism="Homo sapiens + Plasmodium falciparum", tissue="peripheral blood",
        sample_count=len(rows),
        files=inventory_dir(dest, compute_checksums=False),
        notes="Dual RNA-seq; map to combined human+Pf reference, then split.",
    )
    log.info("done. %d fastq files downloaded.", n)


if __name__ == "__main__":
    main()
