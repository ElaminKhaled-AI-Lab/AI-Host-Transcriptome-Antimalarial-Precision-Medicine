#!/usr/bin/env python
"""Stage 2 -- Gambian dual host-parasite transcriptomics (blueprint Section 14).

BioStudies E-MTAB-6413 / ENA study ERP116722 / project PRJEB33892. Dual RNA-seq
of peripheral blood from Gambian children with severe or uncomplicated
P. falciparum malaria. Used to test TRANSPORTABILITY of relationships learned
in Mali (do NOT pool before checking protocol/severity/age/batch -- Section 14).

References:
  BioStudies: https://www.ebi.ac.uk/biostudies/studies/E-MTAB-6413
  ENA study:  ERP116722   project: PRJEB33892
"""
import _bootstrap  # noqa: F401
import argparse

import _ena
from src.utils.config import get_dataset, dataset_dest
from src.utils.provenance import dataset_record, inventory_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
# ENA resolves either the study (ERP...) or project (PRJEB...) accession.
ACCESSION = "PRJEB33892"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = get_dataset("dual_rna_gambia")
    dest = dataset_dest("dual_rna_gambia")

    rows = _ena.fetch_filereport(ACCESSION)
    if not rows:
        log.warning("no runs for %s; trying study accession %s",
                    ACCESSION, ds["ena_study"])
        rows = _ena.fetch_filereport(ds["ena_study"])
    _ena.save_report(rows, dest, ACCESSION)

    n = 0
    if args.download:
        n = _ena.download_fastqs(rows, dest, max_runs=args.max_runs,
                                 overwrite=args.overwrite)
    else:
        log.info("run report only. Re-run with --download to fetch FASTQ.")

    dataset_record(
        accession="E-MTAB-6413 / ERP116722 / PRJEB33892", repository="BioStudies/ENA",
        download_url=ds["url_landing"], dest_dir=dest,
        raw_or_processed_status="raw_fastq",
        organism="Homo sapiens + Plasmodium falciparum", tissue="peripheral blood",
        sample_count=len(rows),
        files=inventory_dir(dest, compute_checksums=False),
        notes="External validation cohort. Check batch/severity/age before pooling.",
    )
    log.info("done. %d fastq files downloaded.", n)


if __name__ == "__main__":
    main()
