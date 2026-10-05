#!/usr/bin/env python
"""Stage 1 -- Host transcriptomics, GEO GSE289197 (blueprint Section 12.1).

IMPORTANT (blueprint Section 12.1): the submitters did NOT deposit raw patient
FASTQ reads openly (identifiable human sequence). Only sample metadata and
processed / count-level expression are open. This script therefore downloads:

  * the GEO Series matrix file  (sample metadata + processed values)
  * the supplementary files      (submitter-processed count matrices)

It fetches from the NCBI GEO FTP mirror, which mirrors each series under
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289197/
Reference: NCBI GEO documentation, https://www.ncbi.nlm.nih.gov/geo/

Do NOT claim complete raw-read reprocessing from this dataset.
"""
import _bootstrap  # noqa: F401
import argparse

from src.utils.config import get_dataset, dataset_dest
from src.utils.download import http_download, check_url
from src.utils.provenance import dataset_record, inventory_dir, write_checksum
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289197"
# Series matrix (metadata + processed) and supplementary directory listing:
FILES = {
    "matrix/GSE289197_series_matrix.txt.gz":
        f"{GEO_FTP}/matrix/GSE289197_series_matrix.txt.gz",
    "soft/GSE289197_family.soft.gz":
        f"{GEO_FTP}/soft/GSE289197_family.soft.gz",
}
SUPPL_INDEX = f"{GEO_FTP}/suppl/"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-suppl", action="store_true",
                    help="skip the supplementary directory (processed matrices)")
    args = ap.parse_args()

    ds = get_dataset("host_transcriptomics")
    dest = dataset_dest("host_transcriptomics")
    log.info("GSE289197 -> %s", dest)

    got = []
    for rel, url in FILES.items():
        if check_url(url):
            p = http_download(url, dest / rel, overwrite=args.overwrite)
            write_checksum(p)
            got.append(str(p))
        else:
            log.warning("not reachable (may not exist for this series): %s", url)

    if not args.skip_suppl:
        log.info("Supplementary files listed at: %s", SUPPL_INDEX)
        log.info("Open that URL to see submitter-processed matrices, then add "
                 "their exact names to FILES and re-run, or use GEOquery in R.")

    dataset_record(
        accession=ds["accession"], repository=ds["repository"],
        download_url=ds["url_landing"], dest_dir=dest,
        license_or_access_terms="open processed/count-level; raw FASTQ NOT open",
        publication="https://www.nature.com/articles/s41467-025-59281-5",
        raw_or_processed_status="processed_only",
        organism="Homo sapiens", tissue="whole blood",
        files=inventory_dir(dest, compute_checksums=False),
        notes="Patient FASTQ withheld (identifiable). Use count-level data only.",
    )
    log.info("done. %d files retrieved.", len(got))


if __name__ == "__main__":
    main()
