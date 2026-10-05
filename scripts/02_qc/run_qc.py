#!/usr/bin/env python
"""Phase QC -- per-dataset quality control + dataset-audit table (Section 93).

Produces the dataset-audit table that the blueprint calls "the true starting
point": one row per dataset with sample counts, omics, raw/processed status,
clinical/treatment label availability, patient-level matching feasibility,
access restrictions, and download/QC completion flags.

This is a STUB. It fills the audit table from the dataset registry written by
the download scripts, and leaves per-omic QC (library size, outliers,
missingness, batch) as TODO hooks per modality.
"""
import _bootstrap  # noqa: F401
import csv
import json

from src.utils.config import get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

AUDIT_COLUMNS = [
    "dataset", "study", "species", "country", "age", "phenotype",
    "sample_count", "omics", "raw_data_status", "processed_data_status",
    "clinical_labels", "treatment_labels", "patient_level_matching",
    "access_restrictions", "download_complete", "qc_complete",
]


def main() -> None:
    reg_dir = get_path("dataset_registry")
    audit = get_path("reports", create=True) / "dataset_audit.tsv"

    rows = []
    for rec_file in sorted(reg_dir.glob("*.json")):
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        rows.append({
            "dataset": rec.get("accession", rec_file.stem),
            "study": "",
            "species": rec.get("organism", ""),
            "country": "",
            "age": "",
            "phenotype": "",
            "sample_count": rec.get("sample_count", ""),
            "omics": "",
            "raw_data_status": rec.get("raw_or_processed_status", ""),
            "processed_data_status": "",
            "clinical_labels": "",
            "treatment_labels": "",
            "patient_level_matching": "",
            "access_restrictions": rec.get("license_or_access_terms", ""),
            "download_complete": "yes" if rec.get("files") else "partial",
            "qc_complete": "no",
        })

    with open(audit, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    log.info("wrote dataset-audit table (%d datasets): %s", len(rows), audit)

    # TODO per-omic QC hooks (Sections 25-31):
    #   - host RNA: library size, low-expression filter, outliers, normalization
    #   - proteomics: technical vs biological replicates, missingness, batch
    #   - metabolomics: QC samples, blanks, drift, annotation confidence
    #   - parasite genome: QC pass, allele-freq/missingness, LD, PCA
    #   - dual RNA: parasite-read fraction, host/parasite separation
    log.info("STUB: add per-modality QC (blueprint Sections 25-31).")


if __name__ == "__main__":
    main()
