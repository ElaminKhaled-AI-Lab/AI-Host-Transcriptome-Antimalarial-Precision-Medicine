#!/usr/bin/env python
"""Phase 0 -- initialise reproducibility scaffolding (blueprint Section 94).

Creates all output directories, initialises a git repository for code, drops
.gitkeep files, and writes an empty dataset-audit table (Section 93) ready to
be filled by the QC step.
"""
import _bootstrap  # noqa: F401
import subprocess
import csv
from pathlib import Path

from src.utils.config import load_config, get_path, project_root
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

AUDIT_COLUMNS = [
    "dataset", "study", "species", "country", "age", "phenotype",
    "sample_count", "omics", "raw_data_status", "processed_data_status",
    "clinical_labels", "treatment_labels", "patient_level_matching",
    "access_restrictions", "download_complete", "qc_complete",
]


def main() -> None:
    cfg = load_config()
    root = project_root()
    log.info("project root: %s", root)

    # Ensure every declared path exists + a .gitkeep so empty dirs persist.
    for key in cfg["paths"]:
        p = get_path(key, create=True)
        (p / ".gitkeep").touch()
        log.info("ensured %s", p)

    # git init for code (data are gitignored).
    if not (root / ".git").exists():
        try:
            subprocess.run(["git", "init"], cwd=root, check=True)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
            log.info("git repository initialised (remember: raw data are gitignored)")
        except Exception as e:
            log.warning("git init skipped: %s", e)

    # Empty dataset-audit table (Section 93 / first computational milestone).
    audit = get_path("reports", create=True) / "dataset_audit.tsv"
    if not audit.exists():
        with open(audit, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh, delimiter="\t").writerow(AUDIT_COLUMNS)
        log.info("wrote empty dataset-audit table: %s", audit)

    log.info("Phase 0 complete. Next: python scripts/00_setup/01_check_environment.py")


if __name__ == "__main__":
    main()
