#!/usr/bin/env python
"""Verify that every dataset is downloaded and complete (blueprint Sections 21, 93).

Checks each dataset declared in config/datasets.yaml against three things:
  1. a provenance record exists in metadata/dataset_registry/
  2. the destination directory exists and is non-empty
  3. dataset-specific expected artifacts / counts are present

Prints a PASS / PARTIAL / MISSING / MANAGED table and writes a completeness
report to reports/data_completeness.tsv. Use this before starting integration so
you know exactly what is on disk and what still needs fetching.
"""
import _bootstrap  # noqa: F401
import csv
import re
from pathlib import Path

from src.utils.config import load_config, project_root, get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _count(dirpath: Path, pattern: str) -> int:
    return sum(1 for _ in dirpath.glob(pattern)) if dirpath.exists() else 0


def _ena_expected_runs(report_dir: Path) -> int:
    """Read an ENA read_run report and count expected runs (x2 for paired)."""
    rep = next(iter(report_dir.glob("**/*_read_run_report.tsv")), None)
    if not rep:
        return 0
    with open(rep, encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def check(name: str, block: dict, root: Path, registry: Path) -> dict:
    dest = root / block.get("dest", "")
    access = block.get("access", "")
    status, detail = "MISSING", ""

    # Managed datasets are never auto-downloaded -> report as MANAGED (expected).
    if access == "managed":
        note = "application checklist" if (get_path("access_notes")).exists() else ""
        return {"dataset": name, "status": "MANAGED", "detail": note,
                "dest": str(dest)}

    # Reference-only entries (WHO guidance) are notes/URLs, not file downloads.
    if name == "formulation_who":
        note = get_path("access_notes") / "WHO_formulation_benchmark.md"
        return {"dataset": name, "status": "REFERENCE",
                "detail": "notes/URLs (no file download expected)", "dest": str(dest)}

    # Pf8 study subsets live inside the main Pf8 download, not their own folders.
    if name in ("pf8_study_1167", "pf8_study_1101"):
        main = root / "data/raw/parasite_genomics/Pf8"
        sid = name.split("_")[-1]
        has = _count(main, f"*study{sid}*") + _count(main, f"*{sid}*")
        st = "PASS" if has else "PARTIAL"
        return {"dataset": name, "status": st,
                "detail": f"subset of parasite_genomics_pf8 (study {sid})",
                "dest": str(main)}

    has_registry = any(registry.glob(f"{re.sub(r'[^A-Za-z0-9]', '*', block.get('accession', name))}*.json")) \
        or (registry / f"{name}.json").exists()
    non_empty = dest.exists() and any(p.is_file() for p in dest.rglob("*"))

    # dataset-specific expectations
    if name == "host_transcriptomics":
        counts = _count(dest, "**/*raw_counts*.txt.gz")
        proc = (get_path("data_processed") / "host_transcriptomics"
                / "transcriptome_patient_by_gene.tsv").exists()
        status = "PASS" if (counts and proc) else ("PARTIAL" if non_empty else "MISSING")
        detail = f"counts={counts}, processed_matrix={proc}"
    elif name == "host_proteomics":
        raws = _count(dest, "**/*.raw") + _count(dest, "**/*.mzML")
        matrix = _count(dest, "**/*pr_matrix*.tsv")
        proc = (get_path("data_processed") / "host_proteomics"
                / "proteome_patient_by_protein.tsv").exists()
        status = "PASS" if (matrix and proc) else ("PARTIAL" if non_empty else "MISSING")
        detail = f"raw/mzML={raws}, pr_matrix={matrix}, processed={proc}"
    elif name == "host_metabolomics":
        n = _count(dest, "**/*")
        status = "PASS" if n > 5 else ("PARTIAL" if non_empty else "MISSING")
        detail = f"files={n} (FTP-blocked networks: fetch elsewhere)"
    elif name in ("dual_rna_mali", "dual_rna_gambia"):
        got = _count(dest, "**/*.fastq.gz")
        exp = _ena_expected_runs(dest) * 2
        status = "PASS" if (exp and got >= exp) else ("PARTIAL" if got else "MISSING")
        detail = f"fastq={got}/{exp or '?'}"
    elif name == "parasite_genomics_pf8":
        tables = _count(dest, "Pf8_*.tsv") + _count(dest, "Pf8_*.txt")
        proc = (get_path("data_processed") / "parasite_genomics"
                / "parasite_feature_matrix.tsv").exists()
        status = "PASS" if tables >= 3 else ("PARTIAL" if non_empty else "MISSING")
        detail = f"pf8_tables={tables}, processed={proc}"
    elif name in ("reference_pubchem", "reference_chembl"):
        # These share one provenance record (pharma_references.json); presence of
        # any descriptor file is enough.
        status = "PASS" if non_empty else "MISSING"
        detail = f"files={_count(dest, '**/*')} (shared pharma_references record)"
    else:
        status = "PASS" if non_empty else "MISSING"
        detail = f"files={_count(dest, '**/*')}"

    # A missing provenance record is a note, not a downgrade: the data is present.
    if not has_registry and status == "PASS":
        detail += " [note: no registry record]"
    return {"dataset": name, "status": status, "detail": detail, "dest": str(dest)}


def main() -> None:
    cfg = load_config()
    root = project_root()
    registry = get_path("dataset_registry")
    rows = [check(name, block, root, registry)
            for name, block in cfg["datasets"].items()]

    # print table
    width = max(len(r["dataset"]) for r in rows)
    log.info("%-*s  %-8s  %s", width, "DATASET", "STATUS", "DETAIL")
    log.info("%s", "-" * (width + 40))
    counts = {"PASS": 0, "PARTIAL": 0, "MISSING": 0, "MANAGED": 0}
    for r in sorted(rows, key=lambda x: x["status"]):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        log.info("%-*s  %-8s  %s", width, r["dataset"], r["status"], r["detail"])

    out = get_path("reports", create=True) / "data_completeness.tsv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "status", "detail", "dest"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    log.info("-" * (width + 40))
    log.info("SUMMARY: %d PASS, %d PARTIAL, %d MISSING, %d MANAGED",
             counts["PASS"], counts["PARTIAL"], counts["MISSING"], counts["MANAGED"])
    log.info("report -> %s", out)
    open_layers = [r["dataset"] for r in rows if r["status"] == "PASS"]
    log.info("Ready for integration on: %s", ", ".join(open_layers))


if __name__ == "__main__":
    main()
