#!/usr/bin/env python
"""ONE COMMAND to acquire everything (blueprint Section 21 acquisition order).

Runs every downloader in dependency order. Open data are fetched; managed data
(IDDO/WWARN clinical-PK, MalariaGEN full human genotypes) cannot be downloaded
automatically, so their application checklists are generated instead.

Usage
-----
    python scripts/01_download/download_all.py                # metadata/lists + open refs (safe, fast)
    python scripts/01_download/download_all.py --heavy        # also pull large FASTQ + proteomics/metabolomics
    python scripts/01_download/download_all.py --stage 1 3    # only stages 1 and 3
    python scripts/01_download/download_all.py --dry-run      # print the plan only

By default the two big sequence stages run in "list/report only" mode so you can
inspect sizes before committing terabytes. Add --heavy to actually pull them.
Everything is idempotent: existing files are skipped (raw data are immutable).
"""
import _bootstrap  # noqa: F401
import argparse
import runpy
import sys
from pathlib import Path

from src.utils.logging_utils import get_logger

log = get_logger(__name__)
HERE = Path(__file__).resolve().parent

# (stage, script, argv-when-light, argv-when-heavy, note)
PLAN = [
    (1, "dl_host_transcriptomics_GSE289197.py", [], [],
     "Host transcriptomics GEO GSE289197 (processed/count-level; FASTQ not open)"),
    (1, "dl_host_proteomics_PXD058621.py", ["--list-only"], [],
     "Host proteomics PRIDE PXD058621 (raw MS). --heavy pulls all files."),
    (1, "dl_host_metabolomics_MSV000096913.py", [], [],
     "Host metabolomics MassIVE MSV000096913 (raw)."),
    (2, "dl_dual_rna_PRJNA962942.py", [], ["--download"],
     "Mali dual RNA PRJNA962942. Light = ENA run report only."),
    (2, "dl_dual_rna_gambia_ERP116722.py", [], ["--download"],
     "Gambia dual RNA PRJEB33892. Light = ENA run report only."),
    (3, "dl_parasite_genomics_Pf8.py", ["--metadata-only"], ["--metadata-only"],
     "MalariaGEN Pf8 metadata + resistance (raw CRAM never auto-pulled)."),
    (3, "dl_parasite_genomics_Pf8.py", ["--metadata-only", "--study", "1167"],
     ["--metadata-only", "--study", "1167"],
     "Pf8 Study 1167 pediatric treatment cohort."),
    (4, "notes_malariagen_managed_human.py", ["--download-open"], ["--download-open"],
     "MalariaGEN matched host-parasite: OPEN Zenodo records + managed note."),
    (5, "prepare_iddo_wwarn_request.py", [], [],
     "IDDO/WWARN clinical-PK: MANAGED. Generates application checklist."),
    (6, "dl_formulation_FDA_IID.py", [], [],
     "FDA Inactive Ingredient Database (excipients)."),
    (6, "dl_reference_pubchem_chembl_who.py", [], [],
     "PubChem + ChEMBL antimalarial descriptors + WHO benchmark notes."),
]


def _run(script: str, argv: list[str]) -> bool:
    path = HERE / script
    log.info(">>> %s %s", script, " ".join(argv))
    old = sys.argv
    try:
        sys.argv = [str(path)] + argv
        runpy.run_path(str(path), run_name="__main__")
        return True
    except SystemExit:
        return True
    except Exception as e:
        log.error("FAILED %s: %s", script, e)
        return False
    finally:
        sys.argv = old


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heavy", action="store_true",
                    help="pull large FASTQ + all proteomics files")
    ap.add_argument("--stage", nargs="*", type=int, default=None,
                    help="run only these stage numbers (1-6)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = set(args.stage) if args.stage else None
    ok = fail = 0
    for stage, script, light, heavy, note in PLAN:
        if stages and stage not in stages:
            continue
        argv = heavy if (args.heavy and heavy) else light
        log.info("STAGE %d | %s", stage, note)
        if args.dry_run:
            log.info("    (dry-run) would call: %s %s", script, " ".join(argv))
            continue
        if _run(script, argv):
            ok += 1
        else:
            fail += 1

    log.info("=" * 60)
    log.info("download_all complete: %d ok, %d failed.", ok, fail)
    log.info("Managed data still require applications: IDDO/WWARN clinical-PK and "
             "MalariaGEN full human genotypes (see metadata/access_notes/).")
    if not args.heavy:
        log.info("Re-run with --heavy to pull large FASTQ + full proteomics.")


if __name__ == "__main__":
    main()
