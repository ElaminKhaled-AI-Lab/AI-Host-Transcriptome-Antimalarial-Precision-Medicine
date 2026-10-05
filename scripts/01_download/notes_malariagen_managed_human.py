#!/usr/bin/env python
"""Stage 4 -- MalariaGEN matched host-parasite genetics notes (Section 18).

The matched host-parasite interaction resource covers >4,000 children with
severe malaria (Banjul, The Gambia; Kilifi County, Kenya). Access is MIXED:

  OPEN (downloadable via DOI):
    * parasite genome sample-accession list  -> 10.5281/zenodo.5723515
    * joint host-parasite genotype dataset   -> 10.5281/zenodo.4973476
    * association summary statistics          -> 10.5281/zenodo.5722497

  MANAGED (application required):
    * full human genotypes                    -> malariagen.net/resource/25/

This helper downloads the OPEN Zenodo records (via their API) and writes an
access note for the managed human-genotype resource.

Reference: https://www.malariagen.net/data_package/host-parasite-interaction/
"""
import _bootstrap  # noqa: F401
import argparse
import json
import re
import urllib.request

from src.utils.config import get_dataset, dataset_dest, get_path
from src.utils.download import http_download
from src.utils.provenance import dataset_record, inventory_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _zenodo_id(doi_url: str) -> str | None:
    m = re.search(r"zenodo\.(\d+)", doi_url)
    return m.group(1) if m else None


def _download_zenodo(record_id: str, dest, overwrite: bool) -> int:
    api = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(api, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rec = json.loads(resp.read().decode())
    n = 0
    for f in rec.get("files", []):
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        name = f.get("key") or f.get("filename")
        if url and name:
            http_download(url, dest / record_id / name, overwrite=overwrite)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download-open", action="store_true",
                    help="download the OPEN Zenodo records")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = get_dataset("host_parasite_genetics")
    dest = dataset_dest("host_parasite_genetics")

    open_dois = {
        "parasite_accession_list": ds["parasite_accession_list"],
        "joint_genotype_dataset": ds["joint_genotype_dataset"],
        "association_summary_stats": ds["association_summary_stats"],
    }

    total = 0
    if args.download_open:
        for label, doi in open_dois.items():
            rid = _zenodo_id(doi)
            if rid:
                log.info("downloading OPEN record %s (%s)", label, rid)
                total += _download_zenodo(rid, dest, args.overwrite)
            else:
                log.warning("could not parse Zenodo id from %s", doi)
    else:
        log.info("re-run with --download-open to fetch the OPEN Zenodo records.")

    # Managed human-genotype access note.
    note = get_path("access_notes", create=True) / "malariagen_human_genotypes_MANAGED.md"
    note.write_text(
        "# MalariaGEN full human genotypes -- MANAGED access\n\n"
        f"Resource: {ds['human_genetics_managed']}\n"
        f"Package landing: {ds['url_landing']}\n\n"
        "Full human genotypes require an application. The parasite genome, the\n"
        "joint host-parasite genotype dataset, and association summary statistics\n"
        "are OPEN (downloaded via --download-open). Apply for full human genotypes\n"
        "only if scientifically justified (host pharmacogenetics / G6PD / Hb variants).\n",
        encoding="utf-8",
    )
    log.info("wrote managed-access note: %s", note)

    dataset_record(
        accession="MalariaGEN_host_parasite", repository="MalariaGEN/Zenodo",
        download_url=ds["url_landing"], dest_dir=dest,
        license_or_access_terms="mixed: parasite/joint/summary OPEN; human genotypes MANAGED",
        organism="Homo sapiens + Plasmodium falciparum",
        files=inventory_dir(dest, compute_checksums=False),
        notes="Open Zenodo DOIs downloaded; human genotypes require application.",
    )
    log.info("done. %d open files downloaded.", total)


if __name__ == "__main__":
    main()
