#!/usr/bin/env python
"""Stage 6 -- FDA Inactive Ingredient Database (blueprint Section 20.1).

FDA = US Food and Drug Administration. The IID lists inactive ingredients
(excipients) with precedent in approved dosage forms. Used for pediatric
formulation design (Section 63). Presence in the IID is NOT proof of safety
for every pediatric age, route, concentration, or exposure (Section 20.1).

The FDA publishes the IID as a downloadable file on:
    https://www.fda.gov/drugs/drug-approvals-and-databases/inactive-ingredients-database-download
The exact ZIP filename changes with each release, so this script fetches the
landing page, finds the current download link, and retrieves it.
"""
import _bootstrap  # noqa: F401
import argparse
import re
import urllib.request

from src.utils.config import get_dataset, dataset_dest
from src.utils.download import http_download
from src.utils.provenance import dataset_record, inventory_dir, write_checksum
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
LANDING = ("https://www.fda.gov/drugs/drug-approvals-and-databases/"
           "inactive-ingredients-database-download")


def _find_download_links(html: str) -> list[str]:
    # FDA hosts the file under /media/.../download; capture zip/xlsx/txt links.
    links = re.findall(r'href="([^"]+\.(?:zip|xlsx|txt|csv))"', html, re.I)
    links += re.findall(r'href="(/media/[^"]+/download)"', html, re.I)
    out = []
    for l in links:
        out.append(l if l.startswith("http") else "https://www.fda.gov" + l)
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=None,
                    help="direct download URL if the auto-detector misses it")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = get_dataset("formulation_fda_iid")
    dest = dataset_dest("formulation_fda_iid")

    urls = [args.url] if args.url else []
    if not urls:
        req = urllib.request.Request(LANDING, headers={"User-Agent": "Malaria_Project/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            html = resp.read().decode(errors="ignore")
        urls = _find_download_links(html)
        log.info("candidate download links: %s", urls or "none found")

    n = 0
    for url in urls:
        try:
            p = http_download(url, dest / (url.split("/")[-2] + ".download"
                              if url.endswith("/download") else url.split("/")[-1]),
                              overwrite=args.overwrite)
            write_checksum(p)
            n += 1
        except Exception as e:
            log.warning("failed %s: %s", url, e)

    if n == 0:
        log.warning("No file auto-downloaded. Open the landing page and pass the "
                    "direct link with --url:\n  %s", LANDING)

    dataset_record(
        accession="FDA_IID", repository="FDA", download_url=LANDING, dest_dir=dest,
        license_or_access_terms="open (US government work)",
        raw_or_processed_status="reference_database",
        files=inventory_dir(dest, compute_checksums=False),
        notes="Excipient precedent only; not a pediatric safety guarantee.",
    )
    log.info("done. %d file(s) retrieved.", n)


if __name__ == "__main__":
    main()
