#!/usr/bin/env python
"""Stage 1 -- Host proteomics, PRIDE PXD058621 (blueprint Section 12.2).

PRIDE = PRoteomics IDEntifications Database. Raw mass-spectrometry files were
submitted openly. Two tracks (Section 26):
  Track A: preserve original raw instrument files + open spectral formats.
  Track B: build the patient x protein integration matrix.

This script lists and downloads the project files via the PRIDE REST API
(https://www.ebi.ac.uk/pride/ws/archive/v2/). If the ``pridepy`` package is
installed it is used (recommended); otherwise it falls back to the REST file
listing + direct download.

References:
  PRIDE Archive:        https://www.ebi.ac.uk/pride/archive/projects/PXD058621
  pridepy:              https://github.com/PRIDE-Archive/pridepy
  ppx (alternative):    https://ppx.readthedocs.io/
"""
import _bootstrap  # noqa: F401
import argparse
import json
import urllib.request

from src.utils.config import get_dataset, dataset_dest
from src.utils.download import http_download
from src.utils.provenance import dataset_record, inventory_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

ACCESSION = "PXD058621"
# PRIDE v3 API (the v2 byProject endpoint was retired and returned non-JSON,
# which caused the "Expecting value: line 1 column 1" failure). v3 paginates.
API_V3 = f"https://www.ebi.ac.uk/pride/ws/archive/v3/projects/{ACCESSION}/files"


def _rest_file_list() -> list[dict]:
    """List all files via the PRIDE v3 API, following pagination."""
    out: list[dict] = []
    page, size = 0, 100
    while True:
        url = f"{API_V3}?pageSize={size}&page={page}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001
            log.warning("v3 API page %d failed: %s", page, e)
            break
        # v3 may return a bare list or an object with an embedded list.
        batch = data if isinstance(data, list) else (
            data.get("_embedded", {}).get("files")
            or data.get("content") or data.get("list") or [])
        if not batch:
            break
        out.extend(batch)
        if len(batch) < size:
            break
        page += 1
    return out


def _url_from_entry(entry: dict) -> str | None:
    """Prefer an HTTP(S) location; fall back to FTP only if that's all there is.

    Preferring HTTP matters on networks where outbound FTP is blocked/slow
    (the same reason MassIVE FTP timed out).
    """
    locs = entry.get("publicFileLocations", []) or entry.get("fileLocations", [])
    http = ftp = None
    for loc in locs:
        val = loc.get("value", "") if isinstance(loc, dict) else str(loc)
        if val.startswith("http"):
            http = http or val
        elif val.startswith("ftp"):
            ftp = ftp or val
    if http:
        return http
    if ftp:
        # Convert the EBI FTP path to its HTTPS mirror where possible.
        return ftp.replace("ftp://ftp.pride.ebi.ac.uk/",
                           "https://ftp.pride.ebi.ac.uk/").replace("ftp://", "https://")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-only", action="store_true",
                    help="only print/download the file manifest, not the data")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="filter by fileCategory value, e.g. RAW SEARCH RESULT")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = get_dataset("host_proteomics")
    dest = dataset_dest("host_proteomics")

    # Prefer pridepy if available.
    try:
        import pridepy  # noqa: F401
        log.info("pridepy is installed -- you may also run:")
        log.info("  pridepy download-all-public-files -a %s -o %s", ACCESSION, dest)
    except Exception:
        log.info("pridepy not installed; using PRIDE REST API directly.")

    files = _rest_file_list()
    log.info("PXD058621 lists %d files", len(files))
    manifest = dest / "PXD058621_file_manifest.json"
    dest.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(files, indent=2), encoding="utf-8")
    log.info("wrote manifest: %s", manifest)

    if args.list_only:
        return

    n = 0
    for entry in files:
        fc = entry.get("fileCategory", "")
        cat = fc.get("value", "") if isinstance(fc, dict) else str(fc or "")
        if args.categories and cat not in args.categories:
            continue
        url = _url_from_entry(entry)
        name = entry.get("fileName") or entry.get("name")
        if not url or not name:
            continue
        try:
            http_download(url, dest / (cat or "files") / name, overwrite=args.overwrite)
            n += 1
        except Exception as e:  # noqa: BLE001 -- keep going on a single bad file
            log.warning("skipping %s: %s", name, e)

    dataset_record(
        accession=ACCESSION, repository=ds["repository"],
        download_url=ds["url_landing"], dest_dir=dest,
        license_or_access_terms="open", raw_or_processed_status="raw+processed",
        organism="Homo sapiens", software="PRIDE REST v2 / pridepy",
        files=inventory_dir(dest, compute_checksums=False),
        notes="Track A raw preserved; Track B build patient x protein matrix.",
    )
    log.info("done. %d files downloaded.", n)


if __name__ == "__main__":
    main()
