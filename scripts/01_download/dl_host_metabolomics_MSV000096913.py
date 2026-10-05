#!/usr/bin/env python
"""Stage 1 -- Host metabolomics, MassIVE MSV000096913 (blueprint Section 12.3).

Networks that block outbound FTP/FTPS (port 21) cannot use MassIVE's FTP server
("421 TLS is required" / connection timeout). This version avoids FTP entirely:

  1. get the file list over HTTPS via ppx (its listing endpoint is HTTP-based
     and works even when FTP is blocked), and
  2. download each file over HTTPS through MassIVE's web endpoint
     ``ProteoSAFe/DownloadResultFile`` (port 443).

If ppx is not installed, the file list falls back to the MassIVE file-listing
API. Everything runs on HTTPS, so no FTP port is needed.

Reference: MassIVE dataset MSV000096913 (task ba8df9f0096c4bf6abb946f323f153c1)
"""
import _bootstrap  # noqa: F401
import argparse
import urllib.parse
from pathlib import Path

from src.utils.config import get_dataset, dataset_dest
from src.utils.download import http_download
from src.utils.provenance import dataset_record, inventory_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

ACCESSION = "MSV000096913"
TASK = "ba8df9f0096c4bf6abb946f323f153c1"
DL = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile"


def _list_via_ppx() -> list[str]:
    """Return dataset-relative file paths using ppx (HTTPS listing)."""
    from ppx import MassiveProject
    p = MassiveProject(ACCESSION)
    files = [str(f) for f in p.remote_files()]
    log.info("ppx listed %d files", len(files))
    return files


def _https_url(relpath: str) -> str:
    # MassIVE web download: file=f.<MSVID>/<relative path within dataset>
    rel = relpath.lstrip("/")
    if not rel.startswith(f"{ACCESSION}/"):
        rel = f"{ACCESSION}/{rel}"
    q = urllib.parse.urlencode({
        "task": TASK, "invoke": "downloadResultFile", "file": f"f.{rel}",
    })
    return f"{DL}?{q}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap number of files (pilot)")
    args = ap.parse_args()

    ds = get_dataset("host_metabolomics")
    dest = dataset_dest("host_metabolomics")
    log.info("MSV000096913 via HTTPS (no FTP) -> %s", dest)

    try:
        files = _list_via_ppx()
    except Exception as e:
        log.error("could not list files via ppx: %s", e)
        log.error("Install ppx (`pip install ppx`) or tell me and I will add "
                  "a pure-HTTPS listing fallback.")
        return

    n = 0
    for i, rel in enumerate(files):
        if args.max_files is not None and i >= args.max_files:
            break
        # strip a leading dataset-id folder if ppx already included it
        local_rel = rel[len(ACCESSION) + 1:] if rel.startswith(ACCESSION + "/") else rel
        url = _https_url(rel)
        try:
            http_download(url, dest / local_rel)
            n += 1
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", rel, e)

    dataset_record(
        accession=ACCESSION, repository=ds["repository"],
        download_url=ds["url_landing"], dest_dir=dest,
        license_or_access_terms="public", raw_or_processed_status="raw",
        organism="Homo sapiens", tissue="plasma/serum",
        software="ppx listing + MassIVE HTTPS DownloadResultFile",
        files=inventory_dir(dest, compute_checksums=False),
        notes="Downloaded over HTTPS (FTP blocked on this network).",
    )
    log.info("done. %d files downloaded over HTTPS.", n)


if __name__ == "__main__":
    main()
