"""Shared ENA (European Nucleotide Archive) download helpers.

Uses the ENA Portal API file-report endpoint to resolve FASTQ FTP URLs for a
study/project accession, then downloads them.

    https://www.ebi.ac.uk/ena/portal/api/filereport
        ?accession=<ACC>&result=read_run&fields=<...>&format=tsv

Reference: ENA Documentation, Retrieving File reports,
https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access/file-reports.html
"""
from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request
from pathlib import Path

from src.utils.download import http_download
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
FIELDS = ",".join([
    "run_accession", "sample_accession", "experiment_accession",
    "study_accession", "scientific_name", "tax_id", "library_strategy",
    "library_source", "read_count", "fastq_ftp", "fastq_bytes", "fastq_md5",
    "submitted_ftp",
])


def fetch_filereport(accession: str) -> list[dict]:
    """Return the ENA read_run file report for a study/project accession."""
    params = urllib.parse.urlencode({
        "accession": accession, "result": "read_run",
        "fields": FIELDS, "format": "tsv", "limit": "0",
    })
    url = f"{FILEREPORT}?{params}"
    log.info("ENA filereport: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Malaria_Project/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode()
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    log.info("%s -> %d runs", accession, len(rows))
    return rows


def save_report(rows: list[dict], dest: Path, accession: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{accession}_read_run_report.tsv"
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    log.info("wrote run report: %s", out)
    return out


def download_fastqs(rows: list[dict], dest: Path, max_runs: int | None = None,
                    overwrite: bool = False) -> int:
    """Download FASTQ files listed in the report. ``max_runs`` caps how many."""
    n = 0
    for i, row in enumerate(rows):
        if max_runs is not None and i >= max_runs:
            break
        ftp_field = row.get("fastq_ftp", "")
        if not ftp_field:
            continue
        for ftp in ftp_field.split(";"):
            url = "https://" + ftp if not ftp.startswith("http") else ftp
            run = row.get("run_accession", "run")
            try:
                http_download(url, dest / "fastq" / run / Path(ftp).name,
                              overwrite=overwrite)
                n += 1
            except Exception as e:  # one bad run must not abort the whole set
                log.warning("skipping %s after repeated failure: %s", run, e)
                continue
    log.info("downloaded %d fastq files (failures skipped and logged)", n)
    return n
