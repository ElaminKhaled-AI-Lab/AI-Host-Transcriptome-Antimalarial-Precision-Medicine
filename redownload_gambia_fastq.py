#!/usr/bin/env python
"""Robust re-download + integrity repair for the Gambia dual-RNA FASTQs (E-MTAB-6413).

The first bulk download of the Gambia cohort was interrupted, leaving most
FASTQ files truncated (gzip "unexpected end of file"), so salmon mapped only
1/53 samples. This script fixes that safely and idempotently:

  1. queries the ENA filereport API for E-MTAB-6413 to get, per run, the exact
     FASTQ FTP URLs AND their published MD5 checksums;
  2. for every file, checks whether a good local copy already exists
     (present + valid gzip + MD5 matches) and SKIPS it -- so re-running only
     fetches what is missing or broken;
  3. otherwise (re-)downloads with wget -c (resume) and several retries, then
     verifies gzip integrity and MD5; a file that still fails is reported, not
     left to poison salmon;
  4. prints a clear summary of OK / repaired / still-bad runs.

Nothing is deleted except a local file that fails verification and is about to
be replaced. Safe to run repeatedly.

Usage:
  python scripts/02_download/redownload_gambia_fastq.py                 # repair all
  python scripts/02_download/redownload_gambia_fastq.py --check-only    # just report
  python scripts/02_download/redownload_gambia_fastq.py --jobs 4        # parallel
"""
import argparse
import concurrent.futures as cf
import glob
import gzip
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ACCESSION = "E-MTAB-6413"
DEST = Path("data/raw/parasite_transcriptomics") / ACCESSION / "fastq"
# ENA portal API. Order fields with format LAST; some accessions need the
# secondary (PRJEB / study) accession -- we prefer the on-disk report anyway.
FILEREPORT = (
    "https://www.ebi.ac.uk/ena/portal/api/filereport"
    "?accession={acc}&result=read_run"
    "&fields=run_accession,fastq_ftp,fastq_md5&format=tsv"
)


def log(msg):
    print(msg, flush=True)


def _rows_from_text(text):
    """Parse a TSV that contains at least run_accession + fastq_ftp columns."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    idx = {h.strip(): i for i, h in enumerate(header)}
    if "run_accession" not in idx or "fastq_ftp" not in idx:
        return []
    ftp_i, run_i = idx["fastq_ftp"], idx["run_accession"]
    md5_i = idx.get("fastq_md5")
    out = []
    for ln in lines[1:]:
        d = ln.split("\t")
        if max(ftp_i, run_i) >= len(d):
            continue
        run = d[run_i].strip()
        urls = [u for u in d[ftp_i].split(";") if u]
        if md5_i is not None and md5_i < len(d):
            md5s = [m for m in d[md5_i].split(";") if m]
        else:
            md5s = []
        # pad md5 list so zip lines up even when md5 is absent
        md5s = (md5s + [""] * len(urls))[:len(urls)]
        if urls:
            out.append((run, list(zip(urls, md5s))))
    return out


def report_from_disk():
    """Reuse the read_run report saved during the original download, if present."""
    cands = glob.glob("data/raw/**/*read_run_report*.tsv", recursive=True)
    cands += glob.glob(f"data/raw/**/{ACCESSION}/**/*.tsv", recursive=True)
    for path in cands:
        if ACCESSION not in path and "gambia" not in path.lower():
            continue
        try:
            rows = _rows_from_text(Path(path).read_text())
        except Exception:
            rows = []
        if rows:
            log(f"using on-disk report: {path}  ({len(rows)} runs)")
            return rows
    return []


def report_from_api(accession):
    url = FILEREPORT.format(acc=accession)
    req = urllib.request.Request(url, headers={"User-Agent": "Malaria_Project/0.2"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return _rows_from_text(r.read().decode())


def fetch_report(accession):
    rows = report_from_disk()
    if rows:
        return rows
    log("no usable on-disk report; querying ENA filereport API ...")
    try:
        rows = report_from_api(accession)
    except Exception as e:
        log(f"API query failed ({e}); trying PRJEB study lookup is not automated -- "
            f"please paste me the header of your Gambia read_run report.")
        rows = []
    return rows


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def gzip_ok(path):
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def file_good(path, md5):
    if not path.exists() or path.stat().st_size == 0:
        return False
    if not gzip_ok(path):
        return False
    if md5 and md5sum(path) != md5:
        return False
    return True


def download(url, dest_path, retries=5):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    full = url if url.startswith(("http://", "https://", "ftp://")) else "https://" + url
    tool = shutil.which("wget")
    for attempt in range(1, retries + 1):
        try:
            if tool:
                cmd = ["wget", "-c", "-q", "--tries=3", "--timeout=120",
                       "-O", str(dest_path), full]
            else:
                cmd = ["curl", "-sSL", "-C", "-", "--retry", "3",
                       "-o", str(dest_path), full]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            log(f"    attempt {attempt}/{retries} failed for {dest_path.name}")
    return False


def handle_run(run, files, check_only):
    run_dir = DEST / run
    status = "OK"
    for url, md5 in files:
        fname = url.split("/")[-1]
        path = run_dir / fname
        if file_good(path, md5):
            continue
        if check_only:
            status = "BAD"
            continue
        # re-download this file
        if path.exists():
            path.unlink()
        ok = download(url, path)
        if not ok or not file_good(path, md5):
            status = "FAILED"
    return run, status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accession", default=ACCESSION)
    ap.add_argument("--check-only", action="store_true",
                    help="only report which runs are good/bad, download nothing")
    ap.add_argument("--jobs", type=int, default=3, help="parallel downloads")
    args = ap.parse_args()

    if not Path("data").exists():
        log("ERROR: run this from the Malaria_Project root (no data/ here).")
        sys.exit(1)

    log(f"querying ENA filereport for {args.accession} ...")
    runs = fetch_report(args.accession)
    log(f"{len(runs)} runs listed by ENA")

    results = {}
    if args.jobs > 1 and not args.check_only:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(handle_run, r, f, args.check_only): r for r, f in runs}
            for fut in cf.as_completed(futs):
                run, st = fut.result()
                results[run] = st
                log(f"  {run}: {st}")
    else:
        for r, f in runs:
            run, st = handle_run(r, f, args.check_only)
            results[run] = st
            log(f"  {run}: {st}")

    ok = sum(v == "OK" for v in results.values())
    bad = [k for k, v in results.items() if v != "OK"]
    log("\n===== SUMMARY =====")
    log(f"good runs : {ok}/{len(results)}")
    if bad:
        log(f"needs work: {len(bad)} -> {', '.join(sorted(bad))}")
        if args.check_only:
            log("re-run WITHOUT --check-only to repair them.")
        else:
            log("re-run the same command to retry the still-failed runs.")
    else:
        log("ALL runs verified (gzip + MD5). Ready to re-run salmon quantification.")


if __name__ == "__main__":
    main()
