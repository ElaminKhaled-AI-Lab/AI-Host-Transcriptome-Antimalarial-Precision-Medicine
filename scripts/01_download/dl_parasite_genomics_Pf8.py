#!/usr/bin/env python
"""Stage 3 -- Parasite genomics, MalariaGEN Pf8 (blueprint Sections 15-17, 28).

Downloads ANALYSIS-READY features first (blueprint Section 15: "Do not
initially download all raw sequence data"):
  1. sample metadata  (Sample, Study, Country, Year, ENA accession, QC pass)
  2. drug-resistance marker genotypes + inferred resistance classifications
  3. copy-number variation / complexity-of-infection where exposed by the API

Uses the official ``malariagen_data`` Python package (accesses Pf8 over Google
Cloud Storage). Raw CRAM alignments are fetched ONLY when explicitly required.

References:
  Pf8 data access: https://malariagen.github.io/parasite-data/pf8/Data_access.html
  malariagen_data: https://pypi.org/project/malariagen_data/
  Pf8 resource:    https://www.malariagen.net/resource/36/
"""
import _bootstrap  # noqa: F401
import argparse
import json
import urllib.request

from src.utils.config import get_dataset, dataset_dest
from src.utils.download import http_download
from src.utils.provenance import dataset_record, write_checksum
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
RESISTANCE_LOCI = ["crt", "dhfr", "dhps", "mdr1", "kelch13", "plasmepsin2_3"]

# Pf8 metadata + resistance mirror on Zenodo (direct HTTPS, no GCS auth).
PF8_ZENODO_RECORD = "18681980"
# Only pull the small analysis-ready tables here, not the multi-GB callset.
_WANT = ("metadata", "resistance", "sample", "qc", "fws", "moi",
         "complexity", "cnv", "copy_number")


def _download_pf8_zenodo(dest) -> int:
    """Fetch Pf8 metadata + resistance tables from Zenodo (no credentials)."""
    api = f"https://zenodo.org/api/records/{PF8_ZENODO_RECORD}"
    try:
        req = urllib.request.Request(api, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            rec = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        log.warning("Zenodo record fetch failed: %s", e)
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in rec.get("files", []):
        name = (f.get("key") or "").lower()
        # Skip the huge variant callset; keep small tabular tables.
        if name.endswith((".zarr", ".zarr.zip", ".vcf", ".vcf.gz", ".bcf", ".cram")):
            continue
        if not (name.endswith((".tsv", ".csv", ".txt", ".xlsx", ".tsv.gz", ".csv.gz"))
                and any(w in name for w in _WANT)):
            continue
        url = f.get("links", {}).get("self")
        if not url:
            continue
        try:
            p = http_download(url, dest / f.get("key"))
            write_checksum(p)
            n += 1
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: %s", name, e)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-only", action="store_true", default=True,
                    help="(default) fetch metadata + resistance, not raw CRAM")
    ap.add_argument("--study", default=None,
                    help="filter to a Pf8 study id, e.g. 1167 or 1101")
    ap.add_argument("--use-gcs", action="store_true",
                    help="use the malariagen_data API (needs Google Cloud auth) "
                         "instead of the default Zenodo direct download")
    args = ap.parse_args()

    ds = get_dataset("parasite_genomics_pf8")
    dest = dataset_dest("parasite_genomics_pf8")

    # PREFERRED PATH: Zenodo direct HTTPS (no Google Cloud credentials).
    # The malariagen_data API serves the callset from Google Cloud Storage and
    # requires GCS authentication (that is the DefaultCredentialsError). But the
    # Pf8 metadata + drug-resistance tables are mirrored on Zenodo as plain
    # files, which is all we need for the metadata-only stage.
    if not args.use_gcs:
        n = _download_pf8_zenodo(dest)
        if n:
            dataset_record(
                accession="Pf8", repository="MalariaGEN/Zenodo",
                download_url="https://zenodo.org/records/18681980", dest_dir=dest,
                license_or_access_terms="open", raw_or_processed_status="metadata+resistance",
                organism="Plasmodium falciparum", software="zenodo-direct",
                notes="Metadata + resistance from Zenodo (no GCS auth). "
                      "Use --use-gcs for the malariagen_data API (needs Google auth).",
            )
            log.info("done via Zenodo (%d files). Filter by study/country in pandas.", n)
            log.info("For the pediatric treatment cohort, subset rows where "
                     "study == 1167 (and verify Country == Tanzania).")
            return
        log.warning("Zenodo path returned nothing; falling back to malariagen_data API.")

    try:
        import malariagen_data
    except Exception:
        log.error("malariagen_data not installed. Install with:")
        log.error("  pip install malariagen_data")
        log.error("Then re-run. Docs: %s", ds["url_docs"])
        return

    log.info("instantiating Pf8 via malariagen_data (Google Cloud Storage)...")
    log.info("NOTE: this path needs Google Cloud auth: `gcloud auth application-default login`")
    pf8 = malariagen_data.Pf8()

    # 1. Sample metadata -----------------------------------------------------
    meta = pf8.sample_metadata()
    if args.study is not None:
        # Study column name may be 'Study' or 'study'; handle both.
        col = "Study" if "Study" in meta.columns else "study"
        meta = meta[meta[col].astype(str) == str(args.study)]
        log.info("filtered to study %s: %d samples", args.study, len(meta))
    meta_path = dest / ("sample_metadata.tsv" if args.study is None
                        else f"sample_metadata_study{args.study}.tsv")
    dest.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_path, sep="\t", index=False)
    write_checksum(meta_path)
    log.info("sample metadata -> %s (%d rows, %d cols)",
             meta_path, len(meta), meta.shape[1])

    # 2. Resistance markers --------------------------------------------------
    # The Pf8 release ships a drug-resistance table. Method names vary by
    # package version; try the common accessors and fall back with guidance.
    got_resistance = False
    for attr in ("drug_resistance", "sample_drug_resistance",
                 "resistance_classifications"):
        fn = getattr(pf8, attr, None)
        if callable(fn):
            try:
                res = fn()
                rp = dest / "drug_resistance.tsv"
                res.to_csv(rp, sep="\t", index=False)
                write_checksum(rp)
                log.info("resistance table via pf8.%s() -> %s", attr, rp)
                got_resistance = True
                break
            except Exception as e:
                log.info("pf8.%s() failed: %s", attr, e)
    if not got_resistance:
        log.warning("No direct resistance accessor in this malariagen_data "
                    "version. The Pf8 drug-resistance marker + inferred-status "
                    "table is distributed with the release; download it from "
                    "the Pf8 documentation: %s", ds["url_docs"])
        (dest / "RESISTANCE_TODO.txt").write_text(
            "Fetch Pf8 drug-resistance marker genotypes + inferred resistance "
            "classifications from: " + ds["url_docs"] + "\n"
            "Loci of interest: " + ", ".join(RESISTANCE_LOCI) + "\n",
            encoding="utf-8",
        )

    dataset_record(
        accession="Pf8" + (f"/study_{args.study}" if args.study else ""),
        repository="MalariaGEN", download_url=ds["url_landing"], dest_dir=dest,
        license_or_access_terms="open", raw_or_processed_status="analysis_ready_metadata",
        organism="Plasmodium falciparum", sample_count=int(len(meta)),
        software="malariagen_data",
        notes="Metadata + resistance only. Raw CRAM downloaded separately if needed.",
    )
    log.info("done. Raw CRAM NOT downloaded (by design). "
             "For a treatment cohort run with --study 1167.")


if __name__ == "__main__":
    main()
