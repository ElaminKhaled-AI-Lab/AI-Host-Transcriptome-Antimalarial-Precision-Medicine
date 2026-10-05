#!/usr/bin/env python
"""Phase 4 -- extract transcriptome sample + clinical table from GEO soft (Section 24/31).

GSE289197's soft file embeds per-sample clinical metadata (match group SMA/CM,
age, sex, hemoglobin, parasitemia, Blantyre coma scale, case/control, ...). This
parses it into a tidy table keyed by the transcriptome sample id so it can (a)
serve as clinical labels for the host cohort and (b) anchor the cross-omic
sample map.

Join key: the counts matrix names samples like 'IGS-MLI-046rCap'; the soft file
records the same as 'Library name: IGS_MLI_046rCAP'. Both are normalised to an
upper-case, dash-delimited key so they line up.

Outputs -> metadata/sample_maps/
  transcriptome_clinical.tsv   (one row per transcriptome sample + clinical vars)
"""
import _bootstrap  # noqa: F401
import gzip
import re
from pathlib import Path

import pandas as pd

from src.utils.config import dataset_dest, get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _norm(s: str) -> str:
    """Normalise a library/sample id to a common join key."""
    return re.sub(r"[-_ ]+", "-", str(s).strip()).upper()


def parse_soft(path: Path) -> pd.DataFrame:
    rows, cur = [], {}
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE"):
                if cur:
                    rows.append(cur)
                cur = {}
            elif line.startswith("!Sample_geo_accession"):
                cur["gsm"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_title"):
                cur["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_description") and "Library name" in line:
                lib = line.split("Library name:", 1)[1].strip()
                cur["library"] = lib
                cur["join_key"] = _norm(lib)
            elif line.startswith("!Sample_characteristics_ch1"):
                kv = line.split("=", 1)[1].strip()
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    cur[f"clin_{k.strip().replace(' ', '_')}"] = v.strip()
        if cur:
            rows.append(cur)
    return pd.DataFrame(rows)


def main() -> None:
    raw = dataset_dest("host_transcriptomics", create=False)
    soft = next(iter(sorted(raw.glob("**/*family.soft.gz"))), None)
    if not soft:
        log.error("soft file not found under %s", raw)
        return
    log.info("parsing %s", soft)
    df = parse_soft(soft)
    log.info("parsed %d transcriptome samples", len(df))

    # cross-check against the processed counts matrix sample ids
    mat = (get_path("data_processed") / "host_transcriptomics"
           / "transcriptome_patient_by_gene.tsv")
    if mat.exists():
        ids = pd.read_csv(mat, sep="\t", index_col=0, nrows=0)
        # counts ids are the row index -> read the index separately
        counts_ids = pd.read_csv(mat, sep="\t", usecols=[0]).iloc[:, 0].astype(str)
        counts_keys = set(counts_ids.map(_norm))
        df["in_counts_matrix"] = df["join_key"].isin(counts_keys)
        log.info("%d/%d soft samples matched to the counts matrix by library key",
                 int(df["in_counts_matrix"].sum()), len(df))

    out = get_path("sample_maps", create=True) / "transcriptome_clinical.tsv"
    df.to_csv(out, sep="\t", index=False)
    log.info("WROTE %s", out)
    log.info("clinical columns: %s",
             [c for c in df.columns if c.startswith("clin_")])
    if "clin_match_group" in df.columns:
        log.info("match-group counts:\n%s",
                 df["clin_match_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
