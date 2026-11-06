#!/usr/bin/env python
"""Phase 5 -- Dual host-parasite RNA-seq processing (blueprint Section 29).

Turns the downloaded ENA dual-RNA FASTQ into REAL, matched host + parasite
expression matrices -- the real-data validation cohort for the integration
methods (synthetic -> real Mali -> external Gambia).

Each sample's reads are quantified against a COMBINED human + Plasmodium
falciparum transcriptome with Salmon; transcripts are then split by organism
into a host-RNA matrix and a parasite-RNA matrix (same samples in both).

Pipeline:
  1. build a combined Salmon index (human GRCh38 cDNA + Pf 3D7 cDNA)     [once]
  2. salmon quant each paired-end sample (resumable; skips completed)
  3. assemble host_rna_counts.tsv + parasite_rna_counts.tsv (samples x transcripts)
  4. record parasite-read fraction per sample (a real QC metric, Section 29.7)

Requirements: salmon (bioconda) + ~15 GB disk for the index. This is compute-
heavy; run it in the background:  nohup python ... &

Usage:
  python scripts/03_preprocess/preprocess_dual_rna.py --dataset dual_rna_mali --build-index
  python scripts/03_preprocess/preprocess_dual_rna.py --dataset dual_rna_mali
  python scripts/03_preprocess/preprocess_dual_rna.py --dataset dual_rna_gambia
"""
import _bootstrap  # noqa: F401
import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.utils.config import get_dataset, dataset_dest, get_path, load_config
from src.utils.download import http_download
from src.utils.provenance import result_manifest
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Reference transcriptomes (override with --human-cdna / --pf-cdna if versions move).
HUMAN_CDNA = ("https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/cdna/"
              "Homo_sapiens.GRCh38.cdna.all.fa.gz")
# EBI FTP (valid TLS cert); EnsemblGenomes is mirrored under /ensemblgenomes/pub/.
PF_CDNA = ("https://ftp.ebi.ac.uk/ensemblgenomes/pub/release-60/protists/fasta/"
           "plasmodium_falciparum/cdna/Plasmodium_falciparum.ASM276v2.cdna.all.fa.gz")


def sh(cmd, **kw):
    log.info("$ %s", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def build_index(ref_dir: Path, threads: int, human_url: str, pf_url: str) -> Path:
    ref_dir.mkdir(parents=True, exist_ok=True)
    human = ref_dir / "human_cdna.fa.gz"
    pf = ref_dir / "pf_cdna.fa.gz"
    combined = ref_dir / "combined_cdna.fa.gz"
    idx = ref_dir / "salmon_index"
    if idx.exists():
        log.info("index exists: %s", idx)
        return idx
    http_download(human_url, human)
    http_download(pf_url, pf)
    # concatenate (gzip streams concatenate fine)
    with open(combined, "wb") as out:
        for f in (human, pf):
            out.write(f.read_bytes())
    if not shutil.which("salmon"):
        raise RuntimeError("salmon not found. Install: conda install -c bioconda salmon")
    sh(["salmon", "index", "-t", combined, "-i", idx, "-k", "31", "-p", threads])
    return idx


def find_samples(dest: Path):
    """Return {run_id: (R1, R2)} for paired FASTQ under dest/fastq/<run>/."""
    out = {}
    fq = dest / "fastq"
    if not fq.exists():
        return out
    for run_dir in sorted(fq.iterdir()):
        if not run_dir.is_dir():
            continue
        r1 = sorted(run_dir.glob("*_1.fastq.gz"))
        r2 = sorted(run_dir.glob("*_2.fastq.gz"))
        if r1 and r2:
            out[run_dir.name] = (r1[0], r2[0])
        else:  # single-end fallback
            se = sorted(run_dir.glob("*.fastq.gz"))
            if se:
                out[run_dir.name] = (se[0], None)
    return out


def quant_sample(idx: Path, run: str, reads, out_dir: Path, threads: int,
                 timeout_s: int = 1800):
    q = out_dir / run
    if (q / "quant.sf").exists():
        log.info("quant exists, skipping: %s", run)
        return q
    r1, r2 = reads
    cmd = ["salmon", "quant", "-i", str(idx), "-l", "A", "-p", str(threads), "-o", str(q)]
    cmd += (["-r", str(r1)] if r2 is None else ["-1", str(r1), "-2", str(r2)])
    log.info("$ %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        log.warning("TIMEOUT after %ds on %s -- skipping this sample", timeout_s, run)
        return None
    except subprocess.CalledProcessError as e:
        log.warning("salmon error on %s: %s -- skipping", run, e)
        return None
    return q


def assemble(quant_root: Path, runs, out_dir: Path):
    """Split combined quant into host vs parasite matrices + parasite-read fraction."""
    host_cols, para_cols, frac = {}, {}, {}
    for run in runs:
        sf = quant_root / run / "quant.sf"
        if not sf.exists():
            continue
        d = pd.read_csv(sf, sep="\t")
        # Revision fix: classify by the HUMAN identifier, not by a parasite name pattern.
        # Ensembl Protists P. falciparum ASM276v2 cDNA uses ENA-style transcript IDs
        # (CAD..., CZT..., CAX...) for most transcripts; the earlier pattern
        # "PF3D7|Plasmodium|ASM276" matched only 149 of ~5,500 and assigned the rest,
        # with their reads, to the host block. Every Ensembl human transcript is ENST*.
        is_pf = ~d["Name"].str.startswith("ENST")
        host_cols[run] = d.loc[~is_pf].set_index("Name")["NumReads"]
        para_cols[run] = d.loc[is_pf].set_index("Name")["NumReads"]
        tot = d["NumReads"].sum()
        frac[run] = float(d.loc[is_pf, "NumReads"].sum() / tot) if tot else 0.0
    host = pd.DataFrame(host_cols).T          # samples x host transcripts
    para = pd.DataFrame(para_cols).T          # samples x parasite transcripts
    out_dir.mkdir(parents=True, exist_ok=True)
    host.to_csv(out_dir / "host_rna_counts.tsv", sep="\t")
    para.to_csv(out_dir / "parasite_rna_counts.tsv", sep="\t")
    pd.Series(frac, name="parasite_read_fraction").to_csv(
        out_dir / "parasite_read_fraction.tsv", sep="\t")
    return host, para, frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dual_rna_mali",
                    choices=["dual_rna_mali", "dual_rna_gambia"])
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="pilot cap")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip salmon; build matrices from existing quant dirs")
    ap.add_argument("--sample-timeout", type=int, default=1800,
                    help="per-sample salmon timeout in seconds (skips a hung sample)")
    ap.add_argument("--human-cdna", default=HUMAN_CDNA)
    ap.add_argument("--pf-cdna", default=PF_CDNA)
    args = ap.parse_args()

    cfg = load_config()
    threads = args.threads or cfg["compute"]["n_jobs"]
    ref_dir = get_path("data_interim", create=True) / "dual_rna_reference"
    dest = dataset_dest(args.dataset, create=False)

    quant_root = get_path("data_interim") / f"salmon_{args.dataset}"
    samples = find_samples(dest)
    if args.max_samples:
        samples = dict(list(samples.items())[: args.max_samples])

    if not args.assemble_only:
        if not shutil.which("salmon"):
            log.error("salmon not installed. In the malaria env: "
                      "conda install -c bioconda salmon")
            return
        idx = build_index(ref_dir, threads, args.human_cdna, args.pf_cdna) \
            if args.build_index or not (ref_dir / "salmon_index").exists() \
            else ref_dir / "salmon_index"
        log.info("%s: %d samples to quantify (per-sample timeout %ds)",
                 args.dataset, len(samples), args.sample_timeout)
        if not samples:
            log.error("no FASTQ found under %s/fastq/. Did the download finish?", dest)
            return
        for run, reads in samples.items():
            quant_sample(idx, run, reads, quant_root, threads, args.sample_timeout)
    else:
        done = sum(1 for r in samples if (quant_root / r / "quant.sf").exists())
        log.info("assemble-only: %d/%d samples have completed quant", done, len(samples))

    out_dir = get_path("data_processed", create=True) / f"dual_rna_{args.dataset.split('_')[-1]}"
    host, para, frac = assemble(quant_root, samples, out_dir)
    result_manifest(f"preprocess_dual_rna_{args.dataset}",
                    extra={"n_samples": int(host.shape[0]),
                           "n_host_transcripts": int(host.shape[1]),
                           "n_parasite_transcripts": int(para.shape[1]),
                           "median_parasite_read_fraction":
                               float(pd.Series(frac).median()),
                           "reference_human": args.human_cdna,
                           "reference_pf": args.pf_cdna})
    log.info("WROTE host %s + parasite %s -> %s", host.shape, para.shape, out_dir)
    log.info("median parasite-read fraction: %.3f", pd.Series(frac).median())
    log.info("This is REAL matched host+parasite RNA -> use for real-data validation.")


if __name__ == "__main__":
    main()
