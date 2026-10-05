#!/usr/bin/env python
"""Phase 1-8 -- single-omics preprocessing dispatcher (blueprint Sections 25-31).

Runs modality-specific preprocessing to produce ML-ready matrices in
data/processed/. Each modality is a function stub tied to its blueprint section.

Design rules enforced downstream:
  * raw files are immutable (Section 23); outputs go to data/interim | processed
  * do not mix technical MS runs with biological patients (Section 26)
  * do not turn uncertain spectral features into definitive names (Section 27)
  * pre-treatment predictors only, when claiming pre-treatment prediction (Section 31)
"""
import _bootstrap  # noqa: F401
import argparse

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

MODALITIES = [
    "host_transcriptome",   # Section 25: counts -> filter -> normalize -> DE -> pathway
    "host_proteome",        # Section 26: Track A raw archive; Track B patient x protein
    "host_metabolome",      # Section 27: peak detect -> align -> annotate -> normalize
    "parasite_genome",      # Section 28: Pf8 features; QC; LD prune; PCA
    "parasite_transcriptome",  # Section 29: split host/parasite; stage; normalize
    "host_genome",          # Section 30: QC; ancestry PCs; pharmacogene extraction
    "clinical_pk",          # Section 31: standardize units/time; pre-treatment design
]


def preprocess(modality: str) -> None:
    log.info("[STUB] preprocess %s (see blueprint section for exact steps)", modality)
    # TODO: implement per Section 25-31 and write to data/processed/<modality>/


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modality", choices=MODALITIES + ["all"], default="all")
    args = ap.parse_args()
    targets = MODALITIES if args.modality == "all" else [args.modality]
    for m in targets:
        preprocess(m)
    log.info("done (stub). Fill in per-modality steps before single-omics analysis.")


if __name__ == "__main__":
    main()
