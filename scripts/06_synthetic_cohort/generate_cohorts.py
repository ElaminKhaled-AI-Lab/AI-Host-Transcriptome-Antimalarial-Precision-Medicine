#!/usr/bin/env python
"""Phase 11 -- generate semi-synthetic cohorts (blueprint Sections 33-36).

Primary design (Section 36): 10 independent cohorts x 2,000 synthetic patients,
each seeded from config/seeds.yaml:synthetic_cohort_seeds. Sensitivity cohorts
at N = 250, 500, 1000, 2000, 5000.

Boundary (Section 7.1 / 36): synthetic patients are NOT clinical evidence and
20,000 simulated records are NOT 20,000 observed children.

Uses src.simulation.generator.SemiSyntheticGenerator. The generator must be
FITTED on real matched cohorts and its outcome model DECLARED before
benchmarking (Section 35.10). This runner wires those steps together.
"""
import _bootstrap  # noqa: F401
import argparse

from src.utils.config import load_config, get_path
from src.utils.logging_utils import get_logger
from src.simulation.generator import SemiSyntheticGenerator, GeneratorConfig

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="patients per cohort")
    ap.add_argument("--n-cohorts", type=int, default=10)
    args = ap.parse_args()

    seeds = load_config()["seeds_cfg"]["synthetic_cohort_seeds"][: args.n_cohorts]
    out_dir = get_path("data_synthetic", create=True)

    for i, seed in enumerate(seeds):
        log.info("cohort %d/%d (seed=%d, N=%d)", i + 1, len(seeds), seed, args.n)
        gen = SemiSyntheticGenerator(GeneratorConfig(n_patients=args.n, seed=seed))
        # TODO wire real inputs:
        #   gen.fit_demographics(real_clinical_df)
        #   gen.fit_host_latent(matched_host_multiomics)   # MOFA+ loadings
        #   gen.fit_parasite_haplotypes(pf8_by_country)
        #   gen.fit_host_parasite_rna(dual_rna_df)
        #   gen.declare_outcome_model(frozen_spec)          # BEFORE benchmarking
        #   cohort = gen.sample()
        #   cohort.to_parquet(out_dir / f"cohort_seed{seed}.parquet")
        log.info("[STUB] provenance: %s", gen.provenance())

    log.info("done (stub). Implement generator.sample() per Sections 34-35.")
    log.info("DECISION GATE 2 (Section 95): proceed only if synthetic data "
             "preserve real joint structure (validated in validate_cohorts.py).")


if __name__ == "__main__":
    main()
