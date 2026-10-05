#!/usr/bin/env python
"""Phase 10,13,14 -- integration benchmark runner (blueprint Sections 39-47, 69).

Runs the mandatory baselines (Section 40) and established integration methods
(Section 69) under the SAME nested CV + seeds so comparisons are fair
(Sections 70, 73, 74). Writes per-method metrics + bootstrap CIs and a
benchmark table.

Method families (config/models.yaml):
  baselines: clinical_only, parasite_resistance_rules, single_omic, concatenation
  integration: mofa, snf, diablo, autoencoder, vae, mogonet, late_fusion

STUB: iterates the configured methods and records where each plugs in.
"""
import _bootstrap  # noqa: F401
from src.utils.config import load_config, seeds
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()["models"]
    log.info("seeds: %s", seeds())
    log.info("CV: %s", cfg["cross_validation"])

    log.info("--- baselines (Section 40) ---")
    for name in cfg["baselines"]:
        log.info("[STUB] baseline: %s -> src.models.baselines", name)

    log.info("--- integration methods (Section 69) ---")
    for name in cfg["integration_methods"]:
        log.info("[STUB] method: %s", name)
        # mofa   -> mofapy2 / MOFA2 (R); latent factors (Section 42)
        # snf    -> snfpy; fused patient network (Section 43)
        # diablo -> mixOmics (R); matched samples required (Section 44)
        # autoencoder / vae -> torch (Sections 45-46)
        # mogonet -> torch GCN + VCDN (Section 47)
        # late_fusion -> per-modality models + learned gating (Section 39.5)

    log.info("DECISION GATE 3 (Section 95): establish baselines BEFORE HPAMM-Net.")
    log.info("done (stub). Fill in each method; write metrics to results/.")


if __name__ == "__main__":
    main()
