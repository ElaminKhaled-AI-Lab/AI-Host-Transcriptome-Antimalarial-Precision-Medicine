"""Semi-synthetic cohort generator (blueprint Sections 33-37).

Design principle (Section 33): a fully independent-random synthetic cohort is
biologically weak. This generator preserves REAL structure learned from the
matched real cohorts -- marginal distributions, cross-feature and cross-omic
covariance, host-parasite dependence, geography, resistance-marker
combinations, treatment distributions, and missingness.

Generation order (Sections 35.1-35.10) is deliberately causal-ish:

    country/site  ->  parasite genome | country
                  ->  host latent factors Z_H  ->  host omics = W Z_H + eps
                  ->  parasite RNA | host RNA, parasitemia, age, severity
                  ->  clinical | host state, parasite state, age
                  ->  treatment (preserve randomization / study probs)
                  ->  PK = f(age, weight, host, drug, dose) + eps
                  ->  outcome = f(resistance, host, PK, drug, dose, clinical)

The outcome generator MUST be declared before benchmarking so it cannot be
tuned to favour HPAMM-Net (Section 35.10). This file is a scaffold: the
``fit`` methods contain the interface and TODO markers where you plug in MOFA+
loadings, real haplotype pools, and empirical conditional distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class GeneratorConfig:
    n_patients: int = 2000            # Section 36 primary design
    seed: int = 11
    modalities: list[str] = field(default_factory=lambda: [
        "host_genome", "host_rna", "host_proteome", "host_metabolome",
        "parasite_genome", "parasite_rna",
    ])
    # Missingness sweep applied post-hoc (Section 37).
    modality_dropout_rate: float = 0.0


class SemiSyntheticGenerator:
    """Fit reference structure on real data, then sample synthetic patients.

    Typical use::

        gen = SemiSyntheticGenerator(GeneratorConfig(n_patients=2000, seed=11))
        gen.fit_demographics(real_clinical_df)
        gen.fit_host_latent(real_host_matched)      # MOFA+ loadings
        gen.fit_parasite_haplotypes(pf8_by_country)
        gen.fit_host_parasite_rna(dual_rna_df)
        gen.declare_outcome_model(...)              # BEFORE benchmarking
        cohort = gen.sample()
    """

    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._fitted: dict[str, bool] = {}

    # ------------------------------------------------------------ fit steps
    def fit_demographics(self, clinical_df) -> None:
        """Learn joint P(age, weight, sex, country) (Section 35.1-35.2).

        TODO: fit a copula or resample real (age, weight, sex, country) tuples
        so age and weight stay correlated. Store the empirical joint here.
        """
        self._demographics = clinical_df  # placeholder: resample rows
        self._fitted["demographics"] = True
        log.info("fit_demographics on %d real records", len(clinical_df))

    def fit_host_latent(self, host_matched) -> None:
        """Learn host latent factors Z_H and loadings W (Section 35.4).

        Recommended: run MOFA+ on matched host RNA + protein + metabolite, keep
        factor scores' distribution and per-omic loadings.  Then
        X_omic = W_omic @ Z_H + eps_omic.
        TODO: store W_rna, W_protein, W_metabolite, noise scales, factor dist.
        """
        self._fitted["host_latent"] = True
        log.info("fit_host_latent (plug in MOFA+ loadings)")

    def fit_parasite_haplotypes(self, pf8_by_country) -> None:
        """Preserve real haplotype blocks + resistance-marker combinations
        conditioned on geography (Section 35.3). Resample real haplotypes
        rather than simulating variants independently.
        TODO: store per-country haplotype pools and CNV / complexity states.
        """
        self._fitted["parasite_haplotypes"] = True
        log.info("fit_parasite_haplotypes (resample real haplotypes by country)")

    def fit_host_parasite_rna(self, dual_rna_df) -> None:
        """Learn P(parasite RNA | host RNA, parasitemia, age, severity)
        from the dual-RNA datasets (Section 35.5). Empirical dependence only;
        no causal claim.
        """
        self._fitted["host_parasite_rna"] = True
        log.info("fit_host_parasite_rna (empirical conditional dependence)")

    def declare_outcome_model(self, spec: dict) -> None:
        """Freeze the outcome generator BEFORE benchmarking (Section 35.10).

        ``spec`` records coefficients for
        outcome = f(resistance, host_state, PK, drug, dose, clinical).
        Persist this so reviewers can confirm it was not tuned to the model.
        """
        self._outcome_spec = spec
        self._fitted["outcome_model"] = True
        log.info("declared outcome model: %s", spec)

    # ------------------------------------------------------------ sampling
    def sample(self):
        """Return a synthetic cohort (pandas DataFrame or dict of matrices).

        TODO: implement the full generation order above using the fitted
        structures. Every synthetic patient carries the fields in blueprint
        Section 34, including MISSINGNESS masks and PROVENANCE (generator
        version, seed, source models).
        """
        missing = [k for k in ("demographics", "host_latent",
                               "parasite_haplotypes", "outcome_model")
                   if not self._fitted.get(k)]
        if missing:
            raise RuntimeError(f"Generator not fully fitted; missing: {missing}")
        raise NotImplementedError(
            "Implement sampling per blueprint Sections 34-35. Scaffold only."
        )

    def provenance(self) -> dict:
        return {
            "generator_version": "0.1.0",
            "random_seed": self.cfg.seed,
            "n_patients": self.cfg.n_patients,
            "source_models": sorted(k for k, v in self._fitted.items() if v),
        }
