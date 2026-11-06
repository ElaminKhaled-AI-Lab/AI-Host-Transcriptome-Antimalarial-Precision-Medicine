# Revision — November 2026

Corrections applied for the resubmission. Every change below is reflected in the
code, the regenerated result files, and the figure scripts in this commit.

## Transcript-split fix (the one that drove the re-analysis)

`scripts/03_preprocess/preprocess_dual_rna.py` classified a transcript as parasite
only when its name matched `PF3D7|Plasmodium|ASM276`. The Ensembl Protists
ASM276v2 cDNA uses ENA-style identifiers for most transcripts, so nearly all
parasite reads were counted as host. The classifier is now the complement of the
human identifier:

```python
is_pf = ~d["Name"].str.startswith("ENST")
```

Median parasite read fraction rises from 0.059% to 12.8% (Mali) and from ~0.20%
to 37.6% (Gambia, 52 sequenced libraries). The parasite layer is substantial, not
sparse, once reads are attributed to the correct organism.

## Clustering re-analysis

With correct library-size normalisation and variance-weighted blocks, neither real
cohort shows reproducible discrete structure:

- Mali is balanced (0.954) but bootstrap stability 0.543 sits below a cluster-free
  reference (null median 0.833, p = 0.979).
- Gambia, using the 45-sample deduplicated set as primary, has stability 0.393
  (null 0.256, p = 0.154).
- Cluster membership is unrelated to cerebral-vs-uncomplicated severity: ARI
  −0.022, Fisher p = 0.746 on 37 labelled children.
- The weak structure tracks parasite transcriptional components, not read-fraction
  burden and not severity.

New/updated result files: `results/dualrna_qc_{real_mali,external_gambia}.json`,
`results/final_clustering_real_mali.json`,
`results/final_clustering_external_gambia_dedup.json`.

## Personalization collapses under a drug-appropriate objective

The nano-DDS objective originally carried a liver-targeting term, which is
inappropriate for lumefantrine (a blood-stage drug). With the term removed
(`src/models/nano_dds.py`, `scripts/13_nano_dds/optimize_nano_dds.py`,
`scripts/14_omics_formulation/build_patient_formulation_dataset.py`):

- All 56 children are assigned polycaprolactone; individual assignment changes the
  process settings for 16.1% and the mean utility gain falls to 0.0006.
- Neither the revised gain nor the larger as-submitted gain (0.0244) exceeds a
  molecular-permutation null (p = 0.53 and 0.58).

The integration-strategy benchmark now uses repeated grouped five-fold
cross-validation over the 56 children. Late/additive integration cannot represent
a patient-by-formulation interaction (interaction-residual rank correlation
−0.130); the joint strategies recover it and tie after Holm correction, so the
result is about the class of integration, not any single architecture.

Result files: `results/personalization_stats.json`,
`results/benchmark_cv_revised.json`.

## New revision analyses

`scripts/15_revision/` adds the analyses requested at review: personalization
nulls, the cross-validated integration benchmark, HPAMM-Net paired statistics, and
a power/sensitivity analysis. `_bootstrap.py` holds the shared resampling helpers.

## Figures

Only three figures changed. F03 (dual-RNA QC) rebuilds from the corrected QC files
with no script change. `F06_real_validation.m` reads the deduplicated Gambia
clustering and its captions are corrected. `F12_precision_formulation.m` is rebuilt
as four panels showing the personalization collapse and the cross-validated
benchmark, replacing the earlier six-panel figure. The other nine figures are
unchanged.

## What this work does not reach

Endpoints (exposure, cure, utility) are simulated; the transcriptomes are real but
the pharmacokinetic and formulation outcomes are model-generated. There is no
clinical validation, no wet-lab formulation, and no demonstration that any
individualization improves a real outcome. The negative clustering and
personalization results are the honest findings; the contribution is
methodological and the digital-twin framework.
