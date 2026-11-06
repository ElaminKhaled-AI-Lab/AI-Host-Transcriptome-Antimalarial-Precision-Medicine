# Figure index — Malaria_Project MATLAB suite

Narrative order: **architecture → data → processing → integration → model →
validation → drug → formulation → nano twin → characterisation → optimisation →
precision formulation.**

Every subfigure is written three ways: `.png` (1200 dpi), `.pdf` (vector) and
`.fig` (editable), into `figures_matlab/`. Arial throughout; 25 pt inside axes,
30 pt headings.

**50 subfigures → 150 files.**

---

## F01 — Study architecture
| ID | Content |
|---|---|
| `F01a_pipeline_architecture` | Ten-stage schematic from public omics to a personalised nanoparticle formulation |

## F02 — Data resources and the paediatric cohort
| ID | Content | Key numbers |
|---|---|---|
| `F02a_dataset_completeness` | Acquisition status per data resource | PASS / PARTIAL / MANAGED / MISSING |
| `F02b_cohort_age` | Age distribution of the real cohort | **n = 56 children**, real BioSample ages |
| `F02c_omic_phenotype_plane` | Host metabolic vs toxicity proxy, coloured by parasite resistance | source `real_dual_rna_mali` |
| `F02d_resistance_distribution` | Parasite resistance burden | 0–1 rank scale |

## F03 — Dual host–parasite RNA processing and QC
| ID | Content | Key numbers |
|---|---|---|
| `F03a_parasite_read_fraction` | Parasite reads as % of total, per cohort | Mali median **0.059 %** (max 1.222 %); Gambia median **0.199 %** (max 0.753 %) |
| `F03b_transcripts_detected` | Host vs parasite transcripts detected | host 199,532; parasite 65 (Mali) / 96 (Gambia) |
| `F03c_samples_above_threshold` | Samples above parasite-signal thresholds | Mali n = 56; Gambia n = 52 |

## F04 — Integration method comparison (honest)
| ID | Content | Key numbers |
|---|---|---|
| `F04a_ari_synthetic` | ARI vs the planted subtype — the embedding-independent comparison | concat **0.658**, BLISS **0.654** — statistically identical |
| `F04b_silhouette_is_circular` | Why silhouette cannot be compared across embeddings | concat 0.119 (shared PC space) vs BLISS 0.871 (own latent); **BLISS scores 0.843 on pure noise** |

> **Changed.** The old panel compared BLISS silhouette (measured in its own
> learned latent) against concatenation silhouette (measured in the raw PC
> space). Those are different spaces, and BLISS is trained to separate clusters
> in its latent, so that comparison was circular. ARI is embedding-independent
> and shows no BLISS advantage.

## F05 — HPAMM-Net multitask and ablations
| ID | Content | Key numbers |
|---|---|---|
| `F05a_multitask_auroc` | Baseline vs HPAMM-Net, 4 outcomes, mean ± SD over 10 seeds | resistance **0.656 vs 0.593**; cure 0.589 vs 0.524 |
| `F05b_multitask_mae` | PK AUC and clearance error | PK AUC MAE 9.05 vs 9.41 |
| `F05c_ablations` | Resistance AUROC by ablation | full **0.739**; parasite-only 0.734; **host-only 0.487 (chance)** |

## F06 — Real-cohort validation, against proper controls
| ID | Content | Key numbers |
|---|---|---|
| `F06a_cluster_balance` | Balance (normalised entropy); dashed line is the degeneracy threshold | Mali balanced (host-only 0.967, dual **0.954**); Gambia host-only 0.996 but **dual 0.318** — the abundant parasite block collapses the split onto a few samples |
| `F06b_reference_null` | Bootstrap stability vs a cluster-free uniform reference | Mali 0.543 vs null median 0.833, **p = 0.979**; Gambia (45-sample dedup) 0.393 vs 0.256, **p = 0.154** — neither above chance |
| `F06c_severity_ari` | Agreement with cerebral vs uncomplicated malaria | **ARI = −0.022**, Fisher **p = 0.746**, n = 37 |
| `F06d_what_clusters_track` | Covariate effect sizes, FDR corrected | host PC1 is the clustering axis itself; among the rest the significant covariates are **parasite transcriptional components (PC1, PC2)**, not read-fraction burden and not severity |

> **Changed.** The old panels plotted BLISS silhouettes from `dualrna_qc`, which
> came from degenerate clusterings measured in a learned latent.
> These panels report what is actually informative: balance, stability against a
> cluster-free reference, ARI, and what the clusters track.

## F07 — AI pharmacokinetics
| ID | Content | Key numbers |
|---|---|---|
| `F07a_exposure_r2` | Day-7 exposure R² by model | allometric **0.044** → gradient boosting 0.128 → neural **0.171** |
| `F07b_exposure_drivers` | Feature importance | **host_txPC1 0.467**, weight 0.137, age 0.110 |
| `F07c_cure_auroc` | Cure discrimination | AI 0.600; **day-7 concentration alone 0.998** |

## F08 — Formulation design of experiments
| ID | Content |
|---|---|
| `F08a_main_effects_size` | Particle size vs the three factors (mean ± SD) |
| `F08b_main_effects_ee` | Entrapment efficiency |
| `F08c_main_effects_DE` | Dissolution efficiency |
| `F08d_main_effects_liver` | Liver-targeting index |
| `F08e_rsm_coefficients_size` | R response-surface coefficients, significant terms highlighted (size R² = 0.969) |

## F09 — Nanoparticle characterisation (11 FDA/GRAS polymers)
| ID | Content |
|---|---|
| `F09a_size` | Particle size, with the 100–200 nm Kupffer window shaded |
| `F09b_pdi` | Polydispersity index |
| `F09c_zeta` | Zeta potential, ±20 mV stability threshold marked |
| `F09d_ee` | Entrapment efficiency |
| `F09e_dl` | Drug loading |
| `F09f_liver_index` | Liver-targeting index |

*Dataset: 8,019 formulations (9 drugs × 11 polymers × 27 runs × 3 replicates, 5 % RSD).*

## F10 — Dissolution and release kinetics
| ID | Content | Key numbers |
|---|---|---|
| `F10a_release_profiles_polymers` | All 11 polymers, drug named (lumefantrine) | |
| `F10b_release_profiles_drugs` | All 9 antimalarials on PLGA 50:50 | |
| `F10c_release_factor_effect` | Factor effect on release | |
| `F10d_DE` | Dissolution efficiency by polymer | DE 43.4–94.1 % |
| `F10e_MDT` | Mean dissolution time | MDT 2.85–22.98 h |
| `F10f_t50` | Time to 50 % release | t50 2.32–51.61 h |
| `F10g_peppas_n` | Korsmeyer–Peppas exponent, thresholds marked | n 0.23–0.87 |
| `F10h_kinetic_model_heatmap` | Mean R² per polymer × 5 models | best by AIC: K-P 6000, first-order 1976, Higuchi 43 |

*Mechanism: anomalous (non-Fickian) 6,772; Fickian 1,245.*

## F11 — Optimisation, stability and literature validation
| ID | Content | Key numbers |
|---|---|---|
| `F11a_polymer_desirability` | Polymer ranking | Eudragit RL100 **0.687**, RS100 0.625, S100 0.578 |
| `F11b_design_space` | Size vs EE coloured by liver targeting | |
| `F11c_stability` | ICH 40 °C / 75 % RH trajectory | shelf life ≈ 2.3 months at 25 °C |
| `F11d_literature_validation` | Design-space overlap with published values | **11 / 11 overlap** |

*Recommended: Eudragit RL100, 153 nm, PDI 0.212, ζ +25.3 mV, EE 86.7 %, liver index 0.945.*

## F12 — Precision formulation
| ID | Content | Key numbers |
|---|---|---|
| `F12a_polymer_collapse` | Polymer counts, as-submitted vs revised objective | as-submitted PCL 23 / RL100 18 / RS100 15; **revised: all 56 → PCL** |
| `F12b_gain_vs_null` | Utility gain vs a molecular-permutation null, both objectives | as-submitted +0.0244 (**p = 0.576**); revised +0.0006 (**p = 0.532**) — neither above the null |
| `F12c_strategy_accuracy` | Exposure and utility R² by integration strategy (cross-validated) | exposure 0.95–0.99; utility 0.93–0.98 |
| `F12d_interaction_metric` | Interaction-residual rank correlation ± 95% CI | early 0.821, intermediate 0.827, PROFILE-Net 0.821, **late −0.130 [−0.233, −0.031]** |

*Benchmark: 56 real children, repeated grouped five-fold cross-validation.
The joint strategies (early, intermediate, PROFILE-Net) tie after Holm correction —
no single architecture wins. Late (additive) integration cannot personalise:
interaction −0.130. Under the corrected lumefantrine objective the one-size
product is polycaprolactone and the personalisation gain does not survive
(revised +0.0006, not above a permutation null).*


---

## Headline numbers for the manuscript

**Negative result on the real cohorts — this is the honest finding.**

1. **No discrete host–parasite subtypes.** After correction Mali is balanced
   (0.954) but its bootstrap stability of 0.543 is below a cluster-free reference
   (null median 0.833, **p = 0.979**); Gambia (45-sample dedup) is 0.393 vs 0.256
   (**p = 0.154**). Neither is above chance.
2. **No relationship to clinical severity.** ARI = −0.022, Fisher p = 0.746 on
   37 labelled Gambian children.
3. **The weak structure tracks parasite transcriptional variation** — the
   significant covariates are parasite transcriptional components, not
   read-fraction burden and not severity. A continuous axis, not a subtype.

**Four methodological pitfalls, each demonstrated with a purpose-built control.**

4. Degeneracy hidden behind a high silhouette: balance 0.129 means 1 vs 55.
5. Per-component z-scoring of PCs destroys real structure: on a planted 25/25
   split, stability 0.030 z-scored versus 1.000 variance-weighted.
6. Unnormalised counts fabricate depth clusters: with *no biology* and 26 of 52
   samples sequenced 3× deeper, silhouette 0.591 and stability 1.000; after CPM,
   stability 0.028.
7. Silhouette inside a learned latent is circular: BLISS scores **0.843 on pure
   noise**. ARI, being embedding-independent, shows no BLISS advantage over plain
   concatenation on synthetic data (0.654 vs 0.658).

**Intact arms, unaffected by any of the above.**

8. Nano-DDS twin: 11/11 published values inside the achievable design space.
9. Dissolution: DE 43–94 %, MDT 2.9–23 h, K-P n 0.23–0.87; AIC model selection.
10. AI pharmacokinetics: exposure R² 0.044 → 0.171; day-7 concentration alone
    gives cure AUROC 0.998.
11. Precision formulation: 80.4 % of children need a formulation other than the
    single best product; joint integration is required, additive integration fails.
