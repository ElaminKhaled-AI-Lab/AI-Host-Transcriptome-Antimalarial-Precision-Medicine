# Nano-DDS formulation-AI + precision-formulation arm (Phases 19-20)

Designs and optimizes polymeric NANOPARTICLE (NP) suspensions of antimalarial
drugs for paediatric oral therapy, characterizes them fully, and matches a
formulation to each individual child using the project's multi-omics layer.
All quantities are in-silico predictions from a documented, literature-anchored
digital twin -- for method development, not a substitute for bench data.

## Run everything

```bash
conda activate malaria
python scripts/13_nano_dds/run_nano_dds_pipeline.py --drug lumefantrine
```

That runs the factorial, the full dissolution analysis (3 replicate batches,
5% batch-to-batch variability), optimization, stability + literature validation,
the whole formulation-omics arm, all nine figures, and the R response-surface
analysis when `Rscript` is available.

## Files

| File | Purpose |
|------|---------|
| `src/models/nano_dds.py` | digital twin: size, PDI, zeta, EE, DL, release, liver targeting, ICH stability |
| `src/models/dissolution_metrics.py` | all five kinetic models + DE, MDT, t25/50/80, burst, f1/f2 |
| `src/models/profile_net.py` | PROFILE-Net and the three comparator integration strategies |
| `src/utils/patient_cohort.py` | paediatric cohort loader (real Mali dual-RNA first, then synthetic) |
| `13_nano_dds/run_factorial_dds.py` | 3^3 factorial x 11 polymers x 9 antimalarials |
| `13_nano_dds/analyze_dissolution.py` | every dissolution parameter, with replicates and noise |
| `13_nano_dds/optimize_nano_dds.py` | RF/GB/MLP surrogates + Box-Behnken RSM + Bayesian optimization |
| `13_nano_dds/validate_and_report.py` | ICH stability + literature validation + architecture |
| `13_nano_dds/make_all_figures.py` | the nine-figure suite (F1-F9) |
| `13_nano_dds/rsm_dissolution_analysis.R` | classical ANOVA / response surface / f2 in R |
| `14_omics_formulation/build_patient_formulation_dataset.py` | child x formulation dataset |
| `14_omics_formulation/benchmark_integration.py` | which integration strategy wins |
| `14_omics_formulation/personalize_all_patients.py` | optimal formulation for every child |

## Figures

F1 characterization by polymer; F2 release profiles (all polymers, drug named);
F3 dissolution parameters; F4 kinetic-model comparison; F5 DoE main effects and
interactions; F6 optimization landscape; F7 ICH stability; F8 literature
validation; F9 precision formulation across the cohort + integration benchmark.
`FR1_rsm_contours.png` comes from the R analysis.

## The method: PROFILE-Net, and what actually won

**PROFILE-Net** -- *Patient-Response Omics-Formulation Interaction Learning
Engine*. Integration type: **intermediate (joint-latent) with low-rank bilinear
cross-attention**, so patient-by-formulation interactions are modelled
explicitly rather than as a sum of main effects.

It was benchmarked head-to-head against the three standard alternatives on a
patient-grouped split (test children never seen in training). The deciding
metric is the **interaction-residual rank correlation**: each formulation's mean
effect across children is removed first, so only the patient-specific component
is scored. A purely additive model scores ~0 there by construction.

Results are averaged over **3 independent seeds** (mean ± SD), because a single
run flips the ranking -- an earlier single-run version of this table reported a
different winner, which is exactly why seed averaging is now the default.

| Strategy | Integration type | exposure R² | utility R² | interaction rank | personalization gain |
|---|---|---|---|---|---|
| early_concat | early | 0.992 ± 0.001 | 0.986 | 0.932 ± 0.004 | +0.0240 |
| late_ensemble | late | 0.985 ± 0.003 | 0.946 | **0.008 ± 0.022** | **−0.0030** |
| intermediate_latent | intermediate | 0.980 ± 0.002 | 0.965 | 0.930 ± 0.009 | +0.0248 |
| PROFILE-Net | intermediate + interaction | 0.973 ± 0.003 | 0.969 | 0.938 ± 0.011 | +0.0246 |

**Honest conclusion, in two parts.**

1. **Late (additive) integration is decisively disqualified.** It scores 0.008 on
   the interaction metric -- statistically zero, exactly as theory predicts for a
   model with no interaction term -- and its personalization gain is *negative*.
   It cannot personalize at all.
2. **The three JOINT strategies are statistically indistinguishable.** PROFILE-Net
   has the highest mean (0.938), but its margin over the runner-up is 0.005 against
   a combined seed standard deviation of 0.016. The benchmark reports this as
   `winner_is_separated_from_noise: false` and it should be read as **a tie**, not
   a win. Claiming PROFILE-Net superiority from these data would be unsupported.

The defensible recommendation is `early_concat` for production -- it ties on
accuracy and interaction with the fewest parameters (36.6k versus 97.7k) -- and
PROFILE-Net only where its explicit interaction term is wanted for
interpretability. The architectural lesson that *is* supported is about
integration TYPE (joint beats additive), not about our particular architecture.

## Precision formulation across the cohort

Every child is assigned the formulation maximizing therapeutic utility
(efficacy net of toxicity risk, via a therapeutic window: too much exposure
harms, and susceptibility scales with the host toxicity proxy).

- 66.7% of children have an optimum **different** from the single best product
- 5 different polymers chosen across the cohort
- mean utility gain +0.027 (max +0.153) over one-size-fits-all

## Why Python *and* R

Python carries the deep learning and Bayesian optimization, where the
PyTorch/scikit-learn ecosystem is strongest. R carries the classical
design-of-experiments and dissolution statistics in the form regulators and
formulation scientists read them -- ANOVA tables with F and p values,
second-order response surfaces with lack-of-fit, canonical stationary-point
analysis (`rsm` package), and f2 similarity. The two are cross-checked.

## GPU note

The deep-learning step verifies that the GPU actually works before using it.
`torch.cuda.is_available()` returns True even when the installed PyTorch has no
compiled kernels for the device -- a Quadro P4000 (sm_61) against an sm_75+ build
fails every kernel launch with `cudaErrorNoKernelImageForDevice`. The selector
checks the compute capability against the build's architecture list and runs a
real tensor operation, falling back to the processor with a clear message.
Force it either way with `--device cpu|cuda` or `MALARIA_DEVICE=cpu`.
On this dataset processor-only training takes roughly 3-5 minutes for all four
strategies at 3 seeds.

## Known limitations (stated deliberately)

- Ranking among the joint architectures is within seed noise, so the benchmark
  supports a conclusion about integration TYPE, not about which joint architecture
  is best. More seeds would be needed to separate them.
- The twin is deterministic; batch variability is *injected* (default 5% RSD,
  3 replicates) rather than measured, so ANOVA p values describe the simulation,
  not laboratory reality.
- Higher polymer:drug ratio raises the release plateau in the twin (because
  entrapment efficiency rises); real matrices often also slow the rate more
  strongly than modelled here.
- The nanosuspension shelf-life estimates are short enough that a real product
  would likely require lyophilization -- the model flags this rather than hiding it.
- For *Plasmodium falciparum* the liver stage is transient; liver-targeted
  delivery is most relevant to causal prophylaxis and to *P. vivax*/*P. ovale*
  hypnozoite radical cure. The galactose module represents asialoglycoprotein-
  receptor hepatocyte targeting.

## Literature anchors (validation)

Primaquine nanochitosan-galactose (AAC 2023, doi:10.1128/aac.00915-23; liver/plasma
0.10 -> 0.58); curcumin-artesunate PLGA 50:50 (PMC5988888); DHA-lumefantrine
polymeric NP (Heliyon 2024). Validation is by **design-space overlap**: 11/11
published values fall inside the twin's achievable range.
