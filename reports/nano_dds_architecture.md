# Nano-DDS formulation-AI module -- architecture

A machine-learning / deep-learning pipeline that designs and optimizes polymeric
NANOPARTICLE (NP) suspensions of antimalarial drugs for pediatric oral therapy,
with liver-stage targeting in view, and hooks into the project's patient
multi-omics layer for precision medicine.

## Layered architecture

```
 (1) DIGITAL TWIN            src/models/nano_dds.py
     Formulation factors  ->  size, PDI, zeta, EE, DL, release profile,
     (polymer, drug,           liver-targeting index, liver/plasma ratio
      3 process factors)       + 5-model release-kinetics fitting
            |
            v
 (2) DESIGN OF EXPERIMENTS   scripts/13_nano_dds/run_factorial_dds.py
     3^3 full factorial (27 runs) x every FDA/GRAS polymer x every antimalarial
     -> data/synthetic/nano_dds_factorial.tsv
            |
            v
 (3) AI SURROGATES + RSM +   scripts/13_nano_dds/optimize_nano_dds.py
     BAYESIAN OPTIMIZATION
     RF / GB / MLP surrogates ; Box-Behnken quadratic RSM ;
     GP + Expected-Improvement search of a desirability
     -> results/nano_dds_optimization.json (recommended formulation)
            |
            v
 (4) STABILITY + VALIDATION  scripts/13_nano_dds/validate_and_report.py
     ICH Q1A accelerated stability (40 C/75% RH, Arrhenius) ;
     validation vs published antimalarial-NP values (citations)
     -> results/nano_dds_stability.json, results/nano_dds_validation.json
            |
            v
 (5) PRECISION-MEDICINE HOOK scripts/13_nano_dds/predict_patient_nano_dds.py
     patient host/parasite omics -> personalized dose + release target +
     recommended NP formulation (links to src/models/pharmacokinetics.py and
     src/models/pharmacomics.py)
```

## Factors (3x3 full factorial, coded -1/0/+1)

| Factor | Meaning                         | Low (-1) | Centre (0) | High (+1) |
|--------|---------------------------------|----------|------------|-----------|
| A      | polymer:drug ratio (w/w)        | 2:1      | 5:1        | 10:1      |
| B      | stabilizer concentration (%w/v) | 0.5      | 1.0        | 2.0       |
| C      | homogenization energy (rel.)    | 0.30     | 0.60       | 1.00      |

## Responses (characterization battery)

Particle size (nm), polydispersity index (PDI), zeta potential (mV), entrapment
efficiency (EE %), drug loading (DL %), cumulative release (%), release
mechanism (Korsmeyer-Peppas exponent n), liver-targeting index and predicted
liver/plasma exposure ratio, and ICH accelerated-stability shelf-life.

## Polymer / excipient library (FDA-approved or GRAS)

PLGA 50:50, PLGA 75:25, PLA (PDLLA), PCL, chitosan, Eudragit RS100 / RL100 /
S100, alginate, ethylcellulose, HPMC; stabilizers PVA, poloxamer 188, Tween 80,
lecithin.

## Release-kinetics models fitted

Zero-order, first-order, Higuchi, Korsmeyer-Peppas (exponent n; sphere
thresholds: n<=0.43 Fickian diffusion, 0.43<n<0.85 anomalous, n>=0.85 case-II),
and Hixson-Crowell; best model chosen by Akaike information criterion (AIC).

All quantities are in-silico predictions from a documented, literature-anchored
digital twin, produced for AI method development -- not a replacement for
laboratory characterization.
