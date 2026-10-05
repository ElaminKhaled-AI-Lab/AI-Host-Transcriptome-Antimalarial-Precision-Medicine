# Dual Host-Parasite Multi-Omics and Artificial Intelligence for Precision Pediatric Antimalarial Therapy and Oral Liquid Formulation

**Master Project Blueprint, Data Plan, Computational Architecture, Synthetic Cohort Strategy, Integration Benchmark, Novel Method Development, Precision Pharmaceutics Module, Validation Plan, and Step-by-Step Execution Roadmap**

**Project root on Linux**

```text
/media/khaled/Flexbay_HDD_TOSHIBA_6TB/Malaria_Project
```

**Document status:** Working master blueprint  
**Prepared:** August 2026  
**Primary disease:** *Plasmodium falciparum* malaria in children  
**Primary application:** Precision antimalarial treatment prioritization linked to pediatric oral formulation development  
**Primary computational paradigm:** Dual host-parasite multi-omics integration with machine learning and deep learning  
**Primary development dataset strategy:** Real-data anchored semi-synthetic cohort with independent real-data validation

---

# 1. Executive Summary

This project aims to develop a precision pediatric antimalarial research platform that integrates molecular information from both the child and the infecting *Plasmodium falciparum* parasite, together with clinical, pharmacokinetic, treatment, and formulation variables.

The central scientific hypothesis is that antimalarial treatment response cannot be fully explained by parasite resistance markers, host biology, dose, or formulation in isolation. A more complete prediction should integrate:

1. host genomics;
2. host transcriptomics;
3. host proteomics;
4. host metabolomics;
5. parasite genomics;
6. parasite transcriptomics;
7. clinical characteristics;
8. pharmacokinetics;
9. treatment identity and dose;
10. pharmaceutical formulation properties.

The project is designed around a modular architecture because no single currently identified public pediatric malaria cohort contains every required layer in the same child.

The project therefore has two complementary goals.

## Goal A. Computational precision antimalarial therapy

Develop and benchmark models that estimate, for each patient and candidate antimalarial treatment:

- probability of treatment success;
- probability of parasite resistance;
- parasite-clearance phenotype;
- probability of recrudescence;
- expected pharmacokinetic exposure;
- toxicity or adverse-event risk;
- confidence and uncertainty of each prediction.

## Goal B. Precision pediatric pharmaceutics

Develop a separate but connected Artificial Intelligence (AI) model that optimizes an age-appropriate oral dosage form for the selected antimalarial treatment, focusing on:

- suspension stability;
- redispersibility;
- sedimentation;
- rheology;
- dissolution;
- dose uniformity;
- chemical stability;
- palatability;
- pediatric excipient suitability;
- predicted drug exposure.

The long-term system is:

```text
Child biology
+
Parasite biology
+
Clinical state
+
Candidate treatment
+
Dose
+
Formulation
        |
        v
Integrated AI model
        |
        +--> Resistance probability
        +--> Pharmacokinetic exposure
        +--> Treatment success
        +--> Recrudescence
        +--> Toxicity
        +--> Prediction uncertainty
        |
        v
Patient-specific treatment prioritization
        |
        v
Pediatric formulation optimization
        |
        v
Experimental validation
```

This must initially be described as a **research decision-support and treatment-prioritization framework**, not as a clinical prescribing system.

---

# 2. Definitions and Abbreviations

- **ACT:** Artemisinin-based Combination Therapy
- **AI:** Artificial Intelligence
- **API:** Active Pharmaceutical Ingredient
- **AUC:** Area Under the concentration-time Curve
- **AUROC:** Area Under the Receiver Operating Characteristic curve
- **AUPRC:** Area Under the Precision-Recall Curve
- **CATE:** Conditional Average Treatment Effect
- **CI:** Confidence Interval
- **CNV:** Copy-Number Variation
- **CRAM:** Compressed Reference-oriented Alignment Map
- **CV:** Cross-Validation
- **DIA:** Data-Independent Acquisition
- **DIABLO:** Data Integration Analysis for Biomarker discovery using Latent variable approaches for Omics studies
- **DL:** Deep Learning
- **DoE:** Design of Experiments
- **DRC:** Democratic Republic of the Congo
- **DUA:** Data Use Agreement
- **EGA:** European Genome-phenome Archive
- **ENA:** European Nucleotide Archive
- **FDA:** United States Food and Drug Administration
- **FDR:** False Discovery Rate
- **G6PD:** Glucose-6-Phosphate Dehydrogenase
- **GCN:** Graph Convolutional Network
- **GEO:** Gene Expression Omnibus
- **GPU:** Graphics Processing Unit
- **HDD:** Hard Disk Drive
- **IDDO:** Infectious Diseases Data Observatory
- **IID:** Inactive Ingredient Database
- **IPD:** Individual Participant Data
- **LD:** Linkage Disequilibrium
- **MAE:** Mean Absolute Error
- **MAR:** Missing At Random
- **MCAR:** Missing Completely At Random
- **MCC:** Matthews Correlation Coefficient
- **ML:** Machine Learning
- **MNAR:** Missing Not At Random
- **MOFA+:** Multi-Omics Factor Analysis Plus
- **MOGONET:** Multi-Omics Graph cOnvolutional NETworks
- **MS:** Mass Spectrometry
- **NCBI:** National Center for Biotechnology Information
- **PCA:** Principal Component Analysis
- **PCR:** Polymerase Chain Reaction
- **PD:** Pharmacodynamics
- **PK:** Pharmacokinetics
- **PRIDE:** PRoteomics IDEntifications Database
- **QC:** Quality Control
- **RNA:** Ribonucleic Acid
- **RNA-seq:** Ribonucleic Acid Sequencing
- **RMSE:** Root Mean Squared Error
- **SHAP:** SHapley Additive exPlanations
- **SNP:** Single-Nucleotide Polymorphism
- **SNF:** Similarity Network Fusion
- **SRA:** Sequence Read Archive
- **SSD:** Solid-State Drive
- **TSV:** Tab-Separated Values
- **VAE:** Variational Autoencoder
- **VCF:** Variant Call Format
- **VCDN:** View Correlation Discovery Network
- **WGS:** Whole-Genome Sequencing
- **WHO:** World Health Organization
- **WWARN:** Worldwide Antimalarial Resistance Network
- **XGBoost:** Extreme Gradient Boosting

---

# 3. Scientific Problem

Current antimalarial treatment decisions are mostly based on:

- parasite species;
- disease severity;
- patient age or weight;
- pregnancy status;
- national treatment guidelines;
- known regional resistance patterns;
- prior treatment history.

These factors are essential, but they do not completely represent the molecular heterogeneity of either the infected child or the parasite.

Two children receiving the same antimalarial combination and nominal weight-based dose may differ in:

- parasite resistance genotype;
- parasite transcriptional state;
- parasite stage distribution;
- host inflammatory state;
- host metabolic state;
- host pharmacogenetic variants;
- nutritional status;
- absorption;
- drug exposure;
- adverse-event susceptibility.

The project therefore treats malaria treatment response as a joint host-pathogen-pharmacology problem.

The research framework is:

\[
P(Y \mid H, P, C, D, F)
\]

where:

- \(Y\) is treatment outcome;
- \(H\) is host molecular state;
- \(P\) is parasite molecular state;
- \(C\) is clinical state;
- \(D\) is drug and dose;
- \(F\) is formulation.

---

# 4. Primary Project Objective

Develop, benchmark, and validate a multi-modal AI framework that integrates pediatric host multi-omics, parasite multi-omics, clinical variables, pharmacokinetics, antimalarial treatment, and formulation properties to predict treatment response and support rational pediatric antimalarial formulation design.

---

# 5. Specific Objectives

## Objective 1. Build a curated pediatric malaria data resource

Create a harmonized resource containing real data from public and managed-access sources, including:

- host transcriptomics;
- host proteomics;
- host metabolomics;
- host genomics;
- parasite genomics;
- parasite transcriptomics;
- clinical variables;
- pharmacokinetic data;
- treatment identity;
- dose;
- treatment outcome.

## Objective 2. Characterize single-omic biology

Analyze each modality independently before integration.

Examples:

- host differential gene expression;
- host pathway activity;
- proteomic inflammatory signatures;
- metabolomic pathway changes;
- parasite resistance markers;
- parasite transcriptional resistance states;
- host-parasite molecular relationships.

## Objective 3. Construct a real-data anchored semi-synthetic cohort

Create synthetic pediatric records while preserving empirically observed:

- marginal distributions;
- cross-feature covariance;
- cross-omic covariance;
- host-parasite dependence;
- geography;
- clinical severity;
- resistance-marker combinations;
- treatment distributions;
- missing-data patterns.

## Objective 4. Benchmark established multi-omics integration methods

Benchmark methods representing distinct methodological families:

- concatenation;
- Principal Component Analysis;
- Multi-Omics Factor Analysis Plus;
- Similarity Network Fusion;
- Data Integration Analysis for Biomarker discovery using Latent variable approaches for Omics studies;
- autoencoders;
- variational autoencoders;
- Multi-Omics Graph cOnvolutional NETworks;
- late-fusion ensembles.

## Objective 5. Develop a malaria-specific integration model

Develop a proposed new method, provisionally named:

**HPAMM-Net: Host-Parasite Adaptive Missingness-aware Multi-task Network**

This is a working research name, not a claim that the name or architecture is globally novel. A formal literature and software novelty search must be completed before publication or naming the method permanently.

## Objective 6. Predict treatment-specific outcomes

For each patient and treatment, predict:

- resistance probability;
- parasite-clearance phenotype;
- cure probability;
- recrudescence probability;
- pharmacokinetic exposure;
- adverse-event probability;
- uncertainty.

## Objective 7. Develop pediatric formulation AI

Generate experimental formulation data and model:

- formulation stability;
- dose uniformity;
- dissolution;
- redispersibility;
- rheology;
- sedimentation;
- chemical degradation;
- palatability;
- exposure-related formulation performance.

## Objective 8. Integrate treatment selection with formulation optimization

The final research system should connect:

```text
patient molecular state
+
parasite molecular state
+
clinical state
        |
        v
treatment-specific prediction
        |
        v
candidate antimalarial combination
        |
        v
dose and formulation optimization
```

---

# 6. Key Scientific Hypotheses

1. Dual host-parasite models will predict clinically relevant malaria phenotypes more accurately than host-only or parasite-only models.
2. Parasite genomics and transcriptomics will explain resistance-related outcomes, while host molecular data will provide complementary information on severity, metabolism, exposure, and toxicity.
3. A missingness-aware multi-modal model will remain more stable than complete-case integration when one or more omics layers are unavailable.
4. A multi-task model trained simultaneously on resistance, pharmacokinetics, clearance, recurrence, and treatment response will learn a more biologically coherent latent representation than a single binary outcome model.
5. Patient-specific treatment ranking will improve when both parasite resistance and host pharmacology are represented.
6. A formulation model using pharmaceutical composition, material properties, processing conditions, and dissolution can optimize pediatric oral formulations more efficiently than one-factor-at-a-time formulation development.

---

# 7. Critical Scientific Boundaries

## 7.1 Synthetic patients are not clinical evidence

Synthetic cohorts are useful for:

- architecture development;
- benchmarking;
- controlled missingness experiments;
- sensitivity analysis;
- stress testing;
- power exploration.

They do not create the evidential value of real patients.

A model that performs well on synthetic data must still be tested on real data.

## 7.2 Unmatched cohorts must not be falsely joined at patient level

If Patient A provides transcriptomics in one study and Patient B provides genomics in another study, those records cannot be joined and described as one real patient.

Instead, different real cohorts can be used to estimate separate components of a generative model.

## 7.3 Treatment ranking is not automatically causal

Observational models may learn treatment-selection bias.

A model output such as:

```text
Treatment A predicted response = 0.85
Treatment B predicted response = 0.71
```

does not prove that Treatment A would cause a better outcome for that same child.

Causal treatment-effect modeling requires stronger assumptions, sufficient treatment overlap, and careful trial design.

## 7.4 A syrup may not be the correct formulation

Many antimalarial active pharmaceutical ingredients have poor aqueous solubility. A true syrup requires substantial drug solubility.

A suspension, granules for oral suspension, or dispersible dosage form may be pharmaceutically more realistic.

The formulation-development stage must select the dosage form from drug properties, not from a predetermined preference for the word "syrup."

## 7.5 Existing pediatric formulations must be respected

The World Health Organization already lists pediatric antimalarial formulations, including dispersible formulations and granules for oral suspension.

The project must demonstrate why a new liquid or suspension provides a measurable advantage.

Possible advantages include:

- flexible dosing in very small children;
- improved dose precision;
- improved acceptability;
- lower administration volume;
- improved exposure;
- new antimalarial compound;
- improved stability;
- improved administration in children unable to use existing formulations.

---

# 8. Overall Project Architecture

```text
                         PUBLIC AND MANAGED-ACCESS DATA
                                      |
             +------------------------+-----------------------+
             |                        |                       |
             v                        v                       v
      HOST MOLECULAR DATA      PARASITE MOLECULAR DATA   CLINICAL/PK DATA
      genome                   genome                    age
      transcriptome            transcriptome             weight
      proteome                 resistance markers        parasitemia
      metabolome               population structure      severity
             |                        |                  treatment
             |                        |                  dose
             |                        |                  outcome
             +------------------------+-----------------------+
                                      |
                                      v
                             DATA HARMONIZATION
                                      |
                                      v
                        REAL COHORT REFERENCE MODELS
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
                  REAL-DATA ANALYSIS        SEMI-SYNTHETIC ENGINE
                                                    |
                                                    v
                                     MULTI-OMICS BENCHMARK COHORTS
                                                    |
                    +-------------------------------+---------------------------+
                    |             |              |              |              |
                    v             v              v              v              v
               MOFA+            SNF           DIABLO       MOGONET       HPAMM-Net
                    |             |              |              |              |
                    +-------------+--------------+--------------+--------------+
                                                    |
                                                    v
                                           MULTI-TASK OUTPUTS
                                                    |
                       +----------------------------+-------------------------+
                       |             |             |             |           |
                       v             v             v             v           v
                 resistance     clearance       PK exposure   toxicity   cure/recurrence
                       |             |             |             |           |
                       +-------------+-------------+-------------+-----------+
                                                    |
                                                    v
                                    TREATMENT-SPECIFIC PRIORITIZATION
                                                    |
                                                    v
                                      FORMULATION OPTIMIZATION MODULE
                                                    |
                                                    v
                                          EXPERIMENTAL VALIDATION
                                                    |
                                                    v
                                         REAL-COHORT VALIDATION
```

---

# 9. Major Work Packages

1. Infrastructure and reproducibility.
2. Data acquisition.
3. Sample and metadata audit.
4. Single-omics preprocessing.
5. Biological annotation.
6. Matched-cohort integration.
7. Semi-synthetic cohort generation.
8. Integration benchmarking.
9. New-method development.
10. Treatment-response modeling.
11. Precision pharmaceutics.
12. External validation.

---

# 10. Required Data Layers

## 10.1 Host genomics

Purpose:

- inherited malaria susceptibility;
- red-cell biology;
- pharmacogenetic variation;
- metabolic capacity;
- toxicity susceptibility.

Candidate features:

- genome-wide single-nucleotide polymorphisms;
- principal components of ancestry;
- known hemoglobin variants;
- glucose-6-phosphate dehydrogenase variants;
- drug-metabolizing enzyme variants;
- transporter variants.

Priority: high, but complete human genotype data may require managed access.

## 10.2 Host transcriptomics

Purpose:

- current immune state;
- disease severity;
- inflammation;
- erythropoiesis;
- endothelial response;
- metabolic response.

Priority: essential.

## 10.3 Host proteomics

Purpose:

- circulating protein abundance;
- complement;
- coagulation;
- inflammation;
- endothelial activation;
- acute-phase proteins.

Priority: essential for the core host reference cohort.

## 10.4 Host metabolomics

Purpose:

- energy metabolism;
- amino-acid metabolism;
- lipid metabolism;
- oxidative stress;
- downstream host phenotype.

Priority: essential for the core host reference cohort.

## 10.5 Parasite genomics

Purpose:

- antimalarial-resistance variants;
- population structure;
- copy-number variation;
- genome-wide diversity;
- within-host infection complexity.

Priority: essential.

## 10.6 Parasite transcriptomics

Purpose:

- functional parasite state;
- developmental stage;
- stress-response pathways;
- resistance-associated transcription.

Priority: essential.

## 10.7 Clinical data

Required variables, where available:

- age;
- weight;
- sex;
- country;
- site;
- parasitemia;
- temperature;
- hemoglobin;
- disease severity;
- nutritional status;
- previous antimalarial use;
- previous malaria episodes;
- vomiting;
- liver function;
- renal function;
- concomitant medication.

Priority: mandatory.

## 10.8 Pharmacokinetic data

Potential variables:

- drug;
- dose;
- dose per kilogram;
- sampling time;
- plasma concentration;
- maximum concentration;
- trough concentration;
- area under the concentration-time curve;
- day-7 concentration where relevant.

Priority: mandatory for the exposure module.

## 10.9 Outcome data

Potential outcomes:

- parasite-clearance time;
- parasite-clearance half-life;
- day-3 parasitemia;
- day-28 cure;
- day-42 cure;
- extended follow-up where available;
- polymerase chain reaction corrected cure;
- recrudescence;
- reinfection;
- treatment failure;
- anemia;
- adverse events.

Priority: mandatory for supervised treatment modeling.

---

# 11. Verified Core Data Sources

| Resource | Accession or Study | Main Data | Pediatric | Raw Data | Access | Primary Role |
|---|---|---|---|---|---|---|
| Gene Expression Omnibus | GSE289197 | Host transcriptomics | Yes | Patient FASTQ not open | Open processed/count-level data | Host expression reference |
| PRIDE | PXD058621 | Host proteomics | Yes | Yes, raw mass spectrometry submitted | Open | Host proteome |
| MassIVE | MSV000096913 | Host metabolomics | Yes | Yes | Public repository | Host metabolome |
| Sequence Read Archive | PRJNA962942 | Host-parasite dual RNA sequencing | Yes | Yes | Open | Host-parasite transcriptome |
| BioStudies/ENA | E-MTAB-6413 / ERP116722 / PRJEB33892 | Host-parasite dual RNA sequencing | Yes | Yes | Open | External dual-transcriptome validation |
| MalariaGEN | Pf8 | Parasite genomics | Mixed, includes pediatric cohorts | Yes | Open | Parasite genome and resistance |
| MalariaGEN | Host-parasite interaction resource | Host and parasite genetics | Yes | Parasite open; full human managed | Mixed | Matched host-parasite genetics |
| MalariaGEN | Study 1167 | Parasite genome plus pediatric randomized treatment context | Yes | Parasite genome open | Open genomic data, clinical linkage must be audited | Treatment-response candidate |
| IDDO/WWARN | Malaria platform | Clinical, treatment, pharmacokinetic, outcome IPD | Yes and adults | Participant-level data through request | Managed | Clinical and PK labels |
| FDA | Inactive Ingredient Database | Excipients | Not patient data | Database download | Open | Formulation design |
| WHO | Malaria guideline and prequalification resources | Treatment and dosage-form context | Yes | Not omics | Open | Clinical and formulation benchmark |

---

# 12. Dataset 1: Mali Host Transcriptome + Proteome + Metabolome

Associated study:

**A shared inflammatory signature across severe malaria syndromes manifested by transcriptomic, proteomic and metabolomic analyses**

Publication:

https://www.nature.com/articles/s41467-025-59281-5

Open full text:

https://pmc.ncbi.nlm.nih.gov/articles/PMC12086225/

This study provides a real pediatric host multi-omics reference.

## 12.1 Transcriptomics

Repository: Gene Expression Omnibus.

Accession:

```text
GSE289197
```

Website:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289197

Important availability point:

The repository explicitly states that raw patient sequence files were not submitted for open access because of concerns about identifiable human sequence data.

Therefore:

```text
Open:
- sample metadata
- processed/count-level expression information

Not openly available:
- original patient FASTQ sequence files
```

Use this dataset for expression-level integration, not for claiming complete raw-read reprocessing.

## 12.2 Proteomics

Repository: PRoteomics IDEntifications Database.

Accession:

```text
PXD058621
```

Website:

https://www.ebi.ac.uk/pride/archive/projects/PXD058621

Recommended project directory:

```text
data/raw/proteomics/PXD058621/
```

Keep:

- original raw mass-spectrometry files;
- open spectral formats if submitted;
- processed protein matrices;
- quantification output;
- metadata;
- sample mapping.

Do not delete raw files after obtaining processed data.

## 12.3 Metabolomics

Repository: MassIVE.

Accession:

```text
MSV000096913
```

Dataset page:

https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=ba8df9f0096c4bf6abb946f323f153c1

File browser:

https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=ba8df9f0096c4bf6abb946f323f153c1

Recommended project directory:

```text
data/raw/metabolomics/MSV000096913/
```

---

# 13. Dataset 2: Mali Dual Host-Parasite Transcriptomics

Repository: National Center for Biotechnology Information Sequence Read Archive.

BioProject:

```text
PRJNA962942
```

Website:

https://www.ncbi.nlm.nih.gov/bioproject/PRJNA962942

Associated open article:

https://pmc.ncbi.nlm.nih.gov/articles/PMC10918175/

The sequencing data were deposited under PRJNA962942.

Primary value:

Reads originate from infected pediatric blood and can be assigned to both:

- human;
- *Plasmodium falciparum*.

This allows estimation of:

\[
HostRNA_i \leftrightarrow ParasiteRNA_i
\]

for the same biological sample.

Primary roles:

- host-parasite interaction modeling;
- age-related host response;
- parasite expression state;
- cross-species transcriptomic dependence;
- semi-synthetic cohort generation.

---

# 14. Dataset 3: Gambian Dual Host-Parasite Transcriptomics

BioStudies accession:

```text
E-MTAB-6413
```

Website:

https://www.ebi.ac.uk/biostudies/studies/E-MTAB-6413

European Nucleotide Archive study:

```text
ERP116722
```

Equivalent project accession:

```text
PRJEB33892
```

Example sequencing record:

https://www.ncbi.nlm.nih.gov/sra/ERX3492172

The project is described as dual RNA sequencing of peripheral blood from Gambian children with severe or uncomplicated *Plasmodium falciparum* malaria.

Primary role:

```text
Develop relationships in Mali
        |
        v
Test transportability in The Gambia
```

Do not pool Mali and The Gambia before checking:

- sequencing protocol;
- disease definition;
- age distribution;
- batch;
- parasite abundance;
- geography;
- severity.

---

# 15. Dataset 4: MalariaGEN Pf8 Parasite Genomics

Official resource:

https://www.malariagen.net/data_package/open-dataset-plasmodium-falciparum-v80/

Interactive study browser:

https://apps.malariagen.net/apps/pf8/studies/

Interactive sample browser:

https://apps.malariagen.net/apps/pf8/samples

Documentation:

https://apps.malariagen.net/apps/pf8/documentation/

The official Pf8 release contains genome variation data for **33,325 *Plasmodium falciparum* samples**.

The release includes:

- study information;
- sample provenance;
- European Nucleotide Archive accessions;
- quality-control information;
- copy-number variation calls;
- complexity-of-infection measures;
- known drug-resistance marker genotypes;
- inferred resistance classifications;
- haplotypes;
- reference genome;
- annotation;
- genetic-distance matrices;
- short-variant genotypes;
- compressed reference-oriented alignment files.

Known resistance-marker loci include:

- `crt`;
- `dhfr`;
- `dhps`;
- `mdr1`;
- `kelch13`;
- `plasmepsin 2-3`.

## Do not initially download all raw sequence data

Start with:

1. metadata;
2. resistance-marker genotypes;
3. inferred resistance classifications;
4. copy-number variation;
5. quality-control variables;
6. analysis-ready genotype resources;
7. selected treatment cohorts.

Raw compressed alignment files should be downloaded only when required.

---

# 16. MalariaGEN Study 1167: High-Priority Pediatric Treatment Cohort

Website:

https://apps.malariagen.net/apps/pf8/studies/1167/

Study description:

**Surveillance of parasite populations and patterns of drug resistance and associated parasite clearance or treatment failure in Tanzania**

The official Pf8 study page reports:

- children aged 6 months to 10 years;
- uncomplicated falciparum malaria;
- randomized artemether-lumefantrine or dihydroartemisinin-piperaquine treatment;
- parasite-clearance monitoring;
- treatment follow-up;
- parasite whole-genome sequencing;
- 347 Pf8 genomic samples listed on the study page.

This is extremely important because it connects:

```text
parasite genome
+
antimalarial treatment
+
treatment outcome
```

Before modeling, establish exactly which clinical variables are downloadable through MalariaGEN and which require retrieval from the original study or IDDO/WWARN.

Do not assume the Pf8 genomic sample table contains the complete trial case-report form.

---

# 17. MalariaGEN Study 1101: Candidate Multi-Treatment Cohort

Study description identified in Pf8:

**Efficacy of three ACTs in treating falciparum malaria in the Democratic Republic of Congo**

Study identifier:

```text
1101
```

Search from:

https://apps.malariagen.net/apps/pf8/studies/

Potential value:

A multi-treatment study can support treatment-specific modeling better than a one-treatment cohort.

Before use, verify:

- age distribution;
- treatment arms;
- genomic sample count;
- outcome availability;
- patient-level data-linkage permissions.

---

# 18. Matched Host-Parasite Genetics Resource

Official page:

https://www.malariagen.net/data_package/host-parasite-interaction/

The official resource reports data from more than 4,000 children with severe malaria from Banjul, The Gambia, and Kilifi County, Kenya.

## Parasite genome sequencing

Access: open.

Sample-accession list:

https://doi.org/10.5281/zenodo.5723515

## Full human genotypes

Access: managed.

MalariaGEN human-genetics resource:

https://www.malariagen.net/resource/25/

## Joint host-parasite genotype dataset

Access: open.

https://doi.org/10.5281/zenodo.4973476

## Association summary statistics

Access: open.

https://doi.org/10.5281/zenodo.5722497

Primary role:

\[
HostGenome_i \leftrightarrow ParasiteGenome_i
\]

Use this resource to estimate:

- host-parasite genotype associations;
- population structure;
- cross-species genetic dependence;
- simulation constraints.

---

# 19. IDDO/WWARN Clinical and Pharmacokinetic Data

Malaria data page:

https://www.iddo.org/wwarn/access-malaria-data

Access instructions:

https://www.iddo.org/data-sharing/accessing-data

The platform reports hundreds of thousands of malaria Individual Participant Data records.

Access is managed.

The official access process includes:

1. explore data;
2. identify study identifiers and variables;
3. review access criteria;
4. define research plan;
5. submit application;
6. undergo review;
7. sign a Data Use Agreement;
8. receive approved data.

## Variables to request

### Demographics

- age;
- sex;
- weight;
- site;
- country.

### Disease

- parasitemia;
- temperature;
- hemoglobin;
- disease severity;
- nutritional measurements.

### Treatment

- drug combination;
- dose;
- dose per kilogram;
- formulation if available;
- vomiting and repeat dose;
- food administration if relevant.

### Pharmacokinetics

- sampling time;
- drug concentration;
- area under the concentration-time curve;
- day-7 concentration where available.

### Outcomes

- parasite-clearance measures;
- day-3 positivity;
- day-28 response;
- day-42 response;
- extended follow-up;
- polymerase chain reaction corrected outcome;
- recrudescence;
- reinfection;
- adverse events.

Important limitation:

Not every study contains every variable. The IDDO access page explicitly warns that variables listed in the data dictionary may not have been collected or shared for every study.

---

# 20. Formulation and Pharmaceutical Reference Resources

## 20.1 FDA Inactive Ingredient Database

https://www.fda.gov/drugs/drug-approvals-and-databases/inactive-ingredients-database-download

Use to identify inactive ingredients with precedent in approved dosage forms.

Do not interpret database presence as proof of safety for every pediatric age, route, concentration, or exposure.

## 20.2 WHO malaria guidance

Global Malaria Programme:

https://www.who.int/teams/global-malaria-programme

Prequalification resources:

https://extranet.who.int/prequal/medicines/prequalified-lists

The project should benchmark any proposed formulation against existing pediatric antimalarial products.

## 20.3 PubChem

https://pubchem.ncbi.nlm.nih.gov/

Use for:

- chemical structures;
- formula;
- molecular weight;
- identifiers;
- physicochemical information.

## 20.4 ChEMBL

https://www.ebi.ac.uk/chembl/

Use for:

- antimalarial bioactivity;
- target information;
- compound structures;
- potency measurements.

---

# 21. Data Acquisition Order

## Stage 1. Host multi-omics

Download:

```text
GSE289197
PXD058621
MSV000096913
```

Goal: build the host transcriptome-proteome-metabolome reference.

## Stage 2. Dual host-parasite transcriptomics

Download:

```text
PRJNA962942
E-MTAB-6413 / ERP116722 / PRJEB33892
```

Goal: model real host-parasite transcriptomic dependence.

## Stage 3. Parasite genomics

Download first:

- Pf8 metadata;
- resistance markers;
- inferred resistance labels;
- copy-number variation;
- quality-control data;
- selected treatment cohorts.

## Stage 4. Matched host-parasite genetics

Download open joint data.

Apply for managed human-genotype data if scientifically justified.

## Stage 5. Clinical and pharmacokinetic data

Prepare IDDO/WWARN application.

## Stage 6. Formulation references

Download excipient and physicochemical reference data.

## Stage 7. Experimental formulation data

Generate experimentally because a sufficiently large open antimalarial suspension formulation dataset has not yet been identified.

---

# 22. Linux Project Architecture

Recommended root:

```text
/media/khaled/Flexbay_HDD_TOSHIBA_6TB/Malaria_Project
```

Recommended structure:

```text
Malaria_Project/
|
+-- README.md
+-- MASTER_PROJECT_BLUEPRINT.md
|
+-- config/
|   +-- project.yaml
|   +-- datasets.yaml
|   +-- models.yaml
|   +-- seeds.yaml
|
+-- data/
|   +-- raw/
|   |   +-- host_transcriptomics/
|   |   +-- host_proteomics/
|   |   +-- host_metabolomics/
|   |   +-- host_genomics/
|   |   +-- parasite_transcriptomics/
|   |   +-- parasite_genomics/
|   |   +-- clinical_pk/
|   |   +-- formulation/
|   +-- interim/
|   +-- processed/
|   +-- synthetic/
|   +-- external_validation/
|
+-- metadata/
|   +-- dataset_registry/
|   +-- sample_maps/
|   +-- checksums/
|   +-- access_notes/
|
+-- scripts/
|   +-- 00_setup/
|   +-- 01_download/
|   +-- 02_qc/
|   +-- 03_preprocess/
|   +-- 04_single_omics/
|   +-- 05_sample_mapping/
|   +-- 06_synthetic_cohort/
|   +-- 07_integration_benchmarks/
|   +-- 08_hpamm_net/
|   +-- 09_treatment_models/
|   +-- 10_formulation/
|   +-- 11_validation/
|   +-- 12_figures/
|
+-- src/
|   +-- data/
|   +-- models/
|   +-- simulation/
|   +-- metrics/
|   +-- utils/
|
+-- notebooks/
+-- models/
+-- results/
+-- figures/
+-- tables/
+-- reports/
+-- logs/
+-- envs/
+-- tmp/
```

---

# 23. Data Provenance Rules

Every downloaded dataset should have a metadata record containing:

```text
accession
repository
download_url
download_date
file_name
file_size
checksum
license_or_access_terms
publication
raw_or_processed_status
sample_count
organism
tissue
platform
software
notes
```

Rules:

- never modify original raw files in place;
- never rename raw files without a mapping table;
- keep raw data immutable;
- derived data go to `data/interim/` or `data/processed/`;
- record checksums.

---

# 24. Sample Matching Is a Dedicated Analysis Step

Create a master sample map:

```text
study_id
repository_sample_id
biological_patient_id
specimen_id
omics
technical_replicate
disease_group
age
sex
treatment
matched_transcriptome
matched_proteome
matched_metabolome
matched_host_genome
matched_parasite_genome
matched_parasite_rna
```

For every study:

1. extract sample identifiers;
2. extract clinical labels;
3. identify technical replicates;
4. identify biological replicates;
5. identify paired cases and controls;
6. identify cross-omic overlap;
7. document ambiguous mappings;
8. exclude unresolved mappings from matched integration.

---

# 25. Host Transcriptomic Preprocessing

Because GSE289197 does not provide open patient FASTQ reads, the primary pipeline starts from count-level expression data.

Recommended steps:

1. import counts;
2. remove extremely low-expression genes;
3. map gene identifiers;
4. inspect library size;
5. detect outliers;
6. normalize counts;
7. model relevant covariates;
8. compute differential expression;
9. derive pathway scores;
10. create machine-learning matrices.

Potential representations:

- normalized counts;
- variance-stabilized expression;
- pathway scores;
- selected genes;
- latent factors.

For small cohorts, pathway-level features may be more stable than thousands of individual genes.

---

# 26. Proteomic Preprocessing

PXD058621 should be handled in two tracks.

## Track A. Raw-data archive

Preserve:

- original raw instrument files;
- open spectral formats;
- quantification files;
- metadata.

## Track B. Integration matrix

Construct:

```text
patient x protein
```

Steps:

1. identify submitted protein matrix;
2. determine quantification type;
3. identify technical replicates;
4. inspect missingness;
5. transform intensities if appropriate;
6. normalize;
7. inspect batch effects;
8. aggregate technical replicates;
9. filter low-information proteins;
10. map proteins to gene symbols and UniProt identifiers;
11. derive pathway-level features.

Do not mix technical mass-spectrometry runs with biological patients.

---

# 27. Metabolomic Preprocessing

For MSV000096913:

1. inventory raw files;
2. identify ionization mode;
3. identify chromatography mode;
4. identify quality-control samples;
5. identify blanks;
6. peak detection;
7. retention-time alignment;
8. peak grouping;
9. blank filtering;
10. drift correction if applicable;
11. metabolite annotation;
12. normalization;
13. missingness analysis;
14. batch inspection;
15. pathway mapping.

Final matrix:

```text
patient x metabolite_feature
```

Separate annotation table:

```text
feature_id
m_z
retention_time
ion_mode
annotation
confidence_level
database_id
```

Do not turn uncertain spectral features into falsely definitive metabolite names.

---

# 28. Parasite Genomic Preprocessing

## Phase A. Analysis-ready Pf8 features

Use:

- quality-control status;
- resistance-marker genotypes;
- copy-number variation;
- infection-complexity measures;
- geography;
- study identity.

## Phase B. Genome-wide variants

Apply:

- quality filtering;
- allele-frequency filtering;
- missingness filtering;
- linkage-disequilibrium pruning where appropriate;
- principal component analysis;
- population-structure analysis.

## Phase C. Representations

Known resistance features:

```text
kelch13
crt
mdr1
dhfr
dhps
plasmepsin2_3
```

Also include:

- genome-wide principal components;
- haplotype summaries;
- copy-number variation;
- complexity of infection.

Geography must be modeled because parasite allele frequencies vary by region.

---

# 29. Parasite Transcriptomic Preprocessing

For dual RNA sequencing:

1. raw-read quality control;
2. adapter trimming;
3. mapping against combined host and parasite references;
4. separation of human and parasite expression;
5. host quantification;
6. parasite quantification;
7. parasite-read fraction assessment;
8. parasite developmental-stage estimation;
9. species-specific normalization;
10. low-information gene filtering;
11. pathway and stress-response scoring.

Avoid direct comparison of absolute human and parasite counts without suitable normalization.

---

# 30. Host Genomic Preprocessing

For managed and open host-genotype resources:

1. sample quality control;
2. variant quality control;
3. relatedness assessment;
4. ancestry principal components;
5. heterozygosity checks;
6. sex checks when allowed;
7. allele-frequency filtering;
8. documented imputation where appropriate;
9. pharmacogene extraction;
10. malaria-protective variant extraction.

Population structure should be represented to reduce confounding.

---

# 31. Clinical and Pharmacokinetic Preprocessing

Standardize clinical variables by:

- units;
- time relative to treatment;
- variable names;
- categorical coding;
- missingness.

Pharmacokinetic observations require time.

Possible representations:

- concentration-time profiles;
- derived pharmacokinetic parameters;
- model-estimated exposure;
- day-7 concentrations.

The safest pretreatment-prediction design is:

```text
pre-treatment predictors
        |
        v
predict post-treatment PK and outcome
```

Do not use measured post-treatment concentrations as baseline predictors when claiming prediction before treatment.

---

# 32. Single-Omics Analyses Before Integration

## Host transcriptome

- differential expression;
- pathway enrichment;
- severity classification;
- age associations.

## Host proteome

- differential abundance;
- pathway analysis;
- transcript-protein correlation.

## Host metabolome

- differential abundance;
- pathway analysis;
- association with severity.

## Parasite genome

- known resistance markers;
- population structure;
- resistance classification.

## Parasite transcriptome

- developmental-stage signatures;
- resistance-related expression;
- host-parasite correlation.

## Host genome

- malaria-protective variants;
- pharmacogenetic candidates;
- population structure.

This stage prevents integration models from hiding basic data-quality problems.

---

# 33. Why the Development Cohort Should Be Semi-Synthetic

A completely synthetic cohort generated from independent random variables would be biologically weak.

The semi-synthetic strategy should preserve real structure.

Real cohorts provide:

```text
Host RNA + protein + metabolite covariance
Host RNA + parasite RNA covariance
Host genome + parasite genome dependence
Parasite genomic resistance structure
Clinical and PK distributions
```

The generator combines these learned structures under explicit assumptions.

The synthetic cohort is not intended to invent biology. It is intended to create a controlled benchmark when no single public cohort contains every modality.

---

# 34. Semi-Synthetic Patient Architecture

Each synthetic patient should contain:

```text
synthetic_patient_id

DEMOGRAPHICS
age
sex
weight
country
site
nutritional_state

CLINICAL
parasitemia
hemoglobin
temperature
severity

HOST_GENOME
pharmacogenetic_variants
malaria_susceptibility_variants
ancestry_components

HOST_TRANSCRIPTOME
selected_genes
pathway_scores
latent_factors

HOST_PROTEOME
selected_proteins
pathway_scores
latent_factors

HOST_METABOLOME
selected_metabolites
pathway_scores
latent_factors

PARASITE_GENOME
resistance_variants
copy_number_variation
genome_principal_components
infection_complexity

PARASITE_TRANSCRIPTOME
selected_genes
stage_scores
stress_scores
resistance_scores

TREATMENT
drug_combination
dose
dose_per_kg

PHARMACOKINETICS
predicted_or_simulated_exposure
concentration_summary

OUTCOME
clearance
cure
recrudescence
toxicity

MISSINGNESS
omics_mask
clinical_missingness_mask

PROVENANCE
generator_version
random_seed
source_models
```

---

# 35. Synthetic Data Generation Strategy

## 35.1 Demographics

Model joint distributions rather than independent variables:

\[
P(Age, Weight, Sex, Country)
\]

Age and weight should remain correlated.

## 35.2 Geography

Generate country or study context before parasite genotype.

\[
P(ParasiteGenome \mid Country)
\]

This avoids geographically unrealistic resistance combinations.

## 35.3 Parasite genome

Do not simulate each variant independently.

Preferred strategies:

1. resample real haplotypes;
2. preserve local haplotype blocks;
3. condition on geography;
4. preserve known resistance-marker combinations;
5. preserve copy-number states;
6. preserve genome principal-component structure.

## 35.4 Host multi-omics

Learn latent factors from real matched host data.

Example:

\[
Z_H = [z_{inflammation}, z_{erythropoiesis}, z_{metabolism}, z_{endothelial}, z_{severity}]
\]

Then:

\[
X_{RNA}=W_{RNA}Z_H+\epsilon_{RNA}
\]

\[
X_{protein}=W_{protein}Z_H+\epsilon_{protein}
\]

\[
X_{metabolite}=W_{metabolite}Z_H+\epsilon_{metabolite}
\]

Multi-Omics Factor Analysis Plus is a strong candidate for estimating this structure.

## 35.5 Host-parasite transcriptome dependence

Learn from dual RNA datasets:

\[
P(ParasiteRNA \mid HostRNA, Parasitemia, Age, Severity)
\]

Use empirical dependence without assuming direct causality.

## 35.6 Host-parasite genomic dependence

Use the MalariaGEN matched genetics resource to preserve observed cross-species dependence where scientifically appropriate.

## 35.7 Clinical phenotype

Simulate:

\[
P(Clinical \mid HostState, ParasiteState, Age)
\]

## 35.8 Treatment

For randomized studies, preserve randomization logic.

For observational studies, preserve study-specific treatment probabilities and record potential confounding.

## 35.9 Pharmacokinetics

Initial model:

\[
PK=f(Age, Weight, HostFactors, Drug, Dose)+\epsilon
\]

A later version should use validated population pharmacokinetic models where possible.

## 35.10 Outcomes

Generate conditionally:

\[
Outcome=f(ParasiteResistance, HostState, PK, Drug, Dose, ClinicalState)
\]

The outcome generator must be declared before benchmarking so it is not tuned to favor HPAMM-Net.

---

# 36. Synthetic Cohort Sizes

Primary development design:

```text
N = 2,000 synthetic patients per cohort
10 independent synthetic cohorts
```

Sensitivity cohorts:

```text
N = 250
N = 500
N = 1,000
N = 2,000
N = 5,000
```

Never describe 20,000 simulated records as equivalent to 20,000 observed children.

---

# 37. Missing-Data Simulation

Model three mechanisms.

## Missing Completely At Random

Missingness independent of measured and unmeasured variables.

## Missing At Random

Missingness depends on observed variables.

## Missing Not At Random

Missingness depends partly on unobserved information.

Test modality dropout:

```text
0%
10%
20%
30%
40%
50%
```

Also test structured absence:

```text
no proteome
no metabolome
no host genome
no parasite RNA
```

---

# 38. Synthetic Cohort Validation

Compare real and synthetic data using:

- univariate distributions;
- pairwise correlations;
- covariance matrices;
- principal-component structure;
- latent-factor distributions;
- pathway distributions;
- resistance-marker frequencies;
- haplotype structure;
- clinical relationships;
- host-parasite correlations;
- missingness.

Use holdout statistics not directly optimized by the generator.

Reject a generator that matches marginal distributions but destroys biologically meaningful joint structure.

---

# 39. Multi-Omics Integration Strategy

## 39.1 Early integration

Combine features before modeling.

```text
omics 1
omics 2
omics 3
        |
        v
normalization and feature control
        |
        v
concatenated feature matrix
        |
        v
predictive model
```

Strengths:

- simple;
- transparent;
- strong baseline.

Weaknesses:

- high dimensionality;
- modality imbalance;
- scale differences;
- poor missing-modality handling.

## 39.2 Intermediate integration

Learn a shared latent space.

Examples:

- MOFA+;
- DIABLO;
- autoencoder;
- VAE;
- modality-specific encoders.

## 39.3 Network integration

Example:

- SNF.

## 39.4 Graph deep learning

Example:

- MOGONET.

## 39.5 Late integration

Train one model per modality and combine predictions.

Options:

- weighted average;
- logistic stacking;
- learned gating;
- attention.

Late fusion is a particularly important comparator for missing modalities.

---

# 40. Mandatory Baseline Methods

## Baseline 1. Clinical-only

Inputs:

```text
age
weight
parasitemia
hemoglobin
severity
treatment
dose
```

Models:

- logistic regression;
- elastic net;
- XGBoost.

Question:

> Does multi-omics improve prediction beyond ordinary clinical information?

## Baseline 2. Parasite resistance rules

Use known molecular markers from Pf8.

Question:

> Does AI improve upon established resistance biology?

## Baseline 3. Single-omic models

Separate:

- host RNA;
- host proteome;
- host metabolome;
- parasite genome;
- parasite RNA.

## Baseline 4. Concatenation

After modality-specific preprocessing:

```text
all_features = concatenate(all_omics)
```

Models:

- logistic regression;
- elastic net;
- Random Forest;
- XGBoost.

---

# 41. Classical Machine-Learning Methods

## Logistic regression

Transparent baseline.

## Elastic net

Useful for sparse, high-dimensional omics.

## Random Forest

Useful for nonlinear interactions and mixed features.

## XGBoost

Strong nonlinear tabular benchmark.

Use nested cross-validation.

## Support Vector Machine

Optional nonlinear comparator.

## Gaussian process regression

Especially useful for the smaller formulation dataset.

---

# 42. MOFA+

Official documentation:

https://biofam.github.io/MOFA2/

Missing-data information:

https://biofam.github.io/MOFA2/faq.html

MOFA+ learns shared and modality-specific latent factors.

Core use:

```text
Host RNA
Host proteome
Host metabolome
        |
        v
MOFA+
        |
        v
latent factors
```

Use latent factors for:

- visualization;
- clustering;
- biological interpretation;
- synthetic generation;
- supervised prediction.

A key advantage is that MOFA+ can handle missing values in the likelihood without hidden imputation.

---

# 43. Similarity Network Fusion

Original article:

https://www.nature.com/articles/nmeth.2810

Similarity Network Fusion constructs patient-similarity networks for each modality and fuses them.

Example:

```text
RNA similarity network
+
protein similarity network
+
metabolite similarity network
        |
        v
fused patient network
```

Primary uses:

- patient stratification;
- cluster discovery;
- network integration.

---

# 44. DIABLO

Official page:

https://mixomics.org/mixdiablo/

Original article:

https://academic.oup.com/bioinformatics/article/35/17/3055/5292387

DIABLO is supervised and identifies correlated multi-omic components that discriminate the outcome.

Strengths:

- supervised integration;
- interpretable selected variables;
- cross-omic signatures.

Limitation:

Standard use requires matched samples across omics blocks.

---

# 45. Autoencoder Baseline

Architecture:

```text
input omics
        |
        v
encoder
        |
        v
latent representation
        |
        v
decoder
```

Test:

- concatenated autoencoder;
- modality-specific autoencoders.

Do not use an oversized network on a small real cohort.

---

# 46. Variational Autoencoder

A Variational Autoencoder learns a probabilistic latent distribution.

Advantages:

- probabilistic representation;
- uncertainty;
- synthetic-generation potential;
- latent regularization.

A multimodal VAE can model:

\[
q(z|x_1,x_2,\dots,x_m)
\]

Missing-modality strategies include:

- product-of-experts;
- mixture-of-experts;
- explicit modality masks.

---

# 47. MOGONET

Original article:

https://www.nature.com/articles/s41467-021-23774-w

MOGONET combines:

- omics-specific Graph Convolutional Networks;
- a View Correlation Discovery Network.

It should be a strong supervised deep-learning benchmark.

Primary question:

> Does the malaria-specific proposed model outperform a strong graph-based supervised integrator?

---

# 48. Proposed New Method: HPAMM-Net

Working full name:

**Host-Parasite Adaptive Missingness-aware Multi-task Network**

This name is provisional.

The method should only be called novel after:

1. systematic literature review;
2. GitHub search;
3. bioRxiv search;
4. arXiv search;
5. benchmarking against relevant methods;
6. formal ablation experiments.

---

# 49. HPAMM-Net Core Architecture

```text
HOST GENOME ----------> Host Genome Encoder -----------+
HOST RNA -------------> Host RNA Encoder --------------+
HOST PROTEOME --------> Host Protein Encoder ----------+
HOST METABOLOME ------> Host Metabolite Encoder -------+
                                                        |
PARASITE GENOME ------> Parasite Genome Encoder -------+--> Gated Fusion
PARASITE RNA ---------> Parasite RNA Encoder ----------+        |
                                                        |        v
CLINICAL -------------> Clinical Encoder ---------------+   Integrated Z
                                                                 |
DRUG + DOSE ----------> Treatment Encoder -----------------------+
                                                                 |
                                                         +-------+--------+
                                                         |       |        |
                                                         v       v        v
                                                    resistance   PK   toxicity
                                                         |       |        |
                                                         +-------+--------+
                                                                 |
                                                                 v
                                                        treatment response
```

---

# 50. Modality-Specific Encoders

Use:

\[
z_m=E_m(x_m)
\]

where:

- \(x_m\) is modality \(m\);
- \(E_m\) is its encoder;
- \(z_m\) is its latent representation.

Suggested latent size:

```text
32 to 128 dimensions per modality
```

Tune rather than assuming larger is better.

---

# 51. Host and Parasite Separation

Model:

\[
Z_H=f(HostGenome,HostRNA,Protein,Metabolite)
\]

and:

\[
Z_P=g(ParasiteGenome,ParasiteRNA)
\]

Then:

\[
Z_{HP}=h(Z_H,Z_P)
\]

This explicitly represents malaria as a dual-organism system.

---

# 52. Missingness-Aware Fusion

Patient modality mask:

\[
M_i=[m_1,m_2,\dots,m_K]
\]

where:

```text
1 = modality available
0 = modality missing
```

Integrated representation:

\[
Z_i=\sum_m \alpha_{im}z_{im}
\]

Fusion weights should depend on:

- modality representation;
- availability mask;
- modality quality;
- patient context.

Missing modalities should contribute zero rather than arbitrary values unless imputation is a deliberate benchmark.

---

# 53. Modality Dropout During Training

Example:

```text
Patient 1: all omics
Patient 2: no proteome
Patient 3: no metabolome
Patient 4: host RNA + parasite genome
Patient 5: clinical + parasite genome
```

Goal:

Learn realistic robustness to incomplete profiling.

---

# 54. Treatment Conditioning

Represent treatment as:

\[
Z_D=E_D(Drug,Dose)
\]

Then model:

\[
P(Y|Z_H,Z_P,Z_C,Z_D)
\]

This allows the same research patient representation to be evaluated under multiple candidate treatment inputs.

---

# 55. Multi-Task Learning

Recommended output heads:

## Resistance

\[
P(resistance)
\]

## Parasite clearance

\[
\hat{t}_{clearance}
\]

## Pharmacokinetic exposure

\[
\widehat{AUC}
\]

## Recrudescence

\[
P(recrudescence)
\]

## Toxicity

\[
P(adverse\ event)
\]

## Cure

\[
P(cure)
\]

Joint loss:

\[
L=\lambda_1L_{resistance}+\lambda_2L_{clearance}+\lambda_3L_{PK}+\lambda_4L_{recrudescence}+\lambda_5L_{toxicity}+\lambda_6L_{cure}
\]

Loss weights should be tuned or learned, not selected solely to maximize a headline metric.

---

# 56. Biological Regularization

Potential prior knowledge:

- resistance pathways;
- pharmacogenes;
- host immune pathways;
- parasite-stage programs;
- protein-gene relationships;
- metabolite-pathway relationships.

Possible formulation:

\[
L_{total}=L_{prediction}+\lambda_bL_{biological}
\]

Biological priors must not encode the final test label.

---

# 57. Geography and Domain Adaptation

Parasite populations differ geographically.

Potential strategies:

- geography covariates;
- adversarial domain adaptation;
- domain-specific normalization;
- country-held-out evaluation;
- study-held-out evaluation.

Strong test:

```text
train on several countries
test on an unseen country
```

This is more informative than only random splitting.

---

# 58. Uncertainty Estimation

Potential methods:

- deep ensembles;
- Monte Carlo dropout;
- Bayesian output layers;
- conformal prediction.

Distinguish:

## Aleatoric uncertainty

Irreducible variation in outcome.

## Epistemic uncertainty

Uncertainty due to limited model knowledge.

High uncertainty should correlate with:

- missing modalities;
- out-of-distribution geography;
- rare resistance genotype;
- unstable bootstrap prediction.

---

# 59. Explainability

Use complementary methods.

## SHAP

For tree-based and compatible models.

## Integrated gradients

For differentiable deep networks.

## Gating weights

Interpret carefully. High gate weight does not prove causal importance.

## Feature and pathway ablation

Remove features or pathways and measure performance change.

Prefer pathway-level interpretation to uncritical ranking of thousands of individual features.

---

# 60. Treatment Prioritization Module

For patient \(i\) and candidate treatment \(d\):

\[
S_{id}=w_1P(cure)-w_2P(resistance)-w_3P(toxicity)-w_4P(recrudescence)+w_5ExposureSuitability
\]

Do not fix weights arbitrarily.

Possible alternatives:

- expert-defined sensitivity analysis;
- Pareto ranking;
- multi-objective optimization.

The primary paper should report component predictions instead of hiding everything in one opaque score.

---

# 61. Optional Causal Modeling

If sufficient multi-treatment data become available, investigate:

- propensity-score adjustment;
- inverse-probability weighting;
- doubly robust estimation;
- causal forests;
- meta-learners;
- Conditional Average Treatment Effects.

Do not make this the first stage.

Randomized treatment cohorts are particularly valuable because they reduce treatment-selection confounding.

---

# 62. Precision Pharmaceutics Module

Question:

> Once a candidate antimalarial combination and exposure target are identified, what pediatric oral formulation best achieves acceptable pharmaceutical performance?

---

# 63. Formulation Inputs

Potential variables:

- active pharmaceutical ingredient concentration;
- fixed-dose ratio;
- suspending agent;
- suspending-agent concentration;
- surfactant;
- wetting agent;
- buffer;
- pH;
- sweetener;
- flavor;
- preservative;
- vehicle;
- viscosity modifier;
- particle size;
- particle-size distribution;
- zeta potential;
- mixing speed;
- mixing time;
- milling parameters;
- storage temperature.

---

# 64. Formulation Outputs

Measure:

- appearance;
- pH;
- viscosity;
- rheology;
- sedimentation volume;
- sedimentation rate;
- caking;
- redispersibility;
- particle-size growth;
- zeta potential;
- drug content;
- content uniformity;
- delivered-dose uniformity;
- dissolution;
- chemical degradation;
- physical stability;
- microbial quality where applicable;
- palatability or validated surrogate;
- administration volume.

---

# 65. Formulation Experimental Design

Use Design of Experiments rather than one-factor-at-a-time testing.

Sequence:

## Screening design

Identify influential variables.

## Response-surface design

Characterize nonlinear effects and interactions.

## AI-guided expansion

Use model uncertainty to select informative next formulations.

Active-learning loop:

```text
initial formulations
        |
        v
experimental measurements
        |
        v
machine-learning model
        |
        v
uncertainty / acquisition function
        |
        v
next formulations
        |
        +------> repeat
```

---

# 66. Formulation Machine-Learning Models

Because the formulation dataset will likely be modest in size, begin with:

- linear and polynomial response-surface models;
- Random Forest;
- XGBoost;
- Gaussian process regression.

Deep neural networks should not automatically be the first choice.

Gaussian process regression is attractive because it provides:

- nonlinear prediction;
- uncertainty;
- compatibility with Bayesian optimization;
- good behavior with smaller experimental datasets.

---

# 67. Bayesian Optimization for Formulation

Goal:

\[
F^*=\arg\max_F[Stability,Dissolution,DoseUniformity,Palatability]
\]

subject to constraints such as:

```text
viscosity range
pH range
excipient constraints
dose-volume limit
chemical-stability threshold
```

Prefer constrained multi-objective optimization over one arbitrary composite score.

---

# 68. Link Between Omics and Formulation

The formulation module should connect to biology through pharmacokinetic exposure.

Conceptual chain:

```text
formulation
        |
        v
dissolution and delivered dose
        |
        v
absorption
        |
        v
PK exposure
        |
        v
parasite kill and toxicity
```

Therefore model:

\[
PK=f(Patient,Drug,Dose,Formulation)
\]

and:

\[
Outcome=g(Parasite,Patient,PK)
\]

This is stronger than directly claiming that a child's transcriptome determines excipient concentration.

---

# 69. Integration Benchmark Table

| Family | Method | Supervision | Main Role |
|---|---|---|---|
| Clinical baseline | Logistic regression | Supervised | Minimal comparator |
| Classical ML | Elastic net | Supervised | Sparse high-dimensional baseline |
| Classical ML | Random Forest | Supervised | Nonlinear baseline |
| Classical ML | XGBoost | Supervised | Strong tabular benchmark |
| Early integration | Concatenation + XGBoost | Supervised | Simple multi-omics benchmark |
| Latent factor | MOFA+ | Unsupervised | Shared latent biology |
| Network | SNF | Unsupervised | Patient-similarity integration |
| Supervised latent | DIABLO | Supervised | Multi-omic biomarker model |
| Deep learning | Autoencoder | Mostly unsupervised | Nonlinear latent benchmark |
| Deep learning | Multimodal VAE | Unsupervised/semi-supervised | Probabilistic latent benchmark |
| Graph deep learning | MOGONET | Supervised | Graph benchmark |
| Late fusion | Stacked modality models | Supervised | Missing-modality benchmark |
| Proposed | HPAMM-Net | Multi-task supervised | Host-parasite and missingness-aware model |

---

# 70. Cross-Validation Design

Do not rely on one random split.

## Nested cross-validation

Outer loop:

- performance estimation.

Inner loop:

- hyperparameter tuning.

## Grouped splitting

Group by:

- patient;
- study;
- site;
- country where appropriate.

Never place technical replicates from the same patient in both training and test sets.

---

# 71. External Validation Hierarchy

1. cross-validation within cohort;
2. study-held-out validation;
3. country-held-out validation;
4. independent repository or publication cohort;
5. prospective cohort.

A random 80:20 split is not the strongest evidence.

---

# 72. Primary Evaluation Metrics

## Binary classification

Report:

- Area Under the Receiver Operating Characteristic curve;
- Area Under the Precision-Recall Curve;
- sensitivity;
- specificity;
- balanced accuracy;
- F1 score;
- Matthews Correlation Coefficient;
- calibration.

## Probability calibration

Report:

- Brier score;
- calibration slope;
- calibration intercept;
- reliability plot;
- Expected Calibration Error if used.

## Regression

For pharmacokinetic exposure and clearance:

- Mean Absolute Error;
- Root Mean Squared Error;
- coefficient of determination.

## Ranking

For treatment prioritization:

- top-1 agreement where a valid reference exists;
- ranking correlation;
- regret under simulation;
- utility analysis only with explicit assumptions.

---

# 73. Statistical Comparison of Models

Use:

- repeated cross-validation;
- bootstrap confidence intervals;
- paired comparisons on identical folds;
- correction for multiple comparisons where appropriate.

Do not report only the best seed.

Report:

```text
mean
standard deviation
median
95% confidence interval
```

---

# 74. Seeds and Reproducibility

Use a predeclared seed set.

Example:

```text
11
23
47
71
101
131
167
197
233
271
```

Store in:

```text
config/seeds.yaml
```

All models should use the same data splits where possible.

---

# 75. Ablation Experiments for HPAMM-Net

Compare the full model against:

1. no host genome;
2. no host RNA;
3. no proteome;
4. no metabolome;
5. no parasite genome;
6. no parasite RNA;
7. no modality gating;
8. no modality dropout;
9. no biological regularization;
10. no multi-task learning;
11. no geography/domain component;
12. no uncertainty component.

This identifies which components actually add value.

---

# 76. Missing-Modality Benchmark

For each method, test:

```text
complete data
10% modality missingness
20%
30%
40%
50%
```

Also test structured absence:

```text
proteome entirely absent
metabolome entirely absent
host genome absent
parasite RNA absent
```

Metrics:

- prediction degradation;
- calibration degradation;
- representation stability;
- uncertainty increase.

A good model should become less confident when information disappears.

---

# 77. Synthetic-to-Real Generalization

A central experiment should be:

```text
semi-synthetic development
        |
        v
model training
        |
        v
real held-out cohort
```

Also compare:

```text
real training only
```

against:

```text
real training + semi-synthetic augmentation
```

using the same real external test set.

If synthetic augmentation does not improve real validation, do not claim biological improvement.

---

# 78. Data Leakage Prevention

Potential leakage sources:

- post-treatment variables in a pretreatment model;
- outcome embedded in resistance label;
- duplicate patients across studies;
- technical replicates across train and test;
- batch-specific identifiers;
- site variables encoding treatment arm;
- use of final test cohort in synthetic-generator fitting.

Create a leakage-audit checklist before every analysis.

---

# 79. Novelty Strategy

Potential novelty axes:

## Novelty 1

Dual host-parasite molecular integration for pediatric antimalarial treatment prediction.

## Novelty 2

Missingness-aware integration across heterogeneous malaria omics.

## Novelty 3

Treatment-conditioned multi-task prediction incorporating resistance, exposure, recurrence, and toxicity.

## Novelty 4

Cross-cohort semi-synthetic generation preserving host-host, parasite-parasite, and host-parasite biological covariance.

## Novelty 5

Country-aware external validation of precision antimalarial models.

## Novelty 6

Explicit connection between molecular treatment prioritization and pediatric formulation optimization through predicted exposure.

## Novelty 7

Active-learning formulation optimization for pediatric antimalarial suspensions, granules, or other justified oral formulations.

---

# 80. What Is Not Sufficient Novelty

These are techniques, not automatically scientific novelty:

```text
using XGBoost on malaria
using deep learning on omics
combining three omics
creating 2,000 synthetic patients
adding SHAP
adding attention
adding a Transformer
optimizing a suspension with AI
```

Novelty must be demonstrated against literature, baselines, and ablation studies.

---

# 81. Novel Method Development Order

Develop HPAMM-Net only after strong baselines.

Sequence:

```text
Step 1: clinical baseline
Step 2: parasite-marker baseline
Step 3: single-omic models
Step 4: concatenation
Step 5: MOFA+
Step 6: SNF
Step 7: DIABLO
Step 8: autoencoder
Step 9: VAE
Step 10: MOGONET
Step 11: late fusion
Step 12: HPAMM-Net
```

If a simple model performs equally well, the complex model must justify itself by:

- missingness robustness;
- calibration;
- external generalization;
- stability;
- interpretability;
- multi-task performance.

---

# 82. HPAMM-Net Development Stages

## Version 0

Clinical + parasite-marker baseline.

## Version 1

Separate host and parasite encoders.

## Version 2

Add modality gating.

## Version 3

Add modality dropout.

## Version 4

Add multi-task heads.

## Version 5

Add treatment conditioning.

## Version 6

Add biological regularization.

## Version 7

Add uncertainty estimation.

## Version 8

Add domain adaptation.

Each version must be compared with the prior version.

---

# 83. Avoiding Overfitting

Real matched cohorts are small.

Therefore:

- reduce dimensionality;
- use pathway scores;
- use sparse models;
- regularize strongly;
- avoid huge Transformers initially;
- use nested cross-validation;
- use repeated seeds;
- perform external validation.

Synthetic data must not be used to justify unnecessarily large models.

---

# 84. Pathway-Level Modeling

Potential host categories:

- innate immunity;
- interferon signaling;
- complement;
- coagulation;
- endothelial activation;
- erythropoiesis;
- oxidative stress;
- amino-acid metabolism.

Potential parasite categories:

- hemoglobin digestion;
- stress response;
- redox metabolism;
- developmental stage;
- drug-resistance pathways.

Pathway features can improve stability and interpretability in small real cohorts.

---

# 85. Data Harmonization Across Studies

Preserve study identity.

Record:

- study;
- sequencing platform;
- batch;
- country;
- collection year;
- phenotype definition.

Possible strategies:

- within-study normalization;
- study-specific standardized features for selected analyses;
- hierarchical models;
- domain adaptation;
- carefully justified batch correction.

Keep an uncorrected representation for sensitivity analysis.

---

# 86. Batch Correction Warning

Batch correction can remove true biology if:

```text
batch == disease group
```

or:

```text
batch == country
```

Inspect confounding before correction.

Do not perform aggressive correction merely because a plot looks cleaner.

---

# 87. Prospective Cohort Required for Final Clinical Claims

The ideal future cohort should collect before treatment:

- host DNA;
- whole-blood RNA;
- parasite DNA;
- parasite RNA;
- plasma or serum for metabolomics;
- plasma or serum for proteomics;
- clinical data.

After treatment:

- pharmacokinetic samples;
- parasite-clearance measurements;
- follow-up;
- recurrence;
- toxicity;
- treatment outcome.

This would allow genuine end-to-end validation.

---

# 88. Suggested Prospective Sampling Scheme

```text
Day 0 before treatment:
host DNA
whole blood RNA
parasite DNA/RNA
plasma/serum
clinical baseline

Early post-treatment:
parasitemia
PK samples

Day 3:
parasitemia

Day 7:
clinical follow-up
drug concentration if relevant

Day 28:
treatment outcome

Day 42:
treatment outcome

Extended:
where required by regimen and study protocol
```

Exact sampling must follow ethics approval, treatment guidelines, and pharmacokinetic design.

---

# 89. Ethics and Governance

For public data:

- respect repository licenses;
- cite source datasets;
- follow attribution requirements.

For managed data:

- follow Data Use Agreement;
- use only approved scientific scope;
- do not re-identify;
- do not perform unauthorized linkage;
- maintain secure storage;
- control access.

Synthetic data derived from managed datasets may remain subject to contractual restrictions.

Do not assume synthetic transformation automatically removes data-governance obligations.

---

# 90. Reproducibility Standards

Every output should record:

```text
dataset version
file checksum
software version
environment
code commit
random seed
model hyperparameters
training split
test split
date
```

Create machine-readable manifests.

---

# 91. Recommended Software Environment

## Python

Core:

- Python 3.12;
- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- matplotlib;
- PyTorch;
- XGBoost;
- Optuna;
- SHAP;
- statsmodels;
- pyarrow;
- zarr;
- malariagen_data.

## R

Core:

- R;
- Bioconductor;
- DESeq2;
- edgeR;
- limma;
- mixOmics;
- MOFA2.

Proteomics and metabolomics may require specialized tools beyond this base environment.

---

# 92. Storage Strategy

Separate:

```text
raw immutable files
processed matrices
temporary files
model checkpoints
```

Large raw data should remain on project storage.

Avoid unnecessary duplication of huge raw files.

Use checksums before deleting redundant copies.

---

# 93. First Computational Milestone

Before AI development, generate a dataset audit table containing:

```text
dataset
study
species
country
age
phenotype
sample_count
omics
raw_data_status
processed_data_status
clinical_labels
treatment_labels
patient_level_matching
access_restrictions
download_complete
qc_complete
```

This is the true starting point.

---

# 94. Step-by-Step Execution Roadmap

## Phase 0. Infrastructure

1. create project directories;
2. finish reproducible Python and R environment;
3. initialize Git for code;
4. create dataset registry;
5. create checksum and log system.

## Phase 1. Host proteomics

1. download complete PXD058621;
2. generate file inventory;
3. identify protein matrix;
4. identify biological and technical samples;
5. reconstruct sample mapping.

## Phase 2. Host metabolomics

1. download MSV000096913;
2. inventory files;
3. identify sample metadata;
4. map samples to study identifiers.

## Phase 3. Host transcriptomics

1. download GSE289197 expression data and metadata;
2. document that patient FASTQ files are not open;
3. map transcriptomic sample identifiers.

## Phase 4. Matched host multi-omics matrix

Determine:

\[
RNA \cap Proteome \cap Metabolome
\]

Create:

```text
matched_host_multiomics_manifest.tsv
```

No synthetic generation should begin until this intersection is audited.

## Phase 5. Dual RNA data

Download:

```text
PRJNA962942
ERP116722 / PRJEB33892
```

Process host and parasite expression.

## Phase 6. Pf8 metadata and resistance

Download:

- sample metadata;
- study metadata;
- resistance markers;
- inferred resistance status;
- copy-number variation;
- infection complexity.

## Phase 7. Treatment-cohort selection

Prioritize:

- Study 1167;
- Study 1101 after audit;
- additional Pf8 studies with suitable treatment and outcome metadata.

## Phase 8. IDDO/WWARN application

Prepare:

- research question;
- rationale;
- study identifiers;
- requested variables;
- analysis plan;
- data security plan;
- publication plan.

## Phase 9. Single-omics analysis

Complete single-omic analyses before integration.

## Phase 10. Real multi-omics integration

Run:

- MOFA+;
- SNF;
- DIABLO where matched samples allow it.

## Phase 11. Semi-synthetic generator

Generate:

```text
10 cohorts x 2,000 synthetic patients
```

## Phase 12. Synthetic quality control

Validate distributions and joint structure.

## Phase 13. Baseline predictive modeling

Run:

- clinical-only;
- resistance-marker-only;
- single-omic;
- concatenation;
- elastic net;
- Random Forest;
- XGBoost.

## Phase 14. Integration benchmark

Run:

- MOFA+;
- SNF;
- DIABLO;
- autoencoder;
- multimodal VAE;
- MOGONET;
- late fusion.

## Phase 15. HPAMM-Net

Build incrementally and perform ablation.

## Phase 16. Missing-modality experiments

Test random and structured missingness.

## Phase 17. External validation

Prioritize:

- study-held-out;
- country-held-out;
- independent dual-RNA cohort;
- independent parasite-genome cohort.

## Phase 18. Formulation experiments

Choose a scientifically justified dosage form and antimalarial target.

Perform Design of Experiments.

## Phase 19. Formulation AI

Benchmark:

- response-surface models;
- Random Forest;
- XGBoost;
- Gaussian process regression.

Use Bayesian optimization.

## Phase 20. Integrated precision-pharmaceutics demonstration

```text
patient molecular profile
+
parasite profile
+
clinical profile
        |
        v
treatment-specific predictions
        |
        v
candidate treatment
        |
        v
formulation optimization
        |
        v
predicted exposure and pharmaceutical performance
```

This remains a research demonstration until prospective clinical validation.

---

# 95. Decision Gates

## Gate 1. Data feasibility

Proceed only if:

- sample identifiers can be mapped;
- useful cross-omic overlap exists;
- raw/processed status is understood.

## Gate 2. Synthetic validity

Proceed only if synthetic data preserve important real structure.

## Gate 3. Baseline performance

Do not develop an unnecessarily complex model before baselines exist.

## Gate 4. Novel-method value

HPAMM-Net must demonstrate a defensible advantage in at least one area:

- external generalization;
- calibration;
- missingness robustness;
- stability;
- multi-task performance;
- biological interpretability.

## Gate 5. Formulation justification

Do not develop a new suspension merely because the project started with that concept.

Demonstrate a pharmaceutical need.

## Gate 6. Clinical claim

Do not make patient-level treatment claims without real prospective validation.

---

# 96. Suggested Publication Strategy

## Paper 1

**Pediatric malaria multi-omics resource and integration benchmark**

Focus:

- data audit;
- host and parasite multi-omics;
- semi-synthetic benchmark;
- established integration methods.

## Paper 2

**HPAMM-Net**

Focus:

- new computational method;
- missing modalities;
- multi-task treatment prediction;
- external validation.

## Paper 3

**AI-guided pediatric antimalarial formulation**

Focus:

- Design of Experiments;
- formulation experiments;
- machine learning;
- Bayesian optimization;
- experimental validation.

## Paper 4

**Integrated precision antimalarial pharmaceutics**

Only after sufficient clinical and experimental evidence.

This staged strategy is stronger than forcing all contributions into one first manuscript.

---

# 97. Highest-Risk Parts of the Project

## Risk 1. Insufficient matched multi-omics

Mitigation:

Use modular real cohorts and clearly labeled semi-synthetic development.

## Risk 2. Treatment data cannot be linked to parasite genomics

Mitigation:

Prioritize studies with explicit treatment design and pursue controlled access.

## Risk 3. Deep learning overfits

Mitigation:

Use compact networks, pathway features, regularization, and external validation.

## Risk 4. Synthetic data inflate performance

Mitigation:

Use real held-out validation and generator-independent tests.

## Risk 5. Formulation work becomes disconnected from biology

Mitigation:

Connect through pharmacokinetic exposure and therapeutic targets.

## Risk 6. AI adds no value beyond known resistance markers

Mitigation:

Include resistance-rule and clinical baselines.

---

# 98. Highest-Value Scientific Opportunities

1. Dual host-parasite modeling recognizes malaria as a two-genome disease process.
2. Missing-modality robustness is directly relevant to real multi-omics.
3. Geographic validation can test transportability across parasite populations.
4. Multi-task learning can link resistance, pharmacokinetics, and clinical outcomes.
5. Exposure creates a mechanistic bridge between systems biology and pharmaceutics.
6. A real-data anchored semi-synthetic framework could become a reusable benchmark for malaria multi-omics methods.

---

# 99. Immediate Next Actions

1. finish the computational environment;
2. download PXD058621 completely;
3. create PXD058621 inventory;
4. identify protein matrix and sample map;
5. download MSV000096913;
6. retrieve GSE289197 expression data and metadata;
7. construct exact host multi-omics intersection;
8. preprocess only after sample matching is verified;
9. download PRJNA962942;
10. download Gambian dual RNA data;
11. download Pf8 metadata and resistance resources;
12. prepare IDDO/WWARN data request;
13. build single-omic baselines;
14. run MOFA+, SNF, and DIABLO benchmarks;
15. design semi-synthetic generator;
16. validate generator;
17. benchmark machine learning and deep learning;
18. develop HPAMM-Net after benchmarks are stable.

---

# 100. Source Registry

## Core pediatric host multi-omics study

Publication:

https://www.nature.com/articles/s41467-025-59281-5

Open full text:

https://pmc.ncbi.nlm.nih.gov/articles/PMC12086225/

Transcriptomics:

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289197

Proteomics:

https://www.ebi.ac.uk/pride/archive/projects/PXD058621

Metabolomics:

https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=ba8df9f0096c4bf6abb946f323f153c1

Metabolomics files:

https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=ba8df9f0096c4bf6abb946f323f153c1

## Mali dual host-parasite transcriptomics

BioProject:

https://www.ncbi.nlm.nih.gov/bioproject/PRJNA962942

Open article:

https://pmc.ncbi.nlm.nih.gov/articles/PMC10918175/

## Gambian dual host-parasite transcriptomics

BioStudies:

https://www.ebi.ac.uk/biostudies/studies/E-MTAB-6413

Example SRA record linked to ERP116722:

https://www.ncbi.nlm.nih.gov/sra/ERX3492172

ENA home:

https://www.ebi.ac.uk/ena/browser/home

## Parasite genomics

MalariaGEN Pf8:

https://www.malariagen.net/data_package/open-dataset-plasmodium-falciparum-v80/

Pf8 studies:

https://apps.malariagen.net/apps/pf8/studies/

Pf8 samples:

https://apps.malariagen.net/apps/pf8/samples

Pf8 documentation:

https://apps.malariagen.net/apps/pf8/documentation/

Study 1167:

https://apps.malariagen.net/apps/pf8/studies/1167/

## Matched host-parasite genetics

Main resource:

https://www.malariagen.net/data_package/host-parasite-interaction/

Parasite accession list:

https://doi.org/10.5281/zenodo.5723515

Joint host-parasite genotype dataset:

https://doi.org/10.5281/zenodo.4973476

Association summary statistics:

https://doi.org/10.5281/zenodo.5722497

Human genetics resource:

https://www.malariagen.net/resource/25/

## Clinical and pharmacokinetic data

IDDO/WWARN malaria:

https://www.iddo.org/wwarn/access-malaria-data

Data access:

https://www.iddo.org/data-sharing/accessing-data

## Integration methods

MOFA+:

https://biofam.github.io/MOFA2/

MOFA+ missing-data documentation:

https://biofam.github.io/MOFA2/faq.html

SNF:

https://www.nature.com/articles/nmeth.2810

DIABLO:

https://mixomics.org/mixdiablo/

DIABLO article:

https://academic.oup.com/bioinformatics/article/35/17/3055/5292387

MOGONET:

https://www.nature.com/articles/s41467-021-23774-w

## Pharmaceutical resources

FDA Inactive Ingredient Database:

https://www.fda.gov/drugs/drug-approvals-and-databases/inactive-ingredients-database-download

WHO Global Malaria Programme:

https://www.who.int/teams/global-malaria-programme

WHO prequalification:

https://extranet.who.int/prequal/medicines/prequalified-lists

PubChem:

https://pubchem.ncbi.nlm.nih.gov/

ChEMBL:

https://www.ebi.ac.uk/chembl/

---

# 101. Final Project Principle

The strongest scientific formulation is:

\[
\boxed{Host\ biology + Parasite\ biology + Clinical\ state + Drug\ and\ dose \rightarrow Exposure,\ resistance,\ efficacy,\ toxicity}
\]

followed by:

\[
\boxed{Therapeutic\ requirement + Drug\ properties + Formulation\ design \rightarrow Optimized\ pediatric\ dosage\ form}
\]

The project should not be described simply as:

```text
AI + multi-omics + malaria + syrup
```

The defensible contribution is the mechanistic and computational connection between:

1. host state;
2. parasite state;
3. pharmacology;
4. treatment outcome;
5. pediatric formulation performance.

That structure should guide every data decision, model choice, experiment, and publication claim.

---

# 102. Project Success Criteria

The project will be considered scientifically successful if it demonstrates:

- reproducible acquisition and processing of core datasets;
- transparent sample mapping;
- biologically plausible single-omics results;
- validated semi-synthetic generation;
- fair benchmarking against established methods;
- real-data validation;
- predictive value beyond simple resistance markers and clinical variables;
- calibrated uncertainty;
- robustness to missing modalities;
- interpretable host and parasite contributions;
- pharmaceutically justified formulation development;
- prospective experimental validation of formulation predictions;
- clear separation between simulation evidence and clinical evidence.

---

# 103. Main Failure Mode to Avoid

The largest risk is producing an impressive computational pipeline that answers a vague biological question.

Do not ask:

```text
Can AI integrate malaria omics?
```

Ask:

```text
Does joint host-parasite molecular information improve prediction of a defined antimalarial response phenotype beyond established clinical and resistance-marker baselines?
```

Then ask:

```text
Can an experimentally validated formulation model identify pediatric oral formulations that achieve predefined pharmaceutical and exposure-related targets?
```

These are testable scientific questions.

---

# 104. Living Document Policy

Update this file whenever:

- a dataset is downloaded;
- an accession changes;
- data access is approved or rejected;
- sample overlap is established;
- a model is added or removed;
- a benchmark is completed;
- the synthetic generator changes;
- a formulation experiment is completed;
- a manuscript claim changes.

Recommended version tags:

```text
v0.1 data-discovery
v0.2 download-complete
v0.3 preprocessing-complete
v0.4 real-integration-complete
v0.5 synthetic-generator-complete
v0.6 benchmark-complete
v0.7 HPAMM-Net-complete
v0.8 external-validation-complete
v0.9 formulation-complete
v1.0 manuscript-ready
```

This blueprint should remain the single source of truth for project scope and methodology.
