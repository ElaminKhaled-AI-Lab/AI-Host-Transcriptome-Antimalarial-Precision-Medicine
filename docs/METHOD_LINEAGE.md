# Method lineage: from BLISS (TCGA colon) to HPAMM-Net (pediatric malaria)

This project deliberately reuses and extends the group's published integration
method so that the new work is a measurable successor, not a fresh start.

## Prior work (the template)

Elamin K.M., Elbashir S.M.I., Adam I. *AI-Driven Multi-Omics Integration of
Synthetic Colon Adenocarcinoma for Cluster-Guided PROTAC Candidate Design
Targeting KRASG12D.* Int. J. Mol. Sci. 2026, 27, 7511.
doi:10.3390/ijms27177511.

Key elements carried over:

| Element | TCGA / BLISS paper | Malaria project (this repo) |
|---|---|---|
| Synthetic anchor | 425 tumors + 41 normals matched to TCGA marginals | 10 x 2,000 semi-synthetic patients matched to real malaria cohorts (Section 36) |
| Benchmark | 16 established integration methods | same families: PCA-concat, MOFA+, SNF, DIABLO, AE, VAE, MOGONET, late fusion (Section 69) |
| Novel method | **BLISS** (triplet + factor + balance losses) | **HPAMM-Net** = BLISS core + dual host-parasite + missingness + multi-task + treatment conditioning |
| Evaluation axes | ARI, silhouette, cluster balance, bootstrap stability, layer ablation, per-patient confidence | identical battery in `src/metrics/integration_eval.py` |
| Synthetic->real | ARI 0.90 -> 0.28 (TCGA) -> 0.13 (CPTAC) | synthetic -> real Mali -> external Gambia (`transportability.py`) |
| Translational layer | cluster-guided de novo PROTAC (BRICS->shape->docking->MM-GBSA->ADMET) | cluster/treatment-guided de novo **formulation** (enumerate->surrogate->constraints->PK-gated desirability->Bayesian) |

## BLISS, reimplemented (`src/models/bliss.py`)

Faithful to paper Section 2.7 / Figure 7:
- separate LINEAR encoders -> 16-d shared latent (concat -> LayerNorm -> GELU -> L2-normalize);
- `L = 1.0*L_triplet + 0.3*L_factor + 0.7*L_balance`;
- triplet over PCA->k-means pseudo-labels (weakly supervised);
- factor = per-layer reconstruction from shared latent;
- balance = KL(cluster marginal || uniform), the term introduced in the paper
  that catches the degenerate SNF collapse.

Shipped as a reusable baseline so the malaria data can reproduce the exact prior
method and benchmark the successor under matched conditions.

## HPAMM-Net, the successor (`src/models/hpamm_net.py`)

Inherits the three BLISS losses (`combined_loss`) and adds the contributions to
be ablated (Section 75) and defended as novel (Sections 48, 79-80):

1. **Dual host-parasite factorization** — separate `Z_H`, `Z_P`, fused `Z_HP`
   (malaria is a two-genome disease; colon cancer is one-genome).
2. **Missingness-aware gated fusion** — missing modality contributes zero, not
   an imputed guess (real malaria omics are missing far more than TCGA).
3. **Modality dropout** during training for robustness.
4. **Treatment conditioning** — the same patient representation evaluated under
   multiple candidate antimalarials (no analog in the cancer paper).
5. **Multi-task supervised heads** — resistance / clearance / PK / recrudescence
   / toxicity / cure jointly (the cancer paper was unsupervised clustering).
6. **Stability selection** built into evaluation (`stability_selection`),
   promoting the paper's bootstrap diagnostic to a first-class model property.

## Why the synthetic-vs-real comparison is the crux

The paper's central honesty is that near-perfect synthetic performance did NOT
transport to real data. For malaria the same test answers the project's core
question — *can an integration method learned on the semi-synthetic cohort be
applied to any other real cohort?* A method that scores high on synthetic but
collapses on real Mali / external Gambia has low transportability and must not
be claimed as broadly applicable (blueprint Sections 7.1, 77). `transportability.py`
computes and reports that gap for every method.
