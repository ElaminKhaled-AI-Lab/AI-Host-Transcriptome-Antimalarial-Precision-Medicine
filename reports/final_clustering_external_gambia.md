# Final clustering on the corrected representation (external_gambia)

- Samples: 52 total, 0 removed by depth quality control, 52 analysed
- k = 2, sizes [25, 27], silhouette 0.164, balance 0.999, bootstrap stability 0.680
- Host PC1 explains 15.2% of host variance; 30 components exceed 1%

## 1. Is the clustering real?

Compared against 1000 cluster-free reference datasets drawn uniformly over the bounding box in the principal-axis frame:

- silhouette 0.164 versus null median 0.0851 (95th percentile 0.1059), p = **0.0010**
- bootstrap stability 0.680 versus null median 0.6825 (95th percentile 0.9034), p = **0.5085**  <- primary criterion

A column-wise permutation was deliberately not used: it preserves each component's marginal distribution, so a bimodal component keeps two clumps and the test loses power.

## 2. Does it correspond to clinical severity?

ARI = **-0.0215** on 44 labelled children; Fisher exact p = 1.0000.

An ARI near zero means the clusters, although reproducible, do not recover the cerebral / uncomplicated distinction.

## 3. What does it track?

| covariate | test | effect | value | p | q (FDR) | significant |
|---|---|---|---|---|---|---|
| host_PC1 | Mann-Whitney U | rank-biserial r | 1.000 | 6.74e-10 | 1.01e-08 | YES |
| parasite_read_fraction | Mann-Whitney U | rank-biserial r | 0.876 | 6.56e-08 | 4.92e-07 | YES |
| parasite_library_size | Mann-Whitney U | rank-biserial r | 0.864 | 9.84e-08 | 4.92e-07 | YES |
| host_transcripts_detected | Mann-Whitney U | rank-biserial r | 0.630 | 0.000103 | 0.000387 | YES |
| parasite_PC2 | Mann-Whitney U | rank-biserial r | 0.422 | 0.0093 | 0.0279 | YES |
| parasite_transcripts_detected | Mann-Whitney U | rank-biserial r | 0.412 | 0.0112 | 0.0279 | YES |
| host_library_size | Mann-Whitney U | rank-biserial r | 0.333 | 0.0402 | 0.0862 | no |
| parasite_PC3 | Mann-Whitney U | rank-biserial r | 0.253 | 0.12 | 0.224 | no |
| host_PC2 | Mann-Whitney U | rank-biserial r | 0.159 | 0.332 | 0.553 | no |
| host_PC4 | Mann-Whitney U | rank-biserial r | 0.141 | 0.389 | 0.584 | no |
| run_accession_order | Mann-Whitney U | rank-biserial r | 0.114 | 0.486 | 0.663 | no |
| host_PC3 | Mann-Whitney U | rank-biserial r | 0.043 | 0.798 | 0.884 | no |
| severity_group | chi-square | Cramer's V | 0.042 | 0.783 | 0.884 | no |
| parasite_PC1 | Mann-Whitney U | rank-biserial r | 0.034 | 0.84 | 0.884 | no |
| host_PC5 | Mann-Whitney U | rank-biserial r | 0.025 | 0.884 | 0.884 | no |

Strongest association: **host_PC1** (rank-biserial r = 1.000, q = 1.01e-08).