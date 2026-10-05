# Final clustering on the corrected representation (real_mali)

- Samples: 56 total, 7 removed by depth quality control, 49 analysed
- k = 2, sizes [4, 45], silhouette 0.282, balance 0.408, bootstrap stability 0.380
- Host PC1 explains 8.7% of host variance; 30 components exceed 1%

## 1. Is the clustering real?

Compared against 1000 cluster-free reference datasets drawn uniformly over the bounding box in the principal-axis frame:

- silhouette 0.282 versus null median 0.1228 (95th percentile 0.1533), p = **0.0010**
- bootstrap stability 0.380 versus null median 0.7835 (95th percentile 0.9650), p = **0.9830**  <- primary criterion

A column-wise permutation was deliberately not used: it preserves each component's marginal distribution, so a bimodal component keeps two clumps and the test loses power.

## 2. Does it correspond to clinical severity?

No usable two-group labels for this cohort.

## 3. What does it track?

| covariate | test | effect | value | p | q (FDR) | significant |
|---|---|---|---|---|---|---|
| host_PC1 | Mann-Whitney U | rank-biserial r | 0.944 | 0.00017 | 0.00272 | YES |
| host_transcripts_detected | Mann-Whitney U | rank-biserial r | 0.622 | 0.0395 | 0.233 | no |
| host_PC5 | Mann-Whitney U | rank-biserial r | 0.600 | 0.0482 | 0.233 | no |
| parasite_PC3 | Mann-Whitney U | rank-biserial r | 0.578 | 0.0582 | 0.233 | no |
| host_PC4 | Mann-Whitney U | rank-biserial r | 0.456 | 0.144 | 0.46 | no |
| host_PC2 | Mann-Whitney U | rank-biserial r | 0.367 | 0.246 | 0.656 | no |
| host_PC3 | Mann-Whitney U | rank-biserial r | 0.289 | 0.366 | 0.837 | no |
| parasite_PC1 | Mann-Whitney U | rank-biserial r | 0.256 | 0.426 | 0.852 | no |
| host_library_size | Mann-Whitney U | rank-biserial r | 0.156 | 0.634 | 0.936 | no |
| parasite_library_size | Mann-Whitney U | rank-biserial r | 0.100 | 0.765 | 0.936 | no |
| parasite_PC2 | Mann-Whitney U | rank-biserial r | 0.089 | 0.792 | 0.936 | no |
| age_years | Mann-Whitney U | rank-biserial r | 0.083 | 0.795 | 0.936 | no |
| parasite_read_fraction | Mann-Whitney U | rank-biserial r | 0.078 | 0.819 | 0.936 | no |
| run_accession_order | Mann-Whitney U | rank-biserial r | 0.078 | 0.819 | 0.936 | no |
| sex | Fisher exact | Cramer's V | 0.031 | 1 | 1 | no |
| parasite_transcripts_detected | Mann-Whitney U | rank-biserial r | 0.028 | 0.942 | 1 | no |

Strongest association: **host_PC1** (rank-biserial r = 0.944, q = 0.00272).