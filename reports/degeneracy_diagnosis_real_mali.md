# Cluster degeneracy diagnosis (real_mali)

- Samples: 56

## Stage 1 - who is isolated, and why

- Minority cluster at k = 2: **1 sample(s)** (SRR25640283)
- Samples failing depth quality control: 7
- Minority fully explained by quality flags: **False**

## Stage 2 - is there any non-degenerate structure?

- Flagged samples removed: 7; retained: 49
- Rejection criteria: balance < 0.5 or a cluster smaller than 5

- Host PC1 explains 9.0% of host variance; 30 host components exceed 1%. The pipeline z-scores all 50 components, equalising informative and noise axes, so both representations are reported below.

| representation | k | silhouette | balance | min size | stability | verdict |
|---|---|---|---|---|---|---|
| z-scored PCs | 2 | 0.209 | 0.144 | 1 | 0.184 | DEGENERATE |
| z-scored PCs | 3 | 0.005 | 0.670 | 1 | 0.108 | DEGENERATE |
| z-scored PCs | 4 | -0.023 | 0.771 | 1 | 0.076 | DEGENERATE |
| z-scored PCs | 5 | -0.015 | 0.746 | 1 | 0.176 | DEGENERATE |
| z-scored PCs | 6 | 0.006 | 0.606 | 1 | 0.096 | DEGENERATE |
| variance-weighted PCs | 2 | 0.075 | 0.931 | 17 | 0.342 | acceptable |
| variance-weighted PCs | 3 | 0.056 | 0.708 | 1 | 0.210 | DEGENERATE |
| variance-weighted PCs | 4 | -0.017 | 0.766 | 1 | 0.277 | DEGENERATE |
| variance-weighted PCs | 5 | 0.060 | 0.543 | 1 | 0.218 | DEGENERATE |
| variance-weighted PCs | 6 | 0.048 | 0.682 | 1 | 0.260 | DEGENERATE |

Best acceptable solution: k = 2, silhouette 0.075, balance 0.931, stability 0.342.
Against a permutation null (median silhouette 0.032), this **exceeds the permutation null** (empirical p = 0.020).

## Stage 3 - is severity graded rather than discrete?

insufficient labelled samples or not a two-group comparison