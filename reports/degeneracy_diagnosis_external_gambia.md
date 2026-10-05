# Cluster degeneracy diagnosis (external_gambia)

- Samples: 52

## Stage 1 - who is isolated, and why

- Minority cluster at k = 2: **4 sample(s)** (ERR3470488, ERR3470509, ERR3470533, ERR3470534)
- Samples failing depth quality control: 0
- Minority fully explained by quality flags: **False**

## Stage 2 - is there any non-degenerate structure?

- Flagged samples removed: 0; retained: 52
- Rejection criteria: balance < 0.5 or a cluster smaller than 5

- Host PC1 explains 16.3% of host variance; 30 host components exceed 1%. The pipeline z-scores all 50 components, equalising informative and noise axes, so both representations are reported below.

| representation | k | silhouette | balance | min size | stability | verdict |
|---|---|---|---|---|---|---|
| z-scored PCs | 2 | 0.031 | 0.391 | 4 | 0.223 | DEGENERATE |
| z-scored PCs | 3 | -0.006 | 0.627 | 1 | 0.137 | DEGENERATE |
| z-scored PCs | 4 | 0.073 | 0.205 | 1 | 0.095 | DEGENERATE |
| z-scored PCs | 5 | 0.015 | 0.739 | 1 | 0.123 | DEGENERATE |
| z-scored PCs | 6 | 0.012 | 0.696 | 1 | 0.107 | DEGENERATE |
| variance-weighted PCs | 2 | 0.162 | 0.990 | 23 | 0.925 | acceptable |
| variance-weighted PCs | 3 | 0.156 | 0.701 | 1 | 0.598 | DEGENERATE |
| variance-weighted PCs | 4 | 0.046 | 0.875 | 2 | 0.537 | DEGENERATE |
| variance-weighted PCs | 5 | 0.106 | 0.661 | 1 | 0.492 | DEGENERATE |
| variance-weighted PCs | 6 | 0.108 | 0.673 | 1 | 0.527 | DEGENERATE |

Best acceptable solution: k = 2, silhouette 0.162, balance 0.990, stability 0.925.
Against a permutation null (median silhouette 0.033), this **exceeds the permutation null** (empirical p = 0.020).

## Stage 3 - is severity graded rather than discrete?

Logistic regression of severity on the continuous components gives a cross-validated AUROC of **0.519** (44 samples, 5-fold).

| component | variance explained | effect r | p |
|---|---|---|---|
| host_PC3 | 0.041 | 0.337 | 0.0578 |
| parasite_PC1 | 0.318 | 0.279 | 0.117 |
| host_PC5 | 0.026 | 0.275 | 0.123 |
| host_PC2 | 0.050 | 0.096 | 0.596 |
| host_PC1 | 0.163 | 0.092 | 0.612 |
| parasite_PC2 | 0.090 | 0.071 | 0.697 |
| parasite_PC3 | 0.068 | 0.067 | 0.715 |
| host_PC4 | 0.032 | 0.012 | 0.953 |