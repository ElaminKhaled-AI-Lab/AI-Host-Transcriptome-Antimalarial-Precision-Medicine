# Semi-synthetic validation (Section 38)

Compares real vs synthetic on transcriptome PC structure.

- tx_PC1: real mean=-0.00 sd=185.18 | synthetic mean=10.28 sd=185.07
- tx_PC2: real mean=0.00 sd=58.54 | synthetic mean=-0.59 sd=57.91
- tx_PC3: real mean=0.00 sd=26.79 | synthetic mean=-0.04 sd=26.94
- tx_PC4: real mean=0.00 sd=24.90 | synthetic mean=0.18 sd=24.65
- tx_PC5: real mean=-0.00 sd=23.14 | synthetic mean=-0.23 sd=22.93

Real transcriptome-phenotype linkage preserved via regression; proteome internal covariance preserved via PCA sampling.
REJECT criterion (Section 38): if synthetic destroys real joint structure. v1 preserves marginals + internal covariance by construction.