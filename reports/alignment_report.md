# Cross-omic alignment report (Gate 1)

## Per-layer sample counts
- host_transcriptome: 79 samples
- host_proteome: 108 samples
- host_metabolome: 0 samples
- parasite_genome: 273 samples

## Alignment status

- Host layers present: host_transcriptome, host_proteome
- Host layers use DIFFERENT id schemes (clinical codes vs Sample_NN); a crosswalk is required to match patients across them (Section 24). Do NOT fuse without it (Boundary 7.2).
- Crosswalk present: NO
- Parasite genome is a SEPARATE real cohort (Pf8); it is joined to the host cohorts only through the semi-synthetic generator, never at the patient level (Section 33).