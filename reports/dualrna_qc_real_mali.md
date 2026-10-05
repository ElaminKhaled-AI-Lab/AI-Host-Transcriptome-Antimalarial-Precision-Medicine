# Dual-RNA QC + robustness (real_mali)

- Samples: 56
- Parasite-read fraction: median 0.059%, max 1.222%
- Samples with parasite fraction >0.1%: 16
- Host transcripts detected (median): 96121
- Parasite transcripts detected (median): 65

The parasite-RNA layer is sparse in peripheral blood, as expected at low peripheral parasitemia. Integration is therefore evaluated host-only, as full dual host+parasite RNA, and on the parasite-informed subset, to show the host integration is stable and to quantify the parasite layer's contribution rather than assume it.

- host_only: BLISS silhouette 0.838, stability 1.000
- dual_full: BLISS silhouette 0.784, stability 0.471
- dual_parasite_informed_subset: BLISS silhouette 0.774, stability 0.657