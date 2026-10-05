# Dual-RNA QC + robustness (external_gambia)

- Samples: 52
- Parasite-read fraction: median 0.199%, max 0.753%
- Samples with parasite fraction >0.1%: 36
- Host transcripts detected (median): 84305
- Parasite transcripts detected (median): 96

The parasite-RNA layer is sparse in peripheral blood, as expected at low peripheral parasitemia. Integration is therefore evaluated host-only, as full dual host+parasite RNA, and on the parasite-informed subset, to show the host integration is stable and to quantify the parasite layer's contribution rather than assume it.

- host_only: BLISS silhouette 0.826, stability 1.000
- dual_full: BLISS silhouette 0.861, stability 1.000
- dual_parasite_informed_subset: BLISS silhouette 0.854, stability 1.000