# Nano-DDS full-factorial (3^3) design -- in-silico digital twin

- Design: 3x3x3 full factorial = 27 runs per polymer x drug
- Polymers swept: 11 (Alginate, Chitosan, Ethylcellulose, Eudragit_RL100, Eudragit_RS100, Eudragit_S100, HPMC, PCL, PLA_PDLLA, PLGA_50_50, PLGA_75_25)
- Drugs swept: 9 (amodiaquine, artemether, artesunate, atovaquone, dihydroartemisinin, lumefantrine, mefloquine, piperaquine, primaquine)
- Total formulations characterized: 2673

## Response ranges (all runs)

- size_nm: 87.30 -- 511.80 (median 184.40)
- pdi: 0.05 -- 0.46 (median 0.17)
- zeta_mv: -31.60 -- 28.00 (median -9.20)
- ee_pct: 32.30 -- 92.60 (median 64.10)
- dl_pct: 6.07 -- 26.71 (median 9.56)
- release_24h_pct: 43.70 -- 96.20 (median 61.60)
- liver_targeting_index: 0.00 -- 0.94 (median 0.53)
- liver_plasma_ratio: 0.10 -- 0.43 (median 0.28)

## Release mechanism distribution (Korsmeyer-Peppas)

- anomalous (non-Fickian) transport: 1640 runs
- Fickian diffusion: 1033 runs

All values are in-silico predictions from a documented, literature-
anchored digital twin, generated for AI method development -- not a
substitute for laboratory characterization.