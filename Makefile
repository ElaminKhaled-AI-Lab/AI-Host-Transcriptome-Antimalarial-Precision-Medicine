# =====================================================================
# Makefile  --  convenience targets for the Malaria_Project pipeline.
# Run `make help` for the list. Targets mirror blueprint Section 94.
# =====================================================================
PY := python

.PHONY: help setup check download-all download-host download-dual download-parasite \
        download-formulation qc single-omics sample-map synthetic \
        benchmarks transportability hpamm treatment formulation validation figures clean-tmp

help:
	@echo "Malaria_Project targets:"
	@echo "  make setup             - init git, registry, log dirs"
	@echo "  make check             - verify config + environment"
	@echo "  make download-all      - fetch ALL open data + managed-access checklists"
	@echo "  make transportability  - synthetic-vs-real transportability battery"
	@echo "  make download-host     - GEO + PRIDE + MassIVE host multi-omics"
	@echo "  make download-dual     - Mali + Gambia dual host-parasite RNA"
	@echo "  make download-parasite - MalariaGEN Pf8 metadata + resistance"
	@echo "  make download-formulation - FDA IID excipients"
	@echo "  make qc                - per-dataset QC + dataset audit table"
	@echo "  make single-omics      - single-omic analyses before integration"
	@echo "  make sample-map        - master sample map + cross-omic intersection"
	@echo "  make synthetic         - semi-synthetic generator + QC"
	@echo "  make benchmarks        - MOFA+/SNF/DIABLO/AE/VAE/MOGONET/late fusion"
	@echo "  make hpamm             - HPAMM-Net incremental build + ablations"
	@echo "  make treatment         - multi-task outcomes + prioritization"
	@echo "  make formulation       - DoE + formulation ML + Bayesian optimization"
	@echo "  make validation        - external validation + leakage audit"
	@echo "  make figures           - figures and tables"

setup:
	$(PY) scripts/00_setup/00_init_project.py

check:
	$(PY) scripts/00_setup/01_check_environment.py

download-all:
	$(PY) scripts/01_download/download_all.py

download-host:
	$(PY) scripts/01_download/dl_host_transcriptomics_GSE289197.py
	$(PY) scripts/01_download/dl_host_proteomics_PXD058621.py
	$(PY) scripts/01_download/dl_host_metabolomics_MSV000096913.py

download-dual:
	$(PY) scripts/01_download/dl_dual_rna_PRJNA962942.py
	$(PY) scripts/01_download/dl_dual_rna_gambia_ERP116722.py

download-parasite:
	$(PY) scripts/01_download/dl_parasite_genomics_Pf8.py --metadata-only

download-formulation:
	$(PY) scripts/01_download/dl_formulation_FDA_IID.py

qc:
	$(PY) scripts/02_qc/run_qc.py

single-omics:
	$(PY) scripts/04_single_omics/run_single_omics.py

sample-map:
	$(PY) scripts/05_sample_mapping/build_sample_map.py

synthetic:
	$(PY) scripts/06_synthetic_cohort/generate_cohorts.py
	$(PY) scripts/06_synthetic_cohort/validate_cohorts.py

benchmarks:
	$(PY) scripts/07_integration_benchmarks/run_benchmarks.py

transportability:
	$(PY) scripts/07_integration_benchmarks/transportability.py --demo

hpamm:
	$(PY) scripts/08_hpamm_net/train_hpamm_net.py
	$(PY) scripts/08_hpamm_net/run_ablations.py

treatment:
	$(PY) scripts/09_treatment_models/run_treatment_models.py

formulation:
	$(PY) scripts/10_formulation/run_formulation_pipeline.py

validation:
	$(PY) scripts/11_validation/run_external_validation.py
	$(PY) scripts/11_validation/leakage_audit.py

figures:
	$(PY) scripts/12_figures/make_figures.py

clean-tmp:
	rm -rf tmp/* && echo "tmp cleared (raw data untouched)"
