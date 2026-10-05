# Data Leakage Audit (Section 78)

Run BEFORE every analysis. Tick each before reporting results.

- [ ] no_post_treatment_predictors_in_pretreatment_model
- [ ] outcome_not_embedded_in_resistance_label
- [ ] no_duplicate_patients_across_studies
- [ ] no_technical_replicates_across_train_test
- [ ] no_batch_specific_identifiers_as_features
- [ ] no_site_variable_encoding_treatment_arm
- [ ] final_test_cohort_excluded_from_generator_fit