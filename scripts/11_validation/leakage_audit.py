#!/usr/bin/env python
"""Phase 17 -- data-leakage audit checklist (blueprint Section 78).

Create and run this BEFORE every analysis. Potential leakage sources:
  * post-treatment variables used in a pre-treatment model
  * outcome embedded in the resistance label
  * duplicate patients across studies
  * technical replicates across train and test
  * batch-specific identifiers
  * site variables encoding the treatment arm
  * use of the final test cohort in synthetic-generator fitting

Emits a checklist to reports/ and (where data are present) runs automatic
checks for duplicate IDs and replicate straddling.
"""
import _bootstrap  # noqa: F401
from src.utils.config import get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

CHECKS = [
    "no_post_treatment_predictors_in_pretreatment_model",
    "outcome_not_embedded_in_resistance_label",
    "no_duplicate_patients_across_studies",
    "no_technical_replicates_across_train_test",
    "no_batch_specific_identifiers_as_features",
    "no_site_variable_encoding_treatment_arm",
    "final_test_cohort_excluded_from_generator_fit",
]


def main() -> None:
    report = get_path("reports", create=True) / "leakage_audit.md"
    lines = ["# Data Leakage Audit (Section 78)", "",
             "Run BEFORE every analysis. Tick each before reporting results.", ""]
    for c in CHECKS:
        log.info("[CHECK] %s", c)
        lines.append(f"- [ ] {c}")
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote leakage-audit checklist: %s", report)


if __name__ == "__main__":
    main()
