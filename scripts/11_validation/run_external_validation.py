#!/usr/bin/env python
"""Phase 17 -- external validation hierarchy (blueprint Sections 71, 77).

Validation hierarchy (Section 71), weakest to strongest:
  1. cross-validation within cohort
  2. study-held-out validation
  3. country-held-out validation
  4. independent repository/publication cohort
  5. prospective cohort

Also the central synthetic-to-real experiment (Section 77):
  train on semi-synthetic -> test on REAL held-out; and compare
  (real only) vs (real + synthetic augmentation) on the SAME real test set.
  If augmentation does not help real validation, claim NO biological improvement.

Uses src.data.splits (study_held_out, country_held_out).
STUB: defines the evaluation matrix.
"""
import _bootstrap  # noqa: F401
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

HIERARCHY = ["within_cohort_cv", "study_held_out", "country_held_out",
             "independent_repository", "prospective"]


def main() -> None:
    for level in HIERARCHY:
        log.info("[STUB] validation level: %s", level)
    log.info("[STUB] synthetic-to-real (Section 77): real-only vs real+synthetic "
             "on identical real test set.")
    log.info("A random 80:20 split is NOT the strongest evidence (Section 71).")


if __name__ == "__main__":
    main()
