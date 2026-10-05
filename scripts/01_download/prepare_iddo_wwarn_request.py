#!/usr/bin/env python
"""Stage 5 -- IDDO/WWARN managed-access application helper (blueprint Section 19).

IDDO  = Infectious Diseases Data Observatory
WWARN = Worldwide Antimalarial Resistance Network

Clinical + pharmacokinetic (PK) Individual Participant Data (IPD) are MANAGED:
they require an application and a Data Use Agreement (DUA). There is no
automated download. This script generates a ready-to-edit application
checklist and the full variable request list (Section 19) into
``metadata/access_notes/`` so the request is complete and auditable.

References:
  Access malaria data: https://www.iddo.org/wwarn/access-malaria-data
  Accessing data:      https://www.iddo.org/data-sharing/accessing-data
"""
import _bootstrap  # noqa: F401
from src.utils.config import get_dataset, get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

CHECKLIST = """# IDDO/WWARN Data Request -- Application Checklist

Access tier: MANAGED (application + Data Use Agreement required).
Landing: {url_access}
Process: {url_process}

## Official access steps (Section 19)
1. [ ] Explore data catalogue; identify candidate study identifiers.
2. [ ] Identify variables available per study (data dictionary).
3. [ ] Review access criteria.
4. [ ] Define research plan (question, rationale, analysis).
5. [ ] Submit application.
6. [ ] Undergo review.
7. [ ] Sign Data Use Agreement (DUA).
8. [ ] Receive approved data into: data/raw/clinical_pk/IDDO_WWARN/

## Research plan components to attach
- [ ] Research question (see blueprint Section 103 -- the defined phenotype).
- [ ] Rationale (dual host-parasite + PK prediction beyond clinical baselines).
- [ ] Study identifiers requested.
- [ ] Requested variables (see variable list below).
- [ ] Analysis plan (nested CV; study/country-held-out validation).
- [ ] Data security plan (encrypted storage on project HDD; access control).
- [ ] Publication plan (staged papers 1-4, Section 96).

## Requested variables (Section 19)

### Demographics
age; sex; weight; site; country

### Disease
parasitemia; temperature; hemoglobin; disease severity; nutritional measurements

### Treatment
drug combination; dose; dose per kilogram; formulation (if available);
vomiting and repeat dose; food administration (if relevant)

### Pharmacokinetics
sampling time; drug concentration; area under the concentration-time curve (AUC);
day-7 concentration (where available)

### Outcomes
parasite-clearance measures; day-3 positivity; day-28 response; day-42 response;
extended follow-up; PCR-corrected outcome; recrudescence; reinfection; adverse events

## Important limitation (Section 19)
Not every study contains every variable. The IDDO data dictionary lists
variables that MAY NOT have been collected/shared for a given study. Record,
per study, which requested variables are actually present.
"""


def main() -> None:
    ds = get_dataset("clinical_pk_iddo")
    out_dir = get_path("access_notes", create=True)
    out = out_dir / "IDDO_WWARN_application_checklist.md"
    out.write_text(
        CHECKLIST.format(url_access=ds["url_access"], url_process=ds["url_process"]),
        encoding="utf-8",
    )
    log.info("wrote application checklist: %s", out)
    log.info("This dataset is MANAGED -- no automated download. Follow the checklist.")


if __name__ == "__main__":
    main()
