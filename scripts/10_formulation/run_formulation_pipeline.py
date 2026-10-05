#!/usr/bin/env python
"""Phase 18-20 -- AI de novo pediatric antimalarial formulation (Sections 62-68).

Formulation analog of the group's cluster-guided de novo PROTAC pipeline
(BLISS paper Sections 2.13-2.14). End-to-end flow:

  molecular/PK arm  ->  candidate antimalarial + target PK exposure (AUC)
        |
        v
  1. ENUMERATE excipient/process candidates (from FDA IID + process limits)
  2. SURROGATE ML predicts all performance responses (GP for small data + uncertainty)
  3. CONSTRAINT FILTER (pH, viscosity, dose-volume, redispersibility, stability) -- the "shape filter"
  4. DESIRABILITY scoring gated on the PK exposure window -- the "MM-GBSA/ADMET" analog
  5. BAYESIAN / active-learning loop proposes the next experiments
        |
        v
  lead pediatric liquid/suspension formulation (computational candidate)

Clinical framing (blueprint Section 7.4-7.5): the dosage form is chosen from
drug properties, not a preset preference for "syrup"; a suspension or granules-
for-suspension may be correct. Any proposed liquid must beat existing WHO
pediatric products on a stated advantage (dose precision, acceptability,
volume, exposure, stability).

RUNNABLE scaffold: uses a documented synthetic response model so the loop
executes end-to-end today; swap in real Design-of-Experiments measurements to
train the surrogate for real proposals. All outputs are COMPUTATIONAL
CANDIDATES requiring experimental validation.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.models.formulation_integration import (
    INPUT_FACTORS, RESPONSES, FormulationConstraints, Objective,
    build_surrogate, enumerate_candidates, satisfies_constraints,
    desirability, acquisition_ucb,
)

log = get_logger(__name__)


def _synthetic_responses(X: np.ndarray, seed: int = 11) -> dict:
    """Stand-in ground-truth response model for the scaffold self-test.

    Encodes plausible monotone relationships (higher suspending-agent conc ->
    lower sedimentation but higher viscosity, etc.) so the pipeline logic is
    demonstrable. REPLACE with a surrogate trained on real DoE measurements.
    """
    rng = np.random.default_rng(seed)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-9)
    def r(*coeffs):
        w = np.zeros(X.shape[1]);
        for i, c in coeffs: w[i] = c
        return Xn @ w + rng.normal(0, 0.3, X.shape[0])
    out = {
        "sedimentation_volume_ratio": 0.7 + 0.1 * r((3, -1.0)),
        "redispersibility_strokes": 8 + 3 * r((3, -0.8), (17, 0.5)),
        "viscosity_cp": 800 + 400 * r((3, 1.0), (13, 0.6)),
        "dissolution_pct_30min": 70 + 15 * r((14, -0.7), (0, 0.4)),
        "content_uniformity_rsd": 3 + 1.5 * r((17, -0.6)),
        "delivered_dose_rsd": 3 + 1.5 * r((3, -0.5)),
        "chemical_assay_pct_t0": 99 + 0.5 * rng.normal(0, 1, X.shape[0]),
        "chemical_assay_pct_accel": 93 + 3 * r((8, -0.5), (20, -0.6)),
        "palatability_score": 6 + 2 * r((9, 0.8), (10, 0.3)),
        "administration_volume_ml": 6 + 2 * r((0, -0.9)),
        "predicted_pk_auc": 100 + 25 * r((3, 0.4), (0, 0.5)),
    }
    # clip to sensible ranges
    out["ph"] = 4 + 3.5 * np.clip((Xn[:, 8] + 3) / 6, 0, 1)
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-candidates", type=int, default=5000,
                    help="excipient/process combinations to enumerate")
    ap.add_argument("--pk-target", type=float, default=None,
                    help="target PK exposure (AUC) from the molecular arm")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    seed = load_config()["reproducibility"]["global_seed"]
    cons = FormulationConstraints()
    obj = Objective(pk_target_auc=args.pk_target)

    log.info("STAGE 1: enumerate %d candidate formulations over %d factors",
             args.n_candidates, len(INPUT_FACTORS))
    # Excipient library normally built from FDA IID + process limits; empty ->
    # placeholder [0,1] ranges (documented in formulation_integration.py).
    excipient_library = {}
    X = enumerate_candidates(excipient_library, args.n_candidates, seed)

    log.info("STAGE 2: predict %d responses (scaffold uses synthetic model)", len(RESPONSES))
    pred = _synthetic_responses(X, seed)   # TODO: replace with trained surrogate

    log.info("STAGE 3: constraint filter (pH, viscosity, dose-volume, stability)")
    ok = satisfies_constraints(pred, cons)
    log.info("  %d / %d candidates pass hard constraints", int(ok.sum()), len(X))

    log.info("STAGE 4: desirability scoring (gated on PK exposure window)")
    d = desirability({k: v[ok] for k, v in pred.items()}, obj)
    idx_pass = np.where(ok)[0]
    order = idx_pass[np.argsort(-d)]

    top = []
    for rank, i in enumerate(order[: args.top], 1):
        entry = {"rank": rank, "candidate_index": int(i),
                 "desirability": float(d[np.argsort(-d)][rank - 1]),
                 "factors": {f: float(X[i, j]) for j, f in enumerate(INPUT_FACTORS)},
                 "responses": {r: float(pred[r][i]) for r in pred}}
        top.append(entry)
    log.info("STAGE 5: active-learning acquisition ready (UCB) for next DoE round")

    out = get_path("results", create=True) / "formulation_candidates.json"
    out.write_text(json.dumps({
        "note": "COMPUTATIONAL CANDIDATES requiring experimental validation.",
        "constraints": cons.__dict__,
        "objective": {"maximize": obj.maximize, "minimize": obj.minimize,
                      "pk_target_auc": obj.pk_target_auc},
        "n_enumerated": int(len(X)), "n_pass_constraints": int(ok.sum()),
        "top_candidates": top,
    }, indent=2), encoding="utf-8")
    log.info("wrote %d lead formulation candidates -> %s", len(top), out)
    log.info("DECISION GATE 5 (Section 95): demonstrate the pharmaceutical NEED "
             "vs existing WHO pediatric products before pursuing a new liquid.")


if __name__ == "__main__":
    main()
