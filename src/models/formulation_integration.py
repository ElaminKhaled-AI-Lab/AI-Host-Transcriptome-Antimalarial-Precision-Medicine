"""Pharmaceutical integration + AI de novo formulation design.

The formulation analog of the group's cluster-guided de novo PROTAC pipeline
(BLISS paper Sections 2.13-2.14): the molecular arm produces a candidate
antimalarial + a target pharmacokinetic (PK) exposure; this module then designs
the pediatric oral dosage form that best hits that target.

The clinical gap this addresses: there is still no adequate age-appropriate
LIQUID dosage form for several pediatric antimalarials, so children rely on
crushed/split adult tablets or dispersibles. We use AI/ML to search excipient +
process space for a stable, palatable, dose-uniform liquid or suspension.

Design mirrors the PROTAC pipeline stage-for-stage:
  PROTAC de novo                        Formulation de novo (this module)
  ---------------------------------     ------------------------------------------
  BRICS fragmentation of PROTAC DB  ->  enumerate excipient/process combinations
  warhead-anchored reassembly       ->  anchor on the fixed API + dose target
  2D/3D shape filter (Tanimoto/USRCAT) -> constraint filter (pH, viscosity, dose-volume)
  Glide docking score               ->  dissolution / redispersibility surrogate
  MM-GBSA binding free energy       ->  predicted PK exposure (AUC) surrogate
  QikProp ADMET                     ->  stability + excipient-safety precedent (FDA IID)
  rank -> lead compound             ->  Bayesian multi-objective -> lead formulation

"Pharmaceutical integration" = jointly modelling composition + material
properties + process + drug physicochemistry to predict multi-response
performance, instead of one-factor-at-a-time development (blueprint Sections
62-68). Uncertainty-guided active learning proposes the next experiments.

Uses scikit-learn (+ optional botorch/gpytorch for Bayesian optimization).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Formulation input factors (blueprint Section 63).
INPUT_FACTORS = [
    "api_concentration", "fixed_dose_ratio", "suspending_agent_id",
    "suspending_agent_conc", "surfactant_id", "surfactant_conc", "wetting_agent_conc",
    "buffer_id", "ph", "sweetener_conc", "flavor_id", "preservative_conc",
    "vehicle_id", "viscosity_modifier_conc", "particle_size_d50",
    "particle_size_span", "zeta_potential", "mixing_speed_rpm", "mixing_time_min",
    "milling_energy", "storage_temp_c",
]
# Performance responses (blueprint Section 64).
RESPONSES = [
    "sedimentation_volume_ratio", "redispersibility_strokes", "viscosity_cp",
    "dissolution_pct_30min", "content_uniformity_rsd", "delivered_dose_rsd",
    "chemical_assay_pct_t0", "chemical_assay_pct_accel", "palatability_score",
    "administration_volume_ml", "predicted_pk_auc",
]


@dataclass
class FormulationConstraints:
    """Hard constraints a candidate must satisfy (blueprint Section 67)."""
    ph_range: tuple[float, float] = (4.0, 7.5)
    viscosity_cp_max: float = 2500.0
    dose_volume_ml_max: float = 10.0            # pediatric acceptability
    redispersibility_strokes_max: int = 20
    chemical_assay_pct_accel_min: float = 90.0  # >=90% API after accelerated storage
    content_uniformity_rsd_max: float = 5.0     # %RSD


@dataclass
class Objective:
    """Multi-objective target (maximize good, respect exposure window)."""
    maximize: list[str] = field(default_factory=lambda: [
        "dissolution_pct_30min", "palatability_score", "chemical_assay_pct_accel"])
    minimize: list[str] = field(default_factory=lambda: [
        "sedimentation_volume_ratio", "delivered_dose_rsd", "administration_volume_ml"])
    pk_target_auc: float | None = None          # from the molecular/PK arm
    pk_tolerance: float = 0.2                     # +/- 20% of target


# --------------------------------------------------------------- surrogates
def build_surrogate(kind: str = "gaussian_process", seed: int = 11):
    """Return a multi-output regressor surrogate for formulation responses.

    GP is preferred for small experimental datasets (blueprint Section 66): it
    gives calibrated uncertainty for the active-learning acquisition step.
    """
    from src.models.baselines import make_regressor
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(make_regressor(kind, seed=seed))


# --------------------------------------------------------------- enumeration
def enumerate_candidates(excipient_library: dict, n: int, seed: int = 11) -> np.ndarray:
    """Sample n candidate formulations from the excipient/process design space.

    ``excipient_library`` provides allowed categorical ids and numeric ranges,
    typically built from the FDA Inactive Ingredient Database (precedented
    excipients + max potency per route) plus process limits. This is the
    formulation analog of BRICS enumeration over the PROTAC database.
    """
    rng = np.random.default_rng(seed)
    cols = []
    for f in INPUT_FACTORS:
        spec = excipient_library.get(f)
        if spec is None:
            cols.append(rng.random(n))                       # placeholder [0,1]
        elif spec.get("type") == "categorical":
            cols.append(rng.integers(0, len(spec["choices"]), n).astype(float))
        else:
            lo, hi = spec.get("range", (0.0, 1.0))
            cols.append(rng.uniform(lo, hi, n))
    return np.column_stack(cols)


def satisfies_constraints(pred: dict, cons: FormulationConstraints) -> np.ndarray:
    """Boolean mask of candidates passing all hard constraints (the shape filter)."""
    ok = np.ones(len(next(iter(pred.values()))), dtype=bool)
    def col(name):
        return np.asarray(pred[name]) if name in pred else None
    checks = {
        "ph": lambda v: (v >= cons.ph_range[0]) & (v <= cons.ph_range[1]),
        "viscosity_cp": lambda v: v <= cons.viscosity_cp_max,
        "administration_volume_ml": lambda v: v <= cons.dose_volume_ml_max,
        "redispersibility_strokes": lambda v: v <= cons.redispersibility_strokes_max,
        "chemical_assay_pct_accel": lambda v: v >= cons.chemical_assay_pct_accel_min,
        "content_uniformity_rsd": lambda v: v <= cons.content_uniformity_rsd_max,
    }
    for name, fn in checks.items():
        v = col(name)
        if v is not None:
            ok &= fn(v)
    return ok


# --------------------------------------------------------------- scoring
def desirability(pred: dict, obj: Objective) -> np.ndarray:
    """Composite desirability in [0,1] (Derringer-style), with PK-window gate.

    Reported ALONGSIDE component responses, never instead of them (mirrors the
    paper's rule of reporting component predictions, not one opaque score).
    """
    n = len(next(iter(pred.values())))
    parts = []
    for name in obj.maximize:
        if name in pred:
            parts.append(_minmax(np.asarray(pred[name])))
    for name in obj.minimize:
        if name in pred:
            parts.append(1.0 - _minmax(np.asarray(pred[name])))
    d = np.prod(parts, axis=0) ** (1.0 / max(len(parts), 1)) if parts else np.zeros(n)
    if obj.pk_target_auc is not None and "predicted_pk_auc" in pred:
        auc = np.asarray(pred["predicted_pk_auc"])
        within = np.abs(auc - obj.pk_target_auc) <= obj.pk_tolerance * obj.pk_target_auc
        d = d * within                                       # gate on exposure window
    return d


def acquisition_ucb(mean_desir: np.ndarray, std_desir: np.ndarray, kappa: float = 1.5):
    """Upper-confidence-bound acquisition for active learning (Section 65)."""
    return mean_desir + kappa * std_desir


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo + 1e-9)
