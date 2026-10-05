"""Pharmacokinetic / pharmacodynamic (PK/PD) core for antimalarial exposure.

A compact, documented compartmental model with allometric + maturation covariate
scaling, exposure metrics, and an Emax pharmacodynamic (PD) link to parasite
clearance and cure. It is the mechanistic backbone the AI PK module learns from
and augments (blueprint Sections 31, 35.9, 68).

Clinical grounding (illustrative priors, not a fitted model):
  * For lumefantrine and piperaquine the day-7 plasma concentration is the
    established predictor of cure (Tarning 2012; PQ IPD meta-analysis PLOS Med
    2016; lumefantrine paediatric popPK JID 2016).
  * Clearance scales allometrically with body weight (exponent 0.75) and volume
    linearly (exponent 1), with an age-based maturation function for young
    children.
Real parameter estimation requires the managed IDDO/WWARN individual PK data;
the priors here reproduce realistic day-7 concentrations for an in-silico study.

Abbreviations: CL clearance; V volume of distribution; ka absorption rate; ke
elimination rate; F bioavailable fraction; AUC area under the concentration-time
curve; Cmax peak concentration; IIV inter-individual variability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Representative paediatric antimalarial priors (per 70 kg reference adult).
# day7_target = concentration (ng/mL) associated with cure in the literature.
# Apparent oral parameters (CL/F, V/F) chosen so the long terminal half-life and
# day-7 concentrations match the antimalarial literature: lumefantrine t1/2 ~4 d
# and day-7 ~200 ng/mL; piperaquine very large apparent V and t1/2 ~2-4 wk with
# day-7 ~30 ng/mL. Illustrative priors, not a fitted popPK model.
DRUGS = {
    "lumefantrine": dict(CL=1.6, V=222.0, ka=0.30, dose_mg=480.0, day7_target=200.0),
    "piperaquine":  dict(CL=1.0, V=30000.0, ka=0.55, dose_mg=960.0, day7_target=30.0),
    "dihydroartemisinin": dict(CL=50.0, V=100.0, ka=1.5, dose_mg=120.0, day7_target=None),
}
WT_REF = 70.0


@dataclass
class PKParams:
    CL: float          # L/h
    V: float           # L
    ka: float          # 1/h
    F: float = 1.0     # bioavailable fraction (formulation-dependent)
    dose_mg: float = 480.0

    @property
    def ke(self):
        return self.CL / self.V


def maturation(age_years):
    """Sigmoidal enzyme-maturation factor in (0,1], ~0.5 near birth -> 1 by ~2-3 y.

    A standard Hill maturation model on post-natal age; reduces clearance in the
    youngest children, which is clinically important for antimalarial exposure.
    """
    a = np.asarray(age_years, float)
    tm50, hill = 0.5, 3.0                     # 50% maturation ~6 months
    return (a ** hill) / (tm50 ** hill + a ** hill) * 0.5 + 0.5


def scale_params(drug: str, weight_kg, age_years, F=1.0):
    """Allometric + maturation covariate scaling of the population priors."""
    p = DRUGS[drug]
    w = np.asarray(weight_kg, float)
    CL = p["CL"] * (w / WT_REF) ** 0.75 * maturation(age_years)
    V = p["V"] * (w / WT_REF)
    return CL, V, p["ka"], p["dose_mg"]


def conc_time(t_h, params: PKParams):
    """One-compartment oral concentration (ng/mL) at times t_h (hours).

    c(t) = F*Dose*ka / (V*(ka-ke)) * (exp(-ke t) - exp(-ka t)).
    Dose mg, V L -> mg/L = ug/mL; *1000 -> ng/mL.
    """
    t = np.asarray(t_h, float)
    ke = params.ke
    ka = params.ka
    if abs(ka - ke) < 1e-9:
        ka = ke * 1.001
    c_mg_per_L = (params.F * params.dose_mg * ka) / (params.V * (ka - ke)) \
        * (np.exp(-ke * t) - np.exp(-ka * t))
    return np.clip(c_mg_per_L, 0, None) * 1000.0    # ng/mL


def exposure_metrics(params: PKParams):
    """AUC_inf (ng*h/mL), Cmax (ng/mL), tmax (h), day-7 concentration (ng/mL)."""
    ke, ka = params.ke, params.ka
    auc = params.F * params.dose_mg / params.CL * 1000.0        # ng*h/mL
    if abs(ka - ke) < 1e-9:
        ka = ke * 1.001
    tmax = np.log(ka / ke) / (ka - ke)
    cmax = conc_time(tmax, params)
    day7 = conc_time(168.0, params)
    return dict(auc=float(auc), cmax=float(cmax), tmax=float(tmax), day7=float(day7))


def simulate_population(drug, weight_kg, age_years, F=None, iiv_cl=0.3, iiv_v=0.3,
                        seed=11):
    """Simulate exposures for a population with log-normal IIV on CL and V.

    Returns a dict of arrays: auc, cmax, day7, CL, V, F. ``F`` may be an array
    (formulation-dependent bioavailability) or None (=1).
    """
    rng = np.random.default_rng(seed)
    n = len(np.atleast_1d(weight_kg))
    CL, V, ka, dose = scale_params(drug, weight_kg, age_years)
    CL = CL * np.exp(rng.normal(0, iiv_cl, n))
    V = V * np.exp(rng.normal(0, iiv_v, n))
    Farr = np.ones(n) if F is None else np.clip(np.asarray(F, float), 0.05, 1.0)
    out = {k: np.zeros(n) for k in ("auc", "cmax", "day7")}
    for i in range(n):
        p = PKParams(CL=CL[i], V=V[i], ka=ka, F=Farr[i], dose_mg=dose)
        m = exposure_metrics(p)
        for k in out:
            out[k][i] = m[k]
    out.update(CL=CL, V=V, F=Farr, ka=np.full(n, ka))
    return out


# ---------------------------------------------------------------- PD link
def emax_kill(exposure, emax=0.99, ec50=None):
    """Fraction of maximal parasite kill as an Emax function of exposure."""
    e = np.asarray(exposure, float)
    if ec50 is None:
        ec50 = np.nanmedian(e)
    return emax * e / (ec50 + e)


def cure_probability(day7, drug, steepness=0.03):
    """Logistic cure probability around the drug's day-7 target concentration.

    Above the target -> high cure; below -> falling cure, matching the clinical
    use of day-7 concentration as the exposure predictor of outcome.
    """
    tgt = DRUGS[drug]["day7_target"]
    if tgt is None:
        tgt = np.nanmedian(day7)
    return 1.0 / (1.0 + np.exp(-steepness * (np.asarray(day7, float) - tgt)))
