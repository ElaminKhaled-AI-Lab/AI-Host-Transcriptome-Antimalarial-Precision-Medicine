"""Nano drug-delivery-system (DDS) digital twin for pediatric antimalarial suspensions.

A documented, literature-anchored mechanistic model of a polymeric NANOPARTICLE
(NP) suspension for oral pediatric antimalarial therapy, with liver-stage
targeting in view. It predicts the full characterization battery from formulation
factors, so the downstream artificial-intelligence (AI) surrogate and optimizer
learn against physically plausible trade-offs. Every relationship is grounded in
established pharmaceutics and in published antimalarial-nanoparticle values; the
output is clearly in-silico decision support for method development, not a
substitute for bench data.

Abbreviations (defined on first use, short form thereafter):
  DDS drug-delivery system; NP nanoparticle; PDI polydispersity index; EE
  entrapment efficiency; DL drug loading; ZP zeta potential; IR immediate
  release; SR sustained release; K-P Korsmeyer-Peppas; ICH International Council
  for Harmonisation; RH relative humidity; logP octanol-water partition
  coefficient; MW molecular weight; AI artificial intelligence.

Mechanistic rationale (documented):
  * Particle size falls with higher stabilizer concentration and higher
    homogenization energy (better emulsification and steric stabilization) and
    rises with a higher polymer-to-drug ratio (more matrix per particle).
  * PDI narrows with stabilizer and energy.
  * ZP is set by the polymer's ionizable groups (chitosan and Eudragit RS/RL
    positive; PLGA, PLA, PCL negative) and shielded toward zero by non-ionic
    stabilizer (PVA/poloxamer).
  * EE rises with polymer-to-drug ratio and drug lipophilicity (logP) and falls
    when excess stabilizer partitions drug into the aqueous phase; DL is the
    inverse trade-off (more polymer dilutes the % load).
  * Release follows Noyes-Whitney surface-area scaling (smaller particles release
    faster) modulated by the polymer's intrinsic degradation/swelling rate; a
    hydrophobic matrix (PLGA/PCL) sustains release, a swellable polymer
    (chitosan/alginate) releases faster with an anomalous mechanism.
  * Liver-targeting index peaks for the ~100-200 nm Kupffer-uptake window and is
    boosted by positive charge / galactosylation (Gomes et al. 2024, AAC
    e00915-23, reported 3-fold higher hepatic half-life and MRT and a liver/plasma
    volume-of-distribution ratio rising from 0.1 to 0.58 in mice for galactosylated
    nanochitosan primaquine; this is a Vd ratio, not a tissue partition
    coefficient). The index is relevant only to
    liver-stage drugs (8-aminoquinolines) and is not used for lumefantrine.
  * Colloidal stability follows DLVO logic: |ZP| < ~20 mV aggregates; growth is
    Arrhenius-accelerated at ICH 40 C / 75% RH.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

# --------------------------------------------------------------------------- #
#  FDA-approved / GRAS polymer + stabilizer library (documented priors).      #
#  base_size_nm : characteristic NP size at the central factor setting.       #
#  zeta_mv      : intrinsic surface charge sign/magnitude (bare NP).          #
#  degrad       : relative matrix erosion/diffusion rate (higher = faster).   #
#  peppas_n0    : central K-P exponent prior (mechanism fingerprint).         #
#  release_class: IR (immediate) vs SR (sustained) tendency.                  #
# --------------------------------------------------------------------------- #
POLYMERS: dict[str, dict] = {
    "PLGA_50_50":   dict(base_size_nm=250, zeta_mv=-20, degrad=1.00, peppas_n0=0.45, release_class="SR", pdi_base=0.15),
    "PLGA_75_25":   dict(base_size_nm=210, zeta_mv=-18, degrad=0.70, peppas_n0=0.43, release_class="SR", pdi_base=0.16),
    "PLA_PDLLA":    dict(base_size_nm=220, zeta_mv=-15, degrad=0.55, peppas_n0=0.43, release_class="SR", pdi_base=0.17),
    "PCL":          dict(base_size_nm=230, zeta_mv=-12, degrad=0.40, peppas_n0=0.42, release_class="SR", pdi_base=0.18),
    "Chitosan":     dict(base_size_nm=430, zeta_mv=+31, degrad=1.60, peppas_n0=0.62, release_class="IR", pdi_base=0.48),
    "Eudragit_RS100": dict(base_size_nm=190, zeta_mv=+22, degrad=0.80, peppas_n0=0.50, release_class="SR", pdi_base=0.24),
    "Eudragit_RL100": dict(base_size_nm=180, zeta_mv=+28, degrad=1.10, peppas_n0=0.55, release_class="IR", pdi_base=0.26),
    "Eudragit_S100":  dict(base_size_nm=200, zeta_mv=-25, degrad=1.30, peppas_n0=0.58, release_class="IR", pdi_base=0.28),
    "Alginate":     dict(base_size_nm=280, zeta_mv=-35, degrad=1.50, peppas_n0=0.60, release_class="IR", pdi_base=0.42),
    "Ethylcellulose": dict(base_size_nm=240, zeta_mv=-10, degrad=0.45, peppas_n0=0.42, release_class="SR", pdi_base=0.20),
    "HPMC":         dict(base_size_nm=260, zeta_mv=-8,  degrad=1.40, peppas_n0=0.70, release_class="IR", pdi_base=0.45),
}

# Non-ionic / ionic stabilizers: charge-shielding strength toward zero ZP.
STABILIZERS: dict[str, dict] = {
    "PVA":          dict(shield=0.55, size_factor=1.00),
    "Poloxamer_188": dict(shield=0.50, size_factor=0.95),
    "Tween_80":     dict(shield=0.45, size_factor=0.98),
    "Lecithin":     dict(shield=0.30, size_factor=1.02),
}

# Representative antimalarial drug descriptors (MW g/mol, logP) from the
# literature / PubChem; used for lipophilicity-driven EE. Override on the
# workstation with data/raw/formulation/PubChem/antimalarial_physicochemical.json.
DRUGS: dict[str, dict] = {
    "lumefantrine":       dict(mw=528.9, logp=8.7),
    "artemether":         dict(mw=298.4, logp=3.1),
    "artesunate":         dict(mw=384.4, logp=2.6),
    "dihydroartemisinin": dict(mw=284.4, logp=2.4),
    "primaquine":         dict(mw=259.4, logp=2.8),
    "atovaquone":         dict(mw=366.8, logp=5.2),
    "piperaquine":        dict(mw=535.5, logp=6.3),
    "amodiaquine":        dict(mw=355.9, logp=3.7),
    "mefloquine":         dict(mw=378.3, logp=3.9),
}


@dataclass
class Formulation:
    """One nanoparticle-suspension formulation (factor settings + choices)."""
    drug: str = "lumefantrine"
    polymer: str = "PLGA_50_50"
    stabilizer: str = "PVA"
    # --- the three full-factorial factors, given as CODED levels in {-1,0,+1} ---
    polymer_drug_ratio: int = 0     # A: 2:1 (-1) / 5:1 (0) / 10:1 (+1)
    stabilizer_pct: int = 0         # B: 0.5% (-1) / 1.0% (0) / 2.0% (+1)
    homogenization: int = 0         # C: low (-1) / medium (0) / high (+1)
    galactosylated: bool = False    # optional liver-targeting ligand


# Coded-level -> real value maps (documented factor ranges).
RATIO_MAP = {-1: 2.0, 0: 5.0, 1: 10.0}       # polymer:drug (w/w)
STAB_MAP  = {-1: 0.5, 0: 1.0, 1: 2.0}        # % w/v
ENERGY_MAP = {-1: 0.30, 0: 0.60, 1: 1.00}    # relative homogenization energy


def _drug_desc(name: str) -> dict:
    return DRUGS.get(name.lower(), dict(mw=400.0, logp=3.0))


# --------------------------------------------------------------------------- #
#  Characterization predictors                                                #
# --------------------------------------------------------------------------- #
def predict_size(f: Formulation) -> float:
    """Particle size (nm): polymer base, up with ratio, down with stabilizer/energy."""
    p = POLYMERS[f.polymer]
    ratio, stab, energy = RATIO_MAP[f.polymer_drug_ratio], STAB_MAP[f.stabilizer_pct], ENERGY_MAP[f.homogenization]
    size = p["base_size_nm"]
    size *= (ratio / 5.0) ** 0.35                      # more polymer -> larger
    size *= (1.0 - 0.22 * math.log10(stab / 0.5 + 1))  # more stabilizer -> smaller
    size *= (1.0 - 0.30 * (energy - 0.30))             # more energy -> smaller
    size *= STABILIZERS[f.stabilizer]["size_factor"]
    return float(np.clip(size, 40, 800))


def predict_pdi(f: Formulation) -> float:
    """PDI: polymer-set base, narrowed by stabilizer and homogenization energy.

    Swellable polysaccharides (chitosan, alginate, HPMC) run broad; PLGA/PLA/PCL
    emulsion-solvent-evaporation systems run narrow -- matching published PDI.
    """
    p = POLYMERS[f.polymer]
    stab, energy = STAB_MAP[f.stabilizer_pct], ENERGY_MAP[f.homogenization]
    pdi = p["pdi_base"] - 0.06 * math.log10(stab / 0.5 + 1) - 0.10 * (energy - 0.30)
    return float(np.clip(pdi, 0.05, 0.65))


def predict_zeta(f: Formulation) -> float:
    """Zeta potential (mV): polymer sign/magnitude shielded by non-ionic stabilizer."""
    p = POLYMERS[f.polymer]
    shield = STABILIZERS[f.stabilizer]["shield"] * (STAB_MAP[f.stabilizer_pct] / 1.0)
    zeta = p["zeta_mv"] * (1.0 - 0.35 * shield)
    if f.galactosylated:                                # galactose ligand adds slight + charge
        zeta += 4.0
    return float(zeta)


def predict_ee(f: Formulation) -> float:
    """Entrapment efficiency (%): up with polymer ratio and drug logP, down with excess stabilizer."""
    d = _drug_desc(f.drug)
    ratio, stab = RATIO_MAP[f.polymer_drug_ratio], STAB_MAP[f.stabilizer_pct]
    ee = 100.0 / (1.0 + math.exp(-(0.20 * (ratio - 4.0) + 0.35 * (d["logp"] - 3.0))))
    ee *= (1.0 - 0.12 * math.log10(stab / 0.5 + 1))    # excess stabilizer leaches drug
    return float(np.clip(ee, 5, 98))


def predict_dl(f: Formulation) -> float:
    """Drug loading (%): inverse trade-off with polymer:drug ratio."""
    ee = predict_ee(f)
    ratio = RATIO_MAP[f.polymer_drug_ratio]
    dl = ee / 100.0 * (1.0 / (1.0 + ratio)) * 100.0
    return float(np.clip(dl, 0.5, 50))


def release_profile(f: Formulation, times_h=None) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative % release Q(t). Underlying Weibull driven by size + polymer erosion.

    Smaller particles (Noyes-Whitney surface area) and faster-eroding polymers
    reach plateau sooner; hydrophobic matrices (PLGA/PCL) sustain release.
    """
    if times_h is None:
        times_h = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24, 48, 72], float)
    size = predict_size(f)
    p = POLYMERS[f.polymer]
    # rate constant: faster for small size + high erosion + high stabilizer wetting
    k = p["degrad"] * (200.0 / size) ** 0.5 * (0.8 + 0.2 * STAB_MAP[f.stabilizer_pct])
    scale = 6.0 / k                                     # Weibull time-scale (h)
    shape = 0.6 + 0.25 * p["peppas_n0"] * 2             # Weibull shape ~ mechanism
    qmax = float(np.clip(predict_ee(f) + 5, 60, 100))   # accessible fraction
    Q = qmax * (1.0 - np.exp(-((times_h / scale) ** shape)))
    return times_h, np.clip(Q, 0, 100)


def liver_targeting_index(f: Formulation) -> float:
    """0-1 index for hepatic delivery via two mechanisms.

    (i) Passive Kupffer-cell uptake: peaks in the ~100-200 nm size window and is
        aided by positive surface charge. (ii) Active, size-independent
        hepatocyte targeting via a galactose ligand binding the asialoglycoprotein
        receptor (Gomes et al. 2024, AAC e00915-23), so galactosylation dominates
        when present. Model construct; relevant to liver-stage drugs only.
    """
    size = predict_size(f)
    win = math.exp(-((size - 150.0) / 90.0) ** 2)              # Kupffer size window
    charge = 1.0 / (1.0 + math.exp(-(predict_zeta(f)) / 12.0))  # + charge favors uptake
    idx = win * (0.5 + 0.5 * charge)
    if f.galactosylated:                                        # receptor-mediated, size-independent
        idx = 0.45 + 0.55 * max(idx, charge)
    return float(np.clip(idx, 0, 1))


def liver_plasma_ratio(f: Formulation) -> float:
    """Model-construct liver/plasma ratio, loosely anchored to Gomes et al. 2024.

    Gomes et al. (AAC e00915-23) report a liver/plasma ratio of the VOLUME OF
    DISTRIBUTION of 0.1 (free primaquine) and 0.58 (galactosylated nanochitosan)
    in mice. That is not a concentration partition ratio; the ceilings below are
    assumptions and this function is not used in any reported analysis.
    """
    idx = liver_targeting_index(f)
    ceiling = 0.65 if f.galactosylated else 0.45
    return float(round(0.10 + (ceiling - 0.10) * idx, 3))


# --------------------------------------------------------------------------- #
#  Release-kinetics model fitting                                             #
# --------------------------------------------------------------------------- #
def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def _aic(y, yhat, k_params):
    n = len(y)
    rss = np.sum((y - yhat) ** 2) + 1e-12
    return n * math.log(rss / n) + 2 * k_params


def fit_release_kinetics(times_h: np.ndarray, Q: np.ndarray) -> dict:
    """Fit zero-order, first-order, Higuchi, Korsmeyer-Peppas, Hixson-Crowell.

    Returns each model's rate constant, R2 and AIC, the best model by AIC, and
    the K-P exponent n with its transport-mechanism interpretation (spheres:
    n<=0.43 Fickian diffusion; 0.43<n<0.85 anomalous/non-Fickian; n>=0.85
    case-II / super case-II).
    """
    t = np.asarray(times_h, float)
    q = np.asarray(Q, float)
    m = (t > 0)
    t, q = t[m], q[m]
    qfrac = q / 100.0
    out: dict = {"models": {}}

    # zero-order: Q = k0 t
    k0 = np.sum(t * q) / np.sum(t * t)
    out["models"]["zero_order"] = {"k": float(k0), "r2": float(_r2(q, k0 * t)),
                                   "aic": float(_aic(q, k0 * t, 1))}
    # first-order: Q = 100(1 - e^{-k1 t})
    try:
        popt, _ = curve_fit(lambda tt, k1: 100 * (1 - np.exp(-k1 * tt)), t, q,
                            p0=[0.1], maxfev=10000)
        yh = 100 * (1 - np.exp(-popt[0] * t))
        out["models"]["first_order"] = {"k": float(popt[0]), "r2": float(_r2(q, yh)),
                                        "aic": float(_aic(q, yh, 1))}
    except Exception:
        out["models"]["first_order"] = {"k": None, "r2": -1, "aic": 1e9}
    # Higuchi: Q = kH sqrt(t)
    kh = np.sum(np.sqrt(t) * q) / np.sum(t)
    yh = kh * np.sqrt(t)
    out["models"]["higuchi"] = {"k": float(kh), "r2": float(_r2(q, yh)),
                                "aic": float(_aic(q, yh, 1))}
    # Korsmeyer-Peppas: Q/Q_inf = k t^n  (fit on <=60% release)
    sel = qfrac <= 0.60
    if sel.sum() >= 3:
        try:
            popt, _ = curve_fit(lambda tt, k, n: k * tt ** n, t[sel], qfrac[sel],
                                p0=[0.2, 0.5], maxfev=10000,
                                bounds=([1e-4, 0.1], [5, 1.5]))
            yh = popt[0] * t[sel] ** popt[1]
            n_exp = float(popt[1])
            out["models"]["korsmeyer_peppas"] = {"k": float(popt[0]), "n": n_exp,
                                                 "r2": float(_r2(qfrac[sel], yh)),
                                                 "aic": float(_aic(qfrac[sel], yh, 2))}
        except Exception:
            n_exp = None
            out["models"]["korsmeyer_peppas"] = {"k": None, "n": None, "r2": -1, "aic": 1e9}
    else:
        n_exp = None
        out["models"]["korsmeyer_peppas"] = {"k": None, "n": None, "r2": -1, "aic": 1e9}
    # Hixson-Crowell: (100)^{1/3} - Q_remaining^{1/3} = k t
    rem = np.clip(100 - q, 1e-6, None)
    lhs = 100 ** (1 / 3) - rem ** (1 / 3)
    khc = np.sum(t * lhs) / np.sum(t * t)
    yh = khc * t
    out["models"]["hixson_crowell"] = {"k": float(khc), "r2": float(_r2(lhs, yh)),
                                       "aic": float(_aic(lhs, yh, 1))}

    # best model by AIC (lower is better)
    best = min(out["models"].items(), key=lambda kv: kv[1].get("aic", 1e9))
    out["best_model"] = best[0]
    out["peppas_n"] = n_exp
    out["mechanism"] = interpret_peppas_n(n_exp)
    return out


def interpret_peppas_n(n) -> str:
    """Transport mechanism from the K-P exponent for spherical particles."""
    if n is None:
        return "undetermined"
    if n <= 0.43:
        return "Fickian diffusion"
    if n < 0.85:
        return "anomalous (non-Fickian) transport"
    if abs(n - 0.85) < 0.02:
        return "case-II transport (erosion-controlled)"
    return "super case-II transport"


# --------------------------------------------------------------------------- #
#  Accelerated stability (ICH Q1A: 40 C / 75% RH; Arrhenius)                  #
# --------------------------------------------------------------------------- #
def accelerated_stability(f: Formulation, months=6, ea_kj=80.0) -> dict:
    """Predict NP growth over ICH accelerated storage and estimate shelf-life.

    Growth rate rises when |ZP| is low (poor colloidal repulsion) and is
    Arrhenius-accelerated at 40 C vs 25 C. Shelf-life = time to +10% size or
    -10% EE, whichever comes first; a 25 C long-term rate is back-extrapolated.
    """
    R = 8.314
    zeta_abs = abs(predict_zeta(f))
    base_monthly = 0.06 * math.exp(-(zeta_abs - 20) / 15)     # fractional size growth /month at 25 C
    base_monthly = float(np.clip(base_monthly, 0.005, 0.5))
    # Arrhenius acceleration factor 25 C (298 K) -> 40 C (313 K)
    accel = math.exp(-ea_kj * 1000 / R * (1 / 313.15 - 1 / 298.15))
    monthly_40 = base_monthly * accel
    t = np.arange(0, months + 1)
    size0 = predict_size(f)
    ee0 = predict_ee(f)
    size_t = size0 * (1 + monthly_40) ** t
    ee_t = ee0 * (1 - 0.5 * monthly_40) ** t
    # shelf-life at 25 C (long-term): time to +10% size
    shelf_25_months = math.log(1.10) / math.log(1 + base_monthly) if base_monthly > 0 else 99
    # Revision: Climatic Zone IVb long-term condition (30 C / 75% RH), the
    # storage condition relevant to deployment in sub-Saharan Africa.
    accel_30 = math.exp(-ea_kj * 1000 / R * (1 / 303.15 - 1 / 298.15))
    monthly_30 = base_monthly * accel_30
    shelf_30_months = math.log(1.10) / math.log(1 + monthly_30) if monthly_30 > 0 else 99
    return {
        "monthly_growth_25C": round(base_monthly, 4),
        "monthly_growth_30C": round(monthly_30, 4),
        "monthly_growth_40C": round(monthly_40, 4),
        "size_trajectory_40C_nm": [round(float(x), 1) for x in size_t],
        "ee_trajectory_40C_pct": [round(float(x), 1) for x in ee_t],
        "predicted_shelf_life_25C_months": round(float(np.clip(shelf_25_months, 0.1, 60)), 2),
        "predicted_shelf_life_30C_months": round(float(np.clip(shelf_30_months, 0.1, 60)), 2),
        "activation_energy_kJ_per_mol": ea_kj,
        "note": "Indicative only: ICH Q1E does not permit shelf-life assignment from "
                "accelerated data alone; the growth model depends on |zeta| only.",
        "colloidally_stable": bool(zeta_abs >= 20),
    }


# --------------------------------------------------------------------------- #
#  Full characterization report for one formulation                          #
# --------------------------------------------------------------------------- #
def characterize(f: Formulation) -> dict:
    t, Q = release_profile(f)
    kin = fit_release_kinetics(t, Q)
    return {
        "drug": f.drug, "polymer": f.polymer, "stabilizer": f.stabilizer,
        "factors": {"polymer_drug_ratio": RATIO_MAP[f.polymer_drug_ratio],
                    "stabilizer_pct": STAB_MAP[f.stabilizer_pct],
                    "homogenization_energy": ENERGY_MAP[f.homogenization],
                    "galactosylated": f.galactosylated},
        "size_nm": round(predict_size(f), 1),
        "pdi": round(predict_pdi(f), 3),
        "zeta_mv": round(predict_zeta(f), 1),
        "ee_pct": round(predict_ee(f), 1),
        "dl_pct": round(predict_dl(f), 2),
        "release_24h_pct": round(float(np.interp(24, t, Q)), 1),
        "release_best_model": kin["best_model"],
        "peppas_n": None if kin["peppas_n"] is None else round(kin["peppas_n"], 3),
        "release_mechanism": kin["mechanism"],
        "liver_targeting_index": round(liver_targeting_index(f), 3),
        "liver_plasma_ratio": liver_plasma_ratio(f),
        "release_class": POLYMERS[f.polymer]["release_class"],
    }
