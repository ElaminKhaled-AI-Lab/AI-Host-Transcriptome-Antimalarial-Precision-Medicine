"""Pharmaco-omics engine: patient omics x drug candidate -> full ADMET/PK/response.

Given a patient's molecular profile and a drug (approved, investigational, or
preclinical) with a dosage form, predict a full spectrum of properties:
bioavailability, absorption/distribution/metabolism/excretion/toxicity (ADMET),
pharmacokinetic exposure, and personalized drug response.

Design philosophy. There is no public dataset of measured patient-by-drug ADMET,
so the engine combines (i) established structure-based in-silico ADMET rules
(SwissADME/BOILED-Egg style, from molecular descriptors), (ii) the mechanistic
pharmacokinetic model (src/models/pharmacokinetics.py), and (iii) patient-omic
personalization that modulates metabolism, response, and toxicity susceptibility.
An AI surrogate (trained separately) can then learn this mapping for fast,
uncertainty-aware inference and to reveal which patient/drug features drive each
endpoint. All outputs are in-silico decision support, not measured values.

Molecular descriptors expected per drug: MolecularWeight, XLogP (logP), TPSA,
HBondDonorCount, HBondAcceptorCount, Complexity, and (optionally) RotatableBondCount.
These are available from PubChem/ChEMBL for approved drugs and can be computed
from SMILES with RDKit for preclinical candidates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Drug:
    name: str
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    complexity: float = 300.0
    rotatable: int = 5
    pk_class: str = "lumefantrine"   # PK prior to borrow (nearest antimalarial)


@dataclass
class Patient:
    """Patient molecular + clinical proxies (normalised, z-scored where noted)."""
    age_years: float = 3.0
    weight_kg: float = 14.0
    host_metabolic_z: float = 0.0      # host transcriptome/proteome PC proxy for
                                       #   metabolic / CYP capacity (higher -> faster CL)
    host_tox_z: float = 0.0            # host pharmacogene proxy for toxicity susceptibility
    parasite_resistance: float = 0.0   # 0..1 parasite resistance burden (from Pf features)


# ---------------------------------------------------------------- ADMET rules
def _lipinski_violations(d: Drug):
    v = 0
    v += d.mw > 500
    v += d.logp > 5
    v += d.hbd > 5
    v += d.hba > 10
    return int(v)


def _veber_ok(d: Drug):
    return (d.tpsa <= 140) and (d.rotatable <= 10)


def gi_absorption(d: Drug):
    """BOILED-Egg style high GI absorption region (TPSA<=142, logP<=5.9)."""
    return "High" if (d.tpsa <= 142 and d.logp <= 5.9) else "Low"


def bbb_permeant(d: Drug):
    """BOILED-Egg yolk region (roughly TPSA<=79 and 0.4<=logP<=6)."""
    return bool(d.tpsa <= 79 and 0.4 <= d.logp <= 6.0)


def abbott_bioavailability_score(d: Drug):
    """Abbott bioavailability score: 0.55 if passes Lipinski/Veber-like rules, else lower."""
    if _lipinski_violations(d) == 0 and _veber_ok(d):
        return 0.55
    if d.tpsa <= 150:
        return 0.11
    return 0.10


def esol_logS(d: Drug):
    """Delaney ESOL aqueous solubility (logS) from descriptors (approx)."""
    return 0.16 - 0.63 * d.logp - 0.0062 * d.mw + 0.066 * d.rotatable - 0.74 * (d.tpsa / d.mw)


def plasma_protein_binding(d: Drug):
    """Fraction bound, rising with lipophilicity (bounded)."""
    return float(np.clip(0.3 + 0.12 * d.logp, 0.1, 0.995))


def cyp3a4_substrate_likelihood(d: Drug):
    """Heuristic: larger, lipophilic molecules are more often CYP3A4 substrates."""
    z = 0.9 * (d.logp - 2.5) + 0.004 * (d.mw - 350)
    return float(1 / (1 + math.exp(-z)))


def hepatotoxicity_risk(d: Drug):
    z = 0.6 * (d.logp - 3.0) + 0.003 * (d.mw - 400) + 0.4 * (_lipinski_violations(d))
    return float(1 / (1 + math.exp(-z)))


def herg_risk(d: Drug):
    z = 0.8 * (d.logp - 3.5) + 0.002 * (d.mw - 400)
    return float(1 / (1 + math.exp(-z)))


# ------------------------------------------------ formulation + omic personalization
def bioavailable_F(d: Drug, dissolution_pct: float | None):
    """Estimated bioavailable fraction from intrinsic absorption + dosage-form dissolution."""
    base = 0.9 if gi_absorption(d) == "High" else 0.4
    base *= abbott_bioavailability_score(d) / 0.55            # penalise rule failures
    if dissolution_pct is not None:                          # dosage form lifts F
        base *= float(np.clip(0.4 + 0.6 * dissolution_pct / 100.0, 0.2, 1.0))
    # poor solubility caps F
    if esol_logS(d) < -5.5:
        base *= 0.6
    return float(np.clip(base, 0.03, 1.0))


def personalized_clearance_factor(p: Patient):
    """Host metabolic capacity modulates clearance (>1 = faster elimination)."""
    return float(math.exp(0.25 * p.host_metabolic_z))


def personalized_toxicity_factor(p: Patient):
    return float(math.exp(0.3 * p.host_tox_z))


# ---------------------------------------------------------------- full report
def predict(patient: Patient, drug: Drug, dissolution_pct: float | None = None):
    """Return the full predicted spectrum for one patient x drug x dosage form."""
    from src.models.pharmacokinetics import (
        DRUGS, scale_params, PKParams, exposure_metrics, cure_probability)

    F = bioavailable_F(drug, dissolution_pct)
    # PK: borrow the nearest antimalarial PK class, scale by patient covariates + F
    cls = drug.pk_class if drug.pk_class in DRUGS else "lumefantrine"
    CL, V, ka, dose = scale_params(cls, patient.weight_kg, patient.age_years, F=F)
    CL = CL * personalized_clearance_factor(patient)
    pk = exposure_metrics(PKParams(CL=float(CL), V=float(V), ka=float(ka), F=F, dose_mg=float(dose)))
    cure = float(cure_probability(np.array([pk["day7"]]), cls)[0])
    # response personalization: resistance reduces efficacy
    response = float(np.clip(cure * (1 - 0.6 * patient.parasite_resistance), 0, 1))

    return {
        "drug": drug.name,
        "bioavailability": {
            "F_estimate": round(F, 3),
            "GI_absorption": gi_absorption(drug),
            "abbott_score": abbott_bioavailability_score(drug),
            "solubility_logS": round(esol_logS(drug), 2),
        },
        "distribution": {
            "BBB_permeant": bbb_permeant(drug),
            "plasma_protein_bound_frac": round(plasma_protein_binding(drug), 3),
        },
        "metabolism": {
            "CYP3A4_substrate_prob": round(cyp3a4_substrate_likelihood(drug), 3),
            "host_metabolic_factor": round(personalized_clearance_factor(patient), 3),
        },
        "excretion": {
            "CL_L_per_h": round(float(CL), 2),
            "half_life_h": round(float(math.log(2) * V / CL), 1),
        },
        "toxicity": {
            "hepatotoxicity_risk": round(hepatotoxicity_risk(drug) * personalized_toxicity_factor(patient), 3),
            "hERG_risk": round(herg_risk(drug), 3),
            "lipinski_violations": _lipinski_violations(drug),
            "veber_pass": _veber_ok(drug),
        },
        "pharmacokinetics": {
            "AUC_ng_h_per_mL": round(pk["auc"], 1),
            "Cmax_ng_per_mL": round(pk["cmax"], 1),
            "day7_ng_per_mL": round(pk["day7"], 1),
            "tmax_h": round(pk["tmax"], 2),
        },
        "response": {
            "predicted_cure_prob": round(cure, 3),
            "resistance_adjusted_response": round(response, 3),
            "parasite_resistance_burden": round(patient.parasite_resistance, 3),
        },
    }
