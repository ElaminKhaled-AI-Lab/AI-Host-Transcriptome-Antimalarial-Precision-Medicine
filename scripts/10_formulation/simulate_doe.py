#!/usr/bin/env python
"""Phase 18 -- In-silico Design of Experiments (DoE) for a pediatric antimalarial
suspension (blueprint Sections 62-68, Goal B).

No open antimalarial-suspension dataset exists (Section 21 Stage 7), so this
builds a DOCUMENTED physical DIGITAL TWIN of an oral suspension and runs a DoE
over it. The response model encodes real pharmaceutical mechanisms so the
downstream AI optimises against physically plausible trade-offs -- it is clearly
labelled in-silico and is for method development, not a substitute for lab data.

Physical rationale baked into the responses (all documented):
  * Stokes' law: sedimentation rate ∝ d^2·Δρ/η -> larger particles + lower
    viscosity settle faster; a higher suspending-agent concentration raises
    viscosity and the sedimentation volume ratio F (good) but too high hurts
    redispersibility and pourability.
  * Noyes-Whitney: dissolution ∝ surface area ∝ 1/d -> smaller particles and a
    wetting surfactant raise 30-min dissolution.
  * Caking/redispersibility worsens with large particles and low suspending agent.
  * Artemisinin chemical stability is worst at extreme pH and high temperature.
  * Palatability rises with sweetener, falls with bitter API load.
  * Predicted PK exposure (AUC) tracks the fraction dissolved/absorbed.

Design: Latin Hypercube over the factor ranges (space-filling screening design),
plus optional centre + axial points. Output is the "experimental" dataset the AI
surrogate/optimiser learns from.

Output -> data/formulation/doe_dataset.tsv  (+ factor_ranges.json)
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.provenance import result_manifest
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Factor ranges (name -> (low, high, unit/notes)). A pragmatic subset of Section 63.
FACTORS = {
    "api_conc_mg_ml":        (5.0, 40.0),    # active pharmaceutical ingredient load
    "suspending_agent_pct":  (0.1, 1.5),     # e.g. xanthan gum
    "surfactant_pct":        (0.0, 0.5),     # wetting agent (polysorbate 80)
    "particle_size_d50_um":  (2.0, 50.0),    # milled particle size
    "ph":                    (3.0, 8.0),
    "sweetener_pct":         (0.0, 15.0),     # sucrose/sorbitol
    "viscosity_modifier_pct":(0.0, 1.0),
    "milling_energy":        (0.0, 1.0),      # normalised
    "storage_temp_c":        (5.0, 40.0),
}


def _n(x, lo, hi):
    return (x - lo) / (hi - lo)             # normalise to [0,1]


def latin_hypercube(n, d, rng):
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    a, b = cut[:n][:, None], cut[1:][:, None]
    pts = a + u * (b - a)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def simulate(df: pd.DataFrame, rng) -> pd.DataFrame:
    f = {k: df[k].to_numpy(float) for k in FACTORS}
    # normalised drivers
    sa = _n(f["suspending_agent_pct"], 0.1, 1.5)
    vm = _n(f["viscosity_modifier_pct"], 0.0, 1.0)
    d50 = f["particle_size_d50_um"]
    d_norm = _n(d50, 2.0, 50.0)
    surf = _n(f["surfactant_pct"], 0.0, 0.5)
    sweet = _n(f["sweetener_pct"], 0.0, 15.0)
    api = _n(f["api_conc_mg_ml"], 5.0, 40.0)
    ph = f["ph"]
    temp = _n(f["storage_temp_c"], 5.0, 40.0)
    mill = f["milling_energy"]
    noise = lambda s: rng.normal(0, s, len(df))

    # viscosity (cP): power-law rise with suspending agent + modifier
    visc = 50 + 2500 * (sa ** 1.3) + 800 * vm + noise(40)
    # sedimentation volume ratio F in (0,1]: higher viscosity + smaller particle -> better
    sed_rate = (d50 ** 2) / (visc + 1e-6)          # Stokes proxy
    F = np.clip(1.0 - 0.02 * _n(sed_rate, sed_rate.min(), sed_rate.max()) * 30
                + 0.5 * sa - 0.3 * d_norm + noise(0.03), 0.05, 1.0)
    # redispersibility (strokes): worse with large particle + low suspending agent
    redisp = np.clip(4 + 25 * d_norm * (1 - sa) + noise(1.5), 1, 40)
    # dissolution % at 30 min: Noyes-Whitney (1/d) + wetting surfactant
    dissolution = np.clip(100 * (0.35 + 0.5 * (1 - d_norm) + 0.2 * surf - 0.1 * api)
                          + noise(4), 5, 100)
    # content / delivered-dose uniformity RSD (%): worse when it settles/cakes
    cu_rsd = np.clip(2 + 8 * (1 - F) + 0.1 * redisp + noise(0.8), 1, 20)
    dd_rsd = np.clip(cu_rsd + 1.0 * (1 - F) + noise(0.6), 1, 25)
    # chemical assay after accelerated storage (% of t0): artemisinin acid/heat labile
    ph_pen = ((ph - 5.5) / 2.5) ** 2               # best near pH 5.5
    assay_accel = np.clip(100 - 18 * ph_pen - 20 * temp - 3 * api + noise(2.5), 60, 100)
    # palatability (0-10): sweetener up, bitter API down
    palat = np.clip(4 + 5 * sweet - 3 * api + 1.0 * surf + noise(0.6), 0, 10)
    # administration volume (mL): dose / api concentration for a 20 mg target dose
    admin_vol = np.clip(20.0 / f["api_conc_mg_ml"] * 5 + noise(0.3), 0.5, 15)
    # predicted PK exposure AUC: tracks dissolved fraction + inverse volume issues
    pk_auc = np.clip(60 + 0.6 * dissolution + 0.1 * assay_accel + noise(6), 20, 160)

    out = df.copy()
    out["visc_cp"] = visc
    out["sedimentation_volume_ratio"] = F
    out["redispersibility_strokes"] = redisp
    out["dissolution_pct_30min"] = dissolution
    out["content_uniformity_rsd"] = cu_rsd
    out["delivered_dose_rsd"] = dd_rsd
    out["chemical_assay_pct_accel"] = assay_accel
    out["palatability_score"] = palat
    out["administration_volume_ml"] = admin_vol
    out["predicted_pk_auc"] = pk_auc
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="DoE experiments")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    seed = args.seed or load_config()["reproducibility"]["global_seed"]
    rng = np.random.default_rng(seed)

    names = list(FACTORS)
    lhs = latin_hypercube(args.n, len(names), rng)
    design = pd.DataFrame({n: FACTORS[n][0] + lhs[:, i] * (FACTORS[n][1] - FACTORS[n][0])
                           for i, n in enumerate(names)})
    df = simulate(design, rng)
    df.insert(0, "formulation_id", [f"F{seed}_{i:04d}" for i in range(len(df))])

    out_dir = get_path("data_raw", create=True) / "formulation" / "in_silico_doe"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "doe_dataset.tsv", sep="\t", index=False)
    (out_dir / "factor_ranges.json").write_text(
        json.dumps({k: {"low": v[0], "high": v[1]} for k, v in FACTORS.items()}, indent=2))
    result_manifest("simulate_doe", seed=seed,
                    extra={"n_experiments": args.n, "n_factors": len(FACTORS),
                           "design": "latin_hypercube", "kind": "in_silico_digital_twin"})
    log.info("WROTE in-silico DoE: %d experiments x %d factors + 10 responses -> %s",
             args.n, len(FACTORS), out_dir / "doe_dataset.tsv")
    log.info("response summary:\n%s",
             df[["dissolution_pct_30min", "sedimentation_volume_ratio",
                 "chemical_assay_pct_accel", "palatability_score",
                 "predicted_pk_auc"]].describe().round(2).T.to_string())
    log.info("Next: optimize_formulation.py -- GP surrogate + Bayesian active learning.")


if __name__ == "__main__":
    main()
