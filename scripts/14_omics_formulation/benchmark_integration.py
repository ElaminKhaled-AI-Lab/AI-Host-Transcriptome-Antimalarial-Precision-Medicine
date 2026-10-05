#!/usr/bin/env python
"""Phase 20 -- which integration strategy is best? PROFILE-Net vs three baselines.

Trains four multi-block integration strategies on identical patient x formulation
data and decides the winner empirically:

  early_concat        EARLY integration   -- raw blocks concatenated, one model
  late_ensemble       LATE integration    -- one model per block, outputs combined
  intermediate_latent INTERMEDIATE        -- per-block encoders, latents concatenated
  PROFILE-Net         INTERMEDIATE + bilinear cross-attention INTERACTION

Evaluation uses a PATIENT-GROUPED split, so every test child is unseen in
training. Targets are log10 day-7 exposure and the therapeutic utility, both continuous.

Metrics, and why each is reported:
  * exposure R^2 / cure R^2 -- plain predictive accuracy.
  * within-patient rank correlation -- ranks formulations inside each child.
    Reported for completeness, but a purely ADDITIVE model can score highly here
    whenever the formulation ordering is similar across children, so it does not
    by itself demonstrate personalization.
  * interaction-residual rank correlation -- THE DECIDING METRIC. Each
    formulation's average effect across patients is removed first, leaving only
    the patient-specific deviation, so the score reflects the
    patient-by-formulation interaction alone. An additive model scores near zero
    by construction.
  * personalization gain -- the true cure probability obtained when each child
    receives the model's top-ranked formulation, minus that of the single
    formulation that is best on average. Positive means personalization pays.

Output -> results/integration_benchmark_profile_net.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.models.profile_net import (
    STRATEGIES, train_model, predict, within_patient_rank_correlation,
    interaction_residual_rank_correlation, personalization_gain, pick_device)

log = get_logger(__name__)

PAT_FEATS = ["age_years", "weight_kg", "host_metabolic_z", "host_tox_z",
             "parasite_resistance"]
FORM_FEATS = ["A", "B", "C", "ratio", "stab_pct", "energy", "size_nm", "pdi",
              "zeta_mv", "ee_pct", "dl_pct", "DE_pct", "MDT_h", "t50_h",
              "n_peppas", "liver_idx", "poly_basesize", "poly_zeta", "poly_degrad"]


def load_dataset():
    syn = get_path("data_synthetic")
    p = syn / "patient_formulation_dataset.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t = syn / "patient_formulation_dataset.tsv"
    if t.exists():
        return pd.read_csv(t, sep="\t")
    raise SystemExit("run build_patient_formulation_dataset.py first")


def standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd == 0] = 1
    return (tr - mu) / sd, (te - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent seeds per strategy; the winner is decided on the "
                         "MEAN across seeds, because a single run can flip the ranking")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda. 'auto' verifies the GPU actually works "
                         "(compute capability + a real kernel launch) before using it.")
    args = ap.parse_args()
    device = pick_device(args.device)
    log.info("compute device: %s", device)
    seed = load_config()["reproducibility"]["global_seed"]
    rng = np.random.default_rng(seed)

    df = load_dataset().dropna(subset=PAT_FEATS + FORM_FEATS + ["log10_day7", "utility"])
    df = df.sort_values(["patient_id", "formulation_id"]).reset_index(drop=True)
    pats = df["patient_id"].unique().copy()
    rng.shuffle(pats)
    n_te = max(2, int(len(pats) * args.test_frac))
    te_ids, tr_ids = set(pats[:n_te]), set(pats[n_te:])
    tr, te = df[df["patient_id"].isin(tr_ids)], df[df["patient_id"].isin(te_ids)]
    log.info("patient-grouped split: %d train children (%d rows) | %d test children (%d rows)",
             len(tr_ids), len(tr), len(te_ids), len(te))

    Xp_tr, Xp_te = standardize(tr[PAT_FEATS].to_numpy(float), te[PAT_FEATS].to_numpy(float))
    Xf_tr, Xf_te = standardize(tr[FORM_FEATS].to_numpy(float), te[FORM_FEATS].to_numpy(float))
    Ytr = np.column_stack([tr["log10_day7"].to_numpy(float), tr["utility"].to_numpy(float)])
    Yte = np.column_stack([te["log10_day7"].to_numpy(float), te["utility"].to_numpy(float)])
    mu, sd = Ytr.mean(0), Ytr.std(0)
    sd[sd == 0] = 1

    from sklearn.metrics import r2_score
    pid, fid = te["patient_id"].to_numpy(), te["formulation_id"].to_numpy()
    results = {}
    for name, cls in STRATEGIES.items():
        log.info("training %s (%d seeds) ...", name, args.repeats)
        runs = {"exposure_r2": [], "utility_r2": [], "within_patient_rank_corr": [],
                "interaction_residual_rank_corr": [], "personalization_gain_utility": []}
        n_params = 0
        for r in range(args.repeats):
            sd_seed = seed + 100 * r
            model = cls(d_pat=Xp_tr.shape[1], d_form=Xf_tr.shape[1], n_reg=2)
            n_params = sum(p.numel() for p in model.parameters())
            model = train_model(model, Xp_tr, Xf_tr, (Ytr - mu) / sd, None,
                                epochs=args.epochs, seed=sd_seed, device=device)
            P, _ = predict(model, Xp_te, Xf_te)
            P = P * sd + mu
            runs["exposure_r2"].append(float(r2_score(Yte[:, 0], P[:, 0])))
            runs["utility_r2"].append(float(r2_score(Yte[:, 1], P[:, 1])))
            runs["within_patient_rank_corr"].append(
                within_patient_rank_correlation(pid, Yte[:, 1], P[:, 1]))
            runs["interaction_residual_rank_corr"].append(
                interaction_residual_rank_correlation(pid, fid, Yte[:, 1], P[:, 1]))
            runs["personalization_gain_utility"].append(
                personalization_gain(pid, Yte[:, 1], P[:, 1]))
        agg = {}
        for k, v in runs.items():
            arr = np.asarray(v, float)
            agg[k] = round(float(np.nanmean(arr)), 4)
            agg[k + "_sd"] = round(float(np.nanstd(arr)), 4)
        agg["n_parameters"] = int(n_params)
        agg["n_seeds"] = int(args.repeats)
        agg["per_seed_interaction_rank"] = [round(x, 4) for x in
                                            runs["interaction_residual_rank_corr"]]
        results[name] = agg
        log.info("  %-20s expR2=%.3f+/-%.3f utilR2=%.3f | within=%.3f | "
                 "INTERACTION=%.3f+/-%.3f | gain=%+.4f | params=%d",
                 name, agg["exposure_r2"], agg["exposure_r2_sd"], agg["utility_r2"],
                 agg["within_patient_rank_corr"],
                 agg["interaction_residual_rank_corr"],
                 agg["interaction_residual_rank_corr_sd"],
                 agg["personalization_gain_utility"], n_params)

    def key(k):
        v = results[k]
        ir = v["interaction_residual_rank_corr"]
        return (ir if np.isfinite(ir) else -9, v["personalization_gain_utility"])

    best = max(results, key=key)
    out = {
        "task": "patient x formulation outcome prediction (paediatric antimalarial nano-DDS)",
        "split": "patient-grouped (test children unseen in training)",
        "n_train_children": len(tr_ids), "n_test_children": len(te_ids),
        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
        "compute_device": str(device),
        "n_seeds_per_strategy": int(args.repeats),
        "epochs": int(args.epochs),
        "integration_strategies": {
            "early_concat": "EARLY -- raw blocks concatenated",
            "late_ensemble": "LATE -- per-block models combined at output",
            "intermediate_latent": "INTERMEDIATE -- per-block encoders, latents concatenated",
            "PROFILE-Net": "INTERMEDIATE + low-rank bilinear cross-attention interaction",
        },
        "metrics": results,
        "winner": best,
        "decision_metric": "interaction-residual rank correlation (patient-by-formulation "
                           "interaction, formulation main effect removed); personalization "
                           "gain as tiebreak",
        "note": "within_patient_rank_corr is reported for completeness but additive models "
                "can score highly on it without modelling interaction.",
    }
    dest = get_path("results", create=True) / "integration_benchmark_profile_net.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    # is the winner separated from the runner-up by more than the seed noise?
    ranked = sorted(results, key=key, reverse=True)
    if len(ranked) > 1:
        a, b = results[ranked[0]], results[ranked[1]]
        margin = a["interaction_residual_rank_corr"] - b["interaction_residual_rank_corr"]
        noise = a["interaction_residual_rank_corr_sd"] + b["interaction_residual_rank_corr_sd"]
        out["winner_margin_over_runner_up"] = round(float(margin), 4)
        out["winner_is_separated_from_noise"] = bool(margin > noise)
        dest.write_text(json.dumps(out, indent=2, default=float))
        log.info("runner-up: %s | margin %.4f vs combined seed SD %.4f -> %s",
                 ranked[1], margin, noise,
                 "SEPARATED" if margin > noise else "NOT separated (treat as a tie)")
    log.info("WINNER (by mean interaction-residual rank over %d seeds): %s",
             args.repeats, best)
    log.info("wrote %s", dest)


if __name__ == "__main__":
    main()
