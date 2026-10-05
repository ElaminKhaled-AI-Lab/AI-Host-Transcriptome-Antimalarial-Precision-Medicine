#!/usr/bin/env python
"""Phase 20 -- the complete figure suite for the nano-DDS + formulation-omics arms.

Produces nine publication-oriented figures, each written separately so it can be
placed independently in the manuscript. Every panel states the drug and polymer
context explicitly.

  F1  nanoparticle characterization across ALL polymers (size, PDI, zeta, EE, DL)
  F2  in-vitro release profiles for ALL polymers (named drug) + drug comparison
  F3  dissolution parameters by polymer (DE, MDT, t50, t80, burst, K-P n)
  F4  release-kinetics model comparison (R^2 heatmap, n distribution, mechanism)
  F5  design-of-experiments main effects and two-factor interactions
  F6  optimization landscape and polymer desirability
  F7  ICH accelerated-stability trajectories
  F8  validation against published values
  F9  precision formulation across the whole paediatric cohort + integration benchmark

Output -> figures/F1_... .png ... figures/F9_... .png
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import get_path
from src.utils.logging_utils import get_logger
from src.models.nano_dds import (
    Formulation, POLYMERS, DRUGS, release_profile, accelerated_stability)

log = get_logger(__name__)

FIG = None          # set in main()
TIMES = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72], float)
PALETTE = plt.cm.tab20(np.linspace(0, 1, 20))
DRUG_MAIN = "lumefantrine"


def _save(fig, name):
    out = FIG / name
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def _polymer_colors():
    return {p: PALETTE[i % 20] for i, p in enumerate(POLYMERS)}


# --------------------------------------------------------------------- F1
def fig1_characterization(dis):
    cols = [("size_nm", "particle size (nm)"), ("pdi", "polydispersity index"),
            ("zeta_mv", "zeta potential (mV)"), ("ee_pct", "entrapment efficiency (%)"),
            ("dl_pct", "drug loading (%)"), ("liver_targeting_index", "liver-targeting index")]
    order = sorted(POLYMERS)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        data = [dis.loc[dis.polymer == p, c].dropna().to_numpy() for p in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
        for patch, p in zip(bp["boxes"], order):
            patch.set_facecolor(_polymer_colors()[p]); patch.set_alpha(0.75)
        ax.set_ylabel(lab); ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(alpha=0.25, axis="y")
        if c == "size_nm":
            ax.axhspan(100, 200, color="grey", alpha=0.18)
            ax.text(0.02, 0.95, "100-200 nm Kupffer window", transform=ax.transAxes,
                    fontsize=7, va="top")
        if c == "zeta_mv":
            for y in (20, -20):
                ax.axhline(y, ls="--", lw=0.8, color="firebrick")
            ax.text(0.02, 0.05, "|zeta| = 20 mV colloidal-stability threshold",
                    transform=ax.transAxes, fontsize=7)
    fig.suptitle("F1  Nanoparticle characterization across all 11 FDA/GRAS polymers\n"
                 f"(all {dis.drug.nunique()} antimalarials x 27-run factorial; n={len(dis)} formulations)",
                 fontsize=13)
    _save(fig, "F1_characterization_by_polymer.png")


# --------------------------------------------------------------------- F2
def fig2_release_profiles(dis):
    order = sorted(POLYMERS)
    colors = _polymer_colors()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) all polymers, one named drug, centre point
    ax = axes[0, 0]
    for p in order:
        f = Formulation(drug=DRUG_MAIN, polymer=p, polymer_drug_ratio=0,
                        stabilizer_pct=0, homogenization=0)
        t, Q = release_profile(f, TIMES)
        ax.plot(t, Q, marker="o", ms=3, lw=1.6, color=colors[p], label=p)
    ax.set_xlabel("time (h)"); ax.set_ylabel("cumulative release (%)")
    ax.set_title(f"(a) All polymers -- drug: {DRUG_MAIN.upper()}, centre point "
                 "(ratio 5:1, stabilizer 1%, medium energy)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25)

    # (b) all drugs on one representative polymer
    ax = axes[0, 1]
    for i, d in enumerate(sorted(DRUGS)):
        f = Formulation(drug=d, polymer="PLGA_50_50", polymer_drug_ratio=0,
                        stabilizer_pct=0, homogenization=0)
        t, Q = release_profile(f, TIMES)
        ax.plot(t, Q, marker="s", ms=3, lw=1.6, color=PALETTE[i % 20], label=d)
    ax.set_xlabel("time (h)"); ax.set_ylabel("cumulative release (%)")
    ax.set_title("(b) All antimalarials -- polymer: PLGA 50:50, centre point")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25)

    # (c) factor effect on release, named drug + polymer
    ax = axes[1, 0]
    for lvl, lab, col in [(-1, "ratio 2:1", "#4C78A8"), (0, "ratio 5:1", "#F58518"),
                          (1, "ratio 10:1", "#54A24B")]:
        f = Formulation(drug=DRUG_MAIN, polymer="PLGA_50_50", polymer_drug_ratio=lvl,
                        stabilizer_pct=0, homogenization=0)
        t, Q = release_profile(f, TIMES)
        ax.plot(t, Q, marker="o", ms=3, color=col, label=lab)
    ax.set_xlabel("time (h)"); ax.set_ylabel("cumulative release (%)")
    ax.set_title(f"(c) Polymer:drug ratio effect -- {DRUG_MAIN.upper()} / PLGA 50:50")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (d) immediate- vs sustained-release classes
    ax = axes[1, 1]
    for cls, col in [("IR", "#E45756"), ("SR", "#4C78A8")]:
        ps = [p for p in order if POLYMERS[p]["release_class"] == cls]
        curves = []
        for p in ps:
            f = Formulation(drug=DRUG_MAIN, polymer=p, polymer_drug_ratio=0,
                            stabilizer_pct=0, homogenization=0)
            t, Q = release_profile(f, TIMES)
            curves.append(Q)
        arr = np.array(curves)
        ax.plot(TIMES, arr.mean(0), color=col, lw=2,
                label=f"{cls} ({'immediate' if cls=='IR' else 'sustained'}) n={len(ps)}")
        ax.fill_between(TIMES, arr.min(0), arr.max(0), color=col, alpha=0.18)
    ax.set_xlabel("time (h)"); ax.set_ylabel("cumulative release (%)")
    ax.set_title(f"(d) Immediate vs sustained release classes -- {DRUG_MAIN.upper()}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle("F2  In-vitro release profiles -- every polymer and every drug identified",
                 fontsize=13)
    _save(fig, "F2_release_profiles_all_polymers.png")


# --------------------------------------------------------------------- F3
def fig3_dissolution_parameters(dis):
    order = sorted(POLYMERS)
    colors = _polymer_colors()
    params = [("DE_pct", "dissolution efficiency DE (%)"),
              ("MDT_h", "mean dissolution time MDT (h)"),
              ("t50_h", "t50 (h)"), ("t80_h", "t80 (h)"),
              ("burst_1h_pct", "burst release at 1 h (%)"),
              ("n_peppas", "Korsmeyer-Peppas exponent n")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (c, lab) in zip(axes.ravel(), params):
        m = dis.groupby("polymer")[c].mean().reindex(order)
        s = dis.groupby("polymer")[c].std().reindex(order)
        ax.bar(order, m.to_numpy(), yerr=s.to_numpy(), capsize=3,
               color=[colors[p] for p in order], alpha=0.85)
        ax.set_ylabel(lab); ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(alpha=0.25, axis="y")
        if c == "n_peppas":
            ax.axhline(0.43, ls="--", color="firebrick", lw=1)
            ax.axhline(0.85, ls="--", color="darkgreen", lw=1)
            ax.text(0.02, 0.93, "n=0.43 Fickian | n=0.85 case-II",
                    transform=ax.transAxes, fontsize=7)
    fig.suptitle("F3  Dissolution parameters by polymer (mean +/- SD over all drugs "
                 "and all 27 factorial runs)", fontsize=13)
    _save(fig, "F3_dissolution_parameters.png")


# --------------------------------------------------------------------- F4
def fig4_kinetics(dis):
    order = sorted(POLYMERS)
    models = [("r2_zero_order", "zero-order"), ("r2_first_order", "first-order"),
              ("r2_higuchi", "Higuchi"), ("r2_peppas", "Korsmeyer-Peppas"),
              ("r2_hixson", "Hixson-Crowell")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) R^2 heatmap polymer x model
    M = np.array([[dis.loc[dis.polymer == p, c].mean() for c, _ in models] for p in order])
    ax = axes[0]
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([l for _, l in models], rotation=45,
                                                          ha="right", fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="mean R$^2$")
    ax.set_title("(a) Kinetic-model fit (R$^2$) by polymer")

    # (b) K-P n distribution by polymer
    ax = axes[1]
    data = [dis.loc[dis.polymer == p, "n_peppas"].dropna().to_numpy() for p in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
    for patch, p in zip(bp["boxes"], order):
        patch.set_facecolor(_polymer_colors()[p]); patch.set_alpha(0.8)
    ax.axhline(0.43, ls="--", color="firebrick"); ax.axhline(0.85, ls="--", color="darkgreen")
    ax.set_ylabel("Korsmeyer-Peppas n"); ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.set_title("(b) Transport exponent n (mechanism)")
    ax.grid(alpha=0.25, axis="y")

    # (c) best model + mechanism counts
    ax = axes[2]
    bm = dis["best_model_aic"].value_counts()
    mech = dis["release_mechanism"].value_counts()
    y = np.arange(len(bm) + len(mech))
    labels = [f"best: {k}" for k in bm.index] + [f"mech: {k}" for k in mech.index]
    vals = list(bm.to_numpy()) + list(mech.to_numpy())
    ax.barh(y, vals, color=["#4C78A8"] * len(bm) + ["#F58518"] * len(mech))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("number of formulations")
    ax.set_title("(c) Best model (AIC, percent scale) and mechanism")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("F4  Release-kinetics model comparison across all formulations", fontsize=13)
    _save(fig, "F4_release_kinetics_models.png")


# --------------------------------------------------------------------- F5
def fig5_doe(dis):
    factors = [("A_ratio", "polymer:drug ratio"), ("B_stabilizer_pct", "stabilizer (% w/v)"),
               ("C_energy", "homogenization energy")]
    resp = [("size_nm", "size (nm)"), ("ee_pct", "EE (%)"), ("DE_pct", "DE (%)"),
            ("liver_targeting_index", "liver index")]
    fig, axes = plt.subplots(len(resp), len(factors) + 1, figsize=(18, 12))
    for i, (rc, rlab) in enumerate(resp):
        for j, (fc, flab) in enumerate(factors):
            ax = axes[i, j]
            g = dis.groupby(fc)[rc].agg(["mean", "std"])
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=4,
                        color="#4C78A8", lw=1.8)
            ax.set_xlabel(flab if i == len(resp) - 1 else "")
            ax.set_ylabel(rlab if j == 0 else "")
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title(f"main effect: {flab}", fontsize=9)
        # interaction heatmap A x B
        ax = axes[i, len(factors)]
        piv = dis.pivot_table(index="A_ratio", columns="B_stabilizer_pct", values=rc,
                              aggfunc="mean")
        im = ax.imshow(piv.to_numpy(), cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        ax.set_xlabel("stabilizer %" if i == len(resp) - 1 else "")
        fig.colorbar(im, ax=ax, label=rlab)
        if i == 0:
            ax.set_title("interaction: ratio x stabilizer", fontsize=9)
    fig.suptitle("F5  Design of experiments: main effects (3 levels) and two-factor "
                 "interaction surfaces, pooled over polymers and drugs", fontsize=13)
    _save(fig, "F5_doe_main_effects_interactions.png")


# --------------------------------------------------------------------- F6
def fig6_optimization(dis, res):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = _polymer_colors()

    ax = axes[0]
    for p in sorted(POLYMERS):
        d = dis[dis.polymer == p]
        ax.scatter(d["size_nm"], d["ee_pct"], s=8, alpha=0.5, color=colors[p], label=p)
    ax.axvspan(100, 200, color="grey", alpha=0.18)
    ax.set_xlabel("particle size (nm)"); ax.set_ylabel("entrapment efficiency (%)")
    ax.set_title("(a) Design space by polymer (grey = Kupffer window)")
    ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.25)

    ax = axes[1]
    sc = ax.scatter(dis["DE_pct"], dis["liver_targeting_index"], c=dis["zeta_mv"],
                    cmap="coolwarm", s=8, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="zeta potential (mV)")
    ax.set_xlabel("dissolution efficiency DE (%)"); ax.set_ylabel("liver-targeting index")
    ax.set_title("(b) Release-versus-targeting trade-off")
    ax.grid(alpha=0.25)

    ax = axes[2]
    rank = (res.get("polymer_ranking_by_desirability") or {})
    if rank:
        items = list(rank.items())[::-1]
        ax.barh([k for k, _ in items], [v for _, v in items],
                color=[colors.get(k, "#888") for k, _ in items])
        ax.set_xlabel("mean desirability")
    ax.set_title(f"(c) Polymer ranking -- drug: {res.get('drug', DRUG_MAIN).upper()}")
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("F6  Optimization landscape and polymer ranking", fontsize=13)
    _save(fig, "F6_optimization_landscape.png")


# --------------------------------------------------------------------- F7
def fig7_stability():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = _polymer_colors()
    months = np.arange(0, 7)
    for p in sorted(POLYMERS):
        f = Formulation(drug=DRUG_MAIN, polymer=p, polymer_drug_ratio=0,
                        stabilizer_pct=0, homogenization=0)
        st = accelerated_stability(f, months=6)
        axes[0].plot(months, st["size_trajectory_40C_nm"], marker="o", ms=3,
                     color=colors[p], label=p)
        axes[1].bar(p, st["predicted_shelf_life_25C_months"], color=colors[p], alpha=0.85)
    axes[0].set_xlabel("months at 40 C / 75% RH"); axes[0].set_ylabel("particle size (nm)")
    axes[0].set_title(f"(a) ICH accelerated stability -- size growth ({DRUG_MAIN.upper()})")
    axes[0].legend(fontsize=6, ncol=2); axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("predicted shelf life at 25 C (months)")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].set_title("(b) Predicted long-term shelf life")
    axes[1].grid(alpha=0.25, axis="y")
    fig.suptitle("F7  ICH Q1A accelerated stability (Arrhenius-scaled)", fontsize=13)
    _save(fig, "F7_stability.png")


# --------------------------------------------------------------------- F8
def fig8_validation(val):
    checks = val.get("checks", [])
    if not checks:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    labels, ok = [], []
    for c in checks:
        labels.append(f"{c['anchor'][:26]}\n{c['metric']}")
        ok.append(1 if c["design_space_overlaps"] else 0)
    y = np.arange(len(labels))
    ax.barh(y, ok, color=["#59A14F" if o else "#E15759" for o in ok], alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1.25); ax.set_xticks([0, 1]); ax.set_xticklabels(["outside", "overlaps"])
    for i, c in enumerate(checks):
        ax.text(1.02, i, f"twin {c['twin_achievable_range']} | lit {c['literature_range']}",
                va="center", fontsize=6)
    ax.set_title(f"F8  Validation against published antimalarial-nanoparticle values\n"
                 f"{val.get('n_design_space_overlaps')}/{val.get('n_checks')} published values "
                 f"fall inside the twin's achievable design space", fontsize=12)
    _save(fig, "F8_literature_validation.png")


# --------------------------------------------------------------------- F9
def fig9_personalization(per, bench, pf):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = _polymer_colors()

    # (a) polymer chosen per child
    ax = axes[0, 0]
    vc = per["polymer"].value_counts()
    ax.bar(vc.index, vc.to_numpy(), color=[colors.get(p, "#888") for p in vc.index], alpha=0.9)
    ax.set_ylabel("number of children"); ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title(f"(a) Personalized polymer choice (n={len(per)} children)")
    ax.grid(alpha=0.25, axis="y")

    # (b) personalization map in the omic plane
    ax = axes[0, 1]
    for p in per["polymer"].unique():
        d = per[per.polymer == p]
        ax.scatter(d["host_metabolic_z"], d["host_tox_z"], s=42, alpha=0.85,
                   color=colors.get(p, "#888"), label=p, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("host metabolic proxy (z)"); ax.set_ylabel("host toxicity proxy (z)")
    ax.set_title("(b) Chosen formulation across the omic phenotype plane")
    ax.legend(fontsize=6); ax.grid(alpha=0.25)

    # (c) gain from personalization
    ax = axes[0, 2]
    ax.hist(per["objective_gain"], bins=25, color="#4C78A8", alpha=0.85)
    ax.axvline(per["objective_gain"].mean(), color="firebrick", ls="--",
               label=f"mean +{per['objective_gain'].mean():.3f}")
    ax.set_xlabel("utility gain vs one-size-fits-all"); ax.set_ylabel("children")
    ax.set_title("(c) Benefit of personalization")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (d) integration benchmark -- the deciding metric
    ax = axes[1, 0]
    mt = bench.get("metrics", {})
    names = list(mt)
    ir = [mt[n]["interaction_residual_rank_corr"] for n in names]
    err = [mt[n].get("interaction_residual_rank_corr_sd", 0) for n in names]
    tie = not bench.get("winner_is_separated_from_noise", True)
    cols = ["#59A14F" if (n == bench.get("winner") and not tie) else "#4C78A8"
            for n in names]
    ax.barh(names, ir, xerr=err, capsize=4, color=cols, alpha=0.9)
    ax.set_xlabel("interaction-residual rank correlation (mean +/- SD over seeds)")
    ttl = "(d) Which integration strategy captures\npatient x formulation interaction?"
    if tie:
        ttl += "\n(joint strategies tie; LATE integration fails)"
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    for i, v in enumerate(ir):
        ax.text(max(v, 0) + max(err) + 0.02, i, f"{v:.3f}", va="center", fontsize=8)

    # (e) accuracy comparison
    ax = axes[1, 1]
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [mt[n]["exposure_r2"] for n in names], w, label="exposure R$^2$",
           color="#4C78A8")
    ax.bar(x + w / 2, [mt[n].get("utility_r2", np.nan) for n in names], w,
           label="utility R$^2$", color="#F58518")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=7, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    ax.set_title("(e) Predictive accuracy by strategy")

    # (f) exposure vs utility coloured by chosen polymer
    ax = axes[1, 2]
    sc = ax.scatter(pf["day7_ng_per_mL"], pf["utility"], c=pf["toxicity_risk"],
                    cmap="magma", s=4, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="toxicity risk")
    ax.set_xlabel("day-7 exposure (ng/mL)"); ax.set_ylabel("therapeutic utility")
    ax.set_xscale("log")
    ax.set_title("(f) Therapeutic window across all\nchild x formulation pairs")
    ax.grid(alpha=0.25)

    fig.suptitle("F9  Precision formulation across the whole paediatric cohort, and the "
                 "integration-strategy benchmark", fontsize=13)
    _save(fig, "F9_precision_formulation_cohort.png")


def main():
    global FIG
    FIG = get_path("figures", create=True)
    tbl, res_dir = get_path("tables"), get_path("results")

    dis = pd.read_csv(tbl / "nano_dds_dissolution_parameters.csv")
    log.info("dissolution table: %d formulations, %d polymers, %d drugs",
             len(dis), dis.polymer.nunique(), dis.drug.nunique())

    def _j(p):
        return json.loads((res_dir / p).read_text()) if (res_dir / p).exists() else {}

    fig1_characterization(dis)
    fig2_release_profiles(dis)
    fig3_dissolution_parameters(dis)
    fig4_kinetics(dis)
    fig5_doe(dis)
    fig6_optimization(dis, _j("nano_dds_optimization.json"))
    fig7_stability()
    fig8_validation(_j("nano_dds_validation.json"))

    per_p = tbl / "personalized_formulation_per_patient.csv"
    if per_p.exists():
        syn = get_path("data_synthetic")
        pf = (pd.read_parquet(syn / "patient_formulation_dataset.parquet")
              if (syn / "patient_formulation_dataset.parquet").exists()
              else pd.read_csv(syn / "patient_formulation_dataset.tsv", sep="\t"))
        fig9_personalization(pd.read_csv(per_p),
                             _j("integration_benchmark_profile_net.json"), pf)
    else:
        log.warning("F9 skipped: run the 14_omics_formulation scripts first")

    log.info("figure suite complete -> %s", FIG)


if __name__ == "__main__":
    main()
