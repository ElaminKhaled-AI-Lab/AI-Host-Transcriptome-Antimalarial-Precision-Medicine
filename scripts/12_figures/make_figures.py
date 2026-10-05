#!/usr/bin/env python
"""Phase all -- publication figures from results/ (blueprint Section 96).

Each panel is guarded by file existence, so this runs at any stage and draws
whatever results are present. Outputs PNG (300 dpi) + PDF to figures/.

Panels:
  F1  integration benchmark   (BLISS vs concat: ARI, silhouette, balance, stability)
  F2  HPAMM-Net multi-task     (AUROC HPAMM vs baseline per outcome, mean±SD over seeds)
  F3  HPAMM-Net missingness    (resistance full vs parasite-dropped)
  F4  ablation study           (mean-class AUROC + resistance across configs)
  F5  formulation optimisation (GP surrogate R2 + active-learning trace)
  F6  real-data validation     (synthetic -> real Mali -> external Gambia)
"""
import _bootstrap  # noqa: F401
import glob
import json

import numpy as np

from src.utils.config import get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _save(fig, name):
    d = get_path("figures", create=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    log.info("wrote %s.png/.pdf", name)


def _load(pattern):
    files = sorted(glob.glob(str(get_path("results") / pattern)))
    return [json.load(open(f)) for f in files]


def fig_integration(plt):
    runs = _load("integration_benchmark_seed*.json")
    if not runs:
        return
    metrics = ["ari", "silhouette", "cluster_balance", "stability_mean_pairwise_ari"]
    labels = ["ARI", "Silhouette", "Cluster balance", "Stability"]
    methods = ["concat_kmeans", "BLISS"]
    vals = {m: [np.mean([r[m][k] for r in runs if m in r]) for k in metrics] for m in methods}
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w/2, vals["concat_kmeans"], w, label="Concat + k-means", color="#9aa7b1")
    ax.bar(x + w/2, vals["BLISS"], w, label="BLISS", color="#c0405f")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("score"); ax.set_title("Integration benchmark (synthetic cohort)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "F1_integration_benchmark"); plt.close(fig)


def _hpamm_agg():
    runs = _load("hpamm_multitask_seed*.json")
    if not runs:
        return None
    CLASS = ["resistance", "cure", "recrudescence", "toxicity"]
    agg = {}
    for t in CLASS:
        hp = [r["hpamm_full"][t]["auroc"] for r in runs]
        bl = [r["baseline_xgb"][t]["auroc"] for r in runs]
        dp = [r["hpamm_parasite_dropped"][t]["auroc"] for r in runs]
        agg[t] = dict(hp=(np.mean(hp), np.std(hp)), bl=(np.mean(bl), np.std(bl)),
                      drop=(np.mean(dp), np.std(dp)))
    return agg, CLASS


def fig_multitask(plt):
    r = _hpamm_agg()
    if not r:
        return
    agg, CLASS = r
    x = np.arange(len(CLASS)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w/2, [agg[t]["bl"][0] for t in CLASS], w,
           yerr=[agg[t]["bl"][1] for t in CLASS], capsize=3, label="XGBoost (per-task)", color="#9aa7b1")
    ax.bar(x + w/2, [agg[t]["hp"][0] for t in CLASS], w,
           yerr=[agg[t]["hp"][1] for t in CLASS], capsize=3, label="HPAMM-Net (multi-task)", color="#2f6f8f")
    ax.axhline(0.5, ls="--", lw=0.8, color="k")
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASS])
    ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 0.85)
    ax.set_title("Multi-task outcome prediction (mean ± SD, 10 cohorts)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "F2_hpamm_multitask"); plt.close(fig)


def fig_missingness(plt):
    r = _hpamm_agg()
    if not r:
        return
    agg, _ = r
    labels = ["resistance\n(parasite-driven)", "toxicity\n(host-driven)"]
    full = [agg["resistance"]["hp"][0], agg["toxicity"]["hp"][0]]
    drop = [agg["resistance"]["drop"][0], agg["toxicity"]["drop"][0]]
    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - w/2, full, w, label="all modalities", color="#2f6f8f")
    ax.bar(x + w/2, drop, w, label="parasite block dropped", color="#e0a458")
    ax.axhline(0.5, ls="--", lw=0.8, color="k")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("AUROC"); ax.set_title("Missingness-aware degradation")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "F3_missingness"); plt.close(fig)


def fig_ablation(plt):
    a = _load("hpamm_ablations.json")
    if not a:
        return
    ab = a[0]
    order = ["full", "no_gating", "no_modality_dropout", "host_only", "parasite_only"]
    order = [o for o in order if o in ab]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x - w/2, [ab[o]["mean_class_auroc"] for o in order], w, label="mean class AUROC", color="#6a8caf")
    ax.bar(x + w/2, [ab[o]["resistance_auroc"] for o in order], w, label="resistance AUROC", color="#c0405f")
    ax.axhline(0.5, ls="--", lw=0.8, color="k")
    ax.set_xticks(x); ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 0.8); ax.set_title("HPAMM-Net ablations")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "F4_ablation"); plt.close(fig)


def fig_formulation(plt):
    cv_files = sorted(glob.glob(str(get_path("results") / "formulation" / "gp_surrogate_cv.json")))
    tr_files = sorted(glob.glob(str(get_path("results") / "formulation" / "optimization_trace.json")))
    if not cv_files:
        return
    cv = json.load(open(cv_files[0]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    keys = [k for k in cv if cv[k] == cv[k]]
    axes[0].barh(range(len(keys)), [cv[k] for k in keys], color="#4c9a76")
    axes[0].set_yticks(range(len(keys))); axes[0].set_yticklabels(keys, fontsize=7)
    axes[0].set_xlabel("cross-validated R²"); axes[0].set_xlim(0, 1)
    axes[0].set_title("GP surrogate accuracy"); axes[0].spines[["top", "right"]].set_visible(False)
    if tr_files:
        tr = json.load(open(tr_files[0]))
        best = [t["best_desirability"] for t in tr]
        axes[1].plot(range(len(best)), best, "-o", color="#c0405f")
        axes[1].set_xlabel("active-learning round"); axes[1].set_ylabel("best desirability")
        axes[1].set_title("Bayesian optimisation trace"); axes[1].spines[["top", "right"]].set_visible(False)
    _save(fig, "F5_formulation"); plt.close(fig)


def fig_real(plt):
    rv = _load("real_validation.json")
    if not rv:
        return
    r = rv[0]
    order = [c for c in ["synthetic", "real_mali", "external_gambia"]
             if c in r.get("transportability", {}).get("per_cohort", {})]
    if not order:
        return
    pc = r["transportability"]["per_cohort"]
    aris = [pc[c].get("ari", np.nan) for c in order]
    sils = [pc[c].get("silhouette", np.nan) for c in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - w/2, aris, w, label="ARI", color="#2f6f8f")
    ax.bar(x + w/2, sils, w, label="silhouette", color="#e0a458")
    ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in order])
    ax.set_ylabel("score"); ax.set_title("Synthetic → real → external transportability")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "F6_real_validation"); plt.close(fig)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "figure.dpi": 120})
    for fn in (fig_integration, fig_multitask, fig_missingness, fig_ablation,
               fig_formulation, fig_real):
        try:
            fn(plt)
        except Exception as e:  # noqa: BLE001
            log.warning("%s skipped: %s", fn.__name__, e)
    log.info("figures written to %s (only panels with available results).", get_path("figures"))


if __name__ == "__main__":
    main()
