#!/usr/bin/env python
"""Phase 19 -- summary figure panel for the nano-DDS arm.

Reads the factorial dataset + optimization + validation outputs and draws a
multi-panel figure: (A) size vs EE colored by liver-targeting index, (B) release
profiles for the recommended vs a burst formulation, (C) polymer desirability
ranking, (D) literature-validation design-space overlap.

Output -> figures/nano_dds_summary.png
"""
import _bootstrap  # noqa: F401
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path
from src.utils.logging_utils import get_logger
from src.models.nano_dds import Formulation, release_profile

log = get_logger(__name__)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syn = get_path("data_synthetic")
    df = pd.read_csv(syn / "nano_dds_factorial.tsv", sep="\t")
    res = get_path("results")
    opt = json.loads((res / "nano_dds_optimization.json").read_text()) if (res / "nano_dds_optimization.json").exists() else {}
    val = json.loads((res / "nano_dds_validation.json").read_text()) if (res / "nano_dds_validation.json").exists() else {}

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # A: size vs EE, colored by liver index
    sc = ax[0, 0].scatter(df["size_nm"], df["ee_pct"], c=df["liver_targeting_index"],
                          cmap="viridis", s=10, alpha=0.6)
    ax[0, 0].axvspan(100, 200, color="grey", alpha=0.15, label="Kupffer 100-200 nm")
    ax[0, 0].set_xlabel("particle size (nm)"); ax[0, 0].set_ylabel("entrapment efficiency (%)")
    ax[0, 0].set_title("A  Design space: size vs EE"); ax[0, 0].legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax[0, 0], label="liver-targeting index")

    # B: release profiles - recommended vs burst
    rec = opt.get("recommended_formulation", {})
    if rec:
        fac = rec.get("optimal_factors_coded", {"A": 0, "B": 0, "C": 0})
        f1 = Formulation(drug=opt.get("drug", "lumefantrine"), polymer=rec.get("polymer", "PLGA_50_50"),
                         polymer_drug_ratio=fac["A"], stabilizer_pct=fac["B"], homogenization=fac["C"])
        f2 = Formulation(drug=opt.get("drug", "lumefantrine"), polymer="Chitosan",
                         polymer_drug_ratio=-1, stabilizer_pct=1, homogenization=1)
        for f, lab in [(f1, f"recommended ({rec.get('polymer')})"), (f2, "fast/burst (Chitosan)")]:
            t, Q = release_profile(f)
            ax[0, 1].plot(t, Q, marker="o", ms=3, label=lab)
    ax[0, 1].set_xlabel("time (h)"); ax[0, 1].set_ylabel("cumulative release (%)")
    ax[0, 1].set_title("B  In-vitro release profiles"); ax[0, 1].legend(fontsize=8)

    # C: polymer desirability ranking
    rank = opt.get("polymer_ranking_by_desirability", {})
    if rank:
        items = list(rank.items())[:8][::-1]
        ax[1, 0].barh([k for k, _ in items], [v for _, v in items], color="#4C78A8")
    ax[1, 0].set_xlabel("mean desirability"); ax[1, 0].set_title("C  Polymer ranking")

    # D: validation overlap
    checks = val.get("checks", [])
    if checks:
        labels = [f"{c['drug'][:6]}:{c['metric'][:8]}" for c in checks]
        ok = [1 if c["design_space_overlaps"] else 0 for c in checks]
        ax[1, 1].bar(range(len(ok)), ok, color=["#59A14F" if o else "#E15759" for o in ok])
        ax[1, 1].set_xticks(range(len(labels))); ax[1, 1].set_xticklabels(labels, rotation=90, fontsize=6)
        ax[1, 1].set_ylabel("overlaps literature (1/0)")
    ax[1, 1].set_title(f"D  Literature validation "
                       f"({val.get('n_design_space_overlaps','?')}/{val.get('n_checks','?')})")

    fig.suptitle("Nano-DDS formulation-AI: in-silico digital twin (method development)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = get_path("figures", create=True) / "nano_dds_summary.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
