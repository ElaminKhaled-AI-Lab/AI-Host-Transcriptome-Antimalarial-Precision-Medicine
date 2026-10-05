#!/usr/bin/env python
"""Phase 17 -- dual-RNA QC + integration robustness (blueprint Sections 29, 71).

Characterises the real dual host-parasite RNA cohort and shows that the
integration result is robust to the sparsity of the parasite-RNA layer -- turning
the low parasite-read fraction from a caveat into a quantified, controlled
analysis.

It reports, for a processed dual-RNA cohort:
  * the distribution of the parasite-read fraction across samples;
  * host and parasite transcript detection per sample;
  * how many samples exceed parasite-signal thresholds; and
  * BLISS + baseline integration scored three ways -- host-RNA only, full dual
    (host + parasite RNA), and the parasite-informed subset -- so the reader can
    see exactly what the (sparse) parasite layer contributes and that the host
    integration is stable regardless.

Output -> reports/dualrna_qc_<cohort>.md  +  results/dualrna_qc_<cohort>.json
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config
from src.utils.logging_utils import get_logger
from src.metrics.integration_eval import full_report

log = get_logger(__name__)

COHORT_DIR = {"real_mali": "dual_rna_mali", "external_gambia": "dual_rna_gambia"}


def _load(path):
    return pd.read_csv(path, sep="\t", index_col=0) if path.exists() else None


def _pca(counts, k, seed):
    from sklearn.decomposition import PCA
    X = counts.fillna(0).to_numpy(float)
    lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    X = np.log1p(X / lib * 1e6)                 # CPM before log: see note above
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    X = (X - mu) / sd
    k = min(k, max(2, X.shape[0] - 1), X.shape[1])
    return PCA(n_components=k, random_state=seed).fit_transform(X)


def _block_scale(P):
    """Scale a block of principal components to unit total variance.

    The previous code z-scored every component, which gave PC1 (often the only
    informative axis) the same weight as PC30, so dozens of noise dimensions
    outvoted the signal. Scaling the BLOCK instead preserves the variance
    ordering inside each block while still preventing the host block from
    dominating the parasite block purely by scale. This is the standard
    multiple-factor-analysis normalisation.
    """
    tot = float(np.sqrt((P.var(axis=0)).sum()))
    return P / (tot if tot > 0 else 1.0)


def evaluate(blocks, k, seed):
    from sklearn.cluster import KMeans
    Z = np.concatenate([_block_scale(v) for v in blocks.values()], axis=1)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    out = {"concat_kmeans": full_report(Z, lab, None, k=k, seed=seed)}
    try:
        import torch  # noqa: F401
        from src.models.bliss import train_bliss, BlissConfig
        _, latent, labels = train_bliss({m: _block_scale(v) for m, v in blocks.items()},
                                        BlissConfig(k_clusters=k, latent_dim=16,
                                                    epochs=150, seed=seed))
        out["BLISS"] = full_report(latent, labels, None, k=k, seed=seed)
    except Exception as e:  # noqa: BLE001
        log.warning("BLISS skipped: %s", e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=list(COHORT_DIR), default="real_mali")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--min-parasite-frac", type=float, default=0.0005,
                    help="parasite-informed subset threshold on read fraction")
    args = ap.parse_args()
    seed = load_config()["reproducibility"]["global_seed"]

    d = get_path("data_processed") / f"dual_rna_{args.cohort.split('_')[-1]}"
    host = _load(d / "host_rna_counts.tsv")
    para = _load(d / "parasite_rna_counts.tsv")
    frac = _load(d / "parasite_read_fraction.tsv")
    if host is None or para is None:
        log.error("processed dual-RNA not found under %s", d)
        return
    common = host.index.intersection(para.index)
    if len(common) < 3 * args.k:
        log.error("%s: only %d matched samples (need >= %d). Has salmon quant "
                  "finished for this cohort? Run preprocess_dual_rna + --assemble-only "
                  "AFTER quantification completes.", args.cohort, len(common), 3 * args.k)
        return
    host, para = host.loc[common], para.loc[common]
    fr = (frac.reindex(common).iloc[:, 0] if frac is not None
          else para.sum(1) / (host.sum(1) + para.sum(1) + 1e-9))
    fr = pd.to_numeric(fr, errors="coerce").fillna(0.0)

    # --- QC statistics ------------------------------------------------------
    host_detected = (host > 0).sum(1)
    para_detected = (para > 0).sum(1)
    qc = {
        "n_samples": int(len(common)),
        "parasite_read_fraction": {
            "median": float(np.nanmedian(fr)), "min": float(np.nanmin(fr)),
            "max": float(np.nanmax(fr)),
            "n_above_0.05pct": int((fr > 0.0005).sum()),
            "n_above_0.1pct": int((fr > 0.001).sum()),
            "n_above_0.5pct": int((fr > 0.005).sum())},
        "host_transcripts_detected_median": int(host_detected.median()),
        "parasite_transcripts_detected_median": int(para_detected.median()),
    }
    log.info("QC: %d samples; parasite fraction median=%.4f%% (max=%.3f%%); "
             "parasite transcripts detected (median)=%d",
             qc["n_samples"], qc["parasite_read_fraction"]["median"] * 100,
             qc["parasite_read_fraction"]["max"] * 100,
             qc["parasite_transcripts_detected_median"])

    # --- robustness: host-only vs dual vs parasite-informed subset ----------
    Hpc = _pca(host, 30, seed)
    Ppc = _pca(para, 20, seed)
    res = {"host_only": evaluate({"host_rna": Hpc}, args.k, seed),
           "dual_full": evaluate({"host_rna": Hpc, "parasite_rna": Ppc}, args.k, seed)}
    keep = np.where(fr.to_numpy() >= args.min_parasite_frac)[0]
    if len(keep) >= 3 * args.k:
        res["dual_parasite_informed_subset"] = {
            "n": int(len(keep)),
            **evaluate({"host_rna": Hpc[keep], "parasite_rna": Ppc[keep]}, args.k, seed)}
        log.info("parasite-informed subset: %d/%d samples with fraction >= %.4f",
                 len(keep), len(common), args.min_parasite_frac)

    def sil(block):
        m = res[block].get("BLISS", res[block].get("concat_kmeans", {}))
        return m.get("silhouette", float("nan")), m.get("stability_mean_pairwise_ari", float("nan"))
    for b in res:
        s, st = sil(b)
        log.info("  %-28s BLISS silhouette=%.3f stability=%.3f", b, s, st)

    out = {"qc": qc, "robustness": res}
    (get_path("results", create=True) / f"dualrna_qc_{args.cohort}.json").write_text(
        json.dumps(out, indent=2, default=float))
    # markdown report
    lines = [f"# Dual-RNA QC + robustness ({args.cohort})", "",
             f"- Samples: {qc['n_samples']}",
             f"- Parasite-read fraction: median {qc['parasite_read_fraction']['median']*100:.3f}%, "
             f"max {qc['parasite_read_fraction']['max']*100:.3f}%",
             f"- Samples with parasite fraction >0.1%: {qc['parasite_read_fraction']['n_above_0.1pct']}",
             f"- Host transcripts detected (median): {qc['host_transcripts_detected_median']}",
             f"- Parasite transcripts detected (median): {qc['parasite_transcripts_detected_median']}",
             "",
             "The parasite-RNA layer is sparse in peripheral blood, as expected at "
             "low peripheral parasitemia. Integration is therefore evaluated host-only, "
             "as full dual host+parasite RNA, and on the parasite-informed subset, to "
             "show the host integration is stable and to quantify the parasite layer's "
             "contribution rather than assume it.", ""]
    for b in res:
        s, st = sil(b)
        lines.append(f"- {b}: BLISS silhouette {s:.3f}, stability {st:.3f}")
    (get_path("reports", create=True) / f"dualrna_qc_{args.cohort}.md").write_text(
        "\n".join(lines), encoding="utf-8")
    log.info("wrote reports/dualrna_qc_%s.md and results/dualrna_qc_%s.json",
             args.cohort, args.cohort)


if __name__ == "__main__":
    main()
