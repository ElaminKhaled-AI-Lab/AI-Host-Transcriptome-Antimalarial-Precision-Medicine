#!/usr/bin/env python
"""Phase 4 -- master sample map + cross-omic alignment (blueprint Sections 24, 95 Gate 1).

Consolidates the sample identifiers of every PROCESSED layer into one master map
and reports how well the layers align. Alignment across host layers is only
honest when a crosswalk exists (the layers use different id schemes:
transcriptome uses clinical/library codes, proteome uses Sample_NN), so this
script:

  1. inventories the sample ids of each processed matrix
  2. loads a crosswalk if the user has placed one in metadata/sample_maps/
     (a supplementary table linking subjects across omics; any .tsv/.csv/.xlsx
      whose name contains 'crosswalk' or 'manifest' or 'sample_map')
  3. attaches the transcriptome clinical table (already keyed by library id)
  4. writes the master map and an alignment report, stating plainly which layers
     are matched and which still need the crosswalk (Gate 1 honesty).

CRITICAL (Section 7.2): unmatched cohorts are NOT joined at the patient level.
Where no crosswalk links two layers, they are reported as separate cohorts, not
fused. This protects every downstream model from a false patient match.
"""
import _bootstrap  # noqa: F401
import re
from pathlib import Path

import pandas as pd

from src.utils.config import get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

PROCESSED = {
    "host_transcriptome": "host_transcriptomics/transcriptome_patient_by_gene.tsv",
    "host_proteome": "host_proteomics/proteome_patient_by_protein.tsv",
    "host_metabolome": "host_metabolomics/metabolome_patient_by_feature.tsv",
    "parasite_genome": "parasite_genomics/parasite_feature_matrix.tsv",
}


def _norm(s: str) -> str:
    return re.sub(r"[-_ ]+", "-", str(s).strip()).upper()


def _sample_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return pd.read_csv(path, sep="\t", usecols=[0]).iloc[:, 0].astype(str).tolist()


def _find_crosswalk(sm_dir: Path) -> Path | None:
    pats = ("crosswalk", "manifest", "sample_map", "sample-map", "samplemap")
    for f in sm_dir.glob("*"):
        if f.suffix.lower() in (".tsv", ".csv", ".xlsx", ".xls") and \
           any(p in f.name.lower() for p in pats) and \
           "master" not in f.name.lower() and "transcriptome_clinical" not in f.name.lower():
            return f
    return None


def main() -> None:
    proc_root = get_path("data_processed")
    sm_dir = get_path("sample_maps", create=True)

    # 1. inventory each processed layer
    inv = {}
    for layer, rel in PROCESSED.items():
        ids = _sample_ids(proc_root / rel)
        inv[layer] = ids
        log.info("%-20s %d samples", layer, len(ids))

    rows = []
    for layer, ids in inv.items():
        for sid in ids:
            rows.append({"layer": layer, "sample_id": sid, "join_key": _norm(sid)})
    master = pd.DataFrame(rows)

    # 2. attach transcriptome clinical (keyed by normalised library id)
    clin_path = sm_dir / "transcriptome_clinical.tsv"
    if clin_path.exists():
        clin = pd.read_csv(clin_path, sep="\t")
        if "join_key" in clin.columns:
            keep = [c for c in clin.columns if c.startswith("clin_") or c in
                    ("join_key", "gsm", "title", "library")]
            master = master.merge(clin[keep], on="join_key", how="left")
            log.info("attached transcriptome clinical labels")

    # 3. load crosswalk if provided
    xwalk = _find_crosswalk(sm_dir)
    if xwalk:
        log.info("found crosswalk: %s", xwalk.name)
        try:
            cw = (pd.read_excel(xwalk) if xwalk.suffix.lower().startswith(".xl")
                  else pd.read_csv(xwalk, sep=None, engine="python"))
            log.info("crosswalk columns: %s", list(cw.columns)[:20])
            log.info("-> A crosswalk is present. Tell me which columns hold the "
                     "per-omics ids and I will wire the exact patient-level join.")
            cw.to_csv(sm_dir / "crosswalk_normalised.tsv", sep="\t", index=False)
        except Exception as e:  # noqa: BLE001
            log.warning("could not parse crosswalk %s: %s", xwalk.name, e)
    else:
        log.info("no crosswalk file found in %s", sm_dir)
        log.info("Drop the paper's supplementary manifest there (name it e.g. "
                 "'sample_crosswalk.xlsx') to enable the patient-level host join.")

    # 4. write master map + alignment report
    out = sm_dir / "master_sample_map.tsv"
    master.to_csv(out, sep="\t", index=False)
    log.info("WROTE master map -> %s (%d rows)", out, len(master))

    report = get_path("reports", create=True) / "alignment_report.md"
    lines = ["# Cross-omic alignment report (Gate 1)", "",
             "## Per-layer sample counts"]
    for layer, ids in inv.items():
        lines.append(f"- {layer}: {len(ids)} samples")
    lines += ["", "## Alignment status", ""]
    host = ["host_transcriptome", "host_proteome", "host_metabolome"]
    present_host = [l for l in host if inv.get(l)]
    lines.append(f"- Host layers present: {', '.join(present_host) or 'none'}")
    lines.append("- Host layers use DIFFERENT id schemes (clinical codes vs "
                 "Sample_NN); a crosswalk is required to match patients across "
                 "them (Section 24). Do NOT fuse without it (Boundary 7.2).")
    lines.append(f"- Crosswalk present: {'YES -> ' + xwalk.name if xwalk else 'NO'}")
    lines.append("- Parasite genome is a SEPARATE real cohort (Pf8); it is joined "
                 "to the host cohorts only through the semi-synthetic generator, "
                 "never at the patient level (Section 33).")
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("alignment report -> %s", report)
    log.info("Gate 1 verdict: %s",
             "crosswalk present, ready to finalise host match" if xwalk else
             "per-layer inventory done; host patient-match needs the crosswalk")


if __name__ == "__main__":
    main()
