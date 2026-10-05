#!/usr/bin/env python
"""Phase 4 -- fetch per-sample severity labels for the dual-RNA cohorts.

The real-data validation (validate_on_real_dualrna.py) can score ARI against the
true clinical group if per-sample labels exist. ENA's standard run fields don't
carry severity, but each sample's BioSample record does (SAMPLE_ATTRIBUTE tags).
This script:
  1. reads the ENA read_run report (run <-> sample) saved during download,
  2. fetches each sample's BioSample XML from ENA,
  3. extracts every attribute tag/value and maps clinical fields to a severity
     group (CM = cerebral malaria, SMA = severe malarial anaemia,
     UM = uncomplicated, SEVERE = unspecified severe),
  4. writes metadata/sample_maps/dualrna_labels_<cohort>.tsv keyed by RUN accession
     (which is how validate_on_real_dualrna.py names samples), and
  5. dumps every distinct attribute tag it saw, so you can confirm the mapping.

If no severity field is found, the script says so and lists the tags that ARE
present -- then tell me which tag holds the group and I will adjust the mapping.

Usage:
  python scripts/05_sample_mapping/fetch_dualrna_labels.py --cohort real_mali
  python scripts/05_sample_mapping/fetch_dualrna_labels.py --cohort external_gambia
"""
import _bootstrap  # noqa: F401
import argparse
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.utils.config import dataset_dest, get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

COHORT_DATASET = {"real_mali": "dual_rna_mali", "external_gambia": "dual_rna_gambia"}

# keyword -> normalised group (checked against lowercased tag + value)
RULES = [
    (r"cerebral", "CM"),
    (r"(severe.*an[ae]mia|\bsma\b|anemia|anaemia)", "SMA"),
    (r"uncomplicated", "UM"),
    (r"\bsevere\b", "SEVERE"),
    (r"\bcontrol\b|healthy", "CONTROL"),
]
CLINICAL_TAG_HINTS = ("disease", "phenotype", "clinical", "status", "group",
                      "syndrome", "condition", "diagnosis", "severity", "cm_sma",
                      "malaria")


def fetch_xml(acc, retries=4, backoff=3.0):
    url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{acc}"
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Malaria_Project/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode(errors="ignore")
        except Exception as e:  # noqa: BLE001
            if i < retries:
                time.sleep(backoff * i)
            else:
                log.warning("XML fetch failed for %s: %s", acc, e)
    return None


def parse_attributes(xml_text):
    out = {}
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out
    for attr in root.iter("SAMPLE_ATTRIBUTE"):
        tag = (attr.findtext("TAG") or "").strip()
        val = (attr.findtext("VALUE") or "").strip()
        if tag:
            out[tag] = val
    return out


def classify(attrs):
    """Return (group, source_tag, source_value) or (None, None, None)."""
    # prefer clinically-named tags first
    items = sorted(attrs.items(),
                   key=lambda kv: 0 if any(h in kv[0].lower() for h in CLINICAL_TAG_HINTS) else 1)
    for tag, val in items:
        text = f"{tag} {val}".lower()
        for pat, grp in RULES:
            if re.search(pat, text):
                return grp, tag, val
    return None, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=list(COHORT_DATASET), required=True)
    ap.add_argument("--max", type=int, default=None, help="cap samples (debug)")
    args = ap.parse_args()

    dataset = COHORT_DATASET[args.cohort]
    dest = dataset_dest(dataset, create=False)
    report = next(iter(sorted(dest.glob("**/*_read_run_report.tsv"))), None)
    if not report:
        log.error("read_run report not found under %s (run the downloader first)", dest)
        return
    df = pd.read_csv(report, sep="\t")
    pairs = df[["run_accession", "sample_accession"]].dropna().drop_duplicates()
    if args.max:
        pairs = pairs.head(args.max)
    log.info("%s: %d runs across %d samples", args.cohort,
             len(pairs), pairs["sample_accession"].nunique())

    # fetch each distinct sample once
    from collections import Counter
    sample_group = {}
    all_tags = set()
    clinical_values = {}   # tag -> Counter(value) for clinically-hinted tags
    for s in sorted(pairs["sample_accession"].unique()):
        xml = fetch_xml(s)
        attrs = parse_attributes(xml) if xml else {}
        all_tags.update(attrs)
        for tag, val in attrs.items():
            if any(h in tag.lower() for h in CLINICAL_TAG_HINTS):
                clinical_values.setdefault(tag, Counter())[val] += 1
        grp, tag, val = classify(attrs)
        # capture age / sex when the BioSample exposes them (used by the
        # formulation-omics arm to restrict the cohort to children)
        age = sex = None
        for k, v in attrs.items():
            kl = k.lower()
            if age is None and kl in ("age", "age_years", "host_age", "patient age"):
                m = re.search(r"[-+]?\d*\.?\d+", str(v))
                if m:
                    age = float(m.group())
                    if "month" in str(v).lower():
                        age /= 12.0
            if sex is None and kl in ("sex", "gender", "host_sex"):
                sex = str(v)
        sample_group[s] = (grp, tag, val, age, sex)

    rows = []
    for _, r in pairs.iterrows():
        grp, tag, val, age, sex = sample_group.get(
            r["sample_accession"], (None, None, None, None, None))
        rows.append({"sample_id": r["run_accession"],
                     "sample_accession": r["sample_accession"],
                     "group": grp, "source_tag": tag, "source_value": val,
                     "age": age, "sex": sex})
    out = pd.DataFrame(rows)
    labelled = out["group"].notna().sum()
    dest_dir = get_path("sample_maps", create=True)
    out_path = dest_dir / f"dualrna_labels_{args.cohort}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    (dest_dir / f"dualrna_attribute_tags_{args.cohort}.txt").write_text(
        "\n".join(sorted(all_tags)), encoding="utf-8")
    # diagnostic: the actual VALUES behind clinically-named tags (disease, phenotype,...)
    diag_lines = []
    for tag in sorted(clinical_values):
        diag_lines.append(f"[{tag}]")
        for val, n in clinical_values[tag].most_common():
            diag_lines.append(f"  {n:>4}x  {val!r}")
        diag_lines.append("")
    diag_path = dest_dir / f"dualrna_clinical_values_{args.cohort}.txt"
    diag_path.write_text("\n".join(diag_lines), encoding="utf-8")

    log.info("WROTE %s: %d/%d runs labelled", out_path, labelled, len(out))
    if labelled:
        log.info("group counts:\n%s", out["group"].value_counts().to_string())
    else:
        log.warning("No severity field matched. Attribute tags seen were:\n%s",
                    "\n".join(sorted(all_tags)))
        log.warning("Distinct values behind clinically-named tags (see %s):", diag_path.name)
        for tag in sorted(clinical_values):
            vals = ", ".join(f"{v!r}({n})" for v, n in clinical_values[tag].most_common(8))
            log.warning("  %-18s -> %s", tag, vals)
        log.warning("Tell me which tag/value holds the clinical group and I will map it.")


if __name__ == "__main__":
    main()
