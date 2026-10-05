#!/usr/bin/env python
"""Stage 6 -- pharmaceutical reference data: PubChem, ChEMBL, WHO (Section 20).

Fetches physicochemical + bioactivity reference data for the antimalarial APIs
that feed the formulation module (Section 68: PK = f(patient, drug, dose,
formulation)). All three are open.

  PubChem PUG REST : structure, formula, MW, logP, solubility descriptors
  ChEMBL API       : antimalarial bioactivity / potency for target compounds
  WHO              : records prequalification + guideline URLs (no bulk omics)

References:
  PubChem PUG REST: https://pubchem.ncbi.nlm.nih.gov/
  ChEMBL API:       https://www.ebi.ac.uk/chembl/
  WHO prequal:      https://extranet.who.int/prequal/medicines/prequalified-lists
"""
import _bootstrap  # noqa: F401
import argparse
import json
import urllib.parse
import urllib.request

from src.utils.config import get_dataset, dataset_dest, get_path
from src.utils.provenance import dataset_record, inventory_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Antimalarial APIs relevant to pediatric ACTs (extend as needed).
ANTIMALARIALS = [
    "artemether", "lumefantrine", "dihydroartemisinin", "piperaquine",
    "artesunate", "amodiaquine", "mefloquine", "pyronaridine",
    "sulfadoxine", "pyrimethamine", "chloroquine", "primaquine",
]
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_PROPS = ("MolecularFormula,MolecularWeight,CanonicalSMILES,"
             "XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,Complexity")


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Malaria_Project/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.warning("request failed: %s (%s)", url, e)
        return None


def fetch_pubchem(dest) -> int:
    rows = []
    for name in ANTIMALARIALS:
        url = f"{PUG}/compound/name/{urllib.parse.quote(name)}/property/{PUG_PROPS}/JSON"
        data = _get_json(url)
        if data and "PropertyTable" in data:
            props = data["PropertyTable"]["Properties"][0]
            props["query_name"] = name
            rows.append(props)
            log.info("PubChem: %s -> CID %s", name, props.get("CID"))
    out = dest / "antimalarial_physicochemical.json"
    dest.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("wrote %s (%d compounds)", out, len(rows))
    return len(rows)


def fetch_chembl(dest) -> int:
    # ChEMBL: resolve molecule by pref_name; save the molecule records.
    base = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
    rows = []
    for name in ANTIMALARIALS:
        url = f"{base}?q={urllib.parse.quote(name)}&format=json&limit=1"
        data = _get_json(url)
        mols = (data or {}).get("molecules", [])
        if mols:
            rows.append({"query_name": name,
                         "chembl_id": mols[0].get("molecule_chembl_id"),
                         "pref_name": mols[0].get("pref_name")})
            log.info("ChEMBL: %s -> %s", name, mols[0].get("molecule_chembl_id"))
    out = dest / "antimalarial_chembl.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("wrote %s (%d compounds)", out, len(rows))
    return len(rows)


def write_who_notes() -> None:
    who = get_dataset("formulation_who")
    note = get_path("access_notes", create=True) / "WHO_formulation_benchmark.md"
    note.write_text(
        "# WHO formulation benchmark references (Section 20.2, 7.5)\n\n"
        f"- Global Malaria Programme: {who['url_gmp']}\n"
        f"- Prequalified medicines list: {who['url_prequal']}\n\n"
        "Task: enumerate existing WHO-listed pediatric antimalarial dosage forms\n"
        "(dispersible tablets, granules for oral suspension) and record, for each,\n"
        "the pharmaceutical advantage a NEW liquid/suspension must beat (Section 7.5):\n"
        "flexible dosing in very small children, dose precision, acceptability,\n"
        "administration volume, exposure, stability.\n",
        encoding="utf-8",
    )
    log.info("wrote WHO benchmark note: %s", note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-pubchem", action="store_true")
    ap.add_argument("--skip-chembl", action="store_true")
    args = ap.parse_args()

    dest = dataset_dest("reference_pubchem")
    n_pc = 0 if args.skip_pubchem else fetch_pubchem(dest)
    dest_ch = dataset_dest("reference_chembl")
    n_ch = 0 if args.skip_chembl else fetch_chembl(dest_ch)
    write_who_notes()

    dataset_record(
        accession="pharma_references", repository="PubChem+ChEMBL+WHO",
        download_url="https://pubchem.ncbi.nlm.nih.gov/", dest_dir=dest,
        license_or_access_terms="open",
        raw_or_processed_status="reference_descriptors",
        files=inventory_dir(dest, compute_checksums=False),
        notes=f"PubChem {n_pc} compounds; ChEMBL {n_ch}; WHO benchmark noted.",
    )
    log.info("done. PubChem=%d ChEMBL=%d.", n_pc, n_ch)


if __name__ == "__main__":
    main()
