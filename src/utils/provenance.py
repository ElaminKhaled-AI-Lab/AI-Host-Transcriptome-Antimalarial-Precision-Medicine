"""Provenance, checksums, and reproducibility manifests.

Implements the data-provenance rules (blueprint Section 23) and the
reproducibility manifest (Section 90). Every download writes a registry
record; every result should write a manifest.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from src.utils.config import get_path, project_root


# ---------------------------------------------------------------- checksums
def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (memory-safe for large raw files)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_checksum(path: str | Path) -> Path:
    """Compute and store a checksum for one file under metadata/checksums/."""
    path = Path(path)
    digest = sha256(path)
    out = get_path("checksums", create=True) / f"{path.name}.sha256"
    out.write_text(f"{digest}  {path}\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------- registry
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def dataset_record(
    accession: str,
    repository: str,
    download_url: str,
    dest_dir: str | Path,
    *,
    license_or_access_terms: str = "",
    publication: str = "",
    raw_or_processed_status: str = "",
    sample_count: int | None = None,
    organism: str = "",
    tissue: str = "",
    platform_: str = "",
    software: str = "",
    files: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> Path:
    """Write a dataset provenance record (Section 23 fields) as JSON.

    ``files`` is a list of ``{"file_name","file_size","checksum"}`` dicts.
    Returns the path of the written record.
    """
    rec = {
        "accession": accession,
        "repository": repository,
        "download_url": download_url,
        "download_date": _now(),
        "dest_dir": str(dest_dir),
        "license_or_access_terms": license_or_access_terms,
        "publication": publication,
        "raw_or_processed_status": raw_or_processed_status,
        "sample_count": sample_count,
        "organism": organism,
        "tissue": tissue,
        "platform": platform_,
        "software": software,
        "files": files or [],
        "notes": notes,
    }
    # Sanitize the accession into a safe filename: some accessions contain
    # slashes/spaces (e.g. "E-MTAB-6413 / ERP116722 / PRJEB33892"), which would
    # otherwise be read as non-existent subdirectories and crash the write.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", accession).strip("_") or "dataset"
    out = get_path("dataset_registry", create=True) / f"{safe}.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return out


def inventory_dir(dest_dir: str | Path, compute_checksums: bool = True) -> list[dict[str, Any]]:
    """Build a file inventory (name, size, optional checksum) for a directory."""
    dest = Path(dest_dir)
    out: list[dict[str, Any]] = []
    for f in sorted(dest.rglob("*")):
        if f.is_file():
            entry = {"file_name": str(f.relative_to(dest)), "file_size": f.stat().st_size}
            if compute_checksums:
                entry["checksum"] = sha256(f)
            out.append(entry)
    return out


# ---------------------------------------------------------------- manifest
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def result_manifest(
    name: str,
    *,
    seed: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
    train_split: Any = None,
    test_split: Any = None,
    inputs: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a reproducibility manifest for a result (Section 90)."""
    man = {
        "name": name,
        "date": _now(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "hyperparameters": hyperparameters or {},
        "train_split": train_split,
        "test_split": test_split,
        "inputs": inputs or [],
        "extra": extra or {},
    }
    out = get_path("results", create=True) / f"{name}.manifest.json"
    out.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return out
