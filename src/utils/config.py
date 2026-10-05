"""Configuration loading and path resolution.

All scripts should start with::

    from src.utils.config import load_config, get_path
    cfg = load_config()

Paths declared in ``config/project.yaml`` are resolved against
``project.root`` so the same code runs on the workstation and on any mirror.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/utils/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and merge all config YAMLs into a single dict.

    Returns keys: ``project``, ``paths``, ``compute``, ``reproducibility``,
    ``logging`` (from project.yaml), plus ``datasets``, ``models``, ``seeds``.
    """
    cfg = _read_yaml(CONFIG_DIR / "project.yaml")
    cfg["datasets"] = _read_yaml(CONFIG_DIR / "datasets.yaml")
    cfg["models"] = _read_yaml(CONFIG_DIR / "models.yaml")
    cfg["seeds_cfg"] = _read_yaml(CONFIG_DIR / "seeds.yaml")
    return cfg


def project_root() -> Path:
    """Absolute project root.

    Uses ``project.root`` from config, but falls back to the repo root if that
    path does not exist (e.g. when running on a machine other than the
    workstation). An env override ``MALARIA_PROJECT_ROOT`` wins if set.
    """
    env = os.environ.get("MALARIA_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_config()
    declared = Path(cfg["project"]["root"]).expanduser()
    return declared if declared.exists() else REPO_ROOT


def get_path(key: str, create: bool = False) -> Path:
    """Resolve a named path from ``config/project.yaml:paths``.

    Parameters
    ----------
    key : one of the keys under ``paths`` (e.g. ``"data_raw"``, ``"logs"``).
    create : if True, create the directory (and parents) if missing.
    """
    cfg = load_config()
    rel = cfg["paths"][key]
    p = Path(rel)
    if not p.is_absolute():
        p = project_root() / rel
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_dataset(name: str) -> dict[str, Any]:
    """Return the config block for a dataset in ``config/datasets.yaml``."""
    ds = load_config()["datasets"]
    if name not in ds:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(ds)}")
    return ds[name]


def dataset_dest(name: str, create: bool = True) -> Path:
    """Absolute destination directory for a dataset's raw files."""
    block = get_dataset(name)
    p = project_root() / block["dest"]
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def seeds() -> list[int]:
    return list(load_config()["seeds_cfg"]["seeds"])


if __name__ == "__main__":
    c = load_config()
    print("project.root  :", c["project"]["root"])
    print("resolved root :", project_root())
    print("seeds         :", seeds())
    print("datasets      :", sorted(c["datasets"]))
