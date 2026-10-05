#!/usr/bin/env python
"""Phase 0 -- verify config, paths, and key tools before downloading.

Checks: config loads, project root resolves, required Python packages import,
and optional bioinformatics CLIs are on PATH. Reports what is missing without
failing hard, so you can fix the environment incrementally.
"""
import _bootstrap  # noqa: F401
import importlib
import shutil

from src.utils.config import load_config, project_root, get_path
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

PY_PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "yaml", "matplotlib"]
PY_OPTIONAL = ["torch", "xgboost", "optuna", "shap", "malariagen_data",
               "ppx", "pridepy", "mofapy2", "snfpy"]
CLI_TOOLS = ["prefetch", "fasterq-dump", "esearch", "wget", "curl", "aria2c",
             "salmon", "fastqc"]


def _check_import(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def main() -> None:
    cfg = load_config()
    log.info("config OK: %d datasets, %d seeds",
             len(cfg["datasets"]), len(cfg["seeds_cfg"]["seeds"]))
    log.info("project.root (declared): %s", cfg["project"]["root"])
    log.info("project root (resolved): %s", project_root())

    log.info("--- required Python packages ---")
    for m in PY_PACKAGES:
        log.info("  [%s] %s", "OK " if _check_import(m) else "MISS", m)

    log.info("--- optional Python packages ---")
    for m in PY_OPTIONAL:
        log.info("  [%s] %s", "OK " if _check_import(m) else "opt ", m)

    log.info("--- command-line tools ---")
    for t in CLI_TOOLS:
        log.info("  [%s] %s", "OK " if shutil.which(t) else "MISS", t)

    log.info("--- writable output dirs ---")
    for key in ("data_raw", "results", "logs", "tmp"):
        p = get_path(key, create=True)
        log.info("  [%s] %s", "OK " if p.exists() else "MISS", p)

    log.info("Environment check complete.")


if __name__ == "__main__":
    main()
