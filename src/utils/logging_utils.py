"""Uniform logging for all scripts.

Usage::

    from src.utils.logging_utils import get_logger
    log = get_logger(__name__)
    log.info("starting download")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    # Local import to avoid a circular import at module load.
    try:
        from src.utils.config import load_config, get_path

        cfg = load_config().get("logging", {})
        level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
        to_file = bool(cfg.get("to_file", True))
        logfile = cfg.get("logfile", "logs/malaria_project.log")
    except Exception:  # config not available yet -- safe defaults
        level, to_file, logfile = logging.INFO, False, "logs/malaria_project.log"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if to_file:
        try:
            from src.utils.config import project_root

            path = project_root() / logfile
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path, encoding="utf-8"))
        except Exception:
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
