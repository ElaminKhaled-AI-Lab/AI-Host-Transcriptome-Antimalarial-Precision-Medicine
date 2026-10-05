#!/usr/bin/env python
"""Phase 15 -- HPAMM-Net ablation experiments (blueprint Section 75).

Compares the full model against systematically disabled components to identify
which parts actually add value. Ablations are listed in config/models.yaml.

STUB: iterates the ablation matrix; each entry toggles a model flag and re-runs.
"""
import _bootstrap  # noqa: F401
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    ablations = load_config()["models"]["ablations"]
    for a in ablations:
        log.info("[STUB] ablation: %s (re-train, compare to full model)", a)
    log.info("done (stub). Report delta vs full model per ablation (Section 75).")


if __name__ == "__main__":
    main()
