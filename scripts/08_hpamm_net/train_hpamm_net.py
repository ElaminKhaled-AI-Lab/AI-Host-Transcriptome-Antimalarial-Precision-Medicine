#!/usr/bin/env python
"""Phase 15 -- train HPAMM-Net incrementally (blueprint Sections 48-56, 82).

HPAMM-Net = Host-Parasite Adaptive Missingness-aware Multi-task Network
(provisional; novelty not assumed -- Section 48/80).

Incremental development (Section 82), each version compared with the prior:
  v0 clinical + parasite-marker baseline
  v1 separate host/parasite encoders
  v2 + modality gating
  v3 + modality dropout
  v4 + multi-task heads
  v5 + treatment conditioning
  v6 + biological regularization
  v7 + uncertainty estimation
  v8 + domain adaptation

Uses src.models.hpamm_net.HPAMMNet (PyTorch). Requires torch.
STUB: wires config -> model construction + training loop skeleton.
"""
import _bootstrap  # noqa: F401
import argparse

from src.utils.config import load_config, seeds
from src.utils.seeds import set_global_seed
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None, help="override hpamm_net.version")
    args = ap.parse_args()

    cfg = load_config()["models"]["hpamm_net"]
    version = args.version or cfg["version"]
    log.info("HPAMM-Net %s", version)

    try:
        import torch  # noqa: F401
        from src.models.hpamm_net import HPAMMNet, multitask_loss  # noqa: F401
    except Exception as e:
        log.error("PyTorch required: %s. See envs/environment.yml.", e)
        return

    for seed in seeds():
        set_global_seed(seed)
        log.info("[STUB] seed=%d: build encoders %s, latent=%d, "
                 "gating=%s, dropout=%.2f, tasks=%s",
                 seed, cfg["encoders"]["modalities"],
                 cfg["encoders"]["latent_dim_per_modality"],
                 cfg["gated_fusion"], cfg["modality_dropout_p"],
                 list(cfg["multitask_heads"]))
        # TODO:
        #   input_dims = {modality: feature_dim, ...} from data/processed
        #   model = HPAMMNet(input_dims, latent_dim=..., tasks=..., ...)
        #   train with multitask_loss; nested CV; save checkpoint + manifest

    log.info("DECISION GATE 4 (Section 95): HPAMM-Net must beat baselines on "
             "generalization / calibration / missingness / stability / "
             "multi-task / interpretability -- not just headline accuracy.")


if __name__ == "__main__":
    main()
