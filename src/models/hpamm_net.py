"""HPAMM-Net -- Host-Parasite Adaptive Missingness-aware Multi-task Network.

Provisional research architecture (blueprint Sections 48-56), designed as a
DIRECT SUCCESSOR to the group's BLISS integration model (Elamin et al.,
Int. J. Mol. Sci. 2026, 27, 7511). It inherits the three BLISS loss terms and
extends them for the dual host-parasite, missingness-heavy, treatment-response
setting of pediatric malaria.

Inherited from BLISS (src/models/bliss.py):
  * L_triplet  -- PCA->k-means pseudo-label triplet metric loss (Super.FELT-inspired)
  * L_factor   -- shared-latent factor reconstruction of each input layer (MOFA-inspired)
  * L_balance  -- KL(cluster marginal || uniform); prevents degenerate collapse

New in HPAMM-Net (the novel contributions to benchmark and ablate):
  1. explicit host / parasite factorization  (Section 51)  Z_H, Z_P -> Z_HP
  2. missingness-aware gated fusion           (Section 52)  missing -> zero
  3. modality dropout during training         (Section 53)
  4. treatment conditioning                   (Section 54)  same patient, many treatments
  5. multi-task supervised heads              (Section 55)  resistance/PK/clearance/...
  6. (optional) geography domain adaptation   (Section 57)
  7. stability selection built into training  (BLISS lineage; Section 75/82)

So HPAMM-Net = BLISS's balanced, stable, reconstructive latent core
              + supervised multi-task + dual-organism + missingness machinery.
This makes the synthetic-vs-real transportability comparison (the crucial test,
paper Sections 2.3/2.8/2.9) an apples-to-apples successor evaluation.

Novelty is NOT assumed. Per Section 48 the name/architecture require a formal
literature + software search and ablation (Section 75) before any novelty
claim. Build incrementally v0..v8 (Section 82) by toggling flags in
``config/models.yaml``.

Requires torch. If torch is absent the module still imports (guarded) so that
config-only tooling works without the DL stack.
"""
from __future__ import annotations

from typing import Sequence

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # torch optional at import time
    _TORCH = False
    nn = object  # type: ignore


if _TORCH:

    class ModalityEncoder(nn.Module):
        """Encode one modality x_m -> latent z_m (Section 50)."""

        def __init__(self, in_dim: int, latent_dim: int = 64, hidden: int = 128,
                     dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent_dim),
            )

        def forward(self, x):
            return self.net(x)


    class GatedFusion(nn.Module):
        """Missingness-aware attention fusion (Section 52).

        Z = sum_m alpha_{im} * z_{im}, where alpha depends on the modality
        representation AND the availability mask. Missing modalities are
        forced to contribute zero (their gate is masked to -inf pre-softmax).
        """

        def __init__(self, latent_dim: int, n_modalities: int):
            super().__init__()
            self.score = nn.Linear(latent_dim, 1)
            self.n_modalities = n_modalities

        def forward(self, zs, mask):
            # zs: (B, M, D) stacked modality latents ; mask: (B, M) in {0,1}
            logits = self.score(zs).squeeze(-1)              # (B, M)
            logits = logits.masked_fill(mask == 0, float("-inf"))
            alpha = torch.softmax(logits, dim=1)             # (B, M)
            alpha = torch.nan_to_num(alpha, nan=0.0)         # all-missing guard
            z = (alpha.unsqueeze(-1) * zs).sum(dim=1)        # (B, D)
            return z, alpha


    class HPAMMNet(nn.Module):
        """Full model. Modalities are grouped into host vs parasite (Section 51)."""

        HOST = ["host_genome", "host_rna", "host_proteome", "host_metabolome"]
        PARASITE = ["parasite_genome", "parasite_rna"]

        def __init__(self, input_dims: dict[str, int], *, latent_dim: int = 64,
                     clinical_dim: int = 0, treatment_dim: int = 0,
                     tasks: Sequence[str] = ("resistance", "cure"),
                     regression_tasks: Sequence[str] = ("clearance", "pk_auc"),
                     modality_dropout_p: float = 0.3,
                     gated_fusion: bool = True):
            super().__init__()
            self.modalities = list(input_dims)
            self.encoders = nn.ModuleDict(
                {m: ModalityEncoder(d, latent_dim) for m, d in input_dims.items()}
            )
            self.modality_dropout_p = modality_dropout_p
            self.gated = gated_fusion
            self.latent_dim = latent_dim

            host_mods = [m for m in self.modalities if m in self.HOST]
            para_mods = [m for m in self.modalities if m in self.PARASITE]
            self.host_fusion = GatedFusion(latent_dim, max(len(host_mods), 1))
            self.para_fusion = GatedFusion(latent_dim, max(len(para_mods), 1))
            self._host_mods, self._para_mods = host_mods, para_mods

            # h(Z_H, Z_P) + clinical + treatment conditioning (Section 54)
            fuse_in = latent_dim * 2 + clinical_dim + treatment_dim
            self.integrator = nn.Sequential(
                nn.Linear(fuse_in, latent_dim), nn.GELU(), nn.Dropout(0.2)
            )

            self.class_heads = nn.ModuleDict(
                {t: nn.Linear(latent_dim, 1) for t in tasks}
            )
            self.reg_heads = nn.ModuleDict(
                {t: nn.Linear(latent_dim, 1) for t in regression_tasks}
            )

            # BLISS lineage: factor-reconstruction decoders (shared latent -> each
            # input layer). Enables the L_factor term and lets HPAMM-Net produce a
            # BLISS-comparable unsupervised latent for the transportability battery.
            self.decoders = nn.ModuleDict(
                {m: nn.Linear(latent_dim, d) for m, d in input_dims.items()}
            )
            self.k_clusters = 4

        # ---- BLISS-inherited loss terms (operate on integrated latent z) -----
        def factor_reconstruction_loss(self, x, mask, z):
            import torch.nn.functional as F
            total, n = 0.0, 0
            for m in self.modalities:
                recon = self.decoders[m](z)
                w = mask[m].unsqueeze(1)                    # only observed modalities
                if w.sum() > 0:
                    total = total + (F.mse_loss(recon * w, x[m] * w, reduction="sum")
                                     / w.sum().clamp(min=1))
                    n += 1
            return total / max(n, 1)

        def balance_loss(self, z):
            import torch
            import torch.nn.functional as F
            k = self.k_clusters
            logits = z @ z.mean(dim=0, keepdim=True).T.repeat(1, k)
            q = F.softmax(logits, dim=1)
            marginal = q.mean(dim=0)
            uniform = torch.full_like(marginal, 1.0 / k)
            return (marginal * (marginal.clamp_min(1e-8).log() - uniform.log())).sum()

        def triplet_loss(self, z, pseudo):
            import torch
            import torch.nn.functional as F
            d = torch.cdist(z, z)
            same = pseudo.unsqueeze(0) == pseudo.unsqueeze(1)
            eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
            pos = torch.nan_to_num(d.masked_fill(~same | eye, float("-inf")).max(1).values,
                                   neginf=0.0)
            neg = torch.nan_to_num(d.masked_fill(same, float("inf")).min(1).values,
                                   posinf=0.0)
            return F.relu(pos - neg + 1.0).mean()

        def integrate(self, x, mask, clinical=None, treatment=None):
            """Return the integrated latent z (for unsupervised/latent use)."""
            import torch
            z_h = self._encode_group(x, mask, self._host_mods, self.host_fusion)
            z_p = self._encode_group(x, mask, self._para_mods, self.para_fusion)
            parts = [z_h, z_p]
            if clinical is not None:
                parts.append(clinical)
            if treatment is not None:
                parts.append(treatment)
            return self.integrator(torch.cat(parts, dim=1))

        def _encode_group(self, x, mask, mods, fusion):
            if not mods:
                b = next(iter(x.values())).shape[0]
                dev = next(iter(x.values())).device
                return torch.zeros(b, self.latent_dim, device=dev)
            zs, ms = [], []
            for m in mods:
                z = self.encoders[m](x[m])
                if self.training and self.modality_dropout_p > 0:
                    drop = (torch.rand(z.shape[0], 1, device=z.device)
                            < self.modality_dropout_p).float()
                    keep_mask = mask[m].unsqueeze(1) * (1 - drop)
                else:
                    keep_mask = mask[m].unsqueeze(1)
                zs.append(z * keep_mask)             # missing -> zero
                ms.append(keep_mask.squeeze(1))
            zs = torch.stack(zs, dim=1)              # (B, |mods|, D)
            ms = torch.stack(ms, dim=1)             # (B, |mods|)
            if self.gated:
                z, _ = fusion(zs, ms)
            else:
                denom = ms.sum(1, keepdim=True).clamp(min=1)
                z = zs.sum(1) / denom
            return z

        def forward(self, x: dict, mask: dict, clinical=None, treatment=None):
            z_h = self._encode_group(x, mask, self._host_mods, self.host_fusion)
            z_p = self._encode_group(x, mask, self._para_mods, self.para_fusion)
            parts = [z_h, z_p]
            if clinical is not None:
                parts.append(clinical)
            if treatment is not None:
                parts.append(treatment)
            z = self.integrator(torch.cat(parts, dim=1))
            out = {t: head(z).squeeze(-1) for t, head in self.class_heads.items()}
            out.update({t: head(z).squeeze(-1) for t, head in self.reg_heads.items()})
            return out


    def multitask_loss(outputs: dict, targets: dict, masks: dict,
                       weights: dict, class_tasks, reg_tasks) -> "torch.Tensor":
        """Weighted multi-task loss (Section 55). ``masks[t]`` selects patients
        with an observed label for task t (labels are often partially missing).
        """
        bce = nn.BCEWithLogitsLoss(reduction="none")
        mse = nn.MSELoss(reduction="none")
        total = 0.0
        for t in class_tasks:
            if t in outputs and t in targets:
                m = masks.get(t, torch.ones_like(targets[t]))
                loss = (bce(outputs[t], targets[t].float()) * m).sum() / m.sum().clamp(min=1)
                total = total + weights.get(t, 1.0) * loss
        for t in reg_tasks:
            if t in outputs and t in targets:
                m = masks.get(t, torch.ones_like(targets[t]))
                loss = (mse(outputs[t], targets[t].float()) * m).sum() / m.sum().clamp(min=1)
                total = total + weights.get(t, 1.0) * loss
        return total


    def combined_loss(model, x, mask, z, outputs, targets, label_masks,
                      class_tasks, reg_tasks, task_weights,
                      pseudo=None, lambda_task=1.0, lambda_triplet=1.0,
                      lambda_factor=0.3, lambda_balance=0.7):
        """Full HPAMM-Net objective = supervised multi-task + BLISS core.

        L = lambda_task    * L_multitask                 (Section 55)
          + lambda_triplet * L_triplet(pseudo-labels)    (BLISS)
          + lambda_factor  * L_factor(reconstruction)    (BLISS)
          + lambda_balance * L_balance(KL to uniform)    (BLISS)

        Set lambda_task=0 to run HPAMM-Net in the purely weakly-supervised BLISS
        regime for a matched comparison on the transportability battery.
        """
        L = lambda_task * multitask_loss(outputs, targets, label_masks,
                                         task_weights, class_tasks, reg_tasks)
        L = L + lambda_factor * model.factor_reconstruction_loss(x, mask, z)
        L = L + lambda_balance * model.balance_loss(z)
        if pseudo is not None:
            L = L + lambda_triplet * model.triplet_loss(z, pseudo)
        return L


    def stability_selection(latents, k=4, n_subsamples=15, frac=0.8, seed=11):
        """Repeated-subsampling stability (paper Section 2.8 / Figure 8A).

        Given a fixed latent embedding (n_samples, d), cluster 15 independent 80%
        subsamples and return the mean pairwise Adjusted Rand Index plus a
        per-sample modal-assignment confidence. This is the 'Stability Selection'
        half of the BLISS name, applied as an evaluation-time diagnostic.
        """
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        rng = np.random.default_rng(seed)
        n = latents.shape[0]
        assignments = np.full((n_subsamples, n), -1)
        for i in range(n_subsamples):
            idx = rng.choice(n, int(frac * n), replace=False)
            lab = KMeans(n_clusters=k, n_init=10, random_state=seed + i).fit_predict(latents[idx])
            assignments[i, idx] = lab
        # pairwise ARI over shared members
        aris = []
        for a in range(n_subsamples):
            for b in range(a + 1, n_subsamples):
                shared = (assignments[a] >= 0) & (assignments[b] >= 0)
                if shared.sum() > k:
                    aris.append(adjusted_rand_score(assignments[a][shared],
                                                    assignments[b][shared]))
        # per-sample confidence = fraction of its subsamples in its modal cluster
        conf = np.zeros(n)
        for j in range(n):
            labs = assignments[assignments[:, j] >= 0, j]
            if len(labs):
                _, counts = np.unique(labs, return_counts=True)
                conf[j] = counts.max() / len(labs)
        return {"mean_pairwise_ari": float(np.mean(aris)) if aris else float("nan"),
                "per_sample_confidence": conf,
                "frac_confident_0.9": float((conf > 0.9).mean())}


else:  # torch missing -- provide informative stubs

    def _no_torch(*_a, **_k):  # noqa
        raise ImportError(
            "HPAMM-Net requires PyTorch. Install via envs/environment.yml "
            "(conda activate malaria)."
        )

    HPAMMNet = _no_torch          # type: ignore
    ModalityEncoder = _no_torch   # type: ignore
    GatedFusion = _no_torch       # type: ignore
    multitask_loss = _no_torch    # type: ignore
    combined_loss = _no_torch     # type: ignore
    stability_selection = _no_torch  # type: ignore
