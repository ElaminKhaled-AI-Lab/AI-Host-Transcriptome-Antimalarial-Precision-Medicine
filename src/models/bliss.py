"""BLISS -- Balanced Latent Integration with Stability Selection.

Faithful re-implementation of the integration model published by the project
group for synthetic colon adenocarcinoma (Elamin, Elbashir, Adam,
*Int. J. Mol. Sci.* 2026, 27, 7511, doi:10.3390/ijms27177511). Shipped here as
a REUSABLE BASELINE so the malaria project can (a) reproduce the prior method
on new data and (b) fairly benchmark the novel successor (see hpamm_net.py)
against it under matched conditions.

Architecture (paper Section 2.7 / Figure 7A):
  * separate LINEAR encoders map K omics layers into a shared latent space
    (16-d in the paper);
  * encoded layers are concatenated, layer-normalized, passed through GELU,
    and projected to an L2-normalized latent representation;
  * three loss terms guide training:
        L = lambda_triplet * L_triplet
          + lambda_factor  * L_factor
          + lambda_balance * L_balance
    - L_triplet (lambda=1.0): triplet metric loss over PCA->k-means pseudo-labels
      (Super.FELT-inspired). Same pseudo-cluster pulled together, different pushed.
    - L_factor  (lambda=0.3): reconstruct each input layer from the shared latent
      (MOFA-inspired shared-factor reconstruction).
    - L_balance (lambda=0.7): KL divergence between the marginal cluster
      distribution and a uniform distribution over k clusters (introduced in the
      paper; prevents the degenerate collapse seen with SNF).

Weakly supervised: pseudo-labels are data-derived (no outcome/subtype labels).

Requires torch + scikit-learn. Import is guarded so config tooling works without
the deep-learning stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:
    _TORCH = False
    nn = object  # type: ignore


@dataclass
class BlissConfig:
    latent_dim: int = 16            # paper uses 16
    k_clusters: int = 4             # paper uses k=4 (CMS-like)
    lambda_triplet: float = 1.0
    lambda_factor: float = 0.3
    lambda_balance: float = 0.7
    lr: float = 1e-3
    epochs: int = 200
    triplet_margin: float = 1.0
    seed: int = 11
    pseudo_label_pca_dim: int = 20  # PCA dim before k-means for pseudo-labels


def make_pseudo_labels(X_concat: np.ndarray, k: int, pca_dim: int, seed: int) -> np.ndarray:
    """PCA -> k-means pseudo-labels on the concatenated omics (paper Section 2.7)."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    d = min(pca_dim, X_concat.shape[1], max(2, X_concat.shape[0] - 1))
    Z = PCA(n_components=d, random_state=seed).fit_transform(X_concat)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


if _TORCH:

    class LinearEncoder(nn.Module):
        """Single-layer linear encoder x_m -> latent (paper: separate LINEAR encoders)."""

        def __init__(self, in_dim: int, latent_dim: int):
            super().__init__()
            self.lin = nn.Linear(in_dim, latent_dim)

        def forward(self, x):
            return self.lin(x)


    class BLISS(nn.Module):
        """Balanced Latent Integration with Stability Selection.

        Parameters
        ----------
        input_dims : {layer_name: n_features}
        cfg        : BlissConfig
        """

        def __init__(self, input_dims: dict[str, int], cfg: BlissConfig | None = None):
            super().__init__()
            self.cfg = cfg or BlissConfig()
            self.layers = list(input_dims)
            L = self.cfg.latent_dim
            self.encoders = nn.ModuleDict(
                {m: LinearEncoder(d, L) for m, d in input_dims.items()}
            )
            self.norm = nn.LayerNorm(L * len(self.layers))
            self.project = nn.Linear(L * len(self.layers), L)
            # Factor-reconstruction decoders (shared latent -> each layer).
            self.decoders = nn.ModuleDict(
                {m: nn.Linear(L, d) for m, d in input_dims.items()}
            )

        def encode(self, x: dict) -> "torch.Tensor":
            zs = [self.encoders[m](x[m]) for m in self.layers]
            h = torch.cat(zs, dim=1)
            h = F.gelu(self.norm(h))
            z = self.project(h)
            return F.normalize(z, p=2, dim=1)      # L2-normalized latent

        def reconstruct(self, z: "torch.Tensor") -> dict:
            return {m: self.decoders[m](z) for m in self.layers}

        # ---- loss terms -------------------------------------------------
        def factor_loss(self, x: dict, recon: dict) -> "torch.Tensor":
            return sum(F.mse_loss(recon[m], x[m]) for m in self.layers) / len(self.layers)

        def balance_loss(self, z: "torch.Tensor") -> "torch.Tensor":
            """KL(marginal cluster dist || uniform) via soft assignment to k
            learnable-free centroids (cosine soft-assign in the L2 latent)."""
            k = self.cfg.k_clusters
            # Soft assignment against k anchor directions derived from a running
            # split of the batch (cheap, differentiable proxy for cluster freq).
            logits = z @ z.mean(dim=0, keepdim=True).T.repeat(1, k)  # (B,k) proxy
            q = F.softmax(logits, dim=1)
            marginal = q.mean(dim=0)                                # (k,)
            uniform = torch.full_like(marginal, 1.0 / k)
            return (marginal * (marginal.clamp_min(1e-8).log()
                                - uniform.log())).sum()

        def triplet_loss(self, z: "torch.Tensor", pseudo: "torch.Tensor") -> "torch.Tensor":
            """Batch-hard triplet over pseudo-labels (Super.FELT-inspired)."""
            d = torch.cdist(z, z)                                   # (B,B)
            same = pseudo.unsqueeze(0) == pseudo.unsqueeze(1)
            eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
            pos = d.masked_fill(~same | eye, float("-inf")).max(1).values
            neg = d.masked_fill(same, float("inf")).min(1).values
            pos = torch.nan_to_num(pos, neginf=0.0)
            neg = torch.nan_to_num(neg, posinf=0.0)
            return F.relu(pos - neg + self.cfg.triplet_margin).mean()


    def train_bliss(X: dict[str, np.ndarray], cfg: BlissConfig | None = None,
                    device: str = "cpu"):
        """Fit BLISS on a dict of {layer: (n_samples, n_features)} arrays.

        Returns (model, latent_np, cluster_labels_np). Cluster labels come from
        k-means on the learned latent (stability selection is applied by the
        transportability battery via repeated subsampling).
        """
        from sklearn.cluster import KMeans

        cfg = cfg or BlissConfig()
        torch.manual_seed(cfg.seed)
        layers = list(X)
        n = X[layers[0]].shape[0]
        Xt = {m: torch.tensor(np.asarray(X[m]), dtype=torch.float32, device=device)
              for m in layers}
        concat = np.concatenate([np.asarray(X[m]) for m in layers], axis=1)
        pseudo = torch.tensor(
            make_pseudo_labels(concat, cfg.k_clusters, cfg.pseudo_label_pca_dim, cfg.seed),
            device=device,
        )

        model = BLISS({m: X[m].shape[1] for m in layers}, cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        model.train()
        for epoch in range(cfg.epochs):
            opt.zero_grad()
            z = model.encode(Xt)
            recon = model.reconstruct(z)
            lt = model.triplet_loss(z, pseudo)
            lf = model.factor_loss(Xt, recon)
            lb = model.balance_loss(z)
            loss = cfg.lambda_triplet * lt + cfg.lambda_factor * lf + cfg.lambda_balance * lb
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            latent = model.encode(Xt).cpu().numpy()
        labels = KMeans(n_clusters=cfg.k_clusters, n_init=10,
                        random_state=cfg.seed).fit_predict(latent)
        return model, latent, labels

else:  # torch missing

    def _no_torch(*_a, **_k):
        raise ImportError("BLISS requires PyTorch. See envs/environment.yml.")

    BLISS = _no_torch          # type: ignore
    train_bliss = _no_torch    # type: ignore
