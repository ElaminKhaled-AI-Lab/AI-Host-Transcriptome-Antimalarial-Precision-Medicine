"""PROFILE-Net and the integration-strategy benchmark it is measured against.

PROFILE-Net -- Patient-Response Omics-Formulation Interaction Learning Engine.

Problem. Precision formulation asks a question that additive models cannot
answer: not "which formulation is best on average" but "which formulation is
best FOR THIS CHILD". Answering it requires learning the INTERACTION between a
patient's molecular state and a formulation's physical behaviour, because the
ranking of formulations must be allowed to change from patient to patient.

Integration taxonomy. Multi-block learning is conventionally grouped as:
  * EARLY integration    -- concatenate raw blocks, then one model.
  * LATE integration     -- one model per block, then combine predictions.
  * INTERMEDIATE (joint-latent) integration -- encode each block to a latent
    space, then combine the latents.
PROFILE-Net is an INTERMEDIATE integration with an explicit BILINEAR
CROSS-ATTENTION interaction: the patient latent both attends over the
formulation latent and enters a low-rank bilinear product with it, so the model
represents patient-by-formulation effects directly rather than as a sum of main
effects. All four strategies are implemented here and benchmarked head to head;
the winner is decided empirically, not assumed.

Abbreviations: MLP multilayer perceptron; ReLU rectified linear unit; MSE
mean-squared error; BCE binary cross-entropy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pick_device(prefer="auto", quiet=False):
    """Choose a device that genuinely works, not merely one that reports itself.

    ``torch.cuda.is_available()`` only says a driver and device are present. It
    returns True even when the installed PyTorch carries no compiled kernels for
    that device's compute capability -- for example a Quadro P4000 (sm_61)
    against a build supporting sm_75 and newer -- and every kernel launch then
    fails with ``cudaErrorNoKernelImageForDevice``. So the capability is checked
    against the build's architecture list AND a real tensor operation is run
    before CUDA is accepted; anything unexpected falls back to the processor.

    Override with the MALARIA_DEVICE environment variable or ``prefer``.
    """
    import os
    prefer = os.environ.get("MALARIA_DEVICE", prefer).lower()
    if prefer.startswith("cpu"):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        arch_list = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
        name = torch.cuda.get_device_name(0)
        if arch_list and sm not in arch_list:
            if not quiet:
                print(f"[device] {name} has compute capability {sm}, but this PyTorch "
                      f"build supports {', '.join(arch_list)}. Falling back to CPU.")
            return torch.device("cpu")
        # smoke test: an actual kernel launch, which is what really fails
        t = torch.zeros(8, device="cuda")
        _ = (t + 1).sum().item()
        torch.cuda.synchronize()
        if not quiet:
            print(f"[device] using GPU: {name} ({sm})")
        return torch.device("cuda")
    except Exception as e:  # noqa: BLE001
        if not quiet:
            print(f"[device] GPU unusable ({type(e).__name__}: {e}); falling back to CPU.")
        return torch.device("cpu")


def _mlp(din, dh, dout, depth=2, p=0.1):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, dh), nn.ReLU(), nn.Dropout(p)]
        d = dh
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
#  1. EARLY integration -- concatenate raw blocks                             #
# --------------------------------------------------------------------------- #
class EarlyIntegration(nn.Module):
    name = "early_concat"

    def __init__(self, d_pat, d_form, d_hidden=128, n_reg=1):
        super().__init__()
        self.body = _mlp(d_pat + d_form, d_hidden, d_hidden, depth=2)
        self.reg = nn.Linear(d_hidden, n_reg)
        self.clf = nn.Linear(d_hidden, 1)

    def forward(self, xp, xf):
        h = self.body(torch.cat([xp, xf], dim=1))
        return self.reg(h), self.clf(h)


# --------------------------------------------------------------------------- #
#  2. LATE integration -- separate per-block models, combined at the output   #
# --------------------------------------------------------------------------- #
class LateIntegration(nn.Module):
    name = "late_ensemble"

    def __init__(self, d_pat, d_form, d_hidden=128, n_reg=1):
        super().__init__()
        self.pat = _mlp(d_pat, d_hidden, d_hidden, depth=2)
        self.form = _mlp(d_form, d_hidden, d_hidden, depth=2)
        self.reg_p, self.reg_f = nn.Linear(d_hidden, n_reg), nn.Linear(d_hidden, n_reg)
        self.clf_p, self.clf_f = nn.Linear(d_hidden, 1), nn.Linear(d_hidden, 1)
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, xp, xf):
        hp, hf = self.pat(xp), self.form(xf)
        w = torch.sigmoid(self.w)
        reg = w * self.reg_p(hp) + (1 - w) * self.reg_f(hf)
        clf = w * self.clf_p(hp) + (1 - w) * self.clf_f(hf)
        return reg, clf


# --------------------------------------------------------------------------- #
#  3. INTERMEDIATE integration -- joint latent, no interaction term           #
# --------------------------------------------------------------------------- #
class IntermediateLatent(nn.Module):
    name = "intermediate_latent"

    def __init__(self, d_pat, d_form, d_hidden=128, d_lat=32, n_reg=1):
        super().__init__()
        self.enc_p = _mlp(d_pat, d_hidden, d_lat, depth=2)
        self.enc_f = _mlp(d_form, d_hidden, d_lat, depth=2)
        self.body = _mlp(2 * d_lat, d_hidden, d_hidden, depth=1)
        self.reg = nn.Linear(d_hidden, n_reg)
        self.clf = nn.Linear(d_hidden, 1)

    def forward(self, xp, xf):
        h = self.body(torch.cat([self.enc_p(xp), self.enc_f(xf)], dim=1))
        return self.reg(h), self.clf(h)


# --------------------------------------------------------------------------- #
#  4. PROFILE-Net -- intermediate + bilinear cross-attention interaction      #
# --------------------------------------------------------------------------- #
class ProfileNet(nn.Module):
    """Patient-Response Omics-Formulation Interaction Learning Engine.

    The patient latent is split into attention heads that query the formulation
    latent (cross-attention), and a low-rank bilinear product supplies an
    explicit multiplicative patient-by-formulation term. Main effects are kept
    alongside, so the model can fall back to additive behaviour when no genuine
    interaction exists -- which makes the comparison against the additive
    baselines a fair test rather than a rigged one.
    """
    name = "PROFILE-Net"

    def __init__(self, d_pat, d_form, d_hidden=128, d_lat=32, rank=16, heads=4, n_reg=1):
        super().__init__()
        self.d_lat, self.heads = d_lat, heads
        self.enc_p = _mlp(d_pat, d_hidden, d_lat, depth=2)
        self.enc_f = _mlp(d_form, d_hidden, d_lat, depth=2)
        # low-rank bilinear interaction: (Up hp) * (Uf hf) -> rank-dim vector
        self.Up = nn.Linear(d_lat, rank, bias=False)
        self.Uf = nn.Linear(d_lat, rank, bias=False)
        # cross-attention: patient queries formulation
        self.attn = nn.MultiheadAttention(d_lat, heads, batch_first=True)
        self.body = _mlp(2 * d_lat + rank + d_lat, d_hidden, d_hidden, depth=2)
        self.reg = nn.Linear(d_hidden, n_reg)
        self.clf = nn.Linear(d_hidden, 1)

    def forward(self, xp, xf):
        hp, hf = self.enc_p(xp), self.enc_f(xf)
        inter = self.Up(hp) * self.Uf(hf)                    # multiplicative term
        q = hp.unsqueeze(1)                                  # (B,1,d)
        kv = hf.unsqueeze(1)
        att, _ = self.attn(q, kv, kv)                        # patient attends to formulation
        att = att.squeeze(1)
        h = self.body(torch.cat([hp, hf, inter, att], dim=1))
        return self.reg(h), self.clf(h)


STRATEGIES = {
    "early_concat": EarlyIntegration,
    "late_ensemble": LateIntegration,
    "intermediate_latent": IntermediateLatent,
    "PROFILE-Net": ProfileNet,
}


# --------------------------------------------------------------------------- #
#  Training / evaluation                                                      #
# --------------------------------------------------------------------------- #
def train_model(model, Xp, Xf, y_reg, y_clf=None, epochs=120, lr=1e-3, batch=512,
                seed=11, verbose=False, device=None):
    """Train on one or more regression targets, with an optional binary head."""
    torch.manual_seed(seed)
    dev = torch.device(device) if device is not None else pick_device(quiet=True)
    model = model.to(dev)
    Xp_t = torch.tensor(Xp, dtype=torch.float32, device=dev)
    Xf_t = torch.tensor(Xf, dtype=torch.float32, device=dev)
    yr = torch.tensor(np.atleast_2d(np.asarray(y_reg, float).T).T
                      if np.asarray(y_reg).ndim > 1 else np.asarray(y_reg, float).reshape(-1, 1),
                      dtype=torch.float32, device=dev)
    yc = (torch.tensor(np.asarray(y_clf, float).reshape(-1, 1), dtype=torch.float32,
                       device=dev) if y_clf is not None else None)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    n = len(Xp_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pr, pc = model(Xp_t[idx], Xf_t[idx])
            loss = mse(pr, yr[idx])
            if yc is not None:
                loss = loss + bce(pc, yc[idx])
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 40 == 0:
            print(f"  epoch {ep+1}/{epochs} loss {loss.item():.4f}")
    return model


@torch.no_grad()
def predict(model, Xp, Xf):
    """Return (regression predictions [n, n_reg], probability from the binary head)."""
    dev = next(model.parameters()).device
    model.eval()
    pr, pc = model(torch.tensor(Xp, dtype=torch.float32, device=dev),
                   torch.tensor(Xf, dtype=torch.float32, device=dev))
    return pr.cpu().numpy(), torch.sigmoid(pc).cpu().numpy().ravel()


def within_patient_rank_correlation(patient_ids, y_true, y_pred):
    """Mean Spearman correlation between true and predicted formulation ranking
    within each patient. NOTE: when the true ranking is similar across patients
    an additive model scores well here WITHOUT modelling any interaction, so
    interpret alongside `interaction_residual_rank_correlation`."""
    from scipy.stats import spearmanr
    rows = []
    ids = np.asarray(patient_ids)
    for pid in np.unique(ids):
        m = ids == pid
        if m.sum() < 3:
            continue
        yt, yp = np.asarray(y_true)[m], np.asarray(y_pred)[m]
        if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
            continue
        r = spearmanr(yt, yp).correlation
        if np.isfinite(r):
            rows.append(r)
    return float(np.mean(rows)) if rows else float("nan")


def _center_by_formulation(form_ids, values):
    """Subtract each formulation's mean across patients, leaving the purely
    patient-specific (interaction) component."""
    s = pd_series(form_ids, values)
    return (s - s.groupby(level=0).transform("mean")).to_numpy()


def pd_series(index, values):
    import pandas as pd
    return pd.Series(np.asarray(values, float), index=np.asarray(index))


def interaction_residual_rank_correlation(patient_ids, form_ids, y_true, y_pred):
    """The interaction-isolating metric.

    Each formulation's average effect across patients is removed from both the
    true and the predicted outcome, leaving only the patient-specific deviation.
    Ranking those residuals within a patient therefore measures ONLY the
    patient-by-formulation interaction: a purely additive model has a constant
    residual within a patient and scores near zero by construction.
    """
    from scipy.stats import spearmanr
    yt = _center_by_formulation(form_ids, y_true)
    yp = _center_by_formulation(form_ids, y_pred)
    ids = np.asarray(patient_ids)
    rows = []
    for pid in np.unique(ids):
        m = ids == pid
        if m.sum() < 3 or np.std(yt[m]) < 1e-12 or np.std(yp[m]) < 1e-12:
            continue
        r = spearmanr(yt[m], yp[m]).correlation
        if np.isfinite(r):
            rows.append(r)
    return float(np.mean(rows)) if rows else float("nan")


def personalization_gain(patient_ids, y_true, y_pred):
    """Outcome gained by choosing per-child versus one-size-fits-all.

    For every test child the model picks its top-predicted formulation; the TRUE
    outcome of that choice is compared with the true outcome of the single
    formulation that is best on average across the cohort. A positive gain means
    personalization genuinely helps.
    """
    import pandas as pd
    d = pd.DataFrame({"pid": np.asarray(patient_ids),
                      "yt": np.asarray(y_true, float),
                      "yp": np.asarray(y_pred, float)})
    # index formulations by their position within each patient block
    d["fidx"] = d.groupby("pid").cumcount()
    global_best = d.groupby("fidx")["yt"].mean().idxmax()
    chosen = d.loc[d.groupby("pid")["yp"].idxmax()]
    baseline = d[d["fidx"] == global_best]
    return float(chosen["yt"].mean() - baseline["yt"].mean())
