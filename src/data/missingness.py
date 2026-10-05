"""Missing-data simulation and modality masks (blueprint Sections 37, 52, 76).

Three mechanisms -- MCAR, MAR, MNAR -- plus structured modality dropout used
by both the semi-synthetic generator and the missing-modality benchmark.
"""
from __future__ import annotations

import numpy as np

MECHANISMS = ("MCAR", "MAR", "MNAR")


def mcar_mask(shape, rate: float, seed: int = 11) -> np.ndarray:
    """Missing Completely At Random: each entry missing with prob ``rate``."""
    rng = np.random.default_rng(seed)
    return rng.random(shape) < rate


def mar_mask(values, driver, rate: float, seed: int = 11) -> np.ndarray:
    """Missing At Random: probability of missing depends on OBSERVED ``driver``.

    ``driver`` (e.g. parasitemia) is scaled to [0,1] and modulates the base
    ``rate``. Returns a boolean mask the shape of ``values``.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(driver, dtype=float)
    d = (d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d) + 1e-9)
    p = np.clip(rate * (0.5 + d), 0, 1)
    p = np.broadcast_to(p.reshape(-1, 1), np.shape(values))
    return rng.random(np.shape(values)) < p


def mnar_mask(values, rate: float, seed: int = 11) -> np.ndarray:
    """Missing Not At Random: probability of missing depends on the value itself.

    Lower values are more likely to be missing (e.g. below detection limit).
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    r = v.argsort(axis=0).argsort(axis=0) / max(v.shape[0] - 1, 1)  # rank in [0,1]
    p = np.clip(rate * (1.5 - r), 0, 1)
    return rng.random(v.shape) < p


def modality_dropout(n_patients: int, modalities: list[str],
                     drop_rate: float, seed: int = 11) -> np.ndarray:
    """Structured modality mask (Section 53). 1 = available, 0 = missing.

    Returns an (n_patients, n_modalities) int array. A patient never loses all
    modalities at once (at least one is kept).
    """
    rng = np.random.default_rng(seed)
    m = (rng.random((n_patients, len(modalities))) >= drop_rate).astype(int)
    empty = m.sum(axis=1) == 0
    if empty.any():
        keep = rng.integers(0, len(modalities), empty.sum())
        m[np.where(empty)[0], keep] = 1
    return m
