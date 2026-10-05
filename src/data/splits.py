"""Leakage-safe data splitting (blueprint Sections 70, 78).

Grouped + stratified nested cross-validation so that technical replicates of
one patient, and samples from one study/site, never straddle train and test.
"""
from __future__ import annotations

import numpy as np


def grouped_outer_folds(groups, n_splits: int = 5, seed: int = 11):
    """Yield (train_idx, test_idx) with all rows of a group kept together.

    ``groups`` is an array-like of group labels (e.g. patient_id or study_id).
    """
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    # GroupKFold is deterministic; shuffle groups first for seed control.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(np.unique(groups)))
    remap = {g: i for i, g in enumerate(np.unique(groups)[order])}
    shuffled = np.array([remap[g] for g in groups])
    gkf = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(groups), 1))
    for tr, te in gkf.split(X_dummy, groups=shuffled):
        yield tr, te


def country_held_out(country, held_out_country):
    """Return (train_idx, test_idx) for country-held-out validation (Section 57).

    Train on every country except ``held_out_country``; test on it. This is a
    stronger transportability test than random splitting.
    """
    country = np.asarray(country)
    test = np.where(country == held_out_country)[0]
    train = np.where(country != held_out_country)[0]
    return train, test


def study_held_out(study, held_out_study):
    """Return (train_idx, test_idx) for study-held-out validation."""
    study = np.asarray(study)
    return (np.where(study != held_out_study)[0],
            np.where(study == held_out_study)[0])
