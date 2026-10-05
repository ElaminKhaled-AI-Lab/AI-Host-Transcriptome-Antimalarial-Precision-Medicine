"""Evaluation metrics (blueprint Section 72) + bootstrap CIs (Section 73).

Classification, calibration, regression, and ranking metrics with the same
names used in ``config/models.yaml``. All functions are pure and depend only
on numpy / scikit-learn so they can be unit-tested without the full pipeline.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- classification
def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """AUROC, AUPRC, sensitivity, specificity, balanced accuracy, F1, MCC."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        matthews_corrcoef, balanced_accuracy_score, confusion_matrix,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    out = {
        "auroc": _safe(roc_auc_score, y_true, y_prob),
        "auprc": _safe(average_precision_score, y_true, y_prob),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return out


def calibration_metrics(y_true, y_prob, n_bins: int = 10) -> dict[str, float]:
    """Brier score, calibration slope/intercept, Expected Calibration Error."""
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y_true, y_prob)

    # Calibration slope/intercept via logistic regression on the logit.
    logit = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression().fit(logit, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope, intercept = float("nan"), float("nan")

    # Expected Calibration Error.
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return {"brier": brier, "calibration_slope": slope,
            "calibration_intercept": intercept, "expected_calibration_error": ece}


# --------------------------------------------------------------- regression
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R^2 for PK exposure and clearance."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse, "r2": float(r2_score(y_true, y_pred))}


# --------------------------------------------------------------- bootstrap CI
def bootstrap_ci(y_true, y_score, metric_fn, n_boot: int = 2000,
                 seed: int = 11, alpha: float = 0.05) -> dict[str, float]:
    """Bootstrap confidence interval for any (y_true, y_score)->float metric."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "ci_low": float(np.quantile(vals, alpha / 2)),
        "ci_high": float(np.quantile(vals, 1 - alpha / 2)),
    }


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")
