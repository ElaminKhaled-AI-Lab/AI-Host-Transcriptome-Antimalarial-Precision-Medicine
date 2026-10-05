"""Baseline + classical ML models (blueprint Sections 40-41).

Factory returning scikit-learn / XGBoost estimators by name, so the benchmark
runner can iterate over ``config/models.yaml`` uniformly. Nested CV and
metrics live in ``scripts/07_integration_benchmarks``.
"""
from __future__ import annotations


def make_classifier(name: str, seed: int = 11, **kw):
    """Return an unfitted classifier by name."""
    name = name.lower()
    if name in ("logistic_regression", "logreg"):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, random_state=seed, **kw)
    if name in ("elastic_net", "elasticnet"):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  l1_ratio=kw.pop("l1_ratio", 0.5),
                                  max_iter=5000, random_state=seed, **kw)
    if name in ("random_forest", "rf"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=kw.pop("n_estimators", 500),
                                      random_state=seed, n_jobs=-1, **kw)
    if name in ("xgboost", "xgb"):
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=-1,
                             eval_metric="logloss", **kw)
    if name in ("svm", "svc"):
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=seed, **kw)
    raise ValueError(f"Unknown classifier '{name}'")


def make_regressor(name: str, seed: int = 11, **kw):
    """Return an unfitted regressor by name (PK exposure, clearance)."""
    name = name.lower()
    if name in ("elastic_net", "elasticnet"):
        from sklearn.linear_model import ElasticNet
        return ElasticNet(random_state=seed, **kw)
    if name in ("random_forest", "rf"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=kw.pop("n_estimators", 500),
                                     random_state=seed, n_jobs=-1, **kw)
    if name in ("xgboost", "xgb"):
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=-1, **kw)
    if name in ("gaussian_process", "gp"):
        from sklearn.gaussian_process import GaussianProcessRegressor
        return GaussianProcessRegressor(normalize_y=True, random_state=seed, **kw)
    raise ValueError(f"Unknown regressor '{name}'")
