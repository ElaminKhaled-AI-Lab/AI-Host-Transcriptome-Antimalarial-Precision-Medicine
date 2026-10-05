"""Complete dissolution / in-vitro release analytics for nanoparticle suspensions.

Everything a formulation scientist reports from a release profile, computed for
every formulation rather than only the winning kinetic model.

Abbreviations (defined on first use, short form thereafter):
  DE dissolution efficiency; MDT mean dissolution time; K-P Korsmeyer-Peppas;
  AIC Akaike information criterion; RMSE root-mean-square error; f1 difference
  factor; f2 similarity factor; Q_t cumulative percent released at time t.

Model set (all fitted, all reported -- constants, R^2, adjusted R^2, AIC, RMSE):
  * Zero-order      Q = Q0 + k0 t                (constant rate, matrix-independent)
  * First-order     Q = 100 (1 - e^{-k1 t})      (concentration-dependent)
  * Higuchi         Q = kH sqrt(t)               (diffusion from a matrix)
  * Korsmeyer-Peppas Q/Qinf = kKP t^n            (mechanism via exponent n)
  * Hixson-Crowell  100^(1/3) - Qr^(1/3) = kHC t (erosion / surface-area change)

Model-independent metrics:
  * DE (Khan 1975): area under the release curve divided by the area of the
    rectangle describing 100% release over the same period, expressed as %.
  * MDT: the mean time for the drug to dissolve, Sum(t_mid dQ) / Sum(dQ).
  * t25 / t50 / t80: time to 25 / 50 / 80 percent release (linear interpolation).
  * Burst release: percent released at 1 h (formulation quality indicator).
  * f1 / f2 (FDA/EMA): difference and similarity factors versus a reference
    profile; f2 >= 50 (and f1 <= 15) indicates equivalent release.

K-P exponent interpretation for SPHERICAL particles: n <= 0.43 Fickian
diffusion; 0.43 < n < 0.85 anomalous (non-Fickian) transport; n approximately
0.85 case-II (erosion-controlled); n > 0.85 super case-II.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


# ------------------------------------------------------------------ goodness of fit
def _gof(y, yhat, k_params):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k_params - 1, 1)
    rmse = math.sqrt(ss_res / n)
    aic = n * math.log(ss_res / n + 1e-12) + 2 * k_params
    return {"r2": round(r2, 4), "adj_r2": round(adj, 4),
            "rmse": round(rmse, 3), "aic": round(aic, 2)}


# ------------------------------------------------------------------ kinetic models
def fit_all_models(t_h, Q) -> dict:
    """Fit all five kinetic models; return constants + full goodness of fit for each."""
    t = np.asarray(t_h, float); q = np.asarray(Q, float)
    keep = t > 0
    t, q = t[keep], q[keep]
    out: dict = {}

    # zero-order
    k0 = float(np.sum(t * q) / np.sum(t * t))
    out["zero_order"] = {"k0_pct_per_h": round(k0, 4), **_gof(q, k0 * t, 1)}

    # first-order
    try:
        p, _ = curve_fit(lambda tt, k: 100 * (1 - np.exp(-k * tt)), t, q,
                         p0=[0.1], maxfev=20000)
        yh = 100 * (1 - np.exp(-p[0] * t))
        out["first_order"] = {"k1_per_h": round(float(p[0]), 4), **_gof(q, yh, 1)}
    except Exception:
        out["first_order"] = {"k1_per_h": None, "r2": None, "adj_r2": None,
                              "rmse": None, "aic": 1e9}

    # Higuchi
    kh = float(np.sum(np.sqrt(t) * q) / np.sum(t))
    out["higuchi"] = {"kH_pct_per_sqrt_h": round(kh, 4), **_gof(q, kh * np.sqrt(t), 1)}

    # Korsmeyer-Peppas (fit on the <=60% release portion, per convention)
    qf = q / 100.0
    sel = qf <= 0.60
    if sel.sum() >= 3:
        try:
            p, _ = curve_fit(lambda tt, k, n: k * tt ** n, t[sel], qf[sel],
                             p0=[0.2, 0.5], maxfev=20000,
                             bounds=([1e-4, 0.1], [5.0, 1.5]))
            yh = p[0] * t[sel] ** p[1]
            out["korsmeyer_peppas"] = {"kKP_per_h^n": round(float(p[0]), 4),
                                       "n": round(float(p[1]), 4),
                                       "n_points_used": int(sel.sum()),
                                       **_gof(qf[sel], yh, 2)}
        except Exception:
            out["korsmeyer_peppas"] = {"kKP_per_h^n": None, "n": None, "aic": 1e9,
                                       "r2": None, "adj_r2": None, "rmse": None}
    else:
        out["korsmeyer_peppas"] = {"kKP_per_h^n": None, "n": None, "aic": 1e9,
                                   "r2": None, "adj_r2": None, "rmse": None,
                                   "n_points_used": int(sel.sum())}

    # Hixson-Crowell
    rem = np.clip(100 - q, 1e-6, None)
    lhs = 100 ** (1 / 3) - rem ** (1 / 3)
    khc = float(np.sum(t * lhs) / np.sum(t * t))
    out["hixson_crowell"] = {"kHC_per_h": round(khc, 4), **_gof(lhs, khc * t, 1)}

    # --- fair model comparison -------------------------------------------------
    # AIC is only comparable when every model is fitted to the SAME response on
    # the SAME points. The conventional K-P fit above uses a fractional scale and
    # only the <=60% portion, so it cannot be ranked against the others. Here all
    # five are refitted to the full percent-scale profile.
    cmp_fits = {}
    cmp_fits["zero_order"] = _gof(q, k0 * t, 1)
    if out["first_order"].get("k1_per_h") is not None:
        cmp_fits["first_order"] = _gof(q, 100 * (1 - np.exp(-out["first_order"]["k1_per_h"] * t)), 1)
    cmp_fits["higuchi"] = _gof(q, kh * np.sqrt(t), 1)
    try:  # K-P as a percent-scale power law over the full profile
        p, _ = curve_fit(lambda tt, A, n: A * tt ** n, t, q, p0=[20.0, 0.5],
                         maxfev=20000, bounds=([1e-3, 0.05], [500.0, 2.0]))
        cmp_fits["korsmeyer_peppas"] = _gof(q, p[0] * t ** p[1], 2)
        out["korsmeyer_peppas_fullrange"] = {"A_pct": round(float(p[0]), 3),
                                             "n_fullrange": round(float(p[1]), 4)}
    except Exception:
        pass
    # Hixson-Crowell back-transformed to percent so it is on the same scale
    q_hc = np.clip(100 - np.clip(100 ** (1 / 3) - khc * t, 0, None) ** 3, 0, 100)
    cmp_fits["hixson_crowell"] = _gof(q, q_hc, 1)

    out["model_comparison_percent_scale"] = cmp_fits
    if cmp_fits:
        out["best_model_by_aic"] = min(cmp_fits, key=lambda k: cmp_fits[k]["aic"])
        out["best_model_by_r2"] = max(cmp_fits, key=lambda k: cmp_fits[k]["r2"])
    else:
        out["best_model_by_aic"] = out["best_model_by_r2"] = None
    return out


def interpret_n(n) -> str:
    """Transport mechanism from the K-P exponent (spherical geometry)."""
    if n is None:
        return "undetermined"
    if n <= 0.43:
        return "Fickian diffusion"
    if n < 0.85:
        return "anomalous (non-Fickian) transport"
    if abs(n - 0.85) < 0.02:
        return "case-II (erosion-controlled)"
    return "super case-II transport"


# ------------------------------------------------------- model-independent metrics
def dissolution_efficiency(t_h, Q) -> float:
    """DE (%): area under the release curve / area of the 100% rectangle (Khan 1975)."""
    t = np.asarray(t_h, float); q = np.asarray(Q, float)
    auc = float(np.trapezoid(q, t)) if hasattr(np, "trapezoid") else float(np.trapz(q, t))
    return float(round(100.0 * auc / (100.0 * (t[-1] - t[0]) + 1e-12), 2))


def mean_dissolution_time(t_h, Q) -> float:
    """MDT (h): Sum(t_mid * dQ) / Sum(dQ)."""
    t = np.asarray(t_h, float); q = np.asarray(Q, float)
    dq = np.diff(q)
    tmid = (t[1:] + t[:-1]) / 2.0
    denom = float(np.sum(dq))
    return float(round(float(np.sum(tmid * dq)) / denom, 3)) if denom > 0 else float("nan")


def time_to_percent(t_h, Q, pct) -> float:
    """Time (h) to reach a given cumulative percent release; NaN if never reached."""
    t = np.asarray(t_h, float); q = np.asarray(Q, float)
    if q.max() < pct:
        return float("nan")
    idx = int(np.argmax(q >= pct))
    if idx == 0:
        return float(t[0])
    q0, q1, t0, t1 = q[idx - 1], q[idx], t[idx - 1], t[idx]
    if q1 == q0:
        return float(t1)
    return float(round(t0 + (pct - q0) * (t1 - t0) / (q1 - q0), 3))


def f1_f2(Q_ref, Q_test) -> dict:
    """FDA/EMA difference (f1) and similarity (f2) factors between two profiles.

    Both profiles must be sampled at the SAME time points. f2 >= 50 with
    f1 <= 15 indicates equivalent release.
    """
    R = np.asarray(Q_ref, float); T = np.asarray(Q_test, float)
    n = len(R)
    f1 = 100.0 * float(np.sum(np.abs(R - T))) / (float(np.sum(R)) + 1e-12)
    f2 = 50.0 * math.log10(100.0 / math.sqrt(1.0 + float(np.sum((R - T) ** 2)) / n))
    return {"f1_difference": round(f1, 2), "f2_similarity": round(f2, 2),
            "equivalent": bool(f2 >= 50 and f1 <= 15)}


# ------------------------------------------------------------------ full report
def full_dissolution_report(t_h, Q, Q_reference=None) -> dict:
    """All kinetic constants + all model-independent dissolution parameters."""
    models = fit_all_models(t_h, Q)
    kp = models.get("korsmeyer_peppas", {})
    rep = {
        "DE_pct": dissolution_efficiency(t_h, Q),
        "MDT_h": mean_dissolution_time(t_h, Q),
        "t25_h": time_to_percent(t_h, Q, 25),
        "t50_h": time_to_percent(t_h, Q, 50),
        "t80_h": time_to_percent(t_h, Q, 80),
        "burst_1h_pct": round(float(np.interp(1.0, t_h, Q)), 2),
        "Q_24h_pct": round(float(np.interp(24.0, t_h, Q)), 2),
        "Q_final_pct": round(float(np.asarray(Q, float)[-1]), 2),
        "peppas_n": kp.get("n"),
        "release_mechanism": interpret_n(kp.get("n")),
        "best_model_by_aic": models.get("best_model_by_aic"),
        "best_model_by_r2": models.get("best_model_by_r2"),
        "models": models,
    }
    if Q_reference is not None:
        rep["similarity_vs_reference"] = f1_f2(Q_reference, Q)
    return rep
