"""
Transformation functions: Hill saturation curve, power transformation,
and adstock (carry-over) models including Weibull with per-lag weights.
"""

import numpy as np
import pandas as pd


# ── Hill transformation ─────────────────────────────────────────────────────

def hill_transform_vec(x: np.ndarray, n: float, S: float) -> np.ndarray:
    x = np.maximum(x, 0.0)
    xn = x ** n
    return xn / (xn + S ** n + 1e-30)


def hill_transform(x: float, n: float, S: float) -> float:
    if x <= 0:
        return 0.0
    return (x ** n) / (x ** n + S ** n + 1e-30)


# ── Power transformation ────────────────────────────────────────────────────

def power_transform_vec(x: np.ndarray, n: float) -> np.ndarray:
    """Element-wise power transform: x^n, clamped to non-negative inputs."""
    return np.maximum(x, 0.0) ** n


def power_transform(x: float, n: float) -> float:
    if x <= 0:
        return 0.0
    return x ** n


# ── Adstock: Nerlove-Arrow (geometric / instant) ────────────────────────────

def adstock_nerlove_arrow(x: pd.Series, lam: float) -> np.ndarray:
    arr = x.values.astype(float)
    out = np.empty_like(arr)
    decay_for_seed = min(lam, 0.9)
    out[0] = arr[0] / (1.0 - decay_for_seed + 1e-8)
    for t in range(1, len(arr)):
        out[t] = arr[t] + lam * out[t - 1]
    return out


# ── Adstock: Weibull with per-lag weights (delayed / distributed) ───────────

def weibull_lag_weights(shape_k: float, scale_lam: float, n_lags: int) -> np.ndarray:
    """
    Compute normalised Weibull PDF weights w_1, w_2, …, w_j for j = n_lags
    past periods (t-1, t-2, …, t-j). The current period (t) is NOT included.

    Formula, for lag index l = 1, 2, …, j:
        w_l = (k / λ) · (l / λ)^(k−1) · exp( −(l / λ)^k )

    Weights are normalised so they sum to exactly 1:
        w_1 + w_2 + … + w_j = 1

    n_lags = j returns exactly j weights. n_lags = 0 returns an empty array
    (no lagged carry-over at all).
    """
    j = int(n_lags)
    if j <= 0:
        return np.zeros(0)
    lags = np.arange(1, j + 1, dtype=float)            # l = 1, 2, …, j
    ratio = lags / scale_lam
    w = (shape_k / scale_lam) * (ratio ** (shape_k - 1)) * np.exp(-(ratio ** shape_k))
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total < 1e-12:
        w = np.ones(j) / j
    else:
        w /= total
    return w


def adstock_weibull_lagged(x: pd.Series,
                            shape_k: float,
                            scale_lam: float,
                            n_lags: int) -> np.ndarray:
    """
    Weighted sum of PAST lagged raw spend using Weibull-derived weights
    (the current period x[t] is excluded — this is a pure lag/delay term):

        adstocked[t] = w_1·x[t-1] + w_2·x[t-2] + … + w_j·x[t-j]

    n_lags = 0 → adstocked[t] = 0 for all t (no lagged carry-over).
    """
    arr = x.values.astype(float)
    T = len(arr)
    weights = weibull_lag_weights(shape_k, scale_lam, n_lags)
    out = np.zeros(T)
    for idx, w in enumerate(weights):
        l = idx + 1                                     # lag = 1, 2, …, j
        if w < 1e-15:
            continue
        out[l:] += w * arr[:-l]
    return out


# ── Dispatcher ───────────────────────────────────────────────────────────────

def _adstock(series: pd.Series, adstock_type: str, params_dict: dict,
             chan_idx: int) -> np.ndarray:
    if adstock_type == "weibull":
        n_lags = int(params_dict.get("adstock_n_lags", 8))
        return adstock_weibull_lagged(
            series,
            params_dict["adstock_shape"][chan_idx],
            params_dict["adstock_scale"][chan_idx],
            n_lags,
        )
    # instant / Nerlove-Arrow
    return adstock_nerlove_arrow(series, params_dict["adstock_lambda"][chan_idx])


def apply_transformation(x: np.ndarray, transform_type: str,
                          n: float, S: float = 1.0) -> np.ndarray:
    """
    Apply either 'power' or 'hill' transformation to array x.
    For power: returns x^n
    For hill:  returns x^n / (x^n + S^n)
    """
    if transform_type == "hill":
        return hill_transform_vec(x, n, S)
    return power_transform_vec(x, n)
