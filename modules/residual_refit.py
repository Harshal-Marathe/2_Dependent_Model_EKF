"""
Two-stage "residual as a variable" re-estimation.

Stage 1: the model you already fitted.
Stage 2: take that model's residuals (y_t − one-step-ahead prediction), add them
         to the data as one more own non-media variable and RE-ESTIMATE THE
         WHOLE MODEL from scratch — every coefficient / hyper-parameter is
         refitted (nothing is held at its stage-1 value).

lag = 0 uses the same-period residual e_t;  lag = 1 uses e_(t-1).

Caveat that is shown in the UI as well: e_t is computed from y_t, so lag 0
partly feeds the KPI to itself — expect an almost perfect in-sample (and
out-of-sample) fit that is not real predictive power, and the "residual"
variable will absorb part of the KPI that would otherwise go to baseline /
media. Compare the contribution and ROI tables to see how much.
"""
import numpy as np

from modules.pipeline import run_full_ekf_pipeline
from modules.refit import add_variable_to_config

RESID_COL = "Model_Residual"


def plain_result(res):
    """The model WITHOUT Cochrane-Orcutt. If `res` was fitted with the AR(1)
    loop, that is its stored ρ = 0 baseline fit; otherwise `res` itself."""
    return res.get("co_baseline") or res


def residual_series(res, lag=0):
    e = np.asarray(res["residuals"], dtype=float).copy()
    e = np.nan_to_num(e, nan=0.0)
    if lag <= 0:
        return e
    return np.r_[np.zeros(lag), e[:-lag]]


def build_residual_dataset(df, config, res, lag=0, col=RESID_COL):
    """Return (df2, cfg2): df plus the residual column, and a config with that
    column added as an own non-media variable (sign unconstrained, instant)."""
    e = residual_series(res, lag)
    if len(e) != len(df):
        raise ValueError(f"Residual length ({len(e)}) ≠ data length ({len(df)}).")
    df2 = df.copy()
    df2[col] = e
    if col in (config.get("non_media") or []):
        cfg2 = dict(config)
    else:
        cfg2 = add_variable_to_config(config, col, "non_media", None, "instant")
    # Stage 2 is the residual method ALONE: never run the AR(1) loop inside it.
    cfg2["co_methods"] = ["residual"]
    cfg2["co_method"] = "residual"
    return df2, cfg2


def run_residual_refit(df, config, res, lag, max_iter, method, ng_cfg=None, col=RESID_COL):
    """Fit stage 2 from scratch on (df + residual). Returns (df2, cfg2, res2)."""
    res = plain_result(res)            # residuals of the model without CO
    df2, cfg2 = build_residual_dataset(df, config, res, lag, col)
    res2 = run_full_ekf_pipeline(df2, cfg2, max_iter, method, ng_cfg=ng_cfg)
    return df2, cfg2, res2
