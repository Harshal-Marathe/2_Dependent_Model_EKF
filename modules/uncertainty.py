"""
Model-uncertainty diagnostics, layered on top of an already-fitted model.
None of this touches how a model is fit: bounds (including the
positivity/negativity beta-sign constraints built in modules/bounds.py
and re-applied post-smoothing in modules/pipeline.py::_postprocess_equation)
are passed through completely unchanged everywhere in this module.

Three pieces:
  1. add_confidence_bands  — normal-approximation 95% bands from the
     posterior state covariance (P_smooth). Kept for older saved results;
     MCMC results already carry exact posterior-draw credible bands.
  2. run_seed_stability    — re-runs MCMC with several different seeds and
     reports how much the ROI / ranking moves (a between-chain sanity check).
  3. compute_vif           — Variance Inflation Factor for the raw predictor
     matrix actually fed to a fitted equation's observation/state
     equations, to flag collinearity that makes individual channel
     coefficients hard to trust even when overall fit looks fine.
"""

import numpy as np
import pandas as pd

Z95 = 1.959963984540054  # two-sided 95% normal critical value


# ─────────────────────────────────────────────────────────────────────────
# 1. Smoother-covariance confidence bands
# ─────────────────────────────────────────────────────────────────────────

def strip_band_columns(contrib_df, g=None, min_frac=0.85):
    """
    Drop credible-band columns (ShortTerm_<ch>_lo / ShortTerm_<ch>_hi) from a
    contribution table and return (clean_df, bands_df), so every variable
    appears ONCE (point estimate) instead of three times (original+lo+hi).

    Name-based and deliberately simple: a ShortTerm_X_lo / ShortTerm_X_hi
    column is a band whenever its base ShortTerm_X column exists. Each
    column is judged on its own (no pairing / adjacency / numeric-ordering
    requirement - those made stale results leak through). The only
    exception: when the fitted model's variable map `g` is supplied and
    "X_lo" / "X_hi" is itself a real model variable, it is kept.
    """
    cols = list(contrib_df.columns)
    colset = set(cols)

    model_vars = set()
    if g:
        model_vars = {"Intercept"}
        for key in ("MEDIA_COLS", "COMP_MEDIA_COLS", "OWN_NONMEDIA_COLS",
                    "COMP_NONMEDIA_COLS", "PRICE_COLS"):
            model_vars.update(g.get(key, []) or [])

    drop = []
    for c in cols:
        if not c.startswith("ShortTerm_") or not (c.endswith("_lo") or c.endswith("_hi")):
            continue
        if c[len("ShortTerm_"):] in model_vars:
            continue                      # genuine model variable
        if c[:-3] in colset:              # base ShortTerm_X exists -> it's a band
            drop.append(c)
    if not drop:
        return contrib_df, pd.DataFrame(index=contrib_df.index)
    return contrib_df.drop(columns=drop), contrib_df[drop]


def _linear_state_index_map(g):
    """
    Ordered (state_index, column, kind) for every state dimension that has
    a direct RAW_VALUE * beta_t contribution in contrib_df. MUST mirror the
    index arithmetic in modules/pipeline.py::_postprocess_equation exactly
    (intercept is state index 0, then media, comp-media, own-nonmedia,
    comp-nonmedia, price, in that order) — keep the two in lockstep.
    """
    idx_map = []
    i = 1
    for col in g["MEDIA_COLS"]:
        idx_map.append((i, col, "media")); i += 1
    for col in g["COMP_MEDIA_COLS"]:
        idx_map.append((i, col, "comp_media")); i += 1
    for col in g["OWN_NONMEDIA_COLS"]:
        idx_map.append((i, col, "non_media")); i += 1
    for col in g["COMP_NONMEDIA_COLS"]:
        idx_map.append((i, col, "comp_nonmedia")); i += 1
    for col in g["PRICE_COLS"]:
        idx_map.append((i, col, "price")); i += 1
    return idx_map


def add_confidence_bands(result, df_full):
    """
    Mutates + returns `result` in place, adding:
      - contrib_df[f"ShortTerm_{col}_lo"/"_hi"]: period-level 95% CI on the
        short-term contribution of every linear (media / comp-media /
        non-media / comp-nonmedia / price) channel, from the RTS-smoother's
        own posterior beta variance (P_smooth diagonal). Exact under the
        model's own assumptions at the period level.
      - roi_df["TotalContrib_lo"/"_hi"] and ["ROI_lo"/"_hi"]: an approximate
        95% CI on the SUM over all periods, for media channels only (same
        rows as roi_df). This sums per-period variances — i.e. it treats
        period-to-period smoothing error as independent. Smoothed states
        are strongly autocorrelated (each beta_t is an AR(1)-like blend of
        its neighbours via the RTS backward pass), so this UNDERSTATES the
        true total uncertainty. Treat the total/ROI band as optimistic — a
        lower bound on how uncertain the number really is, not an exact
        interval.
    Silently no-ops (returns result unchanged) if P_smooth isn't present.
    """
    P_smooth = result.get("P_smooth")
    if P_smooth is None:
        return result

    g = result["g"]
    offset = int(result.get("state_offset", 0))
    contrib_df = result["contrib_df"]
    bands = result.setdefault("contrib_bands_df", pd.DataFrame(index=contrib_df.index))
    roi_df = result["roi_df"]
    media_set = set(g["MEDIA_COLS"])

    tot_lo, tot_hi, roi_lo, roi_hi = {}, {}, {}, {}

    for state_i, col, _kind in _linear_state_index_map(g):
        pcol = offset + state_i
        if pcol >= P_smooth.shape[1]:
            continue
        stcol = f"ShortTerm_{col}"
        if stcol not in contrib_df.columns or col not in df_full.columns:
            continue

        se_beta = np.sqrt(np.clip(P_smooth[:, pcol, pcol], 0.0, None))
        raw = df_full[col].values.astype(float)
        se_contrib = np.abs(raw) * se_beta

        bands[f"{stcol}_lo"] = contrib_df[stcol].values - Z95 * se_contrib
        bands[f"{stcol}_hi"] = contrib_df[stcol].values + Z95 * se_contrib

        if col in media_set:
            row = roi_df.loc[roi_df["Channel"] == col]
            if row.empty:
                continue
            total_mean = float(row["TotalContrib"].iloc[0])
            ts = float(row["TotalSpend"].iloc[0])
            total_se = float(np.sqrt(np.sum(se_contrib ** 2)))
            tot_lo[col] = total_mean - Z95 * total_se
            tot_hi[col] = total_mean + Z95 * total_se
            if ts > 0:
                roi_lo[col] = tot_lo[col] / ts
                roi_hi[col] = tot_hi[col] / ts

    roi_df["TotalContrib_lo"] = roi_df["Channel"].map(tot_lo)
    roi_df["TotalContrib_hi"] = roi_df["Channel"].map(tot_hi)
    roi_df["ROI_lo"] = roi_df["Channel"].map(roi_lo)
    roi_df["ROI_hi"] = roi_df["Channel"].map(roi_hi)

    result["contrib_df"] = contrib_df
    result["roi_df"] = roi_df
    result["uncertainty_note"] = (
        "95% bands come from the posterior state covariance "
        "(P_smooth). Per-period bands are exact under the fitted model. "
        "Total-contribution / ROI bands sum per-period variances (i.e. "
        "assume periods are independent); smoothed betas are actually "
        "autocorrelated across time, so these total bands are optimistic — "
        "read them as a lower bound on the true uncertainty, not an exact CI."
    )
    return result


def shortterm_total_ci(contrib_df, bands_df=None):
    """
    Sum-over-periods 95% CI for the TOTAL short-term contribution of every
    channel that has period-level ShortTerm_{col}_lo/_hi bands (i.e. every
    channel add_confidence_bands touched — media, comp-media, non-media,
    comp-nonmedia, price; NOT Intercept, and NOT LongTerm_* — see that
    function's docstring for why). Self-contained: backs the per-period SE
    out of the _lo/_hi columns already on contrib_df, so callers don't need
    P_smooth / g / df_full again — just a contrib_df that's already been
    through add_confidence_bands.

    Same caveat as roi_df's TotalContrib_lo/_hi: this sums per-period
    variances, i.e. treats periods as independent, which understates the
    true total uncertainty since smoothed betas are autocorrelated. Read it
    as an optimistic lower bound, not an exact interval.

    Returns a DataFrame with columns Channel, ShortTerm_Total, ShortTerm_lo,
    ShortTerm_hi — empty (but correctly-columned) if no bands are present.
    """
    cols = ["Channel", "ShortTerm_Total", "ShortTerm_lo", "ShortTerm_hi"]
    src = bands_df if bands_df is not None else contrib_df
    lo_cols = [c for c in src.columns
               if c.startswith("ShortTerm_") and c.endswith("_lo")]
    rows = []
    for lo_col in lo_cols:
        col = lo_col[len("ShortTerm_"):-len("_lo")]
        base_col, hi_col = f"ShortTerm_{col}", f"ShortTerm_{col}_hi"
        if base_col not in contrib_df.columns or hi_col not in src.columns:
            continue
        per_period_se = (src[hi_col].values - contrib_df[base_col].values) / Z95
        total_mean = float(contrib_df[base_col].sum())
        total_se = float(np.sqrt(np.sum(per_period_se ** 2)))
        rows.append({
            "Channel": col,
            "ShortTerm_Total": total_mean,
            "ShortTerm_lo": total_mean - Z95 * total_se,
            "ShortTerm_hi": total_mean + Z95 * total_se,
        })
    return pd.DataFrame(rows, columns=cols)


# ─────────────────────────────────────────────────────────────────────────
# 2. Multi-seed refit stability
# ─────────────────────────────────────────────────────────────────────────

def run_seed_stability(df_full, config, mcmc_cfg=None, n_seeds=3, base_seed=0, progress_cb=None):
    """
    Re-run the SAME config `n_seeds` times with different MCMC seeds (chain
    starting points and RNG streams; priors/bounds identical) and report how
    much the fitted ROI / channel ranking moves. Seed 0 uses `mcmc_cfg`'s own
    seed. Stable rankings across seeds mean the posterior is well explored;
    movement means run longer / raise target_accept.

    Returns dict: summary, per_seed_roi, per_seed_rank, seed_metrics,
    tau_vs_baseline (same shapes as before).
    """
    from modules.pipeline import run_multi_dependent_pipeline
    from modules.mcmc import mcmc_cfg_with_defaults

    cfg0 = mcmc_cfg_with_defaults(mcmc_cfg)
    seed_rois, seed_metrics = [], []
    for s in range(n_seeds):
        cfg = dict(cfg0); cfg["seed"] = cfg0["seed"] + (base_seed + s) * 1000 if s else cfg0["seed"]
        res, _res2 = run_multi_dependent_pipeline(df_full, config, cfg)
        roi = res["roi_df"][["Channel", "ROI"]].copy(); roi["seed"] = s
        seed_rois.append(roi)
        seed_metrics.append({"seed": s, "mape": res["mape"], "r2": res["r2"],
                              "r2_gelman": res["r2_gelman"], "loglik": res["loglik"],
                              "success": res["success"]})
        if progress_cb:
            progress_cb(s + 1, n_seeds)

    all_roi = pd.concat(seed_rois, ignore_index=True)
    pivot = all_roi.pivot(index="seed", columns="Channel", values="ROI")
    rank_pivot = pivot.rank(axis=1, ascending=False, method="average")
    summary = pd.DataFrame({
        "Channel": pivot.columns, "ROI_mean": pivot.mean(axis=0).values,
        "ROI_std": pivot.std(axis=0).values, "ROI_min": pivot.min(axis=0).values,
        "ROI_max": pivot.max(axis=0).values, "Rank_mean": rank_pivot.mean(axis=0).values,
        "Rank_std": rank_pivot.std(axis=0).values,
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["ROI_CV"] = summary["ROI_std"] / summary["ROI_mean"].abs()
    summary = summary.sort_values("Rank_mean").reset_index(drop=True)

    tau_rows = []
    if n_seeds > 1:
        from scipy.stats import kendalltau
        base = rank_pivot.iloc[0]
        for s in range(1, n_seeds):
            tau, _ = kendalltau(base.values, rank_pivot.iloc[s].values)
            tau_rows.append({"seed": s, "kendall_tau_vs_seed0": tau})
    return {"summary": summary, "per_seed_roi": pivot, "per_seed_rank": rank_pivot,
            "seed_metrics": pd.DataFrame(seed_metrics), "tau_vs_baseline": pd.DataFrame(tau_rows)}


# ─────────────────────────────────────────────────────────────────────────
# 3. VIF / collinearity diagnostic
# ─────────────────────────────────────────────────────────────────────────

def compute_vif(df_full, g, columns=None):
    """
    Variance Inflation Factor for each predictor actually fed to the
    model's observation/state equations — RAW spend/impressions/price
    series, not adstocked (see modules/statespace.py module docstring for why
    the model itself always uses raw regressors: carryover lives in the
    state's own persistence, not in a pre-decayed observation series).

    VIF_i = 1 / (1 - R_i^2), where R_i^2 comes from an OLS regression of
    column i on every other column (with intercept). Rule of thumb: VIF
    >= 5 is worth a look, >= 10 is a real collinearity problem — those
    channels' individual coefficients become unstable / hard to attribute
    even if the model's overall fit (MAPE, R²) looks fine.
    """
    if columns is None:
        columns = (list(g.get("MEDIA_COLS", [])) + list(g.get("COMP_MEDIA_COLS", [])) +
                   list(g.get("OWN_NONMEDIA_COLS", [])) + list(g.get("COMP_NONMEDIA_COLS", [])) +
                   list(g.get("PRICE_COLS", [])))
    columns = [c for c in dict.fromkeys(columns) if c in df_full.columns]

    if len(columns) < 2:
        return pd.DataFrame(columns=["Variable", "VIF", "Flag"])

    X = df_full[columns].astype(float).values
    n = X.shape[0]
    ones = np.ones((n, 1))

    rows = []
    for i, col in enumerate(columns):
        y = X[:, i]
        if np.std(y) < 1e-12:
            rows.append({"Variable": col, "VIF": np.nan})
            continue
        others = [j for j in range(len(columns)) if j != i]
        Xo = np.hstack([ones, X[:, others]])
        try:
            coef, *_ = np.linalg.lstsq(Xo, y, rcond=None)
            yhat = Xo @ coef
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
            r2 = min(max(r2, 0.0), 1 - 1e-9)
            vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = np.nan
        rows.append({"Variable": col, "VIF": vif})

    out = pd.DataFrame(rows).sort_values("VIF", ascending=False, na_position="last").reset_index(drop=True)
    out["Flag"] = np.select(
        [out["VIF"] >= 10, out["VIF"] >= 5],
        ["🔴 High (≥10)", "🟠 Moderate (≥5)"],
        default="🟢 OK (<5)",
    )
    out.loc[out["VIF"].isna(), "Flag"] = "⚪ Constant / undefined"
    return out
