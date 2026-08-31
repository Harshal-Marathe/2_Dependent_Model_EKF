"""
Model-uncertainty diagnostics, layered on top of an already-fitted model.
None of this touches how a model is fit: bounds (including the
positivity/negativity beta-sign constraints built in modules/bounds.py
and re-applied post-smoothing in modules/pipeline.py::_postprocess_equation)
are passed through completely unchanged everywhere in this module.

Three pieces:
  1. add_confidence_bands  — surfaces P_smooth (already computed by the RTS
     smoother, see modules/kalman.py::rts_smoother) as 95% CI bands on
     per-period contributions and on total contribution / ROI.
  2. run_seed_stability    — refits the SAME config from several different
     optimizer starting points (bounds untouched) and reports how much the
     converged ROI / ranking moves.
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

        contrib_df[f"{stcol}_lo"] = contrib_df[stcol].values - Z95 * se_contrib
        contrib_df[f"{stcol}_hi"] = contrib_df[stcol].values + Z95 * se_contrib

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
        "95% bands come from the Kalman smoother's own posterior variance "
        "(P_smooth). Per-period bands are exact under the fitted model. "
        "Total-contribution / ROI bands sum per-period variances (i.e. "
        "assume periods are independent); smoothed betas are actually "
        "autocorrelated across time, so these total bands are optimistic — "
        "read them as a lower bound on the true uncertainty, not an exact CI."
    )
    return result


def shortterm_total_ci(contrib_df):
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
    lo_cols = [c for c in contrib_df.columns
               if c.startswith("ShortTerm_") and c.endswith("_lo")]
    rows = []
    for lo_col in lo_cols:
        col = lo_col[len("ShortTerm_"):-len("_lo")]
        base_col, hi_col = f"ShortTerm_{col}", f"ShortTerm_{col}_hi"
        if base_col not in contrib_df.columns or hi_col not in contrib_df.columns:
            continue
        per_period_se = (contrib_df[hi_col].values - contrib_df[base_col].values) / Z95
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

def _random_theta0(theta0, bounds, rng, jitter_frac=0.35):
    """
    A perturbed starting point WITHIN the exact same `bounds` used for the
    real fit (including the positivity/negativity beta-sign bounds like
    (0, None) / (None, 0)) — only the search's starting guess moves, the
    constraints themselves are untouched. Finite two-sided bounds are
    resampled uniformly; one-sided or unbounded parameters are jittered
    around theta0 and then clipped to whatever finite side does exist.
    """
    theta0 = np.asarray(theta0, dtype=float)
    out = theta0.copy()
    for i, (lo, hi) in enumerate(bounds):
        v = theta0[i]
        lo_fin = lo is not None and np.isfinite(lo)
        hi_fin = hi is not None and np.isfinite(hi)
        if lo_fin and hi_fin and hi > lo:
            out[i] = rng.uniform(lo, hi)
        else:
            scale = max(abs(v), 1e-3) * jitter_frac
            cand = v + rng.normal(0.0, scale)
            if lo_fin:
                cand = max(cand, lo)
            if hi_fin:
                cand = min(cand, hi)
            out[i] = cand
    return out


def run_seed_stability(df_full, config, max_iter, method, n_seeds=5,
                        base_seed=0, ng_cfg=None, progress_cb=None):
    """
    Refit the SAME config `n_seeds` times, each time from a different
    optimizer starting point (bounds — including the positivity/negativity
    constraints — are identical every run; only theta0 moves), and report
    how much the fitted ROI / channel ranking moves across runs. Seed 0
    always uses the model's normal default starting point, so it matches
    what Tab 6 / Tab 8 would produce.

    A model whose ROI ranking is stable across seeds is one you can trust
    more; a lot of movement means the likelihood surface is flat or
    multi-modal for this config, and the point-estimate ROI numbers should
    be read with real caution regardless of how good MAPE/R² look.

    Returns a dict with:
      "summary"        — one row per channel: ROI mean/std/min/max and
                          mean/std of its rank across seeds.
      "per_seed_roi"    — seeds x channels ROI pivot table.
      "per_seed_rank"   — seeds x channels rank pivot table (1 = best ROI).
      "seed_metrics"    — MAPE / R² / log-lik per seed.
      "tau_vs_baseline" — Kendall's tau between each seed's full ranking
                          and seed 0's ranking (1.0 = identical order).
    """
    from modules.pipeline import (
        run_full_ekf_pipeline, run_multi_dependent_pipeline, _prep_joint,
    )
    from modules.params import _make_globals
    from modules.bounds import _build_theta0_and_bounds

    is_joint = bool(config.get("enable_second_dependent") and config.get("target2"))

    if is_joint:
        _, _, theta0_base, bounds, _, _ = _prep_joint(df_full, config)
    else:
        g = _make_globals(config)
        n_train = config["n_train"]
        df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)
        theta0_base, bounds = _build_theta0_and_bounds(df_train, g)

    seed_rois, seed_metrics = [], []
    for s in range(n_seeds):
        theta0_s = None
        if s > 0:
            rng = np.random.default_rng(base_seed + s)
            theta0_s = _random_theta0(theta0_base, bounds, rng)

        if is_joint:
            res, _res2 = run_multi_dependent_pipeline(
                df_full, config, max_iter, method, ng_cfg=ng_cfg,
                theta0_joint_override=theta0_s,
            )
        else:
            res = run_full_ekf_pipeline(
                df_full, config, max_iter, method, ng_cfg=ng_cfg,
                theta0_override=theta0_s,
            )

        roi = res["roi_df"][["Channel", "ROI"]].copy()
        roi["seed"] = s
        seed_rois.append(roi)
        seed_metrics.append({"seed": s, "mape": res["mape"], "r2": res["r2"],
                              "r2_gelman": res["r2_gelman"],
                              "loglik": res["loglik"], "success": res["success"]})
        if progress_cb:
            progress_cb(s + 1, n_seeds)

    all_roi = pd.concat(seed_rois, ignore_index=True)
    pivot = all_roi.pivot(index="seed", columns="Channel", values="ROI")
    rank_pivot = pivot.rank(axis=1, ascending=False, method="average")

    roi_mean = pivot.mean(axis=0)
    summary = pd.DataFrame({
        "Channel": pivot.columns,
        "ROI_mean": roi_mean.values,
        "ROI_std": pivot.std(axis=0).values,
        "ROI_min": pivot.min(axis=0).values,
        "ROI_max": pivot.max(axis=0).values,
        "Rank_mean": rank_pivot.mean(axis=0).values,
        "Rank_std": rank_pivot.std(axis=0).values,
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        summary["ROI_CV"] = summary["ROI_std"] / summary["ROI_mean"].abs()
    summary = summary.sort_values("Rank_mean").reset_index(drop=True)

    tau_rows = []
    if n_seeds > 1:
        from scipy.stats import kendalltau
        baseline_rank = rank_pivot.iloc[0]
        for s in range(1, n_seeds):
            tau, _ = kendalltau(baseline_rank.values, rank_pivot.iloc[s].values)
            tau_rows.append({"seed": s, "kendall_tau_vs_seed0": tau})
    tau_df = pd.DataFrame(tau_rows)

    return {
        "summary": summary,
        "per_seed_roi": pivot,
        "per_seed_rank": rank_pivot,
        "seed_metrics": pd.DataFrame(seed_metrics),
        "tau_vs_baseline": tau_df,
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. VIF / collinearity diagnostic
# ─────────────────────────────────────────────────────────────────────────

def compute_vif(df_full, g, columns=None):
    """
    Variance Inflation Factor for each predictor actually fed to the
    model's observation/state equations — RAW spend/impressions/price
    series, not adstocked (see modules/kalman.py module docstring for why
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
