"""
Full RBE MMM pipeline: optimize on the train split, run the filter +
smoother on the full dataset, and assemble contributions / ROI / parameter
tables for the Results tab.

Two entry points:
  - run_full_ekf_pipeline: single dependent variable (univariate RBE).
  - run_multi_dependent_pipeline: one or two dependent variables. When a
    second dependent variable is configured, the two equations are fitted
    JOINTLY with a bivariate Kalman filter (shared time index, correlated
    observation errors) rather than as two separate, independent fits —
    see modules/kalman.py::run_bivariate_kalman_filter for the state-space
    derivation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.dependencies import NEVERGRAD_AVAILABLE
from modules.params import _make_globals, unpack_theta
from modules.bounds import _build_theta0_and_bounds, build_normalized_problem
from modules.optimizer import run_nevergrad_optimizer, run_nevergrad_optimizer_joint
from modules.kalman import (
    run_kalman_filter, run_bivariate_kalman_filter, rts_smoother,
    _precompute_adstocked, _build_observation_matrix, build_static_cache,
)
from modules.transforms import apply_transformation, hill_transform_vec
from modules.metrics import safe_mape


# ── Shared post-processing (contributions / ROI / parameter tables) ─────────

def _postprocess_equation(df_full, g, params, x_smooth, adstocked_media,
                           cross_beta_contrib, opt_success, opt_nit, loglik,
                           n_train=None):
    """
    Given a fitted/smoothed state trajectory for ONE equation (one
    dependent variable), builds: smoothed yhat, MAPE/R2, the contribution
    table, ROI table, synergy table and parameter table. Used identically
    whether that equation came from the single-dependent univariate filter
    or from one half of the joint bivariate filter.
    """
    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    ADSTOCK_MAP    = g.get("ADSTOCK_MAP", {})
    ADSTOCK_IDX    = g.get("ADSTOCK_IDX", {})
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]

    x_smooth = x_smooth.copy()

    # Re-apply positivity / negativity floors after RTS smoothing
    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    negative_cols = set(g.get("NEGATIVE_BETA_COLS", []))
    for col in positive_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.maximum(x_smooth[:, idx], 0.0)
    for col in negative_cols:
        if col in MEDIA_COLS:
            idx = MEDIA_COLS.index(col) + 1
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in g["OWN_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["OWN_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in g["COMP_NONMEDIA_COLS"]:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] + g["COMP_NONMEDIA_COLS"].index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)
        elif col in PRICE_COLS:
            idx = 1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] + g["N_COMP_NONMEDIA"] + PRICE_COLS.index(col)
            x_smooth[:, idx] = np.minimum(x_smooth[:, idx], 0.0)

    # Baseline floor — a market-mix baseline shouldn't be negative or
    # near-zero. The forward filter already floors it (see
    # modules/kalman.py::_apply_beta_floors), but the RTS backward pass
    # can still pull it below the floor again since smoothing is an
    # unconstrained blend of filtered + next-period-smoothed values.
    min_base_fraction = float(g.get("MIN_BASE_FRACTION", 0.0))
    if min_base_fraction > 0:
        intercept_floor = min_base_fraction * float(df_full[TARGET_COL].mean())
        x_smooth[:, 0] = np.maximum(x_smooth[:, 0], intercept_floor)

    L_mat_full  = _build_observation_matrix(df_full, g, adstocked_media)
    yhat_smooth = (L_mat_full * x_smooth).sum(axis=1)

    target_vals  = df_full[TARGET_COL].values
    resid_smooth = target_vals - yhat_smooth
    # Zero-safe MAPE — see modules/metrics.py for why the naive
    # mean(|resid| / (|actual| + 1e-12)) formula is wrong here: any
    # period where the dependent variable is 0 (common for Dependent 2
    # KPIs like trial counts / leads, unlike a Sales Dependent 1) makes
    # that formula explode to billions of percent and swamp the metric.
    mape  = safe_mape(resid_smooth, target_vals)
    ss_res = np.sum(resid_smooth**2); ss_tot = np.sum((target_vals - target_vals.mean())**2)
    r2    = 1.0 - ss_res / (ss_tot + 1e-12)
    # Gelman R² (Bayesian R², Gelman et al. 2018) — point-estimate version:
    # ratio of explained variance to explained + residual variance, using
    # the final smoothed fit. Bounded in [0, 1] by construction, which makes
    # it more robust than classical R² for this regularized/state-space fit.
    var_yhat_g  = np.var(yhat_smooth, ddof=1)
    var_resid_g = np.var(resid_smooth, ddof=1)
    r2_gelman   = var_yhat_g / (var_yhat_g + var_resid_g + 1e-12)

    # ── In-sample (train) vs. out-of-sample (test/holdout) accuracy ──────
    # The optimizer only ever sees df_train (rows [:n_train]); the smoother
    # above runs over df_full, so everything from n_train onward is a
    # genuine out-of-sample forecast check, not just a fitted residual.
    # Each split's MAPE/R²/Gelman-R² is computed the same way as the
    # full-sample versions above, just restricted to that slice's own rows
    # (and, for R², its own mean as the "no model" baseline).
    def _slice_metrics(idx_slice):
        n_pts = idx_slice.stop - idx_slice.start if isinstance(idx_slice, slice) else None
        if n_pts is not None and n_pts <= 0:
            return {"mape": np.nan, "r2": np.nan, "r2_gelman": np.nan, "n": 0}
        y_s    = target_vals[idx_slice]
        yhat_s = yhat_smooth[idx_slice]
        res_s  = resid_smooth[idx_slice]
        mape_s = safe_mape(res_s, y_s)
        sst_s  = np.sum((y_s - y_s.mean())**2)
        r2_s   = 1.0 - np.sum(res_s**2) / (sst_s + 1e-12)
        if len(y_s) > 1:
            vy_s = np.var(yhat_s, ddof=1); vr_s = np.var(res_s, ddof=1)
            r2g_s = vy_s / (vy_s + vr_s + 1e-12)
        else:
            r2g_s = np.nan
        return {"mape": mape_s, "r2": r2_s, "r2_gelman": r2g_s, "n": len(y_s)}

    n_total = len(df_full)
    if n_train is not None and 0 < n_train <= n_total:
        in_metrics  = _slice_metrics(slice(0, n_train))
        out_metrics = _slice_metrics(slice(n_train, n_total)) if n_train < n_total else \
            {"mape": np.nan, "r2": np.nan, "r2_gelman": np.nan, "n": 0}
    else:
        # No train/test split info available — treat everything as in-sample.
        in_metrics  = _slice_metrics(slice(0, n_total))
        out_metrics = {"mape": np.nan, "r2": np.nan, "r2_gelman": np.nan, "n": 0}

    contrib_df = df_full[[TARGET_COL]].copy()

    G0 = float(params["G0"])
    I0 = float(params.get("I0", 0.0))
    prev_intercept = np.empty(len(df_full))
    prev_intercept[1:] = x_smooth[:-1, 0]
    prev_intercept[0]  = x_smooth[0, 0]
    intercept_carryover = G0 * prev_intercept

    # Short-term view: the intercept as it actually enters the observation
    # equation, Y_t = intercept_t + Σ beta_i,t * media_i,t + ...  (i.e. the
    # full smoothed intercept level, not a residual).
    contrib_df["ShortTerm_Intercept"] = x_smooth[:, 0]

    # Long-term view: decompose that SAME intercept per its own state
    # equation into a persistence/baseline piece and (below) a per-effector
    # boost piece. Named "Intercept Carryover" (not "Intercept") so it
    # doesn't collide with the short-term "Intercept" row when Short-Term +
    # Long-Term are combined.
    #   Carryover dynamics: I_t = G0 * I_(t-1) + Σ_k gamma_k * f(media_k,t)
    #   Simple dynamics:    I_t = I0           + Σ_k gamma_k * f(media_k,t)
    # In "simple" mode G0 is 0 so intercept_carryover is already all-zero;
    # the constant I0 baseline is broken out into its own column instead so
    # the long-term pieces still sum to the full intercept level.
    contrib_df["LongTerm_Intercept Carryover"] = intercept_carryover
    if g.get("INTERCEPT_DYNAMICS_TYPE", "carryover") == "simple":
        contrib_df["LongTerm_Intercept Baseline (I0)"] = np.full(len(df_full), I0)

    for i, col in enumerate(MEDIA_COLS):
        # Matches the observation equation: β_i,t is multiplied by RAW spend/
        # impressions, not adstocked media (carryover already lives in β_i,t
        # via its own λ_i decay / Weibull lag-weighting in the state equation).
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, i+1] * df_full[col].values.astype(float)
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for j, col in enumerate(COMP_MEDIA_COLS):
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, 1+g["N_MEDIA"]+j] * df_full[col].values.astype(float)
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+k
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+k
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0
    for p, col in enumerate(PRICE_COLS):
        si = 1+g["N_MEDIA"]+g["N_COMP"]+g["N_OWN_NONMEDIA"]+g["N_COMP_NONMEDIA"]+p
        contrib_df[f"ShortTerm_{col}"] = x_smooth[:, si] * df_full[col].values
        contrib_df[f"LongTerm_{col}"]  = 0.0

    INTERCEPT_TRANSFORM_TYPE = g.get("INTERCEPT_TRANSFORM_TYPE", "power")
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        if col not in df_full.columns:
            continue
        ni_int = params["n_intercept"][k]
        si_int = params["S_intercept"][k]
        raw = df_full[col].values.astype(float)
        transformed = apply_transformation(raw, INTERCEPT_TRANSFORM_TYPE, ni_int, si_int)
        contrib_df[f"LongTerm_{col}"] = params["gamma"][k] * transformed

    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        contrib_df[f"Synergy_{tgt}_from_{src}"] = cross_beta_contrib[:, k]

    # ROI denominator: for a channel whose raw input is GRP/impressions
    # (not currency), summing that column itself is meaningless as
    # "spend". MEDIA_SPEND_MAP (built in modules/params.py from
    # per_channel_bounds[col]["__spend_col__"], set in Tab 5 · D2 / Tab 8)
    # maps such a channel to its real spend column, whose TOTAL is used
    # as the ROI denominator instead. Channels left as "Spend" (the
    # default) fall back to summing themselves, unchanged from before.
    media_spend_map = g.get("MEDIA_SPEND_MAP", {})
    roi_rows = []
    for col in MEDIA_COLS:
        tc = contrib_df[f"ShortTerm_{col}"].sum() + contrib_df[f"LongTerm_{col}"].sum()
        spend_col = media_spend_map.get(col, col)
        if spend_col in df_full.columns:
            ts = df_full[spend_col].sum()
        else:
            spend_col = col
            ts = df_full[col].sum()
        roi_rows.append({"Channel": col,
                         "InputType": "GRP/Impressions" if spend_col != col else "Spend",
                         "SpendColumn": spend_col, "TotalSpend": ts, "TotalContrib": tc,
                         "ROI": tc/ts if ts > 0 else 0})
    roi_df = pd.DataFrame(roi_rows)

    synergy_rows = []
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        col_name = f"Synergy_{tgt}_from_{src}"
        total_synergy = float(contrib_df[col_name].sum())
        tgt_total = (contrib_df[f"ShortTerm_{tgt}"].sum()
                     + contrib_df[f"LongTerm_{tgt}"].sum()) if tgt in MEDIA_COLS else np.nan
        synergy_rows.append({
            "Source Channel": src,
            "Target Channel": tgt,
            "Total Synergy Contribution": total_synergy,
            "Avg Synergy / Period": float(contrib_df[col_name].mean()),
            "Cross Delta": float(params["cross_delta"][k]),
            "Cross Hill n": float(params["cross_n"][k]),
            "Cross Hill S": float(params["cross_S"][k]),
            "Share of Target's Total Contrib (%)": (
                round(100 * total_synergy / tgt_total, 2)
                if tgt_total and tgt_total != 0 and not np.isnan(tgt_total) else np.nan
            ),
        })
    synergy_df = pd.DataFrame(synergy_rows)

    # ── Parameter table ───────────────────────────────────────────────
    param_rows = []
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        effector_kind = "Media" if col in MEDIA_COLS else "Non-media"
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": "Gamma (boost coeff.)", "Value": params["gamma"][k],
        })
        param_rows.append({
            "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
            "Parameter": f"n_intercept ({'Hill slope' if INTERCEPT_TRANSFORM_TYPE == 'hill' else 'exponent'})",
            "Value": params["n_intercept"][k],
        })
        if INTERCEPT_TRANSFORM_TYPE == "hill":
            param_rows.append({
                "Category": "Intercept Effector", "Variable": f"{col} ({effector_kind})",
                "Parameter": "S_intercept (Half-sat)", "Value": params["S_intercept"][k],
            })

    for i, col in enumerate(MEDIA_COLS):
        if TRANSFORM_TYPE == "hill":
            transform_rows = [
                {"Category":"Media","Variable":col,"Parameter":"n (Hill slope)","Value":params["n_params"][i]},
                {"Category":"Media","Variable":col,"Parameter":"S (Half-sat)",  "Value":params["S_params"][i]},
            ]
        else:
            transform_rows = [
                {"Category":"Media","Variable":col,"Parameter":"n (Power exponent)","Value":params["n_params"][i]},
            ]
        param_rows += [
            {"Category":"Media","Variable":col,"Parameter":"Ls",    "Value":params["Ls"][i]},
            {"Category":"Media","Variable":col,"Parameter":"Delta", "Value":params["delta"][i]},
        ] + transform_rows

        # Per channel now — only channels individually set to weibull get
        # adstock shape/scale rows; instant channels' carryover is fully
        # captured by the "Ls" row above.
        if ADSTOCK_MAP.get(col) == "weibull" and col in ADSTOCK_IDX:
            ai = ADSTOCK_IDX[col]
            param_rows += [
                {"Category":"Media","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][ai]},
                {"Category":"Media","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][ai]},
            ]

    for j, col in enumerate(COMP_MEDIA_COLS):
        param_rows += [
            {"Category":"CompMedia","Variable":col,"Parameter":"Ls_comp",   "Value":params["Ls_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"Delta_comp", "Value":params["delta_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"n_comp",     "Value":params["n_comp"][j]},
            {"Category":"CompMedia","Variable":col,"Parameter":"S_comp",     "Value":params["S_comp"][j]},
        ]
        if ADSTOCK_MAP.get(col) == "weibull" and col in ADSTOCK_IDX:
            ai = ADSTOCK_IDX[col]
            param_rows += [
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][ai]},
                {"Category":"CompMedia","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][ai]},
            ]
        # Instant mode: no separate adstock row — carryover is the "Ls_comp" row above.

    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        param_rows += [
            {"Category":"NonMedia","Variable":col,"Parameter":"Ls",    "Value":params["Ls_own_nonmedia"][k]},
            {"Category":"NonMedia","Variable":col,"Parameter":"Delta", "Value":params["delta_own_nonmedia"][k]},
        ]
        if ADSTOCK_MAP.get(col) == "weibull" and col in ADSTOCK_IDX:
            ai = ADSTOCK_IDX[col]
            param_rows += [
                {"Category":"NonMedia","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][ai]},
                {"Category":"NonMedia","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][ai]},
            ]

    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        param_rows += [
            {"Category":"CompNonMedia","Variable":col,"Parameter":"Ls_comp_nonmedia",    "Value":params["Ls_comp_nonmedia"][k]},
            {"Category":"CompNonMedia","Variable":col,"Parameter":"Delta_comp_nonmedia", "Value":params["delta_comp_nonmedia"][k]},
        ]
        if ADSTOCK_MAP.get(col) == "weibull" and col in ADSTOCK_IDX:
            ai = ADSTOCK_IDX[col]
            param_rows += [
                {"Category":"CompNonMedia","Variable":col,"Parameter":"Adstock shape k","Value":params["adstock_shape"][ai]},
                {"Category":"CompNonMedia","Variable":col,"Parameter":"Adstock scale λ","Value":params["adstock_scale"][ai]},
            ]

    for i, col in enumerate(PRICE_COLS):
        param_rows += [
            {"Category":"Price","Variable":col,"Parameter":"Ls_price",  "Value":params["Ls_price"][i]},
            {"Category":"Price","Variable":col,"Parameter":"Delta_price","Value":params["delta_price"][i]},
        ]
    for k, (tgt, src) in enumerate(g["CROSS_MEDIA_PAIRS"]):
        pair_label = f"{src}→{tgt}"
        param_rows += [
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Delta", "Value":params["cross_delta"][k]},
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Hill n","Value":params["cross_n"][k]},
            {"Category":"Synergy","Variable":pair_label,"Parameter":"Cross Hill S","Value":params["cross_S"][k]},
        ]
    if g.get("INTERCEPT_DYNAMICS_TYPE", "carryover") == "simple":
        param_rows.append({"Category":"Global","Variable":"Intercept","Parameter":"I0",     "Value":params.get("I0", 0.0)})
    else:
        param_rows.append({"Category":"Global","Variable":"Intercept","Parameter":"G0",     "Value":params["G0"]})
    param_rows.append({"Category":"Global","Variable":"Noise",    "Parameter":"sigma_y","Value":params["sigma_y"]})
    if g["USE_ORGANIC_DRIFT"]:
        param_rows.append({"Category":"Global","Variable":"Organic drift","Parameter":"mu","Value":params["mu"]})

    if g.get("ADSTOCK_ANY_WEIBULL"):
        param_rows.append({"Category":"Global","Variable":"Weibull adstock",
                            "Parameter":"n_lags", "Value": g["ADSTOCK_N_LAGS"]})

    param_df = pd.DataFrame(param_rows)

    return {
        "params":params,"yhat_smooth":yhat_smooth,"residuals":resid_smooth,
        "x_smooth":x_smooth,"adstocked_media":adstocked_media,
        "contrib_df":contrib_df,"roi_df":roi_df,"param_df":param_df,"synergy_df":synergy_df,
        "loglik":loglik,"mape":mape,"r2":r2,"r2_gelman":r2_gelman,
        "n_train":n_train, "n_test": (n_total - n_train) if n_train is not None else 0,
        "mape_in":in_metrics["mape"], "r2_in":in_metrics["r2"], "r2_gelman_in":in_metrics["r2_gelman"],
        "mape_out":out_metrics["mape"], "r2_out":out_metrics["r2"], "r2_gelman_out":out_metrics["r2_gelman"],
        "success":opt_success,"nit":opt_nit,"g":g,
    }


# ── Single-dependent-variable pipeline (univariate RBE) ──────────────────────

def run_full_ekf_pipeline(df_full, config, max_iter, method, ng_cfg=None):
    g = _make_globals(config)
    n_train  = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)
    theta0, bounds = _build_theta0_and_bounds(df_train, g)

    # The observation matrix depends only on (df, g), never on the theta
    # being searched over — build it once per optimization run instead of
    # on every single candidate evaluation.
    static_cache_train = build_static_cache(df_train, g)

    if method == "Nevergrad" and NEVERGRAD_AVAILABLE and ng_cfg:
        best_theta, _ = run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg,
                                                  static_cache=static_cache_train)
        opt_success = True; opt_nit = ng_cfg.get("budget", 500)
    else:
        # See modules/bounds.py::build_normalized_problem for why this
        # normalization matters: without it, L-BFGS-B/SLSQP's single
        # scalar finite-difference `eps` applied across theta's wildly
        # different natural scales (Hill's n ~ O(1-15) next to S ~ O(1e8))
        # leaves small-range parameters like `n` stuck exactly at their
        # init value — their true gradient signal is below the numerical
        # noise floor of the Kalman recursion at the default step size.
        theta0_norm, norm_bounds, unscale = build_normalized_problem(theta0, bounds)

        def objective(theta_norm):
            theta = unscale(theta_norm)
            p = unpack_theta(theta, g)
            _, _, _, _, _, _, _, _, loglik = run_kalman_filter(
                df_train, p, g, static_cache=static_cache_train)
            return -loglik
        opt = minimize(objective, theta0_norm, method=method,
                       bounds=norm_bounds,
                       options={"maxiter": max_iter, "ftol": 1e-9, "eps": 1e-6})
        best_theta = unscale(opt.x); opt_success = opt.success; opt_nit = opt.nit

    params = unpack_theta(best_theta, g)
    static_cache_full = build_static_cache(df_full, g)
    yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik = \
        run_kalman_filter(df_full, params, g, static_cache=static_cache_full)
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat)

    adstocked_media = _precompute_adstocked(df_full, g, params)
    result = _postprocess_equation(
        df_full, g, params, x_smooth, adstocked_media, cross_beta_contrib,
        opt_success, opt_nit, loglik, n_train=n_train,
    )
    result["P_smooth"] = P_smooth
    return result


# ── Multi-dependent pipeline — now a genuine JOINT bivariate fit ────────────

def run_multi_dependent_pipeline(df_full, config, max_iter, method, ng_cfg=None):
    """
    Fits Dependent 1 (config["target"]) and, if a second dependent variable
    is configured (config["target2"], e.g. Top-of-Mind / Consideration),
    fits it TOGETHER with Dependent 1 using a single joint bivariate Kalman
    filter:

        [ y1_t ]   [ Intercept_1_t ]   [ beta_1_1_t ... beta_1_M_t ]
        [ y2_t ] = [ Intercept_2_t ] + [ beta_2_1_t ... beta_2_M_t ] · x_t
                                                            + correlated errors

    The two equations share the same time index and the same raw
    regressors x_t, but each keeps its own state dynamics (its own Ls, G0,
    adstock, transform, and per-channel bounds via config["per_channel_bounds_2"]
    if provided). What makes the fit "joint" rather than two independent
    fits stitched together is:
      1. A single optimizer call estimates BOTH equations' parameters
         (theta_1, theta_2), the cross-equation error correlation rho,
         AND the cross-intercept coupling coefficients phi_1/phi_2
         simultaneously, by maximising the bivariate log-likelihood:
             Intercept_1,t = G0_1·Intercept_1,t-1 + phi_1·Intercept_2,t-1 + effectors_1,t
             Intercept_2,t = G0_2·Intercept_2,t-1 + phi_2·Intercept_1,t-1 + effectors_2,t
         Which of phi_1/phi_2 actually get a free theta slot (vs. being
         pinned at exactly 0) is controlled by config
         "cross_intercept_coupling_mode" — "both" (default), "dep1_in_dep2"
         (phi_2 only), "dep2_in_dep1" (phi_1 only), or "none" (neither).
         See the CROSS_INTERCEPT_COUPLING_MODE block below.
      2. At every time step, the Kalman gain is computed from the full
         2x2 observation-noise covariance, so a surprising observation in
         one equation also updates the other equation's state estimate
         (through the off-diagonal covariance terms) at that same t.

    If no second dependent variable is configured, this transparently
    falls back to the single-equation pipeline (results_2 is None).

    Returns
    -------
    (results_1, results_2)
        results_2 is None if no second dependent variable is configured.
        When both are fitted, each result dict also carries "rho_y"
        (estimated error correlation), "phi1"/"phi2" (estimated
        cross-intercept coupling coefficients), and "joint_loglik"
        (shared bivariate log-likelihood) for display/diagnostics.
    """
    target2 = config.get("target2")
    if not (config.get("enable_second_dependent") and target2):
        results_1 = run_full_ekf_pipeline(df_full, config, max_iter, method, ng_cfg=ng_cfg)
        return results_1, None

    # ── Build per-equation configs / globals ─────────────────────────
    # Dependent 2 can reuse Dependent 1's exact predictor set (default,
    # backward-compatible with older saved configs that have no _2 keys),
    # or use its own independently-selected — and potentially overlapping —
    # set of media / non-media / price / competitor variables, configured
    # in Tab 5 · Section A3.
    config_1 = dict(config)
    config_2 = dict(config)
    config_2["target"]          = target2
    config_2["media"]           = config.get("media_2")           or config["media"]
    config_2["non_media"]       = config.get("non_media_2", config["non_media"])
    config_2["comp_media"]      = config.get("comp_media_2", config["comp_media"])
    config_2["comp_nonmedia"]   = config.get("comp_nonmedia_2", config["comp_nonmedia"])
    config_2["price"]           = config.get("price_2", config["price"])
    config_2["use_price"]       = config.get("use_price_2", config["use_price"])
    config_2["cross_media_map"] = config.get("cross_media_map_2", config["cross_media_map"])
    config_2["positive_beta_cols"] = config.get("positive_beta_cols_2", config["positive_beta_cols"])
    config_2["negative_beta_cols"] = config.get("negative_beta_cols_2", config["negative_beta_cols"])
    # Spike/outlier impulse dummies (see modules/spike_dummies.py /
    # Tab 5 · Section A2b): each dependent gets its OWN dummy columns by
    # default (dummy_cols_2, generated against target2's own series) — NOT
    # config_1's dummy_cols, which would otherwise put Dependent 1's spike
    # dummies into Dependent 2's equation too (harmless — an irrelevant
    # dummy just settles near a ~0 beta — but pointless extra state
    # dimensions). Falls back to sharing config["dummy_cols"] only for
    # older saved configs that predate "dummy_cols_2" entirely.
    config_2["dummy_cols"] = config.get("dummy_cols_2", config.get("dummy_cols", []))
    # Re-derive Dep 2's own initial-beta defaults against ITS OWN (possibly
    # different) channel lists rather than reusing Dep 1's, which may not
    # even contain the same columns.
    config_2["initial_media_betas"]         = {c: 0.0     for c in config_2["media"]}
    config_2["initial_comp_betas"]          = {c: -0.0001 for c in config_2["comp_media"]}
    config_2["initial_own_nonmedia_betas"]  = {c: 0.0     for c in config_2["non_media"]}
    config_2["initial_comp_nonmedia_betas"] = {c: -0.01   for c in config_2["comp_nonmedia"]}
    config_2["initial_price_beta"]          = {c: -0.1    for c in config_2["price"]}

    ie2 = config.get("intercept_effectors_2")
    if ie2 is not None:
        config_2["intercept_effectors"] = ie2
    pcb_2 = config.get("per_channel_bounds_2")
    if pcb_2:
        config_2["per_channel_bounds"] = pcb_2

    g1 = _make_globals(config_1)
    g2 = _make_globals(config_2)

    n_train  = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)

    theta0_1, bounds1 = _build_theta0_and_bounds(df_train, g1)
    theta0_2, bounds2 = _build_theta0_and_bounds(df_train, g2)
    n1 = len(theta0_1)
    n2 = len(theta0_2)

    rho0 = 0.0
    rho_bounds = (-0.95, 0.95)
    # Cross-intercept coupling — see modules/kalman.py module docstring:
    #   Intercept_1,t = G0_1·Intercept_1,t-1 + phi_1·Intercept_2,t-1 + effectors_1,t
    #   Intercept_2,t = G0_2·Intercept_2,t-1 + phi_2·Intercept_1,t-1 + effectors_2,t
    # phi is itself a carryover mechanism (it references the OTHER
    # equation's PREVIOUS intercept), so it only makes sense — and only
    # gets a theta slot — when both equations are on "carryover" intercept
    # dynamics. Both g1/g2 share the one "intercept_dynamics_type" config
    # value (no per-dependent override), so checking g1 is sufficient.
    # In "simple" mode the theta_joint layout is simply [theta_1|theta_2|rho]
    # (one fewer block than the carryover-mode layout below) — every
    # downstream reader (optimizer.py, and the extraction code further
    # down) infers which layout it got from len(theta_joint) - (n1+n2)
    # rather than assuming a fixed width, so nothing else needs to branch
    # on this flag.
    #
    # On top of that carryover gate, CROSS_INTERCEPT_COUPLING_MODE (also
    # shared across g1/g2 — checking g1 is sufficient) picks which
    # direction(s) of coupling are actually estimated:
    #   "both"         -> phi_1 and phi_2 both free
    #   "dep1_in_dep2" -> only phi_2 free (Dep1's previous intercept feeds
    #                     Dep2's equation); phi_1 pinned to 0
    #   "dep2_in_dep1" -> only phi_1 free (Dep2's previous intercept feeds
    #                     Dep1's equation); phi_2 pinned to 0
    #   "none"         -> same as "simple" intercept dynamics for coupling
    #                     purposes — no phi theta slots at all
    coupling_mode = g1.get("CROSS_INTERCEPT_COUPLING_MODE", "both")
    use_cross_intercept_coupling = (
        g1.get("INTERCEPT_DYNAMICS_TYPE", "carryover") != "simple"
        and coupling_mode != "none"
    )
    # allow_phi1/allow_phi2 also gate the objective function and the
    # best-theta extraction below, so a direction that's "pinned to 0" via
    # bounds is ALSO forced to exactly 0.0 there — belt and suspenders,
    # since a (0.0, 0.0) bound only constrains the optimizer's search, not
    # the tiny numerical floor `build_normalized_problem` uses internally.
    allow_phi1 = coupling_mode in ("both", "dep2_in_dep1")  # Dep2 -> Dep1
    allow_phi2 = coupling_mode in ("both", "dep1_in_dep2")  # Dep1 -> Dep2
    phi1_0, phi2_0 = 0.0, 0.0
    phi1_bounds = (0.0, None) if allow_phi1 else (0.0, 0.0)
    phi2_bounds = (0.0, None) if allow_phi2 else (0.0, 0.0)
    if use_cross_intercept_coupling:
        theta0_joint = np.concatenate([theta0_1, theta0_2, [rho0, phi1_0, phi2_0]])
        bounds_joint = list(bounds1) + list(bounds2) + [rho_bounds, phi1_bounds, phi2_bounds]
    else:
        theta0_joint = np.concatenate([theta0_1, theta0_2, [rho0]])
        bounds_joint = list(bounds1) + list(bounds2) + [rho_bounds]
    # Safety net (mirrors modules/bounds.py): guarantees theta0_joint[i] always
    # lies inside bounds_joint[i], since the rho/phi appends above happen after
    # theta0_1/theta0_2 were already clipped individually and aren't covered by
    # that earlier clip.
    theta0_joint = np.array([
        float(np.clip(v,
                       lo if lo is not None else -np.inf,
                       hi if hi is not None else  np.inf))
        for v, (lo, hi) in zip(theta0_joint, bounds_joint)
    ])

    static_cache1_train = build_static_cache(df_train, g1)
    static_cache2_train = build_static_cache(df_train, g2)

    if method == "Nevergrad" and NEVERGRAD_AVAILABLE and ng_cfg:
        best_theta_joint, _ = run_nevergrad_optimizer_joint(
            df_train, g1, g2, theta0_joint, bounds_joint, n1, n2, ng_cfg,
            static_cache1=static_cache1_train, static_cache2=static_cache2_train)
        opt_success = True; opt_nit = ng_cfg.get("budget", 500)
    else:
        # Same normalization fix as the single-dependent path above (see
        # modules/bounds.py::build_normalized_problem) — theta_joint mixes
        # the same wide-scale parameters (twice over, once per dependent
        # variable) plus rho/phi, so it needs it just as much.
        theta0_joint_norm, norm_bounds_joint, unscale_joint = build_normalized_problem(
            theta0_joint, bounds_joint)

        def objective(theta_joint_norm):
            theta_joint = unscale_joint(theta_joint_norm)
            theta1 = theta_joint[:n1]
            theta2 = theta_joint[n1:n1+n2]
            rho    = theta_joint[n1+n2]
            if len(theta_joint) - (n1 + n2) >= 3:
                phi1 = theta_joint[n1+n2+1] if allow_phi1 else 0.0
                phi2 = theta_joint[n1+n2+2] if allow_phi2 else 0.0
            else:
                phi1 = phi2 = 0.0
            p1 = unpack_theta(theta1, g1)
            p2 = unpack_theta(theta2, g2)
            (_, _, _, _, _, _, _, _, _, loglik, _, _) = \
                run_bivariate_kalman_filter(df_train, p1, g1, p2, g2, rho, phi1, phi2,
                                             static_cache1=static_cache1_train,
                                             static_cache2=static_cache2_train)
            return -loglik
        opt = minimize(objective, theta0_joint_norm, method=method,
                        bounds=norm_bounds_joint,
                        options={"maxiter": max_iter, "ftol": 1e-9, "eps": 1e-6})
        best_theta_joint = unscale_joint(opt.x); opt_success = opt.success; opt_nit = opt.nit

    best_theta1 = best_theta_joint[:n1]
    best_theta2 = best_theta_joint[n1:n1+n2]
    best_rho    = float(np.clip(best_theta_joint[n1+n2], -0.995, 0.995))
    if len(best_theta_joint) - (n1 + n2) >= 3:
        best_phi1 = float(best_theta_joint[n1+n2+1]) if allow_phi1 else 0.0
        best_phi2 = float(best_theta_joint[n1+n2+2]) if allow_phi2 else 0.0
    else:
        best_phi1 = 0.0
        best_phi2 = 0.0
    params1 = unpack_theta(best_theta1, g1)
    params2 = unpack_theta(best_theta2, g2)

    # ── Run the joint filter on the FULL dataset with the fitted params ──
    static_cache1_full = build_static_cache(df_full, g1)
    static_cache2_full = build_static_cache(df_full, g2)
    (yhat_joint, residuals_joint, x_filt, P_filt, x_pred, P_pred, Tmat_joint,
     cross1, cross2, joint_loglik, dim1, dim2) = run_bivariate_kalman_filter(
        df_full, params1, g1, params2, g2, best_rho, best_phi1, best_phi2,
        static_cache1=static_cache1_full, static_cache2=static_cache2_full)

    # RTS smoother is dimension-agnostic — run once on the joint state
    x_smooth_joint, P_smooth_joint = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat_joint)
    x_smooth_1 = x_smooth_joint[:, :dim1]
    x_smooth_2 = x_smooth_joint[:, dim1:]

    adstocked_media_1 = _precompute_adstocked(df_full, g1, params1)
    adstocked_media_2 = _precompute_adstocked(df_full, g2, params2)

    results_1 = _postprocess_equation(
        df_full, g1, params1, x_smooth_1, adstocked_media_1, cross1,
        opt_success, opt_nit, joint_loglik, n_train=n_train,
    )
    results_2 = _postprocess_equation(
        df_full, g2, params2, x_smooth_2, adstocked_media_2, cross2,
        opt_success, opt_nit, joint_loglik, n_train=n_train,
    )

    for res in (results_1, results_2):
        res["rho_y"] = best_rho
        res["phi1"] = best_phi1  # coefficient of Intercept_Dep2,t-1 in Dep1's intercept eq
        res["phi2"] = best_phi2  # coefficient of Intercept_Dep1,t-1 in Dep2's intercept eq
        res["cross_intercept_coupling_mode"] = coupling_mode  # "both" | "dep1_in_dep2" | "dep2_in_dep1" | "none"
        res["joint_loglik"] = joint_loglik
        res["joint_fit"] = True
        res["P_smooth"] = P_smooth_joint  # joint covariance (block layout: dim1 then dim2)

    return results_1, results_2


# ── Chained / sequential pipeline — Dependent 2 feeds Dependent 1 as x_t ────

def run_chained_dependent_pipeline(df_full, config, max_iter, method, ng_cfg=None):
    """
    Chained (mediation-style) two-stage fit, as an alternative to the joint
    bivariate fit above.

    Stage 1 — Dependent 2 (config["target2"]) is fitted completely on its
    own, exactly like a single-dependent RBE model: its own predictors, own
    adstock/saturation, own equation, own optimizer run
    (run_full_ekf_pipeline). It is a genuine KPI fit with its own
    contributions / ROI / parameter tables.

    Stage 2 — Dependent 2's resulting SMOOTHED trajectory
    (config["chain_use_fitted"] = True, the default) — or, if
    config["chain_use_fitted"] is False, its raw actual column as-is — is
    added to the dataset as one new predictor and handed to Dependent 1
    (config["target"])'s own equation, in the role given by
    config["chain_driver_role"] ("media": gets its own adstock + saturation
    curve like a channel; "non_media": a direct beta, no adstock/saturation,
    like an organic control). Dependent 1 is then fitted on its own too
    (a second, separate run_full_ekf_pipeline call) — so Dependent 2 ends up
    being BOTH a KPI in its own right AND an x-variable driving Dependent 1.

    This differs from run_multi_dependent_pipeline (joint mode): there the
    two equations share a single estimation step and only influence each
    other through a correlated-error / cross-intercept term. Here they are
    two fully separate optimizer calls, run strictly in sequence, connected
    only through the one new predictor column.

    Returns
    -------
    (results_1, results_2, df_with_driver, driver_col)
        results_2 : Dependent 2's own standalone fit (same shape as
            run_full_ekf_pipeline's return).
        results_1 : Dependent 1's fit, run on `df_with_driver` and carrying
            two extra keys: "chain_driver_col" (the new predictor's name)
            and "chain_use_fitted".
        df_with_driver : df_full plus the new driver column (or df_full
            unchanged if chain_use_fitted is False, since the raw target2
            column is already present).
        driver_col : name of the new predictor column fed into Dependent 1
            (None if no second dependent variable is configured at all).
    """
    target2 = config.get("target2")
    if not (config.get("enable_second_dependent") and target2):
        results_1 = run_full_ekf_pipeline(df_full, config, max_iter, method, ng_cfg=ng_cfg)
        return results_1, None, df_full, None

    # ── Stage 1: fit Dependent 2 completely on its own ───────────────────
    config_2 = dict(config)
    config_2["target"]                  = target2
    config_2["enable_second_dependent"] = False
    config_2["target2"]                 = None
    config_2["media"]           = config.get("media_2")           or config["media"]
    config_2["non_media"]       = config.get("non_media_2", config["non_media"])
    config_2["comp_media"]      = config.get("comp_media_2", config["comp_media"])
    config_2["comp_nonmedia"]   = config.get("comp_nonmedia_2", config["comp_nonmedia"])
    config_2["price"]           = config.get("price_2", config["price"])
    config_2["use_price"]       = config.get("use_price_2", config["use_price"])
    config_2["cross_media_map"] = config.get("cross_media_map_2", config["cross_media_map"])
    config_2["positive_beta_cols"] = config.get("positive_beta_cols_2", config["positive_beta_cols"])
    config_2["negative_beta_cols"] = config.get("negative_beta_cols_2", config["negative_beta_cols"])
    # See the identical note in run_multi_dependent_pipeline above — Dep 2
    # gets its own spike/outlier dummies by default, not Dep 1's.
    config_2["dummy_cols"] = config.get("dummy_cols_2", config.get("dummy_cols", []))
    config_2["initial_media_betas"]         = {c: 0.0     for c in config_2["media"]}
    config_2["initial_comp_betas"]          = {c: -0.0001 for c in config_2["comp_media"]}
    config_2["initial_own_nonmedia_betas"]  = {c: 0.0     for c in config_2["non_media"]}
    config_2["initial_comp_nonmedia_betas"] = {c: -0.01   for c in config_2["comp_nonmedia"]}
    config_2["initial_price_beta"]          = {c: -0.1    for c in config_2["price"]}
    ie2 = config.get("intercept_effectors_2")
    if ie2 is not None:
        config_2["intercept_effectors"] = ie2
    pcb_2 = config.get("per_channel_bounds_2")
    if pcb_2:
        config_2["per_channel_bounds"] = pcb_2

    results_2 = run_full_ekf_pipeline(df_full, config_2, max_iter, method, ng_cfg=ng_cfg)

    # ── Stage 2: inject Dependent 2's output as an x-driver, fit Dep 1 ───
    use_fitted  = config.get("chain_use_fitted", True)
    driver_role = config.get("chain_driver_role", "non_media")   # "media" | "non_media"
    force_positive = config.get("chain_driver_positive", True)

    if use_fitted:
        driver_col = f"{target2}__fitted_driver"
        df_with_driver = df_full.copy()
        df_with_driver[driver_col] = results_2["yhat_smooth"]
    else:
        driver_col = target2
        df_with_driver = df_full

    config_1 = dict(config)
    config_1["enable_second_dependent"] = False
    config_1["target2"] = None

    if driver_role == "media":
        config_1["media"] = list(config["media"])
        if driver_col not in config_1["media"]:
            config_1["media"] = config_1["media"] + [driver_col]
        config_1["initial_media_betas"] = dict(config.get("initial_media_betas", {}))
        config_1["initial_media_betas"].setdefault(driver_col, 0.0)
    else:
        config_1["non_media"] = list(config["non_media"])
        if driver_col not in config_1["non_media"]:
            config_1["non_media"] = config_1["non_media"] + [driver_col]
        config_1["initial_own_nonmedia_betas"] = dict(config.get("initial_own_nonmedia_betas", {}))
        config_1["initial_own_nonmedia_betas"].setdefault(driver_col, 0.0)

    if force_positive:
        config_1["positive_beta_cols"] = list(config.get("positive_beta_cols", []))
        if driver_col not in config_1["positive_beta_cols"]:
            config_1["positive_beta_cols"].append(driver_col)
        config_1["negative_beta_cols"] = [
            c for c in config.get("negative_beta_cols", []) if c != driver_col
        ]

    adstock_map = dict(config.get("adstock_map", {}))
    adstock_map.setdefault(driver_col, "instant")
    config_1["adstock_map"] = adstock_map

    results_1 = run_full_ekf_pipeline(df_with_driver, config_1, max_iter, method, ng_cfg=ng_cfg)
    results_1["chained_from_dep2"] = True
    results_1["chain_driver_col"]  = driver_col
    results_1["chain_use_fitted"]  = use_fitted
    results_1["chain_driver_role"] = driver_role
    results_2["chained_into_dep1"] = True

    return results_1, results_2, df_with_driver, driver_col
