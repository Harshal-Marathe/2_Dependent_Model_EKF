"""
Full RBE MMM pipeline: sample the joint posterior (parameters + latent
state path) with NUTS on the training window, then assemble contributions /
ROI / parameter tables and credible bands for the Results tab.

The state-space equations are unchanged (see modules/statespace.py); only
the inference engine changed — the Kalman filter / RTS smoother / point-
estimate optimizers were replaced by MCMC (see modules/mcmc.py).

Entry points:
  - run_full_pipeline:       single dependent variable.
  - run_multi_dependent_pipeline: one or two dependent variables. With a
    second dependent variable the two equations are sampled JOINTLY
    (shared time index, correlated observation errors rho, optional
    cross-intercept coupling phi_1 / phi_2).
  - run_chained_dependent_pipeline: Dependent 2 fitted first, its fitted
    path then drives Dependent 1.

What the numbers mean now:
  * params        posterior MEDIAN of every parameter
  * x_smooth      posterior MEAN of the latent states (the analogue of the
                  old RTS-smoothed states; contributions are linear in it)
  * P_smooth      posterior covariance of the states across draws
  * *_lo / *_hi   95% credible bands from the thinned posterior draws
  * loglik        posterior-mean training log-likelihood
  * holdout rows  true forecasts: the holdout target is never seen
"""

import numpy as np
import pandas as pd

from modules.params import _make_globals, unpack_theta
from modules.bounds import _build_theta0_and_bounds
from modules.statespace import _precompute_adstocked, _build_observation_matrix
from modules.transforms import apply_transformation, hill_transform_vec
from modules.metrics import safe_mape


# ── Shared post-processing (contributions / ROI / parameter tables) ─────────

def _postprocess_equation(df_full, g, params, x_smooth, adstocked_media,
                           cross_beta_contrib, opt_success, opt_nit, loglik,
                           n_train=None, lt_override=None, carry_override=None):
    """
    Given a fitted state trajectory for ONE equation (one dependent
    variable — under MCMC, the posterior-mean states), builds: smoothed yhat, MAPE/R2, the contribution
    table, ROI table, synergy table and parameter table. Used identically
    whether that equation came from a single-dependent fit or from one
    half of the joint bivariate fit.

    `lt_override` (T, N_EFFECTORS) and `carry_override` (T,) carry the
    per-draw posterior means of the long-term intercept pieces
    (gamma_k * f(effector_k) and G0 * I_{t-1}). Those are nonlinear in the
    parameters, so averaging per draw is more faithful than plugging the
    posterior median into the formula. When omitted, the old plug-in
    formulas are used.
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
    # modules/statespace.py::_apply_beta_floors), but posterior averaging
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
    if carry_override is not None:
        intercept_carryover = np.asarray(carry_override, dtype=float)

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
        if lt_override is not None:
            contrib_df[f"LongTerm_{col}"] = np.asarray(lt_override)[:, k]
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


# ── MCMC fit driver (shared by every pipeline below and by refit.py) ────────

def _run_mcmc_fit(df_full, n_train, eq_inputs, mcmc_cfg=None, joint=None,
                  theta_init=None, progress_cb=None):
    """
    Sample the joint posterior over (parameters, latent state path) for one
    equation (single-dependent) or two equations (joint bivariate).

    eq_inputs : list of dict(g=..., theta0=..., bounds=...) — theta0/bounds
                exactly as modules/bounds.py builds them (bounds become the
                prior support; lo == hi pins a parameter).
    joint     : None, or dict(use_coupling, allow_phi1, allow_phi2).
    theta_init: optional flat theta vector used to START the chains
                (warm start for Tab 8 refits); default = prior medians.
    """
    from modules import mcmc as M   # imported lazily: JAX/NumPyro are heavy

    cfg = M.mcmc_cfg_with_defaults(mcmc_cfg)
    df_tr = df_full.iloc[:n_train].reset_index(drop=True)

    eqs, priors, labels = [], [], []
    multi = len(eq_inputs) > 1
    for k, e in enumerate(eq_inputs):
        priors.append(M.build_prior(df_tr, e["g"], e["theta0"], e["bounds"]))
        eqs.append(dict(g=e["g"], st=M.build_eq_static(df_full, e["g"], n_train),
                        n_theta=len(e["theta0"])))
        labels += M.theta_labels(e["g"], prefix=(f"d{k + 1}:" if multi else ""))
    if joint is not None:
        extras = M.joint_extras_prior(joint["use_coupling"], joint["allow_phi1"], joint["allow_phi2"])
        priors.append(extras)
        labels += ["rho"] + (["phi1", "phi2"] if joint["use_coupling"] else [])
    prior = M.concat_priors(priors)

    model, pieces = M.make_model(eqs, prior, joint=joint)
    u_init = M.inverse_map_prior(theta_init, prior) if theta_init is not None else None

    samples, diag = M.run_nuts(model, pieces, cfg, u_init=u_init, progress_cb=progress_cb)
    if progress_cb:
        progress_cb(1.0, "Post-processing posterior draws …")
    post = M.posterior_states(pieces, samples, cfg["n_keep"])

    th_by_chain = samples["theta"]
    theta_med = np.median(th_by_chain.reshape(-1, th_by_chain.shape[-1]), axis=0)
    summary = M.parameter_summary(th_by_chain, labels, prior["kind"])
    rhat = summary["R-hat"].values
    max_rhat = float(np.nanmax(rhat)) if np.isfinite(rhat).any() else float("nan")
    min_ess = float(np.nanmin(summary["ESS"].values)) if len(summary) else float("nan")
    diag = dict(diag, max_rhat=max_rhat, min_ess=min_ess, cfg=cfg)
    converged = diag["divergences"] == 0 and (not np.isfinite(max_rhat) or max_rhat < 1.05)
    return dict(post=post, theta_med=theta_med, summary=summary, diag=diag,
                converged=bool(converged), eqs=eqs)


def _add_posterior_bands(result, df_full, g, post_e):
    """95% credible bands from the thinned posterior draws (replaces the old
    smoother-covariance approximation): exact per-period, and — unlike the
    old sum-of-variances shortcut — the total-contribution / ROI intervals
    keep the autocorrelation of the states because each draw is a whole path."""
    from modules.uncertainty import _linear_state_index_map

    X_keep, lt_keep = post_e["X_keep"], post_e["lt_keep"]
    contrib_df, roi_df = result["contrib_df"], result["roi_df"]
    eff_cols = list(g["INTERCEPT_EFFECTORS"])
    media_set = set(g["MEDIA_COLS"])
    tot_lo, tot_hi, roi_lo, roi_hi, roi_med, p_pos = {}, {}, {}, {}, {}, {}

    for state_i, col, _kind in _linear_state_index_map(g):
        if col not in df_full.columns or f"ShortTerm_{col}" not in contrib_df.columns:
            continue
        raw = df_full[col].values.astype(float)
        draws = X_keep[:, :, state_i] * raw[None, :]                  # (K, T)
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        contrib_df[f"ShortTerm_{col}_lo"] = lo
        contrib_df[f"ShortTerm_{col}_hi"] = hi
        if col in media_set:
            tot = draws.sum(axis=1)
            if col in eff_cols:                                         # + LongTerm_{col}
                tot = tot + lt_keep[:, :, eff_cols.index(col)].sum(axis=1)
            row = roi_df.loc[roi_df["Channel"] == col]
            if row.empty:
                continue
            ts = float(row["TotalSpend"].iloc[0])
            tot_lo[col], tot_hi[col] = (float(v) for v in np.percentile(tot, [2.5, 97.5]))
            if ts > 0:
                roi = tot / ts
                roi_lo[col], roi_hi[col] = (float(v) for v in np.percentile(roi, [2.5, 97.5]))
                roi_med[col] = float(np.median(roi))
                p_pos[col] = float(np.mean(roi > 0))

    roi_df["TotalContrib_lo"] = roi_df["Channel"].map(tot_lo)
    roi_df["TotalContrib_hi"] = roi_df["Channel"].map(tot_hi)
    roi_df["ROI_lo"] = roi_df["Channel"].map(roi_lo)
    roi_df["ROI_hi"] = roi_df["Channel"].map(roi_hi)
    roi_df["ROI_median"] = roi_df["Channel"].map(roi_med)
    roi_df["Prob_ROI_gt_0"] = roi_df["Channel"].map(p_pos)
    result["uncertainty_note"] = (
        "95% bands are credible intervals from the MCMC posterior draws "
        f"({X_keep.shape[0]} thinned draws). Each draw is a complete state path, so "
        "total-contribution and ROI intervals include the autocorrelation "
        "between periods (the old Kalman-smoother band treated periods as independent)."
    )
    return result


def _build_equation_result(df_full, g, fit, e_idx, theta_slice, n_train, loglik=None):
    post_e = fit["post"]["eqs"][e_idx]
    params = unpack_theta(fit["theta_med"][theta_slice], g)
    adstocked = _precompute_adstocked(df_full, g, params)
    result = _postprocess_equation(
        df_full, g, params, post_e["x_mean"], adstocked, post_e["cross_mean"],
        fit["converged"], fit["post"]["n_draws"],
        fit["post"]["loglik_mean"] if loglik is None else loglik,
        n_train=n_train, lt_override=post_e["lt_mean"], carry_override=post_e["carry_mean"],
    )
    result["P_smooth"] = post_e["P"]          # posterior state covariance, this equation's own block
    result["state_offset"] = 0
    _add_posterior_bands(result, df_full, g, post_e)
    result["holdout_mode"] = "forecast"
    result["mcmc"] = dict(summary=fit["summary"], diagnostics=fit["diag"],
                          max_rhat=fit["diag"]["max_rhat"], min_ess=fit["diag"]["min_ess"],
                          divergences=fit["diag"]["divergences"], converged=fit["converged"])
    return result


# ── Single-dependent-variable pipeline ───────────────────────────────────────

def run_full_pipeline(df_full, config, mcmc_cfg=None, progress_cb=None):
    g = _make_globals(config)
    n_train = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)
    theta0, bounds = _build_theta0_and_bounds(df_train, g)
    fit = _run_mcmc_fit(df_full, n_train, [dict(g=g, theta0=theta0, bounds=bounds)],
                        mcmc_cfg, progress_cb=progress_cb)
    return _build_equation_result(df_full, g, fit, 0, slice(0, len(theta0)), n_train)


def _dependent2_config(config):
    """Dependent 2's own config: its own predictor set / bounds / dummies /
    intercept dynamics, falling back to Dependent 1's for older saved configs."""
    config_2 = dict(config)
    config_2["target"]          = config["target2"]
    config_2["media"]           = config.get("media_2")           or config["media"]
    config_2["non_media"]       = config.get("non_media_2", config["non_media"])
    config_2["comp_media"]      = config.get("comp_media_2", config["comp_media"])
    config_2["comp_nonmedia"]   = config.get("comp_nonmedia_2", config["comp_nonmedia"])
    config_2["price"]           = config.get("price_2", config["price"])
    config_2["use_price"]       = config.get("use_price_2", config["use_price"])
    config_2["cross_media_map"] = config.get("cross_media_map_2", config["cross_media_map"])
    config_2["positive_beta_cols"] = config.get("positive_beta_cols_2", config["positive_beta_cols"])
    config_2["negative_beta_cols"] = config.get("negative_beta_cols_2", config["negative_beta_cols"])
    # Each dependent gets its OWN spike/outlier dummies (Tab 5 · A2b).
    config_2["dummy_cols"] = config.get("dummy_cols_2", config.get("dummy_cols", []))
    config_2["intercept_dynamics_type"] = config.get(
        "intercept_dynamics_type_2", config.get("intercept_dynamics_type", "carryover"))
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
    return config_2


# ── Multi-dependent pipeline — a genuine JOINT bivariate posterior ──────────

def run_multi_dependent_pipeline(df_full, config, mcmc_cfg=None, progress_cb=None):
    """
    Fits Dependent 1 (config["target"]) and, if a second dependent variable
    is configured (config["target2"]), fits it TOGETHER with Dependent 1:

        [ y1_t ]   [ Intercept_1_t ]   [ beta_1_1_t ... beta_1_M_t ]
        [ y2_t ] = [ Intercept_2_t ] + [ beta_2_1_t ... beta_2_M_t ] · x_t
                                                            + correlated errors

    Both equations' parameters, both latent state paths, the error
    correlation rho and the cross-intercept coupling phi_1 / phi_2 are one
    posterior, sampled in a single NUTS run:
        Intercept_1,t = G0_1·Intercept_1,t-1 + phi_1·Intercept_2,t-1 + effectors_1,t
        Intercept_2,t = G0_2·Intercept_2,t-1 + phi_2·Intercept_1,t-1 + effectors_2,t
    Which of phi_1/phi_2 are free is set by config
    "cross_intercept_coupling_mode" ("both" | "dep1_in_dep2" | "dep2_in_dep1"
    | "none"); coupling only exists when BOTH intercepts are on carryover.

    Returns (results_1, results_2); results_2 is None with no second dependent.
    """
    target2 = config.get("target2")
    if not (config.get("enable_second_dependent") and target2):
        return run_full_pipeline(df_full, config, mcmc_cfg, progress_cb=progress_cb), None

    config_1 = dict(config)
    config_2 = _dependent2_config(config)
    g1, g2 = _make_globals(config_1), _make_globals(config_2)

    n_train = config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)
    theta0_1, bounds1 = _build_theta0_and_bounds(df_train, g1)
    theta0_2, bounds2 = _build_theta0_and_bounds(df_train, g2)
    n1, n2 = len(theta0_1), len(theta0_2)

    coupling_mode = g1.get("CROSS_INTERCEPT_COUPLING_MODE", "both")
    use_coupling = (
        g1.get("INTERCEPT_DYNAMICS_TYPE", "carryover") != "simple"
        and g2.get("INTERCEPT_DYNAMICS_TYPE", "carryover") != "simple"
        and coupling_mode != "none"
    )
    allow_phi1 = coupling_mode in ("both", "dep2_in_dep1")   # Dep2 -> Dep1
    allow_phi2 = coupling_mode in ("both", "dep1_in_dep2")   # Dep1 -> Dep2

    fit = _run_mcmc_fit(
        df_full, n_train,
        [dict(g=g1, theta0=theta0_1, bounds=bounds1), dict(g=g2, theta0=theta0_2, bounds=bounds2)],
        mcmc_cfg, joint=dict(use_coupling=use_coupling, allow_phi1=allow_phi1, allow_phi2=allow_phi2),
        progress_cb=progress_cb)

    th = fit["theta_med"]
    best_rho = float(np.clip(th[n1 + n2], -0.995, 0.995))
    best_phi1 = float(th[n1 + n2 + 1]) if use_coupling and allow_phi1 else 0.0
    best_phi2 = float(th[n1 + n2 + 2]) if use_coupling and allow_phi2 else 0.0
    joint_loglik = fit["post"]["loglik_mean"]

    results_1 = _build_equation_result(df_full, g1, fit, 0, slice(0, n1), n_train, loglik=joint_loglik)
    results_2 = _build_equation_result(df_full, g2, fit, 1, slice(n1, n1 + n2), n_train, loglik=joint_loglik)

    ybar1 = float(df_full[g1["TARGET_COL"]].mean()) or 1e-8
    ybar2 = float(df_full[g2["TARGET_COL"]].mean()) or 1e-8
    rmse1 = float(np.sqrt(np.mean(results_1["residuals"] ** 2)))
    rmse2 = float(np.sqrt(np.mean(results_2["residuals"] ** 2)))
    for res in (results_1, results_2):
        res["rho_y"] = best_rho
        res["phi1"] = best_phi1
        res["phi2"] = best_phi2
        res["cross_intercept_coupling_mode"] = coupling_mode
        res["joint_loglik"] = joint_loglik
        res["joint_fit"] = True
        # MCMC has no NRMSE penalty in its objective (the posterior is
        # likelihood x priors); these are reported as diagnostics only.
        res["lambda_reg"] = 0.0
        res["loss_function_mode"] = "nll_only"
        res["nrmse_reg"] = rmse1 / abs(ybar1) + rmse2 / abs(ybar2)
        res["rmse_dep1"] = rmse1
        res["rmse_dep2"] = rmse2
    return results_1, results_2


# ── Chained / sequential pipeline — Dependent 2 feeds Dependent 1 as x_t ────

def run_chained_dependent_pipeline(df_full, config, mcmc_cfg=None, progress_cb=None):
    """
    Chained (mediation-style) two-stage fit, as an alternative to the joint
    fit above: Dependent 2 is fitted completely on its own first; its
    posterior-mean fitted path (or its raw actuals, config["chain_use_fitted"]
    = False) is then added as ONE new predictor to Dependent 1's equation
    (config["chain_driver_role"]: "media" or "non_media"), and Dependent 1 is
    fitted on its own. Two separate MCMC runs connected only through that
    one column.

    Returns (results_1, results_2, df_with_driver, driver_col).
    """
    target2 = config.get("target2")
    if not (config.get("enable_second_dependent") and target2):
        return run_full_pipeline(df_full, config, mcmc_cfg, progress_cb=progress_cb), None, df_full, None

    def _stage_cb(lo, hi):
        if progress_cb is None:
            return None
        return lambda frac, txt: progress_cb(lo + (hi - lo) * frac, txt)

    # ── Stage 1: Dependent 2 on its own ──────────────────────────────────
    config_2 = _dependent2_config(config)
    config_2["enable_second_dependent"] = False
    config_2["target2"] = None
    results_2 = run_full_pipeline(df_full, config_2, mcmc_cfg, progress_cb=_stage_cb(0.0, 0.5))

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

    results_1 = run_full_pipeline(df_with_driver, config_1, mcmc_cfg, progress_cb=_stage_cb(0.5, 1.0))
    results_1["chained_from_dep2"] = True
    results_1["chain_driver_col"]  = driver_col
    results_1["chain_use_fitted"]  = use_fitted
    results_1["chain_driver_role"] = driver_role
    results_2["chained_into_dep1"] = True

    return results_1, results_2, df_with_driver, driver_col
