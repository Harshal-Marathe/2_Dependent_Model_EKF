"""
Tab 7 — Results & ROI Analytics: performance metrics, fit charts,
residual diagnostics, channel contributions, ROI, parameters, and
interactive response curves.

`render_full_results()` is the reusable results body used by both this
tab and Tab 8 · Refine & Refit, so a refit in Tab 8 shows exactly the
same charts/tables and the same download options as the officially
saved model here — see modules/tabs/tab7_refine.py. Because both tabs
render in the same Streamlit script run (st.tabs, not separate pages),
every interactive/download widget inside render_full_results() is keyed
with the caller's key_prefix so Tab 7's and Tab 8's copies never collide.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.ui_helpers import section, need_model, safe_multiselect
from modules.transforms import hill_transform, power_transform, apply_transformation, weibull_lag_weights
from modules.exports import build_betas_df, build_master_workbook_bytes, build_full_results_zip_bytes


def _render_tab7_promote_section():
    """If Tab 8 · Refine & Refit has produced a working model that differs
    from the currently-saved official model, surface a compare-and-save
    panel here so the whole 'refit → review → save' loop lives in one
    place: refits happen in Tab 8, saving happens here in Tab 7."""
    refit_result = st.session_state.get("refit_result")
    refit_config = st.session_state.get("refit_config")
    if refit_result is None or refit_config is None:
        return

    steps_taken = len(st.session_state.get("refit_history", [])) - 1
    if steps_taken <= 0:
        return  # nothing refined yet — nothing to promote

    already_saved = refit_result is st.session_state.model_results
    with st.container(border=True):
        st.markdown("### 🔧 Refined Model Available (from Tab 8 · Refine & Refit)")
        base_mape = st.session_state.model_results["mape"]
        base_r2   = st.session_state.model_results["r2"]
        base_r2_g = st.session_state.model_results["r2_gelman"]
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Tab 6 baseline MAPE", f"{base_mape:.2%}")
        dc2.metric("Refined MAPE", f"{refit_result['mape']:.2%}",
                   delta=f"{(base_mape - refit_result['mape'])*100:+.2f} pp (lower is better)")
        dc3.metric("Tab 6 baseline Gelman R²", f"{base_r2_g:.4f}")
        dc4.metric("Refined Gelman R²", f"{refit_result['r2_gelman']:.4f}",
                   delta=f"{refit_result['r2_gelman']-base_r2_g:+.4f}")
        with st.expander("📊 R² metrics"):
            rc1, rc2 = st.columns(2)
            rc1.metric("Tab 6 baseline R²", f"{base_r2:.4f}")
            rc2.metric("Refined R²", f"{refit_result['r2']:.4f}",
                       delta=f"{refit_result['r2']-base_r2:+.4f}")
        st.caption(f"{steps_taken} refinement step(s) taken in Tab 8.")

        if already_saved:
            st.success("✅ This refined model is already saved and is what's shown below.")
        else:
            if st.button("💾 Save this refined model as the official model",
                         type="primary", use_container_width=True, key="tab6_promote_btn"):
                st.session_state.config = refit_config
                st.session_state.model_results = refit_result
                st.session_state.model_fitted = True
                st.success("✅ Saved. Results below now reflect the refined model.")
                st.rerun()
    st.divider()


def _contribution_table(totals, prefix):
    """
    Share % logic:
    1. raw_pct_i = |value_i| / sum(|all values|) * 100
    2. Positives: scale so they sum to 100  =>  raw_pct_i / sum(raw_pct of positives) * 100
    3. Negatives: keep as negative raw_pct  =>  -raw_pct_i
    Result: positive shares sum to +100%, negatives are their true negative weight.
    """
    names  = [c.replace(prefix, "") for c in totals.index]
    values = totals.values.astype(float)

    abs_vals  = np.abs(values)
    total_abs = abs_vals.sum()

    if total_abs < 1e-12:
        shares = np.zeros_like(values)
    else:
        raw_pct     = abs_vals / total_abs * 100
        pos_raw_sum = raw_pct[values > 0].sum()
        shares = np.where(
            values >= 0,
            np.where(pos_raw_sum > 1e-12, raw_pct / pos_raw_sum * 100, 0.0),
            -raw_pct,
        )

    pos_sum = values[values > 0].sum()
    neg_sum = values[values < 0].sum()

    df_out = pd.DataFrame({
        "Channel":       names,
        "Total Contrib": np.round(values, 2),
        "Share (%)":     np.round(shares, 1),
    }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out)+1))
    return df_out, pos_sum, neg_sum


def _render_spend_settings(kp, g, df_full):
    """
    Shared 'Spend Basis' control used by BOTH the Efficiency Index (Section E)
    and ROI Analytics (Section G).

    Why this exists: models are often built on spends that were pre-scaled
    into Lakhs/Crores (or any other unit) before being fed to the optimizer.
    Contributions come out of the model in the *target's* native unit
    regardless, so dividing a native-unit contribution by a Lakhs/Crores
    spend number silently produces a wrong ROI/EI. This lets the user type
    the multiplier that converts the modeled spend unit back to the
    original unit (e.g. modeled in Lakhs -> type 100000; modeled in
    Crores -> type 10000000) and every spend-based number below is
    rescaled consistently.

    Returns:
        rescale_factor (float): multiplier applied to every spend total.
        excluded_media (list[str]): own media channels to LEAVE OUT of the
            spend-proportion pool (e.g. a channel that's really GRPs/organic/
            a non-spend metric, not an actual spend number).
        promo_cols (list[str]): the OWN non-media columns the user has
            flagged as "promotional spend" — these are added to own media
            spend when building the Efficiency Index's spend-proportion
            denominator (Section E). Competitor variables are NEVER
            included, per spec.
    """
    with st.expander("💱 Spend Basis Settings (used by Efficiency Index & ROI below)", expanded=False):
        st.caption(
            "If channel spends were modeled in a scaled unit (Lakhs, Crores, "
            "'000s, etc.), enter the multiplier that converts them back to "
            "the original unit. This is applied to every spend total used "
            "for **EI** (Section E) and **ROI** (Section G) below — it does "
            "**not** change contributions, which already come out of the "
            "model in the target's native unit."
        )
        sc1, sc2 = st.columns([1, 1.4])
        with sc1:
            preset = st.selectbox(
                "Modeled spend unit",
                ["Already in original unit (×1)", "Lakhs (×1,00,000)",
                 "Crores (×1,00,00,000)", "Custom multiplier"],
                key=f"{kp}rescale_preset",
            )
        preset_map = {
            "Already in original unit (×1)": 1.0,
            "Lakhs (×1,00,000)": 100000.0,
            "Crores (×1,00,00,000)": 10000000.0,
        }
        with sc2:
            if preset == "Custom multiplier":
                rescale_factor = st.number_input(
                    "Multiplier (e.g. type 100000 to convert Lakhs → original)",
                    min_value=0.0, value=1.0, step=1.0, format="%.4f",
                    key=f"{kp}rescale_custom",
                )
            else:
                rescale_factor = preset_map[preset]
                st.metric("Multiplier applied", f"×{rescale_factor:,.0f}")
        if rescale_factor is None or rescale_factor <= 0:
            rescale_factor = 1.0

        st.divider()
        media_spend_map = g.get("MEDIA_SPEND_MAP", {})
        media_cols_all = g.get("MEDIA_COLS", [])
        grp_mapped_channels = [c for c in media_cols_all if c in media_spend_map]

        st.caption(
            "**Spend-proportion pool for EI:** own media channels are included "
            "by default. Untick any that are NOT a real spend number and have "
            "**no mapped spend column** (e.g. raw GRPs/impressions with nothing "
            "configured) — they'll be dropped from the spend pool entirely "
            "(no EI/ROI for them), and won't distort other channels' spend "
            "share either. Channels configured in **Tab 5 · Configuration** as "
            "GRP / Impressions mapped to a spend column are always kept in the "
            "pool automatically (below) — that mapped column's total is used "
            "for them instead of the raw GRP/impression numbers, so unticking "
            "them here isn't needed and isn't possible."
        )
        if grp_mapped_channels:
            mapped_list = ", ".join(f"**{c}** → `{media_spend_map[c]}`" for c in grp_mapped_channels)
            st.caption(f"📡 GRP/Impressions channels auto-included via their mapped spend column: {mapped_list}")

        excluded_media = []
        if media_cols_all:
            included_media = safe_multiselect(
                "Own media channels to INCLUDE as spend in the EI pool",
                options=media_cols_all,
                default=list(media_cols_all),
                require=grp_mapped_channels,
                key=f"{kp}media_spend_include",
            )
            excluded_media = [c for c in media_cols_all if c not in included_media]

        own_nonmedia = g.get("OWN_NONMEDIA_COLS", [])
        promo_cols = []
        if own_nonmedia:
            promo_cols = st.multiselect(
                "Which own non-media variable(s) are **promotional spend** "
                "(to include alongside own media spend in the Efficiency "
                "Index's spend base)? Competitor variables are excluded "
                "automatically.",
                own_nonmedia, default=[], key=f"{kp}promo_cols",
            )
        else:
            st.caption("No own non-media variables configured — EI will be based on own media spend only.")
    return rescale_factor, excluded_media, promo_cols


def _roi_value_adj_factor(channel, price_factor, roi_adj):
    """Combine the Tab-2 price-conversion factor with this channel's own
    multiply/divide numbers (from the ROI Value Conversion panel) into a
    single multiplier applied to Beta x Input before it's divided by
    spend for ROI. Missing/zero divide falls back to 1 (no-op)."""
    a = (roi_adj or {}).get(channel, {})
    mult = a.get("multiply", 1.0)
    div  = a.get("divide", 1.0)
    mult = mult if mult not in (None, 0) else 1.0
    div  = div if div not in (None, 0) else 1.0
    return float(price_factor) * float(mult) / float(div)


def _render_roi_value_conversion_panel(kp, g):
    """
    Side panel (used by BOTH the Short-Term table's ROI column and the
    Section G ROI Analytics table): lets the user convert each media
    channel's Beta x Input into VALUE before it's divided by spend.

        ROI_channel = (BetaInput_total * PriceConversionFactor
                        * multiply_channel / divide_channel) / Spend_channel

    PriceConversionFactor comes from Tab 2 · Prophet Decomposition's
    "Sales Modeling Basis" section:
      - Sales Value modeled  -> factor = 1 (contributions are already value,
        ROI is Beta x Input / spend directly, unchanged from before)
      - Sales Volume modeled -> factor = Average Price (x1000 if the
        volume was flagged as Tonnes there, since price is assumed per Kg)

    The multiply/divide numbers below are optional extras on top of that
    (e.g. a channel-specific unit conversion) — they default to 1 (no-op).
    """
    basis = st.session_state.get("sales_modeling_basis", "value")
    price_factor = float(st.session_state.get("price_conversion_factor", 1.0) or 1.0)

    with st.expander("🎛️ ROI Value Conversion (Beta × Input → Value, used for ROI)",
                      expanded=False):
        if basis == "volume":
            unit_note = (" (× 1000 since Tab 2's volume unit = Tonnes, price assumed per Kg)"
                         if st.session_state.get("sales_volume_unit") == "Tonnes" else "")
            st.caption(
                f"📦 **Tab 2** says you're modeling **Sales Volume**. Average price of "
                f"**{st.session_state.get('sales_price_col') or '—'}** = "
                f"**{st.session_state.get('sales_avg_price', 0):,.4f}**{unit_note} "
                f"→ price conversion factor = **{price_factor:,.4f}**. This is applied "
                f"to every media channel's Beta × Input below before it's divided by "
                f"spend for ROI."
            )
        else:
            st.caption(
                "💵 **Tab 2** says you're modeling **Sales Value** (or hasn't been set) "
                "— no price conversion is applied (factor = 1). Change this in "
                "**Tab 2 · Sales Modeling Basis** if that's wrong."
            )
        st.caption(
            "Optionally also multiply and/or divide each media channel's "
            "Beta × Input by an extra number before it's used for ROI "
            "(e.g. a channel-specific unit conversion). Leave both at 1.0 "
            "for no extra adjustment."
        )
        media_cols = g.get("MEDIA_COLS", [])
        roi_adj = {}
        if media_cols:
            hc1, hc2, hc3 = st.columns([2, 1, 1])
            hc1.caption("Channel"); hc2.caption("× multiply"); hc3.caption("÷ divide")
            for col in media_cols:
                rc1, rc2, rc3 = st.columns([2, 1, 1])
                with rc1:
                    st.markdown(f"**{col}**")
                with rc2:
                    mult = st.number_input(
                        "multiply", min_value=0.0, value=1.0, step=0.1,
                        format="%.6g", key=f"{kp}roi_mult_{col}",
                        label_visibility="collapsed")
                with rc3:
                    div = st.number_input(
                        "divide", min_value=0.0, value=1.0, step=0.1,
                        format="%.6g", key=f"{kp}roi_div_{col}",
                        label_visibility="collapsed")
                roi_adj[col] = {"multiply": mult, "divide": div}
        else:
            st.caption("No media channels configured.")
    return price_factor, roi_adj


def _add_efficiency_index(df_st, g, df_full, rescale_factor, excluded_media, promo_cols,
                           price_factor=1.0, roi_adj=None):
    """
    Adds EI (Efficiency Index) and ROI columns to the Short-Term
    Contribution Summary table — computed ONLY for "own" spend-bearing
    variables: own media channels (g['MEDIA_COLS'], minus any the user
    excluded as non-spend) plus whichever own non-media columns were
    flagged as promotional spend. Competitor media/non-media, price, the
    intercept, and any excluded media are left blank since they're not
    part of the spend pool.

        EI  = Contribution Share (%) / Spend Share (%)
              where Spend Share (%) is that variable's share of the
              OWN-spend pool only (own media (minus exclusions) + flagged
              promo spend) — never competitor spend.
        ROI = (Total Contrib * ValueAdjFactor) / Rescaled Spend
              where ValueAdjFactor = PriceConversionFactor (from Tab 2,
              1.0 if modeling Sales Value) * that channel's multiply /
              divide from the ROI Value Conversion panel (1.0 by default,
              media channels only — non-media promo columns get the price
              factor but no multiply/divide adjustment).
    """
    media_cols = [c for c in g.get("MEDIA_COLS", []) if c not in (excluded_media or [])]
    media_spend_map = g.get("MEDIA_SPEND_MAP", {})
    own_vars = media_cols + [c for c in promo_cols if c not in media_cols]

    rescaled_spend = {}
    for col in own_vars:
        if col in media_cols:
            spend_col = media_spend_map.get(col, col)
            if spend_col not in df_full.columns:
                spend_col = col
        else:
            spend_col = col
        raw = float(df_full[spend_col].sum()) if spend_col in df_full.columns else 0.0
        rescaled_spend[col] = raw * rescale_factor

    total_own_spend = sum(rescaled_spend.values())

    ei_vals, roi_vals, spend_share_vals = [], [], []
    for _, row in df_st.iterrows():
        ch = row["Channel"]
        if ch in own_vars and total_own_spend > 1e-12:
            sp = rescaled_spend[ch]
            spend_share = sp / total_own_spend * 100
            ei = row["Share (%)"] / spend_share if spend_share > 1e-9 else np.nan
            value_adj = _roi_value_adj_factor(ch, price_factor, roi_adj) if ch in media_cols \
                else float(price_factor)
            roi = (row["Total Contrib"] * value_adj) / sp if sp > 1e-9 else np.nan
            spend_share_vals.append(round(spend_share, 2))
            ei_vals.append(round(ei, 3) if pd.notna(ei) else np.nan)
            roi_vals.append(round(roi, 4) if pd.notna(roi) else np.nan)
        else:
            spend_share_vals.append(np.nan)
            ei_vals.append(np.nan)
            roi_vals.append(np.nan)

    df_out = df_st.copy()
    df_out["Spend Share (%)"] = spend_share_vals
    df_out["EI"] = ei_vals
    df_out["ROI"] = roi_vals
    return df_out, rescaled_spend, total_own_spend


def _make_response_curve_fig(sel, idx, df, res, g, params, x_max_pct=150,
                              n_points=200, show_ci=True):
    """Builds the Section H response-curve Plotly figure for one channel.
    Shared by the interactive selector AND the batch PNG export used by
    the 'Download All Results (ZIP)' button, so exported images are
    identical to what's shown on screen."""
    transform_type = g.get("TRANSFORM_TYPE", "hill")

    col_data = df[sel]; x_max = col_data.max() * x_max_pct / 100
    x_range  = np.linspace(0, x_max, n_points)
    beta_med = float(np.median(res["x_smooth"][:, idx+1]))
    beta_p25 = float(np.percentile(res["x_smooth"][:, idx+1], 25))
    beta_p75 = float(np.percentile(res["x_smooth"][:, idx+1], 75))
    n_v = params["n_params"][idx]; S_v = params["S_params"][idx]

    def make_resp(b):
        return np.array([b * apply_transformation(
            np.array([x]), transform_type, n_v, S_v)[0] for x in x_range])

    resp_med = make_resp(beta_med)
    resp_p25 = make_resp(beta_p25)
    resp_p75 = make_resp(beta_p75)

    pct_10 = np.percentile(col_data[col_data > 0], 10) if (col_data > 0).any() else 0
    pct_50 = np.percentile(col_data[col_data > 0], 50) if (col_data > 0).any() else 0
    pct_90 = np.percentile(col_data[col_data > 0], 90) if (col_data > 0).any() else 0
    mean_v = col_data.mean()
    def resp_at(v): return float(np.interp(v, x_range, resp_med))

    fig_rc = go.Figure()

    if show_ci:
        fig_rc.add_trace(go.Scatter(
            x=np.concatenate([x_range, x_range[::-1]]),
            y=np.concatenate([resp_p75, resp_p25[::-1]]),
            fill="toself", fillcolor="rgba(99,179,237,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))

    fig_rc.add_trace(go.Scatter(
        x=x_range, y=resp_med, mode="lines",
        line=dict(color="#3b82f6", width=3),
        showlegend=False,
    ))

    marker_specs = [
        (pct_10, "Min",    "#10b981"),
        (mean_v, "Avg",    "#f59e0b"),
        (pct_50, "Median", "#a855f7"),
        (pct_90, "Max",    "#ef4444"),
    ]
    for pv, pn, clr in marker_specs:
        if 0 < pv <= x_max:
            rv = resp_at(pv)
            fig_rc.add_trace(go.Scatter(
                x=[pv], y=[rv], mode="markers+text",
                marker=dict(color=clr, size=13,
                            line=dict(color="white", width=1.5)),
                text=[f"<b>{pn}<br>{pv:.1f}</b>"],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                showlegend=False,
            ))

    fig_rc.update_layout(
        title=dict(
            text=f"Response Curve: {sel}",
            font=dict(color="white", size=14), x=0.5,
        ),
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1"),
        xaxis=dict(
            title=f"Input - {sel}",
            gridcolor="#334155", zerolinecolor="#334155",
        ),
        yaxis=dict(
            title="Response (KPI per unit Input)",
            gridcolor="#334155", zerolinecolor="#334155",
        ),
        height=460,
        margin=dict(t=60, b=60, l=70, r=30),
    )
    return fig_rc, beta_med, beta_p25, beta_p75, n_v, S_v


def _render_stuck_at_init_flag(res):
    """
    Diagnostic-only: flags parameters whose fitted value barely moved from
    theta0 (the optimizer's starting guess), in NORMALIZED units — i.e.
    less than 0.1% of that parameter's own natural range (see
    modules/bounds.py::build_normalized_problem), not raw/absolute units,
    so a huge-range parameter like S (half-saturation, ~O(1e6)) and a
    tiny-range one like Hill's n (~O(1-15)) are compared on an equal,
    proportionate footing.

    This never fixes anything by itself — it's a fast, automatic replacement
    for eyeballing a parameter-table CSV. A parameter landing here CAN be a
    genuine optimum (nothing wrong with that), but it can also mean the
    loglik surface was flat/unidentified around theta0 in that direction and
    the optimizer had no gradient signal to leave it — increasing Tab 5's
    "Restarts (multi-start)" and re-running tells the two apart.
    """
    theta0_norm       = res.get("theta0_norm")
    theta_fitted_norm = res.get("theta_fitted_norm")
    labels             = res.get("theta_labels")
    if theta0_norm is None or theta_fitted_norm is None or labels is None:
        return  # older / pre-diagnostic result (e.g. a saved session) — nothing to show

    theta0_norm       = np.asarray(theta0_norm, dtype=float)
    theta_fitted_norm = np.asarray(theta_fitted_norm, dtype=float)
    move = np.abs(theta_fitted_norm - theta0_norm)

    STUCK_THRESHOLD = 0.001  # < 0.1% of that parameter's own normalized range
    stuck_idx = [i for i in np.where(move < STUCK_THRESHOLD)[0] if i < len(labels)]
    if not stuck_idx:
        return

    n_restarts = res.get("n_restarts", 1)
    stuck_labels = [labels[i] for i in stuck_idx]
    st.warning(
        f"⚠️ **{len(stuck_labels)} parameter(s) moved <0.1% from their initial "
        f"guess** — may indicate a flat/unidentified region of the log-"
        f"likelihood surface rather than a genuine optimum."
    )
    with st.expander(f"Which parameters ({len(stuck_labels)})"):
        st.write(", ".join(stuck_labels))
        if n_restarts <= 1:
            st.caption(
                "This fit used a single optimizer start (Restarts = 1). Raise "
                "**Restarts (multi-start)** on Tab 5 and re-run — if a "
                "randomized restart moves these parameters to a better "
                "log-likelihood, theta0 was just an unlucky start; if every "
                "restart lands back here, it's genuinely flat."
            )
        else:
            st.caption(
                f"This fit already used {n_restarts} multi-start restarts and "
                "still landed here — a stronger sign these parameters are "
                "genuinely in a flat/unidentified region, not just an unlucky "
                "single start."
            )


def render_full_results(df, config, res, target, key_prefix="", pcb_key="per_channel_bounds"):
    """The full Results & ROI Analytics body (sections A-I). Called by
    Tab 7 for the officially-saved model, and by Tab 8 after every refit
    so refits get identical charts, tables and download buttons without
    having to open Tab 7.

    pcb_key: which config key holds this result's per-channel bounds —
    "per_channel_bounds" for Dependent 1 (the default, and the only one
    Tab 8 refits), "per_channel_bounds_2" when Tab 7 is showing Dependent 2.
    """
    g = res["g"]
    kp = key_prefix

    st.markdown("### A · Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE",         f"{res['mape']:.2%}")
    c2.metric("Gelman R²",    f"{res['r2_gelman']:.4f}")
    c3.metric("Log-Lik",      f"{res['loglik']:.2f}")
    c4.metric("Observations", len(df))
    with st.expander("📊 R² metrics"):
        st.metric("R²", f"{res['r2']:.4f}")

    _render_stuck_at_init_flag(res)

    n_test = res.get("n_test", 0)
    if n_test and n_test > 0:
        st.markdown("###### In-Sample (Train) vs. Out-of-Sample (Test) Accuracy")
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Train MAPE",       f"{res['mape_in']:.2%}")
        ic2.metric("Test MAPE",        f"{res['mape_out']:.2%}",
                   delta=f"{(res['mape_out']-res['mape_in'])*100:+.2f} pp vs train",
                   delta_color="inverse")
        ic3.metric("Train / Test obs", f"{res['n_train']} / {n_test}")
        jc1, jc2 = st.columns(2)
        jc1.metric("Train Gelman R²",  f"{res['r2_gelman_in']:.4f}")
        jc2.metric("Test Gelman R²",   f"{res['r2_gelman_out']:.4f}",
                   delta=f"{res['r2_gelman_out']-res['r2_gelman_in']:+.4f} vs train")
        with st.expander("📊 R² metrics (train/test)"):
            kc1, kc2 = st.columns(2)
            kc1.metric("Train R²", f"{res['r2_in']:.4f}")
            kc2.metric("Test R²",  f"{res['r2_out']:.4f}",
                       delta=f"{res['r2_out']-res['r2_in']:+.4f} vs train")
        st.caption("A much larger gap between train and test metrics is a sign of overfitting.")
    else:
        st.caption(
            "No holdout/test period configured — the metrics above are in-sample "
            "(fit) only. Set a train/test split in Tab 5 to see out-of-sample accuracy."
        )

    st.markdown("### B · Actual vs Predicted")
    x_axis  = np.arange(len(df)); n_train = config["n_train"]
    fig_avp = go.Figure()
    fig_avp.add_vrect(x0=n_train, x1=len(df)-1, fillcolor="#fef3c7", opacity=0.35,
                      layer="below", line_width=0,
                      annotation_text="Test period", annotation_position="top left")
    fig_avp.add_trace(go.Scatter(x=x_axis, y=df[target].values,
                                  name="Actual", line=dict(color="#1e40af", width=2)))
    fig_avp.add_trace(go.Scatter(x=x_axis, y=res["yhat_smooth"],
                                  name="RBE Smoothed", line=dict(color="#f59e0b", width=2, dash="dash")))
    fig_avp.update_layout(height=460, template="plotly_white",
                           title="Actual vs RBE Smoothed",
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_avp, use_container_width=True, key=f"{kp}fig_avp")

    st.markdown("### C · Smoothed Intercept State")
    fig_int = go.Figure()
    fig_int.add_trace(go.Scatter(x=x_axis, y=res["x_smooth"][:, 0],
                                  name="Intercept", line=dict(color="#10b981", width=2)))
    fig_int.update_layout(height=340, template="plotly_white",
                           title="Smoothed Intercept (Base Demand) Over Time")
    st.plotly_chart(fig_int, use_container_width=True, key=f"{kp}fig_int")

    st.markdown("### D · Residual Diagnostics")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_h = px.histogram(res["residuals"], nbins=40, title="Residual Distribution",
                              color_discrete_sequence=["#3b82f6"])
        fig_h.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_h, use_container_width=True, key=f"{kp}fig_hist")
    with col_b:
        sr = np.sort(res["residuals"])
        tq = np.sort(np.random.normal(0, np.std(res["residuals"]), len(sr)))
        fig_qq = px.scatter(x=tq, y=sr, title="Q-Q Plot",
                             labels={"x": "Theoretical Normal", "y": "Sample Residuals"})
        fig_qq.add_trace(go.Scatter(x=tq, y=tq, mode="lines",
                                     name="Ideal", line=dict(color="red", dash="dash")))
        fig_qq.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_qq, use_container_width=True, key=f"{kp}fig_qq")

    st.markdown("### E · Channel Contributions")
    contrib_df = res["contrib_df"]
    short_cols = [c for c in contrib_df.columns if c.startswith("ShortTerm_")]
    long_cols  = [c for c in contrib_df.columns if c.startswith("LongTerm_")]
    totals_st  = contrib_df[short_cols].sum()
    totals_lt  = contrib_df[long_cols].sum()

    rescale_factor, excluded_media, promo_cols = _render_spend_settings(kp, g, df)
    price_factor, roi_adj = _render_roi_value_conversion_panel(kp, g)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Short-Term Contribution Summary")
        df_st, pos_st, neg_st = _contribution_table(totals_st, "ShortTerm_")
        df_st_ei, rescaled_spend, total_own_spend = _add_efficiency_index(
            df_st, g, df, rescale_factor, excluded_media, promo_cols,
            price_factor=price_factor, roi_adj=roi_adj)
        st.dataframe(
            df_st_ei.style.format({
                "Total Contrib":   "{:,.2f}",
                "Share (%)":       "{:.1f}",
                "Spend Share (%)": "{:.2f}",
                "EI":              "{:.3f}",
                "ROI":             "{:.4f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True, key=f"{kp}df_st",
        )
        st.caption(
            "**EI (Efficiency Index)** = Contribution Share (%) ÷ Spend Share (%), "
            "computed only for **own** variables (own media + any flagged "
            "promotional spend) — EI > 1 means a variable is punching above its "
            "spend weight, EI < 1 means below. Competitor/price/intercept rows "
            "show **—** since they have no spend to divide by. "
            f"Own-spend pool used: **{total_own_spend:,.2f}** "
            f"(rescale ×{rescale_factor:,.0f}). "
            f"**ROI** = (Total Contrib × Value Conversion Factor) ÷ Spend — see "
            "**🎛️ ROI Value Conversion** panel above "
            f"(current price factor from Tab 2: ×{price_factor:,.4f})."
        )
        check_st = pos_st + neg_st
        c1s, c2s, c3s = st.columns(3)
        c1s.metric("Positive pool", f"{pos_st:,.2f}")
        c2s.metric("Negative pool", f"{neg_st:,.2f}")
        c3s.metric("Net", f"{check_st:,.2f}")
        st.caption("✅ Positive shares sum to **+100 %** · Negative shares sum to **−100 %**")
    with t2:
        st.markdown("#### Long-Term Contribution Summary")
        df_lt, pos_lt, neg_lt = _contribution_table(totals_lt, "LongTerm_")
        st.dataframe(df_lt, use_container_width=True, hide_index=True, key=f"{kp}df_lt")
        check_lt = pos_lt + neg_lt
        c1l, c2l, c3l = st.columns(3)
        c1l.metric("Positive pool", f"{pos_lt:,.2f}")
        c2l.metric("Negative pool", f"{neg_lt:,.2f}")
        c3l.metric("Net", f"{check_lt:,.2f}")
        st.caption("✅ Positive shares sum to **+100 %** · Negative shares sum to **−100 %**")

    synergy_cols = [c for c in contrib_df.columns if c.startswith("Synergy_")]
    if synergy_cols:
        st.markdown("#### 🔗 Cross-Media Synergy Contribution Summary")
        totals_syn = contrib_df[synergy_cols].sum()
        syn_vals = totals_syn.values
        syn_abs_vals  = np.abs(syn_vals)
        syn_total_abs = syn_abs_vals.sum()
        if syn_total_abs < 1e-12:
            syn_shares = np.zeros_like(syn_vals)
        else:
            syn_raw_pct     = syn_abs_vals / syn_total_abs * 100
            syn_pos_raw_sum = syn_raw_pct[syn_vals > 0].sum()
            syn_shares = np.where(
                syn_vals >= 0,
                np.where(syn_pos_raw_sum > 1e-12, syn_raw_pct / syn_pos_raw_sum * 100, 0.0),
                -syn_raw_pct,
            )
        df_syn = pd.DataFrame({
            "Synergy Pair": [c.replace("Synergy_", "").replace("_from_", " <- from ")
                              for c in synergy_cols],
            "Total Contrib": syn_vals.round(2),
            "Share (%)": np.round(syn_shares, 1),
        }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
        df_syn.insert(0, "Rank", range(1, len(df_syn)+1))
        st.dataframe(df_syn, use_container_width=True, hide_index=True, key=f"{kp}df_syn_summary")

        fig_syn = px.bar(
            df_syn.sort_values("Total Contrib"),
            x="Total Contrib", y="Synergy Pair", orientation="h",
            title="Cross-Media Synergy Contribution by Pair",
            color="Total Contrib", color_continuous_scale="Purples",
        )
        fig_syn.update_layout(height=max(300, 45*len(synergy_cols)), template="plotly_white")
        st.plotly_chart(fig_syn, use_container_width=True, key=f"{kp}fig_syn_summary")
    else:
        st.caption("No cross-media synergy pairs were configured in Tab 5 · Section C.")

    with st.expander("📊 Pie Charts — Positive Contributions Only", expanded=False):
        st.caption(
            "Pie charts show only **positive** contributions (share of positive pool = 100%). "
            "Negative channels (competitor, price) are excluded from pies — they are shown "
            "in the summary tables above with negative share %."
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            st_names  = [c.replace("ShortTerm_", "") for c in short_cols]
            st_vals   = totals_st.values
            pos_mask  = st_vals > 0
            if pos_mask.any():
                fig_pie_st = px.pie(
                    values=st_vals[pos_mask],
                    names=[n for n, m in zip(st_names, pos_mask) if m],
                    title="Short-term positive share",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                st.plotly_chart(fig_pie_st, use_container_width=True, key=f"{kp}fig_pie_st")
            else:
                st.info("No positive short-term contributions.")
        with pc2:
            lt_names  = [c.replace("LongTerm_", "") for c in long_cols]
            lt_vals   = totals_lt.values
            pos_mask_lt = lt_vals > 0
            if pos_mask_lt.any():
                fig_pie_lt = px.pie(
                    values=lt_vals[pos_mask_lt],
                    names=[n for n, m in zip(lt_names, pos_mask_lt) if m],
                    title="Long-term positive share",
                    color_discrete_sequence=px.colors.sequential.Greens_r,
                )
                st.plotly_chart(fig_pie_lt, use_container_width=True, key=f"{kp}fig_pie_lt")
            else:
                st.info("No positive long-term contributions.")

    _cplot = contrib_df.copy().reset_index(drop=True)
    _cplot["_period"] = np.arange(len(_cplot))
    fig_st_area = px.area(
        _cplot, x="_period", y=short_cols,
        title="Short-term Contributions Over Time",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_st_area.update_xaxes(title_text="Period")
    fig_st_area.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_st_area, use_container_width=True, key=f"{kp}fig_st_area")

    with st.expander("Full contribution table", expanded=False):
        st.dataframe(contrib_df.style.format("{:,.4f}"), use_container_width=True, key=f"{kp}df_contrib_full")
    st.download_button("📥 Download Contribution Table",
                       contrib_df.to_csv().encode(), "contributions.csv", "text/csv",
                       key=f"{kp}dl_contrib")

    st.markdown("### E2 · Synergy / Cross-Media Effects")
    synergy_df = res.get("synergy_df")
    synergy_cols = [c for c in contrib_df.columns if c.startswith("Synergy_")]
    if synergy_df is None or synergy_df.empty or not synergy_cols:
        st.info(
            "No cross-media synergy relationships are configured for this model. "
            "Set them up in **Tab 5 · Section C (Cross-media Learning)** and re-run."
        )
    else:
        st.caption(
            "One row per configured source → target relationship "
            "(per-variable level, not a single aggregated number)."
        )
        st.dataframe(
            synergy_df.style.format({
                "Total Synergy Contribution": "{:,.4f}",
                "Avg Synergy / Period":       "{:,.4f}",
                "Cross Delta":                "{:.4f}",
                "Cross Hill n":                "{:.3f}",
                "Cross Hill S":                "{:.3f}",
                "Share of Target's Total Contrib (%)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True, key=f"{kp}df_syn_table",
        )
        st.download_button("📥 Download Synergy Table",
                           synergy_df.to_csv(index=False).encode(),
                           "synergy_table.csv", "text/csv", key=f"{kp}dl_synergy")

        pair_labels = [f"{row['Source Channel']} → {row['Target Channel']}"
                        for _, row in synergy_df.iterrows()]
        fig_syn_bar = px.bar(
            x=synergy_df["Total Synergy Contribution"].values, y=pair_labels,
            orientation="h", title="Total Synergy Contribution by Pair",
            labels={"x": "Total Contribution", "y": ""},
            color=synergy_df["Total Synergy Contribution"].values,
            color_continuous_scale="Purples",
        )
        fig_syn_bar.update_layout(height=max(300, 50*len(pair_labels)), template="plotly_white")
        st.plotly_chart(fig_syn_bar, use_container_width=True, key=f"{kp}fig_syn_bar")

        _cplot_syn = contrib_df.copy().reset_index(drop=True)
        _cplot_syn["_period"] = np.arange(len(_cplot_syn))
        fig_syn_area = px.area(
            _cplot_syn, x="_period", y=synergy_cols,
            title="Synergy Contributions Over Time (per pair)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_syn_area.update_xaxes(title_text="Period")
        fig_syn_area.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig_syn_area, use_container_width=True, key=f"{kp}fig_syn_area")

    st.markdown("### F · Estimated Parameters")
    param_df = res["param_df"]
    st.dataframe(param_df.style.format({"Value": "{:.6f}"}),
                 use_container_width=True, hide_index=True, key=f"{kp}df_params")
    st.download_button("📥 Download Parameters",
                       param_df.to_csv(index=False).encode(), "parameters.csv", "text/csv",
                       key=f"{kp}dl_params")

    if g.get("INTERCEPT_DYNAMICS_TYPE", "carryover") == "weibull":
        _iw_params = res["params"]
        _iw_L = int(g.get("INTERCEPT_WEIBULL_N_LAGS", 4))
        _iw_k = float(_iw_params.get("intercept_weibull_shape", 1.5))
        _iw_lam = float(_iw_params.get("intercept_weibull_scale", 1.0))
        _iw_w = weibull_lag_weights(_iw_k, _iw_lam, _iw_L)
        _iw_df = pd.DataFrame({
            "Lag": [f"t-{l}" for l in range(1, _iw_L + 1)],
            "Weight": _iw_w,
        })
        fig_iw = px.bar(_iw_df, x="Lag", y="Weight",
                         title=f"Intercept Weibull carryover weights (k={_iw_k:.3f}, λ={_iw_lam:.3f}) — "
                               f"sum = {_iw_w.sum():.4f}")
        fig_iw.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_iw, use_container_width=True, key=f"{kp}fig_iw_weights")

    st.markdown("### G · ROI Analytics")
    roi_df = res["roi_df"]
    if not roi_df.empty:
        roi_df = roi_df.copy()
        roi_df["TotalSpend"] = roi_df["TotalSpend"] * rescale_factor
        roi_df["ValueAdjFactor"] = roi_df["Channel"].apply(
            lambda ch: _roi_value_adj_factor(ch, price_factor, roi_adj))
        roi_df["AdjTotalContrib"] = roi_df["TotalContrib"] * roi_df["ValueAdjFactor"]
        roi_df["ROI"] = np.where(
            roi_df["TotalSpend"] > 1e-9,
            roi_df["AdjTotalContrib"] / roi_df["TotalSpend"], 0.0,
        )
        if rescale_factor != 1:
            st.caption(
                f"💱 Spend rescale of **×{rescale_factor:,.0f}** from the "
                f"**Spend Basis Settings** panel above has been applied to "
                f"TotalSpend and ROI below."
            )
        if (price_factor != 1.0
                or any(a.get("multiply", 1.0) != 1.0 or a.get("divide", 1.0) != 1.0
                       for a in (roi_adj or {}).values())):
            st.caption(
                "🧮 **ROI Value Conversion** applied — TotalContrib × "
                "**ValueAdjFactor** (price factor from Tab 2 × any per-channel "
                "multiply/divide from the panel above) = **AdjTotalContrib**, "
                "which is what ROI below is computed from."
            )
        n_grp = int((roi_df.get("InputType", pd.Series(dtype=str)) == "GRP/Impressions").sum())
        if n_grp:
            st.caption(
                f"💡 {n_grp} channel(s) are configured as **GRP / Impressions** in "
                f"Tab 5 · D2 (or Tab 8) — their ROI uses their mapped **spend column's** "
                f"total instead of summing the channel itself. See the **SpendColumn** "
                f"column below."
            )
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_roi = px.bar(roi_df.sort_values("ROI", ascending=True),
                              x="ROI", y="Channel", orientation="h",
                              title="ROI by Media Channel",
                              color="ROI", color_continuous_scale="RdYlGn")
            fig_roi.add_vline(x=1.0, line_dash="dash", line_color="black",
                               annotation_text="Break-even")
            fig_roi.update_layout(height=360, template="plotly_white")
            st.plotly_chart(fig_roi, use_container_width=True, key=f"{kp}fig_roi")
        with c2:
            st.dataframe(roi_df.style.format({
                "TotalSpend":      "{:,.2f}",
                "TotalContrib":    "{:,.2f}",
                "ValueAdjFactor":  "{:,.4f}",
                "AdjTotalContrib": "{:,.2f}",
                "ROI":             "{:.4f}",
            }), use_container_width=True, hide_index=True, key=f"{kp}df_roi")
        best  = roi_df.loc[roi_df["ROI"].idxmax()]
        worst = roi_df.loc[roi_df["ROI"].idxmin()]
        st.success(f"🏆 Best ROI: **{best['Channel']}** ({best['ROI']:.4f})")
        st.warning(f"⚠️ Lowest ROI: **{worst['Channel']}** ({worst['ROI']:.4f})")
        st.download_button("📥 Download ROI Report",
                           roi_df.to_csv(index=False).encode(), "roi_report.csv", "text/csv",
                           key=f"{kp}dl_roi")

    st.markdown("### H · Response Curves")
    params       = res["params"]
    media_list   = g["MEDIA_COLS"]
    adstock_map  = g.get("ADSTOCK_MAP", {})
    adstock_idx  = g.get("ADSTOCK_IDX", {})

    if not media_list:
        st.info("No media channels configured.")
    else:
        sel = st.selectbox("Select channel", media_list, key=f"{kp}rc_sel")
        idx = media_list.index(sel)

        rc1, rc2, rc3 = st.columns(3)
        with rc1: x_max_pct = st.slider("X-axis max (%)", 50, 300, 150, 10, key=f"{kp}rc_xmax")
        with rc2: n_points  = st.slider("Curve resolution", 50, 500, 200, 50, key=f"{kp}rc_npts")
        with rc3: show_ci   = st.checkbox("Show beta band", value=True, key=f"{kp}rc_ci")

        fig_rc, beta_med, beta_p25, beta_p75, n_v, S_v = _make_response_curve_fig(
            sel, idx, df, res, g, params, x_max_pct, n_points, show_ci)
        st.plotly_chart(fig_rc, use_container_width=True, key=f"{kp}fig_rc")

        # Download this channel's response curve as a standalone PNG.
        try:
            rc_png_bytes = fig_rc.to_image(format="png", scale=2)
            st.download_button(
                f"🖼️ Download Response Curve Image ({sel})",
                rc_png_bytes, f"response_curve_{sel}.png", "image/png",
                key=f"{kp}dl_rc_png",
            )
        except Exception as e:
            st.caption(
                f"⚠️ Could not render a PNG of this curve ({e}). "
                "Make sure the `kaleido` package is installed."
            )

        # Parameter summary
        pc = st.columns(3)
        transform_type_label = g.get("TRANSFORM_TYPE", "hill")
        if transform_type_label == "hill":
            pc[0].metric("Hill n", f"{n_v:.3f}")
            pc[1].metric("Hill S", f"{S_v:.3f}")
        else:
            pc[0].metric("Power n", f"{n_v:.3f}")
            pc[1].metric("", "")
        sel_is_weibull = adstock_map.get(sel) == "weibull"
        if sel_is_weibull:
            ai = adstock_idx.get(sel)
            pc[2].metric("Weibull k/lam",
                         f"{params['adstock_shape'][ai]:.3f}/{params['adstock_scale'][ai]:.3f}")
        else:
            pc[2].metric("Ls (persistence)", f"{params['Ls'][idx]:.3f}")
        if sel_is_weibull:
            ai = adstock_idx.get(sel)
            ak = params["adstock_shape"][ai]; al = params["adstock_scale"][ai]
            n_lags = int(params.get("adstock_n_lags", 8))
            adstock_label = f"Weibull (k={ak:.3f}, λ={al:.3f}, L={n_lags})"
        else:
            ls_v = params["Ls"][idx]
            adstock_label = f"Instant / Nerlove-Arrow (no separate λ — carryover via Ls={ls_v:.3f})"
        st.caption(f"Adstock: **{adstock_label}**")

        pcb = config.get(pcb_key, {}).get(sel, {})
        pcb_shown = {k: v for k, v in pcb.items() if not k.startswith("__")}
        if pcb_shown:
            st.markdown("#### Per-Variable Bounds Applied")
            st.dataframe(
                pd.DataFrame([{"Parameter": k,
                               "Min": f"{v[0]:.4g}",
                               "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
                              for k, v in pcb_shown.items()]),
                use_container_width=True, hide_index=True, key=f"{kp}df_pcb_applied",
            )
        if pcb.get("__spend_col__"):
            st.caption(f"💰 ROI for **{sel}** uses **{pcb['__spend_col__']}**'s total spend "
                       f"(configured as GRP / Impressions).")

    st.divider()
    st.markdown("### I · Export Center")
    info_text = (
        "Four ways to take results outside the app: just the betas, just the "
        "hyperparameters, one structured Excel workbook with everything laid "
        "out as <b>Raw variables → Transformed variables → Betas → "
        "Contributions (Beta × Transformed)</b> plus hyperparameters / ROI / "
        "synergy on their own sheets, or <b>everything at once</b> as a single ZIP."
    )
    st.markdown(f'<div class="info-box">{info_text}</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        betas_df = build_betas_df(res, df)
        st.download_button(
            "📥 Download Betas (Time Series)",
            betas_df.to_csv(index=False).encode(),
            "betas_timeseries.csv", "text/csv",
            use_container_width=True, key=f"{kp}dl_betas",
        )
    with ec2:
        st.download_button(
            "📥 Download Hyperparameters",
            param_df.to_csv(index=False).encode(),
            "hyperparameters.csv", "text/csv",
            use_container_width=True, key=f"{kp}dl_hyperparams",
        )
    with ec3:
        export_config = dict(config); export_config["target"] = target
        workbook_bytes = build_master_workbook_bytes(res, export_config, df)
        st.download_button(
            "📊 Download Full Excel Workbook",
            workbook_bytes,
            "rainbrain_model_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key=f"{kp}dl_workbook",
        )

    st.markdown("#### 📦 Download Everything")
    st.caption(
        "One ZIP containing the contribution table, hyperparameters, ROI report, "
        "synergy table (if configured), the betas time series, the full Excel "
        "workbook, and a response-curve PNG for every media channel."
    )
    if st.button("🗂️ Build & Download All Results (ZIP)",
                 use_container_width=True, key=f"{kp}build_zip_btn"):
        with st.spinner("Rendering response curves and packaging the ZIP…"):
            rc_images = {}
            for m_col in media_list:
                m_idx = media_list.index(m_col)
                try:
                    fig_m, *_ = _make_response_curve_fig(m_col, m_idx, df, res, g, params)
                    rc_images[f"response_curve_{m_col}.png"] = fig_m.to_image(format="png", scale=2)
                except Exception:
                    pass  # skip a channel's image rather than failing the whole zip
            export_config = dict(config); export_config["target"] = target
            zip_bytes = build_full_results_zip_bytes(res, export_config, df,
                                                       response_curve_images=rc_images)
        st.download_button(
            "📥 Download all_results.zip",
            zip_bytes, "all_results.zip", "application/zip",
            use_container_width=True, key=f"{kp}dl_zip",
        )


def render_tab6():
    section("06", "Results & ROI Analytics")
    if not st.session_state.model_fitted: need_model()

    config = st.session_state.config
    df     = st.session_state.df

    _render_tab7_promote_section()

    has_dep2 = bool(st.session_state.get("model_fitted_2") and st.session_state.get("model_results_2"))
    pcb_key = "per_channel_bounds"
    if has_dep2:
        is_chained = config.get("dependent_relationship") == "chained"
        dep2_suffix = " (fitted independently, feeds Dependent 1)" if is_chained else " (joint bivariate fit)"
        dep_choice = st.radio(
            "📊 Viewing results for:",
            [f"Dependent 1 · {config['target']}", f"Dependent 2 · {config.get('target2')}{dep2_suffix}"],
            horizontal=True, key="tab6_dep_choice",
        )
        if dep_choice.startswith("Dependent 2"):
            res = st.session_state.model_results_2
            target = config.get("target2")
            pcb_key = "per_channel_bounds_2"
        else:
            res = st.session_state.model_results
            target = config["target"]
        if res.get("joint_fit"):
            _coupling_mode_t6 = res.get("cross_intercept_coupling_mode", "both")
            _coupling_note_t6 = {
                "both": "both directions",
                "dep1_in_dep2": "Dep1→Dep2 only",
                "dep2_in_dep1": "Dep2→Dep1 only",
                "none": "off",
            }.get(_coupling_mode_t6, _coupling_mode_t6)
            st.caption(
                f"🔗 Jointly fitted with a bivariate Kalman filter · "
                f"ρ(Dep1, Dep2) = **{res['rho_y']:.3f}** · "
                f"φ₁ (Dep2→Dep1 intercept) = **{res['phi1']:.3f}** · "
                f"φ₂ (Dep1→Dep2 intercept) = **{res['phi2']:.3f}** "
                f"(coupling: {_coupling_note_t6}) · "
                f"joint log-likelihood = **{res['joint_loglik']:.2f}**"
            )
        elif is_chained:
            driver_col = st.session_state.model_results.get("chain_driver_col")
            if dep_choice.startswith("Dependent 1") and driver_col:
                st.caption(
                    f"➡️ Chained mode — this fit includes `{driver_col}` "
                    f"({'smoothed fitted values' if st.session_state.model_results.get('chain_use_fitted') else 'raw actual values'} "
                    f"of `{config.get('target2')}`) as a "
                    f"**{st.session_state.model_results.get('chain_driver_role','non_media').replace('_',' ')}** "
                    f"predictor — look for it in the contribution/ROI/parameter tables below "
                    f"just like any other channel."
                )
            elif driver_col:
                st.caption(
                    f"➡️ Chained mode — this model was fitted independently, then its fitted "
                    f"values feed into Dependent 1 as `{driver_col}`."
                )
        st.divider()
    else:
        res    = st.session_state.model_results
        target = config["target"]

    render_full_results(df, config, res, target, key_prefix="tab6_", pcb_key=pcb_key)
