"""
Tab 5 — Model Configuration: variable roles, positive-beta constraints,
cross-media learning, adstock type, transformation type, lag count,
per-variable hyperparameter bounds, train/test split, and saving config.
"""

import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_helpers import (
    section, info, positive_info, per_channel_info, prophet_info,
    weibull_placeholder, need_data, safe_multiselect,
)
from modules.bounds_ui import render_per_channel_bounds
from modules.spike_dummies import build_spike_dummy_columns, drop_columns_if_present


def render_tab4():
    section("04", "Model Configuration")
    if st.session_state.df is None: need_data()

    df = st.session_state.df
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if st.session_state.prophet_cols_added:
        missing_now = [c for c in st.session_state.prophet_cols_added if c not in num_cols]
        if missing_now:
            st.error(
                f"⚠️ Prophet column(s) were added but are no longer numeric / present in the "
                f"dataset: `{'`, `'.join(missing_now)}`. Re-run the prophet merge in Tab 2."
            )
        prophet_info(
            f"📌 Prophet columns available: "
            f"<b>{', '.join(st.session_state.prophet_cols_added)}</b>. "
            "They are pre-selected in <b>Non-media / organic</b> below — "
            "just save the configuration to include them in the model."
        )

    # ── A. Variable Selection ─────────────────────────────────────────
    st.markdown("### A · Variable Selection")
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("🎯 Dependent variable (KPI)", num_cols, key="cfg_target")
    with col2:
        remaining = [c for c in num_cols if c != target]
        media = safe_multiselect("📺 Media / paid channels", options=remaining,
                                  default=[], key="cfg_media")

    remaining2 = [c for c in remaining if c not in media]
    col3, col4 = st.columns(2)
    with col3:
        prophet_in_scope = [c for c in st.session_state.prophet_cols_added
                             if c in remaining2]
        non_media = safe_multiselect(
            "🗂️ Non-media / organic (incl. prophet columns)",
            options=remaining2,
            require=prophet_in_scope,
            key="cfg_nonmedia",
        )
    with col4:
        remaining3 = [c for c in remaining2 if c not in non_media]
        price_vars = safe_multiselect("💲 Price variables", options=remaining3,
                                       default=[], key="cfg_price")

    remaining4 = [c for c in remaining3 if c not in price_vars]
    col5, col6 = st.columns(2)
    with col5:
        comp_media = safe_multiselect("📉 Competitor media", options=remaining4,
                                       default=[], key="cfg_comp_media")
    with col6:
        remaining5 = [c for c in remaining4 if c not in comp_media]
        comp_nonmedia = safe_multiselect("📉 Competitor non-media", options=remaining5,
                                          default=[], key="cfg_comp_nonmedia")

    st.divider()

    # ── A2. Second Dependent Variable (optional, multi-dependent MMM) ──
    st.markdown("### A2 · Second Dependent Variable (optional)")
    info(
        "Common in MMM when you want two KPIs explained by the <b>same media "
        "mix</b> — e.g. <b>Sales Volume</b> as Dependent 1 and "
        "<b>Top-of-Mind / Consideration</b> as Dependent 2. Both dependents share "
        "the same regressors x_t (media, non-media, price, competitor) and the "
        "same adstock/transformation family, but each keeps its own betas. "
        "Unlike two separate models, Dependent 1 and Dependent 2 are fitted "
        "<b>jointly</b> with a single bivariate Kalman filter: one optimizer run "
        "estimates both equations' parameters together with the correlation "
        "between their errors, so a surprise in one KPI at time t also informs "
        "the state update of the other KPI at that same t."
    )
    enable_second_dependent = st.checkbox(
        "➕ Enable a second dependent variable (jointly fitted, bivariate Kalman filter)",
        key="cfg_enable_target2",
    )
    target2 = None
    if enable_second_dependent:
        used_cols = {target, *media, *non_media, *price_vars, *comp_media, *comp_nonmedia}
        target2_options = [c for c in num_cols if c not in used_cols]
        if target2_options:
            target2 = st.selectbox(
                "🎯 Dependent variable 2 (KPI) — e.g. Top-of-Mind / Consideration",
                target2_options, key="cfg_target2",
            )
            st.caption(
                f"Dependent 2 will be modeled with the **same predictors** as "
                f"Dependent 1 (`{target}`): {len(media)} media · {len(non_media)} "
                f"non-media · {len(price_vars)} price · {len(comp_media)} comp-media · "
                f"{len(comp_nonmedia)} comp-non-media, and the same adstock/transform "
                f"choices set in Section D below. Dependent 1 and Dependent 2 will be "
                f"fitted **jointly** in Tab 6 with a bivariate Kalman filter "
                f"(shared time index, correlated errors) — not as two separate models."
            )
        else:
            st.warning(
                "No remaining numeric columns are available to use as a second "
                "dependent variable (everything is already used as a predictor)."
            )
            enable_second_dependent = False

    # ── A2b. Automatic Spike / Outlier Dummies (optional) ──────────────
    st.markdown("### A2b · Automatic Spike / Outlier Dummies (optional)")
    info(
        "Auto-flag unusually large, short-lived spikes in the dependent "
        "variable(s) and give each one its own single-period impulse dummy "
        "(1 at that observation, 0 elsewhere), so the model can absorb them "
        "instead of forcing the media betas to explain them. Say what "
        "<b>percentage of observations</b> to flag — e.g. 10% of 100 rows "
        "flags the 10 most unusual observations, one dummy each — rather "
        "than hand-picking individual dates."
    )
    enable_spike_dummies = st.checkbox(
        "➕ Auto-detect spike/outlier dummies", key="cfg_enable_spike_dummies",
    )
    spike_dummy_cols_1 = []
    spike_dummy_cols_2 = []
    if enable_spike_dummies:
        sd1, sd2 = st.columns(2)
        with sd1:
            spike_dummy_pct = st.number_input(
                f"Percentage of `{target}`'s observations to flag as spikes (%)",
                min_value=0.0, max_value=50.0, value=5.0, step=0.5,
                key="cfg_spike_dummy_pct",
                help=(
                    "Rounded to the nearest whole observation — e.g. 10% of "
                    "100 rows = 10 dummies. Detection is relative to a local "
                    "(rolling-median) baseline, not the series' overall mean, "
                    "so a real trend/seasonal swing isn't itself flagged."
                ),
            )
        spike_dummy_pct_2 = 0.0
        apply_spike_dep2 = False
        if enable_second_dependent and target2:
            with sd2:
                apply_spike_dep2 = st.checkbox(
                    f"Also flag spikes in `{target2}` (Dependent 2)",
                    value=True, key="cfg_spike_dummy_apply_dep2",
                )
            if apply_spike_dep2:
                spike_dummy_pct_2 = st.number_input(
                    f"Percentage of `{target2}`'s observations to flag as spikes (%)",
                    min_value=0.0, max_value=50.0, value=5.0, step=0.5,
                    key="cfg_spike_dummy_pct_2",
                )

        # Clear out any dummy columns generated on a PREVIOUS run (different
        # percentage / different target) before regenerating, so re-running
        # detection or tweaking the percentage doesn't leave stale impulse
        # columns accumulating in the working dataset across Streamlit reruns.
        drop_columns_if_present(df, st.session_state.get("_spike_dummy_cols_generated", []))

        _date_col_guess = next(
            (c for c in df.columns if "date" in c.lower() or "week" in c.lower()
             or "period" in c.lower()), None
        )

        new_cols_1, flagged_1 = build_spike_dummy_columns(
            df, target, spike_dummy_pct, date_col=_date_col_guess,
        )
        new_cols_2, flagged_2 = [], []
        if apply_spike_dep2:
            new_cols_2, flagged_2 = build_spike_dummy_columns(
                df, target2, spike_dummy_pct_2, existing_cols=new_cols_1,
                date_col=_date_col_guess,
            )

        spike_dummy_cols_1, spike_dummy_cols_2 = new_cols_1, new_cols_2
        spike_dummy_cols_all = new_cols_1 + new_cols_2
        st.session_state["_spike_dummy_cols_generated"] = spike_dummy_cols_all
        st.session_state.df = df  # persist the newly-added columns

        if spike_dummy_cols_all:
            st.success(
                f"✅ {len(new_cols_1)} spike dummy(s) added for `{target}`"
                + (f" · {len(new_cols_2)} spike dummy(s) added for `{target2}`"
                   if apply_spike_dep2 else "")
                + f" — {len(spike_dummy_cols_all)} total. Each is a single-period "
                "impulse column with its own automatically-fitted beta "
                "(no extra optimizer parameters needed)."
            )
            with st.expander(f"🔍 View {len(spike_dummy_cols_all)} flagged observation(s)"):
                preview_df = pd.DataFrame(flagged_1 + flagged_2)
                if not preview_df.empty:
                    preview_df = preview_df.rename(columns={
                        "row": "Row #", "label": "Date/Label",
                        "value": "Value", "column": "Dummy column",
                    })
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No spikes flagged at the current percentage (0%).")
    else:
        # Nothing enabled this run — make sure any dummy columns left over
        # from a previous run (e.g. user unchecked the box) are removed too.
        drop_columns_if_present(df, st.session_state.get("_spike_dummy_cols_generated", []))
        st.session_state["_spike_dummy_cols_generated"] = []
        st.session_state.df = df

    # ── A1b. Relationship between Dependent 1 and Dependent 2 ──────────
    dependent_relationship = "joint"
    chain_use_fitted = True
    chain_driver_role = "non_media"
    chain_driver_positive = True
    if enable_second_dependent and target2:
        st.markdown("#### 🔀 Relationship between Dependent 1 and Dependent 2")
        relationship_choice = st.radio(
            "How should the two dependents be linked?",
            [
                "🔗 Joint — fitted together in one bivariate Kalman filter (correlated errors, cross-intercept coupling)",
                "➡️ Chained — Dependent 2 is fitted on its own first, then its fitted values become an X-driver inside Dependent 1's equation",
            ],
            key="cfg_dep_relationship_mode",
        )
        dependent_relationship = "chained" if relationship_choice.startswith("➡️") else "joint"

        if dependent_relationship == "chained":
            info(
                f"➡️ <b>Chained mode</b> — <code>{target2}</code> is modeled on its own first "
                f"(own predictors, own adstock/saturation, own equation — exactly like a "
                f"single-dependent RBE fit). Its resulting fitted trajectory is then added as a "
                f"brand-new predictor column and used as an <b>X-driver inside "
                f"{target if target else 'Dependent 1'}'s equation</b>, with its own beta "
                "(and, if placed in the Media role, its own adstock + saturation curve). "
                "This is a mediation / funnel relationship "
                "(e.g. Media → Consideration → Sales), not a side-by-side joint fit."
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                chain_use_fitted = st.radio(
                    f"Which values of `{target2}` feed into Dependent 1?",
                    ["Smoothed / fitted values from Dependent 2's own model (recommended)",
                     "Raw actual values (no separate Dependent 2 fit is 'trusted' as ground truth)"],
                    key="cfg_chain_use_fitted",
                ).startswith("Smoothed")
            with cc2:
                chain_driver_role = st.radio(
                    "How does this driver enter Dependent 1's equation?",
                    ["Non-media (direct beta, no adstock/saturation — like an organic control)",
                     "Media-type (gets its own adstock + Hill/Power saturation curve, like a channel)"],
                    key="cfg_chain_driver_role",
                )
                chain_driver_role = "media" if chain_driver_role.startswith("Media") else "non_media"
            chain_driver_positive = st.checkbox(
                f"Require `{target2}`'s effect on `{target}` to be POSITIVE",
                value=True, key="cfg_chain_driver_positive",
                help="Enforces a non-negative beta — appropriate for funnel relationships "
                     "like Consideration → Sales where more of Dep 2 shouldn't hurt Dep 1.",
            )
            st.caption(
                f"📌 Dependent 1's equation will gain one new predictor: the "
                f"{'fitted' if chain_use_fitted else 'raw actual'} values of `{target2}`, "
                f"entering as a **{chain_driver_role.replace('_',' ')}** variable"
                f"{' with a positive-beta constraint' if chain_driver_positive else ''}."
            )

    st.divider()

    # ── A3. Predictor Variables — Dependent 2 (optional independent set) ──
    media_2 = list(media); non_media_2 = list(non_media)
    comp_media_2 = list(comp_media); comp_nonmedia_2 = list(comp_nonmedia)
    price_vars_2 = list(price_vars)
    use_price_2 = False
    different_predictors_2 = False
    if enable_second_dependent and target2:
        st.markdown("### A3 · Predictor Variables — Dependent 2 (optional)")
        info(
            "By default Dependent 2 reuses the exact same media / non-media / price / "
            "competitor variables as Dependent 1 (x_t is shared). If Dependent 2 is "
            "actually driven by a <b>different — but possibly overlapping — set of "
            "variables</b> (e.g. only TV and Digital feed Consideration, while Sales "
            "also responds to Price and a promo flag) enable independent selection "
            "below. Any variable may appear in <b>both</b> Dependent 1's and "
            "Dependent 2's predictor sets at once — the two equations still share the "
            "same underlying time index and are fitted jointly, but each equation's "
            "own observation matrix only includes the variables assigned to it."
        )
        different_predictors_2 = st.checkbox(
            "🔀 Use a different predictor set for Dependent 2",
            key="cfg_diff_predictors_2",
        )
        if different_predictors_2:
            options2 = [c for c in num_cols if c not in {target, target2}]
            dc1, dc2 = st.columns(2)
            with dc1:
                media_2 = safe_multiselect(
                    "📺 Media / paid channels — Dep 2", options=options2,
                    default=[c for c in media if c in options2], key="cfg_media_2")
            with dc2:
                opts_nm2 = [c for c in options2 if c not in media_2]
                non_media_2 = safe_multiselect(
                    "🗂️ Non-media / organic — Dep 2", options=opts_nm2,
                    default=[c for c in non_media if c in opts_nm2], key="cfg_nonmedia_2")

            dc3, dc4 = st.columns(2)
            with dc3:
                opts_price2 = [c for c in options2 if c not in media_2 and c not in non_media_2]
                price_vars_2 = safe_multiselect(
                    "💲 Price variables — Dep 2", options=opts_price2,
                    default=[c for c in price_vars if c in opts_price2], key="cfg_price_2")
            with dc4:
                opts_cm2 = [c for c in options2
                            if c not in media_2 and c not in non_media_2 and c not in price_vars_2]
                comp_media_2 = safe_multiselect(
                    "📉 Competitor media — Dep 2", options=opts_cm2,
                    default=[c for c in comp_media if c in opts_cm2], key="cfg_comp_media_2")

            opts_cnm2 = [c for c in options2
                         if c not in media_2 and c not in non_media_2
                         and c not in price_vars_2 and c not in comp_media_2]
            comp_nonmedia_2 = safe_multiselect(
                "📉 Competitor non-media — Dep 2", options=opts_cnm2,
                default=[c for c in comp_nonmedia if c in opts_cnm2], key="cfg_comp_nonmedia_2")

            use_price_2 = st.checkbox(
                "Include price effects — Dep 2", value=bool(price_vars_2), key="cfg_use_price_2")

            if not media_2:
                st.warning("Dependent 2 needs at least one media channel — falling back to Dependent 1's media list.")
                media_2 = list(media)

            shared = (
                (set(media) | set(non_media) | set(price_vars) | set(comp_media) | set(comp_nonmedia)) &
                (set(media_2) | set(non_media_2) | set(price_vars_2) | set(comp_media_2) | set(comp_nonmedia_2))
            )
            st.caption(
                f"Dep 2 predictors: **{len(media_2)}** media · **{len(non_media_2)}** non-media · "
                f"**{len(price_vars_2)}** price · **{len(comp_media_2)}** comp-media · "
                f"**{len(comp_nonmedia_2)}** comp-non-media. "
                f"**{len(shared)}** variable(s) shared with Dependent 1: "
                f"{', '.join(sorted(shared)) if shared else 'none'}."
            )
        else:
            use_price_2 = None  # resolved after Section F, once use_price (Dep 1) is known

    st.divider()

    # ── B. Beta Sign Constraints ─────────────────────────────────────
    st.markdown("### B · Beta Sign Constraints")

    all_own   = list(media) + list(non_media)
    all_comp  = list(comp_media) + list(comp_nonmedia)
    all_price = list(price_vars)
    all_sign_cols = all_own + all_comp + all_price

    bcol1, bcol2 = st.columns(2)

    with bcol1:
        positive_info(
            "📈 <b>Positive Beta Enforcement</b> — Variables selected here must "
            "contribute <b>positively</b> to the KPI. The optimizer enforces "
            "non-negative <code>delta</code> bounds and floors filtered betas at zero."
        )
        if all_own:
            positive_beta_cols = safe_multiselect(
                "📈 Variables that must have POSITIVE betas",
                options=all_own, default=list(media), key="positive_beta_cols")
        else:
            positive_beta_cols = []
            st.info("Select media or non-media channels in Section A first.")

    with bcol2:
        st.markdown(
            '<div style="background:#1e293b;border-left:4px solid #ef4444;'
            'padding:8px 12px;border-radius:4px;margin-bottom:8px;">'
            '📉 <b>Negative Beta Enforcement</b> — Variables selected here must '
            'contribute <b>negatively</b> to the KPI. The optimizer enforces '
            'non-positive <code>delta</code> bounds and caps filtered betas at zero.'
            '</div>', unsafe_allow_html=True)
        neg_candidates = all_comp + all_price + all_own
        # default: competitor media + price are naturally negative
        neg_defaults = list(comp_media) + list(comp_nonmedia) + list(price_vars)
        neg_defaults = [c for c in neg_defaults if c in neg_candidates]
        if neg_candidates:
            negative_beta_cols = safe_multiselect(
                "📉 Variables that must have NEGATIVE betas",
                options=neg_candidates, default=neg_defaults, key="negative_beta_cols")
            # Remove overlap — positive takes precedence
            negative_beta_cols = [c for c in negative_beta_cols
                                   if c not in positive_beta_cols]
        else:
            negative_beta_cols = []
            st.info("Select channels in Section A first.")

    if positive_beta_cols or negative_beta_cols:
        st.caption(
            f"🔒 Positive: **{', '.join(positive_beta_cols) or 'none'}**  |  "
            f"📉 Negative: **{', '.join(negative_beta_cols) or 'none'}**"
        )

    # ── B2. Beta Sign Constraints — Dependent 2 (only if predictors differ) ──
    positive_beta_cols_2 = list(positive_beta_cols)
    negative_beta_cols_2 = list(negative_beta_cols)
    if enable_second_dependent and target2 and different_predictors_2:
        st.divider()
        st.markdown("### B2 · Beta Sign Constraints — Dependent 2")
        all_own_2  = list(media_2) + list(non_media_2)
        all_comp_2 = list(comp_media_2) + list(comp_nonmedia_2)
        bcol1b, bcol2b = st.columns(2)
        with bcol1b:
            if all_own_2:
                positive_beta_cols_2 = safe_multiselect(
                    "📈 Dep 2 — variables that must have POSITIVE betas",
                    options=all_own_2, default=[c for c in media_2 if c in all_own_2],
                    key="positive_beta_cols_2")
            else:
                positive_beta_cols_2 = []
                st.info("Select Dep 2 media or non-media channels in Section A3 first.")
        with bcol2b:
            neg_candidates_2 = all_comp_2 + list(price_vars_2) + all_own_2
            neg_defaults_2 = [c for c in (list(comp_media_2) + list(comp_nonmedia_2) + list(price_vars_2))
                               if c in neg_candidates_2]
            if neg_candidates_2:
                negative_beta_cols_2 = safe_multiselect(
                    "📉 Dep 2 — variables that must have NEGATIVE betas",
                    options=neg_candidates_2, default=neg_defaults_2,
                    key="negative_beta_cols_2")
                negative_beta_cols_2 = [c for c in negative_beta_cols_2 if c not in positive_beta_cols_2]
            else:
                negative_beta_cols_2 = []
                st.info("Select Dep 2 channels in Section A3 first.")
        st.caption(
            f"🔒 Dep 2 Positive: **{', '.join(positive_beta_cols_2) or 'none'}**  |  "
            f"📉 Dep 2 Negative: **{', '.join(negative_beta_cols_2) or 'none'}**"
        )

    st.divider()

    # ── C. Cross-media Learning ───────────────────────────────────────
    st.markdown("### C · Cross-media Learning")
    info("Define which channels learn from each other.")
    cross_map = {}
    if media:
        for tgt in media:
            sources = safe_multiselect(
                f"Channels that influence **{tgt}**",
                options=[m for m in media if m != tgt], default=[], key=f"cross_{tgt}")
            if sources:
                cross_map[tgt] = set(sources)

    cross_map_2 = dict(cross_map)
    if enable_second_dependent and target2 and different_predictors_2 and media_2:
        st.markdown("#### Cross-media Learning — Dependent 2")
        cross_map_2 = {}
        for tgt in media_2:
            sources = safe_multiselect(
                f"Dep 2 — channels that influence **{tgt}**",
                options=[m for m in media_2 if m != tgt], default=[], key=f"cross2_{tgt}")
            if sources:
                cross_map_2[tgt] = set(sources)

    st.divider()

    # ── D. Adstock & Transformation ───────────────────────────────────
    st.markdown("### D · Adstock Function & Media Transformation")

    info(
        "Choose <b>Adstock</b> (how past spend carries over) and <b>Transformation</b> "
        "(how raw spend maps to marketing effectiveness). These two choices define the "
        "state equation used for all media beta coefficients."
    )

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("#### Adstock (Carry-over) — per channel")
        info(
            "Pick <b>which channels</b> use Delayed (Weibull) adstock and which use "
            "Instant (Nerlove-Arrow) — you're no longer locked into one choice for "
            "the whole model. Any channel not placed in either box below defaults "
            "to <b>Instant</b>. Price variables are always Instant (same-period "
            "elasticity — no carry-over concept applies)."
        )
        adstock_eligible_cols = (
            list(media) + list(comp_media) + list(non_media) + list(comp_nonmedia)
        )
        # Restore any prior per-channel choice (e.g. coming back to this tab)
        # so re-visiting doesn't silently reset everyone to Instant.
        _prev_map = st.session_state.get("cfg_adstock_map", {})
        _default_weibull = [c for c in adstock_eligible_cols if _prev_map.get(c) == "weibull"]

        weibull_channels = safe_multiselect(
            "🌀 Weibull (delayed) channels",
            options=adstock_eligible_cols,
            default=[c for c in _default_weibull if c in adstock_eligible_cols],
            key="cfg_weibull_channels",
            help="β_t = Σ_l w_l·x_{t-l} + δ·f(x_t) — a weighted-lag distribution "
                 "with parameters shape k and scale λ, fitted per channel.",
        )
        instant_options = [c for c in adstock_eligible_cols if c not in weibull_channels]
        instant_channels = safe_multiselect(
            "⚡ Instant (Nerlove-Arrow) channels",
            options=instant_options,
            default=instant_options,
            key="cfg_instant_channels",
            help="β_t = λ·β_{t-1} + δ·f(x_t) — fast geometric decay, "
                 "parameter λ ∈ (0,1) fitted per channel.",
        )
        # Anything left over (e.g. user removed it from the instant box
        # without adding it to weibull) still defaults to instant.
        adstock_map = {
            c: ("weibull" if c in weibull_channels else "instant")
            for c in adstock_eligible_cols
        }
        st.session_state["cfg_adstock_map"] = adstock_map
        use_weibull = len(weibull_channels) > 0  # summary/equation-box flag only

        if weibull_channels:
            n_lags = st.number_input(
                "Number of lags to consider (L) — applies to all Weibull channels",
                min_value=0, max_value=8, value=8, step=1,
                key="adstock_n_lags",
                help=(
                    "Weibull weights are computed for lag = 0, 1, …, L "
                    "(L+1 weights total) and normalised to sum to 1. "
                    "L = 0 means only the current period is used (no carry-over)."
                ),
            )
            st.caption(
                f"📐 Weibull PDF: w_lag = (k/λ) · ((lag+1)/λ)^(k−1) · exp(−((lag+1)/λ)^k), "
                f"normalised over lag = 0…{n_lags} ({n_lags + 1} weights). "
                "Parameters **shape k** and **scale λ** are fitted per channel "
                "(set bounds below)."
            )
            st.caption(
                f"🌀 Weibull: **{', '.join(weibull_channels)}**  ·  "
                f"⚡ Instant: **{', '.join(instant_channels) or 'none'}**"
            )
        else:
            n_lags = 8  # default, unused when nothing is on weibull

    with dcol2:
        st.markdown("#### Transformation (Response Curve)")
        transform_choice = st.radio(
            "Transformation type",
            ["Hill (S-curve saturation)", "Power (diminishing returns)"],
            horizontal=False,
            key="transform_type_radio",
            help=(
                "**Hill**: f(x) = x^n / (x^n + S^n). "
                "S-shaped saturation curve — parameters n ∈ [1,15], S > 0.\n\n"
                "**Power**: f(x) = x^n. "
                "Pure diminishing returns — parameter n ∈ (0,1]."
            ),
        )
        use_hill = transform_choice.startswith("Hill")

        st.markdown("#### Intercept Transformation")
        intercept_transform_choice = st.radio(
            "Intercept transform type",
            ["Power (diminishing returns)", "Hill (S-curve saturation)"],
            horizontal=False,
            key="intercept_transform_type_radio",
            help=(
                "Independent of the media Transformation above — this only "
                "controls how each intercept-effector's boost into the "
                "baseline is shaped.\n\n"
                "**Power**: γ_k · media_k,t^n_k_intercept. "
                "Unbounded diminishing-returns curve, n ∈ (0,1].\n\n"
                "**Hill**: γ_k · media_k,t^n_k_intercept / "
                "(media_k,t^n_k_intercept + S_k_intercept^n_k_intercept). "
                "Bounded 0-1 S-curve with its own half-saturation "
                "S_k_intercept per effector, n ∈ [1,15]."
            ),
        )
        use_hill_intercept = intercept_transform_choice.startswith("Hill")

        st.markdown("#### Intercept Dynamics" + (" — Dependent 1" if (enable_second_dependent and target2) else ""))
        intercept_dynamics_choice = st.radio(
            "Intercept dynamics type" + (f" ({target})" if (enable_second_dependent and target2) else ""),
            ["With carryover (AR(1) baseline)", "With carryover (Weibull adstock)", "Without carryover (simple regression)"],
            horizontal=False,
            key="intercept_dynamics_type_radio",
            help=(
                "Independent of the Intercept Transformation above — this "
                "controls whether the intercept/baseline PERSISTS from one "
                "period to the next at all.\n\n"
                "**With carryover — AR(1)** (default): I_t = G0 · I_{t-1} + "
                "Σ_k γ_k · f(media_k,t). The baseline has its own AR(1) "
                "memory (persistence G0), on top of the effector boost.\n\n"
                "**With carryover — Weibull adstock**: I_t = G0 · Σ_l w_l · "
                "I_{t-l} + Σ_k γ_k · f(media_k,t). The baseline's memory "
                "reaches back over several past periods (window set below) "
                "with Weibull-shaped weights — a delayed/S-curve decay "
                "instead of AR(1)'s single-lag exponential decay. G0 here "
                "plays the same overall-persistence role as AR(1)'s G0 "
                "(fitted, same bounded range), just spread across the "
                "window via the Weibull shape (also fitted: shape k / "
                "scale λ, same mechanism as the per-channel Weibull media "
                "adstock) instead of concentrated at lag 1.\n\n"
                "**Without carryover**: I_t = I0 + Σ_k γ_k · f(media_k,t). "
                "A plain regression on the current period's effectors "
                "around a fitted constant I0 — no dependence on the "
                "previous period's intercept at all.\n\n"
                "For a 2-dependent-variable model, cross-intercept coupling "
                "(φ₁/φ₂) requires BOTH Dependent 1 and Dependent 2 to be on "
                "a carryover dynamics (AR(1) OR Weibull), since that "
                "coupling is itself a carryover mechanism (it references "
                "the OTHER equation's PREVIOUS intercept)."
            ),
        )
        use_simple_intercept = intercept_dynamics_choice.startswith("Without")
        use_weibull_intercept = intercept_dynamics_choice.startswith("With carryover (Weibull")

        intercept_weibull_n_lags = 4
        if use_weibull_intercept:
            intercept_weibull_n_lags = st.number_input(
                "Weibull carryover window — number of lags"
                + (f" ({target})" if (enable_second_dependent and target2) else ""),
                min_value=2, max_value=52, value=4, step=1,
                key="intercept_weibull_n_lags_input",
                help=(
                    "How many past periods (L) the intercept's Weibull "
                    "carryover reaches back over: I_t = Σ_{l=1}^{L} w_l · "
                    "I_{t-l} + effector boost. A fixed window (like the "
                    "media Weibull adstock's own lag count) — shape k and "
                    "scale λ are still fitted, this just sets how far back "
                    "the weighted sum is allowed to look."
                ),
            )

        # Dependent 2 gets its OWN independent choice — no longer forced
        # to mirror Dependent 1's. Shown whenever a second dependent is
        # configured, regardless of joint vs. chained mode, since each
        # dependent fits its own intercept equation either way.
        use_simple_intercept_2 = use_simple_intercept  # fallback for single-dependent models
        use_weibull_intercept_2 = use_weibull_intercept
        intercept_weibull_n_lags_2 = intercept_weibull_n_lags
        if enable_second_dependent and target2:
            intercept_dynamics_choice_2 = st.radio(
                f"Intercept dynamics type ({target2})",
                ["With carryover (AR(1) baseline)", "With carryover (Weibull adstock)", "Without carryover (simple regression)"],
                horizontal=False,
                key="intercept_dynamics_type_radio_2",
                help=(
                    f"Same choice as above, but for **{target2}**'s own intercept "
                    "equation — the two dependents no longer have to match. "
                    "E.g. Dependent 1 can keep AR(1) carryover while Dependent 2 "
                    "runs as a plain constant-baseline regression or a Weibull "
                    "multi-lag carryover, or vice versa."
                ),
            )
            use_simple_intercept_2 = intercept_dynamics_choice_2.startswith("Without")
            use_weibull_intercept_2 = intercept_dynamics_choice_2.startswith("With carryover (Weibull")
            if use_weibull_intercept_2:
                intercept_weibull_n_lags_2 = st.number_input(
                    f"Weibull carryover window — number of lags ({target2})",
                    min_value=2, max_value=52, value=4, step=1,
                    key="intercept_weibull_n_lags_input_2",
                    help=(
                        "Same idea as above, for Dependent 2's own intercept "
                        "equation's Weibull carryover window."
                    ),
                )

    # ── D1. Cross-intercept coupling direction (2-dependent joint fit only) ──

    # Only meaningful when a second dependent is configured, the two are
    # linked in "joint" (bivariate Kalman filter) mode, and BOTH intercepts
    # are on "carryover" dynamics — coupling is itself a carryover mechanism
    # (see modules/kalman.py module docstring), so it's a no-op if either
    # equation has been switched to a simple/constant-baseline regression.
    cross_intercept_coupling_mode_str = "both"
    if enable_second_dependent and target2 and dependent_relationship == "joint":
        st.markdown("#### Cross-intercept Coupling (Dependent 1 ↔ Dependent 2)")
        if use_simple_intercept or use_simple_intercept_2:
            _which = (
                f"{target} and {target2} are" if (use_simple_intercept and use_simple_intercept_2)
                else f"{target} is" if use_simple_intercept
                else f"{target2} is"
            )
            st.caption(
                f"🚫 Not applicable — {_which} set to **Without carryover** "
                "above, so there is no previous-period intercept for that "
                "equation to feed into (or receive from) the other."
            )
        else:
            coupling_options = [
                "🔗 Both directions (Dep1 ↔ Dep2)",
                f"➡️ One-directional — {target} → {target2} only",
                f"⬅️ One-directional — {target2} → {target} only",
                "🚫 None (equations stay coupled only through the shared error correlation ρ)",
            ]
            coupling_choice = st.radio(
                "Cross-intercept coupling direction",
                coupling_options,
                horizontal=False,
                key="cross_intercept_coupling_mode_radio",
                help=(
                    "Each equation's intercept can optionally also depend on the "
                    "OTHER equation's PREVIOUS-period intercept:\n\n"
                    "  Intercept_1,t = G0_1·Intercept_1,t-1 + φ_1·Intercept_2,t-1 + effectors_1,t\n"
                    "  Intercept_2,t = G0_2·Intercept_2,t-1 + φ_2·Intercept_1,t-1 + effectors_2,t\n\n"
                    "**Both directions**: φ_1 and φ_2 are both freely estimated "
                    f"(default, original behaviour).\n\n"
                    f"**{target} → {target2} only**: only φ_2 is estimated — "
                    f"{target}'s previous intercept feeds {target2}'s equation, "
                    f"but not the other way round (φ_1 is fixed at 0).\n\n"
                    f"**{target2} → {target} only**: only φ_1 is estimated — "
                    f"{target2}'s previous intercept feeds {target}'s equation, "
                    f"but not the other way round (φ_2 is fixed at 0).\n\n"
                    "**None**: φ_1 = φ_2 = 0 — the two equations' intercepts "
                    "evolve fully independently; the fit is still \"joint\" only "
                    "through the shared/correlated observation error ρ."
                ),
            )
            if coupling_choice.startswith("🔗"):
                cross_intercept_coupling_mode_str = "both"
            elif coupling_choice.startswith("➡️"):
                cross_intercept_coupling_mode_str = "dep1_in_dep2"
            elif coupling_choice.startswith("⬅️"):
                cross_intercept_coupling_mode_str = "dep2_in_dep1"
            else:
                cross_intercept_coupling_mode_str = "none"

    # ── D2. Loss function (2-dependent joint fit only) ───────────────────
    # Only meaningful for the JOINT bivariate fit — the NRMSE regularization
    # term is defined over both dependents' residuals at once (see
    # modules/kalman.py::joint_composite_loss), so it doesn't apply to a
    # single-dependent fit or to the chained/mediation pipeline.
    loss_function_mode_str = "nll_nrmse"
    if enable_second_dependent and target2 and dependent_relationship == "joint":
        st.markdown("#### Loss Function")
        loss_choice = st.radio(
            "Objective optimized during fitting",
            [
                "🎯 EKF NLL + λ·NRMSE Regularization (default)",
                "📐 EKF NLL only",
            ],
            horizontal=False,
            key="loss_function_mode_radio",
            help=(
                "Loss(Θ) = ½·Σₜ(log|Fₜ| + vₜᵀFₜ⁻¹vₜ) + λ·(RMSE_dep1/ȳ_dep1 + "
                "RMSE_dep2/ȳ_dep2)\n\n"
                "**EKF NLL + λ·NRMSE Regularization**: the bivariate Kalman "
                "negative log-likelihood plus an NRMSE penalty term. λ is "
                "auto-scaled once at the starting point so the two terms "
                "contribute roughly 50/50 to the loss — this pulls the fit "
                "toward lower prediction error on top of maximizing "
                "likelihood, and is the default/original behaviour.\n\n"
                "**EKF NLL only**: pure bivariate EKF negative "
                "log-likelihood — λ is forced to 0, so the NRMSE term "
                "never enters the optimizer's objective. NRMSE is still "
                "computed and shown in Results as a diagnostic either way."
            ),
        )
        loss_function_mode_str = (
            "nll_nrmse" if loss_choice.startswith("🎯") else "nll_only"
        )
        st.caption(
            "**Active objective:** "
            + ("Loss(Θ) = NLL(Θ) + λ·NRMSE(Θ), λ auto-scaled at θ₀"
               if loss_function_mode_str == "nll_nrmse"
               else "Loss(Θ) = NLL(Θ)  *(λ forced to 0 — NRMSE not optimized)*")
        )

    # Summary box showing the active state equation. Adstock is now chosen
    # PER CHANNEL (adstock_map above) — the transform (Hill/Power) is still
    # one global choice for all media betas, so the equation shown here is
    # written generically with "adstock(...)" standing in for whichever
    # per-channel choice (Σw_l·x_{t-l} or λ·β_{t-1}) that channel actually uses.
    transform_label = "Hill(x; n, S)" if use_hill else "x^n"
    transform_type_str = "hill" if use_hill else "power"
    # Legacy global field — kept for old code paths that still read it as a
    # single flag; "weibull" only if every eligible channel is weibull.
    adstock_type_str = "weibull" if (adstock_eligible_cols and not instant_channels) else "instant"
    intercept_transform_type_str = "hill" if use_hill_intercept else "power"
    intercept_dynamics_type_str = (
        "simple" if use_simple_intercept else "weibull" if use_weibull_intercept else "carryover"
    )
    intercept_dynamics_type_2_str = (
        "simple" if use_simple_intercept_2 else "weibull" if use_weibull_intercept_2 else "carryover"
    )

    f_label = "Hill(x_{i,t})" if use_hill else "x_{i,t}^n"
    if use_weibull:
        eq_text = (
            f"β_{{i,t}} = **adstock_i(x)** + δ_i · **{f_label}** + Σ_j δ_{{ij}} · {f_label.replace('_{i,t}', '_{j,t}')}\n\n"
            "where **adstock_i(x)** = Σ_l w_l·x_{i,t-l} (channel on Weibull) "
            "or λ_i·β_{i,t-1} (channel on Instant) — set per channel above."
        )
        params_text = (
            "Parameters: shape k / scale λ for Weibull channels, λ (decay) for "
            "Instant channels, plus n" + (", S" if use_hill else "") + " per channel"
        )
    else:
        eq_text = (
            f"β_{{i,t}} = **λ_i · β_{{i,t-1}}** + δ_i · **{f_label}** "
            f"+ Σ_j δ_{{ij}} · {f_label.replace('_{i,t}', '_{j,t}')}"
        )
        params_text = "Parameters: λ (decay), n" + (", S" if use_hill else "") + " per channel"

    st.info(f"**Active state equation:** {eq_text}\n\n*{params_text}*")

    _intercept_lead = (
        "I0" if use_simple_intercept
        else f"G0 · Σ_l w_l · I_{{t-l}}  (L={intercept_weibull_n_lags})" if use_weibull_intercept
        else "G₀ · I_{t-1}"
    )
    if use_hill_intercept:
        intercept_eq_text = (
            f"I_t = {_intercept_lead} + Σ_k γ_k · "
            "media_k,t^{n_k} / (media_k,t^{n_k} + S_k^{n_k})   *(Hill)*"
        )
    else:
        intercept_eq_text = (
            f"I_t = {_intercept_lead} + Σ_k γ_k · media_k,t^{{n_k}}   *(Power)*"
        )
    _dynamics_label = (
        "Simple / no carryover" if use_simple_intercept
        else "With Weibull carryover" if use_weibull_intercept
        else "With carryover"
    )
    _dep1_or_both_label = f"({target})" if (enable_second_dependent and target2) else ""
    st.markdown(f"**Intercept state {_dep1_or_both_label} ({_dynamics_label}):** {intercept_eq_text}")

    if enable_second_dependent and target2:
        _dynamics_label_2 = (
            "Simple / no carryover" if use_simple_intercept_2
            else f"With Weibull carryover (L={intercept_weibull_n_lags_2})" if use_weibull_intercept_2
            else "With carryover"
        )
        st.markdown(f"**Intercept state ({target2}) ({_dynamics_label_2})**")

    if enable_second_dependent and target2 and dependent_relationship == "joint" \
            and not use_simple_intercept and not use_simple_intercept_2:
        _coupling_label = {
            "both": f"🔗 Both directions (φ₁ and φ₂ both estimated)",
            "dep1_in_dep2": f"➡️ {target} → {target2} only (φ₂ estimated, φ₁ = 0)",
            "dep2_in_dep1": f"⬅️ {target2} → {target} only (φ₁ estimated, φ₂ = 0)",
            "none": "🚫 None (φ₁ = φ₂ = 0)",
        }.get(cross_intercept_coupling_mode_str, cross_intercept_coupling_mode_str)
        st.caption(f"**Cross-intercept coupling:** {_coupling_label}")

    st.divider()

    # ── F. Additional Options ─────────────────────────────────────────
    # (placed here so use_price / intercept_effectors are available for D2 / D3 below)
    st.markdown("### F · Additional Options")
    c1, c2 = st.columns(2)
    with c1: use_organic = st.checkbox("Organic drift (μ) in intercept state", key="use_organic")
    with c2: use_price   = st.checkbox("Include price effects", value=bool(price_vars), key="use_price")
    if use_price_2 is None:
        use_price_2 = use_price  # Dep 2 mirrors Dep 1 unless an independent predictor set was configured

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        min_base_fraction = st.number_input(
            "Baseline floor (% of avg demand)", 0.0, 0.5, 0.03, 0.01,
            key="min_base_fraction",
            help="The intercept/baseline is never reported below this fraction "
                 "of the target's average value — e.g. 0.03 means the baseline "
                 "can't drop below 3% of average demand. Set to 0 to disable "
                 "(old behaviour: baseline can go negative).")
    with bc2:
        intercept_noise_scale = st.number_input(
            "Baseline flexibility (% of avg demand / period)", 0.0, 0.5, 0.02, 0.01,
            key="intercept_noise_scale",
            help="How much the baseline is allowed to drift period-to-period "
                 "(as a fraction of average demand, 1-std per period). Higher "
                 "= the baseline can self-correct more freely instead of "
                 "getting stuck near the floor. Set to 0 for the old, "
                 "nearly-frozen behaviour.")
    with bc3:
        beta_noise_scale = st.number_input(
            "Channel beta flexibility (% of avg demand / period)", 0.0, 0.5, 0.02, 0.01,
            key="beta_noise_scale",
            help="Same idea as baseline flexibility, but for every channel's "
                 "beta (media, comp-media, non-media, price). Without this, "
                 "each beta can only move via Ls·prev + δ·trigger — with "
                 "Ls < 1, that guarantees it decays toward zero whenever its "
                 "trigger weakens, whether or not that's true of the real "
                 "effect. Set to 0 for the old, nearly-frozen behaviour.")
    # Any numeric column from the uploaded file can boost the intercept —
    # not just variables already assigned a role (media/non-media) in the
    # sales equation above. Only the dependent variable(s) are excluded.
    _excluded_targets = {target, target2} if target2 else {target}
    intercept_effector_options = [c for c in num_cols if c not in _excluded_targets]
    intercept_effectors = safe_multiselect(
        "Intercept effectors — Dep 1 (any variable from your data boosting baseline)",
        options=intercept_effector_options, default=list(media), key="intercept_eff")
    if any(c in non_media for c in intercept_effectors):
        info(
            "Non-media effectors boost the intercept through the same "
            f"<b>{'Hill' if use_hill_intercept else 'Power'}</b> intercept transform "
            "as media effectors, each with its own independently-fitted "
            "n_intercept" + (" and S_intercept" if use_hill_intercept else "") + "."
        )

    # Intercept effectors for Dep 2 — independent selection
    intercept_effectors_2 = list(media_2)  # default: Dep 2's own media list
    if enable_second_dependent and target2:
        info(
            "🎯 <b>Dep 2 Intercept Effectors</b> — because Dep 2 (e.g. Top-of-Mind / "
            "Consideration) is also driven by media spend boosting the baseline, you can "
            "choose which media (and non-media) channels feed into the Dep 2 intercept "
            "state. Defaults to Dep 2's own media channels — deselect any that should not "
            "influence Dep 2's baseline."
        )
        intercept_effectors_2 = safe_multiselect(
            f"Intercept effectors — Dep 2 · {target2}",
            options=intercept_effector_options,
            default=list(media_2),
            key="intercept_eff_2",
        )
        if any(c in non_media_2 for c in intercept_effectors_2):
            info(
                "Non-media effectors for Dep 2 boost its intercept through the same "
                f"<b>{'Hill' if use_hill_intercept else 'Power'}</b> intercept transform "
                "as media effectors (Dep 2 shares Dep 1's Intercept Transform Type)."
            )

    st.divider()

    # Per-variable hyperparameter bounds widget (shared with Tab 8 · Refine & Refit)
    _render_per_channel_bounds = render_per_channel_bounds

    # ── D2. Per-Variable Hyperparameter Bounds — Dependent 1 ─────────
    st.markdown("### D2 · Per-Variable Hyperparameter Bounds — Dependent 1")
    per_channel_info(
        "🎛️ <b>Hyperparameter bounds for Dependent 1.</b> "
        "Own-media, competitor-media, price, and non-media/control variables each have "
        "separate bounds — so media betas, competition betas, price betas, and control "
        "betas are constrained independently. "
        "Defaults are derived from each channel's data distribution. "
        "Expand a channel to customise its bounds. "
        "Each own-media channel also has a <b>Media Input Type</b> toggle: leave it as "
        "<b>Spend</b> if the column is currency (ROI uses that column's own total as "
        "before), or switch to <b>GRP / Impressions</b> and pick the matching spend "
        "column — that spend column's total will then be used as the ROI denominator "
        "for this channel instead."
    )

    all_channel_cols = list(media) + list(comp_media)
    per_channel_bounds: dict = _render_per_channel_bounds(
        channel_cols=all_channel_cols,
        comp_cols=comp_media,
        key_prefix="d1_",
        df=df,
        use_hill=use_hill,
        adstock_map=adstock_map,
        price_cols=price_vars if use_price else [],
        nonmedia_cols=non_media,
    )

    if per_channel_bounds:
        rows = [
            {"Channel": col,
             "Type": (
                 "Competitor Media" if col in comp_media else
                 "Price" if col in price_vars else
                 "Non-Media / Control" if col in non_media else
                 "Own Media"
             ),
             "Parameter": param,
             "Min": f"{v[0]:.4g}",
             "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
            for col, bdict in per_channel_bounds.items()
            for param, v in bdict.items()
            if not param.startswith("__")
        ]
        if rows:
            with st.expander("📋 Dep 1 — all per-variable bounds summary", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── D3. Per-Variable Hyperparameter Bounds — Dependent 2 ─────────
    per_channel_bounds_2: dict = {}
    if enable_second_dependent and target2:
        st.divider()
        st.markdown("### D3 · Per-Variable Hyperparameter Bounds — Dependent 2")
        per_channel_info(
            f"🎛️ <b>Independent hyperparameter bounds for Dependent 2 "
            f"(<code>{target2}</code>).</b> "
            "Because Dep 2 may have a very different scale and dynamics from Dep 1, "
            "you can set separate bounds here for own-media betas, competition betas, "
            "price betas, and non-media / control betas. "
            "The model structure (state equation, adstock, transformation type) is "
            "shared with Dep 1; only the fitted betas and these bounds differ."
        )
        per_channel_bounds_2 = _render_per_channel_bounds(
            channel_cols=list(media_2) + list(comp_media_2),
            comp_cols=comp_media_2,
            key_prefix="d2_",
            df=df,
            use_hill=use_hill,
            adstock_map=adstock_map,
            price_cols=price_vars_2 if use_price_2 else [],
            nonmedia_cols=non_media_2,
        )
        if per_channel_bounds_2:
            rows2 = [
                {"Channel": col,
                 "Type": (
                     "Competitor Media" if col in comp_media_2 else
                     "Price" if col in price_vars_2 else
                     "Non-Media / Control" if col in non_media_2 else
                     "Own Media"
                 ),
                 "Parameter": param,
                 "Min": f"{v[0]:.4g}",
                 "Max": f"{v[1]:.4g}" if v[1] is not None else "∞"}
                for col, bdict in per_channel_bounds_2.items()
                for param, v in bdict.items()
                if not param.startswith("__")
            ]
            if rows2:
                with st.expander("📋 Dep 2 — all per-variable bounds summary", expanded=False):
                    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

    st.divider()

    # ── E. Train / Test Split ─────────────────────────────────────────
    st.markdown("### E · Train / Test Split")
    train_ratio = st.slider("Training proportion", 0.50, 0.95, 0.80, 0.05,
                             format="%.0f%%", key="train_ratio")
    n_total = len(df); n_train = int(n_total * train_ratio); n_test = n_total - n_train
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", n_total); c2.metric("Train", n_train); c3.metric("Test", n_test)

    st.divider()

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        if not media:
            st.error("Select at least one media channel.")
        else:
            n_bounds_set = sum(len(v) for v in per_channel_bounds.values())
            n_grp_mapped = sum(1 for v in per_channel_bounds.values() if v.get("__spend_col__"))
            prophet_in_model = [c for c in non_media if c.startswith("prophet_")]
            st.session_state.config = {
                "target": target,
                "target2": target2 if (enable_second_dependent and target2) else None,
                "enable_second_dependent": bool(enable_second_dependent and target2),
                "dependent_relationship": dependent_relationship,
                "chain_use_fitted": chain_use_fitted,
                "chain_driver_role": chain_driver_role,
                "chain_driver_positive": chain_driver_positive,
                "media": media,
                "non_media": non_media,
                "price": price_vars if use_price else [],
                "comp_media": comp_media,
                "comp_nonmedia": comp_nonmedia,
                "dummy_cols": spike_dummy_cols_1,
                "dummy_cols_2": spike_dummy_cols_2,
                "intercept_effectors": intercept_effectors,
                "intercept_effectors_2": intercept_effectors_2,
                "cross_media_map": cross_map,
                "cross_media_map_2": cross_map_2,
                "different_predictors_2": bool(different_predictors_2),
                "media_2": media_2,
                "non_media_2": non_media_2,
                "comp_media_2": comp_media_2,
                "comp_nonmedia_2": comp_nonmedia_2,
                "price_2": price_vars_2 if use_price_2 else [],
                "use_price_2": use_price_2,
                "positive_beta_cols_2": positive_beta_cols_2,
                "negative_beta_cols_2": negative_beta_cols_2,
                "adstock_type": adstock_type_str,
                "adstock_map": adstock_map,
                "transform_type": transform_type_str,
                "intercept_transform_type": intercept_transform_type_str,
                "intercept_dynamics_type": intercept_dynamics_type_str,
                "intercept_dynamics_type_2": intercept_dynamics_type_2_str,
                "intercept_weibull_n_lags": int(intercept_weibull_n_lags),
                "intercept_weibull_n_lags_2": int(intercept_weibull_n_lags_2),
                "cross_intercept_coupling_mode": cross_intercept_coupling_mode_str,
                "loss_function_mode": loss_function_mode_str,
                "adstock_n_lags": int(n_lags),
                "use_organic": use_organic,
                "use_price": use_price,
                "min_base_fraction": float(min_base_fraction),
                "intercept_noise_scale": float(intercept_noise_scale),
                "beta_noise_scale": float(beta_noise_scale),
                "train_ratio": train_ratio,
                "n_train": n_train,
                "n_test": n_test,
                "positive_beta_cols": positive_beta_cols,
                "negative_beta_cols": negative_beta_cols,
                "per_channel_bounds": per_channel_bounds,
                "per_channel_bounds_2": per_channel_bounds_2,
                "initial_media_betas":         {c: 0.0     for c in media},
                "initial_comp_betas":          {c: -0.0001 for c in comp_media},
                "initial_own_nonmedia_betas":  {c: 0.0     for c in non_media},
                "initial_comp_nonmedia_betas": {c: -0.01   for c in comp_nonmedia},
                "initial_price_beta":          {c: -0.1    for c in price_vars},
            }
            n_weibull_ch = sum(1 for v in adstock_map.values() if v == "weibull")
            n_instant_ch = len(adstock_map) - n_weibull_ch
            if n_weibull_ch and n_instant_ch:
                adstock_label_summary = f"Mixed ({n_weibull_ch} Weibull / {n_instant_ch} Instant)"
            elif n_weibull_ch:
                adstock_label_summary = "Weibull (all channels)"
            else:
                adstock_label_summary = "Instant (all channels)"
            combo_label = f"{adstock_label_summary} × {'Hill' if use_hill else 'Power'}"
            st.success(
                f"✅ Saved — {len(media)} own-media · {len(comp_media)} competitor · "
                f"{len(non_media)} non-media "
                f"({len(prophet_in_model)} prophet col{'s' if len(prophet_in_model)!=1 else ''}) · "
                f"adstock×transform: **{combo_label}** · "
                f"{'lags: ' + str(n_lags) + ' · ' if use_weibull else ''}"
                f"train/test: **{n_train}/{n_test}** · "
                f"positive-beta: **{len(positive_beta_cols)}** · "
                f"negative-beta: **{len(negative_beta_cols)}** · "
                f"per-variable bound params: **{n_bounds_set}**" +
                (f" · GRP/Impression channels mapped to a spend column: **{n_grp_mapped}**"
                 if n_grp_mapped else "")
            )
            if prophet_in_model:
                prophet_info(
                    f"📌 Prophet controls included in model: "
                    f"<b>{', '.join(prophet_in_model)}</b>"
                )
            if st.session_state.config["enable_second_dependent"]:
                if dependent_relationship == "chained":
                    st.success(
                        f"➡️ Second dependent variable enabled: **{target2}** — will be "
                        f"fitted **on its own first**, then its "
                        f"{'fitted' if chain_use_fitted else 'raw actual'} values will be "
                        f"injected as a new **{chain_driver_role.replace('_',' ')}** predictor "
                        f"driving **{target}** in Tab 6 (chained / mediation mode)."
                    )
                else:
                    st.success(
                        f"➕ Second dependent variable enabled: **{target2}** — will be "
                        f"fitted **jointly** with **{target}** in Tab 6 using a bivariate "
                        f"Kalman filter (shared predictors x_t, correlated errors)."
                    )
                    st.caption(
                        "**Loss function:** "
                        + ("EKF NLL + λ·NRMSE Regularization"
                           if loss_function_mode_str == "nll_nrmse"
                           else "EKF NLL only")
                    )
                if different_predictors_2:
                    st.info(
                        f"🔀 Dependent 2 uses its **own predictor set**: "
                        f"{len(media_2)} media · {len(non_media_2)} non-media · "
                        f"{len(price_vars_2) if use_price_2 else 0} price · "
                        f"{len(comp_media_2)} comp-media · {len(comp_nonmedia_2)} comp-non-media."
                    )
