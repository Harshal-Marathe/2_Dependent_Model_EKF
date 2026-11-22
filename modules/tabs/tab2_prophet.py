"""
Tab 2 — Prophet Decomposition (trend / seasonality / holidays), and
merging the resulting components back into the working dataset.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.ui_helpers import section, info, prophet_info


def render_tab2(prophet_available: bool, holidays_available: bool):
    from modules.ui_helpers import need_data
    section("02", "Prophet Decomposition")
    if st.session_state.df is None: need_data()

    df = st.session_state.df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── 0 · Sales Modeling Basis (Volume vs Value) ──────────────────────
    # This does not affect Prophet itself — it just captures a setting
    # that Tab 7 (Results) and Tab 8 (Refine & Refit) need later to
    # calculate ROI correctly. Beta x Input contributions come out of the
    # model in whatever unit the dependent variable itself is in. If that
    # is Sales VALUE (currency), dividing by spend gives a correct ROI
    # directly. If it's Sales VOLUME (units/kg/tonnes), the contribution
    # must first be converted to value (x average price) before it's
    # divided by spend — otherwise ROI is in volume/currency, not
    # currency/currency.
    st.markdown("#### 📦 0 · Sales Modeling Basis (for ROI value conversion)")
    info(
        "Tell the app whether the dependent variable you are modeling is "
        "<b>Sales Value</b> (currency) or <b>Sales Volume</b> (units/kg/tonnes). "
        "This is only used later, by <b>Tab 7 · Results &amp; ROI Analytics</b> and "
        "<b>Tab 8 · Refine &amp; Refit</b>, to correctly convert Beta × Input "
        "contributions into value before they're divided by spend for ROI."
    )
    mb1, mb2, mb3 = st.columns(3)
    with mb1:
        modeling_basis_label = st.radio(
            "I am modeling:", ["Sales Value", "Sales Volume"],
            key="sales_modeling_basis_choice", horizontal=True,
        )

    price_col = None
    avg_price = 1.0
    volume_unit = None
    price_conversion_factor = 1.0

    if modeling_basis_label == "Sales Volume":
        with mb2:
            price_col = st.selectbox(
                "💰 Price column", num_cols, key="sales_price_col_select")
        with mb3:
            volume_unit = st.selectbox(
                "📏 Volume is currently in", ["Kg", "Tonnes"],
                key="sales_volume_unit_select",
                help="If Tonnes, the volume will be converted to Kg "
                     "(× 1000) before multiplying by price, since price "
                     "is assumed to be quoted per Kg.")
        if price_col:
            avg_price = float(pd.to_numeric(df[price_col], errors="coerce").mean())
            unit_mult = 1000.0 if volume_unit == "Tonnes" else 1.0
            price_conversion_factor = avg_price * unit_mult
            st.success(
                f"✅ Average price of **{price_col}** = **{avg_price:,.4f}**"
                + (f" · Volume unit = **Tonnes** → × 1000 to convert to Kg "
                   f"(price assumed per Kg)" if volume_unit == "Tonnes" else "")
                + f" → **price conversion factor stored = {price_conversion_factor:,.4f}**. "
                "This will be used automatically in the ROI panel of Tab 7 / Tab 8."
            )
    else:
        st.caption(
            "Sales Value modeling selected — ROI in Tab 7/Tab 8 will use "
            "Beta × Input directly (no price conversion applied)."
        )

    st.session_state.sales_modeling_basis = (
        "volume" if modeling_basis_label == "Sales Volume" else "value")
    st.session_state.sales_price_col = price_col
    st.session_state.sales_avg_price = avg_price
    st.session_state.sales_volume_unit = volume_unit
    st.session_state.price_conversion_factor = price_conversion_factor

    st.divider()

    info("Prophet extracts <b>trend</b>, <b>seasonality</b>, and <b>holiday effects</b>. "
         "After fitting, click <b>Add prophet components to dataset</b> — those columns "
         "will then appear in <b>Tab 4</b> and <b>Tab 5</b> immediately.")

    col1, col2, col3 = st.columns(3)
    with col1: p_target = st.selectbox("🎯 Dependent variable", num_cols, key="p_target")
    with col2: p_date   = st.selectbox("📅 Date column", df.columns.tolist(), key="p_date")
    with col3: p_freq   = st.selectbox("📊 Frequency", ["Weekly","Monthly","Daily"], key="p_freq")

    st.markdown("#### 🗓️ Holiday Effects")
    COUNTRY_CODES = {
        "None (disable holidays)": None, "India (IN)":"IN", "United States (US)":"US",
        "United Kingdom (UK)":"GB", "Germany (DE)":"DE", "France (FR)":"FR",
        "Brazil (BR)":"BR", "Canada (CA)":"CA", "Australia (AU)":"AU",
        "Japan (JP)":"JP", "China (CN)":"CN", "Singapore (SG)":"SG",
        "UAE (AE)":"AE", "South Africa (ZA)":"ZA", "Mexico (MX)":"MX",
    }
    hc1, hc2 = st.columns([2, 1])
    with hc1:
        country_label   = st.selectbox("Country for public holidays",
                                        list(COUNTRY_CODES.keys()), index=1, key="holiday_country")
        holiday_country = COUNTRY_CODES[country_label]
    with hc2:
        custom_holidays_text = st.text_area(
            "Custom holidays (YYYY-MM-DD, one per line)",
            height=100, placeholder="2023-11-01\n2024-01-26", key="custom_holidays")

    if not prophet_available:
        st.error("Prophet not installed. Run `pip install prophet` and restart.")
    else:
        from prophet import Prophet
        if holidays_available:
            from prophet.make_holidays import make_holidays_df
        else:
            make_holidays_df = None

        if st.button("🚀 Run Prophet Decomposition", type="primary", use_container_width=True):
            with st.spinner("Fitting Prophet…"):
                try:
                    df_work = st.session_state.df.copy()
                    df_work[p_date] = pd.to_datetime(df_work[p_date])
                    df_work = df_work.sort_values(p_date).reset_index(drop=True)
                    prophet_df = df_work[[p_date, p_target]].rename(
                        columns={p_date: "ds", p_target: "y"})
                    year_min = int(prophet_df["ds"].dt.year.min())
                    year_max = int(prophet_df["ds"].dt.year.max()) + 1
                    holidays_df = None
                    if holiday_country and holidays_available:
                        try:
                            holidays_df = make_holidays_df(
                                year_list=list(range(year_min, year_max+1)),
                                country=holiday_country)
                        except Exception:
                            holidays_df = None
                    custom_rows = []
                    if custom_holidays_text.strip():
                        for line in custom_holidays_text.strip().split("\n"):
                            line = line.strip()
                            if line:
                                try:
                                    custom_rows.append({
                                        "ds": pd.to_datetime(line),
                                        "holiday": "custom_holiday",
                                        "lower_window": 0, "upper_window": 1,
                                    })
                                except Exception:
                                    pass
                    if custom_rows:
                        cdf = pd.DataFrame(custom_rows)
                        holidays_df = (pd.concat([holidays_df, cdf], ignore_index=True)
                                       if holidays_df is not None else cdf)
                    m_kwargs = dict(
                        yearly_seasonality=True,
                        weekly_seasonality=(p_freq == "Weekly"),
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05,
                        seasonality_mode="additive",
                    )
                    if holidays_df is not None:
                        m_kwargs["holidays"] = holidays_df
                    m = Prophet(**m_kwargs)
                    if holiday_country and holidays_df is None:
                        try: m.add_country_holidays(country_name=holiday_country)
                        except Exception: pass
                    m.fit(prophet_df)
                    forecast = m.predict(m.make_future_dataframe(periods=0))
                    st.session_state.prophet_results = {
                        "forecast": forecast, "model": m,
                        "target_col": p_target, "date_col": p_date,
                        "df_prophet": df_work, "freq": p_freq,
                        "holiday_country": holiday_country,
                    }
                    st.success("✅ Prophet fitted! Now click 'Add prophet components to dataset' below.")
                except Exception as e:
                    st.error(f"Prophet error: {e}")

    if st.session_state.pop("_prophet_stale", False):
        st.warning("The saved Prophet results didn't match the current dataset "
                   "(their target/date column is not in the data), so they were cleared. "
                   "Pick the columns and click the run button again.")

    if st.session_state.prophet_results:
        pr = st.session_state.prophet_results
        fc = pr["forecast"]; df_p = pr["df_prophet"]
        # Everything below must use the columns Prophet was actually FITTED on
        # (stored in the results), not whatever the selectboxes say right now —
        # otherwise changing the selection, uploading a new file or loading a
        # saved workspace makes df_p[p_target] raise a KeyError.
        p_target = pr.get("target_col", p_target)
        p_date   = pr.get("date_col", p_date)
        if (p_target not in df_p.columns or p_date not in df_p.columns
                or st.session_state.df is None
                or p_date not in st.session_state.df.columns):
            st.session_state.prophet_results = None
            st.session_state["_prophet_stale"] = True
            st.rerun()

        c1, c2, c3 = st.columns(3)
        c1.metric("Observed mean", f"{df_p[p_target].mean():,.2f}")
        c2.metric("Periods", len(df_p))
        c3.metric("Holidays", pr.get("holiday_country") or "None")

        fig_av = go.Figure()
        fig_av.add_trace(go.Scatter(x=df_p[p_date], y=df_p[p_target],
                                     name="Actual", line=dict(color="#3b82f6", width=2)))
        fig_av.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"],
                                     name="Prophet fit", line=dict(color="#f59e0b", width=2, dash="dash")))
        fig_av.update_layout(height=380, template="plotly_white", title="Actual vs Prophet",
                              legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_av, use_container_width=True)

        components = [c for c in ["trend", "weekly", "yearly", "holidays"] if c in fc.columns]
        if components:
            fig_d = make_subplots(rows=len(components), cols=1, shared_xaxes=True,
                                   subplot_titles=[c.title() for c in components])
            for i, comp in enumerate(components, 1):
                fig_d.add_trace(go.Scatter(x=fc["ds"], y=fc[comp], name=comp.title()), row=i, col=1)
            fig_d.update_layout(height=260*len(components), showlegend=False, template="plotly_white")
            st.plotly_chart(fig_d, use_container_width=True)

        st.divider()
        st.markdown("#### 📌 Add Prophet Components to Dataset")
        info(
            "Clicking this merges <code>prophet_trend</code>, <code>prophet_yearly</code>, etc. "
            "into the working dataset. They will <b>immediately appear</b> in "
            "<b>Tab 4 (Correlation)</b> and <b>Tab 5 (Non-media controls)</b> on the next render."
        )

        already = st.session_state.prophet_cols_added
        if already:
            prophet_info(f"✅ Already in dataset: <b>{', '.join(already)}</b>. "
                         f"Re-clicking will re-merge (replaces old columns).")

        comp_cols_avail = [c for c in ["trend", "weekly", "yearly", "holidays"] if c in fc.columns]
        sel_comps = st.multiselect(
            "Choose components to add",
            comp_cols_avail,
            default=comp_cols_avail,
            key="prophet_comp_select",
        )

        if st.button("📌 Add selected prophet components to dataset", type="primary",
                     use_container_width=True):
            if not sel_comps:
                st.warning("Select at least one component.")
            else:
                # ── Build merge ───────────────────────────────────────
                new_col_names = [f"prophet_{c}" for c in sel_comps]
                comp_df = fc[["ds"] + sel_comps].copy()
                comp_df.columns = ["ds"] + new_col_names
                comp_df["ds"] = pd.to_datetime(comp_df["ds"])

                base_df = st.session_state.df.copy()
                # Drop any old prophet columns to avoid duplicates on re-merge
                old_pcols = [c for c in base_df.columns if c.startswith("prophet_")]
                if old_pcols:
                    base_df = base_df.drop(columns=old_pcols)
                base_df[p_date] = pd.to_datetime(base_df[p_date])

                merged = base_df.merge(comp_df, left_on=p_date, right_on="ds", how="left")
                if "ds" in merged.columns and "ds" != p_date:
                    merged = merged.drop(columns=["ds"])

                # Force the new prophet columns to a clean float dtype.
                # (A left-merge with no date overlap would otherwise leave
                # an all-NaN object-dtype column in rare edge cases, which
                # could make select_dtypes(include=np.number) skip it on
                # a later render — a second, sneakier way prophet columns
                # could "disappear". Explicit cast removes that risk.)
                for c in new_col_names:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").astype(float)

                # ── Write back ────────────────────────────────────────
                st.session_state.df = merged
                st.session_state.prophet_cols_added = new_col_names

                # ── Pre-seed widget session-state keys BEFORE rerun ──
                # This still helps the very next render. The durable
                # protection across ALL future reruns comes from
                # safe_multiselect() in Tab 4 / Tab 5, which sanitizes
                # against current options on every single run.
                all_numeric_after = merged.select_dtypes(include=np.number).columns.tolist()
                st.session_state["corr_vars"] = all_numeric_after

                prev_nonmedia = st.session_state.get("cfg_nonmedia", [])
                valid_prev = [c for c in prev_nonmedia if c in merged.columns]
                new_nonmedia = sorted(set(valid_prev) | set(new_col_names))
                st.session_state["cfg_nonmedia"] = new_nonmedia

                # Update saved config non_media if a full config already exists
                if (st.session_state.config is not None
                        and "media" in st.session_state.config):
                    cfg = st.session_state.config
                    ctrl = set(cfg.get("non_media", [])) | set(new_col_names)
                    cfg["non_media"] = sorted(ctrl)
                    st.session_state.config = cfg

                st.success(
                    f"✅ Added {len(new_col_names)} prophet column(s): "
                    f"`{'`, `'.join(new_col_names)}` — "
                    f"dataset now has **{merged.shape[1]}** columns. "
                    f"Switching to Tab 4 or Tab 5 will show them immediately."
                )

                # Verify merge quality
                null_counts = merged[new_col_names].isnull().sum()
                if null_counts.any():
                    st.warning(
                        f"⚠️ Some rows have nulls after merge (date mismatch?): "
                        f"{null_counts[null_counts > 0].to_dict()}"
                    )

                with st.expander("🔍 Preview merged prophet columns", expanded=True):
                    prev_cols = ([p_date] if p_date in merged.columns else []) + new_col_names
                    st.dataframe(merged[prev_cols].head(8), use_container_width=True)

                # Trigger full rerun so all tabs see the updated dataframe
                st.rerun()
