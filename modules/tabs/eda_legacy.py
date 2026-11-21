"""
Legacy EDA tools (dependent-variable stats, spends mix, dependent vs. independent + univariate OLS).
Kept from the original EDA tab; now shown under "More tools" in the new EDA tab.

Three sections:
  A. Summary statistics for the selected dependent variable.
  B. Spends summary table — sum & proportion of total for selected spend
     (media / cost) variables.
  C. Dependent vs. Independent line-chart explorer, with optional
     secondary axis, a min/max/average/correlation summary table, and a
     univariate OLS fit for each selected independent variable.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

from modules.ui_helpers import section, info, need_data, safe_multiselect


def _fmt(v):
    """Small helper — format a stat number for display, keep NaN as '—'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return round(float(v), 4)


def render_legacy_eda():
    if st.session_state.df is None:
        return

    df = st.session_state.df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.warning("No numeric columns found in the uploaded dataset.")
        st.stop()

    # Optional date column for the x-axis of the line chart in Section C.
    date_guess_idx = 0
    date_like = [c for c in df.columns if "date" in c.lower()]
    all_cols = df.columns.tolist()
    if date_like:
        date_guess_idx = all_cols.index(date_like[0]) + 1  # +1 for "(row index)"
    x_axis_col = st.selectbox(
        "X-axis for the line chart (used in Section C)",
        options=["(row index)"] + all_cols,
        index=date_guess_idx,
        key="eda_x_axis_col",
    )
    if x_axis_col != "(row index)":
        try:
            x_series_full = pd.to_datetime(df[x_axis_col])
        except (ValueError, TypeError):
            x_series_full = df[x_axis_col]
    else:
        x_series_full = pd.Series(df.index, index=df.index)

    st.divider()

    # ════════════════════════════════════════════════════════════════
    # SECTION A — Summary statistics for the dependent variable
    # ════════════════════════════════════════════════════════════════
    st.markdown("### A · Dependent Variable — Summary Statistics")
    dep_var = st.selectbox("Select dependent variable", num_cols, key="eda_dep_var")

    y_full = df[dep_var].astype(float)
    y_nonzero = y_full[y_full != 0]

    stats_a = {
        "Count":            y_full.count(),
        "Missing":          int(y_full.isna().sum()),
        "Sum":              y_full.sum(),
        "Mean":             y_full.mean(),
        "Median":           y_full.median(),
        "Std Dev":          y_full.std(),
        "Min":              y_full.min(),
        "Max":              y_full.max(),
        "Non-zero Count":   int(y_nonzero.count()),
        "Non-zero Mean":    y_nonzero.mean() if not y_nonzero.empty else np.nan,
        "Non-zero Min":     y_nonzero.min() if not y_nonzero.empty else np.nan,
    }
    stats_a_df = pd.DataFrame({
        "Statistic": list(stats_a.keys()),
        dep_var: [_fmt(v) if k not in ("Count", "Missing", "Non-zero Count")
                  else int(v) for k, v in stats_a.items()],
    })
    st.dataframe(stats_a_df, use_container_width=True, hide_index=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════
    # SECTION B — Spends summary (sum & proportion)
    # ════════════════════════════════════════════════════════════════
    st.markdown("### B · Spends Summary")
    info("Select the spend / cost variables to see their total spend and share of "
         "the combined total. If a channel was set up in <b>Tab 5 · Configuration</b> "
         "as <b>GRP / Impressions</b> mapped to an actual spend column, that mapped "
         "spend column is used automatically here instead of the raw GRP/impression "
         "numbers.")

    spend_default = [c for c in num_cols
                      if any(kw in c.lower() for kw in
                             ["spend", "cost", "media", "tv", "digital",
                              "social", "search", "display", "radio",
                              "ooh", "print", "paid"])]
    spend_vars = safe_multiselect(
        "Select spend variables",
        options=num_cols,
        default=spend_default,
        key="eda_spend_vars",
    )

    if spend_vars:
        # ── Resolve GRP/Impressions channels to their mapped spend column ──
        # In Tab 5 · Configuration, an own-media channel can be marked as
        # "GRP / Impressions" with an actual spend column picked alongside
        # it (stored as bounds[channel]["__spend_col__"]). Summing the raw
        # GRP/impression numbers as "spend" is meaningless, so if the
        # variable selected here matches such a channel, use its mapped
        # spend column's values instead — same convention the model itself
        # already uses for ROI (see modules/bounds_ui.py / pipeline.py).
        cfg = st.session_state.get("config")
        grp_map: dict = {}
        if cfg:
            for bounds_key in ("per_channel_bounds", "per_channel_bounds_2"):
                for ch, bdict in (cfg.get(bounds_key) or {}).items():
                    mapped = bdict.get("__spend_col__")
                    if mapped:
                        grp_map[ch] = mapped

        resolved_col, is_mapped, label = {}, {}, {}
        for var in spend_vars:
            mapped_col = grp_map.get(var)
            if mapped_col and mapped_col in df.columns:
                resolved_col[var] = mapped_col
                is_mapped[var] = True
                label[var] = f"{var} → {mapped_col} (GRP/Impr. mapped)"
            else:
                resolved_col[var] = var
                is_mapped[var] = False
                label[var] = var

        if any(is_mapped.values()):
            mapped_list = ", ".join(f"**{v}** → `{resolved_col[v]}`"
                                     for v in spend_vars if is_mapped[v])
            info(f"📡 GRP/Impressions channels resolved via Tab 5 config: {mapped_list}")

        exclude_vars = safe_multiselect(
            "Exclude from Proportion calculation "
            "(e.g. a non-spend variable picked above by mistake) — stays visible in "
            "the table below, just left out of the % math",
            options=spend_vars,
            key="eda_spend_exclude",
        )

        sums = pd.Series({var: df[resolved_col[var]].astype(float).sum() for var in spend_vars})
        included_vars = [v for v in spend_vars if v not in exclude_vars]
        total_spend = sums[included_vars].sum() if included_vars else 0.0

        rows = []
        for var in spend_vars:
            excluded = var in exclude_vars
            var_sum = sums[var]
            prop = (var_sum / total_spend * 100) if (not excluded and total_spend != 0) else np.nan
            rows.append({
                "Spend Variable": label[var],
                "Sum": round(var_sum, 2),
                "Proportion of Total (%)": _fmt(prop),
                "Included in %": "❌ Excluded" if excluded else "✅ Included",
            })
        spend_df = pd.DataFrame(rows).sort_values("Sum", ascending=False)
        st.dataframe(spend_df, use_container_width=True, hide_index=True)

        fig_spend = px.bar(
            spend_df.sort_values("Sum"), x="Sum", y="Spend Variable",
            orientation="h", color="Sum", color_continuous_scale="Blues",
            title="Total Spend by Variable",
        )
        fig_spend.update_layout(height=max(320, 30 * len(spend_df)),
                                 template="plotly_white")
        st.plotly_chart(fig_spend, use_container_width=True)

        st.download_button(
            "📥 Download Spends Summary",
            spend_df.to_csv(index=False).encode(), "spends_summary.csv", "text/csv",
            key="eda_spend_dl",
        )
    else:
        st.info("Select at least 1 spend variable.")

    st.divider()

    # ════════════════════════════════════════════════════════════════
    # SECTION C — Dependent vs. Independent variable(s)
    # ════════════════════════════════════════════════════════════════
    st.markdown("### C · Dependent vs. Independent Variable(s)")
    info("Compare the dependent variable (selected in Section A above) against "
         "one or more independent variables over time.")

    indep_options = [c for c in num_cols if c != dep_var]
    compare_vars = safe_multiselect(
        "Select one or more independent variable(s) to compare",
        options=indep_options,
        key="eda_compare_vars",
    )

    if compare_vars:
        secondary_vars = safe_multiselect(
            "Plot on secondary axis (use if scale differs from the dependent variable)",
            options=compare_vars,
            key="eda_secondary_vars",
        )

        use_secondary = len(secondary_vars) > 0
        fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

        fig.add_trace(
            go.Scatter(x=x_series_full, y=y_full, mode="lines", name=dep_var,
                       line=dict(color="#1f77b4", width=2.5)),
            secondary_y=False,
        )

        palette = px.colors.qualitative.Set2
        for i, var in enumerate(compare_vars):
            on_secondary = var in secondary_vars
            fig.add_trace(
                go.Scatter(
                    x=x_series_full, y=df[var].astype(float), mode="lines",
                    name=f"{var}" + (" (secondary)" if on_secondary else ""),
                    line=dict(color=palette[i % len(palette)], width=1.8),
                ),
                secondary_y=on_secondary,
            )

        fig.update_layout(
            title=f"{dep_var} vs. {', '.join(compare_vars)}",
            template="plotly_white", height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text=dep_var, secondary_y=False)
        if use_secondary:
            fig.update_yaxes(title_text="Secondary axis", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary table: min, non-zero min, max, average, non-zero average, correlation ──
        rows = []
        for var in compare_vars:
            x_full = df[var].astype(float)
            x_nonzero = x_full[x_full != 0]
            mask = x_full.notna() & y_full.notna()
            corr = x_full[mask].corr(y_full[mask]) if mask.sum() >= 2 else np.nan
            rows.append({
                "Variable": var,
                "Min": _fmt(x_full.min()),
                "Non-zero Min": _fmt(x_nonzero.min() if not x_nonzero.empty else np.nan),
                "Max": _fmt(x_full.max()),
                "Average": _fmt(x_full.mean()),
                "Non-zero Average": _fmt(x_nonzero.mean() if not x_nonzero.empty else np.nan),
                f"Correlation with {dep_var}": _fmt(corr),
            })
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Summary Table",
            summary_df.to_csv(index=False).encode(), "eda_summary_table.csv", "text/csv",
            key="eda_summary_dl",
        )

        # ── Univariate OLS: dependent ~ each selected independent variable ──
        st.markdown("#### Univariate OLS (dependent ~ each selected variable)")
        ols_rows = []
        for var in compare_vars:
            x = df[var].astype(float)
            mask = x.notna() & y_full.notna()
            x_clean, y_clean = x[mask], y_full[mask]
            if len(x_clean) < 3 or x_clean.std() == 0:
                ols_rows.append({
                    "Variable": var, "b0 (Intercept)": np.nan, "b1 (Slope)": np.nan,
                    "R²": np.nan, "p-value": np.nan, "Std Err (b1)": np.nan,
                    "Significant (p<0.05)": "—", "n": int(mask.sum()),
                })
                continue
            res = sp_stats.linregress(x_clean, y_clean)
            ols_rows.append({
                "Variable": var,
                "b0 (Intercept)": round(res.intercept, 4),
                "b1 (Slope)": round(res.slope, 4),
                "R²": round(res.rvalue ** 2, 4),
                "p-value": round(res.pvalue, 4),
                "Std Err (b1)": round(res.stderr, 4),
                "Significant (p<0.05)": "✅ Yes" if res.pvalue < 0.05 else "❌ No",
                "n": int(mask.sum()),
            })
        ols_df = pd.DataFrame(ols_rows)
        st.dataframe(ols_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Univariate OLS Table",
            ols_df.to_csv(index=False).encode(), "eda_univariate_ols.csv", "text/csv",
            key="eda_ols_dl",
        )

        # One fit chart per selected variable, so each of the "2 variables →
        # 2 univariate OLS" the user asked for is visible individually.
        for var in compare_vars:
            x = df[var].astype(float)
            mask = x.notna() & y_full.notna()
            x_clean, y_clean = x[mask], y_full[mask]
            if len(x_clean) < 3 or x_clean.std() == 0:
                continue
            res = sp_stats.linregress(x_clean, y_clean)
            with st.expander(f"🔎 OLS fit — {dep_var} ~ {var}"):
                fig_ols = go.Figure()
                fig_ols.add_trace(go.Scatter(
                    x=x_clean, y=y_clean, mode="markers", name="Observed",
                    marker=dict(color="#4C72B0", opacity=0.7),
                ))
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = res.intercept + res.slope * x_line
                fig_ols.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines", name="OLS fit",
                    line=dict(color="#C44E52", width=2),
                ))
                fig_ols.update_layout(
                    title=f"{dep_var} = {res.intercept:.4f} + {res.slope:.4f} × {var}  "
                          f"(R²={res.rvalue**2:.3f}, p={res.pvalue:.4f})",
                    xaxis_title=var, yaxis_title=dep_var, template="plotly_white",
                )
                st.plotly_chart(fig_ols, use_container_width=True)
    else:
        st.info("Select at least 1 independent variable to compare with the dependent variable.")
