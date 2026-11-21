"""
Tab 3 — EDA (Exploratory Data Analysis), interactive version.

Flow
----
 1. Vintage setup   : date column, granularity, start month -> 12-month
                      "vintage years" (+ completeness check and charts).
 2. Categorise vars : Sales / Media / Price / Distribution / Promotions /
                      Competition (auto-guessed, fully editable).
 3. One EDA window per category + a final Manual EDA window. Each shows
    summary stats (Avg, Non-zero Avg, Min, Non-zero Min, Max, Sum, CAGR,
    Growth, YoY Growth), yearly values per vintage and interactive charts.
    Yearly basis: Sum for everything except Price (Average).
 4. "More tools": the original dependent-vs-independent / OLS explorer.
"""

import io
import math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.ui_helpers import section, info, need_data, safe_multiselect
from modules import eda_core as core
from modules.tabs.eda_legacy import render_legacy_eda

_PALETTE = px.colors.qualitative.Set2
_BAND = ["#3b82f6", "#94a3b8"]


# ════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════
def _short(label: str) -> str:
    """'Y2 (Apr-24 – Mar-25)' -> 'Y2'"""
    return label.split(" ")[0]


def _excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _facet_bars(wide: pd.DataFrame, labels, title, pct=False, signed=False):
    """Small-multiple bar chart: one panel per variable (free y-axis)."""
    vars_ = wide["Variable"].tolist()
    n = len(vars_)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=vars_,
                        horizontal_spacing=0.08,
                        vertical_spacing=min(0.18, 0.5 / max(nrows, 1)))
    for i, var in enumerate(vars_):
        r, c = divmod(i, ncols)
        vals = wide.loc[wide["Variable"] == var, labels].iloc[0].astype(float).values
        color = ([("#16a34a" if (not np.isnan(v) and v >= 0) else "#dc2626") for v in vals]
                 if signed else _PALETTE[i % len(_PALETTE)])
        txt = [("" if np.isnan(v) else (f"{v:.1f}%" if pct else f"{v:,.2f}")) for v in vals]
        fig.add_trace(go.Bar(
            x=[_short(l) for l in labels], y=vals, text=txt, textposition="outside",
            marker_color=color, customdata=labels, showlegend=False,
            hovertemplate="%{customdata}<br>%{y:,.2f}" + ("%" if pct else "") + "<extra></extra>",
        ), row=r + 1, col=c + 1)
    fig.update_layout(title=title, template="plotly_white", height=max(320, 300 * nrows),
                      margin=dict(t=80))
    return fig


def _timeseries(df, date_col, vars_, vt, scale, secondary_vars, show_bands, title):
    use_sec = len(secondary_vars) > 0 and scale == "Raw values"
    fig = make_subplots(specs=[[{"secondary_y": use_sec}]])
    for i, v in enumerate(vars_):
        y = pd.to_numeric(df[v], errors="coerce")
        if scale == "Indexed (first non-zero = 100)":
            nz = y[(y != 0) & y.notna()]
            y = y / nz.iloc[0] * 100 if len(nz) else y
        elif scale == "Z-score":
            y = (y - y.mean()) / (y.std() or 1)
        fig.add_trace(go.Scatter(x=df[date_col], y=y, mode="lines", name=v,
                                 line=dict(color=_PALETTE[i % len(_PALETTE)], width=2)),
                      secondary_y=(use_sec and v in secondary_vars))
    if show_bands and len(vt):
        for _, r in vt.iterrows():
            fig.add_vrect(x0=r["Window start"], x1=r["Window end"],
                          fillcolor=_BAND[int(r["Vintage"]) % 2], opacity=0.07, line_width=0,
                          annotation_text=_short(r["Label"]) + ("" if r["Complete"] else "*"),
                          annotation_position="top left")
    fig.update_layout(title=title, template="plotly_white", height=460,
                      hovermode="x unified", legend=dict(orientation="h", y=1.08))
    return fig


# ════════════════════════════════════════════════════════════════════
# Category window
# ════════════════════════════════════════════════════════════════════
def _render_panel(name, variables, ctx, *, default_agg, cat_by_var, manual=False):
    """Generic EDA window used for every category and for Manual EDA."""
    df, date_col, vint, vt = ctx["df"], ctx["date_col"], ctx["vint"], ctx["vt"]
    k = f"eda_{name.lower()}"

    if manual:
        variables = safe_multiselect(
            "Select the variables you want to explore (any category)",
            options=ctx["numeric_cols"], key=f"{k}_vars",
            format_func=lambda v: f"{v}  ·  {cat_by_var.get(v, core.UNASSIGNED)}")
    else:
        if not variables:
            st.info(f"No variables are assigned to **{name}**. Assign some in the "
                    f"*Variable categories* section above.")
            return
        variables = safe_multiselect(
            f"Variables shown in {name}", options=variables, default=variables,
            key=f"{k}_vars")

    if not variables:
        st.info("Select at least one variable.")
        return

    # ── Controls ────────────────────────────────────────────────
    c1, c2 = st.columns([1, 1])
    if manual:
        agg_choice = c1.selectbox(
            "Yearly basis", ["Auto (Price = Average, others = Sum)", "Sum", "Average"],
            key=f"{k}_agg")
    else:
        agg_choice = c1.selectbox("Yearly basis", [f"Default ({default_agg})", "Sum", "Average"],
                                  key=f"{k}_agg")

    def _agg_for(v):
        if agg_choice in ("Sum", "Average"):
            return agg_choice
        if manual:
            return core.DEFAULT_AGG.get(cat_by_var.get(v, core.UNASSIGNED), "Sum")
        return default_agg

    agg_by_var = {v: _agg_for(v) for v in variables}
    ign_zero = c2.checkbox(
        "Ignore zeros when averaging (treat 0 as missing)", value=(name == "Price"),
        key=f"{k}_ignz",
        help="Only affects variables whose yearly basis is Average. Useful for price "
             "columns where 0 means 'no data'.")
    zero_by_var = {v: ign_zero for v in variables}

    summary, yearly, yoy = core.build_category_tables(
        df, variables, vint, vt, agg_by_var, zero_by_var,
        include_partial=ctx["include_partial"], category_by_var=cat_by_var)

    used = [c for c in yearly.columns if c != "Variable"]
    if len(used) < 2:
        st.warning("Fewer than 2 vintage years available — CAGR / Growth / YoY need at least "
                   "two. Try an earlier start month, or tick *include incomplete years* in "
                   "the vintage setup.")

    # ── Summary stats ───────────────────────────────────────────
    st.markdown("##### Summary statistics")
    show = summary.copy()
    num_cols = [c for c in show.columns if c not in ("Variable", "Category", "Yearly basis")]
    show[num_cols] = show[num_cols].astype(float).round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Avg / Min / Max / Sum use every period from the start month onward. "
               "CAGR, Growth (first→last year) and Latest YoY use the yearly values below. "
               "CAGR = (last ÷ first)^(1/(years−1)) − 1.")

    # ── Yearly tables ───────────────────────────────────────────
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### Yearly values by vintage")
        st.dataframe(yearly.set_index("Variable").astype(float).round(2),
                     use_container_width=True)
    with t2:
        st.markdown("##### YoY growth % by vintage")
        st.dataframe(yoy.set_index("Variable").astype(float).round(2),
                     use_container_width=True)

    st.download_button(
        f"📥 Download {name} EDA (Excel)",
        _excel_bytes({"Summary": summary, "Yearly": yearly, "YoY %": yoy}),
        f"eda_{name.lower().replace(' ', '_')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{k}_dl")

    # ── Charts ──────────────────────────────────────────────────
    st.markdown("##### Charts")
    views = ["Yearly values", "YoY growth", "CAGR & Growth", "Time series"]
    all_sum = all(a == "Sum" for a in agg_by_var.values())
    if len(variables) >= 2 and all_sum and len(used) >= 1:
        views.insert(3, "Mix (share of yearly total)")
    view = st.radio("View", views, horizontal=True, key=f"{k}_view")

    if view == "Yearly values":
        if not used:
            st.info("No vintage years to plot.")
        else:
            st.plotly_chart(_facet_bars(yearly, used, f"{name} — yearly values"),
                            use_container_width=True, key=f"{k}_c1")
    elif view == "YoY growth":
        if len(used) < 2:
            st.info("Need at least 2 vintage years.")
        else:
            st.plotly_chart(_facet_bars(yoy, used[1:], f"{name} — YoY growth %",
                                        pct=True, signed=True),
                            use_container_width=True, key=f"{k}_c2")
    elif view == "CAGR & Growth":
        d = summary[["Variable", "CAGR %", "Growth %"]].melt("Variable", var_name="Metric",
                                                            value_name="%")
        fig = px.bar(d, x="Variable", y="%", color="Metric", barmode="group", text_auto=".1f",
                     color_discrete_sequence=["#1e40af", "#f97316"],
                     title=f"{name} — CAGR vs total growth (first → last year)")
        fig.update_layout(template="plotly_white", height=430)
        st.plotly_chart(fig, use_container_width=True, key=f"{k}_c3")
    elif view == "Mix (share of yearly total)":
        w = yearly.set_index("Variable")[used].astype(float)
        share = (w / w.sum() * 100).T.reset_index(names="Vintage")
        share["Vintage"] = share["Vintage"].map(_short)
        fig = px.bar(share.melt("Vintage", var_name="Variable", value_name="Share %"),
                     x="Vintage", y="Share %", color="Variable", text_auto=".1f",
                     title=f"{name} — share of yearly total (%)",
                     color_discrete_sequence=_PALETTE)
        fig.update_layout(template="plotly_white", height=430, barmode="stack")
        st.plotly_chart(fig, use_container_width=True, key=f"{k}_c4")
    else:  # Time series
        a, b, c = st.columns([1.3, 1.5, 1])
        scale = a.selectbox("Scale", ["Raw values", "Indexed (first non-zero = 100)", "Z-score"],
                            key=f"{k}_scale")
        sec = []
        if scale == "Raw values" and len(variables) > 1:
            sec = b.multiselect("Secondary axis", variables, key=f"{k}_sec")
        bands = c.checkbox("Show vintage bands", value=True, key=f"{k}_bands")
        st.plotly_chart(_timeseries(df, date_col, variables, vt, scale, sec, bands,
                                    f"{name} — time series"),
                        use_container_width=True, key=f"{k}_c5")


# ════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════
def render_tab_eda():
    section("03", "EDA · Exploratory Data Analysis")
    if st.session_state.df is None:
        need_data()

    raw = st.session_state.df.copy()

    # ────────────────────────────────────────────────────────────
    # 1 · VINTAGE SETUP
    # ────────────────────────────────────────────────────────────
    st.markdown("### 1 · Data vintage")
    info("Pick the date column, the data granularity and the <b>start month</b>. The data is "
         "cut into 12-month <b>vintage years</b> from that month (e.g. start = Apr → "
         "Apr–Mar years). Yearly sums / averages, CAGR, growth and YoY growth are all "
         "computed on these vintage years.")

    date_cands = core.find_date_columns(raw)
    if not date_cands:
        st.error("No date column could be detected in the uploaded data, so vintage-based "
                 "EDA can't run. Add a date column (e.g. `Date`) and re-upload.")
        with st.expander("More tools (dependent vs. independent, OLS)"):
            render_legacy_eda()
        return

    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date column", date_cands, key="eda_date_col")

    work = raw.copy()
    work[date_col] = core.parse_dates(work[date_col])
    n_bad = int(work[date_col].isna().sum())
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if work.empty:
        st.error(f"Column `{date_col}` has no valid dates.")
        return
    if n_bad:
        st.warning(f"{n_bad} row(s) with unparseable dates were ignored in EDA.")
    if work[date_col].duplicated().any():
        st.warning("Duplicate dates found — if the data has several rows per date "
                   "(e.g. multiple regions) yearly sums will add them together.")

    detected = core.detect_granularity(work[date_col])
    gran_opts = ["Daily", "Weekly", "Monthly"]
    granularity = c2.selectbox("Granularity", gran_opts, index=gran_opts.index(detected),
                               key="eda_gran",
                               help=f"Auto-detected: {detected}. Override if wrong.")

    months = core.month_options(work[date_col])
    start_month = c3.selectbox("Vintage start month", months, index=0,
                               format_func=lambda d: d.strftime("%b %Y"), key="eda_start_month")
    include_partial = st.checkbox(
        "Include incomplete vintage years in growth / CAGR (not recommended)", value=False,
        key="eda_incl_partial",
        help="By default only full 12-month vintage years feed CAGR / Growth / YoY, "
             "otherwise a half-year looks like a collapse.")

    vint = core.assign_vintage(work[date_col], start_month)
    vt = core.vintage_table(work[date_col], start_month, granularity)

    if work[date_col].min() > start_month + pd.Timedelta(days=31):
        st.warning("Data starts after the selected start month — the first vintage year is "
                   "partial and will be treated as incomplete.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Date range", f"{work[date_col].min():%b-%y} → {work[date_col].max():%b-%y}")
    m2.metric("Rows in analysis window", f"{int(vint.notna().sum()):,}",
              delta=f"-{int(vint.isna().sum())} before start" if vint.isna().any() else None,
              delta_color="off")
    m3.metric("Vintage years", len(vt))
    m4.metric("Complete years", int(vt["Complete"].sum()) if len(vt) else 0)

    if len(vt):
        disp = vt.copy()
        disp["Window start"] = disp["Window start"].dt.strftime("%d %b %Y")
        disp["Window end"] = disp["Window end"].dt.strftime("%d %b %Y")
        disp["Complete"] = disp["Complete"].map({True: "✅ Complete", False: "⚠️ Partial"})
        st.dataframe(disp.drop(columns=["Vintage"]), use_container_width=True, hide_index=True)

    numeric_cols = [c for c in raw.select_dtypes(include=np.number).columns if c != date_col]
    if not numeric_cols:
        st.warning("No numeric columns found in the uploaded dataset.")
        return

    # Vintage charts
    with st.expander("📈 Vintage charts", expanded=True):
        cc1, cc2 = st.columns([1, 1])
        preview_var = cc1.selectbox("Variable to preview over the vintage years", numeric_cols,
                                    key="eda_vintage_var")
        cc2.caption("Shaded bands = vintage years. `*` = incomplete year.")
        st.plotly_chart(
            _timeseries(work, date_col, [preview_var], vt, "Raw values", [], True,
                        f"{preview_var} across vintage years"),
            use_container_width=True, key="eda_vint_ts")

        if len(vt):
            cov = go.Figure()
            cov.add_trace(go.Bar(
                x=[_short(l) for l in vt["Label"]], y=vt["Observations"],
                marker_color=["#16a34a" if c else "#f59e0b" for c in vt["Complete"]],
                text=vt["Observations"], textposition="outside", name="Observations",
                customdata=vt["Label"], hovertemplate="%{customdata}<br>%{y} obs<extra></extra>"))
            cov.add_hline(y=core.EXPECTED_PERIODS[granularity], line_dash="dot",
                          annotation_text=f"Expected ≈ {core.EXPECTED_PERIODS[granularity]}")
            cov.update_layout(title="Observations per vintage year (green = complete)",
                              template="plotly_white", height=320, showlegend=False)
            st.plotly_chart(cov, use_container_width=True, key="eda_vint_cov")

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 2 · VARIABLE CATEGORIES
    # ────────────────────────────────────────────────────────────
    st.markdown("### 2 · Variable categories")
    info("Variables are auto-assigned from their names (and from your Step 5 configuration if "
         "one exists). Check the table and change any category — each variable belongs to "
         "exactly one category. <b>Unassigned</b> variables are still available in Manual EDA.")

    sig = tuple(numeric_cols)
    if st.session_state.get("_eda_cat_sig") != sig:
        st.session_state["_eda_cat_sig"] = sig
        guess = core.auto_categorize(numeric_cols, st.session_state.get("config"))
        st.session_state["_eda_cat_base"] = pd.DataFrame({
            "Variable": numeric_cols, "Category": [guess[c] for c in numeric_cols]})
        st.session_state.pop("eda_cat_editor", None)

    with st.expander("🗂️ Edit variable categories", expanded=False):
        rb1, _ = st.columns([1, 4])
        if rb1.button("↺ Re-detect", key="eda_cat_reset",
                      help="Discard manual edits and re-guess from names / Step 5 config."):
            st.session_state["_eda_cat_sig"] = None
            st.session_state.pop("eda_cat_editor", None)
            st.rerun()
        edited = st.data_editor(
            st.session_state["_eda_cat_base"], key="eda_cat_editor", hide_index=True,
            use_container_width=True, disabled=["Variable"],
            column_config={"Category": st.column_config.SelectboxColumn(
                "Category", options=core.ALL_CATEGORY_CHOICES, required=True)})
    cat_by_var = dict(zip(edited["Variable"], edited["Category"]))

    counts = pd.Series(cat_by_var).value_counts()
    cols = st.columns(len(core.ALL_CATEGORY_CHOICES))
    for col, cat in zip(cols, core.ALL_CATEGORY_CHOICES):
        col.metric(cat, int(counts.get(cat, 0)))

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 3 · EDA WINDOWS
    # ────────────────────────────────────────────────────────────
    st.markdown("### 3 · EDA windows")
    ctx = dict(df=work, date_col=date_col, vint=vint, vt=vt, numeric_cols=numeric_cols,
               include_partial=include_partial)

    windows = core.CATEGORIES + ["Manual EDA"]
    tabs = st.tabs([("🛠️ " if w == "Manual EDA" else "") + w for w in windows])
    for tab, w in zip(tabs, windows):
        with tab:
            if w == "Manual EDA":
                info("Pick any variables from any category and see the same statistics and charts.")
                _render_panel("Manual", [], ctx, default_agg="Sum",
                              cat_by_var=cat_by_var, manual=True)
            else:
                vars_in = [v for v in numeric_cols if cat_by_var.get(v) == w]
                basis = "average" if w == "Price" else "sum"
                info(f"Yearly value = <b>{basis}</b> of the periods in each vintage year. "
                     f"You can change the basis below.")
                _render_panel(w, vars_in, ctx, default_agg=core.DEFAULT_AGG[w],
                              cat_by_var=cat_by_var)

    st.divider()
    with st.expander("🔧 More tools — dependent vs. independent, spends mix, univariate OLS"):
        render_legacy_eda()
