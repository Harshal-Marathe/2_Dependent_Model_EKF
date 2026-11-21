"""
Tab 3 — EDA (Exploratory Data Analysis), dashboard-style.

Flow
----
 1. Data vintage    : date column, granularity, start month -> 12-month
                      vintage years (FY22-23 style) with completeness check.
 2. Variable groups : Sales / Media / Distribution / Price / Promotions /
                      Competitors (auto-guessed, editable).
 3. Windows         : DATA QUALITY | SALES | MEDIA | DISTRIBUTION | PRICE |
                      PROMOTIONS | COMPETITORS | CORRELATION | MANUAL EDA

Every variable window shows: trend chart with rolling average, KPI cards
(CAGR, total growth, seasonality strength, trend strength), yearly bar +
YoY growth bar, STL decomposition, summary-statistics table and
year-on-year table (CSV export). Yearly basis = sum, except Price = average.
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

# Colours matched to the reference dashboard
C_LINE = "#4f46e5"      # indigo  – main series / yearly bars
C_ROLL = "#f59e0b"      # amber   – rolling average
C_POS = "#10b981"       # green   – positive YoY
C_NEG = "#ef4444"       # red     – negative YoY
_PALETTE = px.colors.qualitative.Set2
_BAND = ["#3b82f6", "#94a3b8"]

# UI name shown on the tab -> category key used in the data
WINDOWS = [
    ("DATA QUALITY", None), ("SALES", "Sales"), ("MEDIA", "Media"),
    ("DISTRIBUTION", "Distribution"), ("PRICE", "Price"),
    ("PROMOTIONS", "Promotions"), ("COMPETITORS", "Competition"),
    ("CORRELATION", None), ("MANUAL EDA", None),
]

_CSS = """
<style>
.kpi-card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:14px 18px;
          box-shadow:0 2px 8px rgba(15,23,42,.06);}
.kpi-label{font-size:.72rem;letter-spacing:.07em;font-weight:700;color:#9ca3af;text-transform:uppercase;}
.kpi-value{font-size:1.65rem;font-weight:700;color:#111827;line-height:1.35;}
.eda-h{font-size:.78rem;letter-spacing:.08em;color:#9ca3af;font-weight:700;
       text-transform:uppercase;margin:22px 0 6px 0;}
.eda-note{font-size:.82rem;color:#6b7280;margin:10px 0 4px 0;}
</style>
"""


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def _kpi(col, label, value):
    col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def _heading(text):
    st.markdown(f'<div class="eda-h">{text}</div>', unsafe_allow_html=True)


def _fmt_pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}%"


def _fmt_num(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def _excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def _base_layout(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=18)),
        template="plotly_white", height=height, paper_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0.02),
    )
    return fig


def _trend_chart(df, date_col, var, roll, vt, show_bands):
    y = pd.to_numeric(df[var], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=y, mode="lines", name=var,
                             line=dict(color=C_LINE, width=2)))
    if roll and roll > 1:
        fig.add_trace(go.Scatter(x=df[date_col], y=y.rolling(roll).mean(), mode="lines",
                                 name=f"{roll}-period rolling avg",
                                 line=dict(color=C_ROLL, width=2)))
    if show_bands and len(vt):
        for _, r in vt.iterrows():
            fig.add_vrect(x0=r["Window start"], x1=r["Window end"],
                          fillcolor=_BAND[int(r["Vintage"]) % 2], opacity=0.07, line_width=0,
                          annotation_text=r["Label"] + ("" if r["Complete"] else "*"),
                          annotation_position="top left")
    return _base_layout(fig, f"{var} trend", 400)


def _yearly_bar(labels, values, var, basis):
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=C_LINE,
                           hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
    fig.update_yaxes(tickformat="~s")
    fig.update_layout(bargap=0.25)
    return _base_layout(fig, f"Yearly {basis.lower()} — {var}", 360)


def _yoy_bar(labels, values):
    colors = [C_POS if (not np.isnan(v) and v >= 0) else C_NEG for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(bargap=0.25)
    return _base_layout(fig, "YoY growth %", 360)


def _decomp_chart(dec, dates):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    for i, (key, color) in enumerate([("observed", C_LINE), ("trend", "#0ea5e9"),
                                      ("seasonal", C_ROLL), ("resid", "#94a3b8")], start=1):
        fig.add_trace(go.Scatter(x=dates, y=dec[key], mode="lines", showlegend=False,
                                 line=dict(color=color, width=1.8)), row=i, col=1)
    fig.update_layout(title=dict(text="Decomposition (trend & seasonality)", x=0.02,
                                 font=dict(size=18)),
                      template="plotly_white", height=620, margin=dict(l=50, r=20, t=70, b=30))
    return fig


# ════════════════════════════════════════════════════════════════════
# Single-variable dashboard (the "reference" layout)
# ════════════════════════════════════════════════════════════════════
def _variable_dashboard(var, agg, ign_zero, roll, ctx, key):
    df, date_col, vint, vt = ctx["df"], ctx["date_col"], ctx["vint"], ctx["vt"]
    gran = ctx["granularity"]
    in_win = vint.notna()
    dfw = df[in_win].reset_index(drop=True)

    complete = set(vt.loc[vt["Complete"], "Vintage"])
    use_years = set(vt["Vintage"]) if ctx["include_partial"] else complete
    labels_map = dict(zip(vt["Vintage"], vt["Label"]))

    yv_all = core.yearly_values(df[var], vint, agg, ign_zero)
    yv = yv_all[yv_all.index.isin(use_years)].sort_index()
    yoy = core.yoy_growth(yv)
    labels = [labels_map[k] for k in yv.index]

    stats = core.variable_summary(df[var], vint, agg, use_years, ign_zero)

    # ── decomposition (needed for KPI cards) ─────────────────────
    period = ctx["seasonal_period"]
    dec = core.decompose(pd.to_numeric(dfw[var], errors="coerce"), period)

    # ── trend chart ──────────────────────────────────────────────
    st.plotly_chart(_trend_chart(dfw, date_col, var, roll, vt, ctx["show_bands"]),
                    use_container_width=True, key=f"{key}_trend")

    # ── KPI cards ────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    _kpi(k1, "CAGR", _fmt_pct(stats["CAGR %"]))
    _kpi(k2, "Total growth", _fmt_pct(stats["Growth %"]))
    _kpi(k3, "Seasonality strength", _fmt_num(dec["seasonal_strength"]) if dec else "—")
    _kpi(k4, "Trend strength", _fmt_num(dec["trend_strength"]) if dec else "—")

    n_years = len(yv.dropna())
    basis_txt = "each year's total" if agg == "Sum" else "each year's average"
    if n_years >= 2:
        st.markdown(
            f'<div class="eda-note">CAGR &amp; growth read over {labels[0]} → {labels[-1]} '
            f'({n_years} years, {n_years - 1} compounding period{"s" if n_years > 2 else ""}) '
            f'· {basis_txt}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="eda-note">Need at least 2 vintage years for CAGR / growth. '
                    'Try an earlier start month or include incomplete years.</div>',
                    unsafe_allow_html=True)
    if not dec:
        st.markdown(f'<div class="eda-note">Seasonality / trend strength need at least 2 full '
                    f'seasonal cycles ({2 * period} {gran.lower()} periods for period = {period}).'
                    f'</div>', unsafe_allow_html=True)

    # ── yearly bar + YoY bar ─────────────────────────────────────
    if n_years >= 1:
        b1, b2 = st.columns(2)
        b1.plotly_chart(_yearly_bar(labels, yv.values, var, "Sum" if agg == "Sum" else "Average"),
                        use_container_width=True, key=f"{key}_yearly")
        b2.plotly_chart(_yoy_bar(labels, yoy.reindex(yv.index).values),
                        use_container_width=True, key=f"{key}_yoy")

    # ── decomposition chart ──────────────────────────────────────
    if dec:
        st.plotly_chart(_decomp_chart(dec, dfw[date_col]), use_container_width=True,
                        key=f"{key}_decomp")

    # ── tables: summary statistics + year-on-year ────────────────
    s = pd.to_numeric(dfw[var], errors="coerce")
    nz = s[s != 0]
    summary_df = pd.DataFrame([{
        "Avg": stats["Avg"], "Non-zero Avg": stats["Non-zero Avg"],
        "Min": stats["Min"], "Non-zero Min": stats["Non-zero Min"],
        "Max": stats["Max"], "Sum": stats["Sum"],
        "Count": int(s.count()), "Zero Count": int((s == 0).sum()),
    }]).round(3)
    yoy_df = pd.DataFrame({"Year": labels, var: yv.values.round(2),
                           "YoY %": yoy.reindex(yv.index).values.round(1)})

    t1, t2 = st.columns(2)
    with t1:
        _heading("Summary statistics")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button("⬇ EXPORT CSV", summary_df.to_csv(index=False).encode(),
                           f"summary_{var}.csv", "text/csv", key=f"{key}_dl_sum")
    with t2:
        _heading("Year-on-year")
        st.dataframe(yoy_df, use_container_width=True, hide_index=True)
        st.download_button("⬇ EXPORT CSV", yoy_df.to_csv(index=False).encode(),
                           f"yoy_{var}.csv", "text/csv", key=f"{key}_dl_yoy")


# ════════════════════════════════════════════════════════════════════
# Overview of every variable in a window (comparison tables + charts)
# ════════════════════════════════════════════════════════════════════
def _facet_bars(wide, labels, title, pct=False, signed=False):
    vars_ = wide["Variable"].tolist()
    n = len(vars_)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=vars_, horizontal_spacing=0.08,
                        vertical_spacing=min(0.18, 0.5 / max(nrows, 1)))
    for i, var in enumerate(vars_):
        r, c = divmod(i, ncols)
        vals = wide.loc[wide["Variable"] == var, labels].iloc[0].astype(float).values
        color = ([(C_POS if (not np.isnan(v) and v >= 0) else C_NEG) for v in vals]
                 if signed else C_LINE)
        txt = [("" if np.isnan(v) else (f"{v:.1f}%" if pct else f"{v:,.2f}")) for v in vals]
        fig.add_trace(go.Bar(x=labels, y=vals, text=txt, textposition="outside",
                             marker_color=color, showlegend=False),
                      row=r + 1, col=c + 1)
    fig.update_layout(title=title, template="plotly_white", height=max(320, 300 * nrows),
                      margin=dict(t=80))
    return fig


def _overview(name, variables, agg_by_var, zero_by_var, ctx, cat_by_var, key):
    summary, yearly, yoy = core.build_category_tables(
        ctx["df"], variables, ctx["vint"], ctx["vt"], agg_by_var, zero_by_var,
        include_partial=ctx["include_partial"], category_by_var=cat_by_var)
    used = [c for c in yearly.columns if c != "Variable"]

    _heading("Summary statistics — all variables")
    show = summary.copy()
    num_cols = [c for c in show.columns if c not in ("Variable", "Category", "Yearly basis")]
    show[num_cols] = show[num_cols].astype(float).round(2)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Avg / Min / Max / Sum use every period from the start month onward. "
               "CAGR = (last ÷ first)^(1/(years−1)) − 1; Growth = last ÷ first − 1; "
               "YoY = latest year vs the year before.")

    c1, c2 = st.columns(2)
    with c1:
        _heading("Yearly values by vintage")
        st.dataframe(yearly.set_index("Variable").astype(float).round(2), use_container_width=True)
    with c2:
        _heading("YoY growth % by vintage")
        st.dataframe(yoy.set_index("Variable").astype(float).round(2), use_container_width=True)

    st.download_button(
        f"⬇ EXPORT {name.upper()} (EXCEL)",
        _excel_bytes({"Summary": summary, "Yearly": yearly, "YoY %": yoy}),
        f"eda_{name.lower().replace(' ', '_')}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key}_dl_all")

    if len(variables) < 2 and len(used) < 2:
        return
    views = ["Yearly values", "YoY growth", "CAGR & Growth"]
    if len(variables) >= 2 and all(a == "Sum" for a in agg_by_var.values()) and used:
        views.append("Mix (share of yearly total)")
    view = st.radio("Comparison chart", views, horizontal=True, key=f"{key}_ovview")
    if view == "Yearly values" and used:
        st.plotly_chart(_facet_bars(yearly, used, f"{name} — yearly values"),
                        use_container_width=True, key=f"{key}_ov1")
    elif view == "YoY growth":
        if len(used) < 2:
            st.info("Need at least 2 vintage years.")
        else:
            st.plotly_chart(_facet_bars(yoy, used[1:], f"{name} — YoY growth %", pct=True,
                                        signed=True), use_container_width=True, key=f"{key}_ov2")
    elif view == "CAGR & Growth":
        d = summary[["Variable", "CAGR %", "Growth %"]].melt("Variable", var_name="Metric",
                                                            value_name="%")
        fig = px.bar(d, x="Variable", y="%", color="Metric", barmode="group", text_auto=".1f",
                     color_discrete_sequence=[C_LINE, C_ROLL],
                     title=f"{name} — CAGR vs total growth")
        fig.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_ov3")
    elif view == "Mix (share of yearly total)":
        w = yearly.set_index("Variable")[used].astype(float)
        share = (w / w.sum() * 100).T.reset_index(names="Vintage")
        fig = px.bar(share.melt("Vintage", var_name="Variable", value_name="Share %"),
                     x="Vintage", y="Share %", color="Variable", text_auto=".1f",
                     title=f"{name} — share of yearly total (%)", color_discrete_sequence=_PALETTE)
        fig.update_layout(template="plotly_white", height=420, barmode="stack")
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_ov4")


# ════════════════════════════════════════════════════════════════════
# Variable window (category or manual)
# ════════════════════════════════════════════════════════════════════
def _variable_window(label, cat, variables, ctx, cat_by_var, manual=False):
    k = f"eda_{label.lower().replace(' ', '_')}"
    default_agg = core.DEFAULT_AGG.get(cat, "Sum") if cat else "Sum"

    if manual:
        info("Pick any variables from any category — you get the same dashboard for each.")
        variables = safe_multiselect(
            "Variables", options=ctx["numeric_cols"], key=f"{k}_vars",
            format_func=lambda v: f"{v}  ·  {cat_by_var.get(v, core.UNASSIGNED)}")
    else:
        if not variables:
            st.info(f"No variables are assigned to **{label.title()}**. Assign some in "
                    f"*2 · Variable categories* above.")
            return
        variables = safe_multiselect("Variables in this window", options=variables,
                                     default=variables, key=f"{k}_vars")
    if not variables:
        st.info("Select at least one variable.")
        return

    # controls row
    c1, c2, c3 = st.columns([1.2, 1.4, 1])
    if manual:
        opts = ["Auto (Price = Average, others = Sum)", "Sum", "Average"]
    else:
        opts = [f"Default ({default_agg})", "Sum", "Average"]
    agg_choice = c1.selectbox("Yearly basis", opts, key=f"{k}_agg")
    ign_zero = c2.checkbox("Ignore zeros when averaging", value=(cat == "Price"),
                           key=f"{k}_ignz",
                           help="Treat 0 as missing for variables averaged yearly.")
    roll = c3.number_input("Rolling avg window", min_value=1, max_value=52, value=4,
                           key=f"{k}_roll")

    def _agg_for(v):
        if agg_choice in ("Sum", "Average"):
            return agg_choice
        if manual:
            return core.DEFAULT_AGG.get(cat_by_var.get(v, core.UNASSIGNED), "Sum")
        return default_agg

    agg_by_var = {v: _agg_for(v) for v in variables}
    zero_by_var = {v: ign_zero for v in variables}

    focus = st.selectbox("Variable to analyse", variables, key=f"{k}_focus") \
        if len(variables) > 1 else variables[0]
    if len(variables) == 1:
        st.markdown(f"**{focus}**")

    _variable_dashboard(focus, agg_by_var[focus], ign_zero, roll, ctx, f"{k}_dash")

    if len(variables) > 1:
        with st.expander(f"📊 Compare all {len(variables)} variables", expanded=manual):
            _overview(label.title(), variables, agg_by_var, zero_by_var, ctx, cat_by_var,
                      f"{k}_ov")


# ════════════════════════════════════════════════════════════════════
# Data-quality window
# ════════════════════════════════════════════════════════════════════
def _data_quality_window(ctx, cat_by_var):
    df, vint, vt = ctx["df"], ctx["vint"], ctx["vt"]
    cols = ctx["numeric_cols"]
    dq = core.data_quality_table(df, cols, vint)
    dq.insert(1, "Category", [cat_by_var.get(c, core.UNASSIGNED) for c in dq["Variable"]])
    gaps = core.date_gap_count(df[ctx["date_col"]], ctx["granularity"])
    dups = int(df[ctx["date_col"]].duplicated().sum())

    k1, k2, k3, k4 = st.columns(4)
    _kpi(k1, "Rows in window", f"{int(vint.notna().sum()):,}")
    _kpi(k2, "Missing periods", gaps)
    _kpi(k3, "Duplicate dates", dups)
    _kpi(k4, "Variables with issues",
         int(((dq["Missing"] > 0) | dq["Constant"] | (dq["Negative"] > 0)).sum()))

    _heading("Vintage years")
    disp = vt.copy()
    disp["Complete"] = disp["Complete"].map({True: "✅ Complete", False: "⚠️ Partial"})
    st.dataframe(disp[["Label", "Window", "Observations", "Expected", "Complete"]],
                 use_container_width=True, hide_index=True)

    _heading("Variable health")
    st.dataframe(dq, use_container_width=True, hide_index=True)
    st.download_button("⬇ EXPORT CSV", dq.to_csv(index=False).encode(),
                       "data_quality.csv", "text/csv", key="eda_dq_dl")

    fig = px.bar(dq.sort_values("Zero %"), x="Zero %", y="Variable", orientation="h",
                 color_discrete_sequence=[C_LINE])
    fig = _base_layout(fig, "% of periods equal to zero", max(300, 28 * len(dq) + 100))
    st.plotly_chart(fig, use_container_width=True, key="eda_dq_zero")


# ════════════════════════════════════════════════════════════════════
# Correlation window
# ════════════════════════════════════════════════════════════════════
def _correlation_window(ctx, cat_by_var):
    df, vint = ctx["df"], ctx["vint"]
    numeric = ctx["numeric_cols"]
    assigned = [c for c in numeric if cat_by_var.get(c) != core.UNASSIGNED] or numeric

    sel = safe_multiselect("Variables", options=numeric, default=assigned, key="eda_corr_vars",
                           format_func=lambda v: f"{v}  ·  {cat_by_var.get(v, core.UNASSIGNED)}")
    if len(sel) < 2:
        st.info("Select at least two variables.")
        return
    c1, c2 = st.columns(2)
    method = c1.selectbox("Method", ["pearson", "spearman"], key="eda_corr_method")
    sales_vars = [v for v in sel if cat_by_var.get(v) == "Sales"]
    target = c2.selectbox("Rank correlations against", sel,
                          index=sel.index(sales_vars[0]) if sales_vars else 0,
                          key="eda_corr_target")

    d = df.loc[vint.notna(), sel].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method=method)

    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r")
    fig = _base_layout(fig, "Correlation matrix", max(420, 34 * len(sel) + 150))
    st.plotly_chart(fig, use_container_width=True, key="eda_corr_heat")

    _heading(f"Correlation with {target}")
    rk = corr[target].drop(target).dropna().sort_values()
    bar = go.Figure(go.Bar(x=rk.values, y=rk.index, orientation="h",
                           marker_color=[C_POS if v >= 0 else C_NEG for v in rk.values],
                           text=[f"{v:.2f}" for v in rk.values], textposition="outside"))
    bar.update_xaxes(range=[-1, 1])
    bar = _base_layout(bar, f"Correlation with {target}", max(300, 28 * len(rk) + 110))
    st.plotly_chart(bar, use_container_width=True, key="eda_corr_bar")
    st.download_button("⬇ EXPORT MATRIX CSV", corr.round(4).to_csv().encode(),
                       "correlation_matrix.csv", "text/csv", key="eda_corr_dl")


# ════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════
def render_tab_eda():
    section("03", "EDA · Exploratory Data Analysis")
    if st.session_state.df is None:
        need_data()
    st.markdown(_CSS, unsafe_allow_html=True)

    raw = st.session_state.df.copy()

    # ────────────────────────────────────────────────────────────
    # 1 · VINTAGE SETUP
    # ────────────────────────────────────────────────────────────
    st.markdown("### 1 · Data vintage")
    info("Pick the date column, the granularity and the <b>start month</b>. The data is cut "
         "into 12-month <b>vintage years</b> from that month (start = Apr → FY Apr–Mar). "
         "Yearly totals / averages, CAGR, growth and YoY are computed on these years.")

    date_cands = core.find_date_columns(raw)
    if not date_cands:
        st.error("No date column could be detected in the uploaded data, so vintage-based "
                 "EDA can't run. Add a date column (e.g. `Date`, format YYYY-MM-DD) and re-upload.")
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
                               key="eda_gran", help=f"Auto-detected: {detected}.")
    months = core.month_options(work[date_col])
    start_month = c3.selectbox("Vintage start month", months, index=0,
                               format_func=lambda d: d.strftime("%b %Y"), key="eda_start_month")

    o1, o2, o3 = st.columns([2, 1, 1])
    include_partial = o1.checkbox("Include incomplete vintage years in growth / CAGR "
                                  "(not recommended)", value=False, key="eda_incl_partial")
    show_bands = o2.checkbox("Show vintage bands on trend charts", value=True,
                             key="eda_show_bands")
    seasonal_period = o3.number_input(
        "Seasonal period", min_value=2, max_value=400,
        value=core.DEFAULT_SEASONAL_PERIOD[granularity], key=f"eda_season_{granularity}",
        help="Periods per seasonal cycle for the decomposition "
             "(Monthly 12, Weekly 52, Daily 7).")

    vint = core.assign_vintage(work[date_col], start_month)
    vt = core.vintage_table(work[date_col], start_month, granularity)

    if work[date_col].min() > start_month + pd.Timedelta(days=31):
        st.warning("Data starts after the selected start month — the first vintage year is "
                   "partial and will be treated as incomplete.")

    m1, m2, m3, m4 = st.columns(4)
    _kpi(m1, "Date range", f"{work[date_col].min():%b-%y} → {work[date_col].max():%b-%y}")
    _kpi(m2, "Rows in window", f"{int(vint.notna().sum()):,}")
    _kpi(m3, "Vintage years", len(vt))
    _kpi(m4, "Complete years", int(vt["Complete"].sum()) if len(vt) else 0)

    if len(vt):
        disp = vt.copy()
        disp["Complete"] = disp["Complete"].map({True: "✅ Complete", False: "⚠️ Partial"})
        st.dataframe(disp[["Label", "Window", "Observations", "Expected", "Complete"]],
                     use_container_width=True, hide_index=True)

    numeric_cols = [c for c in raw.select_dtypes(include=np.number).columns if c != date_col]
    if not numeric_cols:
        st.warning("No numeric columns found in the uploaded dataset.")
        return

    with st.expander("📈 Vintage charts", expanded=False):
        pv = st.selectbox("Variable to preview", numeric_cols, key="eda_vintage_var")
        wk = work[vint.notna()].reset_index(drop=True)
        st.plotly_chart(_trend_chart(wk, date_col, pv, 0, vt, True),
                        use_container_width=True, key="eda_vint_ts")
        if len(vt):
            cov = go.Figure(go.Bar(
                x=vt["Label"], y=vt["Observations"], text=vt["Observations"],
                textposition="outside",
                marker_color=[C_POS if c else C_ROLL for c in vt["Complete"]]))
            cov.add_hline(y=core.EXPECTED_PERIODS[granularity], line_dash="dot",
                          annotation_text=f"Expected ≈ {core.EXPECTED_PERIODS[granularity]}")
            cov = _base_layout(cov, "Observations per vintage year (green = complete)", 320)
            st.plotly_chart(cov, use_container_width=True, key="eda_vint_cov")

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 2 · VARIABLE CATEGORIES
    # ────────────────────────────────────────────────────────────
    st.markdown("### 2 · Variable categories")
    info("Variables are auto-assigned from their names (and your Step 5 configuration, if "
         "saved). Edit any row — each variable belongs to one category. <b>Unassigned</b> "
         "variables stay available in Manual EDA.")

    sig = tuple(numeric_cols)
    if st.session_state.get("_eda_cat_sig") != sig:
        st.session_state["_eda_cat_sig"] = sig
        guess = core.auto_categorize(numeric_cols, st.session_state.get("config"))
        st.session_state["_eda_cat_base"] = pd.DataFrame({
            "Variable": numeric_cols, "Category": [guess[c] for c in numeric_cols]})
        st.session_state.pop("eda_cat_editor", None)

    with st.expander("🗂️ Edit variable categories", expanded=False):
        rb1, _ = st.columns([1, 4])
        if rb1.button("↺ Re-detect", key="eda_cat_reset"):
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
        _kpi(col, cat, int(counts.get(cat, 0)))

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 3 · EDA WINDOWS
    # ────────────────────────────────────────────────────────────
    st.markdown("### 3 · EDA windows")
    ctx = dict(df=work, date_col=date_col, vint=vint, vt=vt, numeric_cols=numeric_cols,
               include_partial=include_partial, granularity=granularity,
               show_bands=show_bands, seasonal_period=int(seasonal_period))

    tabs = st.tabs([w for w, _ in WINDOWS])
    for tab, (w, cat) in zip(tabs, WINDOWS):
        with tab:
            if w == "DATA QUALITY":
                _data_quality_window(ctx, cat_by_var)
            elif w == "CORRELATION":
                _correlation_window(ctx, cat_by_var)
            elif w == "MANUAL EDA":
                _variable_window("Manual", None, [], ctx, cat_by_var, manual=True)
            else:
                vars_in = [v for v in numeric_cols if cat_by_var.get(v) == cat]
                _variable_window(w, cat, vars_in, ctx, cat_by_var)

    st.divider()
    with st.expander("🔧 More tools — dependent vs. independent, spends mix, univariate OLS"):
        render_legacy_eda()
