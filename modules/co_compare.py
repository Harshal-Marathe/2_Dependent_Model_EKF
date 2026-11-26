"""
With vs without Cochrane-Orcutt — side-by-side comparison.

When the model is fitted with CO ON, the pipeline also keeps the ordinary
(rho = 0) fit that is iteration 1 of the CO loop and runs it through the same
filter / smoother / post-processing (result["co_baseline"]). This module turns
the two result dicts into comparison tables and charts so you can see exactly
what CO changed: fit quality, residual autocorrelation, how the KPI is split
between baseline and each channel, ROI, and the fitted parameters.

Data builders are plain pandas (testable); `render_co_comparison` is the
Streamlit/plotly view.
"""
import numpy as np
import pandas as pd

from modules.cochrane_orcutt import estimate_ar1_rho, durbin_watson


# ── data ─────────────────────────────────────────────────────────────
def _train_resid(res):
    n = int(res.get("n_train") or len(res["residuals"]))
    e = np.asarray(res["residuals"], dtype=float)[:n]
    return e[1:]          # t = 0 only reflects the initial-state guess


LBL_CO = ("Without CO", "With CO")


def metrics_table(base, co, lbl=LBL_CO, include_rho=True):
    """Fit quality / autocorrelation, left (base) vs right (co) result."""
    def pct(x):
        return None if x is None or not np.isfinite(x) else float(x) * 100
    rows = [
        ("In-sample MAPE (%)",      pct(base["mape_in"]),  pct(co["mape_in"]),  True),
        ("Out-of-sample MAPE (%)",  pct(base["mape_out"]), pct(co["mape_out"]), True),
        ("In-sample R²",            base["r2_in"],  co["r2_in"],  False),
        ("Out-of-sample R²",        base["r2_out"], co["r2_out"], False),
        ("R² (Gelman)",             base["r2_gelman"], co["r2_gelman"], False),
    ]
    eb, ec = _train_resid(base), _train_resid(co)
    rows += [
        ("Residual lag-1 autocorrelation (train)", estimate_ar1_rho(eb), estimate_ar1_rho(ec), False),
        ("Durbin-Watson of residuals (train)",     durbin_watson(eb),    durbin_watson(ec),    False),
    ]
    if include_rho:
        rows.append(("AR(1) ρ used in the fit",
                     float(base["g"].get("CO_RHO", 0.0) or 0.0),
                     float(co["g"].get("CO_RHO", 0.0) or 0.0), False))
    out = pd.DataFrame(rows, columns=["Metric", lbl[0], lbl[1], "_lower_better"])
    out["Change"] = out[lbl[1]].astype(float) - out[lbl[0]].astype(float)
    return out.drop(columns="_lower_better")


def _components(res):
    """Total contribution per component (short + long term merged)."""
    cd = res["contrib_df"]
    tgt = cd.columns[0]
    comp = {}
    for c in cd.columns[1:]:
        name = c.split("_", 1)[1] if c.startswith(("ShortTerm_", "LongTerm_")) else c
        if name.startswith("Intercept"):
            name = "Baseline (intercept)"
        comp[name] = comp.get(name, 0.0) + float(np.nansum(cd[c].values))
    return comp, float(np.nansum(cd[tgt].values))


def decomposition_table(base, co, lbl=LBL_CO):
    cb, tb = _components(base)
    cc, tc = _components(co)
    names = list(dict.fromkeys(list(cb) + list(cc)))
    rows = []
    for n in names:
        b, c = cb.get(n, 0.0), cc.get(n, 0.0)
        sb = b / tb * 100 if tb else np.nan
        sc = c / tc * 100 if tc else np.nan
        rows.append({"Component": n,
                     f"Contribution ({lbl[0]})": b, f"Contribution ({lbl[1]})": c,
                     f"Share % ({lbl[0]})": sb, f"Share % ({lbl[1]})": sc,
                     "Δ share (pp)": sc - sb})
    out = pd.DataFrame(rows)
    out["_k"] = out[[f"Share % ({lbl[0]})", f"Share % ({lbl[1]})"]].abs().max(axis=1)
    return out.sort_values("_k", ascending=False).drop(columns="_k").reset_index(drop=True)


def roi_table(base, co, lbl=LBL_CO):
    a, b = f"ROI ({lbl[0]})", f"ROI ({lbl[1]})"
    m = base["roi_df"][["Channel", "ROI"]].merge(
        co["roi_df"][["Channel", "ROI"]], on="Channel", how="outer",
        suffixes=(f" ({lbl[0]})", f" ({lbl[1]})"))
    m["Δ ROI"] = m[b] - m[a]
    m["Δ ROI %"] = np.where(m[a].abs() > 1e-12, m["Δ ROI"] / m[a].abs() * 100, np.nan)
    return m


def param_table(base, co, lbl=LBL_CO):
    k = ["Category", "Variable", "Parameter"]
    a, b = f"Value ({lbl[0]})", f"Value ({lbl[1]})"
    m = base["param_df"][k + ["Value"]].merge(
        co["param_df"][k + ["Value"]], on=k, how="outer",
        suffixes=(f" ({lbl[0]})", f" ({lbl[1]})"))
    for c in (a, b):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["Δ"] = m[b] - m[a]
    den = m[a].abs().where(lambda s: s > 1e-9)
    m["Δ %"] = m["Δ"] / den * 100
    return m.reindex(m["Δ %"].abs().sort_values(ascending=False, na_position="last").index
                     ).reset_index(drop=True)


# ── view ─────────────────────────────────────────────────────────────
def render_pair(base, res, lbl, key_prefix="", x=None, include_rho=True, file_stub="comparison"):
    """Tables + charts comparing two results of the same dependent variable."""
    import streamlit as st
    import plotly.graph_objects as go

    a, b = lbl
    st.markdown("**Fit quality & residual autocorrelation**")
    mt = metrics_table(base, res, lbl, include_rho)
    st.dataframe(mt.style.format({a: "{:.4f}", b: "{:.4f}", "Change": "{:+.4f}"}, na_rep="–"),
                 use_container_width=True, hide_index=True)

    dec = decomposition_table(base, res, lbl)
    st.markdown("**Contribution — how the KPI is split**")
    st.dataframe(dec.style.format({
        f"Contribution ({a})": "{:,.0f}", f"Contribution ({b})": "{:,.0f}",
        f"Share % ({a})": "{:.2f}", f"Share % ({b})": "{:.2f}",
        "Δ share (pp)": "{:+.2f}"}, na_rep="–"),
        use_container_width=True, hide_index=True)
    top = dec.head(12)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top["Component"], y=top[f"Share % ({a})"], name=a,
                         marker_color="#9ca3af"))
    fig.add_trace(go.Bar(x=top["Component"], y=top[f"Share % ({b})"], name=b,
                         marker_color="#4f46e5"))
    fig.update_layout(barmode="group", height=380, title="Share of KPI by component (%)",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}cmp_share")

    st.markdown("**ROI**")
    rt = roi_table(base, res, lbl)
    st.dataframe(rt.style.format({
        f"ROI ({a})": "{:.4f}", f"ROI ({b})": "{:.4f}",
        "Δ ROI": "{:+.4f}", "Δ ROI %": "{:+.1f}%"}, na_rep="–"),
        use_container_width=True, hide_index=True)

    tgt = res["contrib_df"].columns[0]
    xs = x if x is not None and len(x) == len(res["yhat_smooth"]) else np.arange(len(res["yhat_smooth"]))
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=xs, y=res["contrib_df"][tgt].values, name="Actual",
                            line=dict(color="#111827", width=2)))
    f2.add_trace(go.Scatter(x=xs, y=base["yhat_smooth"], name=f"Fitted · {a}",
                            line=dict(color="#9ca3af", width=2, dash="dot")))
    f2.add_trace(go.Scatter(x=xs, y=res["yhat_smooth"], name=f"Fitted · {b}",
                            line=dict(color="#4f46e5", width=2)))
    f2.update_layout(height=380, title="Actual vs fitted", margin=dict(l=10, r=10, t=50, b=10),
                     legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(f2, use_container_width=True, key=f"{key_prefix}cmp_fit")

    with st.expander("Parameter changes (largest relative change first)"):
        pt = param_table(base, res, lbl)
        st.dataframe(pt.style.format({
            f"Value ({a})": "{:.5g}", f"Value ({b})": "{:.5g}",
            "Δ": "{:+.5g}", "Δ %": "{:+.1f}%"}, na_rep="–"),
            use_container_width=True, hide_index=True, height=420)

    buf = []
    for name, d in (("metrics", mt), ("contribution", dec), ("roi", rt)):
        d2 = d.copy(); d2.insert(0, "Table", name); buf.append(d2)
    st.download_button("⬇ Download comparison (CSV)",
                       pd.concat(buf, ignore_index=True).to_csv(index=False).encode(),
                       f"{file_stub}.csv", "text/csv", key=f"{key_prefix}cmp_dl")


def render_co_comparison(res, key_prefix="", x=None, expanded=False):
    """Expander showing the with-vs-without-CO comparison for one result.
    No-op if the model wasn't fitted with CO (no baseline stored)."""
    import streamlit as st

    base = res.get("co_baseline")
    if not base or not res.get("co_info"):
        return
    rho = float(res["co_info"]["rho"])
    with st.expander("🔁 Cochrane-Orcutt impact — model with vs without CO", expanded=expanded):
        st.caption(
            f"“Without CO” is the ordinary fit (ρ = 0, iteration 1 of the CO loop, same "
            f"data/config/start values). “With CO” is the final model fitted with ρ = {rho:.3f}. "
            "Everything is on the original (non-differenced) scale. CO's job is to reduce "
            "autocorrelation in the errors, so look at the lag-1 autocorrelation moving toward 0 "
            "(Durbin-Watson toward 2). Log-likelihood isn't compared: the CO likelihood is on the "
            "whitened scale, so the two aren't comparable.")
        render_pair(base, res, LBL_CO, key_prefix=f"{key_prefix}co_", x=x,
                    include_rho=True, file_stub="cochrane_orcutt_comparison")
