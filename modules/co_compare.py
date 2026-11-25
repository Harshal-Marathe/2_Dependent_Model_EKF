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


def metrics_table(base, co):
    """Fit quality / autocorrelation, without vs with CO."""
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
        ("AR(1) ρ used in the fit", 0.0, float(co["g"].get("CO_RHO", 0.0) or 0.0), False),
    ]
    out = pd.DataFrame(rows, columns=["Metric", "Without CO", "With CO", "_lower_better"])
    out["Change"] = out["With CO"].astype(float) - out["Without CO"].astype(float)
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


def decomposition_table(base, co):
    cb, tb = _components(base)
    cc, tc = _components(co)
    names = list(dict.fromkeys(list(cb) + list(cc)))
    rows = []
    for n in names:
        b, c = cb.get(n, 0.0), cc.get(n, 0.0)
        sb = b / tb * 100 if tb else np.nan
        sc = c / tc * 100 if tc else np.nan
        rows.append({"Component": n,
                     "Contribution (without CO)": b, "Contribution (with CO)": c,
                     "Share % (without CO)": sb, "Share % (with CO)": sc,
                     "Δ share (pp)": sc - sb})
    out = pd.DataFrame(rows)
    out["_k"] = out[["Share % (without CO)", "Share % (with CO)"]].abs().max(axis=1)
    return out.sort_values("_k", ascending=False).drop(columns="_k").reset_index(drop=True)


def roi_table(base, co):
    m = base["roi_df"][["Channel", "ROI"]].merge(
        co["roi_df"][["Channel", "ROI"]], on="Channel", how="outer",
        suffixes=(" (without CO)", " (with CO)"))
    m["Δ ROI"] = m["ROI (with CO)"] - m["ROI (without CO)"]
    m["Δ ROI %"] = np.where(m["ROI (without CO)"].abs() > 1e-12,
                            m["Δ ROI"] / m["ROI (without CO)"].abs() * 100, np.nan)
    return m


def param_table(base, co):
    k = ["Category", "Variable", "Parameter"]
    m = base["param_df"][k + ["Value"]].merge(
        co["param_df"][k + ["Value"]], on=k, how="outer",
        suffixes=(" (without CO)", " (with CO)"))
    for c in ("Value (without CO)", "Value (with CO)"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["Δ"] = m["Value (with CO)"] - m["Value (without CO)"]
    den = m["Value (without CO)"].abs().where(lambda s: s > 1e-9)
    m["Δ %"] = m["Δ"] / den * 100
    return m.reindex(m["Δ %"].abs().sort_values(ascending=False, na_position="last").index
                     ).reset_index(drop=True)


# ── view ─────────────────────────────────────────────────────────────
def render_co_comparison(res, key_prefix="", x=None, expanded=False):
    """Expander showing the with-vs-without-CO comparison for one result.
    No-op if the model wasn't fitted with CO (no baseline stored)."""
    import streamlit as st
    import plotly.graph_objects as go

    base = res.get("co_baseline")
    if not base or not res.get("co_info"):
        return
    rho = float(res["co_info"]["rho"])
    with st.expander("🔁 Cochrane-Orcutt impact — model with vs without CO", expanded=expanded):
        st.caption(
            f"“Without CO” is the ordinary fit (ρ = 0, iteration 1 of the CO loop, same "
            f"data/config/start values). “With CO” is the final model fitted with ρ = {rho:.3f}. "
            "Everything is on the original (non-differenced) scale.")

        # 1 · fit quality
        st.markdown("**Fit quality & residual autocorrelation**")
        mt = metrics_table(base, res)
        st.dataframe(mt.style.format({"Without CO": "{:.4f}", "With CO": "{:.4f}",
                                      "Change": "{:+.4f}"}, na_rep="–"),
                     use_container_width=True, hide_index=True)
        st.caption("CO's job is to reduce autocorrelation in the errors, so look at the "
                   "lag-1 autocorrelation moving toward 0 (Durbin-Watson toward 2). Accuracy "
                   "(MAPE / R²) may go up or down slightly. Log-likelihood isn't compared: "
                   "the CO likelihood is on the whitened scale, so the two aren't comparable.")

        # 2 · where the KPI is attributed
        dec = decomposition_table(base, res)
        st.markdown("**Contribution — how CO shifts the split of the KPI**")
        st.dataframe(dec.style.format({
            "Contribution (without CO)": "{:,.0f}", "Contribution (with CO)": "{:,.0f}",
            "Share % (without CO)": "{:.2f}", "Share % (with CO)": "{:.2f}",
            "Δ share (pp)": "{:+.2f}"}, na_rep="–"),
            use_container_width=True, hide_index=True)
        top = dec.head(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top["Component"], y=top["Share % (without CO)"],
                             name="Without CO", marker_color="#9ca3af"))
        fig.add_trace(go.Bar(x=top["Component"], y=top["Share % (with CO)"],
                             name="With CO", marker_color="#4f46e5"))
        fig.update_layout(barmode="group", height=380, title="Share of KPI by component (%)",
                          margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}co_cmp_share")

        # 3 · ROI
        st.markdown("**ROI**")
        rt = roi_table(base, res)
        st.dataframe(rt.style.format({
            "ROI (without CO)": "{:.4f}", "ROI (with CO)": "{:.4f}",
            "Δ ROI": "{:+.4f}", "Δ ROI %": "{:+.1f}%"}, na_rep="–"),
            use_container_width=True, hide_index=True)

        # 4 · fitted vs actual
        tgt = res["contrib_df"].columns[0]
        xs = x if x is not None and len(x) == len(res["yhat_smooth"]) else np.arange(len(res["yhat_smooth"]))
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=xs, y=res["contrib_df"][tgt].values, name="Actual",
                                line=dict(color="#111827", width=2)))
        f2.add_trace(go.Scatter(x=xs, y=base["yhat_smooth"], name="Fitted · without CO",
                                line=dict(color="#9ca3af", width=2, dash="dot")))
        f2.add_trace(go.Scatter(x=xs, y=res["yhat_smooth"], name="Fitted · with CO",
                                line=dict(color="#4f46e5", width=2)))
        f2.update_layout(height=380, title="Actual vs fitted", margin=dict(l=10, r=10, t=50, b=10),
                         legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(f2, use_container_width=True, key=f"{key_prefix}co_cmp_fit")

        # 5 · parameters
        with st.expander("Parameter changes (largest relative change first)"):
            pt = param_table(base, res)
            st.dataframe(pt.style.format({
                "Value (without CO)": "{:.5g}", "Value (with CO)": "{:.5g}",
                "Δ": "{:+.5g}", "Δ %": "{:+.1f}%"}, na_rep="–"),
                use_container_width=True, hide_index=True, height=420)

        # download
        buf = []
        for name, d in (("metrics", mt), ("contribution", dec), ("roi", rt)):
            d2 = d.copy(); d2.insert(0, "Table", name); buf.append(d2)
        st.download_button("⬇ Download comparison (CSV)",
                           pd.concat(buf, ignore_index=True).to_csv(index=False).encode(),
                           "cochrane_orcutt_comparison.csv", "text/csv",
                           key=f"{key_prefix}co_cmp_dl")
