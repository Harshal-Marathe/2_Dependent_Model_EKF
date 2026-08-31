"""
Shared result-rendering helpers.

`render_fit_and_contrib()` draws an Actual-vs-Predicted chart plus a
Short-term / Long-term / Both contribution summary for any fitted result
dict (whatever produced it — Tab 6's run_full_ekf_pipeline or Tab 8's
run_refit_pipeline — the shapes are identical). Used by Tab 8 so every
refit immediately shows its own fit + contribution results, and could be
reused anywhere else a "before/after" comparison is needed.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


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

    df_out = pd.DataFrame({
        "Channel":       names,
        "Total Contrib": np.round(values, 2),
        "Share (%)":     np.round(shares, 1),
    }).sort_values("Total Contrib", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out) + 1))
    return df_out


def render_fit_and_contrib(df, config, res, target, key_prefix=""):
    """Actual vs Predicted + Short/Long/Both contribution summary.

    df       : the full working dataframe
    config   : the config dict that produced `res` (used for n_train / test split)
    res      : a fitted result dict (mape, r2, yhat_smooth, contrib_df, ...)
    target   : the dependent-variable column name in df
    key_prefix: unique prefix so widget/chart keys don't collide across calls
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAPE", f"{res['mape']:.2%}")
    c2.metric("R²", f"{res['r2']:.4f}")
    c3.metric("Gelman R²", f"{res['r2_gelman']:.4f}")
    c4.metric("Log-Lik", f"{res['loglik']:.2f}")

    st.markdown("##### In-Sample vs. Out-of-Sample Accuracy")
    n_test = res.get("n_test", 0)
    if n_test and n_test > 0:
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("In-Sample MAPE (Train)", f"{res['mape_in']:.2%}")
        ic2.metric("In-Sample R² (Train)", f"{res['r2_in']:.4f}")
        ic3.metric("Out-of-Sample MAPE (Test)", f"{res['mape_out']:.2%}",
                   delta=f"{(res['mape_out']-res['mape_in'])*100:+.2f} pp vs train",
                   delta_color="inverse")
        ic4.metric("Out-of-Sample R² (Test)", f"{res['r2_out']:.4f}",
                   delta=f"{res['r2_out']-res['r2_in']:+.4f} vs train")
        jc1, jc2 = st.columns(2)
        jc1.metric("In-Sample Gelman R² (Train)", f"{res['r2_gelman_in']:.4f}")
        jc2.metric("Out-of-Sample Gelman R² (Test)", f"{res['r2_gelman_out']:.4f}",
                   delta=f"{res['r2_gelman_out']-res['r2_gelman_in']:+.4f} vs train")
        st.caption(
            f"Train: {res['n_train']} obs · Test (holdout): {n_test} obs. "
            "A much larger gap between train and test metrics is a sign of overfitting."
        )
    else:
        st.caption(
            "No holdout/test period configured — all figures above are in-sample "
            "(fit) metrics only. Set a train/test split (Tab 5) to see genuine "
            "out-of-sample accuracy."
        )

    st.markdown("##### Actual vs Predicted")
    n_train = config.get("n_train")
    x_axis = np.arange(len(df))
    fig = go.Figure()
    if n_train is not None and 0 < n_train < len(df):
        fig.add_vrect(x0=n_train, x1=len(df) - 1, fillcolor="#fef3c7", opacity=0.35,
                      layer="below", line_width=0,
                      annotation_text="Test period", annotation_position="top left")
    fig.add_trace(go.Scatter(x=x_axis, y=df[target].values,
                              name="Actual", line=dict(color="#1e40af", width=2)))
    fig.add_trace(go.Scatter(x=x_axis, y=res["yhat_smooth"],
                              name="Fitted (RBE Smoothed)",
                              line=dict(color="#f59e0b", width=2, dash="dash")))
    fig.update_layout(height=380, template="plotly_white",
                       title="Actual vs Fitted", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}avp")

    st.markdown("##### Channel Contributions")
    contrib_df = res["contrib_df"]
    short_cols = [c for c in contrib_df.columns if c.startswith("ShortTerm_")]
    long_cols  = [c for c in contrib_df.columns if c.startswith("LongTerm_")]
    totals_st  = contrib_df[short_cols].sum()
    totals_lt  = contrib_df[long_cols].sum()

    view = st.radio(
        "Contribution view", ["Short-term", "Long-term", "Both (combined)"],
        horizontal=True, key=f"{key_prefix}contrib_view",
    )

    if view == "Short-term":
        df_out = _contribution_table(totals_st, "ShortTerm_")
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        fig_bar = px.bar(df_out.sort_values("Total Contrib"), x="Total Contrib", y="Channel",
                          orientation="h", title="Short-term Contribution",
                          color="Total Contrib", color_continuous_scale="Blues")
        fig_bar.update_layout(height=max(280, 35 * len(df_out)), template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}st_bar")

    elif view == "Long-term":
        df_out = _contribution_table(totals_lt, "LongTerm_")
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        fig_bar = px.bar(df_out.sort_values("Total Contrib"), x="Total Contrib", y="Channel",
                          orientation="h", title="Long-term Contribution",
                          color="Total Contrib", color_continuous_scale="Greens")
        fig_bar.update_layout(height=max(280, 35 * len(df_out)), template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}lt_bar")

    else:  # Both (combined)
        st_map = {c.replace("ShortTerm_", ""): v for c, v in totals_st.items()}
        lt_map = {c.replace("LongTerm_", ""): v for c, v in totals_lt.items()}
        names = sorted(set(st_map) | set(lt_map))
        combo = pd.DataFrame({
            "Channel":    names,
            "Short-Term": [st_map.get(n, 0.0) for n in names],
            "Long-Term":  [lt_map.get(n, 0.0) for n in names],
        })
        combo["Total"] = combo["Short-Term"] + combo["Long-Term"]
        combo = combo.sort_values("Total", ascending=False).reset_index(drop=True)
        combo.insert(0, "Rank", range(1, len(combo) + 1))
        st.dataframe(combo.round(2), use_container_width=True, hide_index=True)

        plot_df = combo.sort_values("Total")
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(y=plot_df["Channel"], x=plot_df["Short-Term"],
                                    name="Short-Term", orientation="h", marker_color="#3b82f6"))
        fig_combo.add_trace(go.Bar(y=plot_df["Channel"], x=plot_df["Long-Term"],
                                    name="Long-Term", orientation="h", marker_color="#10b981"))
        fig_combo.update_layout(barmode="stack", height=max(280, 35 * len(plot_df)),
                                 template="plotly_white",
                                 title="Short + Long-Term Contribution (combined)",
                                 legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_combo, use_container_width=True, key=f"{key_prefix}combo_bar")
