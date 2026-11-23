"""
Shared per-channel hyperparameter bounds widget — used by Tab 5
(Configuration, Dependent 1 & 2) and Tab 8 (Refine & Refit, for the bounds
of a newly added or re-opened variable). Extracted so both tabs render the
exact same widgets/keys/defaults instead of drifting apart.
"""

import pandas as pd
import streamlit as st


def render_per_channel_bounds(channel_cols, comp_cols, key_prefix, df,
                               use_hill, adstock_map,
                               price_cols=None, nonmedia_cols=None):
    """
    Renders per-channel bound widgets for media/comp-media columns (with
    adstock + transformation bounds) and optionally simplified Ls+delta
    bounds for price and non-media/control columns.

    `adstock_map`: either a plain bool (applied to every channel — kept for
    backward compatibility with old call sites) or a dict {col: "weibull"|
    "instant"} giving each channel's OWN adstock choice. Any channel not
    covered by an eligible group (currently price) never shows adstock
    controls.

    Returns a dict {col: {param: (lo, hi), ...}}.
    """
    price_cols     = price_cols or []
    nonmedia_cols  = nonmedia_cols or []
    bounds: dict   = {}

    def _is_weibull(col):
        if isinstance(adstock_map, dict):
            return adstock_map.get(col) == "weibull"
        return bool(adstock_map)

    # ── Media / competitor-media channels ─────────────────────────
    all_adstock_cols = list(channel_cols)
    if all_adstock_cols:
        st.markdown("#### Own Media & Competitor Media")
        for col in all_adstock_cols:
            is_comp  = col in comp_cols
            col_label = f"📉 {col}  *(competitor)*" if is_comp else f"📺 {col}"
            vals    = df[col][df[col] > 0] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_p90 = float(vals.quantile(0.90)) if len(vals) > 0 else 1.0
            col_max = float(vals.max())    if len(vals) > 0 else 1.0
            def_s_lo = max(1e-6, col_med * 0.01)
            def_s_hi = col_max * 5.0

            with st.expander(col_label, expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · P90: **{col_p90:,.2f}** · "
                           f"max: **{col_max:,.2f}**")
                bounds[col] = {}

                # Media input type — Spend vs GRP/Impressions. ROI is
                # Total Contribution ÷ Total Spend; if this channel's raw
                # column is GRP/impressions rather than currency, summing
                # it directly is meaningless as "spend", so the actual
                # spend column is picked here and its TOTAL is used as
                # the ROI denominator instead (see modules/pipeline.py).
                if not is_comp:
                    st.markdown("**Media Input Type**")
                    input_type = st.radio(
                        "Is this channel's data Spend or GRP / Impressions?",
                        ["Spend", "GRP / Impressions"],
                        horizontal=True,
                        key=f"{key_prefix}input_type_{col}",
                        help="Leave as **Spend** if this column is currency — its own "
                             "total is used for ROI as before. Pick **GRP / Impressions** "
                             "if this column is not currency; you'll then pick the actual "
                             "spend column below, and ITS total (not this column's total) "
                             "is used as the ROI denominator for this channel.",
                    )
                    if input_type == "GRP / Impressions":
                        spend_options = [c for c in df.columns if c != col]
                        if spend_options:
                            spend_col = st.selectbox(
                                f"Spend column for {col}'s ROI",
                                spend_options,
                                key=f"{key_prefix}spend_col_{col}",
                            )
                            bounds[col]["__spend_col__"] = spend_col
                            st.caption(
                                f"💰 ROI for **{col}** = Total Contribution ÷ Total "
                                f"**{spend_col}** (not ÷ Total {col})."
                            )
                        else:
                            st.warning("No other numeric column available to use as the spend column.")
                    st.divider()

                # Beta persistence
                st.markdown("**Beta Persistence (Ls)**")
                lsc1, lsc2 = st.columns(2)
                with lsc1:
                    ls_lo = st.number_input("Ls min", 0.0, 1.0, 0.2, 0.01,
                                             key=f"{key_prefix}ls_lo_{col}")
                with lsc2:
                    ls_hi = st.number_input("Ls max", 0.0, 1.0, 0.8, 0.01,
                                             key=f"{key_prefix}ls_hi_{col}")
                bounds[col]["ls"] = (ls_lo, ls_hi)
                st.divider()

                # Transformation parameters
                if use_hill:
                    st.markdown("**Hill Adstock** — a_t = x_t + adstock·a_(t-1)  (range 0.1 – 0.9)")
                    hac1, hac2 = st.columns(2)
                    with hac1:
                        ha_lo = st.number_input("adstock min", 0.1, 0.9, 0.1, 0.05,
                                                 key=f"{key_prefix}ha_lo_{col}")
                    with hac2:
                        ha_hi = st.number_input("adstock max", 0.1, 0.9, 0.9, 0.05,
                                                 key=f"{key_prefix}ha_hi_{col}")
                    bounds[col]["hill_adstock"] = (ha_lo, ha_hi)
                    st.divider()

                    st.markdown("**Hill Alpha** — curve shape; above 1 gives an S-curve  (range 0.5 – 3.0)")
                    hnc1, hnc2 = st.columns(2)
                    with hnc1:
                        hn_lo = st.number_input("alpha min", 0.5, 3.0, 0.5, 0.1,
                                                 key=f"{key_prefix}hn_lo_{col}")
                    with hnc2:
                        hn_hi = st.number_input("alpha max", 0.5, 3.0, 3.0, 0.1,
                                                 key=f"{key_prefix}hn_hi_{col}")
                    bounds[col]["hill_n"] = (hn_lo, hn_hi)
                    st.divider()

                    st.markdown("**Hill Gamma** — half-saturation as a fraction of max(adstocked series)  (range 0.1 – 0.9)")
                    hsc1, hsc2 = st.columns(2)
                    with hsc1:
                        hs_lo = st.number_input("gamma min", 0.1, 0.9, 0.1, 0.05,
                                                 key=f"{key_prefix}hs_lo_{col}")
                    with hsc2:
                        hs_hi = st.number_input("gamma max", 0.1, 0.9, 0.9, 0.05,
                                                 key=f"{key_prefix}hs_hi_{col}")
                    bounds[col]["hill_s"] = (hs_lo, hs_hi)
                else:
                    st.markdown("**Power Exponent (n)** — n ∈ (0,1] for diminishing returns")
                    pnc1, pnc2 = st.columns(2)
                    with pnc1:
                        pn_lo = st.number_input("n min", 0.001, 1.0, 0.01, 0.01,
                                                 key=f"{key_prefix}pn_lo_{col}")
                    with pnc2:
                        pn_hi = st.number_input("n max", 0.001, 1.0, 1.0, 0.01,
                                                 key=f"{key_prefix}pn_hi_{col}")
                    bounds[col]["transform_n"] = (pn_lo, pn_hi)
                st.divider()

                # Adstock parameters
                if _is_weibull(col):
                    st.markdown("**Weibull Adstock — Shape (k) and Scale (λ)**")
                    st.caption(
                        "Shape k controls the weight distribution across lags "
                        "(k < 1: front-loaded, k = 1: exponential, k > 1: bell-shaped). "
                        "Scale λ controls how quickly weights decay."
                    )
                    wa1, wa2, wa3, wa4 = st.columns(4)
                    with wa1: wa_klo = st.number_input("k min", 0.01, 10.0, 0.1, 0.1,
                                                        key=f"{key_prefix}wa_klo_{col}")
                    with wa2: wa_khi = st.number_input("k max", 0.01, 10.0, 5.0, 0.1,
                                                        key=f"{key_prefix}wa_khi_{col}")
                    with wa3: wl_lo  = st.number_input("λ min", 0.01, 10.0, 0.1, 0.1,
                                                        key=f"{key_prefix}wl_lo_{col}")
                    with wa4: wl_hi  = st.number_input("λ max", 0.01, 10.0, 5.0, 0.1,
                                                        key=f"{key_prefix}wl_hi_{col}")
                    bounds[col]["adstock_shape"] = (wa_klo, wa_khi)
                    bounds[col]["adstock_scale"] = (wl_lo,  wl_hi)
                else:
                    st.caption(
                        "ℹ️ Instant adstock has no separate λ decay parameter — "
                        "carryover is carried entirely by this channel's **Beta "
                        "Persistence (Ls)** above. A separate adstock λ on top of "
                        "Ls would double-count the same carryover."
                    )

                _shown_params = [p for p in bounds[col].keys() if not p.startswith("__")]
                st.caption(f"✅ Bounds set for: {', '.join(_shown_params)}")
    else:
        st.info("Select media channels in Section A first.")

    # ── Price columns ──────────────────────────────────────────────
    if price_cols:
        st.markdown("#### Price Variables")
        for col in price_cols:
            vals    = df[col][df[col] > 0] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_max = float(vals.max())    if len(vals) > 0 else 1.0
            with st.expander(f"💲 {col}", expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · max: **{col_max:,.2f}**")
                bounds[col] = {}
                st.markdown("**Beta Persistence (Ls_price)**")
                pc1, pc2 = st.columns(2)
                with pc1:
                    p_ls_lo = st.number_input("Ls min", 0.0, 1.0, 0.2, 0.01,
                                               key=f"{key_prefix}p_ls_lo_{col}")
                with pc2:
                    p_ls_hi = st.number_input("Ls max", 0.0, 1.0, 0.8, 0.01,
                                               key=f"{key_prefix}p_ls_hi_{col}")
                bounds[col]["ls"] = (p_ls_lo, p_ls_hi)
                st.caption(f"✅ Bounds set for: {', '.join(bounds[col].keys())}")

    # ── Non-media / control columns ────────────────────────────────
    if nonmedia_cols:
        st.markdown("#### Non-Media / Control Variables")
        for col in nonmedia_cols:
            vals    = df[col] if col in df.columns else pd.Series([1.0])
            col_med = float(vals.median()) if len(vals) > 0 else 1.0
            col_max = float(abs(vals).max()) if len(vals) > 0 else 1.0
            with st.expander(f"🗂️ {col}", expanded=False):
                st.caption(f"📊 median: **{col_med:,.2f}** · max: **{col_max:,.2f}**")
                bounds[col] = {}
                st.markdown("**Beta Persistence (Ls_nonmedia)**")
                nm1, nm2 = st.columns(2)
                with nm1:
                    nm_ls_lo = st.number_input("Ls min", 0.0, 1.0, 0.2, 0.01,
                                                key=f"{key_prefix}nm_ls_lo_{col}")
                with nm2:
                    nm_ls_hi = st.number_input("Ls max", 0.0, 1.0, 0.8, 0.01,
                                                key=f"{key_prefix}nm_ls_hi_{col}")
                bounds[col]["ls"] = (nm_ls_lo, nm_ls_hi)
                st.divider()

                if _is_weibull(col):
                    st.markdown("**Weibull Adstock — Shape (k) and Scale (λ)**")
                    st.caption(
                        "Shape k controls the weight distribution across lags "
                        "(k < 1: front-loaded, k = 1: exponential, k > 1: bell-shaped). "
                        "Scale λ controls how quickly weights decay."
                    )
                    nwa1, nwa2, nwa3, nwa4 = st.columns(4)
                    with nwa1: nwa_klo = st.number_input("k min", 0.01, 10.0, 0.1, 0.1,
                                                          key=f"{key_prefix}nwa_klo_{col}")
                    with nwa2: nwa_khi = st.number_input("k max", 0.01, 10.0, 5.0, 0.1,
                                                          key=f"{key_prefix}nwa_khi_{col}")
                    with nwa3: nwl_lo  = st.number_input("λ min", 0.01, 10.0, 0.1, 0.1,
                                                          key=f"{key_prefix}nwl_lo_{col}")
                    with nwa4: nwl_hi  = st.number_input("λ max", 0.01, 10.0, 5.0, 0.1,
                                                          key=f"{key_prefix}nwl_hi_{col}")
                    bounds[col]["adstock_shape"] = (nwa_klo, nwa_khi)
                    bounds[col]["adstock_scale"] = (nwl_lo,  nwl_hi)
                else:
                    st.caption(
                        "ℹ️ Instant adstock has no separate λ decay parameter — "
                        "carryover is carried entirely by this channel's **Beta "
                        "Persistence (Ls)** above."
                    )
                st.caption(f"✅ Bounds set for: {', '.join(k for k in bounds[col].keys() if not k.startswith('__'))}")

    return bounds
