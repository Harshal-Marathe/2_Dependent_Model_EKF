"""
Tab 6 — Run RBE Model: MCMC (NUTS) settings and kicking off the full pipeline.
"""

import streamlit as st

from modules.ui_helpers import section, info, ng_info, prophet_info, need_data, need_config
from modules.pipeline import run_multi_dependent_pipeline, run_chained_dependent_pipeline


def render_tab5(nevergrad_available: bool = False):
    section("05", "Run RBE Model")
    if st.session_state.df is None:     need_data()
    if st.session_state.config is None: need_config()

    config = st.session_state.config
    if "media" not in config:
        st.error("Configuration is incomplete — please re-save it in **Tab 5**.")
        st.stop()

    df = st.session_state.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Media channels", len(config["media"]))
    c2.metric("Price vars",     len(config.get("price", [])))
    adstock_map = config.get("adstock_map", {})
    if adstock_map:
        n_w = sum(1 for v in adstock_map.values() if v == "weibull")
        n_i = len(adstock_map) - n_w
        adstock_label = f"Mixed ({n_w}W/{n_i}I)" if (n_w and n_i) else ("Weibull" if n_w else "Instant")
    else:
        adstock_label = config.get("adstock_type", "instant").title()
    combo = f"{adstock_label} × {config.get('transform_type','hill').title()}"
    c3.metric("Adstock×Transform", combo)
    c4.metric("Train / Test",   f"{config['n_train']} / {config['n_test']}")

    pb_cols = config.get("positive_beta_cols", [])
    pcb     = config.get("per_channel_bounds", {})
    if pb_cols: st.info(f"🔒 Positive-beta: {', '.join(pb_cols)}")
    if pcb:
        n_pcb = sum(len(v) for v in pcb.values())
        st.info(f"🎛️ Per-variable bounds: {len(pcb)} channel(s), {n_pcb} params")

    prophet_in_model = [c for c in config.get("non_media", []) if c.startswith("prophet_")]
    if prophet_in_model:
        prophet_info(f"📌 Prophet control variables in model: <b>{', '.join(prophet_in_model)}</b>")

    # Validate all config columns exist in df
    dep2_active = config.get("enable_second_dependent") and config.get("target2")
    dep2_cols = (
        config.get("media_2", []) + config.get("non_media_2", []) +
        config.get("comp_media_2", []) + config.get("comp_nonmedia_2", []) +
        config.get("price_2", [])
    ) if dep2_active else []
    missing_cols = [
        col for col in (
            config.get("media", []) + config.get("non_media", []) +
            config.get("comp_media", []) + config.get("comp_nonmedia", []) +
            config.get("price", []) + dep2_cols +
            [config.get("target", "")] +
            ([config["target2"]] if dep2_active else [])
        )
        if col and col not in df.columns
    ]
    if missing_cols:
        st.error(
            f"❌ Columns in saved config are missing from dataset: "
            f"`{'`, `'.join(missing_cols)}`. Re-save configuration in Tab 5."
        )
        st.stop()

    dep_relationship = config.get("dependent_relationship", "joint")
    if config.get("enable_second_dependent") and config.get("target2"):
        if dep_relationship == "chained":
            st.info(
                f"➡️ **Chained mode**: Dependent 2 (`{config['target2']}`) will be fitted "
                f"**on its own first**, then its "
                f"{'fitted' if config.get('chain_use_fitted', True) else 'raw actual'} values "
                f"will be added as a new **{config.get('chain_driver_role','non_media').replace('_',' ')}** "
                f"predictor driving Dependent 1 (`{config['target']}`) — two separate MCMC "
                f"runs, connected only through that one new column."
            )
        else:
            st.info(
                f"➕ **Joint (bivariate) mode**: Dependent 1 (`{config['target']}`) and "
                f"Dependent 2 (`{config['target2']}`) will be fitted **together** in a "
                f"single bivariate state-space posterior — one MCMC run over both equations' "
                f"parameters plus the correlation (ρ) between their errors, rather than "
                f"two separate independent fits."
            )
        if config.get("different_predictors_2"):
            st.caption(
                f"🔀 Dependent 2 uses its own predictor set: "
                f"{len(config.get('media_2', []))} media · {len(config.get('non_media_2', []))} non-media · "
                f"{len(config.get('price_2', []))} price · {len(config.get('comp_media_2', []))} comp-media · "
                f"{len(config.get('comp_nonmedia_2', []))} comp-non-media."
            )

    st.divider()
    st.markdown("### MCMC Settings (NUTS)")
    ng_info(
        "🟣 <b>MCMC / NUTS</b><br>"
        "Parameters <i>and</i> the full latent state path are sampled from their joint posterior "
        "(same state-space equations as before). Holdout rows are true forecasts — the holdout target "
        "is never seen. Fewer draws = faster; check R-hat / divergences below before trusting a fit."
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        num_warmup = st.number_input("Warm-up iterations / chain", 100, 5000, 500, 100)
        num_chains = st.number_input("Chains", 1, 4, 2, 1,
                                     help="≥2 needed for R-hat. Chains run in parallel on separate CPU devices.")
    with m2:
        num_samples = st.number_input("Kept draws / chain", 100, 5000, 500, 100)
        target_accept = st.slider("Target accept", 0.80, 0.99, 0.90, 0.01,
                                  help="Raise toward 0.95–0.99 if divergences appear (slower, safer).")
    with m3:
        seed = st.number_input("Random seed", 0, 10**6, 0, 1)
        max_tree_depth = st.number_input("Max tree depth", 5, 12, 8, 1)
    mcmc_cfg = {"num_warmup": int(num_warmup), "num_samples": int(num_samples),
                "num_chains": int(num_chains), "target_accept": float(target_accept),
                "seed": int(seed), "max_tree_depth": int(max_tree_depth)}

    if st.button("🚀 Run RBE MMM", type="primary", use_container_width=True):
        joint_active = bool(config.get("enable_second_dependent") and config.get("target2"))
        chained_mode = joint_active and dep_relationship == "chained"
        spinner_text = (
            "Running chained two-stage MCMC… (minutes)" if chained_mode else
            "Running joint bivariate MCMC… (minutes)" if joint_active else
            "Running MCMC… (minutes)"
        )
        prog = st.progress(0.0, text="Starting …")
        _cb = lambda f, t: prog.progress(float(min(max(f, 0.0), 1.0)), text=t)
        with st.spinner(spinner_text):
            try:
                if chained_mode:
                    results_1, results_2, df_with_driver, driver_col = \
                        run_chained_dependent_pipeline(df, config, mcmc_cfg, progress_cb=_cb)
                    # Persist the new driver column into the working dataset — same
                    # pattern Tab 2 uses for prophet columns — so every downstream
                    # tab (Results, Refine & Refit, exports) sees it automatically.
                    if driver_col is not None and driver_col not in df.columns:
                        st.session_state.df = df_with_driver
                        existing = set(st.session_state.get("chain_driver_cols_added", []))
                        existing.add(driver_col)
                        st.session_state.chain_driver_cols_added = sorted(existing)
                    # Also fold the driver into the SAVED config's Dependent-1
                    # predictor lists — otherwise Tab 8 · Refine & Refit (which
                    # deep-copies st.session_state.config, not results_1["g"])
                    # would silently lose this predictor on the next refit.
                    if driver_col is not None:
                        role_key = "media" if results_1.get("chain_driver_role") == "media" else "non_media"
                        new_config = dict(st.session_state.config)
                        role_list = list(new_config.get(role_key, []))
                        if driver_col not in role_list:
                            role_list.append(driver_col)
                        new_config[role_key] = role_list
                        if config.get("chain_driver_positive", True):
                            pos_list = list(new_config.get("positive_beta_cols", []))
                            if driver_col not in pos_list:
                                pos_list.append(driver_col)
                            new_config["positive_beta_cols"] = pos_list
                        adstock_map = dict(new_config.get("adstock_map", {}))
                        adstock_map.setdefault(driver_col, "instant")
                        new_config["adstock_map"] = adstock_map
                        st.session_state.config = new_config
                        config = new_config
                else:
                    results_1, results_2 = run_multi_dependent_pipeline(
                        df, config, mcmc_cfg, progress_cb=_cb)
                st.session_state.model_results   = results_1
                st.session_state.model_fitted    = True
                st.session_state.model_results_2 = results_2
                st.session_state.model_fitted_2  = results_2 is not None

                prog.empty()
                st.success("✅ Model fitted!")
                _dg = results_1["mcmc"]
                if _dg["converged"]:
                    st.caption(f"MCMC: max R-hat {_dg['max_rhat']:.3f} · min ESS {_dg['min_ess']:.0f} · 0 divergences ✅")
                else:
                    st.warning(f"MCMC diagnostics need attention — max R-hat {_dg['max_rhat']:.3f}, "
                               f"{_dg['divergences']} divergence(s), min ESS {_dg['min_ess']:.0f}. "
                               "Try more warm-up/draws or a higher target accept.")
                st.markdown(f"#### Dependent 1 · `{config['target']}`")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("MAPE",       f"{results_1['mape']:.2%}")
                c2.metric("Train MAPE", f"{results_1['mape_in']:.2%}")
                c3.metric("Test MAPE",  f"{results_1['mape_out']:.2%}")
                c4.metric("Gelman R²",  f"{results_1['r2_gelman']:.4f}")
                c5.metric("Post. mean log-lik", f"{results_1['loglik']:.1f}")
                c6.metric("Diagnostics OK", "Yes ✅" if results_1["success"] else "Check ⚠️")
                with st.expander("📊 R² metrics"):
                    st.metric("R²", f"{results_1['r2']:.4f}")

                if results_2 is not None and chained_mode:
                    st.markdown(f"#### Dependent 2 · `{config.get('target2')}` (fitted independently)")
                    d1, d2, d3, d4, d5, d6 = st.columns(6)
                    d1.metric("MAPE",       f"{results_2['mape']:.2%}")
                    d2.metric("Train MAPE", f"{results_2['mape_in']:.2%}")
                    d3.metric("Test MAPE",  f"{results_2['mape_out']:.2%}")
                    d4.metric("Gelman R²",  f"{results_2['r2_gelman']:.4f}")
                    d5.metric("Post. mean log-lik",    f"{results_2['loglik']:.1f}")
                    d6.metric("Diagnostics OK",  "Yes ✅" if results_2["success"] else "Check ⚠️")
                    with st.expander("📊 R² metrics"):
                        st.metric("R²", f"{results_2['r2']:.4f}")
                    st.info(
                        f"➡️ **Chained into Dependent 1**: `{results_1['chain_driver_col']}` "
                        f"({'fitted' if results_1['chain_use_fitted'] else 'raw actual'} values of "
                        f"`{config.get('target2')}`) was added as a "
                        f"**{results_1['chain_driver_role'].replace('_',' ')}** predictor in "
                        f"Dependent 1's equation — see its beta/contribution/ROI in Tab 6 "
                        f"alongside the other channels."
                    )
                elif results_2 is not None:
                    st.markdown(f"#### Dependent 2 · `{config.get('target2')}` (joint bivariate fit)")
                    d1, d2, d3, d4, d5, d6 = st.columns(6)
                    d1.metric("MAPE",       f"{results_2['mape']:.2%}")
                    d2.metric("Train MAPE", f"{results_2['mape_in']:.2%}")
                    d3.metric("Test MAPE",  f"{results_2['mape_out']:.2%}")
                    d4.metric("Gelman R²",  f"{results_2['r2_gelman']:.4f}")
                    d5.metric("Post. mean log-lik",    f"{results_2['loglik']:.1f}")
                    d6.metric("Diagnostics OK",  "Yes ✅" if results_2["success"] else "Check ⚠️")
                    with st.expander("📊 R² metrics"):
                        st.metric("R²", f"{results_2['r2']:.4f}")
                    _coupling_mode_2 = results_2.get("cross_intercept_coupling_mode", "both")
                    _coupling_note_2 = {
                        "both": "both directions",
                        "dep1_in_dep2": "one-directional, Dep1→Dep2 only",
                        "dep2_in_dep1": "one-directional, Dep2→Dep1 only",
                        "none": "off",
                    }.get(_coupling_mode_2, _coupling_mode_2)
                    st.info(
                        f"🔗 Joint bivariate log-likelihood: **{results_2['joint_loglik']:.1f}** · "
                        f"estimated error correlation ρ(Dep1, Dep2) = **{results_2['rho_y']:.3f}** · "
                        f"cross-intercept coupling ({_coupling_note_2}): φ₁ (Dep2→Dep1) = **{results_2['phi1']:.3f}**, "
                        f"φ₂ (Dep1→Dep2) = **{results_2['phi2']:.3f}**"
                    )
                    # NOTE: the loss-formula + breakdown display lives in the
                    # PERSISTENT results section below (driven by
                    # st.session_state.model_fitted), not here — this whole
                    # `if st.button(...)` block only renders for the one
                    # rerun right after the click, and disappears on any
                    # later Streamlit rerun (widget interaction, tab switch,
                    # etc.). Putting it only here would make it flicker away.
            except Exception as e:
                st.exception(e)

    if st.session_state.model_fitted:
        res = st.session_state.model_results; st.divider()

        def _show_train_test(r, key_prefix):
            n_test = r.get("n_test", 0)
            if n_test and n_test > 0:
                tc1, tc2 = st.columns(2)
                tc1.metric("Train MAPE", f"{r['mape_in']:.2%}")
                tc2.metric("Test MAPE",  f"{r['mape_out']:.2%}",
                           delta=f"{(r['mape_out']-r['mape_in'])*100:+.2f} pp", delta_color="inverse")
                st.caption(f"({r['n_train']} train / {n_test} test obs)")
            else:
                st.caption("No holdout/test split configured — metrics above are in-sample only.")

        st.markdown(f"**Dependent 1 · `{config['target']}`**")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE", f"{res['mape']:.2%}")
        c2.metric("Gelman R²", f"{res['r2_gelman']:.4f}")
        c3.metric("Post. mean log-lik", f"{res['loglik']:.2f}")
        with st.expander("📊 R² metrics"):
            st.metric("R²", f"{res['r2']:.4f}")
        _show_train_test(res, "dep1")
        _m = res.get("mcmc")
        if _m:
            with st.expander("🔬 MCMC diagnostics & posterior summary"):
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Max R-hat", f"{_m['max_rhat']:.3f}")
                k2.metric("Min ESS", f"{_m['min_ess']:.0f}")
                k3.metric("Divergences", f"{_m['divergences']}")
                k4.metric("Runtime", f"{_m['diagnostics']['runtime_s']:.0f}s")
                st.dataframe(_m["summary"].round(5), use_container_width=True, hide_index=True)

        if st.session_state.get("model_fitted_2") and st.session_state.get("model_results_2"):
            res2 = st.session_state.model_results_2
            mode_label = "fitted independently" if res2.get("chained_into_dep1") else "joint bivariate fit"
            st.markdown(f"**Dependent 2 · `{config.get('target2')}`** ({mode_label})")
            e1, e2, e3 = st.columns(3)
            e1.metric("MAPE", f"{res2['mape']:.2%}")
            e2.metric("Gelman R²", f"{res2['r2_gelman']:.4f}")
            e3.metric("Post. mean log-lik", f"{res2['loglik']:.2f}")
            with st.expander("📊 R² metrics"):
                st.metric("R²", f"{res2['r2']:.4f}")
            _show_train_test(res2, "dep2")
            if res2.get("joint_fit"):
                _coupling_mode_disp = res2.get("cross_intercept_coupling_mode", "both")
                _coupling_note_disp = {
                    "both": "both directions",
                    "dep1_in_dep2": "Dep1→Dep2 only",
                    "dep2_in_dep1": "Dep2→Dep1 only",
                    "none": "off",
                }.get(_coupling_mode_disp, _coupling_mode_disp)
                st.caption(f"ρ(Dep1, Dep2) = {res2['rho_y']:.3f} · "
                           f"φ₁ (Dep2→Dep1) = {res2['phi1']:.3f} · φ₂ (Dep1→Dep2) = {res2['phi2']:.3f} "
                           f"[coupling: {_coupling_note_disp}] · "
                           f"joint log-lik = {res2['joint_loglik']:.2f}")

                st.caption("Objective: joint posterior over both equations' parameters and latent state paths "
                           "(bivariate-Gaussian likelihood on the training rows × priors). "
                           f"RMSE (full data): {res2['rmse_dep1']:.4g} / {res2['rmse_dep2']:.4g}.")
            elif res2.get("chained_into_dep1") and st.session_state.model_results.get("chain_driver_col"):
                st.caption(
                    f"➡️ Feeds Dependent 1 as `{st.session_state.model_results['chain_driver_col']}` "
                    f"({st.session_state.model_results.get('chain_driver_role','non_media').replace('_',' ')} role)."
                )

        st.caption("Proceed to **Tab 7** for full results. Use the selector there to "
                   "switch between Dependent 1 and Dependent 2.")
