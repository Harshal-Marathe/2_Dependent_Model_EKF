"""
Tab 8 — Refine & Refit: after the model in Tab 6 is fitted and saved,
iteratively (a) add a new variable with its own bounds, and/or (b) reopen
an existing variable's bounds for adjustment — then refit. Everything
already fitted keeps its previous parameter values (frozen) unless you
explicitly reopen it; only the new/reopened variable(s) are actually
searched over, starting from the bounds you give them. Each refit becomes
the new baseline for the next round, so you can keep layering changes on.
"""

import copy

import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_helpers import section, info, per_channel_info, need_data, need_model
from modules.bounds_ui import render_per_channel_bounds
from modules.dependencies import NEVERGRAD_AVAILABLE
from modules.tabs.tab6_results import render_full_results
from modules.refit import (
    ROLE_LABELS, add_variable_to_config,
    variable_role_lists, run_refit_pipeline, get_current_value,
    editable_params_for_role,
)


def _init_refit_state():
    if st.session_state.get("refit_result") is None:
        st.session_state.refit_config = copy.deepcopy(st.session_state.config)
        st.session_state.refit_result = st.session_state.model_results
        st.session_state.refit_history = [{
            "Step": 0, "Action": "Baseline (Tab 6 fit)", "Variable": "—",
            "MAPE": st.session_state.model_results["mape"],
            "R²": st.session_state.model_results["r2"],
            "Gelman R²": st.session_state.model_results["r2_gelman"],
        }]


def _reset_refit_state():
    st.session_state.refit_config = copy.deepcopy(st.session_state.config)
    st.session_state.refit_result = st.session_state.model_results
    st.session_state.refit_history = [{
        "Step": 0, "Action": "Baseline (Tab 6 fit)", "Variable": "—",
        "MAPE": st.session_state.model_results["mape"],
        "R²": st.session_state.model_results["r2"],
        "Gelman R²": st.session_state.model_results["r2_gelman"],
    }]


def _bounds_widget_for(col, role, refit_config, df, key_prefix, adstock_choice="instant"):
    """Route to render_per_channel_bounds with the right column bucket for
    the chosen role, returning just this column's bounds dict.

    `adstock_choice`: this specific column's OWN "weibull"/"instant" pick
    (adstock is per-channel now — see modules/bounds_ui.py) — NOT a global
    model-wide flag."""
    use_hill = refit_config.get("transform_type", "hill") == "hill"
    adstock_map = {col: adstock_choice}

    if role == "media":
        b = render_per_channel_bounds([col], [], key_prefix, df, use_hill, adstock_map)
    elif role == "comp_media":
        b = render_per_channel_bounds([col], [col], key_prefix, df, use_hill, adstock_map)
    elif role == "price":
        b = render_per_channel_bounds([], [], key_prefix, df, use_hill, adstock_map, price_cols=[col])
    else:  # non_media / comp_nonmedia
        b = render_per_channel_bounds([], [], key_prefix, df, use_hill, adstock_map, nonmedia_cols=[col])
    return b.get(col, {})


def _run_and_record(df, new_config, action_label, var_label, unfreeze_cols,
                     freeze_existing, refit_sigma, refit_G0, method, max_iter,
                     manual_overrides=None):
    with st.spinner("Refitting… warm-started from the current baseline (30–300 s)"):
        try:
            result = run_refit_pipeline(
                df, new_config, st.session_state.refit_result, max_iter, method,
                unfreeze_cols=unfreeze_cols, freeze_existing=freeze_existing,
                refit_sigma=refit_sigma, refit_G0=refit_G0,
                manual_overrides=manual_overrides,
            )
        except Exception as e:
            st.exception(e)
            return
    st.session_state.refit_config = new_config
    st.session_state.refit_result = result
    st.session_state.refit_history.append({
        "Step": len(st.session_state.refit_history),
        "Action": action_label, "Variable": var_label,
        "MAPE": result["mape"], "R²": result["r2"],
        "Gelman R²": result["r2_gelman"],
    })
    # The results panel (Actual vs Predicted / Contributions) is rendered
    # further up the script than these buttons, so within a single script
    # pass it would still be showing the *previous* result. Stash a message
    # and force a full rerun so the whole page redraws top-to-bottom with
    # the fresh refit_result already in session state.
    st.session_state.refit_last_message = (
        f"✅ Refit complete — MAPE {result['mape']:.2%} · R² {result['r2']:.4f} · "
        f"Gelman R² {result['r2_gelman']:.4f}"
    )
    st.rerun()


def render_tab7():
    section("07", "Refine & Refit")
    if st.session_state.df is None: need_data()
    if not st.session_state.model_fitted: need_model()

    info(
        "Fit the baseline model in <b>Tab 6</b> first. Here you can <b>add a new "
        "variable</b> with its own bounds, or <b>reopen an existing variable's "
        "bounds</b> for adjustment, then refit. Anything already fitted that you "
        "don't touch keeps <b>exactly the same parameter values</b> — only the "
        "new/reopened variable(s) are actually re-optimized, starting from the "
        "bounds you set for them. Every refit's <b>Actual vs Predicted</b> and "
        "<b>Short/Long-term contribution</b> results are shown below as soon as "
        "it completes. When you're happy with a refit, go to <b>Tab 7</b> to "
        "save it as the official model."
    )
    if st.session_state.config.get("enable_second_dependent") and st.session_state.config.get("target2"):
        if st.session_state.config.get("dependent_relationship") == "chained":
            driver_col = st.session_state.model_results.get("chain_driver_col")
            st.warning(
                "⚠️ A chained second dependent variable is configured, but Tab 8 "
                "currently refines **Dependent 1 only**. Refits here won't re-fit "
                f"Dependent 2 (`{st.session_state.config.get('target2')}`) — "
                + (f"its fitted driver `{driver_col}` stays frozen and is treated as "
                   "an ordinary predictor of Dependent 1." if driver_col else
                   "its values feed Dependent 1 as an ordinary predictor.")
            )
        else:
            st.warning(
                "⚠️ A second dependent variable is configured, but Tab 8 currently "
                "refines **Dependent 1 only** (the joint bivariate fit is not "
                "warm-startable yet). Refits here won't touch Dependent 2."
            )

    _init_refit_state()
    df = st.session_state.df
    refit_config = st.session_state.refit_config
    refit_result = st.session_state.refit_result
    target = refit_config["target"]

    # ── Baseline status ─────────────────────────────────────────────
    st.markdown(f"### Current Working Model · `{target}`")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAPE", f"{refit_result['mape']:.2%}")
    c2.metric("R²", f"{refit_result['r2']:.4f}")
    c3.metric("Gelman R²", f"{refit_result['r2_gelman']:.4f}")
    c4.metric("Log-Lik", f"{refit_result['loglik']:.2f}")
    c5.metric("Refit steps taken", len(st.session_state.refit_history) - 1)

    with st.expander("📜 Refinement history", expanded=False):
        hist_df = pd.DataFrame(st.session_state.refit_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    if st.button("↩️ Reset — discard all refinements, start over from Tab 6 fit"):
        _reset_refit_state()
        st.rerun()

    if st.session_state.get("refit_last_message"):
        st.success(st.session_state.pop("refit_last_message"))

    with st.expander("📈 Current Working Model — Full Results & ROI Analytics (same as Tab 7)",
                      expanded=True):
        st.caption(
            "Everything below reflects the **current working (refit) model** — "
            "identical sections and download buttons to Tab 7, so you don't need "
            "to switch tabs to review or export a refit before deciding whether "
            "to save it as the official model."
        )
        render_full_results(df, refit_config, refit_result, target, key_prefix="tab7_")

    st.divider()

    # ── Advanced / shared refit options ──────────────────────────────
    with st.expander("⚙️ Refit options", expanded=False):
        oc1, oc2 = st.columns(2)
        with oc1:
            OPTIMIZER_OPTIONS = ["L-BFGS-B", "SLSQP"]
            method = st.selectbox("Optimizer", OPTIMIZER_OPTIONS, key="refit_method")
            max_iter = st.number_input("Max iterations", 100, 5000, 500, 100, key="refit_max_iter")
        with oc2:
            refit_sigma = st.checkbox(
                "Re-optimize noise (sigma_y)", value=True, key="refit_sigma",
                help="Recommended — residual variance usually shifts slightly "
                     "once a new variable explains part of it.")
            refit_G0 = st.checkbox(
                "Also re-optimize global intercept persistence/baseline "
                "(G0, I0, or Weibull shape/scale)",
                value=False, key="refit_G0",
                help="Covers whichever applies to this model's Intercept "
                     "Dynamics setting — G0 (Carryover/AR(1) mode), I0 "
                     "(Simple/no-carryover mode), or the Weibull shape/"
                     "scale (Weibull carryover mode; that mode has no G0).")
        freeze_existing = st.checkbox(
            "🔒 Freeze all previously-fitted parameters (recommended)", value=True,
            key="refit_freeze",
            help="If off, EVERY parameter is free to move again — this stops "
                 "being an incremental refit and behaves like a full refit.")

    st.divider()

    # ── Section A: add a new variable ────────────────────────────────
    st.markdown("### A · Add a New Variable")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    used_cols = {target, refit_config.get("target2")} | {
        c for lst in variable_role_lists(refit_config).values() for c in lst
    }
    available = [c for c in num_cols if c not in used_cols]

    if not available:
        st.info("Every numeric column is already in the model.")
    else:
        ac1, ac2 = st.columns(2)
        with ac1:
            new_col = st.selectbox("Variable to add", available, key="refit_new_col")
        with ac2:
            role_key = st.selectbox(
                "Role in the model", list(ROLE_LABELS.keys()),
                format_func=lambda k: ROLE_LABELS[k], key="refit_new_role",
            )

        new_adstock_choice = "instant"
        if role_key != "price":
            adstock_pick = st.radio(
                f"Adstock type for `{new_col}`",
                ["⚡ Instant (Nerlove-Arrow)", "🌀 Weibull (delayed)"],
                horizontal=True, key="refit_new_adstock",
                help="Chosen per channel, same as Tab 5 · Section D — this "
                     "only affects this one new variable.",
            )
            new_adstock_choice = "weibull" if adstock_pick.startswith("🌀") else "instant"

        st.caption(f"Set bounds for **{new_col}** — these are the ONLY parameters "
                   f"that will actually be searched over in this refit.")
        new_bounds = _bounds_widget_for(new_col, role_key, refit_config, df,
                                         key_prefix=f"refit_new_{new_col}_",
                                         adstock_choice=new_adstock_choice)

        if st.button(f"➕ Add `{new_col}` & Refit", type="primary", key="refit_add_btn"):
            new_config = add_variable_to_config(refit_config, new_col, role_key, new_bounds,
                                                 adstock_choice=new_adstock_choice)
            _run_and_record(
                df, new_config, "Added variable", new_col,
                unfreeze_cols=set(), freeze_existing=freeze_existing,
                refit_sigma=refit_sigma, refit_G0=refit_G0,
                method=method, max_iter=int(max_iter),
            )

    st.divider()

    # ── Section B: set an existing variable's parameter to an exact value ──
    st.markdown("### B · Set an Existing Variable's Parameter to a New Value")
    per_channel_info(
        "🎯 No ranges here — pick a variable, tick the parameter(s) you want to "
        "change, type the <b>exact value</b> you want (e.g. Ls = 0.7), and refit. "
        "That parameter is pinned to exactly that number — the model is NOT free "
        "to search around it. Everything else (this variable's other parameters, "
        "and every other variable) stays frozen at its current fitted value."
    )
    role_lists = variable_role_lists(refit_config)
    existing_vars = [(c, r) for r, cols in role_lists.items() for c in cols]

    if not existing_vars:
        st.info("No variables in the current model yet.")
    else:
        labels = [f"{c}  ({ROLE_LABELS[r]})" for c, r in existing_vars]
        pick = st.selectbox("Variable to edit", labels, key="refit_adjust_pick")
        adj_col, adj_role = existing_vars[labels.index(pick)]

        g_now = refit_result["g"]
        specs = editable_params_for_role(adj_role, g_now, col=adj_col)
        overrides = {}
        for block_name, param_label, is_unit_interval in specs:
            current = get_current_value(refit_result, adj_col, block_name)
            if current is None:
                continue
            oc1, oc2 = st.columns([1, 2])
            with oc1:
                do_override = st.checkbox(
                    f"Set {param_label}", key=f"refit_ov_chk_{adj_col}_{block_name}")
            with oc2:
                if is_unit_interval:
                    new_val = st.number_input(
                        f"New value (current: {current:.4f})", 0.0, 1.0,
                        float(np.clip(current, 0.0, 1.0)), 0.01,
                        key=f"refit_ov_val_{adj_col}_{block_name}",
                        disabled=not do_override,
                    )
                else:
                    new_val = st.number_input(
                        f"New value (current: {current:.4g})",
                        value=float(current), format="%.6g",
                        key=f"refit_ov_val_{adj_col}_{block_name}",
                        disabled=not do_override,
                    )
            if do_override:
                overrides[block_name] = new_val

        if st.button(f"🎯 Set Value(s) for `{adj_col}` & Refit",
                     type="primary", key="refit_adjust_btn"):
            if not overrides:
                st.warning("Tick at least one parameter to override.")
            else:
                _run_and_record(
                    df, refit_config, "Set parameter value(s)",
                    f"{adj_col} → " + ", ".join(f"{k}={v:.4g}" for k, v in overrides.items()),
                    unfreeze_cols=set(), freeze_existing=freeze_existing,
                    refit_sigma=refit_sigma, refit_G0=refit_G0,
                    method=method, max_iter=int(max_iter),
                    manual_overrides={adj_col: overrides},
                )

    st.divider()

    # ── Section C: parameter table (saving now happens in Tab 7) ─────
    st.markdown("### C · Current Parameter Table")
    with st.expander("📋 Current parameter table", expanded=False):
        st.dataframe(refit_result["param_df"], use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download parameters", refit_result["param_df"].to_csv(index=False).encode(),
            "refit_parameters.csv", "text/csv", key="refit_param_dl",
        )
    st.info(
        "➡️ Head to **Tab 7 · Results & ROI Analytics** to compare this refined "
        "model against the Tab 6 baseline and **save it as the official model**."
    )
