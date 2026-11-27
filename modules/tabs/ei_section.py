"""
Tab 6 section: EI refinement + spend-distributed media (see modules/ei_refine.py).
"""
import numpy as np
import pandas as pd
import streamlit as st

from modules import ei_refine as E


def _slug(x):
    return "".join(ch if ch.isalnum() else "_" for ch in str(x))


def _summary(res):
    return (f"R² {res['r2'] * 100:.2f}%  ·  MAPE {res['mape'] * 100:.2f}%  ·  "
            f"Media {E.media_pct_of_kpi(res):.1f}%")


def render_ei_section(df, config, res, target, pcb_key=None):
    if res.get("joint_fit") or target != config.get("target"):
        return
    g = res["g"]; cols = list(g["MEDIA_COLS"])
    if len(cols) < 2:
        return
    if res.get("theta") is None:
        return

    st.divider()
    st.markdown("### 🎯 Effectiveness-index (EI) refinement & spend-distributed media")
    st.caption(
        "EI = a channel's share of media contribution ÷ its share of media spend. It is "
        "**relative**: the spend-weighted average EI is always exactly 1, so every channel "
        "cannot be above 1 — raising one channel's EI lowers another's. What you control "
        "here is the spread: which channels sit above/below 1, and by how much. Both options "
        "below refit the model (every parameter re-estimated) with a penalty added to the "
        "likelihood; neither is a finding about your data.")

    _, ce0, se0 = E.ei_from_result(res)

    # ── 1 · funnel classification & bands ──────────────────────────
    lo, hi = [], []
    with st.expander("Funnel classification & EI bands", expanded=False):
        st.caption("Each channel's band is guessed from its name — override where the guess "
                   "is wrong. Pick **Custom** to type your own min/max.")
        for c in cols:
            guess = E.guess_funnel(c)
            opts = list(E.FUNNEL_PRESETS) + [E.CUSTOM]
            cls = st.selectbox(c, opts, index=opts.index(guess), key=f"ei_cls_{_slug(c)}")
            a, b = E.FUNNEL_PRESETS.get(cls, E.FUNNEL_PRESETS[guess])
            k1, k2 = st.columns(2)
            amin = k1.number_input("EI min", 0.0, 10.0, float(a), 0.05,
                                   key=f"ei_min_{_slug(c)}_{_slug(cls)}", disabled=cls != E.CUSTOM)
            amax = k2.number_input("EI max", 0.0, 10.0, float(b), 0.05,
                                   key=f"ei_max_{_slug(c)}_{_slug(cls)}", disabled=cls != E.CUSTOM)
            lo.append(min(amin, amax)); hi.append(max(amin, amax))
    lo, hi = np.array(lo), np.array(hi)
    note = E.feasibility_note(lo, hi, se0)
    if note:
        st.warning(note)

    # ── 2 · EI refinement settings ─────────────────────────────────
    with st.expander("EI refinement", expanded=True):
        do_ref = st.checkbox("Apply EI refinement", True, key="ei_do_ref",
                             help="Pulls each channel's EI into its band, then refits.")
        mode = st.radio("Refinement mode", ["Soft prior (guided)", "Hard band (clamped)"],
                        key="ei_mode", horizontal=True)
        st.caption("Soft pulls toward the band but lets the data move a channel. Hard clamps "
                   "strictly, which can pile several channels onto one bound.")
        strength = st.number_input("Soft prior strength", 0.0, 100.0, 1.0, 0.5, key="ei_strength")
        diff = st.checkbox("Auto-differentiate EI (fan channels off their bounds)", True,
                           key="ei_diff", help="Channels on a bound, or tied, are nudged to "
                           "distinct EIs ordered by their raw-model effectiveness.")
        d1, d2 = st.columns(2)
        diff_strength = d1.number_input("Differentiation strength", 0.0, 100.0, 4.0, 0.5,
                                        key="ei_diff_strength")
        gap = d2.number_input("Minimum EI gap between channels", 0.0, 1.0, 0.05, 0.01,
                              key="ei_gap")

    # ── 3 · spend-distributed settings ─────────────────────────────
    with st.expander("Spread media by spend share", expanded=False):
        do_sd = st.checkbox("Distribute media by spend (EI ≈ 1)", False, key="ei_do_sd",
                            help="Ties each channel's contribution to its share of spend. "
                                 "An assumption, not a finding.")
        auto = st.checkbox("Auto-tune target media % (sweep, keep the best fit)", True,
                           key="ei_auto")
        s1, s2, s3 = st.columns(3)
        sw_lo = s1.number_input("Sweep from %", 1.0, 90.0, 6.0, 1.0, key="ei_sw_lo", disabled=not auto)
        sw_hi = s2.number_input("to %", 1.0, 90.0, 20.0, 1.0, key="ei_sw_hi", disabled=not auto)
        sw_st = s3.number_input("step", 0.5, 20.0, 2.0, 0.5, key="ei_sw_st", disabled=not auto)
        tgt_pct = st.number_input("Target total media (% of sales)", 0.5, 95.0, 10.0, 0.5,
                                  key="ei_tgt_pct", disabled=auto)
        tol = st.number_input("Per-channel tolerance (± of target EI, %)", 0.0, 100.0, 30.0, 5.0,
                              key="ei_tol")
        fw = st.checkbox("Differentiate EI by funnel position", False, key="ei_fw")
        sdiff = st.checkbox("Auto-differentiate EI so every channel is distinct", True, key="ei_sdiff")
        sgap = st.number_input("Minimum EI gap between channels ", 0.0, 1.0, 0.05, 0.01, key="ei_sgap")
        sens = st.checkbox("Show media-budget sensitivity (10% to 60%)", False, key="ei_sens",
                           help="Refits at each target level and reports R² / MAPE (slower).")
        n_fits = (len(np.arange(sw_lo, sw_hi + 1e-9, sw_st)) if auto else 1) + (6 if sens else 0)
        if do_sd:
            st.caption(f"This will run about {n_fits} refit(s).")

    # ── 4 · run every model in one pass ────────────────────────────
    st.markdown("#### ⚡ Run all models at once")
    st.caption("Runs every ticked model in one go. Each one then appears in the model chooser at "
               "the top of the page, so you can switch between them without pressing each button. "
               "Uses the bands, strength and sweep settings above.")
    run_pick = st.multiselect("Models to run", list(E.ALL_MODEL_LABELS),
                              default=list(E.ALL_MODEL_LABELS),
                              format_func=lambda k: E.ALL_MODEL_LABELS[k], key="ei_all_pick")
    sw_grid = list(np.arange(sw_lo, sw_hi + 1e-9, sw_st)) if auto else [float(tgt_pct)]
    n_all = E.count_fits(run_pick, len(sw_grid))
    st.caption(f"About {n_all} full refit(s) in total — each takes roughly 30–300 s, so this can "
               "take a while. Leave the page open until it finishes.")
    if st.button("▶ Run all selected models", type="primary", use_container_width=True,
                 key="ei_all_run", disabled=not run_pick):
        s = st.session_state.get("last_fit_settings") or {}
        mi, me, ng = s.get("max_iter", 800), s.get("method", "L-BFGS-B"), s.get("ng_cfg")
        slot = st.empty()
        soft_cfg = dict(mode="soft", strength=strength, differentiate=diff,
                        diff_strength=diff_strength, gap=gap)
        hard_cfg = dict(soft_cfg, mode="hard")
        sd_cfg = dict(funnel_weights=fw, differentiate=sdiff, gap=sgap, tol_pct=tol,
                      strength=max(strength, 1.0), auto_tune=auto, grid=sw_grid,
                      target_pct=tgt_pct, sensitivity=False, sens_grid=[10, 20, 30, 40, 50, 60])
        sig_all = (target, round(float(res.get("loglik", 0.0)), 6))
        ok = False
        try:
            models = E.run_all_models(df, config, res, lo, hi, soft_cfg, hard_cfg, sd_cfg,
                                      set(run_pick), mi, me, ng,
                                      progress=lambda m: slot.info(m))
            co = models.pop("co", None)
            st.session_state.ei_all = {"sig": sig_all, "models": models}
            if co:
                st.session_state.residual_refit = {
                    "df": co["df"], "cfg": co["cfg"], "res": co["res"],
                    "sig": (target, sig_all[1], 0), "lag": 0}
            st.session_state["_tab6_pick_next"] = "co" if co else next(iter(models), "none")
            ok = True
        except Exception as e:
            st.session_state.ei_all = None
            slot.error(f"Run-all failed: {e}")
        if ok:
            st.rerun()

    if not (do_ref or do_sd):
        st.info("Tick at least one option above, then run.")
        return

    sig = (target, round(float(res.get("loglik", 0.0)), 6))
    if st.button("▶ Run refinement", type="primary", use_container_width=True, key="ei_run"):
        s = st.session_state.get("last_fit_settings") or {}
        mi, me, ng = s.get("max_iter", 800), s.get("method", "L-BFGS-B"), s.get("ng_cfg")
        slot = st.empty()
        out = {"sig": sig, "lo": lo, "hi": hi}
        try:
            if do_ref:
                slot.info("EI refinement: refitting every parameter…")
                out["refined"] = E.run_refinement(
                    df, config, res, (lo, hi),
                    dict(mode="hard" if mode.startswith("Hard") else "soft", strength=strength,
                         differentiate=diff, diff_strength=diff_strength, gap=gap), mi, me, ng)
            if do_sd:
                sd = dict(funnel_weights=fw, differentiate=sdiff, gap=sgap, tol_pct=tol,
                          strength=max(strength, 1.0), auto_tune=auto,
                          grid=list(np.arange(sw_lo, sw_hi + 1e-9, sw_st)), target_pct=tgt_pct,
                          sensitivity=sens, sens_grid=[10, 20, 30, 40, 50, 60])
                r = E.run_spend_distributed(df, config, res, (lo, hi), sd, mi, me, ng,
                                            progress=lambda m: slot.info(m))
                out["spend"] = r["model"]; out["sweep"] = r["sweep"]; out["sens"] = r["sens"]
            st.session_state.ei_models = out
            st.session_state["_tab6_pick_next"] = "spend" if "spend" in out else "refined"
            ok = True
        except Exception as e:
            st.session_state.ei_models = None
            slot.error(f"Refinement failed: {e}")
            ok = False
        if ok:
            st.rerun()          # refresh the model chooser at the top of the page

    em = st.session_state.get("ei_models")
    if not em or em.get("sig") != sig:
        return

    st.success("Refinement done — pick **EI refinement** or **Spend-distributed media** in the "
               "model chooser at the top of this page to see each as a complete model.")
    if "spend" in em:
        with st.expander("Spend-distributed tuning details", expanded=False):
            st.markdown(f"**Target total media tuned to {em['spend'].get('chosen_pct', 0):g}% of "
                        "sales** (applied to the training-window direct contribution; the "
                        "achieved media % of the final smoothed model is in the table).")
            st.dataframe(em["sweep"].round(2), use_container_width=True, hide_index=True)
            if em.get("sens") is not None:
                st.markdown("**Media-budget sensitivity**")
                st.dataframe(em["sens"].round(2), use_container_width=True, hide_index=True)
