"""
Sidebar: branding, dependency status, step checklist, prophet column log.
"""

import streamlit as st

from modules.persistence import build_workspace_bytes, restore_workspace


def render_sidebar(nevergrad_available: bool):
    with st.sidebar:
        st.markdown("## 📡 2 dependent model")
        st.markdown("**Recursive Bayesian Estimation**  \nMarketing Mix Modeling")
        
        st.divider()
        steps = {
            "1 · Data Upload":    st.session_state.df is not None,
            "2 · Prophet Decomp": st.session_state.prophet_results is not None,
            "3 · Correlation":    st.session_state.df is not None,
            "4 · Configuration":  st.session_state.config is not None,
            "5 · Run Model":      st.session_state.model_fitted,
            "6 · Results":        st.session_state.model_fitted,
        }
        for label, done in steps.items():
            st.markdown(f"`{'✅' if done else '○'}` {label}")

        if st.session_state.prophet_cols_added:
            st.divider()
            st.caption("📌 Prophet cols in dataset:")
            for pc in st.session_state.prophet_cols_added:
                st.caption(f"  • {pc}")

        st.divider()
        with st.expander("💾 Save / Load Model", expanded=False):
            st.caption(
                "Save your data, configuration, and fitted model(s) to one "
                "file you can download now and load back later — in this "
                "session or a brand-new one — to pick up exactly where you "
                "left off, without redoing Tabs 1-8."
            )
            if st.session_state.df is None:
                st.caption("Upload data in **Tab 1** first to enable saving.")
            else:
                if st.button("🧮 Prepare model file", use_container_width=True,
                             key="sidebar_prep_save"):
                    from datetime import datetime
                    st.session_state["_workspace_bytes"] = build_workspace_bytes()
                    st.session_state["_workspace_fname"] = (
                        f"model_workspace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rbe"
                    )
                if st.session_state.get("_workspace_bytes"):
                    st.download_button(
                        "⬇️ Download saved model",
                        st.session_state["_workspace_bytes"],
                        st.session_state.get("_workspace_fname") or "model_workspace.rbe",
                        "application/octet-stream",
                        use_container_width=True, key="sidebar_dl_workspace",
                    )

            st.markdown("---")
            uploaded_ws = st.file_uploader(
                "⬆️ Load a saved model", type=["rbe", "pkl"],
                key="sidebar_load_workspace_uploader",
                help="Overwrites everything currently loaded in this app "
                     "(data, configuration, fitted model, refit history) "
                     "with what's in the file.",
            )
            if uploaded_ws is not None:
                file_id = (uploaded_ws.name, uploaded_ws.size)
                if st.session_state.get("_last_loaded_workspace_id") != file_id:
                    ok, msg = restore_workspace(uploaded_ws.getvalue())
                    st.session_state["_last_loaded_workspace_id"] = file_id
                    if ok:
                        st.session_state["_load_workspace_msg"] = ("success", msg)
                        st.rerun()
                    else:
                        st.session_state["_load_workspace_msg"] = ("error", msg)

            load_msg = st.session_state.pop("_load_workspace_msg", None)
            if load_msg:
                kind, msg = load_msg
                (st.success if kind == "success" else st.error)(msg)

        st.divider()
        st.caption("Complete steps 1 → 5 in order.")
