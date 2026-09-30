# Rainbrain 2 v8 — MMM Platform (MCMC core)

Modularized version of the original single-file Streamlit app.

## Structure

```
rainbrain_app/
├── app.py                       # Entry point — run this with `streamlit run app.py`
├── requirements.txt
└── modules/
    ├── dependencies.py          # Optional-package detection (Prophet, nevergrad)
    ├── styles.py                # Global CSS
    ├── state.py                 # st.session_state defaults
    ├── sidebar.py                # Sidebar (branding, step checklist)
    ├── ui_helpers.py             # section()/info()/safe_multiselect() etc.
    ├── transforms.py             # Hill saturation + adstock functions
    ├── params.py                 # _make_globals() / unpack_theta()
    ├── statespace.py              # State-space equations: obs matrix, process noise, adstock
    ├── mcmc.py                    # NUTS core: JAX equations, priors, sampler, posterior summaries
    ├── layout.py                  # Flat theta layout (shared by refit + priors)
    ├── bounds.py                  # theta0 + per-channel bounds builder
    ├── pipeline.py                 # run_full_pipeline() / joint / chained — ties it all together
    └── tabs/
        ├── tab1_data_upload.py
        ├── tab2_prophet.py
        ├── tab3_correlation.py
        ├── tab4_configuration.py
        ├── tab5_run_model.py
        └── tab6_results.py
```

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Prophet is optional (Tab 2 is disabled with a message if missing).
JAX + NumPyro are required for fitting.

## What changed in the core

Same state-space equations; the Kalman filter, RTS smoother and the
L-BFGS-B/SLSQP/Nevergrad optimizers were replaced by NUTS. Parameters and
the whole latent state path are sampled jointly. Holdout rows are true
forecasts (holdout target never enters the likelihood). Results carry
R-hat / ESS / divergences and 95% credible bands from posterior draws.

## Key fix preserved from the original file

`modules/ui_helpers.py::safe_multiselect()` is a drop-in replacement for
`st.multiselect` that sanitizes any stored selection against the current
`options=` list on every render, so Streamlit can never raise
`StreamlitAPIException` on a stale value (e.g. a prophet column that
was merged into the dataset out-of-band in Tab 2). It also supports a
`require=` argument to permanently re-inject specific values (like new
prophet columns) into a selection on every rerun, regardless of what
else the user has clicked.
