# Rainbrain 2 v8 — RBE Marketing Mix Modeling Platform

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
    ├── kalman.py                  # RBE forward filter + RTS smoother
    ├── cochrane_orcutt.py         # OPTIONAL AR(1) error correction (ρ estimation + CO iteration loop)
    ├── bounds.py                  # theta0 + per-channel bounds builder
    ├── optimizer.py               # Nevergrad multi-objective optimizer
    ├── pipeline.py                 # run_full_ekf_pipeline() — ties it all together
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

Prophet and Nevergrad are optional — the app detects their availability
at import time (`modules/dependencies.py`) and degrades gracefully
(Tab 2 / the Nevergrad optimizer option are disabled with a clear
message if not installed).

## Key fix preserved from the original file

`modules/ui_helpers.py::safe_multiselect()` is a drop-in replacement for
`st.multiselect` that sanitizes any stored selection against the current
`options=` list on every render, so Streamlit can never raise
`StreamlitAPIException` on a stale value (e.g. a prophet column that
was merged into the dataset out-of-band in Tab 2). It also supports a
`require=` argument to permanently re-inject specific values (like new
prophet columns) into a selection on every rerun, regardless of what
else the user has clicked.

## Cochrane-Orcutt AR(1) error correction (optional)

Off by default. Tick **Fit with Cochrane-Orcutt AR(1) error correction** in the
Configuration tab (section "Autocorrelated Errors") and save the configuration;
every fit after that (single, joint, chained, and Tab 8 refits) runs the loop in
`modules/cochrane_orcutt.py`:

1. fit all parameters (ρ = 0 on the first pass — the ordinary fit)
2. estimate ρ from the one-step-ahead train residuals: ρ̂ = Σ eₜeₜ₋₁ / Σ eₜ₋₁²
3. refit with the observation quasi-differenced:
   yₜ − ρ·yₜ₋₁ = (Lₜ − ρ·Lₜ₋₁)·xₜ + uₜ
4. repeat until |ρ̂ − ρ| < tolerance or the max number of iterations is reached

Notes:
- Fitted values, contributions, ROI and metrics are still computed on the
  original (non-differenced) Lₜ·xₜ, so betas keep their structural meaning.
- With two dependents each equation gets its own ρ (serial correlation within
  an equation); the joint fit's ρ(Dep1, Dep2) is the separate cross-equation
  correlation and is unchanged.
- The first observation is kept with Prais-Winsten scaling by default (it anchors
  the initial state); untick it for classical Cochrane-Orcutt (first obs dropped).
- Each CO iteration is a full optimizer run → runtime ≈ iterations × normal run.
- Results tab / Run Model tab show ρ, iteration history and Durbin-Watson
  (before, and whitened after).
- With the option off, results are identical to the previous version.
