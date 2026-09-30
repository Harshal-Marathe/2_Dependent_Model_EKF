"""
Flat theta-vector layout — the single source of truth for "which slot of
theta is which parameter".

The layout is the same one modules/bounds.py::_build_theta0_and_bounds
builds theta0/bounds in and modules/params.py::unpack_theta reads back
out of, so the three must stay in lockstep. It is used by:
  - modules/refit.py   (warm-start / freeze / unfreeze by variable)
  - modules/mcmc.py    (per-parameter priors and posterior summary labels)
"""

import numpy as np

_BLOCK_TO_GKEY = {
    "Ls": "MEDIA_COLS", "delta": "MEDIA_COLS",
    "n_params": "MEDIA_COLS", "S_params": "MEDIA_COLS",
    "gamma": "INTERCEPT_EFFECTORS", "n_intercept": "INTERCEPT_EFFECTORS",
    "S_intercept": "INTERCEPT_EFFECTORS",
    "Ls_own_nonmedia": "OWN_NONMEDIA_COLS", "delta_own_nonmedia": "OWN_NONMEDIA_COLS",
    "Ls_comp_nonmedia": "COMP_NONMEDIA_COLS", "delta_comp_nonmedia": "COMP_NONMEDIA_COLS",
    "Ls_comp": "COMP_MEDIA_COLS", "delta_comp": "COMP_MEDIA_COLS",
    "n_comp": "COMP_MEDIA_COLS", "S_comp": "COMP_MEDIA_COLS",
    "Ls_price": "PRICE_COLS", "delta_price": "PRICE_COLS",
}


def theta_blocks(g: dict):
    """Ordered list of theta blocks with (start, length, cols/pairs/scalar).
    MUST stay in lockstep with bounds.py::_build_theta0_and_bounds and
    params.py::unpack_theta — the three are the same layout, read three ways.
    """
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_CROSS = g["N_CROSS"]; N_EFFECTORS = g["N_EFFECTORS"]
    USE_ORGANIC_DRIFT = g["USE_ORGANIC_DRIFT"]
    # Per-channel now: only channels individually on "weibull" (in ANY of
    # media/comp_media/own_nonmedia/comp_nonmedia) get a shape/scale slot,
    # in the fixed order g["ADSTOCK_WEIBULL_COLS"] (see modules/params.py).
    adstock_weibull_cols = g.get("ADSTOCK_WEIBULL_COLS", [])
    N_ADSTOCK = len(adstock_weibull_cols)

    blocks = []
    idx = 0

    def add(name, length, cols=None, pairs=None, scalar=False):
        nonlocal idx
        blocks.append({"name": name, "start": idx, "length": length,
                        "cols": cols, "pairs": pairs, "scalar": scalar})
        idx += length

    INTERCEPT_DYNAMICS_TYPE = g.get("INTERCEPT_DYNAMICS_TYPE", "carryover")

    add("Ls", N_MEDIA, cols=g["MEDIA_COLS"])
    if INTERCEPT_DYNAMICS_TYPE == "simple":
        add("I0", 1, scalar=True)
    else:
        add("G0", 1, scalar=True)
    add("delta", N_MEDIA, cols=g["MEDIA_COLS"])
    add("gamma", N_EFFECTORS, cols=g["INTERCEPT_EFFECTORS"])
    add("n_params", N_MEDIA, cols=g["MEDIA_COLS"])
    add("S_params", N_MEDIA, cols=g["MEDIA_COLS"])
    add("n_intercept", N_EFFECTORS, cols=g["INTERCEPT_EFFECTORS"])
    add("S_intercept", N_EFFECTORS, cols=g["INTERCEPT_EFFECTORS"])
    if N_ADSTOCK:
        add("adstock_shape", N_ADSTOCK, cols=adstock_weibull_cols)
        add("adstock_scale", N_ADSTOCK, cols=adstock_weibull_cols)
    add("Ls_own_nonmedia", N_OWN_NONMEDIA, cols=g["OWN_NONMEDIA_COLS"])
    add("Ls_comp_nonmedia", N_COMP_NONMEDIA, cols=g["COMP_NONMEDIA_COLS"])
    add("delta_own_nonmedia", N_OWN_NONMEDIA, cols=g["OWN_NONMEDIA_COLS"])
    add("delta_comp_nonmedia", N_COMP_NONMEDIA, cols=g["COMP_NONMEDIA_COLS"])
    add("Ls_comp", N_COMP, cols=g["COMP_MEDIA_COLS"])
    add("delta_comp", N_COMP, cols=g["COMP_MEDIA_COLS"])
    add("n_comp", N_COMP, cols=g["COMP_MEDIA_COLS"])
    add("S_comp", N_COMP, cols=g["COMP_MEDIA_COLS"])
    add("cross_delta", N_CROSS, pairs=g["CROSS_MEDIA_PAIRS"])
    add("cross_n", N_CROSS, pairs=g["CROSS_MEDIA_PAIRS"])
    add("cross_S", N_CROSS, pairs=g["CROSS_MEDIA_PAIRS"])
    add("Ls_price", N_PRICE, cols=g["PRICE_COLS"])
    add("delta_price", N_PRICE, cols=g["PRICE_COLS"])
    if USE_ORGANIC_DRIFT:
        add("mu", 1, scalar=True)
    add("sigma_y", 1, scalar=True)
    return blocks


_block_slices = theta_blocks  # legacy name used by modules/refit.py
