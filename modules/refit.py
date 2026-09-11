"""
Incremental model refinement: warm-start a refit from a previously fitted
model's parameters, optionally FREEZING every parameter that was already
fitted (so the original model's behaviour doesn't change) while a newly
added variable (or a variable the user explicitly chose to re-open) is
fitted fresh within the bounds supplied for it.

This is what powers Tab 8 · Refine & Refit:
  1. Fit a baseline model in Tab 6 (saved automatically).
  2. In Tab 8, add a new variable with its own bounds, and/or reopen an
     existing variable's bounds for adjustment.
  3. Refit — parameters belonging to variables that were already in the
     model and were NOT reopened keep their previous fitted value (frozen,
     i.e. lo == hi == fitted value, so the optimizer can't move them).
     The new variable's parameters (and any reopened ones) are free to
     move within the bounds given, starting the search from a sensible
     default init rather than from scratch for the whole model.

The flat theta-vector layout mirrors modules/bounds.py::_build_theta0_and_bounds
and modules/params.py::unpack_theta exactly — see `_block_slices` below.
"""

import copy

import numpy as np
from scipy.optimize import minimize

from modules.dependencies import NEVERGRAD_AVAILABLE
from modules.params import _make_globals, unpack_theta
from modules.bounds import _build_theta0_and_bounds, build_normalized_problem
from modules.optimizer import run_nevergrad_optimizer
from modules.kalman import run_kalman_filter, rts_smoother, _precompute_adstocked, build_static_cache
from modules.pipeline import _postprocess_equation


# ────────────────────────────────────────────────────────────────────
# Config helpers — add a new variable / adjust bounds on the working config
# ────────────────────────────────────────────────────────────────────
ROLE_LABELS = {
    "media": "📺 Media / paid channel",
    "non_media": "🗂️ Non-media / control",
    "price": "💲 Price",
    "comp_media": "📉 Competitor media",
    "comp_nonmedia": "📉 Competitor non-media",
}

# Which theta block a (role, UI-param-name) pair lives in — used to let the
# user type an exact NEW VALUE for an existing variable's parameter (e.g.
# Ls 0.6 -> 0.7) rather than a search range. Mirrors the parameter set
# modules/bounds_ui.py actually exposes per role.
ROLE_PARAM_BLOCK = {
    ("media", "ls"): "Ls",
    ("media", "hill_n"): "n_params", ("media", "transform_n"): "n_params",
    ("media", "hill_s"): "S_params",
    ("media", "adstock_shape"): "adstock_shape", ("media", "adstock_scale"): "adstock_scale",
    ("comp_media", "ls"): "Ls_comp",
    ("comp_media", "hill_n"): "n_comp", ("comp_media", "transform_n"): "n_comp",
    ("comp_media", "hill_s"): "S_comp",
    ("comp_media", "adstock_shape"): "adstock_shape", ("comp_media", "adstock_scale"): "adstock_scale",
    ("price", "ls"): "Ls_price",
    ("non_media", "ls"): "Ls_own_nonmedia",
    ("non_media", "adstock_shape"): "adstock_shape", ("non_media", "adstock_scale"): "adstock_scale",
    ("comp_nonmedia", "ls"): "Ls_comp_nonmedia",
    ("comp_nonmedia", "adstock_shape"): "adstock_shape", ("comp_nonmedia", "adstock_scale"): "adstock_scale",
}


def add_variable_to_config(base_config: dict, col: str, role: str, bounds_dict: dict | None,
                            adstock_choice: str | None = None):
    """Return a NEW config dict with `col` appended to the right role list.

    `adstock_choice`: "weibull" | "instant" | None — this channel's OWN
    adstock choice (only meaningful for media/comp_media/non_media/
    comp_nonmedia roles; price never gets an adstock option). If given,
    stored into cfg["adstock_map"][col] so the per-channel choice made
    here survives into modules/params.py::_make_globals exactly like a
    channel configured in Tab 5. Defaults to "instant" if not given.
    """
    cfg = copy.deepcopy(base_config)
    bounds_dict = bounds_dict or {}

    if role == "media":
        cfg["media"] = list(cfg.get("media", [])) + [col]
        cfg["initial_media_betas"] = dict(cfg.get("initial_media_betas", {}))
        cfg["initial_media_betas"][col] = 0.0
    elif role == "non_media":
        cfg["non_media"] = list(cfg.get("non_media", [])) + [col]
        cfg["initial_own_nonmedia_betas"] = dict(cfg.get("initial_own_nonmedia_betas", {}))
        cfg["initial_own_nonmedia_betas"][col] = 0.0
    elif role == "price":
        cfg["price"] = list(cfg.get("price", [])) + [col]
        cfg["use_price"] = True
        cfg["initial_price_beta"] = dict(cfg.get("initial_price_beta", {}))
        cfg["initial_price_beta"][col] = -0.1
    elif role == "comp_media":
        cfg["comp_media"] = list(cfg.get("comp_media", [])) + [col]
        cfg["initial_comp_betas"] = dict(cfg.get("initial_comp_betas", {}))
        cfg["initial_comp_betas"][col] = -0.0001
    elif role == "comp_nonmedia":
        cfg["comp_nonmedia"] = list(cfg.get("comp_nonmedia", [])) + [col]
        cfg["initial_comp_nonmedia_betas"] = dict(cfg.get("initial_comp_nonmedia_betas", {}))
        cfg["initial_comp_nonmedia_betas"][col] = -0.01
    else:
        raise ValueError(f"Unknown role: {role}")

    if bounds_dict:
        cfg["per_channel_bounds"] = dict(cfg.get("per_channel_bounds", {}))
        cfg["per_channel_bounds"][col] = dict(bounds_dict)
    if role != "price":
        cfg["adstock_map"] = dict(cfg.get("adstock_map", {}))
        cfg["adstock_map"][col] = adstock_choice if adstock_choice in ("weibull", "instant") else "instant"
    return cfg


def apply_bound_adjustment(base_config: dict, col: str, bounds_dict: dict,
                            adstock_choice: str | None = None):
    """Return a NEW config dict with `col`'s per_channel_bounds overridden
    (and, if given, `col`'s adstock choice updated too — reopening a
    channel in Tab 8 lets you flip it between Weibull/Instant)."""
    cfg = copy.deepcopy(base_config)
    cfg["per_channel_bounds"] = dict(cfg.get("per_channel_bounds", {}))
    cfg["per_channel_bounds"][col] = dict(bounds_dict)
    if adstock_choice in ("weibull", "instant"):
        cfg["adstock_map"] = dict(cfg.get("adstock_map", {}))
        cfg["adstock_map"][col] = adstock_choice
    return cfg


def variable_role_lists(config: dict):
    """All variable names currently in the model, tagged by role."""
    return {
        "media": list(config.get("media", [])),
        "non_media": list(config.get("non_media", [])),
        "price": list(config.get("price", [])) if config.get("use_price") else [],
        "comp_media": list(config.get("comp_media", [])),
        "comp_nonmedia": list(config.get("comp_nonmedia", [])),
    }


# ────────────────────────────────────────────────────────────────────
# Flat theta-vector block layout (must mirror bounds.py / params.py)
# ────────────────────────────────────────────────────────────────────
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


def _block_slices(g: dict):
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
    elif INTERCEPT_DYNAMICS_TYPE == "weibull":
        add("intercept_weibull_shape", 1, scalar=True)
        add("intercept_weibull_scale", 1, scalar=True)
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


def _bound_clip(b):
    lo, hi = b
    return (-np.inf if lo is None else lo), (np.inf if hi is None else hi)


def build_warm_started_theta(g_new, theta0_default, bounds_default,
                              prev_params, prev_g, unfreeze_cols=None,
                              freeze_existing=True, refit_sigma=True, refit_G0=False,
                              manual_overrides=None):
    """
    Start from the fresh theta0/bounds for g_new, then for every parameter
    whose owning column ALSO existed in prev_g:
      - if `manual_overrides[col][block_name]` is given, PIN that parameter
        to exactly that value (theta0 = value, bounds = (value, value)) —
        no searching, no range, regardless of freeze_existing/unfreeze_cols.
      - otherwise warm-start theta0 at its previously fitted value (clipped
        to the new bounds so it's always feasible), and if freeze_existing
        and that column isn't in `unfreeze_cols`, pin its bounds to
        (value, value) so the optimizer can't move it either.
    Columns that are new (not in prev_g) are left at their fresh default
    init/bounds — they are exactly what gets fitted/searched.
    """
    unfreeze_cols = set(unfreeze_cols or [])
    manual_overrides = manual_overrides or {}
    theta0 = np.array(theta0_default, dtype=float).copy()
    bounds = list(bounds_default)

    if prev_params is None or prev_g is None:
        return theta0, bounds

    blocks = _block_slices(g_new)
    for blk in blocks:
        name, start, length = blk["name"], blk["start"], blk["length"]
        if length == 0:
            continue

        if blk["scalar"]:
            if name == "G0" and "G0" in prev_params:
                lo, hi = _bound_clip(bounds[start])
                val = float(np.clip(prev_params["G0"], lo, hi))
                theta0[start] = val
                if freeze_existing and not refit_G0:
                    bounds[start] = (val, val)
            elif name == "I0" and "I0" in prev_params:
                # Simple (no-carryover) intercept dynamics' baseline
                # constant — same warm-start/freeze treatment as G0 above,
                # gated by the same refit_G0 checkbox (it's the analogous
                # "global intercept" knob for this dynamics mode).
                lo, hi = _bound_clip(bounds[start])
                val = float(np.clip(prev_params["I0"], lo, hi))
                theta0[start] = val
                if freeze_existing and not refit_G0:
                    bounds[start] = (val, val)
            elif name in ("intercept_weibull_shape", "intercept_weibull_scale") and name in prev_params:
                # Weibull multi-lag intercept dynamics' shape/scale —
                # same warm-start/freeze treatment as G0/I0 above, gated
                # by the same refit_G0 checkbox.
                lo, hi = _bound_clip(bounds[start])
                val = float(np.clip(prev_params[name], lo, hi))
                theta0[start] = val
                if freeze_existing and not refit_G0:
                    bounds[start] = (val, val)
            elif name == "mu" and prev_g.get("USE_ORGANIC_DRIFT") and "mu" in prev_params:
                lo, hi = _bound_clip(bounds[start])
                val = float(np.clip(prev_params["mu"], lo, hi))
                theta0[start] = val
                if freeze_existing:
                    bounds[start] = (val, val)
            elif name == "sigma_y" and "sigma_y" in prev_params:
                lo, hi = _bound_clip(bounds[start])
                val = float(np.clip(prev_params["sigma_y"], lo, hi))
                theta0[start] = val
                if freeze_existing and not refit_sigma:
                    bounds[start] = (val, val)
            continue

        prev_arr = prev_params.get(name)
        if prev_arr is None:
            continue

        if blk["pairs"] is not None:
            new_keys = blk["pairs"]
            old_keys = prev_g.get("CROSS_MEDIA_PAIRS", [])
        elif name in ("adstock_shape", "adstock_scale"):
            new_keys = blk["cols"]
            old_keys = prev_g.get("ADSTOCK_WEIBULL_COLS", [])
        else:
            new_keys = blk["cols"]
            old_keys = prev_g.get(_BLOCK_TO_GKEY[name], [])

        old_index = {k: i for i, k in enumerate(old_keys)}
        for local_i, key in enumerate(new_keys):
            if key not in old_index:
                continue  # brand-new column — keep fresh default init/bounds
            flat_i = start + local_i

            override_val = manual_overrides.get(key, {}).get(name) if isinstance(key, str) else None
            if override_val is not None:
                val = float(override_val)
                theta0[flat_i] = val
                bounds[flat_i] = (val, val)  # pinned exactly — no search, no range
                continue

            lo, hi = _bound_clip(bounds[flat_i])
            val = float(np.clip(prev_arr[old_index[key]], lo, hi))
            theta0[flat_i] = val
            if freeze_existing and key not in unfreeze_cols:
                bounds[flat_i] = (val, val)

    return theta0, bounds


# ────────────────────────────────────────────────────────────────────
# Reading current fitted values (for the "set this to a new value" UI)
# ────────────────────────────────────────────────────────────────────
def get_current_value(result, col, block_name):
    """Current fitted value of `block_name` (e.g. "Ls", "delta_price") for
    `col`, read out of a result dict's params/g — or None if not found."""
    g = result["g"]; params = result["params"]
    if block_name in ("adstock_shape", "adstock_scale"):
        cols = g.get("ADSTOCK_WEIBULL_COLS", [])
    else:
        gkey = _BLOCK_TO_GKEY.get(block_name)
        cols = g.get(gkey) if gkey else None
    if not cols or col not in cols:
        return None
    idx = cols.index(col)
    arr = params.get(block_name)
    if arr is None:
        return None
    try:
        return float(arr[idx])
    except (TypeError, IndexError):
        return None


def editable_params_for_role(role, g, col=None):
    """(block_name, display_label, is_unit_interval) tuples of the
    parameters that make sense to manually override for a variable of
    this role, given the model's transform type and — since adstock is
    now chosen PER CHANNEL — this specific channel's own adstock choice
    (pass `col`; if omitted, falls back to "any channel is on weibull",
    matching the old global behaviour, for backward-compat callers)."""
    transform_hill = g.get("TRANSFORM_TYPE", "hill") == "hill"
    adstock_map = g.get("ADSTOCK_MAP", {})
    if col is not None:
        weibull = adstock_map.get(col) == "weibull"
    else:
        weibull = g.get("ADSTOCK_ANY_WEIBULL", g.get("ADSTOCK_TYPE", "instant") == "weibull")

    if role in ("media", "comp_media"):
        is_comp = role == "comp_media"
        ls_key    = "Ls_comp" if is_comp else "Ls"
        delta_key = "delta_comp" if is_comp else "delta"
        n_key     = "n_comp" if is_comp else "n_params"
        s_key     = "S_comp" if is_comp else "S_params"
        specs = [
            (ls_key, "Beta persistence (Ls)", True),
            (delta_key, "Delta (beta coefficient)", False),
            (n_key, "Hill slope (n)" if (transform_hill or is_comp) else "Power exponent (n)", False),
        ]
        if transform_hill or is_comp:
            specs.append((s_key, "Hill half-saturation (S)", False))
        if weibull:
            specs += [("adstock_shape", "Adstock shape (k)", False),
                      ("adstock_scale", "Adstock scale (λ)", False)]
        return specs
    if role == "price":
        return [("Ls_price", "Beta persistence (Ls_price)", True),
                ("delta_price", "Delta_price", False)]
    if role == "non_media":
        specs = [("Ls_own_nonmedia", "Beta persistence (Ls)", True),
                 ("delta_own_nonmedia", "Delta", False)]
        if weibull:
            specs += [("adstock_shape", "Adstock shape (k)", False),
                      ("adstock_scale", "Adstock scale (λ)", False)]
        return specs
    if role == "comp_nonmedia":
        specs = [("Ls_comp_nonmedia", "Beta persistence (Ls_comp_nonmedia)", True),
                 ("delta_comp_nonmedia", "Delta_comp_nonmedia", False)]
        if weibull:
            specs += [("adstock_shape", "Adstock shape (k)", False),
                      ("adstock_scale", "Adstock scale (λ)", False)]
        return specs
    return []


# ────────────────────────────────────────────────────────────────────
# Refit entry point
# ────────────────────────────────────────────────────────────────────
def run_refit_pipeline(df_full, new_config, prev_result, max_iter, method,
                        unfreeze_cols=None, freeze_existing=True,
                        refit_sigma=True, refit_G0=False, ng_cfg=None,
                        manual_overrides=None):
    """
    Fit `new_config` warm-started from `prev_result` (a result dict from
    run_full_ekf_pipeline / a previous call to this function — has "params"
    and "g" keys). Returns a result dict shaped exactly like
    run_full_ekf_pipeline's, so it can be dropped straight into Tab 7.

    `manual_overrides`: {col: {block_name: value}} — pins those specific
    parameters to an exact value (no searching) regardless of everything
    else. Everything else follows the usual freeze/unfreeze/warm-start
    rules above.
    """
    g_new = _make_globals(new_config)
    n_train = new_config["n_train"]
    df_train = df_full.iloc[:n_train].copy().reset_index(drop=True)

    theta0_default, bounds_default = _build_theta0_and_bounds(df_train, g_new)

    prev_params = prev_result.get("params") if prev_result else None
    prev_g = prev_result.get("g") if prev_result else None

    theta0, bounds = build_warm_started_theta(
        g_new, theta0_default, bounds_default, prev_params, prev_g,
        unfreeze_cols=unfreeze_cols, freeze_existing=freeze_existing,
        refit_sigma=refit_sigma, refit_G0=refit_G0,
        manual_overrides=manual_overrides,
    )

    static_cache_train = build_static_cache(df_train, g_new)

    if method == "Nevergrad" and NEVERGRAD_AVAILABLE and ng_cfg:
        best_theta, _ = run_nevergrad_optimizer(df_train, g_new, theta0, bounds, ng_cfg,
                                                  static_cache=static_cache_train)
        opt_success, opt_nit = True, ng_cfg.get("budget", 500)
    else:
        # Same fix as run_full_ekf_pipeline (see
        # modules/bounds.py::build_normalized_problem) — refit's theta has
        # the same wide-scale mix, so it's just as prone to `n` (and other
        # small-range params) getting stuck at their warm-started/init
        # value under a single unscaled finite-difference `eps`.
        theta0_norm, norm_bounds, unscale = build_normalized_problem(theta0, bounds)

        def objective(theta_norm):
            theta = unscale(theta_norm)
            p = unpack_theta(theta, g_new)
            _, _, _, _, _, _, _, _, loglik = run_kalman_filter(
                df_train, p, g_new, static_cache=static_cache_train)
            return -loglik
        opt = minimize(objective, theta0_norm, method=method,
                        bounds=norm_bounds,
                        options={"maxiter": max_iter, "ftol": 1e-9, "eps": 1e-6})
        best_theta, opt_success, opt_nit = unscale(opt.x), opt.success, opt.nit

    params = unpack_theta(best_theta, g_new)
    static_cache_full = build_static_cache(df_full, g_new)
    yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik = \
        run_kalman_filter(df_full, params, g_new, static_cache=static_cache_full)
    x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat)

    adstocked_media = _precompute_adstocked(df_full, g_new, params)
    result = _postprocess_equation(
        df_full, g_new, params, x_smooth, adstocked_media, cross_beta_contrib,
        opt_success, opt_nit, loglik, n_train=n_train,
    )
    result["P_smooth"] = P_smooth
    return result
