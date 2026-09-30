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

from modules.params import _make_globals, unpack_theta
from modules.bounds import _build_theta0_and_bounds


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
from modules.layout import _BLOCK_TO_GKEY, theta_blocks, _block_slices  # noqa: F401


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
def run_refit_pipeline(df_full, new_config, prev_result, mcmc_cfg=None,
                        unfreeze_cols=None, freeze_existing=True,
                        refit_sigma=True, refit_G0=False,
                        manual_overrides=None, progress_cb=None):
    """
    Re-sample `new_config`, warm-started from `prev_result` (a result dict
    from run_full_pipeline / a previous call to this function — has "params"
    and "g" keys). Returns a result dict shaped exactly like
    run_full_pipeline's, so it can be dropped straight into Tab 7.

    Frozen parameters (lo == hi) stay FIXED in the posterior; free ones get
    their usual prior and the chains START at the warm-started values.
    `manual_overrides`: {col: {block_name: value}} pins those parameters.
    """
    from modules.pipeline import _run_mcmc_fit, _build_equation_result

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
    fit = _run_mcmc_fit(df_full, n_train, [dict(g=g_new, theta0=theta0, bounds=bounds)],
                        mcmc_cfg, theta_init=theta0, progress_cb=progress_cb)
    return _build_equation_result(df_full, g_new, fit, 0, slice(0, len(theta0)), n_train)
