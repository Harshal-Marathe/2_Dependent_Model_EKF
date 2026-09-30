"""
MCMC (NUTS) inference core — replaces the Kalman filter, RTS smoother and
point-estimate optimizers (L-BFGS-B / SLSQP / Nevergrad).

WHAT CHANGED AND WHAT DID NOT
-----------------------------
The state-space equations are exactly the ones documented in
modules/statespace.py (Instant/Weibull adstock x Power/Hill transform,
intercept carryover/simple dynamics, cross-media synergy, competitor and
price states, spike dummies, joint bivariate fit with rho / phi coupling).

  Before:  parameters theta  = argmax of the Kalman-filter likelihood
           states x_t        = Kalman filter + RTS smoother given theta
  Now:     theta and the whole latent state path x_{0:T} are sampled
           TOGETHER from their joint posterior with NUTS:

      x_0 ~ N(x0, diag(sd0^2))
      x_t = clip( Td * x_{t-1} + u_t(theta) + sqrt(Q) * w_t ),  w_t ~ N(0, I)
      y_t ~ N( L_t . x_t , sigma_y^2 )                   (t in training window)

  (joint mode: y_t ~ N2( [L1_t.x1_t, L2_t.x2_t], R(sigma1, sigma2, rho) ),
   with the phi_1 / phi_2 cross-intercept terms inside the recursion.)

Td (transition diagonal) and u_t (forcing: delta*f(x_t) + Weibull lag-sum +
synergy + intercept boost) are the same quantities the numpy Kalman code
built; here they are written in JAX so NUTS can differentiate them.

  * Sign constraints / baseline floor: the same clamps the filter applied
    (positive/negative betas, competitor & price betas <= 0, intercept
    floor) are applied inside the recursion with jnp.clip.
  * Process noise Q is unchanged (fixed, scaled off the training target).
  * Parameter bounds (theta bounds from modules/bounds.py, including the
    user's per-channel bounds and the positive/negative beta constraints)
    become the SUPPORT of the priors; pinned/frozen parameters
    (lo == hi, used by Tab 8 refits) stay pinned.
  * HOLDOUT: the likelihood only sees the training rows, but the latent
    states run over the whole dataset. Holdout periods are therefore
    genuine forecasts (states propagate with process noise, the holdout
    target is never seen) — unlike the earlier build, where the filter
    still ingested the holdout target. Test metrics will look more
    conservative as a result; they are now honest out-of-sample numbers.

Parameterisation: every parameter is a deterministic transform of a
standard-normal base variable (probit-uniform for bounded parameters,
inverse-CDF truncated normal for scale-free ones, log-normal for
half-saturation S), so NUTS always works on well-conditioned N(0,1)
coordinates; states use the non-centered form (standard-normal shocks).
"""

import os
import time

# Must be set BEFORE jax is first imported so chains can run in parallel on
# separate CPU "devices". Respect any XLA_FLAGS the user already set.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import ndtr, ndtri

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as _np_summary

from scipy.special import ndtr as _sp_ndtr, ndtri as _sp_ndtri

from modules.params import unpack_theta
from modules.layout import theta_blocks
from modules.statespace import _build_observation_matrix, _build_process_noise
from modules.transforms import apply_transformation, hill_transform_vec

numpyro.enable_x64()

DEFAULT_MCMC_CFG = {
    "num_warmup": 500,       # NUTS adaptation iterations per chain
    "num_samples": 500,      # kept draws per chain
    "num_chains": 2,
    "target_accept": 0.90,   # raise toward 0.95-0.99 if divergences appear
    "max_tree_depth": 8,
    "seed": 0,
    "n_keep": 400,           # draws kept for credible bands / summaries
}

# prior kinds
K_FIXED, K_UNIF, K_TN, K_LOGTN = 0, 1, 2, 3


def mcmc_cfg_with_defaults(cfg=None):
    out = dict(DEFAULT_MCMC_CFG)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None})
    for k in ("num_warmup", "num_samples", "num_chains", "max_tree_depth", "seed", "n_keep"):
        out[k] = int(out[k])
    out["target_accept"] = float(out["target_accept"])
    return out


# ═════════════════════════════════════════════════════════════════════════
# 1. STATIC (parameter-independent) DATA PER EQUATION
# ═════════════════════════════════════════════════════════════════════════

def build_eq_static(df_full, g, n_train):
    """
    Everything about one equation that does not depend on the sampled
    parameters: observation matrix, raw regressor blocks, Weibull lag
    tables, process-noise sd, initial-state prior, and per-state clamps.
    Scale statistics (target mean/std, regressor means, baseline floor) come
    from the TRAINING window only, so nothing about the holdout leaks in.
    """
    T = len(df_full)
    n_train = int(min(max(n_train, 2), T))
    df_tr = df_full.iloc[:n_train]
    TARGET = g["TARGET_COL"]
    MEDIA, COMP = list(g["MEDIA_COLS"]), list(g["COMP_MEDIA_COLS"])
    ONM, CNM = list(g["OWN_NONMEDIA_COLS"]), list(g["COMP_NONMEDIA_COLS"])
    PRICE, DUM = list(g["PRICE_COLS"]), list(g["DUMMY_COLS"])
    N_MEDIA, N_COMP, N_ONM, N_CNM = len(MEDIA), len(COMP), len(ONM), len(CNM)
    N_PRICE, N_DUM = len(PRICE), len(DUM)
    dim = 1 + N_MEDIA + N_COMP + N_ONM + N_CNM + N_PRICE + N_DUM + g["SEASONAL_DIM"]
    adstock_map = g.get("ADSTOCK_MAP", {})
    adstock_idx = g.get("ADSTOCK_IDX", {})
    wb_cols = list(g.get("ADSTOCK_WEIBULL_COLS", []))
    n_lags = int(g.get("ADSTOCK_N_LAGS", 8))

    def block(cols):
        if not cols:
            return np.zeros((T, 0))
        return np.column_stack([df_full[c].values.astype(float) for c in cols])

    def ai(col):
        return adstock_idx[col] if (adstock_map.get(col) == "weibull" and col in adstock_idx) else -1

    # Weibull lag tables: Xlag[a, t, l-1] = x_a[t-l]  (0 for t < l)
    if wb_cols:
        Xlag = np.zeros((len(wb_cols), T, max(n_lags, 0)))
        for a, c in enumerate(wb_cols):
            arr = df_full[c].values.astype(float)
            for l in range(1, n_lags + 1):
                Xlag[a, l:, l - 1] = arr[:-l]
    else:
        Xlag = np.zeros((0, T, max(n_lags, 0)))

    # cross-media synergy pairs
    media_idx = {c: i for i, c in enumerate(MEDIA)}
    cross_tgt_state = np.array([media_idx[t] + 1 for t, _ in g["CROSS_MEDIA_PAIRS"]], dtype=int)
    cross_src_raw = block([s for _, s in g["CROSS_MEDIA_PAIRS"]])
    cross_src_ai = [ai(s) for _, s in g["CROSS_MEDIA_PAIRS"]]

    # intercept effectors (columns absent from df are skipped, as before)
    eff_cols = list(g["INTERCEPT_EFFECTORS"])
    eff_present = np.array([c in df_full.columns for c in eff_cols], dtype=float)
    X_eff = np.column_stack([
        df_full[c].values.astype(float) if c in df_full.columns else np.zeros(T)
        for c in eff_cols]) if eff_cols else np.zeros((T, 0))

    y = df_full[TARGET].values.astype(float)
    y_tr = y[:n_train]
    y_mean, y_std = float(np.mean(y_tr)), float(np.std(y_tr))
    y_std = y_std if y_std > 1e-12 else 1.0

    # ── process noise (unchanged: fixed, scaled off the training target) ──
    Qm = _build_process_noise(df_tr, g)
    qsd = np.sqrt(np.diag(Qm))

    # ── initial-state prior ──────────────────────────────────────────────
    x0 = np.zeros(dim)
    x0[0] = y_mean * 0.8
    for i, col in enumerate(MEDIA):
        x0[1 + i] = g["INITIAL_MEDIA_BETAS"].get(col, 0.0)
    for j, col in enumerate(COMP):
        x0[1 + N_MEDIA + j] = g["INITIAL_COMP_BETAS"].get(col, -0.0001)
    for k, col in enumerate(ONM):
        x0[1 + N_MEDIA + N_COMP + k] = g["INITIAL_OWN_NONMEDIA_BETAS"].get(col, 0.0)
    for k, col in enumerate(CNM):
        x0[1 + N_MEDIA + N_COMP + N_ONM + k] = g["INITIAL_COMP_NONMEDIA_BETAS"].get(col, -0.01)
    for p, col in enumerate(PRICE):
        x0[1 + N_MEDIA + N_COMP + N_ONM + N_CNM + p] = g["INITIAL_PRICE_BETA"].get(col, -0.1)

    # Diffuse-but-scaled prior on the initial state. The Kalman build used
    # P0 = var(y) for every state, which for a beta in raw units (spend ~1e6)
    # is astronomically wide; here each beta's initial sd is "3 target-sds
    # worth of effect at typical regressor level" instead.
    sd0 = np.full(dim, y_std)
    sd0[0] = 2.0 * y_std
    base = 1 + N_MEDIA + N_COMP + N_ONM + N_CNM + N_PRICE
    reg_cols = MEDIA + COMP + ONM + CNM + PRICE
    for s, col in enumerate(reg_cols):
        reg_mean = float(np.mean(np.abs(df_tr[col].values.astype(float))))
        sd0[1 + s] = 3.0 * y_std / reg_mean if reg_mean > 1e-9 else 1e-2
    # dummy betas keep the original sd = std(y)

    # ── per-state clamps (same rules the filter applied) ─────────────────
    lo = np.full(dim, -np.inf)
    hi = np.full(dim, np.inf)
    min_base_fraction = float(g.get("MIN_BASE_FRACTION", 0.0))
    if min_base_fraction > 0:
        lo[0] = min_base_fraction * y_mean
    pos, neg = set(g.get("POSITIVE_BETA_COLS", [])), set(g.get("NEGATIVE_BETA_COLS", []))
    for i, col in enumerate(MEDIA):
        if col in pos:   lo[1 + i] = 1e-8
        elif col in neg: hi[1 + i] = -1e-8
    for j in range(N_COMP):
        hi[1 + N_MEDIA + j] = -1e-8
    for k, col in enumerate(ONM):
        idx = 1 + N_MEDIA + N_COMP + k
        if col in pos:   lo[idx] = 1e-8
        elif col in neg: hi[idx] = -1e-8
    for k in range(N_CNM):
        hi[1 + N_MEDIA + N_COMP + N_ONM + k] = -1e-8
    for p in range(N_PRICE):
        hi[1 + N_MEDIA + N_COMP + N_ONM + N_CNM + p] = -1e-8

    return dict(
        T=T, n_train=n_train, dim=dim,
        y=y, obs_mask=(np.arange(T) < n_train).astype(float),
        y_mean=y_mean, y_std=y_std,
        L=_build_observation_matrix(df_full, g, None),
        X_own=block(MEDIA), X_comp=block(COMP), X_onm=block(ONM),
        X_cnm=block(CNM), X_price=block(PRICE), X_eff=X_eff, eff_present=eff_present,
        Xlag=Xlag, n_lags=n_lags, n_ads=len(wb_cols),
        own_ai=[ai(c) for c in MEDIA], comp_ai=[ai(c) for c in COMP],
        onm_ai=[ai(c) for c in ONM], cnm_ai=[ai(c) for c in CNM],
        wb_own=np.array([ai(c) >= 0 for c in MEDIA], dtype=bool),
        wb_comp=np.array([ai(c) >= 0 for c in COMP], dtype=bool),
        wb_onm=np.array([ai(c) >= 0 for c in ONM], dtype=bool),
        wb_cnm=np.array([ai(c) >= 0 for c in CNM], dtype=bool),
        cross_tgt_state=cross_tgt_state, cross_src_raw=cross_src_raw, cross_src_ai=cross_src_ai,
        n_cross=len(g["CROSS_MEDIA_PAIRS"]), n_eff=len(eff_cols), n_dummies=N_DUM,
        N_MEDIA=N_MEDIA, N_COMP=N_COMP, N_ONM=N_ONM, N_CNM=N_CNM, N_PRICE=N_PRICE,
        transform_type=g["TRANSFORM_TYPE"],
        intercept_transform_type=g.get("INTERCEPT_TRANSFORM_TYPE", "power"),
        use_drift=bool(g["USE_ORGANIC_DRIFT"]),
        x0=x0, sd0=sd0, qsd=qsd, lo=lo, hi=hi,
    )


# ═════════════════════════════════════════════════════════════════════════
# 2. THE EQUATIONS IN JAX  (Td, u_t, recursion)
# ═════════════════════════════════════════════════════════════════════════

def _safe_pow(x, n):
    """x**n with a finite gradient at x == 0 (d/dn of 0**n is 0*log 0 = nan)."""
    pos = x > 0
    xs = jnp.where(pos, x, 1.0)
    return jnp.where(pos, xs ** n, 0.0)


def _tf(x, ttype, n, S):
    x = jnp.maximum(x, 0.0)
    xn = _safe_pow(x, n)
    if ttype == "hill":
        return xn / (xn + S ** n + 1e-30)
    return xn


def _hill(x, n, S):
    return _tf(x, "hill", n, S)


def _wb_weights(k, lam, nl):
    """Normalised Weibull PDF weights for lags 1..nl (same as
    modules/transforms.py::weibull_lag_weights, differentiable)."""
    lags = jnp.arange(1, nl + 1, dtype=jnp.float64)
    ratio = lags / lam
    w = (k / lam) * ratio ** (k - 1.0) * jnp.exp(-ratio ** k)
    w = jnp.maximum(w, 0.0)
    tot = w.sum()
    return jnp.where(tot < 1e-12, jnp.ones(nl) / nl, w / jnp.maximum(tot, 1e-300))


def build_dynamics(p, st):
    """
    Parameter-dependent part of one equation.

    Returns
      Td      (dim,)      transition diagonal
      U       (T, dim)    forcing added at every step (row 0 unused)
      cross   (T, N_CROSS) synergy contribution booked against each target
      lt_eff  (T, N_EFF)  per-effector intercept boost  gamma_k * f(x_k,t)
    """
    T, dim, nl = st["T"], st["dim"], st["n_lags"]
    tt, itt = st["transform_type"], st["intercept_transform_type"]
    zero = jnp.zeros(T)

    lag_series = []
    for a in range(st["n_ads"]):
        w = _wb_weights(p["adstock_shape"][a], p["adstock_scale"][a], nl)
        lag_series.append(jnp.asarray(st["Xlag"][a]) @ w if nl > 0 else zero)

    def wl(a):
        return lag_series[a] if a >= 0 else zero

    # ── own media:  Σ_l w_l x_{t-l}  +  delta * f(x_t)   (+ synergy) ──────
    own_cols = []
    for i in range(st["N_MEDIA"]):
        f = _tf(jnp.asarray(st["X_own"][:, i]), tt, p["n_params"][i], p["S_params"][i])
        own_cols.append(wl(st["own_ai"][i]) + p["delta"][i] * f)

    cross_rows = []
    for k in range(st["n_cross"]):
        a = st["cross_src_ai"][k]
        src = lag_series[a] if a >= 0 else jnp.asarray(st["cross_src_raw"][:, k])
        contrib = p["cross_delta"][k] * _hill(src, p["cross_n"][k], p["cross_S"][k])
        own_cols[int(st["cross_tgt_state"][k]) - 1] = own_cols[int(st["cross_tgt_state"][k]) - 1] + contrib
        cross_rows.append(contrib)
    cross = jnp.stack(cross_rows, axis=1) if cross_rows else jnp.zeros((T, 0))

    # ── competitor media (Hill on raw, or on the Weibull lag series) ─────
    comp_cols = []
    for j in range(st["N_COMP"]):
        a = st["comp_ai"][j]
        series = lag_series[a] if a >= 0 else jnp.asarray(st["X_comp"][:, j])
        comp_cols.append(wl(a) + p["delta_comp"][j] * _hill(series, p["n_comp"][j], p["S_comp"][j]))

    onm_cols = [wl(st["onm_ai"][k]) + p["delta_own_nonmedia"][k] * jnp.asarray(st["X_onm"][:, k])
                for k in range(st["N_ONM"])]
    cnm_cols = [wl(st["cnm_ai"][k]) + p["delta_comp_nonmedia"][k] * jnp.asarray(st["X_cnm"][:, k])
                for k in range(st["N_CNM"])]
    price_cols = [p["delta_price"][q] * jnp.asarray(st["X_price"][:, q]) for q in range(st["N_PRICE"])]

    # ── intercept boost:  Σ_k gamma_k f(effector_k,t)  (+ I0 + mu) ───────
    lt_cols = []
    for k in range(st["n_eff"]):
        lt_cols.append(p["gamma"][k] * st["eff_present"][k]
                       * _tf(jnp.asarray(st["X_eff"][:, k]), itt, p["n_intercept"][k], p["S_intercept"][k]))
    lt_eff = jnp.stack(lt_cols, axis=1) if lt_cols else jnp.zeros((T, 0))
    u0 = lt_eff.sum(axis=1) + p["I0"]
    if st["use_drift"]:
        u0 = u0 + p["mu"]

    cols = [u0] + own_cols + comp_cols + onm_cols + cnm_cols + price_cols
    cols += [zero] * (dim - len(cols))          # spike dummies (and any seasonal dims): no forcing
    U = jnp.stack(cols, axis=1)

    # ── transition diagonal (weibull channels: 0 — the lag-sum is the memory) ──
    def diag(ls, wb):
        ls = jnp.asarray(ls)
        return jnp.where(jnp.asarray(wb), 0.0, ls) if len(wb) else ls

    Td = jnp.concatenate([
        jnp.atleast_1d(jnp.asarray(p["G0"], dtype=jnp.float64)),
        diag(p["Ls"], st["wb_own"]),
        diag(p["Ls_comp"], st["wb_comp"]),
        diag(p["Ls_own_nonmedia"], st["wb_onm"]),
        diag(p["Ls_comp_nonmedia"], st["wb_cnm"]),
        jnp.asarray(p["Ls_price"]),
        jnp.full(st["n_dummies"], 0.98),
        jnp.ones(dim - 1 - st["N_MEDIA"] - st["N_COMP"] - st["N_ONM"] - st["N_CNM"]
                 - st["N_PRICE"] - st["n_dummies"]),
    ])
    return Td, U, cross, lt_eff


def rollout(Td, U, x0, sd0, z, qsd, lo, hi, cpl_idx=None, phi1=0.0, phi2=0.0):
    """Non-centered latent state path  x_0 .. x_{T-1}  (T, dim)."""
    x_init = jnp.clip(x0 + sd0 * z[0], lo, hi)

    def step(x_prev, inp):
        u_t, z_t = inp
        x = Td * x_prev + u_t + qsd * z_t
        if cpl_idx is not None:
            x = x.at[0].add(phi1 * x_prev[cpl_idx])
            x = x.at[cpl_idx].add(phi2 * x_prev[0])
        x = jnp.clip(x, lo, hi)
        return x, x

    _, xs = lax.scan(step, x_init, (U[1:], z[1:]))
    return jnp.concatenate([x_init[None, :], xs], axis=0)


# ═════════════════════════════════════════════════════════════════════════
# 3. PRIORS  (theta bounds  ->  prior support; data-scaled scales)
# ═════════════════════════════════════════════════════════════════════════

_UNIF_BLOCKS = {"Ls", "Ls_own_nonmedia", "Ls_comp_nonmedia", "Ls_comp", "Ls_price", "G0",
                "n_params", "n_intercept", "n_comp", "cross_n",
                "adstock_shape", "adstock_scale", "mu"}
_LOGTN_BLOCKS = {"S_params", "S_intercept", "S_comp", "cross_S"}
_DELTA_BLOCKS = {"delta", "delta_own_nonmedia", "delta_comp_nonmedia", "delta_comp",
                 "delta_price", "cross_delta", "gamma"}


def _mean_or(v, default):
    v = float(v)
    return v if np.isfinite(v) and v > 1e-300 else default


def build_prior(df_tr, g, theta0, bounds):
    """
    Per-parameter prior over the flat theta vector.

    Bounds become support. Within the support:
      * Ls, G0, Hill n / power exponent, Weibull shape/scale, cross_n, mu:
          Uniform over their bounds (the same "box" the optimizer searched).
      * S (half-saturation): log-normal centred on its data-driven init
          (the channel median), log-sd 1.0, truncated to the bounds.
      * delta / gamma / cross_delta: (half-)normal, scaled so ONE prior sd
          is a channel contributing ~10% of the average target level.
          Sign comes from the bounds (positive / negative constraints).
      * sigma_y: half-normal with scale = sd of the training target.
      * I0: normal around its init, truncated at its lower bound.
      * rho: uniform(-0.95, 0.95); phi: half-normal(0.3).
      * lo == hi (frozen / pinned): fixed.
    """
    n = len(theta0)
    kind = np.full(n, K_TN, dtype=int)
    loc = np.zeros(n); scale = np.ones(n)
    lo = np.array([-np.inf if b[0] is None else float(b[0]) for b in bounds])
    hi = np.array([np.inf if b[1] is None else float(b[1]) for b in bounds])

    y = df_tr[g["TARGET_COL"]].values.astype(float)
    tm = _mean_or(abs(np.mean(y)), 1.0)
    sy = _mean_or(np.std(y), 1.0)
    p0 = unpack_theta(np.asarray(theta0, dtype=float), g)
    TT = g["TRANSFORM_TYPE"]; ITT = g.get("INTERCEPT_TRANSFORM_TYPE", "power")
    carry = g.get("INTERCEPT_DYNAMICS_TYPE", "carryover") != "simple"
    PRIOR_FRAC = 0.10

    def col(c):
        return df_tr[c].values.astype(float)

    def delta_scale(mag):
        mag = _mean_or(mag, 0.0)
        return PRIOR_FRAC * tm / mag if mag > 0 else 1.0

    for blk in theta_blocks(g):
        name, start, length = blk["name"], blk["start"], blk["length"]
        for j in range(length):
            i = start + j
            if lo[i] > -np.inf and hi[i] < np.inf and hi[i] - lo[i] <= 1e-12:
                kind[i] = K_FIXED; continue
            finite2 = np.isfinite(lo[i]) and np.isfinite(hi[i])
            x0i = float(theta0[i])

            if name in _UNIF_BLOCKS and finite2:
                kind[i] = K_UNIF
            elif name in _LOGTN_BLOCKS:
                kind[i] = K_LOGTN; loc[i] = max(x0i, 1e-6); scale[i] = 1.0
            elif name in _DELTA_BLOCKS:
                kind[i] = K_TN; loc[i] = 0.0
                if name == "delta":
                    c = blk["cols"][j]; x = col(c)
                    f = apply_transformation(x, TT, p0["n_params"][j], p0["S_params"][j])
                    scale[i] = delta_scale(np.mean(f * np.abs(x)))
                elif name == "delta_comp":
                    x = col(blk["cols"][j])
                    f = hill_transform_vec(x, p0["n_comp"][j], p0["S_comp"][j])
                    scale[i] = delta_scale(np.mean(f * np.abs(x)))
                elif name in ("delta_own_nonmedia", "delta_comp_nonmedia", "delta_price"):
                    x = col(blk["cols"][j]); scale[i] = delta_scale(np.mean(x ** 2))
                elif name == "cross_delta":
                    tgt, src = blk["pairs"][j]
                    f = hill_transform_vec(col(src), p0["cross_n"][j], p0["cross_S"][j])
                    scale[i] = delta_scale(np.mean(f * np.abs(col(tgt))))
                elif name == "gamma":
                    c = blk["cols"][j]
                    if c in df_tr.columns:
                        f = apply_transformation(col(c), ITT, p0["n_intercept"][j], p0["S_intercept"][j])
                        mag = _mean_or(np.mean(f), 0.0)
                        sc = PRIOR_FRAC * tm * ((1.0 - float(p0["G0"])) if carry else 1.0)
                        scale[i] = sc / mag if mag > 0 else 1.0
                    else:
                        scale[i] = 1.0
            elif name == "I0":
                kind[i] = K_TN; loc[i] = x0i; scale[i] = tm
            elif name == "sigma_y":
                kind[i] = K_TN; loc[i] = 0.0; scale[i] = sy
            else:  # unknown block / half-open bounds on a Uniform-type block
                kind[i] = K_TN; loc[i] = x0i; scale[i] = max(2.0 * abs(x0i), 0.5)
    return dict(kind=kind, loc=loc, scale=scale, lo=lo, hi=hi)


def concat_priors(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def joint_extras_prior(use_coupling, allow_phi1, allow_phi2):
    """Prior block for [rho | phi_1 | phi_2] (joint bivariate fit)."""
    kind = [K_UNIF]; loc = [0.0]; scale = [1.0]; lo = [-0.95]; hi = [0.95]
    if use_coupling:
        for allowed in (allow_phi1, allow_phi2):
            if allowed:
                kind.append(K_TN); loc.append(0.0); scale.append(0.3); lo.append(0.0); hi.append(np.inf)
            else:
                kind.append(K_FIXED); loc.append(0.0); scale.append(1.0); lo.append(0.0); hi.append(0.0)
    return dict(kind=np.array(kind), loc=np.array(loc), scale=np.array(scale),
                lo=np.array(lo), hi=np.array(hi))


def _prior_arrays(pr):
    """Sanitised arrays for the JAX transform (benign values on unused
    branches, so jnp.where never carries NaNs into the gradient)."""
    kind = pr["kind"]
    n = len(kind)
    unif_lo = np.where(kind == K_UNIF, pr["lo"], 0.0)
    unif_hi = np.where(kind == K_UNIF, pr["hi"], 1.0)
    fixed = np.where(kind == K_FIXED, pr["lo"], 0.0)
    tn_loc = np.zeros(n); tn_scale = np.ones(n)
    a = np.full(n, -np.inf); b = np.full(n, np.inf)
    is_log = kind == K_LOGTN
    for i in range(n):
        if kind[i] == K_TN:
            tn_loc[i], tn_scale[i] = pr["loc"][i], pr["scale"][i]
            a[i] = (pr["lo"][i] - tn_loc[i]) / tn_scale[i] if np.isfinite(pr["lo"][i]) else -np.inf
            b[i] = (pr["hi"][i] - tn_loc[i]) / tn_scale[i] if np.isfinite(pr["hi"][i]) else np.inf
        elif kind[i] == K_LOGTN:
            tn_loc[i] = np.log(max(pr["loc"][i], 1e-300)); tn_scale[i] = pr["scale"][i]
            lo_l = np.log(max(pr["lo"][i], 1e-300)) if np.isfinite(pr["lo"][i]) else -np.inf
            hi_l = np.log(pr["hi"][i]) if np.isfinite(pr["hi"][i]) else np.inf
            a[i] = (lo_l - tn_loc[i]) / tn_scale[i] if np.isfinite(lo_l) else -np.inf
            b[i] = (hi_l - tn_loc[i]) / tn_scale[i] if np.isfinite(hi_l) else np.inf
    return dict(kind=kind, unif_lo=unif_lo, unif_hi=unif_hi, fixed=fixed,
                tn_loc=tn_loc, tn_scale=tn_scale, a=a, b=b, is_log=is_log)


def _tn_std(u, a, b):
    """Standard truncated normal on [a, b] from a N(0,1) base variable, by
    inverse CDF (lower-tail form, mirrored to the upper tail if a > 0 for
    numerical stability)."""
    p = ndtr(u)
    Fa, Fb = ndtr(a), ndtr(b)
    zl = ndtri(jnp.clip(Fa + p * (Fb - Fa), 1e-300, 1.0 - 1e-16))
    Sa, Sb = ndtr(-a), ndtr(-b)
    zu = -ndtri(jnp.clip(Sa - p * (Sa - Sb), 1e-300, 1.0 - 1e-16))
    return jnp.where(a > 0, zu, zl)


def map_prior(u, pa):
    """u ~ N(0, I)  ->  theta with the configured prior."""
    th_unif = pa["unif_lo"] + (pa["unif_hi"] - pa["unif_lo"]) * ndtr(u)
    raw = pa["tn_loc"] + pa["tn_scale"] * _tn_std(u, pa["a"], pa["b"])
    th_tn = jnp.where(pa["is_log"], jnp.exp(jnp.clip(raw, -700.0, 700.0)), raw)
    k = pa["kind"]
    return jnp.where(k == K_UNIF, th_unif, jnp.where(k == K_FIXED, pa["fixed"], th_tn))


def inverse_map_prior(theta, pr):
    """numpy inverse of map_prior — used to start chains at a warm-start theta."""
    kind = pr["kind"]; u = np.zeros(len(kind))
    for i in range(len(kind)):
        th = float(theta[i])
        if kind[i] == K_UNIF:
            F = (th - pr["lo"][i]) / max(pr["hi"][i] - pr["lo"][i], 1e-300)
        elif kind[i] in (K_TN, K_LOGTN):
            loc, sc = pr["loc"][i], pr["scale"][i]
            lo_, hi_ = pr["lo"][i], pr["hi"][i]
            if kind[i] == K_LOGTN:
                th = np.log(max(th, 1e-300)); loc = np.log(max(loc, 1e-300))
                lo_ = np.log(max(lo_, 1e-300)) if np.isfinite(lo_) else -np.inf
                hi_ = np.log(hi_) if np.isfinite(hi_) else np.inf
            xs = (th - loc) / sc
            a = (lo_ - loc) / sc if np.isfinite(lo_) else -np.inf
            b = (hi_ - loc) / sc if np.isfinite(hi_) else np.inf
            Fa, Fb = _sp_ndtr(a), _sp_ndtr(b)
            F = (_sp_ndtr(xs) - Fa) / max(Fb - Fa, 1e-300)
        else:
            continue
        u[i] = float(np.clip(_sp_ndtri(np.clip(F, 1e-6, 1 - 1e-6)), -3.5, 3.5))
    return u


# ═════════════════════════════════════════════════════════════════════════
# 4. THE PROBABILISTIC MODEL
# ═════════════════════════════════════════════════════════════════════════

def _split_joint(th, n1, n2, use_coupling):
    th1, th2 = th[:n1], th[n1:n1 + n2]
    rho = th[n1 + n2]
    if use_coupling:
        return th1, th2, rho, th[n1 + n2 + 1], th[n1 + n2 + 2]
    return th1, th2, rho, 0.0, 0.0


def make_model(eqs, prior, joint=None):
    """
    eqs   : list with one (single) or two (joint) dicts {g, st, n_theta}
    prior : concatenated prior over the flat parameter vector
            (joint: [theta_1 | theta_2 | rho (| phi_1 | phi_2)])
    joint : None, or dict(use_coupling=bool)
    Returns (model_fn, pieces) where pieces exposes the pure functions used
    for post-processing draws.
    """
    pa = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in _prior_arrays(prior).items()}
    n_total = len(prior["kind"])
    n_thetas = [e["n_theta"] for e in eqs]
    T = eqs[0]["st"]["T"]
    dims = [e["st"]["dim"] for e in eqs]
    consts = [{k: (jnp.asarray(e["st"][k]) if k in ("x0", "sd0", "qsd", "lo", "hi", "y", "L", "obs_mask") else e["st"][k])
               for k in e["st"]} for e in eqs]
    use_coupling = bool(joint and joint.get("use_coupling"))

    def simulate(u, zs):
        """Pure function: base variables -> (theta, states, per-eq extras)."""
        th = map_prior(u, pa)
        if joint is None:
            ths = [th]; rho = phi1 = phi2 = 0.0
        else:
            th1, th2, rho, phi1, phi2 = _split_joint(th, n_thetas[0], n_thetas[1], use_coupling)
            ths = [th1, th2]
        ps = [unpack_theta(t, e["g"]) for t, e in zip(ths, eqs)]
        dyn = [build_dynamics(p, c) for p, c in zip(ps, consts)]
        if joint is None:
            Td, U, cross, lt = dyn[0]; c = consts[0]
            X = rollout(Td, U, c["x0"], c["sd0"], zs[0], c["qsd"], c["lo"], c["hi"])
            Xs = [X]
        else:
            Td = jnp.concatenate([d[0] for d in dyn]); U = jnp.concatenate([d[1] for d in dyn], axis=1)
            x0 = jnp.concatenate([c["x0"] for c in consts]); sd0 = jnp.concatenate([c["sd0"] for c in consts])
            qsd = jnp.concatenate([c["qsd"] for c in consts])
            lo = jnp.concatenate([c["lo"] for c in consts]); hi = jnp.concatenate([c["hi"] for c in consts])
            z = jnp.concatenate(zs, axis=1)
            X = rollout(Td, U, x0, sd0, z, qsd, lo, hi, cpl_idx=dims[0], phi1=phi1, phi2=phi2)
            Xs = [X[:, :dims[0]], X[:, dims[0]:]]
        mus = [(c["L"] * Xe).sum(axis=1) for c, Xe in zip(consts, Xs)]
        return th, ps, dyn, Xs, mus, rho, phi1, phi2

    def loglik_terms(ps, mus, rho):
        """Log-likelihood of the TRAINING rows (mask), plus per-row terms."""
        m = consts[0]["obs_mask"]
        if joint is None:
            s = ps[0]["sigma_y"]
            r = (consts[0]["y"] - mus[0]) / s
            row = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(s) - 0.5 * r ** 2
        else:
            s1, s2 = ps[0]["sigma_y"], ps[1]["sigma_y"]
            rho_c = jnp.clip(rho, -0.995, 0.995)
            r1 = (consts[0]["y"] - mus[0]) / s1
            r2 = (consts[1]["y"] - mus[1]) / s2
            quad = (r1 ** 2 - 2 * rho_c * r1 * r2 + r2 ** 2) / (1 - rho_c ** 2)
            row = (-jnp.log(2 * jnp.pi) - jnp.log(s1) - jnp.log(s2)
                   - 0.5 * jnp.log(1 - rho_c ** 2) - 0.5 * quad)
        return row * m

    def model():
        u = numpyro.sample("u", dist.Normal(jnp.zeros(n_total), 1.0).to_event(1))
        zs = [numpyro.sample(f"z{e}", dist.Normal(0.0, 1.0).expand([T, d]).to_event(2))
              for e, d in enumerate(dims)]
        th, ps, dyn, Xs, mus, rho, phi1, phi2 = simulate(u, zs)
        ll = loglik_terms(ps, mus, rho).sum()
        numpyro.factor("y_obs", ll)
        numpyro.deterministic("theta", th)
        numpyro.deterministic("loglik", ll)

    return model, dict(simulate=simulate, loglik_terms=loglik_terms, prior_arrays=pa,
                       n_total=n_total, dims=dims, T=T, consts=consts)


# ═════════════════════════════════════════════════════════════════════════
# 5. RUN NUTS
# ═════════════════════════════════════════════════════════════════════════

def run_nuts(model, pieces, cfg, u_init=None, progress_cb=None):
    """Sample the joint posterior. Returns (samples_by_chain, diagnostics)."""
    cfg = mcmc_cfg_with_defaults(cfg)
    C, W, S = cfg["num_chains"], cfg["num_warmup"], cfg["num_samples"]
    n_total, dims, T = pieces["n_total"], pieces["dims"], pieces["T"]
    rng = np.random.default_rng(cfg["seed"])

    u0 = np.zeros(n_total) if u_init is None else np.asarray(u_init, dtype=float)
    init = {"u": u0[None, :] + 0.05 * rng.standard_normal((C, n_total))}
    for e, d in enumerate(dims):
        init[f"z{e}"] = 0.01 * rng.standard_normal((C, T, d))

    chain_method = "parallel" if (C > 1 and jax.local_device_count() >= C) else "sequential"
    kernel = NUTS(model, target_accept_prob=cfg["target_accept"],
                  max_tree_depth=cfg["max_tree_depth"], dense_mass=False)
    chunk = max(25, S // 10)
    mcmc = MCMC(kernel, num_warmup=W, num_samples=S, num_chains=C,
                chain_method=chain_method, progress_bar=False)

    t0 = time.time()
    key = jax.random.PRNGKey(cfg["seed"])
    if progress_cb:
        progress_cb(0.0, f"NUTS warm-up ({W} iterations × {C} chain(s)) …")
    mcmc.warmup(key, init_params=init, collect_warmup=False,
                extra_fields=("diverging", "accept_prob", "num_steps"))

    # Draw in chunks so the UI can show progress.
    got, chunks, extras = 0, [], []
    run_key = mcmc.post_warmup_state.rng_key
    while got < S:
        m = min(chunk, S - got)
        mcmc.num_samples = m
        mcmc.run(run_key, extra_fields=("diverging", "accept_prob", "num_steps"))
        chunks.append({k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()})
        ex = mcmc.get_extra_fields(group_by_chain=True)
        extras.append({k: np.asarray(v) for k, v in ex.items()})
        mcmc.post_warmup_state = mcmc.last_state
        run_key = mcmc.post_warmup_state.rng_key
        got += m
        if progress_cb:
            progress_cb(got / S, f"NUTS sampling — {got}/{S} draws per chain")

    samples = {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0]}
    ex_all = {k: np.concatenate([e[k] for e in extras], axis=1) for k in extras[0]}
    diag = dict(
        divergences=int(np.sum(ex_all["diverging"])),
        mean_accept=float(np.mean(ex_all["accept_prob"])),
        mean_leapfrog=float(np.mean(ex_all["num_steps"])),
        runtime_s=time.time() - t0, chain_method=chain_method,
        num_warmup=W, num_samples=S, num_chains=C, target_accept=cfg["target_accept"],
    )
    return samples, diag


# ═════════════════════════════════════════════════════════════════════════
# 6. POSTERIOR SUMMARIES
# ═════════════════════════════════════════════════════════════════════════

def theta_labels(g, prefix=""):
    labels = []
    for blk in theta_blocks(g):
        for j in range(blk["length"]):
            if blk["cols"] is not None:
                tag = f"{blk['name']}[{blk['cols'][j]}]"
            elif blk["pairs"] is not None:
                tgt, src = blk["pairs"][j]; tag = f"{blk['name']}[{src}→{tgt}]"
            else:
                tag = blk["name"]
            labels.append(prefix + tag)
    return labels


def parameter_summary(theta_by_chain, labels, kind):
    """r_hat / ESS / quantiles for every free parameter."""
    free = np.where(kind != K_FIXED)[0]
    sm = _np_summary({"theta": theta_by_chain[..., free]}, prob=0.95, group_by_chain=True)["theta"]
    flat = theta_by_chain.reshape(-1, theta_by_chain.shape[-1])[:, free]
    rows = pd.DataFrame({
        "Parameter": [labels[i] for i in free],
        "Mean": flat.mean(0), "SD": flat.std(0), "Median": np.median(flat, 0),
        "2.5%": np.percentile(flat, 2.5, 0), "97.5%": np.percentile(flat, 97.5, 0),
        "ESS": np.asarray(sm["n_eff"]), "R-hat": np.asarray(sm["r_hat"]),
    })
    return rows


def posterior_states(pieces, samples, n_keep, chunk=40):
    """
    Push posterior draws back through the model equations to get, per
    equation: posterior-mean states, the state covariance across draws
    (used as P_smooth), per-draw-averaged long-term intercept pieces, and a
    thinned set of full draws for credible bands.
    """
    sim = jax.jit(jax.vmap(lambda u, *zs: _sim_out(pieces, u, list(zs))))
    u_all = samples["u"].reshape(-1, samples["u"].shape[-1])
    zs_all = [samples[f"z{e}"].reshape(-1, *samples[f"z{e}"].shape[2:]) for e in range(len(pieces["dims"]))]
    S = u_all.shape[0]
    keep_idx = np.unique(np.linspace(0, S - 1, min(n_keep, S)).astype(int))
    keep_set = set(keep_idx.tolist())

    n_eq = len(pieces["dims"])
    sum_x = [0.0] * n_eq; sum_xx = [0.0] * n_eq
    sum_lt = [0.0] * n_eq; sum_cross = [0.0] * n_eq; sum_carry = [0.0] * n_eq
    sum_mu = [0.0] * n_eq
    kept = dict(theta=[], X=[[] for _ in range(n_eq)], lt=[[] for _ in range(n_eq)], ll=[])
    ll_sum = 0.0

    for s0 in range(0, S, chunk):
        sl = slice(s0, min(s0 + chunk, S))
        out = sim(jnp.asarray(u_all[sl]), *[jnp.asarray(z[sl]) for z in zs_all])
        out = jax.tree_util.tree_map(np.asarray, out)
        th, Xs, lts, crosses, carries, mus, ll = out
        ll_sum += float(ll.sum())
        for e in range(n_eq):
            sum_x[e] = sum_x[e] + Xs[e].sum(0)
            sum_xx[e] = sum_xx[e] + np.einsum("ctd,cte->tde", Xs[e], Xs[e])
            sum_lt[e] = sum_lt[e] + lts[e].sum(0)
            sum_cross[e] = sum_cross[e] + crosses[e].sum(0)
            sum_carry[e] = sum_carry[e] + carries[e].sum(0)
            sum_mu[e] = sum_mu[e] + mus[e].sum(0)
        for local, gi in enumerate(range(sl.start, sl.stop)):
            if gi in keep_set:
                kept["theta"].append(th[local]); kept["ll"].append(float(ll[local]))
                for e in range(n_eq):
                    kept["X"][e].append(Xs[e][local]); kept["lt"][e].append(lts[e][local])

    eqs_out = []
    for e in range(n_eq):
        mean_x = sum_x[e] / S
        cov = sum_xx[e] / S - np.einsum("td,te->tde", mean_x, mean_x)
        eqs_out.append(dict(
            x_mean=mean_x, P=cov, lt_mean=sum_lt[e] / S, cross_mean=sum_cross[e] / S,
            carry_mean=sum_carry[e] / S, mu_mean=sum_mu[e] / S,
            X_keep=np.stack(kept["X"][e]), lt_keep=np.stack(kept["lt"][e]),
        ))
    return dict(eqs=eqs_out, theta_keep=np.stack(kept["theta"]),
                loglik_mean=ll_sum / S, loglik_keep=np.array(kept["ll"]), n_draws=S)


def _sim_out(pieces, u, zs):
    th, ps, dyn, Xs, mus, rho, phi1, phi2 = pieces["simulate"](u, zs)
    ll = pieces["loglik_terms"](ps, mus, rho).sum()
    lts = [d[3] for d in dyn]
    crosses = [d[2] for d in dyn]
    # intercept carryover piece  G0_s * I_{t-1,s}  (per draw — nonlinear in G0 x I)
    carries = []
    for p, X in zip(ps, Xs):
        prev = jnp.concatenate([X[:1, 0], X[:-1, 0]])
        carries.append(jnp.asarray(p["G0"], dtype=jnp.float64) * prev)
    return th, Xs, lts, crosses, carries, mus, ll
