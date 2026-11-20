"""
State-space equation building blocks shared by the MCMC core
(modules/mcmc.py) and the result post-processing (modules/pipeline.py).

The EQUATIONS are unchanged from the earlier Kalman-filter build — only the
inference engine changed (Kalman filter + RTS smoother + point-estimate
optimizer  ->  MCMC / NUTS over parameters AND latent states). What is left
in this module is the numpy-side plumbing that does not depend on the
parameters being inferred:

  _build_observation_matrix  the L_t matrix (raw regressors, intercept first)
  _build_process_noise       Q, the per-state process-noise variances
  _precompute_adstocked      Weibull lag-sum series (for display / export)
  _transform_media           Hill / power dispatcher

The parameter-dependent part of each equation (transition diagonal,
forcing terms, intercept boost) lives in modules/mcmc.py as JAX code so it
can be differentiated by NUTS; it is a line-for-line port of the equations
below (see tests in the change notes: forcing terms and transition
diagonal match the original numpy implementation to machine precision).

State equations (all 4 adstock × transform combinations), per dependent
variable:

Instant + Power:
  β_i,t = Ls_i · β_i,t-1  +  δ_i · x_i,t^n_i
           + Σ_j∈synergy  δ_ij · x_j,t^n_j
  (no adstock λ: carryover lives entirely in Ls_i, the beta-persistence
  term. Multiplying the shock by an additionally-decayed/adstocked series
  here would double-count the same carryover twice — once via Ls_i's own
  AR(1) memory, once via λ's geometric decay — so the shock always uses
  RAW spend/impressions, never adstocked media.)

Instant + Hill:
  β_i,t = Ls_i · β_i,t-1  +  δ_i · Hill(x_i,t; n_i, S_i)
           + Σ_j∈synergy  δ_ij · Hill(x_j,t; n_j, S_j)
  (same reasoning — x_i,t and every synergy x_j,t are RAW, not adstocked.
  Competitor-media betas follow the identical pattern: Ls_comp · β_comp,t-1
  + δ_comp · Hill(raw comp spend).)

Weibull + Power:
  β_i,t = Σ_{l=1}^{j} w_l · x_i,t-l   (weighted PAST-lag sum, j = user-selected lags, 0–8)
           +  δ_i · x_i,t^n_i
           + Σ_j∈synergy  δ_ij · x_j,t^n_j

Weibull + Hill:
  β_i,t = Σ_{l=1}^{j} w_l · x_i,t-l
           +  δ_i · Hill(x_i,t; n_i, S_i)
           + Σ_j∈synergy  δ_ij · Hill(x_j,t; n_j, S_j)

  Weibull per-lag weight (l = 1, 2, …, j), normalised to sum to 1
  (w_1 + w_2 + … + w_j = 1). The current period (l=0 / x_t) is NOT part
  of this lag sum — it only enters via the δ_i·f(x_i,t) shock term above:
    w_l = (k/λ) · (l/λ)^(k−1) · exp( −(l/λ)^k )

Intercept — two independent switches: a Transform Type (Power or Hill,
config "intercept_transform_type") for the effector boost shape, and a
Dynamics Type (Carryover or Simple, config "intercept_dynamics_type")
for whether the intercept persists period-to-period at all:

  Carryover (default) + Power:
    I_t = G0 · I_{t-1}  +  Σ_k γ_k · media_k,t^{n_k_intercept}

  Carryover (default) + Hill:
    I_t = G0 · I_{t-1}  +  Σ_k γ_k ·
              media_k,t^{n_k_intercept} /
              ( media_k,t^{n_k_intercept} + S_k_intercept^{n_k_intercept} )

  Simple (no carryover — pure regression on current-period effectors) + Power:
    I_t = I0  +  Σ_k γ_k · media_k,t^{n_k_intercept}

  Simple (no carryover — pure regression on current-period effectors) + Hill:
    I_t = I0  +  Σ_k γ_k ·
              media_k,t^{n_k_intercept} /
              ( media_k,t^{n_k_intercept} + S_k_intercept^{n_k_intercept} )

  Every intercept-effector column — whether or not it is also a media
  channel with its own beta — is transformed the same way, with its own
  independently-fitted n_k_intercept (and S_k_intercept, Hill only).

  In Simple mode, G0 is fixed at 0 (dropped out of theta entirely) and a
  fitted constant baseline I0 takes its place — implemented as: Tmat[0,0]
  stays 0 (so the Tmat @ x_prev matrix multiply contributes nothing to the
  intercept row) and I0 is added alongside the effector boost in the
  additive (nonlinear) part of the predict step, exactly where G0·I_{t-1}
  would otherwise have been folded in via the matrix multiply. See
  modules/params.py::unpack_theta and _predict_step below.

──────────────────────────────────────────────────────────────────────────
Joint (bivariate) fit
──────────────────────────────────────────────────────────────────────────
When a second dependent variable is configured, the two equations share a
time index; their observation errors are correlated through a freely-
estimated rho, and (in carryover mode) each intercept picks up the other's
previous intercept through phi_1 / phi_2:

  Intercept_1,t = G0_1 · Intercept_1,t-1 + phi_1 · Intercept_2,t-1 + effectors_1,t
  Intercept_2,t = G0_2 · Intercept_2,t-1 + phi_2 · Intercept_1,t-1 + effectors_2,t

Under MCMC these are simply two coupled latent-state blocks inside one
posterior; rho, phi_1 and phi_2 are sampled together with everything else.
"""

import numpy as np
import pandas as pd

from modules.transforms import (
    apply_transformation,
    adstock_weibull_lagged,
)


def _precompute_adstocked(df, g, params):
    """
    Per-channel now: for every channel (in MEDIA_COLS, COMP_MEDIA_COLS,
    OWN_NONMEDIA_COLS, or COMP_NONMEDIA_COLS) individually set to
    "weibull" in g["ADSTOCK_MAP"], compute its weighted-lag sum using its
    own fitted shape/scale (looked up via g["ADSTOCK_IDX"]). Channels left
    on "instant" (Nerlove-Arrow) are NOT computed — their carryover is
    carried entirely by that state's own Ls persistence (β_i,t = Ls_i ·
    β_i,t-1 + shock); building a separately-decayed adstocked series for
    them would only invite a double-carryover bug (Ls persistence stacked
    on top of a λ decay). Returns dict col -> np.ndarray of length T,
    containing only the weibull-selected channels.
    """
    adstock_map = g.get("ADSTOCK_MAP", {})
    adstock_idx = g.get("ADSTOCK_IDX", {})
    n_lags = int(params.get("adstock_n_lags", 8))
    adstocked = {}

    for col in (list(g["MEDIA_COLS"]) + list(g["COMP_MEDIA_COLS"]) +
                list(g["OWN_NONMEDIA_COLS"]) + list(g["COMP_NONMEDIA_COLS"])):
        if adstock_map.get(col) != "weibull" or col not in adstock_idx:
            continue
        ci = adstock_idx[col]
        adstocked[col] = adstock_weibull_lagged(
            df[col], params["adstock_shape"][ci], params["adstock_scale"][ci], n_lags)

    return adstocked


def _build_observation_matrix(df, g, adstocked_media):
    """
    y_t = I_t · 1
          + Σ β_i,t · raw_spend_i,t             (own media — raw spend/impressions,
                                                  NOT adstocked; carryover is already
                                                  carried by β_i,t's own λ_i decay in
                                                  the state equation, so multiplying by
                                                  adstocked media here would double-count
                                                  the carryover effect)
          + Σ β_j,t · raw_spend_j,t             (comp media — raw, same reasoning)
          + Σ β_k  · nonmedia_k,t               (own non-media — raw)
          + Σ β_k  · comp_nonmedia_k,t          (comp non-media — raw)
          + Σ β_p  · price_p,t
          + Σ β_d  · dummy_d,t

    Note: `adstocked_media` is still passed in and still used elsewhere (Weibull
    per-lag state transitions, Hill-on-adstocked comp/synergy shock terms in
    _prepare_equation) — it is simply no longer what the observation equation
    multiplies β by.
    """
    T = len(df)
    cols = [np.ones(T)]
    for c in g["MEDIA_COLS"]:         cols.append(df[c].values.astype(float))
    for c in g["COMP_MEDIA_COLS"]:    cols.append(df[c].values.astype(float))
    for c in g["OWN_NONMEDIA_COLS"]:  cols.append(df[c].values.astype(float))
    for c in g["COMP_NONMEDIA_COLS"]: cols.append(df[c].values.astype(float))
    for c in g["PRICE_COLS"]:         cols.append(df[c].values.astype(float))
    for c in g["DUMMY_COLS"]:         cols.append(df[c].values.astype(float))
    return np.column_stack(cols)


def _build_process_noise(df, g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + N_PRICE + N_DUMMIES + SEASONAL_DIM
    Q = np.eye(dim) * 1e-6
    target_mean = float(df[g["TARGET_COL"]].mean())

    # Intercept process noise: previously a flat 1e-4 regardless of the
    # target's actual scale/units, which made it negligible for most
    # business data and left the intercept almost frozen wherever its very
    # first (loosely-constrained) update landed — including negative.
    # Scaling it off the target's own mean gives the intercept genuine,
    # data-scale-appropriate period-to-period flexibility so it can drift
    # back toward a sensible level instead of getting stuck.
    intercept_noise_scale = float(g.get("INTERCEPT_NOISE_SCALE", 0.0))
    if intercept_noise_scale > 0 and target_mean > 0:
        Q[0, 0] = (intercept_noise_scale * target_mean) ** 2
    else:
        Q[0, 0] = 1e-6  # legacy fallback (effectively frozen)

    # Beta states (media / comp-media / non-media / comp-non-media / price):
    # each beta evolves as β_t = Ls·β_{t-1} + δ·trigger_t. With a flat,
    # unit-free 1e-6 process noise the filter treated that as almost
    # perfectly deterministic — no real ability to wiggle back up on its
    # own. Combined with Ls < 1, that guarantees geometric decay toward
    # zero for EVERY channel whenever its forcing term weakens, regardless
    # of whether that's actually true of that channel's real-world effect.
    #
    # Scale each beta's process noise so a "beta_noise_scale" fraction of
    # this is honoured consistently across channels of very different
    # units (spend in thousands vs. a 0/1 flag, say): allow the beta to
    # wander enough, per period, that — once multiplied by that channel's
    # own typical magnitude — the resulting contribution could move by
    # roughly `beta_noise_scale` × the target's average value. This is
    # the direct beta-level analogue of the intercept treatment above.
    beta_noise_scale = float(g.get("BETA_NOISE_SCALE", 0.0))

    def _beta_q(col):
        if beta_noise_scale <= 0 or target_mean <= 0 or col not in df.columns:
            return 1e-6  # legacy fallback (effectively frozen)
        reg_mean = float(np.mean(np.abs(df[col].values)))
        if reg_mean < 1e-9:
            return 1e-6
        return (beta_noise_scale * target_mean / reg_mean) ** 2

    for i, col in enumerate(g["MEDIA_COLS"]):
        Q[i + 1, i + 1] = _beta_q(col)
    for j, col in enumerate(g["COMP_MEDIA_COLS"]):
        Q[1 + N_MEDIA + j, 1 + N_MEDIA + j] = _beta_q(col)
    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        idx = 1 + N_MEDIA + N_COMP + k
        Q[idx, idx] = _beta_q(col)
    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        idx = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + k
        Q[idx, idx] = _beta_q(col)
    for p, col in enumerate(g["PRICE_COLS"]):
        idx = 1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + p
        Q[idx, idx] = _beta_q(col)

    # Spike dummies sit AFTER the price states in the state vector (see
    # _build_observation_matrix). The Kalman build omitted N_PRICE from this
    # index, so whenever a price column and a spike dummy were both in the
    # model the dummy's 5e-3 noise landed on the PRICE state instead (and the
    # dummy kept 1e-6). Fixed here; the same slip existed for the dummy's
    # 0.98 persistence in the old transition matrix (now correct in mcmc.py).
    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+N_PRICE+d; Q[idx, idx] = 5e-3
    return Q


def _transform_media(x: np.ndarray, transform_type: str,
                     n: float, S: float) -> np.ndarray:
    """Apply configured transformation to a media array."""
    return apply_transformation(x, transform_type, n, S)
