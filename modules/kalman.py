"""
Recursive Bayesian Estimation core: observation/transition/process-noise matrix
builders, the forward filter, and the RTS (Rauch-Tung-Striebel) smoother.

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

  Weibull (multi-lag Weibull-adstocked carryover — a distributed-lag,
  potentially delayed/S-shaped persistence, reaching back L =
  INTERCEPT_WEIBULL_N_LAGS periods, in place of AR(1)'s single-lag G0):
    I_t = G0 · Σ_{l=1}^{L} w_l · I_{t-l}  +  Σ_k γ_k · f(media_k,t)
    (w_1..w_L are normalised Weibull PDF weights summing to 1 — same
    weibull_lag_weights() function used for per-channel media adstock,
    fitted shape k / scale λ. G0 here plays the SAME overall-persistence
    role/bound as AR(1)'s G0 — spread across L lags via the Weibull shape
    instead of concentrated at lag 1 — and keeps the AR(L) feedback
    stationary/mean-reverting rather than a unit root, since the Weibull
    weights alone always sum to exactly 1.)

  Implemented via L-1 extra "shadow lag" states appended to the END of the
  state vector (after every other block — media/comp/non-media/price/
  dummies/seasonal), holding I_{t-1}..I_{t-L+1} — the intercept's own
  index-0 slot still holds I_t as always. The transition matrix Tmat
  encodes this as an AR(L) companion form: row 0 gets the Weibull weights
  w_1 (on Tmat[0,0], i.e. its own previous value) and w_2..w_L (on the
  shadow-lag columns); each shadow-lag row is a pure 1.0 shift-copy of the
  slot "in front of" it. This keeps every other state's index unchanged
  (media betas etc. are indexed identically to non-weibull-intercept
  models) and lets the existing `Tmat @ x_prev` matrix multiply in
  _predict_step do the entire AR(L) propagation — including the lag
  shifting — with zero changes to _predict_step itself. See
  modules/kalman.py::_intercept_extra_lags / _build_transition_matrix /
  _build_observation_matrix / _initial_state.

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
Joint (bivariate) Kalman-filter fit
──────────────────────────────────────────────────────────────────────────
When a second dependent variable is configured (e.g. Consideration /
Top-of-Mind alongside Sales), the two equations above are NOT fitted as
two separate, independent RBE runs. Instead they are stacked into a single
joint (bivariate) state-space model and filtered together:

    [ sales_t         ]   [ Intercept_sales_t         ]   [ error_sales_t         ]
    [ consideration_t ] = [ Intercept_consideration_t ] + [ error_consideration_t ]

                            [ beta_sales_1_t         ... beta_sales_M_t         ]
                          + [ beta_consid_1_t        ... beta_consid_M_t       ] · x_t

where x_t is the vector of raw media/control regressors (e.g. raw TV
GRPs) — media/comp-media columns are RAW spend/impressions, not
adstocked; carryover is carried entirely by β_i,t's own Ls_i persistence
(instant mode) or, in Weibull mode, the weighted-lag sum, in the state
equation — never by adstocking the observation-side regressor, and
never by adstocking the shock term either (that would double-count the
same carryover twice). Concretely:

  • The joint transition matrix and process-noise matrix are block
    diagonal EXCEPT for two off-diagonal cross-intercept coupling terms
    (see "Cross-intercept coupling" below) — each equation's betas evolve
    according to their own dynamics (own Ls/adstock), but both blocks are
    propagated and updated within the same filter recursion, at the same
    time index t.
  • The joint observation matrix L_t is 2 × dim_joint, with the sales row
    loading only on the sales-equation block and the consideration row
    loading only on the consideration-equation block:
        L_t = [ L_sales,t        0            ]
              [ 0                L_consid,t   ]
  • The observation noise is a full 2×2 covariance matrix
        R = [ σ_sales²                  ρ·σ_sales·σ_consid ]
            [ ρ·σ_sales·σ_consid        σ_consid²           ]
    where ρ is a freely-estimated correlation between the two equations'
    contemporaneous shocks. This is what makes the fit genuinely "joint"
    rather than two separate univariate fits glued together: information
    about one equation's residual at time t informs the state update of
    the other equation at the same t (via the off-diagonal Kalman gain
    terms), and ρ itself is estimated jointly with all other parameters
    by maximising the bivariate Gaussian log-likelihood.
  • The RTS smoother is dimension-agnostic and is re-used unmodified on
    the joint (dim_1+dim_2)-length state.

──────────────────────────────────────────────────────────────────────────
Cross-intercept coupling (phi_1, phi_2)
──────────────────────────────────────────────────────────────────────────
Each equation's intercept also depends on the OTHER equation's previous
intercept, in addition to its own persistence and its own intercept
effectors:

  Intercept_1,t = G0_1 · Intercept_1,t-1  +  phi_1 · Intercept_2,t-1
                  + Σ_k γ_1,k · effector_1,k,t^{n_1,k}

  Intercept_2,t = G0_2 · Intercept_2,t-1  +  phi_2 · Intercept_1,t-1
                  + Σ_k γ_2,k · effector_2,k,t^{n_2,k}

phi_1 and phi_2 are estimated jointly with everything else (they only
exist/are fitted when a second dependent variable is configured). This
is implemented as two off-diagonal entries in the joint transition
matrix — Tmat_joint[0, dim1] = phi_1 and Tmat_joint[dim1, 0] = phi_2 —
and the matching additive terms in the mean prediction step.

Cross-intercept coupling only applies when BOTH equations are on
"carryover" intercept dynamics — it is itself a carryover mechanism (each
equation's intercept picking up the OTHER equation's PREVIOUS value), so
it is dropped (phi_1 = phi_2 = 0, and their theta slots omitted entirely)
whenever config "intercept_dynamics_type" is "simple". See
modules/pipeline.py::run_multi_dependent_pipeline.
"""

import numpy as np
import pandas as pd

from modules.transforms import (
    apply_transformation,
    hill_transform_vec,
    adstock_weibull_lagged,
    adstock_nerlove_arrow,
    weibull_lag_weights,
)


# ── Intercept Weibull carryover: state-dimension helpers ─────────────────────

def _intercept_extra_lags(g):
    """Number of extra 'shadow lag' states appended to the END of the state
    vector for the intercept's own multi-lag Weibull carryover (0 unless
    INTERCEPT_DYNAMICS_TYPE == 'weibull'). L = INTERCEPT_WEIBULL_N_LAGS
    total lags needs L-1 extra states — I_t itself still lives at index 0,
    exactly as in "carryover"/"simple" mode."""
    if g.get("INTERCEPT_DYNAMICS_TYPE", "carryover") == "weibull":
        return max(int(g.get("INTERCEPT_WEIBULL_N_LAGS", 4)) - 1, 0)
    return 0


def _base_state_dim(g):
    """State dimension WITHOUT the intercept's extra Weibull shadow-lag
    states — i.e. the original single-intercept-slot dim formula, used as
    both dependencies' components. This is where the shadow-lag block
    (if any) starts."""
    return (1 + g["N_MEDIA"] + g["N_COMP"] + g["N_OWN_NONMEDIA"] +
            g["N_COMP_NONMEDIA"] + g["N_PRICE"] + g["N_DUMMIES"] + g["SEASONAL_DIM"])


# ── Adstock pre-computation ──────────────────────────────────────────────────

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


# ── Observation matrix ───────────────────────────────────────────────────────

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
    # Intercept Weibull shadow-lag states (if any) carry I_{t-1}..I_{t-L+1}
    # for the state transition's own use only — they don't separately enter
    # the observation equation (I_t at index 0 already does, via the
    # leading all-ones column above), so they load as zero columns here.
    for _ in range(_intercept_extra_lags(g)):
        cols.append(np.zeros(T))
    return np.column_stack(cols)


# ── Transition matrix ────────────────────────────────────────────────────────

def _build_transition_matrix(g, params):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    base_dim = _base_state_dim(g)
    extra_lags = _intercept_extra_lags(g)
    dim = base_dim + extra_lags
    Tmat = np.eye(dim)
    if g.get("INTERCEPT_DYNAMICS_TYPE", "carryover") == "weibull" and extra_lags >= 0:
        # AR(L) companion form: row 0 (I_t) picks up the Weibull-weighted
        # sum of I_{t-1} (its own previous value, Tmat[0,0]) and
        # I_{t-2}..I_{t-L} (the shadow-lag states, Tmat[0, base_dim+j]).
        # Each shadow-lag row is a pure shift-copy of the slot in front of
        # it (weight 1.0) — together these fully implement
        # I_t = Σ_l w_l·I_{t-l} AND the lag-history shift, so the plain
        # `Tmat @ x_prev` multiply in _predict_step needs no other change.
        n_lags = extra_lags + 1
        w = weibull_lag_weights(params["intercept_weibull_shape"],
                                 params["intercept_weibull_scale"], n_lags)
        # w sums to exactly 1 by construction (weibull_lag_weights
        # normalises it) — multiplying by G0 (bounded < 1, see
        # modules/bounds.py) keeps the AR(L) companion matrix's
        # coefficients summing to G0 < 1, i.e. stationary/mean-reverting,
        # instead of a unit root.
        G0 = params["G0"]
        Tmat[0, 0] = G0 * w[0]
        for j in range(extra_lags):
            Tmat[0, base_dim + j] = G0 * w[j + 1]
        if extra_lags >= 1:
            # Clear the shadow-lag block's identity diagonal first — np.eye
            # left a 1.0 there, which (left in place alongside the shift
            # entries below) would double-count each shadow-lag state every
            # step and blow up exponentially instead of just shifting.
            for j in range(extra_lags):
                Tmat[base_dim + j, base_dim + j] = 0.0
            Tmat[base_dim, 0] = 1.0
            for j in range(1, extra_lags):
                Tmat[base_dim + j, base_dim + j - 1] = 1.0
    else:
        Tmat[0, 0] = params["G0"]
    ADSTOCK_MAP = g.get("ADSTOCK_MAP", {})
    # Per channel now — weibull mode: β_i,t = Σ_l w_l·x_i,t-l + δ_i·f(x_i,t) + synergy
    # (no λ·β_t-1 term — the weighted-lag sum itself supplies the state's
    # "memory"). Instant mode: β_i,t = λ_i·β_i,t-1 + δ_i·f(x_i,t) + synergy.
    # This weibull-vs-instant branch now applies independently to each
    # channel in every group below (own media, comp media, own non-media,
    # comp non-media) — price never gets a weibull option.
    for i, col in enumerate(g["MEDIA_COLS"]):
        Tmat[i+1, i+1] = 0.0 if ADSTOCK_MAP.get(col) == "weibull" else params["Ls"][i]
    for j, col in enumerate(g["COMP_MEDIA_COLS"]):
        idx = 1+N_MEDIA+j
        Tmat[idx, idx] = 0.0 if ADSTOCK_MAP.get(col) == "weibull" else params["Ls_comp"][j]
    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        idx = 1+N_MEDIA+N_COMP+k
        Tmat[idx, idx] = 0.0 if ADSTOCK_MAP.get(col) == "weibull" else params["Ls_own_nonmedia"][k]
    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k
        Tmat[idx, idx] = 0.0 if ADSTOCK_MAP.get(col) == "weibull" else params["Ls_comp_nonmedia"][k]
    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+d; Tmat[idx, idx] = 0.98
    for p in range(N_PRICE):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p; Tmat[idx, idx] = params["Ls_price"][p]
    return Tmat


# ── Process noise ────────────────────────────────────────────────────────────

def _build_process_noise(df, g):
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    dim = _base_state_dim(g) + _intercept_extra_lags(g)
    # Intercept Weibull shadow-lag states (indices base_dim..dim-1, if any)
    # keep the default 1e-6 (effectively frozen) below — they're exact
    # deterministic shift-copies of a previous slot via Tmat, not
    # independently-noisy states, mirroring how weibull-adstocked media
    # betas' own Tmat diagonal is 0 (no separate noise source needed
    # beyond what index 0's Q[0,0] already supplies).
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

    for d in range(N_DUMMIES):
        idx = 1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+d; Q[idx, idx] = 5e-3
    return Q


# ── Transformation helper ────────────────────────────────────────────────────

def _transform_media(x: np.ndarray, transform_type: str,
                     n: float, S: float) -> np.ndarray:
    """Apply configured transformation to a media array."""
    return apply_transformation(x, transform_type, n, S)


# ── Per-equation precompute (shared by single & joint filters) ──────────────

def build_static_cache(df, g):
    """
    Precomputes the pieces of one equation's state-space setup that depend
    ONLY on (df, g) — never on the parameters being optimized — so they can
    be built ONCE per optimization run and reused across every candidate
    evaluation, instead of being rebuilt from scratch on every single one
    (which is what happened before this existed: `_prepare_equation` ran
    `_build_observation_matrix` fresh every call, even though its result
    is 100% identical across the whole run for a fixed df/g).

    Pass the result as `static_cache=` to `run_kalman_filter` /
    `run_bivariate_kalman_filter` inside an optimization loop. Safe to
    omit anywhere — every consumer falls back to building things fresh.
    """
    return {"L_mat": _build_observation_matrix(df, g, None)}


def _prepare_equation(df, g, params, static_cache=None):
    """
    Builds every array needed to run one equation's (one dependent
    variable's) state-space recursion: observation/transition/process-noise
    matrices, adstocked media, transformed shock terms, intercept boosts,
    Weibull lag sums, and the positivity/negativity index sets.

    This is called once per equation. For a single-dependent model it is
    called once; for the joint bivariate model it is called twice (once
    per dependent variable) and the two results are combined into the
    joint state-space system by run_bivariate_kalman_filter.

    `static_cache` (optional): output of `build_static_cache(df, g)`. The
    observation matrix (`L_mat`) depends only on `df`/`g` — never on the
    parameters being optimized — so during optimization it's identical on
    every single candidate evaluation. Previously it was rebuilt from
    scratch every call; passing a static_cache lets it be built once per
    optimization run instead. Safe to omit — falls back to building it
    fresh, exactly as before.
    """
    TARGET_COL = g["TARGET_COL"]; MEDIA_COLS = g["MEDIA_COLS"]
    COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]; OWN_NONMEDIA_COLS = g["OWN_NONMEDIA_COLS"]
    COMP_NONMEDIA_COLS = g["COMP_NONMEDIA_COLS"]; PRICE_COLS = g["PRICE_COLS"]
    CROSS_MEDIA_PAIRS = g["CROSS_MEDIA_PAIRS"]
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]
    N_PRICE = g["N_PRICE"]; N_DUMMIES = g["N_DUMMIES"]
    N_CROSS = g["N_CROSS"]; SEASONAL_DIM = g["SEASONAL_DIM"]
    ADSTOCK_MAP = g.get("ADSTOCK_MAP", {})
    TRANSFORM_TYPE = g["TRANSFORM_TYPE"]

    T_len = len(df)
    base_dim = _base_state_dim(g)
    dim = base_dim + _intercept_extra_lags(g)

    adstocked_media = _precompute_adstocked(df, g, params)
    if static_cache is not None and static_cache.get("L_mat") is not None:
        L_mat = static_cache["L_mat"]
    else:
        L_mat = _build_observation_matrix(df, g, adstocked_media)
    Tmat  = _build_transition_matrix(g, params)
    Q     = _build_process_noise(df, g)

    # Per-timestep raw column access, precomputed once here instead of via
    # repeated df[col].iloc[t] / list.index() calls inside the T-step loop
    # in _predict_step (pandas scalar indexing and repeated list lookups
    # are both meaningfully slower than a numpy array read / dict lookup
    # when done T times per filter run, on every optimizer candidate).
    own_nonmedia_vals  = {c: df[c].values.astype(float) for c in OWN_NONMEDIA_COLS}
    comp_nonmedia_vals = {c: df[c].values.astype(float) for c in COMP_NONMEDIA_COLS}
    price_vals         = {c: df[c].values.astype(float) for c in PRICE_COLS}
    media_idx          = {c: i for i, c in enumerate(MEDIA_COLS)}

    # Baseline floor: the intercept can never be reported below this
    # fraction of the target's average value — a market-mix baseline
    # representing organic demand shouldn't be negative or near-zero.
    # Set MIN_BASE_FRACTION to 0 (config "min_base_fraction") to disable.
    min_base_fraction = float(g.get("MIN_BASE_FRACTION", 0.0))
    intercept_floor = (
        min_base_fraction * float(df[TARGET_COL].mean())
        if min_base_fraction > 0 else -np.inf
    )

    # Applied to RAW spend (not adstocked) — this is the δ_i · f(x_i,t) term
    # in the state transition equation.
    transformed_own = np.stack([
        _transform_media(
            df[c].values.astype(float),
            TRANSFORM_TYPE,
            params["n_params"][i],
            params["S_params"][i],
        )
        for i, c in enumerate(MEDIA_COLS)
    ], axis=1) if MEDIA_COLS else np.zeros((T_len, 0))

    # Competitor media: Hill on RAW spend in instant mode (Ls_comp already
    # supplies the carryover — Hill-ing an adstocked series on top would
    # double count it, same as own media). Weibull mode keeps Hill-on-
    # adstocked, since there the weighted-lag series IS the delay model.
    transformed_comp = np.stack([
        hill_transform_vec(
            (adstocked_media[c] if ADSTOCK_MAP.get(c) == "weibull" else df[c].values.astype(float)),
            params["n_comp"][j], params["S_comp"][j])
        for j, c in enumerate(COMP_MEDIA_COLS)
    ], axis=1) if COMP_MEDIA_COLS else np.zeros((T_len, 0))

    # Cross-media synergy: Hill on RAW source for instant-adstock sources,
    # for the same reason — the synergy contribution folds into the target
    # channel's own beta, which already persists forward via its own Ls;
    # adstocking the source on top of that would carry its effect over
    # twice. Decided per the SOURCE channel's own adstock choice.
    transformed_cross = np.zeros((T_len, N_CROSS))
    for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
        src_series = adstocked_media[src] if ADSTOCK_MAP.get(src) == "weibull" else df[src].values.astype(float)
        transformed_cross[:, k] = hill_transform_vec(
            src_series, params["cross_n"][k], params["cross_S"][k])

    # Intercept boosts — Power or Hill per INTERCEPT_TRANSFORM_TYPE (set
    # independently of the media betas' own TRANSFORM_TYPE above):
    #   Power: I_t += Σ γ_k · media_k,t^{n_k_intercept}
    #   Hill:  I_t += Σ γ_k · media_k,t^{n_k_intercept} /
    #                     (media_k,t^{n_k_intercept} + S_k_intercept^{n_k_intercept})
    # Applies uniformly to every intercept effector, whether or not it is
    # also a media channel — each gets its own independently-fitted
    # n_intercept (and S_intercept, Hill only), never the media beta's.
    INTERCEPT_TRANSFORM_TYPE = g.get("INTERCEPT_TRANSFORM_TYPE", "power")
    intercept_boost = np.zeros(T_len)
    for k, col in enumerate(g["INTERCEPT_EFFECTORS"]):
        if col not in df.columns:
            continue
        ni_int = params["n_intercept"][k]
        si_int = params["S_intercept"][k]
        raw = df[col].values.astype(float)
        intercept_boost += params["gamma"][k] * _transform_media(
            raw, INTERCEPT_TRANSFORM_TYPE, ni_int, si_int)

    # Weibull weighted-lag arrays for state transition — per channel now.
    # A channel on "instant" contributes an all-zero column here (harmless
    # to add in _predict_step, since its Tmat diagonal already carries its
    # Ls persistence instead). Built for every group that can use weibull
    # adstock: own media, comp media, own non-media, comp non-media.
    def _lagsum_block(cols, n):
        if not cols:
            return np.zeros((T_len, n))
        return np.stack([
            adstocked_media[c] if ADSTOCK_MAP.get(c) == "weibull" else np.zeros(T_len)
            for c in cols
        ], axis=1)

    weibull_lagsum_own          = _lagsum_block(MEDIA_COLS, N_MEDIA)
    weibull_lagsum_comp         = _lagsum_block(COMP_MEDIA_COLS, N_COMP)
    weibull_lagsum_own_nonmedia = _lagsum_block(OWN_NONMEDIA_COLS, N_OWN_NONMEDIA)
    weibull_lagsum_comp_nonmedia = _lagsum_block(COMP_NONMEDIA_COLS, N_COMP_NONMEDIA)

    positive_cols = set(g.get("POSITIVE_BETA_COLS", []))
    negative_cols = set(g.get("NEGATIVE_BETA_COLS", []))
    positive_state_idx = (
        [i + 1 for i, col in enumerate(MEDIA_COLS) if col in positive_cols] +
        [1 + N_MEDIA + N_COMP + k for k, col in enumerate(OWN_NONMEDIA_COLS)
         if col in positive_cols]
    )
    negative_state_idx = (
        [i + 1 for i, col in enumerate(MEDIA_COLS) if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + k for k, col in enumerate(OWN_NONMEDIA_COLS)
         if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + k
         for k, col in enumerate(COMP_NONMEDIA_COLS) if col in negative_cols] +
        [1 + N_MEDIA + N_COMP + N_OWN_NONMEDIA + N_COMP_NONMEDIA + p
         for p, col in enumerate(PRICE_COLS) if col in negative_cols]
    )

    return dict(
        g=g, params=params, dim=dim, base_dim=base_dim, T_len=T_len,
        TARGET_COL=TARGET_COL, MEDIA_COLS=MEDIA_COLS, COMP_MEDIA_COLS=COMP_MEDIA_COLS,
        OWN_NONMEDIA_COLS=OWN_NONMEDIA_COLS, COMP_NONMEDIA_COLS=COMP_NONMEDIA_COLS,
        PRICE_COLS=PRICE_COLS, CROSS_MEDIA_PAIRS=CROSS_MEDIA_PAIRS,
        N_MEDIA=N_MEDIA, N_COMP=N_COMP, N_OWN_NONMEDIA=N_OWN_NONMEDIA,
        N_COMP_NONMEDIA=N_COMP_NONMEDIA, N_PRICE=N_PRICE, N_CROSS=N_CROSS,
        ADSTOCK_MAP=ADSTOCK_MAP, USE_ORGANIC_DRIFT=g["USE_ORGANIC_DRIFT"],
        adstocked_media=adstocked_media, L_mat=L_mat, Tmat=Tmat, Q=Q,
        transformed_own=transformed_own, transformed_comp=transformed_comp,
        transformed_cross=transformed_cross, intercept_boost=intercept_boost,
        weibull_lagsum_own=weibull_lagsum_own,
        weibull_lagsum_comp=weibull_lagsum_comp,
        weibull_lagsum_own_nonmedia=weibull_lagsum_own_nonmedia,
        weibull_lagsum_comp_nonmedia=weibull_lagsum_comp_nonmedia,
        positive_cols=positive_cols, negative_cols=negative_cols,
        positive_state_idx=positive_state_idx, negative_state_idx=negative_state_idx,
        intercept_floor=intercept_floor,
        own_nonmedia_vals=own_nonmedia_vals, comp_nonmedia_vals=comp_nonmedia_vals,
        price_vals=price_vals, media_idx=media_idx,
        target_vals=df[TARGET_COL].values,
    )


def _initial_state(df, g, params, pc):
    """Builds x0 / P0 for one equation, identical to the previous
    single-equation initialisation logic."""
    dim = pc["dim"]; TARGET_COL = g["TARGET_COL"]
    MEDIA_COLS = g["MEDIA_COLS"]; COMP_MEDIA_COLS = g["COMP_MEDIA_COLS"]
    OWN_NONMEDIA_COLS = g["OWN_NONMEDIA_COLS"]; COMP_NONMEDIA_COLS = g["COMP_NONMEDIA_COLS"]
    PRICE_COLS = g["PRICE_COLS"]
    N_MEDIA = g["N_MEDIA"]; N_COMP = g["N_COMP"]
    N_OWN_NONMEDIA = g["N_OWN_NONMEDIA"]; N_COMP_NONMEDIA = g["N_COMP_NONMEDIA"]

    x0 = np.zeros(dim)
    x0[0] = df[TARGET_COL].mean() * 0.8
    # Intercept Weibull shadow-lag states (if any): seed with a flat
    # history assumption — same starting level as I_t itself.
    base_dim = pc.get("base_dim", dim)
    for lag_idx in range(base_dim, dim):
        x0[lag_idx] = x0[0]
    for i, col in enumerate(MEDIA_COLS):
        x0[i+1] = g["INITIAL_MEDIA_BETAS"].get(col, 0.0)
    for j, col in enumerate(COMP_MEDIA_COLS):
        x0[1+N_MEDIA+j] = g["INITIAL_COMP_BETAS"].get(col, -0.0001)
    for k, col in enumerate(OWN_NONMEDIA_COLS):
        x0[1+N_MEDIA+N_COMP+k] = g["INITIAL_OWN_NONMEDIA_BETAS"].get(col, 0.0)
    for k, col in enumerate(COMP_NONMEDIA_COLS):
        x0[1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+k] = g["INITIAL_COMP_NONMEDIA_BETAS"].get(col, -0.01)
    for p, col in enumerate(PRICE_COLS):
        x0[1+N_MEDIA+N_COMP+N_OWN_NONMEDIA+N_COMP_NONMEDIA+p] = g["INITIAL_PRICE_BETA"].get(col, -0.1)

    target_var = float(np.var(df[TARGET_COL].values))
    base_var   = max(target_var, 1e-6)
    P0 = np.eye(dim) * base_var
    P0[0, 0] = base_var * 4.0
    return x0, P0


def _predict_step(t, x_prev, df, pc):
    """
    One equation's nonlinear state-prediction step (the x_p = f(x_{t-1}, x_t)
    part of the RBE), identical to the per-timestep logic previously inlined
    in run_kalman_filter's forward loop. Used by both the single-equation
    filter and (twice per t, once per dependent variable) by the joint
    bivariate filter.

    Returns (x_p, cross_contrib_row) where cross_contrib_row has length
    N_CROSS (synergy contribution booked against the target channel at t).
    """
    params = pc["params"]
    MEDIA_COLS = pc["MEDIA_COLS"]; OWN_NONMEDIA_COLS = pc["OWN_NONMEDIA_COLS"]
    COMP_NONMEDIA_COLS = pc["COMP_NONMEDIA_COLS"]; PRICE_COLS = pc["PRICE_COLS"]
    CROSS_MEDIA_PAIRS = pc["CROSS_MEDIA_PAIRS"]
    N_MEDIA = pc["N_MEDIA"]; N_COMP = pc["N_COMP"]
    positive_cols = pc["positive_cols"]; negative_cols = pc["negative_cols"]

    x_p = pc["Tmat"] @ x_prev
    cross_contrib_row = np.zeros(pc["N_CROSS"])

    # ── Own-media state equations ────────────────────────────
    # Per channel now: weibull_lagsum_own[t, i] is 0 for any channel on
    # instant adstock (its carryover already lives in Tmat's diagonal via
    # Ls, applied above by the matrix multiply), so this add is safe to do
    # unconditionally for every channel regardless of its individual choice.
    for i in range(N_MEDIA):
        x_p[i+1] += pc["weibull_lagsum_own"][t, i] + params["delta"][i] * pc["transformed_own"][t, i]
        if MEDIA_COLS[i] in positive_cols:
            x_p[i+1] = max(x_p[i+1], 1e-8)
        elif MEDIA_COLS[i] in negative_cols:
            x_p[i+1] = min(x_p[i+1], -1e-8)

    # ── Competitor media ─────────────────────────────────────
    for j in range(N_COMP):
        idx = 1+N_MEDIA+j
        x_p[idx] += pc["weibull_lagsum_comp"][t, j] + params["delta_comp"][j] * pc["transformed_comp"][t, j]
        x_p[idx]  = min(x_p[idx], -1e-8)

    # ── Cross-media synergy ───────────────────────────────────
    for k, (tgt, src) in enumerate(CROSS_MEDIA_PAIRS):
        tgt_idx = pc["media_idx"][tgt]
        contrib = params["cross_delta"][k] * pc["transformed_cross"][t, k]
        x_p[tgt_idx+1] += contrib
        cross_contrib_row[k] = contrib

    # ── Own non-media ─────────────────────────────────────────
    # Same per-channel pattern as own media: weibull_lagsum_own_nonmedia
    # is 0 for any channel left on instant adstock.
    for k, col in enumerate(OWN_NONMEDIA_COLS):
        si = 1+N_MEDIA+N_COMP+k
        x_p[si] += pc["weibull_lagsum_own_nonmedia"][t, k] + params["delta_own_nonmedia"][k] * pc["own_nonmedia_vals"][col][t]
        if col in positive_cols:
            x_p[si] = max(x_p[si], 1e-8)

    # ── Competitor non-media ──────────────────────────────────
    for k, col in enumerate(COMP_NONMEDIA_COLS):
        si = 1+N_MEDIA+N_COMP+pc["N_OWN_NONMEDIA"]+k
        x_p[si] += pc["weibull_lagsum_comp_nonmedia"][t, k] + params["delta_comp_nonmedia"][k] * pc["comp_nonmedia_vals"][col][t]
        x_p[si]  = min(x_p[si], -1e-8)

    # ── Price ─────────────────────────────────────────────────
    for p, col in enumerate(PRICE_COLS):
        si = 1+N_MEDIA+N_COMP+pc["N_OWN_NONMEDIA"]+pc["N_COMP_NONMEDIA"]+p
        x_p[si] += params["delta_price"][p] * pc["price_vals"][col][t]
        x_p[si]  = min(x_p[si], -1e-8)

    # ── Intercept boost ───────────────────────────────────────
    # I0 is 0.0 in "carryover" mode (dynamics fully covered by Tmat[0,0] =
    # G0 already applied via the Tmat @ x_prev multiply above) and the
    # fitted baseline constant in "simple" mode (where Tmat[0,0] = 0, so
    # this add is literally the whole of I_t besides the effector boost).
    x_p[0] += pc["intercept_boost"][t] + params.get("I0", 0.0)
    if pc["USE_ORGANIC_DRIFT"]:
        x_p[0] += params["mu"]

    return x_p, cross_contrib_row


def _apply_beta_floors(x_vec, pc):
    floor = pc.get("intercept_floor", -np.inf)
    if x_vec[0] < floor:
        x_vec[0] = floor
    for idx in pc["positive_state_idx"]:
        if x_vec[idx] < 0:
            x_vec[idx] = 1e-8
    for idx in pc["negative_state_idx"]:
        if x_vec[idx] > 0:
            x_vec[idx] = -1e-8
    return x_vec


# ── Main (single-equation) Kalman filter ─────────────────────────────────────

def run_kalman_filter(df, params, g, static_cache=None):
    """Single-dependent-variable RBE (used when no second dependent
    variable is configured, and internally re-uses the same building
    blocks as the joint bivariate filter below).

    `static_cache`: optional, from `build_static_cache(df, g)` — reuses
    the observation matrix across repeated calls instead of rebuilding it
    every time (see build_static_cache's docstring). Safe to omit."""
    pc = _prepare_equation(df, g, params, static_cache=static_cache)
    dim = pc["dim"]; T_len = pc["T_len"]
    L_mat = pc["L_mat"]; Tmat = pc["Tmat"]; Q = pc["Q"]
    R = params["sigma_y"] ** 2

    x_filt = np.zeros((T_len, dim)); P_filt = np.zeros((T_len, dim, dim))
    x_pred = np.zeros((T_len, dim)); P_pred = np.zeros((T_len, dim, dim))
    cross_beta_contrib = np.zeros((T_len, pc["N_CROSS"]))
    yhat = np.zeros(T_len); residuals = np.zeros(T_len); loglik = 0.0

    x0, P0 = _initial_state(df, g, params, pc)
    x_filt[0] = x0; P_filt[0] = P0
    target_vals = pc["target_vals"]

    for t in range(T_len):
        if t > 0:
            x_p, cross_row = _predict_step(t, x_filt[t-1], df, pc)
            cross_beta_contrib[t] = cross_row
            x_pred[t] = x_p
            P_pred[t] = Tmat @ P_filt[t-1] @ Tmat.T + Q
        else:
            x_pred[0] = x_filt[0].copy(); P_pred[0] = P_filt[0].copy()

        L_t = L_mat[t]
        yhat[t]      = L_t @ x_pred[t]
        residuals[t] = target_vals[t] - yhat[t]
        LP  = L_t @ P_pred[t]
        S_t = max(LP @ L_t + R, 1e-8)
        K   = (P_pred[t] @ L_t) / S_t
        x_filt[t] = x_pred[t] + K * residuals[t]
        x_filt[t] = _apply_beta_floors(x_filt[t], pc)
        P_filt[t] = (np.eye(dim) - np.outer(K, L_t)) @ P_pred[t]
        loglik += -0.5 * (np.log(2 * np.pi * S_t) + residuals[t]**2 / S_t)

    return yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat, cross_beta_contrib, loglik


# ── Joint (bivariate) Kalman filter ──────────────────────────────────────────

def run_bivariate_kalman_filter(df, params1, g1, params2, g2, rho, phi1=0.0, phi2=0.0,
                                 static_cache1=None, static_cache2=None):
    """
    Joint (bivariate) Kalman-filter fit of:

        [ sales_t         ]   [ Intercept_sales_t         ]   [ error_sales_t         ]
        [ consideration_t ] = [ Intercept_consideration_t ] + [ error_consideration_t ]

                                [ beta_sales_1_t         ... beta_sales_M_t         ]
                              + [ beta_consid_1_t        ... beta_consid_M_t       ] · x_t

    Both dependent variables are fitted SIMULTANEOUSLY in one state-space
    system: a single (dim_1+dim_2)-length state vector is propagated and
    updated at every t, with a full 2×2 observation-noise covariance matrix
    R (parameterised by σ_1, σ_2 and a jointly-estimated correlation ρ)
    coupling the two equations' contemporaneous errors, AND each equation's
    intercept feeds into the other's next-period intercept via phi_1 / phi_2
    (see the module docstring's "Cross-intercept coupling" section):

        Intercept_1,t = G0_1 · Intercept_1,t-1 + phi_1 · Intercept_2,t-1 + effectors_1,t
        Intercept_2,t = G0_2 · Intercept_2,t-1 + phi_2 · Intercept_1,t-1 + effectors_2,t

    Parameters
    ----------
    phi1 : coefficient of Intercept_2,t-1 in equation 1's intercept update.
    phi2 : coefficient of Intercept_1,t-1 in equation 2's intercept update.

    Returns
    -------
    yhat, residuals : (T, 2) arrays — column 0 = dep 1, column 1 = dep 2
    x_filt, P_filt, x_pred, P_pred : joint filtered/predicted state & covariance
    Tmat_joint : joint transition matrix (block-diagonal plus the two
                 phi_1/phi_2 cross-intercept off-diagonal entries)
    cross1, cross2 : (T, N_CROSS) per-equation synergy contribution arrays
    loglik  : joint bivariate log-likelihood (single scalar, NOT split per equation)
    dim1, dim2 : state dimensions of equation 1 / equation 2 (for splitting x_smooth later)

    `static_cache1`/`static_cache2`: optional, from `build_static_cache(df, g1)`
    / `build_static_cache(df, g2)` — reuses each equation's observation
    matrix across repeated calls instead of rebuilding it every time.
    Safe to omit.
    """
    pc1 = _prepare_equation(df, g1, params1, static_cache=static_cache1)
    pc2 = _prepare_equation(df, g2, params2, static_cache=static_cache2)
    dim1, dim2 = pc1["dim"], pc2["dim"]
    dim = dim1 + dim2
    T_len = pc1["T_len"]

    Tmat_joint = np.zeros((dim, dim))
    Tmat_joint[:dim1, :dim1] = pc1["Tmat"]
    Tmat_joint[dim1:, dim1:] = pc2["Tmat"]
    # Cross-intercept coupling: intercept is always state index 0 within
    # each equation's own block, i.e. joint indices 0 (eq 1) and dim1 (eq 2).
    Tmat_joint[0, dim1] = phi1
    Tmat_joint[dim1, 0] = phi2

    Q_joint = np.zeros((dim, dim))
    Q_joint[:dim1, :dim1] = pc1["Q"]
    Q_joint[dim1:, dim1:] = pc2["Q"]

    x0_1, P0_1 = _initial_state(df, g1, params1, pc1)
    x0_2, P0_2 = _initial_state(df, g2, params2, pc2)
    x0 = np.concatenate([x0_1, x0_2])
    P0 = np.zeros((dim, dim))
    P0[:dim1, :dim1] = P0_1
    P0[dim1:, dim1:] = P0_2

    x_filt = np.zeros((T_len, dim)); P_filt = np.zeros((T_len, dim, dim))
    x_pred = np.zeros((T_len, dim)); P_pred = np.zeros((T_len, dim, dim))
    cross1 = np.zeros((T_len, pc1["N_CROSS"])); cross2 = np.zeros((T_len, pc2["N_CROSS"]))
    yhat = np.zeros((T_len, 2)); residuals = np.zeros((T_len, 2)); loglik = 0.0

    x_filt[0] = x0; P_filt[0] = P0
    targets = np.column_stack([pc1["target_vals"], pc2["target_vals"]])

    sigma1 = float(params1["sigma_y"]); sigma2 = float(params2["sigma_y"])
    rho = float(np.clip(rho, -0.995, 0.995))
    R = np.array([
        [sigma1 ** 2,            rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2,  sigma2 ** 2],
    ])

    for t in range(T_len):
        if t > 0:
            x_p1, cross_row1 = _predict_step(t, x_filt[t-1, :dim1], df, pc1)
            x_p2, cross_row2 = _predict_step(t, x_filt[t-1, dim1:], df, pc2)
            # Cross-intercept coupling: each equation's intercept also picks
            # up phi_i · (other equation's PREVIOUS intercept).
            x_p1[0] += phi1 * x_filt[t-1, dim1]
            x_p2[0] += phi2 * x_filt[t-1, 0]
            cross1[t] = cross_row1; cross2[t] = cross_row2
            x_pred[t] = np.concatenate([x_p1, x_p2])
            P_pred[t] = Tmat_joint @ P_filt[t-1] @ Tmat_joint.T + Q_joint
        else:
            x_pred[0] = x_filt[0].copy(); P_pred[0] = P_filt[0].copy()

        # ── Joint observation matrix (2 × dim) ────────────────────────
        L_t = np.zeros((2, dim))
        L_t[0, :dim1] = pc1["L_mat"][t]
        L_t[1, dim1:] = pc2["L_mat"][t]

        yhat[t] = L_t @ x_pred[t]
        residuals[t] = targets[t] - yhat[t]

        PLt = P_pred[t] @ L_t.T                    # (dim, 2)
        S_t = L_t @ PLt + R                         # (2, 2) innovation covariance
        S_t = S_t + np.eye(2) * 1e-10               # numerical safety
        S_inv = np.linalg.inv(S_t)
        K = PLt @ S_inv                             # (dim, 2) joint Kalman gain

        x_filt[t] = x_pred[t] + K @ residuals[t]
        x_filt[t, :dim1] = _apply_beta_floors(x_filt[t, :dim1].copy(), pc1)
        x_filt[t, dim1:] = _apply_beta_floors(x_filt[t, dim1:].copy(), pc2)

        P_filt[t] = (np.eye(dim) - K @ L_t) @ P_pred[t]

        sign, logdet = np.linalg.slogdet(S_t)
        if sign <= 0:
            logdet = np.log(1e-12)
        quad = residuals[t] @ S_inv @ residuals[t]
        loglik += -0.5 * (2 * np.log(2 * np.pi) + logdet + quad)

    return (yhat, residuals, x_filt, P_filt, x_pred, P_pred, Tmat_joint,
            cross1, cross2, loglik, dim1, dim2)


# ── Joint composite loss: EKF NLL + λ·NRMSE regularization ──────────────────
#
#   Loss(Θ) = (1/2)·Σ_t (log|F_t| + v_t' F_t^-1 v_t)
#             + λ·( RMSE_sales/ȳ_sales + RMSE_visits/ȳ_visits )
#
# The first term is exactly `-loglik` from run_bivariate_kalman_filter
# above (that function's `loglik` already accumulates
# -0.5*(2*log(2*pi) + logdet + quad) per step — the extra 2*log(2*pi)
# constant per step doesn't depend on Θ, so it doesn't change where the
# optimum sits, it only offsets the reported log-lik value by a constant).
# The second term penalizes scale-normalized RMSE on BOTH dependent
# variables jointly, so the optimizer can't trade one equation's fit for
# the other's.

def compute_joint_nrmse(residuals, df, g1, g2):
    """NRMSE regularization term: RMSE_1/ȳ_1 + RMSE_2/ȳ_2.

    `residuals` is the (T, 2) array returned by run_bivariate_kalman_filter
    (column 0 = dependent 1, column 1 = dependent 2). ȳ_i is that
    dependent's own observed mean over `df` (guarded away from 0 for
    near-zero-mean series).
    """
    rmse1 = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
    rmse2 = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
    ybar1 = float(df[g1["TARGET_COL"]].mean())
    ybar2 = float(df[g2["TARGET_COL"]].mean())
    ybar1 = ybar1 if abs(ybar1) > 1e-8 else 1e-8
    ybar2 = ybar2 if abs(ybar2) > 1e-8 else 1e-8
    nrmse = rmse1 / abs(ybar1) + rmse2 / abs(ybar2)
    return nrmse, rmse1, rmse2


def joint_composite_loss(df, p1, g1, p2, g2, rho, phi1, phi2, lambda_reg,
                          static_cache1=None, static_cache2=None):
    """EKF NLL + λ·NRMSE, evaluated on `df` at the given joint parameters.

    Returns (loss, nll, nrmse, rmse1, rmse2) — the pieces are all handed
    back too so callers can log/inspect them without recomputing.
    """
    (_, residuals, _, _, _, _, _, _, _, loglik, _, _) = run_bivariate_kalman_filter(
        df, p1, g1, p2, g2, rho, phi1, phi2,
        static_cache1=static_cache1, static_cache2=static_cache2)
    nll = -loglik
    nrmse, rmse1, rmse2 = compute_joint_nrmse(residuals, df, g1, g2)
    loss = nll + lambda_reg * nrmse
    return loss, nll, nrmse, rmse1, rmse2


def rts_smoother(x_filt, P_filt, x_pred, P_pred, Tmat):
    """Dimension-agnostic RTS smoother — used as-is for both the
    single-equation state and the joint (dim_1+dim_2) bivariate state."""
    T_len, dim = x_filt.shape
    x_smooth = x_filt.copy(); P_smooth = P_filt.copy()
    for t in range(T_len - 2, -1, -1):
        C = P_filt[t] @ Tmat.T @ np.linalg.inv(P_pred[t+1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t+1] - P_pred[t+1]) @ C.T
    return x_smooth, P_smooth
