"""
Cochrane-Orcutt (CO) correction for AR(1) observation errors — OPTIONAL.

Off by default. Switched on in Tab 4 (config key "use_cochrane_orcutt");
when it is off nothing in this module is touched and every fit is
bit-for-bit what it was before.

──────────────────────────────────────────────────────────────────────────
The model
──────────────────────────────────────────────────────────────────────────
The state-space observation equation is

    y_t = L_t · x_t + ε_t

The plain filter assumes ε_t is white noise, ε_t ~ N(0, σ²). CO instead
assumes first-order serially correlated errors

    ε_t = ρ · ε_{t-1} + u_t ,    u_t ~ N(0, σ²) white,   |ρ| < 1

and quasi-differences the observation equation:

    y*_t = y_t − ρ·y_{t-1}
    L*_t = L_t − ρ·L_{t-1}
    ⇒  y*_t = L*_t · x_t + u_t              (t = 2 … T)

The Kalman filter then runs on (y*_t, L*_t) with white noise u_t, so its
innovations, gain and log-likelihood are all computed on the whitened
equation. The state equations (betas, intercept) are untouched.

First observation (config "co_keep_first_obs", default True): classical CO
drops it (no update, no likelihood term). By default it is instead kept
with the Prais-Winsten rescale y*_1 = sqrt(1-ρ²)·y_1, L*_1 = sqrt(1-ρ²)·L_1
(variance σ²/(1-ρ²) → σ²), because in a state-space model the first
observation is what anchors the INITIAL STATE; dropping it leaves the
first couple of fitted values noticeably off.

Everything reported to the user — fitted values, contributions, ROI,
MAPE, R² — still uses the ORIGINAL (non-differenced) L_t · x_t. So the
betas keep their structural meaning, and the residuals you see are the
structural ε_t (which ARE serially correlated, by assumption); it is
u_t = ε_t − ρ·ε_{t-1} that is whitened.

──────────────────────────────────────────────────────────────────────────
The iteration (classical CO)
──────────────────────────────────────────────────────────────────────────
    ρ ← 0
    repeat:
        1. fit all parameters (the normal optimizer run) with the current ρ
        2. take the structural one-step-ahead residuals e_t = y_t − L_t·x_{t|t-1}
           on the TRAIN window
        3. ρ̂ = Σ e_t·e_{t-1} / Σ e_{t-1}²          (OLS of e_t on e_{t-1},
                                                      no intercept)
        4. stop if |ρ̂ − ρ| < tol, else ρ ← ρ̂ and go to 1
Iteration 1 therefore IS the ordinary (ρ = 0) fit; it is kept so the
"before" Durbin-Watson / ρ̂ can be reported next to the "after".
Each later iteration warm-starts from the previous theta.

With two dependents (joint or chained) each equation gets its OWN ρ.
In the joint fit this is separate from the existing cross-equation error
correlation (rho_y): ρ_AR is serial correlation WITHIN an equation, rho_y
is correlation BETWEEN the two equations' (whitened) shocks.

Where ρ lives: g["CO_RHO"] (set by modules/params.py, default 0.0). The
filters in modules/kalman.py read it, so the optimizers, static caches
and loss functions needed no signature changes.
"""

import numpy as np

RHO_CLIP = 0.99


def estimate_ar1_rho(resid, clip=RHO_CLIP):
    """CO step 3: OLS slope of e_t on e_{t-1} through the origin, clipped
    to (−clip, clip) so the quasi-difference stays well-posed."""
    e = np.asarray(resid, dtype=float)
    e = e[np.isfinite(e)] if not np.all(np.isfinite(e)) else e
    if len(e) < 3:
        return 0.0
    num = float(np.sum(e[1:] * e[:-1]))
    den = float(np.sum(e[:-1] ** 2))
    if den < 1e-12:
        return 0.0
    return float(np.clip(num / den, -clip, clip))


def durbin_watson(resid):
    """Durbin-Watson statistic: ≈2 means no first-order autocorrelation,
    <2 positive, >2 negative (DW ≈ 2·(1 − ρ̂))."""
    e = np.asarray(resid, dtype=float)
    if len(e) < 3:
        return float("nan")
    den = float(np.sum(e ** 2))
    if den < 1e-12:
        return float("nan")
    return float(np.sum(np.diff(e) ** 2) / den)


def quasi_difference_obs(y, L_mat, rho, keep_first=True):
    """Returns (y*, L*) of the same length as the inputs.

    Rows t >= 1:  y*_t = y_t - rho*y_{t-1},  L*_t = L_t - rho*L_{t-1}.

    Row 0, two choices (config "co_keep_first_obs", default True):
      keep_first=True  — Prais-Winsten: y*_0 = sqrt(1-rho^2)*y_0,
                         L*_0 = sqrt(1-rho^2)*L_0. Since Var(eps_0) =
                         sigma^2/(1-rho^2), this rescales it to variance
                         sigma^2, i.e. the same white-noise scale as every
                         other row. Keeps the first observation, which in a
                         state-space model is what anchors the INITIAL
                         STATE (intercept level / starting betas).
      keep_first=False — classical Cochrane-Orcutt: row 0 is left
                         un-differenced as a placeholder and the caller
                         must skip the update at t = 0 (see
                         modules/kalman.py). Cheaper to explain, but with
                         no data at t = 0 the initial state is only pinned
                         down indirectly, so the first couple of fitted
                         values can be noticeably off.
    """
    y = np.asarray(y, dtype=float); L_mat = np.asarray(L_mat, dtype=float)
    y_star = y.copy(); L_star = L_mat.copy()
    if abs(rho) > 0.0:
        y_star[1:] = y[1:] - rho * y[:-1]
        L_star[1:] = L_mat[1:] - rho * L_mat[:-1]
        if keep_first:
            w = float(np.sqrt(max(1.0 - rho * rho, 1e-12)))
            y_star[0] = w * y[0]
            L_star[0] = w * L_mat[0]
    return y_star, L_star


def ar1_diagnostics(resid, rho_used):
    """DW of the structural residuals e_t and of the whitened
    u_t = e_t − ρ_used·e_{t-1} (the latter should be ≈ 2 if CO worked)."""
    e = np.asarray(resid, dtype=float)
    u = e[1:] - rho_used * e[:-1] if len(e) > 1 else e
    return {"dw_structural": durbin_watson(e), "dw_whitened": durbin_watson(u)}


def run_cochrane_orcutt(fit_fn, resid_fn, n_eq, max_iter=5, tol=0.01,
                         progress_cb=None):
    """
    Generic CO driver, shared by the single / joint / refit pipelines.

    fit_fn(rhos, theta_init) -> theta
        Runs the normal optimizer with each equation's ρ already set to
        `rhos[i]` (the caller writes it into g["CO_RHO"]). `theta_init` is
        None on the first call (use the default start) and the previous
        iteration's best theta afterwards (warm start). Whatever `theta`
        is (flat array for one equation, joint vector for two), it is
        passed straight back into resid_fn and returned.
    resid_fn(theta, rhos) -> list of n_eq 1-D arrays
        Structural one-step-ahead train residuals per equation, from the
        filter run at that theta and those ρ.

    Returns (theta, rhos, info). `rhos` is the ρ the FINAL theta was
    fitted with (so theta and ρ are always self-consistent, even if the
    loop stopped on max_iter rather than convergence). info = {
      "converged", "n_iter", "rho" (list), "rho_hat_last" (list),
      "tol", "max_iter", "history": [ {iteration, eq, rho_used, rho_hat,
      dw_structural, dw_whitened}, … ]  (one row per iteration per equation)
    }
    """
    rhos = [0.0] * n_eq
    theta = None
    history = []
    converged = False
    rho_hat_last = [0.0] * n_eq

    for k in range(1, int(max_iter) + 1):
        theta = fit_fn(list(rhos), theta)
        resids = resid_fn(theta, list(rhos))
        rho_hat = [estimate_ar1_rho(e) for e in resids]
        for i in range(n_eq):
            d = ar1_diagnostics(resids[i], rhos[i])
            history.append({
                "iteration": k, "eq": i + 1,
                "rho_used": float(rhos[i]), "rho_hat": float(rho_hat[i]),
                **d,
            })
        rho_hat_last = rho_hat
        if progress_cb:
            progress_cb(k, int(max_iter), rho_hat)
        delta = max(abs(rho_hat[i] - rhos[i]) for i in range(n_eq))
        if delta < tol:
            converged = True
            break
        if k < int(max_iter):
            rhos = rho_hat      # next iteration fits with the new ρ̂
        # else: keep rhos as the ρ the last theta was fitted with

    info = {
        "converged": converged, "n_iter": k,
        "rho": [float(r) for r in rhos],
        "rho_hat_last": [float(r) for r in rho_hat_last],
        "tol": float(tol), "max_iter": int(max_iter),
        "history": history,
    }
    return theta, rhos, info
