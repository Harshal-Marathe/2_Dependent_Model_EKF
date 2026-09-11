"""
Nevergrad optimizer support.

Loss = -loglik   (identical objective to the L-BFGS-B / SLSQP path)
"""

import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

from modules.params import unpack_theta
from modules.bounds import build_normalized_problem
from modules.kalman import (
    run_kalman_filter, run_bivariate_kalman_filter, build_static_cache,
    joint_composite_loss,
)


def _composite_loss(theta, df_train, g, static_cache=None):
    try:
        p = unpack_theta(theta, g)
        _, _, _, _, _, _, _, _, loglik = run_kalman_filter(
            df_train, p, g, static_cache=static_cache)
        return -loglik
    except Exception:
        return 1e12


def _ask_eval_tell_loop(optimizer, budget, num_workers, progress_label, loss_fn):
    """
    Shared ask/evaluate/tell driver for both the single-equation and joint
    Nevergrad optimizers.

    Previously this always asked for and evaluated ONE candidate at a time,
    regardless of the `num_workers` setting in the UI — so raising "workers"
    had zero effect on wall-clock time. This now genuinely batches
    `num_workers` candidates per round and evaluates them concurrently via a
    thread pool (nevergrad's ask-many/tell-many pattern is explicitly
    designed for this). With num_workers=1 (the default) this is exactly
    the old serial behaviour — nothing changes unless you raise it.

    Note: the Kalman filter's per-timestep loop is Python-level, not a
    single vectorized numpy call, so GIL contention limits how much thread
    parallelism can help — the numpy matrix multiplies inside each step do
    release the GIL, so there's a real, if likely partial (not N×), speedup
    from raising num_workers. It is, at minimum, no longer a no-op.
    """
    best_loss = np.inf; best_theta = None
    progress = st.progress(0, text=f"{progress_label} — 0/{budget} evals")
    executor = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    evaluated = 0
    report_every = max(1, budget // 50)
    try:
        while evaluated < budget:
            batch_n = min(num_workers, budget - evaluated)
            cands = [optimizer.ask() for _ in range(batch_n)]
            if executor is not None:
                losses = list(executor.map(lambda c: loss_fn(c.value), cands))
            else:
                losses = [loss_fn(c.value) for c in cands]
            for cand, loss in zip(cands, losses):
                optimizer.tell(cand, loss)
                if loss < best_loss:
                    best_loss = loss; best_theta = cand.value.copy()
            evaluated += batch_n
            if (evaluated // report_every) != ((evaluated - batch_n) // report_every):
                progress.progress(min(100, int(evaluated / budget * 100)),
                                  text=f"{progress_label} — {evaluated}/{budget} | best: {best_loss:.4f}")
    finally:
        if executor is not None:
            executor.shutdown(wait=False)
    progress.progress(100, text=f"✅ {progress_label} done — best loss: {best_loss:.4f}")
    return best_theta, best_loss


def _ng_bounds_arrays(norm_bounds, sentinel=100.0):
    """
    ng.p.Array.set_bounds() needs finite numeric arrays, but
    build_normalized_problem() legitimately leaves a side as None when the
    ORIGINAL parameter bound was None on that side (see its docstring —
    that's a real "no constraint here" that we don't want to silently
    reintroduce, so we don't turn it into a tight box). Since every
    dimension is already normalized to a comparable, roughly-O(1) scale by
    that point (unlike the raw ±1e6-in-real-units sentinel this replaces),
    a single generous, uniform normalized-space sentinel works fine here —
    it's proportionate for every dimension instead of swamping the ones
    that started out small in real units.
    """
    lows  = np.array([b[0] if b[0] is not None else -sentinel for b in norm_bounds])
    highs = np.array([b[1] if b[1] is not None else  sentinel for b in norm_bounds])
    return lows, highs


# ── Multi-start local optimization (L-BFGS-B / SLSQP) ───────────────────────
#
# A single scipy.optimize.minimize call from one fixed theta0 can get stuck
# exactly at (or very near) its starting point in some dimensions if the
# loglik surface is flat/insensitive there around theta0 — the optimizer
# reads "no local gradient signal" as "already at the optimum" even when a
# real, better optimum exists elsewhere in the search space. Restarting the
# SAME local optimizer from several different (randomized) starting points
# and keeping whichever restart reaches the best loglik directly tests
# whether that's actually true, instead of just trusting the one run that
# happened to start at theta0.

def _jittered_normalized_starts(theta0_norm, norm_bounds, n_starts, seed=42):
    """
    Builds `n_starts` starting points in the SAME normalized theta space
    scipy's L-BFGS-B/SLSQP already searches in (see
    modules/bounds.py::build_normalized_problem).

    Start 0 is always the exact, unperturbed theta0_norm — so multi-start
    can never do WORSE than the original single-start fit, only find
    something better (or confirm theta0 already was the optimum). Starts
    1..n_starts-1 are randomized:
      - Fully-bounded normalized dims (already scaled to exactly [0, 1])
        are drawn uniformly across the whole [0, 1] box — a genuine
        global restart for that dimension, not just a small nudge.
      - Partially/unbounded dims (no finite normalized box to sample
        uniformly from) are jittered around theta0_norm with Gaussian
        noise in normalized units, then clipped back to whatever real
        bound does exist on that side (if any).
    """
    rng = np.random.default_rng(seed)
    theta0_norm = np.asarray(theta0_norm, dtype=float)
    starts = [theta0_norm.copy()]
    n = len(theta0_norm)
    for _ in range(max(0, n_starts - 1)):
        cand = np.empty(n)
        for i, (lo, hi) in enumerate(norm_bounds):
            x0 = theta0_norm[i]
            if lo is not None and hi is not None:
                cand[i] = rng.uniform(lo, hi)
            else:
                val = x0 + rng.normal(0.0, 0.5)
                if lo is not None:
                    val = max(val, lo)
                if hi is not None:
                    val = min(val, hi)
                cand[i] = val
        starts.append(cand)
    return starts


def run_multistart_local_optimizer(objective, theta0_norm, norm_bounds, method, max_iter,
                                    n_restarts=1, seed=42, max_workers=None,
                                    progress_label="Local optimizer"):
    """
    Runs scipy.optimize.minimize (L-BFGS-B or SLSQP) from `n_restarts`
    different starting points — theta0_norm itself, plus `n_restarts - 1`
    randomized ones (see `_jittered_normalized_starts`) — and returns the
    best-loglik result across all of them.

    Restarts are independent optimizer runs, so they're run concurrently
    via a thread pool (same ask/evaluate/tell parallelization pattern
    `_ask_eval_tell_loop` already uses for Nevergrad's `num_workers`) —
    N restarts costs roughly N/max_workers wall-clock time, not N×.

    Returns
    -------
    (best_opt, all_opts, best_idx)
        best_opt  : the scipy OptimizeResult with the lowest `.fun` across
            all restarts.
        all_opts  : every restart's OptimizeResult, in start order (index 0
            is always the unperturbed-theta0 restart).
        best_idx  : index into all_opts/starts of the winning restart —
            0 means "theta0 itself was already the best start found",
            >0 means a randomized restart found something better.
    """
    n_restarts = max(1, int(n_restarts))
    starts = _jittered_normalized_starts(theta0_norm, norm_bounds, n_restarts, seed=seed)
    if max_workers is None:
        max_workers = min(len(starts), 8)

    def _run_one(x0):
        return minimize(objective, x0, method=method, bounds=norm_bounds,
                         options={"maxiter": max_iter, "ftol": 1e-9, "eps": 1e-6})

    results = [None] * len(starts)
    completed = 0
    progress = None
    if len(starts) > 1:
        progress = st.progress(0, text=f"{progress_label} — 0/{len(starts)} restarts")

    if len(starts) == 1 or max_workers <= 1:
        for i, x0 in enumerate(starts):
            results[i] = _run_one(x0)
            completed += 1
            if progress is not None:
                progress.progress(int(completed / len(starts) * 100),
                                   text=f"{progress_label} — {completed}/{len(starts)} restarts")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_run_one, x0): i for i, x0 in enumerate(starts)}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                results[i] = future.result()
                completed += 1
                progress.progress(int(completed / len(starts) * 100),
                                   text=f"{progress_label} — {completed}/{len(starts)} restarts")

    best_idx = int(np.argmin([r.fun for r in results]))
    best_opt = results[best_idx]
    if progress is not None:
        origin = "theta0 (unperturbed)" if best_idx == 0 else f"random restart #{best_idx}"
        progress.progress(100, text=(f"✅ {progress_label} done — best loss {best_opt.fun:.4f} "
                                      f"from {origin} ({len(starts)} restarts total)"))
    return best_opt, results, best_idx


def run_nevergrad_optimizer(df_train, g, theta0, bounds, ng_cfg, static_cache=None):
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NGOpt"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1)))
    if static_cache is None:
        static_cache = build_static_cache(df_train, g)

    # Same fix as the L-BFGS-B/SLSQP path (modules/bounds.py::build_
    # normalized_problem): search in per-parameter-normalized space
    # instead of raw theta. Previously every unbounded dimension (gamma,
    # sigma_y, several deltas) got an arbitrary ±1e6 box, which — for a
    # population/mutation-based search like NGOpt — let those huge-range
    # dimensions dominate exploration and mutation step sizes, drowning
    # out small-range ones like Hill's n (1-15) and letting a couple of
    # parameters wander into nonsensical territory that corrupted the
    # whole loglik surface for everything else. Normalizing first makes
    # every dimension's search box proportionate, the same way it already
    # fixed L-BFGS-B/SLSQP.
    theta0_norm, norm_bounds, unscale, _scale = build_normalized_problem(theta0, bounds)
    lows, highs = _ng_bounds_arrays(norm_bounds)
    param = ng.p.Array(init=theta0_norm).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta_norm: _composite_loss(unscale(theta_norm), df_train, g, static_cache)
    best_theta_norm, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}]", loss_fn)
    best_theta = unscale(best_theta_norm) if best_theta_norm is not None else theta0.copy()
    return best_theta, best_loss


# ── Joint (bivariate) composite loss & optimizer ─────────────────────────────

def _composite_loss_joint(theta_joint, df_train, g1, g2, n1, n2,
                           static_cache1=None, static_cache2=None,
                           lambda_reg=0.0):
    """
    Same composite-loss idea as `_composite_loss`, but evaluated on the
    JOINT bivariate Kalman filter so both dependent variables (the error
    correlation rho, and the cross-intercept coupling phi_1/phi_2) are
    optimised together in a single Nevergrad run, rather than as separate
    sequential optimizer calls.

    `lambda_reg` weights an NRMSE regularization term added on top of the
    EKF negative log-likelihood — see modules/kalman.py::joint_composite_loss
    for the exact formula. Defaults to 0.0 (pure NLL, old behaviour) so
    existing callers that don't pass it are unaffected.

    theta_joint = [theta_1 (len n1) | theta_2 (len n2) | rho | phi_1 | phi_2]

    The trailing phi_1/phi_2 pair is OMITTED entirely (theta_joint ends
    right after rho) when the model is configured with "simple" (no
    carryover) intercept dynamics, OR when g1["CROSS_INTERCEPT_COUPLING_MODE"]
    is "none" — cross-intercept coupling is itself a carryover mechanism,
    so it doesn't apply there. Detected here from theta_joint's actual
    length rather than a separate flag, so this stays correct regardless
    of which caller (scipy or Nevergrad) built it.

    When the pair IS present, g1["CROSS_INTERCEPT_COUPLING_MODE"] (shared
    with g2 — see modules/pipeline.py) further decides whether ONE of the
    two directions is masked back to exactly 0.0 even though its theta
    slot exists (kept for a stable, fixed-width theta_joint layout — see
    modules/pipeline.py::run_multi_dependent_pipeline):
      "both"          -> phi_1 and phi_2 both free
      "dep1_in_dep2"  -> only phi_2 free (phi_1 forced to 0)
      "dep2_in_dep1"  -> only phi_1 free (phi_2 forced to 0)
    """
    try:
        theta1 = theta_joint[:n1]
        theta2 = theta_joint[n1:n1+n2]
        rho    = theta_joint[n1+n2]
        if len(theta_joint) - (n1 + n2) >= 3:
            coupling_mode = g1.get("CROSS_INTERCEPT_COUPLING_MODE", "both")
            allow_phi1 = coupling_mode in ("both", "dep2_in_dep1")
            allow_phi2 = coupling_mode in ("both", "dep1_in_dep2")
            phi1 = theta_joint[n1+n2+1] if allow_phi1 else 0.0
            phi2 = theta_joint[n1+n2+2] if allow_phi2 else 0.0
        else:
            phi1 = phi2 = 0.0
        p1 = unpack_theta(theta1, g1)
        p2 = unpack_theta(theta2, g2)
        loss, _, _, _, _ = joint_composite_loss(
            df_train, p1, g1, p2, g2, rho, phi1, phi2, lambda_reg,
            static_cache1=static_cache1, static_cache2=static_cache2)
        return loss
    except Exception:
        return 1e12


def run_nevergrad_optimizer_joint(df_train, g1, g2, theta0_joint, bounds_joint, n1, n2, ng_cfg,
                                   static_cache1=None, static_cache2=None, lambda_reg=0.0):
    """Joint-mode counterpart of run_nevergrad_optimizer: optimises
    theta_1, theta_2, rho, and the cross-intercept coupling phi_1/phi_2
    together against the bivariate loglik (plus the NRMSE regularization
    term, weighted by `lambda_reg` — see joint_composite_loss)."""
    import nevergrad as ng
    strategy_name = ng_cfg.get("strategy", "NGOpt"); budget = ng_cfg.get("budget", 500)
    num_workers = max(1, int(ng_cfg.get("num_workers", 1)))
    if static_cache1 is None:
        static_cache1 = build_static_cache(df_train, g1)
    if static_cache2 is None:
        static_cache2 = build_static_cache(df_train, g2)

    # Same normalization fix as run_nevergrad_optimizer above.
    theta0_joint_norm, norm_bounds_joint, unscale_joint, _scale_joint = build_normalized_problem(
        theta0_joint, bounds_joint)
    lows, highs = _ng_bounds_arrays(norm_bounds_joint)
    param = ng.p.Array(init=theta0_joint_norm).set_bounds(lows, highs)
    optimizer_cls = getattr(ng.optimizers, strategy_name, None) or ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=param, budget=budget, num_workers=num_workers)

    loss_fn = lambda theta_joint_norm: _composite_loss_joint(
        unscale_joint(theta_joint_norm), df_train, g1, g2, n1, n2, static_cache1, static_cache2,
        lambda_reg=lambda_reg)
    best_theta_norm, best_loss = _ask_eval_tell_loop(
        optimizer, budget, num_workers, f"Nevergrad [{strategy_name}] (joint bivariate)", loss_fn)
    best_theta = unscale_joint(best_theta_norm) if best_theta_norm is not None else theta0_joint.copy()
    return best_theta, best_loss
