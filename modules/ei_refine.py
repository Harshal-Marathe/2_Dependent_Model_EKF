"""
Effectiveness-index (EI) refinement and spend-distributed media.

EI_i = (channel i's share of total media contribution) / (channel i's share of
total media spend).  It is RELATIVE: the spend-weighted average of EI over the
media channels is always exactly 1, so "every channel above 1" is impossible —
a channel can only be above 1 if another is below it.  What can be steered is
the *spread*: which channels sit above/below 1 and by how much.

Two refits, both warm-started from the solver's own fit and both re-estimating
every parameter with an extra penalty added to -loglik (see
run_full_ekf_pipeline(extra_loss=...)):

  • EI refinement      — penalise a channel's EI for sitting outside its band
                         (soft prior, or a much stiffer "hard band"); optional
                         "auto-differentiate" fans channels that sit on a bound
                         or tie onto distinct EIs ordered by raw effectiveness.
  • Spend-distributed  — pull every channel's effect share to its spend share
                         (EI ≈ 1, or funnel-weighted), within a ± tolerance, and
                         pin the total media contribution to a % of sales
                         (fixed or swept and the best fit kept).  This is an
                         ASSUMPTION imposed on the model, not a finding.

The penalty acts on the DIRECT (short-term) contribution beta_i,t · x_i,t from the
filtered states of the training window; the tables report the final
(short + long term) contributions from the smoothed model.
"""
import numpy as np
import pandas as pd

from modules.pipeline import run_full_ekf_pipeline

FUNNEL_PRESETS = {
    "Brand / upper-funnel":       (0.40, 1.30),   # from the reference screens
    "Mid-funnel":                 (0.70, 1.50),   # assumption — edit as needed
    "Lower-funnel / performance": (1.00, 2.00),   # assumption — edit as needed
}
CUSTOM = "Custom"
_UPPER = ("tv", "ooh", "print", "radio", "cinema", "outdoor", "newspaper", "magazine")
_LOWER = ("search", "performance", "affiliate", "retail", "direct", "crm", "email", "sms", "whatsapp")


def guess_funnel(col):
    low = str(col).lower()
    toks = low.replace("-", "_").split("_")
    if any(w in low for w in _LOWER):
        return "Lower-funnel / performance"
    if any(w in toks for w in _UPPER):
        return "Brand / upper-funnel"
    return "Mid-funnel"


# ── EI maths ─────────────────────────────────────────────────────────
def spend_shares(df, g):
    cols = g["MEDIA_COLS"]; smap = g.get("MEDIA_SPEND_MAP", {})
    sp = np.array([float(df[smap.get(c, c)].sum()) if smap.get(c, c) in df.columns
                   else float(df[c].sum()) for c in cols])
    tot = sp.sum()
    return sp / tot if tot > 0 else np.full(len(cols), 1 / max(len(cols), 1))


def ei_from_result(res):
    """EI per media channel from a result's final (short+long term) contributions."""
    r = res["roi_df"]
    c = r["TotalContrib"].to_numpy(float); s = r["TotalSpend"].to_numpy(float)
    ce = c / c.sum() if c.sum() != 0 else np.full(len(c), np.nan)
    se = s / s.sum() if s.sum() > 0 else np.full(len(s), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, ce / se, np.nan), ce, se


def media_pct_of_kpi(res):
    cd = res["contrib_df"]
    tgt = cd.columns[0]
    return float(res["roi_df"]["TotalContrib"].sum() / np.nansum(cd[tgt].values) * 100)


def feasibility_note(lo, hi, s):
    """Because the spend-weighted mean EI is 1, bands must straddle it."""
    lo_w, hi_w = float(np.sum(s * lo)), float(np.sum(s * hi))
    if lo_w > 1 + 1e-9:
        return (f"These bands can't all be met: even with every channel at its minimum the "
                f"spend-weighted average EI would be {lo_w:.2f}, but it is always exactly 1. "
                f"Lower some minimums.")
    if hi_w < 1 - 1e-9:
        return (f"These bands can't all be met: even with every channel at its maximum the "
                f"spend-weighted average EI would be {hi_w:.2f}, but it is always exactly 1. "
                f"Raise some maximums.")
    return None


def distinct_targets(raw, lo, hi, gap):
    """Target EI per channel inside its band, ordered by raw EI, ≥ `gap` apart."""
    raw = np.nan_to_num(np.asarray(raw, float), nan=1.0)
    t = np.clip(raw, lo, hi)
    o = np.argsort(raw, kind="stable")
    for k in range(1, len(o)):
        t[o[k]] = max(t[o[k]], t[o[k - 1]] + gap)
    for k in range(len(o) - 1, -1, -1):
        t[o[k]] = min(t[o[k]], hi[o[k]])
        if k < len(o) - 1:
            t[o[k]] = min(t[o[k]], t[o[k + 1]] - gap)
    return np.clip(t, lo, hi)


def spread_targets(raw, base_w, gap):
    """Spend-distributed targets: start at base weights (1 or funnel midpoints),
    separate ties by `gap` ordered by raw EI, centre so spend-weighted mean = 1
    (centring is done by the caller, which knows the spend shares)."""
    raw = np.nan_to_num(np.asarray(raw, float), nan=1.0)
    t = np.asarray(base_w, float).copy()
    o = np.argsort(raw, kind="stable")
    for k in range(1, len(o)):
        t[o[k]] = max(t[o[k]], t[o[k - 1]] + gap)
    return t


def centre_to_one(t, s, floor=0.05):
    t = np.asarray(t, float) + (1.0 - float(np.sum(s * t)))
    return np.maximum(t, floor)


# ── the penalty ──────────────────────────────────────────────────────
def make_ei_loss(df_train, g, lo, hi, targets=None, strength=1.0, band_mult=1.0,
                 diff_strength=0.0, media_pct_target=None, media_mult=1.0, unit=0.05):
    """Return extra_loss(theta, params, xf_train) for run_full_ekf_pipeline."""
    cols = g["MEDIA_COLS"]; n = len(df_train)
    X = df_train[cols].to_numpy(float)
    s = np.maximum(spend_shares(df_train, g), 1e-6)
    y_tot = float(df_train[g["TARGET_COL"]].sum())
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    tg = None if targets is None else np.asarray(targets, float)
    N = len(cols)

    def extra(theta, p, xf):
        c = (xf[:, 1:1 + N] * X).sum(axis=0)
        cm = float(c.sum())
        if not np.isfinite(cm) or cm <= 1e-9:
            return 1e9
        ei = (c / cm) / s
        dist = np.maximum(lo - ei, 0.0) + np.maximum(ei - hi, 0.0)
        pen = band_mult * strength * 0.5 * n * float(np.sum(s * (dist / unit) ** 2))
        if tg is not None and diff_strength > 0:
            pen += diff_strength * 0.1 * 0.5 * n * float(np.sum(s * ((ei - tg) / unit) ** 2))
        if media_pct_target is not None and y_tot > 0:
            m = cm / y_tot
            pen += media_mult * strength * 0.5 * n * ((m - media_pct_target / 100.0) / 0.01) ** 2
        return pen

    return extra


# ── fits ─────────────────────────────────────────────────────────────
def _fit(df, config, base, extra, max_iter, method, ng_cfg):
    return run_full_ekf_pipeline(df, config, max_iter, method, ng_cfg=ng_cfg,
                                 theta_start=base.get("theta"), extra_loss=extra)


def run_refinement(df, config, base, bands, s_cfg, max_iter, method, ng_cfg=None):
    """EI refinement. bands = (lo, hi) arrays. s_cfg: mode ('soft'|'hard'),
    strength, differentiate, diff_strength, gap."""
    g = base["g"]; df_train = df.iloc[:config["n_train"]].reset_index(drop=True)
    lo, hi = bands
    raw, _, _ = ei_from_result(base)
    tg = distinct_targets(raw, lo, hi, s_cfg["gap"]) if s_cfg["differentiate"] else None
    extra = make_ei_loss(df_train, g, lo, hi, tg, s_cfg["strength"],
                         band_mult=(50.0 if s_cfg["mode"] == "hard" else 1.0),
                         diff_strength=s_cfg["diff_strength"] if s_cfg["differentiate"] else 0.0)
    res = _fit(df, config, base, extra, max_iter, method, ng_cfg)
    return {"res": res, "lo": lo, "hi": hi, "targets": tg}


def spend_targets(base, lo, hi, s_train, sd_cfg):
    raw, _, _ = ei_from_result(base)
    if sd_cfg["funnel_weights"]:
        w = (np.asarray(lo, float) + np.asarray(hi, float)) / 2
        w = w / max(float(np.sum(s_train * w)), 1e-9)
    else:
        w = np.ones(len(raw))
    t = spread_targets(raw, w, sd_cfg["gap"]) if sd_cfg["differentiate"] else w
    t = centre_to_one(t, s_train)
    tol = sd_cfg["tol_pct"] / 100.0
    return t, t * (1 - tol), t * (1 + tol)


def run_spend_distributed(df, config, base, bands, sd_cfg, max_iter, method, ng_cfg=None,
                          progress=None):
    """Spend-distributed media. Returns dict(model=..., sweep=DataFrame, sens=DataFrame|None)."""
    g = base["g"]; df_train = df.iloc[:config["n_train"]].reset_index(drop=True)
    s_train = spend_shares(df_train, g)
    t, lo, hi = spend_targets(base, bands[0], bands[1], s_train, sd_cfg)
    kpi = float(np.nansum(base["contrib_df"][base["contrib_df"].columns[0]].values))

    def one(m_pct):
        extra = make_ei_loss(df_train, g, lo, hi, t, sd_cfg["strength"], band_mult=50.0,
                             diff_strength=sd_cfg["strength"], media_pct_target=m_pct, media_mult=20.0)
        r = _fit(df, config, base, extra, max_iter, method, ng_cfg)
        return r

    grid = list(sd_cfg["grid"]) if sd_cfg["auto_tune"] else [float(sd_cfg["target_pct"])]
    trials, best = [], None
    for i, m in enumerate(grid):
        if progress: progress(f"Spend-distributed fit: target media {m:g}% ({i + 1}/{len(grid)})")
        r = one(m)
        trials.append({"Target media % of sales": m,
                       "Achieved media %": media_pct_of_kpi(r),
                       "In-sample MAPE %": r["mape_in"] * 100, "In-sample R² %": r["r2_in"] * 100,
                       "R² % (full)": r["r2"] * 100})
        key = (r["mape_in"], -r["r2_in"])
        if best is None or key < best[0]:
            best = (key, r, m)
    sens = None
    if sd_cfg.get("sensitivity"):
        rows = []
        for j, m in enumerate(sd_cfg["sens_grid"]):
            if progress: progress(f"Media-budget sensitivity: {m:g}% ({j + 1}/{len(sd_cfg['sens_grid'])})")
            r = one(m)
            rows.append({"Target media % of sales": m, "Achieved media %": media_pct_of_kpi(r),
                         "In-sample MAPE %": r["mape_in"] * 100, "In-sample R² %": r["r2_in"] * 100,
                         "R² % (full)": r["r2"] * 100})
        sens = pd.DataFrame(rows)
    return {"model": {"res": best[1], "lo": lo, "hi": hi, "targets": t, "chosen_pct": best[2]},
            "sweep": pd.DataFrame(trials), "sens": sens}


def ei_table(res, lo=None, hi=None, base=None):
    ei, ce, se = ei_from_result(res)
    r = res["roi_df"]
    out = pd.DataFrame({"Channel": r["Channel"], "Spend share %": se * 100,
                        "Effect share %": ce * 100, "EI": ei, "ROI": r["ROI"]})
    if base is not None:
        out["EI (solver's own fit)"] = ei_from_result(base)[0]
    if lo is not None:
        out["Band"] = [f"{a:.2f}–{b:.2f}" for a, b in zip(lo, hi)]
        out["In band?"] = ["✓" if (np.isfinite(e) and a - 0.005 <= e <= b + 0.005) else "✗"
                           for e, a, b in zip(ei, lo, hi)]
    return out.sort_values("Spend share %", ascending=False, ignore_index=True)


# ── run every model in one pass ──────────────────────────────────────
ALL_MODEL_LABELS = {
    "ei_soft":      "EI refinement · soft prior",
    "ei_hard":      "EI refinement · hard band",
    "spend_flat":   "Spend-distributed · flat",
    "spend_funnel": "Spend-distributed · funnel-weighted",
    "co":           "Cochrane-Orcutt · residual re-estimation",
}


def count_fits(which, grid_len):
    """Rough number of full refits a run-all selection will perform."""
    n = 0
    n += 1 if "ei_soft" in which else 0
    n += 1 if "ei_hard" in which else 0
    n += grid_len * (("spend_flat" in which) + ("spend_funnel" in which))
    n += 1 if "co" in which else 0
    return n


def run_all_models(df, config, base, lo, hi, soft_cfg, hard_cfg, sd_cfg, which,
                   max_iter, method, ng_cfg=None, progress=None):
    """Run every model in `which` once and return {key: entry}.

    Each entry has title, desc, res, lo, hi. Spend variants also carry the
    tuning sweep. The 'co' entry carries df, cfg, res (residual re-estimation)
    instead of bands, so the caller can store it in residual_refit."""
    from modules.residual_refit import run_residual_refit

    say = progress or (lambda m: None)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    out = {}

    if "ei_soft" in which:
        say(f"{ALL_MODEL_LABELS['ei_soft']}…")
        r = run_refinement(df, config, base, (lo, hi), soft_cfg, max_iter, method, ng_cfg)
        out["ei_soft"] = dict(title=ALL_MODEL_LABELS["ei_soft"],
                              desc="Each channel's effectiveness index pulled into its band (soft prior), then refit.",
                              res=r["res"], lo=lo, hi=hi)

    if "ei_hard" in which:
        say(f"{ALL_MODEL_LABELS['ei_hard']}…")
        r = run_refinement(df, config, base, (lo, hi), hard_cfg, max_iter, method, ng_cfg)
        out["ei_hard"] = dict(title=ALL_MODEL_LABELS["ei_hard"],
                              desc="Each channel's effectiveness index clamped strictly into its band, then refit.",
                              res=r["res"], lo=lo, hi=hi)

    for key, funnel, desc in (
        ("spend_flat", False, "Media split by share of spend, equal weights — an assumption, not a finding."),
        ("spend_funnel", True, "Media split by share of spend, weighted by funnel position — an assumption, not a finding."),
    ):
        if key not in which:
            continue
        say(f"{ALL_MODEL_LABELS[key]}…")
        r = run_spend_distributed(df, config, base, (lo, hi),
                                  dict(sd_cfg, funnel_weights=funnel, sensitivity=False),
                                  max_iter, method, ng_cfg, progress=say)
        m = r["model"]
        out[key] = dict(title=ALL_MODEL_LABELS[key], desc=desc, res=m["res"], lo=m["lo"], hi=m["hi"],
                        sweep=r["sweep"], chosen_pct=m["chosen_pct"])

    if "co" in which:
        say(f"{ALL_MODEL_LABELS['co']}…")
        df2, cfg2, res2 = run_residual_refit(df, config, base, 0, max_iter, method, ng_cfg)
        out["co"] = dict(title=ALL_MODEL_LABELS["co"],
                         desc="Fitted, then the residuals were added as a variable and every coefficient was re-estimated.",
                         df=df2, cfg=cfg2, res=res2, lo=None, hi=None)
    return out
