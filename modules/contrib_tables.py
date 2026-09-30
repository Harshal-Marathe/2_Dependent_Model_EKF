"""
Coefficient table + coefficient-based Short-Term Contribution table.

Coefficient  = time-average of the smoothed beta_t (posterior mean).
95% CI       = 2.5 / 97.5 percentiles, across posterior draws, of each
               draw's time-averaged beta (stored by the pipeline in
               result["coef_ci_df"]).
Contribution = Coefficient x Sum of Input   (point coefficient ONLY - no
               lo / hi bands are used anywhere in the contribution maths).
"""

import html as _html

import numpy as np
import pandas as pd

_TYPE_LABEL = {
    "intercept": "Intercept", "media": "Own Media", "comp_media": "Comp Media",
    "own_nonmedia": "Own Non-Media", "comp_nonmedia": "Comp Non-Media",
    "price": "Price",
}


def _variable_index(g):
    """Ordered (name, group, state_index) - mirrors exports._build_variable_index."""
    items = [("Intercept", "intercept", 0)]
    base = 1
    for i, col in enumerate(g["MEDIA_COLS"]):
        items.append((col, "media", base + i))
    base += g["N_MEDIA"]
    for j, col in enumerate(g["COMP_MEDIA_COLS"]):
        items.append((col, "comp_media", base + j))
    base += g["N_COMP"]
    for k, col in enumerate(g["OWN_NONMEDIA_COLS"]):
        items.append((col, "own_nonmedia", base + k))
    base += g["N_OWN_NONMEDIA"]
    for k, col in enumerate(g["COMP_NONMEDIA_COLS"]):
        items.append((col, "comp_nonmedia", base + k))
    base += g["N_COMP_NONMEDIA"]
    for p, col in enumerate(g["PRICE_COLS"]):
        items.append((col, "price", base + p))
    return items


def coefficient_table(res, g):
    """One row per variable: Coefficient with its 95% credible interval.
    CI columns are NaN for results fitted before CIs were stored (re-run the
    model to get them)."""
    x_smooth = res["x_smooth"]
    ci = res.get("coef_ci_df")
    rows = []
    for name, group, si in _variable_index(g):
        r = dict(Variable=name, Type=_TYPE_LABEL[group],
                 Coefficient=float(np.mean(x_smooth[:, si])),
                 ci_lo=np.nan, ci_hi=np.nan, p_pos=np.nan)
        if ci is not None and name in ci.index:
            r["ci_lo"] = float(ci.loc[name, "Coef_lo"])
            r["ci_hi"] = float(ci.loc[name, "Coef_hi"])
            r["p_pos"] = float(ci.loc[name, "Prob_gt_0"])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(0, "#", np.arange(1, len(out) + 1))
    out = out.rename(columns={"ci_lo": "95% CI Low", "ci_hi": "95% CI High",
                              "p_pos": "P(Coef > 0)"})
    has = out["95% CI Low"].notna() & out["95% CI High"].notna()
    out["CI Excludes 0"] = np.where(
        has, np.where((out["95% CI Low"] > 0) | (out["95% CI High"] < 0), "Yes", "No"), "—")
    return out


def coefficient_contrib_frame(res, g, df_full):
    """Per-period short-term contribution using the single coefficient:
    ShortTerm_<var>[t] = Coefficient x Input[t]  (Intercept input = 1)."""
    x_smooth = res["x_smooth"]
    T = len(df_full)
    cols = {}
    for name, group, si in _variable_index(g):
        coef = float(np.mean(x_smooth[:, si]))
        inp = np.ones(T) if group == "intercept" else df_full[name].values.astype(float)
        cols[f"ShortTerm_{name}"] = coef * inp
    return pd.DataFrame(cols)


def shortterm_table(res, g, df_full, rescale_factor, excluded_media, promo_cols,
                    value_adj):
    """
    Short-Term Contribution table.

      Contribution            = Coefficient x Sum of Input
      Contri % (Pos=100)      = positives scaled to sum to 100 (negatives -> 0)
      Contri % (Pos/Neg=100)  = |Contribution| / sum |Contribution| x 100
      ROAS  (own spend vars)  = Contribution x value_adj(var) / (Raw Spend x rescale)
      EI    (own spend vars)  = ROAS / pooled ROAS  (pooled over own spend vars)

    value_adj(var) -> multiplier (Tab-2 price factor x ROI Value Conversion).
    Returns the table as a DataFrame.
    """
    x_smooth = res["x_smooth"]
    media_cols = [c for c in g.get("MEDIA_COLS", []) if c not in (excluded_media or [])]
    spend_map = g.get("MEDIA_SPEND_MAP", {})
    own_vars = media_cols + [c for c in (promo_cols or []) if c not in media_cols]
    T = len(df_full)

    rows = []
    for name, group, si in _variable_index(g):
        coef = float(np.mean(x_smooth[:, si]))
        if group == "intercept":
            raw_sum, in_sum = 0.0, float(T)
        else:
            raw_sum = in_sum = float(df_full[name].astype(float).sum())
        raw_spend = 0.0
        if name in own_vars:
            sc = spend_map.get(name, name) if name in media_cols else name
            if sc not in df_full.columns:
                sc = name
            raw_spend = float(df_full[sc].sum()) if sc in df_full.columns else 0.0
        rows.append(dict(Variable=name, raw_sum=raw_sum, raw_spend=raw_spend,
                         in_sum=in_sum, contrib=coef * in_sum, coef=coef))
    d = pd.DataFrame(rows)

    c = d["contrib"].values
    pos_total = c[c > 0].sum()
    abs_total = np.abs(c).sum()
    d["pct_pos"] = np.where(c > 0, c / pos_total * 100, 0.0) if pos_total > 1e-12 else 0.0
    d["pct_abs"] = np.abs(c) / abs_total * 100 if abs_total > 1e-12 else 0.0

    roas = np.zeros(len(d)); ei = np.zeros(len(d))
    in_pool = d["Variable"].isin(own_vars) & (d["raw_spend"] * rescale_factor > 1e-9)
    adj = np.array([value_adj(v) for v in d["Variable"]], dtype=float)
    sp = d["raw_spend"].values * rescale_factor
    roas[in_pool.values] = (c * adj)[in_pool.values] / sp[in_pool.values]
    pool_sp = sp[in_pool.values].sum()
    pooled = (c * adj)[in_pool.values].sum() / pool_sp if pool_sp > 1e-9 else 0.0
    if abs(pooled) > 1e-12:
        ei[in_pool.values] = roas[in_pool.values] / pooled
    d["ROAS"], d["EI"] = roas, ei

    out = pd.DataFrame({
        "#": np.arange(1, len(d) + 1),
        "Variable": d["Variable"],
        "Raw Sum of Input": d["raw_sum"],
        "Sum of Raw Spend": d["raw_spend"],
        "Sum of Input": d["in_sum"],
        "Contribution": d["contrib"],
        "Contri % (Pos=100)": d["pct_pos"],
        "Contri % (Pos/Neg=100)": d["pct_abs"],
        "EI": d["EI"], "ROAS": d["ROAS"],
        "Coefficient": d["coef"],
    })
    return out


_CSS = """
<style>
.stt-wrap{overflow-x:auto;border-radius:14px;background:#fff;
  box-shadow:0 2px 10px rgba(15,76,129,.15);margin:.25rem 0 .75rem 0}
table.stt{border-collapse:collapse;width:100%;font-family:'Segoe UI',Arial,sans-serif;
  font-size:14px;color:#1f2937}
table.stt thead th{background:linear-gradient(180deg,#0f5aa0,#0a4478);color:#fff;
  font-weight:600;text-align:left;padding:14px 16px;white-space:nowrap}
table.stt td{padding:14px 16px;border-bottom:1px solid #edf0f4;white-space:nowrap;
  text-align:left;font-variant-numeric:tabular-nums;background:#fff}
table.stt tbody tr:nth-child(even) td{background:#f5f7fa}
.stt-bar{display:inline-flex;align-items:center;gap:10px}
.stt-track{width:100px;height:8px;border-radius:6px;background:#e6ecf5;overflow:hidden}
.stt-fill{height:100%;background:#3b8cf0;border-radius:6px}
.stt-val{color:#1d4ed8;font-weight:600}
</style>
"""


def shortterm_table_html(t):
    """Render the Short-Term table as the styled HTML table."""
    two = lambda v: f"{v:.2f}"
    hdr = "".join(f"<th>{_html.escape(c)}</th>" for c in t.columns)
    body = []
    for _, r in t.iterrows():
        pct = float(r["Contri % (Pos=100)"])
        bar = (f'<span class="stt-bar"><span class="stt-track"><span class="stt-fill" '
               f'style="width:{max(0.0, min(100.0, pct)):.1f}%;display:block"></span></span>'
               f'<span class="stt-val">{pct:.2f}</span></span>')
        cells = [
            f"{int(r['#'])}", _html.escape(str(r["Variable"])),
            two(r["Raw Sum of Input"]), two(r["Sum of Raw Spend"]),
            f"{r['Sum of Input']:.0f}" if float(r["Sum of Input"]).is_integer() else two(r["Sum of Input"]),
            two(r["Contribution"]), bar, two(r["Contri % (Pos/Neg=100)"]),
            two(r["EI"]), two(r["ROAS"]), f"{r['Coefficient']:.6g}",
        ]
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return (_CSS + '<div class="stt-wrap"><table class="stt"><thead><tr>' + hdr
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
