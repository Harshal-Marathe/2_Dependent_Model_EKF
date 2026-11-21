"""
EDA core logic (no Streamlit imports) — categorisation, vintage periods,

Kept separate from the UI so it can be tested on its own.

Key concepts
------------
Vintage year : a 12-month window that starts on the month the user picks
               (e.g. start = Apr  ->  Apr-Mar years). "Vintage 1" is the first
               12 months from the start month, "Vintage 2" the next 12, etc.
Complete     : a vintage year is complete when the data covers the whole
               12-month window. Growth / CAGR use complete years only by
               default, otherwise a half-year would look like a crash.
"""

import re
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────
# Categories
# ────────────────────────────────────────────────────────────────────
CATEGORIES = ["Sales", "Media", "Price", "Distribution", "Promotions", "Competition"]
UNASSIGNED = "Unassigned"
ALL_CATEGORY_CHOICES = CATEGORIES + [UNASSIGNED]

# Default yearly aggregation per category (Price = average, rest = sum)
DEFAULT_AGG = {
    "Sales": "Sum", "Media": "Sum", "Price": "Average",
    "Distribution": "Sum", "Promotions": "Sum", "Competition": "Sum",
    UNASSIGNED: "Sum",
}

# Keyword lists. Order of evaluation matters (see CHECK_ORDER): Competition
# is tested first so that e.g. "comp_tv_spend" lands in Competition, not Media.
# Keywords of <= 3 chars must match a whole token (so "tv" does not hit
# "outlets"); longer ones match as substrings.
_KEYWORDS = {
    "Competition": ["comp", "competitor", "competition", "rival", "cmp", "brandx"],
    "Sales": ["sales", "sale", "volume", "vol", "revenue", "units", "unit", "value",
              "gmv", "orders", "qty", "quantity", "target", "dependent", "kg", "tonnes"],
    "Price": ["price", "asp", "mrp", "rsp", "pricing", "rpk"],
    "Distribution": ["distribution", "dist", "acv", "wd", "nd", "tdp", "outlet", "outlets",
                     "store", "stores", "availability", "coverage", "penetration", "reach"],
    "Promotions": ["promo", "promotion", "promotions", "discount", "offer", "coupon", "deal",
                   "tpr", "trade", "flyer", "feature", "markdown", "cashback", "bogo"],
    "Media": ["media", "tv", "digital", "social", "search", "display", "radio", "ooh",
              "print", "paid", "spend", "grp", "trp", "impression", "impressions", "youtube",
              "facebook", "google", "video", "cinema", "outdoor", "influencer", "ctv",
              "ads", "advertising", "sponsorship", "affiliate", "programmatic"],
}
CHECK_ORDER = ["Competition", "Sales", "Price", "Distribution", "Promotions", "Media"]


def _tokens(name: str):
    # split camelCase, then on any non-alphanumeric
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def guess_category(col: str) -> str:
    toks = _tokens(col)
    low = str(col).lower()
    for cat in CHECK_ORDER:
        for kw in _KEYWORDS[cat]:
            if len(kw) <= 3:
                if kw in toks:
                    return cat
            elif kw in low:
                return cat
    return UNASSIGNED


def auto_categorize(columns, config=None) -> dict:
    """
    Return {column: category}. If the user has already saved a model config
    (Step 5) its media / price / competition lists win over keyword guessing.
    """
    out = {c: guess_category(c) for c in columns}
    if config:
        def _apply(keys, cat):
            for k in keys:
                for c in (config.get(k) or []):
                    if c in out:
                        out[c] = cat
        _apply(["media"], "Media")
        _apply(["price"], "Price")
        _apply(["comp_media", "comp_nonmedia", "comp_media_2", "comp_nonmedia_2"], "Competition")
        tgt = config.get("target")
        if tgt in out:
            out[tgt] = "Sales"
    return out


# ────────────────────────────────────────────────────────────────────
# Date handling & granularity
# ────────────────────────────────────────────────────────────────────
def find_date_columns(df: pd.DataFrame):
    """Columns that parse cleanly as dates (name hint first, then dtype)."""
    cands = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            cands.append(c)
        elif df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.95:
                    cands.append(c)
            except Exception:
                pass
    cands.sort(key=lambda c: 0 if "date" in str(c).lower() else 1)
    return cands


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def detect_granularity(dates: pd.Series) -> str:
    d = pd.Series(dates).dropna().sort_values().drop_duplicates()
    if len(d) < 2:
        return "Monthly"
    med = d.diff().dropna().dt.days.median()
    if med <= 1.5:
        return "Daily"
    if med <= 10:
        return "Weekly"
    return "Monthly"


# days of slack used for the completeness test, per granularity
_STEP_DAYS = {"Daily": 1, "Weekly": 7, "Monthly": 31}
EXPECTED_PERIODS = {"Daily": 365, "Weekly": 52, "Monthly": 12}


def month_options(dates: pd.Series):
    """Month-start timestamps from first to last month present in the data."""
    d = pd.Series(dates).dropna()
    return list(pd.period_range(d.min().to_period("M"), d.max().to_period("M"), freq="M")
                .to_timestamp())


# ────────────────────────────────────────────────────────────────────
# Vintage assignment
# ────────────────────────────────────────────────────────────────────
def assign_vintage(dates: pd.Series, start_month: pd.Timestamp) -> pd.Series:
    """
    Vintage number (1, 2, 3 …) for each date; NaN for rows before the start.
    Vintage k covers months [start + 12(k-1), start + 12k).
    """
    d = pd.to_datetime(dates)
    months_from_start = (d.dt.year - start_month.year) * 12 + (d.dt.month - start_month.month)
    vint = (months_from_start // 12) + 1
    return vint.where(months_from_start >= 0)


def vintage_table(dates: pd.Series, start_month: pd.Timestamp, granularity: str) -> pd.DataFrame:
    """One row per vintage year with window, observation count and completeness."""
    v = assign_vintage(dates, start_month)
    d = pd.to_datetime(dates)
    max_d, min_d = d.max(), d[v.notna()].min() if v.notna().any() else None
    step = pd.Timedelta(days=_STEP_DAYS[granularity])
    rows = []
    if v.notna().sum() == 0:
        return pd.DataFrame(columns=["Vintage", "Label", "Window start", "Window end",
                                     "Observations", "Expected", "Complete"])
    for k in range(1, int(v.max()) + 1):
        w_start = start_month + pd.DateOffset(months=12 * (k - 1))
        w_end = start_month + pd.DateOffset(months=12 * k)      # exclusive
        n = int((v == k).sum())
        covers_end = max_d >= (w_end - step)
        covers_start = (k > 1) or (min_d is not None and min_d <= w_start + step)
        rows.append({
            "Vintage": k,
            "Label": f"Y{k} ({w_start:%b-%y} – {(w_end - pd.DateOffset(days=1)):%b-%y})",
            "Window start": w_start, "Window end": w_end - pd.DateOffset(days=1),
            "Observations": n,
            "Expected": EXPECTED_PERIODS[granularity],
            "Complete": bool(covers_end and covers_start and n > 0),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Yearly aggregation & growth metrics
# ────────────────────────────────────────────────────────────────────
def yearly_values(series: pd.Series, vint: pd.Series, agg: str,
                  ignore_zeros: bool = False) -> pd.Series:
    """Aggregate one variable by vintage year. Index = vintage number."""
    s = pd.to_numeric(series, errors="coerce")
    if ignore_zeros:
        s = s.where(s != 0)
    g = s.groupby(vint)
    out = g.sum(min_count=1) if agg == "Sum" else g.mean()
    out.index = out.index.astype(int)
    return out


def yoy_growth(yearly: pd.Series) -> pd.Series:
    """YoY % per vintage year (first year = NaN). NaN where previous <= 0."""
    prev = yearly.shift(1)
    out = (yearly / prev - 1) * 100
    return out.where(prev > 0)


def total_growth(yearly: pd.Series) -> float:
    """(last / first - 1) in %, across the supplied years."""
    y = yearly.dropna()
    if len(y) < 2 or y.iloc[0] <= 0:
        return np.nan
    return (y.iloc[-1] / y.iloc[0] - 1) * 100


def cagr(yearly: pd.Series) -> float:
    """CAGR in % = (last/first)^(1/(n-1)) - 1, n = number of years supplied."""
    y = yearly.dropna()
    if len(y) < 2 or y.iloc[0] <= 0 or y.iloc[-1] <= 0:
        return np.nan
    return ((y.iloc[-1] / y.iloc[0]) ** (1 / (len(y) - 1)) - 1) * 100


def variable_summary(series: pd.Series, vint: pd.Series, agg: str,
                     complete_years, ignore_zeros: bool = False) -> dict:
    """
    One summary row for a variable.

    Descriptive stats (Avg, Min, Max, Sum…) use every period inside the
    analysis window (i.e. from the start month on). Growth metrics use the
    yearly aggregates restricted to `complete_years`.
    """
    s = pd.to_numeric(series, errors="coerce")
    in_window = vint.notna()
    s = s[in_window]
    nz = s[s != 0]
    yearly_all = yearly_values(series, vint, agg, ignore_zeros)
    yearly_c = yearly_all[yearly_all.index.isin(complete_years)]
    yoy = yoy_growth(yearly_c)
    return {
        "Avg": s.mean(),
        "Non-zero Avg": nz.mean() if len(nz) else np.nan,
        "Min": s.min(),
        "Non-zero Min": nz.min() if len(nz) else np.nan,
        "Max": s.max(),
        "Sum": s.sum(),
        "CAGR %": cagr(yearly_c),
        "Growth %": total_growth(yearly_c),
        "Latest YoY %": yoy.dropna().iloc[-1] if yoy.notna().any() else np.nan,
    }


def build_category_tables(df: pd.DataFrame, variables, vint: pd.Series, vt: pd.DataFrame,
                          agg_by_var: dict, zero_by_var: dict, include_partial: bool,
                          category_by_var: dict = None):
    """
    Returns (summary_df, yearly_df, yoy_df) for the given variables.
      summary_df : one row per variable with all stats
      yearly_df  : rows = variables, columns = vintage labels
      yoy_df     : same shape, YoY %
    """
    labels = dict(zip(vt["Vintage"], vt["Label"]))
    complete = set(vt.loc[vt["Complete"], "Vintage"])
    use_years = set(vt["Vintage"]) if include_partial else complete

    s_rows, y_rows, g_rows = [], [], []
    for var in variables:
        agg = agg_by_var.get(var, "Sum")
        iz = zero_by_var.get(var, False)
        stats = variable_summary(df[var], vint, agg, use_years, iz)
        row = {"Variable": var}
        if category_by_var is not None:
            row["Category"] = category_by_var.get(var, UNASSIGNED)
        row["Yearly basis"] = agg + (" (ex-zero)" if iz and agg == "Average" else "")
        row.update(stats)
        s_rows.append(row)

        yv = yearly_values(df[var], vint, agg, iz)
        yv_u = yv[yv.index.isin(use_years)]
        yy = yoy_growth(yv_u)
        y_rows.append({"Variable": var, **{labels[k]: yv_u.get(k, np.nan) for k in sorted(use_years)}})
        g_rows.append({"Variable": var, **{labels[k]: yy.get(k, np.nan) for k in sorted(use_years)}})

    return pd.DataFrame(s_rows), pd.DataFrame(y_rows), pd.DataFrame(g_rows)
