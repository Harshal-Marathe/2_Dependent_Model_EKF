"""
Automatic spike / outlier impulse-dummy detection.

Lets the user say "flag the top X% most unusual observations in the
dependent variable" (e.g. 10% of 100 rows -> 10 dummies) instead of hand-
picking individual dates. Each flagged observation gets its OWN single-period
impulse dummy column (1 at that row, 0 everywhere else) — these plug directly
into the existing `dummy_cols` / `DUMMY_COLS` mechanism already wired through
modules/params.py -> modules/kalman.py (own dedicated, mildly-persistent,
Kalman-filtered beta per dummy column; see kalman.py's `_build_transition_matrix`
/ `_build_process_noise`, which give every dummy state Ls=0.98 and a
comparatively large process-noise 5e-3 — i.e. its "beta" is essentially a
free, per-period level-shift the filter fits on its own, not something that
needs its own theta slot / optimizer bounds).

Detection method
-----------------
"Unusual" is judged relative to a local baseline, not the series' global
mean/level (so a genuine trend or seasonal swing isn't itself flagged as a
spike): a centered rolling-median baseline, then a robust (MAD-based)
z-score of the residual from that baseline. This only needs numpy/pandas —
no extra dependency — and is insensitive to the handful of huge outliers
it's specifically trying to find (unlike a mean/std z-score, which those
same outliers would drag around).
"""

import numpy as np
import pandas as pd


def _robust_z_scores(values: np.ndarray, window: int = None) -> np.ndarray:
    """
    Centered rolling-median baseline + MAD-based robust z-score of the
    residual. Returns an array of |z| (absolute robust z-scores), same
    length as `values`, NaN-safe (NaNs in `values` get z = 0, never selected).
    """
    s = pd.Series(values, dtype=float)
    n = len(s)
    if n == 0:
        return np.zeros(0)

    if window is None:
        # ~1/20th of the series, odd, at least 3 — small enough to track a
        # short-lived spike as a deviation from its immediate neighbourhood,
        # large enough to not itself get dragged onto the spike.
        window = max(3, int(round(n / 20)))
        if window % 2 == 0:
            window += 1

    baseline = s.rolling(window=window, center=True, min_periods=1).median()
    resid = (s - baseline).values
    resid = np.where(np.isnan(resid), 0.0, resid)

    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad  # MAD -> std-equivalent for a normal distribution
    if scale < 1e-9:
        scale = float(np.std(resid))
    if scale < 1e-9:
        scale = 1.0

    z = np.abs(resid - med) / scale
    z = np.where(np.isnan(values), 0.0, z)
    return z


def detect_spike_indices(values: np.ndarray, pct: float, window: int = None) -> list:
    """
    Returns the (sorted, ascending) row indices of the top `pct`% most
    unusual observations in `values` by robust z-score.

    pct=10 on a 100-row series -> exactly 10 indices (round-to-nearest,
    clipped to [0, n]).
    """
    n = len(values)
    if n == 0 or pct is None or pct <= 0:
        return []
    k = int(round(float(pct) / 100.0 * n))
    k = max(0, min(k, n))
    if k == 0:
        return []
    z = _robust_z_scores(values, window=window)
    # argsort ascending on -z == descending on z; stable enough for ties
    order = np.argsort(-z, kind="stable")
    return sorted(int(i) for i in order[:k])


def build_spike_dummy_columns(df: pd.DataFrame, target_col: str, pct: float,
                               existing_cols=None, prefix: str = "dummy_spike",
                               date_col: str = None, window: int = None):
    """
    Detects the top `pct`% most unusual rows in df[target_col] and adds one
    single-period impulse dummy column per flagged row directly onto `df`
    (mutated in place — same object the caller's `st.session_state.df`
    already points at).

    Parameters
    ----------
    existing_cols : optional iterable of column names to avoid colliding
        with (e.g. columns already used as dummies for another target).
    date_col : optional column to use for a human-readable suffix in the
        generated column name (falls back to the row index if omitted / not
        present / not unique-safe).

    Returns
    -------
    (new_col_names, flagged_rows_info)
        new_col_names   : list[str], the dummy columns actually added,
                           in ROW order (not creation order).
        flagged_rows_info: list[dict] with keys "row", "label", "value",
                           "column" — one per flagged row, for a UI preview.
    """
    if target_col not in df.columns:
        return [], []

    idxs = detect_spike_indices(df[target_col].values, pct, window=window)
    if not idxs:
        return [], []

    taken = set(existing_cols or [])
    new_cols = []
    flagged_rows_info = []

    for i in idxs:
        if date_col and date_col in df.columns:
            raw_label = str(df[date_col].iloc[i])
        else:
            raw_label = f"t{i}"
        label = "".join(ch if (ch.isalnum()) else "_" for ch in raw_label).strip("_")
        if not label:
            label = f"t{i}"
        col_name = f"{prefix}_{target_col}_{label}"
        base_name = col_name
        suffix = 1
        while col_name in df.columns or col_name in taken:
            col_name = f"{base_name}_{suffix}"
            suffix += 1

        dummy = np.zeros(len(df))
        dummy[i] = 1.0
        df[col_name] = dummy
        taken.add(col_name)
        new_cols.append(col_name)
        flagged_rows_info.append({
            "row": i,
            "label": str(df[date_col].iloc[i]) if (date_col and date_col in df.columns) else str(i),
            "value": float(df[target_col].iloc[i]),
            "column": col_name,
        })

    return new_cols, flagged_rows_info


def drop_columns_if_present(df: pd.DataFrame, cols) -> None:
    """Drops `cols` from `df` in place, ignoring any that aren't present —
    used to clear out a previous run's auto-generated dummy columns before
    regenerating (so re-running detection, or changing the percentage,
    doesn't leave stale columns accumulating in the working dataset)."""
    present = [c for c in (cols or []) if c in df.columns]
    if present:
        df.drop(columns=present, inplace=True)
