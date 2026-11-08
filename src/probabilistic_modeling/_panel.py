"""
Internal helpers for dense ``series x period`` panel lookups and O(1) window sums.

What this module does
---------------------
Converts a long-format demand DataFrame ``(ts_id_cols, period_col, var_col)`` into a
dense 2D NumPy matrix ``(n_series, n_periods)`` plus a row-wise cumulative prefix-sum
array ``prefix`` of shape ``(n_series, n_periods + 1)``.

Why this module is needed
-------------------------
Two components in the library need fast random access to historical or future demand slices:
1. :class:`EmpiricalQuantileForecaster` needs to slice trailing windows ``[t - W, t)`` or
   seasonal same-weekday lags ``t + h - 7k`` across thousands of series and decision days.
2. :func:`evaluate_ddlt_calibration` needs the realized demand summed over ``[t, t + L + R)``
   for every possible lead time ``L`` in the support of the lead-time distribution.

Doing these lookups with repeated Pandas ``groupby`` / ``merge`` operations is orders of
magnitude slower than indexing a dense 2D NumPy matrix and subtracting two prefix-sum
entries ``prefix[r, c + m] - prefix[r, c]`` in ``O(1)`` time.

Where this module is used
-------------------------
- ``probabilistic_modeling.models.baseline.EmpiricalQuantileForecaster``
- ``probabilistic_modeling.evaluation.ddlt_calibration.evaluate_ddlt_calibration``
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DenseHistory:
    """
    Demand history stored as a dense ``(n_series, n_periods)`` float64 matrix plus index maps.

    Attributes
    ----------
    matrix : np.ndarray, shape ``(n_series, n_periods)``
        Dense demand grid; missing ``(series, period)`` combinations default to ``0.0``.
    prefix : np.ndarray, shape ``(n_series, n_periods + 1)``
        Row-wise cumulative sums with a leading zero column (`prefix[:, 0] == 0`), so the
        total demand of series ``r`` over ``[c, c + m)`` is ``prefix[r, c + m] - prefix[r, c]``.
    series_index : dict
        Maps each series key (scalar if ``len(ts_id_cols) == 1``, else tuple) to row ``0..n_series-1``.
    period_index : dict
        Maps each sorted unique period value to column ``0..n_periods-1``.
    periods : np.ndarray
        Sorted 1D array of unique period values.
    ts_id_cols : list[str]
        Column names identifying a time series.
    period_col : str
        Column name identifying the time period.
    """

    matrix: np.ndarray
    prefix: np.ndarray
    series_index: dict
    period_index: dict
    periods: np.ndarray
    ts_id_cols: list[str]
    period_col: str

    @classmethod
    def from_frame(
        cls,
        history: pd.DataFrame,
        ts_id_cols: list[str],
        period_col: str,
        var_col: str,
    ) -> "DenseHistory":
        """
        Build a :class:`DenseHistory` from a long-format DataFrame.

        What it does
        ------------
        1. Unwraps any Pandas ``Categorical`` columns to their underlying dtype so dictionary
           lookups do not fail when comparing categories from different DataFrames.
        2. Assigns each unique series a row index ``0..n_series-1`` and each sorted unique
           period a column index ``0..n_periods-1``.
        3. Accumulates ``var_col`` into ``matrix`` via ``np.add.at`` (handling any duplicate
           rows by summing) and computes the row-wise cumulative sum ``prefix``.

        Where it is used
        ----------------
        Called by :meth:`EmpiricalQuantileForecaster.fit` / :meth:`update_history` and by
        :func:`evaluate_ddlt_calibration`.
        """
        ts_id_cols = list(ts_id_cols)
        h = history[ts_id_cols + [period_col, var_col]].copy()
        for c in ts_id_cols:
            if isinstance(h[c].dtype, pd.CategoricalDtype):
                h[c] = h[c].astype(h[c].cat.categories.dtype)
        if isinstance(h[period_col].dtype, pd.CategoricalDtype):
            h[period_col] = h[period_col].astype(h[period_col].cat.categories.dtype)

        keys = _row_keys(h, ts_id_cols)
        uniq_keys = pd.unique(keys)
        series_index = {k: i for i, k in enumerate(uniq_keys)}
        periods = np.sort(h[period_col].unique())
        period_index = {p: j for j, p in enumerate(periods)}

        rows = np.fromiter((series_index[k] for k in keys), dtype=np.int64, count=len(keys))
        cols = h[period_col].map(period_index).to_numpy(dtype=np.int64)
        matrix = np.zeros((len(uniq_keys), len(periods)), dtype=np.float64)
        np.add.at(matrix, (rows, cols), h[var_col].fillna(0).to_numpy(dtype=np.float64))

        prefix = np.zeros((matrix.shape[0], matrix.shape[1] + 1), dtype=np.float64)
        np.cumsum(matrix, axis=1, out=prefix[:, 1:])
        return cls(matrix, prefix, series_index, period_index, periods, ts_id_cols, period_col)

    def locate(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Map each row of ``df`` to its ``(row_idx, col_idx)`` coordinates in ``self.matrix``.

        Returns ``-1`` for any series or period not present in ``self``, allowing callers
        to mask or warn on unseen series/periods cleanly.
        """
        d = df[self.ts_id_cols + [self.period_col]].copy()
        for c in self.ts_id_cols + [self.period_col]:
            if isinstance(d[c].dtype, pd.CategoricalDtype):
                d[c] = d[c].astype(d[c].cat.categories.dtype)
        keys = _row_keys(d, self.ts_id_cols)
        r = np.fromiter((self.series_index.get(k, -1) for k in keys), dtype=np.int64, count=len(keys))
        c = d[self.period_col].map(self.period_index).fillna(-1).to_numpy(dtype=np.int64)
        return r, c


def _row_keys(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Build a 1D array of hashable series keys (scalars when ``len(cols) == 1``, tuples otherwise).

    Why it is needed
    ----------------
    Passing a list of tuples directly to ``np.array(...)`` would unpack the tuples into a 2D
    array; pre-allocating a 1D ``dtype=object`` array and assigning ``out[:] = list(...)``
    preserves each tuple as a single hashable element for ``pd.unique`` and ``dict`` lookup.
    """
    if len(cols) == 1:
        return df[cols[0]].to_numpy()
    out = np.empty(len(df), dtype=object)  # 1-D array of tuples (np.array would build 2-D)
    out[:] = list(df[cols].itertuples(index=False, name=None))
    return out
