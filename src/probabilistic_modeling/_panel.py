"""
Internal helpers for dense ``series x period`` panel lookups (used by the empirical baseline
forecaster and the DDLT calibration report).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DenseHistory:
    """
    Demand history as a dense ``(n_series, n_periods)`` matrix plus key -> index maps.

    Missing ``(series, period)`` combinations are filled with 0 (no recorded sales).
    ``prefix`` holds row-wise cumulative sums with a leading zero column, so the demand
    summed over periods ``[c, c + m)`` is ``prefix[r, c + m] - prefix[r, c]``.
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
        """Row / column indices of ``df``'s ``(series, period)`` pairs (``-1`` if unknown)."""
        d = df[self.ts_id_cols + [self.period_col]].copy()
        for c in self.ts_id_cols + [self.period_col]:
            if isinstance(d[c].dtype, pd.CategoricalDtype):
                d[c] = d[c].astype(d[c].cat.categories.dtype)
        keys = _row_keys(d, self.ts_id_cols)
        r = np.fromiter((self.series_index.get(k, -1) for k in keys), dtype=np.int64, count=len(keys))
        c = d[self.period_col].map(self.period_index).fillna(-1).to_numpy(dtype=np.int64)
        return r, c


def _row_keys(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if len(cols) == 1:
        return df[cols[0]].to_numpy()
    out = np.empty(len(df), dtype=object)  # 1-D array of tuples (np.array would build 2-D)
    out[:] = list(df[cols].itertuples(index=False, name=None))
    return out
