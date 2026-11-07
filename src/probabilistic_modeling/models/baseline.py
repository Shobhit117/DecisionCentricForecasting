"""
Naive empirical-quantile baseline forecaster.

Predicts each horizon's demand quantiles directly from the series' own recent history,
with no covariates and no model fitting:

- ``method="trailing"``: quantiles of the last ``window`` observed days (same for every horizon);
- ``method="seasonal"``: quantiles of the same weekday over the last ``n_weeks`` weeks
  (horizon-specific, captures day-of-week patterns).

Its output has exactly the same columns as :class:`CatBoostQuantileForecaster`
(``target_t_plus_<h>_q<tau>`` and ``target_t_plus_<h>_mu``), so it runs through the same
DDLT, target-inventory, simulation and calibration code. Use it as the yardstick a real
model must beat on decision KPIs.

Only demand observed strictly *before* the decision period is used (no look-ahead).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probabilistic_modeling._panel import DenseHistory
from probabilistic_modeling.ddlt import DEFAULT_SUPPORT_TOL, DDLTCalculator
from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.distributions.quantile import DEFAULT_QUANTILES, validate_quantile_levels
from probabilistic_modeling.models.catboost_quantile import quantile_col_name
from probabilistic_modeling.models.forecasters import ProbabilisticForecaster, _apply_ddlt_across_groups

logger = logging.getLogger(__name__)


class EmpiricalQuantileForecaster(ProbabilisticForecaster):
    """
    Covariate-free baseline: per-series empirical quantiles of recent demand.

    Parameters
    ----------
    quantiles : sequence of float
        Quantile levels to predict (same convention as the CatBoost forecaster).
    method : {"trailing", "seasonal"}
        Window definition (see module docstring).
    window : int
        Trailing window length in periods (``method="trailing"``).
    n_weeks : int
        Number of same-weekday observations (``method="seasonal"``); assumes daily periods.
    horizon : int
        Number of forecast horizons ``target_t_plus_0 .. target_t_plus_{horizon-1}``. When fitted
        on a ``DiscreteTimeSeries`` its ``target_cols`` are used instead.
    ts_id_cols, period_col, var_col :
        Panel column names when fitting on a plain DataFrame.
    """

    forecaster_type = "empirical"

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
        method: str = "trailing",
        window: int = 56,
        n_weeks: int = 8,
        horizon: int = 7,
        ts_id_cols: list[str] | None = None,
        period_col: str = "day",
        var_col: str = "demand",
        tol: float = DEFAULT_SUPPORT_TOL,
    ):
        super().__init__()
        if method not in ("trailing", "seasonal"):
            raise ValueError("'method' must be 'trailing' or 'seasonal'.")
        if window < 1 or n_weeks < 1 or horizon < 1:
            raise ValueError("'window', 'n_weeks' and 'horizon' must be >= 1.")
        self.quantiles = tuple(float(q) for q in validate_quantile_levels(quantiles))
        self.method = method
        self.window = int(window)
        self.n_weeks = int(n_weeks)
        self.horizon = int(horizon)
        self.tol = float(tol)
        self.ts_id_cols_ = list(ts_id_cols) if ts_id_cols else []
        self.period_col_ = period_col
        self.var_col_ = var_col
        self.target_cols_ = [f"target_t_plus_{h}" for h in range(self.horizon)]
        self.history_: DenseHistory | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, ts: Any, history: pd.DataFrame | None = None) -> "EmpiricalQuantileForecaster":
        """
        Stores the demand history. ``ts`` may be a ``DiscreteTimeSeries`` (its target columns and
        id columns are adopted) or a long DataFrame ``(ts_id_cols, period_col, var_col)``.

        Pass the complete raw panel as ``history`` when the time series was built with zero
        down-sampling, so the empirical windows see every observed day.
        """
        if isinstance(ts, pd.DataFrame):
            hist = ts
            if not self.ts_id_cols_:
                raise ValueError("'ts_id_cols' must be given when fitting on a DataFrame.")
        else:
            self._bind_training_context(ts)
            self.feature_cols_ = []
            self.cat_feature_cols_ = []
            self.horizon = len(self.target_cols_)
            hist = history if history is not None else ts.df
        self.history_ = DenseHistory.from_frame(hist, self.ts_id_cols_, self.period_col_, self.var_col_)
        return self

    def update_history(self, history: pd.DataFrame) -> "EmpiricalQuantileForecaster":
        """Replaces the stored history (e.g. with newly observed sales) without other changes."""
        self.history_ = DenseHistory.from_frame(history, self.ts_id_cols_, self.period_col_, self.var_col_)
        return self

    # --------------------------------------------------------------- prediction
    def _window_values(self, r: np.ndarray, c: np.ndarray, h: int) -> np.ndarray:
        """``(n_rows, n_obs)`` matrix of history values used for horizon ``h`` (NaN = unavailable)."""
        hist = self.history_.matrix
        if self.method == "trailing":
            offsets = np.arange(-self.window, 0)
        else:
            k0 = h // 7 + 1  # first lag (in weeks) that is strictly before the decision period
            offsets = h - 7 * np.arange(k0, k0 + self.n_weeks)
        idx = c[:, None] + offsets[None, :]
        valid = (idx >= 0) & (idx < hist.shape[1]) & (r[:, None] >= 0) & (c[:, None] >= 0)
        vals = hist[np.clip(r, 0, None)[:, None], np.clip(idx, 0, hist.shape[1] - 1)]
        return np.where(valid, vals, np.nan)

    def predict(
        self,
        ts: Any = None,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
    ) -> pd.DataFrame:
        if self.history_ is None:
            raise ValueError("EmpiricalQuantileForecaster needs a history: call fit() or update_history() first.")
        t0 = time.perf_counter()
        if isinstance(ts, pd.DataFrame) and df is None:
            df, ts = ts, None
        if df is not None:
            test_data = df.copy()
            if period is not None:
                periods = list(period) if isinstance(period, (list, tuple, set, np.ndarray, pd.Series)) else [period]
                test_data = test_data[test_data[self.period_col_].isin(periods)].copy()
        elif ts is not None:
            test_data, _, _, _ = self._resolve_inference_inputs(ts=ts, period=period)
        else:
            raise ValueError("Provide a DiscreteTimeSeries or an inference DataFrame 'df'.")

        r, c = self.history_.locate(test_data)
        if np.any(r < 0):
            logger.warning("%d inference rows belong to series without history; predicting zero demand.", int(np.sum(r < 0)))
        levels = np.asarray(self.quantiles)
        new_cols: dict[str, np.ndarray] = {}
        trailing_cache = None
        for h, target in enumerate(self.target_cols_):
            if self.method == "trailing" and trailing_cache is not None:
                q_mat, mu = trailing_cache
            else:
                vals = self._window_values(r, c, h)
                has_obs = np.isfinite(vals).any(axis=1)
                q_mat = np.zeros((len(test_data), len(levels)), dtype=np.float64)
                mu = np.zeros(len(test_data), dtype=np.float64)
                if np.any(has_obs):
                    q_mat[has_obs] = np.nanquantile(vals[has_obs], levels, axis=1).T
                    mu[has_obs] = np.nanmean(vals[has_obs], axis=1)
                trailing_cache = (q_mat, mu)
            for k, lv in enumerate(levels):
                new_cols[quantile_col_name(target, lv)] = q_mat[:, k]
            new_cols[f"{target}_mu"] = mu

        out = pd.concat([test_data.reset_index(drop=True), pd.DataFrame(new_cols)], axis=1)
        logger.info(
            "Empirical %s baseline predicted %d rows x %d horizons in %.2fs",
            self.method, len(out), len(self.target_cols_), time.perf_counter() - t0,
        )
        return out

    def compute_ddlt(
        self,
        ts: Any = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
        max_service_level: float | None = None,
    ) -> pd.DataFrame:
        if lead_time is None:
            raise ValueError("'lead_time' distribution must be provided.")
        predictions = self.predict(ts, df=df, period=period)
        target_cols = list(self.target_cols_)
        return _apply_ddlt_across_groups(
            predictions=predictions,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            service_levels=service_levels,
            return_distribution=return_distribution,
            group_fn=lambda gdf, lt_distr: DDLTCalculator(
                lt_distr, cycle_time, tol=self.tol, max_service_level=max_service_level
            ).from_quantiles_group(
                gdf,
                lt_distr,
                target_cols,
                self.quantiles,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=self.period_col_,
            ),
        )

    # -------------------------------------------------------------- persistence
    def save(self, out_dir: Path | str) -> Path:
        """Saves the configuration (the history is data, not a model: re-attach it with ``update_history``)."""
        out_path = Path(out_dir)
        self._save_base_context(
            out_path,
            extra_meta={
                "quantiles": list(self.quantiles),
                "method": self.method,
                "window": self.window,
                "n_weeks": self.n_weeks,
                "horizon": self.horizon,
                "tol": self.tol,
            },
        )
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str) -> "EmpiricalQuantileForecaster":
        out_path = Path(out_dir)
        meta = json.loads((out_path / "forecaster_meta.json").read_text(encoding="utf-8"))
        forecaster = cls(
            quantiles=meta.get("quantiles", DEFAULT_QUANTILES),
            method=meta.get("method", "trailing"),
            window=int(meta.get("window", 56)),
            n_weeks=int(meta.get("n_weeks", 8)),
            horizon=int(meta.get("horizon", 7)),
            ts_id_cols=meta.get("ts_id_cols"),
            period_col=meta.get("period_col", "day"),
            var_col=meta.get("var_col", "demand"),
            tol=float(meta.get("tol", DEFAULT_SUPPORT_TOL)),
        )
        forecaster._load_base_context(out_path, meta)
        return forecaster

