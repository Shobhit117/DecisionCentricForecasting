"""
Naive empirical-quantile baseline forecaster.

What this module does
---------------------
Predicts each horizon's demand quantiles directly from each series' own historical demand
using a dense matrix lookup (:class:`DenseHistory`), with **no covariates and no regression model**:

- ``method="trailing"``: empirical quantiles of the last ``window`` observed days ``[t - window, t)``
  strictly before decision day ``t`` (identical across all horizons ``h = 0..H-1``);
- ``method="seasonal"``: empirical quantiles of the last ``n_weeks`` observations falling on the
  **same weekday** as ``t + h`` strictly before decision day ``t`` (horizon-specific; captures
  day-of-week seasonality).

Why this module is needed
-------------------------
A complex ML/GLM model is only valuable if its inventory decisions beat a zero-training,
per-SKU historical empirical quantile rule on downstream KPIs (fill rate for a given stock
investment, or net profit). Because :class:`EmpiricalQuantileForecaster` outputs the exact
same columns as :class:`CatBoostQuantileForecaster` (``target_t_plus_<h>_q<tau>`` and
``target_t_plus_<h>_mu``), it plugs directly into the same DDLT convolution, newsvendor/quantile
target policy, inventory simulator, and DDLT calibration report.

Where this module is used
-------------------------
- Re-exported by ``probabilistic_modeling`` and ``probabilistic_modeling.models``
- Loaded by :func:`probabilistic_modeling.models.forecasters.load_forecaster` when ``forecaster_type == "empirical"``
- Used in ``scripts/run_m5_pipeline.py`` when ``--baseline trailing`` or ``--baseline seasonal`` is passed.
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
from probabilistic_modeling.models.forecasters import (
    ProbabilisticForecaster,
    _apply_ddlt_across_groups,
    compute_forecasting_metrics,
)

logger = logging.getLogger(__name__)


class EmpiricalQuantileForecaster(ProbabilisticForecaster):
    """
    Covariate-free baseline forecaster using per-series empirical quantiles of recent demand.

    Parameters
    ----------
    quantiles : sequence of float, default DEFAULT_QUANTILES
        Quantile levels to predict (same convention as :class:`CatBoostQuantileForecaster`).
    method : {"trailing", "seasonal"}, default "trailing"
        - ``"trailing"``: last ``window`` periods ``[t - window, t)``.
        - ``"seasonal"``: last ``n_weeks`` same-weekday observations ``t + h - 7 * k`` before ``t``.
    window : int, default 56
        Trailing window length in periods (used when ``method="trailing"``).
    n_weeks : int, default 8
        Number of same-weekday historical observations (used when ``method="seasonal"``).
    horizon : int, default 7
        Number of forecast horizons ``target_t_plus_0 .. target_t_plus_{horizon-1}`` (overridden
        automatically when fitted on a :class:`DiscreteTimeSeries`).
    ts_id_cols, period_col, var_col :
        Panel column names when fitting directly on a long DataFrame rather than a ``DiscreteTimeSeries``.
    tol : float, default DEFAULT_SUPPORT_TOL
        Upper-tail truncation tolerance passed to :class:`DDLTCalculator` during :meth:`compute_ddlt`.
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
        Index the demand history into a :class:`DenseHistory` matrix.

        ``ts`` may be either a :class:`DiscreteTimeSeries` (from which ``ts_id_cols``, ``period_col``,
        ``var_col``, ``target_cols``, and ``series_metadata`` are adopted) or a raw long DataFrame.
        When ``ts`` is a :class:`DiscreteTimeSeries` that was built with zero-row down-sampling
        (`downsample_zeros_frac`), pass the complete raw sales DataFrame as ``history`` so that
        the empirical windows contain all historical zero-demand days.
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
        """Replace the stored :class:`DenseHistory` with newly observed sales data (e.g. after ``load()``)."""
        self.history_ = DenseHistory.from_frame(history, self.ts_id_cols_, self.period_col_, self.var_col_)
        return self

    # --------------------------------------------------------------- prediction
    def _window_values(self, r: np.ndarray, c: np.ndarray, h: int) -> np.ndarray:
        """
        Extract the ``(n_rows, n_obs)`` matrix of historical demand values strictly before
        column index ``c`` for horizon ``h = 0, ..., H - 1`` (`NaN` where out of bounds).

        - ``method="trailing"``: column offsets ``[-window, ..., -1]`` relative to ``c``.
        - ``method="seasonal"``: column offsets ``h - 7 * k`` for ``k = k0, ..., k0 + n_weeks - 1``
          where ``k0 = h // 7 + 1`` is the smallest integer number of weeks that places ``h - 7 * k0 < 0``
          (strictly before decision period ``c``, preventing any look-ahead leakage).
        """
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
        """
        Predict empirical quantiles (``target_t_plus_<h>_q<tau>``) and sample means
        (``target_t_plus_<h>_mu``) for every inference row and horizon.

        Note: For ``method="trailing"``, the trailing window ``[c - window, c)`` does not depend
        on ``h``, so ``q_mat`` and ``mu`` are computed once at ``h = 0`` and reused across all
        horizons via ``trailing_cache``.
        """
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
        if all(t in out.columns for t in self.target_cols_):
            compute_forecasting_metrics(out, list(self.target_cols_), [f"{t}_mu" for t in self.target_cols_])
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
        """
        Predict empirical quantiles and propagate them into DDLT distributions or target
        inventories via :meth:`DDLTCalculator.from_quantiles_group`.
        """
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
        """Save the baseline configuration to ``out_dir`` (re-attach sales history after loading via ``update_history``)."""
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
        """Load an :class:`EmpiricalQuantileForecaster` configuration from ``out_dir``."""
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
