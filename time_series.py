import json
import logging
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from ddlt import DDLTCalculator
from discrete_distribution import DiscreteDistribution, compute_best_fit_distribution
from forecasters import (
    CatBoostQuantileForecaster,
    GLMForecaster,
    ProbabilisticForecaster,
    compute_forecasting_metrics,
    evaluate_quantile_forecasts,
    train_single_glm,
)
from quantile_distribution import DEFAULT_QUANTILES

if TYPE_CHECKING:
    from catboost_quantile import CatBoostQuantileModel

logger = logging.getLogger(__name__)


def train_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str = "nbinom",
) -> tuple[dict, Any]:
    """Trains a Generalized Linear Model (GLM) on the provided DataFrame."""
    return train_single_glm(df, feature_cols, target_col, model_type)


def predict_glm(
    model_data: dict[str, Any],
    feature_dict: dict[str, Any] | pd.Series,
) -> tuple[float, float | None]:
    """
    Computes the expected value (mu) and dispersion parameter (alpha)
    for a given feature set using trained GLM coefficients.
    """
    coefficients = model_data["coefficients"]
    z = coefficients.get("Intercept", 0.0)
    for feature_name, coef_value in coefficients.items():
        if feature_name != "Intercept":
            z += feature_dict.get(feature_name, 0.0) * coef_value
    mu = np.exp(z)
    alpha = model_data.get("alpha", np.nan)
    return mu, alpha


def load_glm_model_data(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    for col in df.columns:
        if col.startswith("target"):
            df[col] = df[col].apply(json.loads)
    return df


class DiscreteTimeSeries:
    """
    Panel time-series builder and feature engineering pipeline for discrete demand.

    Supports both historical backtesting (``num_test_days > 0``) and production
    inference (``num_test_days = 0``), where future target actuals are unavailable
    at the latest decision date(s) but future-known covariates (calendar, events,
    scheduled prices) can still be attached across the forecast horizon.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ts_id_cols: list[str],
        period_col: str,
        var_col: str,
        to_datetime: bool = False,
        default_value: int = 0,
        lags: tuple[int, ...] = (1, 7, 14),
        rolling_windows: tuple[int, ...] = (7, 28),
        num_future_targets: int = 7,
        num_test_days: int = 1,
        include_std_dev: bool = False,
        use_best_fit_feature: bool = False,
        use_one_hot_encoding: bool = True,
        downsample_zeros_frac: float | None = 0.1,
        fit_historical_distributions: bool = True,
    ):
        t0 = time.perf_counter()
        self.ts_id_cols = list(ts_id_cols)
        self.var_col = var_col
        self.period_col = period_col
        self.num_future_targets = num_future_targets
        self.num_test_days = num_test_days or 0
        self.downsample_zeros_frac = downsample_zeros_frac
        self.df = df[self.ts_id_cols + [period_col, var_col]].copy()
        for col in self.ts_id_cols:
            self.df[col] = self.df[col].astype("category")
        if to_datetime:
            self.df[period_col] = pd.to_datetime(self.df[period_col])
        self.start_period = self.df[period_col].min()
        self.end_period = self.df[period_col].max()
        self.periods = sorted(self.df[period_col].unique())
        self.num_periods = len(self.periods)
        self.df[period_col] = self.df[period_col].astype("category")
        self.feature_cols: list[str] = []
        self.target_cols: list[str] = []
        self.categorical_feature_cols: list[str] = []
        self.test_start_period = None
        self.use_one_hot_encoding = use_one_hot_encoding
        self.logger = logger

        # Compute all ts_id x period combinations:
        unique_ts_id = self.df[self.ts_id_cols].drop_duplicates()
        self.num_ts = len(unique_ts_id)
        self.logger.info(
            "Building panel for %d series across %d periods (%s to %s)...",
            self.num_ts,
            self.num_periods,
            self.start_period,
            self.end_period,
        )
        unique_ts_id_period = pd.merge(
            unique_ts_id, pd.DataFrame({period_col: self.periods}), how="cross"
        )
        self.df = pd.merge(
            unique_ts_id_period, self.df, on=self.ts_id_cols + [period_col], how="left"
        ).fillna({var_col: default_value})

        self._add_test_flag(self.num_test_days)
        grouped = self.df.groupby(self.ts_id_cols, observed=True)
        self._add_overdispersion_flag(grouped)
        if fit_historical_distributions or use_best_fit_feature:
            self._fit_distributions(grouped, use_best_fit_feature, use_one_hot_encoding)
        self._engineer_features(lags, rolling_windows, num_future_targets, include_std_dev)
        self.logger.info(
            "Panel ready in %.2fs: %d rows, %d numerical features, %d categorical features, %d target horizons",
            time.perf_counter() - t0,
            len(self.df),
            len(self.feature_cols),
            len(self.categorical_feature_cols),
            len(self.target_cols),
        )

    def _add_test_flag(self, num_test_days: int):
        if num_test_days is not None and num_test_days > 0:
            self.test_start_period = self.periods[-num_test_days]
            self.df["is_test"] = np.where(self.df[self.period_col] >= self.test_start_period, 1, 0)
        else:
            # Production mode: all observed periods are training history
            self.test_start_period = None
            self.df["is_test"] = 0

    def get_inference_data(self, period: Any = None) -> pd.DataFrame:
        """
        Returns the rows to forecast on:
        - If ``period`` is specified, returns rows for that period.
        - In backtesting mode (``num_test_days > 0``), returns rows with ``is_test == 1``.
        - In production mode (``num_test_days == 0``), returns rows at the latest observed
          period (``self.end_period``) so future demand ``t, ..., t+H-1`` can be predicted.
        """
        if period is not None:
            return self.df[self.df[self.period_col] == period].copy()
        if self.test_start_period is not None:
            return self.df[self.df["is_test"] == 1].copy()
        return self.df[self.df[self.period_col] == self.end_period].copy()

    def _engineer_features(
        self,
        lags: tuple[int, ...],
        rolling_windows: tuple[int, ...],
        num_future_targets: int,
        include_std_dev: bool,
        apply_log_transformation: bool = True,
    ):
        t0 = time.perf_counter()
        self.logger.info(
            "Engineering time-series features (lags=%s, rolling_windows=%s, include_std_dev=%s, horizons=%d)...",
            lags,
            rolling_windows,
            include_std_dev,
            num_future_targets,
        )
        feature_df = self.df.copy().sort_values(by=self.ts_id_cols + [self.period_col])
        grouped_df = feature_df.groupby(self.ts_id_cols, observed=True)

        # Create Lag Features:
        for lag in lags:
            col_name = f"lag_{lag}"
            shifted_val = grouped_df[self.var_col].shift(lag)
            feature_df[col_name] = np.log1p(shifted_val) if apply_log_transformation else shifted_val
            self.feature_cols.append(col_name)

        # Create Rolling Window Features (vectorized Cython groupby rolling on shifted series):
        shifted_var = grouped_df[self.var_col].shift(1)
        group_keys = [feature_df[c] for c in self.ts_id_cols]
        for window in rolling_windows:
            col_name = f"rolling_mean_{window}"
            roll_obj = shifted_var.groupby(group_keys, observed=True).rolling(window=window, min_periods=1)
            rmean = roll_obj.mean().reset_index(level=list(range(len(self.ts_id_cols))), drop=True)
            feature_df[col_name] = np.log1p(rmean) if apply_log_transformation else rmean
            self.feature_cols.append(col_name)

            if include_std_dev:
                col_name = f"rolling_std_{window}"
                rstd = (
                    roll_obj.std()
                    .reset_index(level=list(range(len(self.ts_id_cols))), drop=True)
                    .fillna(0.0)
                )
                feature_df[col_name] = np.log1p(rstd) if apply_log_transformation else rstd
                self.feature_cols.append(col_name)

        # Future Targets:
        self.target_cols = []
        for i in range(0, num_future_targets):
            col_name = f"target_t_plus_{i}"
            feature_df[col_name] = grouped_df[self.var_col].shift(-i)
            if self.test_start_period is not None and i > 0:
                target_in_test = grouped_df["is_test"].shift(-i) == 1
                feature_df.loc[(feature_df["is_test"] == 0) & target_in_test, col_name] = np.nan
            self.target_cols.append(col_name)

        # Remove rows with missing features:
        feature_df = feature_df[feature_df[self.feature_cols].isna().sum(axis=1) == 0]

        # Optionally downsample rows where both features and targets are all 0s in training data:
        if self.downsample_zeros_frac is not None and 0.0 < self.downsample_zeros_frac < 1.0:
            rows_before = len(feature_df)
            mask = (feature_df[self.feature_cols].sum(axis=1) == 0) & (
                feature_df[self.target_cols].fillna(0).sum(axis=1) == 0
            )
            if self.test_start_period is not None:
                mask = mask & (feature_df[self.period_col] < self.test_start_period)
            else:
                # In production mode, always keep the latest period intact for inference
                mask = mask & (feature_df[self.period_col] != self.end_period)
            zero_rows = feature_df[mask].sample(frac=self.downsample_zeros_frac, random_state=42)
            non_zero_rows = feature_df[~mask]
            feature_df = pd.concat([zero_rows, non_zero_rows])
            self.logger.info(
                "Downsampled all-zero training rows (frac=%.2f): %d -> %d rows in %.2fs",
                self.downsample_zeros_frac,
                rows_before,
                len(feature_df),
                time.perf_counter() - t0,
            )
        self.df = feature_df

    def _create_tmp_col_for_train(self, col_name: str) -> str:
        tmp_col = f"tmp_{col_name}"
        if self.test_start_period is not None:
            self.df[tmp_col] = np.where(
                self.df[self.period_col] >= self.test_start_period, np.nan, self.df[self.var_col]
            )
        else:
            self.df[tmp_col] = self.df[self.var_col].copy()
        return tmp_col

    def _add_overdispersion_flag(self, grouped: DataFrameGroupBy):
        tmp_col = self._create_tmp_col_for_train(self.var_col)
        mean = grouped[tmp_col].transform("mean")
        var = grouped[tmp_col].transform("var")
        self.df["overdispersion"] = (var > mean).astype(int)
        self.df.drop(columns=[tmp_col], inplace=True)

    def _fit_distributions(
        self,
        grouped: DataFrameGroupBy,
        use_best_fit_feature: bool = False,
        one_hot: bool = True,
    ):
        t0 = time.perf_counter()
        tmp_col = self._create_tmp_col_for_train(self.var_col)
        self.logger.info("Fitting historical parametric distributions across %d series...", self.num_ts)

        def distr_fit(group_df: pd.DataFrame) -> pd.Series:
            data = group_df[tmp_col].dropna().to_numpy(dtype=int)
            best_fit, best_fit_params = compute_best_fit_distribution(data)
            params = list(best_fit_params) + [np.nan] * (3 - len(best_fit_params))
            return pd.Series({
                "best_fit": best_fit,
                "distr_param_0": params[0],
                "distr_param_1": params[1],
                "distr_param_2": params[2],
            })

        try:
            distr_df = grouped[[tmp_col]].apply(distr_fit, include_groups=False).reset_index()
        except TypeError:
            distr_df = grouped[[tmp_col]].apply(distr_fit).reset_index()
        self.df = pd.merge(self.df, distr_df, on=self.ts_id_cols, how="left")
        self.df["best_fit"] = self.df["best_fit"].astype("category")

        self.df.drop(columns=[tmp_col], inplace=True)
        self.logger.info("Historical distribution fitting completed in %.2fs", time.perf_counter() - t0)

        if use_best_fit_feature:
            if one_hot:
                dummies = pd.get_dummies(
                    self.df["best_fit"], prefix="best_fit", dtype=int, drop_first=True
                )
                self.df = pd.concat([self.df, dummies], axis=1)
                self.feature_cols.extend([col for col in self.df.columns if col.startswith("best_fit_")])
            else:
                self.categorical_feature_cols.append("best_fit")

    def add_calendar_features(
        self,
        df_calendar: pd.DataFrame,
        day_of_week_col: str | None = None,
        use_week_end_feature: bool = False,
        use_day_of_week_features: bool = True,
        weekend_values: set[Any] | None = None,
        future_horizons: tuple[int, ...] | None = None,
    ):
        """
        Adds calendar features (day of week or weekend indicator) to the panel.

        Parameters
        ----------
        df_calendar : pd.DataFrame
            Calendar table containing ``self.period_col`` and ``day_of_week_col``.
            In production, ``df_calendar`` may extend into future periods beyond
            ``self.end_period`` so ``future_horizons`` leads can be attached.
        weekend_values : set, optional
            Values in ``day_of_week_col`` that represent weekend days. Defaults to
            ``{1, 2}`` when ``day_of_week_col`` is provided (M5 convention) or
            ``{5, 6}`` when inferred from a datetime ``period_col``.
        future_horizons : tuple[int, ...], optional
            Future offsets ``(1, 2, ...)`` for which future calendar features from
            ``df_calendar`` are attached to origin period ``t`` (works even when
            ``t + h > self.end_period``).
        """
        if use_week_end_feature and use_day_of_week_features:
            raise ValueError("Only one of 'use_week_end_feature' or 'use_day_of_week_features' can be True")

        self.logger.info("Adding calendar features (col=%s, weekend=%s, day_of_week=%s)...", day_of_week_col, use_week_end_feature, use_day_of_week_features)
        if day_of_week_col is not None and future_horizons:
            lead_lags = tuple(-h for h in future_horizons if h > 0)
            self.add_feature(
                df_calendar,
                day_of_week_col,
                join_on=[],
                is_cat_feature=use_day_of_week_features,
                lags=lead_lags,
            )
            return

        df_cal = df_calendar.copy()
        df_cal[self.period_col] = df_cal[self.period_col].astype(self.df[self.period_col].dtype)
        if day_of_week_col is not None:
            df_cal[day_of_week_col] = df_cal[day_of_week_col].astype("category")
            self.df = pd.merge(
                self.df,
                df_cal[[self.period_col, day_of_week_col]].drop_duplicates(),
                on=self.period_col,
                how="left",
            )
        if use_week_end_feature:
            self._add_weekend_feature(day_of_week_col, weekend_values=weekend_values)
        if use_day_of_week_features:
            self._add_day_of_week_features(day_of_week_col, self.use_one_hot_encoding)

    def _add_weekend_feature(self, day_of_week_col: str | None = None, weekend_values: set[Any] | None = None):
        if day_of_week_col is None:
            wknd = weekend_values if weekend_values is not None else {5, 6}
            self.df["is_weekend"] = np.where(
                pd.to_datetime(self.df[self.period_col]).dt.dayofweek.isin(wknd), 1, 0
            )
        else:
            # Default {1, 2} preserves M5 Saturday/Sunday convention when not overridden:
            wknd = weekend_values if weekend_values is not None else {1, 2}
            self.df["is_weekend"] = np.where(self.df[day_of_week_col].isin(wknd), 1, 0)
        self.feature_cols.append("is_weekend")

    def _add_day_of_week_features(self, day_of_week_col: str | None, one_hot: bool = True):
        if day_of_week_col is None:
            day_of_week_col = "wday"
            self.df["wday"] = pd.to_datetime(self.df[self.period_col]).dt.dayofweek
        if one_hot:
            self.df = pd.get_dummies(self.df, columns=[day_of_week_col], dtype=int, drop_first=True)
            self.feature_cols.extend(
                [col for col in self.df.columns if col.startswith(f"{day_of_week_col}_")]
            )
        else:
            self.df[day_of_week_col] = self.df[day_of_week_col].astype("category")
            self.categorical_feature_cols.append(day_of_week_col)

    def add_feature(
        self,
        df_feature: pd.DataFrame,
        feature_name: str,
        join_on: list[str],
        is_cat_feature: bool = False,
        default_value: Any = np.nan,
        lags: tuple[int, ...] | None = None,
        rolling_windows: tuple[int, ...] | None = None,
    ):
        """
        Merges an external numerical or categorical feature onto the panel, with optional
        lags (including negative lags ``-h`` for future-known covariates at ``t + h``)
        and rolling windows.

        When ``df_feature`` contains future periods beyond ``self.end_period`` and negative
        lags are requested, those leads are computed on ``df_feature`` before merging so
        that the final historical periods still receive future covariate values in production.
        """
        t0 = time.perf_counter()
        self.logger.info(
            "Adding external feature '%s' (join_on=%s, cat=%s, lags=%s, rolling=%s)...",
            feature_name,
            join_on,
            is_cat_feature,
            lags,
            rolling_windows,
        )
        df_feat = df_feature.copy()
        keys = list(join_on) + [self.period_col]
        df_feat = df_feat[keys + [feature_name]].drop_duplicates()

        # Pre-compute negative lags (future leads) on df_feat before merging so future rows beyond
        # self.end_period are preserved for production inference at self.end_period:
        precomputed_lead_cols: list[str] = []
        remaining_lags: list[int] = []
        if lags is not None:
            for lag in lags:
                if lag < 0:
                    col_name = f"{feature_name}_lag_{lag}".replace("-", "minus_")
                    df_feat = df_feat.sort_values(by=keys)
                    if join_on:
                        df_feat[col_name] = df_feat.groupby(join_on, observed=True)[feature_name].shift(lag)
                    else:
                        df_feat[col_name] = df_feat[feature_name].shift(lag)
                    precomputed_lead_cols.append(col_name)
                else:
                    remaining_lags.append(lag)

        df_feat[self.period_col] = df_feat[self.period_col].astype(self.df[self.period_col].dtype)
        for col in join_on:
            df_feat[col] = df_feat[col].astype("category")

        self.df = pd.merge(
            self.df, df_feat[keys + [feature_name] + precomputed_lead_cols], on=keys, how="left"
        )

        self.df.sort_values(by=self.ts_id_cols + [self.period_col], inplace=True)
        gdf = self.df.groupby(self.ts_id_cols, observed=True)
        self.df[feature_name] = gdf[feature_name].ffill().bfill().fillna(default_value)
        for col_name in precomputed_lead_cols:
            self.df[col_name] = gdf[col_name].ffill().bfill().fillna(default_value)

        # Positive lags on the merged panel:
        cols_lags = list(precomputed_lead_cols)
        for lag in remaining_lags:
            col_name = f"{feature_name}_lag_{lag}".replace("-", "minus_")
            self.df[col_name] = gdf[feature_name].shift(lag)
            cols_lags.append(col_name)

        # Rolling means on the merged panel:
        cols_rolling = []
        if (rolling_windows is not None) and (not is_cat_feature):
            group_keys = [self.df[c] for c in self.ts_id_cols]
            for window in rolling_windows:
                col_name = f"{feature_name}_rolling_mean_{window}"
                roll_obj = self.df[feature_name].groupby(group_keys, observed=True).rolling(
                    window=window, min_periods=1
                )
                self.df[col_name] = roll_obj.mean().reset_index(
                    level=list(range(len(self.ts_id_cols))), drop=True
                )
                cols_rolling.append(col_name)

        generated_cols = cols_lags + cols_rolling
        for col in generated_cols:
            self.df[col] = gdf[col].ffill().bfill().fillna(default_value)

        feature_list = [feature_name] + generated_cols
        if is_cat_feature and self.use_one_hot_encoding:
            original_cols = self.df.columns.copy()
            self.df = pd.get_dummies(self.df, columns=feature_list, dtype=int, drop_first=True)
            new_cols = self.df.columns.difference(original_cols)
            self.feature_cols.extend(new_cols)
        elif is_cat_feature and not self.use_one_hot_encoding:
            for f in feature_list:
                self.df[f] = self.df[f].astype("category")
                self.categorical_feature_cols.append(f)
        else:
            self.feature_cols.extend(feature_list)
        self.logger.info("Feature '%s' added in %.2fs", feature_name, time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Delegation methods to GLMForecaster, CatBoostQuantileForecaster, and DDLTCalculator
    # ------------------------------------------------------------------
    def _period_positions(self, df: pd.DataFrame) -> pd.Series:
        """Position (0-based index into self.periods) of every row's period."""
        values = df[self.period_col]
        if isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype(values.cat.categories.dtype)
        positions = np.searchsorted(np.asarray(self.periods), values.to_numpy())
        return pd.Series(positions, index=df.index)

    def _target_offset(self, target_col: str) -> int:
        return int(target_col.rsplit("_", 1)[-1])

    def train_glm(self, train_groups: tuple[str, ...] | list[str] | None = None) -> pd.DataFrame:
        return GLMForecaster(train_groups=train_groups).fit(self)

    def predict_glm(self, model_data: pd.DataFrame) -> pd.DataFrame:
        return GLMForecaster().predict(self, model_data=model_data)

    def _compute_forecasting_metrics(self, pred_df: pd.DataFrame, pred_columns: list[str]):
        compute_forecasting_metrics(pred_df, self.target_cols, pred_columns)

    def compute_lead_time_demand_glm(
        self,
        gdf: pd.DataFrame,
        lead_time: DiscreteDistribution,
        cycle_time: int,
        glm_mu_cols: list[str] | None = None,
        glm_alpha_cols: list[str] | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
    ) -> pd.DataFrame:
        return DDLTCalculator(lead_time, cycle_time).from_glm_group(
            gdf,
            lead_time,
            self.target_cols,
            glm_mu_cols=glm_mu_cols,
            glm_alpha_cols=glm_alpha_cols,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
            period_col=self.period_col,
        )

    def train_catboost(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        train_groups: tuple[str, ...] | None = None,
        extra_cat_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id"),
        num_valid_days: int = 28,
        max_train_days: int | None = None,
        catboost_params: dict[str, Any] | None = None,
    ) -> "CatBoostQuantileModel":
        forecaster = CatBoostQuantileForecaster(
            quantiles=quantiles,
            train_groups=train_groups,
            extra_cat_cols=extra_cat_cols,
            num_valid_days=num_valid_days,
            max_train_days=max_train_days,
            catboost_params=catboost_params,
        )
        return forecaster.fit(self)

    def predict_catboost(self, model: "CatBoostQuantileModel", tol: float = 1e-4) -> pd.DataFrame:
        return CatBoostQuantileForecaster(quantiles=model.quantiles, tol=tol).predict(self, model=model)

    def evaluate_quantile_forecasts(
        self,
        pred_df: pd.DataFrame,
        quantiles: tuple[float, ...],
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        return evaluate_quantile_forecasts(pred_df, self.target_cols, quantiles, group_cols)

    def compute_lead_time_demand_catboost(
        self,
        gdf: pd.DataFrame,
        lead_time: DiscreteDistribution,
        cycle_time: int,
        quantiles: tuple[float, ...],
        return_distribution: bool = False,
        pred_first_day: bool = True,
        tol: float = 1e-4,
    ) -> pd.DataFrame:
        return DDLTCalculator(lead_time, cycle_time, tol=tol).from_quantiles_group(
            gdf,
            lead_time,
            self.target_cols,
            quantiles=quantiles,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
            period_col=self.period_col,
        )


def compute_lead_time_demand(
    ts: DiscreteTimeSeries,
    model_data: "pd.DataFrame | CatBoostQuantileModel | ProbabilisticForecaster",
    lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
    lead_time_keys: list[str] | None,
    cycle_time: int,
    service_levels: pd.DataFrame | None = None,
    model_type: str = "glm",
    return_distribution: bool = False,
    pred_first_day: bool = True,
) -> pd.DataFrame:
    if isinstance(model_data, ProbabilisticForecaster):
        return model_data.compute_ddlt(
            ts,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            cycle_time=cycle_time,
            service_levels=service_levels,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
        )
    if model_type == "glm":
        return GLMForecaster().compute_ddlt(
            ts,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            cycle_time=cycle_time,
            service_levels=service_levels,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
            model_data=model_data,
        )
    if model_type == "catboost":
        return CatBoostQuantileForecaster(quantiles=model_data.quantiles).compute_ddlt(
            ts,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            cycle_time=cycle_time,
            service_levels=service_levels,
            return_distribution=service_levels is None if return_distribution else return_distribution,
            pred_first_day=pred_first_day,
            model=model_data,
        )
    raise ValueError(f"Unsupported model type: '{model_type}'. Supported options are 'glm', 'catboost'.")


def compute_distribution_dict(
    df: pd.DataFrame,
    value: str = "lead_time",
    keys: list[str] | None = None,
    method: str = "actual",
) -> tuple[dict[Any, DiscreteDistribution], list[str]]:
    if keys is None:
        keys = [col for col in df.columns if col != value]
    distr = df.groupby(keys, observed=True)[value].apply(lambda x: DiscreteDistribution.from_data(x, method)).to_dict()
    return distr, keys
