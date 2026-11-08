"""
Panel time-series construction and leak-free feature engineering pipeline (`time_series.py`).

What this module does
---------------------
Transforms raw transactional or daily demand data into a balanced `(series x period)`
panel with:
1. Optional pre-launch zero trimming (`_trim_pre_launch_zeros`),
2. Leak-free historical statistics (`overdispersion` flag and method-of-moments
   `best_fit` fallback distribution parameters computed strictly on training periods),
3. Autoregressive lag features (`lag_1`, `lag_7`, ...) and Cython-vectorized rolling
   window means/standard deviations (`rolling_mean_7`, `rolling_std_7`, ...) computed
   on 1-step-shifted series (`shift(1)`) so day `t`'s features never include day `t`'s
   realized demand,
4. Multi-horizon targets (`target_t_plus_0`, ..., `target_t_plus_{H-1}`) with boundary
   masking so training rows never see test-period realizations as targets,
5. Calendar features (`add_calendar_features`) and external covariates (`add_feature`)
   supporting negative lags (`-h`) for future-known covariates at `t + h`,
6. Serializable feature replay (`TimeSeriesFeatureSpec`) so fitted forecasters can
   transform new raw data at inference time without rebuilding the training dataset.

Where it is used
----------------
- Primary data preparation entry point before calling `GLMForecaster.fit`,
  `CatBoostQuantileForecaster.fit`, or `EmpiricalQuantileForecaster.fit`.
- Replayed automatically inside `ProbabilisticForecaster.transform` and
  `ProbabilisticForecaster.predict_from_raw`.
"""
import ast
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from probabilistic_modeling.ddlt import DDLTCalculator
from probabilistic_modeling.distributions.discrete import DiscreteDistribution, compute_best_fit_distribution
from probabilistic_modeling.models.forecasters import (
    CatBoostQuantileForecaster,
    GLMForecaster,
    ProbabilisticForecaster,
    _hurdle_conditional_mean,
    compute_forecasting_metrics,
    evaluate_quantile_forecasts,
    train_single_glm,
)
from probabilistic_modeling.distributions.quantile import DEFAULT_QUANTILES

if TYPE_CHECKING:
    from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel

logger = logging.getLogger(__name__)


def _parse_serialized_dict(val: Any) -> Any:
    """
    Parses a dictionary that was serialized to CSV either as JSON or as a Python literal string.

    What it does
    ------------
    Returns `val` unchanged if it is already a `dict` or non-string; otherwise attempts
    `json.loads(val)` first and falls back to `ast.literal_eval(val)`.

    Why it is needed
    ----------------
    When a DataFrame containing dictionary cells (such as GLM coefficient dictionaries
    per horizon in `GLMForecaster.model_data`) is saved to CSV via `df.to_csv()`, Pandas
    formats each dictionary using Python's `repr()` (single quotes) rather than strict JSON
    (double quotes). Supporting both JSON and `ast.literal_eval` makes CSV round-tripping
    reliable.

    Where it is used
    ----------------
    - `load_glm_model_data` in this module.
    - `GLMModel.from_dataframe` in `models/forecasters.py`.
    """
    if isinstance(val, dict):
        return val
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(val)


@dataclass
class CalendarFeatureSpec:
    """
    Serializable specification for calendar features attached to a `DiscreteTimeSeries` panel.

    What it does
    ------------
    Records the arguments passed to `DiscreteTimeSeries.add_calendar_features` so the exact
    same calendar feature transformation can be replayed on new data during inference.

    Where it is used
    ----------------
    Stored in `TimeSeriesFeatureSpec.calendar_specs` and replayed inside
    `TimeSeriesFeatureSpec.transform`.
    """
    day_of_week_col: str | None = None
    use_week_end_feature: bool = False
    use_day_of_week_features: bool = True
    weekend_values: list[Any] | None = None
    future_horizons: list[int] | None = None


@dataclass
class ExternalFeatureSpec:
    """
    Serializable specification for an external covariate merged onto a `DiscreteTimeSeries` panel.

    What it does
    ------------
    Records the arguments passed to `DiscreteTimeSeries.add_feature` (`feature_name`,
    `join_on`, `is_cat_feature`, `default_value`, `lags`, `rolling_windows`) so external
    covariates (such as prices, promotions, or SNAP indicators) can be merged and lagged
    identically at inference time.

    Where it is used
    ----------------
    Stored in `TimeSeriesFeatureSpec.external_specs` and replayed inside
    `TimeSeriesFeatureSpec.transform`.
    """
    feature_name: str
    join_on: list[str]
    is_cat_feature: bool = False
    default_value: Any = None
    lags: list[int] | None = None
    rolling_windows: list[int] | None = None


@dataclass
class TimeSeriesFeatureSpec:
    """
    Serializable feature-engineering specification learned from a training
    `DiscreteTimeSeries` panel.

    What it does
    ------------
    Captures every hyperparameter and column list produced during training panel construction:
    - Series identifier, period, and target variable column names (`ts_id_cols`, `period_col`, `var_col`),
    - Lag and rolling window configurations (`lags`, `rolling_windows`, `include_std_dev`,
      `apply_log_transformation`),
    - Series-level metadata flags (`use_best_fit_feature`, `use_one_hot_encoding`, `trim_leading_zeros`),
    - Ordered history of `add_calendar_features` (`calendar_specs`) and `add_feature` (`external_specs`) calls,
    - Exact resulting `feature_cols`, `categorical_feature_cols`, `target_cols`, and
      `future_lead_horizon_map` (mapping future-known lead features to their lead offset `h`).

    Why it is needed
    ----------------
    In production, a trained forecaster is saved to disk and later loaded in a separate
    process to score newly arrived raw sales and covariate tables. Without `TimeSeriesFeatureSpec`,
    callers would have to manually reconstruct the entire `DiscreteTimeSeries` pipeline and risk
    feature-column mismatches (e.g., missing one-hot dummy levels or re-estimating historical
    distributions on a short inference window).

    Where it is used
    ----------------
    - Created and updated automatically on `DiscreteTimeSeries.feature_spec`.
    - Saved and loaded alongside forecaster weights in `ProbabilisticForecaster._save_base_context`
      and `_load_base_context` (`feature_spec.json`).
    - Invoked by `DiscreteTimeSeries.transform`, `ProbabilisticForecaster.transform`, and
      `ProbabilisticForecaster.predict_from_raw`.
    """
    ts_id_cols: list[str]
    period_col: str
    var_col: str
    to_datetime: bool = False
    default_value: int = 0
    lags: list[int] = field(default_factory=lambda: [1, 7, 14])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 28])
    num_future_targets: int = 7
    include_std_dev: bool = False
    apply_log_transformation: bool = True
    use_best_fit_feature: bool = False
    use_one_hot_encoding: bool = True
    trim_leading_zeros: bool = False
    calendar_specs: list[CalendarFeatureSpec] = field(default_factory=list)
    external_specs: list[ExternalFeatureSpec] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)
    categorical_feature_cols: list[str] = field(default_factory=list)
    target_cols: list[str] = field(default_factory=list)
    future_lead_horizon_map: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts this specification into a JSON-serializable dictionary (replacing `NaN` defaults with `None`)."""
        d = asdict(self)
        for ext in d.get("external_specs", []):
            val = ext.get("default_value")
            if isinstance(val, float) and np.isnan(val):
                ext["default_value"] = None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSeriesFeatureSpec":
        """Reconstructs a `TimeSeriesFeatureSpec` from a dictionary parsed from JSON."""
        cal_specs = [CalendarFeatureSpec(**c) for c in data.get("calendar_specs", [])]
        ext_specs = []
        for e in data.get("external_specs", []):
            e_copy = dict(e)
            if e_copy.get("default_value") is None:
                e_copy["default_value"] = np.nan
            ext_specs.append(ExternalFeatureSpec(**e_copy))
        return cls(
            ts_id_cols=list(data["ts_id_cols"]),
            period_col=str(data["period_col"]),
            var_col=str(data["var_col"]),
            to_datetime=bool(data.get("to_datetime", False)),
            default_value=int(data.get("default_value", 0)),
            lags=[int(x) for x in data.get("lags", [1, 7, 14])],
            rolling_windows=[int(x) for x in data.get("rolling_windows", [7, 28])],
            num_future_targets=int(data.get("num_future_targets", 7)),
            include_std_dev=bool(data.get("include_std_dev", False)),
            apply_log_transformation=bool(data.get("apply_log_transformation", True)),
            use_best_fit_feature=bool(data.get("use_best_fit_feature", False)),
            use_one_hot_encoding=bool(data.get("use_one_hot_encoding", True)),
            trim_leading_zeros=bool(data.get("trim_leading_zeros", False)),
            calendar_specs=cal_specs,
            external_specs=ext_specs,
            feature_cols=list(data.get("feature_cols", [])),
            categorical_feature_cols=list(data.get("categorical_feature_cols", [])),
            target_cols=list(data.get("target_cols", [])),
            future_lead_horizon_map={str(k): int(v) for k, v in data.get("future_lead_horizon_map", {}).items()},
        )

    def save(self, path: str | Path) -> None:
        """Writes this feature specification to a JSON file at `path`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesFeatureSpec":
        """Loads a `TimeSeriesFeatureSpec` from a JSON file at `path`."""
        with open(Path(path), encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def get_features_for_horizon(
        self,
        target_col: str,
        restrict_future_leads: bool = False,
        exclude_cols: set[str] | None = None,
    ) -> list[str]:
        """
        Returns the numerical feature list for a specific forecast horizon `target_t_plus_<h>`.

        What it does
        ------------
        Filters `self.feature_cols` to exclude any columns in `exclude_cols`. When
        `restrict_future_leads=True`, also excludes any future-known lead covariate
        (created via negative lags `-k`, representing period `t + k`) whose future offset
        `k` exceeds the target horizon `h`.

        Why it is needed
        ----------------
        When future-known covariates (such as future day-of-week or scheduled promotions at
        `t + 1, ..., t + H - 1`) are added to the panel, a model for horizon `target_t_plus_h`
        may optionally be restricted (`restrict_future_leads=True`) so it only conditions on
        covariates up to period `t + h` rather than covariates after `t + h`.

        Where it is used
        ----------------
        - `DiscreteTimeSeries.get_features_for_horizon`
        - `GLMForecaster.fit` and `CatBoostQuantileForecaster.fit` in `models/forecasters.py`
        """
        excl = exclude_cols or set()
        if not restrict_future_leads or not self.future_lead_horizon_map:
            return [c for c in self.feature_cols if c not in excl]
        h = int(target_col.rsplit("_", 1)[-1])
        return [
            c
            for c in self.feature_cols
            if c not in excl and self.future_lead_horizon_map.get(c, 0) <= h
        ]

    def transform(
        self,
        sales_df: pd.DataFrame,
        series_metadata: pd.DataFrame | None = None,
        calendar_df: pd.DataFrame | None = None,
        external_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        period: Any = None,
        num_test_days: int = 0,
    ) -> pd.DataFrame:
        """
        Reconstructs the engineered feature panel on a new `sales_df` using this specification
        and pre-fitted `series_metadata` (without re-estimating training distributions).

        What it does
        ------------
        1. Instantiates a `DiscreteTimeSeries` with `fit_historical_distributions=False`,
           `downsample_zeros_frac=None`, and `pretrained_series_metadata=series_metadata`.
        2. Replays every `CalendarFeatureSpec` in `self.calendar_specs`.
        3. Replays every `ExternalFeatureSpec` in `self.external_specs` (matching each
           feature from `external_features` or `calendar_df`).
        4. Guarantees schema alignment with training: any expected numerical column (e.g.,
           a one-hot dummy category that did not appear in the new inference slice) is added
           with value `0`, and any expected categorical column is ensured to have `"category"`
           dtype.
        5. Returns either the full engineered panel (`period="all"`) or the requested
           inference slice via `ts_new.get_inference_data(period=period)`.

        Where it is used
        ----------------
        - `DiscreteTimeSeries.transform`
        - `ProbabilisticForecaster.transform` and `ProbabilisticForecaster.predict_from_raw`
        """
        ts_new = DiscreteTimeSeries(
            sales_df,
            ts_id_cols=self.ts_id_cols,
            period_col=self.period_col,
            var_col=self.var_col,
            to_datetime=self.to_datetime,
            default_value=self.default_value,
            lags=tuple(self.lags),
            rolling_windows=tuple(self.rolling_windows),
            num_future_targets=self.num_future_targets,
            num_test_days=num_test_days,
            include_std_dev=self.include_std_dev,
            use_best_fit_feature=self.use_best_fit_feature,
            use_one_hot_encoding=self.use_one_hot_encoding,
            downsample_zeros_frac=None,
            fit_historical_distributions=False,
            trim_leading_zeros=self.trim_leading_zeros,
            pretrained_series_metadata=series_metadata,
        )
        if calendar_df is not None:
            for cal_spec in self.calendar_specs:
                ts_new.add_calendar_features(
                    calendar_df,
                    day_of_week_col=cal_spec.day_of_week_col,
                    use_week_end_feature=cal_spec.use_week_end_feature,
                    use_day_of_week_features=cal_spec.use_day_of_week_features,
                    weekend_values=set(cal_spec.weekend_values) if cal_spec.weekend_values is not None else None,
                    future_horizons=tuple(cal_spec.future_horizons) if cal_spec.future_horizons is not None else None,
                )
        for ext_spec in self.external_specs:
            src_df = None
            if isinstance(external_features, dict) and ext_spec.feature_name in external_features:
                src_df = external_features[ext_spec.feature_name]
            elif isinstance(external_features, pd.DataFrame) and ext_spec.feature_name in external_features.columns:
                src_df = external_features
            elif calendar_df is not None and ext_spec.feature_name in calendar_df.columns:
                src_df = calendar_df
            if src_df is not None:
                ts_new.add_feature(
                    src_df,
                    feature_name=ext_spec.feature_name,
                    join_on=ext_spec.join_on,
                    is_cat_feature=ext_spec.is_cat_feature,
                    default_value=ext_spec.default_value,
                    lags=tuple(ext_spec.lags) if ext_spec.lags is not None else None,
                    rolling_windows=tuple(ext_spec.rolling_windows) if ext_spec.rolling_windows is not None else None,
                )
        # Ensure all expected training feature columns exist (e.g. missing one-hot dummy levels filled with 0):
        for col in self.feature_cols:
            if col not in ts_new.df.columns:
                ts_new.df[col] = 0
        for col in self.categorical_feature_cols:
            if col not in ts_new.df.columns:
                ts_new.df[col] = pd.Series(["unknown"] * len(ts_new.df), index=ts_new.df.index, dtype="category")
            elif not isinstance(ts_new.df[col].dtype, pd.CategoricalDtype):
                ts_new.df[col] = ts_new.df[col].astype("category")
        ts_new.feature_cols = list(self.feature_cols)
        ts_new.categorical_feature_cols = list(self.categorical_feature_cols)

        if period == "all":
            return ts_new.df.copy()
        return ts_new.get_inference_data(period=period)


def train_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str = "nbinom",
    hurdle: bool = False,
) -> tuple[dict, Any]:
    """
    Trains a Generalized Linear Model (GLM) on the provided DataFrame.

    What it does
    ------------
    Thin convenience wrapper around `train_single_glm` in `models/forecasters.py` that fits
    a log-link Poisson or Negative Binomial GLM (`target_col ~ feature_cols`) on non-test rows,
    either as a standard count GLM (`hurdle=False`) or a two-part zero-hurdle GLM (`hurdle=True`).

    Where it is used
    ----------------
    Exported at package root for standalone single-horizon GLM fitting and backward compatibility.
    """
    return train_single_glm(df, feature_cols, target_col, model_type, hurdle=hurdle)


def predict_glm(
    model_data: dict[str, Any],
    feature_dict: dict[str, Any] | pd.Series,
) -> tuple[float, float | None]:
    """
    Computes the expected value (`mu`) and dispersion parameter (`alpha`) for a single
    observation using trained GLM coefficients (standard or zero-hurdle).

    What it does
    ------------
    Evaluates the linear predictor `z = Intercept + sum_j x_j * beta_j`, applies the
    inverse log-link `lam = exp(z)`, and extracts `alpha = model_data.get("alpha", np.nan)`.
    For zero-hurdle models (`"hurdle_coefficients"` present in `model_data`), also evaluates
    `pi = sigmoid(z_hurdle)` and returns `mu = pi * lam / (1 - P_0(lam, alpha))`.

    Where it is used
    ----------------
    Single-row GLM scoring helper and unit tests (`tests/test_glm.py`). Note that bulk
    multi-row scoring in `GLMForecaster.predict` uses vectorized matrix multiplication
    (`X_aligned.dot(coef_series)`) instead.
    """
    coefficients = model_data["coefficients"]
    z = coefficients.get("Intercept", 0.0)
    for feature_name, coef_value in coefficients.items():
        if feature_name != "Intercept":
            z += feature_dict.get(feature_name, 0.0) * coef_value
    lam = float(np.exp(np.clip(z, -50.0, 50.0)))
    alpha = model_data.get("alpha", np.nan)
    if "hurdle_coefficients" in model_data or str(model_data.get("model_type", "")).startswith("hurdle_"):
        h_coefs = model_data.get("hurdle_coefficients", {"Intercept": 0.0})
        z_h = h_coefs.get("Intercept", 0.0)
        for feature_name, coef_value in h_coefs.items():
            if feature_name != "Intercept":
                z_h += feature_dict.get(feature_name, 0.0) * coef_value
        pi = float(1.0 / (1.0 + np.exp(-np.clip(z_h, -50.0, 50.0))))
        mu = float(pi * _hurdle_conditional_mean(lam, alpha))
    else:
        mu = lam
    return mu, alpha


def load_glm_model_data(file_name: str | Path) -> pd.DataFrame:
    """
    Loads a saved GLM model DataFrame from CSV and parses serialized horizon dictionaries.

    What it does
    ------------
    Reads `file_name` via `pd.read_csv` and applies `_parse_serialized_dict` to every
    `target*` column so the cells are restored to Python `dict` objects.

    Where it is used
    ----------------
    Used when loading legacy CSV exports of `GLMForecaster.model_data` (modern code
    can also use `GLMForecaster.save` / `GLMForecaster.load` or `load_forecaster`).
    """
    df = pd.read_csv(file_name)
    for col in df.columns:
        if col.startswith("target"):
            df[col] = df[col].apply(_parse_serialized_dict)
    return df


class DiscreteTimeSeries:
    """
    Panel time-series builder and feature engineering pipeline for discrete demand.

    What it does
    ------------
    Given a raw demand DataFrame `df` with series identifiers `ts_id_cols`, timestamp/day
    index `period_col`, and demand column `var_col`, constructs a balanced panel and
    engineers lag, rolling, target, and series-metadata columns:

    1. **Cross-Join Panel Balancing**: Builds the full Cartesian product of unique
       `ts_id_cols` and sorted unique `periods`, filling missing `(series, period)` pairs
       with `default_value` (typically `0`).
    2. **Pre-Launch Zero Trimming** (optional, `trim_leading_zeros=True`): Drops leading
       zero-demand rows before each series' first positive demand observation so pre-launch
       structural zeros do not bias mean/variance or autoregressive features.
    3. **Train / Test Split Flagging**: Marks the last `num_test_days` periods with
       `is_test = 1` (backtesting mode) or sets `is_test = 0` everywhere when
       `num_test_days = 0` (production mode).
    4. **Leak-Free Series Metadata**: Computes per-series `overdispersion` (`var > mean`
       on training rows) and optionally fits historical method-of-moments fallback
       distributions (`best_fit`, `distr_param_0..2`) strictly on training rows (`is_test == 0`).
       When transforming new data at inference time, attaches `pretrained_series_metadata`
       from training instead of re-estimating.
    5. **Autoregressive & Rolling Features**: Computes `lag_<k>` and 1-step-shifted rolling
       statistics (`rolling_mean_<w>`, `rolling_std_<w>`) with `np.log1p` scaling.
    6. **Multi-Horizon Targets**: Creates `target_t_plus_0, ..., target_t_plus_{H-1}` and
       masks out any target on a training row (`is_test == 0`) whose future realization
       date `t + i` falls inside the test window (`is_test == 1`).
    7. **All-Zero Row Downsampling** (optional, `downsample_zeros_frac`): Randomly subsamples
       training rows where all features and targets are zero to speed up model fitting on
       intermittent retail panels.

    Where it is used
    ----------------
    Instantiated directly by users and pipelines (`scripts/run_m5_pipeline.py`) and
    internally by `TimeSeriesFeatureSpec.transform`.
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
        trim_leading_zeros: bool = False,
        pretrained_series_metadata: pd.DataFrame | None = None,
    ):
        t0 = time.perf_counter()
        self.ts_id_cols = list(ts_id_cols)
        self.var_col = var_col
        self.period_col = period_col
        self.num_future_targets = num_future_targets
        self.num_test_days = num_test_days or 0
        self.downsample_zeros_frac = downsample_zeros_frac
        self.trim_leading_zeros = bool(trim_leading_zeros)
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
        self.future_lead_horizon_map: dict[str, int] = {}
        self.test_start_period = None
        self.use_one_hot_encoding = use_one_hot_encoding
        self.series_metadata: pd.DataFrame = pd.DataFrame()
        self.feature_spec = TimeSeriesFeatureSpec(
            ts_id_cols=list(self.ts_id_cols),
            period_col=period_col,
            var_col=var_col,
            to_datetime=to_datetime,
            default_value=default_value,
            lags=list(lags),
            rolling_windows=list(rolling_windows),
            num_future_targets=num_future_targets,
            include_std_dev=include_std_dev,
            apply_log_transformation=True,
            use_best_fit_feature=use_best_fit_feature,
            use_one_hot_encoding=use_one_hot_encoding,
            trim_leading_zeros=self.trim_leading_zeros,
        )
        self.logger = logger

        # Step 1: Compute all ts_id x period combinations to form a complete balanced grid:
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

        # Step 2: Optionally drop pre-launch leading zeros for each series:
        if self.trim_leading_zeros:
            self._trim_pre_launch_zeros()

        # Step 3: Flag test periods and compute/attach series-level metadata:
        self._add_test_flag(self.num_test_days)
        if pretrained_series_metadata is not None and not pretrained_series_metadata.empty:
            self._attach_pretrained_series_metadata(
                pretrained_series_metadata, use_best_fit_feature, use_one_hot_encoding
            )
        else:
            grouped = self.df.groupby(self.ts_id_cols, observed=True)
            self._add_overdispersion_flag(grouped)
            if fit_historical_distributions or use_best_fit_feature:
                self._fit_distributions(grouped, use_best_fit_feature, use_one_hot_encoding)
            self._extract_series_metadata()

        # Step 4: Engineer lags, rolling windows, and multi-horizon targets:
        self._engineer_features(lags, rolling_windows, num_future_targets, include_std_dev)
        self._sync_feature_spec()
        self.logger.info(
            "Panel ready in %.2fs: %d rows, %d numerical features, %d categorical features, %d target horizons",
            time.perf_counter() - t0,
            len(self.df),
            len(self.feature_cols),
            len(self.categorical_feature_cols),
            len(self.target_cols),
        )

    def _trim_pre_launch_zeros(self):
        """
        Removes leading zero-demand periods prior to each series' first active (`> 0`) period.

        Why it is needed
        ----------------
        In retail datasets (such as M5), many items are launched mid-way through the multi-year
        calendar. Prior to launch, their recorded sales are `0` for hundreds of consecutive days.
        Including pre-launch zeros artificially inflates zero-mass and depresses rolling means
        and parametric distribution fits. If a series is `0` across the entire history, all rows
        for that series are retained so the series is not dropped from the panel.
        """
        self.df.sort_values(by=self.ts_id_cols + [self.period_col], inplace=True)
        active = (self.df[self.var_col] > 0).astype(int)
        cum_active = active.groupby([self.df[c] for c in self.ts_id_cols], observed=True).cumsum()
        # Keep rows from the first positive observation onward; if a series is all-zero, keep all rows:
        has_any_positive = active.groupby([self.df[c] for c in self.ts_id_cols], observed=True).transform("max") > 0
        self.df = self.df[(cum_active > 0) | (~has_any_positive)].copy()

    def _extract_series_metadata(self):
        """
        Extracts one row per series containing static training metadata (`overdispersion`,
        `positive_overdispersion`, `best_fit`, `distr_param_0..2`) into `self.series_metadata`.

        Why it is needed
        ----------------
        Persisting `self.series_metadata` allows `TimeSeriesFeatureSpec.transform` and
        `ProbabilisticForecaster` to attach training-time overdispersion flags and fallback
        distribution parameters to new inference data without re-fitting on test periods.
        """
        meta_cols = list(self.ts_id_cols)
        for c in ("overdispersion", "positive_overdispersion", "best_fit", "distr_param_0", "distr_param_1", "distr_param_2"):
            if c in self.df.columns:
                meta_cols.append(c)
        self.series_metadata = self.df[meta_cols].drop_duplicates(subset=self.ts_id_cols).reset_index(drop=True)

    def _attach_pretrained_series_metadata(
        self,
        pretrained_meta: pd.DataFrame,
        use_best_fit_feature: bool,
        one_hot: bool,
    ):
        """
        Merges pre-fitted `series_metadata` from training onto `self.df` during `transform()`.

        Why it is needed
        ----------------
        Ensures that at inference time every series uses the exact same `overdispersion`
        and `positive_overdispersion` routing (Poisson vs. Negative Binomial) and `best_fit`
        fallback parameters learned during training.
        """
        meta = pretrained_meta.copy()
        for col in self.ts_id_cols:
            if col in meta.columns and col in self.df.columns:
                meta[col] = meta[col].astype(self.df[col].dtype)
        meta_cols = [
            c
            for c in ("overdispersion", "positive_overdispersion", "best_fit", "distr_param_0", "distr_param_1", "distr_param_2")
            if c in meta.columns
        ]
        self.df = pd.merge(
            self.df,
            meta[self.ts_id_cols + meta_cols].drop_duplicates(subset=self.ts_id_cols),
            on=self.ts_id_cols,
            how="left",
        )
        if "overdispersion" in self.df.columns:
            self.df["overdispersion"] = self.df["overdispersion"].fillna(0).astype(int)
        if "positive_overdispersion" in self.df.columns:
            self.df["positive_overdispersion"] = self.df["positive_overdispersion"].fillna(0).astype(int)
        if "best_fit" in self.df.columns:
            self.df["best_fit"] = self.df["best_fit"].astype("category")
            if use_best_fit_feature:
                if one_hot:
                    dummies = pd.get_dummies(
                        self.df["best_fit"], prefix="best_fit", dtype=int, drop_first=True
                    )
                    self.df = pd.concat([self.df, dummies], axis=1)
                    self.feature_cols.extend([col for col in self.df.columns if col.startswith("best_fit_")])
                else:
                    self.categorical_feature_cols.append("best_fit")
        self.series_metadata = meta.copy()

    def _sync_feature_spec(self):
        """Synchronizes `self.feature_spec` lists with the current panel feature and target columns."""
        self.feature_spec.feature_cols = list(self.feature_cols)
        self.feature_spec.categorical_feature_cols = list(self.categorical_feature_cols)
        self.feature_spec.target_cols = list(self.target_cols)
        self.feature_spec.future_lead_horizon_map = dict(self.future_lead_horizon_map)

    def _add_test_flag(self, num_test_days: int):
        """
        Adds the binary `is_test` indicator column to `self.df`.

        What it does
        ------------
        - When `num_test_days > 0` (backtesting mode): sets `self.test_start_period` to the
          `num_test_days`-th period from the end of `self.periods` and marks rows with
          `period >= test_start_period` as `is_test = 1` (else `0`).
        - When `num_test_days == 0` (production mode): sets `self.test_start_period = None`
          and marks all rows as `is_test = 0` so the entire observed history is available
          for training.
        """
        if num_test_days is not None and num_test_days > 0:
            self.test_start_period = self.periods[-num_test_days]
            self.df["is_test"] = np.where(self.df[self.period_col] >= self.test_start_period, 1, 0)
        else:
            # Production mode: all observed periods are training history
            self.test_start_period = None
            self.df["is_test"] = 0

    def get_inference_data(self, period: Any = None) -> pd.DataFrame:
        """
        Returns the subset of rows in `self.df` to generate forecasts for:
        - If `period` is a scalar or collection of periods, returns rows matching `period`.
        - In backtesting mode (`num_test_days > 0`), returns rows with `is_test == 1`.
        - In production mode (`num_test_days == 0`), returns rows at the latest observed
          period (`self.end_period`) so future demand `t, ..., t+H-1` can be predicted.

        Where it is used
        ----------------
        Called by `ProbabilisticForecaster._resolve_inference_inputs` and
        `TimeSeriesFeatureSpec.transform`.
        """
        if period is not None:
            if isinstance(period, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
                return self.df[self.df[self.period_col].isin(list(period))].copy()
            return self.df[self.df[self.period_col] == period].copy()
        if self.test_start_period is not None:
            return self.df[self.df["is_test"] == 1].copy()
        return self.df[self.df[self.period_col] == self.end_period].copy()

    def get_features_for_horizon(
        self,
        target_col: str,
        restrict_future_leads: bool = False,
        exclude_cols: set[str] | None = None,
    ) -> list[str]:
        """Delegates to `self.feature_spec.get_features_for_horizon`."""
        return self.feature_spec.get_features_for_horizon(
            target_col, restrict_future_leads=restrict_future_leads, exclude_cols=exclude_cols
        )

    def transform(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
        external_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        period: Any = None,
        num_test_days: int = 0,
    ) -> pd.DataFrame:
        """
        Applies this panel's learned feature specification and training `series_metadata`
        to a new `sales_df` for repeated inference without retraining.
        """
        return self.feature_spec.transform(
            sales_df=sales_df,
            series_metadata=self.series_metadata,
            calendar_df=calendar_df,
            external_features=external_features,
            period=period,
            num_test_days=num_test_days,
        )

    def _engineer_features(
        self,
        lags: tuple[int, ...],
        rolling_windows: tuple[int, ...],
        num_future_targets: int,
        include_std_dev: bool,
        apply_log_transformation: bool = True,
    ):
        """
        Computes lag features, 1-step-shifted rolling window statistics, and multi-horizon
        target columns on `self.df`.

        How look-ahead leakage is prevented
        -----------------------------------
        1. **Autoregressive features**: Because `target_t_plus_0` is the demand on day `t`
           itself, all lags (`lag >= 1`) and rolling windows (`shifted_var = shift(1)`)
           strictly use observations up to day `t - 1`.
        2. **Cross-boundary targets**: For a training row at day `t` (`is_test == 0`), if
           future target `target_t_plus_i` (`i > 0`) lands on a day `t + i >= test_start_period`
           (`target_in_test == True`), that cell is masked to `NaN` (lines 557-559) so no
           model trained on `is_test == 0` rows can ever observe a test-window realization.
        """
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

        # Create Lag Features (log1p-transformed when apply_log_transformation=True):
        for lag in lags:
            col_name = f"lag_{lag}"
            shifted_val = grouped_df[self.var_col].shift(lag)
            feature_df[col_name] = np.log1p(shifted_val) if apply_log_transformation else shifted_val
            self.feature_cols.append(col_name)

        # Create Rolling Window Features (vectorized Cython groupby rolling on 1-step-shifted series):
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

        # Create Future Target Columns (target_t_plus_0 .. target_t_plus_{H-1}):
        self.target_cols = []
        for i in range(0, num_future_targets):
            col_name = f"target_t_plus_{i}"
            feature_df[col_name] = grouped_df[self.var_col].shift(-i)
            # Prevent training rows (is_test == 0) from seeing targets that fall into the test window:
            if self.test_start_period is not None and i > 0:
                target_in_test = grouped_df["is_test"].shift(-i) == 1
                feature_df.loc[(feature_df["is_test"] == 0) & target_in_test, col_name] = np.nan
            self.target_cols.append(col_name)

        # Remove warm-up rows at the start of each series where lag features are NaN:
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
        """
        Creates a temporary column `tmp_<col_name>` on `self.df` whose values are masked
        to `NaN` on all test rows (`period >= test_start_period`).

        Why it is needed
        ----------------
        Allows Pandas `.groupby().transform("mean")` / `.transform("var")` and distribution
        fitting (`_fit_distributions`) to operate on the full panel while strictly ignoring
        test-period observations.
        """
        tmp_col = f"tmp_{col_name}"
        if self.test_start_period is not None:
            self.df[tmp_col] = np.where(
                self.df[self.period_col] >= self.test_start_period, np.nan, self.df[self.var_col]
            )
        else:
            self.df[tmp_col] = self.df[self.var_col].copy()
        return tmp_col

    def _add_overdispersion_flag(self, grouped: DataFrameGroupBy):
        """
        Computes per-series binary indicators:
        - `overdispersion = 1` iff the training-window sample variance exceeds the
          training-window sample mean (`var > mean`) across all training observations.
        - `positive_overdispersion = 1` iff the sample variance exceeds the sample mean
          strictly across positive training observations (`demand > 0`).

        Where it is used
        ----------------
        Used by `GLMForecaster.fit` to automatically select a Negative Binomial vs. Poisson GLM
        for standard GLMs (`overdispersion`) and Zero-Hurdle GLMs (`positive_overdispersion`).
        """
        tmp_col = self._create_tmp_col_for_train(self.var_col)
        mean = grouped[tmp_col].transform("mean")
        var = grouped[tmp_col].transform("var")
        self.df["overdispersion"] = (var > mean).astype(int)

        pos_col = f"{tmp_col}_pos"
        self.df[pos_col] = np.where(self.df[tmp_col] > 0, self.df[tmp_col], np.nan)
        pos_mean = grouped[pos_col].transform("mean")
        pos_var = grouped[pos_col].transform("var")
        self.df["positive_overdispersion"] = (pos_var > pos_mean).fillna(False).astype(int)
        self.df.drop(columns=[tmp_col, pos_col], inplace=True)

    def _fit_distributions(
        self,
        grouped: DataFrameGroupBy,
        use_best_fit_feature: bool = False,
        one_hot: bool = True,
    ):
        """
        Fits the best-fitting discrete parametric family (`nbinom`, `binom`, `poisson`,
        `geom`, `uniform`) to each series' training observations via `compute_best_fit_distribution`.

        Why it is needed
        ----------------
        1. Populates `best_fit`, `distr_param_0`, `distr_param_1`, and `distr_param_2` on
           each row. During `DDLTCalculator` execution, if a lead-time support point `l`
           exceeds the number of explicitly forecasted horizons `H`, DDLT uses this series'
           historical `best_fit` distribution as the stationary fallback distribution for
           horizons `>= H`.
        2. Optionally exposes `best_fit` as a one-hot or categorical model feature when
           `use_best_fit_feature=True`.
        """
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
            Calendar table containing `self.period_col` and `day_of_week_col`.
            In production, `df_calendar` may extend into future periods beyond
            `self.end_period` so `future_horizons` leads can be attached.
        weekend_values : set, optional
            Values in `day_of_week_col` that represent weekend days. Defaults to
            `{1, 2}` when `day_of_week_col` is provided (M5 convention where `wday=1` is
            Saturday and `wday=2` is Sunday) or `{5, 6}` when inferred from a datetime
            `period_col` (Pandas Monday=0 .. Sunday=6 convention).
        future_horizons : tuple[int, ...], optional
            Future offsets `(1, 2, ...)` for which future calendar features from
            `df_calendar` are attached to origin period `t` via negative lags `(-1, -2, ...)`
            on `add_feature` (works even when `t + h > self.end_period`).
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

        self.feature_spec.calendar_specs.append(
            CalendarFeatureSpec(
                day_of_week_col=day_of_week_col,
                use_week_end_feature=use_week_end_feature,
                use_day_of_week_features=use_day_of_week_features,
                weekend_values=sorted(weekend_values) if weekend_values is not None else None,
                future_horizons=list(future_horizons) if future_horizons is not None else None,
            )
        )

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
        self._sync_feature_spec()

    def _add_weekend_feature(self, day_of_week_col: str | None = None, weekend_values: set[Any] | None = None):
        """Creates binary `is_weekend` feature column and appends it to `self.feature_cols`."""
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
        """Encodes `day_of_week_col` either as `drop_first=True` one-hot dummies or as a categorical feature."""
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
        Merges an external numerical or categorical covariate onto the panel, with optional
        lags (including negative lags `-h` for future-known covariates at `t + h`)
        and rolling windows.

        Why negative lags are precomputed before merging
        ------------------------------------------------
        In production mode (`num_test_days = 0`), `self.df` only contains periods up to the
        latest observed sales date `self.end_period`. However, `df_feature` (such as a future
        calendar, SNAP schedule, or planned price table) often contains rows for future periods
        `self.end_period + 1, ..., self.end_period + H`. If we merged `df_feature` onto `self.df`
        first and then shifted by `-h`, those future rows would be lost during the left join!
        By shifting negative lags (`lag < 0`, named `<feature>_lag_minus_<h>`) on `df_feature`
        *before* merging onto `self.df`, the row at `self.end_period` receives the future
        covariate values from `self.end_period + h`.
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
        self.feature_spec.external_specs.append(
            ExternalFeatureSpec(
                feature_name=feature_name,
                join_on=list(join_on),
                is_cat_feature=is_cat_feature,
                default_value=default_value,
                lags=list(lags) if lags is not None else None,
                rolling_windows=list(rolling_windows) if rolling_windows is not None else None,
            )
        )
        df_feat = df_feature.copy()
        keys = list(join_on) + [self.period_col]
        df_feat = df_feat[keys + [feature_name]].drop_duplicates()

        # Pre-compute negative lags (future leads) on df_feat before merging so future rows beyond
        # self.end_period are preserved for production inference at self.end_period:
        precomputed_lead_cols: list[str] = []
        lead_col_to_horizon: dict[str, int] = {}
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
                    lead_col_to_horizon[col_name] = abs(int(lag))
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
            new_cols = list(self.df.columns.difference(original_cols))
            self.feature_cols.extend(new_cols)
            for lead_prefix, h_val in lead_col_to_horizon.items():
                for c in new_cols:
                    if c.startswith(f"{lead_prefix}_"):
                        self.future_lead_horizon_map[c] = h_val
        elif is_cat_feature and not self.use_one_hot_encoding:
            for f in feature_list:
                self.df[f] = self.df[f].astype("category")
                self.categorical_feature_cols.append(f)
            self.future_lead_horizon_map.update(lead_col_to_horizon)
        else:
            self.feature_cols.extend(feature_list)
            self.future_lead_horizon_map.update(lead_col_to_horizon)
        self._sync_feature_spec()
        self.logger.info("Feature '%s' added in %.2fs", feature_name, time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Delegation methods to GLMForecaster, CatBoostQuantileForecaster, and DDLTCalculator
    # ------------------------------------------------------------------
    def _period_positions(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns the 0-based integer index into `self.periods` for every row's period in `df`.

        Where it is used
        ----------------
        Used by `CatBoostQuantileForecaster.fit` to compute exact period offsets (`target_pos =
        group_pos + h`) when partitioning training and validation windows without look-ahead leakage.
        """
        values = df[self.period_col]
        if isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype(values.cat.categories.dtype)
        positions = np.searchsorted(np.asarray(self.periods), values.to_numpy())
        return pd.Series(positions, index=df.index)

    def _target_offset(self, target_col: str) -> int:
        """Extracts integer horizon offset `h` from column name `'target_t_plus_<h>'`."""
        return int(target_col.rsplit("_", 1)[-1])

    def train_glm(
        self,
        train_groups: tuple[str, ...] | list[str] | None = None,
        hurdle: bool = False,
    ) -> pd.DataFrame:
        """Convenience method delegating to `GLMForecaster(train_groups=train_groups, hurdle=hurdle).fit(self)`."""
        return GLMForecaster(train_groups=train_groups, hurdle=hurdle).fit(self)

    def predict_glm(self, model_data: pd.DataFrame) -> pd.DataFrame:
        """Convenience method delegating to `GLMForecaster().predict(self, model=model_data)`."""
        return GLMForecaster().predict(self, model=model_data)

    def _compute_forecasting_metrics(self, pred_df: pd.DataFrame, pred_columns: list[str]):
        """Delegates to `compute_forecasting_metrics`."""
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
        glm_pi_cols: list[str] | None = None,
        glm_lambda_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Delegates single-group GLM DDLT calculation to `DDLTCalculator.from_glm_group`."""
        return DDLTCalculator(lead_time, cycle_time).from_glm_group(
            gdf,
            lead_time,
            self.target_cols,
            glm_mu_cols=glm_mu_cols,
            glm_alpha_cols=glm_alpha_cols,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
            period_col=self.period_col,
            glm_pi_cols=glm_pi_cols,
            glm_lambda_cols=glm_lambda_cols,
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
        """Convenience method delegating to `CatBoostQuantileForecaster(...).fit(self)`."""
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
        """Convenience method delegating to `CatBoostQuantileForecaster(...).predict(self, model=model)`."""
        return CatBoostQuantileForecaster(quantiles=model.quantiles, tol=tol).predict(self, model=model)

    def evaluate_quantile_forecasts(
        self,
        pred_df: pd.DataFrame,
        quantiles: tuple[float, ...],
        group_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Delegates to `evaluate_quantile_forecasts`."""
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
        """Delegates single-group CatBoost quantile DDLT calculation to `DDLTCalculator.from_quantiles_group`."""
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
    ts: "DiscreteTimeSeries | pd.DataFrame | None",
    model_data: "pd.DataFrame | CatBoostQuantileModel | ProbabilisticForecaster",
    lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
    lead_time_keys: list[str] | None,
    cycle_time: int,
    service_levels: pd.DataFrame | None = None,
    model_type: str = "glm",
    return_distribution: bool = False,
    pred_first_day: bool = True,
    period: Any = None,
    df: pd.DataFrame | None = None,
    max_service_level: float | None = None,
) -> pd.DataFrame:
    """
    Unified functional dispatcher that computes Demand During Lead Time (DDLT) distributions
    (`return_distribution=True`) or service-level target inventories (`return_distribution=False`).

    What it does
    ------------
    Routes the request to `model_data.compute_ddlt(...)` if `model_data` is already a
    `ProbabilisticForecaster` instance, or wraps raw `model_data` in a `GLMForecaster`
    (`model_type="glm"`) or `CatBoostQuantileForecaster` (`model_type="catboost"`).

    Why `max_service_level` matters
    -------------------------------
    `max_service_level` (e.g. `0.9999` when solving a high-margin newsvendor problem)
    tightens the support truncation tolerance of the daily demand distributions via
    `support_tol_for_service_level(max_service_level, max_days)` so that downstream
    quantiles up to that critical ratio do not land on a truncated upper-tail bin.
    When `return_distribution=False`, the maximum value in the `service_level` column is
    used automatically.

    Where it is used
    ----------------
    Exported at the top-level package (`probabilistic_modeling.compute_lead_time_demand`)
    and used across scripts and tests.
    """
    if isinstance(model_data, ProbabilisticForecaster):
        return model_data.compute_ddlt(
            ts,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            cycle_time=cycle_time,
            service_levels=service_levels,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
            df=df,
            period=period,
            max_service_level=max_service_level,
        )
    if model_type in ("glm", "glm_hurdle", "hurdle_glm"):
        return GLMForecaster(hurdle=(model_type in ("glm_hurdle", "hurdle_glm"))).compute_ddlt(
            ts,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            cycle_time=cycle_time,
            service_levels=service_levels,
            return_distribution=return_distribution,
            pred_first_day=pred_first_day,
            df=df,
            period=period,
            model=model_data,
            max_service_level=max_service_level,
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
            df=df,
            period=period,
            model=model_data,
            max_service_level=max_service_level,
        )
    raise ValueError(
        f"Unsupported model type: '{model_type}'. Supported options are 'glm', 'glm_hurdle', 'catboost'."
    )


def compute_distribution_dict(
    df: pd.DataFrame,
    value: str = "lead_time",
    keys: list[str] | None = None,
    method: str = "actual",
) -> tuple[dict[Any, DiscreteDistribution], list[str]]:
    """
    Builds a dictionary mapping group keys (e.g., `store_id` or `(store_id, dept_id)`)
    to empirical or parametric `DiscreteDistribution` objects estimated from `df[value]`.

    What it does
    ------------
    Groups `df` by `keys` (defaulting to all columns except `value`) and fits
    `DiscreteDistribution.from_data(x, method)` on each group's `value` observations.
    Returns `(distr_dict, keys)`.

    Where it is used
    ----------------
    Used in `scripts/run_m5_pipeline.py` and user workflows to construct store-level
    or supplier-level lead-time distribution dictionaries from historical shipment logs
    before calling `forecaster.compute_ddlt(..., lead_time=distr_dict, lead_time_keys=keys)`.
    """
    if keys is None:
        keys = [col for col in df.columns if col != value]
    distr = df.groupby(keys, observed=True)[value].apply(lambda x: DiscreteDistribution.from_data(x, method)).to_dict()
    return distr, keys
