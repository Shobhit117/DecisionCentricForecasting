"""
Clean, composable probabilistic forecaster abstractions (`GLMForecaster` and
`CatBoostQuantileForecaster`) with a shared interface for training, reusable
multi-time prediction, persistence (`save` / `load` / `load_forecaster`),
evaluation, and DDLT propagation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from probabilistic_modeling.ddlt import DDLTCalculator
from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.distributions.quantile import (
    DEFAULT_QUANTILES,
    pinball_loss,
    quantile_crps,
    quantiles_to_mean,
    validate_quantile_levels,
)

if TYPE_CHECKING:
    from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel
    from probabilistic_modeling.time_series import DiscreteTimeSeries, TimeSeriesFeatureSpec

logger = logging.getLogger(__name__)


def compute_forecasting_metrics(
    pred_df: pd.DataFrame,
    target_cols: list[str],
    pred_columns: list[str],
) -> pd.DataFrame:
    """
    Computes multi-horizon point forecast error columns (``actual``, ``pred``,
    ``under_diff``, ``over_diff``, ``abs_diff``) in-place on ``pred_df``.

    Works both in backtesting (where ``target_cols`` contain realized demand)
    and in production inference (where ``target_cols`` may be ``NaN`` or absent).
    """
    if len(target_cols) != len(pred_columns):
        raise ValueError(
            f"Number of pred columns ({len(pred_columns)}) does not match "
            f"the number of target columns ({len(target_cols)})."
        )

    target_to_pred = dict(zip(sorted(target_cols), sorted(pred_columns)))
    present_targets = [c for c in target_cols if c in pred_df.columns]

    if not present_targets or pred_df[present_targets].notna().sum().sum() == 0:
        # Production mode: no realized future targets available yet
        pred_df["actual"] = np.nan
        pred_df["pred"] = pred_df[pred_columns].sum(axis=1)
        pred_df["under_diff"] = np.nan
        pred_df["over_diff"] = np.nan
        pred_df["abs_diff"] = np.nan
        return pred_df

    tmp_cols = []
    for col_name in target_cols:
        tmp_col = f"tmp_{col_name}"
        pred_df[tmp_col] = np.where(pred_df[col_name].isna(), 0, pred_df[target_to_pred[col_name]])
        tmp_cols.append(tmp_col)

    pred_df["actual"] = pred_df[target_cols].fillna(0).sum(axis=1)
    pred_df["pred"] = pred_df[tmp_cols].sum(axis=1)
    pred_df.drop(columns=tmp_cols, inplace=True)
    pred_df["under_diff"] = np.maximum(pred_df["actual"] - pred_df["pred"], 0)
    pred_df["over_diff"] = np.maximum(pred_df["pred"] - pred_df["actual"], 0)
    pred_df["abs_diff"] = pred_df["under_diff"] + pred_df["over_diff"]
    return pred_df


def evaluate_quantile_forecasts(
    pred_df: pd.DataFrame,
    target_cols: list[str],
    quantiles: tuple[float, ...] | list[float],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Computes pinball loss, empirical coverage P(y <= q_hat), and quantile-approximated
    CRPS per horizon and quantile level.
    """
    from probabilistic_modeling.models.catboost_quantile import quantile_col_name

    levels = tuple(float(q) for q in validate_quantile_levels(quantiles))
    group_cols = [c for c in (group_cols or []) if c in pred_df.columns]
    groups = pred_df.groupby(group_cols, observed=True) if group_cols else [((), pred_df)]
    rows = []
    for group_values, gdf in groups:
        group_values = group_values if isinstance(group_values, tuple) else (group_values,)
        for target in target_cols:
            if target not in gdf.columns:
                continue
            mask = gdf[target].notna()
            y = gdf.loc[mask, target].to_numpy(dtype=float)
            if len(y) == 0:
                continue
            q_cols = [quantile_col_name(target, q) for q in levels]
            q_mat = gdf.loc[mask, q_cols].to_numpy(dtype=float)
            crps_mean = float(np.mean(quantile_crps(y, q_mat, levels)))
            for j, q in enumerate(levels):
                q_hat = q_mat[:, j]
                rows.append({
                    **dict(zip(group_cols, group_values)),
                    "target": target,
                    "quantile": q,
                    "n_obs": len(y),
                    "pinball_loss": float(pinball_loss(y, q_hat, q).mean()),
                    "coverage": float(np.mean(y <= q_hat)),
                    "crps": crps_mean,
                })
    return pd.DataFrame(rows)


def train_single_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str = "nbinom",
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
) -> tuple[dict[str, Any], Any]:
    """
    Trains a single Poisson or Negative Binomial GLM for one group and horizon.

    Separates the dispersion parameter ``alpha`` from regression ``coefficients``
    so ``coefficients`` strictly maps ``Intercept`` and covariates to their weights.
    """
    train_df = df[~df[target_col].isna()]
    if "is_test" in train_df.columns:
        train_df = train_df[train_df["is_test"] == 0]
    formula = f"{target_col} ~ {' + '.join(feature_cols)}"
    if model_type == "nbinom":
        model = smf.negativebinomial(formula=formula, data=train_df)
    elif model_type == "poisson":
        model = smf.poisson(formula=formula, data=train_df)
    else:
        raise ValueError(f"Unsupported distribution: '{model_type}'. Supported options are 'nbinom', 'poisson'.")

    if regularization_alpha is not None and regularization_alpha > 0.0:
        result = model.fit_regularized(alpha=float(regularization_alpha), L1_wt=float(l1_wt), maxiter=100, disp=False)
        summary = None
    else:
        result = model.fit(maxiter=100, disp=False)
        summary = result.summary()

    raw_params = {str(k): float(v) for k, v in result.params.to_dict().items()}
    alpha_val = raw_params.pop("alpha", None)
    prod_model = {
        "model_type": model_type,
        "horizon": target_col,
        "coefficients": raw_params,
        "alpha": float(alpha_val) if alpha_val is not None and np.isfinite(alpha_val) else None,
    }
    return prod_model, summary


@dataclass
class GLMModel:
    """
    Serializable container for grouped, multi-horizon Poisson / Negative-Binomial GLMs.

    Provides the same persistence ergonomics (`.save()`, `.load()`, `.coefficients_table()`)
    as :class:`CatBoostQuantileModel` while remaining 100% compatible with the tabular
    DataFrame representation (`to_dataframe` / `from_dataframe`).
    """
    target_cols: list[str]
    feature_cols: list[str]
    train_groups: list[str] = field(default_factory=list)
    models: dict[tuple[str, ...], dict[str, dict[str, Any]]] = field(default_factory=dict)
    horizon_feature_cols: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def group_key(values: Any) -> tuple[str, ...]:
        if not isinstance(values, tuple):
            values = (values,)
        return tuple(str(v) for v in values)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for group_key, horizon_dict in self.models.items():
            row: dict[str, Any] = {}
            for col_name, key_val in zip(self.train_groups, group_key):
                if col_name == "overdispersion":
                    try:
                        row[col_name] = int(key_val)
                    except ValueError:
                        row[col_name] = key_val
                else:
                    row[col_name] = key_val
            for target in self.target_cols:
                if target in horizon_dict:
                    row[target] = horizon_dict[target]
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
        feature_cols: list[str] | None = None,
    ) -> "GLMModel":
        from probabilistic_modeling.time_series import _parse_serialized_dict

        if target_cols is None:
            target_cols = [c for c in df.columns if c.startswith("target_")]
        train_groups = [c for c in df.columns if c not in target_cols]
        models: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        inferred_features: list[str] = list(feature_cols) if feature_cols else []

        for _, row in df.iterrows():
            g_vals = tuple(str(row[c]) for c in train_groups)
            h_map: dict[str, dict[str, Any]] = {}
            for t in target_cols:
                if t in row and pd.notna(row[t]):
                    m_dict = dict(_parse_serialized_dict(row[t]))
                    coefs = dict(m_dict.get("coefficients", {}))
                    if "alpha" in coefs and m_dict.get("alpha") is None:
                        m_dict["alpha"] = float(coefs.pop("alpha"))
                        m_dict["coefficients"] = coefs
                    h_map[t] = m_dict
                    if not inferred_features:
                        inferred_features = [k for k in coefs.keys() if k != "Intercept"]
            models[g_vals] = h_map
        return cls(
            target_cols=list(target_cols),
            feature_cols=inferred_features,
            train_groups=train_groups,
            models=models,
        )

    def coefficients_table(self) -> pd.DataFrame:
        """Returns a tidy DataFrame of fitted GLM coefficients across groups and horizons."""
        rows = []
        for group_key, horizon_dict in self.models.items():
            group_info = dict(zip(self.train_groups, group_key))
            for target, m_dict in horizon_dict.items():
                model_type = m_dict.get("model_type")
                alpha = m_dict.get("alpha")
                for feat, coef in m_dict.get("coefficients", {}).items():
                    rows.append({
                        **group_info,
                        "target": target,
                        "model_type": model_type,
                        "alpha": alpha,
                        "feature": feat,
                        "coefficient": coef,
                    })
        return pd.DataFrame(rows)

    def save(self, out_dir: Path | str, prefix: str = "glm") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            "train_groups": self.train_groups,
            "horizon_feature_cols": self.horizon_feature_cols,
            "models": [
                {"group_key": list(k), "horizons": v}
                for k, v in self.models.items()
            ],
        }
        with open(out_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "glm") -> "GLMModel":
        out_dir = Path(out_dir)
        with open(out_dir / f"{prefix}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            target_cols=list(meta["target_cols"]),
            feature_cols=list(meta["feature_cols"]),
            train_groups=list(meta["train_groups"]),
            horizon_feature_cols=meta.get("horizon_feature_cols", {}),
        )
        for entry in meta.get("models", []):
            obj.models[tuple(entry["group_key"])] = entry["horizons"]
        return obj


class ProbabilisticForecaster(ABC):
    """
    Abstract base class for multi-horizon probabilistic demand forecasters.

    Supports reusable multi-prediction workflows:
    - Fit once via ``forecaster.fit(ts)``;
    - Save/load the complete forecaster state (weights + feature specification +
      per-series training metadata) via ``forecaster.save(dir)`` and
      ``load_forecaster(dir)``;
    - Predict repeatedly on ``ts``, on specific ``period`` slices, on pre-engineered
      DataFrames ``forecaster.predict(df)``, or on new raw sales data via
      ``forecaster.predict_from_raw(sales_df, ...)`` without needing the original
      training ``DiscreteTimeSeries`` object.
    """

    forecaster_type: str = "base"

    def __init__(self) -> None:
        self.target_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.cat_feature_cols_: list[str] = []
        self.ts_id_cols_: list[str] = []
        self.period_col_: str = "day"
        self.var_col_: str = "demand"
        self.feature_spec_: "TimeSeriesFeatureSpec | None" = None
        self.series_metadata_: pd.DataFrame | None = None

    def _bind_training_context(self, ts: "DiscreteTimeSeries") -> None:
        self.target_cols_ = list(ts.target_cols)
        self.feature_cols_ = list(ts.feature_cols)
        self.cat_feature_cols_ = list(ts.categorical_feature_cols)
        self.ts_id_cols_ = list(ts.ts_id_cols)
        self.period_col_ = ts.period_col
        self.var_col_ = ts.var_col
        self.feature_spec_ = ts.feature_spec
        self.series_metadata_ = ts.series_metadata.copy() if not ts.series_metadata.empty else None

    def _resolve_inference_inputs(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        period: Any = None,
        fallback_target_cols: list[str] | None = None,
        fallback_feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str], list[str], str]:
        """
        Resolves the inference DataFrame, target columns, feature columns, and period column
        whether the caller passed a ``DiscreteTimeSeries``, a standalone ``pd.DataFrame``, or both.
        Also attaches training ``series_metadata_`` (``overdispersion``, ``best_fit``, etc.)
        if missing from a standalone inference DataFrame.
        """
        if isinstance(ts, pd.DataFrame):
            if df is None:
                df = ts
            ts = None

        target_cols = (
            list(ts.target_cols)
            if ts is not None
            else (list(self.target_cols_) if self.target_cols_ else list(fallback_target_cols or []))
        )
        feature_cols = (
            list(ts.feature_cols)
            if ts is not None
            else (list(self.feature_cols_) if self.feature_cols_ else list(fallback_feature_cols or []))
        )
        period_col = ts.period_col if ts is not None else self.period_col_

        if df is not None:
            test_data = df.copy()
            if period is not None and period_col in test_data.columns:
                if isinstance(period, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
                    test_data = test_data[test_data[period_col].isin(list(period))].copy()
                else:
                    test_data = test_data[test_data[period_col] == period].copy()
        elif ts is not None:
            test_data = ts.get_inference_data(period=period)
        else:
            raise ValueError("Provide either a 'DiscreteTimeSeries' or an inference DataFrame 'df'.")

        # Attach training series metadata (overdispersion, best_fit, distr_param_*) if missing:
        meta = self.series_metadata_
        if meta is None and ts is not None and not ts.series_metadata.empty:
            meta = ts.series_metadata
        if meta is not None and not meta.empty:
            join_keys = [c for c in self.ts_id_cols_ if c in meta.columns and c in test_data.columns]
            if not join_keys and ts is not None:
                join_keys = [c for c in ts.ts_id_cols if c in meta.columns and c in test_data.columns]
            missing_meta_cols = [c for c in meta.columns if c not in join_keys and c not in test_data.columns]
            if join_keys and missing_meta_cols:
                meta_sub = meta[join_keys + missing_meta_cols].drop_duplicates(subset=join_keys).copy()
                for k in join_keys:
                    meta_sub[k] = meta_sub[k].astype(test_data[k].dtype)
                test_data = pd.merge(test_data, meta_sub, on=join_keys, how="left")
                if "overdispersion" in missing_meta_cols:
                    test_data["overdispersion"] = test_data["overdispersion"].fillna(0).astype(int)

        return test_data, target_cols, feature_cols, period_col

    def transform(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
        external_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        period: Any = None,
        num_test_days: int = 0,
    ) -> pd.DataFrame:
        """
        Transforms raw ``sales_df`` and external feature tables into an inference-ready
        feature DataFrame using the ``feature_spec_`` and ``series_metadata_`` captured
        during training.
        """
        if self.feature_spec_ is None:
            raise ValueError("No 'feature_spec_' bound to this forecaster. Fit or load a forecaster first.")
        return self.feature_spec_.transform(
            sales_df=sales_df,
            series_metadata=self.series_metadata_,
            calendar_df=calendar_df,
            external_features=external_features,
            period=period,
            num_test_days=num_test_days,
        )

    def predict_from_raw(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
        external_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        period: Any = None,
    ) -> pd.DataFrame:
        """
        End-to-end reusable inference from raw ``sales_df`` + covariates without
        requiring the original training ``DiscreteTimeSeries``.
        """
        inf_df = self.transform(
            sales_df=sales_df,
            calendar_df=calendar_df,
            external_features=external_features,
            period=period,
            num_test_days=0,
        )
        return self.predict(df=inf_df)

    def _save_base_context(self, out_dir: Path, extra_meta: dict[str, Any] | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "forecaster_type": self.forecaster_type,
            "target_cols": self.target_cols_,
            "feature_cols": self.feature_cols_,
            "cat_feature_cols": self.cat_feature_cols_,
            "ts_id_cols": self.ts_id_cols_,
            "period_col": self.period_col_,
            "var_col": self.var_col_,
            **(extra_meta or {}),
        }
        with open(out_dir / "forecaster_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.feature_spec_ is not None:
            self.feature_spec_.save(out_dir / "feature_spec.json")
        if self.series_metadata_ is not None and not self.series_metadata_.empty:
            self.series_metadata_.to_csv(out_dir / "series_metadata.csv", index=False)

    def _load_base_context(self, out_dir: Path, meta: dict[str, Any]) -> None:
        from probabilistic_modeling.time_series import TimeSeriesFeatureSpec

        self.target_cols_ = list(meta.get("target_cols", []))
        self.feature_cols_ = list(meta.get("feature_cols", []))
        self.cat_feature_cols_ = list(meta.get("cat_feature_cols", []))
        self.ts_id_cols_ = list(meta.get("ts_id_cols", []))
        self.period_col_ = str(meta.get("period_col", "day"))
        self.var_col_ = str(meta.get("var_col", "demand"))
        spec_path = out_dir / "feature_spec.json"
        if spec_path.exists():
            self.feature_spec_ = TimeSeriesFeatureSpec.load(spec_path)
        series_meta_path = out_dir / "series_metadata.csv"
        if series_meta_path.exists():
            self.series_metadata_ = pd.read_csv(series_meta_path)

    @abstractmethod
    def fit(self, ts: "DiscreteTimeSeries") -> Any:
        ...

    @abstractmethod
    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
    ) -> pd.DataFrame:
        """
        Predicts every forecast horizon. ``model`` optionally overrides the fitted model
        container (``GLMModel`` / coefficient table for GLMs, ``CatBoostQuantileModel`` for CatBoost).
        """
        ...

    @abstractmethod
    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
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
        """Shared signature for all forecasters (``model`` has the same meaning as in ``predict``)."""
        ...

    @abstractmethod
    def save(self, out_dir: Path | str) -> Path:
        ...

    @classmethod
    @abstractmethod
    def load(cls, out_dir: Path | str) -> "ProbabilisticForecaster":
        ...


class GLMForecaster(ProbabilisticForecaster):
    """
    Grouped Poisson / Negative-Binomial GLM forecaster across multiple horizons.
    """

    forecaster_type = "glm"

    def __init__(
        self,
        train_groups: tuple[str, ...] | list[str] | None = None,
        regularization_alpha: float | None = None,
        l1_wt: float = 0.0,
        restrict_future_leads: bool = False,
    ):
        super().__init__()
        self.train_groups = list(train_groups) if train_groups is not None else []
        self.regularization_alpha = regularization_alpha
        self.l1_wt = l1_wt
        self.restrict_future_leads = restrict_future_leads
        self.model_data: pd.DataFrame | None = None
        self.model: GLMModel | None = None

    def fit(self, ts: "DiscreteTimeSeries") -> pd.DataFrame:
        t0 = time.perf_counter()
        self._bind_training_context(ts)
        partition_cols = list(self.train_groups)
        if "overdispersion" not in partition_cols:
            partition_cols.append("overdispersion")

        train_df = ts.df[ts.df["is_test"] == 0] if ts.test_start_period is not None else ts.df
        n_groups = train_df[partition_cols].drop_duplicates().shape[0]
        logger.info(
            "Starting GLM training across %d group(s) and %d horizon(s) (%d training rows)...",
            n_groups,
            len(ts.target_cols),
            len(train_df),
        )

        horizon_feature_cols = {
            target: ts.get_features_for_horizon(target, restrict_future_leads=self.restrict_future_leads)
            for target in ts.target_cols
        }

        def _train_group(group_df: pd.DataFrame) -> pd.Series:
            group_name = getattr(group_df, "name", "unknown")
            if isinstance(group_name, tuple):
                idx = partition_cols.index("overdispersion")
                is_overdispersed = group_name[idx]
            else:
                is_overdispersed = group_name

            model_type = "nbinom" if is_overdispersed == 1 else "poisson"
            group_result = {}
            n_horizons = len(ts.target_cols)
            for h_idx, target in enumerate(ts.target_cols, start=1):
                h_t0 = time.perf_counter()
                h_features = horizon_feature_cols[target]
                logger.info(
                    "Training %s GLM for group %s, horizon [%d/%d] (%s)...",
                    model_type,
                    group_name,
                    h_idx,
                    n_horizons,
                    target,
                )
                model_dict, summary = train_single_glm(
                    group_df,
                    h_features,
                    target,
                    model_type,
                    regularization_alpha=self.regularization_alpha,
                    l1_wt=self.l1_wt,
                )
                group_result[target] = model_dict
                if summary is not None:
                    logger.debug("%s", summary)
                logger.info(
                    "Completed %s GLM for group %s, %s in %.2fs",
                    model_type,
                    group_name,
                    target,
                    time.perf_counter() - h_t0,
                )
            return pd.Series(group_result)

        try:
            result_df = train_df.groupby(partition_cols, observed=True).apply(_train_group, include_groups=False).reset_index()
        except TypeError:
            result_df = train_df.groupby(partition_cols, observed=True).apply(_train_group).reset_index()
        self.model_data = result_df
        self.model = GLMModel.from_dataframe(result_df, target_cols=list(ts.target_cols), feature_cols=list(ts.feature_cols))
        self.model.horizon_feature_cols = horizon_feature_cols
        logger.info("Finished GLM training in %.2fs", time.perf_counter() - t0)
        return result_df

    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: pd.DataFrame | GLMModel | None = None,
        period: Any = None,
        model_data: pd.DataFrame | GLMModel | None = None,
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        if model is None:
            model = model_data  # backward-compatible alias
        model_data = model
        if isinstance(model_data, GLMModel):
            glm_container = model_data
            model_df = glm_container.to_dataframe()
        elif isinstance(model_data, pd.DataFrame):
            glm_container = GLMModel.from_dataframe(model_data)
            model_df = model_data
        elif self.model is not None:
            glm_container = self.model
            model_df = self.model_data if self.model_data is not None else glm_container.to_dataframe()
        elif self.model_data is not None:
            glm_container = GLMModel.from_dataframe(self.model_data)
            model_df = self.model_data
        else:
            raise ValueError("GLMForecaster has not been fitted yet.")

        test_data, target_cols, feature_cols, _ = self._resolve_inference_inputs(
            ts=ts,
            df=df,
            period=period,
            fallback_target_cols=glm_container.target_cols,
            fallback_feature_cols=glm_container.feature_cols,
        )
        train_groups = [col for col in model_df.columns if col not in target_cols]
        logger.info("Generating GLM predictions for %d rows across %d horizons...", len(test_data), len(target_cols))

        for target in target_cols:
            test_data[f"{target}_mu"] = np.nan
            test_data[f"{target}_alpha"] = np.nan

        model_dict_map = model_df.set_index(train_groups).to_dict(orient="index")
        # Also prepare string-keyed fallback map so loaded JSON models match categorical/int columns seamlessly:
        str_model_dict_map = {
            (tuple(str(x) for x in k) if isinstance(k, tuple) else str(k)): v
            for k, v in model_dict_map.items()
        }

        present_features = [c for c in feature_cols if c in test_data.columns]
        for group_keys, group_df in test_data.groupby(train_groups, observed=True):
            lookup_key = group_keys[0] if isinstance(group_keys, tuple) and len(group_keys) == 1 else group_keys
            row_models = model_dict_map.get(lookup_key)
            if row_models is None:
                str_key = tuple(str(x) for x in group_keys) if isinstance(group_keys, tuple) else str(group_keys)
                if isinstance(str_key, tuple) and len(str_key) == 1:
                    str_key = str_key[0]
                row_models = str_model_dict_map.get(str_key)
            if row_models is None:
                logger.warning("No trained GLM found for group %s; skipping %d rows", lookup_key, len(group_df))
                continue

            group_idx = group_df.index
            X = group_df[present_features].copy()
            X["Intercept"] = 1.0
            for target in target_cols:
                model_dict = row_models.get(target)
                if not isinstance(model_dict, dict) or "coefficients" not in model_dict:
                    continue
                coefs = dict(model_dict["coefficients"])
                alpha_val = model_dict.get("alpha", coefs.pop("alpha", np.nan))
                coef_series = pd.Series(coefs, dtype=float)
                X_aligned = X.reindex(columns=coef_series.index).fillna(0.0)
                z = X_aligned.dot(coef_series)
                test_data.loc[group_idx, f"{target}_mu"] = np.exp(z)
                test_data.loc[group_idx, f"{target}_alpha"] = float(alpha_val) if alpha_val is not None else np.nan

        compute_forecasting_metrics(test_data, target_cols, [f"{target}_mu" for target in target_cols])
        logger.info("GLM predictions completed in %.2fs", time.perf_counter() - t0)
        return test_data

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: pd.DataFrame | GLMModel | None = None,
        period: Any = None,
        max_service_level: float | None = None,
        model_data: pd.DataFrame | GLMModel | None = None,
    ) -> pd.DataFrame:
        if lead_time is None:
            raise ValueError("'lead_time' distribution must be provided.")
        predictions = self.predict(ts, df=df, model=model if model is not None else model_data, period=period)
        _, target_cols, _, period_col = self._resolve_inference_inputs(
            ts=ts,
            df=predictions,
            fallback_target_cols=self.model.target_cols if self.model is not None else None,
        )
        return _apply_ddlt_across_groups(
            predictions=predictions,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            service_levels=service_levels,
            return_distribution=return_distribution,
            group_fn=lambda gdf, lt_distr: DDLTCalculator(
                lt_distr, cycle_time, max_service_level=max_service_level
            ).from_glm_group(
                gdf,
                lt_distr,
                target_cols,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=period_col,
            ),
        )

    def save(self, out_dir: Path | str, prefix: str = "glm") -> Path:
        if self.model is None and self.model_data is not None:
            self.model = GLMModel.from_dataframe(self.model_data, self.target_cols_, self.feature_cols_)
        if self.model is None:
            raise ValueError("GLMForecaster has not been fitted yet.")
        out_path = Path(out_dir)
        self._save_base_context(
            out_path,
            extra_meta={
                "train_groups": self.train_groups,
                "regularization_alpha": self.regularization_alpha,
                "l1_wt": self.l1_wt,
                "restrict_future_leads": self.restrict_future_leads,
                "prefix": prefix,
            },
        )
        self.model.save(out_path, prefix=prefix)
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "glm") -> "GLMForecaster":
        out_path = Path(out_dir)
        meta_file = out_path / "forecaster_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        saved_prefix = meta.get("prefix", prefix)
        glm_model = GLMModel.load(out_path, prefix=saved_prefix)
        forecaster = cls(
            train_groups=meta.get("train_groups", [c for c in glm_model.train_groups if c != "overdispersion"]),
            regularization_alpha=meta.get("regularization_alpha"),
            l1_wt=float(meta.get("l1_wt", 0.0)),
            restrict_future_leads=bool(meta.get("restrict_future_leads", False)),
        )
        forecaster.model = glm_model
        forecaster.model_data = glm_model.to_dataframe()
        if meta:
            forecaster._load_base_context(out_path, meta)
        else:
            forecaster.target_cols_ = list(glm_model.target_cols)
            forecaster.feature_cols_ = list(glm_model.feature_cols)
        return forecaster


class CatBoostQuantileForecaster(ProbabilisticForecaster):
    """
    Multi-horizon CatBoost ``MultiQuantile`` forecaster with Numba-accelerated
    quantile-to-distribution reconstruction and DDLT calculation.
    """

    forecaster_type = "catboost"

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
        train_groups: tuple[str, ...] | list[str] | None = None,
        extra_cat_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id"),
        num_valid_days: int = 28,
        max_train_days: int | None = None,
        catboost_params: dict[str, Any] | None = None,
        tol: float = 1e-4,
        restrict_future_leads: bool = False,
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in validate_quantile_levels(quantiles))
        self.train_groups = list(train_groups) if train_groups else []
        self.extra_cat_cols = tuple(extra_cat_cols)
        self.num_valid_days = num_valid_days
        self.max_train_days = max_train_days
        self.catboost_params = catboost_params
        self.tol = tol
        self.restrict_future_leads = restrict_future_leads
        self.model: "CatBoostQuantileModel | None" = None

    def fit(self, ts: "DiscreteTimeSeries") -> "CatBoostQuantileModel":
        from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel, train_catboost_quantile

        t0 = time.perf_counter()
        self._bind_training_context(ts)
        partition_cols = list(self.train_groups)
        available_extra_cats = [
            c for c in self.extra_cat_cols if c in ts.df.columns and c not in partition_cols
        ]
        cat_cols = list(dict.fromkeys(list(ts.categorical_feature_cols) + available_extra_cats))
        feature_cols = [c for c in ts.feature_cols if c not in cat_cols]
        self.feature_cols_ = list(feature_cols)
        self.cat_feature_cols_ = list(cat_cols)

        horizon_feature_cols = {}
        horizon_cat_feature_cols = {}
        for target in ts.target_cols:
            allowed_num = set(ts.get_features_for_horizon(target, restrict_future_leads=self.restrict_future_leads))
            horizon_feature_cols[target] = [c for c in feature_cols if c in allowed_num]
            if self.restrict_future_leads and ts.future_lead_horizon_map:
                h_offset = ts._target_offset(target)
                horizon_cat_feature_cols[target] = [
                    c for c in cat_cols if ts.future_lead_horizon_map.get(c, 0) <= h_offset
                ]
            else:
                horizon_cat_feature_cols[target] = list(cat_cols)

        result = CatBoostQuantileModel(
            self.quantiles,
            list(ts.target_cols),
            feature_cols,
            cat_cols,
            partition_cols,
            horizon_feature_cols=horizon_feature_cols,
            horizon_cat_feature_cols=horizon_cat_feature_cols,
        )

        test_start_pos = (
            len(ts.periods) if ts.test_start_period is None else ts.periods.index(ts.test_start_period)
        )
        period_pos = ts._period_positions(ts.df)
        mask = ts.df["is_test"] == 0
        if self.max_train_days is not None:
            mask &= period_pos >= test_start_pos - self.num_valid_days - self.max_train_days
        train_df = ts.df[mask]
        train_pos = period_pos[mask]

        groups = list(train_df.groupby(partition_cols, observed=True)) if partition_cols else [((), train_df)]
        logger.info(
            "Starting CatBoost MultiQuantile training (%d quantiles, %d group(s), %d horizon(s), %d num features, %d cat features)...",
            len(self.quantiles),
            len(groups),
            len(ts.target_cols),
            len(feature_cols),
            len(cat_cols),
        )

        n_horizons = len(ts.target_cols)
        for group_values, group_df in groups:
            key = CatBoostQuantileModel.group_key(group_values) if partition_cols else ()
            group_pos = train_pos.loc[group_df.index]
            for h_idx, target in enumerate(ts.target_cols, start=1):
                h_t0 = time.perf_counter()
                target_pos = group_pos + ts._target_offset(target)
                has_target = group_df[target].notna() & (target_pos < test_start_pos)
                valid_mask = has_target & (target_pos >= test_start_pos - self.num_valid_days)
                fit_mask = has_target & ~valid_mask
                h_num_cols, h_cat_cols = result.features_for_target(target)
                logger.info(
                    "Training CatBoost model for group %s, horizon [%d/%d] (%s): %d train rows, %d validation rows...",
                    key or "(all)",
                    h_idx,
                    n_horizons,
                    target,
                    int(fit_mask.sum()),
                    int(valid_mask.sum()),
                )
                cb_model = train_catboost_quantile(
                    group_df[fit_mask],
                    h_num_cols,
                    h_cat_cols,
                    target,
                    self.quantiles,
                    eval_df=group_df[valid_mask] if self.num_valid_days > 0 else None,
                    params=self.catboost_params,
                )
                result.models.setdefault(key, {})[target] = cb_model
                logger.info(
                    "Completed CatBoost model for group %s, %s in %.2fs (best iteration: %s)",
                    key or "(all)",
                    target,
                    time.perf_counter() - h_t0,
                    cb_model.get_best_iteration(),
                )
        self.model = result
        logger.info("Finished all CatBoost horizon models in %.2fs", time.perf_counter() - t0)
        return result

    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: "CatBoostQuantileModel | None" = None,
        period: Any = None,
    ) -> pd.DataFrame:
        from probabilistic_modeling.models.catboost_quantile import predict_catboost_quantile

        t0 = time.perf_counter()
        cb_container = model if model is not None else self.model
        if cb_container is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")

        test_data, target_cols, _, _ = self._resolve_inference_inputs(
            ts=ts,
            df=df,
            period=period,
            fallback_target_cols=cb_container.target_cols,
            fallback_feature_cols=cb_container.feature_cols,
        )
        levels = np.asarray(cb_container.quantiles, dtype=np.float64)
        logger.info(
            "Generating CatBoost quantile predictions for %d rows across %d horizons (%d quantiles)...",
            len(test_data),
            len(cb_container.target_cols),
            len(levels),
        )
        preds = {target: np.full((len(test_data), len(levels)), np.nan) for target in cb_container.target_cols}

        if cb_container.train_groups:
            groups = test_data.groupby(cb_container.train_groups, observed=True).indices
        else:
            groups = {(): np.arange(len(test_data))}

        for group_values, idx in groups.items():
            key = cb_container.group_key(group_values) if cb_container.train_groups else ()
            group_models = cb_container.models.get(key)
            if group_models is None:
                logger.warning("No CatBoost model available for group %s; skipping %d rows", key, len(idx))
                continue
            group_df = test_data.iloc[idx]
            for target, cb_model in group_models.items():
                h_num_cols, h_cat_cols = cb_container.features_for_target(target)
                preds[target][idx] = predict_catboost_quantile(
                    cb_model, group_df, h_num_cols, h_cat_cols
                )

        new_cols = {}
        for target, arr in preds.items():
            for j, col in enumerate(cb_container.quantile_cols(target)):
                new_cols[col] = arr[:, j]
            mu = np.full(len(arr), np.nan)
            valid = ~np.isnan(arr).any(axis=1)
            if np.any(valid):
                mu[valid] = quantiles_to_mean(levels, arr[valid], self.tol)
            new_cols[f"{target}_mu"] = mu
        test_data = pd.concat([test_data, pd.DataFrame(new_cols, index=test_data.index)], axis=1)

        compute_forecasting_metrics(test_data, target_cols, [f"{target}_mu" for target in cb_container.target_cols])
        logger.info("CatBoost predictions and distribution means completed in %.2fs", time.perf_counter() - t0)
        return test_data

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: "CatBoostQuantileModel | None" = None,
        period: Any = None,
        max_service_level: float | None = None,
    ) -> pd.DataFrame:
        if lead_time is None:
            raise ValueError("'lead_time' distribution must be provided.")
        cb_container = model if model is not None else self.model
        if cb_container is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")

        predictions = self.predict(ts, df=df, model=cb_container, period=period)
        _, target_cols, _, period_col = self._resolve_inference_inputs(
            ts=ts,
            df=predictions,
            fallback_target_cols=cb_container.target_cols,
        )
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
                cb_container.quantiles,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=period_col,
            ),
        )

    def save(self, out_dir: Path | str, prefix: str = "catboost") -> Path:
        if self.model is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")
        out_path = Path(out_dir)
        self._save_base_context(
            out_path,
            extra_meta={
                "quantiles": list(self.quantiles),
                "train_groups": self.train_groups,
                "extra_cat_cols": list(self.extra_cat_cols),
                "num_valid_days": self.num_valid_days,
                "max_train_days": self.max_train_days,
                "catboost_params": self.catboost_params,
                "tol": self.tol,
                "restrict_future_leads": self.restrict_future_leads,
                "prefix": prefix,
            },
        )
        self.model.save(out_path, prefix=prefix)
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "catboost") -> "CatBoostQuantileForecaster":
        from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel

        out_path = Path(out_dir)
        meta_file = out_path / "forecaster_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        saved_prefix = meta.get("prefix", prefix)
        cb_model = CatBoostQuantileModel.load(out_path, prefix=saved_prefix)
        forecaster = cls(
            quantiles=cb_model.quantiles,
            train_groups=cb_model.train_groups,
            extra_cat_cols=tuple(meta.get("extra_cat_cols", ("item_id", "dept_id", "cat_id"))),
            num_valid_days=int(meta.get("num_valid_days", 28)),
            max_train_days=meta.get("max_train_days"),
            catboost_params=meta.get("catboost_params"),
            tol=float(meta.get("tol", 1e-4)),
            restrict_future_leads=bool(meta.get("restrict_future_leads", False)),
        )
        forecaster.model = cb_model
        if meta:
            forecaster._load_base_context(out_path, meta)
        else:
            forecaster.target_cols_ = list(cb_model.target_cols)
            forecaster.feature_cols_ = list(cb_model.feature_cols)
            forecaster.cat_feature_cols_ = list(cb_model.cat_feature_cols)
        return forecaster


def load_forecaster(out_dir: Path | str) -> ProbabilisticForecaster:
    """
    Universal loader that inspects ``out_dir`` and reconstructs the appropriate
    fitted :class:`ProbabilisticForecaster` (``GLMForecaster`` or ``CatBoostQuantileForecaster``),
    complete with its feature specification and series metadata.
    """
    out_path = Path(out_dir)
    meta_file = out_path / "forecaster_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        f_type = meta.get("forecaster_type")
        if f_type == "empirical":
            from probabilistic_modeling.models.baseline import EmpiricalQuantileForecaster

            return EmpiricalQuantileForecaster.load(out_path)
        if f_type == "glm":
            return GLMForecaster.load(out_path)
        if f_type == "catboost":
            return CatBoostQuantileForecaster.load(out_path)
        raise ValueError(f"Unknown forecaster_type '{f_type}' in {meta_file}.")
    if (out_path / "catboost_meta.json").exists():
        return CatBoostQuantileForecaster.load(out_path)
    if (out_path / "glm_meta.json").exists():
        return GLMForecaster.load(out_path)
    raise FileNotFoundError(f"No recognized forecaster bundle found in {out_path}.")


def _apply_ddlt_across_groups(
    predictions: pd.DataFrame,
    lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
    lead_time_keys: list[str] | None,
    service_levels: pd.DataFrame | None,
    return_distribution: bool,
    group_fn,
) -> pd.DataFrame:
    if not return_distribution:
        if service_levels is not None:
            join_cols = [col for col in service_levels.columns if col != "service_level"]
            predictions = pd.merge(predictions, service_levels, on=join_cols, how="left").fillna(
                {"service_level": 0.95}
            )
        elif "service_level" not in predictions.columns:
            predictions["service_level"] = 0.95

    if isinstance(lead_time, DiscreteDistribution) or not lead_time_keys:
        lt_distr = lead_time if isinstance(lead_time, DiscreteDistribution) else next(iter(lead_time.values()))
        return group_fn(predictions, lt_distr)

    cols = predictions.columns.to_list()
    return predictions.groupby(lead_time_keys, as_index=False, observed=True)[cols].apply(
        lambda x: group_fn(x, lead_time[x.name])
    )
