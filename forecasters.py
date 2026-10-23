"""
Clean, composable probabilistic forecaster abstractions (`GLMForecaster` and
`CatBoostQuantileForecaster`) with a shared interface for training, prediction,
evaluation, and DDLT propagation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ddlt import DDLTCalculator
from discrete_distribution import DiscreteDistribution
from quantile_distribution import (
    DEFAULT_QUANTILES,
    pinball_loss,
    quantiles_to_mean,
    validate_quantile_levels,
)

if TYPE_CHECKING:
    from catboost_quantile import CatBoostQuantileModel
    from time_series import DiscreteTimeSeries

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
    """Computes pinball loss and empirical coverage P(y <= q_hat) per horizon and quantile level."""
    from catboost_quantile import quantile_col_name

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
            for q in quantiles:
                q_hat = gdf.loc[mask, quantile_col_name(target, q)].to_numpy(dtype=float)
                rows.append({
                    **dict(zip(group_cols, group_values)),
                    "target": target,
                    "quantile": q,
                    "n_obs": len(y),
                    "pinball_loss": pinball_loss(y, q_hat, q).mean(),
                    "coverage": np.mean(y <= q_hat),
                })
    return pd.DataFrame(rows)


def train_single_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str = "nbinom",
) -> tuple[dict[str, Any], Any]:
    """Trains a single Poisson or Negative Binomial GLM for one group and horizon."""
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
    result = model.fit(maxiter=100, disp=False)
    prod_model = {
        "model_type": model_type,
        "horizon": target_col,
        "coefficients": result.params.to_dict(),
    }
    return prod_model, result.summary()


class ProbabilisticForecaster(ABC):
    """Abstract base class for multi-horizon probabilistic demand forecasters."""

    @abstractmethod
    def fit(self, ts: "DiscreteTimeSeries") -> Any:
        ...

    @abstractmethod
    def predict(self, ts: "DiscreteTimeSeries", df: pd.DataFrame | None = None) -> pd.DataFrame:
        ...

    @abstractmethod
    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries",
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...


class GLMForecaster(ProbabilisticForecaster):
    """
    Grouped Poisson / Negative-Binomial GLM forecaster across multiple horizons.
    """

    def __init__(self, train_groups: tuple[str, ...] | list[str] | None = None):
        self.train_groups = list(train_groups) if train_groups is not None else []
        self.model_data: pd.DataFrame | None = None

    def fit(self, ts: "DiscreteTimeSeries") -> pd.DataFrame:
        t0 = time.perf_counter()
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
                logger.info(
                    "Training %s GLM for group %s, horizon [%d/%d] (%s)...",
                    model_type,
                    group_name,
                    h_idx,
                    n_horizons,
                    target,
                )
                model_dict, summary = train_single_glm(group_df, ts.feature_cols, target, model_type)
                group_result[target] = model_dict
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
        logger.info("Finished GLM training in %.2fs", time.perf_counter() - t0)
        return result_df

    def predict(
        self,
        ts: "DiscreteTimeSeries",
        df: pd.DataFrame | None = None,
        model_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        model_df = model_data if model_data is not None else self.model_data
        if model_df is None:
            raise ValueError("GLMForecaster has not been fitted yet.")

        test_data = df.copy() if df is not None else ts.get_inference_data()
        train_groups = [col for col in model_df.columns if col not in ts.target_cols]
        logger.info("Generating GLM predictions for %d rows across %d horizons...", len(test_data), len(ts.target_cols))

        for target in ts.target_cols:
            test_data[f"{target}_mu"] = np.nan
            test_data[f"{target}_alpha"] = np.nan

        model_dict_map = model_df.set_index(train_groups).to_dict(orient="index")
        for group_keys, group_df in test_data.groupby(train_groups, observed=True):
            if isinstance(group_keys, tuple) and len(group_keys) == 1:
                group_keys = group_keys[0]
            if group_keys not in model_dict_map:
                logger.warning("No trained GLM found for group %s; skipping %d rows", group_keys, len(group_df))
                continue
            row_models = model_dict_map[group_keys]
            group_idx = group_df.index
            X = group_df[ts.feature_cols].copy()
            X["Intercept"] = 1.0
            for target in ts.target_cols:
                model_dict = row_models.get(target)
                if not isinstance(model_dict, dict) or "coefficients" not in model_dict:
                    continue
                coef_series = pd.Series(model_dict["coefficients"])
                X_aligned = X.reindex(columns=coef_series.index).fillna(0.0)
                z = X_aligned.dot(coef_series)
                test_data.loc[group_idx, f"{target}_mu"] = np.exp(z)
                test_data.loc[group_idx, f"{target}_alpha"] = model_dict.get(
                    "alpha", coef_series.get("alpha", np.nan)
                )

        compute_forecasting_metrics(test_data, ts.target_cols, [f"{target}_mu" for target in ts.target_cols])
        logger.info("GLM predictions completed in %.2fs", time.perf_counter() - t0)
        return test_data

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries",
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        predictions = self.predict(ts, df=df, model_data=model_data)
        return _apply_ddlt_across_groups(
            predictions=predictions,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            service_levels=service_levels,
            return_distribution=return_distribution,
            group_fn=lambda gdf, lt_distr: DDLTCalculator(lt_distr, cycle_time).from_glm_group(
                gdf,
                lt_distr,
                ts.target_cols,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=ts.period_col,
            ),
        )


class CatBoostQuantileForecaster(ProbabilisticForecaster):
    """
    Multi-horizon CatBoost ``MultiQuantile`` forecaster with Numba-accelerated
    quantile-to-distribution reconstruction and DDLT calculation.
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
        train_groups: tuple[str, ...] | list[str] | None = None,
        extra_cat_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id"),
        num_valid_days: int = 28,
        max_train_days: int | None = None,
        catboost_params: dict[str, Any] | None = None,
        tol: float = 1e-4,
    ):
        self.quantiles = tuple(float(q) for q in validate_quantile_levels(quantiles))
        self.train_groups = list(train_groups) if train_groups else []
        self.extra_cat_cols = tuple(extra_cat_cols)
        self.num_valid_days = num_valid_days
        self.max_train_days = max_train_days
        self.catboost_params = catboost_params
        self.tol = tol
        self.model: "CatBoostQuantileModel | None" = None

    def fit(self, ts: "DiscreteTimeSeries") -> "CatBoostQuantileModel":
        from catboost_quantile import CatBoostQuantileModel, train_catboost_quantile

        t0 = time.perf_counter()
        partition_cols = list(self.train_groups)
        available_extra_cats = [
            c for c in self.extra_cat_cols if c in ts.df.columns and c not in partition_cols
        ]
        cat_cols = list(dict.fromkeys(list(ts.categorical_feature_cols) + available_extra_cats))
        feature_cols = [c for c in ts.feature_cols if c not in cat_cols]
        result = CatBoostQuantileModel(self.quantiles, list(ts.target_cols), feature_cols, cat_cols, partition_cols)

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
                    feature_cols,
                    cat_cols,
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
        ts: "DiscreteTimeSeries",
        df: pd.DataFrame | None = None,
        model: "CatBoostQuantileModel | None" = None,
    ) -> pd.DataFrame:
        from catboost_quantile import predict_catboost_quantile

        t0 = time.perf_counter()
        cb_container = model if model is not None else self.model
        if cb_container is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")

        test_data = df.copy() if df is not None else ts.get_inference_data()
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
                preds[target][idx] = predict_catboost_quantile(
                    cb_model, group_df, cb_container.feature_cols, cb_container.cat_feature_cols
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

        compute_forecasting_metrics(test_data, ts.target_cols, [f"{target}_mu" for target in ts.target_cols])
        logger.info("CatBoost predictions and distribution means completed in %.2fs", time.perf_counter() - t0)
        return test_data

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries",
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: "CatBoostQuantileModel | None" = None,
    ) -> pd.DataFrame:
        cb_container = model if model is not None else self.model
        if cb_container is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")

        predictions = self.predict(ts, df=df, model=cb_container)
        return _apply_ddlt_across_groups(
            predictions=predictions,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            service_levels=service_levels,
            return_distribution=return_distribution,
            group_fn=lambda gdf, lt_distr: DDLTCalculator(lt_distr, cycle_time, tol=self.tol).from_quantiles_group(
                gdf,
                lt_distr,
                ts.target_cols,
                cb_container.quantiles,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=ts.period_col,
            ),
        )


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
