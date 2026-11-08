"""
Unified probabilistic forecaster abstractions (`models/forecasters.py`).

What this module does
---------------------
Provides a shared, stateful object-oriented interface (`ProbabilisticForecaster`) and
concrete implementations (`GLMForecaster` and `CatBoostQuantileForecaster`) for:
1. Multi-horizon model training (`fit`),
2. Reusable multi-time prediction on engineered panels, specific period slices, or raw
   sales + covariate tables (`predict`, `transform`, `predict_from_raw`),
3. Complete bundle persistence (`save`, `load`, and universal `load_forecaster`),
4. Multi-horizon point & quantile forecast evaluation (`compute_forecasting_metrics`,
   `evaluate_quantile_forecasts`), and
5. Compound Demand During Lead Time propagation (`compute_ddlt` via `DDLTCalculator`).

Where it is used
----------------
- Primary model training, scoring, and DDLT interface used by `scripts/run_m5_pipeline.py`,
  `DiscreteTimeSeries` delegation methods, and `EmpiricalQuantileForecaster` in `models/baseline.py`.
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
    Computes multi-horizon cumulative point forecast error columns (`actual`, `pred`,
    `under_diff`, `over_diff`, `abs_diff`) in-place on `pred_df`.

    What it does
    ------------
    - Pairs each horizon target column (`target_t_plus_h`) with its corresponding expected-value
      prediction column (`target_t_plus_h_mu`).
    - In **production mode** (when `target_cols` are absent or all `NaN`), sets `pred` to the
      sum of predicted horizon means across the forecast window and sets `actual`, `under_diff`,
      `over_diff`, and `abs_diff` to `NaN`.
    - In **backtesting mode**, sums realized demand across all non-missing horizons on each row
      (`actual`), sums the corresponding predicted means over those exact same non-missing
      horizons (`pred`, masking out any horizon where `target_t_plus_h` is `NaN` near the end
      of the panel), and computes:
      - `under_diff = max(actual - pred, 0)` (under-forecasting error),
      - `over_diff = max(pred - actual, 0)` (over-forecasting error),
      - `abs_diff = under_diff + over_diff = |actual - pred|`.

    Why it is needed
    ----------------
    Near the end of a backtest panel, row `t` may have realized observations for `target_t_plus_0`
    and `target_t_plus_1`, while `target_t_plus_2 .. target_t_plus_6` lie beyond the end of the
    dataset (`NaN`). Masking `pred` to zero on horizons where `target_col` is `NaN` ensures
    `actual` and `pred` always compare the exact same set of realized horizons.

    Where it is used
    ----------------
    Called automatically at the end of `GLMForecaster.predict`, `CatBoostQuantileForecaster.predict`,
    and `EmpiricalQuantileForecaster.predict`.
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
    Evaluates multi-horizon quantile forecasts by computing pinball loss, empirical
    coverage `P(y <= q_hat)`, and quantile-approximated CRPS per horizon and quantile level.

    What it does
    ------------
    For each optional group in `group_cols` and each horizon `target` in `target_cols`:
    1. Extracts non-missing realizations `y` and the `(n_obs, K)` matrix of predicted
       quantiles `q_mat` (using column naming `quantile_col_name(target, q)`).
    2. Computes the mean Continuous Ranked Probability Score (`crps_mean`) across all
       observations via `quantile_crps(y, q_mat, levels)`.
    3. For each quantile level `q_j`, computes mean `pinball_loss(y, q_hat_j, q_j)` and
       empirical coverage `mean(y <= q_hat_j)` (which should be close to `q_j` for a
       well-calibrated forecaster).

    Where it is used
    ----------------
    Called by `DiscreteTimeSeries.evaluate_quantile_forecasts`, `scripts/run_m5_pipeline.py`,
    and quantile evaluation tests.
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


def _hurdle_conditional_mean(lam: np.ndarray | float, alpha: float | None) -> np.ndarray | float:
    """
    Computes the conditional positive expectation `E[Y | Y > 0] = lambda / (1 - P_0(lambda, alpha))`
    for a Zero-Truncated Poisson (`alpha is None`, `NaN`, or `<= 0`) or Zero-Truncated NB2 (`alpha > 0`).

    Why this helper is needed
    -------------------------
    In a Zero-Hurdle GLM, the count stage models `Y | Y > 0` using a zero-truncated distribution
    with untruncated rate parameter `lambda = exp(X @ beta_count)`. The unconditional mean is:

        mu = E[Y] = pi * E[Y | Y > 0] = pi * lambda / (1 - P_0(lambda, alpha))

    where `P_0(lambda) = exp(-lambda)` for Poisson and `P_0(lambda, alpha) = (1 + alpha * lambda)^(-1/alpha)`
    for NB2. Using `-expm1(...)` and clamping the conditional mean at `>= 1.0` preserves full
    floating-point precision as `lambda -> 0+` (where `lambda / (1 - P_0) -> 1.0`).
    """
    is_scalar = np.ndim(lam) == 0
    lam_arr = np.asarray(lam, dtype=np.float64)
    use_nb = alpha is not None and np.isfinite(alpha) and float(alpha) > 0.0
    if use_nb:
        a = float(alpha)
        log_p0 = -(1.0 / a) * np.log1p(np.maximum(a * lam_arr, 0.0))
        pos_mass = -np.expm1(log_p0)
    else:
        pos_mass = -np.expm1(-np.maximum(lam_arr, 0.0))

    cond_mu = np.where(pos_mass > 1e-12, np.maximum(lam_arr / np.maximum(pos_mass, 1e-12), 1.0), 1.0)
    return float(cond_mu) if is_scalar else cond_mu


def _fit_hurdle_binary_stage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
) -> tuple[dict[str, float], Any]:
    """
    Fits the binary hurdle stage `P(Y > 0 | X)` using a logistic regression GLM (`smf.logit`),
    with graceful fallback when a group's target is all-zero, all-positive, or exhibits
    perfect separation / constant covariates.
    """
    zero_coefs = {str(f): 0.0 for f in feature_cols}
    if len(train_df) == 0:
        return {"Intercept": -30.0, **zero_coefs}, None

    is_pos = (train_df[target_col] > 0).astype(int)
    n_pos = int(is_pos.sum())
    if n_pos == 0:
        return {"Intercept": -30.0, **zero_coefs}, None
    if n_pos == len(train_df):
        return {"Intercept": 30.0, **zero_coefs}, None

    active_features = [
        f for f in feature_cols if f in train_df.columns and train_df[f].nunique(dropna=True) > 1
    ]
    work_df = train_df[active_features].copy() if active_features else pd.DataFrame(index=train_df.index)
    work_df["_hurdle_y"] = is_pos.to_numpy()
    formula = f"_hurdle_y ~ {' + '.join(active_features)}" if active_features else "_hurdle_y ~ 1"

    summary = None
    params_dict: dict[str, float] | None = None
    try:
        logit_model = smf.logit(formula=formula, data=work_df)
        if regularization_alpha is not None and regularization_alpha > 0.0:
            res = logit_model.fit_regularized(
                alpha=float(regularization_alpha), L1_wt=float(l1_wt), maxiter=100, disp=False
            )
        else:
            try:
                res = logit_model.fit(maxiter=100, disp=False)
                summary = res.summary()
            except Exception:
                res = logit_model.fit(method="bfgs", maxiter=100, disp=False)
        cand = {str(k): float(v) for k, v in res.params.to_dict().items()}
        if all(np.isfinite(v) and abs(v) < 50.0 for v in cand.values()):
            params_dict = cand
        else:
            res_reg = logit_model.fit_regularized(alpha=1e-4, L1_wt=0.0, maxiter=100, disp=False)
            cand_reg = {str(k): float(v) for k, v in res_reg.params.to_dict().items()}
            if all(np.isfinite(v) for v in cand_reg.values()):
                params_dict = {k: float(np.clip(v, -30.0, 30.0)) for k, v in cand_reg.items()}
    except Exception:
        params_dict = None

    if params_dict is None:
        p_bar = float(np.clip(is_pos.mean(), 1e-6, 1.0 - 1e-6))
        return {"Intercept": float(np.log(p_bar / (1.0 - p_bar))), **zero_coefs}, None

    hurdle_params = {
        "Intercept": float(params_dict.get("Intercept", 0.0)),
        **{str(f): float(params_dict.get(f, 0.0)) for f in feature_cols},
    }
    return hurdle_params, summary


def _fit_hurdle_count_stage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    base_type: str,
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
) -> tuple[dict[str, float], float | None, Any]:
    """
    Fits the zero-truncated count stage `P(Y = k | Y > 0, X)` (`k >= 1`) on positive rows
    using `TruncatedLFNegativeBinomialP` (`base_type="nbinom"`) or `TruncatedLFPoisson`
    (`base_type="poisson"`), with automatic fallback if positive observations are all `1`
    or if Newton optimization encounters a singular Hessian on sparse subsets.
    """
    from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP, TruncatedLFPoisson

    zero_coefs = {str(f): 0.0 for f in feature_cols}
    pos_df = train_df[train_df[target_col] > 0]
    if len(pos_df) == 0 or float(pos_df[target_col].max()) <= 1.0:
        return {"Intercept": -20.0, **zero_coefs}, None, None

    active_features = [
        f for f in feature_cols if f in pos_df.columns and pos_df[f].nunique(dropna=True) > 1
    ]
    if len(pos_df) <= len(active_features) + 1:
        active_features = []
    formula = f"{target_col} ~ {' + '.join(active_features)}" if active_features else f"{target_col} ~ 1"

    def _try_truncated_fit(dist_cls, f_str: str) -> tuple[dict[str, float] | None, float | None, Any]:
        try:
            mod = dist_cls.from_formula(f_str, data=pos_df)
            if regularization_alpha is not None and regularization_alpha > 0.0:
                res = mod.fit_regularized(
                    alpha=float(regularization_alpha), L1_wt=float(l1_wt), maxiter=100, disp=False
                )
                summ = None
            else:
                try:
                    res = mod.fit(maxiter=100, disp=False)
                    summ = res.summary()
                except Exception:
                    res = mod.fit(method="bfgs", maxiter=100, disp=False)
                    summ = None
            raw = {str(k): float(v) for k, v in res.params.to_dict().items()}
            a_val = raw.pop("alpha", None)
            if not all(np.isfinite(v) and abs(v) < 50.0 for v in raw.values()):
                return None, None, None
            if dist_cls is TruncatedLFNegativeBinomialP:
                if a_val is None or not np.isfinite(a_val) or a_val <= 0.0 or a_val > 1e4:
                    return None, None, None
                return raw, float(a_val), summ
            return raw, None, summ
        except Exception:
            return None, None, None

    raw_params: dict[str, float] | None = None
    alpha_val: float | None = None
    summary = None

    if base_type == "nbinom":
        raw_params, alpha_val, summary = _try_truncated_fit(TruncatedLFNegativeBinomialP, formula)
        if raw_params is None and active_features:
            raw_params, alpha_val, summary = _try_truncated_fit(
                TruncatedLFNegativeBinomialP, f"{target_col} ~ 1"
            )

    if raw_params is None:
        raw_params, alpha_val, summary = _try_truncated_fit(TruncatedLFPoisson, formula)
        if raw_params is None and active_features:
            raw_params, alpha_val, summary = _try_truncated_fit(TruncatedLFPoisson, f"{target_col} ~ 1")

    if raw_params is None:
        mu_pos = max(float(pos_df[target_col].mean()) - 1.0, 1e-4)
        return {"Intercept": float(np.log(mu_pos)), **zero_coefs}, None, None

    count_params = {
        "Intercept": float(raw_params.get("Intercept", 0.0)),
        **{str(f): float(raw_params.get(f, 0.0)) for f in feature_cols},
    }
    return count_params, alpha_val, summary


def train_single_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str = "nbinom",
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
    hurdle: bool = False,
) -> tuple[dict[str, Any], Any]:
    """
    Trains a single log-link Poisson or Negative Binomial GLM (standard or zero-hurdle)
    for one group and one horizon.

    What it does
    ------------
    1. Filters `df` to training rows (`is_test == 0`) where `target_col` is not `NaN`.
    2. **Standard GLM (`hurdle=False`, `model_type in ("nbinom", "poisson")`)**:
       Builds the Statsmodels formula `target_col ~ feature_1 + feature_2 + ...` and
       instantiates either `smf.negativebinomial` (`model_type="nbinom"`) or `smf.poisson`
       (`model_type="poisson"`), fitting via MLE or penalized likelihood.
    3. **Zero-Hurdle GLM (`hurdle=True` or `model_type in ("hurdle_nbinom", "hurdle_poisson")`)**:
       Fits a two-part hurdle model:
       - *Binary hurdle stage*: logistic regression (`smf.logit`) on `1{target_col > 0}`
         across all training rows, producing `"hurdle_coefficients"`.
       - *Zero-truncated count stage*: `TruncatedLFNegativeBinomialP` (`"hurdle_nbinom"`)
         or `TruncatedLFPoisson` (`"hurdle_poisson"`) fitted strictly on positive training
         rows (`target_col > 0`), producing `"coefficients"` and `"alpha"`.
    4. Separates the Negative Binomial dispersion parameter `"alpha"` from the regression
       `"coefficients"` dictionary so `"coefficients"` strictly maps `"Intercept"` and
       covariate names to their linear weights.

    Where it is used
    ----------------
    Called by `GLMForecaster.fit` and `time_series.train_glm`.
    """
    if hurdle and model_type in ("nbinom", "poisson"):
        model_type = f"hurdle_{model_type}"

    train_df = df[~df[target_col].isna()]
    if "is_test" in train_df.columns:
        train_df = train_df[train_df["is_test"] == 0]

    if model_type in ("hurdle_nbinom", "hurdle_poisson"):
        base_type = "nbinom" if model_type == "hurdle_nbinom" else "poisson"
        hurdle_params, hurdle_summary = _fit_hurdle_binary_stage(
            train_df,
            feature_cols,
            target_col,
            regularization_alpha=regularization_alpha,
            l1_wt=l1_wt,
        )
        count_params, alpha_val, count_summary = _fit_hurdle_count_stage(
            train_df,
            feature_cols,
            target_col,
            base_type=base_type,
            regularization_alpha=regularization_alpha,
            l1_wt=l1_wt,
        )
        prod_model = {
            "model_type": model_type,
            "horizon": target_col,
            "coefficients": count_params,
            "hurdle_coefficients": hurdle_params,
            "alpha": float(alpha_val) if alpha_val is not None and np.isfinite(alpha_val) else None,
        }
        return prod_model, count_summary if count_summary is not None else hurdle_summary

    formula = f"{target_col} ~ {' + '.join(feature_cols)}"
    if model_type == "nbinom":
        model = smf.negativebinomial(formula=formula, data=train_df)
    elif model_type == "poisson":
        model = smf.poisson(formula=formula, data=train_df)
    else:
        raise ValueError(
            f"Unsupported distribution: '{model_type}'. Supported options are "
            "'nbinom', 'poisson', 'hurdle_nbinom', 'hurdle_poisson'."
        )

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

    What it does
    ------------
    Stores fitted per-group, per-horizon GLM parameter dictionaries (`model_type`,
    `coefficients`, `alpha`) along with feature and target metadata, and provides:
    - `.to_dataframe()` / `.from_dataframe()` for round-tripping with the tabular
      DataFrame representation (`GLMForecaster.model_data`),
    - `.coefficients_table()` for inspecting a tidy long-format DataFrame of all
      regression weights across groups and horizons, and
    - `.save()` / `.load()` for clean JSON persistence (`glm_meta.json`).

    Where it is used
    ----------------
    Created by `GLMForecaster.fit`, persisted by `GLMForecaster.save`, and used by
    `GLMForecaster.predict`.
    """
    target_cols: list[str]
    feature_cols: list[str]
    train_groups: list[str] = field(default_factory=list)
    models: dict[tuple[str, ...], dict[str, dict[str, Any]]] = field(default_factory=dict)
    horizon_feature_cols: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def group_key(values: Any) -> tuple[str, ...]:
        """Normalizes a scalar or tuple of group column values into a tuple of strings."""
        if not isinstance(values, tuple):
            values = (values,)
        return tuple(str(v) for v in values)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the nested `models` dictionary into a wide DataFrame with one row per group and one column per horizon."""
        rows = []
        for group_key, horizon_dict in self.models.items():
            row: dict[str, Any] = {}
            for col_name, key_val in zip(self.train_groups, group_key):
                if col_name in ("overdispersion", "positive_overdispersion"):
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
        """Reconstructs a `GLMModel` from a wide DataFrame (such as `GLMForecaster.model_data` or `load_glm_model_data`)."""
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
        """Returns a tidy long-format DataFrame of fitted GLM coefficients across groups and horizons."""
        has_hurdle = any(
            "hurdle_coefficients" in m_dict
            for horizon_dict in self.models.values()
            for m_dict in horizon_dict.values()
        )
        rows = []
        for group_key, horizon_dict in self.models.items():
            group_info = dict(zip(self.train_groups, group_key))
            for target, m_dict in horizon_dict.items():
                model_type = m_dict.get("model_type")
                alpha = m_dict.get("alpha")
                for feat, coef in m_dict.get("coefficients", {}).items():
                    row = {
                        **group_info,
                        "target": target,
                        "model_type": model_type,
                        "alpha": alpha,
                        "feature": feat,
                        "coefficient": coef,
                    }
                    if has_hurdle:
                        row["component"] = "count"
                    rows.append(row)
                if "hurdle_coefficients" in m_dict:
                    for feat, coef in m_dict.get("hurdle_coefficients", {}).items():
                        rows.append({
                            **group_info,
                            "target": target,
                            "model_type": model_type,
                            "alpha": None,
                            "feature": feat,
                            "coefficient": coef,
                            "component": "hurdle",
                        })
        return pd.DataFrame(rows)

    def save(self, out_dir: Path | str, prefix: str = "glm") -> None:
        """Saves all group/horizon GLM parameters and feature metadata to `<out_dir>/<prefix>_meta.json`."""
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
        """Loads a `GLMModel` from `<out_dir>/<prefix>_meta.json`."""
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

    What it does
    ------------
    Defines the common lifecycle and state management shared by `GLMForecaster`,
    `CatBoostQuantileForecaster`, and `EmpiricalQuantileForecaster`:
    - **Fit once**: `forecaster.fit(ts)` binds the training panel's feature specification
      (`feature_spec_`), per-series metadata (`series_metadata_`), and column names via
      `_bind_training_context(ts)`.
    - **Flexible inference input resolution**: `_resolve_inference_inputs(...)` allows
      `predict()` and `compute_ddlt()` to accept a `DiscreteTimeSeries`, a pre-engineered
      `pd.DataFrame` (passed either as `df=` or positionally as `ts`), and an optional
      `period` filter, automatically merging missing training `series_metadata_` onto
      standalone DataFrames.
    - **Raw-data inference**: `transform(sales_df, ...)` and `predict_from_raw(sales_df, ...)`
      replay the saved `TimeSeriesFeatureSpec` on brand-new raw sales and covariate tables
      without needing the original training `DiscreteTimeSeries` object.
    - **Self-contained persistence**: `_save_base_context` and `_load_base_context` persist
      `forecaster_meta.json`, `feature_spec.json`, and `series_metadata.csv` alongside the
      subclass's model weights so `load_forecaster(dir)` restores a complete, ready-to-score
      pipeline.
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
        """Captures column lists, `feature_spec`, and `series_metadata` from `ts` during `fit()`."""
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
        whether the caller passed a `DiscreteTimeSeries`, a standalone `pd.DataFrame`, or both.

        Why it is needed
        ----------------
        Callers may invoke `forecaster.predict(ts)`, `forecaster.predict(ts, period=1914)`,
        `forecaster.predict(df=custom_df)`, or `forecaster.predict(custom_df)`. Furthermore,
        when a standalone `df` is passed (or after loading a saved forecaster), per-series
        training metadata (`overdispersion`, `best_fit`, `distr_param_0..2`) might not be
        present on `df` yet. This helper normalizes all calling conventions and left-joins
        any missing `series_metadata_` columns onto `test_data`.
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
                if "positive_overdispersion" in missing_meta_cols:
                    test_data["positive_overdispersion"] = test_data["positive_overdispersion"].fillna(0).astype(int)

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
        Transforms raw `sales_df` and external feature tables into an inference-ready
        feature DataFrame using the `feature_spec_` and `series_metadata_` captured
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
        End-to-end reusable inference from raw `sales_df` + covariates without
        requiring the original training `DiscreteTimeSeries`.
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
        """Writes `forecaster_meta.json`, `feature_spec.json`, and `series_metadata.csv` to `out_dir`."""
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
        """Restores `target_cols_`, `feature_cols_`, `feature_spec_`, and `series_metadata_` from `out_dir`."""
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
        Predicts every forecast horizon. `model` optionally overrides the fitted model
        container (`GLMModel` / coefficient table for GLMs, `CatBoostQuantileModel` for CatBoost).
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
        """Shared signature for all forecasters (`model` has the same meaning as in `predict`)."""
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
    Grouped Poisson / Negative-Binomial GLM forecaster (standard or zero-hurdle) across multiple horizons.

    What it does
    ------------
    - **Training (`fit`)**:
      * When `hurdle=False` (default): Partitions the training panel by `train_groups + ["overdispersion"]`.
        Fits a Negative Binomial GLM (`model_type="nbinom"`) if `overdispersion == 1` or a Poisson GLM
        (`model_type="poisson"`) if `overdispersion == 0`.
      * When `hurdle=True`: Partitions the training panel by `train_groups + ["positive_overdispersion"]`,
        where `positive_overdispersion` is computed strictly on positive training observations (`demand > 0`).
        Fits a Zero-Hurdle Negative Binomial GLM (`model_type="hurdle_nbinom"`) if `positive_overdispersion == 1`
        or a Zero-Hurdle Poisson GLM (`model_type="hurdle_poisson"`) if `positive_overdispersion == 0`.
    - **Prediction (`predict`)**:
      * Standard GLM: Evaluates `target_t_plus_h_mu = exp(X @ beta)` and attaches `target_t_plus_h_alpha`.
      * Zero-Hurdle GLM: Evaluates hurdle probability `target_t_plus_h_pi = sigmoid(X @ gamma)`,
        untruncated count rate `target_t_plus_h_lambda = exp(X @ beta)`, `target_t_plus_h_alpha`,
        and unconditional expected value `target_t_plus_h_mu = pi * lambda / (1 - P_0(lambda, alpha))`.
    - **DDLT (`compute_ddlt`)**: Passes the predicted GLM parameter columns to
      `DDLTCalculator.from_glm_group`, which constructs truncated Poisson/NB2 or Zero-Hurdle
      Poisson/NB2 PMFs in compiled Numba log-space and convolves them over the lead-time distribution.
    """

    forecaster_type = "glm"

    def __init__(
        self,
        train_groups: tuple[str, ...] | list[str] | None = None,
        regularization_alpha: float | None = None,
        l1_wt: float = 0.0,
        restrict_future_leads: bool = False,
        hurdle: bool = False,
    ):
        super().__init__()
        self.train_groups = list(train_groups) if train_groups is not None else []
        self.regularization_alpha = regularization_alpha
        self.l1_wt = l1_wt
        self.restrict_future_leads = restrict_future_leads
        self.hurdle = bool(hurdle)
        self.model_data: pd.DataFrame | None = None
        self.model: GLMModel | None = None

    def fit(self, ts: "DiscreteTimeSeries") -> pd.DataFrame:
        """
        Fits grouped multi-horizon Poisson / Negative Binomial GLMs (standard or zero-hurdle) on `ts`
        and returns the wide `model_data` DataFrame (while also storing `self.model: GLMModel`).
        """
        t0 = time.perf_counter()
        self._bind_training_context(ts)
        dispersion_col = "positive_overdispersion" if self.hurdle else "overdispersion"
        if dispersion_col not in ts.df.columns:
            # Ensure positive_overdispersion is present if ts was constructed without it:
            if self.hurdle and ts.var_col in ts.df.columns and ts.ts_id_cols:
                tmp_pos = np.where(
                    (ts.df["is_test"] == 0 if "is_test" in ts.df.columns else True) & (ts.df[ts.var_col] > 0),
                    ts.df[ts.var_col],
                    np.nan,
                )
                pos_s = pd.Series(tmp_pos, index=ts.df.index)
                grp = pos_s.groupby([ts.df[c] for c in ts.ts_id_cols], observed=True)
                ts.df["positive_overdispersion"] = (grp.transform("var") > grp.transform("mean")).fillna(False).astype(int)
                if self.series_metadata_ is not None and "positive_overdispersion" not in self.series_metadata_.columns:
                    meta_pos = ts.df[list(ts.ts_id_cols) + ["positive_overdispersion"]].drop_duplicates(subset=list(ts.ts_id_cols))
                    self.series_metadata_ = pd.merge(self.series_metadata_, meta_pos, on=list(ts.ts_id_cols), how="left")
            elif "overdispersion" in ts.df.columns:
                dispersion_col = "overdispersion"

        partition_cols = list(self.train_groups)
        if dispersion_col not in partition_cols:
            partition_cols.append(dispersion_col)

        train_df = ts.df[ts.df["is_test"] == 0] if ts.test_start_period is not None else ts.df
        n_groups = train_df[partition_cols].drop_duplicates().shape[0]
        logger.info(
            "Starting %sGLM training across %d group(s) and %d horizon(s) (%d training rows)...",
            "Hurdle " if self.hurdle else "",
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
                idx = partition_cols.index(dispersion_col)
                is_overdispersed = group_name[idx]
            else:
                is_overdispersed = group_name

            if self.hurdle:
                model_type = "hurdle_nbinom" if is_overdispersed == 1 else "hurdle_poisson"
            else:
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
                    hurdle=self.hurdle,
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
        """
        Generates `<target>_mu` and `<target>_alpha` predictions (plus `<target>_pi` and
        `<target>_lambda` for zero-hurdle models) for every horizon in `target_cols`.

        Why string-keyed fallback lookup is included
        --------------------------------------------
        When a `GLMModel` is loaded from JSON (`glm_meta.json`), its group keys are tuples of
        strings (e.g., `('CA_1', '1')`), whereas an in-memory DataFrame may have integer or
        categorical group keys (e.g., `('CA_1', 1)`). Maintaining both `model_dict_map` and
        `str_model_dict_map` guarantees seamless group matching regardless of whether the
        model was just fitted in memory or loaded from disk.
        """
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
        # Fallback if a standalone DataFrame was passed without `positive_overdispersion`:
        if "positive_overdispersion" in train_groups and "positive_overdispersion" not in test_data.columns:
            if "overdispersion" in test_data.columns:
                test_data["positive_overdispersion"] = test_data["overdispersion"].fillna(0).astype(int)
            else:
                test_data["positive_overdispersion"] = 0
        if "overdispersion" in train_groups and "overdispersion" not in test_data.columns:
            if "positive_overdispersion" in test_data.columns:
                test_data["overdispersion"] = test_data["positive_overdispersion"].fillna(0).astype(int)
            else:
                test_data["overdispersion"] = 0

        has_hurdle = self.hurdle or any(
            isinstance(m_dict, dict)
            and ("hurdle_coefficients" in m_dict or str(m_dict.get("model_type", "")).startswith("hurdle_"))
            for h_dict in glm_container.models.values()
            for m_dict in h_dict.values()
        )
        logger.info("Generating GLM predictions for %d rows across %d horizons...", len(test_data), len(target_cols))

        for target in target_cols:
            if has_hurdle:
                test_data[f"{target}_pi"] = np.nan
                test_data[f"{target}_lambda"] = np.nan
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
                alpha_clean = float(alpha_val) if alpha_val is not None and np.isfinite(alpha_val) else np.nan
                coef_series = pd.Series(coefs, dtype=float)
                X_aligned = X.reindex(columns=coef_series.index).fillna(0.0)
                z = X_aligned.dot(coef_series)
                lam = np.exp(np.clip(z.to_numpy(dtype=np.float64), -50.0, 50.0))

                if "hurdle_coefficients" in model_dict or str(model_dict.get("model_type", "")).startswith("hurdle_"):
                    h_coefs = dict(model_dict.get("hurdle_coefficients", {"Intercept": 0.0}))
                    h_series = pd.Series(h_coefs, dtype=float)
                    X_h = X.reindex(columns=h_series.index).fillna(0.0)
                    z_h = np.clip(X_h.dot(h_series).to_numpy(dtype=np.float64), -50.0, 50.0)
                    pi = 1.0 / (1.0 + np.exp(-z_h))
                    mu = pi * _hurdle_conditional_mean(lam, alpha_clean)
                    test_data.loc[group_idx, f"{target}_pi"] = pi
                    test_data.loc[group_idx, f"{target}_lambda"] = lam
                    test_data.loc[group_idx, f"{target}_mu"] = mu
                else:
                    test_data.loc[group_idx, f"{target}_mu"] = lam
                test_data.loc[group_idx, f"{target}_alpha"] = alpha_clean

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
        """
        Generates GLM predictions and propagates them through `DDLTCalculator.from_glm_group`
        across all `lead_time_keys` groups.
        """
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
        """Persists the fitted `GLMForecaster` bundle (`forecaster_meta.json`, `feature_spec.json`, `series_metadata.csv`, and `glm_meta.json`) to `out_dir`."""
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
                "hurdle": self.hurdle,
                "prefix": prefix,
            },
        )
        self.model.save(out_path, prefix=prefix)
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "glm") -> "GLMForecaster":
        """Restores a fitted `GLMForecaster` from `out_dir`."""
        out_path = Path(out_dir)
        meta_file = out_path / "forecaster_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        saved_prefix = meta.get("prefix", prefix)
        glm_model = GLMModel.load(out_path, prefix=saved_prefix)
        inferred_hurdle = any(
            isinstance(m_dict, dict)
            and ("hurdle_coefficients" in m_dict or str(m_dict.get("model_type", "")).startswith("hurdle_"))
            for h_dict in glm_model.models.values()
            for m_dict in h_dict.values()
        )
        forecaster = cls(
            train_groups=meta.get(
                "train_groups",
                [c for c in glm_model.train_groups if c not in ("overdispersion", "positive_overdispersion")],
            ),
            regularization_alpha=meta.get("regularization_alpha"),
            l1_wt=float(meta.get("l1_wt", 0.0)),
            restrict_future_leads=bool(meta.get("restrict_future_leads", False)),
            hurdle=bool(meta.get("hurdle", inferred_hurdle)),
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
    Multi-horizon CatBoost `MultiQuantile` forecaster with Numba-accelerated
    quantile-to-distribution reconstruction and DDLT calculation.

    What it does
    ------------
    - **Training (`fit`)**: For each partition in `train_groups` and each horizon
      `target_t_plus_h`, trains a joint `CatBoostRegressor(loss_function='MultiQuantile:alpha=...')`
      using a leak-free temporal validation split (`num_valid_days` immediately prior to
      `test_start_period`).
    - **Prediction (`predict`)**: Predicts all `K` quantiles simultaneously per horizon,
      applies non-crossing rearrangement and non-negativity clipping (`predict_catboost_quantile`),
      and computes the implied discrete expected value `<target>_mu` via `quantiles_to_mean`.
    - **DDLT (`compute_ddlt`)**: Passes predicted quantile columns to
      `DDLTCalculator.from_quantiles_group`, which converts each row's quantiles into a
      `DiscreteDistribution` via `_quantiles_to_pmf_numba` and convolves them over the
      lead-time distribution.
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
        """
        Trains one `CatBoostRegressor` per `(group, horizon)` pair and returns the
        fitted `CatBoostQuantileModel` container.

        How the temporal validation split works (lines 869-872)
        -------------------------------------------------------
        For a row at period position `group_pos` and a horizon `target_t_plus_h` with offset
        `h = ts._target_offset(target)`:
        - The target realization occurs at period position `target_pos = group_pos + h`.
        - To prevent any look-ahead into the test set, `has_target` requires `target_pos < test_start_pos`.
        - The validation set (`valid_mask`) consists of rows whose realization date `target_pos`
          falls in the last `num_valid_days` before `test_start_period`
          (`test_start_pos - num_valid_days <= target_pos < test_start_pos`).
        - The training set (`fit_mask`) consists of all earlier rows (`target_pos < test_start_pos - num_valid_days`),
          optionally bounded by `max_train_days` so training on multi-year panels stays fast
          and focused on recent regime dynamics.
        """
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
        """
        Generates quantile forecast columns (`<target>_q_<level>`) and implied discrete
        mean columns (`<target>_mu`) across all horizons, plus cumulative error metrics.
        """
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
        """
        Generates CatBoost quantile predictions and propagates them through
        `DDLTCalculator.from_quantiles_group` across all `lead_time_keys` groups.
        """
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
        """Persists the fitted `CatBoostQuantileForecaster` bundle (`forecaster_meta.json`, `feature_spec.json`, `series_metadata.csv`, `catboost_meta.json`, and `.cbm` model files) to `out_dir`."""
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
        """Restores a fitted `CatBoostQuantileForecaster` from `out_dir`."""
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
    Universal loader that inspects `out_dir` and reconstructs the appropriate
    fitted `ProbabilisticForecaster` (`GLMForecaster`, `CatBoostQuantileForecaster`, or
    `EmpiricalQuantileForecaster`), complete with its feature specification and series metadata.

    What it does
    ------------
    1. Checks `<out_dir>/forecaster_meta.json` for `"forecaster_type"` (`"empirical"`,
       `"glm"`, or `"catboost"`) and dispatches to the corresponding `.load(out_path)`.
    2. Falls back to checking `<out_dir>/catboost_meta.json` or `<out_dir>/glm_meta.json`
       so directories saved directly via `CatBoostQuantileModel.save` or `GLMModel.save`
       are also loaded seamlessly.

    Where it is used
    ----------------
    Exported at the top-level package (`probabilistic_modeling.load_forecaster`) for
    reloading any saved model artifact directory without hardcoding the model class.
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
    """
    Internal helper that attaches optional per-item/group `service_levels` and applies
    `group_fn(gdf, lt_distr)` either to the entire `predictions` DataFrame (when `lead_time`
    is a single `DiscreteDistribution`) or per `lead_time_keys` group (when `lead_time`
    is a dictionary mapping group keys to `DiscreteDistribution` objects).

    Where it is used
    ----------------
    Shared by `GLMForecaster.compute_ddlt`, `CatBoostQuantileForecaster.compute_ddlt`,
    and `EmpiricalQuantileForecaster.compute_ddlt`.
    """
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
