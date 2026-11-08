"""
CatBoost multi-quantile regression for multi-horizon demand forecasting.

What this module does
---------------------
Provides low-level training, prediction, feature preparation, and `.cbm` + JSON persistence
for CatBoost ``MultiQuantile`` models (:class:`CatBoostQuantileModel`). One CatBoost regressor
is trained per forecast horizon (``target_t_plus_<h>``) and, optionally, per training group.

Why this module is needed
-------------------------
Instead of fitting 12 separate quantile regression trees per horizon (one per quantile level
``tau``), CatBoost's ``MultiQuantile:alpha=tau_1,...,tau_K`` loss optimizes the sum of pinball
losses across all ``K`` quantile levels simultaneously in a single multi-output tree ensemble.
This is ~10x faster to train and score than separate models and drastically reduces quantile
crossing. Any residual crossing is removed via :func:`rearrange_quantiles`.

Where this module is used
-------------------------
- Wrapped by :class:`probabilistic_modeling.models.forecasters.CatBoostQuantileForecaster`
- ``quantile_col_name`` is also used by :class:`EmpiricalQuantileForecaster`,
  :func:`evaluate_quantile_forecasts`, and :meth:`DDLTCalculator.from_quantiles_group`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probabilistic_modeling.distributions.quantile import rearrange_quantiles, validate_quantile_levels

logger = logging.getLogger(__name__)

# Default hyperparameters for CatBoostRegressor when training MultiQuantile models.
# `allow_writing_files=False` prevents CatBoost from creating `catboost_info/` directories on disk.
DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": 100,
    "allow_writing_files": False,
}


def multiquantile_loss(quantiles: list[float] | tuple[float, ...] | np.ndarray) -> str:
    """
    Format the CatBoost loss-function string ``"MultiQuantile:alpha=0.05,0.1,..."`` after
    validating that ``quantiles`` are strictly increasing in ``(0, 1)``.
    """
    levels = validate_quantile_levels(quantiles)
    return "MultiQuantile:alpha=" + ",".join(f"{q:g}" for q in levels)


def prepare_features(df: pd.DataFrame, feature_cols: list[str], cat_feature_cols: list[str]) -> pd.DataFrame:
    """
    Select numerical (`feature_cols`) and categorical (`cat_feature_cols`) columns from ``df``
    and cast categorical columns to ``str`` (required by ``catboost.Pool``).
    """
    X = df[feature_cols + cat_feature_cols].copy()
    for col in cat_feature_cols:
        X[col] = X[col].astype(str)
    return X


def train_catboost_quantile(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_feature_cols: list[str],
    target_col: str,
    quantiles: list[float] | tuple[float, ...],
    eval_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
):
    """
    Fit a single ``CatBoostRegressor`` with ``MultiQuantile`` loss for one forecast horizon ``target_col``.

    If a temporal validation split ``eval_df`` is provided, enables early stopping and keeps
    the iteration with the lowest validation ``MultiQuantile`` loss (`use_best_model=True`).

    Where it is used
    ----------------
    Called inside the group x horizon training loop of :meth:`CatBoostQuantileForecaster.fit`.
    """
    from catboost import CatBoostRegressor, Pool  # imported lazily: optional dependency

    model_params = {**DEFAULT_CATBOOST_PARAMS, **(params or {})}
    model_params["loss_function"] = multiquantile_loss(quantiles)
    if eval_df is None or eval_df.empty:
        model_params.pop("early_stopping_rounds", None)

    train_pool = Pool(prepare_features(train_df, feature_cols, cat_feature_cols), label=train_df[target_col].to_numpy(dtype=float), cat_features=cat_feature_cols)
    eval_pool = None
    if eval_df is not None and not eval_df.empty:
        eval_pool = Pool(prepare_features(eval_df, feature_cols, cat_feature_cols), label=eval_df[target_col].to_numpy(dtype=float), cat_features=cat_feature_cols)

    model = CatBoostRegressor(**model_params)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
    return model


def predict_catboost_quantile(model, df: pd.DataFrame, feature_cols: list[str], cat_feature_cols: list[str]) -> np.ndarray:
    """
    Predict all quantiles for ``df`` using a fitted ``CatBoostRegressor`` and apply
    :func:`rearrange_quantiles` so the returned ``(n_obs, n_quantiles)`` matrix is guaranteed
    monotonically non-decreasing and non-negative.

    Where it is used
    ----------------
    Called inside :meth:`CatBoostQuantileForecaster.predict`.
    """
    from catboost import Pool

    pool = Pool(prepare_features(df, feature_cols, cat_feature_cols), cat_features=cat_feature_cols)
    pred = np.asarray(model.predict(pool), dtype=float)
    return rearrange_quantiles(np.atleast_2d(pred).reshape(len(df), -1))


def quantile_col_name(target_col: str, level: float) -> str:
    """
    Canonical column name ``f"{target_col}_q{level:g}"`` (e.g. ``"target_t_plus_0_q0.95"``)
    used to store predicted quantiles in DataFrames across all quantile forecasters and DDLT.
    """
    return f"{target_col}_q{level:g}"


@dataclass
class CatBoostQuantileModel:
    """
    Serializable container holding the fitted per-group, per-horizon ``CatBoostRegressor``
    objects and their feature/quantile metadata.

    What it does
    ------------
    - Stores ``models[group_key][target_col] -> CatBoostRegressor`` where ``group_key`` is a
      tuple of strings (or ``()`` when trained globally without ``train_groups``).
    - Tracks horizon-specific feature lists (``horizon_feature_cols``, ``horizon_cat_feature_cols``)
      when ``restrict_future_leads=True``.
    - Exports tidy feature importances via :meth:`feature_importance` and saves/loads binary
      ``.cbm`` files plus ``catboost_meta.json`` via :meth:`save` / :meth:`load`.

    Where it is used
    ----------------
    Returned by :meth:`CatBoostQuantileForecaster.fit` and stored on ``forecaster.model``.
    """
    quantiles: tuple[float, ...]
    target_cols: list[str]
    feature_cols: list[str]
    cat_feature_cols: list[str]
    train_groups: list[str] = field(default_factory=list)
    # models[group_key][target_col] -> CatBoostRegressor; group_key is a tuple of strings (empty for a global model)
    models: dict[tuple, dict[str, Any]] = field(default_factory=dict)
    horizon_feature_cols: dict[str, list[str]] = field(default_factory=dict)
    horizon_cat_feature_cols: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def group_key(values: Any) -> tuple[str, ...]:
        """Normalize a scalar or tuple of group values into a canonical tuple of strings."""
        if not isinstance(values, tuple):
            values = (values,)
        return tuple(str(v) for v in values)

    def features_for_target(self, target_col: str) -> tuple[list[str], list[str]]:
        """Return ``(numerical_cols, categorical_cols)`` used by horizon ``target_col``."""
        num_cols = self.horizon_feature_cols.get(target_col, self.feature_cols)
        cat_cols = self.horizon_cat_feature_cols.get(target_col, self.cat_feature_cols)
        return num_cols, cat_cols

    def quantile_cols(self, target_col: str) -> list[str]:
        """Return the ordered list of quantile column names for ``target_col``."""
        return [quantile_col_name(target_col, q) for q in self.quantiles]

    def feature_importance(self) -> pd.DataFrame:
        """Return a tidy DataFrame of CatBoost feature importances across all groups and horizons."""
        rows = []
        for group_key, group_models in self.models.items():
            for target, model in group_models.items():
                for feat, imp in zip(model.feature_names_, model.get_feature_importance()):
                    rows.append({**dict(zip(self.train_groups, group_key)), "target": target, "feature": feat, "importance": imp})
        return pd.DataFrame(rows)

    def save(self, out_dir: Path | str, prefix: str = "catboost") -> None:
        """Save all ``CatBoostRegressor`` models as ``.cbm`` files plus ``<prefix>_meta.json`` in ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "quantiles": list(self.quantiles),
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            "cat_feature_cols": self.cat_feature_cols,
            "train_groups": self.train_groups,
            "horizon_feature_cols": self.horizon_feature_cols,
            "horizon_cat_feature_cols": self.horizon_cat_feature_cols,
            "models": [],
        }
        for group_key, group_models in self.models.items():
            group_str = "_".join(str(k) for k in group_key) or "all"
            for target, model in group_models.items():
                file_name = f"{prefix}_{group_str}_{target}.cbm"
                model.save_model(str(out_dir / file_name))
                meta["models"].append({"group_key": [str(k) for k in group_key], "target": target, "file": file_name,
                                       "best_iteration": model.get_best_iteration()})
        with open(out_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "catboost") -> "CatBoostQuantileModel":
        """Reconstruct a :class:`CatBoostQuantileModel` from ``<prefix>_meta.json`` and ``.cbm`` files in ``out_dir``."""
        from catboost import CatBoostRegressor

        out_dir = Path(out_dir)
        with open(out_dir / f"{prefix}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            tuple(meta["quantiles"]),
            meta["target_cols"],
            meta["feature_cols"],
            meta["cat_feature_cols"],
            meta["train_groups"],
            horizon_feature_cols=meta.get("horizon_feature_cols", {}),
            horizon_cat_feature_cols=meta.get("horizon_cat_feature_cols", {}),
        )
        for entry in meta["models"]:
            model = CatBoostRegressor()
            model.load_model(str(out_dir / entry["file"]))
            obj.models.setdefault(tuple(entry["group_key"]), {})[entry["target"]] = model
        return obj
