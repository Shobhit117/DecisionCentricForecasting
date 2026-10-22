"""
CatBoost multi-quantile regression for multi-horizon demand forecasting.

One CatBoost model is trained per forecast horizon (``target_t_plus_<h>``) and, optionally,
per training group. Each model uses CatBoost's ``MultiQuantile`` loss, i.e. the sum of the
pinball losses over all requested quantile levels, so a single model returns the full
vector of quantiles for every observation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantile_distribution import rearrange_quantiles, validate_quantile_levels

logger = logging.getLogger(__name__)

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
    levels = validate_quantile_levels(quantiles)
    return "MultiQuantile:alpha=" + ",".join(f"{q:g}" for q in levels)


def prepare_features(df: pd.DataFrame, feature_cols: list[str], cat_feature_cols: list[str]) -> pd.DataFrame:
    """Selects model features and casts categorical features to strings (as required by CatBoost)."""
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
    """Fits a single CatBoost ``MultiQuantile`` model for one forecast horizon."""
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
    """Returns a ``(n_obs, n_quantiles)`` matrix of non-crossing, non-negative quantiles."""
    from catboost import Pool

    pool = Pool(prepare_features(df, feature_cols, cat_feature_cols), cat_features=cat_feature_cols)
    pred = np.asarray(model.predict(pool), dtype=float)
    return rearrange_quantiles(np.atleast_2d(pred).reshape(len(df), -1))


def quantile_col_name(target_col: str, level: float) -> str:
    return f"{target_col}_q{level:g}"


@dataclass
class CatBoostQuantileModel:
    """Container for the per-group, per-horizon CatBoost models and their metadata."""
    quantiles: tuple[float, ...]
    target_cols: list[str]
    feature_cols: list[str]
    cat_feature_cols: list[str]
    train_groups: list[str] = field(default_factory=list)
    # models[group_key][target_col] -> CatBoostRegressor; group_key is a tuple of strings (empty for a global model)
    models: dict[tuple, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def group_key(values: Any) -> tuple[str, ...]:
        if not isinstance(values, tuple):
            values = (values,)
        return tuple(str(v) for v in values)

    def quantile_cols(self, target_col: str) -> list[str]:
        return [quantile_col_name(target_col, q) for q in self.quantiles]

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for group_key, group_models in self.models.items():
            for target, model in group_models.items():
                for feat, imp in zip(model.feature_names_, model.get_feature_importance()):
                    rows.append({**dict(zip(self.train_groups, group_key)), "target": target, "feature": feat, "importance": imp})
        return pd.DataFrame(rows)

    def save(self, out_dir: Path, prefix: str = "catboost") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "quantiles": list(self.quantiles),
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            "cat_feature_cols": self.cat_feature_cols,
            "train_groups": self.train_groups,
            "models": [],
        }
        for group_key, group_models in self.models.items():
            group_str = "_".join(str(k) for k in group_key) or "all"
            for target, model in group_models.items():
                file_name = f"{prefix}_{group_str}_{target}.cbm"
                model.save_model(str(out_dir / file_name))
                meta["models"].append({"group_key": [str(k) for k in group_key], "target": target, "file": file_name,
                                       "best_iteration": model.get_best_iteration()})
        with open(out_dir / f"{prefix}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, out_dir: Path, prefix: str = "catboost") -> "CatBoostQuantileModel":
        from catboost import CatBoostRegressor

        with open(out_dir / f"{prefix}_meta.json") as f:
            meta = json.load(f)
        obj = cls(tuple(meta["quantiles"]), meta["target_cols"], meta["feature_cols"], meta["cat_feature_cols"], meta["train_groups"])
        for entry in meta["models"]:
            model = CatBoostRegressor()
            model.load_model(str(out_dir / entry["file"]))
            obj.models.setdefault(tuple(entry["group_key"]), {})[entry["target"]] = model
        return obj
