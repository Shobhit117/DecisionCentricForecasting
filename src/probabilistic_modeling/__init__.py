"""
probabilistic_modeling
======================

Probabilistic demand forecasting, discrete distribution algebra, and
inventory optimization / simulation.

Subpackages
-----------
- ``probabilistic_modeling.distributions`` : discrete & quantile-based predictive distributions
- ``probabilistic_modeling.models``        : GLM and CatBoost multi-quantile forecasters
- ``probabilistic_modeling.inventory``     : multi-product newsvendor and inventory simulation
- ``probabilistic_modeling.time_series``   : panel time-series feature engineering
- ``probabilistic_modeling.ddlt``          : Numba kernels and demand-during-lead-time calculator

The most commonly used objects are re-exported lazily at the top level, so
``import probabilistic_modeling`` stays cheap (heavy dependencies such as
CatBoost and statsmodels are only imported when first accessed).
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.3.0"

# public name -> defining module (relative to this package)
_LAZY_EXPORTS: dict[str, str] = {
    # distributions
    "DiscreteDistribution": "distributions.discrete",
    "PredictiveDistribution": "distributions.discrete",
    "ContinuousQuantileDistribution": "distributions.quantile",
    "DEFAULT_QUANTILES": "distributions.quantile",
    "quantiles_to_distribution": "distributions.quantile",
    "quantiles_to_mean": "distributions.quantile",
    "rearrange_quantiles": "distributions.quantile",
    "pinball_loss": "distributions.quantile",
    "quantile_crps": "distributions.quantile",
    # lead-time demand
    "DDLTCalculator": "ddlt",
    "support_tol_for_service_level": "ddlt",
    "TruncatedSupportWarning": "distributions.discrete",
    # time series
    "DiscreteTimeSeries": "time_series",
    "TimeSeriesFeatureSpec": "time_series",
    "compute_lead_time_demand": "time_series",
    # models
    "ProbabilisticForecaster": "models.forecasters",
    "GLMForecaster": "models.forecasters",
    "CatBoostQuantileForecaster": "models.forecasters",
    "EmpiricalQuantileForecaster": "models.baseline",
    "load_forecaster": "models.forecasters",
    "compute_forecasting_metrics": "models.forecasters",
    "evaluate_quantile_forecasts": "models.forecasters",
    # evaluation
    "evaluate_ddlt_calibration": "evaluation.ddlt_calibration",
    "DDLTCalibrationResult": "evaluation.ddlt_calibration",
    # inventory
    "InventorySimulator": "inventory.simulation",
    "InventorySimulationResult": "inventory.simulation",
    "simulate_inventory": "inventory.simulation",
    "resolve_target_inventory": "inventory.simulation",
    "compute_simulation_kpis": "inventory.simulation",
    # utils
    "setup_logging": "logging_utils",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

