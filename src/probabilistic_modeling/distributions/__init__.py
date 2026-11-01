"""
Discrete and quantile-based predictive distributions.

Names are resolved lazily (PEP 562) because ``probabilistic_modeling.ddlt`` and the
modules in this subpackage import each other; eager re-exports here would create an
import cycle whenever ``ddlt`` is imported first.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    "DiscreteDistribution": "discrete",
    "PredictiveDistribution": "discrete",
    "compute_best_fit_distribution": "discrete",
    "DEFAULT_QUANTILES": "quantile",
    "ContinuousQuantileDistribution": "quantile",
    "distribution_mean": "quantile",
    "pinball_loss": "quantile",
    "quantile_crps": "quantile",
    "quantiles_to_distribution": "quantile",
    "quantiles_to_mean": "quantile",
    "rearrange_quantiles": "quantile",
    "validate_quantile_levels": "quantile",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
