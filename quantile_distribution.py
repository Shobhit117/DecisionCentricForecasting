"""
Reconstruction of discrete demand distributions from a grid of predicted quantiles.

A quantile regression model (e.g. CatBoost with the ``MultiQuantile`` loss) returns,
for every observation, a vector of predicted quantiles ``q_1 <= ... <= q_K`` at the
levels ``tau_1 < ... < tau_K``. This module turns such a vector into a
:class:`DiscreteDistribution` over non-negative integer demand, without any
parametric assumption inside the fitted quantile range:

1. **Quantile crossing** is removed by monotone rearrangement (sorting), and the
   quantiles are clipped at zero (demand is non-negative).
2. A **continuous quantile function** ``Q(tau)`` is built by linear interpolation
   between the knots ``(tau_k, q_k)``.
3. **Tails** outside ``[tau_1, tau_K]``:

   * lower tail: linear extrapolation of the first segment down to ``tau = 0``,
     clipped at zero;
   * upper tail: exponential tail ``Q(tau) = q_K + theta * ln((1 - tau_K) / (1 - tau))``,
     with ``theta`` chosen so that the tail passes through the last two knots. The tail
     is truncated at ``tau = 1 - tol`` (consistent with ``DiscreteDistribution.from_parametric``).

4. **Discretisation**: integer demand is defined as ``D = round(Q(U))`` with
   ``U ~ Uniform(0, 1)``, i.e. ``P(D <= y) = P(Q(U) < y + 0.5)``. When the predicted
   quantiles are integers (which is what the pinball loss favours on count data), this
   places ``F(y)`` midway between the bounds implied by the discrete quantile definition.
"""
from __future__ import annotations

import numpy as np

from discrete_distribution import DiscreteDistribution
from ddlt import _quantiles_to_pmf_numba, _quantiles_to_mean_batch_numba

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

# Small offset used to evaluate the left limit P(Q(U) < x) instead of P(Q(U) <= x).
_LEFT_LIMIT_EPS = 1e-9


def validate_quantile_levels(levels: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2:
        raise ValueError("At least two quantile levels are required.")
    if np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("Quantile levels must lie strictly between 0 and 1.")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("Quantile levels must be strictly increasing.")
    return levels


def rearrange_quantiles(values: np.ndarray, lower_bound: float = 0.0) -> np.ndarray:
    """
    Removes quantile crossing by monotone rearrangement (sorting along the last axis)
    and enforces the lower bound of the support. Works on a single vector or on a
    ``(n_obs, n_quantiles)`` matrix.
    """
    values = np.sort(np.asarray(values, dtype=float), axis=-1)
    return np.maximum(values, lower_bound)


def quantiles_to_pmf(levels: np.ndarray, values: np.ndarray, tol: float = 1e-4, max_support: int | None = None) -> tuple[np.ndarray, int]:
    """
    Converts one vector of predicted quantiles into the PMF of a non-negative integer
    random variable using the Numba-compiled kernel.
    """
    levels_arr = np.ascontiguousarray(levels, dtype=np.float64)
    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    cap = -1 if max_support is None else int(max_support)
    pmf, min_val = _quantiles_to_pmf_numba(levels_arr, values_arr, float(tol), cap)
    return pmf, int(min_val)


def quantiles_to_distribution(levels: list[float] | np.ndarray, values: list[float] | np.ndarray, tol: float = 1e-4, max_support: int | None = None) -> DiscreteDistribution:
    """Builds a :class:`DiscreteDistribution` from a vector of predicted quantiles."""
    levels = validate_quantile_levels(levels)
    values = np.asarray(values, dtype=float)
    if values.shape != levels.shape:
        raise ValueError(f"Expected {len(levels)} quantile values, got {values.shape}.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Quantile values must be finite: {values}.")
    pmf, min_value = quantiles_to_pmf(levels, values, tol, max_support)
    return DiscreteDistribution(pmf, min_value)


def distribution_mean(distr: DiscreteDistribution) -> float:
    support = distr.min + distr.bin_size * np.arange(len(distr.pmf))
    return float(np.dot(support, distr.pmf))


def quantiles_to_mean(levels: np.ndarray, values_matrix: np.ndarray, tol: float = 1e-4, max_support: int | None = None) -> np.ndarray:
    """Mean of the reconstructed distribution for every row of a ``(n_obs, n_quantiles)`` matrix."""
    levels_arr = np.ascontiguousarray(validate_quantile_levels(levels), dtype=np.float64)
    values_arr = np.ascontiguousarray(np.atleast_2d(np.asarray(values_matrix, dtype=np.float64)))
    cap = -1 if max_support is None else int(max_support)
    return _quantiles_to_mean_batch_numba(levels_arr, values_arr, float(tol), cap)


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, level: float) -> np.ndarray:
    """Element-wise pinball (quantile) loss."""
    diff = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return np.maximum(level * diff, (level - 1.0) * diff)
