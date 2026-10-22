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
    random variable.

    Parameters
    ----------
    levels : np.ndarray
        Strictly increasing quantile levels in (0, 1). Assumed validated.
    values : np.ndarray
        Predicted quantiles, one per level (crossing is fixed internally).
    tol : float
        Upper-tail truncation probability.
    max_support : int, optional
        Hard cap on the largest support value (protects against extreme tails).

    Returns
    -------
    (pmf, min_value) : tuple[np.ndarray, int]
        PMF over the integers ``min_value, min_value + 1, ...``.
    """
    q = rearrange_quantiles(values)

    # Lower tail: extend the first segment linearly to tau = 0, clipped at zero.
    slope_lo = (q[1] - q[0]) / (levels[1] - levels[0])
    q_zero = max(0.0, q[0] - slope_lo * levels[0])

    # Upper tail: exponential tail through the last two knots.
    theta = (q[-1] - q[-2]) / np.log((1.0 - levels[-2]) / (1.0 - levels[-1]))
    q_upper = q[-1] + theta * np.log((1.0 - levels[-1]) / tol) if tol < 1.0 - levels[-1] else q[-1]

    y_max = int(np.floor(q_upper + 0.5))
    if max_support is not None:
        y_max = min(y_max, int(max_support))
    y_max = max(y_max, 0)

    # P(D <= y) = P(Q(U) < y + 0.5)
    x = np.arange(y_max + 1, dtype=float) + 0.5 - _LEFT_LIMIT_EPS
    knots_q = np.concatenate(([q_zero], q))
    knots_tau = np.concatenate(([0.0], levels))
    cdf = np.interp(x, knots_q, knots_tau)  # 0 below q_zero, tau_K above q_K (overwritten below)

    upper = x > q[-1]
    if theta > 0:
        cdf[upper] = 1.0 - (1.0 - levels[-1]) * np.exp(-(x[upper] - q[-1]) / theta)
    else:
        cdf[upper] = 1.0
    cdf[-1] = 1.0  # the truncated tail mass is assigned to the last support point

    pmf = np.maximum(np.diff(cdf, prepend=0.0), 0.0)
    nonzero = np.flatnonzero(pmf > 0)
    first, last = nonzero[0], nonzero[-1]
    return pmf[first:last + 1], int(first)


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
    levels = validate_quantile_levels(levels)
    values_matrix = np.atleast_2d(np.asarray(values_matrix, dtype=float))
    means = np.empty(len(values_matrix))
    for i, row in enumerate(values_matrix):
        pmf, min_value = quantiles_to_pmf(levels, row, tol, max_support)
        means[i] = min_value + np.dot(np.arange(len(pmf)), pmf)
    return means


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, level: float) -> np.ndarray:
    """Element-wise pinball (quantile) loss."""
    diff = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return np.maximum(level * diff, (level - 1.0) * diff)

