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

from probabilistic_modeling.distributions.discrete import DiscreteDistribution, PredictiveDistribution
from probabilistic_modeling.ddlt import _quantiles_to_pmf_numba, _quantiles_to_mean_batch_numba

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


def rearrange_quantiles(values: np.ndarray, lower_bound: float | None = 0.0) -> np.ndarray:
    """
    Removes quantile crossing by monotone rearrangement (sorting along the last axis)
    and optionally enforces a lower bound on the support (pass ``lower_bound=None``
    for unbounded continuous domains). Works on a single vector or on a
    ``(n_obs, n_quantiles)`` matrix.
    """
    values = np.sort(np.asarray(values, dtype=float), axis=-1)
    if lower_bound is not None and np.isfinite(lower_bound):
        return np.maximum(values, lower_bound)
    return values


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


def quantile_crps(y: np.ndarray | float, q_pred: np.ndarray, levels: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """
    Computes the Continuous Ranked Probability Score (CRPS) approximated from a
    grid of predicted quantiles via the identity ``CRPS(F, y) = 2 * int_0^1 rho_tau(y - Q(tau)) d_tau``.

    Parameters
    ----------
    y : array-like or float
        Realized observations of shape ``(n_obs,)`` or scalar.
    q_pred : np.ndarray
        Predicted quantiles of shape ``(n_obs, n_quantiles)`` or ``(n_quantiles,)``.
    levels : array-like
        Strictly increasing quantile levels in ``(0, 1)`` of length ``n_quantiles``.
    """
    tau = validate_quantile_levels(levels)
    q_arr = np.asarray(q_pred, dtype=float)
    single = q_arr.ndim == 1
    if single:
        q_arr = q_arr[None, :]
    if q_arr.shape[1] != len(tau):
        raise ValueError(f"Expected {len(tau)} quantile columns, got {q_arr.shape[1]}.")

    y_arr = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1, 1)
    diff = y_arr - q_arr  # (n_obs, K)
    losses = np.maximum(tau[None, :] * diff, (tau[None, :] - 1.0) * diff)
    # Trapezoid integration over [tau_0, tau_{K-1}] plus boundary triangles to 0 and 1:
    trapz = np.trapezoid(losses, tau, axis=1) if hasattr(np, "trapezoid") else np.trapz(losses, tau, axis=1)
    left_tail = 0.5 * tau[0] * losses[:, 0]
    right_tail = 0.5 * (1.0 - tau[-1]) * losses[:, -1]
    out = 2.0 * (trapz + left_tail + right_tail)
    return float(out[0]) if single and np.ndim(y) == 0 else out


class ContinuousQuantileDistribution(PredictiveDistribution):
    """
    Continuous predictive distribution reconstructed from a grid of quantiles
    ``(levels, values)`` without integer discretization.

    Useful for continuous domains (energy load, weight, volume, financial cashflows)
    while sharing the ``PredictiveDistribution`` interface with ``DiscreteDistribution``.
    """

    def __init__(
        self,
        levels: list[float] | tuple[float, ...] | np.ndarray,
        values: list[float] | tuple[float, ...] | np.ndarray,
        lower_bound: float | None = 0.0,
        tol: float = 1e-4,
    ):
        self.levels = validate_quantile_levels(levels)
        raw_vals = np.asarray(values, dtype=float)
        if raw_vals.shape != self.levels.shape:
            raise ValueError(f"Expected {len(self.levels)} quantile values, got {raw_vals.shape}.")
        if np.any(~np.isfinite(raw_vals)):
            raise ValueError(f"Quantile values must be finite: {raw_vals}.")
        self.values = rearrange_quantiles(raw_vals, lower_bound=lower_bound)
        self.lower_bound = lower_bound
        self.tol = float(tol)

        # Lower tail knot at tau = 0:
        slope_lo = (self.values[1] - self.values[0]) / (self.levels[1] - self.levels[0])
        q_zero = self.values[0] - slope_lo * self.levels[0]
        if lower_bound is not None and np.isfinite(lower_bound):
            q_zero = max(float(lower_bound), float(q_zero))
        self.q_zero = float(q_zero)

        # Upper exponential tail scale parameter theta:
        self.theta = float(
            (self.values[-1] - self.values[-2])
            / np.log((1.0 - self.levels[-2]) / (1.0 - self.levels[-1]))
        )
        if self.tol < 1.0 - self.levels[-1]:
            self.q_upper = float(self.values[-1] + self.theta * np.log((1.0 - self.levels[-1]) / self.tol))
        else:
            self.q_upper = float(self.values[-1])

        self._knots_tau = np.concatenate(([0.0], self.levels))
        self._knots_q = np.concatenate(([self.q_zero], self.values))

    def quantile(self, q: list | np.ndarray | float) -> np.ndarray:
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("Quantiles must be between 0 and 1.")
        q_clipped = np.clip(q_arr, 0.0, 1.0 - self.tol)
        out = np.interp(q_clipped, self._knots_tau, self._knots_q)
        upper = q_clipped > self.levels[-1]
        if np.any(upper) and self.theta > 0.0:
            out[upper] = self.values[-1] + self.theta * np.log(
                (1.0 - self.levels[-1]) / np.maximum(1.0 - q_clipped[upper], self.tol)
            )
        return out

    def cdf_at(self, x: list | np.ndarray | float) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        cdf = np.interp(x_arr, self._knots_q, self._knots_tau, left=0.0, right=1.0)
        upper = x_arr > self.values[-1]
        if np.any(upper):
            if self.theta > 0.0:
                cdf[upper] = 1.0 - (1.0 - self.levels[-1]) * np.exp(-(x_arr[upper] - self.values[-1]) / self.theta)
            else:
                cdf[upper] = 1.0
        cdf[x_arr >= self.q_upper] = 1.0
        return np.clip(cdf, 0.0, 1.0)

    def mean(self) -> float:
        # Analytical piecewise-linear integral on [0, tau_K] + truncated exponential tail on [tau_K, 1 - tol]:
        dtau = np.diff(self._knots_tau)
        mid_q = 0.5 * (self._knots_q[:-1] + self._knots_q[1:])
        interior = float(np.dot(dtau, mid_q))
        tau_last = float(self.levels[-1])
        if self.tol < 1.0 - tau_last:
            rem = (1.0 - tau_last) - self.tol
            tail = rem * self.values[-1] + self.theta * (
                rem - self.tol * np.log((1.0 - tau_last) / self.tol)
            )
            total_mass = 1.0 - self.tol
            return float((interior + tail) / total_mass)
        return float(interior / tau_last)

    def variance(self) -> float:
        u = np.linspace(0.5 * self.tol, 1.0 - self.tol, 512)
        q_vals = self.quantile(u)
        mu = float(np.mean(q_vals))
        return float(np.mean((q_vals - mu) ** 2))

    def crps(self, y: float | list | np.ndarray) -> float | np.ndarray:
        return quantile_crps(y, self.values, self.levels)

    def sample(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        u = gen.uniform(0.0, 1.0 - self.tol, size=size)
        return self.quantile(u)

    def to_discrete(self, max_support: int | None = None) -> DiscreteDistribution:
        """Discretizes this continuous quantile distribution onto non-negative integers."""
        return quantiles_to_distribution(self.levels, np.maximum(self.values, 0.0), tol=self.tol, max_support=max_support)
