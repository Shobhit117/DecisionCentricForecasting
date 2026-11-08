"""
Reconstruction of discrete and continuous predictive distributions from a grid of quantiles,
plus proper quantile scoring rules (pinball loss and quantile-grid CRPS).

What this module does
---------------------
A quantile regression model (e.g. :class:`CatBoostQuantileForecaster` or
:class:`EmpiricalQuantileForecaster`) predicts a vector of quantiles ``q_1 <= ... <= q_K``
at probability levels ``0 < tau_1 < ... < tau_K < 1`` for each ``(series, period, horizon)``.
This module bridges those quantile vectors to full probability distributions:

1. **Quantile crossing removal** (:func:`rearrange_quantiles`): sorts raw predictions along the
   quantile axis (Chernozhukov et al., 2010 monotone rearrangement) and clips at ``lower_bound`` (0).
2. **Interior interpolation**: linear interpolation between the knots ``(tau_k, q_k)``.
3. **Tail extrapolation**:
   - *Lower tail* (`[0, tau_1]`): linear extrapolation of the first segment down to ``tau = 0``,
     clipped at ``lower_bound``.
   - *Upper tail* (`[tau_K, 1 - tol]`): exponential tail
     ``Q(tau) = q_K + theta * ln((1 - tau_K) / (1 - tau))`` with scale ``theta`` matched to the
     last two knots ``(tau_{K-1}, q_{K-1})`` and ``(tau_K, q_K)``, truncated at ``1 - tol``.
4. **Discretization** (:func:`quantiles_to_pmf`, :func:`quantiles_to_distribution`):
   Defines integer demand ``D = round(Q(U))`` for ``U ~ Uniform(0, 1)``, so
   ``P(D <= y) = P(Q(U) < y + 0.5)``, delegated to the compiled Numba kernel
   :func:`probabilistic_modeling.ddlt._quantiles_to_pmf_numba`.
5. **Continuous counterpart** (:class:`ContinuousQuantileDistribution`):
   Preserves the continuous piecewise-linear + exponential-tail quantile function without
   integer rounding (for continuous domains like weight, volume, energy, cashflow).

Where this module is used
-------------------------
- ``models/catboost_quantile.py`` & ``models/forecasters.py`` (training, prediction, evaluation)
- ``models/baseline.py`` (default quantile grid and validation)
- ``ddlt.py`` (validating quantile levels in :meth:`DDLTCalculator.from_quantiles_group`)
"""
from __future__ import annotations

import numpy as np

from probabilistic_modeling.distributions.discrete import DiscreteDistribution, PredictiveDistribution
from probabilistic_modeling.ddlt import _quantiles_to_pmf_numba, _quantiles_to_mean_batch_numba

# Default 12-level quantile grid spanning the body (0.05..0.90) and upper safety-stock tail (0.95, 0.99).
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

# Small offset used to evaluate the left limit P(Q(U) < x) instead of P(Q(U) <= x).
_LEFT_LIMIT_EPS = 1e-9


def validate_quantile_levels(levels: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """
    Validate that ``levels`` is a 1D strictly increasing array of at least 2 probabilities in ``(0, 1)``.

    Why it is needed
    ----------------
    At least two quantile knots are required to estimate both the lower-tail slope and the
    upper-tail exponential scale ``theta``. Used across ``quantile.py``, ``catboost_quantile.py``,
    ``forecasters.py``, ``baseline.py``, and ``ddlt.py``.
    """
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
    Remove quantile crossing by monotone rearrangement (sorting along the last axis) and
    optionally clip below ``lower_bound`` (`0.0` by default for non-negative demand).

    Why it is needed
    ----------------
    Even with CatBoost's joint ``MultiQuantile`` loss, unconstrained tree predictions can
    occasionally violate ``q(tau_1) <= q(tau_2)`` for ``tau_1 < tau_2``. Sorting along the
    quantile axis restores monotonicity and is guaranteed by theory (Chernozhukov et al., 2010)
    to weakly improve pinball loss.

    Where it is used
    ----------------
    - :func:`probabilistic_modeling.models.catboost_quantile.predict_catboost_quantile`
    - :class:`ContinuousQuantileDistribution`
    """
    values = np.sort(np.asarray(values, dtype=float), axis=-1)
    if lower_bound is not None and np.isfinite(lower_bound):
        return np.maximum(values, lower_bound)
    return values


def quantiles_to_pmf(levels: np.ndarray, values: np.ndarray, tol: float = 1e-4, max_support: int | None = None) -> tuple[np.ndarray, int]:
    """
    Convert one 1D vector of predicted quantiles ``values`` at ``levels`` into ``(pmf, min_value)``
    of a non-negative integer distribution via the compiled Numba kernel :func:`_quantiles_to_pmf_numba`.

    Where it is used
    ----------------
    Called by :func:`quantiles_to_distribution`.
    """
    levels_arr = np.ascontiguousarray(levels, dtype=np.float64)
    values_arr = np.ascontiguousarray(values, dtype=np.float64)
    cap = -1 if max_support is None else int(max_support)
    pmf, min_val = _quantiles_to_pmf_numba(levels_arr, values_arr, float(tol), cap)
    return pmf, int(min_val)


def quantiles_to_distribution(levels: list[float] | np.ndarray, values: list[float] | np.ndarray, tol: float = 1e-4, max_support: int | None = None) -> DiscreteDistribution:
    """
    Build a :class:`DiscreteDistribution` from a 1D vector of predicted quantiles ``values`` at ``levels``.

    The extrapolated upper exponential tail is truncated at ``1 - tol`` (recorded as
    ``truncation_tol=tol`` on the returned distribution unless an explicit ``max_support`` cap
    is applied).

    Where it is used
    ----------------
    - :meth:`ContinuousQuantileDistribution.to_discrete`
    - Public API for converting custom quantile forecasts into :class:`DiscreteDistribution` objects.
    """
    levels = validate_quantile_levels(levels)
    values = np.asarray(values, dtype=float)
    if values.shape != levels.shape:
        raise ValueError(f"Expected {len(levels)} quantile values, got {values.shape}.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Quantile values must be finite: {values}.")
    pmf, min_value = quantiles_to_pmf(levels, values, tol, max_support)
    # An explicit ``max_support`` cap may cut off more mass than ``tol`` -> unknown truncation.
    return DiscreteDistribution(pmf, min_value, truncation_tol=float(tol) if max_support is None else None)


def distribution_mean(distr: DiscreteDistribution) -> float:
    """Return the expected value ``E[X]`` of a :class:`DiscreteDistribution`."""
    support = distr.min + distr.bin_size * np.arange(len(distr.pmf))
    return float(np.dot(support, distr.pmf))


def quantiles_to_mean(levels: np.ndarray, values_matrix: np.ndarray, tol: float = 1e-4, max_support: int | None = None) -> np.ndarray:
    """
    Compute the expected value ``E[D]`` of the reconstructed discrete distribution for every row
    of a ``(n_obs, n_quantiles)`` matrix using the parallel Numba kernel :func:`_quantiles_to_mean_batch_numba`.

    Where it is used
    ----------------
    Called by :meth:`CatBoostQuantileForecaster.predict` to populate the point-forecast column
    ``target_t_plus_<h>_mu`` from the predicted quantile grid.
    """
    levels_arr = np.ascontiguousarray(validate_quantile_levels(levels), dtype=np.float64)
    values_arr = np.ascontiguousarray(np.atleast_2d(np.asarray(values_matrix, dtype=np.float64)))
    cap = -1 if max_support is None else int(max_support)
    return _quantiles_to_mean_batch_numba(levels_arr, values_arr, float(tol), cap)


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, level: float) -> np.ndarray:
    """
    Element-wise pinball (check / quantile) loss at probability level ``level`` (`tau` in `(0, 1)`):

        rho_tau(y - q) = max(tau * (y - q), (tau - 1) * (y - q))

    Why it is needed
    ----------------
    The unique proper scoring rule whose expected value is minimized by the true ``tau``-quantile
    of ``Y``. Used by :func:`evaluate_quantile_forecasts`.
    """
    diff = np.asarray(y, dtype=float) - np.asarray(q_pred, dtype=float)
    return np.maximum(level * diff, (level - 1.0) * diff)


def quantile_crps(y: np.ndarray | float, q_pred: np.ndarray, levels: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    """
    Compute the Continuous Ranked Probability Score (CRPS) from a grid of predicted quantiles
    using the Gneiting & Ranjan (2011) integral identity:

        CRPS(F, y) = 2 * int_0^1 rho_tau(y - Q(tau)) d_tau

    What it does
    ------------
    Evaluates the pinball loss ``rho_tau(y - q_pred)`` at each grid knot ``tau_k`` and integrates
    over ``[0, 1]`` using the composite trapezoidal rule on ``[tau_0, tau_{K-1}]`` plus linear
    boundary triangles on ``[0, tau_0]`` and ``[tau_{K-1}, 1]`` (where pinball loss vanishes at
    the endpoints ``tau = 0`` and ``tau = 1``).

    Where it is used
    ----------------
    - :func:`probabilistic_modeling.models.forecasters.evaluate_quantile_forecasts`
    - :meth:`ContinuousQuantileDistribution.crps`
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
    Continuous predictive distribution reconstructed from a quantile grid ``(levels, values)``
    without integer discretization.

    What it does
    ------------
    Models ``Q(tau)`` as piecewise-linear across ``[0, tau_0, ..., tau_{K-1}]`` and exponential
    on ``(tau_{K-1}, 1 - tol]`` with scale parameter ``theta``. Implements the full
    :class:`PredictiveDistribution` interface (`mean`, `variance`, `std`, `quantile`, `cdf_at`,
    `crps`, `sample`) and can be discretized onto non-negative integers via :meth:`to_discrete`.

    Why it is needed
    ----------------
    Allows using the same quantile-reconstruction math for continuous quantities (e.g. weight,
    volume, energy load, revenue) as for discrete unit counts.
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
        """Evaluate the continuous quantile function ``Q(q)`` at probability level(s) ``q`` in ``[0, 1]``."""
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
        """Evaluate the continuous CDF ``F(x) = P(X <= x)`` at one or more values ``x``."""
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
        """
        Analytical expected value ``E[X] = (1 / (1 - tol)) * int_0^{1-tol} Q(tau) d_tau``,
        combining trapezoidal integration on ``[0, tau_{K-1}]`` with the closed-form integral
        of the exponential tail on ``[tau_{K-1}, 1 - tol]``.
        """
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
        """Numerical variance ``Var(X) = int_0^1 (Q(u) - E[X])^2 du`` over a 512-point grid."""
        u = np.linspace(0.5 * self.tol, 1.0 - self.tol, 512)
        q_vals = self.quantile(u)
        mu = float(np.mean(q_vals))
        return float(np.mean((q_vals - mu) ** 2))

    def crps(self, y: float | list | np.ndarray) -> float | np.ndarray:
        """Evaluate CRPS against observation(s) ``y`` via :func:`quantile_crps`."""
        return quantile_crps(y, self.values, self.levels)

    def sample(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Draw ``size`` continuous random samples via inverse-CDF sampling ``Q(U)``."""
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        u = gen.uniform(0.0, 1.0 - self.tol, size=size)
        return self.quantile(u)

    def to_discrete(self, max_support: int | None = None) -> DiscreteDistribution:
        """Discretize this continuous quantile distribution onto non-negative integers."""
        return quantiles_to_distribution(self.levels, np.maximum(self.values, 0.0), tol=self.tol, max_support=max_support)
