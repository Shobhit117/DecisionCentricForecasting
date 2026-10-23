"""
Numba-accelerated Demand-During-Lead-Time (DDLT) calculation and discrete convolution kernels.

Provides:
- Fast 1D discrete convolution and random-sum mixture kernels compiled with Numba,
  requiring zero 2D matrix allocations and zero temporary CDF allocations.
- Fast O(1)-auxiliary-memory quantile lookup directly on a PMF.
- ``DDLTCalculator``: a model-agnostic calculator that converts daily predictive
  distributions (from quantiles, GLM parameters, or ``DiscreteDistribution`` objects)
  and a lead-time distribution into DDLT distributions or target inventory quantities.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List as NumbaList

from discrete_distribution import DiscreteDistribution

logger = logging.getLogger(__name__)

_LEFT_LIMIT_EPS = 1e-9
DEFAULT_TAIL_TOL = 1e-9


@njit(cache=True)
def _convolve_1d_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1D discrete convolution of two PMFs."""
    na = len(a)
    nb = len(b)
    out = np.zeros(na + nb - 1, dtype=np.float64)
    for i in range(na):
        ai = a[i]
        if ai > 0.0:
            for j in range(nb):
                out[i + j] += ai * b[j]
    return out


@njit(cache=True)
def _trim_pmf_numba(
    pmf: np.ndarray,
    min_val: int,
    bin_size: int,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Trims leading and trailing bins with probability ``<= tail_tol``, shifts
    ``min_val`` when leading bins are removed, and renormalizes the PMF.
    """
    n = len(pmf)
    if n <= 1 or tail_tol < 0.0:
        return pmf, min_val

    first = 0
    while first < n - 1 and pmf[first] <= tail_tol:
        first += 1

    last = n - 1
    while last > first and pmf[last] <= tail_tol:
        last -= 1

    if first == 0 and last == n - 1:
        return pmf, min_val

    trimmed = pmf[first : last + 1].copy()
    s = np.sum(trimmed)
    if s > 0.0:
        trimmed /= s
    return trimmed, min_val + first * bin_size


@njit(cache=True)
def _pmf_quantile_scalar_numba(pmf: np.ndarray, min_val: int, bin_size: int, q: float) -> int:
    """
    Computes the discrete quantile at level ``q`` directly from ``pmf`` in O(1) auxiliary
    memory (without allocating or caching a CDF array).
    Matches ``np.searchsorted(np.cumsum(pmf), q, side='left')``.
    """
    n = len(pmf)
    if q <= 0.0 or n <= 1:
        return min_val
    csum = 0.0
    for i in range(n):
        csum += pmf[i]
        if csum >= q:
            return min_val + i * bin_size
    return min_val + (n - 1) * bin_size


@njit(cache=True)
def _pmf_quantiles_numba(pmf: np.ndarray, min_val: int, bin_size: int, q_arr: np.ndarray) -> np.ndarray:
    """Vectorized quantile evaluation on a single PMF with O(1) extra memory."""
    m = len(q_arr)
    out = np.empty(m, dtype=np.int64)
    for k in range(m):
        out[k] = _pmf_quantile_scalar_numba(pmf, min_val, bin_size, q_arr[k])
    return out


@njit(cache=True)
def _batch_pmf_quantiles_numba(
    pmfs: NumbaList,
    mins: np.ndarray,
    bin_sizes: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Evaluates one quantile ``alphas[i]`` per product distribution in compiled code."""
    n = len(pmfs)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = _pmf_quantile_scalar_numba(pmfs[i], int(mins[i]), int(bin_sizes[i]), float(alphas[i]))
    return out


@njit(cache=True)
def _random_sum_aligned_numba(
    lt_pmf: np.ndarray,
    lt_min: int,
    rv_pmfs: NumbaList,
    rv_mins: np.ndarray,
    bin_size: int,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Computes the PMF and minimum support value of the random sum
    ``S = X_1 + ... + X_L`` where ``L ~ (lt_pmf, lt_min, bin_size=1)`` and each ``X_j``
    already shares ``bin_size`` with ``rv_mins[j]`` divisible by ``bin_size``.

    Trims negligible tails (``<= tail_tol``) after each convolution step so intermediate
    support sizes stay compact across long lead times, and accumulates weighted
    convolutions into a single 1D output vector without allocating a 2D matrix.
    """
    lt_max = lt_min + len(lt_pmf) - 1
    n_terms = len(lt_pmf)

    saved_pmfs = NumbaList()
    saved_mins = np.empty(n_terms, dtype=np.int64)
    saved_weights = np.empty(n_terms, dtype=np.float64)
    n_saved = 0

    curr_pmf = np.ones(1, dtype=np.float64)
    curr_min = 0
    overall_min = 0
    overall_max = 0

    for n in range(lt_max + 1):
        if n >= lt_min:
            w = lt_pmf[n - lt_min]
            if w > 0.0:
                c_max = curr_min + bin_size * (len(curr_pmf) - 1)
                if n_saved == 0:
                    overall_min = curr_min
                    overall_max = c_max
                else:
                    if curr_min < overall_min:
                        overall_min = curr_min
                    if c_max > overall_max:
                        overall_max = c_max
                saved_pmfs.append(curr_pmf)
                saved_mins[n_saved] = curr_min
                saved_weights[n_saved] = w
                n_saved += 1
        if n < lt_max:
            r_pmf, r_min = _trim_pmf_numba(rv_pmfs[n], int(rv_mins[n]), bin_size, tail_tol)
            curr_pmf = _convolve_1d_numba(curr_pmf, r_pmf)
            s = np.sum(curr_pmf)
            if s > 0.0:
                curr_pmf /= s
            curr_min += r_min
            curr_pmf, curr_min = _trim_pmf_numba(curr_pmf, curr_min, bin_size, tail_tol)

    if n_saved == 0:
        return np.ones(1, dtype=np.float64), 0

    width = (overall_max - overall_min) // bin_size + 1
    out_pmf = np.zeros(width, dtype=np.float64)
    for idx in range(n_saved):
        p = saved_pmfs[idx]
        w = saved_weights[idx]
        start = (saved_mins[idx] - overall_min) // bin_size
        for k in range(len(p)):
            out_pmf[start + k] += w * p[k]

    total = np.sum(out_pmf)
    if total > 0.0:
        out_pmf /= total
    out_pmf, overall_min = _trim_pmf_numba(out_pmf, overall_min, bin_size, tail_tol)
    return out_pmf, overall_min


@njit(cache=True)
def _quantiles_to_pmf_numba(
    levels: np.ndarray,
    values: np.ndarray,
    tol: float = 1e-4,
    max_support: int = -1,
) -> tuple[np.ndarray, int]:
    """Numba-compiled conversion of a quantile vector into ``(pmf, min_value)``."""
    k_len = len(levels)
    q = np.sort(values.astype(np.float64))
    for i in range(k_len):
        if q[i] < 0.0:
            q[i] = 0.0

    slope_lo = (q[1] - q[0]) / (levels[1] - levels[0])
    q_zero = q[0] - slope_lo * levels[0]
    if q_zero < 0.0:
        q_zero = 0.0

    theta = (q[k_len - 1] - q[k_len - 2]) / np.log((1.0 - levels[k_len - 2]) / (1.0 - levels[k_len - 1]))
    if tol < 1.0 - levels[k_len - 1]:
        q_upper = q[k_len - 1] + theta * np.log((1.0 - levels[k_len - 1]) / tol)
    else:
        q_upper = q[k_len - 1]

    y_max = int(np.floor(q_upper + 0.5))
    if max_support >= 0 and y_max > max_support:
        y_max = max_support
    if y_max < 0:
        y_max = 0

    knots_q = np.empty(k_len + 1, dtype=np.float64)
    knots_tau = np.empty(k_len + 1, dtype=np.float64)
    knots_q[0] = q_zero
    knots_tau[0] = 0.0
    for i in range(k_len):
        knots_q[i + 1] = q[i]
        knots_tau[i + 1] = levels[i]

    x = np.arange(y_max + 1, dtype=np.float64) + 0.5 - _LEFT_LIMIT_EPS
    cdf = np.interp(x, knots_q, knots_tau)

    q_last = q[k_len - 1]
    tau_last = levels[k_len - 1]
    for i in range(y_max + 1):
        if x[i] > q_last:
            if theta > 0.0:
                cdf[i] = 1.0 - (1.0 - tau_last) * np.exp(-(x[i] - q_last) / theta)
            else:
                cdf[i] = 1.0
    cdf[y_max] = 1.0

    pmf_full = np.empty(y_max + 1, dtype=np.float64)
    prev = 0.0
    first = -1
    last = 0
    for i in range(y_max + 1):
        diff = cdf[i] - prev
        prev = cdf[i]
        if diff < 0.0:
            diff = 0.0
        pmf_full[i] = diff
        if diff > 0.0:
            if first == -1:
                first = i
            last = i

    if first == -1:
        out = np.ones(1, dtype=np.float64)
        return out, 0

    out = pmf_full[first : last + 1].copy()
    s = np.sum(out)
    if s > 0.0:
        out /= s
    return out, first


@njit(parallel=True, cache=True)
def _quantiles_to_mean_batch_numba(
    levels: np.ndarray,
    values_matrix: np.ndarray,
    tol: float = 1e-4,
    max_support: int = -1,
) -> np.ndarray:
    """Parallel Numba computation of the distribution mean for every row of ``values_matrix``."""
    n_rows = values_matrix.shape[0]
    means = np.empty(n_rows, dtype=np.float64)
    for i in prange(n_rows):
        pmf, min_val = _quantiles_to_pmf_numba(levels, values_matrix[i], tol, max_support)
        acc = 0.0
        for k in range(len(pmf)):
            acc += (min_val + k) * pmf[k]
        means[i] = acc
    return means


@njit(cache=True)
def _ddlt_from_quantiles_row_numba(
    lt_pmf: np.ndarray,
    lt_min: int,
    levels: np.ndarray,
    row_quantiles: np.ndarray,
    default_pmf: np.ndarray,
    default_min: int,
    tol: float = 1e-4,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Converts one item's multi-horizon quantile matrix ``(horizon, n_quantiles)`` into
    daily PMFs and computes the random-sum DDLT distribution in a single compiled call.
    """
    horizon = row_quantiles.shape[0]
    lt_max = lt_min + len(lt_pmf) - 1
    total_days = lt_max if lt_max > horizon else horizon

    rv_pmfs = NumbaList()
    rv_mins = np.empty(total_days, dtype=np.int64)
    for h in range(horizon):
        pmf, min_val = _quantiles_to_pmf_numba(levels, row_quantiles[h], tol, -1)
        rv_pmfs.append(pmf)
        rv_mins[h] = min_val
    for h in range(horizon, total_days):
        rv_pmfs.append(default_pmf)
        rv_mins[h] = default_min

    return _random_sum_aligned_numba(lt_pmf, lt_min, rv_pmfs, rv_mins, 1, tail_tol)


class DDLTCalculator:
    """
    Model-agnostic Demand-During-Lead-Time (DDLT) calculator.

    Combines multi-horizon daily demand forecasts (represented either as quantile grids,
    GLM ``(mu, alpha)`` parameters, or lists of :class:`DiscreteDistribution`) with a
    lead-time distribution and review cycle time.
    """

    def __init__(
        self,
        lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution],
        cycle_time: int = 0,
        lead_time_keys: list[str] | None = None,
        tol: float = 1e-4,
        tail_tol: float = DEFAULT_TAIL_TOL,
    ):
        if cycle_time < 0:
            raise ValueError("'cycle_time' must be non-negative.")
        self.lead_time = lead_time
        self.cycle_time = cycle_time
        self.lead_time_keys = list(lead_time_keys) if lead_time_keys else []
        self.tol = tol
        self.tail_tol = tail_tol

    def _shifted_lead_time(self, lt: DiscreteDistribution) -> DiscreteDistribution:
        lt_with_cycle = lt.copy()
        lt_with_cycle.shift(self.cycle_time)
        if lt_with_cycle.min < 0:
            raise ValueError("Lead time plus cycle time must have non-negative support.")
        if lt_with_cycle.bin_size != 1:
            lt_with_cycle.split_bins()
        return lt_with_cycle

    @staticmethod
    def _build_default_demand(row: pd.Series, max_lt: int, horizon: int) -> DiscreteDistribution | None:
        if max_lt <= horizon:
            return None
        best_fit = row.get("best_fit")
        if pd.isna(best_fit) or best_fit is None:
            raise ValueError(
                f"Lead time + cycle time ({max_lt}) exceeds forecast horizon ({horizon}), "
                "and no fallback 'best_fit' distribution is present."
            )
        params = tuple(
            row[c]
            for c in ("distr_param_0", "distr_param_1", "distr_param_2")
            if c in row and not pd.isna(row[c])
        )
        return DiscreteDistribution.from_parametric(str(best_fit), params)

    def from_quantiles_group(
        self,
        gdf: pd.DataFrame,
        lead_time: DiscreteDistribution,
        target_cols: list[str],
        quantiles: tuple[float, ...] | list[float] | np.ndarray,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        period_col: str = "day",
    ) -> pd.DataFrame:
        """Computes DDLT from multi-horizon quantile columns for one lead-time group."""
        from catboost_quantile import quantile_col_name
        from quantile_distribution import validate_quantile_levels

        t0 = time.perf_counter()
        if pred_first_day:
            first_day = gdf[period_col].min()
            gdf = gdf[gdf[period_col] == first_day].copy()
        else:
            gdf = gdf.copy()

        levels = validate_quantile_levels(quantiles)
        horizon = len(target_cols)
        if horizon == 0:
            raise ValueError("At least one forecast horizon is required.")

        if not return_distribution:
            service_levels_arr = gdf["service_level"].to_numpy(dtype=float)
            if np.any((service_levels_arr < 0) | (service_levels_arr > 1)):
                raise ValueError("'service_levels' values must be between 0 and 1.")

        q_cols = [quantile_col_name(t, q) for t in target_cols for q in levels]
        missing_cols = [c for c in q_cols if c not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required to compute target inventory: {missing_cols}.")

        lt_with_cycle = self._shifted_lead_time(lead_time)
        lt_pmf = np.ascontiguousarray(lt_with_cycle.pmf, dtype=np.float64)
        lt_min = int(lt_with_cycle.min)
        needs_default = lt_with_cycle.max > horizon
        if needs_default:
            param_cols = ["best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
            missing_param = [c for c in param_cols if c not in gdf.columns]
            if missing_param:
                raise ValueError(f"Missing columns required to compute target inventory: {missing_param}.")

        logger.info(
            "Computing Numba DDLT from quantiles for %d rows (horizon=%d, quantiles=%d, L+R support=[%d, %d])...",
            len(gdf),
            horizon,
            len(levels),
            lt_with_cycle.min,
            lt_with_cycle.max,
        )

        num_q = len(levels)
        q_matrix = np.ascontiguousarray(
            gdf[q_cols].to_numpy(dtype=np.float64).reshape(len(gdf), horizon, num_q)
        )
        dummy_pmf = np.ones(1, dtype=np.float64)

        results = []
        for row_num in range(len(gdf)):
            row_quantiles = q_matrix[row_num]
            if np.any(~np.isfinite(row_quantiles)):
                raise ValueError(f"Invalid CatBoost quantile forecast for row {row_num}: {row_quantiles}.")

            if needs_default:
                default_distr = self._build_default_demand(gdf.iloc[row_num], lt_with_cycle.max, horizon)
                default_pmf = np.ascontiguousarray(default_distr.pmf, dtype=np.float64)
                default_min = int(default_distr.min)
            else:
                default_pmf = dummy_pmf
                default_min = 0

            ddlt_pmf, ddlt_min = _ddlt_from_quantiles_row_numba(
                lt_pmf, lt_min, levels, row_quantiles, default_pmf, default_min, self.tol, self.tail_tol
            )
            if return_distribution:
                results.append(DiscreteDistribution(ddlt_pmf, ddlt_min, 1))
            else:
                q_val = _pmf_quantile_scalar_numba(ddlt_pmf, ddlt_min, 1, float(service_levels_arr[row_num]))
                results.append(int(q_val))

        if return_distribution:
            gdf["demand_distribution"] = results
        else:
            gdf["target_inventory"] = results
        logger.info("DDLT from quantiles completed for %d rows in %.2fs", len(gdf), time.perf_counter() - t0)
        return gdf

    def from_glm_group(
        self,
        gdf: pd.DataFrame,
        lead_time: DiscreteDistribution,
        target_cols: list[str],
        glm_mu_cols: list[str] | None = None,
        glm_alpha_cols: list[str] | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        period_col: str = "day",
    ) -> pd.DataFrame:
        """Computes DDLT from GLM ``(mu, alpha)`` columns for one lead-time group."""
        t0 = time.perf_counter()
        if pred_first_day:
            first_day = gdf[period_col].min()
            gdf = gdf[gdf[period_col] == first_day].copy()
        else:
            gdf = gdf.copy()

        if glm_mu_cols is None:
            glm_mu_cols = [f"{col_name}_mu" for col_name in target_cols]
        if glm_alpha_cols is None:
            glm_alpha_cols = [f"{col_name}_alpha" for col_name in target_cols]
        horizon = len(glm_mu_cols)
        if horizon != len(glm_alpha_cols):
            raise ValueError("Number of 'mu' and 'alpha' columns do not match.")
        if horizon == 0:
            raise ValueError("At least one GLM forecast horizon is required.")

        if not return_distribution:
            service_levels_arr = gdf["service_level"].to_numpy(dtype=float)
            if np.any((service_levels_arr < 0) | (service_levels_arr > 1)):
                raise ValueError("'service_levels' values must be between 0 and 1.")

        lt_with_cycle = self._shifted_lead_time(lead_time)
        needs_default = lt_with_cycle.max > horizon
        param_cols = ["best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
        required_cols = glm_mu_cols + glm_alpha_cols + (param_cols if needs_default else [])
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required to compute target inventory: {missing_cols}.")

        logger.info(
            "Computing DDLT from GLM parameters for %d rows (horizon=%d, L+R support=[%d, %d])...",
            len(gdf),
            horizon,
            lt_with_cycle.min,
            lt_with_cycle.max,
        )

        mu_arr = gdf[glm_mu_cols].to_numpy(dtype=float)
        alpha_arr = gdf[glm_alpha_cols].to_numpy(dtype=float)

        results = []
        for row_num in range(len(gdf)):
            mu = mu_arr[row_num]
            alpha = alpha_arr[row_num]
            default_demand = (
                self._build_default_demand(gdf.iloc[row_num], lt_with_cycle.max, horizon)
                if needs_default
                else None
            )

            demand = []
            for curr_mu, curr_alpha in zip(mu, alpha):
                if pd.isna(curr_mu) or curr_mu < 0:
                    raise ValueError(f"Invalid GLM mean for row {row_num}: {curr_mu}.")
                if pd.isna(curr_alpha) or curr_alpha <= 0:
                    demand.append(DiscreteDistribution.from_parametric("poisson", (float(curr_mu), 0), tol=self.tol))
                else:
                    n = 1.0 / curr_alpha
                    p = 1.0 / (1.0 + curr_mu * curr_alpha)
                    demand.append(DiscreteDistribution.from_parametric("nbinom", (n, p, 0), tol=self.tol))

            demand_during_lead_time = lt_with_cycle.random_sum(demand, default_demand, tail_tol=self.tail_tol)
            if return_distribution:
                results.append(demand_during_lead_time)
            else:
                results.append(demand_during_lead_time.quantile(service_levels_arr[row_num]).item())

        if return_distribution:
            gdf["demand_distribution"] = results
        else:
            gdf["target_inventory"] = results
        logger.info("DDLT from GLM completed for %d rows in %.2fs", len(gdf), time.perf_counter() - t0)
        return gdf
