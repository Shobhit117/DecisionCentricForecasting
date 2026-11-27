"""
Forecast evaluation framework: multi-quantile WMAPE, discrete PIT distributions, and M5 sectioning.

What this module does
---------------------
1. **Rolling M5 dataset sectioning** (:class:`M5Section`, :func:`build_m5_sections`):
   Slices a multi-year daily sales panel (such as the 1,941-day M5 dataset) into clean
   non-overlapping or rolling sections where each section uses ``train_days`` (default 365,
   ~1 year) of training observations and evaluates forecasts over the following ``test_days``
   (default 14) days.
2. **Unified quantile extraction across all model families** (:func:`populate_model_quantiles`):
   - Quantile forecasters (:class:`CatBoostQuantileForecaster`, :class:`EmpiricalQuantileForecaster`)
     natively output ``<target>_q<level>`` columns.
   - Parametric count forecasters (:class:`GLMForecaster` with ``hurdle=False`` or ``hurdle=True``)
     output ``<target>_mu`` and ``<target>_alpha`` (plus ``<target>_pi`` and ``<target>_lambda``
     when ``hurdle=True``). :func:`populate_model_quantiles` evaluates the exact discrete
     Poisson / NB2 / Zero-Hurdle PMF in parallel via Numba and populates ``<target>_q<level>``
     columns so all models share an identical quantile schema.
3. **Quantile WMAPE & Pinball evaluation** (:func:`evaluate_quantile_wmape`):
   Evaluates every predicted quantile ``q in quantiles`` (plus the point forecast ``<target>_mu``)
   across each individual horizon ``target_t_plus_h``, pooled across all horizons
   (``target="all_horizons"``), and summed across the forecast horizon per row
   (``target="horizon_sum"``), reporting:
   - ``wmape`` (``100 * sum(|y - q_hat|) / sum(y)``),
   - ``under_wmape`` (``100 * sum(max(y - q_hat, 0)) / sum(y)``),
   - ``over_wmape`` (``100 * sum(max(q_hat - y, 0)) / sum(y)``),
   - ``pinball_loss`` (mean quantile check loss ``mean(q * max(y - q_hat, 0) + (1 - q) * max(q_hat - y, 0))``),
   - ``pinball_wmape`` (``100 * 2 * sum(pinball) / sum(y)``, which equals ``wmape`` at ``q = 0.5``),
   - ``coverage`` (empirical fraction ``P(Y <= q_hat)``).
4. **Probability Integral Transform (PIT) evaluation** (:class:`ForecastPITResult`,
   :func:`evaluate_forecast_pit`):
   Evaluates the non-randomized discrete PIT of Czado, Gneiting & Held (2009) across all model
   families (parametric GLM, Zero-Hurdle GLM, and non-parametric quantile forecasters via
   piecewise-linear CDF reconstruction) per horizon and pooled across horizons, reporting
   PIT histograms, PIT coverage at target service levels, and PIT uniformity diagnostics
   (``pit_mean``, ``pit_std``, ``pit_mae``, ``pit_rmse``, ``pit_ks``, ``pit_tail_lower``,
   ``pit_tail_upper``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import warnings

import numpy as np
import pandas as pd
from numba import njit, prange

from probabilistic_modeling._panel import DenseHistory
from probabilistic_modeling.ddlt import (
    DEFAULT_SUPPORT_TOL,
    _count_pmf_truncated_numba,
    _hurdle_count_pmf_truncated_numba,
    _pmf_crps_scalar_numba,
    _pmf_quantile_scalar_numba,
    _pmf_quantiles_numba,
    _quantiles_to_pmf_numba,
)
from probabilistic_modeling.distributions.quantile import (
    DEFAULT_QUANTILES,
    validate_quantile_levels,
)
from probabilistic_modeling.evaluation.ddlt_calibration import (
    DEFAULT_CALIBRATION_LEVELS,
    _pit_cdf,
)
from probabilistic_modeling.models.catboost_quantile import quantile_col_name


# ---------------------------------------------------------------------------
# 1. M5 Rolling Section Builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class M5Section:
    """
    Specification of a single train/test temporal section on a daily panel.

    Attributes
    ----------
    section_id : int
        1-based section index.
    name : str
        Human-readable identifier (e.g. ``"sec_01_d1_379"``).
    train_start_day : int
        First inclusive training day in the section.
    train_end_day : int
        Last inclusive training day in the section (`train_start_day + train_days - 1`).
    test_start_day : int
        First inclusive test day (`train_end_day + 1`).
    test_end_day : int
        Last inclusive test day (`test_start_day + test_days - 1`).
    """

    section_id: int
    name: str
    train_start_day: int
    train_end_day: int
    test_start_day: int
    test_end_day: int

    @property
    def train_days(self) -> int:
        """Number of days in the training window."""
        return self.train_end_day - self.train_start_day + 1

    @property
    def test_days(self) -> int:
        """Number of days in the test window."""
        return self.test_end_day - self.test_start_day + 1


def build_m5_sections(
    min_day: int = 1,
    max_day: int = 1941,
    train_days: int = 365,
    test_days: int = 14,
    step_days: int | None = None,
    max_sections: int | None = None,
    start_day: int | None = None,
) -> list[M5Section]:
    """
    Partition the day range ``[min_day, max_day]`` into ``M5Section`` windows, each with
    ``train_days`` (default 365, ~1 year) of training data followed by ``test_days``
    (default 14) days of forecast evaluation.

    Parameters
    ----------
    min_day : int, default 1
        Earliest day available in the dataset.
    max_day : int, default 1941
        Latest day available in the dataset.
    train_days : int, default 365
        Number of training days per section (~1 year).
    test_days : int, default 14
        Number of test/evaluation days immediately following the training window.
    step_days : int, optional
        Stride (in days) between consecutive sections. Defaults to ``train_days`` (365) so
        sections cover consecutive 1-year training blocks across the dataset; pass a smaller
        value (e.g. ``90`` or ``180``) for overlapping rolling windows.
    max_sections : int, optional
        Maximum number of sections to return.
    start_day : int, optional
        Explicit starting day for the first section (defaults to ``min_day``).
    """
    if train_days < 1 or test_days < 1:
        raise ValueError("'train_days' and 'test_days' must be >= 1.")
    stride = int(step_days) if step_days is not None else int(train_days)
    if stride < 1:
        raise ValueError("'step_days' must be >= 1.")

    cur_start = int(start_day) if start_day is not None else int(min_day)
    if cur_start < min_day:
        raise ValueError(f"'start_day' ({cur_start}) cannot be less than 'min_day' ({min_day}).")

    sections: list[M5Section] = []
    sec_id = 1
    while cur_start + train_days + test_days - 1 <= max_day:
        train_end = cur_start + train_days - 1
        test_start = train_end + 1
        test_end = test_start + test_days - 1
        sections.append(
            M5Section(
                section_id=sec_id,
                name=f"sec_{sec_id:02d}_d{cur_start}_{test_end}",
                train_start_day=cur_start,
                train_end_day=train_end,
                test_start_day=test_start,
                test_end_day=test_end,
            )
        )
        if max_sections is not None and len(sections) >= int(max_sections):
            break
        cur_start += stride
        sec_id += 1

    if not sections:
        raise ValueError(
            f"Cannot build any section with train_days={train_days}, test_days={test_days} "
            f"within [{cur_start}, {max_day}]."
        )
    return sections


# ---------------------------------------------------------------------------
# 2. Fast Numba Kernels for GLM Quantiles & Discrete PIT
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _glm_quantiles_batch_numba(
    mu_mat: np.ndarray,
    alpha_mat: np.ndarray,
    quantiles: np.ndarray,
    tol: float,
) -> np.ndarray:
    """
    Compute exact discrete quantiles ``F^{-1}(q)`` for ``(n_rows, n_horizons)`` Poisson / NB2
    predictions in parallel.

    Returns an array of shape ``(n_rows, n_horizons, n_quantiles)``.
    """
    n_rows, n_horizons = mu_mat.shape
    n_q = len(quantiles)
    out = np.full((n_rows, n_horizons, n_q), np.nan, dtype=np.float64)
    for i in prange(n_rows):
        for h in range(n_horizons):
            mu = mu_mat[i, h]
            if np.isnan(mu):
                continue
            if mu <= 0.0:
                for k in range(n_q):
                    out[i, h, k] = 0.0
                continue
            alpha = alpha_mat[i, h]
            pmf = _count_pmf_truncated_numba(mu, alpha, tol)
            out[i, h, :] = _pmf_quantiles_numba(pmf, 0, 1, quantiles)
    return out


@njit(parallel=True, cache=True)
def _hurdle_glm_quantiles_batch_numba(
    pi_mat: np.ndarray,
    lam_mat: np.ndarray,
    alpha_mat: np.ndarray,
    quantiles: np.ndarray,
    tol: float,
) -> np.ndarray:
    """
    Compute exact discrete quantiles ``F^{-1}(q)`` for ``(n_rows, n_horizons)`` Zero-Hurdle
    Poisson / NB2 predictions in parallel.

    Returns an array of shape ``(n_rows, n_horizons, n_quantiles)``.
    """
    n_rows, n_horizons = pi_mat.shape
    n_q = len(quantiles)
    out = np.full((n_rows, n_horizons, n_q), np.nan, dtype=np.float64)
    for i in prange(n_rows):
        for h in range(n_horizons):
            pi = pi_mat[i, h]
            lam = lam_mat[i, h]
            if np.isnan(pi) or np.isnan(lam):
                continue
            if pi <= 0.0 or lam <= 0.0:
                for k in range(n_q):
                    out[i, h, k] = 0.0
                continue
            alpha = alpha_mat[i, h]
            pmf = _hurdle_count_pmf_truncated_numba(pi, lam, alpha, tol)
            out[i, h, :] = _pmf_quantiles_numba(pmf, 0, 1, quantiles)
    return out


def populate_model_quantiles(
    pred_df: pd.DataFrame,
    target_cols: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    tol: float = DEFAULT_SUPPORT_TOL,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Ensure that ``pred_df`` contains ``<target>_q<level>`` columns for every ``target`` in
    ``target_cols`` and every ``level`` in ``quantiles``.

    What it does
    ------------
    - If ``pred_df`` already has all ``<target>_q<level>`` columns (as produced by
      :class:`CatBoostQuantileForecaster` and :class:`EmpiricalQuantileForecaster`), returns
      ``pred_df`` unchanged (or a copy when ``inplace=False``).
    - If ``pred_df`` comes from :class:`GLMForecaster` (``hurdle=True`` with ``<target>_pi``
      and ``<target>_lambda``, or ``hurdle=False`` with ``<target>_mu`` and ``<target>_alpha``),
      evaluates the exact discrete predictive PMFs in parallel via Numba and populates all
      ``<target>_q<level>`` columns.
    """
    q_arr = validate_quantile_levels(quantiles)
    target_list = list(target_cols)
    out = pred_df if inplace else pred_df.copy()

    expected_cols = [quantile_col_name(t, float(q)) for t in target_list for q in q_arr]
    if all(col in out.columns for col in expected_cols):
        return out

    n_rows = len(out)
    n_horizons = len(target_list)
    if n_rows == 0 or n_horizons == 0:
        for col in expected_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out

    is_hurdle = all(
        f"{t}_pi" in out.columns and f"{t}_lambda" in out.columns for t in target_list
    )
    alpha_mat = np.column_stack(
        [
            out[f"{t}_alpha"].to_numpy(dtype=np.float64)
            if f"{t}_alpha" in out.columns
            else np.full(n_rows, np.nan, dtype=np.float64)
            for t in target_list
        ]
    )

    if is_hurdle:
        pi_mat = np.column_stack([out[f"{t}_pi"].to_numpy(dtype=np.float64) for t in target_list])
        lam_mat = np.column_stack([out[f"{t}_lambda"].to_numpy(dtype=np.float64) for t in target_list])
        q_tensor = _hurdle_glm_quantiles_batch_numba(
            np.ascontiguousarray(pi_mat),
            np.ascontiguousarray(lam_mat),
            np.ascontiguousarray(alpha_mat),
            np.ascontiguousarray(q_arr, dtype=np.float64),
            float(tol),
        )
    else:
        missing_mu = [t for t in target_list if f"{t}_mu" not in out.columns]
        if missing_mu:
            raise ValueError(
                f"Cannot populate quantiles: predictions are missing both quantile columns and "
                f"'<target>_mu' columns for {missing_mu}."
            )
        mu_mat = np.column_stack([out[f"{t}_mu"].to_numpy(dtype=np.float64) for t in target_list])
        q_tensor = _glm_quantiles_batch_numba(
            np.ascontiguousarray(mu_mat),
            np.ascontiguousarray(alpha_mat),
            np.ascontiguousarray(q_arr, dtype=np.float64),
            float(tol),
        )

    new_cols: dict[str, np.ndarray] = {}
    for h, t in enumerate(target_list):
        for k, q in enumerate(q_arr):
            col = quantile_col_name(t, float(q))
            if col not in out.columns:
                new_cols[col] = q_tensor[:, h, k]

    if new_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for col, vals in new_cols.items():
                out[col] = vals
    return out


# ---------------------------------------------------------------------------
# 3. Multi-Quantile WMAPE & Pinball Loss Evaluation
# ---------------------------------------------------------------------------


def _compute_quantile_wmape_row(
    y: np.ndarray,
    q_hat: np.ndarray,
    q_level: float | None,
    quantile_label: str,
) -> dict[str, float | int | str]:
    """Compute WMAPE, under/over WMAPE, pinball loss, and coverage on valid `(y, q_hat)` pairs."""
    valid = np.isfinite(y) & np.isfinite(q_hat)
    n_obs = int(valid.sum())
    if n_obs == 0:
        return {
            "quantile": quantile_label,
            "quantile_level": float(q_level) if q_level is not None else np.nan,
            "n_obs": 0,
            "actual": 0.0,
            "pred": 0.0,
            "under_diff": 0.0,
            "over_diff": 0.0,
            "abs_diff": 0.0,
            "wmape": np.nan,
            "under_wmape": np.nan,
            "over_wmape": np.nan,
            "pinball_loss": np.nan,
            "pinball_wmape": np.nan,
            "coverage": np.nan,
        }

    yv = y[valid]
    qv = q_hat[valid]
    actual = float(yv.sum())
    pred = float(qv.sum())
    diff = yv - qv
    under_diff = float(np.maximum(diff, 0.0).sum())
    over_diff = float(np.maximum(-diff, 0.0).sum())
    abs_diff = under_diff + over_diff

    if actual > 0.0:
        wmape = 100.0 * abs_diff / actual
        under_wmape = 100.0 * under_diff / actual
        over_wmape = 100.0 * over_diff / actual
    else:
        wmape = np.nan
        under_wmape = np.nan
        over_wmape = np.nan

    if q_level is not None and np.isfinite(q_level):
        tau = float(q_level)
        pinball_sum = tau * under_diff + (1.0 - tau) * over_diff
        pinball_loss = pinball_sum / n_obs
        pinball_wmape = (100.0 * 2.0 * pinball_sum / actual) if actual > 0.0 else np.nan
    else:
        pinball_loss = abs_diff / (2.0 * n_obs)
        pinball_wmape = wmape

    coverage = float(np.mean(yv <= qv))
    return {
        "quantile": quantile_label,
        "quantile_level": float(q_level) if q_level is not None else np.nan,
        "n_obs": n_obs,
        "actual": actual,
        "pred": pred,
        "under_diff": under_diff,
        "over_diff": over_diff,
        "abs_diff": abs_diff,
        "wmape": wmape,
        "under_wmape": under_wmape,
        "over_wmape": over_wmape,
        "pinball_loss": pinball_loss,
        "pinball_wmape": pinball_wmape,
        "coverage": coverage,
    }


def evaluate_quantile_wmape(
    pred_df: pd.DataFrame,
    target_cols: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    group_cols: Sequence[str] | None = None,
    include_mean: bool = True,
    include_horizon_sum: bool = True,
    tol: float = DEFAULT_SUPPORT_TOL,
) -> pd.DataFrame:
    """
    Evaluate WMAPE, under/over WMAPE, pinball loss, and empirical coverage across all
    quantiles ``q in quantiles`` (and optionally ``"mean"`` from ``<target>_mu``).

    What it does
    ------------
    1. Automatically populates ``<target>_q<level>`` columns via :func:`populate_model_quantiles`
       if ``pred_df`` comes from :class:`GLMForecaster`.
    2. Evaluates metrics at three aggregation levels of ``target``:
       - Each individual horizon ``target_t_plus_0 .. target_t_plus_{H-1}``,
       - ``target="all_horizons"``: pooled across all ``(row, horizon)`` pairs,
       - ``target="horizon_sum"`` (when ``include_horizon_sum=True``): per-row sum of actuals
         vs. per-row sum of quantile predictions across all non-missing horizons on that row.
    3. Computes results overall (``group="overall"``) and broken down by ``group_cols`` if provided.
    """
    q_arr = validate_quantile_levels(quantiles)
    target_list = [t for t in target_cols if t in pred_df.columns]
    if not target_list:
        raise ValueError(f"None of the target_cols {list(target_cols)} are present in 'pred_df'.")

    df_with_q = populate_model_quantiles(pred_df, target_list, quantiles=q_arr, tol=tol, inplace=False)
    has_mean = include_mean and all(f"{t}_mu" in df_with_q.columns for t in target_list)

    def _eval_subset(sub: pd.DataFrame, group_label: str, group_dict: dict) -> list[dict]:
        rows_out: list[dict] = []
        y_mat = np.column_stack([sub[t].to_numpy(dtype=np.float64) for t in target_list])
        valid_y = np.isfinite(y_mat)

        # Evaluate each quantile level
        for q in q_arr:
            q_label = f"q{float(q):g}"
            q_mat = np.column_stack(
                [sub[quantile_col_name(t, float(q))].to_numpy(dtype=np.float64) for t in target_list]
            )
            for h, t in enumerate(target_list):
                stat = _compute_quantile_wmape_row(y_mat[:, h], q_mat[:, h], float(q), q_label)
                rows_out.append({"group": group_label, **group_dict, "target": t, **stat})

            stat_all = _compute_quantile_wmape_row(y_mat.ravel(), q_mat.ravel(), float(q), q_label)
            rows_out.append({"group": group_label, **group_dict, "target": "all_horizons", **stat_all})

            if include_horizon_sum:
                pair_ok = valid_y & np.isfinite(q_mat)
                any_ok = pair_ok.any(axis=1)
                y_sum = np.where(any_ok, np.where(pair_ok, y_mat, 0.0).sum(axis=1), np.nan)
                q_sum = np.where(any_ok, np.where(pair_ok, q_mat, 0.0).sum(axis=1), np.nan)
                stat_sum = _compute_quantile_wmape_row(y_sum, q_sum, float(q), q_label)
                rows_out.append({"group": group_label, **group_dict, "target": "horizon_sum", **stat_sum})

        # Evaluate point forecast (<target>_mu)
        if has_mean:
            mu_mat = np.column_stack([sub[f"{t}_mu"].to_numpy(dtype=np.float64) for t in target_list])
            for h, t in enumerate(target_list):
                stat = _compute_quantile_wmape_row(y_mat[:, h], mu_mat[:, h], None, "mean")
                rows_out.append({"group": group_label, **group_dict, "target": t, **stat})

            stat_all = _compute_quantile_wmape_row(y_mat.ravel(), mu_mat.ravel(), None, "mean")
            rows_out.append({"group": group_label, **group_dict, "target": "all_horizons", **stat_all})

            if include_horizon_sum:
                pair_ok = valid_y & np.isfinite(mu_mat)
                any_ok = pair_ok.any(axis=1)
                y_sum = np.where(any_ok, np.where(pair_ok, y_mat, 0.0).sum(axis=1), np.nan)
                mu_sum = np.where(any_ok, np.where(pair_ok, mu_mat, 0.0).sum(axis=1), np.nan)
                stat_sum = _compute_quantile_wmape_row(y_sum, mu_sum, None, "mean")
                rows_out.append({"group": group_label, **group_dict, "target": "horizon_sum", **stat_sum})

        return rows_out

    all_rows = _eval_subset(df_with_q, "overall", {})
    valid_groups = [g for g in (group_cols or []) if g in df_with_q.columns]
    if valid_groups:
        for key, sub in df_with_q.groupby(valid_groups, observed=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            kd = dict(zip(valid_groups, key_tuple))
            label = "|".join(f"{k}={v}" for k, v in kd.items())
            all_rows.extend(_eval_subset(sub, label, kd))

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# 4. Non-Randomized Discrete PIT Evaluation Framework (All Forecasters)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _eval_pmf_pit_point(
    p: np.ndarray,
    mn: int,
    yy: float,
    levels: np.ndarray,
    n_bins: int,
    cov_row: np.ndarray,
    pitcov_row: np.ndarray,
    q_row: np.ndarray,
    pit_row: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    Evaluate mean, variance, CRPS, squared error, PIT first/second moments, target coverage,
    and PIT histogram bins for a single discrete PMF ``p`` (starting at ``mn`` with unit bin size)
    against realized count ``yy``.
    """
    m = len(p)
    mu = 0.0
    for k in range(m):
        mu += (mn + k) * p[k]
    v = 0.0
    for k in range(m):
        d = (mn + k) - mu
        v += d * d * p[k]

    n_lv = len(levels)
    for a in range(n_lv):
        q_val = _pmf_quantile_scalar_numba(p, mn, 1, levels[a])
        q_row[a] = q_val
        cov_row[a] = 1.0 if yy <= q_val else 0.0

    cr = _pmf_crps_scalar_numba(p, mn, 1, yy)
    se = (yy - mu) * (yy - mu)

    f_lo = 0.0
    f_hi = 0.0
    iy = int(np.floor(yy)) - mn
    if iy >= m:
        f_lo = 1.0
        f_hi = 1.0
    elif iy >= 0:
        c = 0.0
        for k in range(iy):
            c += p[k]
        f_lo = min(c, 1.0)
        f_hi = min(c + p[iy], 1.0)

    for a in range(n_lv):
        pitcov_row[a] = _pit_cdf(levels[a], f_lo, f_hi)

    prev = 0.0
    for b in range(n_bins):
        if b == n_bins - 1:
            cdf_u = 1.0
        else:
            cdf_u = _pit_cdf((b + 1.0) / n_bins, f_lo, f_hi)
        pit_row[b] = cdf_u - prev
        prev = cdf_u

    # Exact 1st and 2nd moments of Uniform[f_lo, f_hi]
    u_mean = 0.5 * (f_lo + f_hi)
    u_sq = (f_lo * f_lo + f_lo * f_hi + f_hi * f_hi) / 3.0
    return mu, v, cr, se, u_mean, u_sq


@njit(parallel=True, cache=True)
def _glm_pit_kernel(
    y: np.ndarray,
    mu_arr: np.ndarray,
    alpha_arr: np.ndarray,
    levels: np.ndarray,
    n_bins: int,
    tol: float,
):
    """Parallel Numba PIT evaluation kernel for standard Poisson / NB2 GLM forecasts."""
    n = len(y)
    n_lv = len(levels)
    mean = np.full(n, np.nan, dtype=np.float64)
    var = np.full(n, np.nan, dtype=np.float64)
    crps = np.full(n, np.nan, dtype=np.float64)
    sq_err = np.full(n, np.nan, dtype=np.float64)
    u_mean = np.full(n, np.nan, dtype=np.float64)
    u_sq = np.full(n, np.nan, dtype=np.float64)
    coverage = np.full((n, n_lv), np.nan, dtype=np.float64)
    pit_cov = np.full((n, n_lv), np.nan, dtype=np.float64)
    q_out = np.full((n, n_lv), np.nan, dtype=np.float64)
    pit = np.full((n, n_bins), np.nan, dtype=np.float64)

    for i in prange(n):
        yy = y[i]
        mu = mu_arr[i]
        if np.isnan(yy) or np.isnan(mu) or mu < 0.0:
            continue
        p = _count_pmf_truncated_numba(mu, alpha_arr[i], tol)
        m_i, v_i, cr_i, se_i, um_i, usq_i = _eval_pmf_pit_point(
            p, 0, yy, levels, n_bins, coverage[i], pit_cov[i], q_out[i], pit[i]
        )
        mean[i] = m_i
        var[i] = v_i
        crps[i] = cr_i
        sq_err[i] = se_i
        u_mean[i] = um_i
        u_sq[i] = usq_i
    return mean, var, crps, sq_err, u_mean, u_sq, coverage, pit_cov, q_out, pit


@njit(parallel=True, cache=True)
def _hurdle_glm_pit_kernel(
    y: np.ndarray,
    pi_arr: np.ndarray,
    lam_arr: np.ndarray,
    alpha_arr: np.ndarray,
    levels: np.ndarray,
    n_bins: int,
    tol: float,
):
    """Parallel Numba PIT evaluation kernel for Zero-Hurdle Poisson / NB2 GLM forecasts."""
    n = len(y)
    n_lv = len(levels)
    mean = np.full(n, np.nan, dtype=np.float64)
    var = np.full(n, np.nan, dtype=np.float64)
    crps = np.full(n, np.nan, dtype=np.float64)
    sq_err = np.full(n, np.nan, dtype=np.float64)
    u_mean = np.full(n, np.nan, dtype=np.float64)
    u_sq = np.full(n, np.nan, dtype=np.float64)
    coverage = np.full((n, n_lv), np.nan, dtype=np.float64)
    pit_cov = np.full((n, n_lv), np.nan, dtype=np.float64)
    q_out = np.full((n, n_lv), np.nan, dtype=np.float64)
    pit = np.full((n, n_bins), np.nan, dtype=np.float64)

    for i in prange(n):
        yy = y[i]
        pi = pi_arr[i]
        lam = lam_arr[i]
        if np.isnan(yy) or np.isnan(pi) or np.isnan(lam) or pi < 0.0 or lam < 0.0:
            continue
        p = _hurdle_count_pmf_truncated_numba(pi, lam, alpha_arr[i], tol)
        m_i, v_i, cr_i, se_i, um_i, usq_i = _eval_pmf_pit_point(
            p, 0, yy, levels, n_bins, coverage[i], pit_cov[i], q_out[i], pit[i]
        )
        mean[i] = m_i
        var[i] = v_i
        crps[i] = cr_i
        sq_err[i] = se_i
        u_mean[i] = um_i
        u_sq[i] = usq_i
    return mean, var, crps, sq_err, u_mean, u_sq, coverage, pit_cov, q_out, pit


@njit(parallel=True, cache=True)
def _quantile_pit_kernel(
    y: np.ndarray,
    q_mat: np.ndarray,
    q_levels: np.ndarray,
    levels: np.ndarray,
    n_bins: int,
    tol: float,
):
    """
    Parallel Numba PIT evaluation kernel for quantile-regression forecasters
    (:class:`CatBoostQuantileForecaster` and :class:`EmpiricalQuantileForecaster`), converting
    each row's quantile grid into a discrete PMF via :func:`_quantiles_to_pmf_numba`.
    """
    n = len(y)
    n_lv = len(levels)
    mean = np.full(n, np.nan, dtype=np.float64)
    var = np.full(n, np.nan, dtype=np.float64)
    crps = np.full(n, np.nan, dtype=np.float64)
    sq_err = np.full(n, np.nan, dtype=np.float64)
    u_mean = np.full(n, np.nan, dtype=np.float64)
    u_sq = np.full(n, np.nan, dtype=np.float64)
    coverage = np.full((n, n_lv), np.nan, dtype=np.float64)
    pit_cov = np.full((n, n_lv), np.nan, dtype=np.float64)
    q_out = np.full((n, n_lv), np.nan, dtype=np.float64)
    pit = np.full((n, n_bins), np.nan, dtype=np.float64)

    for i in prange(n):
        yy = y[i]
        if np.isnan(yy) or np.isnan(q_mat[i, 0]):
            continue
        p, mn = _quantiles_to_pmf_numba(q_levels, q_mat[i], tol, -1)
        m_i, v_i, cr_i, se_i, um_i, usq_i = _eval_pmf_pit_point(
            p, mn, yy, levels, n_bins, coverage[i], pit_cov[i], q_out[i], pit[i]
        )
        mean[i] = m_i
        var[i] = v_i
        crps[i] = cr_i
        sq_err[i] = se_i
        u_mean[i] = um_i
        u_sq[i] = usq_i
    return mean, var, crps, sq_err, u_mean, u_sq, coverage, pit_cov, q_out, pit


def compute_pit_uniformity_metrics(
    pit_mass: np.ndarray,
    u_mean: float | None = None,
    u_sq: float | None = None,
) -> dict[str, float]:
    """
    Compute summary diagnostics measuring how closely a discrete PIT distribution matches
    ``Uniform(0, 1)``.

    Metrics returned
    ----------------
    - ``pit_mean``: Expected PIT value ``E[U]`` (calibrated = ``0.5``; ``> 0.5`` indicates
      systematic under-prediction, ``< 0.5`` indicates systematic over-prediction).
    - ``pit_std``: Standard deviation ``sqrt(Var(U))`` (calibrated = ``1 / sqrt(12) ~= 0.2887``;
      ``> 0.2887`` indicates under-dispersed/too-narrow forecasts, ``< 0.2887`` indicates
      over-dispersed/too-wide forecasts).
    - ``pit_mae``: Mean absolute deviation of histogram bin densities from ``1.0``
      (``mean(|density_b - 1.0|)``, calibrated = ``0.0``).
    - ``pit_rmse``: Root mean squared deviation of bin densities from ``1.0``
      (``sqrt(mean((density_b - 1.0)^2))``, calibrated = ``0.0``).
    - ``pit_ks``: Maximum absolute deviation between the empirical PIT CDF and the uniform
      diagonal ``u`` across bin boundaries (Kolmogorov-Smirnov distance, calibrated = ``0.0``).
    - ``pit_tail_lower``: Relative density in the lowest bin ``[0, 1/B]`` (calibrated = ``1.0``).
    - ``pit_tail_upper``: Relative density in the highest bin ``[1 - 1/B, 1]`` (calibrated = ``1.0``).
    """
    n_bins = len(pit_mass)
    if n_bins == 0 or not np.all(np.isfinite(pit_mass)):
        return {
            "pit_mean": np.nan,
            "pit_std": np.nan,
            "pit_mae": np.nan,
            "pit_rmse": np.nan,
            "pit_ks": np.nan,
            "pit_tail_upper": np.nan,
            "pit_tail_lower": np.nan,
        }

    density = pit_mass * n_bins
    pit_mae = float(np.mean(np.abs(density - 1.0)))
    pit_rmse = float(np.sqrt(np.mean((density - 1.0) ** 2)))
    cdf_emp = np.cumsum(pit_mass)
    cdf_uni = np.arange(1, n_bins + 1, dtype=np.float64) / n_bins
    pit_ks = float(np.max(np.abs(cdf_emp - cdf_uni)))

    if u_mean is None or not np.isfinite(u_mean):
        centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
        u_mean = float(np.sum(pit_mass * centers))
    if u_sq is None or not np.isfinite(u_sq):
        lo = np.arange(n_bins, dtype=np.float64) / n_bins
        hi = (np.arange(n_bins, dtype=np.float64) + 1.0) / n_bins
        u_sq = float(np.sum(pit_mass * (lo * lo + lo * hi + hi * hi) / 3.0))

    pit_var = max(float(u_sq) - float(u_mean) ** 2, 0.0)
    return {
        "pit_mean": float(u_mean),
        "pit_std": float(np.sqrt(pit_var)),
        "pit_mae": pit_mae,
        "pit_rmse": pit_rmse,
        "pit_ks": pit_ks,
        "pit_tail_upper": float(density[-1]),
        "pit_tail_lower": float(density[0]),
    }


@dataclass
class ForecastPITResult:
    """
    Container for the outputs of :func:`evaluate_forecast_pit`.

    Attributes
    ----------
    summary_df : pd.DataFrame
        Overall, per-horizon, and per-group PIT summary metrics (`n_obs`, `mean_pred`,
        `mean_realized`, `bias_pct`, `crps`, `dispersion_ratio`, `pit_mean`, `pit_std`,
        `pit_mae`, `pit_rmse`, `pit_ks`, `pit_tail_upper`, `pit_tail_lower`).
    coverage_df : pd.DataFrame
        Quantile coverage and discrete PIT coverage table (`group`, `target`, `level`,
        `coverage`, `pit_coverage`, `gap_pp`, `n_obs`).
    pit_df : pd.DataFrame
        Non-randomized discrete PIT histogram (`group`, `target`, `bin`, `lower`, `upper`,
        `mass`, `density`).
    levels : tuple[float, ...]
        Target calibration levels evaluated.
    """

    summary_df: pd.DataFrame
    coverage_df: pd.DataFrame
    pit_df: pd.DataFrame
    levels: tuple[float, ...] = field(default=DEFAULT_CALIBRATION_LEVELS)

    def report(self) -> str:
        """Format a compact human-readable text report of the pooled (`all_horizons`) PIT results."""
        overall = self.summary_df[
            (self.summary_df["group"] == "overall") & (self.summary_df["target"] == "all_horizons")
        ]
        cov = self.coverage_df[
            (self.coverage_df["group"] == "overall") & (self.coverage_df["target"] == "all_horizons")
        ]
        pit = self.pit_df[
            (self.pit_df["group"] == "overall") & (self.pit_df["target"] == "all_horizons")
        ]
        lines = ["Forecast PIT Evaluation (all_horizons)", "======================================"]
        cols = [
            "n_obs",
            "mean_pred",
            "mean_realized",
            "bias_pct",
            "crps",
            "dispersion_ratio",
            "pit_mean",
            "pit_std",
            "pit_mae",
            "pit_ks",
            "pit_tail_upper",
        ]
        lines.append(overall[[c for c in cols if c in overall.columns]].to_string(index=False))
        lines.append("")
        lines.append(cov[["level", "coverage", "pit_coverage", "gap_pp"]].to_string(index=False))
        lines.append("")
        lines.append(
            "PIT density by decile (1.0 = calibrated): "
            + " ".join(f"{v:.2f}" for v in pit["density"].to_numpy())
        )
        return "\n".join(lines)


def evaluate_forecast_pit(
    pred_df: pd.DataFrame,
    target_cols: Sequence[str],
    levels: Sequence[float] = DEFAULT_CALIBRATION_LEVELS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    group_cols: Sequence[str] | None = None,
    include_by_horizon: bool = True,
    n_pit_bins: int = 10,
    tol: float = DEFAULT_SUPPORT_TOL,
) -> ForecastPITResult:
    """
    Evaluate non-randomized discrete Probability Integral Transform (PIT) distributions
    across all forecast horizons for any forecaster in the library.

    How predictive distributions are reconstructed per model family
    ---------------------------------------------------------------
    1. **Zero-Hurdle GLM** (``<target>_pi`` and ``<target>_lambda`` columns present):
       Uses the exact zero-hurdle Poisson / NB2 PMF via :func:`_hurdle_glm_pit_kernel`.
    2. **Standard GLM** (``<target>_mu`` and ``<target>_alpha`` columns present without quantile grid):
       Uses the exact Poisson / NB2 PMF via :func:`_glm_pit_kernel`.
    3. **Quantile Forecasters** (:class:`CatBoostQuantileForecaster`, :class:`EmpiricalQuantileForecaster`
       with ``<target>_q<level>`` columns):
       Reconstructs each row's discrete PMF via piecewise-linear quantile interpolation
       (:func:`_quantile_Row_pmf_numba` inside :func:`_quantile_pit_kernel`).
    """
    levels_arr = np.asarray(sorted(levels), dtype=np.float64)
    target_list = [t for t in target_cols if t in pred_df.columns]
    if not target_list:
        raise ValueError(f"None of the target_cols {list(target_cols)} are present in 'pred_df'.")

    n_rows = len(pred_df)
    n_horizons = len(target_list)
    n_lv = len(levels_arr)
    n_bins = int(n_pit_bins)

    q_arr = validate_quantile_levels(quantiles)
    has_hurdle_cols = all(
        f"{t}_pi" in pred_df.columns and f"{t}_lambda" in pred_df.columns for t in target_list
    )
    has_glm_alpha = any(f"{t}_alpha" in pred_df.columns for t in target_list)
    has_mu_cols = all(f"{t}_mu" in pred_df.columns for t in target_list)
    has_q_cols = all(
        quantile_col_name(t, float(q)) in pred_df.columns for t in target_list for q in q_arr
    )

    y_mat = np.column_stack([pred_df[t].to_numpy(dtype=np.float64) for t in target_list])
    y_flat = np.ascontiguousarray(y_mat.ravel(), dtype=np.float64)
    n_total = n_rows * n_horizons

    mean_f = np.full(n_total, np.nan, dtype=np.float64)
    var_f = np.full(n_total, np.nan, dtype=np.float64)
    crps_f = np.full(n_total, np.nan, dtype=np.float64)
    sq_f = np.full(n_total, np.nan, dtype=np.float64)
    um_f = np.full(n_total, np.nan, dtype=np.float64)
    usq_f = np.full(n_total, np.nan, dtype=np.float64)
    cov_f = np.full((n_total, n_lv), np.nan, dtype=np.float64)
    pitcov_f = np.full((n_total, n_lv), np.nan, dtype=np.float64)
    pit_f = np.full((n_total, n_bins), np.nan, dtype=np.float64)
    assigned = np.zeros(n_total, dtype=bool)

    alpha_flat = np.ascontiguousarray(
        np.column_stack(
            [
                pred_df[f"{t}_alpha"].to_numpy(dtype=np.float64)
                if f"{t}_alpha" in pred_df.columns
                else np.full(n_rows, np.nan, dtype=np.float64)
                for t in target_list
            ]
        ).ravel()
    )

    # 1. Zero-Hurdle GLM entries (supports both pure hurdle DataFrames and hybrid router DataFrames):
    if has_hurdle_cols:
        pi_flat = np.ascontiguousarray(
            np.column_stack([pred_df[f"{t}_pi"].to_numpy(dtype=np.float64) for t in target_list]).ravel()
        )
        lam_flat = np.ascontiguousarray(
            np.column_stack([pred_df[f"{t}_lambda"].to_numpy(dtype=np.float64) for t in target_list]).ravel()
        )
        h_mask = np.isfinite(pi_flat) & np.isfinite(lam_flat)
        if np.any(h_mask):
            idx_h = np.flatnonzero(h_mask)
            m1, v1, c1, s1, um1, usq1, cv1, pcv1, _, pt1 = _hurdle_glm_pit_kernel(
                np.ascontiguousarray(y_flat[idx_h]),
                np.ascontiguousarray(pi_flat[idx_h]),
                np.ascontiguousarray(lam_flat[idx_h]),
                np.ascontiguousarray(alpha_flat[idx_h]),
                levels_arr,
                n_bins,
                float(tol),
            )
            mean_f[idx_h], var_f[idx_h], crps_f[idx_h], sq_f[idx_h] = m1, v1, c1, s1
            um_f[idx_h], usq_f[idx_h], cov_f[idx_h], pitcov_f[idx_h], pit_f[idx_h] = um1, usq1, cv1, pcv1, pt1
            assigned[idx_h] = True

    # 2. Quantile-grid entries (pure quantile models, or intermittent quantile branch in RegimeRouterForecaster):
    is_router_intermittent = (
        np.repeat((pred_df["model_route"].astype(str).to_numpy() == "intermittent"), n_horizons)
        if "model_route" in pred_df.columns
        else np.zeros(n_total, dtype=bool)
    )
    if has_q_cols and (not has_glm_alpha or np.any(~assigned & (is_router_intermittent | (not has_mu_cols)))):
        q_3d = np.empty((n_rows, n_horizons, len(q_arr)), dtype=np.float64)
        for h, t in enumerate(target_list):
            for k, q in enumerate(q_arr):
                q_3d[:, h, k] = pred_df[quantile_col_name(t, float(q))].to_numpy(dtype=np.float64)
        q_flat = q_3d.reshape(n_total, len(q_arr))
        q_mask = (~assigned) & np.isfinite(q_flat[:, 0])
        if has_glm_alpha and has_mu_cols:
            q_mask = q_mask & (is_router_intermittent | (~np.isfinite(alpha_flat) & has_hurdle_cols))
        if np.any(q_mask):
            idx_q = np.flatnonzero(q_mask)
            m2, v2, c2, s2, um2, usq2, cv2, pcv2, _, pt2 = _quantile_pit_kernel(
                np.ascontiguousarray(y_flat[idx_q]),
                np.ascontiguousarray(q_flat[idx_q]),
                np.ascontiguousarray(q_arr, dtype=np.float64),
                levels_arr,
                n_bins,
                float(tol),
            )
            mean_f[idx_q], var_f[idx_q], crps_f[idx_q], sq_f[idx_q] = m2, v2, c2, s2
            um_f[idx_q], usq_f[idx_q], cov_f[idx_q], pitcov_f[idx_q], pit_f[idx_q] = um2, usq2, cv2, pcv2, pt2
            assigned[idx_q] = True

    # 3. Standard Poisson / NB2 GLM entries for any remaining rows:
    if has_mu_cols and np.any(~assigned):
        mu_flat = np.ascontiguousarray(
            np.column_stack([pred_df[f"{t}_mu"].to_numpy(dtype=np.float64) for t in target_list]).ravel()
        )
        g_mask = (~assigned) & np.isfinite(mu_flat)
        if np.any(g_mask):
            idx_g = np.flatnonzero(g_mask)
            m3, v3, c3, s3, um3, usq3, cv3, pcv3, _, pt3 = _glm_pit_kernel(
                np.ascontiguousarray(y_flat[idx_g]),
                np.ascontiguousarray(mu_flat[idx_g]),
                np.ascontiguousarray(alpha_flat[idx_g]),
                levels_arr,
                n_bins,
                float(tol),
            )
            mean_f[idx_g], var_f[idx_g], crps_f[idx_g], sq_f[idx_g] = m3, v3, c3, s3
            um_f[idx_g], usq_f[idx_g], cov_f[idx_g], pitcov_f[idx_g], pit_f[idx_g] = um3, usq3, cv3, pcv3, pt3
            assigned[idx_g] = True

    # Reshape back to (n_rows, n_horizons, ...)
    mean_mat = mean_f.reshape(n_rows, n_horizons)
    var_mat = var_f.reshape(n_rows, n_horizons)
    crps_mat = crps_f.reshape(n_rows, n_horizons)
    sq_mat = sq_f.reshape(n_rows, n_horizons)
    um_mat = um_f.reshape(n_rows, n_horizons)
    usq_mat = usq_f.reshape(n_rows, n_horizons)
    cov_3d = cov_f.reshape(n_rows, n_horizons, n_lv)
    pitcov_3d = pitcov_f.reshape(n_rows, n_horizons, n_lv)
    pit_3d = pit_f.reshape(n_rows, n_horizons, n_bins)

    def _agg_slice(
        mask_2d: np.ndarray,
        group_label: str,
        group_dict: dict,
        target_label: str,
    ) -> tuple[dict, list[dict], list[dict]]:
        n_obs = int(mask_2d.sum())
        if n_obs == 0:
            uni = compute_pit_uniformity_metrics(np.full(n_bins, np.nan))
            summ = {
                "group": group_label,
                **group_dict,
                "target": target_label,
                "n_obs": 0,
                "mean_pred": np.nan,
                "mean_realized": np.nan,
                "bias_pct": np.nan,
                "crps": np.nan,
                "dispersion_ratio": np.nan,
                **uni,
            }
            cov_rows = [
                {
                    "group": group_label,
                    **group_dict,
                    "target": target_label,
                    "level": float(lv),
                    "coverage": np.nan,
                    "pit_coverage": np.nan,
                    "gap_pp": np.nan,
                    "n_obs": 0,
                }
                for lv in levels_arr
            ]
            pit_rows = [
                {
                    "group": group_label,
                    **group_dict,
                    "target": target_label,
                    "bin": b,
                    "lower": b / n_bins,
                    "upper": (b + 1) / n_bins,
                    "mass": np.nan,
                    "density": np.nan,
                }
                for b in range(n_bins)
            ]
            return summ, cov_rows, pit_rows

        mp = float(mean_mat[mask_2d].mean())
        mr = float(y_mat[mask_2d].mean())
        v_sum = float(var_mat[mask_2d].sum())
        se_sum = float(sq_mat[mask_2d].sum())
        pit_mass = pit_3d[mask_2d].mean(axis=0)
        u_mean_val = float(um_mat[mask_2d].mean())
        u_sq_val = float(usq_mat[mask_2d].mean())
        uni = compute_pit_uniformity_metrics(pit_mass, u_mean=u_mean_val, u_sq=u_sq_val)

        summ = {
            "group": group_label,
            **group_dict,
            "target": target_label,
            "n_obs": n_obs,
            "mean_pred": mp,
            "mean_realized": mr,
            "bias_pct": 100.0 * (mp - mr) / mr if mr > 0.0 else np.nan,
            "crps": float(crps_mat[mask_2d].mean()),
            "dispersion_ratio": se_sum / v_sum if v_sum > 0.0 else np.nan,
            **uni,
        }
        cov_m = cov_3d[mask_2d].mean(axis=0)
        pitcov_m = pitcov_3d[mask_2d].mean(axis=0)
        cov_rows = [
            {
                "group": group_label,
                **group_dict,
                "target": target_label,
                "level": float(lv),
                "coverage": float(cov_m[a]),
                "pit_coverage": float(pitcov_m[a]),
                "gap_pp": 100.0 * (float(pitcov_m[a]) - float(lv)),
                "n_obs": n_obs,
            }
            for a, lv in enumerate(levels_arr)
        ]
        pit_rows = [
            {
                "group": group_label,
                **group_dict,
                "target": target_label,
                "bin": b,
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "mass": float(pit_mass[b]),
                "density": float(pit_mass[b] * n_bins),
            }
            for b in range(n_bins)
        ]
        return summ, cov_rows, pit_rows

    valid_2d = np.isfinite(mean_mat) & np.isfinite(y_mat)
    summaries: list[dict] = []
    cov_all: list[dict] = []
    pit_all: list[dict] = []

    def _collect_for_row_mask(row_mask: np.ndarray, group_label: str, group_dict: dict) -> None:
        mask_all = valid_2d & row_mask[:, None]
        s, cv, pt = _agg_slice(mask_all, group_label, group_dict, "all_horizons")
        summaries.append(s)
        cov_all.extend(cv)
        pit_all.extend(pt)
        if include_by_horizon:
            for h, t in enumerate(target_list):
                mask_h = np.zeros_like(valid_2d, dtype=bool)
                mask_h[:, h] = valid_2d[:, h] & row_mask
                sh, cvh, pth = _agg_slice(mask_h, group_label, group_dict, t)
                summaries.append(sh)
                cov_all.extend(cvh)
                pit_all.extend(pth)

    _collect_for_row_mask(np.ones(n_rows, dtype=bool), "overall", {})
    valid_groups = [g for g in (group_cols or []) if g in pred_df.columns]
    if valid_groups:
        row_positions = pd.Series(np.arange(n_rows), index=pred_df.index)
        for key, sub_idx in row_positions.groupby([pred_df[g] for g in valid_groups], observed=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            kd = dict(zip(valid_groups, key_tuple))
            label = "|".join(f"{k}={v}" for k, v in kd.items())
            rmask = np.zeros(n_rows, dtype=bool)
            rmask[sub_idx.to_numpy()] = True
            _collect_for_row_mask(rmask, label, kd)

    return ForecastPITResult(
        summary_df=pd.DataFrame(summaries),
        coverage_df=pd.DataFrame(cov_all),
        pit_df=pd.DataFrame(pit_all),
        levels=tuple(float(x) for x in levels_arr),
    )


# ---------------------------------------------------------------------------
# 5. Weighted Scaled Pinball Loss (WSPL / SPL) — M5 & Long et al. (2023)
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _compute_insample_naive_scale_numba(
    matrix: np.ndarray,
    n_train_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-series post-first-sale one-step-ahead naive MAE scale factor and active
    training volume strictly over ``matrix[:, :n_train_cols]`` (Long et al., 2023, Sec. 4.3):

        scale_i = (1 / (T - t_{i,0})) * sum_{t = t_{i,0} + 1}^{T} |y_{i,t} - y_{i,t-1}|

    where ``t_{i,0}`` is the 0-based column index of the first non-zero sale in the training
    window. Series with no training sales, fewer than 2 observations from ``t_{i,0}`` to ``T``,
    or zero variation (e.g. identical sales on consecutive final timestamps) receive ``NaN``
    and are omitted from WSPL aggregation per Long et al. (2023, p. 13).
    """
    n_series = matrix.shape[0]
    scales = np.full(n_series, np.nan, dtype=np.float64)
    volumes = np.zeros(n_series, dtype=np.float64)

    for i in prange(n_series):
        first_pos = -1
        vol = 0.0
        for t in range(n_train_cols):
            val = matrix[i, t]
            if val > 0.0:
                vol += val
                if first_pos < 0:
                    first_pos = t
        volumes[i] = vol
        if first_pos < 0 or (n_train_cols - first_pos) < 2:
            continue

        diff_sum = 0.0
        cnt = 0
        for t in range(first_pos + 1, n_train_cols):
            diff_sum += abs(matrix[i, t] - matrix[i, t - 1])
            cnt += 1
        if cnt > 0:
            sc = diff_sum / cnt
            if sc > 1e-12:
                scales[i] = sc
    return scales, volumes


def evaluate_wspl(
    pred_df: pd.DataFrame,
    history: pd.DataFrame | object,
    target_cols: Sequence[str],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ts_id_cols: Sequence[str] | None = None,
    period_col: str = "day",
    var_col: str = "demand",
    train_end_period: float | int | None = None,
    weights: str | dict | pd.DataFrame = "equal",
    group_cols: Sequence[str] | None = None,
    stratify_by_regime: bool = True,
    include_by_horizon: bool = False,
    tol: float = DEFAULT_SUPPORT_TOL,
) -> pd.DataFrame:
    """
    Evaluate **Weighted Scaled Pinball Loss (WSPL)** and per-quantile **Scaled Pinball Loss (SPL)**
    following the M5 Uncertainty competition and Long et al. (2023, Section 4.3, p. 13).

    Why WSPL complements WMAPE
    --------------------------
    In unscaled or volume-weighted metrics (`WMAPE`), fast-moving `Smooth` and `Erratic` SKUs
    dominate the total error sum while the 70%–95% of `Lumpy` and `Intermittent` SKUs have
    minimal influence. WSPL normalizes each series' pinball loss by its own post-first-sale
    in-sample 1-step naive MAE before aggregating across series, and by default (`stratify_by_regime=True`)
    reports WSPL both overall and stratified by SBC demand class (`demand_regime_name`).

    Parameters
    ----------
    pred_df : pd.DataFrame
        Forecast DataFrame from any forecaster (`GLMForecaster`, `CatBoostQuantileForecaster`,
        `EmpiricalQuantileForecaster`, or `RegimeRouterForecaster`). Missing `<target>_q<level>`
        columns on parametric GLM outputs are populated automatically via :func:`populate_model_quantiles`.
    history : DiscreteTimeSeries or pd.DataFrame
        Training or full historical demand panel used to compute each series' post-first-sale
        in-sample naive 1-step MAE denominator.
    target_cols : sequence of str
        Horizon target column names (`target_t_plus_0`, ...).
    quantiles : sequence of float, default DEFAULT_QUANTILES
        Quantile levels `u_1, ..., u_q` evaluated.
    ts_id_cols : sequence of str, optional
        Series identifier columns (inferred automatically if `history` is a `DiscreteTimeSeries`
        or from shared `store_id`/`item_id` columns).
    train_end_period : optional
        Last inclusive training period `T`. If omitted, inferred from `history.test_start_period`
        (strictly `< test_start_period`) or strictly `< pred_df[period_col].min()`.
    weights : {"equal", "volume"} or dict or pd.DataFrame, default "equal"
        Series weighting scheme `w_i`:
        - `"equal"`: `w_i = 1 / n_valid` (Long et al., 2023, Section 4.3).
        - `"volume"`: `w_i` proportional to each series' training unit volume.
        - `dict` or `pd.DataFrame`: custom non-negative weight per series (`ts_id_cols` + `"weight"`).
    group_cols : sequence of str, optional
        Optional breakdown columns (e.g. `["cat_id"]`).
    stratify_by_regime : bool, default True
        If `True` and `"demand_regime_name"` is present (or derivable from `history`), automatically
        includes per-demand-regime (`smooth`, `erratic`, `intermittent`, `lumpy`) WSPL breakdowns.
    include_by_horizon : bool, default False
        If `True`, also reports per-horizon SPL/WSPL rows in addition to `target="all_horizons"`.
    """
    from probabilistic_modeling.time_series import classify_demand_regimes

    q_arr = validate_quantile_levels(quantiles)
    target_list = [t for t in target_cols if t in pred_df.columns]
    if not target_list:
        raise ValueError(f"None of the target_cols {list(target_cols)} are present in 'pred_df'.")

    # Extract raw history DataFrame and metadata from DiscreteTimeSeries if passed:
    inferred_test_start = None
    if hasattr(history, "ts_id_cols") and hasattr(history, "df"):
        ts_obj = history
        if ts_id_cols is None:
            ts_id_cols = list(ts_obj.ts_id_cols)
        period_col = ts_obj.period_col
        var_col = ts_obj.var_col
        inferred_test_start = ts_obj.test_start_period
        hist_df = (
            ts_obj.raw_df
            if (hasattr(ts_obj, "raw_df") and isinstance(ts_obj.raw_df, pd.DataFrame) and not ts_obj.raw_df.empty)
            else ts_obj.df
        )
    elif isinstance(history, pd.DataFrame):
        hist_df = history
    else:
        raise TypeError("'history' must be a DiscreteTimeSeries or a pandas DataFrame.")

    if ts_id_cols is None:
        id_cols = [c for c in ("store_id", "item_id") if c in pred_df.columns and c in hist_df.columns]
        if not id_cols:
            id_cols = [c for c in ("id", "item_id") if c in pred_df.columns and c in hist_df.columns]
        if not id_cols:
            raise ValueError("Could not infer 'ts_id_cols'; please specify explicitly.")
    else:
        id_cols = list(ts_id_cols)

    dense_hist = DenseHistory.from_frame(hist_df, id_cols, period_col, var_col)
    periods_sorted = dense_hist.periods
    if train_end_period is not None:
        n_train_cols = int(np.searchsorted(periods_sorted, train_end_period, side="right"))
    elif inferred_test_start is not None:
        n_train_cols = int(np.searchsorted(periods_sorted, inferred_test_start, side="left"))
    elif period_col in pred_df.columns and len(pred_df) > 0:
        min_pred_p = pred_df[period_col].min()
        n_train_cols = int(np.searchsorted(periods_sorted, min_pred_p, side="left"))
    else:
        n_train_cols = len(periods_sorted)

    if n_train_cols < 2:
        raise ValueError(
            f"At least 2 training periods are required to compute WSPL naive scale (got {n_train_cols})."
        )

    scales_per_series, vol_per_series = _compute_insample_naive_scale_numba(dense_hist.matrix, n_train_cols)

    df_q = populate_model_quantiles(pred_df, target_list, quantiles=q_arr, tol=tol, inplace=False)

    # Attach demand_regime_name from training history if requested and not yet on df_q:
    if stratify_by_regime and "demand_regime_name" not in df_q.columns:
        max_train_p = periods_sorted[n_train_cols - 1]
        sbc_df = classify_demand_regimes(
            hist_df, ts_id_cols=id_cols, period_col=period_col, var_col=var_col, train_end_period=max_train_p
        )
        sbc_sub = sbc_df[id_cols + ["demand_regime", "demand_regime_name"]].copy()
        for c in id_cols:
            sbc_sub[c] = sbc_sub[c].astype(df_q[c].dtype)
        df_q = pd.merge(df_q, sbc_sub, on=id_cols, how="left")

    row_series_idx, _ = dense_hist.locate(df_q)
    n_rows = len(df_q)
    n_h = len(target_list)
    n_q = len(q_arr)

    y_mat = np.column_stack([df_q[t].to_numpy(dtype=np.float64) for t in target_list])  # (n_rows, n_h)
    # Build pinball tensor of shape (n_rows, n_h, n_q):
    pb_3d = np.full((n_rows, n_h, n_q), np.nan, dtype=np.float64)
    for k, q in enumerate(q_arr):
        tau = float(q)
        q_mat = np.column_stack([df_q[quantile_col_name(t, tau)].to_numpy(dtype=np.float64) for t in target_list])
        valid = np.isfinite(y_mat) & np.isfinite(q_mat)
        diff = np.where(valid, y_mat - q_mat, 0.0)
        pb = tau * np.maximum(diff, 0.0) + (1.0 - tau) * np.maximum(-diff, 0.0)
        pb_3d[:, :, k] = np.where(valid, pb, np.nan)

    # Resolve custom series weights if provided:
    n_total_hist_series = len(scales_per_series)
    if isinstance(weights, str):
        w_mode = weights.strip().lower()
        if w_mode == "equal":
            raw_series_w = np.ones(n_total_hist_series, dtype=np.float64)
        elif w_mode == "volume":
            raw_series_w = np.maximum(vol_per_series, 0.0)
        else:
            raise ValueError(f"Unsupported weights='{weights}'. Use 'equal', 'volume', a dict, or a DataFrame.")
    elif isinstance(weights, dict):
        raw_series_w = np.array(
            [float(weights.get(k, 0.0)) for k, _ in sorted(dense_hist.series_index.items(), key=lambda kv: kv[1])],
            dtype=np.float64,
        )
    elif isinstance(weights, pd.DataFrame):
        w_col = "weight" if "weight" in weights.columns else [c for c in weights.columns if c not in id_cols][0]
        r_w, _ = dense_hist.locate(weights.assign(**{period_col: periods_sorted[0]}))
        raw_series_w = np.zeros(n_total_hist_series, dtype=np.float64)
        valid_rw = r_w >= 0
        raw_series_w[r_w[valid_rw]] = weights.loc[valid_rw, w_col].to_numpy(dtype=np.float64)
    else:
        raise TypeError("'weights' must be 'equal', 'volume', a dict, or a DataFrame.")

    def _eval_wspl_slice(row_mask: np.ndarray, group_label: str, group_dict: dict, target_label: str, h_idx: int | None) -> list[dict]:
        sub_r = row_series_idx[row_mask]
        known_mask = sub_r >= 0
        sub_r = sub_r[known_mask]
        uniq_s = np.unique(sub_r)
        n_series_tot = int(len(uniq_s))
        if n_series_tot == 0:
            return []

        if h_idx is None:
            # Pool across all horizons on rows in row_mask: shape (n_masked_rows, n_h, n_q)
            pb_sub = pb_3d[row_mask][known_mask]  # (M, n_h, n_q)
            pb_sum_s = np.zeros((n_total_hist_series, n_q), dtype=np.float64)
            pb_cnt_s = np.zeros((n_total_hist_series, n_q), dtype=np.int64)
            valid_pb = np.isfinite(pb_sub)
            pb_zero = np.where(valid_pb, pb_sub, 0.0)
            np.add.at(pb_sum_s, sub_r, pb_zero.sum(axis=1))
            np.add.at(pb_cnt_s, sub_r, valid_pb.sum(axis=1))
        else:
            pb_sub = pb_3d[row_mask, h_idx, :][known_mask]  # (M, n_q)
            pb_sum_s = np.zeros((n_total_hist_series, n_q), dtype=np.float64)
            pb_cnt_s = np.zeros((n_total_hist_series, n_q), dtype=np.int64)
            valid_pb = np.isfinite(pb_sub)
            pb_zero = np.where(valid_pb, pb_sub, 0.0)
            np.add.at(pb_sum_s, sub_r, pb_zero)
            np.add.at(pb_cnt_s, sub_r, valid_pb.astype(np.int64))

        has_eval_obs = (pb_cnt_s[uniq_s] > 0).all(axis=1)
        valid_scale = np.isfinite(scales_per_series[uniq_s]) & (scales_per_series[uniq_s] > 0.0)
        keep = has_eval_obs & valid_scale
        valid_s = uniq_s[keep]
        n_valid = int(len(valid_s))
        n_omitted = int(n_series_tot - n_valid)

        if n_valid == 0:
            out_empty = [
                {
                    "group": group_label,
                    **group_dict,
                    "target": target_label,
                    "quantile": f"q{float(q):g}",
                    "quantile_level": float(q),
                    "n_series_total": n_series_tot,
                    "n_series_valid": 0,
                    "n_series_omitted": n_omitted,
                    "mean_scale": np.nan,
                    "wspl": np.nan,
                }
                for q in q_arr
            ]
            out_empty.append(
                {
                    "group": group_label,
                    **group_dict,
                    "target": target_label,
                    "quantile": "wspl",
                    "quantile_level": np.nan,
                    "n_series_total": n_series_tot,
                    "n_series_valid": 0,
                    "n_series_omitted": n_omitted,
                    "mean_scale": np.nan,
                    "wspl": np.nan,
                }
            )
            return out_empty

        mean_pl = pb_sum_s[valid_s] / pb_cnt_s[valid_s]  # (n_valid, n_q)
        spl_mat = mean_pl / scales_per_series[valid_s, None]  # (n_valid, n_q)

        w_vec = raw_series_w[valid_s].copy()
        w_sum = float(w_vec.sum())
        if w_sum <= 0.0:
            w_vec = np.full(n_valid, 1.0 / n_valid, dtype=np.float64)
        else:
            w_vec /= w_sum

        wspl_per_q = w_vec @ spl_mat  # (n_q,)
        mean_sc = float(np.dot(w_vec, scales_per_series[valid_s]))

        rows_res: list[dict] = []
        for k, q in enumerate(q_arr):
            rows_res.append(
                {
                    "group": group_label,
                    **group_dict,
                    "target": target_label,
                    "quantile": f"q{float(q):g}",
                    "quantile_level": float(q),
                    "n_series_total": n_series_tot,
                    "n_series_valid": n_valid,
                    "n_series_omitted": n_omitted,
                    "mean_scale": mean_sc,
                    "wspl": float(wspl_per_q[k]),
                }
            )
        rows_res.append(
            {
                "group": group_label,
                **group_dict,
                "target": target_label,
                "quantile": "wspl",
                "quantile_level": np.nan,
                "n_series_total": n_series_tot,
                "n_series_valid": n_valid,
                "n_series_omitted": n_omitted,
                "mean_scale": mean_sc,
                "wspl": float(np.mean(wspl_per_q)),
            }
        )
        return rows_res

    out_rows: list[dict] = []

    def _run_for_mask(rmask: np.ndarray, glabel: str, gdict: dict) -> None:
        out_rows.extend(_eval_wspl_slice(rmask, glabel, gdict, "all_horizons", None))
        if include_by_horizon:
            for h_i, t_name in enumerate(target_list):
                out_rows.extend(_eval_wspl_slice(rmask, glabel, gdict, t_name, h_i))

    _run_for_mask(np.ones(n_rows, dtype=bool), "overall", {})

    effective_groups = [g for g in (group_cols or []) if g in df_q.columns]
    if effective_groups:
        row_pos = pd.Series(np.arange(n_rows), index=df_q.index)
        for key, sub_idx in row_pos.groupby([df_q[g] for g in effective_groups], observed=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            kd = dict(zip(effective_groups, key_tuple))
            label = "|".join(f"{k}={v}" for k, v in kd.items())
            rmask = np.zeros(n_rows, dtype=bool)
            rmask[sub_idx.to_numpy()] = True
            _run_for_mask(rmask, label, kd)

    if stratify_by_regime and "demand_regime_name" in df_q.columns and "demand_regime_name" not in effective_groups:
        row_pos = pd.Series(np.arange(n_rows), index=df_q.index)
        for reg_name, sub_idx in row_pos.groupby(df_q["demand_regime_name"], observed=True):
            kd = {"demand_regime_name": str(reg_name)}
            rmask = np.zeros(n_rows, dtype=bool)
            rmask[sub_idx.to_numpy()] = True
            _run_for_mask(rmask, f"demand_regime_name={reg_name}", kd)

    return pd.DataFrame(out_rows)
