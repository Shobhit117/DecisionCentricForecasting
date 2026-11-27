"""
Unified probabilistic forecaster abstractions (`models/forecasters.py`).

What this module does
---------------------
Provides a shared, stateful object-oriented interface (`ProbabilisticForecaster`) and
concrete implementations (`GLMForecaster` and `CatBoostQuantileForecaster`) for:
1. Multi-horizon model training (`fit`),
2. Reusable multi-time prediction on engineered panels, specific period slices, or raw
   sales + covariate tables (`predict`, `transform`, `predict_from_raw`),
3. Complete bundle persistence (`save`, `load`, and universal `load_forecaster`),
4. Multi-horizon point & quantile forecast evaluation (`compute_forecasting_metrics`,
   `evaluate_quantile_forecasts`), and
5. Compound Demand During Lead Time propagation (`compute_ddlt` via `DDLTCalculator`).

Where it is used
----------------
- Primary model training, scoring, and DDLT interface used by `scripts/run_m5_pipeline.py`,
  `DiscreteTimeSeries` delegation methods, and `EmpiricalQuantileForecaster` in `models/baseline.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

from numba import njit, prange
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf

from probabilistic_modeling.ddlt import DDLTCalculator
from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.distributions.quantile import (
    DEFAULT_QUANTILES,
    pinball_loss,
    quantile_crps,
    quantiles_to_mean,
    validate_quantile_levels,
)

if TYPE_CHECKING:
    from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel
    from probabilistic_modeling.time_series import DiscreteTimeSeries, TimeSeriesFeatureSpec

logger = logging.getLogger(__name__)


def compute_forecasting_metrics(
    pred_df: pd.DataFrame,
    target_cols: list[str],
    pred_columns: list[str],
) -> pd.DataFrame:
    """
    Computes multi-horizon cumulative point forecast error columns (`actual`, `pred`,
    `under_diff`, `over_diff`, `abs_diff`) in-place on `pred_df`.

    What it does
    ------------
    - Pairs each horizon target column (`target_t_plus_h`) with its corresponding expected-value
      prediction column (`target_t_plus_h_mu`).
    - In **production mode** (when `target_cols` are absent or all `NaN`), sets `pred` to the
      sum of predicted horizon means across the forecast window and sets `actual`, `under_diff`,
      `over_diff`, and `abs_diff` to `NaN`.
    - In **backtesting mode**, sums realized demand across all non-missing horizons on each row
      (`actual`), sums the corresponding predicted means over those exact same non-missing
      horizons (`pred`, masking out any horizon where `target_t_plus_h` is `NaN` near the end
      of the panel), and computes:
      - `under_diff = max(actual - pred, 0)` (under-forecasting error),
      - `over_diff = max(pred - actual, 0)` (over-forecasting error),
      - `abs_diff = under_diff + over_diff = |actual - pred|`.

    Why it is needed
    ----------------
    Near the end of a backtest panel, row `t` may have realized observations for `target_t_plus_0`
    and `target_t_plus_1`, while `target_t_plus_2 .. target_t_plus_6` lie beyond the end of the
    dataset (`NaN`). Masking `pred` to zero on horizons where `target_col` is `NaN` ensures
    `actual` and `pred` always compare the exact same set of realized horizons.

    Where it is used
    ----------------
    Called automatically at the end of `GLMForecaster.predict`, `CatBoostQuantileForecaster.predict`,
    and `EmpiricalQuantileForecaster.predict`.
    """
    if len(target_cols) != len(pred_columns):
        raise ValueError(
            f"Number of pred columns ({len(pred_columns)}) does not match "
            f"the number of target columns ({len(target_cols)})."
        )

    target_to_pred = dict(zip(sorted(target_cols), sorted(pred_columns)))
    present_targets = [c for c in target_cols if c in pred_df.columns]

    if not present_targets or pred_df[present_targets].notna().sum().sum() == 0:
        # Production mode: no realized future targets available yet
        pred_df["actual"] = np.nan
        pred_df["pred"] = pred_df[pred_columns].sum(axis=1)
        pred_df["under_diff"] = np.nan
        pred_df["over_diff"] = np.nan
        pred_df["abs_diff"] = np.nan
        return pred_df

    sorted_targets = sorted(target_cols)
    sorted_preds = [target_to_pred[c] for c in sorted_targets]
    t_mat = pred_df[sorted_targets].to_numpy(dtype=np.float64, copy=False)
    p_mat = pred_df[sorted_preds].to_numpy(dtype=np.float64, copy=False)
    valid = ~np.isnan(t_mat)

    actual = np.where(valid, t_mat, 0.0).sum(axis=1)
    pred = np.where(valid, p_mat, 0.0).sum(axis=1)
    pred_df["actual"] = actual
    pred_df["pred"] = pred
    pred_df["under_diff"] = np.maximum(actual - pred, 0.0)
    pred_df["over_diff"] = np.maximum(pred - actual, 0.0)
    pred_df["abs_diff"] = pred_df["under_diff"] + pred_df["over_diff"]
    return pred_df


def evaluate_quantile_forecasts(
    pred_df: pd.DataFrame,
    target_cols: list[str],
    quantiles: tuple[float, ...] | list[float],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluates multi-horizon quantile forecasts by computing pinball loss, empirical
    coverage `P(y <= q_hat)`, and quantile-approximated CRPS per horizon and quantile level.

    What it does
    ------------
    For each optional group in `group_cols` and each horizon `target` in `target_cols`:
    1. Extracts non-missing realizations `y` and the `(n_obs, K)` matrix of predicted
       quantiles `q_mat` (using column naming `quantile_col_name(target, q)`).
    2. Computes the mean Continuous Ranked Probability Score (`crps_mean`) across all
       observations via `quantile_crps(y, q_mat, levels)`.
    3. For each quantile level `q_j`, computes mean `pinball_loss(y, q_hat_j, q_j)` and
       empirical coverage `mean(y <= q_hat_j)` (which should be close to `q_j` for a
       well-calibrated forecaster).

    Where it is used
    ----------------
    Called by `DiscreteTimeSeries.evaluate_quantile_forecasts`, `scripts/run_m5_pipeline.py`,
    and quantile evaluation tests.
    """
    from probabilistic_modeling.models.catboost_quantile import quantile_col_name

    levels = tuple(float(q) for q in validate_quantile_levels(quantiles))
    group_cols = [c for c in (group_cols or []) if c in pred_df.columns]
    groups = pred_df.groupby(group_cols, observed=True) if group_cols else [((), pred_df)]
    rows = []
    for group_values, gdf in groups:
        group_values = group_values if isinstance(group_values, tuple) else (group_values,)
        for target in target_cols:
            if target not in gdf.columns:
                continue
            mask = gdf[target].notna()
            y = gdf.loc[mask, target].to_numpy(dtype=float)
            if len(y) == 0:
                continue
            q_cols = [quantile_col_name(target, q) for q in levels]
            q_mat = gdf.loc[mask, q_cols].to_numpy(dtype=float)
            crps_mean = float(np.mean(quantile_crps(y, q_mat, levels)))
            for j, q in enumerate(levels):
                q_hat = q_mat[:, j]
                rows.append({
                    **dict(zip(group_cols, group_values)),
                    "target": target,
                    "quantile": q,
                    "n_obs": len(y),
                    "pinball_loss": float(pinball_loss(y, q_hat, q).mean()),
                    "coverage": float(np.mean(y <= q_hat)),
                    "crps": crps_mean,
                })
    return pd.DataFrame(rows)


def _hurdle_conditional_mean(lam: np.ndarray | float, alpha: float | None) -> np.ndarray | float:
    """
    Computes the conditional positive expectation `E[Y | Y > 0] = lambda / (1 - P_0(lambda, alpha))`
    for a Zero-Truncated Poisson (`alpha is None`, `NaN`, or `<= 0`) or Zero-Truncated NB2 (`alpha > 0`).

    Why this helper is needed
    -------------------------
    In a Zero-Hurdle GLM, the count stage models `Y | Y > 0` using a zero-truncated distribution
    with untruncated rate parameter `lambda = exp(X @ beta_count)`. The unconditional mean is:

        mu = E[Y] = pi * E[Y | Y > 0] = pi * lambda / (1 - P_0(lambda, alpha))

    where `P_0(lambda) = exp(-lambda)` for Poisson and `P_0(lambda, alpha) = (1 + alpha * lambda)^(-1/alpha)`
    for NB2. Using `-expm1(...)` and clamping the conditional mean at `>= 1.0` preserves full
    floating-point precision as `lambda -> 0+` (where `lambda / (1 - P_0) -> 1.0`).
    """
    is_scalar = np.ndim(lam) == 0
    lam_arr = np.asarray(lam, dtype=np.float64)
    use_nb = alpha is not None and np.isfinite(alpha) and float(alpha) > 0.0
    if use_nb:
        a = float(alpha)
        log_p0 = -(1.0 / a) * np.log1p(np.maximum(a * lam_arr, 0.0))
        pos_mass = -np.expm1(log_p0)
    else:
        pos_mass = -np.expm1(-np.maximum(lam_arr, 0.0))

    cond_mu = np.where(pos_mass > 1e-12, np.maximum(lam_arr / np.maximum(pos_mass, 1e-12), 1.0), 1.0)
    return float(cond_mu) if is_scalar else cond_mu


@njit(parallel=True, nogil=True, cache=True)
def _predict_glm_batch_numba(
    X_mat: np.ndarray,
    row_group_id: np.ndarray,
    count_intercepts: np.ndarray,
    count_weights: np.ndarray,
    hurdle_intercepts: np.ndarray,
    hurdle_weights: np.ndarray,
    alphas: np.ndarray,
    is_hurdle_mat: np.ndarray,
    valid_model_mat: np.ndarray,
    has_any_hurdle: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-core parallel Numba kernel (`numba.prange`) that evaluates all `H` forecast horizons
    simultaneously across all `N` rows for standard Poisson/NB2 and Zero-Hurdle Poisson/NB2 GLMs.

    Returns
    -------
    mu_out : np.ndarray of shape (N, H)
    alpha_out : np.ndarray of shape (N, H)
    pi_out : np.ndarray of shape (N, H) when `has_any_hurdle=True` else (0, 0)
    lam_out : np.ndarray of shape (N, H) when `has_any_hurdle=True` else (0, 0)
    """
    n_rows, n_feats = X_mat.shape
    n_horizons = count_intercepts.shape[1]
    mu_out = np.empty((n_rows, n_horizons), dtype=np.float64)
    alpha_out = np.empty((n_rows, n_horizons), dtype=np.float64)
    if has_any_hurdle:
        pi_out = np.empty((n_rows, n_horizons), dtype=np.float64)
        lam_out = np.empty((n_rows, n_horizons), dtype=np.float64)
    else:
        pi_out = np.empty((0, 0), dtype=np.float64)
        lam_out = np.empty((0, 0), dtype=np.float64)

    for i in prange(n_rows):
        g = row_group_id[i]
        if g < 0:
            for h in range(n_horizons):
                mu_out[i, h] = np.nan
                alpha_out[i, h] = np.nan
                if has_any_hurdle:
                    pi_out[i, h] = np.nan
                    lam_out[i, h] = np.nan
            continue

        for h in range(n_horizons):
            if valid_model_mat[g, h] == 0:
                mu_out[i, h] = np.nan
                alpha_out[i, h] = np.nan
                if has_any_hurdle:
                    pi_out[i, h] = np.nan
                    lam_out[i, h] = np.nan
                continue

            z = count_intercepts[g, h]
            for p in range(n_feats):
                z += X_mat[i, p] * count_weights[g, h, p]
            if z < -50.0:
                z = -50.0
            elif z > 50.0:
                z = 50.0
            lam = math.exp(z)
            a = alphas[g, h]
            alpha_out[i, h] = a

            if is_hurdle_mat[g, h] == 1:
                z_h = hurdle_intercepts[g, h]
                for p in range(n_feats):
                    z_h += X_mat[i, p] * hurdle_weights[g, h, p]
                if z_h < -50.0:
                    z_h = -50.0
                elif z_h > 50.0:
                    z_h = 50.0
                pi = 1.0 / (1.0 + math.exp(-z_h))
                if a > 0.0:
                    u = a * lam if a * lam > 0.0 else 0.0
                    log_p0 = -(1.0 / a) * math.log1p(u)
                    pos_mass = -math.expm1(log_p0)
                else:
                    pos_mass = -math.expm1(-(lam if lam > 0.0 else 0.0))
                if pos_mass > 1e-12:
                    ratio = lam / pos_mass
                    cond_mu = ratio if ratio > 1.0 else 1.0
                else:
                    cond_mu = 1.0
                mu_out[i, h] = pi * cond_mu
                if has_any_hurdle:
                    pi_out[i, h] = pi
                    lam_out[i, h] = lam
            else:
                mu_out[i, h] = lam
                if has_any_hurdle:
                    pi_out[i, h] = np.nan
                    lam_out[i, h] = np.nan

    return mu_out, alpha_out, pi_out, lam_out


@njit(nogil=True, cache=True)
def _digamma_scalar(x: float) -> float:
    """Fast scalar digamma function psi(x) = d/dx ln(Gamma(x)) for x > 0."""
    res = 0.0
    while x < 8.0:
        res -= 1.0 / x
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    res += math.log(x) - 0.5 * inv_x
    res -= inv_x2 * (
        (1.0 / 12.0)
        - inv_x2 * (
            (1.0 / 120.0)
            - inv_x2 * ((1.0 / 252.0) - inv_x2 * ((1.0 / 240.0) - inv_x2 * (5.0 / 660.0)))
        )
    )
    return res


@njit(nogil=True, cache=True)
def _logit_nll_and_grad_eta(
    y: np.ndarray, eta: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Analytical weighted average negative log-likelihood and linear-predictor gradient for
    binary Logistic Regression (`Y in {0, 1}`, `pi = 1 / (1 + exp(-eta))`).
    """
    n = y.shape[0]
    grad_eta = np.empty(n, dtype=np.float64)
    nll = 0.0
    w_sum = 0.0
    for i in range(n):
        w_sum += weights[i]
    inv_w = 1.0 / w_sum if w_sum > 0.0 else (1.0 / max(n, 1))
    for i in range(n):
        yi = y[i]
        wi = weights[i]
        ei = eta[i]
        if ei < -30.0:
            ei = -30.0
        elif ei > 30.0:
            ei = 30.0
        if ei >= 0.0:
            exp_neg = math.exp(-ei)
            pi = 1.0 / (1.0 + exp_neg)
            log1p_exp = ei + math.log1p(exp_neg)
        else:
            exp_pos = math.exp(ei)
            pi = exp_pos / (1.0 + exp_pos)
            log1p_exp = math.log1p(exp_pos)
        ll_i = yi * ei - log1p_exp
        nll -= wi * ll_i
        grad_eta[i] = wi * (pi - yi) * inv_w
    return nll * inv_w, grad_eta


@njit(nogil=True, cache=True)
def _poisson_nll_and_grad_eta(
    y: np.ndarray, eta: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Analytical weighted average negative log-likelihood and linear-predictor gradient for
    standard Poisson Regression (`Y >= 0`, `mu = exp(eta)`).
    """
    n = y.shape[0]
    grad_eta = np.empty(n, dtype=np.float64)
    nll = 0.0
    w_sum = 0.0
    for i in range(n):
        w_sum += weights[i]
    inv_w = 1.0 / w_sum if w_sum > 0.0 else (1.0 / max(n, 1))
    for i in range(n):
        yi = y[i]
        wi = weights[i]
        ei = eta[i]
        if ei < -30.0:
            ei = -30.0
        elif ei > 30.0:
            ei = 30.0
        mu = math.exp(ei)
        ll_i = yi * ei - mu - math.lgamma(yi + 1.0)
        nll -= wi * ll_i
        grad_eta[i] = wi * (mu - yi) * inv_w
    return nll * inv_w, grad_eta


@njit(nogil=True, cache=True)
def _nb2_nll_and_grad_eta(
    y: np.ndarray, eta: np.ndarray, log_alpha: float, weights: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """
    Analytical weighted average negative log-likelihood and gradients w.r.t. linear predictor `eta`
    and `log_alpha = ln(alpha)` for standard Negative Binomial NB2 (`Y >= 0`, `mu = exp(eta)`).
    """
    n = y.shape[0]
    grad_eta = np.empty(n, dtype=np.float64)
    if log_alpha < -15.0:
        log_alpha = -15.0
    elif log_alpha > 10.0:
        log_alpha = 10.0
    alpha = math.exp(log_alpha)
    r = math.exp(-log_alpha)
    lgamma_r = math.lgamma(r)
    digamma_r = _digamma_scalar(r)
    r_log_r = -r * log_alpha

    nll = 0.0
    grad_tau = 0.0
    w_sum = 0.0
    for i in range(n):
        w_sum += weights[i]
    inv_w = 1.0 / w_sum if w_sum > 0.0 else (1.0 / max(n, 1))

    for i in range(n):
        yi = y[i]
        wi = weights[i]
        ei = eta[i]
        if ei < -30.0:
            ei = -30.0
        elif ei > 30.0:
            ei = 30.0
        mu = math.exp(ei)
        u = alpha * mu
        denom = 1.0 + u
        log1p_u = math.log1p(u)

        ll_i = (
            math.lgamma(yi + r)
            - lgamma_r
            - math.lgamma(yi + 1.0)
            + r_log_r
            + yi * ei
            - (yi + r) * math.log(r + mu)
        )
        nll -= wi * ll_i
        grad_eta[i] = wi * ((mu - yi) / denom) * inv_w

        d_ll_dr = (
            _digamma_scalar(yi + r)
            - digamma_r
            - log1p_u
            + (mu - yi) / (r + mu)
        )
        # Since r = exp(-log_alpha), dr / d(log_alpha) = -r, so d(-ll_i)/d(log_alpha) = r * d_ll_dr
        grad_tau += wi * (r * d_ll_dr) * inv_w

    return nll * inv_w, grad_eta, grad_tau


@njit(nogil=True, cache=True)
def _ztp_nll_and_grad_eta(y: np.ndarray, eta: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Analytical average negative log-likelihood and linear-predictor gradient for
    Zero-Truncated Poisson (`Y | Y > 0`, `y >= 1`).
    """
    n = y.shape[0]
    grad_eta = np.empty(n, dtype=np.float64)
    nll = 0.0
    inv_n = 1.0 / n
    for i in range(n):
        yi = y[i]
        ei = eta[i]
        if ei < -30.0:
            ei = -30.0
        elif ei > 30.0:
            ei = 30.0
        lam = math.exp(ei)
        if lam < 1e-12:
            log_pos = ei - 0.5 * lam
            cond_mu = 1.0 + 0.5 * lam
        else:
            pos_mass = -math.expm1(-lam)
            log_pos = math.log(pos_mass)
            cond_mu = lam / pos_mass
        ll_i = yi * ei - lam - math.lgamma(yi + 1.0) - log_pos
        nll -= ll_i
        grad_eta[i] = (cond_mu - yi) * inv_n
    return nll * inv_n, grad_eta


@njit(nogil=True, cache=True)
def _ztnb2_nll_and_grad_eta(
    y: np.ndarray, eta: np.ndarray, log_alpha: float
) -> tuple[float, np.ndarray, float]:
    """
    Analytical average negative log-likelihood and gradients w.r.t. linear predictor `eta`
    and `log_alpha = ln(alpha)` for Zero-Truncated Negative Binomial NB2 (`Y | Y > 0`, `y >= 1`).
    """
    n = y.shape[0]
    grad_eta = np.empty(n, dtype=np.float64)
    if log_alpha < -15.0:
        log_alpha = -15.0
    elif log_alpha > 10.0:
        log_alpha = 10.0
    alpha = math.exp(log_alpha)
    r = math.exp(-log_alpha)
    lgamma_r = math.lgamma(r)
    digamma_r = _digamma_scalar(r)
    r_log_r = -r * log_alpha

    nll = 0.0
    grad_tau = 0.0
    inv_n = 1.0 / n

    for i in range(n):
        yi = y[i]
        ei = eta[i]
        if ei < -30.0:
            ei = -30.0
        elif ei > 30.0:
            ei = 30.0
        lam = math.exp(ei)
        u = alpha * lam
        denom = 1.0 + u
        log1p_u = math.log1p(u)
        log_p0 = -r * log1p_u

        if log_p0 > -1e-12:
            pos_mass = max(-log_p0, 1e-15)
            log_pos = math.log(pos_mass)
            p0_over_pos = (1.0 + log_p0) / pos_mass
            cond_mu = 1.0 + 0.5 * (1.0 + alpha) * lam
        else:
            pos_mass = -math.expm1(log_p0)
            log_pos = math.log(pos_mass)
            p0 = math.exp(log_p0)
            p0_over_pos = p0 / pos_mass
            cond_mu = lam / pos_mass

        ll_i = (
            math.lgamma(yi + r)
            - lgamma_r
            - math.lgamma(yi + 1.0)
            + r_log_r
            + yi * ei
            - (yi + r) * math.log(r + lam)
            - log_pos
        )
        nll -= ll_i
        grad_eta[i] = ((cond_mu - yi) / denom) * inv_n

        d_logp0_dr = -log1p_u + (u / denom)
        d_ll_dr = (
            _digamma_scalar(yi + r)
            - digamma_r
            - log1p_u
            + (lam - yi) / (r + lam)
            + p0_over_pos * d_logp0_dr
        )
        # Since r = exp(-log_alpha), dr / d(log_alpha) = -r, so d(-ll_i)/d(log_alpha) = r * d_ll_dr
        grad_tau += (r * d_ll_dr) * inv_n

    return nll * inv_n, grad_eta, grad_tau


def _build_design_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Constructs a contiguous `float64` design matrix `X` with column 0 = `1.0` (`Intercept`)
    followed by all non-constant active columns in `feature_cols`, bypassing Patsy formula parsing.
    """
    n_rows = len(df)
    present = [f for f in feature_cols if f in df.columns]
    if not present or n_rows <= 2:
        return np.ones((n_rows, 1), dtype=np.float64), []

    raw_X = df[present].to_numpy(dtype=np.float64, copy=False)
    col_min = np.nanmin(raw_X, axis=0)
    col_max = np.nanmax(raw_X, axis=0)
    keep_mask = np.isfinite(col_min) & np.isfinite(col_max) & (col_max > col_min)
    active_idx = np.flatnonzero(keep_mask)
    if len(active_idx) == 0 or n_rows <= len(active_idx) + 1:
        return np.ones((n_rows, 1), dtype=np.float64), []

    active_features = [present[i] for i in active_idx]
    X = np.empty((n_rows, 1 + len(active_idx)), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = raw_X[:, active_idx]
    return X, active_features


def _prepare_weights(n: int, sample_weight: np.ndarray | None) -> np.ndarray:
    """Return a contiguous positive float64 weight vector of length `n`."""
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    w = np.ascontiguousarray(sample_weight, dtype=np.float64)
    if len(w) != n:
        raise ValueError(f"Expected sample_weight of length {n}, got {len(w)}.")
    return np.maximum(w, 0.0)


def _fit_logit_fast(
    y_bin: np.ndarray,
    X: np.ndarray,
    reg_alpha: float = 0.0,
    l1_wt: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray | None:
    """Fits (optionally weighted) binary Logistic Regression via L-BFGS-B using compiled Numba gradients."""
    k = X.shape[1]
    w = _prepare_weights(len(y_bin), sample_weight)
    w_sum = float(w.sum())
    p_bar = float(np.clip(float(np.dot(w, y_bin)) / max(w_sum, 1e-12), 1e-6, 1.0 - 1e-6))
    beta0 = np.zeros(k, dtype=np.float64)
    beta0[0] = math.log(p_bar / (1.0 - p_bar))
    l2_pen = float(reg_alpha) * (1.0 - float(l1_wt)) if reg_alpha > 0.0 else 0.0

    def _obj(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        nll, grad_eta = _logit_nll_and_grad_eta(y_bin, eta, w)
        grad_beta = X.T @ grad_eta
        if l2_pen > 0.0:
            nll += 0.5 * l2_pen * float(np.dot(beta[1:], beta[1:]))
            grad_beta[1:] += l2_pen * beta[1:]
        return float(nll), grad_beta

    bounds = [(-30.0, 30.0)] * k
    res = minimize(_obj, beta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 150, "ftol": 1e-9})
    if res.x is not None and np.all(np.isfinite(res.x)) and np.all(np.abs(res.x) < 50.0):
        return res.x
    return None


def _fit_poisson_fast(
    y: np.ndarray,
    X: np.ndarray,
    reg_alpha: float = 0.0,
    l1_wt: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray | None:
    """Fits (optionally weighted) standard Poisson Regression via L-BFGS-B using compiled Numba gradients."""
    k = X.shape[1]
    w = _prepare_weights(len(y), sample_weight)
    w_sum = float(w.sum())
    mu0 = max(float(np.dot(w, y)) / max(w_sum, 1e-12), 1e-4)
    beta0 = np.zeros(k, dtype=np.float64)
    beta0[0] = math.log(mu0)
    l2_pen = float(reg_alpha) * (1.0 - float(l1_wt)) if reg_alpha > 0.0 else 0.0

    def _obj(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        nll, grad_eta = _poisson_nll_and_grad_eta(y, eta, w)
        grad_beta = X.T @ grad_eta
        if l2_pen > 0.0:
            nll += 0.5 * l2_pen * float(np.dot(beta[1:], beta[1:]))
            grad_beta[1:] += l2_pen * beta[1:]
        return float(nll), grad_beta

    bounds = [(-30.0, 30.0)] * k
    res = minimize(_obj, beta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 150, "ftol": 1e-9})
    if res.x is not None and np.all(np.isfinite(res.x)) and np.all(np.abs(res.x) < 50.0):
        return res.x
    return None


def _fit_nb2_fast(
    y: np.ndarray,
    X: np.ndarray,
    reg_alpha: float = 0.0,
    l1_wt: float = 0.0,
    init_beta: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float | None]:
    """Fits (optionally weighted) standard Negative Binomial (NB2) via L-BFGS-B using compiled Numba gradients."""
    k = X.shape[1]
    w = _prepare_weights(len(y), sample_weight)
    w_sum = max(float(w.sum()), 1e-12)
    y_mean = float(np.dot(w, y)) / w_sum
    y_var = float(np.dot(w, (y - y_mean) ** 2)) / w_sum if len(y) > 1 else y_mean
    alpha0 = max((y_var - y_mean) / max(y_mean * y_mean, 1e-2), 0.1)
    alpha0 = float(np.clip(alpha0, 1e-3, 50.0))

    theta0 = np.zeros(k + 1, dtype=np.float64)
    if init_beta is not None and len(init_beta) == k:
        theta0[:k] = np.clip(init_beta, -25.0, 25.0)
    else:
        theta0[0] = math.log(max(y_mean, 1e-4))
    theta0[k] = math.log(alpha0)
    l2_pen = float(reg_alpha) * (1.0 - float(l1_wt)) if reg_alpha > 0.0 else 0.0

    def _obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:k]
        log_alpha = float(theta[k])
        eta = X @ beta
        nll, grad_eta, grad_tau = _nb2_nll_and_grad_eta(y, eta, log_alpha, w)
        grad = np.empty(k + 1, dtype=np.float64)
        grad[:k] = X.T @ grad_eta
        grad[k] = grad_tau
        if l2_pen > 0.0:
            nll += 0.5 * l2_pen * float(np.dot(beta[1:], beta[1:]))
            grad[1:k] += l2_pen * beta[1:]
        return float(nll), grad

    bounds = [*([(-30.0, 30.0)] * k), (-11.5, 9.2)]
    res = minimize(_obj, theta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
    if res.x is None or not np.all(np.isfinite(res.x)):
        return None, None
    beta_hat = res.x[:k]
    alpha_hat = float(math.exp(float(res.x[k])))
    if not np.all(np.abs(beta_hat) < 50.0) or not (1e-5 < alpha_hat < 1e4):
        return None, None
    return beta_hat, alpha_hat


def _fit_hurdle_binary_stage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
) -> tuple[dict[str, float], Any]:
    """
    Fits the binary hurdle stage `P(Y > 0 | X)` using compiled Numba analytical gradients + L-BFGS-B
    (`_fit_logit_fast`), honoring `sample_weight` if present on `train_df` (e.g. from `downsample_zeros_frac`).
    """
    zero_coefs = {str(f): 0.0 for f in feature_cols}
    if len(train_df) == 0:
        return {"Intercept": -30.0, **zero_coefs}, None

    y_bin = np.ascontiguousarray(
        (train_df[target_col].to_numpy(dtype=np.float64, copy=False) > 0.0).astype(np.float64)
    )
    n_pos = int(y_bin.sum())
    if n_pos == 0:
        return {"Intercept": -30.0, **zero_coefs}, None
    if n_pos == len(train_df):
        return {"Intercept": 30.0, **zero_coefs}, None

    w = (
        np.ascontiguousarray(train_df["sample_weight"].to_numpy(dtype=np.float64, copy=False))
        if "sample_weight" in train_df.columns
        else None
    )
    X, active_features = _build_design_matrix(train_df, feature_cols)
    if n_pos <= len(active_features) + 1 or (len(train_df) - n_pos) <= len(active_features) + 1:
        X = np.ones((len(train_df), 1), dtype=np.float64)
        active_features = []
    col_names = ["Intercept", *active_features]
    reg_a = float(regularization_alpha) if regularization_alpha is not None and regularization_alpha > 0.0 else 0.0

    params_dict: dict[str, float] | None = None
    if reg_a > 0.0 and float(l1_wt) > 0.0 and w is None:
        try:
            logit_model = sm.Logit(y_bin, X)
            res = logit_model.fit_regularized(
                alpha=reg_a, L1_wt=float(l1_wt), maxiter=100, disp=False
            )
            raw_vals = np.asarray(res.params, dtype=np.float64)
            if np.all(np.isfinite(raw_vals)):
                params_dict = {
                    col_names[i]: float(np.clip(raw_vals[i], -30.0, 30.0))
                    for i in range(len(col_names))
                }
        except Exception:
            params_dict = None

    if params_dict is None:
        beta_hat = _fit_logit_fast(y_bin, X, reg_alpha=reg_a, l1_wt=l1_wt, sample_weight=w)
        if beta_hat is not None:
            params_dict = {col_names[i]: float(beta_hat[i]) for i in range(len(col_names))}

    if params_dict is None:
        p_mean = (
            float(np.dot(w, y_bin)) / max(float(w.sum()), 1e-12)
            if w is not None
            else float(y_bin.mean())
        )
        p_bar = float(np.clip(p_mean, 1e-6, 1.0 - 1e-6))
        return {"Intercept": float(np.log(p_bar / (1.0 - p_bar))), **zero_coefs}, None

    hurdle_params = {
        "Intercept": float(params_dict.get("Intercept", 0.0)),
        **{str(f): float(params_dict.get(f, 0.0)) for f in feature_cols},
    }
    return hurdle_params, None


def _fit_ztp_fast(
    y: np.ndarray,
    X: np.ndarray,
    reg_alpha: float = 0.0,
    l1_wt: float = 0.0,
) -> np.ndarray | None:
    """Fits Zero-Truncated Poisson via L-BFGS-B using compiled analytical gradients."""
    k = X.shape[1]
    mu0 = max(float(y.mean()) - 1.0, 0.1)
    beta0 = np.zeros(k, dtype=np.float64)
    beta0[0] = math.log(mu0)
    l2_pen = float(reg_alpha) * (1.0 - float(l1_wt)) if reg_alpha > 0.0 else 0.0

    def _obj(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        nll, grad_eta = _ztp_nll_and_grad_eta(y, eta)
        grad_beta = X.T @ grad_eta
        if l2_pen > 0.0:
            nll += 0.5 * l2_pen * float(np.dot(beta[1:], beta[1:]))
            grad_beta[1:] += l2_pen * beta[1:]
        return float(nll), grad_beta

    bounds = [(-30.0, 30.0)] * k
    res = minimize(_obj, beta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 150, "ftol": 1e-9})
    if res.x is not None and np.all(np.isfinite(res.x)) and np.all(np.abs(res.x) < 50.0):
        return res.x
    return None


def _fit_ztnb2_fast(
    y: np.ndarray,
    X: np.ndarray,
    reg_alpha: float = 0.0,
    l1_wt: float = 0.0,
    init_beta: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float | None]:
    """Fits Zero-Truncated Negative Binomial (NB2) via L-BFGS-B using compiled analytical gradients."""
    k = X.shape[1]
    y_mean = float(y.mean())
    y_var = float(y.var()) if len(y) > 1 else y_mean
    alpha0 = max((y_var - y_mean) / max(y_mean * y_mean, 1e-2), 0.1)
    alpha0 = float(np.clip(alpha0, 1e-3, 50.0))

    theta0 = np.zeros(k + 1, dtype=np.float64)
    if init_beta is not None and len(init_beta) == k:
        theta0[:k] = np.clip(init_beta, -25.0, 25.0)
    else:
        theta0[0] = math.log(max(y_mean - 1.0, 0.1))
    theta0[k] = math.log(alpha0)
    l2_pen = float(reg_alpha) * (1.0 - float(l1_wt)) if reg_alpha > 0.0 else 0.0

    def _obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:k]
        log_alpha = float(theta[k])
        eta = X @ beta
        nll, grad_eta, grad_tau = _ztnb2_nll_and_grad_eta(y, eta, log_alpha)
        grad = np.empty(k + 1, dtype=np.float64)
        grad[:k] = X.T @ grad_eta
        grad[k] = grad_tau
        if l2_pen > 0.0:
            nll += 0.5 * l2_pen * float(np.dot(beta[1:], beta[1:]))
            grad[1:k] += l2_pen * beta[1:]
        return float(nll), grad

    bounds = [*([(-30.0, 30.0)] * k), (-11.5, 9.2)]
    res = minimize(_obj, theta0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
    if res.x is None or not np.all(np.isfinite(res.x)):
        return None, None
    beta_hat = res.x[:k]
    alpha_hat = float(math.exp(float(res.x[k])))
    if not np.all(np.abs(beta_hat) < 50.0) or not (1e-5 < alpha_hat < 1e4):
        return None, None
    return beta_hat, alpha_hat


def _fit_hurdle_count_stage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    base_type: str,
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
) -> tuple[dict[str, float], float | None, Any]:
    """
    Fits the zero-truncated count stage `P(Y = k | Y > 0, X)` (`k >= 1`) on positive rows
    using compiled Numba analytical gradients + L-BFGS-B (`_fit_ztnb2_fast` for `base_type="nbinom"`
    or `_fit_ztp_fast` for `base_type="poisson"`), with automatic fallback if positive observations
    are all `1` or if optimization encounters degenerate subsets.
    """
    zero_coefs = {str(f): 0.0 for f in feature_cols}
    y_all = train_df[target_col].to_numpy(dtype=np.float64, copy=False)
    pos_mask = y_all > 0.0
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return {"Intercept": -20.0, **zero_coefs}, None, None
    y_pos = np.ascontiguousarray(y_all[pos_mask], dtype=np.float64)
    if float(y_pos.max()) <= 1.0:
        return {"Intercept": -20.0, **zero_coefs}, None, None

    pos_df = train_df.loc[pos_mask]
    X, active_features = _build_design_matrix(pos_df, feature_cols)
    col_names = ["Intercept", *active_features]
    reg_a = float(regularization_alpha) if regularization_alpha is not None and regularization_alpha > 0.0 else 0.0

    # If L1 penalty is requested, use statsmodels fit_regularized on (y_pos, X)
    if reg_a > 0.0 and float(l1_wt) > 0.0:
        from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP, TruncatedLFPoisson

        try:
            dist_cls = TruncatedLFNegativeBinomialP if base_type == "nbinom" else TruncatedLFPoisson
            mod = dist_cls(y_pos, X)
            res = mod.fit_regularized(alpha=reg_a, L1_wt=float(l1_wt), maxiter=100, disp=False)
            params_arr = np.asarray(res.params, dtype=np.float64)
            if np.all(np.isfinite(params_arr)):
                beta_arr = params_arr[: len(col_names)]
                a_val = float(params_arr[-1]) if base_type == "nbinom" else None
                raw_params = {col_names[i]: float(beta_arr[i]) for i in range(len(col_names))}
                count_params = {
                    "Intercept": float(raw_params.get("Intercept", 0.0)),
                    **{str(f): float(raw_params.get(f, 0.0)) for f in feature_cols},
                }
                if a_val is not None and (not np.isfinite(a_val) or a_val <= 0.0 or a_val > 1e4):
                    a_val = None
                return count_params, a_val, None
        except Exception:
            pass

    ztp_beta = _fit_ztp_fast(y_pos, X, reg_alpha=reg_a, l1_wt=l1_wt)
    beta_hat: np.ndarray | None = None
    alpha_val: float | None = None

    if base_type == "nbinom":
        beta_hat, alpha_val = _fit_ztnb2_fast(
            y_pos, X, reg_alpha=reg_a, l1_wt=l1_wt, init_beta=ztp_beta
        )
        if beta_hat is None and active_features:
            X_int = np.ones((n_pos, 1), dtype=np.float64)
            b_int, a_int = _fit_ztnb2_fast(y_pos, X_int, reg_alpha=reg_a, l1_wt=l1_wt)
            if b_int is not None:
                return {"Intercept": float(b_int[0]), **zero_coefs}, a_int, None

    if beta_hat is None:
        beta_hat = ztp_beta
        alpha_val = None
        if beta_hat is None and active_features:
            X_int = np.ones((n_pos, 1), dtype=np.float64)
            b_int = _fit_ztp_fast(y_pos, X_int, reg_alpha=reg_a, l1_wt=l1_wt)
            if b_int is not None:
                return {"Intercept": float(b_int[0]), **zero_coefs}, None, None

    if beta_hat is None:
        mu_pos = max(float(y_pos.mean()) - 1.0, 1e-4)
        return {"Intercept": float(np.log(mu_pos)), **zero_coefs}, None, None

    raw_params = {col_names[i]: float(beta_hat[i]) for i in range(len(col_names))}
    count_params = {
        "Intercept": float(raw_params.get("Intercept", 0.0)),
        **{str(f): float(raw_params.get(f, 0.0)) for f in feature_cols},
    }
    return count_params, alpha_val, None


def _fit_standard_count_stage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str,
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
) -> tuple[dict[str, float], float | None, Any]:
    """
    Fits a standard (untruncated) Poisson (`model_type="poisson"`) or Negative Binomial NB2
    (`model_type="nbinom"`) GLM on all non-missing training rows (`y >= 0`) using compiled
    Numba analytical gradients + L-BFGS-B (`_fit_nb2_fast` or `_fit_poisson_fast`), honoring
    `sample_weight` if present on `train_df`, warm-starting NB2 from the fast Poisson solution,
    and falling back gracefully if L1 regularization is requested or if a subgroup is degenerate.
    """
    zero_coefs = {str(f): 0.0 for f in feature_cols}
    if len(train_df) == 0:
        return {"Intercept": -20.0, **zero_coefs}, None, None

    y_vec = np.ascontiguousarray(train_df[target_col].to_numpy(dtype=np.float64, copy=False))
    if float(y_vec.max()) <= 0.0:
        return {"Intercept": -20.0, **zero_coefs}, None, None

    w = (
        np.ascontiguousarray(train_df["sample_weight"].to_numpy(dtype=np.float64, copy=False))
        if "sample_weight" in train_df.columns
        else None
    )
    eff_n = float(w.sum()) if w is not None else float(len(train_df))

    n_pos = int((y_vec > 0.0).sum())
    X, active_features = _build_design_matrix(train_df, feature_cols)
    if n_pos <= len(active_features) + 1:
        X = np.ones((len(train_df), 1), dtype=np.float64)
        active_features = []
    col_names = ["Intercept", *active_features]
    reg_a = float(regularization_alpha) if regularization_alpha is not None and regularization_alpha > 0.0 else 0.0

    # If L1 penalty is requested, use statsmodels fit_regularized on (y_vec, X)
    if reg_a > 0.0 and float(l1_wt) > 0.0 and w is None:
        try:
            mod = sm.NegativeBinomial(y_vec, X) if model_type == "nbinom" else sm.Poisson(y_vec, X)
            res = mod.fit_regularized(alpha=reg_a, L1_wt=float(l1_wt), maxiter=100, disp=False)
            params_arr = np.asarray(res.params, dtype=np.float64)
            if np.all(np.isfinite(params_arr)):
                k_cols = len(col_names)
                beta_arr = params_arr[:k_cols]
                a_val = float(params_arr[k_cols]) if (model_type == "nbinom" and len(params_arr) > k_cols) else None
                raw_params = {col_names[i]: float(beta_arr[i]) for i in range(k_cols)}
                coef_dict = {
                    "Intercept": float(raw_params.get("Intercept", 0.0)),
                    **{str(f): float(raw_params.get(f, 0.0)) for f in feature_cols},
                }
                if a_val is not None and (not np.isfinite(a_val) or a_val <= 0.0 or a_val > 1e4):
                    a_val = None
                return coef_dict, a_val, None
        except Exception:
            pass

    pois_beta = _fit_poisson_fast(y_vec, X, reg_alpha=reg_a, l1_wt=l1_wt, sample_weight=w)
    # Guard against collinearity / quasi-separation slope explosion on sparse count subsets
    # (e.g. log1p(rolling_mean_w) vs. zero_freq_w when positive counts are rare {0, 1} events):
    if reg_a == 0.0 and len(active_features) > 0:
        needs_ridge = (
            pois_beta is None
            or float(np.max(np.abs(pois_beta[1:]))) > 5.0
            or n_pos < 20 * len(active_features)
        )
        if needs_ridge:
            reg_a = max(1e-3, 1.0 / max(eff_n, 1.0))
            pois_beta = _fit_poisson_fast(y_vec, X, reg_alpha=reg_a, l1_wt=0.0, sample_weight=w)

    beta_hat: np.ndarray | None = None
    alpha_val: float | None = None

    if model_type == "nbinom":
        beta_hat, alpha_val = _fit_nb2_fast(
            y_vec, X, reg_alpha=reg_a, l1_wt=l1_wt, init_beta=pois_beta, sample_weight=w
        )
        if (
            reg_a == 0.0
            and len(active_features) > 0
            and (beta_hat is None or float(np.max(np.abs(beta_hat[1:]))) > 5.0)
        ):
            reg_a = max(1e-3, 1.0 / max(eff_n, 1.0))
            if pois_beta is None or float(np.max(np.abs(pois_beta[1:]))) > 5.0:
                pois_beta = _fit_poisson_fast(y_vec, X, reg_alpha=reg_a, l1_wt=0.0, sample_weight=w)
            beta_hat, alpha_val = _fit_nb2_fast(
                y_vec, X, reg_alpha=reg_a, l1_wt=0.0, init_beta=pois_beta, sample_weight=w
            )
        if beta_hat is None and active_features:
            X_int = np.ones((len(train_df), 1), dtype=np.float64)
            b_int, a_int = _fit_nb2_fast(y_vec, X_int, reg_alpha=reg_a, l1_wt=l1_wt, sample_weight=w)
            if b_int is not None:
                return {"Intercept": float(b_int[0]), **zero_coefs}, a_int, None

    if beta_hat is None:
        beta_hat = pois_beta
        alpha_val = None
        if beta_hat is None and active_features:
            X_int = np.ones((len(train_df), 1), dtype=np.float64)
            b_int = _fit_poisson_fast(y_vec, X_int, reg_alpha=reg_a, l1_wt=l1_wt, sample_weight=w)
            if b_int is not None:
                return {"Intercept": float(b_int[0]), **zero_coefs}, None, None

    if beta_hat is None:
        mu_mean = float(np.dot(w, y_vec)) / max(eff_n, 1e-12) if w is not None else float(y_vec.mean())
        mu_bar = max(mu_mean, 1e-6)
        return {"Intercept": float(np.log(mu_bar)), **zero_coefs}, None, None

    raw_params = {col_names[i]: float(beta_hat[i]) for i in range(len(col_names))}
    coef_dict = {
        "Intercept": float(raw_params.get("Intercept", 0.0)),
        **{str(f): float(raw_params.get(f, 0.0)) for f in feature_cols},
    }
    return coef_dict, alpha_val, None


def train_single_glm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str = "nbinom",
    regularization_alpha: float | None = None,
    l1_wt: float = 0.0,
    hurdle: bool = False,
) -> tuple[dict[str, Any], Any]:
    """
    Trains a single log-link Poisson or Negative Binomial GLM (standard or zero-hurdle)
    for one group and one horizon.

    What it does
    ------------
    1. Filters `df` to training rows (`is_test == 0`) where `target_col` is not `NaN`.
    2. **Standard GLM (`hurdle=False`, `model_type in ("nbinom", "poisson")`)**:
       Builds the numeric design matrix `X` (`Intercept` + active features) directly and fits
       standard Negative Binomial NB2 (`_fit_nb2_fast`) or Poisson (`_fit_poisson_fast`) via
       compiled Numba analytical gradients + L-BFGS-B.
    3. **Zero-Hurdle GLM (`hurdle=True` or `model_type in ("hurdle_nbinom", "hurdle_poisson")`)**:
       Fits a two-part hurdle model using compiled Numba analytical gradients + L-BFGS-B:
       - *Binary hurdle stage*: logistic regression (`_fit_logit_fast`) on `1{target_col > 0}`
         across all training rows, producing `"hurdle_coefficients"`.
       - *Zero-truncated count stage*: Zero-Truncated NB2 (`"hurdle_nbinom"`, `_fit_ztnb2_fast`)
         or Zero-Truncated Poisson (`"hurdle_poisson"`, `_fit_ztp_fast`) fitted strictly on positive
         training rows (`target_col > 0`), producing `"coefficients"` and `"alpha"`.
    4. Separates the Negative Binomial dispersion parameter `"alpha"` from the regression
       `"coefficients"` dictionary so `"coefficients"` strictly maps `"Intercept"` and
       covariate names to their linear weights.

    Where it is used
    ----------------
    Called by `GLMForecaster.fit` and `time_series.train_glm`.
    """
    if hurdle and model_type in ("nbinom", "poisson"):
        model_type = f"hurdle_{model_type}"

    train_df = df[~df[target_col].isna()]
    if "is_test" in train_df.columns:
        train_df = train_df[train_df["is_test"] == 0]

    if model_type in ("hurdle_nbinom", "hurdle_poisson"):
        base_type = "nbinom" if model_type == "hurdle_nbinom" else "poisson"
        hurdle_params, hurdle_summary = _fit_hurdle_binary_stage(
            train_df,
            feature_cols,
            target_col,
            regularization_alpha=regularization_alpha,
            l1_wt=l1_wt,
        )
        count_params, alpha_val, count_summary = _fit_hurdle_count_stage(
            train_df,
            feature_cols,
            target_col,
            base_type=base_type,
            regularization_alpha=regularization_alpha,
            l1_wt=l1_wt,
        )
        prod_model = {
            "model_type": model_type,
            "horizon": target_col,
            "coefficients": count_params,
            "hurdle_coefficients": hurdle_params,
            "alpha": float(alpha_val) if alpha_val is not None and np.isfinite(alpha_val) else None,
        }
        return prod_model, count_summary if count_summary is not None else hurdle_summary

    if model_type not in ("nbinom", "poisson"):
        raise ValueError(
            f"Unsupported distribution: '{model_type}'. Supported options are "
            "'nbinom', 'poisson', 'hurdle_nbinom', 'hurdle_poisson'."
        )

    coef_dict, alpha_val, summary = _fit_standard_count_stage(
        train_df,
        feature_cols,
        target_col,
        model_type=model_type,
        regularization_alpha=regularization_alpha,
        l1_wt=l1_wt,
    )
    prod_model = {
        "model_type": model_type,
        "horizon": target_col,
        "coefficients": coef_dict,
        "alpha": float(alpha_val) if alpha_val is not None and np.isfinite(alpha_val) else None,
    }
    return prod_model, summary


@dataclass
class GLMModel:
    """
    Serializable container for grouped, multi-horizon Poisson / Negative-Binomial GLMs.

    What it does
    ------------
    Stores fitted per-group, per-horizon GLM parameter dictionaries (`model_type`,
    `coefficients`, `alpha`) along with feature and target metadata, and provides:
    - `.to_dataframe()` / `.from_dataframe()` for round-tripping with the tabular
      DataFrame representation (`GLMForecaster.model_data`),
    - `.coefficients_table()` for inspecting a tidy long-format DataFrame of all
      regression weights across groups and horizons, and
    - `.save()` / `.load()` for clean JSON persistence (`glm_meta.json`).

    Where it is used
    ----------------
    Created by `GLMForecaster.fit`, persisted by `GLMForecaster.save`, and used by
    `GLMForecaster.predict`.
    """
    target_cols: list[str]
    feature_cols: list[str]
    train_groups: list[str] = field(default_factory=list)
    models: dict[tuple[str, ...], dict[str, dict[str, Any]]] = field(default_factory=dict)
    horizon_feature_cols: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def group_key(values: Any) -> tuple[str, ...]:
        """Normalizes a scalar or tuple of group column values into a tuple of strings."""
        if not isinstance(values, tuple):
            values = (values,)
        return tuple(str(v) for v in values)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the nested `models` dictionary into a wide DataFrame with one row per group and one column per horizon."""
        rows = []
        for group_key, horizon_dict in self.models.items():
            row: dict[str, Any] = {}
            for col_name, key_val in zip(self.train_groups, group_key):
                if col_name in ("overdispersion", "positive_overdispersion", "velocity_tier", "demand_regime"):
                    try:
                        row[col_name] = int(key_val)
                    except ValueError:
                        row[col_name] = key_val
                else:
                    row[col_name] = key_val
            for target in self.target_cols:
                if target in horizon_dict:
                    row[target] = horizon_dict[target]
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
        feature_cols: list[str] | None = None,
    ) -> "GLMModel":
        """Reconstructs a `GLMModel` from a wide DataFrame (such as `GLMForecaster.model_data` or `load_glm_model_data`)."""
        from probabilistic_modeling.time_series import _parse_serialized_dict

        if target_cols is None:
            target_cols = [c for c in df.columns if c.startswith("target_")]
        train_groups = [c for c in df.columns if c not in target_cols]
        models: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        inferred_features: list[str] = list(feature_cols) if feature_cols else []

        for _, row in df.iterrows():
            g_vals = tuple(str(row[c]) for c in train_groups)
            h_map: dict[str, dict[str, Any]] = {}
            for t in target_cols:
                if t in row and pd.notna(row[t]):
                    m_dict = dict(_parse_serialized_dict(row[t]))
                    coefs = dict(m_dict.get("coefficients", {}))
                    if "alpha" in coefs and m_dict.get("alpha") is None:
                        m_dict["alpha"] = float(coefs.pop("alpha"))
                        m_dict["coefficients"] = coefs
                    h_map[t] = m_dict
                    if not inferred_features:
                        inferred_features = [k for k in coefs.keys() if k != "Intercept"]
            models[g_vals] = h_map
        return cls(
            target_cols=list(target_cols),
            feature_cols=inferred_features,
            train_groups=train_groups,
            models=models,
        )

    def coefficients_table(self) -> pd.DataFrame:
        """Returns a tidy long-format DataFrame of fitted GLM coefficients across groups and horizons."""
        has_hurdle = any(
            "hurdle_coefficients" in m_dict
            for horizon_dict in self.models.values()
            for m_dict in horizon_dict.values()
        )
        rows = []
        for group_key, horizon_dict in self.models.items():
            group_info = dict(zip(self.train_groups, group_key))
            for target, m_dict in horizon_dict.items():
                model_type = m_dict.get("model_type")
                alpha = m_dict.get("alpha")
                for feat, coef in m_dict.get("coefficients", {}).items():
                    row = {
                        **group_info,
                        "target": target,
                        "model_type": model_type,
                        "alpha": alpha,
                        "feature": feat,
                        "coefficient": coef,
                    }
                    if has_hurdle:
                        row["component"] = "count"
                    rows.append(row)
                if "hurdle_coefficients" in m_dict:
                    for feat, coef in m_dict.get("hurdle_coefficients", {}).items():
                        rows.append({
                            **group_info,
                            "target": target,
                            "model_type": model_type,
                            "alpha": None,
                            "feature": feat,
                            "coefficient": coef,
                            "component": "hurdle",
                        })
        return pd.DataFrame(rows)

    def save(self, out_dir: Path | str, prefix: str = "glm") -> None:
        """Saves all group/horizon GLM parameters and feature metadata to `<out_dir>/<prefix>_meta.json`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_cols": self.target_cols,
            "feature_cols": self.feature_cols,
            "train_groups": self.train_groups,
            "horizon_feature_cols": self.horizon_feature_cols,
            "models": [
                {"group_key": list(k), "horizons": v}
                for k, v in self.models.items()
            ],
        }
        with open(out_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "glm") -> "GLMModel":
        """Loads a `GLMModel` from `<out_dir>/<prefix>_meta.json`."""
        out_dir = Path(out_dir)
        with open(out_dir / f"{prefix}_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            target_cols=list(meta["target_cols"]),
            feature_cols=list(meta["feature_cols"]),
            train_groups=list(meta["train_groups"]),
            horizon_feature_cols=meta.get("horizon_feature_cols", {}),
        )
        for entry in meta.get("models", []):
            obj.models[tuple(entry["group_key"])] = entry["horizons"]
        return obj


class ProbabilisticForecaster(ABC):
    """
    Abstract base class for multi-horizon probabilistic demand forecasters.

    What it does
    ------------
    Defines the common lifecycle and state management shared by `GLMForecaster`,
    `CatBoostQuantileForecaster`, and `EmpiricalQuantileForecaster`:
    - **Fit once**: `forecaster.fit(ts)` binds the training panel's feature specification
      (`feature_spec_`), per-series metadata (`series_metadata_`), and column names via
      `_bind_training_context(ts)`.
    - **Flexible inference input resolution**: `_resolve_inference_inputs(...)` allows
      `predict()` and `compute_ddlt()` to accept a `DiscreteTimeSeries`, a pre-engineered
      `pd.DataFrame` (passed either as `df=` or positionally as `ts`), and an optional
      `period` filter, automatically merging missing training `series_metadata_` onto
      standalone DataFrames.
    - **Raw-data inference**: `transform(sales_df, ...)` and `predict_from_raw(sales_df, ...)`
      replay the saved `TimeSeriesFeatureSpec` on brand-new raw sales and covariate tables
      without needing the original training `DiscreteTimeSeries` object.
    - **Self-contained persistence**: `_save_base_context` and `_load_base_context` persist
      `forecaster_meta.json`, `feature_spec.json`, and `series_metadata.csv` alongside the
      subclass's model weights so `load_forecaster(dir)` restores a complete, ready-to-score
      pipeline.
    """

    forecaster_type: str = "base"

    def __init__(self) -> None:
        self.target_cols_: list[str] = []
        self.feature_cols_: list[str] = []
        self.cat_feature_cols_: list[str] = []
        self.ts_id_cols_: list[str] = []
        self.period_col_: str = "day"
        self.var_col_: str = "demand"
        self.feature_spec_: "TimeSeriesFeatureSpec | None" = None
        self.series_metadata_: pd.DataFrame | None = None

    def _bind_training_context(self, ts: "DiscreteTimeSeries") -> None:
        """Captures column lists, `feature_spec`, and `series_metadata` from `ts` during `fit()`."""
        self.target_cols_ = list(ts.target_cols)
        self.feature_cols_ = list(ts.feature_cols)
        self.cat_feature_cols_ = list(ts.categorical_feature_cols)
        self.ts_id_cols_ = list(ts.ts_id_cols)
        self.period_col_ = ts.period_col
        self.var_col_ = ts.var_col
        self.feature_spec_ = ts.feature_spec
        self.series_metadata_ = ts.series_metadata.copy() if not ts.series_metadata.empty else None

    def _resolve_inference_inputs(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        period: Any = None,
        fallback_target_cols: list[str] | None = None,
        fallback_feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str], list[str], str]:
        """
        Resolves the inference DataFrame, target columns, feature columns, and period column
        whether the caller passed a `DiscreteTimeSeries`, a standalone `pd.DataFrame`, or both.

        Why it is needed
        ----------------
        Callers may invoke `forecaster.predict(ts)`, `forecaster.predict(ts, period=1914)`,
        `forecaster.predict(df=custom_df)`, or `forecaster.predict(custom_df)`. Furthermore,
        when a standalone `df` is passed (or after loading a saved forecaster), per-series
        training metadata (`overdispersion`, `best_fit`, `distr_param_0..2`) might not be
        present on `df` yet. This helper normalizes all calling conventions and left-joins
        any missing `series_metadata_` columns onto `test_data`.
        """
        if isinstance(ts, pd.DataFrame):
            if df is None:
                df = ts
            ts = None

        target_cols = (
            list(ts.target_cols)
            if ts is not None
            else (list(self.target_cols_) if self.target_cols_ else list(fallback_target_cols or []))
        )
        feature_cols = (
            list(ts.feature_cols)
            if ts is not None
            else (list(self.feature_cols_) if self.feature_cols_ else list(fallback_feature_cols or []))
        )
        period_col = ts.period_col if ts is not None else self.period_col_

        if df is not None:
            test_data = df.copy()
            if period is not None and period_col in test_data.columns:
                if isinstance(period, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
                    test_data = test_data[test_data[period_col].isin(list(period))].copy()
                else:
                    test_data = test_data[test_data[period_col] == period].copy()
        elif ts is not None:
            test_data = ts.get_inference_data(period=period)
        else:
            raise ValueError("Provide either a 'DiscreteTimeSeries' or an inference DataFrame 'df'.")

        # Attach training series metadata (overdispersion, best_fit, distr_param_*) if missing:
        meta = self.series_metadata_
        if meta is None and ts is not None and not ts.series_metadata.empty:
            meta = ts.series_metadata
        if meta is not None and not meta.empty:
            join_keys = [c for c in self.ts_id_cols_ if c in meta.columns and c in test_data.columns]
            if not join_keys and ts is not None:
                join_keys = [c for c in ts.ts_id_cols if c in meta.columns and c in test_data.columns]
            missing_meta_cols = [c for c in meta.columns if c not in join_keys and c not in test_data.columns]
            if join_keys and missing_meta_cols:
                meta_sub = meta[join_keys + missing_meta_cols].drop_duplicates(subset=join_keys).copy()
                for k in join_keys:
                    meta_sub[k] = meta_sub[k].astype(test_data[k].dtype)
                test_data = pd.merge(test_data, meta_sub, on=join_keys, how="left")
                for int_col in ("overdispersion", "positive_overdispersion", "velocity_tier"):
                    if int_col in missing_meta_cols and int_col in test_data.columns:
                        test_data[int_col] = test_data[int_col].fillna(0).astype(int)
                if "demand_regime" in missing_meta_cols and "demand_regime" in test_data.columns:
                    test_data["demand_regime"] = test_data["demand_regime"].fillna(2).astype(int)
                if "demand_regime_name" in missing_meta_cols and "demand_regime_name" in test_data.columns:
                    test_data["demand_regime_name"] = (
                        test_data["demand_regime_name"]
                        .astype(str)
                        .replace({"nan": "intermittent", "None": "intermittent"})
                        .astype("category")
                    )

        return test_data, target_cols, feature_cols, period_col

    def transform(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
        external_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        period: Any = None,
        num_test_days: int = 0,
    ) -> pd.DataFrame:
        """
        Transforms raw `sales_df` and external feature tables into an inference-ready
        feature DataFrame using the `feature_spec_` and `series_metadata_` captured
        during training.
        """
        if self.feature_spec_ is None:
            raise ValueError("No 'feature_spec_' bound to this forecaster. Fit or load a forecaster first.")
        return self.feature_spec_.transform(
            sales_df=sales_df,
            series_metadata=self.series_metadata_,
            calendar_df=calendar_df,
            external_features=external_features,
            period=period,
            num_test_days=num_test_days,
        )

    def predict_from_raw(
        self,
        sales_df: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
        external_features: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
        period: Any = None,
    ) -> pd.DataFrame:
        """
        End-to-end reusable inference from raw `sales_df` + covariates without
        requiring the original training `DiscreteTimeSeries`.
        """
        inf_df = self.transform(
            sales_df=sales_df,
            calendar_df=calendar_df,
            external_features=external_features,
            period=period,
            num_test_days=0,
        )
        return self.predict(df=inf_df)

    def _save_base_context(self, out_dir: Path, extra_meta: dict[str, Any] | None = None) -> None:
        """Writes `forecaster_meta.json`, `feature_spec.json`, and `series_metadata.csv` to `out_dir`."""
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "forecaster_type": self.forecaster_type,
            "target_cols": self.target_cols_,
            "feature_cols": self.feature_cols_,
            "cat_feature_cols": self.cat_feature_cols_,
            "ts_id_cols": self.ts_id_cols_,
            "period_col": self.period_col_,
            "var_col": self.var_col_,
            **(extra_meta or {}),
        }
        with open(out_dir / "forecaster_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.feature_spec_ is not None:
            self.feature_spec_.save(out_dir / "feature_spec.json")
        if self.series_metadata_ is not None and not self.series_metadata_.empty:
            self.series_metadata_.to_csv(out_dir / "series_metadata.csv", index=False)

    def _load_base_context(self, out_dir: Path, meta: dict[str, Any]) -> None:
        """Restores `target_cols_`, `feature_cols_`, `feature_spec_`, and `series_metadata_` from `out_dir`."""
        from probabilistic_modeling.time_series import TimeSeriesFeatureSpec

        self.target_cols_ = list(meta.get("target_cols", []))
        self.feature_cols_ = list(meta.get("feature_cols", []))
        self.cat_feature_cols_ = list(meta.get("cat_feature_cols", []))
        self.ts_id_cols_ = list(meta.get("ts_id_cols", []))
        self.period_col_ = str(meta.get("period_col", "day"))
        self.var_col_ = str(meta.get("var_col", "demand"))
        spec_path = out_dir / "feature_spec.json"
        if spec_path.exists():
            self.feature_spec_ = TimeSeriesFeatureSpec.load(spec_path)
        series_meta_path = out_dir / "series_metadata.csv"
        if series_meta_path.exists():
            self.series_metadata_ = pd.read_csv(series_meta_path)

    @abstractmethod
    def fit(self, ts: "DiscreteTimeSeries") -> Any:
        ...

    @abstractmethod
    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
    ) -> pd.DataFrame:
        """
        Predicts every forecast horizon. `model` optionally overrides the fitted model
        container (`GLMModel` / coefficient table for GLMs, `CatBoostQuantileModel` for CatBoost).
        """
        ...

    @abstractmethod
    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
        max_service_level: float | None = None,
    ) -> pd.DataFrame:
        """Shared signature for all forecasters (`model` has the same meaning as in `predict`)."""
        ...

    @abstractmethod
    def save(self, out_dir: Path | str) -> Path:
        ...

    @classmethod
    @abstractmethod
    def load(cls, out_dir: Path | str) -> "ProbabilisticForecaster":
        ...


class GLMForecaster(ProbabilisticForecaster):
    """
    Grouped Poisson / Negative-Binomial GLM forecaster (standard or zero-hurdle) across multiple horizons.

    What it does
    ------------
    - **Training (`fit`)**:
      * When `hurdle=False` (default): Partitions the training panel by `train_groups + ["overdispersion"]`.
        Fits a Negative Binomial GLM (`model_type="nbinom"`) if `overdispersion == 1` or a Poisson GLM
        (`model_type="poisson"`) if `overdispersion == 0`.
      * When `hurdle=True`: Partitions the training panel by `train_groups + ["positive_overdispersion"]`,
        where `positive_overdispersion` is computed strictly on positive training observations (`demand > 0`).
        Fits a Zero-Hurdle Negative Binomial GLM (`model_type="hurdle_nbinom"`) if `positive_overdispersion == 1`
        or a Zero-Hurdle Poisson GLM (`model_type="hurdle_poisson"`) if `positive_overdispersion == 0`.
    - **Prediction (`predict`)**:
      * Standard GLM: Evaluates `target_t_plus_h_mu = exp(X @ beta)` and attaches `target_t_plus_h_alpha`.
      * Zero-Hurdle GLM: Evaluates hurdle probability `target_t_plus_h_pi = sigmoid(X @ gamma)`,
        untruncated count rate `target_t_plus_h_lambda = exp(X @ beta)`, `target_t_plus_h_alpha`,
        and unconditional expected value `target_t_plus_h_mu = pi * lambda / (1 - P_0(lambda, alpha))`.
    - **DDLT (`compute_ddlt`)**: Passes the predicted GLM parameter columns to
      `DDLTCalculator.from_glm_group`, which constructs truncated Poisson/NB2 or Zero-Hurdle
      Poisson/NB2 PMFs in compiled Numba log-space and convolves them over the lead-time distribution.
    """

    forecaster_type = "glm"

    def __init__(
        self,
        train_groups: tuple[str, ...] | list[str] | None = None,
        regularization_alpha: float | None = None,
        l1_wt: float = 0.0,
        restrict_future_leads: bool = False,
        hurdle: bool = False,
        velocity_tiers: bool | str = False,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.train_groups = list(train_groups) if train_groups is not None else []
        self.regularization_alpha = regularization_alpha
        self.l1_wt = l1_wt
        self.restrict_future_leads = restrict_future_leads
        self.hurdle = bool(hurdle)
        self.velocity_tiers = velocity_tiers
        self.n_jobs = int(n_jobs)
        self.model_data: pd.DataFrame | None = None
        self.model: GLMModel | None = None

    def _resolve_velocity_col(self) -> str | None:
        if not self.velocity_tiers:
            return None
        if isinstance(self.velocity_tiers, str):
            val = self.velocity_tiers.strip().lower()
            if val in ("demand_regime", "sbc", "syntetos_boylan"):
                return "demand_regime"
            if val in ("velocity_tier", "velocity", "true"):
                return "velocity_tier"
            if val in ("false", "none", ""):
                return None
            raise ValueError(
                f"Unsupported velocity_tiers='{self.velocity_tiers}'. "
                "Use False, True, 'velocity_tier', or 'demand_regime'."
            )
        return "velocity_tier"

    def _ensure_tier_and_dispersion_columns(self, ts: "DiscreteTimeSeries", needed_cols: list[str]) -> None:
        """Computes any missing `positive_overdispersion`, `velocity_tier`, or `demand_regime` columns on `ts.df` using post-first-sale history."""
        missing = [c for c in needed_cols if c not in ts.df.columns]
        if not missing or not ts.ts_id_cols:
            return
        from probabilistic_modeling.time_series import classify_demand_regimes

        src_df = (
            ts.raw_df
            if (hasattr(ts, "raw_df") and isinstance(ts.raw_df, pd.DataFrame) and not ts.raw_df.empty)
            else ts.df
        )
        if ts.var_col not in src_df.columns:
            return
        if ts.test_start_period is not None:
            train_slice = src_df[src_df[ts.period_col] < ts.test_start_period]
        elif "is_test" in src_df.columns:
            train_slice = src_df[src_df["is_test"] == 0]
        else:
            train_slice = src_df

        sbc_df = classify_demand_regimes(
            train_slice, ts_id_cols=ts.ts_id_cols, period_col=ts.period_col, var_col=ts.var_col
        )
        add_cols = [c for c in missing if c in sbc_df.columns]
        if add_cols:
            ts.df = pd.merge(ts.df, sbc_df[list(ts.ts_id_cols) + add_cols], on=list(ts.ts_id_cols), how="left")
            for int_col in ("overdispersion", "positive_overdispersion", "velocity_tier"):
                if int_col in add_cols:
                    ts.df[int_col] = ts.df[int_col].fillna(0).astype(int)
            if "demand_regime" in add_cols:
                ts.df["demand_regime"] = ts.df["demand_regime"].fillna(2).astype(int)

        if self.series_metadata_ is not None:
            new_meta_cols = [c for c in missing if c in ts.df.columns and c not in self.series_metadata_.columns]
            if new_meta_cols:
                meta_sub = ts.df[list(ts.ts_id_cols) + new_meta_cols].drop_duplicates(subset=list(ts.ts_id_cols))
                self.series_metadata_ = pd.merge(self.series_metadata_, meta_sub, on=list(ts.ts_id_cols), how="left")

    def _resolve_n_jobs(self, n_tasks: int) -> int:
        if n_tasks <= 1 or self.n_jobs == 1:
            return 1
        cpu_cnt = os.cpu_count() or 1
        if self.n_jobs <= 0:
            workers = max(1, cpu_cnt + 1 + self.n_jobs) if self.n_jobs < -1 else cpu_cnt
        else:
            workers = self.n_jobs
        return max(1, min(workers, n_tasks))

    def fit(self, ts: "DiscreteTimeSeries") -> pd.DataFrame:
        """
        Fits grouped multi-horizon Poisson / Negative Binomial GLMs (standard or zero-hurdle) on `ts`
        and returns the wide `model_data` DataFrame (while also storing `self.model: GLMModel`).
        """
        t0 = time.perf_counter()
        self._bind_training_context(ts)
        dispersion_col = "positive_overdispersion" if self.hurdle else "overdispersion"
        vel_col = self._resolve_velocity_col()

        needed_cols = [dispersion_col]
        if vel_col is not None:
            needed_cols.append(vel_col)
        for col in ("velocity_tier", "demand_regime"):
            if col in self.train_groups and col not in needed_cols:
                needed_cols.append(col)
        self._ensure_tier_and_dispersion_columns(ts, needed_cols)
        if dispersion_col not in ts.df.columns and "overdispersion" in ts.df.columns:
            dispersion_col = "overdispersion"

        partition_cols = list(self.train_groups)
        if vel_col is not None and vel_col not in partition_cols and vel_col in ts.df.columns:
            partition_cols.append(vel_col)
        if dispersion_col not in partition_cols:
            partition_cols.append(dispersion_col)

        train_df = ts.df[ts.df["is_test"] == 0] if ts.test_start_period is not None else ts.df
        groups_list = list(train_df.groupby(partition_cols, observed=True))
        n_groups = len(groups_list)
        n_horizons = len(ts.target_cols)
        n_tasks = n_groups * n_horizons
        workers = self._resolve_n_jobs(n_tasks)
        logger.info(
            "Starting %sGLM training across %d group(s) and %d horizon(s) (%d training rows, %d worker(s))...",
            "Hurdle " if self.hurdle else "",
            n_groups,
            n_horizons,
            len(train_df),
            workers,
        )

        horizon_feature_cols = {
            target: ts.get_features_for_horizon(target, restrict_future_leads=self.restrict_future_leads)
            for target in ts.target_cols
        }
        disp_idx = partition_cols.index(dispersion_col)

        tasks: list[tuple[int, tuple[Any, ...], pd.DataFrame, int, str, str]] = []
        group_rows: list[dict[str, Any]] = []
        for g_idx, (group_name, group_df) in enumerate(groups_list):
            g_tuple = group_name if isinstance(group_name, tuple) else (group_name,)
            is_overdispersed = g_tuple[disp_idx]
            if self.hurdle:
                model_type = "hurdle_nbinom" if is_overdispersed == 1 else "hurdle_poisson"
            else:
                model_type = "nbinom" if is_overdispersed == 1 else "poisson"
            group_rows.append(dict(zip(partition_cols, g_tuple)))
            for h_idx, target in enumerate(ts.target_cols, start=1):
                tasks.append((g_idx, g_tuple, group_df, h_idx, target, model_type))

        def _fit_one_task(
            task: tuple[int, tuple[Any, ...], pd.DataFrame, int, str, str]
        ) -> tuple[int, str, dict[str, Any]]:
            g_idx, g_tuple, g_df, h_idx, target, model_type = task
            h_t0 = time.perf_counter()
            h_features = horizon_feature_cols[target]
            display_group = g_tuple[0] if len(g_tuple) == 1 else g_tuple
            logger.debug(
                "Training %s GLM for group %s, horizon [%d/%d] (%s)...",
                model_type,
                display_group,
                h_idx,
                n_horizons,
                target,
            )
            effective_reg_alpha = (
                self.regularization_alpha
                if self.regularization_alpha is not None
                else (None if self.hurdle else 1e-3)
            )
            model_dict, summary = train_single_glm(
                g_df,
                h_features,
                target,
                model_type,
                regularization_alpha=effective_reg_alpha,
                l1_wt=self.l1_wt,
                hurdle=self.hurdle,
            )
            if summary is not None:
                logger.debug("%s", summary)
            logger.debug(
                "Completed %s GLM for group %s, %s in %.2fs",
                model_type,
                display_group,
                target,
                time.perf_counter() - h_t0,
            )
            return g_idx, target, model_dict

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                task_outputs = list(pool.map(_fit_one_task, tasks))
        else:
            task_outputs = [_fit_one_task(t) for t in tasks]

        for g_idx, target, model_dict in task_outputs:
            group_rows[g_idx][target] = model_dict

        result_df = pd.DataFrame(group_rows)
        self.model_data = result_df
        self.model = GLMModel.from_dataframe(result_df, target_cols=list(ts.target_cols), feature_cols=list(ts.feature_cols))
        self.model.horizon_feature_cols = horizon_feature_cols
        logger.info("Finished %sGLM training in %.2fs", "Hurdle " if self.hurdle else "", time.perf_counter() - t0)
        return result_df

    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: pd.DataFrame | GLMModel | None = None,
        period: Any = None,
        model_data: pd.DataFrame | GLMModel | None = None,
    ) -> pd.DataFrame:
        """
        Generates `<target>_mu` and `<target>_alpha` predictions (plus `<target>_pi` and
        `<target>_lambda` for zero-hurdle models) for every horizon in `target_cols`.

        Why string-keyed fallback lookup is included
        --------------------------------------------
        When a `GLMModel` is loaded from JSON (`glm_meta.json`), its group keys are tuples of
        strings (e.g., `('CA_1', '1')`), whereas an in-memory DataFrame may have integer or
        categorical group keys (e.g., `('CA_1', 1)`). Maintaining both `model_dict_map` and
        `str_model_dict_map` guarantees seamless group matching regardless of whether the
        model was just fitted in memory or loaded from disk.
        """
        t0 = time.perf_counter()
        if model is None:
            model = model_data  # backward-compatible alias
        model_data = model
        if isinstance(model_data, GLMModel):
            glm_container = model_data
            model_df = glm_container.to_dataframe()
        elif isinstance(model_data, pd.DataFrame):
            glm_container = GLMModel.from_dataframe(model_data)
            model_df = model_data
        elif self.model is not None:
            glm_container = self.model
            model_df = self.model_data if self.model_data is not None else glm_container.to_dataframe()
        elif self.model_data is not None:
            glm_container = GLMModel.from_dataframe(self.model_data)
            model_df = self.model_data
        else:
            raise ValueError("GLMForecaster has not been fitted yet.")

        test_data, target_cols, feature_cols, _ = self._resolve_inference_inputs(
            ts=ts,
            df=df,
            period=period,
            fallback_target_cols=glm_container.target_cols,
            fallback_feature_cols=glm_container.feature_cols,
        )
        train_groups = [col for col in model_df.columns if col not in target_cols]
        # Fallback if a standalone DataFrame was passed without dispersion or velocity/regime columns:
        if "positive_overdispersion" in train_groups and "positive_overdispersion" not in test_data.columns:
            if "overdispersion" in test_data.columns:
                test_data["positive_overdispersion"] = test_data["overdispersion"].fillna(0).astype(int)
            else:
                test_data["positive_overdispersion"] = 0
        if "overdispersion" in train_groups and "overdispersion" not in test_data.columns:
            if "positive_overdispersion" in test_data.columns:
                test_data["overdispersion"] = test_data["positive_overdispersion"].fillna(0).astype(int)
            else:
                test_data["overdispersion"] = 0
        for tier_col in ("velocity_tier", "demand_regime"):
            if tier_col in train_groups and tier_col not in test_data.columns:
                test_data[tier_col] = 0

        has_hurdle = self.hurdle or any(
            isinstance(m_dict, dict)
            and ("hurdle_coefficients" in m_dict or str(m_dict.get("model_type", "")).startswith("hurdle_"))
            for h_dict in glm_container.models.values()
            for m_dict in h_dict.values()
        )
        logger.info("Generating GLM predictions for %d rows across %d horizons...", len(test_data), len(target_cols))

        model_dict_map = model_df.set_index(train_groups).to_dict(orient="index")
        # Also prepare string-keyed fallback map so loaded JSON models match categorical/int columns seamlessly:
        str_model_dict_map = {
            (tuple(str(x) for x in k) if isinstance(k, tuple) else str(k)): v
            for k, v in model_dict_map.items()
        }

        def _lookup_group_model(group_keys: Any) -> dict[str, Any] | None:
            lookup_key = group_keys[0] if isinstance(group_keys, tuple) and len(group_keys) == 1 else group_keys
            row_models = model_dict_map.get(lookup_key)
            if row_models is not None:
                return row_models
            str_key = tuple(str(x) for x in group_keys) if isinstance(group_keys, tuple) else str(group_keys)
            if isinstance(str_key, tuple) and len(str_key) == 1:
                str_key = str_key[0]
            row_models = str_model_dict_map.get(str_key)
            if row_models is not None:
                return row_models
            if not isinstance(group_keys, tuple) or len(group_keys) < 2:
                return None

            # Fallback 1: flip trailing overdispersion / positive_overdispersion flag (0 <-> 1)
            if train_groups and train_groups[-1] in ("overdispersion", "positive_overdispersion"):
                try:
                    alt_flag = 1 - int(group_keys[-1])
                    alt_key = (*group_keys[:-1], alt_flag)
                    row_models = model_dict_map.get(alt_key) or str_model_dict_map.get(tuple(str(x) for x in alt_key))
                    if row_models is not None:
                        return row_models
                except (ValueError, TypeError):
                    pass

            # Fallback 2: if velocity_tier or demand_regime is in train_groups and a specific tier
            # was unseen in training for this parent group, match the nearest available tier:
            tier_indices = [i for i, col in enumerate(train_groups) if col in ("velocity_tier", "demand_regime")]
            if tier_indices:
                t_idx = tier_indices[-1]
                for candidate_tier in (1, 0, 2, 3):
                    cand_list = list(group_keys)
                    cand_list[t_idx] = candidate_tier
                    cand_key = tuple(cand_list)
                    row_models = model_dict_map.get(cand_key) or str_model_dict_map.get(tuple(str(x) for x in cand_key))
                    if row_models is not None:
                        return row_models
                    if train_groups[-1] in ("overdispersion", "positive_overdispersion"):
                        try:
                            cand_list[-1] = 1 - int(group_keys[-1])
                            cand_alt = tuple(cand_list)
                            row_models = model_dict_map.get(cand_alt) or str_model_dict_map.get(
                                tuple(str(x) for x in cand_alt)
                            )
                            if row_models is not None:
                                return row_models
                        except (ValueError, TypeError):
                            pass
            return None

        present_features = [c for c in feature_cols if c in test_data.columns]
        feat_to_idx = {f: idx for idx, f in enumerate(present_features)}
        n_rows = len(test_data)
        n_feats = len(present_features)
        n_horizons = len(target_cols)

        if n_rows > 0:
            X_mat = (
                np.ascontiguousarray(test_data[present_features].fillna(0.0).to_numpy(dtype=np.float64))
                if n_feats > 0
                else np.empty((n_rows, 0), dtype=np.float64)
            )
            row_group_id = np.full(n_rows, -1, dtype=np.int64)
            grouped_indices = test_data.groupby(train_groups, observed=True).indices
            matched_models: list[dict[str, Any]] = []

            for group_keys, pos_indices in grouped_indices.items():
                row_models = _lookup_group_model(group_keys)
                if row_models is None:
                    lookup_key = (
                        group_keys[0] if isinstance(group_keys, tuple) and len(group_keys) == 1 else group_keys
                    )
                    logger.warning("No trained GLM found for group %s; skipping %d rows", lookup_key, len(pos_indices))
                    continue
                g_id = len(matched_models)
                matched_models.append(row_models)
                row_group_id[pos_indices] = g_id

            n_groups = max(len(matched_models), 1)
            count_intercepts = np.zeros((n_groups, n_horizons), dtype=np.float64)
            count_weights = np.zeros((n_groups, n_horizons, n_feats), dtype=np.float64)
            hurdle_intercepts = np.zeros((n_groups, n_horizons), dtype=np.float64)
            hurdle_weights = np.zeros((n_groups, n_horizons, n_feats), dtype=np.float64)
            alphas = np.full((n_groups, n_horizons), np.nan, dtype=np.float64)
            is_hurdle_mat = np.zeros((n_groups, n_horizons), dtype=np.uint8)
            valid_model_mat = np.zeros((n_groups, n_horizons), dtype=np.uint8)

            for g_id, row_models in enumerate(matched_models):
                for h_idx, target in enumerate(target_cols):
                    model_dict = row_models.get(target)
                    if not isinstance(model_dict, dict) or "coefficients" not in model_dict:
                        continue
                    valid_model_mat[g_id, h_idx] = 1
                    coefs = model_dict["coefficients"]
                    count_intercepts[g_id, h_idx] = float(coefs.get("Intercept", 0.0))
                    for f_name, f_val in coefs.items():
                        if f_name != "Intercept":
                            f_idx = feat_to_idx.get(f_name)
                            if f_idx is not None and f_val:
                                count_weights[g_id, h_idx, f_idx] = float(f_val)

                    alpha_val = model_dict.get("alpha", coefs.get("alpha", np.nan))
                    if alpha_val is not None and np.isfinite(alpha_val):
                        alphas[g_id, h_idx] = float(alpha_val)

                    if "hurdle_coefficients" in model_dict or str(model_dict.get("model_type", "")).startswith(
                        "hurdle_"
                    ):
                        is_hurdle_mat[g_id, h_idx] = 1
                        h_coefs = model_dict.get("hurdle_coefficients", {"Intercept": 0.0})
                        hurdle_intercepts[g_id, h_idx] = float(h_coefs.get("Intercept", 0.0))
                        for f_name, f_val in h_coefs.items():
                            if f_name != "Intercept":
                                f_idx = feat_to_idx.get(f_name)
                                if f_idx is not None and f_val:
                                    hurdle_weights[g_id, h_idx, f_idx] = float(f_val)

            mu_out, alpha_out, pi_out, lam_out = _predict_glm_batch_numba(
                X_mat,
                row_group_id,
                count_intercepts,
                count_weights,
                hurdle_intercepts,
                hurdle_weights,
                alphas,
                is_hurdle_mat,
                valid_model_mat,
                has_hurdle,
            )

            for h_idx, target in enumerate(target_cols):
                if has_hurdle:
                    test_data[f"{target}_pi"] = pi_out[:, h_idx]
                    test_data[f"{target}_lambda"] = lam_out[:, h_idx]
                test_data[f"{target}_mu"] = mu_out[:, h_idx]
                test_data[f"{target}_alpha"] = alpha_out[:, h_idx]
        else:
            for target in target_cols:
                if has_hurdle:
                    test_data[f"{target}_pi"] = np.nan
                    test_data[f"{target}_lambda"] = np.nan
                test_data[f"{target}_mu"] = np.nan
                test_data[f"{target}_alpha"] = np.nan

        compute_forecasting_metrics(test_data, target_cols, [f"{target}_mu" for target in target_cols])
        logger.info("GLM predictions completed in %.2fs", time.perf_counter() - t0)
        return test_data

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: pd.DataFrame | GLMModel | None = None,
        period: Any = None,
        max_service_level: float | None = None,
        model_data: pd.DataFrame | GLMModel | None = None,
    ) -> pd.DataFrame:
        """
        Generates GLM predictions and propagates them through `DDLTCalculator.from_glm_group`
        across all `lead_time_keys` groups.
        """
        if lead_time is None:
            raise ValueError("'lead_time' distribution must be provided.")
        predictions = self.predict(ts, df=df, model=model if model is not None else model_data, period=period)
        _, target_cols, _, period_col = self._resolve_inference_inputs(
            ts=ts,
            df=predictions,
            fallback_target_cols=self.model.target_cols if self.model is not None else None,
        )
        return _apply_ddlt_across_groups(
            predictions=predictions,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            service_levels=service_levels,
            return_distribution=return_distribution,
            group_fn=lambda gdf, lt_distr: DDLTCalculator(
                lt_distr, cycle_time, max_service_level=max_service_level
            ).from_glm_group(
                gdf,
                lt_distr,
                target_cols,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=period_col,
            ),
        )

    def save(self, out_dir: Path | str, prefix: str = "glm") -> Path:
        """Persists the fitted `GLMForecaster` bundle (`forecaster_meta.json`, `feature_spec.json`, `series_metadata.csv`, and `glm_meta.json`) to `out_dir`."""
        if self.model is None and self.model_data is not None:
            self.model = GLMModel.from_dataframe(self.model_data, self.target_cols_, self.feature_cols_)
        if self.model is None:
            raise ValueError("GLMForecaster has not been fitted yet.")
        out_path = Path(out_dir)
        self._save_base_context(
            out_path,
            extra_meta={
                "train_groups": self.train_groups,
                "regularization_alpha": self.regularization_alpha,
                "l1_wt": self.l1_wt,
                "restrict_future_leads": self.restrict_future_leads,
                "hurdle": self.hurdle,
                "velocity_tiers": self.velocity_tiers,
                "prefix": prefix,
            },
        )
        self.model.save(out_path, prefix=prefix)
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "glm") -> "GLMForecaster":
        """Restores a fitted `GLMForecaster` from `out_dir`."""
        out_path = Path(out_dir)
        meta_file = out_path / "forecaster_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        saved_prefix = meta.get("prefix", prefix)
        glm_model = GLMModel.load(out_path, prefix=saved_prefix)
        inferred_hurdle = any(
            isinstance(m_dict, dict)
            and ("hurdle_coefficients" in m_dict or str(m_dict.get("model_type", "")).startswith("hurdle_"))
            for h_dict in glm_model.models.values()
            for m_dict in h_dict.values()
        )
        inferred_vel: bool | str = False
        if "demand_regime" in glm_model.train_groups:
            inferred_vel = "demand_regime"
        elif "velocity_tier" in glm_model.train_groups:
            inferred_vel = True
        forecaster = cls(
            train_groups=meta.get(
                "train_groups",
                [
                    c
                    for c in glm_model.train_groups
                    if c not in ("overdispersion", "positive_overdispersion", "velocity_tier", "demand_regime")
                ],
            ),
            regularization_alpha=meta.get("regularization_alpha"),
            l1_wt=float(meta.get("l1_wt", 0.0)),
            restrict_future_leads=bool(meta.get("restrict_future_leads", False)),
            hurdle=bool(meta.get("hurdle", inferred_hurdle)),
            velocity_tiers=meta.get("velocity_tiers", inferred_vel),
        )
        forecaster.model = glm_model
        forecaster.model_data = glm_model.to_dataframe()
        if meta:
            forecaster._load_base_context(out_path, meta)
        else:
            forecaster.target_cols_ = list(glm_model.target_cols)
            forecaster.feature_cols_ = list(glm_model.feature_cols)
        return forecaster


class CatBoostQuantileForecaster(ProbabilisticForecaster):
    """
    Multi-horizon CatBoost `MultiQuantile` forecaster with Numba-accelerated
    quantile-to-distribution reconstruction and DDLT calculation.

    What it does
    ------------
    - **Training (`fit`)**: For each partition in `train_groups` and each horizon
      `target_t_plus_h`, trains a joint `CatBoostRegressor(loss_function='MultiQuantile:alpha=...')`
      using a leak-free temporal validation split (`num_valid_days` immediately prior to
      `test_start_period`).
    - **Prediction (`predict`)**: Predicts all `K` quantiles simultaneously per horizon,
      applies non-crossing rearrangement and non-negativity clipping (`predict_catboost_quantile`),
      and computes the implied discrete expected value `<target>_mu` via `quantiles_to_mean`.
    - **DDLT (`compute_ddlt`)**: Passes predicted quantile columns to
      `DDLTCalculator.from_quantiles_group`, which converts each row's quantiles into a
      `DiscreteDistribution` via `_quantiles_to_pmf_numba` and convolves them over the
      lead-time distribution.
    """

    forecaster_type = "catboost"

    def __init__(
        self,
        quantiles: tuple[float, ...] | list[float] = DEFAULT_QUANTILES,
        train_groups: tuple[str, ...] | list[str] | None = None,
        extra_cat_cols: tuple[str, ...] = ("item_id", "dept_id", "cat_id", "demand_regime_name"),
        num_valid_days: int = 28,
        max_train_days: int | None = None,
        catboost_params: dict[str, Any] | None = None,
        tol: float = 1e-4,
        restrict_future_leads: bool = False,
        velocity_tiers: bool | str = False,
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in validate_quantile_levels(quantiles))
        self.train_groups = list(train_groups) if train_groups else []
        self.extra_cat_cols = tuple(extra_cat_cols)
        self.num_valid_days = num_valid_days
        self.max_train_days = max_train_days
        self.catboost_params = catboost_params
        self.tol = tol
        self.restrict_future_leads = restrict_future_leads
        self.velocity_tiers = velocity_tiers
        self.model: "CatBoostQuantileModel | None" = None

    def _resolve_velocity_col(self) -> str | None:
        if not self.velocity_tiers:
            return None
        if isinstance(self.velocity_tiers, str):
            val = self.velocity_tiers.strip().lower()
            if val in ("demand_regime", "sbc", "syntetos_boylan"):
                return "demand_regime"
            if val in ("demand_regime_name",):
                return "demand_regime_name"
            if val in ("velocity_tier", "velocity", "true"):
                return "velocity_tier"
            if val in ("false", "none", ""):
                return None
            raise ValueError(
                f"Unsupported velocity_tiers='{self.velocity_tiers}'. "
                "Use False, True, 'velocity_tier', 'demand_regime', or 'demand_regime_name'."
            )
        return "velocity_tier"

    def fit(self, ts: "DiscreteTimeSeries") -> "CatBoostQuantileModel":
        """
        Trains one `CatBoostRegressor` per `(group, horizon)` pair and returns the
        fitted `CatBoostQuantileModel` container.

        How the temporal validation split works (lines 869-872)
        -------------------------------------------------------
        For a row at period position `group_pos` and a horizon `target_t_plus_h` with offset
        `h = ts._target_offset(target)`:
        - The target realization occurs at period position `target_pos = group_pos + h`.
        - To prevent any look-ahead into the test set, `has_target` requires `target_pos < test_start_pos`.
        - The validation set (`valid_mask`) consists of rows whose realization date `target_pos`
          falls in the last `num_valid_days` before `test_start_period`
          (`test_start_pos - num_valid_days <= target_pos < test_start_pos`).
        - The training set (`fit_mask`) consists of all earlier rows (`target_pos < test_start_pos - num_valid_days`),
          optionally bounded by `max_train_days` so training on multi-year panels stays fast
          and focused on recent regime dynamics.
        """
        from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel, train_catboost_quantile

        t0 = time.perf_counter()
        self._bind_training_context(ts)
        partition_cols = list(self.train_groups)
        vel_col = self._resolve_velocity_col()
        if vel_col is not None and vel_col not in partition_cols and vel_col in ts.df.columns:
            partition_cols.append(vel_col)
        # Exclude redundant demand_regime_name from cat features if already partitioned by demand_regime:
        redundant_cats = {"demand_regime_name"} if "demand_regime" in partition_cols else set()
        available_extra_cats = [
            c
            for c in self.extra_cat_cols
            if c in ts.df.columns and c not in partition_cols and c not in redundant_cats
        ]
        cat_cols = list(dict.fromkeys(list(ts.categorical_feature_cols) + available_extra_cats))
        feature_cols = [c for c in ts.feature_cols if c not in cat_cols]
        self.feature_cols_ = list(feature_cols)
        self.cat_feature_cols_ = list(cat_cols)

        horizon_feature_cols = {}
        horizon_cat_feature_cols = {}
        for target in ts.target_cols:
            allowed_num = set(ts.get_features_for_horizon(target, restrict_future_leads=self.restrict_future_leads))
            horizon_feature_cols[target] = [c for c in feature_cols if c in allowed_num]
            horizon_cat_feature_cols[target] = ts.get_cat_features_for_horizon(
                target,
                restrict_future_leads=self.restrict_future_leads,
                extra_cat_cols=available_extra_cats,
            )

        result = CatBoostQuantileModel(
            self.quantiles,
            list(ts.target_cols),
            feature_cols,
            cat_cols,
            partition_cols,
            horizon_feature_cols=horizon_feature_cols,
            horizon_cat_feature_cols=horizon_cat_feature_cols,
        )

        test_start_pos = (
            len(ts.periods) if ts.test_start_period is None else ts.periods.index(ts.test_start_period)
        )
        period_pos = ts._period_positions(ts.df)
        mask = ts.df["is_test"] == 0
        if self.max_train_days is not None:
            mask &= period_pos >= test_start_pos - self.num_valid_days - self.max_train_days
        train_df = ts.df[mask]
        train_pos = period_pos[mask]

        groups = list(train_df.groupby(partition_cols, observed=True)) if partition_cols else [((), train_df)]
        logger.info(
            "Starting CatBoost MultiQuantile training (%d quantiles, %d group(s), %d horizon(s), %d num features, %d cat features)...",
            len(self.quantiles),
            len(groups),
            len(ts.target_cols),
            len(feature_cols),
            len(cat_cols),
        )

        n_horizons = len(ts.target_cols)
        for group_values, group_df in groups:
            key = CatBoostQuantileModel.group_key(group_values) if partition_cols else ()
            group_pos = train_pos.loc[group_df.index]
            for h_idx, target in enumerate(ts.target_cols, start=1):
                h_t0 = time.perf_counter()
                target_pos = group_pos + ts._target_offset(target)
                has_target = group_df[target].notna() & (target_pos < test_start_pos)
                valid_mask = has_target & (target_pos >= test_start_pos - self.num_valid_days)
                fit_mask = has_target & ~valid_mask
                h_num_cols, h_cat_cols = result.features_for_target(target)
                logger.info(
                    "Training CatBoost model for group %s, horizon [%d/%d] (%s): %d train rows, %d validation rows...",
                    key or "(all)",
                    h_idx,
                    n_horizons,
                    target,
                    int(fit_mask.sum()),
                    int(valid_mask.sum()),
                )
                cb_model = train_catboost_quantile(
                    group_df[fit_mask],
                    h_num_cols,
                    h_cat_cols,
                    target,
                    self.quantiles,
                    eval_df=group_df[valid_mask] if self.num_valid_days > 0 else None,
                    params=self.catboost_params,
                )
                result.models.setdefault(key, {})[target] = cb_model
                logger.info(
                    "Completed CatBoost model for group %s, %s in %.2fs (best iteration: %s)",
                    key or "(all)",
                    target,
                    time.perf_counter() - h_t0,
                    cb_model.get_best_iteration(),
                )
        self.model = result
        logger.info("Finished all CatBoost horizon models in %.2fs", time.perf_counter() - t0)
        return result

    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: "CatBoostQuantileModel | None" = None,
        period: Any = None,
    ) -> pd.DataFrame:
        """
        Generates quantile forecast columns (`<target>_q_<level>`) and implied discrete
        mean columns (`<target>_mu`) across all horizons, plus cumulative error metrics.
        """
        from probabilistic_modeling.models.catboost_quantile import predict_catboost_quantile

        t0 = time.perf_counter()
        cb_container = model if model is not None else self.model
        if cb_container is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")

        test_data, target_cols, _, _ = self._resolve_inference_inputs(
            ts=ts,
            df=df,
            period=period,
            fallback_target_cols=cb_container.target_cols,
            fallback_feature_cols=cb_container.feature_cols,
        )
        levels = np.asarray(cb_container.quantiles, dtype=np.float64)
        for tier_col in ("velocity_tier", "demand_regime"):
            if tier_col in cb_container.train_groups and tier_col not in test_data.columns:
                test_data[tier_col] = 2 if tier_col == "demand_regime" else 0
        if (
            "demand_regime_name" in cb_container.train_groups or "demand_regime_name" in cb_container.cat_feature_cols
        ) and "demand_regime_name" not in test_data.columns:
            from probabilistic_modeling.time_series import DEMAND_REGIME_NAMES

            if "demand_regime" in test_data.columns:
                test_data["demand_regime_name"] = (
                    test_data["demand_regime"].map(DEMAND_REGIME_NAMES).fillna("intermittent").astype("category")
                )
            else:
                test_data["demand_regime_name"] = pd.Series(
                    ["intermittent"] * len(test_data), index=test_data.index, dtype="category"
                )
        logger.info(
            "Generating CatBoost quantile predictions for %d rows across %d horizons (%d quantiles)...",
            len(test_data),
            len(cb_container.target_cols),
            len(levels),
        )
        preds = {target: np.full((len(test_data), len(levels)), np.nan) for target in cb_container.target_cols}

        if cb_container.train_groups:
            groups = test_data.groupby(cb_container.train_groups, observed=True).indices
        else:
            groups = {(): np.arange(len(test_data))}

        for group_values, idx in groups.items():
            key = cb_container.group_key(group_values) if cb_container.train_groups else ()
            group_models = cb_container.models.get(key)
            if group_models is None and cb_container.train_groups:
                # Fallback if an inference demand_regime / velocity_tier was unseen in training:
                tier_indices = [
                    i
                    for i, c in enumerate(cb_container.train_groups)
                    if c in ("velocity_tier", "demand_regime", "demand_regime_name")
                ]
                if tier_indices:
                    t_idx = tier_indices[-1]
                    for cand in ("1", "0", "2", "3", "intermittent", "lumpy", "erratic", "smooth"):
                        cand_key = list(key)
                        cand_key[t_idx] = cand
                        group_models = cb_container.models.get(tuple(cand_key))
                        if group_models is not None:
                            break
            if group_models is None:
                logger.warning("No CatBoost model available for group %s; skipping %d rows", key, len(idx))
                continue
            group_df = test_data.iloc[idx]
            for target, cb_model in group_models.items():
                h_num_cols, h_cat_cols = cb_container.features_for_target(target)
                preds[target][idx] = predict_catboost_quantile(
                    cb_model, group_df, h_num_cols, h_cat_cols
                )

        new_cols = {}
        for target, arr in preds.items():
            for j, col in enumerate(cb_container.quantile_cols(target)):
                new_cols[col] = arr[:, j]
            mu = np.full(len(arr), np.nan)
            valid = ~np.isnan(arr).any(axis=1)
            if np.any(valid):
                mu[valid] = quantiles_to_mean(levels, arr[valid], self.tol)
            new_cols[f"{target}_mu"] = mu
        test_data = pd.concat([test_data, pd.DataFrame(new_cols, index=test_data.index)], axis=1)

        compute_forecasting_metrics(test_data, target_cols, [f"{target}_mu" for target in cb_container.target_cols])
        logger.info("CatBoost predictions and distribution means completed in %.2fs", time.perf_counter() - t0)
        return test_data

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: "CatBoostQuantileModel | None" = None,
        period: Any = None,
        max_service_level: float | None = None,
    ) -> pd.DataFrame:
        """
        Generates CatBoost quantile predictions and propagates them through
        `DDLTCalculator.from_quantiles_group` across all `lead_time_keys` groups.
        """
        if lead_time is None:
            raise ValueError("'lead_time' distribution must be provided.")
        cb_container = model if model is not None else self.model
        if cb_container is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")

        predictions = self.predict(ts, df=df, model=cb_container, period=period)
        _, target_cols, _, period_col = self._resolve_inference_inputs(
            ts=ts,
            df=predictions,
            fallback_target_cols=cb_container.target_cols,
        )
        return _apply_ddlt_across_groups(
            predictions=predictions,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            service_levels=service_levels,
            return_distribution=return_distribution,
            group_fn=lambda gdf, lt_distr: DDLTCalculator(
                lt_distr, cycle_time, tol=self.tol, max_service_level=max_service_level
            ).from_quantiles_group(
                gdf,
                lt_distr,
                target_cols,
                cb_container.quantiles,
                return_distribution=return_distribution,
                pred_first_day=pred_first_day,
                period_col=period_col,
            ),
        )

    def save(self, out_dir: Path | str, prefix: str = "catboost") -> Path:
        """Persists the fitted `CatBoostQuantileForecaster` bundle (`forecaster_meta.json`, `feature_spec.json`, `series_metadata.csv`, `catboost_meta.json`, and `.cbm` model files) to `out_dir`."""
        if self.model is None:
            raise ValueError("CatBoostQuantileForecaster has not been fitted yet.")
        out_path = Path(out_dir)
        self._save_base_context(
            out_path,
            extra_meta={
                "quantiles": list(self.quantiles),
                "train_groups": self.train_groups,
                "extra_cat_cols": list(self.extra_cat_cols),
                "num_valid_days": self.num_valid_days,
                "max_train_days": self.max_train_days,
                "catboost_params": self.catboost_params,
                "tol": self.tol,
                "restrict_future_leads": self.restrict_future_leads,
                "velocity_tiers": self.velocity_tiers,
                "prefix": prefix,
            },
        )
        self.model.save(out_path, prefix=prefix)
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str, prefix: str = "catboost") -> "CatBoostQuantileForecaster":
        """Restores a fitted `CatBoostQuantileForecaster` from `out_dir`."""
        from probabilistic_modeling.models.catboost_quantile import CatBoostQuantileModel

        out_path = Path(out_dir)
        meta_file = out_path / "forecaster_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        saved_prefix = meta.get("prefix", prefix)
        cb_model = CatBoostQuantileModel.load(out_path, prefix=saved_prefix)
        inferred_vel: bool | str = False
        if "demand_regime" in cb_model.train_groups:
            inferred_vel = "demand_regime"
        elif "demand_regime_name" in cb_model.train_groups:
            inferred_vel = "demand_regime_name"
        elif "velocity_tier" in cb_model.train_groups:
            inferred_vel = True
        forecaster = cls(
            quantiles=cb_model.quantiles,
            train_groups=meta.get(
                "train_groups",
                [
                    c
                    for c in cb_model.train_groups
                    if c not in ("velocity_tier", "demand_regime", "demand_regime_name")
                ],
            ),
            extra_cat_cols=tuple(meta.get("extra_cat_cols", ("item_id", "dept_id", "cat_id", "demand_regime_name"))),
            num_valid_days=int(meta.get("num_valid_days", 28)),
            max_train_days=meta.get("max_train_days"),
            catboost_params=meta.get("catboost_params"),
            tol=float(meta.get("tol", 1e-4)),
            restrict_future_leads=bool(meta.get("restrict_future_leads", False)),
            velocity_tiers=meta.get("velocity_tiers", inferred_vel),
        )
        forecaster.model = cb_model
        if meta:
            forecaster._load_base_context(out_path, meta)
        else:
            forecaster.target_cols_ = list(cb_model.target_cols)
            forecaster.feature_cols_ = list(cb_model.feature_cols)
            forecaster.cat_feature_cols_ = list(cb_model.cat_feature_cols)
        return forecaster


def load_forecaster(out_dir: Path | str) -> ProbabilisticForecaster:
    """
    Universal loader that inspects `out_dir` and reconstructs the appropriate
    fitted `ProbabilisticForecaster` (`GLMForecaster`, `CatBoostQuantileForecaster`, or
    `EmpiricalQuantileForecaster`), complete with its feature specification and series metadata.

    What it does
    ------------
    1. Checks `<out_dir>/forecaster_meta.json` for `"forecaster_type"` (`"empirical"`,
       `"glm"`, or `"catboost"`) and dispatches to the corresponding `.load(out_path)`.
    2. Falls back to checking `<out_dir>/catboost_meta.json` or `<out_dir>/glm_meta.json`
       so directories saved directly via `CatBoostQuantileModel.save` or `GLMModel.save`
       are also loaded seamlessly.

    Where it is used
    ----------------
    Exported at the top-level package (`probabilistic_modeling.load_forecaster`) for
    reloading any saved model artifact directory without hardcoding the model class.
    """
    out_path = Path(out_dir)
    meta_file = out_path / "forecaster_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        f_type = meta.get("forecaster_type")
        if f_type == "empirical":
            from probabilistic_modeling.models.baseline import EmpiricalQuantileForecaster

            return EmpiricalQuantileForecaster.load(out_path)
        if f_type == "regime_router":
            from probabilistic_modeling.models.router import RegimeRouterForecaster

            return RegimeRouterForecaster.load(out_path)
        if f_type == "glm":
            return GLMForecaster.load(out_path)
        if f_type == "catboost":
            return CatBoostQuantileForecaster.load(out_path)
        raise ValueError(f"Unknown forecaster_type '{f_type}' in {meta_file}.")
    if (out_path / "catboost_meta.json").exists():
        return CatBoostQuantileForecaster.load(out_path)
    if (out_path / "glm_meta.json").exists():
        return GLMForecaster.load(out_path)
    raise FileNotFoundError(f"No recognized forecaster bundle found in {out_path}.")


def _apply_ddlt_across_groups(
    predictions: pd.DataFrame,
    lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution,
    lead_time_keys: list[str] | None,
    service_levels: pd.DataFrame | None,
    return_distribution: bool,
    group_fn,
) -> pd.DataFrame:
    """
    Internal helper that attaches optional per-item/group `service_levels` and applies
    `group_fn(gdf, lt_distr)` either to the entire `predictions` DataFrame (when `lead_time`
    is a single `DiscreteDistribution`) or per `lead_time_keys` group (when `lead_time`
    is a dictionary mapping group keys to `DiscreteDistribution` objects).

    Where it is used
    ----------------
    Shared by `GLMForecaster.compute_ddlt`, `CatBoostQuantileForecaster.compute_ddlt`,
    and `EmpiricalQuantileForecaster.compute_ddlt`.
    """
    if not return_distribution:
        if service_levels is not None:
            join_cols = [col for col in service_levels.columns if col != "service_level"]
            predictions = pd.merge(predictions, service_levels, on=join_cols, how="left").fillna(
                {"service_level": 0.95}
            )
        elif "service_level" not in predictions.columns:
            predictions["service_level"] = 0.95

    if isinstance(lead_time, DiscreteDistribution) or not lead_time_keys:
        lt_distr = lead_time if isinstance(lead_time, DiscreteDistribution) else next(iter(lead_time.values()))
        return group_fn(predictions, lt_distr)

    cols = predictions.columns.to_list()
    return predictions.groupby(lead_time_keys, as_index=False, observed=True)[cols].apply(
        lambda x: group_fn(x, lead_time[x.name])
    )
