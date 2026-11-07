"""
Calibration report for demand-during-lead-time (DDLT) distributions.

Daily forecast metrics (pinball loss, coverage) do not tell you whether the *decision*
distribution is right. Inventory targets are quantiles of the DDLT distribution, so the
question that matters is:

    Does the realised demand over the replenishment window fall below the τ-quantile of the
    DDLT distribution τ of the time?

For a decision at period ``t`` with review period ``R`` and lead time ``L``, the realised
quantity is ``Y_L = sum(demand[t : t + L + R])``. In a backtest the lead time of a
hypothetical order is unknown, so every statistic is computed as an exact expectation over
the lead-time distribution, ``E_L[stat(F, Y_L)] = sum_l P(L = l) * stat(F, Y_l)``. Under a
correct model ``Y_L ~ F`` exactly, so:

- ``pit_coverage`` at level τ          → τ   (``gap_pp`` = 100 * (pit_coverage - τ))
- non-randomised PIT histogram       → flat (Czado, Gneiting & Held, 2009)
- dispersion ratio E[(Y-μ)²] / E[σ²] → 1   (> 1: DDLT too narrow, e.g. ignored
  day-to-day correlation; < 1: too wide)
- CRPS                               → lower is better (compare models)

``coverage`` is the raw ``P(Y <= q_τ)``, i.e. the cycle service level actually achieved by
ordering up to the τ-quantile. For *discrete* distributions it is ``>= τ`` even for a perfect
model (the atom at ``q_τ`` counts as covered), which is why calibration is judged on
``pit_coverage`` = E[P(PIT <= τ)], the discrete-correct analogue.

Note: decision periods overlap in time, so effective sample sizes are smaller than the row
counts; treat small coverage gaps (< ~1pp) with caution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

from probabilistic_modeling._panel import DenseHistory
from probabilistic_modeling.ddlt import _pmf_crps_scalar_numba, _pmf_quantile_scalar_numba
from probabilistic_modeling.distributions.discrete import DiscreteDistribution

DEFAULT_CALIBRATION_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)


@njit(cache=True)
def _pit_cdf(u: float, f_lo: float, f_hi: float) -> float:
    """CDF at ``u`` of the non-randomised PIT, uniform on ``[F(y-1), F(y)]``."""
    if u <= f_lo:
        return 0.0
    if u >= f_hi:
        return 1.0
    return (u - f_lo) / (f_hi - f_lo)


@njit(parallel=True, cache=True)
def _ddlt_calibration_kernel(
    pmf_flat: np.ndarray,
    offsets: np.ndarray,
    mins: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    levels: np.ndarray,
    n_bins: int,
):
    """
    Per-row lead-time-weighted calibration statistics.

    Row ``i``'s DDLT PMF is ``pmf_flat[offsets[i]:offsets[i+1]]`` starting at ``mins[i]``.
    ``y[i, j]`` is the realised demand if the lead time equals the ``j``-th support point of row
    ``i``'s lead-time distribution, with probability ``w[i, j]`` (zero-padded).
    """
    n = len(mins)
    n_lv = len(levels)
    mean = np.empty(n)
    var = np.empty(n)
    y_mean = np.empty(n)
    crps = np.empty(n)
    sq_err = np.empty(n)
    coverage = np.zeros((n, n_lv))
    pit_cov = np.zeros((n, n_lv))
    q_out = np.empty((n, n_lv))
    pit = np.zeros((n, n_bins))
    for i in prange(n):
        p = pmf_flat[offsets[i]:offsets[i + 1]]
        mn = mins[i]
        m = len(p)
        mu = 0.0
        for k in range(m):
            mu += (mn + k) * p[k]
        v = 0.0
        for k in range(m):
            d = (mn + k) - mu
            v += d * d * p[k]
        mean[i] = mu
        var[i] = v
        for a in range(n_lv):
            q_out[i, a] = _pmf_quantile_scalar_numba(p, mn, 1, levels[a])

        ym = 0.0
        cr = 0.0
        se = 0.0
        for j in range(y.shape[1]):
            wj = w[i, j]
            if wj <= 0.0:
                continue
            yy = y[i, j]
            ym += wj * yy
            cr += wj * _pmf_crps_scalar_numba(p, mn, 1, yy)
            se += wj * (yy - mu) * (yy - mu)
            # F(y - 1) and F(y) for the non-randomised PIT
            f_lo = 0.0
            f_hi = 0.0
            iy = int(yy) - mn
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
                if yy <= q_out[i, a]:
                    coverage[i, a] += wj
                pit_cov[i, a] += wj * _pit_cdf(levels[a], f_lo, f_hi)
            prev = 0.0
            for b in range(n_bins):
                if b == n_bins - 1:
                    cdf_u = 1.0  # PIT <= 1 always (also covers y beyond the support: F(y-1) = 1)
                else:
                    cdf_u = _pit_cdf((b + 1.0) / n_bins, f_lo, f_hi)
                pit[i, b] += wj * (cdf_u - prev)
                prev = cdf_u
        y_mean[i] = ym
        crps[i] = cr
        sq_err[i] = se
    return mean, var, y_mean, crps, sq_err, coverage, pit_cov, q_out, pit


@dataclass
class DDLTCalibrationResult:
    """Outputs of :func:`evaluate_ddlt_calibration`."""

    row_df: pd.DataFrame
    summary_df: pd.DataFrame
    coverage_df: pd.DataFrame
    pit_df: pd.DataFrame
    levels: tuple[float, ...] = field(default=DEFAULT_CALIBRATION_LEVELS)
    n_dropped: int = 0

    def report(self) -> str:
        """Compact human-readable report (overall rows only)."""
        overall = self.summary_df[self.summary_df["group"] == "overall"]
        cov = self.coverage_df[self.coverage_df["group"] == "overall"]
        pit = self.pit_df[self.pit_df["group"] == "overall"]
        lines = ["DDLT calibration", "================"]
        cols = ["n_rows", "mean_pred", "mean_realized", "bias_pct", "crps", "dispersion_ratio", "pit_tail_upper"]
        lines.append(overall[[c for c in cols if c in overall.columns]].to_string(index=False))
        lines.append("")
        lines.append(cov[["level", "coverage", "pit_coverage", "gap_pp"]].to_string(index=False))
        lines.append("")
        lines.append("PIT density by decile (1.0 = calibrated): "
                     + " ".join(f"{v:.2f}" for v in pit["density"].to_numpy()))
        if self.n_dropped:
            lines.append(f"({self.n_dropped} rows dropped: realised window extends beyond the observed history)")
        return "\n".join(lines)


def _lead_time_terms(
    rows: pd.DataFrame,
    lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution],
    lead_time_keys: list[str] | None,
    cycle_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(m_values, weights)`` matrices with ``m = L + R`` (zero-padded)."""

    def terms(lt: DiscreteDistribution) -> tuple[np.ndarray, np.ndarray]:
        lt1 = lt.split_bins(inplace=False) if lt.bin_size != 1 else lt
        m = lt1.min + cycle_time + np.arange(len(lt1.pmf))
        keep = lt1.pmf > 0
        return m[keep].astype(np.int64), lt1.pmf[keep].astype(np.float64)

    if isinstance(lead_time, DiscreteDistribution) or not lead_time_keys:
        lt = lead_time if isinstance(lead_time, DiscreteDistribution) else next(iter(lead_time.values()))
        m, wts = terms(lt)
        return np.tile(m, (len(rows), 1)), np.tile(wts, (len(rows), 1))

    per_key = {k: terms(v) for k, v in lead_time.items()}
    width = max(len(m) for m, _ in per_key.values())
    m_mat = np.zeros((len(rows), width), dtype=np.int64)
    w_mat = np.zeros((len(rows), width), dtype=np.float64)
    if len(lead_time_keys) == 1:
        key_vals = rows[lead_time_keys[0]].to_numpy()
    else:
        key_vals = list(rows[lead_time_keys].itertuples(index=False, name=None))
    for i, k in enumerate(key_vals):
        hit = per_key.get(k, per_key.get(str(k)))
        if hit is None:
            raise KeyError(f"No lead-time distribution for key {k!r}.")
        m, wts = hit
        m_mat[i, : len(m)] = m
        w_mat[i, : len(m)] = wts
    return m_mat, w_mat


def evaluate_ddlt_calibration(
    ddlt_df: pd.DataFrame,
    history: pd.DataFrame,
    lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution],
    ts_id_cols: list[str],
    lead_time_keys: list[str] | None = None,
    cycle_time: int = 1,
    period_col: str = "day",
    var_col: str = "demand",
    dist_col: str = "demand_distribution",
    levels: tuple[float, ...] = DEFAULT_CALIBRATION_LEVELS,
    group_cols: list[str] | None = None,
    n_pit_bins: int = 10,
) -> DDLTCalibrationResult:
    """
    Evaluates DDLT distributions against realised demand over the replenishment window.

    Parameters
    ----------
    ddlt_df :
        One row per ``(series, decision period)`` with a :class:`DiscreteDistribution` in
        ``dist_col`` (e.g. the output of ``compute_lead_time_demand(..., return_distribution=True,
        pred_first_day=False)``).
    history :
        Long demand panel ``(ts_id_cols, period_col, var_col)`` covering the decision periods
        *and* the following ``max(L) + R`` periods.
    lead_time, lead_time_keys, cycle_time :
        The same lead-time specification used to build the DDLT distributions.
    group_cols :
        Optional breakdown columns present in ``ddlt_df`` (e.g. ``["cat_id"]``).
    """
    levels_arr = np.asarray(sorted(levels), dtype=np.float64)
    hist = DenseHistory.from_frame(history, ts_id_cols, period_col, var_col)
    r, c = hist.locate(ddlt_df)
    m_mat, w_mat = _lead_time_terms(ddlt_df, lead_time, lead_time_keys, cycle_time)

    n_periods = hist.matrix.shape[1]
    max_m = np.where(w_mat > 0, m_mat, 0).max(axis=1)
    ok = (r >= 0) & (c >= 0) & (c + max_m <= n_periods)
    n_dropped = int((~ok).sum())
    rows = ddlt_df.loc[ok].reset_index(drop=True)
    r, c, m_mat, w_mat = r[ok], c[ok], m_mat[ok], w_mat[ok]
    if len(rows) == 0:
        raise ValueError("No decision rows have a fully observed replenishment window in 'history'.")

    end = np.clip(c[:, None] + m_mat, 0, n_periods)
    y = hist.prefix[r[:, None], end] - hist.prefix[r, c][:, None]
    y = np.where(w_mat > 0, y, 0.0)
    w_mat = w_mat / w_mat.sum(axis=1, keepdims=True)

    pmf_list = []
    mins = np.empty(len(rows), dtype=np.int64)
    for i, d in enumerate(rows[dist_col]):
        if d.bin_size != 1:
            d = d.split_bins(inplace=False)
        pmf_list.append(d.pmf)
        mins[i] = d.min
    offsets = np.zeros(len(pmf_list) + 1, dtype=np.int64)
    np.cumsum([len(p) for p in pmf_list], out=offsets[1:])
    pmf_flat = np.ascontiguousarray(np.concatenate(pmf_list), dtype=np.float64)

    mean, var, y_mean, crps, sq_err, coverage, pit_cov, q_out, pit = _ddlt_calibration_kernel(
        pmf_flat, offsets, mins, np.ascontiguousarray(y), np.ascontiguousarray(w_mat), levels_arr, int(n_pit_bins)
    )

    keep_cols = [col for col in list(ts_id_cols) + [period_col] + list(group_cols or []) if col in rows.columns]
    row_df = rows[list(dict.fromkeys(keep_cols))].copy()
    row_df["pred_mean"] = mean
    row_df["pred_std"] = np.sqrt(var)
    row_df["realized_mean"] = y_mean
    row_df["crps"] = crps
    row_df["sq_err"] = sq_err
    row_df["pred_var"] = var
    for a, lv in enumerate(levels_arr):
        row_df[f"cov_{lv:g}"] = coverage[:, a]
        row_df[f"pitcov_{lv:g}"] = pit_cov[:, a]
        row_df[f"q_{lv:g}"] = q_out[:, a]
    pit_cols = [f"pit_{b}" for b in range(n_pit_bins)]
    row_df[pit_cols] = pit

    def summarize(sub: pd.DataFrame, label: str, keys: dict) -> tuple[dict, list[dict], list[dict]]:
        mp, mr = float(sub["pred_mean"].mean()), float(sub["realized_mean"].mean())
        pit_mass = sub[pit_cols].mean().to_numpy()
        summ = {
            "group": label, **keys,
            "n_rows": int(len(sub)),
            "mean_pred": mp,
            "mean_realized": mr,
            "bias_pct": 100.0 * (mp - mr) / mr if mr > 0 else np.nan,
            "crps": float(sub["crps"].mean()),
            "dispersion_ratio": float(sub["sq_err"].sum() / sub["pred_var"].sum()) if sub["pred_var"].sum() > 0 else np.nan,
            # mass in the top / bottom PIT bin relative to the calibrated 1 / n_bins
            "pit_tail_upper": float(pit_mass[-1] * n_pit_bins),
            "pit_tail_lower": float(pit_mass[0] * n_pit_bins),
        }
        cov_rows = []
        for lv in levels_arr:
            cv = float(sub[f"cov_{lv:g}"].mean())
            pcv = float(sub[f"pitcov_{lv:g}"].mean())
            cov_rows.append({"group": label, **keys, "level": float(lv),
                             "coverage": cv, "pit_coverage": pcv,
                             "gap_pp": 100.0 * (pcv - lv), "n_rows": int(len(sub))})
        pit_rows = [{"group": label, **keys, "bin": b, "lower": b / n_pit_bins, "upper": (b + 1) / n_pit_bins,
                     "mass": float(pit_mass[b]), "density": float(pit_mass[b] * n_pit_bins)}
                    for b in range(n_pit_bins)]
        return summ, cov_rows, pit_rows

    summaries, cov_all, pit_all = [], [], []
    s, cv, pt = summarize(row_df, "overall", {})
    summaries.append(s); cov_all += cv; pit_all += pt
    valid_groups = [g for g in (group_cols or []) if g in row_df.columns]
    if valid_groups:
        for key, sub in row_df.groupby(valid_groups, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            kd = dict(zip(valid_groups, key))
            label = "|".join(f"{k}={v}" for k, v in kd.items())
            s, cv, pt = summarize(sub, label, kd)
            summaries.append(s); cov_all += cv; pit_all += pt

    return DDLTCalibrationResult(
        row_df=row_df,
        summary_df=pd.DataFrame(summaries),
        coverage_df=pd.DataFrame(cov_all),
        pit_df=pd.DataFrame(pit_all),
        levels=tuple(float(x) for x in levels_arr),
        n_dropped=n_dropped,
    )
