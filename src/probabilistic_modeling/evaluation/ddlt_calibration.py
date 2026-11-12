"""
Calibration report for demand-during-lead-time (DDLT) distributions.

What this module does
---------------------
Evaluates whether predicted DDLT distributions ``F`` are probabilistically calibrated against
the actual cumulative demand ``Y_L = sum_{j=0}^{L+R-1} demand[t + j]`` realized over the
replenishment protection window.

Why this module is needed
-------------------------
1. **Daily calibration != DDLT calibration**: Even if each single-day forecast ``D_{t+h}`` is
   well calibrated marginally, convolving them under independence ignores positive day-to-day
   demand autocorrelation (e.g. momentum, multi-day promotions, stockout-induced shifts),
   which makes the multi-day sum distribution ``F`` **too narrow** (under-dispersed) and causes
   systematic under-stocking at high service levels (90%–99%).
2. **Discrete PIT (Czado, Gneiting & Held, 2009)**: For discrete count data, the raw coverage
   ``P(Y <= q_tau)`` is strictly ``>= tau`` even for a perfect model because ``F`` jumps by
   ``P(Y = q_tau) > 0`` at the integer ``q_tau``. The non-randomized Probability Integral
   Transform (PIT) distributes the probability mass of observation ``y`` uniformly across
   the jump interval ``[F(y - 1), F(y)]``; its expected CDF at ``tau`` (`pit_coverage`) equals
   ``tau`` **exactly** under a calibrated discrete model.
3. **Random lead times in backtesting**: Since a backtest row does not have a single realized
   physical shipment lead time, every calibration metric (`pit_coverage`, `coverage`, PIT
   histogram bins, `crps`, `sq_err`) is computed as an exact expectation over the lead-time
   distribution ``sum_l P(L = l) * stat(F, Y_l)``.

Where this module is used
-------------------------
- Re-exported by ``probabilistic_modeling`` and ``probabilistic_modeling.evaluation``
- Called by ``scripts/run_m5_pipeline.py`` when ``--evaluate-ddlt`` is passed.
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
    """
    Evaluate the conditional CDF at ``u`` in ``[0, 1]`` of the non-randomized discrete PIT
    (which is Uniform on ``[f_lo, f_hi] = [F(y - 1), F(y)]``) given observation ``y``:

        G(u | y) = 0                        if u <= F(y - 1)
                 = (u - F(y-1)) / P(Y = y)  if F(y - 1) < u < F(y)
                 = 1                        if u >= F(y)

    Why it is needed
    ----------------
    By Czado, Gneiting & Held (2009), if ``Y ~ F`` for any discrete distribution ``F``, then
    ``E_Y[G(u | Y)] = u`` for all ``u in [0, 1]``. This gives an exact calibration check free
    from discrete step-function conservatism and free from Monte Carlo randomization noise.

    Where it is used
    ----------------
    Called inside :func:`_ddlt_calibration_kernel` to compute both ``pit_coverage`` at each
    target quantile level and the mass falling into each decile bin of the PIT histogram.
    """
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
    Parallel Numba kernel computing lead-time-weighted DDLT calibration statistics for ``n`` rows.

    What it does
    ------------
    For each decision row ``i = 0, ..., n - 1`` (in parallel via ``prange``):
    1. Slices row ``i``'s DDLT PMF ``p = pmf_flat[offsets[i] : offsets[i + 1]]`` starting at
       ``mn = mins[i]``, and computes its predictive mean ``mu``, variance ``v``, and target
       quantiles ``q_out[i, a] = F_i^{-1}(levels[a])``.
    2. Loops over each support point ``j`` of the lead-time distribution with weight ``wj = w[i, j]``
       and realized ``(L_j + R)``-day demand ``yy = y[i, j]``:
       - Accumulates expected realized demand ``ym += wj * yy``, expected CRPS, and expected
         squared error ``se += wj * (yy - mu)^2``.
       - Computes ``f_lo = F_i(yy - 1)`` and ``f_hi = F_i(yy)``.
       - Accumulates raw coverage ``1{yy <= q_out[i, a]}`` and discrete-correct PIT coverage
         ``_pit_cdf(levels[a], f_lo, f_hi)``.
       - Accumulates the PIT mass ``G((b+1)/n_bins) - G(b/n_bins)`` in each histogram bin ``b``.

    Where it is used
    ----------------
    Called by :func:`evaluate_ddlt_calibration`.
    """
    n = len(mins)
    n_lv = len(levels)
    mean = np.empty(n)
    var = np.empty(n)
    y_mean = np.empty(n)
    crps = np.empty(n)
    sq_err = np.empty(n)
    u_mean = np.empty(n)
    u_sq = np.empty(n)
    coverage = np.zeros((n, n_lv))
    pit_cov = np.zeros((n, n_lv))
    under_diff = np.zeros((n, n_lv))
    over_diff = np.zeros((n, n_lv))
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
        um = 0.0
        usq = 0.0
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
            um += wj * 0.5 * (f_lo + f_hi)
            usq += wj * (f_lo * f_lo + f_lo * f_hi + f_hi * f_hi) / 3.0
            for a in range(n_lv):
                qa = q_out[i, a]
                if yy <= qa:
                    coverage[i, a] += wj
                    over_diff[i, a] += wj * (qa - yy)
                else:
                    under_diff[i, a] += wj * (yy - qa)
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
        u_mean[i] = um
        u_sq[i] = usq
    return mean, var, y_mean, crps, sq_err, u_mean, u_sq, coverage, pit_cov, under_diff, over_diff, q_out, pit


@dataclass
class DDLTCalibrationResult:
    """
    Container for the outputs of :func:`evaluate_ddlt_calibration`.

    Attributes
    ----------
    row_df : pd.DataFrame
        Per-(series, decision period) statistics (`pred_mean`, `pred_std`, `realized_mean`,
        `crps`, `sq_err`, `pred_var`, `cov_<lv>`, `pitcov_<lv>`, `q_<lv>`, `pit_0..pit_{B-1}`).
    summary_df : pd.DataFrame
        Overall and per-group summary metrics (`n_rows`, `mean_pred`, `mean_realized`,
        `bias_pct`, `crps`, `dispersion_ratio`, `pit_tail_upper`, `pit_tail_lower`).
    coverage_df : pd.DataFrame
        Overall and per-group quantile calibration table (`level`, `coverage`, `pit_coverage`, `gap_pp`).
    pit_df : pd.DataFrame
        Overall and per-group non-randomized PIT histogram (`bin`, `lower`, `upper`, `mass`, `density`).
    levels : tuple[float, ...]
        Quantile levels evaluated.
    n_dropped : int
        Number of decision rows dropped because ``t + max(L) + R`` extended past the end of ``history``.
    """

    row_df: pd.DataFrame
    summary_df: pd.DataFrame
    coverage_df: pd.DataFrame
    pit_df: pd.DataFrame
    levels: tuple[float, ...] = field(default=DEFAULT_CALIBRATION_LEVELS)
    n_dropped: int = 0

    def report(self) -> str:
        """Format a compact human-readable text report of the overall calibration tables."""
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
    """
    Build zero-padded 2D arrays ``(m_mat, w_mat)`` of shape ``(len(rows), max_lt_support)``
    where ``m_mat[i, j] = l_j + cycle_time`` is the protection window length (in periods) and
    ``w_mat[i, j] = P(L = l_j)`` is its probability weight for row ``i``.

    Where it is used
    ----------------
    Called by :func:`evaluate_ddlt_calibration`.
    """

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
    Evaluate predicted DDLT distributions against realized demand over the replenishment window.

    What it does
    ------------
    1. Builds a :class:`DenseHistory` prefix-sum matrix from ``history`` and locates each row
       of ``ddlt_df``.
    2. Filters out decision rows near the end of the test set whose worst-case protection window
       ``t + max(L) + R`` extends past the last observed period in ``history`` (recorded in
       ``result.n_dropped``).
    3. Computes the realized cumulative demand ``y[i, j] = sum_{t}^{t + L_j + R - 1} demand``
       in ``O(1)`` per entry via ``hist.prefix[r, c + m] - hist.prefix[r, c]``.
    4. Packs all DDLT PMFs into a flat contiguous array and runs :func:`_ddlt_calibration_kernel`.
    5. Aggregates ``summary_df``, ``coverage_df``, and ``pit_df`` overall and by ``group_cols``.

    Parameters
    ----------
    ddlt_df : pd.DataFrame
        One row per ``(series, decision period)`` with a :class:`DiscreteDistribution` in
        ``dist_col`` (e.g. output of ``compute_lead_time_demand(..., return_distribution=True,
        pred_first_day=False)``).
    history : pd.DataFrame
        Long demand panel ``(ts_id_cols, period_col, var_col)`` covering the decision periods
        *and* the following ``max(L) + R`` periods.
    lead_time, lead_time_keys, cycle_time :
        The same lead-time specification used to build the DDLT distributions.
    group_cols : list[str], optional
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

    (
        mean,
        var,
        y_mean,
        crps,
        sq_err,
        u_mean,
        u_sq,
        coverage,
        pit_cov,
        under_diff,
        over_diff,
        q_out,
        pit,
    ) = _ddlt_calibration_kernel(
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
    row_df["pit_u_mean"] = u_mean
    row_df["pit_u_sq"] = u_sq
    for a, lv in enumerate(levels_arr):
        row_df[f"cov_{lv:g}"] = coverage[:, a]
        row_df[f"pitcov_{lv:g}"] = pit_cov[:, a]
        row_df[f"under_{lv:g}"] = under_diff[:, a]
        row_df[f"over_{lv:g}"] = over_diff[:, a]
        row_df[f"q_{lv:g}"] = q_out[:, a]
    pit_cols = [f"pit_{b}" for b in range(n_pit_bins)]
    row_df[pit_cols] = pit

    def summarize(sub: pd.DataFrame, label: str, keys: dict) -> tuple[dict, list[dict], list[dict]]:
        n_sub = int(len(sub))
        mp, mr = float(sub["pred_mean"].mean()), float(sub["realized_mean"].mean())
        pit_mass = sub[pit_cols].mean().to_numpy()
        density = pit_mass * n_pit_bins
        pit_mae = float(np.mean(np.abs(density - 1.0)))
        pit_rmse = float(np.sqrt(np.mean((density - 1.0) ** 2)))
        cdf_emp = np.cumsum(pit_mass)
        cdf_uni = np.arange(1, n_pit_bins + 1, dtype=np.float64) / n_pit_bins
        pit_ks = float(np.max(np.abs(cdf_emp - cdf_uni)))
        um_val = float(sub["pit_u_mean"].mean())
        usq_val = float(sub["pit_u_sq"].mean())
        pit_std = float(np.sqrt(max(usq_val - um_val * um_val, 0.0)))
        summ = {
            "group": label, **keys,
            "n_rows": n_sub,
            "mean_pred": mp,
            "mean_realized": mr,
            "bias_pct": 100.0 * (mp - mr) / mr if mr > 0 else np.nan,
            "crps": float(sub["crps"].mean()),
            "dispersion_ratio": float(sub["sq_err"].sum() / sub["pred_var"].sum()) if sub["pred_var"].sum() > 0 else np.nan,
            "pit_mean": um_val,
            "pit_std": pit_std,
            "pit_mae": pit_mae,
            "pit_rmse": pit_rmse,
            "pit_ks": pit_ks,
            # mass in the top / bottom PIT bin relative to the calibrated 1 / n_bins
            "pit_tail_upper": float(density[-1]),
            "pit_tail_lower": float(density[0]),
        }
        actual_sum = float(sub["realized_mean"].sum())
        cov_rows = []
        for lv in levels_arr:
            tau = float(lv)
            cv = float(sub[f"cov_{lv:g}"].mean())
            pcv = float(sub[f"pitcov_{lv:g}"].mean())
            pred_sum = float(sub[f"q_{lv:g}"].sum())
            u_sum = float(sub[f"under_{lv:g}"].sum())
            o_sum = float(sub[f"over_{lv:g}"].sum())
            a_sum = u_sum + o_sum
            pinball_sum = tau * u_sum + (1.0 - tau) * o_sum
            cov_rows.append({
                "group": label, **keys, "level": tau,
                "coverage": cv, "pit_coverage": pcv,
                "gap_pp": 100.0 * (pcv - tau), "n_rows": n_sub,
                "actual": actual_sum,
                "pred": pred_sum,
                "under_diff": u_sum,
                "over_diff": o_sum,
                "abs_diff": a_sum,
                "wmape": 100.0 * a_sum / actual_sum if actual_sum > 0 else np.nan,
                "under_wmape": 100.0 * u_sum / actual_sum if actual_sum > 0 else np.nan,
                "over_wmape": 100.0 * o_sum / actual_sum if actual_sum > 0 else np.nan,
                "pinball_loss": pinball_sum / n_sub if n_sub > 0 else np.nan,
                "pinball_wmape": 100.0 * 2.0 * pinball_sum / actual_sum if actual_sum > 0 else np.nan,
            })
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
