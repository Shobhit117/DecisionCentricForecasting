"""
Numba-accelerated multi-period retail & supply-chain inventory replenishment simulation (`inventory/simulation.py`).

What this module does
---------------------
Evaluates the real-world operational and financial performance of probabilistic demand
forecasts and target-inventory policies over a multi-period horizon:
1. **Target Inventory Resolution (`resolve_target_inventory`)**: Converts DDLT distributions
   into daily order-up-to targets (`target_inventory`) using either:
   - `policy="quantile"` (service-level quantile `F^{-1}(alpha)` via Numba `O(1)`-memory scan),
   - `policy="newsvendor"` (budget-constrained multi-product newsvendor via Lagrangian bisection),
   - `policy="precomputed"` / `"auto"` (existing `target_inventory` column).
2. **Order-Up-To (Base-Stock) Trajectory Simulation (`_simulate_inventory_trajectory_numba`)**:
   Tracks daily opening inventory, arriving shipments from in-transit orders, inventory position
   (`on_hand + in_transit - backlog`), periodic replenishment orders every `cycle_time` days,
   stochastic or deterministic shipment lead times, demand fulfillment, lost sales vs. backorders,
   closing inventory, and daily P&L (revenue, purchase spend, holding cost, shortage cost, net profit).
3. **Monte Carlo Lead-Time Simulation (`_simulate_inventory_mc_numba`)**: Simulates `n_sims`
   independent stochastic lead-time trajectories in parallel to quantify confidence intervals
   (`mean`, `std`, `p05`, `p95`) on fill rate, cycle service level, stockout frequency,
   average inventory, and total order volume.
4. **KPI Aggregation (`compute_simulation_kpis`, `InventorySimulationResult`)**: Computes
   overall, per-item, and per-group operational and financial KPIs.

Where it is used
----------------
- Final evaluation stage of `scripts/run_m5_pipeline.py` (Stage 6).
- Exported at top-level package (`InventorySimulator`, `InventorySimulationResult`,
  `simulate_inventory`, `resolve_target_inventory`, `compute_simulation_kpis`).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List as NumbaList
from scipy.stats import norm

from probabilistic_modeling._panel import DenseHistory
from probabilistic_modeling.ddlt import (
    _batch_pmf_quantiles_numba,
    _batch_pmf_quantiles_round_numba,
    _pmf_quantile_scalar_numba,
)
from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.inventory.newsvendor import _solve_multi_product_newsvendor

logger = logging.getLogger(__name__)


@njit(cache=True)
def _sample_lead_times_numba(
    pmfs: NumbaList,
    mins: np.ndarray,
    bin_sizes: np.ndarray,
    item_lt_idx: np.ndarray,
    u_mat: np.ndarray,
) -> np.ndarray:
    """
    Maps a `(n_items, n_days)` matrix of uniform random numbers `u_mat` in `[0, 1)`
    into integer lead times using `O(1)`-memory PMF quantile inversion.

    What it does
    ------------
    For each item `i`, looks up its lead-time distribution index `k = item_lt_idx[i]`
    and evaluates `_pmf_quantile_scalar_numba(pmfs[k], mins[k], bin_sizes[k], u_mat[i, t])`
    for every day `t`.

    Why it is needed
    ----------------
    Calling `DiscreteDistribution.sample()` inside a Python loop over thousands of items
    and days would allocate temporary CDF arrays on every call. Using `_pmf_quantile_scalar_numba`
    over a pre-generated uniform matrix `u_mat` performs inverse-CDF sampling in compiled
    machine code with zero heap allocations.

    Where it is used
    ----------------
    Called by `InventorySimulator.run` when `lead_time_mode="sample"` (with random `u_mat`)
    or `lead_time_mode="median"` (with `u_mat = 0.5`).
    """
    n_items, n_days = u_mat.shape
    out = np.empty((n_items, n_days), dtype=np.int64)
    for i in range(n_items):
        k = item_lt_idx[i]
        pmf = pmfs[k]
        mn = int(mins[k])
        bs = int(bin_sizes[k])
        for t in range(n_days):
            out[i, t] = _pmf_quantile_scalar_numba(pmf, mn, bs, float(u_mat[i, t]))
    return out


@njit(parallel=True, cache=True)
def _simulate_inventory_trajectory_numba(
    demand_mat: np.ndarray,
    target_mat: np.ndarray,
    lead_times_mat: np.ndarray,
    initial_on_hand: np.ndarray,
    cycle_time: int = 1,
    allow_backlog: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Parallel Numba kernel simulating multi-period order-up-to replenishment across
    `n_items` and `n_days`.

    How the look-ahead arrival buffer (`arrivals_buf`) works
    --------------------------------------------------------
    Each item `i` maintains a 1D integer array `arrivals_buf` of length `n_days + max_lt + 2`
    initialized to zero. When an order of size `q > 0` is placed on day `t` with realized
    lead time `lt > 0`, we simply add `q` to `arrivals_buf[t + lt]` and increment `in_transit += q`.
    Even if lead times are stochastic and two orders cross in time (an order placed on day `t+1`
    with a short lead time arrives on the same day or earlier than an order placed on day `t`
    with a long lead time), `arrivals_buf[day]` automatically sums all shipments arriving on
    that exact day in `O(1)` time without needing a priority queue or list of open orders.

    Sequence of daily events for item `i` on day `t`
    ------------------------------------------------
    1. **Opening Inventory**: Record `opening_inv[i, t] = on_hand` (equal to yesterday's closing stock).
    2. **Receive Shipments**: Add `arr = arrivals_buf[t]` to `on_hand` and subtract `arr` from
       `in_transit`. Record `on_hand_before_demand[i, t]` and `in_transit_before_order[i, t]`.
    3. **Review & Order Placement**: Compute `inv_pos = on_hand + in_transit - backlog`.
       If `t % cycle_time == 0` (review day), place order `q = max(0, target_mat[i, t] - inv_pos)`.
       - If `lt <= 0` (instantaneous delivery), `q` is added immediately to `on_hand` before demand.
       - If `lt > 0`, `q` is scheduled in `arrivals_buf[t + lt]` and added to `in_transit`.
    4. **Demand Fulfillment**: Realize demand `d = max(0, demand_mat[i, t])`. Total quantity
       needed is `needed = d + backlog`.
       - If `on_hand >= needed`: fulfill all `needed`, decrement `on_hand`, set `unfulfilled = 0`
         and `backlog = 0`.
       - If `on_hand < needed`: sell all available `on_hand`, set `unfulfilled = needed - on_hand`,
         set `on_hand = 0`, and if `allow_backlog=True` carry `backlog = unfulfilled` to day `t + 1`
         (otherwise `unfulfilled` is permanently lost in lost-sales retail mode).

    Where it is used
    ----------------
    Called by `InventorySimulator.run` to produce the detailed daily `trajectory_df`.
    """
    n_items, n_days = demand_mat.shape
    max_lt = 0
    for i in range(n_items):
        for t in range(n_days):
            if lead_times_mat[i, t] > max_lt:
                max_lt = int(lead_times_mat[i, t])

    opening_inv = np.empty((n_items, n_days), dtype=np.int64)
    arrivals_out = np.empty((n_items, n_days), dtype=np.int64)
    on_hand_before_demand = np.empty((n_items, n_days), dtype=np.int64)
    in_transit_before_order = np.empty((n_items, n_days), dtype=np.int64)
    inv_pos_out = np.empty((n_items, n_days), dtype=np.int64)
    order_placed = np.empty((n_items, n_days), dtype=np.int64)
    sold = np.empty((n_items, n_days), dtype=np.int64)
    lost_sales = np.empty((n_items, n_days), dtype=np.int64)
    closing_inv = np.empty((n_items, n_days), dtype=np.int64)
    in_transit_end = np.empty((n_items, n_days), dtype=np.int64)

    buf_len = n_days + max_lt + 2

    for i in prange(n_items):
        arrivals_buf = np.zeros(buf_len, dtype=np.int64)
        on_hand = int(initial_on_hand[i])
        if on_hand < 0:
            on_hand = 0
        in_transit = 0
        backlog = 0

        for t in range(n_days):
            opening_inv[i, t] = on_hand

            # 1. Receive arriving shipments at start of day t:
            arr = arrivals_buf[t]
            arrivals_out[i, t] = arr
            on_hand += arr
            in_transit -= arr
            on_hand_before_demand[i, t] = on_hand
            in_transit_before_order[i, t] = in_transit

            # 2. Compute inventory position & place replenishment order:
            inv_pos = on_hand + in_transit - backlog
            inv_pos_out[i, t] = inv_pos
            if cycle_time <= 1 or (t % cycle_time == 0):
                q = int(target_mat[i, t]) - inv_pos
                if q < 0:
                    q = 0
            else:
                q = 0
            order_placed[i, t] = q

            if q > 0:
                lt = int(lead_times_mat[i, t])
                if lt <= 0:
                    on_hand += q
                    arrivals_out[i, t] += q
                    on_hand_before_demand[i, t] = on_hand
                else:
                    arrivals_buf[t + lt] += q
                    in_transit += q

            # 3. Fulfill demand:
            d = int(demand_mat[i, t])
            if d < 0:
                d = 0
            needed = d + backlog
            if on_hand >= needed:
                s = needed
                on_hand -= needed
                unfulfilled = 0
                backlog = 0
            else:
                s = on_hand
                unfulfilled = needed - on_hand
                on_hand = 0
                if allow_backlog:
                    backlog = unfulfilled

            sold[i, t] = s
            lost_sales[i, t] = unfulfilled
            closing_inv[i, t] = on_hand
            in_transit_end[i, t] = in_transit

    return (
        opening_inv,
        arrivals_out,
        on_hand_before_demand,
        in_transit_before_order,
        inv_pos_out,
        order_placed,
        sold,
        lost_sales,
        closing_inv,
        in_transit_end,
    )


@njit(parallel=True, cache=True)
def _simulate_inventory_mc_numba(
    demand_mat: np.ndarray,
    target_mat: np.ndarray,
    pmfs: NumbaList,
    mins: np.ndarray,
    bin_sizes: np.ndarray,
    item_lt_idx: np.ndarray,
    initial_on_hand: np.ndarray,
    u_cube: np.ndarray,
    cycle_time: int = 1,
    allow_backlog: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Monte Carlo kernel simulating `n_sims` independent stochastic lead-time
    trajectories per item.

    What it does
    ------------
    Executes the exact same daily order-up-to replenishment logic as
    `_simulate_inventory_trajectory_numba` across `prange(n_sims)`, drawing lead times
    on the fly from `u_cube[sim, i, t]` via `_pmf_quantile_scalar_numba`, and accumulates
    only per-`(sim, item)` summary totals:
    `(tot_sold, tot_lost, tot_ordered, sum_closing_inv, stockout_days)`.

    Why it is needed
    ----------------
    Materializing full `(n_sims, n_items, n_days)` trajectory arrays for `n_sims = 100+`
    replications would consume gigabytes of RAM. Accumulating scalar counters in registers
    per `(sim, item)` keeps memory `O(n_sims * n_items)` and runs at native multi-core speed.

    Where it is used
    ----------------
    Called by `InventorySimulator.run` when `n_mc_simulations > 1` to populate
    `InventorySimulationResult.mc_summary_df`.
    """
    n_sims, n_items, n_days = u_cube.shape
    max_lt = 0
    for k in range(len(pmfs)):
        hi = int(mins[k]) + int(bin_sizes[k]) * (len(pmfs[k]) - 1)
        if hi > max_lt:
            max_lt = hi
    buf_len = n_days + max_lt + 2

    tot_sold = np.empty((n_sims, n_items), dtype=np.int64)
    tot_lost = np.empty((n_sims, n_items), dtype=np.int64)
    tot_ordered = np.empty((n_sims, n_items), dtype=np.int64)
    sum_closing_inv = np.empty((n_sims, n_items), dtype=np.int64)
    stockout_days = np.empty((n_sims, n_items), dtype=np.int64)

    for sim in prange(n_sims):
        for i in range(n_items):
            arrivals_buf = np.zeros(buf_len, dtype=np.int64)
            on_hand = int(initial_on_hand[i])
            if on_hand < 0:
                on_hand = 0
            in_transit = 0
            backlog = 0
            k_lt = item_lt_idx[i]
            pmf = pmfs[k_lt]
            mn = int(mins[k_lt])
            bs = int(bin_sizes[k_lt])

            s_acc = 0
            l_acc = 0
            o_acc = 0
            c_acc = 0
            so_acc = 0

            for t in range(n_days):
                arr = arrivals_buf[t]
                on_hand += arr
                in_transit -= arr

                inv_pos = on_hand + in_transit - backlog
                if cycle_time <= 1 or (t % cycle_time == 0):
                    q = int(target_mat[i, t]) - inv_pos
                    if q < 0:
                        q = 0
                else:
                    q = 0
                o_acc += q

                if q > 0:
                    lt = _pmf_quantile_scalar_numba(pmf, mn, bs, float(u_cube[sim, i, t]))
                    if lt <= 0:
                        on_hand += q
                    else:
                        arrivals_buf[t + lt] += q
                        in_transit += q

                d = int(demand_mat[i, t])
                if d < 0:
                    d = 0
                needed = d + backlog
                if on_hand >= needed:
                    s = needed
                    on_hand -= needed
                    unfulfilled = 0
                    backlog = 0
                else:
                    s = on_hand
                    unfulfilled = needed - on_hand
                    on_hand = 0
                    if allow_backlog:
                        backlog = unfulfilled

                s_acc += s
                l_acc += unfulfilled
                c_acc += on_hand
                if unfulfilled > 0:
                    so_acc += 1

            tot_sold[sim, i] = s_acc
            tot_lost[sim, i] = l_acc
            tot_ordered[sim, i] = o_acc
            sum_closing_inv[sim, i] = c_acc
            stockout_days[sim, i] = so_acc

    return tot_sold, tot_lost, tot_ordered, sum_closing_inv, stockout_days


def _lead_time_mean_var_vectors(
    df: pd.DataFrame,
    lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution] | int,
    lead_time_keys: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve per-row expected lead time ``E[L]`` and lead-time variance ``Var(L)`` for ``df``.
    """
    n_rows = len(df)
    if isinstance(lead_time, (int, float, np.integer, np.floating)):
        return np.full(n_rows, float(lead_time), dtype=np.float64), np.zeros(n_rows, dtype=np.float64)

    def _moments(d: DiscreteDistribution) -> tuple[float, float]:
        sup = d.min + np.arange(len(d.pmf), dtype=np.float64) * d.bin_size
        mu = float(np.dot(sup, d.pmf))
        var = float(np.dot((sup - mu) ** 2, d.pmf))
        return mu, var

    if isinstance(lead_time, DiscreteDistribution) or not lead_time_keys:
        d = lead_time if isinstance(lead_time, DiscreteDistribution) else next(iter(lead_time.values()))
        mu_l, var_l = _moments(d)
        return np.full(n_rows, mu_l, dtype=np.float64), np.full(n_rows, var_l, dtype=np.float64)

    moment_map: dict[Any, tuple[float, float]] = {}
    for k, d in lead_time.items():
        mv = _moments(d)
        moment_map[k] = mv
        moment_map[str(k)] = mv
    default_mv = next(iter(moment_map.values()))

    mu_vec = np.empty(n_rows, dtype=np.float64)
    var_vec = np.empty(n_rows, dtype=np.float64)
    if len(lead_time_keys) == 1:
        col = lead_time_keys[0]
        for i, val in enumerate(df[col].to_numpy()):
            mv = moment_map.get(val, moment_map.get(str(val), default_mv))
            mu_vec[i] = mv[0]
            var_vec[i] = mv[1]
    else:
        for i, row_vals in enumerate(df[lead_time_keys].itertuples(index=False, name=None)):
            mv = moment_map.get(row_vals, moment_map.get(str(row_vals), default_mv))
            mu_vec[i] = mv[0]
            var_vec[i] = mv[1]
    return mu_vec, var_vec


def compute_kings_target_inventory(
    df: pd.DataFrame,
    lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution] | int = 1,
    lead_time_keys: list[str] | None = None,
    cycle_time: int = 1,
    service_level: float = 0.95,
    history: pd.DataFrame | None = None,
    ts_id_cols: list[str] | None = None,
    period_col: str = "day",
    var_col: str = "demand",
    window: int = 56,
    method: str = "trailing",
    rounding: str = "round",
    target_col: str = "target_inventory",
    dist_col: str = "demand_distribution",
) -> pd.DataFrame:
    """
    Computes order-up-to target inventory ``S_t`` using **King's Formula** (Normal approximation
    for demand during stochastic lead time + review period):

        mu_DDLT    = (E[L] + R) * mu_D
        sigma_King = sqrt((E[L] + R) * sigma_D^2 + Var(L) * mu_D^2)
        SS_King    = Phi^{-1}(alpha) * sigma_King
        S_King     = round_or_ceil(max(0, mu_DDLT + SS_King))

    Parameters
    ----------
    df : pd.DataFrame
        Decision rows (one row per SKU and decision period ``period_col``).
    lead_time : DiscreteDistribution | dict | int
        Replenishment lead-time distribution(s) or deterministic integer lead time.
    lead_time_keys : list[str], optional
        Columns mapping each row to its lead-time distribution when ``lead_time`` is a dict.
    cycle_time : int, default 1
        Review period ``R`` in periods.
    service_level : float, default 0.95
        Target cycle service level (instock quantile ``alpha in (0, 1)``).
    history : pd.DataFrame, optional
        Long historical demand panel ``(ts_id_cols, period_col, var_col)`` required when
        ``method in ("trailing", "seasonal")``.
    ts_id_cols : list[str], optional
        Series identifier columns (inferred from ``df`` if omitted).
    period_col, var_col : str
        Period and demand column names in ``history``.
    window : int, default 56
        Trailing historical window length in periods for ``method="trailing"`` (or ``window // 7``
        same-weekday weeks for ``method="seasonal"``).
    method : {"trailing", "seasonal", "model"}, default "trailing"
        How daily demand mean ``mu_D`` and standard deviation ``sigma_D`` are estimated:
        - ``"trailing"``: sample mean and sample std (``ddof=1``) over the trailing ``window``
          periods ``[t - window, t)`` strictly before decision period ``t``.
        - ``"seasonal"``: sample mean and sample std over the last ``max(2, window // 7)``
          same-weekday observations strictly before decision period ``t``.
        - ``"model"``: uses the forecaster's predicted distribution moments on ``df`` (from
          ``dist_col`` if present, or from ``target_t_plus_<h>_mu`` and trailing ``sigma_D``)
          combined with Gaussian quantile ``Phi^{-1}(alpha)``.
    rounding : {"round", "ceil"}, default "round"
        Whether to round continuous ``mu_DDLT + z_alpha * sigma_King`` to nearest integer
        (``"round"``) or round up (``"ceil"``).
    target_col : str, default "target_inventory"
        Output column name written to the returned DataFrame copy.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``target_col``, ``kings_cycle_stock``, ``kings_safety_stock``,
        and ``kings_sigma`` columns populated.
    """
    if not (0.0 < float(service_level) < 1.0):
        raise ValueError("'service_level' must be strictly between 0 and 1 for King's formula.")
    method_norm = method.lower().strip()
    if method_norm not in ("trailing", "seasonal", "model"):
        raise ValueError(f"Unsupported King's formula method '{method}'. Choose from 'trailing', 'seasonal', 'model'.")
    round_norm = rounding.lower().strip()
    if round_norm not in ("round", "ceil"):
        raise ValueError(f"Unsupported rounding '{rounding}'. Choose from 'round', 'ceil'.")

    out = df.copy()
    mu_lt, var_lt = _lead_time_mean_var_vectors(out, lead_time, lead_time_keys)
    exposure_mean = mu_lt + float(cycle_time)
    z_alpha = float(norm.ppf(float(service_level)))

    if method_norm == "model" and dist_col in out.columns and out[dist_col].notna().all():
        distrs = list(out[dist_col])
        mu_ddlt = np.array([float(d.mean()) for d in distrs], dtype=np.float64)
        sigma_king = np.array([float(d.std()) for d in distrs], dtype=np.float64)
    else:
        if history is None:
            raise ValueError(f"'history' DataFrame is required when method='{method_norm}' and '{dist_col}' is absent.")
        id_cols = list(ts_id_cols) if ts_id_cols else [
            c for c in ("store_id", "item_id") if c in out.columns and c in history.columns
        ]
        if not id_cols:
            id_cols = [c for c in ("id", "item_id") if c in out.columns and c in history.columns]
        if not id_cols:
            raise ValueError("Could not infer 'ts_id_cols' shared between 'df' and 'history'.")

        dense_hist = DenseHistory.from_frame(history, id_cols, period_col, var_col)
        r, c = dense_hist.locate(out)
        hist_mat = dense_hist.matrix
        if method_norm == "seasonal":
            n_weeks = max(2, int(window) // 7)
            offsets = -7 * np.arange(1, n_weeks + 1)
        else:
            offsets = np.arange(-max(2, int(window)), 0)

        idx = c[:, None] + offsets[None, :]
        valid = (idx >= 0) & (idx < hist_mat.shape[1]) & (r[:, None] >= 0) & (c[:, None] >= 0)
        vals = np.where(
            valid,
            hist_mat[np.clip(r, 0, None)[:, None], np.clip(idx, 0, hist_mat.shape[1] - 1)],
            np.nan,
        )
        n_valid = np.sum(np.isfinite(vals), axis=1)
        mu_d = np.where(n_valid > 0, np.nanmean(vals, axis=1), 0.0)
        std_d = np.where(n_valid > 1, np.nanstd(vals, axis=1, ddof=1), 0.0)
        mu_d = np.nan_to_num(mu_d, nan=0.0, posinf=0.0, neginf=0.0)
        std_d = np.nan_to_num(std_d, nan=0.0, posinf=0.0, neginf=0.0)

        if method_norm == "model":
            mu_cols = sorted(col for col in out.columns if col.startswith("target_t_plus_") and col.endswith("_mu"))
            if mu_cols:
                mu_d = np.nanmean(out[mu_cols].to_numpy(dtype=np.float64), axis=1)
                mu_d = np.nan_to_num(mu_d, nan=0.0, posinf=0.0, neginf=0.0)

        mu_ddlt = exposure_mean * mu_d
        sigma_king = np.sqrt(np.maximum(0.0, exposure_mean * (std_d ** 2) + var_lt * (mu_d ** 2)))

    safety_stock = z_alpha * sigma_king
    raw_target = np.maximum(0.0, mu_ddlt + safety_stock)
    if round_norm == "ceil":
        target_inv = np.ceil(raw_target).astype(np.int64)
    else:
        target_inv = np.round(raw_target).astype(np.int64)

    out["kings_cycle_stock"] = mu_ddlt
    out["kings_safety_stock"] = safety_stock
    out["kings_sigma"] = sigma_king
    out[target_col] = target_inv
    return out


def resolve_target_inventory(
    df: pd.DataFrame,
    policy: str = "auto",
    service_level: float = 0.95,
    budget: float | None = None,
    target_col: str = "target_inventory",
    dist_col: str = "demand_distribution",
    group_cols: list[str] | None = None,
    quantile_rounding: str = "ceil",
    lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution] | int = 1,
    lead_time_keys: list[str] | None = None,
    cycle_time: int = 1,
    history: pd.DataFrame | None = None,
    ts_id_cols: list[str] | None = None,
    period_col: str = "day",
    var_col: str = "demand",
    kings_window: int = 56,
    kings_method: str = "trailing",
    kings_rounding: str = "round",
) -> pd.DataFrame:
    """
    Computes or validates `target_col` on `df` according to `policy`:

    - `policy="quantile"`: sets `target_col` to the `service_level` quantile of
      `df[dist_col]` using Numba-compiled `O(1)`-memory quantile lookup
      (`_batch_pmf_quantiles_numba` when `quantile_rounding="ceil"`, or
      `_batch_pmf_quantiles_round_numba` when `quantile_rounding="round"`).
    - `policy="kings"`: sets `target_col` using King's safety-stock formula via
      :func:`compute_kings_target_inventory`.
    - `policy="newsvendor"`: solves the budget-constrained multi-product newsvendor problem
      using `budget` and economic columns (`sell_price`, `purchase_cost`, `shortage_penalty`)
      via `_solve_multi_product_newsvendor`.
    - `policy="precomputed"`: uses the existing `df[target_col]` column.
    - `policy="auto"`: uses `df[target_col]` if already present; otherwise uses `"newsvendor"`
      if `budget` is provided and economic columns exist, else `"quantile"` at `service_level`.

    Where it is used
    ----------------
    Called at the start of `InventorySimulator.run` and exported at the package root
    for standalone target-inventory computation.
    """
    out = df.copy()
    # Normalize common economic column aliases if canonical names are absent:
    if "sell_price" not in out.columns and "price" in out.columns:
        out["sell_price"] = out["price"]
    if "purchase_cost" not in out.columns and "cost" in out.columns:
        out["purchase_cost"] = out["cost"]
    if "shortage_penalty" not in out.columns and "shortage_cost" in out.columns:
        out["shortage_penalty"] = out["shortage_cost"]

    policy_norm = policy.lower().strip()
    if policy_norm not in ("auto", "quantile", "kings", "newsvendor", "precomputed"):
        raise ValueError(
            f"Unsupported policy '{policy}'. Choose from 'auto', 'quantile', 'kings', 'newsvendor', 'precomputed'."
        )

    if policy_norm == "auto":
        if target_col in out.columns and out[target_col].notna().all():
            return out
        has_econ = all(c in out.columns for c in ("sell_price", "purchase_cost", "shortage_penalty"))
        if budget is not None and has_econ and dist_col in out.columns:
            policy_norm = "newsvendor"
        elif dist_col in out.columns:
            policy_norm = "quantile"
        else:
            raise ValueError(
                f"Cannot auto-resolve '{target_col}': provide either '{target_col}' or '{dist_col}'."
            )

    if policy_norm == "precomputed":
        if target_col not in out.columns:
            raise ValueError(f"Column '{target_col}' is required when policy='precomputed'.")
        return out

    if policy_norm == "kings":
        sl = 0.95 if service_level is None else float(service_level)
        return compute_kings_target_inventory(
            out,
            lead_time=lead_time,
            lead_time_keys=lead_time_keys,
            cycle_time=cycle_time,
            service_level=sl,
            history=history,
            ts_id_cols=ts_id_cols,
            period_col=period_col,
            var_col=var_col,
            window=kings_window,
            method=kings_method,
            rounding=kings_rounding,
            target_col=target_col,
            dist_col=dist_col,
        )

    if dist_col not in out.columns:
        if target_col in out.columns and policy_norm == "quantile":
            logger.info("'%s' not present; using existing '%s' column.", dist_col, target_col)
            return out
        raise ValueError(f"Column '{dist_col}' is required to compute target inventory with policy='{policy_norm}'.")

    if policy_norm == "quantile":
        if "service_level" in out.columns and service_level is None:
            alphas = out["service_level"].to_numpy(dtype=np.float64)
        else:
            sl = 0.95 if service_level is None else float(service_level)
            if not (0.0 <= sl <= 1.0):
                raise ValueError("'service_level' must be between 0 and 1.")
            alphas = np.full(len(out), sl, dtype=np.float64)

        q_round = quantile_rounding.lower().strip()
        if q_round not in ("ceil", "round"):
            raise ValueError(f"Unsupported quantile_rounding '{quantile_rounding}'. Choose from 'ceil', 'round'.")

        distrs = list(out[dist_col])
        pmfs = NumbaList()
        mins = np.empty(len(distrs), dtype=np.int64)
        bin_sizes = np.empty(len(distrs), dtype=np.int64)
        for idx, d in enumerate(distrs):
            pmfs.append(d.pmf)
            mins[idx] = int(d.min)
            bin_sizes[idx] = int(d.bin_size)

        if q_round == "round":
            out[target_col] = _batch_pmf_quantiles_round_numba(pmfs, mins, bin_sizes, alphas)
        else:
            out[target_col] = _batch_pmf_quantiles_numba(pmfs, mins, bin_sizes, alphas)
        return out

    # policy_norm == "newsvendor":
    if budget is None:
        raise ValueError("'budget' must be provided when policy='newsvendor'.")
    return _solve_multi_product_newsvendor(
        out, total_budget=float(budget), out_col_name=target_col, group_cols=group_cols
    )


def compute_simulation_kpis(
    trajectory_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    item_id_cols: list[str] | None = None,
    period_col: str = "day",
) -> pd.DataFrame:
    """
    Aggregates operational and economic inventory simulation KPIs overall or by `group_cols`.

    KPI Definitions
    ---------------
    - `fill_rate`: `total_sold / total_demand` (proportion of unit demand fulfilled immediately; `1.0` if `total_demand == 0`).
    - `stockout_frequency`: fraction of item-days with `lost_sales > 0`.
    - `cycle_service_level`: `1.0 - stockout_frequency` (fraction of item-days with zero stockout).
    - `avg_closing_inv`: mean daily closing on-hand inventory per item-day.
    - `avg_on_hand`: mean daily on-hand inventory after shipment arrivals and before demand.
    - `avg_in_transit`: mean daily in-transit pipeline inventory at end of day.
    - `inventory_turnover`: `total_sold / avg_closing_inv`.
    - Financial KPIs (included automatically when price/cost columns exist):
      `total_revenue`, `total_purchase_cost`, `total_holding_cost`, `total_shortage_cost`,
      `total_profit`, and `profit_margin_pct` (`100 * total_profit / total_revenue`).

    Where it is used
    ----------------
    Called by `InventorySimulator.run` (for `summary_df`, `item_summary_df`, `group_summary_df`)
    and `InventorySimulationResult.summarize`.
    """
    valid_groups = [c for c in (group_cols or []) if c in trajectory_df.columns]
    id_cols = [c for c in (item_id_cols or ["item_id"]) if c in trajectory_df.columns]
    has_econ = all(
        c in trajectory_df.columns
        for c in ("revenue", "purchase_spend", "holding_cost", "shortage_cost", "net_profit")
    )

    if valid_groups:
        grouped = trajectory_df.groupby(valid_groups, observed=True)
    else:
        grouped = [((), trajectory_df)]

    rows = []
    for g_key, gdf in grouped:
        if not isinstance(g_key, tuple):
            g_key = (g_key,)
        n_item_days = len(gdf)
        n_days = gdf[period_col].nunique() if period_col in gdf.columns else n_item_days
        n_items = gdf[id_cols].drop_duplicates().shape[0] if id_cols else max(1, n_item_days // max(1, n_days))

        tot_demand = int(gdf["demand"].sum())
        tot_sold = int(gdf["sold"].sum())
        tot_lost = int(gdf["lost_sales"].sum())
        tot_ordered = int(gdf["order_placed"].sum())
        tot_arrivals = int(gdf["arrivals"].sum())

        fill_rate = float(tot_sold / tot_demand) if tot_demand > 0 else 1.0
        stockout_freq = float((gdf["lost_sales"] > 0).mean()) if n_item_days > 0 else 0.0
        cycle_sl = 1.0 - stockout_freq

        avg_target = float(gdf["target_inventory"].mean()) if (n_item_days > 0 and "target_inventory" in gdf.columns) else 0.0
        avg_closing = float(gdf["closing_inv"].mean()) if n_item_days > 0 else 0.0
        avg_on_hand = float(gdf["on_hand_before_demand"].mean()) if n_item_days > 0 else 0.0
        avg_in_transit = float(gdf["in_transit_end"].mean()) if n_item_days > 0 else 0.0
        turnover = float(tot_sold / avg_closing) if avg_closing > 0.0 else np.nan

        row: dict[str, Any] = {
            **dict(zip(valid_groups, g_key)),
            "n_items": int(n_items),
            "n_days": int(n_days),
            "n_item_days": int(n_item_days),
            "total_demand": tot_demand,
            "total_sold": tot_sold,
            "total_lost_sales": tot_lost,
            "total_ordered": tot_ordered,
            "total_arrivals": tot_arrivals,
            "fill_rate": round(fill_rate, 4),
            "cycle_service_level": round(cycle_sl, 4),
            "stockout_frequency": round(stockout_freq, 4),
            "avg_target_inventory": round(avg_target, 2),
            "avg_closing_inv": round(avg_closing, 2),
            "avg_on_hand": round(avg_on_hand, 2),
            "avg_in_transit": round(avg_in_transit, 2),
            "inventory_turnover": round(turnover, 4) if np.isfinite(turnover) else np.nan,
        }
        if has_econ:
            rev = float(gdf["revenue"].sum())
            pur = float(gdf["purchase_spend"].sum())
            hld = float(gdf["holding_cost"].sum())
            sht = float(gdf["shortage_cost"].sum())
            prf = float(gdf["net_profit"].sum())
            row.update({
                "total_revenue": round(rev, 2),
                "total_purchase_cost": round(pur, 2),
                "total_holding_cost": round(hld, 2),
                "total_shortage_cost": round(sht, 2),
                "total_profit": round(prf, 2),
                "profit_margin_pct": round(100.0 * prf / rev, 2) if rev > 0.0 else np.nan,
            })
        rows.append(row)

    return pd.DataFrame(rows)


@dataclass
class InventorySimulationResult:
    """
    Container for multi-period inventory simulation outputs and KPIs.

    Attributes
    ----------
    trajectory_df : pd.DataFrame
        Item-by-day simulation log with columns `opening_inv`, `arrivals`,
        `on_hand_before_demand`, `in_transit_before_order`, `inventory_position`,
        `target_inventory`, `order_placed`, `lead_time_drawn`, `demand`, `sold`,
        `lost_sales`, `closing_inv`, `in_transit_end` (and financial columns when
        prices/costs are provided).
    summary_df : pd.DataFrame
        Single-row overall KPI summary across all simulated items and days.
    item_summary_df : pd.DataFrame
        Per-item KPI breakdown grouped by `item_id_cols`.
    group_summary_df : pd.DataFrame | None
        Optional KPI breakdown grouped by user-specified `group_cols` (e.g., `store_id` or `cat_id`).
    mc_summary_df : pd.DataFrame | None
        Optional Monte Carlo summary table (`mean`, `std`, `p05`, `p95`) when `n_mc_simulations > 1`.
    """
    trajectory_df: pd.DataFrame
    summary_df: pd.DataFrame
    item_summary_df: pd.DataFrame
    group_summary_df: pd.DataFrame | None = None
    mc_summary_df: pd.DataFrame | None = None
    item_id_cols: list[str] | None = None
    period_col: str = "day"

    def summarize(self, group_cols: list[str] | None = None) -> pd.DataFrame:
        """Computes simulation KPIs grouped by any subset of columns on `trajectory_df`."""
        return compute_simulation_kpis(
            self.trajectory_df,
            group_cols=group_cols,
            item_id_cols=self.item_id_cols,
            period_col=self.period_col,
        )


class InventorySimulator:
    """
    Numba-accelerated multi-period inventory replenishment simulator.

    Parameters
    ----------
    lead_time : DiscreteDistribution | dict[Any, DiscreteDistribution] | int
        Store/item lead-time distribution(s) or a fixed integer lead time.
    lead_time_keys : list[str], optional
        Columns mapping each item to its corresponding distribution when `lead_time` is a dict.
    cycle_time : int, default 1
        Review period `R` in days (orders are placed when `day_idx % cycle_time == 0`).
    lead_time_mode : str, default "sample"
        How lead times are generated on each order day:
        - `"sample"`: drawn stochastically from `lead_time` using `random_seed`.
        - `"mean"`: deterministic rounded expected value `round(E[L])`.
        - `"median"`: deterministic median `L.quantile(0.5)`.
    initial_inventory : str | int | np.ndarray, default "target_day_0"
        Initial on-hand stock at the start of the first test day:
        - `"target_day_0"`: starts at day 0's `target_inventory` (steady-state backtest).
        - `"zero"` or `int` (e.g. `0`): starts every item at that fixed integer quantity.
        - 1D `np.ndarray`: custom initial inventory per item.
    allow_backlog : bool, default False
        If `False`, unfulfilled demand is lost (standard retail lost-sales assumption).
        If `True`, unfulfilled demand is backordered and fulfilled from future arrivals.
    holding_cost_per_unit : float | str, default 0.0
        Daily holding cost per unit of closing inventory (or column name in `df`, or
        fraction of `purchase_cost` via `holding_cost_rate`).
    holding_cost_rate : float, default 0.0
        If `> 0` and `purchase_cost` is present, daily holding cost is
        `holding_cost_rate * purchase_cost` per unit of closing inventory.
    random_seed : int, default 42
        Seed for reproducible lead-time sampling.
    """

    def __init__(
        self,
        lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution] | int = 1,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        lead_time_mode: str = "sample",
        initial_inventory: str | int | float | np.ndarray = "target_day_0",
        allow_backlog: bool = False,
        holding_cost_per_unit: float = 0.0,
        holding_cost_rate: float = 0.0,
        random_seed: int = 42,
        random_state: int | None = None,
        item_id_cols: list[str] | None = None,
        ts_id_cols: list[str] | None = None,
        period_col: str = "day",
        demand_col: str = "target_t_plus_0",
        target_col: str = "target_inventory",
        policy: str = "auto",
        service_level: float = 0.95,
        budget: float | None = None,
        group_cols: list[str] | None = None,
    ):
        if cycle_time < 1:
            raise ValueError("'cycle_time' must be >= 1.")
        if lead_time_mode not in ("sample", "mean", "median"):
            raise ValueError("'lead_time_mode' must be 'sample', 'mean', or 'median'.")
        self.lead_time = lead_time
        self.lead_time_keys = list(lead_time_keys) if lead_time_keys else []
        self.cycle_time = int(cycle_time)
        self.lead_time_mode = lead_time_mode
        self.initial_inventory = initial_inventory
        self.allow_backlog = bool(allow_backlog)
        self.holding_cost_per_unit = float(holding_cost_per_unit)
        self.holding_cost_rate = float(holding_cost_rate)
        self.random_seed = int(random_state if random_state is not None else random_seed)
        self.default_item_id_cols = list(item_id_cols or ts_id_cols) if (item_id_cols or ts_id_cols) else None
        self.default_period_col = period_col
        self.default_demand_col = demand_col
        self.default_target_col = target_col
        self.default_policy = policy
        self.default_service_level = service_level
        self.default_budget = budget
        self.default_group_cols = list(group_cols) if group_cols else None

    def _prepare_lead_time_Lookup(
        self,
        items_df: pd.DataFrame,
    ) -> tuple[NumbaList, np.ndarray, np.ndarray, np.ndarray]:
        """
        Packs the lead-time distribution(s) into Numba-compatible arrays `(pmfs, mins, bin_sizes)`
        and an integer index array `item_lt_idx` mapping each item `0 .. n_items-1` to its
        corresponding lead-time distribution.

        Why it is needed
        ----------------
        Numba `@njit` kernels cannot access Python dictionaries of `DiscreteDistribution`
        objects directly. Packing the unique lead-time PMFs into a `NumbaList` and resolving
        each item's dictionary key once into `item_lt_idx[i]` lets `_sample_lead_times_numba`
        and `_simulate_inventory_mc_numba` index lead-time distributions in `O(1)` time.
        """
        pmfs = NumbaList()
        mins_list: list[int] = []
        bins_list: list[int] = []

        if isinstance(self.lead_time, (int, np.integer)):
            pmfs.append(np.ones(1, dtype=np.float64))
            mins_list.append(int(self.lead_time))
            bins_list.append(1)
            item_lt_idx = np.zeros(len(items_df), dtype=np.int64)
        elif isinstance(self.lead_time, DiscreteDistribution) or not self.lead_time_keys:
            d = (
                self.lead_time
                if isinstance(self.lead_time, DiscreteDistribution)
                else next(iter(self.lead_time.values()))
            )
            pmfs.append(np.ascontiguousarray(d.pmf, dtype=np.float64))
            mins_list.append(int(d.min))
            bins_list.append(int(d.bin_size))
            item_lt_idx = np.zeros(len(items_df), dtype=np.int64)
        else:
            key_to_idx: dict[Any, int] = {}
            for k, d in self.lead_time.items():
                idx = len(mins_list)
                key_to_idx[k] = idx
                key_to_idx[str(k)] = idx
                pmfs.append(np.ascontiguousarray(d.pmf, dtype=np.float64))
                mins_list.append(int(d.min))
                bins_list.append(int(d.bin_size))

            item_lt_idx = np.zeros(len(items_df), dtype=np.int64)
            if len(self.lead_time_keys) == 1:
                col = self.lead_time_keys[0]
                for i, val in enumerate(items_df[col].to_numpy()):
                    item_lt_idx[i] = key_to_idx.get(val, key_to_idx.get(str(val), 0))
            else:
                for i, row_vals in enumerate(items_df[self.lead_time_keys].itertuples(index=False, name=None)):
                    item_lt_idx[i] = key_to_idx.get(row_vals, key_to_idx.get(str(row_vals), 0))

        return (
            pmfs,
            np.asarray(mins_list, dtype=np.int64),
            np.asarray(bins_list, dtype=np.int64),
            item_lt_idx,
        )

    def run(
        self,
        predictions_df: pd.DataFrame,
        item_id_cols: list[str] | None = None,
        ts_id_cols: list[str] | None = None,
        period_col: str | None = None,
        demand_col: str | None = None,
        target_col: str | None = None,
        policy: str | None = None,
        service_level: float | None = None,
        budget: float | None = None,
        group_cols: list[str] | None = None,
        n_mc_simulations: int = 0,
        quantile_rounding: str = "ceil",
        history: pd.DataFrame | None = None,
        kings_window: int = 56,
        kings_method: str = "trailing",
        kings_rounding: str = "round",
    ) -> InventorySimulationResult:
        """
        Runs the multi-period inventory simulation over `predictions_df`.

        Step-by-step execution
        ----------------------
        1. Resolves `target_inventory` on `predictions_df` via `resolve_target_inventory`
           and drops the heavy `demand_distribution` object column to save memory.
        2. Sorts `df` into deterministic `(item_id_cols, period_col)` order (pivoting onto
           a balanced `n_items x n_days` grid if any item-day pairs are missing) and reshapes
           `demand_mat` and `target_mat` into contiguous `(n_items, n_days)` `int64` matrices.
        3. Initializes `init_oh` (`"target_day_0"`, `"zero"`, scalar `int`, or custom 1D array).
        4. Generates the `(n_items, n_days)` matrix of lead times according to `lead_time_mode`
           (`"sample"`, `"median"`, or `"mean"`).
        5. Executes `_simulate_inventory_trajectory_numba` and attaches all 10 trajectory
           state columns to `df` (plus daily financial P&L columns when price/cost exist).
        6. Computes `summary_df`, `item_summary_df`, `group_summary_df`, and optional
           Monte Carlo `mc_summary_df` (via `_simulate_inventory_mc_numba`).
        """
        t0 = time.perf_counter()
        item_id_cols = item_id_cols or ts_id_cols or self.default_item_id_cols
        period_col = period_col or self.default_period_col
        demand_col = demand_col or self.default_demand_col
        target_col = target_col or self.default_target_col
        policy = policy or self.default_policy
        service_level = self.default_service_level if service_level is None else service_level
        budget = self.default_budget if budget is None else budget
        group_cols = group_cols or self.default_group_cols
        if item_id_cols is None:
            item_id_cols = [c for c in ("store_id", "item_id") if c in predictions_df.columns]
            if not item_id_cols and "id" in predictions_df.columns:
                item_id_cols = ["id"]
            if not item_id_cols:
                raise ValueError("Could not infer 'item_id_cols'; please specify explicitly.")

        nv_group_cols = (
            ([item_id_cols[0], period_col] if len(item_id_cols) > 1 else [period_col])
            if (group_cols is None or period_col not in group_cols)
            else group_cols
        )
        df = resolve_target_inventory(
            predictions_df,
            policy=policy,
            service_level=service_level,
            budget=budget,
            target_col=target_col,
            group_cols=nv_group_cols,
            quantile_rounding=quantile_rounding,
            lead_time=self.lead_time,
            lead_time_keys=self.lead_time_keys,
            cycle_time=self.cycle_time,
            history=history,
            ts_id_cols=item_id_cols,
            period_col=period_col,
            kings_window=kings_window,
            kings_method=kings_method,
            kings_rounding=kings_rounding,
        )
        if "demand_distribution" in df.columns:
            df = df.drop(columns=["demand_distribution"])

        if demand_col not in df.columns:
            if "actual" in df.columns:
                demand_col = "actual"
            else:
                raise ValueError(f"Demand column '{demand_col}' not found in DataFrame.")

        # Sort panel in deterministic (item_id_cols, period_col) order:
        df = df.copy()
        if isinstance(df[period_col].dtype, pd.CategoricalDtype):
            df[period_col] = df[period_col].astype(df[period_col].cat.categories.dtype)
        df.sort_values(by=item_id_cols + [period_col], inplace=True, ignore_index=True)

        periods = sorted(df[period_col].unique())
        n_days = len(periods)
        item_meta_cols = list(item_id_cols)
        if self.lead_time_keys is not None:
            for k_col in self.lead_time_keys:
                if k_col not in item_meta_cols and k_col in df.columns:
                    item_meta_cols.append(k_col)
        items_df = df[item_meta_cols].drop_duplicates(subset=item_id_cols).reset_index(drop=True)
        n_items = len(items_df)

        if len(df) != n_items * n_days:
            # Pivot onto complete balanced (item x period) grid if needed:
            grid = pd.merge(items_df, pd.DataFrame({period_col: periods}), how="cross")
            df = pd.merge(grid, df, on=item_id_cols + [period_col], how="left")
            df.sort_values(by=item_id_cols + [period_col], inplace=True, ignore_index=True)

        demand_mat = np.ascontiguousarray(
            df[demand_col].fillna(0).to_numpy(dtype=np.int64).reshape(n_items, n_days)
        )
        target_mat = np.ascontiguousarray(
            df[target_col].fillna(0).to_numpy(dtype=np.int64).reshape(n_items, n_days)
        )

        # Resolve initial on-hand inventory per item:
        if isinstance(self.initial_inventory, str):
            if self.initial_inventory == "target_day_0":
                init_oh = np.ascontiguousarray(target_mat[:, 0], dtype=np.int64)
            elif self.initial_inventory == "zero":
                init_oh = np.zeros(n_items, dtype=np.int64)
            else:
                raise ValueError(
                    f"Unsupported initial_inventory string '{self.initial_inventory}'. "
                    "Use 'target_day_0', 'zero', an int, or a 1D array."
                )
        elif isinstance(self.initial_inventory, (int, float, np.integer, np.floating)):
            init_oh = np.full(n_items, int(self.initial_inventory), dtype=np.int64)
        else:
            init_oh = np.ascontiguousarray(np.asarray(self.initial_inventory, dtype=np.int64))
            if len(init_oh) != n_items:
                raise ValueError(f"Expected initial_inventory of length {n_items}, got {len(init_oh)}.")

        pmfs, mins, bin_sizes, item_lt_idx = self._prepare_lead_time_Lookup(items_df)
        rng = np.random.default_rng(self.random_seed)

        if self.lead_time_mode == "sample":
            u_mat = rng.random((n_items, n_days), dtype=np.float64)
            lead_times_mat = _sample_lead_times_numba(pmfs, mins, bin_sizes, item_lt_idx, u_mat)
        elif self.lead_time_mode == "median":
            u_mat = np.full((n_items, n_days), 0.5, dtype=np.float64)
            lead_times_mat = _sample_lead_times_numba(pmfs, mins, bin_sizes, item_lt_idx, u_mat)
        else:
            # Deterministic expected lead time per distribution:
            lt_means = np.empty(len(pmfs), dtype=np.int64)
            for k in range(len(pmfs)):
                p = pmfs[k]
                sup = mins[k] + bin_sizes[k] * np.arange(len(p))
                lt_means[k] = int(np.round(np.dot(sup, p)))
            lead_times_mat = np.repeat(lt_means[item_lt_idx][:, None], n_days, axis=1)

        logger.info(
            "Running Numba inventory simulation for %d items across %d days (cycle_time=%d, mode=%s, init=%s)...",
            n_items,
            n_days,
            self.cycle_time,
            self.lead_time_mode,
            self.initial_inventory if not isinstance(self.initial_inventory, np.ndarray) else "custom_array",
        )

        (
            opening_inv,
            arrivals_out,
            on_hand_before_demand,
            in_transit_before_order,
            inv_pos_out,
            order_placed,
            sold,
            lost_sales,
            closing_inv,
            in_transit_end,
        ) = _simulate_inventory_trajectory_numba(
            demand_mat,
            target_mat,
            lead_times_mat,
            init_oh,
            cycle_time=self.cycle_time,
            allow_backlog=self.allow_backlog,
        )

        df["opening_inv"] = opening_inv.ravel()
        df["arrivals"] = arrivals_out.ravel()
        df["on_hand_before_demand"] = on_hand_before_demand.ravel()
        df["in_transit_before_order"] = in_transit_before_order.ravel()
        df["inventory_position"] = inv_pos_out.ravel()
        df["target_inventory"] = target_mat.ravel()
        df["order_placed"] = order_placed.ravel()
        df["lead_time_drawn"] = lead_times_mat.ravel()
        df["demand"] = demand_mat.ravel()
        df["sold"] = sold.ravel()
        df["lost_sales"] = lost_sales.ravel()
        df["closing_inv"] = closing_inv.ravel()
        df["in_transit_end"] = in_transit_end.ravel()

        # Compute economic P&L columns when price/cost inputs exist:
        has_price = "sell_price" in df.columns
        has_cost = "purchase_cost" in df.columns
        if has_price or has_cost:
            sell_p = df["sell_price"].fillna(0.0).to_numpy(dtype=np.float64) if has_price else np.zeros(len(df))
            purch_c = df["purchase_cost"].fillna(0.0).to_numpy(dtype=np.float64) if has_cost else np.zeros(len(df))
            short_p = (
                df["shortage_penalty"].fillna(0.0).to_numpy(dtype=np.float64)
                if "shortage_penalty" in df.columns
                else np.zeros(len(df))
            )
            if "holding_cost" in df.columns:
                unit_hold = df["holding_cost"].fillna(0.0).to_numpy(dtype=np.float64)
            elif self.holding_cost_rate > 0.0 and has_cost:
                unit_hold = self.holding_cost_rate * purch_c
            else:
                unit_hold = np.full(len(df), self.holding_cost_per_unit, dtype=np.float64)

            df["revenue"] = np.round(df["sold"].to_numpy(dtype=np.float64) * sell_p, 4)
            df["purchase_spend"] = np.round(df["order_placed"].to_numpy(dtype=np.float64) * purch_c, 4)
            df["holding_cost"] = np.round(df["closing_inv"].to_numpy(dtype=np.float64) * unit_hold, 4)
            df["shortage_cost"] = np.round(df["lost_sales"].to_numpy(dtype=np.float64) * short_p, 4)
            df["net_profit"] = np.round(
                df["revenue"] - df["purchase_spend"] - df["holding_cost"] - df["shortage_cost"], 4
            )

        summary_df = compute_simulation_kpis(
            df, group_cols=None, item_id_cols=item_id_cols, period_col=period_col
        )
        item_summary_df = compute_simulation_kpis(
            df, group_cols=item_id_cols, item_id_cols=item_id_cols, period_col=period_col
        )
        group_summary_df = (
            compute_simulation_kpis(df, group_cols=group_cols, item_id_cols=item_id_cols, period_col=period_col)
            if group_cols
            else None
        )

        mc_summary_df = None
        if n_mc_simulations > 1:
            u_cube = rng.random((int(n_mc_simulations), n_items, n_days), dtype=np.float64)
            tot_s, tot_l, tot_o, sum_c, so_days = _simulate_inventory_mc_numba(
                demand_mat,
                target_mat,
                pmfs,
                mins,
                bin_sizes,
                item_lt_idx,
                init_oh,
                u_cube,
                cycle_time=self.cycle_time,
                allow_backlog=self.allow_backlog,
            )
            total_dem_all = float(demand_mat.sum())
            sim_fill_rates = tot_s.sum(axis=1) / max(1.0, total_dem_all)
            sim_stockout_freq = so_days.sum(axis=1) / float(n_items * n_days)
            sim_avg_inv = sum_c.sum(axis=1) / float(n_items * n_days)
            sim_tot_ordered = tot_o.sum(axis=1)
            mc_summary_df = pd.DataFrame({
                "metric": ["fill_rate", "cycle_service_level", "stockout_frequency", "avg_closing_inv", "total_ordered"],
                "mean": [
                    float(np.mean(sim_fill_rates)),
                    float(np.mean(1.0 - sim_stockout_freq)),
                    float(np.mean(sim_stockout_freq)),
                    float(np.mean(sim_avg_inv)),
                    float(np.mean(sim_tot_ordered)),
                ],
                "std": [
                    float(np.std(sim_fill_rates)),
                    float(np.std(1.0 - sim_stockout_freq)),
                    float(np.std(sim_stockout_freq)),
                    float(np.std(sim_avg_inv)),
                    float(np.std(sim_tot_ordered)),
                ],
                "p05": [
                    float(np.quantile(sim_fill_rates, 0.05)),
                    float(np.quantile(1.0 - sim_stockout_freq, 0.05)),
                    float(np.quantile(sim_stockout_freq, 0.05)),
                    float(np.quantile(sim_avg_inv, 0.05)),
                    float(np.quantile(sim_tot_ordered, 0.05)),
                ],
                "p95": [
                    float(np.quantile(sim_fill_rates, 0.95)),
                    float(np.quantile(1.0 - sim_stockout_freq, 0.95)),
                    float(np.quantile(sim_stockout_freq, 0.95)),
                    float(np.quantile(sim_avg_inv, 0.95)),
                    float(np.quantile(sim_tot_ordered, 0.95)),
                ],
            })

        logger.info(
            "Inventory simulation finished in %.2fs | Fill Rate=%.2f%%, Stockout Freq=%.2f%%, Avg Closing Inv=%.2f",
            time.perf_counter() - t0,
            100.0 * float(summary_df["fill_rate"].iloc[0]),
            100.0 * float(summary_df["stockout_frequency"].iloc[0]),
            float(summary_df["avg_closing_inv"].iloc[0]),
        )
        return InventorySimulationResult(
            trajectory_df=df,
            summary_df=summary_df,
            item_summary_df=item_summary_df,
            group_summary_df=group_summary_df,
            mc_summary_df=mc_summary_df,
            item_id_cols=item_id_cols,
            period_col=period_col,
        )


def simulate_inventory(
    predictions_df: pd.DataFrame,
    lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution] | int = 1,
    lead_time_keys: list[str] | None = None,
    cycle_time: int = 1,
    policy: str = "auto",
    service_level: float = 0.95,
    budget: float | None = None,
    initial_inventory: str | int | float | np.ndarray = "target_day_0",
    lead_time_mode: str = "sample",
    allow_backlog: bool = False,
    holding_cost_per_unit: float = 0.0,
    holding_cost_rate: float = 0.0,
    item_id_cols: list[str] | None = None,
    ts_id_cols: list[str] | None = None,
    period_col: str = "day",
    demand_col: str = "target_t_plus_0",
    target_col: str = "target_inventory",
    group_cols: list[str] | None = None,
    n_mc_simulations: int = 0,
    random_seed: int = 42,
    random_state: int | None = None,
    quantile_rounding: str = "ceil",
    history: pd.DataFrame | None = None,
    kings_window: int = 56,
    kings_method: str = "trailing",
    kings_rounding: str = "round",
) -> InventorySimulationResult:
    """
    Functional convenience wrapper around `InventorySimulator(...).run(predictions_df, ...)`.

    Where it is used
    ----------------
    Exported at the package root (`probabilistic_modeling.simulate_inventory`) for one-line
    inventory backtesting.
    """
    simulator = InventorySimulator(
        lead_time=lead_time,
        lead_time_keys=lead_time_keys,
        cycle_time=cycle_time,
        lead_time_mode=lead_time_mode,
        initial_inventory=initial_inventory,
        allow_backlog=allow_backlog,
        holding_cost_per_unit=holding_cost_per_unit,
        holding_cost_rate=holding_cost_rate,
        random_seed=random_seed,
        random_state=random_state,
    )
    return simulator.run(
        predictions_df=predictions_df,
        item_id_cols=item_id_cols or ts_id_cols,
        period_col=period_col,
        demand_col=demand_col,
        target_col=target_col,
        policy=policy,
        service_level=service_level,
        budget=budget,
        group_cols=group_cols,
        n_mc_simulations=n_mc_simulations,
        quantile_rounding=quantile_rounding,
        history=history,
        kings_window=kings_window,
        kings_method=kings_method,
        kings_rounding=kings_rounding,
    )
