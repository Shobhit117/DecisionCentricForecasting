import logging
import time
import numpy as np
import pandas as pd
from numba.typed import List as NumbaList
from discrete_distribution import DiscreteDistribution
from ddlt import _batch_pmf_quantiles_numba

logger = logging.getLogger(__name__)


def solve_multi_product_newsvendor(
    demand_distributions: list[DiscreteDistribution] | pd.Series,
    selling_price: list[float] | np.ndarray,
    purchase_cost: list[float] | np.ndarray,
    shortage_penalty: list[float] | np.ndarray,
    available_budget: float,
    max_iters: int = 200,
    tol: float = 1e-4,
) -> np.ndarray:
    demand_list = list(demand_distributions)
    num_products = len(demand_list)

    if num_products != len(selling_price):
        raise ValueError(f"Selling price length mismatch: expected {num_products}, got {len(selling_price)}.")
    if num_products != len(purchase_cost):
        raise ValueError(f"Purchase cost length mismatch: expected {num_products}, got {len(purchase_cost)}.")
    if num_products != len(shortage_penalty):
        raise ValueError(f"Shortage penalty length mismatch: expected {num_products}, got {len(shortage_penalty)}.")

    selling_price = np.asarray(selling_price, dtype=float)
    purchase_cost = np.asarray(purchase_cost, dtype=float)
    shortage_penalty = np.asarray(shortage_penalty, dtype=float)

    if np.any(selling_price <= 0):
        raise ValueError("Selling price must be positive.")
    if np.any(purchase_cost <= 0):
        raise ValueError("Purchase cost must be positive.")
    if np.any(shortage_penalty < 0):
        raise ValueError("Shortage penalty must be non-negative.")
    if available_budget < 0:
        raise ValueError("Available budget must be non-negative.")

    mins = np.array([distr.min for distr in demand_list], dtype=np.int64)
    if np.any(mins < 0):
        raise ValueError("Demand must be non-negative.")

    if available_budget == 0 or num_products == 0:
        return np.zeros(num_products, dtype=np.int64)

    bin_sizes = np.array([distr.bin_size for distr in demand_list], dtype=np.int64)
    pmfs = NumbaList()
    for distr in demand_list:
        pmfs.append(distr.pmf)

    denom = selling_price + shortage_penalty
    base_num = selling_price + shortage_penalty

    def opt_inventory(k: float) -> np.ndarray:
        alpha = np.clip((base_num - purchase_cost * (1.0 + k)) / denom, 0.0, 1.0)
        return _batch_pmf_quantiles_numba(pmfs, mins, bin_sizes, alpha)

    x = opt_inventory(0.0)
    unconstrained_spend = float(np.sum(x * purchase_cost))
    if unconstrained_spend <= available_budget:
        logger.debug(
            "Unconstrained newsvendor solution satisfies budget (spend=%.2f <= %.2f)",
            unconstrained_spend,
            available_budget,
        )
        return x
    logger.debug("Iteration 0 (unconstrained): budget used = %.2f > %.2f", unconstrained_spend, available_budget)

    # Find upper bound on k large enough to force low inventory.
    k_lb = 0.0
    k_ub = float(np.max(denom / purchase_cost - 1.0))

    # Initiate binary search on k:
    for i in range(max_iters):
        k_mid = 0.5 * (k_lb + k_ub)
        x = opt_inventory(k_mid)
        budget_used = float(np.sum(x * purchase_cost))
        logger.debug("Iteration %d: lb = %.6f, ub = %.6f, budget used = %.2f", i, k_lb, k_ub, budget_used)

        if budget_used <= available_budget:
            k_ub = k_mid
        else:
            k_lb = k_mid

        if (k_ub - k_lb < tol):
            break
    x = opt_inventory(k_ub)  # k_ub always satisfies the budget constraint (the last k_mid may not)
    return x


def _solve_multi_product_newsvendor(
    df: pd.DataFrame,
    total_budget: float,
    out_col_name: str = 'target_inventory',
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Solves the budget-constrained multi-product newsvendor problem.

    When ``group_cols`` is provided (e.g. ``['store_id', 'day']``), an independent
    newsvendor problem with ``total_budget`` is solved for each group (Option A:
    one budget per store per decision day).
    """
    t0 = time.perf_counter()
    if group_cols is not None:
        valid_group_cols = [c for c in group_cols if c in df.columns]
        if valid_group_cols and df[valid_group_cols].drop_duplicates().shape[0] > 1:
            groups = list(df.groupby(valid_group_cols, observed=True).groups.items())
            logger.info(
                "Solving %d independent multi-product newsvendor problems (grouped by %s, budget=%.2f each)...",
                len(groups),
                valid_group_cols,
                total_budget,
            )
            result_df = df.drop(columns=['demand_distribution']).copy()
            result_df[out_col_name] = 0
            spends = []
            for group_key, idx in groups:
                sub = df.loc[idx]
                inv = solve_multi_product_newsvendor(
                    sub['demand_distribution'],
                    sub['sell_price'].to_numpy(),
                    sub['purchase_cost'].to_numpy(),
                    sub['shortage_penalty'].to_numpy(),
                    total_budget,
                )
                result_df.loc[idx, out_col_name] = inv
                spends.append(float(np.sum(inv * sub['purchase_cost'].to_numpy())))
            logger.info(
                "Completed %d newsvendor problems in %.2fs (budget used min=%.2f, mean=%.2f, max=%.2f / %.2f)",
                len(groups),
                time.perf_counter() - t0,
                min(spends),
                float(np.mean(spends)),
                max(spends),
                total_budget,
            )
            return result_df

    logger.info(
        "Solving multi-product newsvendor problem for %d products (budget=%.2f)...",
        len(df),
        total_budget,
    )
    demand_distributions = df['demand_distribution']  # do not create a copy of this column
    selling_price = df['sell_price'].to_numpy()
    purchase_cost = df['purchase_cost'].to_numpy()
    shortage_penalty = df['shortage_penalty'].to_numpy()
    result_df = df.drop(columns=['demand_distribution']).copy()
    inv = solve_multi_product_newsvendor(
        demand_distributions, selling_price, purchase_cost, shortage_penalty, total_budget
    )
    result_df[out_col_name] = inv
    logger.info(
        "Completed newsvendor problem in %.2fs (budget used=%.2f / %.2f, total units=%d)",
        time.perf_counter() - t0,
        float(np.sum(inv * purchase_cost)),
        total_budget,
        int(np.sum(inv)),
    )
    return result_df
