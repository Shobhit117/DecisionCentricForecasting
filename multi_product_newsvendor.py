import numpy as np
import pandas as pd
from discrete_distribution import DiscreteDistribution
import logging
logger = logging.getLogger(__name__)

def solve_multi_product_newsvendor(
    demand_distributions: list[DiscreteDistribution],
    selling_price: list[float] | np.ndarray,
    purchase_cost: list[float] | np.ndarray,
    shortage_penalty: list[float] | np.ndarray,
    available_budget: float,
    max_iters: int = 200,
    tol: float = 1e-4
) -> np.ndarray:
    num_products = len(demand_distributions)

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

    min_demand_values = np.array([distr.min for distr in demand_distributions])
    if np.any(min_demand_values < 0):
        raise ValueError("Demand must be non-negative.")

    if available_budget == 0:
        return np.zeros(num_products)

    def opt_inventory(k: float) -> np.ndarray:
        alpha = (
            selling_price
            - purchase_cost * (1.0 + k)
            + shortage_penalty
        ) / (selling_price + shortage_penalty)

        alpha = np.clip(alpha, 0.0, 1.0)

        return np.array([
            distr.quantile(alpha[i])[0]
            for i, distr in enumerate(demand_distributions)
        ])
    
    logger.info('== Solving Multi-Product Newsvendor Problem ==')
    x = opt_inventory(0.0)
    if np.sum(x * purchase_cost) <= available_budget:
        return x
    logger.info(f'Iteration 0: Budget used = {np.sum(x * purchase_cost)}')
    
    # Find upper bound on k large enough to force low inventory.
    k_lb = 0.0
    k_ub = np.max((selling_price + shortage_penalty) / purchase_cost - 1.0)
    
    # Initiate binary search on k:
    for i in range(max_iters):
        k_mid = 0.5 * (k_lb + k_ub)
        x = opt_inventory(k_mid)
        print(np.sum(x))
        budget_used = np.sum(x * purchase_cost)
        logger.info(f'Iteration {i}: lb = {k_lb}, ub = {k_ub}, budget used = {budget_used}')
        
        if budget_used <= available_budget:
            k_ub = k_mid
        else:
            k_lb = k_mid
        
        if (k_ub - k_lb < tol):
            break
    logger.info('== done ==')
    print(np.sum(x))
    return x

def _solve_multi_product_newsvendor(df : pd.DataFrame, total_budget : float, out_col_name : str = 'target_inventory') -> pd.DataFrame:
    demand_distributions = df['demand_distribution'] # do not create a copy of this column
    selling_price = df['sell_price'].to_list()
    purchase_cost = df['purchase_cost'].to_list()
    shortage_penalty = df['shortage_penalty'].to_list()
    result_df = df.drop(columns=['demand_distribution'])
    result_df[out_col_name] = solve_multi_product_newsvendor(demand_distributions, selling_price, purchase_cost, shortage_penalty, total_budget)
    return result_df