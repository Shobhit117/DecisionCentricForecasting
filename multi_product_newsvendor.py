import numpy as np
from discrete_distribution import DiscreteDistribution

def solve_multi_product_newsvendor(
    demand_distributions: list[DiscreteDistribution],
    selling_price: list[float] | np.ndarray,
    purchase_cost: list[float] | np.ndarray,
    shortage_penalty: list[float] | np.ndarray,
    available_budget: float,
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
            distr.quantile(alpha[i])
            for i, distr in enumerate(demand_distributions)
        ])

    x0 = opt_inventory(0.0)

    if np.sum(x0 * purchase_cost) <= available_budget:
        return x0

    # Find upper bound on k large enough to force low inventory.
    k_lb = 0.0
    k_ub = np.max((selling_price + shortage_penalty) / purchase_cost - 1.0)
    k_ub = max(k_ub, 1.0)

    tol = 1e-4
    max_iters = 200

    best_feasible_x = np.zeros(num_products)

    for _ in range(max_iters):
        k_mid = 0.5 * (k_lb + k_ub)
        x = opt_inventory(k_mid)
        budget_used = np.sum(x * purchase_cost)

        if budget_used <= available_budget:
            best_feasible_x = x
            k_ub = k_mid
        else:
            k_lb = k_mid

        if k_ub - k_lb < tol:
            break

    return best_feasible_x

    