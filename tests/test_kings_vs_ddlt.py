import numpy as np
import pandas as pd

from probabilistic_modeling import (
    DiscreteDistribution,
    compute_kings_target_inventory,
    resolve_target_inventory,
    simulate_inventory,
)


def test_compute_kings_target_inventory_and_simulation():
    # Build synthetic 2-SKU history over 30 training days + 5 test decision days
    rows = []
    rng = np.random.default_rng(123)
    for item_id, rate in [("ITEM_A", 2.0), ("ITEM_B", 6.0)]:
        demands = rng.poisson(rate, size=35)
        for d_idx, dem in enumerate(demands, start=1):
            rows.append({"store_id": "CA_1", "item_id": item_id, "day": d_idx, "demand": int(dem)})
    history_df = pd.DataFrame(rows)

    test_df = history_df[history_df["day"] > 30].rename(columns={"demand": "target_t_plus_0"}).copy()
    lt_dist = DiscreteDistribution(np.array([0.2, 0.5, 0.3]), 1, 1)

    # 1. Test compute_kings_target_inventory directly
    kings_80 = compute_kings_target_inventory(
        test_df,
        lead_time=lt_dist,
        cycle_time=1,
        service_level=0.80,
        history=history_df,
        ts_id_cols=["store_id", "item_id"],
        window=28,
        method="trailing",
        rounding="round",
    )
    kings_99 = compute_kings_target_inventory(
        test_df,
        lead_time=lt_dist,
        cycle_time=1,
        service_level=0.99,
        history=history_df,
        ts_id_cols=["store_id", "item_id"],
        window=28,
        method="trailing",
        rounding="ceil",
    )
    assert "target_inventory" in kings_80.columns
    assert "kings_safety_stock" in kings_80.columns
    assert np.all(kings_99["target_inventory"].to_numpy() >= kings_80["target_inventory"].to_numpy())

    # 2. Test simulate_inventory with policy='kings'
    sim_king = simulate_inventory(
        test_df,
        lead_time=lt_dist,
        cycle_time=1,
        policy="kings",
        service_level=0.95,
        history=history_df,
        kings_window=28,
        kings_method="trailing",
        kings_rounding="round",
    )
    assert "avg_target_inventory" in sim_king.summary_df.columns
    assert float(sim_king.summary_df["fill_rate"].iloc[0]) > 0.0

    # 3. Test quantile_rounding='round' vs 'ceil' on discrete distributions
    ddlt_df = test_df.copy()
    ddlt_df["demand_distribution"] = [
        DiscreteDistribution.from_parametric("poisson", (6.0, 0)) for _ in range(len(ddlt_df))
    ]
    res_ceil = resolve_target_inventory(ddlt_df, policy="quantile", service_level=0.85, quantile_rounding="ceil")
    res_round = resolve_target_inventory(ddlt_df, policy="quantile", service_level=0.85, quantile_rounding="round")
    assert np.all(res_ceil["target_inventory"].to_numpy() >= res_round["target_inventory"].to_numpy())
