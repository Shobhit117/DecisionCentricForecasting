import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddlt import DDLTCalculator, _ddlt_from_quantiles_row_numba  # noqa: E402
from discrete_distribution import DiscreteDistribution  # noqa: E402
from forecasters import CatBoostQuantileForecaster, GLMForecaster  # noqa: E402
from multi_product_newsvendor import _solve_multi_product_newsvendor  # noqa: E402
from quantile_distribution import (  # noqa: E402
    DEFAULT_QUANTILES,
    distribution_mean,
    quantiles_to_distribution,
    quantiles_to_mean,
    rearrange_quantiles,
)
from time_series import DiscreteTimeSeries, compute_lead_time_demand  # noqa: E402

LEVELS = np.array(DEFAULT_QUANTILES)


def test_rearrange_fixes_crossing_and_negatives():
    q = rearrange_quantiles(np.array([[2.0, 1.0, -0.5, 3.0]]))
    assert np.all(np.diff(q, axis=-1) >= 0)
    assert q.min() >= 0


def test_all_zero_quantiles_is_point_mass_at_zero():
    d = quantiles_to_distribution(LEVELS, np.zeros(len(LEVELS)))
    assert d.min == 0 and np.allclose(d.pmf, [1.0])


def test_pmf_is_valid_and_respects_integer_quantile_bounds():
    true = stats.poisson(4.0)
    values = true.ppf(LEVELS)
    d = quantiles_to_distribution(LEVELS, values)
    assert d.min >= 0
    assert np.isclose(d.pmf.sum(), 1.0) and np.all(d.pmf >= 0)
    support = d.min + np.arange(len(d.pmf))
    cdf = dict(zip(support, d.cdf))
    for tau, y in zip(LEVELS, values.astype(int)):
        assert cdf.get(y, 1.0) >= tau - 1e-9
        assert cdf.get(y - 1, 0.0) < tau + 1e-9


@pytest.mark.parametrize("dist", [stats.poisson(3.0), stats.nbinom(2, 0.1), stats.poisson(40.0)])
def test_reconstruction_approximates_true_distribution(dist):
    d = quantiles_to_distribution(LEVELS, dist.ppf(LEVELS))
    assert abs(distribution_mean(d) - dist.mean()) / dist.mean() < 0.1
    assert d.quantile(0.5)[0] == dist.ppf(0.5)


def test_quantiles_to_mean_matches_distribution_mean():
    rng = np.random.default_rng(0)
    q = rearrange_quantiles(rng.gamma(2.0, 3.0, size=(20, len(LEVELS))))
    means = quantiles_to_mean(LEVELS, q)
    for row, m in zip(q, means):
        assert np.isclose(distribution_mean(quantiles_to_distribution(LEVELS, row)), m)


def test_reconstructed_distribution_supports_random_sum():
    daily = [quantiles_to_distribution(LEVELS, stats.poisson(mu).ppf(LEVELS)) for mu in (2.0, 3.0, 4.0)]
    lead_time = DiscreteDistribution(np.array([0.5, 0.5]), min_value=2)
    ddlt = lead_time.random_sum(daily)
    assert np.isclose(ddlt.pmf.sum(), 1.0)
    expected = 0.5 * sum(map(distribution_mean, daily[:2])) + 0.5 * sum(map(distribution_mean, daily))
    assert np.isclose(distribution_mean(ddlt), expected)


def test_numba_ddlt_from_quantiles_matches_random_sum():
    q_rows = np.vstack([stats.poisson(mu).ppf(LEVELS) for mu in (2.0, 3.5, 5.0)])
    daily = [quantiles_to_distribution(LEVELS, row) for row in q_rows]
    lead_time = DiscreteDistribution(np.array([0.3, 0.7]), min_value=2)
    expected = lead_time.random_sum(daily)

    pmf, min_val = _ddlt_from_quantiles_row_numba(
        lead_time.pmf, lead_time.min, LEVELS, q_rows, np.ones(1, dtype=np.float64), 0, 1e-4
    )
    assert min_val == expected.min
    assert np.allclose(pmf, expected.pmf)


def test_numba_quantile_matches_searchsorted_cdf():
    d = DiscreteDistribution.from_parametric("nbinom", (3.0, 0.25, 2))
    d.merge_bins(3)
    qs = np.linspace(0.0, 1.0, 101)
    indices = np.clip(np.searchsorted(d.cdf, qs, side="left"), 0, len(d.pmf) - 1)
    expected = d.min + indices * d.bin_size
    assert np.array_equal(d.quantile(qs), expected)


def test_invalid_levels_raise():
    with pytest.raises(ValueError):
        quantiles_to_distribution([0.5, 0.5], [1.0, 2.0])
    with pytest.raises(ValueError):
        quantiles_to_distribution([0.0, 0.5], [1.0, 2.0])


def test_no_future_target_leakage_across_train_test_boundary():
    days = list(range(1, 51))
    df = pd.DataFrame({
        "id": ["item_1"] * len(days),
        "day": days,
        "demand": list(range(1, 51)),
    })
    ts = DiscreteTimeSeries(
        df,
        ts_id_cols=["id"],
        period_col="day",
        var_col="demand",
        lags=(1, 2),
        rolling_windows=(3,),
        num_future_targets=4,
        num_test_days=10,
        fit_historical_distributions=False,
    )
    train_rows = ts.df[ts.df["is_test"] == 0]
    test_rows = ts.df[ts.df["is_test"] == 1]

    for h, col in enumerate(ts.target_cols):
        non_null_targets = train_rows[col].dropna()
        assert (non_null_targets < 41).all(), f"Leakage in {col}"
        if h > 0:
            leaked_days = list(range(41 - h, 41))
            assert train_rows.loc[train_rows["day"].isin(leaked_days), col].isna().all()

    assert test_rows.loc[test_rows["day"] == 41, "target_t_plus_3"].item() == 44.0


def test_per_day_newsvendor_allocates_full_budget_to_each_day():
    # Two days (day 1, day 2), two items per day, budget = 50 per day
    d1 = DiscreteDistribution.from_parametric("poisson", (10.0, 0))
    d2 = DiscreteDistribution.from_parametric("poisson", (10.0, 0))
    df = pd.DataFrame({
        "store_id": ["S1", "S1", "S1", "S1"],
        "day": [1, 1, 2, 2],
        "item_id": ["A", "B", "A", "B"],
        "sell_price": [10.0, 10.0, 10.0, 10.0],
        "purchase_cost": [5.0, 5.0, 5.0, 5.0],
        "shortage_penalty": [2.0, 2.0, 2.0, 2.0],
        "demand_distribution": [d1, d2, d1, d2],
    })
    budget = 50.0
    res = _solve_multi_product_newsvendor(df, budget, group_cols=["store_id", "day"])
    day1_spend = (res.loc[res["day"] == 1, "target_inventory"] * 5.0).sum()
    day2_spend = (res.loc[res["day"] == 2, "target_inventory"] * 5.0).sum()
    assert 0 < day1_spend <= budget
    assert np.isclose(day1_spend, day2_spend)
    # Without per-day grouping, combined spend across both days would be capped at 50;
    # with per-day grouping, each day uses the full 50 (total 100 across the 2 days).
    assert day1_spend + day2_spend > budget


def test_production_mode_with_future_covariates_and_forecasters():
    rng = np.random.default_rng(7)
    days = list(range(1, 61))
    records = []
    for item_id, base in [("item_A", 4.0), ("item_B", 8.0)]:
        for d in days:
            records.append({
                "store_id": "S1",
                "item_id": item_id,
                "day": d,
                "demand": int(rng.poisson(base + (2.0 if d % 7 in (1, 2) else 0.0))),
            })
    sales_df = pd.DataFrame(records)

    # Calendar extends into future unseen days 61..65 (production scenario):
    cal_days = list(range(1, 66))
    calendar_df = pd.DataFrame({
        "day": cal_days,
        "wday": [(d - 1) % 7 + 1 for d in cal_days],
        "promo_flag": [1 if d >= 61 else 0 for d in cal_days],
    })

    # Production mode: num_test_days=0 (all 60 observed days are training data)
    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=3,
        num_test_days=0,
        use_one_hot_encoding=False,
        downsample_zeros_frac=None,
        fit_historical_distributions=False,
    )
    ts.add_calendar_features(calendar_df, "wday", use_day_of_week_features=False, use_week_end_feature=True)
    # Attach future promo_flag at t+1 and t+2 using negative lags:
    ts.add_feature(calendar_df, "promo_flag", join_on=[], lags=(-1, -2))

    # Verify the last observed day (day 60) received the future promo_flag from days 61 and 62:
    latest = ts.get_inference_data()
    assert len(latest) == 2
    assert (latest["day"] == 60).all()
    assert (latest["promo_flag"] == 0).all()
    assert (latest["promo_flag_lag_minus_1"] == 1).all()
    assert (latest["promo_flag_lag_minus_2"] == 1).all()

    # Fit CatBoostQuantileForecaster and compute DDLT on the production decision day (day 60):
    forecaster = CatBoostQuantileForecaster(
        quantiles=(0.1, 0.5, 0.9),
        extra_cat_cols=("item_id",),
        num_valid_days=5,
        catboost_params={"iterations": 15, "depth": 4, "verbose": False},
    )
    forecaster.fit(ts)
    lt_distr = {"S1": DiscreteDistribution(np.array([0.5, 0.5]), min_value=1)}
    ddlt_df = compute_lead_time_demand(
        ts,
        forecaster,
        lt_distr,
        lead_time_keys=["store_id"],
        cycle_time=1,
        return_distribution=False,
        pred_first_day=True,
    )
    assert len(ddlt_df) == 2
    assert (ddlt_df["target_inventory"] >= 0).all()
