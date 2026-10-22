import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discrete_distribution import DiscreteDistribution  # noqa: E402
from quantile_distribution import (  # noqa: E402
    DEFAULT_QUANTILES,
    distribution_mean,
    quantiles_to_distribution,
    quantiles_to_mean,
    rearrange_quantiles,
)

LEVELS = np.array(DEFAULT_QUANTILES)


def test_rearrange_fixes_crossing_and_negatives():
    q = rearrange_quantiles(np.array([[2.0, 1.0, -0.5, 3.0]]))
    assert np.all(np.diff(q, axis=-1) >= 0)
    assert q.min() >= 0


def test_all_zero_quantiles_is_point_mass_at_zero():
    d = quantiles_to_distribution(LEVELS, np.zeros(len(LEVELS)))
    assert d.min == 0 and np.allclose(d.pmf, [1.0])


def test_pmf_is_valid_and_respects_integer_quantile_bounds():
    # Discrete quantile semantics: q_tau = y  =>  F(y - 1) < tau <= F(y)
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
    # The reconstructed median should match the true median.
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


def test_invalid_levels_raise():
    with pytest.raises(ValueError):
        quantiles_to_distribution([0.5, 0.5], [1.0, 2.0])
    with pytest.raises(ValueError):
        quantiles_to_distribution([0.0, 0.5], [1.0, 2.0])


def test_no_future_target_leakage_across_train_test_boundary():
    import pandas as pd
    from time_series import DiscreteTimeSeries

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
    )
    train_rows = ts.df[ts.df["is_test"] == 0]
    test_rows = ts.df[ts.df["is_test"] == 1]

    # test_start_period is day 41 (days 41..50 are test, with demand 41..50).
    # No training row may contain a target value >= 41 in any horizon column.
    for h, col in enumerate(ts.target_cols):
        non_null_targets = train_rows[col].dropna()
        assert (non_null_targets < 41).all(), f"Leakage in {col}"
        if h > 0:
            leaked_days = list(range(41 - h, 41))
            assert train_rows.loc[train_rows["day"].isin(leaked_days), col].isna().all()

    # Test rows still keep their ground-truth future targets for evaluation.
    assert test_rows.loc[test_rows["day"] == 41, "target_t_plus_3"].item() == 44.0


