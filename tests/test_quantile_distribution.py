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


def test_setup_logging_writes_log_file(tmp_path):
    import logging
    from logging_utils import setup_logging

    log_path = tmp_path / "pipeline_run.log"
    setup_logging(log_file=log_path, level="INFO")
    test_logger = logging.getLogger("test_logger")
    test_logger.info("Progress check: step %d/%d", 1, 3)

    for h in logging.getLogger().handlers:
        h.flush()

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "Progress check: step 1/3" in content
    assert "test_logger" in content


def test_continuous_and_discrete_predictive_distribution_interface():
    from discrete_distribution import PredictiveDistribution
    from quantile_distribution import ContinuousQuantileDistribution, quantile_crps

    # 1. DiscreteDistribution implements PredictiveDistribution, moments, cdf_at, crps, sample
    d = DiscreteDistribution.from_parametric("poisson", (5.0, 0))
    assert isinstance(d, PredictiveDistribution)
    assert np.isclose(d.mean(), 5.0, rtol=1e-2)
    assert np.isclose(d.variance(), 5.0, rtol=1e-2)
    assert np.isclose(d.std(), np.sqrt(5.0), rtol=1e-2)
    assert np.allclose(d.cdf_at([d.min - 1, d.max + 10]), [0.0, 1.0])
    crps_scalar = d.crps(5.0)
    crps_vec = d.crps([3.0, 5.0, 8.0])
    assert crps_scalar > 0.0 and crps_vec.shape == (3,)
    assert crps_vec[1] < crps_vec[2]  # CRPS is minimized near the median/mean
    samples = d.sample(size=200, rng=42)
    assert samples.shape == (200,) and (samples >= 0).all()

    # 2. ContinuousQuantileDistribution supports both non-negative and unbounded (negative) domains
    norm_dist = stats.norm(loc=-2.5, scale=4.0)
    q_vals = norm_dist.ppf(LEVELS)
    cont = ContinuousQuantileDistribution(LEVELS, q_vals, lower_bound=None)
    assert isinstance(cont, PredictiveDistribution)
    assert cont.quantile(0.1)[0] < 0.0  # Preserves negative values when lower_bound=None
    assert np.isclose(cont.mean(), -2.5, atol=0.25)
    assert np.isclose(cont.std(), 4.0, atol=0.5)
    assert np.isclose(cont.cdf_at(-2.5)[0], 0.5, atol=0.05)
    assert cont.crps(-2.5) < cont.crps(10.0)
    assert quantile_crps(np.array([-2.5, 0.0]), np.vstack([q_vals, q_vals]), LEVELS).shape == (2,)


def test_fft_convolution_matches_direct_numba_convolution():
    from ddlt import DEFAULT_TAIL_TOL, _direct_convolve_1d_numba, _fft_convolve_1d_numba, _trim_pmf_numba

    # Create two 600-point distributions so len(a)*len(b) = 360,000 > FFT_CONVOLVE_THRESHOLD (262,144)
    p = np.linspace(1.0, 3.0, 600)
    p /= p.sum()
    d1 = DiscreteDistribution(p, min_value=5)
    d2 = DiscreteDistribution(p[::-1], min_value=10)
    conv_fft = d1 + d2

    raw_direct = _direct_convolve_1d_numba(d1.pmf, d2.pmf)
    raw_fft = _fft_convolve_1d_numba(d1.pmf, d2.pmf)
    assert np.allclose(raw_fft, raw_direct, atol=1e-11)

    raw_direct /= raw_direct.sum()
    expected_pmf, expected_min = _trim_pmf_numba(raw_direct, 15, 1, DEFAULT_TAIL_TOL)
    assert conv_fft.min == expected_min
    assert np.allclose(conv_fft.pmf, expected_pmf, atol=1e-10)


def test_trim_leading_zeros_and_horizon_restricted_leads():
    # item_late launches on day 11 (days 1..10 are 0); item_early is active from day 1
    days = list(range(1, 26))
    rows = []
    for d in days:
        rows.append({"id": "item_early", "day": d, "demand": 3})
        rows.append({"id": "item_late", "day": d, "demand": 0 if d <= 10 else 4})
    sales_df = pd.DataFrame(rows)
    cal_df = pd.DataFrame({"day": list(range(1, 30)), "promo": [1 if d % 5 == 0 else 0 for d in range(1, 30)]})

    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["id"],
        period_col="day",
        var_col="demand",
        lags=(1, 2),
        rolling_windows=(2,),
        num_future_targets=3,
        num_test_days=0,
        downsample_zeros_frac=None,
        fit_historical_distributions=False,
        trim_leading_zeros=True,
    )
    ts.add_feature(cal_df, "promo", join_on=[], lags=(-1, -2))

    # item_late's first active period is day 11; after lag(1, 2), its first engineered row is day 13:
    late_days = ts.df.loc[ts.df["id"] == "item_late", "day"].astype(int)
    assert late_days.min() == 13

    # Horizon feature restriction excludes future leads beyond h:
    f_h0 = ts.get_features_for_horizon("target_t_plus_0", restrict_future_leads=True)
    f_h1 = ts.get_features_for_horizon("target_t_plus_1", restrict_future_leads=True)
    f_h2 = ts.get_features_for_horizon("target_t_plus_2", restrict_future_leads=True)
    assert "promo_lag_minus_1" not in f_h0 and "promo_lag_minus_2" not in f_h0
    assert "promo_lag_minus_1" in f_h1 and "promo_lag_minus_2" not in f_h1
    assert "promo_lag_minus_1" in f_h2 and "promo_lag_minus_2" in f_h2


def test_reusable_forecaster_save_load_and_multi_prediction(tmp_path):
    """
    Verifies that both GLMForecaster and CatBoostQuantileForecaster can be:
    1. Trained once;
    2. Saved to a self-contained directory bundle and loaded via `load_forecaster`;
    3. Used repeatedly to predict on different periods, standalone DataFrames (without `ts`),
       and brand-new raw sales + covariate DataFrames (`predict_from_raw`), as well as DDLT
       with fallback distributions when L + R > horizon.
    """
    from forecasters import evaluate_quantile_forecasts, load_forecaster

    rng = np.random.default_rng(123)
    days = list(range(1, 51))
    records = []
    # item_A: low-variance (Poisson-like); item_B: overdispersed (NB-like)
    for d in days:
        records.append({
            "store_id": "S1",
            "cat_id": "FOODS",
            "item_id": "item_A",
            "day": d,
            "demand": int(rng.poisson(4.0)),
        })
        records.append({
            "store_id": "S1",
            "cat_id": "FOODS",
            "item_id": "item_B",
            "day": d,
            "demand": int(rng.negative_binomial(2, 0.25)),
        })
    sales_df = pd.DataFrame(records)
    calendar_df = pd.DataFrame({
        "day": list(range(1, 65)),
        "wday": [(d - 1) % 7 + 1 for d in range(1, 65)],
        "promo": [1 if d % 6 == 0 else 0 for d in range(1, 65)],
    })

    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "cat_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=2,
        num_test_days=5,
        use_one_hot_encoding=False,
        downsample_zeros_frac=None,
        fit_historical_distributions=True,
    )
    ts.add_calendar_features(calendar_df, "wday", use_day_of_week_features=False, use_week_end_feature=True)
    ts.add_feature(calendar_df, "promo", join_on=[], lags=(-1,))

    # Lead time + cycle time max = 3 > horizon (2), exercising fallback best_fit from series_metadata:
    lt_distr = {"S1": DiscreteDistribution(np.array([0.5, 0.5]), min_value=2)}

    # --- A. Test GLMForecaster save / load / reusable multi-prediction ---
    glm = GLMForecaster(train_groups=["store_id", "cat_id"], restrict_future_leads=True)
    glm.fit(ts)
    assert glm.model is not None
    coef_table = glm.model.coefficients_table()
    assert "alpha" not in coef_table["feature"].values  # alpha is cleanly separated from regression covariates

    glm_dir = tmp_path / "glm_bundle"
    glm.save(glm_dir)
    loaded_glm = load_forecaster(glm_dir)
    assert isinstance(loaded_glm, GLMForecaster)

    # 1) Predict on specific single period and multiple periods:
    pred_day46 = loaded_glm.predict(ts, period=46)
    pred_days_46_47 = loaded_glm.predict(ts, period=[46, 47])
    assert len(pred_day46) == 2 and (pred_day46["day"] == 46).all()
    assert len(pred_days_46_47) == 4

    # 2) Predict on a standalone DataFrame stripped of training metadata (overdispersion, best_fit) without `ts`:
    bare_df = ts.get_inference_data(period=48).drop(
        columns=["overdispersion", "best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
    )
    pred_bare = loaded_glm.predict(bare_df)
    assert len(pred_bare) == 2 and pred_bare["target_t_plus_0_mu"].notna().all()

    # 3) Compute DDLT directly from bare_df without `ts` (uses persisted series_metadata_ for best_fit fallback):
    ddlt_glm = loaded_glm.compute_ddlt(bare_df, lead_time=lt_distr, lead_time_keys=["store_id"], cycle_time=1)
    assert len(ddlt_glm) == 2 and (ddlt_glm["target_inventory"] >= 0).all()

    # --- B. Test CatBoostQuantileForecaster save / load / predict_from_raw ---
    cb = CatBoostQuantileForecaster(
        quantiles=(0.1, 0.5, 0.9),
        extra_cat_cols=("item_id", "cat_id"),
        num_valid_days=5,
        catboost_params={"iterations": 12, "depth": 4, "verbose": False},
        restrict_future_leads=True,
    )
    cb.fit(ts)
    cb_dir = tmp_path / "cb_bundle"
    cb.save(cb_dir)

    loaded_cb = load_forecaster(cb_dir)
    assert isinstance(loaded_cb, CatBoostQuantileForecaster)

    # Evaluate quantile forecasts (including new CRPS column):
    cb_test_preds = loaded_cb.predict(ts)
    q_eval = evaluate_quantile_forecasts(cb_test_preds, loaded_cb.target_cols_, loaded_cb.quantiles)
    assert "crps" in q_eval.columns and (q_eval["crps"] >= 0).all()

    # 4) Reusable inference on brand-new future days (days 35..55) from raw tables without `ts`:
    new_records = [r for r in records if r["day"] >= 35]
    for d in range(51, 56):
        for item_id in ("item_A", "item_B"):
            new_records.append({
                "store_id": "S1",
                "cat_id": "FOODS",
                "item_id": item_id,
                "day": d,
                "demand": 5,
            })
    new_sales_df = pd.DataFrame(new_records)

    raw_preds_day55 = loaded_cb.predict_from_raw(new_sales_df, calendar_df=calendar_df, period=55)
    assert len(raw_preds_day55) == 2
    assert (raw_preds_day55["day"] == 55).all()
    assert raw_preds_day55["target_t_plus_1_mu"].notna().all()

    # And DDLT on the transformed new raw data without `ts`:
    transformed_day55 = loaded_cb.transform(new_sales_df, calendar_df=calendar_df, period=55)
    ddlt_cb = loaded_cb.compute_ddlt(transformed_day55, lead_time=lt_distr, lead_time_keys=["store_id"], cycle_time=1)
    assert len(ddlt_cb) == 2 and (ddlt_cb["target_inventory"] >= 0).all()


def test_inventory_simulation_quantile_newsvendor_and_mc():
    """Verify multi-period inventory simulation with quantile policy, newsvendor policy, and Monte Carlo."""
    from inventory_simulation import InventorySimulator, simulate_inventory

    rng = np.random.default_rng(123)
    rows = []
    for day in range(1, 15):
        for item_idx, (item_id, cat_id, mu, price, cost, shortage) in enumerate(
            [
                ("item_1", "FOODS", 4.0, 5.0, 2.0, 1.0),
                ("item_2", "FOODS", 6.0, 8.0, 3.0, 2.0),
                ("item_3", "HOUSEHOLD", 3.0, 12.0, 5.0, 3.0),
            ]
        ):
            # DDLT distribution over ~2-3 days of demand
            ddlt_dist = DiscreteDistribution.from_parametric("poisson", (mu * 2.5, 0))
            actual_demand = int(rng.poisson(mu))
            rows.append(
                {
                    "store_id": "CA_1",
                    "cat_id": cat_id,
                    "item_id": item_id,
                    "day": day,
                    "target_t_plus_0": actual_demand,
                    "demand_distribution": ddlt_dist,
                    "price": price,
                    "cost": cost,
                    "shortage_cost": shortage,
                }
            )
    panel_df = pd.DataFrame(rows)
    lead_time_dist = {"CA_1": DiscreteDistribution(pmf=np.array([0.3, 0.5, 0.2]), min_value=1)}

    # 1. Policy = "quantile" (optional Newsvendor bypassed; uses simple DDLT quantile)
    sim_q = simulate_inventory(
        panel_df.drop(columns=["price", "cost", "shortage_cost"]),
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        demand_col="target_t_plus_0",
        policy="quantile",
        service_level=0.95,
        lead_time=lead_time_dist,
        lead_time_keys=["store_id"],
        group_cols=["store_id", "cat_id"],
        n_mc_simulations=25,
        random_state=42,
    )
    traj_q = sim_q.trajectory_df
    assert len(traj_q) == len(panel_df)

    # Check exact physical conservation identities on every row
    np.testing.assert_allclose(
        traj_q["on_hand_before_demand"],
        traj_q["opening_inv"] + traj_q["arrivals"],
    )
    np.testing.assert_allclose(
        traj_q["sold"] + traj_q["lost_sales"],
        traj_q["demand"],
    )
    np.testing.assert_allclose(
        traj_q["closing_inv"],
        traj_q["on_hand_before_demand"] - traj_q["sold"],
    )
    np.testing.assert_allclose(
        traj_q["order_placed"],
        np.maximum(0.0, traj_q["target_inventory"] - traj_q["inventory_position"]),
    )
    np.testing.assert_allclose(
        traj_q["in_transit_end"],
        traj_q["in_transit_before_order"] + traj_q["order_placed"],
    )

    # Check multi-day carry-over per item: opening_inv[t] == closing_inv[t-1]
    for _, grp in traj_q.sort_values("day").groupby(["store_id", "item_id"]):
        np.testing.assert_allclose(
            grp["opening_inv"].to_numpy()[1:],
            grp["closing_inv"].to_numpy()[:-1],
        )

    # Check KPIs and Monte Carlo summary
    assert 0.0 <= float(sim_q.summary_df["fill_rate"].iloc[0]) <= 1.0
    assert 0.0 <= float(sim_q.summary_df["cycle_service_level"].iloc[0]) <= 1.0
    assert len(sim_q.item_summary_df) == 3
    assert sim_q.group_summary_df is not None and len(sim_q.group_summary_df) == 2
    assert sim_q.mc_summary_df is not None and "fill_rate" in set(sim_q.mc_summary_df["metric"])

    # 2. Policy = "newsvendor" with economic KPIs
    sim_nv = simulate_inventory(
        panel_df,
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        demand_col="target_t_plus_0",
        policy="newsvendor",
        budget=60.0,
        lead_time=2,
        holding_cost_per_unit=0.10,
        group_cols=["store_id", "cat_id"],
        n_mc_simulations=10,
        random_state=7,
    )
    assert "net_profit" in sim_nv.trajectory_df.columns
    assert "total_profit" in sim_nv.summary_df.columns
    assert "total_holding_cost" in sim_nv.summary_df.columns
    assert sim_nv.mc_summary_df is not None and set(sim_nv.mc_summary_df.columns) >= {"metric", "mean", "std", "p05", "p95"}

    # 3. Backlog mode (allow_backlog=True) vs Lost-Sales mode from zero initial stock
    sim_ls_zero = InventorySimulator(
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        demand_col="target_t_plus_0",
        policy="quantile",
        service_level=0.95,
        lead_time=2,
        initial_inventory=0,
        allow_backlog=False,
    ).run(panel_df)
    sim_bl_zero = InventorySimulator(
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        demand_col="target_t_plus_0",
        policy="quantile",
        service_level=0.95,
        lead_time=2,
        initial_inventory=0,
        allow_backlog=True,
    ).run(panel_df)
    # In backlog mode, unfulfilled demand from days 1-2 is satisfied once replenishment arrives:
    assert float(sim_bl_zero.summary_df["total_sold"].iloc[0]) > float(sim_ls_zero.summary_df["total_sold"].iloc[0])



