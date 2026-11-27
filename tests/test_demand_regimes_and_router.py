"""
Unit tests for P0 & P1 improvements from Long et al. (2023):
1. Post-first-sale Syntetos-Boylan-Croston (SBC) demand classification (`ADI`, `CV^2`, `demand_regime_name`).
2. Reciprocal `sample_weight` unbiasedness when `downsample_zeros_frac < 1` in GLM & CatBoost.
3. `RegimeRouterForecaster` training-row reduction, mixed-branch DDLT, evaluation, and persistence.
4. `CatBoostQuantileForecaster(velocity_tiers="demand_regime")` partitioning & reload.
5. Weighted Scaled Pinball Loss (`evaluate_wspl`) exact math & constant-tail omission.
6. Regime-aware inventory policies (`service_level` dict & `quantile_rounding="regime_adaptive"`).
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from probabilistic_modeling import (
    CatBoostQuantileForecaster,
    DiscreteDistribution,
    DiscreteTimeSeries,
    EmpiricalQuantileForecaster,
    GLMForecaster,
    ProbabilisticForecaster,
    REGIME_ADAPTIVE_ROUNDING,
    RegimeRouterForecaster,
    classify_demand_regimes,
    evaluate_ddlt_calibration,
    evaluate_forecast_pit,
    evaluate_quantile_wmape,
    evaluate_wspl,
    load_forecaster,
    resolve_target_inventory,
    simulate_inventory,
)
from probabilistic_modeling.models.catboost_quantile import quantile_col_name


def test_post_first_sale_sbc_classification_ignores_pre_launch_zeros():
    rng = np.random.default_rng(101)
    n_train = 85
    n_test = 5
    days = list(range(1, n_train + n_test + 1))
    records = []

    for d in days:
        # 1. Newly launched smooth SKU: 60 pre-launch zeros, then steady 9..11 every day!
        dem_launch_smooth = 0 if d <= 60 else int(rng.choice([9, 10, 11]))
        # 2. Erratic SKU: sells almost daily (~92%), right-skewed bursty sizes -> ADI < 1.32, CV^2 >= 0.49
        dem_erratic = int(rng.choice([1, 1, 1, 30])) if rng.random() < 0.92 else 0
        # 3. Intermittent SKU: sells ~15% of days, constant positive size 2 -> CV^2 == 0 < 0.49
        dem_intermittent = 2 if rng.random() < 0.15 else 0
        # 4. Lumpy SKU: sells ~25% of days, right-skewed bursty sizes -> ADI >= 1.32, CV^2 >= 0.49
        dem_lumpy = int(rng.choice([1, 1, 1, 30])) if rng.random() < 0.25 else 0
        # 5. Single fluke sale on last training day (day 85): must NOT be classified as smooth!
        dem_fluke = 3 if d == n_train else 0
        # 6. All-zero SKU across training window
        dem_zero = 0

        for item_id, dem in (
            ("SKU_launch_smooth", dem_launch_smooth),
            ("SKU_erratic", dem_erratic),
            ("SKU_intermittent", dem_intermittent),
            ("SKU_lumpy", dem_lumpy),
            ("SKU_fluke", dem_fluke),
            ("SKU_zero", dem_zero),
        ):
            records.append({"store_id": "S1", "cat_id": "FOODS", "item_id": item_id, "day": d, "demand": dem})

    sales_df = pd.DataFrame(records)
    # Keep trim_leading_zeros=False (the default) to verify post-first-sale classification still works!
    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "cat_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=2,
        num_test_days=n_test,
        trim_leading_zeros=False,
        downsample_zeros_frac=None,
        fit_historical_distributions=True,
    )

    meta = ts.series_metadata.set_index("item_id")
    # Newly launched smooth SKU: active days 61..85 (25 days, 25 sales -> ADI = 1.0, smooth, not overdispersed!)
    assert float(meta.loc["SKU_launch_smooth", "adi"]) == pytest.approx(1.0)
    assert int(meta.loc["SKU_launch_smooth", "demand_regime"]) == 0
    assert str(meta.loc["SKU_launch_smooth", "demand_regime_name"]) == "smooth"
    assert int(meta.loc["SKU_launch_smooth", "velocity_tier"]) == 2
    assert int(meta.loc["SKU_launch_smooth", "overdispersion"]) == 0

    assert str(meta.loc["SKU_erratic", "demand_regime_name"]) == "erratic"
    assert str(meta.loc["SKU_intermittent", "demand_regime_name"]) == "intermittent"
    assert str(meta.loc["SKU_lumpy", "demand_regime_name"]) == "lumpy"
    assert str(meta.loc["SKU_fluke", "demand_regime_name"]) == "intermittent"
    assert str(meta.loc["SKU_zero", "demand_regime_name"]) == "intermittent"
    assert np.isinf(float(meta.loc["SKU_zero", "adi"]))


def test_zero_downsampling_sample_weights_unbiased_for_glm_and_catboost():
    rng = np.random.default_rng(2028)
    # 12 sparse SKUs over 180 days with constant Bernoulli(0.08) * 2 demand -> mostly all-zero rows
    rows = []
    for i in range(12):
        for d in range(1, 181):
            dem = 2 if rng.random() < 0.08 else 0
            rows.append({"store_id": "S1", "item_id": f"I{i}", "day": d, "demand": dem})
    sales_df = pd.DataFrame(rows)

    ts_full = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=1,
        num_test_days=5,
        downsample_zeros_frac=None,
        fit_historical_distributions=False,
    )
    ts_down = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=1,
        num_test_days=5,
        downsample_zeros_frac=0.1,
        fit_historical_distributions=False,
    )

    assert "sample_weight" in ts_down.df.columns
    assert "sample_weight" not in ts_down.feature_cols
    assert set(np.round(ts_down.df["sample_weight"].unique(), 6)).issubset({1.0, 10.0})

    # 1. Hurdle GLM: without sample_weight, downsampling 90% of zero rows inflates pi ~8x!
    glm_full = GLMForecaster(train_groups=["store_id"], hurdle=True)
    glm_full.fit(ts_full)
    glm_down = GLMForecaster(train_groups=["store_id"], hurdle=True)
    glm_down.fit(ts_down)

    p_full = glm_full.predict(ts_full)
    p_down = glm_down.predict(ts_down)
    assert float(p_down["target_t_plus_0_pi"].mean()) == pytest.approx(
        float(p_full["target_t_plus_0_pi"].mean()), abs=0.025
    )
    assert float(p_down["target_t_plus_0_mu"].mean()) == pytest.approx(
        float(p_full["target_t_plus_0_mu"].mean()), abs=0.05
    )

    # 2. Standard base GLM:
    base_full = GLMForecaster(train_groups=["store_id"], hurdle=False)
    base_full.fit(ts_full)
    base_down = GLMForecaster(train_groups=["store_id"], hurdle=False)
    base_down.fit(ts_down)
    assert float(base_down.predict(ts_down)["target_t_plus_0_mu"].mean()) == pytest.approx(
        float(base_full.predict(ts_full)["target_t_plus_0_mu"].mean()), abs=0.05
    )

    # 3. CatBoost MultiQuantile: without Pool weights, q0.70 on 92%-zero data jumps from 0 to ~2!
    cb_params = {"iterations": 40, "depth": 4, "verbose": False}
    cb_full = CatBoostQuantileForecaster(quantiles=(0.5, 0.7, 0.95), num_valid_days=0, catboost_params=cb_params)
    cb_full.fit(ts_full)
    cb_down = CatBoostQuantileForecaster(quantiles=(0.5, 0.7, 0.95), num_valid_days=0, catboost_params=cb_params)
    cb_down.fit(ts_down)

    q70_full = cb_full.predict(ts_full)[quantile_col_name("target_t_plus_0", 0.7)].mean()
    q70_down = cb_down.predict(ts_down)[quantile_col_name("target_t_plus_0", 0.7)].mean()
    assert float(q70_down) == pytest.approx(float(q70_full), abs=0.25)


def test_regime_router_forecaster_end_to_end_and_training_row_reduction(tmp_path):
    # Check signature consistency with ProbabilisticForecaster:
    for method in ("predict", "compute_ddlt"):
        base_params = [p for p in inspect.signature(getattr(ProbabilisticForecaster, method)).parameters if p != "self"]
        sub_params = [p for p in inspect.signature(getattr(RegimeRouterForecaster, method)).parameters if p != "self"]
        assert sub_params[: len(base_params)] == base_params

    rng = np.random.default_rng(77)
    days = list(range(1, 96))
    rows = []
    # 2 active SKUs (1 smooth, 1 lumpy) + 8 intermittent SKUs (80% intermittent assortment!)
    for d in days:
        rows.append({"store_id": "S1", "cat_id": "FOODS", "item_id": "SMOOTH_1", "day": d, "demand": int(rng.poisson(8.0))})
        rows.append(
            {
                "store_id": "S1",
                "cat_id": "FOODS",
                "item_id": "LUMPY_1",
                "day": d,
                "demand": int(rng.choice([1, 1, 1, 25])) if rng.random() < 0.35 else 0,
            }
        )
        for k in range(8):
            rows.append(
                {
                    "store_id": "S1",
                    "cat_id": "HOBBIES",
                    "item_id": f"INT_{k}",
                    "day": d,
                    "demand": 1 if rng.random() < 0.06 else 0,
                }
            )
    sales_df = pd.DataFrame(rows)
    cal_df = pd.DataFrame({"day": list(range(1, 105)), "wday": [(d - 1) % 7 + 1 for d in range(1, 105)]})

    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "cat_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=3,
        num_test_days=6,
        use_one_hot_encoding=False,
        downsample_zeros_frac=0.2,
        fit_historical_distributions=True,
    )
    ts.add_calendar_features(cal_df, "wday", use_day_of_week_features=False, use_week_end_feature=True)

    quantiles = (0.1, 0.5, 0.9, 0.95)
    router = RegimeRouterForecaster(
        primary_forecaster=GLMForecaster(train_groups=["store_id"], hurdle=True, velocity_tiers="demand_regime"),
        intermittent_forecaster=EmpiricalQuantileForecaster(quantiles=quantiles, method="trailing", window=28),
        intermittent_regimes=("intermittent",),
        quantiles=quantiles,
    ).fit(ts)

    # Primary forecaster only trained on 2 of the 10 series!
    assert router.routing_summary_["n_series_total"] == 10
    assert router.routing_summary_["n_series_primary"] == 2
    assert router.routing_summary_["n_series_intermittent"] == 8
    total_train_rows = int((ts.df["is_test"] == 0).sum())
    assert router.routing_summary_["n_train_rows_primary"] < 0.5 * total_train_rows

    preds = router.predict(ts)
    orig_test = ts.get_inference_data()
    # Exact row order preserved:
    pd.testing.assert_series_equal(
        preds["item_id"].astype(str).reset_index(drop=True),
        orig_test["item_id"].astype(str).reset_index(drop=True),
    )
    for t in ts.target_cols:
        assert f"{t}_mu" in preds.columns
        for q in quantiles:
            assert quantile_col_name(t, q) in preds.columns
            assert preds[quantile_col_name(t, q)].notna().all()

    # Regime-stratified WMAPE, PIT, and WSPL all work seamlessly on hybrid predictions:
    wmape_df = evaluate_quantile_wmape(preds, ts.target_cols, quantiles=quantiles, group_cols=["demand_regime_name"])
    assert "demand_regime_name=smooth" in set(wmape_df["group"])
    assert "demand_regime_name=intermittent" in set(wmape_df["group"])

    pit_res = evaluate_forecast_pit(
        preds, ts.target_cols, levels=(0.5, 0.9), quantiles=quantiles, group_cols=["demand_regime_name"]
    )
    overall_pit = pit_res.summary_df[
        (pit_res.summary_df["group"] == "overall") & (pit_res.summary_df["target"] == "all_horizons")
    ].iloc[0]
    # All non-missing test observations across BOTH branches were evaluated (no intermittent rows dropped!):
    expected_n_obs = int(preds[ts.target_cols].notna().sum().sum())
    assert int(overall_pit["n_obs"]) == expected_n_obs

    wspl_df = evaluate_wspl(preds, ts, ts.target_cols, quantiles=quantiles)
    assert "demand_regime_name=smooth" in set(wspl_df["group"])
    assert np.isfinite(wspl_df[(wspl_df["group"] == "overall") & (wspl_df["quantile"] == "wspl")]["wspl"].iloc[0])

    # Mixed-branch DDLT + DDLT calibration:
    lt = {"S1": DiscreteDistribution(np.array([0.5, 0.5]), 1, 1)}
    ddlt_df = router.compute_ddlt(
        ts, lead_time=lt, lead_time_keys=["store_id"], cycle_time=1, return_distribution=True, pred_first_day=False
    )
    assert len(ddlt_df) == len(orig_test)
    assert all(isinstance(d, DiscreteDistribution) for d in ddlt_df["demand_distribution"])

    calib = evaluate_ddlt_calibration(
        ddlt_df,
        sales_df,
        lt,
        ts_id_cols=["store_id", "item_id"],
        lead_time_keys=["store_id"],
        cycle_time=1,
        group_cols=["demand_regime_name"],
    )
    assert "demand_regime_name=intermittent" in set(calib.summary_df["group"])

    # Persistence & reload via universal load_forecaster (including saved EmpiricalQuantile history!):
    bundle_dir = tmp_path / "router_bundle"
    router.save(bundle_dir)
    loaded = load_forecaster(bundle_dir)
    assert isinstance(loaded, RegimeRouterForecaster)
    preds_loaded = loaded.predict(ts)
    for t in ts.target_cols:
        np.testing.assert_allclose(preds[f"{t}_mu"], preds_loaded[f"{t}_mu"])


def test_catboost_velocity_tiers_demand_regime_and_persistence(tmp_path):
    rng = np.random.default_rng(55)
    rows = []
    for d in range(1, 75):
        rows.append({"store_id": "S1", "cat_id": "FOODS", "item_id": "I_SMOOTH", "day": d, "demand": int(rng.choice([5, 6, 7]))})
        rows.append({"store_id": "S1", "cat_id": "FOODS", "item_id": "I_INT", "day": d, "demand": 2 if rng.random() < 0.2 else 0})
    sales_df = pd.DataFrame(rows)

    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=["store_id", "cat_id", "item_id"],
        period_col="day",
        var_col="demand",
        lags=(1, 7),
        rolling_windows=(7,),
        num_future_targets=2,
        num_test_days=4,
        use_one_hot_encoding=False,
        downsample_zeros_frac=None,
        fit_historical_distributions=False,
    )

    cb = CatBoostQuantileForecaster(
        quantiles=(0.1, 0.5, 0.9),
        train_groups=["store_id"],
        num_valid_days=5,
        catboost_params={"iterations": 12, "depth": 3, "verbose": False},
        velocity_tiers="demand_regime",
    )
    cb_model = cb.fit(ts)
    assert "demand_regime" in cb_model.train_groups
    assert len(cb_model.models) == 2

    save_dir = tmp_path / "cb_regime"
    cb.save(save_dir)
    loaded = load_forecaster(save_dir)
    assert isinstance(loaded, CatBoostQuantileForecaster)
    assert loaded.velocity_tiers == "demand_regime"

    p1 = cb.predict(ts)
    p2 = loaded.predict(ts)
    np.testing.assert_allclose(p1["target_t_plus_0_mu"], p2["target_t_plus_0_mu"])


def test_evaluate_wspl_exact_hand_computed_and_constant_tail_omission():
    # 3 series over training days 1..6, evaluated at test day 7 (horizon = 1, quantiles = (0.1, 0.9)):
    # - S_prelaunch: [0, 0, 2, 4, 2, 4] -> post-first-sale [2, 4, 2, 4] -> |diffs| = [2, 2, 2] -> scale = 2.0, vol = 12
    # - S_regular  : [10, 20, 10, 20, 10, 20] -> |diffs| = [10]*5 -> scale = 10.0, vol = 90
    # - S_const_tail: [0, 0, 0, 5, 5, 5] -> post-first-sale [5, 5, 5] -> |diffs| = [0, 0] -> scale = 0 -> OMITTED!
    hist = pd.DataFrame(
        [
            *[
                {"store_id": "CA_1", "item_id": "S_prelaunch", "day": d, "demand": v}
                for d, v in enumerate([0, 0, 2, 4, 2, 4, 3], start=1)
            ],
            *[
                {"store_id": "CA_1", "item_id": "S_regular", "day": d, "demand": v}
                for d, v in enumerate([10, 20, 10, 20, 10, 20, 15], start=1)
            ],
            *[
                {"store_id": "CA_1", "item_id": "S_const_tail", "day": d, "demand": v}
                for d, v in enumerate([0, 0, 0, 5, 5, 5, 5], start=1)
            ],
        ]
    )

    pred_df = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "CA_1"],
            "item_id": ["S_prelaunch", "S_regular", "S_const_tail"],
            "day": [7, 7, 7],
            "target_t_plus_0": [3.0, 15.0, 5.0],
            "target_t_plus_0_q0.1": [1.0, 10.0, 4.0],
            "target_t_plus_0_q0.9": [5.0, 25.0, 6.0],
        }
    )
    # Hand calculation:
    # S_prelaunch (y=3, scale=2.0):
    #   q0.1=1 <= 3 -> PL = 0.1 * (3 - 1) = 0.2 -> SPL[0.1] = 0.2 / 2.0 = 0.10
    #   q0.9=5 > 3  -> PL = 0.1 * (5 - 3) = 0.2 -> SPL[0.9] = 0.2 / 2.0 = 0.10
    # S_regular (y=15, scale=10.0):
    #   q0.1=10 <= 15 -> PL = 0.1 * (15 - 10) = 0.5 -> SPL[0.1] = 0.5 / 10.0 = 0.05
    #   q0.9=25 > 15  -> PL = 0.1 * (25 - 15) = 1.0 -> SPL[0.9] = 1.0 / 10.0 = 0.10
    # Equal weights (w = [0.5, 0.5]):
    #   WSPL[0.1] = 0.5 * 0.10 + 0.5 * 0.05 = 0.075
    #   WSPL[0.9] = 0.5 * 0.10 + 0.5 * 0.10 = 0.100
    #   Overall WSPL = 0.5 * (0.075 + 0.100) = 0.0875
    res_eq = evaluate_wspl(
        pred_df,
        hist,
        target_cols=["target_t_plus_0"],
        quantiles=(0.1, 0.9),
        ts_id_cols=["store_id", "item_id"],
        weights="equal",
        stratify_by_regime=False,
    )
    ov = res_eq[res_eq["group"] == "overall"].set_index("quantile")
    assert int(ov.loc["wspl", "n_series_total"]) == 3
    assert int(ov.loc["wspl", "n_series_valid"]) == 2
    assert int(ov.loc["wspl", "n_series_omitted"]) == 1
    assert float(ov.loc["q0.1", "wspl"]) == pytest.approx(0.075)
    assert float(ov.loc["q0.9", "wspl"]) == pytest.approx(0.100)
    assert float(ov.loc["wspl", "wspl"]) == pytest.approx(0.0875)

    # Volume weights: vol = [12, 90] -> w = [12/102, 90/102]
    res_vol = evaluate_wspl(
        pred_df,
        hist,
        target_cols=["target_t_plus_0"],
        quantiles=(0.1, 0.9),
        ts_id_cols=["store_id", "item_id"],
        weights="volume",
        stratify_by_regime=False,
    )
    ov_vol = res_vol[res_vol["group"] == "overall"].set_index("quantile")
    expected_q01_vol = (12.0 / 102.0) * 0.10 + (90.0 / 102.0) * 0.05
    assert float(ov_vol.loc["q0.1", "wspl"]) == pytest.approx(expected_q01_vol)


def test_regime_aware_inventory_service_levels_and_adaptive_rounding():
    # Intermittent SKU with PMF [0.94, 0.06] on {0, 1}:
    # At service_level=0.95:
    # - ceiling quantile F^{-1}(0.95) = 1 (orders 1 unit even though 94% chance of 0 demand)
    # - nearest-integer rounded quantile: (0.95 - 0.94) = 0.01 < 0.5 * 0.06 = 0.03 -> 0 units!
    dist_sparse = DiscreteDistribution(np.array([0.94, 0.06]), 0, 1)
    dist_smooth = DiscreteDistribution(np.array([0.94, 0.06]), 10, 1)  # on {10, 11}

    df = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["SKU_INT", "SKU_SMOOTH"],
            "day": [1, 1],
            "demand_regime_name": ["intermittent", "smooth"],
            "target_t_plus_0": [0, 10],
            "demand_distribution": [dist_sparse, dist_smooth],
        }
    )

    assert REGIME_ADAPTIVE_ROUNDING["intermittent"] == "round"
    assert REGIME_ADAPTIVE_ROUNDING["smooth"] == "ceil"

    decided_ceil = resolve_target_inventory(df, policy="quantile", service_level=0.95, quantile_rounding="ceil")
    decided_adapt = resolve_target_inventory(
        df, policy="quantile", service_level=0.95, quantile_rounding="regime_adaptive"
    )
    assert list(decided_ceil["target_inventory"]) == [1, 11]
    assert list(decided_adapt["target_inventory"]) == [0, 11]

    # Per-regime service_level dictionary + simulation KPI breakdown by demand_regime_name:
    sl_by_regime = {"smooth": 0.99, "erratic": 0.95, "lumpy": 0.90, "intermittent": 0.80}
    sim_res = simulate_inventory(
        df,
        lead_time=1,
        cycle_time=1,
        policy="quantile",
        service_level=sl_by_regime,
        quantile_rounding="regime_adaptive",
        group_cols=["demand_regime_name"],
    )
    assert "demand_regime_name" in sim_res.item_summary_df.columns
    assert set(sim_res.group_summary_df["demand_regime_name"]) == {"intermittent", "smooth"}
    by_item = sim_res.item_summary_df.set_index("item_id")
    assert float(by_item.loc["SKU_INT", "avg_target_inventory"]) == 0.0
    assert float(by_item.loc["SKU_SMOOTH", "avg_target_inventory"]) == 11.0
