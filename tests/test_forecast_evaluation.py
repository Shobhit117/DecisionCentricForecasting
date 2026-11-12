"""Unit tests for multi-quantile WMAPE evaluation, discrete PIT distributions, and M5 sectioning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from probabilistic_modeling import (
    DEFAULT_QUANTILES,
    DiscreteDistribution,
    build_m5_sections,
    compute_pit_uniformity_metrics,
    evaluate_ddlt_calibration,
    evaluate_forecast_pit,
    evaluate_quantile_wmape,
    populate_model_quantiles,
)
from probabilistic_modeling.models.catboost_quantile import quantile_col_name


def test_build_m5_sections_default_and_rolling():
    # M5 default: days 1..1941, 365 train days + 14 test days, non-overlapping 365-day step
    sections = build_m5_sections(min_day=1, max_day=1941, train_days=365, test_days=14, step_days=365)
    assert len(sections) == 5
    assert sections[0].train_start_day == 1
    assert sections[0].train_end_day == 365
    assert sections[0].test_start_day == 366
    assert sections[0].test_end_day == 379
    assert sections[0].train_days == 365
    assert sections[0].test_days == 14
    assert sections[-1].test_end_day <= 1941

    # Rolling 180-day step with max_sections=3
    rolling = build_m5_sections(
        min_day=1, max_day=1941, train_days=365, test_days=14, step_days=180, max_sections=3
    )
    assert len(rolling) == 3
    assert rolling[1].train_start_day == 181
    assert rolling[2].train_start_day == 361


def test_populate_model_quantiles_glm_and_hurdle():
    quantiles = (0.1, 0.5, 0.9)
    target_cols = ["target_t_plus_0", "target_t_plus_1"]

    # Standard Poisson (alpha=NaN) and NB2 (alpha=0.5)
    df_glm = pd.DataFrame(
        {
            "target_t_plus_0": [3.0, 5.0],
            "target_t_plus_1": [4.0, 6.0],
            "target_t_plus_0_mu": [3.5, 6.0],
            "target_t_plus_0_alpha": [np.nan, 0.5],
            "target_t_plus_1_mu": [4.0, 8.0],
            "target_t_plus_1_alpha": [0.0, 0.25],
        }
    )
    out_glm = populate_model_quantiles(df_glm, target_cols, quantiles=quantiles)
    for t in target_cols:
        q_vals = [out_glm[quantile_col_name(t, q)].to_numpy() for q in quantiles]
        assert np.all(q_vals[0] <= q_vals[1])
        assert np.all(q_vals[1] <= q_vals[2])
    # Exact check for row 0, target_t_plus_0 (Poisson(3.5))
    for q in quantiles:
        assert out_glm.loc[0, quantile_col_name("target_t_plus_0", q)] == poisson.ppf(q, 3.5)

    # Zero-Hurdle GLM
    df_hurdle = pd.DataFrame(
        {
            "target_t_plus_0": [0.0, 4.0],
            "target_t_plus_1": [2.0, 5.0],
            "target_t_plus_0_pi": [0.05, 0.85],
            "target_t_plus_0_lambda": [2.0, 5.0],
            "target_t_plus_0_mu": [0.1, 4.3],
            "target_t_plus_0_alpha": [np.nan, 0.4],
            "target_t_plus_1_pi": [0.5, 0.9],
            "target_t_plus_1_lambda": [3.0, 6.0],
            "target_t_plus_1_mu": [1.6, 5.5],
            "target_t_plus_1_alpha": [np.nan, 0.3],
        }
    )
    out_hurdle = populate_model_quantiles(df_hurdle, target_cols, quantiles=quantiles)
    # Row 0 has pi=0.05 -> 95% zero mass, so q0.1, q0.5, q0.9 must all be 0
    for q in quantiles:
        assert out_hurdle.loc[0, quantile_col_name("target_t_plus_0", q)] == 0.0
    assert out_hurdle.loc[1, quantile_col_name("target_t_plus_0", 0.9)] >= out_hurdle.loc[
        1, quantile_col_name("target_t_plus_0", 0.5)
    ]


def test_evaluate_quantile_wmape_identities():
    rng = np.random.default_rng(42)
    n = 200
    quantiles = (0.1, 0.5, 0.9)
    target_cols = ["target_t_plus_0", "target_t_plus_1"]
    pred_df = pd.DataFrame(
        {
            "cat_id": rng.choice(["FOODS", "HOBBIES"], size=n),
            "target_t_plus_0": rng.poisson(4.0, size=n).astype(float),
            "target_t_plus_1": rng.poisson(5.0, size=n).astype(float),
            "target_t_plus_0_mu": np.full(n, 4.0),
            "target_t_plus_0_alpha": np.full(n, np.nan),
            "target_t_plus_1_mu": np.full(n, 5.0),
            "target_t_plus_1_alpha": np.full(n, np.nan),
        }
    )

    res = evaluate_quantile_wmape(
        pred_df,
        target_cols=target_cols,
        quantiles=quantiles,
        group_cols=["cat_id"],
        include_mean=True,
        include_horizon_sum=True,
    )

    assert set(res["group"].unique()) == {"overall", "cat_id=FOODS", "cat_id=HOBBIES"}
    assert set(res["target"].unique()) == {"target_t_plus_0", "target_t_plus_1", "all_horizons", "horizon_sum"}
    assert set(res["quantile"].unique()) == {"q0.1", "q0.5", "q0.9", "mean"}

    # Identity checks: abs_diff = under_diff + over_diff, wmape = under_wmape + over_wmape
    assert np.allclose(res["abs_diff"], res["under_diff"] + res["over_diff"])
    assert np.allclose(res["wmape"], res["under_wmape"] + res["over_wmape"])

    # At median q0.5, pinball_wmape == wmape
    q50 = res[res["quantile"] == "q0.5"]
    assert np.allclose(q50["pinball_wmape"], q50["wmape"])

    # Coverage increases monotonically with quantile level
    overall_all = res[(res["group"] == "overall") & (res["target"] == "all_horizons") & (res["quantile"] != "mean")]
    covs = overall_all.sort_values("quantile_level")["coverage"].to_numpy()
    assert np.all(np.diff(covs) >= 0.0)


def test_evaluate_forecast_pit_calibrated_across_model_families():
    rng = np.random.default_rng(123)
    n = 3000
    mu = 5.0
    alpha = 0.4
    r_nb = 1.0 / alpha
    p_nb = r_nb / (r_nb + mu)
    y0 = rng.negative_binomial(r_nb, p_nb, size=n).astype(float)
    y1 = rng.negative_binomial(r_nb, p_nb, size=n).astype(float)
    target_cols = ["target_t_plus_0", "target_t_plus_1"]
    levels = (0.5, 0.8, 0.9, 0.95)

    # 1. Standard GLM representation
    df_glm = pd.DataFrame(
        {
            "cat_id": np.where(np.arange(n) < n // 2, "FOODS", "HOUSEHOLD"),
            "target_t_plus_0": y0,
            "target_t_plus_1": y1,
            "target_t_plus_0_mu": np.full(n, mu),
            "target_t_plus_0_alpha": np.full(n, alpha),
            "target_t_plus_1_mu": np.full(n, mu),
            "target_t_plus_1_alpha": np.full(n, alpha),
        }
    )
    pit_glm = evaluate_forecast_pit(df_glm, target_cols, levels=levels, group_cols=["cat_id"])
    overall = pit_glm.summary_df[
        (pit_glm.summary_df["group"] == "overall") & (pit_glm.summary_df["target"] == "all_horizons")
    ].iloc[0]
    assert abs(overall["pit_mean"] - 0.5) < 0.02
    assert abs(overall["pit_std"] - (1.0 / np.sqrt(12.0))) < 0.02
    assert overall["pit_mae"] < 0.08
    assert overall["pit_ks"] < 0.03

    cov_overall = pit_glm.coverage_df[
        (pit_glm.coverage_df["group"] == "overall") & (pit_glm.coverage_df["target"] == "all_horizons")
    ]
    assert np.all(np.abs(cov_overall["gap_pp"]) < 2.5)

    # 2. Quantile-grid representation (CatBoost / Empirical style)
    df_q = populate_model_quantiles(df_glm.drop(columns=["target_t_plus_0_alpha", "target_t_plus_1_alpha"]), target_cols)
    # Populate with true NB2 quantiles
    for t in target_cols:
        for q in DEFAULT_QUANTILES:
            df_q[quantile_col_name(t, float(q))] = float(nbinom.ppf(q, r_nb, p_nb))
    pit_q = evaluate_forecast_pit(df_q, target_cols, levels=levels)
    overall_q = pit_q.summary_df[
        (pit_q.summary_df["group"] == "overall") & (pit_q.summary_df["target"] == "all_horizons")
    ].iloc[0]
    assert abs(overall_q["pit_mean"] - 0.5) < 0.03
    assert "Forecast PIT Evaluation" in pit_q.report()


def test_ddlt_calibration_includes_pit_uniformity_and_quantile_wmape():
    rng = np.random.default_rng(7)
    lam = 4.0
    pmf = poisson.pmf(np.arange(35), lam)
    pmf /= pmf.sum()
    dist = DiscreteDistribution(0, len(pmf) - 1, 1, pmf)

    n_skus, n_days = 40, 20
    history = pd.DataFrame(
        {
            "item_id": np.repeat(np.arange(n_skus), n_days),
            "day": np.tile(np.arange(1, n_days + 1), n_skus),
            "demand": rng.poisson(lam, size=n_skus * n_days),
        }
    )
    ddlt_df = history[history["day"] <= n_days - 2][["item_id", "day"]].copy()
    ddlt_df["demand_distribution"] = [dist] * len(ddlt_df)
    lt = DiscreteDistribution(0, 0, 1, np.array([1.0]))

    res = evaluate_ddlt_calibration(
        ddlt_df,
        history,
        lead_time=lt,
        ts_id_cols=["item_id"],
        cycle_time=1,
        levels=(0.5, 0.9),
    )
    for col in ("pit_mean", "pit_std", "pit_mae", "pit_rmse", "pit_ks"):
        assert col in res.summary_df.columns
        assert np.isfinite(res.summary_df[col].iloc[0])

    for col in ("actual", "pred", "under_diff", "over_diff", "abs_diff", "wmape", "under_wmape", "over_wmape", "pinball_loss", "pinball_wmape"):
        assert col in res.coverage_df.columns
        assert np.all(np.isfinite(res.coverage_df[col]))

    q50_row = res.coverage_df[res.coverage_df["level"] == 0.5].iloc[0]
    assert np.isclose(q50_row["wmape"], q50_row["pinball_wmape"])
    assert np.isclose(q50_row["wmape"], q50_row["under_wmape"] + q50_row["over_wmape"])
