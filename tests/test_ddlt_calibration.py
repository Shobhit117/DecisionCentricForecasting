"""
Tests for the DDLT calibration report and the empirical-quantile baseline.

The calibration tests use synthetic data where the truth is known:
- correctly specified DDLT distributions must look calibrated;
- DDLT distributions built under a (wrong) day-to-day independence assumption for
  autocorrelated demand must be flagged as too narrow.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.evaluation import evaluate_ddlt_calibration
from probabilistic_modeling.models.baseline import EmpiricalQuantileForecaster
from probabilistic_modeling.models.catboost_quantile import quantile_col_name
from probabilistic_modeling.models.forecasters import load_forecaster

LEAD_TIME = DiscreteDistribution(np.array([0.3, 0.5, 0.2]), 2)  # L in {2, 3, 4}
CYCLE = 1                                                       # m = L + R in {3, 4, 5}
M_VALUES = np.array([3, 4, 5])
M_WEIGHTS = np.array([0.3, 0.5, 0.2])


def _mixture_of_sums(daily: DiscreteDistribution) -> DiscreteDistribution:
    """DDLT = sum_m P(m) * (daily convolved m times), assuming independent days."""
    sums, cur = {}, daily
    for m in range(1, M_VALUES.max() + 1):
        if m > 1:
            cur = cur + daily
        sums[m] = cur
    lo = min(sums[m].min for m in M_VALUES)
    hi = max(sums[m].max for m in M_VALUES)
    pmf = np.zeros(hi - lo + 1)
    for m, w in zip(M_VALUES, M_WEIGHTS):
        s = sums[m]
        pmf[s.min - lo : s.min - lo + len(s.pmf)] += w * s.pmf
    return DiscreteDistribution(pmf, lo)


def _panel(demand: np.ndarray, start_day: int = 1) -> pd.DataFrame:
    n_series, n_days = demand.shape
    return pd.DataFrame({
        "store_id": "S1",
        "item_id": np.repeat([f"I{i}" for i in range(n_series)], n_days),
        "day": np.tile(np.arange(start_day, start_day + n_days), n_series),
        "demand": demand.ravel(),
    })


def _decision_rows(n_series, decision_days, dists_by_series):
    rows = []
    for i in range(n_series):
        for d in decision_days:
            rows.append({"store_id": "S1", "item_id": f"I{i}", "cat_id": "A" if i % 2 else "B", "day": d,
                         "demand_distribution": dists_by_series[i]})
    return pd.DataFrame(rows)


def test_correct_ddlt_is_calibrated():
    rng = np.random.default_rng(0)
    n_series, n_days = 300, 120
    mus = rng.uniform(0.5, 8.0, n_series)
    demand = rng.poisson(mus[:, None], size=(n_series, n_days))
    dists = {i: _mixture_of_sums(DiscreteDistribution.from_parametric("poisson", (mus[i], 0), tol=1e-10))
             for i in range(n_series)}
    ddlt_df = _decision_rows(n_series, range(10, 110), dists)

    res = evaluate_ddlt_calibration(ddlt_df, _panel(demand), LEAD_TIME, ["store_id", "item_id"],
                                    cycle_time=CYCLE, group_cols=["cat_id"])
    overall = res.summary_df.iloc[0]
    assert overall["n_rows"] == len(ddlt_df) and res.n_dropped == 0
    assert overall["dispersion_ratio"] == pytest.approx(1.0, abs=0.05)
    assert abs(overall["bias_pct"]) < 1.5
    cov = res.coverage_df[res.coverage_df["group"] == "overall"]
    assert np.all(np.abs(cov["gap_pp"]) < 1.5), cov
    # raw coverage of a discrete forecast is >= the nominal level even when calibrated
    assert np.all(cov["coverage"].to_numpy() >= cov["level"].to_numpy() - 0.01)
    pit = res.pit_df[res.pit_df["group"] == "overall"]["density"].to_numpy()
    assert np.isclose(pit.sum() / len(pit), 1.0) and np.all(np.abs(pit - 1.0) < 0.12), pit
    assert set(res.summary_df["group"]) == {"overall", "cat_id=A", "cat_id=B"}
    assert "DDLT calibration" in res.report()


def test_independence_assumption_with_autocorrelated_demand_is_flagged():
    rng = np.random.default_rng(1)
    n_series, n_days, rho, sigma = 300, 160, 0.85, 0.6
    mus = rng.uniform(1.0, 6.0, n_series)
    # latent stationary AR(1) log-rate -> positively correlated daily demand
    z = np.empty((n_series, n_days))
    z[:, 0] = rng.normal(0, sigma, n_series)
    eps_sd = sigma * np.sqrt(1 - rho**2)
    for t in range(1, n_days):
        z[:, t] = rho * z[:, t - 1] + rng.normal(0, eps_sd, n_series)
    rate = mus[:, None] * np.exp(z - sigma**2 / 2)
    demand = rng.poisson(rate)

    # Correct *marginal* daily distribution (Poisson-lognormal, by Monte Carlo) but
    # independent days in the DDLT -> spread too small.
    dists = {}
    for i in range(n_series):
        draws = rng.poisson(mus[i] * np.exp(rng.normal(0, sigma, 200_000) - sigma**2 / 2))
        dists[i] = _mixture_of_sums(DiscreteDistribution.from_data(draws))
    ddlt_df = _decision_rows(n_series, range(10, 150), dists)

    res = evaluate_ddlt_calibration(ddlt_df, _panel(demand), LEAD_TIME, ["store_id", "item_id"], cycle_time=CYCLE)
    overall = res.summary_df.iloc[0]
    assert overall["dispersion_ratio"] > 1.5                 # realised spread >> predicted
    assert abs(overall["bias_pct"]) < 3.0                    # ... while the mean is right
    cov = res.coverage_df.set_index("level")["pit_coverage"]
    assert cov[0.95] < 0.93 and cov[0.99] < 0.975            # upper quantiles under-cover
    pit = res.pit_df["density"].to_numpy()
    assert pit[-1] > 1.3 and pit[0] > 1.1                    # U-shaped PIT


def test_pit_edges_and_dropped_rows():
    # Point-mass DDLT at 5: realised 0 -> PIT in first bin; realised far above -> last bin.
    d = DiscreteDistribution(np.array([1.0]), 5)
    demand = np.array([[0] * 10, [10] * 10])
    ddlt_df = pd.DataFrame({"store_id": "S1", "item_id": ["I0", "I1", "I1"], "day": [1, 1, 9],
                            "demand_distribution": [d, d, d]})
    res = evaluate_ddlt_calibration(ddlt_df, _panel(demand), LEAD_TIME, ["store_id", "item_id"], cycle_time=CYCLE)
    assert res.n_dropped == 1  # day 9 + up to 5 days runs past day 10
    rows = res.row_df
    assert rows.loc[0, "pit_0"] == pytest.approx(1.0) and rows.loc[1, "pit_9"] == pytest.approx(1.0)
    assert rows.loc[0, "cov_0.95"] == 1.0 and rows.loc[1, "cov_0.95"] == 0.0


def test_calibration_with_per_store_lead_times():
    lt = {"S1": DiscreteDistribution(np.array([1.0]), 2), "S2": DiscreteDistribution(np.array([1.0]), 4)}
    demand = np.ones((2, 20))
    hist = pd.DataFrame({"store_id": np.repeat(["S1", "S2"], 20), "item_id": "I0",
                         "day": np.tile(np.arange(1, 21), 2), "demand": demand.ravel()})
    point = lambda v: DiscreteDistribution(np.array([1.0]), v)  # noqa: E731
    ddlt_df = pd.DataFrame({"store_id": ["S1", "S2"], "item_id": "I0", "day": [3, 3],
                            "demand_distribution": [point(3), point(5)]})  # exact: (L + R) days x 1
    res = evaluate_ddlt_calibration(ddlt_df, hist, lt, ["store_id", "item_id"], lead_time_keys=["store_id"], cycle_time=1)
    np.testing.assert_allclose(res.row_df["realized_mean"], [3.0, 5.0])
    np.testing.assert_allclose(res.row_df["crps"], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Empirical-quantile baseline
# ---------------------------------------------------------------------------

def _baseline_panel(seed=0, n_series=3, n_days=80):
    rng = np.random.default_rng(seed)
    demand = rng.poisson(rng.uniform(1, 10, n_series)[:, None] * (1 + 0.5 * (np.arange(n_days) % 7 == 5)),
                         size=(n_series, n_days))
    return _panel(demand), demand


def test_trailing_baseline_matches_numpy_and_has_no_lookahead():
    hist, demand = _baseline_panel()
    fc = EmpiricalQuantileForecaster(quantiles=(0.1, 0.5, 0.9), window=28, horizon=3,
                                     ts_id_cols=["store_id", "item_id"]).fit(hist)
    rows = hist[hist["day"].isin([60, 61])]
    pred = fc.predict(df=rows)
    for _, row in pred.iterrows():
        i, t = int(row["item_id"][1:]), int(row["day"])
        window = demand[i, t - 1 - 28 : t - 1]  # days t-28 .. t-1 (column t-1 is day t)
        exp = np.quantile(window, [0.1, 0.5, 0.9])
        got = [row[quantile_col_name("target_t_plus_2", q)] for q in (0.1, 0.5, 0.9)]
        np.testing.assert_allclose(got, exp)
        assert row["target_t_plus_0_mu"] == pytest.approx(window.mean())

    # Changing demand on/after the latest decision day (61) must not change any forecast
    hist2 = hist.copy()
    hist2.loc[hist2["day"] >= 61, "demand"] += 100
    pred2 = EmpiricalQuantileForecaster(quantiles=(0.1, 0.5, 0.9), window=28, horizon=3,
                                        ts_id_cols=["store_id", "item_id"]).fit(hist2).predict(df=rows)
    q_cols = [c for c in pred.columns if "_q" in c or c.endswith("_mu")]
    pd.testing.assert_frame_equal(pred[q_cols], pred2[q_cols])


def test_seasonal_baseline_uses_same_weekday_strictly_in_the_past():
    hist, demand = _baseline_panel()
    fc = EmpiricalQuantileForecaster(quantiles=(0.25, 0.5), method="seasonal", n_weeks=4, horizon=7,
                                     ts_id_cols=["store_id", "item_id"])
    with pytest.raises(ValueError):
        EmpiricalQuantileForecaster(quantiles=(0.25, 0.5), method="bogus")
    fc.fit(hist)
    t = 61
    pred = fc.predict(df=hist[(hist["day"] == t) & (hist["item_id"] == "I0")])
    for h in range(7):
        k0 = h // 7 + 1
        days = [t + h - 7 * k for k in range(k0, k0 + 4)]
        assert max(days) <= t - 1
        exp = np.quantile(demand[0, np.array(days) - 1], 0.5)
        assert pred[quantile_col_name(f"target_t_plus_{h}", 0.5)].iloc[0] == pytest.approx(exp)


def test_baseline_ddlt_end_to_end_and_persistence(tmp_path):
    hist, _ = _baseline_panel(n_series=4)
    fc = EmpiricalQuantileForecaster(window=28, horizon=7, ts_id_cols=["store_id", "item_id"]).fit(hist)
    rows = hist[hist["day"].between(50, 60)]
    ddlt = fc.compute_ddlt(df=rows, lead_time={"S1": LEAD_TIME}, lead_time_keys=["store_id"], cycle_time=1,
                           return_distribution=True, pred_first_day=False)
    assert len(ddlt) == len(rows)
    assert all(isinstance(d, DiscreteDistribution) for d in ddlt["demand_distribution"])

    res = evaluate_ddlt_calibration(ddlt, hist, {"S1": LEAD_TIME}, ["store_id", "item_id"],
                                    lead_time_keys=["store_id"], cycle_time=1)
    assert res.summary_df.iloc[0]["n_rows"] == len(rows)

    fc.save(tmp_path / "emp")
    loaded = load_forecaster(tmp_path / "emp")
    assert isinstance(loaded, EmpiricalQuantileForecaster) and loaded.window == 28
    with pytest.raises(ValueError, match="needs a history"):
        loaded.predict(df=rows)
    pd.testing.assert_frame_equal(loaded.update_history(hist).predict(df=rows), fc.predict(df=rows))
