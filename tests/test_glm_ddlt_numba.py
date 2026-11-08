"""
Tests for the compiled GLM -> DDLT path and the API / mutability quick fixes.

Benchmarks (skipped by default):

    python -m pytest tests/test_glm_ddlt_numba.py -m benchmark -s -q
"""
import inspect
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from probabilistic_modeling.ddlt import (
    DEFAULT_SUPPORT_TOL,
    DDLTCalculator,
    _count_pmf_truncated_numba,
    _hurdle_count_pmf_truncated_numba,
)
from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.models.forecasters import (
    CatBoostQuantileForecaster,
    GLMForecaster,
    ProbabilisticForecaster,
    _hurdle_conditional_mean,
    load_forecaster,
)
from probabilistic_modeling.time_series import DiscreteTimeSeries, predict_glm, train_glm

SL_GRID = (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


# ---------------------------------------------------------------------------
# 1. Truncated Poisson / NB2 PMF kernel vs. DiscreteDistribution.from_parametric
# ---------------------------------------------------------------------------

def _scipy_reference(mu, alpha, tol):
    if not mu > 0:
        return DiscreteDistribution(np.ones(1), 0)
    if np.isnan(alpha) or alpha <= 0:
        return DiscreteDistribution.from_parametric("poisson", (float(mu), 0), tol=tol)
    return DiscreteDistribution.from_parametric("nbinom", (1.0 / alpha, 1.0 / (1.0 + mu * alpha), 0), tol=tol)


@pytest.mark.parametrize("mu", [0.0, 1e-6, 0.3, 2.0, 15.0, 120.0, 2_500.0])
@pytest.mark.parametrize("alpha", [np.nan, 0.0, 0.05, 0.5, 3.0])
@pytest.mark.parametrize("tol", [1e-4, 1e-7])
def test_count_pmf_matches_scipy_from_parametric(mu, alpha, tol):
    got = _count_pmf_truncated_numba(mu, alpha, tol)
    ref = _scipy_reference(mu, alpha, tol)

    assert got.min() >= 0.0 and np.isclose(got.sum(), 1.0)
    # Support end = smallest k with F(k) >= 1 - tol: identical up to a possible 1-bin
    # floating-point tie at the cut-off.
    assert abs(len(got) - len(ref.pmf)) <= 1
    n = min(len(got), len(ref.pmf)) - 1  # compare everything below the lumped last bin
    np.testing.assert_allclose(got[:n], ref.pmf[:n], rtol=1e-8, atol=1e-14)
    # Quantiles are exact below the truncation point
    d = DiscreteDistribution(got, 0)
    levels = np.array([q for q in (0.01, 0.1, 0.5, 0.9, 0.99, 1 - 10 * tol) if q < 1 - 10 * tol or q == 1 - 10 * tol])
    np.testing.assert_array_equal(d.quantile(levels), ref.quantile(levels))


def test_count_pmf_large_mean_does_not_underflow():
    for mu, alpha in [(1e4, np.nan), (1e5, np.nan), (5e3, 0.01)]:
        pmf = _count_pmf_truncated_numba(mu, alpha, 1e-6)
        mean = float(np.dot(np.arange(len(pmf)), pmf))
        assert mean == pytest.approx(mu, rel=1e-3)


# ---------------------------------------------------------------------------
# 2. from_glm_group: compiled kernel vs. the previous scipy + random_sum implementation
# ---------------------------------------------------------------------------

def _reference_glm_ddlt(lt_with_cycle, mus, alphas, default, tol, tail_tol):
    """Pre-change implementation: from_parametric per day + DiscreteDistribution.random_sum."""
    demand = [_scipy_reference(m, a, tol) for m, a in zip(mus, alphas)]
    return lt_with_cycle.random_sum(demand, default, tail_tol=tail_tol)


def _glm_frame(n_rows, horizon, seed, with_fallback=False):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        row = {"store_id": "S1", "item_id": f"I{i}", "day": 1, "service_level": float(rng.choice(SL_GRID))}
        for h in range(horizon):
            row[f"target_t_plus_{h}_mu"] = float(rng.gamma(2.0, 4.0))
            row[f"target_t_plus_{h}_alpha"] = np.nan if i % 3 == 0 else float(rng.uniform(0.01, 1.5))
        if with_fallback:
            if i % 2 == 0:
                row.update(best_fit="poisson", distr_param_0=float(rng.uniform(1, 10)), distr_param_1=0.0,
                           distr_param_2=np.nan)
            else:
                row.update(best_fit="nbinom", distr_param_0=float(rng.uniform(1, 5)),
                           distr_param_1=float(rng.uniform(0.2, 0.8)), distr_param_2=0.0)
        rows.append(row)
    return pd.DataFrame(rows), [f"target_t_plus_{h}" for h in range(horizon)]


@pytest.mark.parametrize("with_fallback", [False, True])
def test_glm_group_numba_matches_reference(with_fallback):
    horizon = 4
    lead_time = DiscreteDistribution(np.array([0.2, 0.5, 0.3]), 2)  # L in {2,3,4}
    cycle_time = 2 if with_fallback else 0  # L + R up to 6 > horizon -> fallback distribution
    gdf, target_cols = _glm_frame(60, horizon, seed=3, with_fallback=with_fallback)

    calc = DDLTCalculator(lead_time, cycle_time=cycle_time)
    dist_df = calc.from_glm_group(gdf, lead_time, target_cols, return_distribution=True)
    tgt_df = calc.from_glm_group(gdf, lead_time, target_cols, return_distribution=False)

    lt_with_cycle = calc._shifted_lead_time(lead_time)
    tol = calc.effective_tol(max(int(lt_with_cycle.max), 1), None)
    tol_t = calc.effective_tol(max(int(lt_with_cycle.max), 1), gdf["service_level"].to_numpy())
    mus = gdf[[f"{t}_mu" for t in target_cols]].to_numpy()
    alphas = gdf[[f"{t}_alpha" for t in target_cols]].to_numpy()
    for i in range(len(gdf)):
        default = calc._build_default_demand(gdf.iloc[i], lt_with_cycle.max, horizon, tol=tol) if with_fallback else None
        ref = _reference_glm_ddlt(lt_with_cycle, mus[i], alphas[i], default, tol, calc.tail_tol)
        got = dist_df["demand_distribution"].iloc[i]
        assert got.mean() == pytest.approx(ref.mean(), rel=1e-6, abs=1e-9)
        assert got.std() == pytest.approx(ref.std(), rel=1e-5, abs=1e-9)
        np.testing.assert_array_equal(got.quantile(SL_GRID), ref.quantile(SL_GRID))

        default_t = calc._build_default_demand(gdf.iloc[i], lt_with_cycle.max, horizon, tol=tol_t) if with_fallback else None
        ref_t = _reference_glm_ddlt(lt_with_cycle, mus[i], alphas[i], default_t, tol_t, calc.tail_tol)
        assert int(tgt_df["target_inventory"].iloc[i]) == int(ref_t.quantile(gdf["service_level"].iloc[i])[0])


def test_glm_group_rejects_invalid_means():
    lead_time = DiscreteDistribution(np.array([1.0]), 2)
    gdf, target_cols = _glm_frame(5, 2, seed=0)
    gdf.loc[3, "target_t_plus_1_mu"] = -1.0
    with pytest.raises(ValueError, match="Invalid GLM mean for row 3"):
        DDLTCalculator(lead_time).from_glm_group(gdf, lead_time, target_cols, return_distribution=True)


# ---------------------------------------------------------------------------
# 3. Quick fixes: no hidden mutation, consistent forecaster signatures, lazy matplotlib
# ---------------------------------------------------------------------------

def test_matmul_does_not_mutate_left_operand():
    n = DiscreteDistribution(np.array([0.25, 0.5, 0.25]), 2, bin_size=2)  # N in {2, 4, 6}
    x = DiscreteDistribution.from_parametric("poisson", (1.5, 0))
    before = (n.pmf.copy(), n.min, n.bin_size)
    _ = n @ x
    assert np.array_equal(n.pmf, before[0]) and n.min == before[1] and n.bin_size == before[2]


def test_non_inplace_bin_and_shift_operations():
    d = DiscreteDistribution(np.array([0.1, 0.2, 0.3, 0.4]), 0)
    merged = d.merge_bins(2, inplace=False)
    assert merged is not d and merged.bin_size == 2 and d.bin_size == 1 and len(d.pmf) == 4
    split = merged.split_bins(inplace=False)
    assert split.bin_size == 1 and merged.bin_size == 2
    shifted = d.shift(5, inplace=False)
    assert shifted.min == 5 and d.min == 0
    # in-place calls keep the old behaviour and now return self (chainable)
    assert d.shift(1) is d and d.min == 1


def test_forecaster_predict_and_compute_ddlt_signatures_are_consistent():
    def names(fn):
        return [p for p in inspect.signature(fn).parameters if p != "self"]

    for method in ("predict", "compute_ddlt"):
        base = names(getattr(ProbabilisticForecaster, method))
        for cls in (GLMForecaster, CatBoostQuantileForecaster):
            sub = names(getattr(cls, method))
            assert sub[: len(base)] == base, f"{cls.__name__}.{method} diverges from the base signature"
    # GLM keeps the old keyword as a backward-compatible alias
    assert "model_data" in names(GLMForecaster.predict)


def test_core_import_does_not_load_matplotlib():
    import subprocess
    import sys

    code = (
        "import sys; import probabilistic_modeling.distributions.discrete, probabilistic_modeling.ddlt, "
        "probabilistic_modeling.inventory.simulation; print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# 4. Speed (benchmark marker)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_glm_ddlt_numba_speedup():
    horizon = 7
    lead_time = DiscreteDistribution(np.array([0.1, 0.3, 0.3, 0.2, 0.1]), 3)  # L in {3..7}
    gdf, target_cols = _glm_frame(300, horizon, seed=9)
    calc = DDLTCalculator(lead_time, cycle_time=0)
    lt_with_cycle = calc._shifted_lead_time(lead_time)
    mus = gdf[[f"{t}_mu" for t in target_cols]].to_numpy()
    alphas = gdf[[f"{t}_alpha" for t in target_cols]].to_numpy()

    calc.from_glm_group(gdf.head(3), lead_time, target_cols, return_distribution=True)  # warm-up / JIT
    t0 = time.perf_counter()
    calc.from_glm_group(gdf, lead_time, target_cols, return_distribution=True)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(len(gdf)):
        _reference_glm_ddlt(lt_with_cycle, mus[i], alphas[i], None, DEFAULT_SUPPORT_TOL, calc.tail_tol)
    t_old = time.perf_counter() - t0
    print(f"\n[GLM DDLT, 300 rows x 7 horizons] scipy+random_sum {t_old*1e3:.1f} ms "
          f"({t_old/300*1e3:.2f} ms/row) | numba {t_new*1e3:.1f} ms ({t_new/300*1e3:.3f} ms/row) "
          f"-> {t_old/t_new:.1f}x | projected 85,372 rows: {t_old/300*85372:.0f} s -> {t_new/300*85372:.1f} s")
    assert t_new < t_old


# ---------------------------------------------------------------------------
# 5. Zero-Hurdle Poisson & Negative-Binomial PMF, DDLT & GLMForecaster(hurdle=True)
# ---------------------------------------------------------------------------

def _scipy_hurdle_reference(pi, lam, alpha, tol):
    if np.isnan(pi):
        return _scipy_reference(lam, alpha, tol)
    if pi <= 0.0:
        return DiscreteDistribution(np.ones(1), 0)
    pi = min(float(pi), 1.0)
    if lam <= 1e-12:
        return DiscreteDistribution(np.array([1.0 - pi, pi]), 0)
    base = _scipy_reference(lam, alpha, tol)
    if len(base.pmf) <= 1 or base.pmf[0] >= 1.0 - 1e-15:
        return DiscreteDistribution(np.array([1.0 - pi, pi]), 0)
    pos_mass = 1.0 - float(base.pmf[0])
    pmf = np.empty(len(base.pmf), dtype=np.float64)
    pmf[0] = 1.0 - pi
    pmf[1:] = pi * base.pmf[1:] / pos_mass
    return DiscreteDistribution(pmf, 0)


@pytest.mark.parametrize("pi", [0.0, 1e-4, 0.25, 0.7, 1.0])
@pytest.mark.parametrize("lam", [0.0, 1e-8, 0.5, 4.0, 35.0])
@pytest.mark.parametrize("alpha", [np.nan, 0.0, 0.2, 1.5])
@pytest.mark.parametrize("tol", [1e-5, 1e-7])
def test_hurdle_count_pmf_matches_scipy_reference(pi, lam, alpha, tol):
    got = _hurdle_count_pmf_truncated_numba(pi, lam, alpha, tol)
    ref = _scipy_hurdle_reference(pi, lam, alpha, tol)

    assert got.min() >= 0.0 and np.isclose(got.sum(), 1.0)
    assert got[0] == pytest.approx(ref.pmf[0], rel=1e-12, abs=1e-14)
    d = DiscreteDistribution(got, 0)
    expected_mu = 0.0 if pi <= 0.0 else float(pi * _hurdle_conditional_mean(lam, alpha))
    assert d.mean() == pytest.approx(expected_mu, rel=1e-3, abs=1e-4)
    assert d.mean() == pytest.approx(ref.mean(), rel=1e-3, abs=1e-4)

    n = min(len(got), len(ref.pmf)) - 1
    if n >= 1:
        np.testing.assert_allclose(got[:n], ref.pmf[:n], rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("with_fallback", [False, True])
def test_hurdle_glm_group_ddlt_matches_reference(with_fallback):
    horizon = 3
    lead_time = DiscreteDistribution(np.array([0.3, 0.5, 0.2]), 2)  # L in {2,3,4}
    cycle_time = 1 if with_fallback else 0
    gdf, target_cols = _glm_frame(40, horizon, seed=21, with_fallback=with_fallback)
    rng = np.random.default_rng(99)
    for h in range(horizon):
        pis = rng.uniform(0.05, 0.95, size=len(gdf))
        lams = gdf[f"target_t_plus_{h}_mu"].to_numpy()
        alphas = gdf[f"target_t_plus_{h}_alpha"].to_numpy()
        gdf[f"target_t_plus_{h}_pi"] = pis
        gdf[f"target_t_plus_{h}_lambda"] = lams
        gdf[f"target_t_plus_{h}_mu"] = [
            float(p * _hurdle_conditional_mean(float(l), None if np.isnan(a) else float(a)))
            for p, l, a in zip(pis, lams, alphas)
        ]

    calc = DDLTCalculator(lead_time, cycle_time=cycle_time)
    dist_df = calc.from_glm_group(gdf, lead_time, target_cols, return_distribution=True)
    tgt_df = calc.from_glm_group(gdf, lead_time, target_cols, return_distribution=False)

    lt_with_cycle = calc._shifted_lead_time(lead_time)
    tol = calc.effective_tol(max(int(lt_with_cycle.max), 1), None)
    pis_mat = gdf[[f"{t}_pi" for t in target_cols]].to_numpy()
    lams_mat = gdf[[f"{t}_lambda" for t in target_cols]].to_numpy()
    alphas_mat = gdf[[f"{t}_alpha" for t in target_cols]].to_numpy()

    for i in range(len(gdf)):
        default = calc._build_default_demand(gdf.iloc[i], lt_with_cycle.max, horizon, tol=tol) if with_fallback else None
        daily = [_scipy_hurdle_reference(pis_mat[i, h], lams_mat[i, h], alphas_mat[i, h], tol) for h in range(horizon)]
        ref = lt_with_cycle.random_sum(daily, default, tail_tol=calc.tail_tol)
        got = dist_df["demand_distribution"].iloc[i]
        assert got.mean() == pytest.approx(ref.mean(), rel=1e-4, abs=1e-6)
        assert got.std() == pytest.approx(ref.std(), rel=1e-4, abs=1e-6)
        assert int(tgt_df["target_inventory"].iloc[i]) >= 0


def test_hurdle_glm_forecaster_positive_overdispersion_and_end_to_end(tmp_path):
    """
    Verifies that:
    1. `DiscreteTimeSeries` computes `positive_overdispersion` strictly on positive training
       observations (`demand > 0`), distinguishing a zero-heavy item whose positive sales are
       under-dispersed (`overdispersion == 1` overall, `positive_overdispersion == 0` on positives)
       from an item whose positive sales are over-dispersed (`positive_overdispersion == 1`).
    2. `GLMForecaster(hurdle=True)` routes `positive_overdispersion == 0` to `hurdle_poisson`
       and `positive_overdispersion == 1` to `hurdle_nbinom`.
    3. `predict()`, `save()` / `load_forecaster()`, `coefficients_table()`, `predict_glm()`,
       and `compute_ddlt()` work end-to-end.
    """
    rng = np.random.default_rng(2026)
    days = list(range(1, 81))
    records = []
    for d in days:
        # item_A: 70% zeros -> overall var > mean (overdispersion == 1),
        # but when positive, demand is always 3 or 4 -> positive var (~0.25) < positive mean (~3.5),
        # so positive_overdispersion == 0 (routed to hurdle_poisson)!
        is_pos_a = rng.random() < 0.30
        dem_a = int(rng.choice([3, 4])) if is_pos_a else 0
        records.append({
            "store_id": "S1",
            "cat_id": "FOODS",
            "item_id": "item_A",
            "day": d,
            "demand": dem_a,
        })
        # item_B: 50% zeros, and when positive, drawn from 1 + NegBin(2, 0.2) -> positive var >> positive mean,
        # so positive_overdispersion == 1 (routed to hurdle_nbinom)!
        is_pos_b = rng.random() < 0.50
        dem_b = int(1 + rng.negative_binomial(2, 0.2)) if is_pos_b else 0
        records.append({
            "store_id": "S1",
            "cat_id": "FOODS",
            "item_id": "item_B",
            "day": d,
            "demand": dem_b,
        })

    sales_df = pd.DataFrame(records)
    calendar_df = pd.DataFrame({
        "day": list(range(1, 90)),
        "wday": [(d - 1) % 7 + 1 for d in range(1, 90)],
        "promo": [1 if d % 5 == 0 else 0 for d in range(1, 90)],
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

    meta_a = ts.series_metadata[ts.series_metadata["item_id"] == "item_A"].iloc[0]
    meta_b = ts.series_metadata[ts.series_metadata["item_id"] == "item_B"].iloc[0]
    # Unconditional demand for item_A is overdispersed due to zeros, but positive demand is underdispersed:
    assert int(meta_a["overdispersion"]) == 1
    assert int(meta_a["positive_overdispersion"]) == 0
    assert int(meta_b["positive_overdispersion"]) == 1

    hurdle_glm = GLMForecaster(train_groups=["store_id", "cat_id"], restrict_future_leads=True, hurdle=True)
    model_df = hurdle_glm.fit(ts)
    assert "positive_overdispersion" in model_df.columns
    row_under = model_df[model_df["positive_overdispersion"] == 0].iloc[0]
    row_over = model_df[model_df["positive_overdispersion"] == 1].iloc[0]
    assert row_under["target_t_plus_0"]["model_type"] == "hurdle_poisson"
    assert row_under["target_t_plus_0"]["alpha"] is None
    assert "hurdle_coefficients" in row_under["target_t_plus_0"]
    assert row_over["target_t_plus_0"]["model_type"] == "hurdle_nbinom"
    assert row_over["target_t_plus_0"]["alpha"] is not None and row_over["target_t_plus_0"]["alpha"] > 0

    coef_table = hurdle_glm.model.coefficients_table()
    assert set(coef_table["component"].unique()) == {"count", "hurdle"}
    assert "alpha" not in coef_table["feature"].values

    # Save & reload via universal load_forecaster:
    bundle_dir = tmp_path / "hurdle_glm_bundle"
    hurdle_glm.save(bundle_dir)
    loaded = load_forecaster(bundle_dir)
    assert isinstance(loaded, GLMForecaster) and loaded.hurdle is True

    preds = loaded.predict(ts, period=76)
    assert len(preds) == 2
    for t in ts.target_cols:
        assert f"{t}_pi" in preds.columns
        assert f"{t}_lambda" in preds.columns
        assert f"{t}_mu" in preds.columns
        assert f"{t}_alpha" in preds.columns
        assert ((preds[f"{t}_pi"] > 0.0) & (preds[f"{t}_pi"] < 1.0)).all()
        assert (preds[f"{t}_lambda"] > 0.0).all()
        assert (preds[f"{t}_mu"] > 0.0).all()

    # Check single-row predict_glm helper matches vectorized GLMForecaster.predict:
    sample_row = ts.get_inference_data(period=76).iloc[0]
    m_dict = row_under["target_t_plus_0"] if sample_row["item_id"] == "item_A" else row_over["target_t_plus_0"]
    mu_single, _ = predict_glm(m_dict, sample_row)
    assert mu_single == pytest.approx(float(preds.iloc[0]["target_t_plus_0_mu"]), rel=1e-9)

    # Check standalone bare_df (without positive_overdispersion column) and DDLT propagation:
    bare_df = ts.get_inference_data(period=77).drop(
        columns=["overdispersion", "positive_overdispersion", "best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
    )
    lt_distr = {"S1": DiscreteDistribution(np.array([0.5, 0.5]), min_value=2)}
    ddlt_df = loaded.compute_ddlt(
        bare_df, lead_time=lt_distr, lead_time_keys=["store_id"], cycle_time=1, return_distribution=True
    )
    assert len(ddlt_df) == 2
    for dist in ddlt_df["demand_distribution"]:
        assert isinstance(dist, DiscreteDistribution)
        assert dist.mean() > 0.0


