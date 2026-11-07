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
)
from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.models.forecasters import (
    CatBoostQuantileForecaster,
    GLMForecaster,
    ProbabilisticForecaster,
)

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

