"""
Timing + equivalence tests comparing the Numba DDLT implementation against the
legacy pure-NumPy implementation (`tests/legacy_discrete_distribution.py`, taken from
commit 31da9af, before the Numba rewrite).

Run with timings printed:

    python -m pytest tests/test_ddlt_benchmark.py -s -q
"""
import os
import sys
import time

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import legacy_discrete_distribution as legacy  # noqa: E402
from ddlt import _ddlt_from_quantiles_row_numba, _quantiles_to_pmf_numba  # noqa: E402
from discrete_distribution import DiscreteDistribution  # noqa: E402
from quantile_distribution import DEFAULT_QUANTILES  # noqa: E402

LEVELS = np.asarray(DEFAULT_QUANTILES, dtype=np.float64)
HORIZON = 7
N_ROWS = 300  # rows timed per implementation (legacy is slow, so keep this modest)


def legacy_quantiles_to_pmf(levels, values, tol=1e-4):
    """The original NumPy implementation of quantile -> PMF reconstruction."""
    q = np.maximum(np.sort(np.asarray(values, dtype=float)), 0.0)
    slope_lo = (q[1] - q[0]) / (levels[1] - levels[0])
    q_zero = max(0.0, q[0] - slope_lo * levels[0])
    theta = (q[-1] - q[-2]) / np.log((1.0 - levels[-2]) / (1.0 - levels[-1]))
    q_upper = q[-1] + theta * np.log((1.0 - levels[-1]) / tol) if tol < 1.0 - levels[-1] else q[-1]
    y_max = max(int(np.floor(q_upper + 0.5)), 0)
    x = np.arange(y_max + 1, dtype=float) + 0.5 - 1e-9
    cdf = np.interp(x, np.concatenate(([q_zero], q)), np.concatenate(([0.0], levels)))
    upper = x > q[-1]
    cdf[upper] = 1.0 - (1.0 - levels[-1]) * np.exp(-(x[upper] - q[-1]) / theta) if theta > 0 else 1.0
    cdf[-1] = 1.0
    pmf = np.maximum(np.diff(cdf, prepend=0.0), 0.0)
    nz = np.flatnonzero(pmf > 0)
    return pmf[nz[0]:nz[-1] + 1], int(nz[0])


def legacy_ddlt(lt_pmf, lt_min, q_row):
    daily = [legacy.DiscreteDistribution(*legacy_quantiles_to_pmf(LEVELS, q)) for q in q_row]
    lt = legacy.DiscreteDistribution(lt_pmf.copy(), lt_min)
    return lt.random_sum(daily)


def make_quantile_rows(n, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    mu = rng.gamma(2.0, scale, size=(n, 1)) + 0.5
    rows = np.empty((n, HORIZON, len(LEVELS)))
    for h in range(HORIZON):
        rows[:, h, :] = stats.poisson(mu * (1 + 0.05 * h)).ppf(LEVELS[None, :])
    return np.ascontiguousarray(rows)


def _timeit(fn, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.fixture(scope="module")
def lead_time():
    return DiscreteDistribution(np.array([0.2, 0.5, 0.3]), min_value=2)  # L + R in {2, 3, 4}


@pytest.fixture(scope="module")
def warmed_up(lead_time):
    """Trigger Numba JIT compilation so timings exclude compile time."""
    q = make_quantile_rows(2)
    _ddlt_from_quantiles_row_numba(lead_time.pmf, lead_time.min, LEVELS, q[0], np.ones(1), 0, 1e-4)
    return True


def test_numba_ddlt_matches_legacy(lead_time, warmed_up):
    q_rows = make_quantile_rows(25, seed=1)
    for q_row in q_rows:
        expected = legacy_ddlt(lead_time.pmf, lead_time.min, q_row)
        pmf, mn = _ddlt_from_quantiles_row_numba(lead_time.pmf, lead_time.min, LEVELS, q_row, np.ones(1), 0, 1e-4)
        assert mn == expected.min
        assert len(pmf) == len(expected.pmf)
        assert np.allclose(pmf, expected.pmf, atol=1e-10)


def test_quantile_to_pmf_speed(warmed_up):
    q_rows = make_quantile_rows(N_ROWS, seed=2)[:, 0, :]
    for r in q_rows[:5]:
        np.testing.assert_allclose(legacy_quantiles_to_pmf(LEVELS, r)[0], _quantiles_to_pmf_numba(LEVELS, r, 1e-4, -1)[0], atol=1e-12)

    t_old = _timeit(lambda: [legacy_quantiles_to_pmf(LEVELS, r) for r in q_rows])
    t_new = _timeit(lambda: [_quantiles_to_pmf_numba(LEVELS, r, 1e-4, -1) for r in q_rows])
    print(f"\n[quantiles -> PMF, {N_ROWS} vectors]  legacy {t_old*1e3:8.1f} ms | numba {t_new*1e3:8.1f} ms | speedup {t_old/t_new:6.1f}x")
    assert t_new < t_old


def test_random_sum_speed(lead_time, warmed_up):
    """Isolates the random-sum (convolution + mixture) step, given precomputed daily PMFs."""
    q_rows = make_quantile_rows(N_ROWS, seed=3)
    daily_old = [[legacy.DiscreteDistribution(*legacy_quantiles_to_pmf(LEVELS, q)) for q in row] for row in q_rows]
    daily_new = [[DiscreteDistribution(*legacy_quantiles_to_pmf(LEVELS, q)) for q in row] for row in q_rows]
    lt_old = legacy.DiscreteDistribution(lead_time.pmf.copy(), lead_time.min)

    # correctness spot-check
    a, b = lt_old.random_sum(list(daily_old[0])), lead_time.random_sum(list(daily_new[0]))
    assert a.min == b.min and np.allclose(a.pmf, b.pmf, atol=1e-10)

    lead_time.random_sum(list(daily_new[0]))  # warm-up the DiscreteDistribution -> numba path
    t_old = _timeit(lambda: [lt_old.random_sum(list(d)) for d in daily_old])
    t_new = _timeit(lambda: [lead_time.random_sum(list(d)) for d in daily_new])
    print(f"\n[random_sum, {N_ROWS} rows]            legacy {t_old*1e3:8.1f} ms | numba {t_new*1e3:8.1f} ms | speedup {t_old/t_new:6.1f}x")
    assert t_new < t_old


def test_end_to_end_ddlt_speed(lead_time, warmed_up):
    """Full pipeline per row: quantiles -> daily PMFs -> DDLT PMF -> 95th percentile."""
    q_rows = make_quantile_rows(N_ROWS, seed=4)
    lt_pmf, lt_min = lead_time.pmf, lead_time.min
    dummy = np.ones(1)

    def run_old():
        return [legacy_ddlt(lt_pmf, lt_min, r).quantile(0.95)[0] for r in q_rows]

    def run_new():
        out = []
        for r in q_rows:
            pmf, mn = _ddlt_from_quantiles_row_numba(lt_pmf, lt_min, LEVELS, r, dummy, 0, 1e-4)
            out.append(mn + int(np.searchsorted(np.cumsum(pmf), 0.95)))
        return out

    assert run_old() == run_new()  # identical inventory targets

    t_old = _timeit(run_old, repeats=2)
    t_new = _timeit(run_new, repeats=2)
    per_old, per_new = t_old / N_ROWS * 1e3, t_new / N_ROWS * 1e3
    projected_old = per_old * 85_372 / 1e3
    projected_new = per_new * 85_372 / 1e3
    print(
        f"\n[end-to-end DDLT, {N_ROWS} rows]      legacy {t_old*1e3:8.1f} ms ({per_old:.3f} ms/row) | "
        f"numba {t_new*1e3:8.1f} ms ({per_new:.3f} ms/row) | speedup {t_old/t_new:6.1f}x"
    )
    print(f"   projected for one M5 store (85,372 rows): legacy ~{projected_old:,.0f} s | numba ~{projected_new:,.1f} s")
    assert t_new < t_old
