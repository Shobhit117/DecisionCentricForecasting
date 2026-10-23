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
    q_grid = np.array([0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    for q_row in q_rows:
        expected = legacy_ddlt(lead_time.pmf, lead_time.min, q_row)
        # 1) Untrimmed mode (tail_tol=-1.0) matches legacy bit-for-bit
        pmf_untrimmed, mn_untrimmed = _ddlt_from_quantiles_row_numba(
            lead_time.pmf, lead_time.min, LEVELS, q_row, np.ones(1), 0, 1e-4, -1.0
        )
        assert mn_untrimmed == expected.min
        assert len(pmf_untrimmed) == len(expected.pmf)
        assert np.allclose(pmf_untrimmed, expected.pmf, atol=1e-10)

        # 2) Default trimmed mode (tail_tol=1e-9) trims negligible tail mass and preserves quantiles
        pmf, mn = _ddlt_from_quantiles_row_numba(lead_time.pmf, lead_time.min, LEVELS, q_row, np.ones(1), 0, 1e-4)
        trimmed_distr = DiscreteDistribution(pmf, mn)
        assert len(pmf) <= len(expected.pmf)
        np.testing.assert_array_equal(trimmed_distr.quantile(q_grid), expected.quantile(q_grid))


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
    b_untrimmed = lead_time.random_sum(list(daily_new[0]), tail_tol=-1.0)
    assert a.min == b_untrimmed.min and np.allclose(a.pmf, b_untrimmed.pmf, atol=1e-10)
    assert len(b.pmf) <= len(a.pmf)
    np.testing.assert_array_equal(a.quantile([0.1, 0.5, 0.9, 0.95, 0.99]), b.quantile([0.1, 0.5, 0.9, 0.95, 0.99]))

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


# ----------------------------------------------------------------------------------
# Long-support scenario: ~100-point daily demand PMFs and a ~10-day lead-time PMF.
# ----------------------------------------------------------------------------------
LONG_SUPPORT = 100
LT_DAYS = 10
LT_MIN = 3  # lead time + review period supported on {3, ..., 12}
N_LONG_ROWS = 40


def _long_daily_pmfs(n_rows, seed=0):
    """Per row: LT_MIN + LT_DAYS - 1 daily negative-binomial PMFs, each truncated to 100 support points."""
    rng = np.random.default_rng(seed)
    n_days = LT_MIN + LT_DAYS - 1
    support = np.arange(LONG_SUPPORT)
    rows = []
    for _ in range(n_rows):
        mean = rng.uniform(25.0, 45.0)
        days = []
        for _ in range(n_days):
            m = mean * rng.uniform(0.9, 1.1)
            r = 6.0  # moderate over-dispersion -> long right tail
            pmf = stats.nbinom(r, r / (r + m)).pmf(support)
            days.append(pmf / pmf.sum())
        rows.append(days)
    return rows


@pytest.fixture(scope="module")
def long_lead_time():
    pmf = np.linspace(1.0, 2.0, LT_DAYS)
    return pmf / pmf.sum()


def test_long_support_ddlt_speed(long_lead_time, warmed_up):
    from numba.typed import List as NumbaList

    from ddlt import _random_sum_aligned_numba

    rows = _long_daily_pmfs(N_LONG_ROWS)
    lt_old = legacy.DiscreteDistribution(long_lead_time.copy(), LT_MIN)
    lt_new = DiscreteDistribution(long_lead_time.copy(), LT_MIN)
    old_days = [[legacy.DiscreteDistribution(p.copy(), 0) for p in r] for r in rows]
    new_days = [[DiscreteDistribution(p.copy(), 0) for p in r] for r in rows]

    # Raw compiled kernel inputs (no Python wrapper overhead)
    typed = []
    for r in rows:
        lst = NumbaList()
        for p in r:
            lst.append(np.ascontiguousarray(p))
        typed.append(lst)
    zero_mins = np.zeros(LT_MIN + LT_DAYS - 1, dtype=np.int64)

    # correctness
    a = lt_old.random_sum(list(old_days[0]))
    b = lt_new.random_sum(list(new_days[0]))
    c, c_min = _random_sum_aligned_numba(long_lead_time, LT_MIN, typed[0], zero_mins, 1)
    c_untrimmed, c_untrimmed_min = _random_sum_aligned_numba(long_lead_time, LT_MIN, typed[0], zero_mins, 1, -1.0)
    assert a.min == c_untrimmed_min and np.allclose(a.pmf, c_untrimmed, atol=1e-10)
    assert b.min == c_min and np.allclose(b.pmf, c, atol=1e-12)
    assert len(b.pmf) < len(a.pmf)
    np.testing.assert_array_equal(a.quantile([0.1, 0.5, 0.9, 0.95, 0.99]), b.quantile([0.1, 0.5, 0.9, 0.95, 0.99]))
    print(
        f"\n   result DDLT support: legacy untrimmed={len(a.pmf)} pts [{a.min}, {a.max}] | "
        f"numba trimmed={len(b.pmf)} pts [{b.min}, {b.max}]"
    )

    t_old = _timeit(lambda: [lt_old.random_sum(list(d)) for d in old_days], repeats=2)
    t_wrap = _timeit(lambda: [lt_new.random_sum(list(d)) for d in new_days], repeats=2)
    t_raw = _timeit(lambda: [_random_sum_aligned_numba(long_lead_time, LT_MIN, t, zero_mins, 1) for t in typed], repeats=2)
    n = N_LONG_ROWS
    print(f"[long support: {LONG_SUPPORT}-pt daily PMFs, {LT_DAYS}-day lead time, {n} rows]")
    print(f"   legacy random_sum            {t_old*1e3:9.1f} ms ({t_old/n*1e3:7.2f} ms/row)")
    print(f"   numba via DiscreteDistribution {t_wrap*1e3:7.1f} ms ({t_wrap/n*1e3:7.2f} ms/row) | speedup {t_old/t_wrap:5.1f}x")
    print(f"   numba raw kernel             {t_raw*1e3:9.1f} ms ({t_raw/n*1e3:7.2f} ms/row) | speedup {t_old/t_raw:5.1f}x")
    print(f"   projected for 85,372 rows: legacy ~{t_old/n*85372:,.0f} s | numba wrapper ~{t_wrap/n*85372:,.0f} s | raw kernel ~{t_raw/n*85372:,.0f} s")
    assert t_raw < t_old
