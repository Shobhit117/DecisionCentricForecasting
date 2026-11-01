"""
Tests for the adaptive quantile search (linear scan vs. CDF + binary search), the
precomputed-CDF newsvendor, and service-level-aware support truncation.

Timings are printed with:

    python -m pytest tests/test_quantile_search.py -s -q
"""
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from probabilistic_modeling.ddlt import (
    DEFAULT_SUPPORT_TOL,
    QUANTILE_BSEARCH_MIN_QUERIES,
    QUANTILE_BSEARCH_MIN_SUPPORT,
    DDLTCalculator,
    _batch_cdf_quantiles_numba,
    _batch_pmf_quantiles_numba,
    _cdf_quantile_scalar_numba,
    _pmf_quantile_scalar_numba,
    _pmf_quantiles_linear_numba,
    _pmf_quantiles_numba,
    _pmf_to_cdf_numba,
    build_cdf_table,
    support_tol_for_service_level,
)
from probabilistic_modeling.distributions.discrete import DiscreteDistribution, TruncatedSupportWarning
from probabilistic_modeling.inventory.newsvendor import solve_multi_product_newsvendor


def _edge_quantiles(rng, m):
    q = rng.random(m)
    edges = np.array([0.0, 1e-300, 1e-12, 0.5, 0.95, 0.99, 0.9999, 0.99999, 1 - 1e-12, 1 - 1e-16, 1.0])
    return np.ascontiguousarray(np.concatenate([q, edges]))


# ---------------------------------------------------------------------------
# 1. Adaptive single-distribution kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 10, 63, 64, 1_000, 50_000])
@pytest.mark.parametrize("m", [1, 7, 8, 500])
def test_adaptive_quantiles_match_linear_and_searchsorted(n, m):
    rng = np.random.default_rng(n * 1000 + m)
    pmf = rng.random(n) ** 3  # skewed, with many tiny entries
    pmf /= pmf.sum()
    q = _edge_quantiles(rng, m)

    adaptive = _pmf_quantiles_numba(pmf, 5, 2, q)
    linear = _pmf_quantiles_linear_numba(pmf, 5, 2, q)
    np.testing.assert_array_equal(adaptive, linear)

    # Reference semantics: first index with cumsum >= q, clamped to the last index
    idx = np.minimum(np.searchsorted(np.cumsum(pmf), q, side="left"), n - 1)
    idx[q <= 0.0] = 0
    np.testing.assert_array_equal(adaptive, 5 + 2 * idx)


def test_binary_search_path_is_actually_taken_and_identical():
    rng = np.random.default_rng(1)
    n, m = QUANTILE_BSEARCH_MIN_SUPPORT * 10, QUANTILE_BSEARCH_MIN_QUERIES * 10
    pmf = rng.random(n)
    pmf /= pmf.sum()
    q = _edge_quantiles(rng, m)
    cdf = _pmf_to_cdf_numba(pmf)
    bs = np.array([_cdf_quantile_scalar_numba(cdf, 0, n, 0, 1, x) for x in q])
    lin = np.array([_pmf_quantile_scalar_numba(pmf, 0, 1, x) for x in q])
    np.testing.assert_array_equal(bs, lin)


def test_bsearch_handles_cdf_ending_below_one_and_tiny_negative_pmf_entries():
    # CDF that ends at 1 - 1e-13 due to round-off: q = 1.0 must return the last support point.
    pmf = np.full(1000, 1.0 / 1000)
    pmf[-1] -= 1e-13
    q = np.full(QUANTILE_BSEARCH_MIN_QUERIES, 1.0)
    assert np.all(_pmf_quantiles_numba(pmf, 0, 1, q) == 999)

    # FFT convolution can create tiny negative values -> non-monotone raw cumsum.
    rng = np.random.default_rng(7)
    pmf = rng.random(2_000)
    pmf /= pmf.sum()
    pmf[::50] = -1e-17
    q = _edge_quantiles(rng, 2_000)
    np.testing.assert_array_equal(_pmf_quantiles_numba(pmf, 0, 1, q), _pmf_quantiles_linear_numba(pmf, 0, 1, q))
    assert np.all(np.diff(_pmf_to_cdf_numba(pmf)) >= 0.0)


def test_distribution_quantile_and_sample_use_adaptive_kernel():
    d = DiscreteDistribution.from_parametric("nbinom", (5.0, 0.01, 0))  # long support (~1500 bins)
    q = np.linspace(0.0, 0.999, 1_000)
    np.testing.assert_array_equal(d.quantile(q), _pmf_quantiles_linear_numba(d.pmf, d.min, d.bin_size, q))
    s = d.sample(20_000, rng=0)
    assert abs(s.mean() - d.mean()) < 0.05 * d.mean()


# ---------------------------------------------------------------------------
# 2. Precomputed CDF table + newsvendor
# ---------------------------------------------------------------------------

def test_batch_cdf_quantiles_match_batch_pmf_quantiles():
    from numba.typed import List as NumbaList

    rng = np.random.default_rng(3)
    pmfs = []
    for _ in range(300):
        p = rng.random(int(rng.integers(1, 3_000)))
        pmfs.append(p / p.sum())
    mins = rng.integers(0, 50, len(pmfs)).astype(np.int64)
    bins = rng.integers(1, 4, len(pmfs)).astype(np.int64)
    alphas = rng.random(len(pmfs))
    alphas[:5] = [0.0, 1.0, 0.99999, 1 - 1e-15, 1e-300]

    cdf_flat, offsets = build_cdf_table(pmfs)
    nl = NumbaList()
    for p in pmfs:
        nl.append(p)
    np.testing.assert_array_equal(
        _batch_cdf_quantiles_numba(cdf_flat, offsets, mins, bins, alphas),
        _batch_pmf_quantiles_numba(nl, mins, bins, alphas),
    )


def test_prefix_cdfs_answer_all_lower_queries_exactly():
    from numba.typed import List as NumbaList

    from probabilistic_modeling.ddlt import _build_prefix_cdfs_numba

    rng = np.random.default_rng(21)
    nl = NumbaList()
    for size in rng.integers(1, 5_000, 400):
        p = rng.random(int(size)) ** 4
        nl.append(p / p.sum())
    mins = rng.integers(0, 10, len(nl)).astype(np.int64)
    bins = rng.integers(1, 3, len(nl)).astype(np.int64)
    a0 = rng.random(len(nl))
    a0[:4] = [0.0, 1.0, 1 - 1e-15, 1e-300]

    x0 = _batch_pmf_quantiles_numba(nl, mins, bins, a0)
    cdf_flat, offsets = _build_prefix_cdfs_numba(nl, (x0 - mins) // bins + 1)
    # prefix stops at the answer index: no work beyond it
    np.testing.assert_array_equal(np.diff(offsets), (x0 - mins) // bins + 1)
    for shrink in (1.0, 0.999, 0.7, 0.3, 0.0):
        a = a0 * shrink
        np.testing.assert_array_equal(
            _batch_cdf_quantiles_numba(cdf_flat, offsets, mins, bins, a),
            _batch_pmf_quantiles_numba(nl, mins, bins, a),
        )


def _reference_newsvendor(dists, price, cost, shortage, budget, max_iters=200, tol=1e-4):
    """Pre-change implementation (linear scan in every shadow-price iteration)."""
    from numba.typed import List as NumbaList

    pmfs = NumbaList()
    for d in dists:
        pmfs.append(d.pmf)
    mins = np.array([d.min for d in dists], dtype=np.int64)
    bins = np.array([d.bin_size for d in dists], dtype=np.int64)
    denom = price + shortage

    def opt(k):
        return _batch_pmf_quantiles_numba(pmfs, mins, bins, np.clip((denom - cost * (1 + k)) / denom, 0, 1))

    x = opt(0.0)
    if float(np.sum(x * cost)) <= budget:
        return x
    lb, ub = 0.0, float(np.max(denom / cost - 1.0))
    for _ in range(max_iters):
        mid = 0.5 * (lb + ub)
        if float(np.sum(opt(mid) * cost)) <= budget:
            ub = mid
        else:
            lb = mid
        if ub - lb < tol:
            break
    return opt(ub)


def _random_products(n_products, mu_scale, seed):
    rng = np.random.default_rng(seed)
    dists = [
        DiscreteDistribution.from_parametric("nbinom", (float(rng.uniform(1, 5)), float(rng.uniform(0.002, 0.05)), 0))
        if mu_scale == "large"
        else DiscreteDistribution.from_parametric("poisson", (float(rng.uniform(1, 30)), 0))
        for _ in range(n_products)
    ]
    price = rng.uniform(5, 20, n_products)
    cost = price * rng.uniform(0.3, 0.7, n_products)
    shortage = rng.uniform(0, 5, n_products)
    return dists, price, cost, shortage


@pytest.mark.parametrize("mu_scale", ["small", "large"])
def test_newsvendor_with_cdf_table_matches_reference(mu_scale):
    dists, price, cost, shortage = _random_products(200, mu_scale, seed=11)
    unconstrained = float(np.sum(_reference_newsvendor(dists, price, cost, shortage, np.inf) * cost))
    for frac in (0.2, 0.5, 0.9, 1.5):
        budget = frac * unconstrained
        new = solve_multi_product_newsvendor(dists, price, cost, shortage, budget)
        ref = _reference_newsvendor(dists, price, cost, shortage, budget)
        np.testing.assert_array_equal(new, ref)
        assert float(np.sum(new * cost)) <= budget + 1e-9


# ---------------------------------------------------------------------------
# 3. Service-level-aware support truncation
# ---------------------------------------------------------------------------

def test_support_tol_for_service_level_values():
    assert support_tol_for_service_level(None) == DEFAULT_SUPPORT_TOL
    assert support_tol_for_service_level(0.95) == DEFAULT_SUPPORT_TOL
    assert support_tol_for_service_level(0.999) == DEFAULT_SUPPORT_TOL
    assert support_tol_for_service_level(0.9999) == pytest.approx(1e-5)
    assert support_tol_for_service_level(0.99999) == pytest.approx(1e-6)
    assert support_tol_for_service_level(0.9999, n_terms=10) == pytest.approx(1e-6)
    assert support_tol_for_service_level(1.0) > 0.0  # floored, never zero
    # never loosens an explicitly tighter base tolerance
    assert support_tol_for_service_level(0.95, base_tol=1e-8) == 1e-8


@pytest.mark.parametrize("q", [0.9999, 0.99999, 0.999999])
def test_from_parametric_high_service_level_is_exact(q):
    for name, params in [("poisson", (50.0, 0)), ("nbinom", (3.0, 0.05, 0))]:
        exact = int(SCIPY[name].ppf(q, *params))
        d = DiscreteDistribution.from_parametric(name, params, max_service_level=q)
        with warnings.catch_warnings():
            warnings.simplefilter("error", TruncatedSupportWarning)
            assert d.quantile(q)[0] == exact


SCIPY = {"poisson": stats.poisson, "nbinom": stats.nbinom}


def test_truncation_warning_is_raised_only_beyond_support_tolerance():
    d = DiscreteDistribution.from_parametric("poisson", (50.0, 0))  # default tol = 1e-4
    assert d.truncation_tol == pytest.approx(1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncatedSupportWarning)
        d.quantile([0.5, 0.95, 0.999])
    with pytest.warns(TruncatedSupportWarning):
        biased = d.quantile(0.99999)[0]
    assert biased < int(stats.poisson.ppf(0.99999, 50.0))  # documents the bias being warned about

    # Bounded families are exact -> never warn
    b = DiscreteDistribution.from_parametric("binom", (20, 0.3, 0))
    assert b.truncation_tol == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncatedSupportWarning)
        b.quantile(0.999999)

    # Truncation bounds propagate through convolution and copies
    s = d + d
    assert s.truncation_tol == pytest.approx(2e-4)
    assert s.copy().truncation_tol == pytest.approx(2e-4)


def _glm_ddlt_frame(mus, service_level):
    horizon = len(mus)
    row = {"store_id": "S1", "item_id": "A", "day": 1, "service_level": service_level}
    for h, mu in enumerate(mus):
        row[f"target_t_plus_{h}_mu"] = mu
        row[f"target_t_plus_{h}_alpha"] = np.nan  # Poisson
    return pd.DataFrame([row]), [f"target_t_plus_{h}" for h in range(horizon)]


@pytest.mark.parametrize("service_level", [0.95, 0.9999, 0.99999])
def test_ddlt_glm_target_is_exact_at_high_service_levels(service_level):
    # Deterministic lead time of 3 days and Poisson daily demand -> DDLT ~ Poisson(sum mu) exactly.
    mus = [20.0, 25.0, 30.0]
    lead_time = DiscreteDistribution(np.array([1.0]), 3)
    gdf, target_cols = _glm_ddlt_frame(mus, service_level)

    calc = DDLTCalculator(lead_time, cycle_time=0)
    out = calc.from_glm_group(gdf, lead_time, target_cols, return_distribution=False)
    exact = int(stats.poisson.ppf(service_level, sum(mus)))
    assert int(out["target_inventory"].iloc[0]) == exact

    # Distribution mode: pass max_service_level explicitly
    calc_d = DDLTCalculator(lead_time, cycle_time=0, max_service_level=service_level)
    dist = calc_d.from_glm_group(gdf, lead_time, target_cols, return_distribution=True)["demand_distribution"].iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncatedSupportWarning)
        assert int(dist.quantile(service_level)[0]) == exact


def test_ddlt_glm_without_max_service_level_warns_at_extreme_quantile():
    mus = [20.0, 25.0, 30.0]
    lead_time = DiscreteDistribution(np.array([1.0]), 3)
    gdf, target_cols = _glm_ddlt_frame(mus, 0.95)
    dist = DDLTCalculator(lead_time, cycle_time=0).from_glm_group(
        gdf, lead_time, target_cols, return_distribution=True
    )["demand_distribution"].iloc[0]
    assert dist.truncation_tol == pytest.approx(3 * DEFAULT_SUPPORT_TOL)
    with pytest.warns(TruncatedSupportWarning):
        dist.quantile(0.99999)


def test_ddlt_effective_tol_uses_max_of_column_and_argument():
    lt = DiscreteDistribution(np.array([0.5, 0.5]), 2)
    calc = DDLTCalculator(lt, cycle_time=1)
    assert calc.effective_tol(4, None) == DEFAULT_SUPPORT_TOL
    assert calc.effective_tol(4, np.array([0.9, 0.9999])) == pytest.approx(1e-4 / (10 * 4))  # 2.5e-6
    calc2 = DDLTCalculator(lt, cycle_time=1, max_service_level=0.99999)
    assert calc2.effective_tol(4, np.array([0.95])) == pytest.approx(1e-5 / 10 / 4)


def test_ddlt_quantile_path_extends_tail_for_high_service_levels():
    """CatBoost-style quantile grid: higher service level -> longer extrapolated tail -> larger target."""
    levels = np.array([0.1, 0.5, 0.9, 0.99])
    lead_time = DiscreteDistribution(np.array([1.0]), 2)
    rows = []
    for sl in (0.95, 0.99999):
        row = {"store_id": "S1", "item_id": f"A{sl}", "day": 1, "service_level": sl}
        for h in range(2):
            for lv, v in zip(levels, [2.0, 5.0, 9.0, 14.0]):
                row[f"target_t_plus_{h}_q{lv:g}"] = v
        rows.append(row)
    gdf = pd.DataFrame(rows)
    from probabilistic_modeling.models.catboost_quantile import quantile_col_name

    gdf = gdf.rename(columns={f"target_t_plus_{h}_q{lv:g}": quantile_col_name(f"target_t_plus_{h}", lv)
                              for h in range(2) for lv in levels})
    calc = DDLTCalculator(lead_time, cycle_time=0)
    out = calc.from_quantiles_group(gdf, lead_time, ["target_t_plus_0", "target_t_plus_1"], levels)
    t95, t99999 = out["target_inventory"].tolist()
    assert t99999 > t95

    # With the default tol (1e-4) each day's extrapolated tail is cut at 1 - 1e-4, so the
    # 0.99999 DDLT quantile is biased low (and a TruncatedSupportWarning is emitted):
    dist_default = calc.from_quantiles_group(
        gdf.iloc[[1]], lead_time, ["target_t_plus_0", "target_t_plus_1"], levels, return_distribution=True
    )["demand_distribution"].iloc[0]
    with pytest.warns(TruncatedSupportWarning):
        biased = int(dist_default.quantile(0.99999)[0])
    assert biased < t99999


# ---------------------------------------------------------------------------
# 4. Speed
# ---------------------------------------------------------------------------

def _best_time(fn, *args, reps=5):
    fn(*args)
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def test_many_quantile_queries_speedup():
    rng = np.random.default_rng(0)
    pmf = rng.random(20_000)
    pmf /= pmf.sum()
    q = rng.random(5_000)
    t_lin = _best_time(_pmf_quantiles_linear_numba, pmf, 0, 1, q)
    t_ada = _best_time(_pmf_quantiles_numba, pmf, 0, 1, q)
    print(f"\n[quantiles] n=20k, m=5k: linear {t_lin*1e3:.2f} ms, adaptive {t_ada*1e3:.3f} ms -> {t_lin/t_ada:.1f}x")
    assert t_ada * 5 < t_lin

    # Single query: adaptive must not regress (it stays on the linear path)
    q1 = np.array([0.99])
    t_lin1 = _best_time(_pmf_quantiles_linear_numba, pmf, 0, 1, q1, reps=50)
    t_ada1 = _best_time(_pmf_quantiles_numba, pmf, 0, 1, q1, reps=50)
    print(f"[quantiles] n=20k, m=1: linear {t_lin1*1e6:.1f} us, adaptive {t_ada1*1e6:.1f} us")
    assert t_ada1 < 3 * t_lin1 + 5e-6


def test_newsvendor_speedup_on_long_supports():
    dists, price, cost, shortage = _random_products(500, "large", seed=5)
    budget = 0.5 * float(np.sum(_reference_newsvendor(dists, price, cost, shortage, np.inf) * cost))
    t_ref = _best_time(_reference_newsvendor, dists, price, cost, shortage, budget, reps=15)
    t_new = _best_time(solve_multi_product_newsvendor, dists, price, cost, shortage, budget, reps=15)
    sizes = [len(d.pmf) for d in dists]
    print(f"\n[newsvendor] 500 products, mean support {np.mean(sizes):.0f} bins: "
          f"reference {t_ref*1e3:.1f} ms, CDF+bsearch {t_new*1e3:.1f} ms -> {t_ref/t_new:.1f}x")
    # Modest by design: the early-exit scan already stops at the answer and alpha(k) shrinks
    # during bisection, so the dominant cost is the unconstrained (k = 0) pass shared by both.
    # ~1 ms total -> timing noise is large; this is a no-regression guard.
    assert t_new < 1.5 * t_ref
