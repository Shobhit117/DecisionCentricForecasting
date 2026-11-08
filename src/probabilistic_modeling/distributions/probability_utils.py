"""
Method-of-moments fitting and goodness-of-fit utilities for discrete parametric distributions.

What this module does
---------------------
Provides fast closed-form parameter estimators (`fit_nbinom`, `fit_binom`, `fit_poisson`,
`fit_geom`, `fit_uniform`) matching SciPy's `scipy.stats` discrete distribution signatures,
plus `rmse_discrete` to score how closely a fitted PMF matches the empirical histogram.

Why this module is needed
-------------------------
SciPy does not provide built-in `.fit()` methods for discrete distributions (`rv_discrete`),
and numerical maximum-likelihood across 3,000+ series would be slow. Closed-form method-of-moments
estimators fit each series in microseconds.

Where this module is used
-------------------------
Called by :func:`probabilistic_modeling.distributions.discrete.compute_best_fit_distribution`.
"""
import numpy as np
from scipy import stats


def rmse_discrete(data : np.ndarray, distr : stats.rv_discrete, params : tuple) -> float:
    """
    Compute the Root Mean Squared Error (scaled by 100, in percentage points) between an
    empirical histogram of ``data`` and the theoretical PMF ``distr.pmf(x, *params)``
    over ``[min(data), max(data)]``.

    Used by :func:`compute_best_fit_distribution` to select the best parametric family.
    """
    low = np.min(data)
    high = np.max(data)
    x = np.arange(low, high+1)
    (values, counts) = np.unique(data, return_counts=True)
    counts = counts / np.sum(counts)
    hist = dict(zip(values, counts))
    pmf = distr.pmf(x, *params)
    hist_vals = np.array([hist.get(val, 0) for val in x])
    rmse = 100 * np.sqrt(np.mean((pmf - hist_vals)**2))
    return rmse


def fit_nbinom(data: np.ndarray) -> tuple[int, float, int]:
    """
    Method-of-moments estimator for ``scipy.stats.nbinom(n, p, loc)``.

    Shifts ``x = data - min(data)`` and matches sample mean ``mu`` and variance ``var``:
    since ``E[X] = n * (1 - p) / p`` and ``Var(X) = n * (1 - p) / p^2 = E[X] / p``,
    we have ``p = min(mu / var, 0.999)`` and ``n = p * mu / (1 - p)``.
    """
    loc = np.min(data)
    x = data - loc
    mu = np.mean(x)
    var = np.var(x)
    if mu == 0:
        return (1.0, 1.0, loc)
    if var > 0:
        p0 = min(mu / var, 0.999)
    else:
        p0 = 0.999
    n0 = p0 * mu / (1-p0)
    return (n0, p0, loc)


def fit_binom(data: np.ndarray) -> tuple[int, float, int]:
    """
    Method-of-moments estimator for ``scipy.stats.binom(n, p, loc)`` with ``loc = min(data)``,
    ``n = max(data) - min(data)``, and ``p = mean(data - loc) / n``.
    """
    loc = np.min(data)
    high = np.max(data)
    n = int(high - loc)
    mu = np.mean(data - loc)
    p = mu / n
    return (n, p, loc)


def fit_poisson(data: np.ndarray) -> tuple[float, int]:
    """
    Method-of-moments estimator for ``scipy.stats.poisson(mu, loc)`` with ``loc = min(data)``
    and ``mu = mean(data - loc)``.
    """
    loc = np.min(data)
    mu = np.mean(data - loc)
    return (mu, loc)


def fit_geom(data: np.ndarray) -> tuple[float, int]:
    """
    Method-of-moments estimator for ``scipy.stats.geom(p, loc)`` (supported on ``{loc+1, loc+2, ...}``)
    with ``loc = min(data) - 1`` and ``p = 1 / mean(data - min(data) + 1)``.
    """
    loc = np.min(data)
    mu = np.mean(data - loc + 1)
    return (1/mu, loc-1)


def fit_uniform(data: np.ndarray) -> tuple[int, int, int]:
    """
    Estimator for ``scipy.stats.randint(low, high, loc=0)`` over ``[min(data), max(data)]``
    (note: SciPy's ``randint(low, high)`` has support ``{low, ..., high - 1}``).
    """
    low = np.min(data)
    high = np.max(data)+1
    return (low, high, 0)


def plot_distr_fit(data: np.ndarray, distr: stats.rv_discrete, params: tuple):
    """Plot the empirical histogram of ``data`` against a fitted discrete PMF ``distr(*params)``."""
    import matplotlib.pyplot as plt  # optional, imported lazily (plotting only)
    from matplotlib.ticker import MaxNLocator

    data = np.array(np.round(data), dtype=int)
    (x, y) = np.unique(data, return_counts=True)
    y = y / np.sum(y)
    hist = dict(zip(x, y))
    low, high = np.min(data), np.max(data)
    x = np.arange(low, high+1)
    y_fit = distr.pmf(x, *params)
    y = np.array([hist.get(k, 0) for k in x])
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.bar(x, y, color='orange', label='data')
    plt.vlines(x, 0, y_fit, color='blue', label='fit')
    plt.legend(loc='best')
