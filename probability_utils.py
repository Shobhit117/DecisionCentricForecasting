import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def rmse_discrete(data : np.ndarray, distr : stats.rv_discrete, params : tuple) -> float:
    # Compute RMSE for discrete RVs:
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
    loc = np.min(data)
    high = np.max(data)
    n = int(high - loc)
    mu = np.mean(data - loc)
    p = mu / n
    return (n, p, loc)

def fit_poisson(data: np.ndarray) -> tuple[float, int]:
    loc = np.min(data)
    mu = np.mean(data - loc)
    return (mu, loc)

def fit_geom(data: np.ndarray) -> tuple[float, int]:
    loc = np.min(data)
    mu = np.mean(data - loc + 1)
    return (1/mu, loc-1)

def fit_uniform(data: np.ndarray) -> tuple[int, int, int]:
    low = np.min(data)
    high = np.max(data)+1
    return (low, high, 0)

def plot_distr_fit(data: np.ndarray, distr: stats.rv_discrete, params: tuple):
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
