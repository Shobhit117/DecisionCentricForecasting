from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy import signal, stats
from probabilistic_modeling.distributions import probability_utils as prob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numba.typed import List as NumbaList

# Product of support lengths above which FFT convolution is faster than O(N*M) direct loop:
FFT_CONVOLVE_THRESHOLD = 65_536

SUPPORTED_DISTRS = {'nbinom': stats.nbinom,
                    'binom': stats.binom,
                    'poisson': stats.poisson,
                    'geom': stats.geom,
                    'uniform': stats.randint}


class TruncatedSupportWarning(UserWarning):
    """
    Raised when a quantile is requested beyond ``1 - truncation_tol``: the distribution's
    finite support dropped that much upper-tail mass, so the returned value may be biased low.
    Rebuild the distribution with a smaller ``tol`` (or pass ``max_service_level``).
    """


def _combine_truncation(*tols: "float | None") -> "float | None":
    """Upper bound on the truncated tail mass of a sum of independent truncated variables."""
    if any(t is None for t in tols):
        return None
    return float(sum(tols))


def _lump_upper_tail(distr, params: tuple, min_value: int, max_value: int) -> np.ndarray:
    """
    PMF on ``[min_value, max_value]`` with the truncated upper-tail mass ``P(X > max_value)``
    added to the last bin (instead of renormalising the whole PMF).

    Renormalising by ``1 - tail`` would inflate the CDF at *every* support point by up to
    ``tail`` and bias all high quantiles low. Lumping keeps ``F(x)`` exact for every
    ``x < max_value``, so quantiles are exact up to ``F(max_value - 1)`` (~``1 - tol``).
    """
    pmf = distr.pmf(np.arange(min_value, max_value + 1), *params).astype(np.float64)
    pmf[-1] += float(distr.sf(max_value, *params))
    return pmf


class PredictiveDistribution(ABC):
    """
    Common interface for 1D predictive probability distributions (both discrete and continuous).
    """

    @abstractmethod
    def mean(self) -> float:
        """Expected value E[X]."""
        ...

    @abstractmethod
    def variance(self) -> float:
        """Variance Var(X)."""
        ...

    def std(self) -> float:
        """Standard deviation sqrt(Var(X))."""
        return float(np.sqrt(max(0.0, self.variance())))

    @abstractmethod
    def quantile(self, q: list | np.ndarray | float) -> np.ndarray:
        """Evaluates the quantile function F^{-1}(q) at one or more probability levels in [0, 1]."""
        ...

    @abstractmethod
    def cdf_at(self, x: list | np.ndarray | float) -> np.ndarray:
        """Evaluates the cumulative distribution function P(X <= x) at one or more values."""
        ...

    @abstractmethod
    def crps(self, y: float | list | np.ndarray) -> float | np.ndarray:
        """Continuous/Discrete Ranked Probability Score for observation(s) ``y``."""
        ...

    @abstractmethod
    def sample(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Draws ``size`` random samples from the distribution."""
        ...

def compute_best_fit_distribution(data : np.ndarray) -> tuple[str, tuple[float]]:
    best_fit = None
    best_fit_params = None
    best_rmse = np.inf
    for distr_name, distr in SUPPORTED_DISTRS.items():
        if distr_name == 'nbinom':
            params = prob.fit_nbinom(data)
        elif distr_name == 'binom':
            params = prob.fit_binom(data)
        elif distr_name == 'poisson':
            params = prob.fit_poisson(data)
        elif distr_name == 'geom':
            params = prob.fit_geom(data)
        elif distr_name == 'uniform':
            params = prob.fit_uniform(data)
        rmse = prob.rmse_discrete(data, distr, params)
        if rmse < best_rmse:
            best_rmse = rmse
            best_fit = distr_name
            best_fit_params = params
    return best_fit, best_fit_params

class DiscreteDistribution(PredictiveDistribution):
    def __init__(self, pmf: np.ndarray, min_value: int, bin_size: int = 1, distr_name: str = None, distr_params: tuple = None,
                 truncation_tol: float | None = None):
        self.pmf = np.ascontiguousarray(pmf / np.sum(pmf), dtype=np.float64)
        self.min = int(min_value)
        if (distr_name is not None) and (distr_name not in SUPPORTED_DISTRS.keys()):
            raise ValueError(f"Distribution {distr_name} is not supported. Available distributions are: {list(SUPPORTED_DISTRS.keys())}.")
        self.best_fit = distr_name
        self.best_fit_params = distr_params
        self.bin_size = int(bin_size)
        # Upper bound on the upper-tail probability mass dropped when building this finite
        # support (``None`` = unknown, ``0.0`` = exact). Quantiles above ``1 - truncation_tol``
        # trigger a ``TruncatedSupportWarning``.
        self.truncation_tol = truncation_tol
    
    def copy(self):
        return DiscreteDistribution(self.pmf.copy(), self.min, self.bin_size, self.best_fit, self.best_fit_params,
                                    truncation_tol=self.truncation_tol)
    
    @property
    def max(self) -> int:
        return self.min + self.bin_size * len(self.pmf) - 1

    @property
    def support(self) -> np.ndarray:
        return self.min + self.bin_size * np.arange(len(self.pmf), dtype=np.int64)

    @property
    def cdf(self) -> np.ndarray:
        return np.cumsum(self.pmf)

    def mean(self) -> float:
        return float(np.dot(self.support, self.pmf))

    def variance(self) -> float:
        sup = self.support.astype(np.float64)
        mu = float(np.dot(sup, self.pmf))
        return float(np.dot((sup - mu) ** 2, self.pmf))

    def cdf_at(self, x: list | np.ndarray | float) -> np.ndarray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        idx = np.floor((x_arr - self.min) / self.bin_size).astype(np.int64)
        cdf_vals = self.cdf
        out = np.zeros(len(x_arr), dtype=np.float64)
        in_range = (idx >= 0) & (idx < len(cdf_vals))
        out[in_range] = cdf_vals[idx[in_range]]
        out[idx >= len(cdf_vals)] = 1.0
        return out

    def crps(self, y: float | list | np.ndarray) -> float | np.ndarray:
        from probabilistic_modeling.ddlt import _pmf_crps_numba, _pmf_crps_scalar_numba

        if np.ndim(y) == 0:
            return float(_pmf_crps_scalar_numba(self.pmf, self.min, self.bin_size, float(y)))
        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        return _pmf_crps_numba(self.pmf, self.min, self.bin_size, y_arr).reshape(np.shape(y))

    def sample(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        # Inverse-CDF sampling via the adaptive quantile kernel (CDF + binary search for large
        # ``size``). Bypasses the truncation warning: occasional draws above ``1 - tol`` are
        # expected and only cap a vanishing fraction of samples at the support maximum.
        from probabilistic_modeling.ddlt import _pmf_quantiles_numba

        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        u = gen.random(size=size)
        flat = _pmf_quantiles_numba(self.pmf, self.min, self.bin_size, np.ascontiguousarray(np.ravel(u)))
        return flat.reshape(np.shape(u))

    @classmethod
    def from_data(cls, data: np.ndarray, method : str ='actual', tol : float = 1e-4):
        data = np.array(data, dtype=int)
        min_value = np.min(data)
        max_value = np.max(data)
        bin_size = 1

        if method == 'actual':
            (values, counts) = np.unique(data, return_counts=True)
            counts = counts / np.sum(counts)
            hist = dict(zip(values, counts))
            pmf = np.zeros(max_value - min_value + 1)
            for i in range(len(pmf)):
                pmf[i] = hist.get(i + min_value, 0)
            best_fit = None
            best_fit_params = None
        
        elif method == 'best_fit':
            best_fit, best_fit_params = compute_best_fit_distribution(data)
            max_value = int(SUPPORTED_DISTRS[best_fit].ppf(1 - tol, *best_fit_params))
            pmf = _lump_upper_tail(SUPPORTED_DISTRS[best_fit], best_fit_params, min_value, max_value)
            return cls(pmf, min_value, bin_size, best_fit, best_fit_params, truncation_tol=float(tol))
        else:
            raise ValueError(f"Invalid probability distribution initialization method: {method}. Available methods are: 'actual' and 'best_fit'.")
        return cls(pmf, min_value, bin_size, best_fit, best_fit_params, truncation_tol=0.0)
    
    @classmethod
    def from_parametric(cls, distr_name : str, params : tuple, tol : float | None = None,
                        max_service_level: float | None = None):
        """
        Builds a discrete distribution from a supported parametric family.

        Unbounded supports are truncated at the ``1 - tol`` quantile. ``tol`` defaults to
        ``DEFAULT_SUPPORT_TOL`` (1e-4) and is automatically tightened when
        ``max_service_level`` is given so that quantiles up to that level stay exact.
        """
        from probabilistic_modeling.ddlt import DEFAULT_SUPPORT_TOL, support_tol_for_service_level

        if distr_name not in SUPPORTED_DISTRS.keys():
            raise ValueError(f"Distribution: {distr_name} is not supported.")
        base_tol = DEFAULT_SUPPORT_TOL if tol is None else float(tol)
        tol = support_tol_for_service_level(max_service_level, base_tol=base_tol)
        distr = SUPPORTED_DISTRS[distr_name]
        low, high = distr.support(*params)
        min_value = int(low)
        bounded = bool(np.isfinite(high))
        max_value = int(high) if bounded else int(distr.ppf(1-tol, *params))
        pmf = _lump_upper_tail(distr, params, min_value, max_value)
        return cls(pmf, min_value, 1, distr_name, params, truncation_tol=0.0 if bounded else tol)
    
    def merge_bins(self, level : int = 10):
        if level == 1:
            return
        pad_size = (level - (len(self.pmf) % level)) % level
        padded_pmf = self.pmf
        if pad_size > 0:
            padded_pmf = np.pad(self.pmf, (0, pad_size), mode='constant')
        self.pmf = np.ascontiguousarray(padded_pmf.reshape(-1, level).sum(axis=1), dtype=np.float64)
        self.bin_size *= level
    
    def split_bins(self, level : int = None):
        if level == 1:
            return
        if level is None:
            level = self.bin_size
        if self.bin_size % level != 0:
            raise ValueError(f"Failed to split_bins pmf by {level} levels. The bin size must be divisible by the number of levels to split_bins.")

        if self.best_fit is None or self.best_fit_params is None:
            self.pmf = np.ascontiguousarray(np.repeat(self.pmf / level, level), dtype=np.float64)
            self.bin_size //= level
        else:
            self.pmf = SUPPORTED_DISTRS[self.best_fit].pmf(np.arange(self.min, self.max+1), *self.best_fit_params)
            self.pmf = np.ascontiguousarray(self.pmf / np.sum(self.pmf), dtype=np.float64)
            new_bin_size = self.bin_size // level
            self.bin_size = 1
            self.merge_bins(new_bin_size)
    
    def quantile(self, q: list | np.ndarray | float) -> np.ndarray:
        """
        Discrete quantiles ``min{x : F(x) >= q}``.

        Uses an adaptive Numba kernel: an early-exit linear scan for few queries, and a
        single CDF build + binary search per query when many quantiles are requested
        (e.g. ``sample(size=...)``). Results are identical in both modes.
        """
        from probabilistic_modeling.ddlt import _pmf_quantiles_numba

        q_arr = np.ascontiguousarray(np.atleast_1d(np.asarray(q, dtype=np.float64)).ravel())
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("Quantiles must be between 0 and 1.")
        trunc = self.truncation_tol
        if trunc is not None and trunc > 0.0 and np.any(q_arr > 1.0 - trunc):
            warnings.warn(
                f"Quantile level {float(np.max(q_arr)):.6g} exceeds 1 - truncation_tol ({1.0 - trunc:.6g}); "
                "the finite support dropped up to that much upper-tail mass, so the result may be biased low. "
                "Rebuild the distribution with a smaller 'tol' or pass 'max_service_level'.",
                TruncatedSupportWarning,
                stacklevel=2,
            )
        return _pmf_quantiles_numba(self.pmf, self.min, self.bin_size, q_arr)

    def __add__(self, other : "DiscreteDistribution") -> "DiscreteDistribution":
        from probabilistic_modeling.ddlt import DEFAULT_TAIL_TOL, _convolve_1d_numba, _trim_pmf_numba

        trunc = _combine_truncation(self.truncation_tol, other.truncation_tol)
        if self.bin_size != other.bin_size:
            self = self.copy()
            other = other.copy()
            new_bin_size = int(np.gcd(self.bin_size, other.bin_size))
            self.split_bins(self.bin_size // new_bin_size)
            other.split_bins(other.bin_size // new_bin_size)
        else:
            new_bin_size = self.bin_size
        new_min = self.min + other.min
        new_pmf = _convolve_1d_numba(self.pmf, other.pmf)
        new_pmf /= np.sum(new_pmf)
        new_pmf, new_min = _trim_pmf_numba(new_pmf, new_min, new_bin_size, DEFAULT_TAIL_TOL)
        return DiscreteDistribution(new_pmf, new_min, new_bin_size, truncation_tol=trunc)

    def _align_bin_size(self):
        if (self.bin_size == 1) or (self.min % self.bin_size == 0):
            return
        original_bin_size = self.bin_size
        self.split_bins()
        new_min = original_bin_size * (self.min // original_bin_size)
        shift = self.min - new_min
        self.min -= int(shift)
        self.pmf = np.ascontiguousarray(np.pad(self.pmf, (shift, 0), mode='constant'), dtype=np.float64)
        self.merge_bins(original_bin_size)
    
    def _trim_pmf(self, tol : float = 1e-9):
        from probabilistic_modeling.ddlt import _trim_pmf_numba

        self.pmf, self.min = _trim_pmf_numba(self.pmf, self.min, self.bin_size, tol)

    def shift(self, by : int):
        self.min += int(by)
        if (self.best_fit is not None) and (self.best_fit_params is not None):
            params = list(self.best_fit_params)
            params[-1] += by # shift the location parameter
            self.best_fit_params = tuple(params)

    def __matmul__(self, other : "DiscreteDistribution") -> "DiscreteDistribution":
        if self.min < 0:
            raise ValueError("Left operand (number of summands) must have non-negative support.")
        
        if other.min < 0:
            raise ValueError("Summand distribution must have non-negative support.")
        
        if self.bin_size != 1:
            self.split_bins()
        
        if other.min % other.bin_size != 0:
            other = other.copy()
            other._align_bin_size()
        
        overall_min = self.min * other.min
        overall_max = self.max * other.max
        
        curr = np.array([1.0])
        step_size = other.min // other.bin_size
        height = len(self.pmf)
        width = int(np.ceil((1 + overall_max - overall_min) / other.bin_size))
        start = 0
        mat = np.zeros((height, width))
        for n in range(0, self.max + 1):
            if n >= self.min:
                end = start + len(curr)
                mat[n - self.min, start:end] = curr
                start += step_size
            if n == self.max:
                break
            curr = np.convolve(curr, other.pmf)
        pmf = self.pmf @ mat
        sum_distr = DiscreteDistribution(pmf, overall_min, other.bin_size)
        sum_distr._trim_pmf(0)
        return sum_distr

    def random_sum(
        self,
        rv_list: list["DiscreteDistribution"],
        default_distr: "DiscreteDistribution" = None,
        tail_tol: float = 1e-9,
    ) -> "DiscreteDistribution":
        from probabilistic_modeling.ddlt import _random_sum_aligned_numba

        if self.min < 0:
            raise ValueError("Number of summands must have non-negative support.")
        
        if self.max > len(rv_list):
            if default_distr is None:
                raise ValueError(f"Number of RVs in rv_list: {len(rv_list)}. Expected: {self.max}.")
            rv_list = list(rv_list) + [default_distr] * (self.max - len(rv_list))
        else:
            rv_list = list(rv_list[:self.max])
        
        if self.bin_size != 1:
            self = self.copy()
            self.split_bins()

        if self.max == 0:
            return DiscreteDistribution(np.array([1.0]), 0, 1)

        # Fast path when all summands already have bin_size == 1 (no copy/alignment needed):
        all_unit_bins = all(rv.bin_size == 1 for rv in rv_list)
        if all_unit_bins:
            bin_size = 1
            aligned_rvs = rv_list
        else:
            bin_sizes = [rv.bin_size for rv in rv_list]
            bin_size = int(np.gcd.reduce(bin_sizes))
            aligned_rvs = []
            for rv in rv_list:
                rv_copy = rv.copy()
                rv_copy.split_bins(rv_copy.bin_size // bin_size)
                rv_copy._align_bin_size()
                aligned_rvs.append(rv_copy)

        rv_pmfs = NumbaList()
        rv_mins = np.empty(len(aligned_rvs), dtype=np.int64)
        for idx, rv in enumerate(aligned_rvs):
            rv_pmfs.append(rv.pmf)
            rv_mins[idx] = rv.min

        pmf, overall_min = _random_sum_aligned_numba(self.pmf, self.min, rv_pmfs, rv_mins, bin_size, tail_tol)
        # Each outcome sums at most ``self.max`` truncated summands:
        rv_truncs = [rv.truncation_tol for rv in aligned_rvs]
        trunc = None if any(t is None for t in rv_truncs) else float(self.max * max(rv_truncs, default=0.0))
        return DiscreteDistribution(pmf, overall_min, bin_size, truncation_tol=trunc)

    def plot(self, label : str = None
                 , title : str = None
                 , xlabel : str = None
                 , ylabel : str = None):
        plt.figure()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        shift = self.bin_size / 2 if self.bin_size > 1 else 0
        x = np.arange(self.min, self.max + 1, self.bin_size) + shift
        plt.bar(x, self.pmf, width=self.bin_size, label=label, color='orange')
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
