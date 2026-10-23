import numpy as np
from scipy import stats
import probability_utils as prob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numba.typed import List as NumbaList

SUPPORTED_DISTRS = {'nbinom': stats.nbinom,
                    'binom': stats.binom,
                    'poisson': stats.poisson,
                    'geom': stats.geom,
                    'uniform': stats.randint}

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

class DiscreteDistribution:
    def __init__(self, pmf: np.ndarray, min_value: int, bin_size: int = 1, distr_name: str = None, distr_params: tuple = None):
        self.pmf = np.ascontiguousarray(pmf / np.sum(pmf), dtype=np.float64)
        self.min = int(min_value)
        if (distr_name is not None) and (distr_name not in SUPPORTED_DISTRS.keys()):
            raise ValueError(f"Distribution {distr_name} is not supported. Available distributions are: {list(SUPPORTED_DISTRS.keys())}.")
        self.best_fit = distr_name
        self.best_fit_params = distr_params
        self.bin_size = int(bin_size)
    
    def copy(self):
        return DiscreteDistribution(self.pmf.copy(), self.min, self.bin_size, self.best_fit, self.best_fit_params)
    
    @property
    def max(self) -> int:
        return self.min + self.bin_size * len(self.pmf) - 1

    @property
    def cdf(self) -> np.ndarray:
        return np.cumsum(self.pmf)

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
            pmf = SUPPORTED_DISTRS[best_fit].pmf(np.arange(min_value, max_value+1), *best_fit_params)
            pmf /= np.sum(pmf)
        else:
            raise ValueError(f"Invalid probability distribution initialization method: {method}. Available methods are: 'actual' and 'best_fit'.")
        return cls(pmf, min_value, bin_size, best_fit, best_fit_params)
    
    @classmethod
    def from_parametric(cls, distr_name : str, params : tuple, tol : float = 1e-4):
        if distr_name not in SUPPORTED_DISTRS.keys():
            raise ValueError(f"Distribution: {distr_name} is not supported.")
        distr = SUPPORTED_DISTRS[distr_name]
        low, high = distr.support(*params)
        min_value = int(low)
        max_value = int(high) if np.isfinite(high) else int(distr.ppf(1-tol, *params))
        pmf = distr.pmf(np.arange(min_value, max_value+1), *params)
        pmf /= np.sum(pmf)
        return cls(pmf, min_value, 1, distr_name, params)
    
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
        from ddlt import _pmf_quantiles_numba

        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("Quantiles must be between 0 and 1.")
        return _pmf_quantiles_numba(self.pmf, self.min, self.bin_size, q_arr)

    def __add__(self, other : "DiscreteDistribution") -> "DiscreteDistribution":
        from ddlt import DEFAULT_TAIL_TOL, _convolve_1d_numba, _trim_pmf_numba

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
        return DiscreteDistribution(new_pmf, new_min, new_bin_size)

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
        from ddlt import _trim_pmf_numba

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
        from ddlt import _random_sum_aligned_numba

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
        return DiscreteDistribution(pmf, overall_min, bin_size)

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
