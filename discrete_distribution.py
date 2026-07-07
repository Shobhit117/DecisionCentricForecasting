import numpy as np
from scipy import stats
import probability_utils as prob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
        self.pmf = pmf / np.sum(pmf)
        self.min = min_value
        if (distr_name is not None) and (distr_name not in SUPPORTED_DISTRS.keys()):
            raise ValueError(f"Distribution {distr_name} is not supported. Available distributions are: {list(SUPPORTED_DISTRS.keys())}.")
        self.best_fit = distr_name
        self.best_fit_params = distr_params
        self.bin_size = bin_size
    
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
        self.pmf = padded_pmf.reshape(-1, level).sum(axis=1)
        self.bin_size *= level
    
    def split_bins(self, level : int = None):
        if level == 1:
            return
        if level is None:
            level = self.bin_size
        if self.bin_size % level != 0:
            raise ValueError(f"Failed to split_bins pmf by {level} levels. The bin size must be divisible by the number of levels to split_bins.")

        if self.best_fit is None or self.best_fit_params is None:
            self.pmf = np.repeat(self.pmf / level, level)
            self.bin_size //= level
        else:
            self.pmf = SUPPORTED_DISTRS[self.best_fit].pmf(np.arange(self.min, self.max+1), *self.best_fit_params)
            self.pmf /= np.sum(self.pmf)
            new_bin_size = self.bin_size // level
            self.bin_size = 1
            self.merge_bins(new_bin_size)
    
    def quantile(self, q: list | np.ndarray | float) -> np.ndarray:
        q_arr = np.atleast_1d(q)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("Quantiles must be between 0 and 1.")
        # searchsorted with side='left' gives the first index where cdf >= q
        indices = np.searchsorted(self.cdf, q_arr, side='left')
        # Protect against minor floating point errors (e.g., if cdf[-1] is 0.9999999999)
        indices = np.clip(indices, 0, len(self.pmf) - 1)
        return self.min + indices * self.bin_size

    def __add__(self, other : "DiscreteDistribution") -> "DiscreteDistribution":
        if self.bin_size != other.bin_size:
            # self = DiscreteDistribution(self.pmf.copy(), self.min, self.bin_size, self.best_fit, self.best_fit_params)
            # other = DiscreteDistribution(other.pmf.copy(), other.min, other.bin_size, other.best_fit, other.best_fit_params)
            self = self.copy()
            other = other.copy()
            new_bin_size = int(np.gcd(self.bin_size, other.bin_size))
            self.split_bins(self.bin_size // new_bin_size)
            other.split_bins(other.bin_size // new_bin_size)
        else:
            new_bin_size = self.bin_size
        new_min = self.min + other.min
        new_pmf = np.convolve(self.pmf, other.pmf)
        new_pmf /= np.sum(new_pmf)
        return DiscreteDistribution(new_pmf, new_min, new_bin_size)

    def _align_bin_size(self):
        if (self.bin_size == 1) or (self.min % self.bin_size == 0):
            return
        original_bin_size = self.bin_size
        self.split_bins()
        new_min = original_bin_size * (self.min // original_bin_size)
        shift = self.min - new_min
        self.min -= int(shift)
        self.pmf = np.pad(self.pmf, (shift, 0), mode='constant')
        self.merge_bins(original_bin_size)
    
    def _trim_pmf(self, tol : float = 0.0):
        i = len(self.pmf) - 1
        while self.pmf[i] <= tol:
            i -= 1
        self.pmf = self.pmf[:(i+1)]
        self.pmf /= np.sum(self.pmf)

    def shift(self, by : int):
        self.min += by
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
            # other = DiscreteDistribution(other.pmf, other.min, other.bin_size, other.best_fit, other.best_fit_params)
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

    def random_sum(self, rv_list: list["DiscreteDistribution"], default_distr : "DiscreteDistribution" = None) -> "DiscreteDistribution":
        if self.min < 0:
            raise ValueError("Number of summands must have non-negative support.")
        if self.max > len(rv_list):
            rv_list.extend([default_distr for _ in range(self.max - len(rv_list))])
        if self.bin_size != 1:
            self = DiscreteDistribution(self.pmf.copy(), self.min, self.bin_size, self.best_fit, self.best_fit_params)
            self.split_bins()
        
        bin_sizes = [rv.bin_size for rv in rv_list]
        bin_size_gcd = np.gcd.reduce(bin_sizes)
        for rv in rv_list:
            rv.split_bins(rv.bin_size // bin_size_gcd) # ensure equal bin size
            rv._align_bin_size() # ensure that the min value is divisible by the bin size

        # compute sum of RVs:
        rv_sum_list = []
        curr = DiscreteDistribution(np.array([1.0]), 0)
        for n in range(0, self.max + 1):
            if n >= self.min:
                rv_sum_list.append(curr)
            if n != self.max:
                curr = curr + rv_list[n]
        
        # compute overall min and max values:
        overall_min = np.inf
        overall_max = -np.inf
        for rv in rv_sum_list:
            overall_min = min(overall_min, rv.min)
            overall_max = max(overall_max, rv.max)
        
        # compute the distribution of the random sum:
        bin_size = rv_sum_list[0].bin_size
        height = len(self.pmf)
        width = int(np.ceil((1 + overall_max - overall_min) / bin_size))
        mat = np.zeros((height, width))
        for n, rv in enumerate(rv_sum_list):
            start = (rv.min - overall_min) // bin_size
            end = start + len(rv.pmf)
            mat[n, start:end] = rv.pmf
        pmf = self.pmf @ mat
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

# N = DiscreteDistribution(
#     pmf=np.array([0.5, 0.5]),
#     min_value=1
# )

# # support = {3, 5}
# D = DiscreteDistribution(
#     pmf=np.array([0.7, 0.3]),
#     min_value=3,
#     bin_size=2
# )

# S = N @ D

# # print(S.min)
# # print(S.bin_size)
# print(S.pmf)
# # print(N.random_sum([D.copy(), D.copy(), D.copy(), D.copy(), D.copy()]).pmf)
# S.plot()
# plt.show()