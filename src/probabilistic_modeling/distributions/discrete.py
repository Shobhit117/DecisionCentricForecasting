"""
Discrete probability distributions and distribution algebra for count demand and lead times.

What this module does
---------------------
Defines:
- :class:`PredictiveDistribution`: the abstract interface shared by discrete and continuous
  predictive distributions (`mean`, `variance`, `std`, `quantile`, `cdf_at`, `crps`, `sample`).
- :class:`DiscreteDistribution`: a 1D discrete probability distribution over an integer grid
  ``min, min + bin_size, ..., max`` with full distribution algebra:
  * ``d1 + d2``: independent sum via Numba-accelerated 1D convolution;
  * ``N @ X``: compound random sum ``X_1 + ... + X_N`` of i.i.d. copies of ``X``;
  * ``N.random_sum([X_1, X_2, ...])``: compound random sum of *non-identically* distributed
    daily demands over a random lead time ``N``;
  * ``shift``, ``merge_bins``, ``split_bins``, ``from_data``, ``from_parametric``.
- :class:`TruncatedSupportWarning` and upper-tail lumping utilities that track and warn when
  high quantiles exceed a finite support's truncation boundary.

Where this module is used
-------------------------
Used across every layer of the library (`ddlt.py`, `time_series.py`, `models/`, `inventory/`,
`evaluation/`).
"""
from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy import stats
from probabilistic_modeling.distributions import probability_utils as prob
from numba.typed import List as NumbaList

# Mapping from supported parametric family names to their `scipy.stats` discrete distributions.
# Used by `compute_best_fit_distribution`, `DiscreteDistribution.from_parametric`, and `split_bins`.
SUPPORTED_DISTRS = {'nbinom': stats.nbinom,
                    'binom': stats.binom,
                    'poisson': stats.poisson,
                    'geom': stats.geom,
                    'uniform': stats.randint}


class TruncatedSupportWarning(UserWarning):
    """
    Warning emitted by :meth:`DiscreteDistribution.quantile` when a requested quantile level
    ``q`` exceeds ``1 - truncation_tol``.

    Why it is needed
    ----------------
    Unbounded distributions (Poisson, NB2, exponential quantile tails) must be truncated to a
    finite integer array by cutting off ``truncation_tol`` upper-tail probability mass. Asking
    for a quantile inside that cut-off tail (e.g. ``q = 0.99995`` when ``truncation_tol = 1e-4``)
    returns the support cap rather than the true unbounded quantile. Emitting a warning alerts
    the caller to pass a smaller ``tol`` or ``max_service_level``.
    """


def _combine_truncation(*tols: "float | None") -> "float | None":
    """
    Upper bound on the truncated upper-tail probability mass of a sum of independent truncated RVs.

    By the union bound, if ``X`` truncated at most ``tol_X`` upper-tail mass and ``Y`` truncated
    at most ``tol_Y``, then ``X + Y`` is affected above probability ``1 - (tol_X + tol_Y)``.
    Returns ``None`` if any operand has unknown truncation (``None``).

    Used in :meth:`DiscreteDistribution.__add__`.
    """
    if any(t is None for t in tols):
        return None
    return float(sum(tols))


def _lump_upper_tail(distr, params: tuple, min_value: int, max_value: int) -> np.ndarray:
    """
    Evaluate ``distr.pmf`` on ``[min_value, max_value]`` and lump the truncated upper-tail
    probability mass ``P(X > max_value) = distr.sf(max_value, *params)`` into the final bin.

    Why lumping is used instead of renormalizing
    --------------------------------------------
    If we instead divided ``pmf`` by ``1 - tail`` (renormalization), every interior CDF value
    ``F(x)`` would be inflated to ``F(x) / (1 - tail)``, which biases *all* high quantiles
    downward. By adding ``P(X > max_value)`` exclusively to ``pmf[-1]``, the CDF ``F(x)`` stays
    **100% exact** for every ``x < max_value``, so quantiles are exact all the way up to
    ``F(max_value - 1) >= 1 - tol``.

    Where it is used
    ----------------
    Called by :meth:`DiscreteDistribution.from_data` (`method='best_fit'`) and
    :meth:`DiscreteDistribution.from_parametric`.
    """
    pmf = distr.pmf(np.arange(min_value, max_value + 1), *params).astype(np.float64)
    pmf[-1] += float(distr.sf(max_value, *params))
    return pmf


class PredictiveDistribution(ABC):
    """
    Abstract base class defining the common interface for 1D predictive probability distributions.

    Implemented by:
    - :class:`DiscreteDistribution` (integer-valued demand and lead-time distributions)
    - :class:`probabilistic_modeling.distributions.quantile.ContinuousQuantileDistribution`
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
    """
    Fit all 5 candidate parametric families in ``SUPPORTED_DISTRS`` (`nbinom`, `binom`,
    `poisson`, `geom`, `uniform`) to ``data`` via method-of-moments and return the family
    ``(best_fit_name, best_fit_params)`` that minimizes empirical PMF RMSE.

    Why it is needed
    ----------------
    When the replenishment protection window ``max(L + R)`` is longer than the model's explicit
    forecast horizon ``H`` (e.g. ``H = 7`` days but ``L + R`` can reach 9 days), :class:`DDLTCalculator`
    fills days ``H + 1 .. max(L + R)`` using each series' historical best-fit distribution. Also
    optionally used as a categorical feature (`use_best_fit_feature=True`) in :class:`DiscreteTimeSeries`.

    Where it is used
    ----------------
    - :meth:`DiscreteTimeSeries._fit_distributions`
    - :meth:`DiscreteDistribution.from_data` (when ``method='best_fit'``)
    """
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
    """
    1D discrete probability distribution over an integer grid ``min, min + bin_size, ..., max``.

    Attributes
    ----------
    pmf : np.ndarray (1D float64, contiguous)
        Probability mass function normalized to sum to ``1.0``; ``pmf[i] = P(X = min + i * bin_size)``.
    min : int
        Minimum support value corresponding to ``pmf[0]``.
    bin_size : int, default 1
        Step size between consecutive bins (``1`` for standard unit-integer grids; ``> 1`` after
        calling :meth:`merge_bins`).
    best_fit : str | None
        Optional name of the parametric family (`'nbinom'`, `'poisson'`, etc.) that generated this PMF.
    best_fit_params : tuple | None
        Optional SciPy parameter tuple for ``best_fit``.
    truncation_tol : float | None
        Upper bound on the upper-tail probability mass truncated when constructing the finite
        support (``0.0`` = exact bounded support, ``None`` = unknown). Used by :meth:`quantile`
        to emit :class:`TruncatedSupportWarning` if a query exceeds ``1 - truncation_tol``.
    """

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
        """Return a deep copy of this distribution."""
        return DiscreteDistribution(self.pmf.copy(), self.min, self.bin_size, self.best_fit, self.best_fit_params,
                                    truncation_tol=self.truncation_tol)
    
    @property
    def max(self) -> int:
        """Largest integer covered by the support (inclusive)."""
        return self.min + self.bin_size * len(self.pmf) - 1

    @property
    def support(self) -> np.ndarray:
        """1D integer array of grid points ``min, min + bin_size, ..., min + (len(pmf)-1)*bin_size``."""
        return self.min + self.bin_size * np.arange(len(self.pmf), dtype=np.int64)

    @property
    def cdf(self) -> np.ndarray:
        """Cumulative distribution function ``F(x) = cumsum(pmf)`` evaluated at ``self.support``."""
        return np.cumsum(self.pmf)

    def mean(self) -> float:
        """Expected value ``E[X] = sum_i support[i] * pmf[i]``."""
        return float(np.dot(self.support, self.pmf))

    def variance(self) -> float:
        """Variance ``Var(X) = sum_i (support[i] - E[X])^2 * pmf[i]``."""
        sup = self.support.astype(np.float64)
        mu = float(np.dot(sup, self.pmf))
        return float(np.dot((sup - mu) ** 2, self.pmf))

    def cdf_at(self, x: list | np.ndarray | float) -> np.ndarray:
        """
        Evaluate ``P(X <= x)`` at one or more values ``x`` (returns ``0.0`` below ``self.min``
        and ``1.0`` at or above the last bin).
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        idx = np.floor((x_arr - self.min) / self.bin_size).astype(np.int64)
        cdf_vals = self.cdf
        out = np.zeros(len(x_arr), dtype=np.float64)
        in_range = (idx >= 0) & (idx < len(cdf_vals))
        out[in_range] = cdf_vals[idx[in_range]]
        out[idx >= len(cdf_vals)] = 1.0
        return out

    def crps(self, y: float | list | np.ndarray) -> float | np.ndarray:
        """
        Compute the exact Continuous/Discrete Ranked Probability Score (CRPS) for observation(s) ``y``
        using the Numba-compiled step-function integral (:func:`_pmf_crps_scalar_numba`).
        """
        from probabilistic_modeling.ddlt import _pmf_crps_numba, _pmf_crps_scalar_numba

        if np.ndim(y) == 0:
            return float(_pmf_crps_scalar_numba(self.pmf, self.min, self.bin_size, float(y)))
        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
        return _pmf_crps_numba(self.pmf, self.min, self.bin_size, y_arr).reshape(np.shape(y))

    def sample(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """
        Draw ``size`` random integer samples via inverse-CDF transform (`F^{-1}(U)` with `U ~ Uniform(0,1)`).

        Uses :func:`_pmf_quantiles_numba` directly (which builds the CDF once and binary-searches
        when ``size >= 8``) and intentionally bypasses :class:`TruncatedSupportWarning` because
        random ``U`` draws occasionally exceed ``1 - truncation_tol`` by chance.
        """
        from probabilistic_modeling.ddlt import _pmf_quantiles_numba

        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        u = gen.random(size=size)
        flat = _pmf_quantiles_numba(self.pmf, self.min, self.bin_size, np.ascontiguousarray(np.ravel(u)))
        return flat.reshape(np.shape(u))

    @classmethod
    def from_data(cls, data: np.ndarray, method : str ='actual', tol : float = 1e-4):
        """
        Construct a :class:`DiscreteDistribution` from an empirical sample ``data`` of integers.

        Parameters
        ----------
        data : array-like of int
            Observed count values (e.g. historical supplier lead times or daily sales).
        method : {'actual', 'best_fit'}, default 'actual'
            - ``'actual'``: builds the exact empirical histogram over ``[min(data), max(data)]``
              with ``truncation_tol = 0.0``.
            - ``'best_fit'``: fits the best parametric family via :func:`compute_best_fit_distribution`
              and truncates its upper tail at ``1 - tol``.
        tol : float, default 1e-4
            Upper-tail truncation tolerance when ``method='best_fit'``.

        Where it is used
        ----------------
        Called by :func:`probabilistic_modeling.time_series.compute_distribution_dict` to build
        per-store lead-time distributions from historical shipment records.
        """
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
        Build a :class:`DiscreteDistribution` from a parametric family in ``SUPPORTED_DISTRS``
        (`'nbinom'`, `'binom'`, `'poisson'`, `'geom'`, `'uniform'`).

        Unbounded supports are truncated at the ``1 - tol`` quantile and the remaining tail mass
        is lumped into the last bin via :func:`_lump_upper_tail`. ``tol`` defaults to
        ``DEFAULT_SUPPORT_TOL`` (`1e-4`) and is automatically tightened via
        :func:`support_tol_for_service_level` when ``max_service_level`` is provided.
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
    
    def merge_bins(self, level : int = 10, inplace: bool = True) -> "DiscreteDistribution":
        """
        Coarsen the distribution by summing every ``level`` consecutive bins into one bin
        of width ``bin_size * level``.

        Why it is needed
        ----------------
        Useful for reducing support length on very wide distributions before plotting or
        coarse approximations. Mutates ``self`` unless ``inplace=False``.
        """
        target = self if inplace else self.copy()
        target._merge_bins_inplace(level)
        return target

    def _merge_bins_inplace(self, level: int) -> None:
        if level == 1:
            return
        pad_size = (level - (len(self.pmf) % level)) % level
        padded_pmf = self.pmf
        if pad_size > 0:
            padded_pmf = np.pad(self.pmf, (0, pad_size), mode='constant')
        self.pmf = np.ascontiguousarray(padded_pmf.reshape(-1, level).sum(axis=1), dtype=np.float64)
        self.bin_size *= level
    
    def split_bins(self, level : int = None, inplace: bool = True) -> "DiscreteDistribution":
        """
        Refine each bin into ``level`` sub-bins of width ``bin_size // level`` (defaults to
        splitting all the way down to unit bins ``bin_size == 1``).

        If ``self.best_fit`` is known, re-evaluates the parametric PMF on the finer grid;
        otherwise splits each bin's mass uniformly across its ``level`` sub-bins.
        """
        target = self if inplace else self.copy()
        target._split_bins_inplace(level)
        return target

    def _split_bins_inplace(self, level: int | None) -> None:
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
            self._merge_bins_inplace(new_bin_size)
    
    def quantile(self, q: list | np.ndarray | float) -> np.ndarray:
        """
        Evaluate discrete quantile(s) ``F^{-1}(q) = min{x : F(x) >= q}`` for ``q`` in ``[0, 1]``.

        Uses the adaptive Numba kernel :func:`_pmf_quantiles_numba` (early-exit linear scan for
        few queries, single CDF build + binary search when ``len(q) >= 8`` and ``len(pmf) >= 64``).
        Emits :class:`TruncatedSupportWarning` if any ``q > 1 - self.truncation_tol``.
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
        """
        Distribution of the sum ``X + Y`` of two independent discrete random variables via
        1D convolution (:func:`_convolve_1d_numba`).

        If ``self`` and ``other`` have different ``bin_size`` values, both are first split to
        ``gcd(self.bin_size, other.bin_size)`` without mutating the operands. Negligible tails
        (``<= 1e-9``) are trimmed and ``truncation_tol`` is summed via :func:`_combine_truncation`.
        """
        from probabilistic_modeling.ddlt import DEFAULT_TAIL_TOL, _convolve_1d_numba, _trim_pmf_numba

        trunc = _combine_truncation(self.truncation_tol, other.truncation_tol)
        if self.bin_size != other.bin_size:
            new_bin_size = int(np.gcd(self.bin_size, other.bin_size))
            self = self.split_bins(self.bin_size // new_bin_size, inplace=False)
            other = other.split_bins(other.bin_size // new_bin_size, inplace=False)
        else:
            new_bin_size = self.bin_size
        new_min = self.min + other.min
        new_pmf = _convolve_1d_numba(self.pmf, other.pmf)
        new_pmf /= np.sum(new_pmf)
        new_pmf, new_min = _trim_pmf_numba(new_pmf, new_min, new_bin_size, DEFAULT_TAIL_TOL)
        return DiscreteDistribution(new_pmf, new_min, new_bin_size, truncation_tol=trunc)

    def _align_bin_size(self):
        """
        Align ``self.min`` to be an exact multiple of ``self.bin_size`` by splitting to unit bins,
        left-padding with zeros down to ``bin_size * (min // bin_size)``, and re-merging.

        Why it is needed
        ----------------
        When convolving coarse-binned distributions (`bin_size > 1`) in :meth:`__matmul__` or
        :meth:`random_sum`, each step shifts the minimum support by ``other.min``. Having ``min``
        divisible by ``bin_size`` ensures the shifted supports land on the exact same coarse grid.
        """
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
        """Trim leading and trailing bins with probability ``<= tol`` in-place via :func:`_trim_pmf_numba`."""
        from probabilistic_modeling.ddlt import _trim_pmf_numba

        self.pmf, self.min = _trim_pmf_numba(self.pmf, self.min, self.bin_size, tol)

    def shift(self, by : int, inplace: bool = True) -> "DiscreteDistribution":
        """
        Translate the distribution's support by an integer constant ``by`` (i.e. ``X + by``).

        Used by :meth:`DDLTCalculator._shifted_lead_time` to add the deterministic review period
        ``R = cycle_time`` to the stochastic lead time ``L``. Mutates ``self`` unless ``inplace=False``.
        """
        target = self if inplace else self.copy()
        target.min += int(by)
        if (target.best_fit is not None) and (target.best_fit_params is not None):
            params = list(target.best_fit_params)
            params[-1] += by # shift the location parameter
            target.best_fit_params = tuple(params)
        return target

    def __matmul__(self, other : "DiscreteDistribution") -> "DiscreteDistribution":
        """
        Compound random sum ``S = X_1 + ... + X_N`` where ``N ~ self`` and ``X_j`` are i.i.d.
        copies of ``other`` (invoked via ``N @ X``).

        For non-identically distributed summands ``[X_1, X_2, ...]`` (such as multi-horizon daily
        forecasts), use :meth:`random_sum` instead.
        """
        if self.min < 0:
            raise ValueError("Left operand (number of summands) must have non-negative support.")
        
        if other.min < 0:
            raise ValueError("Summand distribution must have non-negative support.")
        
        if self.bin_size != 1:
            self = self.split_bins(inplace=False)  # never mutate the caller's operand
        
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
        """
        Compute the distribution of the random sum ``S = X_1 + X_2 + ... + X_N`` where the count
        of terms ``N ~ self`` is random and the summands ``rv_list = [X_1, X_2, ...]`` are
        independent but **non-identically** distributed.

        What it does
        ------------
        1. If ``self.max > len(rv_list)``, extends ``rv_list`` up to ``self.max`` using
           ``default_distr`` (or raises ``ValueError`` if ``default_distr is None``).
        2. Aligns bin sizes across all summands if any ``bin_size != 1``.
        3. Delegates the forward convolution and mixture accumulation to the compiled Numba
           kernel :func:`_random_sum_aligned_numba`.
        4. Propagates the worst-case upper-tail truncation bound ``self.max * max(truncation_tols)``.
        """
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
        """Plot the PMF as a bar chart using Matplotlib (imported lazily)."""
        import matplotlib.pyplot as plt  # optional, imported lazily (plotting only)
        from matplotlib.ticker import MaxNLocator

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
