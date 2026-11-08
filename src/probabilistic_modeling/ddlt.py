"""
Numba-accelerated Demand-During-Lead-Time (DDLT) calculation and discrete convolution kernels.

What this module does
---------------------
This module is the computational engine of the library. It takes daily demand distributions
over a forecast horizon ``t, t+1, ..., t+H-1`` and a discrete distribution of replenishment
lead time ``L`` (plus a deterministic review period ``R = cycle_time``), and computes the
distribution of **Demand During Lead Time plus Review Period (DDLT)**:

    D^{L+R}_t = X_1 + X_2 + ... + X_{L + R}

where each ``X_j`` is the demand on day ``t + j - 1`` and ``L + R`` is random. Because both
the daily demand distributions ``X_j`` and the number of days ``L + R`` are random, ``D^{L+R}_t``
is a **probability mixture of convolutions**:

    P(D^{L+R}_t = y) = sum_{m = min(L+R)}^{max(L+R)} P(L + R = m) * (X_1 * X_2 * ... * X_m)(y)

Why this module is needed
-------------------------
Computing thousands of discrete convolutions, quantile reconstructions, and parametric PMFs
in pure Python/NumPy/SciPy is orders of magnitude too slow and allocates gigabytes of
temporary arrays. This module compiles all hot loops to native machine code via Numba (`@njit`):

1. **Support-truncation control** (:func:`support_tol_for_service_level`):
   Unbounded distributions (Poisson, Negative Binomial, exponential upper quantile tails) must
   be truncated to a finite integer grid ``[0, K]``. If we truncate ``tol`` probability mass on
   each of ``n_terms`` daily distributions, the sum can lose up to ``n_terms * tol`` upper-tail
   mass. To keep a high quantile (e.g. 99.99% service level) accurate, ``tol`` is tightened
   automatically so ``n_terms * tol <= (1 - service_level) / 10``.
2. **Adaptive 1D discrete convolution** (:func:`_convolve_1d_numba`):
   Uses a simple ``O(N * M)`` nested loop for small PMFs (zero FFT overhead) and switches
   automatically to an ``O(N log N)`` FFT convolution when ``N * M > 262,144`` and both
   supports have at least 256 bins. When ``rocket-fft`` is not installed, a pure-Numba
   Cooley-Tukey Radix-2 FFT packs *both* real PMFs into a single complex FFT (`z = a + i*b`).
3. **Adaptive quantile & CDF kernels** (:func:`_pmf_quantile_scalar_numba`, :func:`_pmf_quantiles_numba`):
   For a single quantile query (e.g. computing one target inventory), a linear scan that stops
   as soon as the cumulative sum reaches ``q`` is optimal (``O(answer_index)`` time, ``O(1)``
   memory). When many quantiles are requested from the same PMF (e.g. Monte Carlo sampling),
   it builds a monotonic CDF once in ``O(N)`` and binary-searches each query in ``O(log N)``.
4. **Fused random-sum & row kernels** (:func:`_random_sum_aligned_numba`,
   :func:`_ddlt_from_quantiles_row_numba`, :func:`_ddlt_from_glm_row_numba`):
   Step forward through ``m = 1, 2, ..., max(L+R)``, convolving one day at a time, trimming
   negligible tails after each step, and accumulating weighted partial convolutions into a
   single 1D output array (no 2D matrix allocations).
5. **High-level orchestrator** (:class:`DDLTCalculator`):
   Wraps the Numba kernels into a clean DataFrame-in / DataFrame-out interface used by all
   forecasters (:class:`GLMForecaster`, :class:`CatBoostQuantileForecaster`,
   :class:`EmpiricalQuantileForecaster`) and by :func:`compute_lead_time_demand`.

Where this module is used
-------------------------
- ``distributions/discrete.py``: uses convolution, tail trimming, quantile, CRPS, and random-sum kernels.
- ``distributions/quantile.py``: uses ``_quantiles_to_pmf_numba`` and ``_quantiles_to_mean_batch_numba``.
- ``models/forecasters.py`` & ``models/baseline.py``: use :class:`DDLTCalculator` inside ``compute_ddlt``.
- ``time_series.py``: uses :class:`DDLTCalculator` in legacy helpers and ``compute_lead_time_demand``.
- ``inventory/newsvendor.py`` & ``inventory/simulation.py``: use batch quantile kernels.
- ``evaluation/ddlt_calibration.py``: uses scalar quantile and CRPS kernels.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import List as NumbaList

from probabilistic_modeling.distributions.discrete import DiscreteDistribution

logger = logging.getLogger(__name__)

# Optional dependency: `rocket-fft` enables `np.fft.rfft` / `np.fft.irfft` inside Numba `@njit`
# functions by binding SciPy's PocketFFT C++ implementation. If not installed, we fall back to
# `_fft_convolve_radix2_numba` (our self-contained pure-Numba Cooley-Tukey Radix-2 FFT below).
try:
    import rocket_fft as _rocket_fft  # type: ignore[import-not-found]  # noqa: F401
    HAS_ROCKET_FFT = True
except ImportError:
    HAS_ROCKET_FFT = False

# =============================================================================
# Section 1: Module Constants & Support-Truncation Math
# =============================================================================

# Tiny offset subtracted when discretizing a continuous quantile function Q(U) onto integers.
# Because integer demand D = round(Q(U)) satisfies D <= y iff Q(U) < y + 0.5 (strict inequality),
# evaluating the continuous CDF at `y + 0.5 - _LEFT_LIMIT_EPS` approximates the left limit
# P(Q(U) < y + 0.5) when Q has flat spots (atoms) at half-integers.
# Used in: `_quantiles_to_pmf_numba`.
_LEFT_LIMIT_EPS = 1e-9

# Probability threshold for trimming negligible leading/trailing bins from a PMF after convolution.
# Bins at the extreme left or right with mass <= 1e-9 are dropped and the remaining PMF is
# renormalized. Without this, convolving 10+ days would grow the support array with thousands
# of near-zero floating-point dust bins, slowing down subsequent convolutions.
# Used in: `_trim_pmf_numba`, `_random_sum_aligned_numba`, `DiscreteDistribution.__add__`.
DEFAULT_TAIL_TOL = 1e-9

# Crossover thresholds for switching from direct O(N * M) convolution to O(N log N) FFT convolution.
# Direct nested loops have zero setup cost and are faster when the product of lengths N * M <= 262,144
# (e.g. two 512-bin PMFs) or when either PMF is shorter than 256 bins. Above both thresholds,
# FFT convolution wins by 5x-49x.
# Used in: `_convolve_1d_numba`.
FFT_CONVOLVE_THRESHOLD = 262_144
FFT_MIN_SIDE = 256

# Default upper-tail probability mass `tol` cut off when building a finite support for an
# unbounded distribution (e.g. Poisson, Negative Binomial, or exponential quantile tail).
# By default, the support stops at the `1 - 1e-4` (99.99th percentile) quantile, and the
# remaining `<= 1e-4` tail probability is lumped into the last bin.
# Used in: `support_tol_for_service_level`, `DDLTCalculator`, `DiscreteDistribution.from_parametric`.
DEFAULT_SUPPORT_TOL = 1e-4

# Safety factor ensuring that the total upper-tail probability mass truncated across all `n_terms`
# daily distributions in a DDLT sum is at least 10x smaller than `1 - max_service_level`.
# Used in: `support_tol_for_service_level`.
SUPPORT_TOL_SAFETY_FACTOR = 10.0

# Hard floor on support truncation tolerance (1e-12) to prevent float64 precision exhaustion
# or astronomically large support arrays when `max_service_level` is extremely close to 1.0.
# Used in: `support_tol_for_service_level`.
MIN_SUPPORT_TOL = 1e-12

# Crossover thresholds for `_pmf_quantiles_numba` (adaptive quantile lookup):
# Building a full CDF array takes O(N) time across all N bins, whereas a single linear scan
# stops early at the answer index. Binary search only pays off when we have at least
# `QUANTILE_BSEARCH_MIN_QUERIES` (8) quantile queries on a PMF with at least
# `QUANTILE_BSEARCH_MIN_SUPPORT` (64) bins.
# Used in: `_pmf_quantiles_numba`.
QUANTILE_BSEARCH_MIN_QUERIES = 8
QUANTILE_BSEARCH_MIN_SUPPORT = 64


def support_tol_for_service_level(
    max_service_level: float | None,
    n_terms: int = 1,
    base_tol: float = DEFAULT_SUPPORT_TOL,
    safety_factor: float = SUPPORT_TOL_SAFETY_FACTOR,
) -> float:
    """
    Compute the per-day upper-tail truncation tolerance ``tol`` required so that quantiles
    up to ``max_service_level`` on a sum of ``n_terms`` distributions are not biased low.

    What it does
    ------------
    Computes::

        tol = max(MIN_SUPPORT_TOL, min(base_tol, (1 - max_service_level) / (safety_factor * n_terms)))

    Why it is needed
    ----------------
    Unbounded distributions (Poisson, NB2, exponential quantile tails) must be stored on a
    finite integer grid ``[0, K]`` where ``F(K) >= 1 - tol``. Even though we lump the tail
    mass ``P(X > K)`` into the last bin ``K`` (so ``F(x)`` is exact for ``x < K``), any
    quantile above ``1 - tol`` gets capped at ``K``. When we convolve ``n_terms`` daily
    distributions ``X_1 + ... + X_{n_terms}``, by the union bound the sum's upper tail
    can be affected at probability levels above ``1 - n_terms * tol``. Dividing the target
    tail ``1 - max_service_level`` by ``safety_factor * n_terms`` guarantees that the
    truncation point sits safely above the requested service-level quantile.

    Examples (with ``n_terms=1, safety_factor=10``):
    - ``max_service_level = 0.95``    -> ``tol = 1e-4`` (kept at ``base_tol``)
    - ``max_service_level = 0.999``   -> ``tol = 1e-4``
    - ``max_service_level = 0.9999``  -> ``tol = 1e-5``
    - ``max_service_level = 0.99999`` -> ``tol = 1e-6``

    Where it is used
    ----------------
    - :meth:`DDLTCalculator.effective_tol` (automatically tightens daily PMF supports during DDLT).
    - :meth:`DiscreteDistribution.from_parametric` (when ``max_service_level`` is passed).
    """
    if max_service_level is None or not np.isfinite(max_service_level):
        return float(base_tol)
    sl = float(max_service_level)
    if sl <= 0.0:
        return float(base_tol)
    tail = max(1.0 - sl, 0.0)
    required = tail / (float(safety_factor) * max(int(n_terms), 1))
    return float(max(min(base_tol, required), MIN_SUPPORT_TOL))


# =============================================================================
# Section 2: 1D Discrete Convolution Kernels (Direct & Radix-2 FFT)
# =============================================================================

@njit(cache=True)
def _direct_convolve_1d_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Direct ``O(N * M)`` 1D discrete convolution of two PMF arrays ``a`` and ``b``.

    What it does
    ------------
    Computes ``out[k] = sum_{i+j=k} a[i] * b[j]`` for ``k = 0, ..., len(a) + len(b) - 2``,
    skipping outer-loop iterations where ``a[i] == 0``.

    Why it is needed
    ----------------
    For compact supports (e.g., daily retail demand of 0..50 units), a simple nested loop in
    compiled C/Numba has zero allocation/twiddle overhead and beats FFT convolution. Also
    exploits sparsity when ``a[i] == 0``.

    Where it is used
    ----------------
    Called by :func:`_convolve_1d_numba` when ``len(a) * len(b) <= FFT_CONVOLVE_THRESHOLD``
    or ``min(len(a), len(b)) < FFT_MIN_SIDE``.
    """
    na = len(a)
    nb = len(b)
    out = np.zeros(na + nb - 1, dtype=np.float64)
    for i in range(na):
        ai = a[i]
        if ai > 0.0:
            for j in range(nb):
                out[i + j] += ai * b[j]
    return out


@njit(cache=True)
def _inplace_fft_radix2_numba(re: np.ndarray, im: np.ndarray, inverse: bool) -> None:
    """
    In-place iterative Cooley-Tukey Radix-2 Fast Fourier Transform (FFT) on separate
    contiguous ``float64`` real (`re`) and imaginary (`im`) arrays.

    What it does
    ------------
    Computes the Discrete Fourier Transform (when ``inverse=False``):
        X[k] = sum_{n=0}^{m-1} x[n] * exp(-2*pi*i * k * n / m)
    or the Inverse Discrete Fourier Transform (when ``inverse=True``):
        x[n] = (1 / m) * sum_{k=0}^{m-1} X[k] * exp(+2*pi*i * k * n / m)
    in-place in ``O(m log m)`` time, where ``m = len(re)`` must be a power of 2.

    Algorithm step-by-step
    ----------------------
    1. **Bit-reversal permutation**: Reorders ``(re, im)`` in-place so that index ``i`` is
       swapped with the integer formed by reversing the ``log2(m)`` binary bits of ``i``.
       This puts the elements in the exact order needed for bottom-up butterfly combination.
    2. **Butterfly stages (`length = 2, 4, 8, ..., m`)**:
       - Precomputes the ``length / 2`` twiddle factors ``w[k] = exp(+-2*pi*i * k / length)``
         using a complex multiplication recurrence (requiring only **one** ``cos`` and ``sin``
         call per stage instead of ``m log m`` calls).
       - Applies the Cooley-Tukey butterfly to each block of size ``length``:
         ``u = x[u_idx]``, ``v = x[v_idx] * w[k]``, setting ``x[u_idx] = u + v`` and
         ``x[v_idx] = u - v``.
    3. **Normalization**: When ``inverse=True``, divides all entries by ``m``.

    Why it is needed
    ----------------
    Numba's standard ``@njit`` mode does not support ``np.fft`` out of the box unless the
    third-party ``rocket-fft`` package is installed. Having a self-contained, zero-dependency
    Numba FFT guarantees ``O(N log N)`` convolution on any machine out of the box. Separate
    ``re`` and ``im`` arrays avoid complex128 struct boxing overhead and enable SIMD vectorization.

    Where it is used
    ----------------
    Called by :func:`_fft_convolve_radix2_numba` (forward pass and inverse pass).
    """
    m = len(re)
    # --- Step 1: In-place bit-reversal permutation ---
    j = 0
    for i in range(1, m):
        bit = m >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]

    w_re = np.empty(m >> 1, dtype=np.float64)
    w_im = np.empty(m >> 1, dtype=np.float64)

    # --- Step 2: Bottom-up butterfly stages for sub-DFT sizes 2, 4, 8, ..., m ---
    length = 2
    while length <= m:
        half = length >> 1
        angle = (2.0 * np.pi / length) if inverse else (-2.0 * np.pi / length)
        wlen_re = np.cos(angle)
        wlen_im = np.sin(angle)
        cur_re = 1.0
        cur_im = 0.0
        # Build twiddle table w[k] = (wlen)^k via recurrence:
        for k in range(half):
            w_re[k] = cur_re
            w_im[k] = cur_im
            nxt_re = cur_re * wlen_re - cur_im * wlen_im
            cur_im = cur_re * wlen_im + cur_im * wlen_re
            cur_re = nxt_re

        # Apply Radix-2 butterflies across all blocks of width `length`:
        for i in range(0, m, length):
            for k in range(half):
                u_idx = i + k
                v_idx = u_idx + half
                # Complex multiply: v = x[v_idx] * w[k]
                vr = re[v_idx] * w_re[k] - im[v_idx] * w_im[k]
                vi = re[v_idx] * w_im[k] + im[v_idx] * w_re[k]
                ur = re[u_idx]
                ui = im[u_idx]
                re[u_idx] = ur + vr
                im[u_idx] = ui + vi
                re[v_idx] = ur - vr
                im[v_idx] = ui - vi
        length <<= 1

    # --- Step 3: 1/m scaling for the inverse DFT ---
    if inverse:
        inv_m = 1.0 / m
        for i in range(m):
            re[i] *= inv_m
            im[i] *= inv_m


@njit(cache=True)
def _fft_convolve_radix2_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    ``O(N log N)`` 1D linear convolution of two real PMFs ``a`` and ``b`` using the
    "two-real-signals-in-one-complex-FFT" packing trick.

    What it does
    ------------
    Convolves ``a`` (length ``na``) and ``b`` (length ``nb``) into an output array of
    length ``out_len = na + nb - 1`` using only **two** complex FFT calls (one forward,
    one inverse) instead of three.

    Mathematical derivation of the 2-in-1 packing trick
    ---------------------------------------------------
    Because both ``a`` and ``b`` are purely real sequences, a naive complex FFT of ``a``
    alone would waste the imaginary channel. Instead, we zero-pad to the next power of two
    ``m >= na + nb - 1`` and form a single complex sequence:

        z[n] = a[n] + i * b[n]

    By linearity of the DFT, ``Z[k] = DFT(z)[k] = A[k] + i * B[k]``.
    Because ``a`` and ``b`` are real, their individual DFTs satisfy Hermitian conjugate
    symmetry: ``A[m - k] = conj(A[k])`` and ``B[m - k] = conj(B[k])``. Therefore:

        Z[k]     = (A_re[k] - B_im[k]) + i * (A_im[k] + B_re[k])
        Z[m - k] = (A_re[k] + B_im[k]) + i * (-A_im[k] + B_re[k])

    Adding and subtracting ``Z[k]`` and ``conj(Z[m - k])`` recovers ``A[k]`` and ``B[k]``
    in closed form for every frequency ``k = 0, ..., m // 2``:

        A[k] = 0.5 * (Z[k] + conj(Z[m - k]))
        B[k] = -0.5 * i * (Z[k] - conj(Z[m - k]))

    We then multiply ``C[k] = A[k] * B[k]`` in-place, set ``C[m - k] = conj(C[k])``, and
    run a single inverse FFT on ``C``. Finally, negative floating-point round-off dust
    (e.g. ``-1e-17``) is clipped to ``0.0`` since probabilities are non-negative.

    Why it is needed
    ----------------
    Cuts the number of FFT passes from 3 (``FFT(a)``, ``FFT(b)``, ``IFFT(A*B)``) to 2,
    reducing FFT convolution runtime by ~33% with zero external dependencies.

    Where it is used
    ----------------
    Bound to ``_fft_convolve_1d_numba`` when ``rocket-fft`` is not installed, and called
    by :func:`_convolve_1d_numba` for large PMF supports.
    """
    na = len(a)
    nb = len(b)
    out_len = na + nb - 1
    # Next power of 2 >= out_len (prevents circular wrap-around aliasing):
    m = 1
    while m < out_len:
        m <<= 1

    # Pack a into real part and b into imaginary part: z = a + i * b
    re = np.zeros(m, dtype=np.float64)
    im = np.zeros(m, dtype=np.float64)
    re[:na] = a
    im[:nb] = b

    # 1. Single forward FFT: Z = DFT(a + i * b)
    _inplace_fft_radix2_numba(re, im, False)

    # 2. Unpack A[k] = DFT(a) and B[k] = DFT(b) via conjugate symmetry, and compute C[k] = A[k] * B[k]:
    half_m = m >> 1
    mask = m - 1  # Since m is a power of 2, `(m - k) & (m - 1)` is `(m - k) % m`
    for k in range(half_m + 1):
        mk = (m - k) & mask
        r1 = re[k]
        i1 = im[k]
        r2 = re[mk]
        i2 = im[mk]
        # A[k] = 0.5 * (Z[k] + conj(Z[mk]))
        a_re = 0.5 * (r1 + r2)
        a_im = 0.5 * (i1 - i2)
        # B[k] = -0.5i * (Z[k] - conj(Z[mk]))
        b_re = 0.5 * (i1 + i2)
        b_im = 0.5 * (r2 - r1)
        # C[k] = A[k] * B[k]
        c_re = a_re * b_re - a_im * b_im
        c_im = a_re * b_im + a_im * b_re
        re[k] = c_re
        im[k] = c_im
        if mk != k:
            # Hermitian symmetry for the upper half of the spectrum: C[m - k] = conj(C[k])
            re[mk] = c_re
            im[mk] = -c_im

    # 3. Single inverse FFT: c = IDFT(C)
    _inplace_fft_radix2_numba(re, im, True)

    # 4. Extract valid linear convolution length and clip tiny negative round-off to 0:
    out = np.empty(out_len, dtype=np.float64)
    for i in range(out_len):
        val = re[i]
        out[i] = val if val > 0.0 else 0.0
    return out


if HAS_ROCKET_FFT:
    @njit(cache=True)
    def _fft_convolve_rocket_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        ``O(N log N)`` Numba-compiled 1D real convolution powered by ``rocket-fft`` (PocketFFT).

        What it does
        ------------
        Uses ``np.fft.rfft`` and ``np.fft.irfft`` at the next power-of-two size ``m >= len(a)+len(b)-1``
        and clips any negative floating-point round-off to ``0.0``.

        Where it is used
        ----------------
        Bound to ``_fft_convolve_1d_numba`` when the optional ``rocket-fft`` extra is installed.
        """
        na = len(a)
        nb = len(b)
        out_len = na + nb - 1
        m = 1
        while m < out_len:
            m <<= 1
        fa = np.fft.rfft(a, m)
        fb = np.fft.rfft(b, m)
        conv = np.fft.irfft(fa * fb, m)
        out = np.empty(out_len, dtype=np.float64)
        for i in range(out_len):
            val = conv[i]
            out[i] = val if val > 0.0 else 0.0
        return out

    _fft_convolve_1d_numba = _fft_convolve_rocket_numba
else:
    _fft_convolve_1d_numba = _fft_convolve_radix2_numba


@njit(cache=True)
def _convolve_1d_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Adaptive 1D discrete convolution of two PMFs ``a`` and ``b`` in compiled Numba code.

    What it does
    ------------
    Dispatches between :func:`_direct_convolve_1d_numba` and :func:`_fft_convolve_1d_numba`:
    - Uses direct ``O(N * M)`` loop when ``len(a) * len(b) <= 262,144`` or ``min(len(a), len(b)) < 256``.
    - Switches to ``O(N log N)`` FFT convolution when both supports are large.

    Why it is needed
    ----------------
    During DDLT propagation, early convolution steps (e.g. day 1 + day 2) often have small
    supports where direct convolution is fastest, whereas later steps or high-volume items
    (or tight ``tol`` for 99.99% service levels) produce 1,000–50,000-bin PMFs where direct
    convolution would stall. Adaptive dispatch gets the best of both worlds automatically.

    Where it is used
    ----------------
    - :func:`_random_sum_aligned_numba` (core DDLT convolution loop)
    - :meth:`DiscreteDistribution.__add__` (``d1 + d2`` operator)
    """
    na = len(a)
    nb = len(b)
    if na >= FFT_MIN_SIDE and nb >= FFT_MIN_SIDE and na * nb > FFT_CONVOLVE_THRESHOLD:
        return _fft_convolve_1d_numba(a, b)
    return _direct_convolve_1d_numba(a, b)


# =============================================================================
# Section 3: PMF Trimming, Adaptive Quantile Search, CDF Tables & CRPS Kernels
# =============================================================================

@njit(cache=True)
def _trim_pmf_numba(
    pmf: np.ndarray,
    min_val: int,
    bin_size: int,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Trim negligible leading and trailing probability bins (``<= tail_tol``) from a PMF.

    What it does
    ------------
    1. Scans from the left to find the first bin ``first`` with ``pmf[first] > tail_tol``.
    2. Scans from the right to find the last bin ``last`` with ``pmf[last] > tail_tol``.
    3. Slices ``pmf[first : last + 1]``, renormalizes so the sum is ``1.0``, and updates
       the minimum support value to ``min_val + first * bin_size``.

    Why it is needed
    ----------------
    Each convolution of two PMFs adds their support lengths (`na + nb - 1`), even when the
    extreme tails contain only ``1e-15`` probability mass. Trimming bins ``<= 1e-9`` after
    each convolution step keeps intermediate PMFs compact across long lead times without
    affecting any meaningful quantile or moment.

    Where it is used
    ----------------
    - :func:`_random_sum_aligned_numba` (before and after each convolution step, and on output)
    - :meth:`DiscreteDistribution.__add__` and :meth:`DiscreteDistribution._trim_pmf`
    """
    n = len(pmf)
    if n <= 1 or tail_tol < 0.0:
        return pmf, min_val

    first = 0
    while first < n - 1 and pmf[first] <= tail_tol:
        first += 1

    last = n - 1
    while last > first and pmf[last] <= tail_tol:
        last -= 1

    if first == 0 and last == n - 1:
        return pmf, min_val

    trimmed = pmf[first : last + 1].copy()
    s = np.sum(trimmed)
    if s > 0.0:
        trimmed /= s
    return trimmed, min_val + first * bin_size


@njit(cache=True)
def _pmf_quantile_scalar_numba(pmf: np.ndarray, min_val: int, bin_size: int, q: float) -> int:
    """
    Evaluate a single discrete quantile ``F^{-1}(q) = min{x : F(x) >= q}`` directly from a PMF.

    What it does
    ------------
    Accumulates ``csum += pmf[i]`` from ``i = 0`` upward and returns ``min_val + i * bin_size``
    at the first index where ``csum >= q`` (or the maximum support point if round-off leaves
    ``sum(pmf) < q``).

    Why it is needed
    ----------------
    For a **single** quantile query on a PMF, building a full CDF array first would cost ``O(n)``
    time and ``O(n)`` memory, whereas an early-exit linear scan costs ``O(answer_index) <= O(n)``
    time and **zero** memory allocations.

    Where it is used
    ----------------
    - :func:`_pmf_quantiles_numba` (when number of queries < 8 or support < 64)
    - :func:`_batch_pmf_quantiles_numba` (one quantile per product in newsvendor / target resolution)
    - :meth:`DDLTCalculator.from_quantiles_group` & :meth:`DDLTCalculator.from_glm_group`
    - :func:`_sample_lead_times_numba` & :func:`_simulate_inventory_mc_numba` (lead-time draws)
    - :func:`_ddlt_calibration_kernel` (evaluating DDLT quantiles in calibration reports)
    """
    n = len(pmf)
    if q <= 0.0 or n <= 1:
        return min_val
    csum = 0.0
    for i in range(n):
        csum += pmf[i]
        if csum >= q:
            return min_val + i * bin_size
    return min_val + (n - 1) * bin_size


@njit(cache=True)
def _pmf_to_cdf_numba(pmf: np.ndarray) -> np.ndarray:
    """
    Build a guaranteed non-decreasing CDF array from ``pmf`` using a running maximum.

    What it does
    ------------
    Computes the sequential cumulative sum ``csum += pmf[i]`` in the exact same floating-point
    addition order as :func:`_pmf_quantile_scalar_numba`, and stores ``cdf[i] = max_{j <= i} csum_j``.

    Why it is needed
    ----------------
     Binary search (:func:`_cdf_quantile_scalar_numba`) requires a monotonically non-decreasing
    array. More subtly, tracking the running maximum ensures that the first index ``i`` where
    ``cdf[i] >= q`` is **bit-for-bit identical** to the first index where the early-exit linear
    scan in :func:`_pmf_quantile_scalar_numba` reaches ``q`` (even if ``pmf`` had tiny negative
    values).

    Where it is used
    ----------------
    Called by :func:`_pmf_quantiles_numba` when batch-querying many quantiles from one PMF.
    """
    n = len(pmf)
    cdf = np.empty(n, dtype=np.float64)
    csum = 0.0
    run_max = -np.inf
    for i in range(n):
        csum += pmf[i]
        if csum > run_max:
            run_max = csum
        cdf[i] = run_max
    return cdf


@njit(cache=True)
def _cdf_quantile_scalar_numba(
    cdf: np.ndarray, start: int, n: int, min_val: int, bin_size: int, q: float
) -> int:
    """
    Binary search (``O(log n)``) for the smallest index ``i`` in ``cdf[start : start + n]``
    satisfying ``cdf[start + i] >= q``.

    What it does
    ------------
    Performs standard lower-bound binary search over ``[0, n - 1]`` and maps the winning bin
    index ``lo`` back to the support value ``min_val + lo * bin_size``.

    Why it is needed
    ----------------
    Once a CDF is built, binary search answers each quantile query in ``O(log n)`` instead of
    ``O(n)``. Initializing ``hi = n - 1`` (the last valid bin index) also gracefully handles
    the edge case ``q > cdf[-1]`` caused by floating-point sum ``1 - eps``: the loop converges
    to ``lo = n - 1`` (the maximum support value), matching :func:`_pmf_quantile_scalar_numba`.

    Where it is used
    ----------------
    - :func:`_pmf_quantiles_numba` (multi-query quantile lookup and ``DiscreteDistribution.sample``)
    - :func:`_batch_cdf_quantiles_numba` (batch lookup over precomputed flat CDF tables)
    """
    if q <= 0.0 or n <= 1:
        return min_val
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if cdf[start + mid] >= q:
            hi = mid
        else:
            lo = mid + 1
    return min_val + lo * bin_size


@njit(cache=True)
def _pmf_quantiles_numba(pmf: np.ndarray, min_val: int, bin_size: int, q_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized quantile evaluation on a single PMF with adaptive algorithm selection.

    What it does
    ------------
    Given ``m = len(q_arr)`` probability levels and a PMF of length ``n = len(pmf)``:
    - If ``m >= 8`` (`QUANTILE_BSEARCH_MIN_QUERIES`) and ``n >= 64`` (`QUANTILE_BSEARCH_MIN_SUPPORT`),
      builds the CDF once (`O(n)`) and binary-searches each query (`O(m log n)`).
    - Otherwise, runs the zero-allocation early-exit linear scan per query.

    Why it is needed
    ----------------
    :meth:`DiscreteDistribution.quantile` and :meth:`DiscreteDistribution.sample` are called
    both with 1–3 queries (where allocating a CDF is wasted work) and with 10,000+ uniform
    random numbers during Monte Carlo sampling (where linear scan would be 70x slower).
    Both paths return identical integers.

    Where it is used
    ----------------
    - :meth:`DiscreteDistribution.quantile`
    - :meth:`DiscreteDistribution.sample`
    """
    m = len(q_arr)
    n = len(pmf)
    out = np.empty(m, dtype=np.int64)
    if m >= QUANTILE_BSEARCH_MIN_QUERIES and n >= QUANTILE_BSEARCH_MIN_SUPPORT:
        cdf = _pmf_to_cdf_numba(pmf)
        for k in range(m):
            out[k] = _cdf_quantile_scalar_numba(cdf, 0, n, min_val, bin_size, q_arr[k])
    else:
        for k in range(m):
            out[k] = _pmf_quantile_scalar_numba(pmf, min_val, bin_size, q_arr[k])
    return out


@njit(cache=True)
def _pmf_quantiles_linear_numba(pmf: np.ndarray, min_val: int, bin_size: int, q_arr: np.ndarray) -> np.ndarray:
    """
    Reference linear-scan quantile implementation (``O(m * n)``).

    Where it is used
    ----------------
    Used in ``tests/test_quantile_search.py`` to verify that adaptive binary search
    (:func:`_pmf_quantiles_numba`) produces bit-identical results to linear scan.
    """
    m = len(q_arr)
    out = np.empty(m, dtype=np.int64)
    for k in range(m):
        out[k] = _pmf_quantile_scalar_numba(pmf, min_val, bin_size, q_arr[k])
    return out


@njit(cache=True)
def _build_cdfs_flat_numba(pmf_flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Build running-max CDFs for a batch of variable-length PMFs packed into a single 1D array.

    What it does
    ------------
    ``pmf_flat`` stores ``K`` PMFs back-to-back, where PMF ``i`` occupies slice
    ``pmf_flat[offsets[i] : offsets[i + 1]]``. Computes the cumulative sum (with running max)
    within each segment and returns ``cdf_flat`` with the exact same offsets.

    Why it is needed
    ----------------
    Packing variable-length PMFs into a single contiguous 1D array avoids Numba `typed.List`
    boxing overhead when repeatedly querying a collection of distributions.

    Where it is used
    ----------------
    Called by :func:`build_cdf_table`.
    """
    cdf_flat = np.empty(len(pmf_flat), dtype=np.float64)
    for i in range(len(offsets) - 1):
        s = offsets[i]
        e = offsets[i + 1]
        csum = 0.0
        run_max = -np.inf
        for j in range(s, e):
            csum += pmf_flat[j]
            if csum > run_max:
                run_max = csum
            cdf_flat[j] = run_max
    return cdf_flat


@njit(cache=True)
def _batch_cdf_quantiles_numba(
    cdf_flat: np.ndarray,
    offsets: np.ndarray,
    mins: np.ndarray,
    bin_sizes: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    Binary-search one quantile level ``alphas[i]`` per product on precomputed flat CDFs.

    What it does
    ------------
    For each distribution ``i = 0, ..., n - 1``, binary-searches ``cdf_flat[offsets[i]:offsets[i+1]]``
    for ``alphas[i]`` in ``O(log n_i)`` time.

    Where it is used
    ----------------
    Used with :func:`build_cdf_table` or :func:`_build_prefix_cdfs_numba` when evaluating
    repeated quantile vectors across a batch of distributions (e.g. in ``tests/test_quantile_search.py``).
    """
    n = len(offsets) - 1
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = offsets[i]
        out[i] = _cdf_quantile_scalar_numba(
            cdf_flat, s, offsets[i + 1] - s, int(mins[i]), int(bin_sizes[i]), float(alphas[i])
        )
    return out


@njit(cache=True)
def _build_prefix_cdfs_numba(pmfs: NumbaList, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build only the first ``lengths[i]`` CDF bins of each PMF in ``pmfs`` into a flat layout.

    What it does
    ------------
    Allocates a flat array sized to ``sum(lengths)`` and computes the running-max CDF of each
    ``pmfs[i]`` only up to index ``lengths[i] - 1``.

    Why it is needed
    ----------------
    When an upper bound on the answer index is already known for every distribution ``i``
    (for instance, after computing the unconstrained ``k = 0`` newsvendor solution, increasing
    the shadow price ``k > 0`` can only *decrease* each product's critical fractile and order
    quantity), we never need CDF bins beyond the unconstrained index. Building only the prefix
    avoids summing long upper tails that binary search will never visit.

    Where it is used
    ----------------
    Tested and benchmarked in ``tests/test_quantile_search.py``.
    """
    n = len(pmfs)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        li = lengths[i]
        if li > len(pmfs[i]):
            li = len(pmfs[i])
        if li < 1:
            li = 1
        offsets[i + 1] = offsets[i] + li
    cdf_flat = np.empty(offsets[n], dtype=np.float64)
    for i in range(n):
        p = pmfs[i]
        s = offsets[i]
        csum = 0.0
        run_max = -np.inf
        for j in range(offsets[i + 1] - s):
            csum += p[j]
            if csum > run_max:
                run_max = csum
            cdf_flat[s + j] = run_max
    return cdf_flat, offsets


def build_cdf_table(pmfs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate a list of PMF arrays and build all of their CDFs in one compiled pass.

    What it does
    ------------
    Packs ``pmfs`` into a single contiguous array ``pmf_flat`` with index boundaries ``offsets``
    and calls :func:`_build_cdfs_flat_numba`.

    Returns
    -------
    cdf_flat : np.ndarray
        1D array containing all CDFs concatenated back-to-back.
    offsets : np.ndarray
        1D integer array of length ``len(pmfs) + 1`` where distribution ``i`` occupies
        ``cdf_flat[offsets[i] : offsets[i + 1]]``.

    Where it is used
    ----------------
    Paired with :func:`_batch_cdf_quantiles_numba` when querying the same list of distributions
    across many iterations.
    """
    lengths = np.fromiter((len(p) for p in pmfs), dtype=np.int64, count=len(pmfs))
    offsets = np.zeros(len(pmfs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if len(pmfs) == 0:
        return np.empty(0, dtype=np.float64), offsets
    pmf_flat = np.ascontiguousarray(np.concatenate(pmfs), dtype=np.float64)
    return _build_cdfs_flat_numba(pmf_flat, offsets), offsets


@njit(cache=True)
def _batch_pmf_quantiles_numba(
    pmfs: NumbaList,
    mins: np.ndarray,
    bin_sizes: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    Evaluate one quantile ``alphas[i]`` for each product distribution ``i`` in compiled code.

    What it does
    ------------
    Loops over ``i = 0, ..., len(pmfs) - 1`` and calls :func:`_pmf_quantile_scalar_numba`
    for ``(pmfs[i], mins[i], bin_sizes[i], alphas[i])``.

    Why it is needed
    ----------------
    Eliminates Python loop overhead and temporary CDF allocations when computing target
    inventories across thousands of SKUs (either at a fixed service level or inside each
    step of the multi-product newsvendor binary search).

    Where it is used
    ----------------
    - :func:`probabilistic_modeling.inventory.newsvendor.solve_multi_product_newsvendor`
    - :func:`probabilistic_modeling.inventory.simulation.resolve_target_inventory`
    """
    n = len(pmfs)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = _pmf_quantile_scalar_numba(pmfs[i], int(mins[i]), int(bin_sizes[i]), float(alphas[i]))
    return out


@njit(cache=True)
def _pmf_crps_scalar_numba(pmf: np.ndarray, min_val: int, bin_size: int, y: float) -> float:
    """
    Compute the exact Continuous/Discrete Ranked Probability Score (CRPS) for one observation ``y``
    directly from a discrete PMF in ``O(1)`` auxiliary memory.

    What it does
    ------------
    Evaluates the integral:

        CRPS(F, y) = int_{-inf}^{+inf} (F(x) - 1{x >= y})^2 dx

    Because a discrete distribution has a step-function CDF that is constant (`F(x) = csum`) on
    each interval ``[x_lo, x_hi) = [min_val + i * bin_size, min_val + (i + 1) * bin_size)``:
    - For ``x < x_first`` (`F(x) = 0`): if ``y < x_first``, ``(0 - 1)^2 = 1`` integrates to ``x_first - y``.
    - On each interior step ``[x_lo, x_hi)`` of width ``step = bin_size``:
      * If ``y <= x_lo``, ``1{x >= y} = 1``, so the integrand is ``(1 - csum)^2 * step``.
      * If ``y >= x_hi``, ``1{x >= y} = 0``, so the integrand is ``csum^2 * step``.
      * If ``x_lo < y < x_hi`` (non-integer ``y`` or ``bin_size > 1``), the interval splits at ``y``.
    - For ``x >= x_last`` (`F(x) = 1`): if ``y > x_last``, ``(1 - 0)^2 = 1`` integrates to ``y - x_last``.

    Why it is needed
    ----------------
    CRPS is the standard proper scoring rule for evaluating full predictive distributions in
    the same units as demand. Computing it on the fly without allocating a CDF array makes
    evaluating tens of thousands of DDLT distributions fast and memory-free.

    Where it is used
    ----------------
    - :func:`_pmf_crps_numba`
    - :meth:`DiscreteDistribution.crps`
    - :func:`probabilistic_modeling.evaluation.ddlt_calibration._ddlt_calibration_kernel`
    """
    n = len(pmf)
    x_first = float(min_val)
    x_last = float(min_val + (n - 1) * bin_size)
    score = 0.0
    # Contribution from (-inf, x_first) when observation y is below the support minimum:
    if y < x_first:
        score += x_first - y

    csum = 0.0
    step = float(bin_size)
    # Contribution from each step interval [x_lo, x_hi) between support points:
    for i in range(n - 1):
        csum += pmf[i]
        if csum > 1.0:
            csum = 1.0
        x_lo = float(min_val + i * bin_size)
        x_hi = x_lo + step
        if y <= x_lo:
            diff = 1.0 - csum
            score += step * diff * diff
        elif y >= x_hi:
            score += step * csum * csum
        else:
            diff = 1.0 - csum
            score += (y - x_lo) * csum * csum + (x_hi - y) * diff * diff

    # Contribution from [x_last, +inf) when observation y exceeds the support maximum:
    if y > x_last:
        score += y - x_last
    return score


@njit(cache=True)
def _pmf_crps_numba(pmf: np.ndarray, min_val: int, bin_size: int, y_arr: np.ndarray) -> np.ndarray:
    """
    Vectorized CRPS evaluation on a single PMF across an array of observations ``y_arr``.

    Where it is used
    ----------------
    Called by :meth:`DiscreteDistribution.crps` when given a 1D/ND array of observations.
    """
    m = len(y_arr)
    out = np.empty(m, dtype=np.float64)
    for k in range(m):
        out[k] = _pmf_crps_scalar_numba(pmf, min_val, bin_size, float(y_arr[k]))
    return out


# =============================================================================
# Section 4: Random-Sum Mixture & Fused Quantile / GLM -> DDLT Kernels
# =============================================================================

@njit(cache=True)
def _random_sum_aligned_numba(
    lt_pmf: np.ndarray,
    lt_min: int,
    rv_pmfs: NumbaList,
    rv_mins: np.ndarray,
    bin_size: int,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Compute the PMF and minimum support value of a random sum of non-identically distributed
    discrete random variables:

        S = X_1 + X_2 + ... + X_N,   where N ~ (lt_pmf, lt_min, bin_size=1)

    and each summand ``X_j`` has PMF ``rv_pmfs[j - 1]`` starting at ``rv_mins[j - 1]`` on a
    shared grid spacing ``bin_size``.

    What it does (step-by-step)
    ---------------------------
    1. **Forward convolution pass (`n = 0, 1, ..., lt_max`)**:
       Starts with ``curr_pmf = [1.0]`` at ``curr_min = 0`` (the 0-day sum ``S_0 = 0``).
       - Whenever ``n >= lt_min`` and ``w = P(N = n) = lt_pmf[n - lt_min] > 0``, saves the
         current partial convolution ``(curr_pmf, curr_min)`` along with weight ``w``, and
         updates the global support bounds ``[overall_min, overall_max]``.
       - If ``n < lt_max``, trims ``rv_pmfs[n]``, convolves ``curr_pmf`` with ``rv_pmfs[n]``
         via :func:`_convolve_1d_numba`, renormalizes, adds ``r_min`` to ``curr_min``, and
         trims negligible outer tails (``<= tail_tol``).
    2. **1D mixture accumulation**:
       Allocates a single 1D array ``out_pmf`` covering ``[overall_min, overall_max]`` and
       adds ``w * saved_pmfs[idx]`` at the appropriate offset ``(saved_mins[idx] - overall_min) // bin_size``.
    3. **Final normalization and tail trim**:
       Renormalizes ``out_pmf`` to sum to ``1.0`` and trims any leading/trailing bins ``<= tail_tol``.

    Why it is needed
    ----------------
    In inventory replenishment, both the daily forecasts ``X_1, X_2, ...`` (which vary by day of
    week, promotion, horizon) and the replenishment window length ``N = L + R`` are random.
    A naive implementation that builds a 2D ``(n_lead_times, max_support)`` matrix and multiplies
    by ``lt_pmf`` wastes massive memory when supports reach 10,000+ bins. Accumulating directly
    into a 1D output vector and trimming ``1e-9`` tails after each step makes DDLT computation
    fast and cache-friendly.

    Where it is used
    ----------------
    - :func:`_ddlt_from_quantiles_row_numba` (CatBoost / Empirical quantile -> DDLT path)
    - :func:`_ddlt_from_glm_row_numba` (Poisson / NB2 GLM -> DDLT path)
    - :meth:`DiscreteDistribution.random_sum` (general distribution algebra)
    """
    lt_max = lt_min + len(lt_pmf) - 1
    n_terms = len(lt_pmf)

    saved_pmfs = NumbaList()
    saved_mins = np.empty(n_terms, dtype=np.int64)
    saved_weights = np.empty(n_terms, dtype=np.float64)
    n_saved = 0

    # Base case for n = 0 summands: point mass of 1.0 at 0
    curr_pmf = np.ones(1, dtype=np.float64)
    curr_min = 0
    overall_min = 0
    overall_max = 0

    # Step 1: Forward pass through n = 0, 1, ..., lt_max summands
    for n in range(lt_max + 1):
        # If n is in the support of N = L + R with positive weight, snapshot S_n:
        if n >= lt_min:
            w = lt_pmf[n - lt_min]
            if w > 0.0:
                c_max = curr_min + bin_size * (len(curr_pmf) - 1)
                if n_saved == 0:
                    overall_min = curr_min
                    overall_max = c_max
                else:
                    if curr_min < overall_min:
                        overall_min = curr_min
                    if c_max > overall_max:
                        overall_max = c_max
                saved_pmfs.append(curr_pmf)
                saved_mins[n_saved] = curr_min
                saved_weights[n_saved] = w
                n_saved += 1
        # Advance from S_n to S_{n+1} = S_n + X_{n+1}:
        if n < lt_max:
            r_pmf, r_min = _trim_pmf_numba(rv_pmfs[n], int(rv_mins[n]), bin_size, tail_tol)
            curr_pmf = _convolve_1d_numba(curr_pmf, r_pmf)
            s = np.sum(curr_pmf)
            if s > 0.0:
                curr_pmf /= s
            curr_min += r_min
            curr_pmf, curr_min = _trim_pmf_numba(curr_pmf, curr_min, bin_size, tail_tol)

    if n_saved == 0:
        return np.ones(1, dtype=np.float64), 0

    # Step 2: Accumulate the mixture sum_n P(N = n) * P(S_n = y) into a single 1D array
    width = (overall_max - overall_min) // bin_size + 1
    out_pmf = np.zeros(width, dtype=np.float64)
    for idx in range(n_saved):
        p = saved_pmfs[idx]
        w = saved_weights[idx]
        start = (saved_mins[idx] - overall_min) // bin_size
        for k in range(len(p)):
            out_pmf[start + k] += w * p[k]

    # Step 3: Normalize and trim negligible outer tails
    total = np.sum(out_pmf)
    if total > 0.0:
        out_pmf /= total
    out_pmf, overall_min = _trim_pmf_numba(out_pmf, overall_min, bin_size, tail_tol)
    return out_pmf, overall_min


@njit(cache=True)
def _quantiles_to_pmf_numba(
    levels: np.ndarray,
    values: np.ndarray,
    tol: float = 1e-4,
    max_support: int = -1,
) -> tuple[np.ndarray, int]:
    """
    Reconstruct a discrete non-negative integer PMF ``(pmf, min_value)`` from a single
    vector of predicted quantiles ``values`` at probability levels ``levels``.

    What it does (4-stage algorithm)
    --------------------------------
    Given quantile levels ``0 < tau_0 < tau_1 < ... < tau_{K-1} < 1`` and raw predicted
    quantiles ``values``:
    1. **Monotone rearrangement & non-negativity**:
       Sorts ``values`` in ascending order (eliminating quantile crossing without increasing
       pinball loss) and clips negative predictions to ``0.0``.
    2. **Tail extrapolation**:
       - *Lower tail* (down to ``tau = 0``): linearly extrapolates the first segment
         ``(tau_0, q_0) -> (tau_1, q_1)`` back to ``tau = 0`` and clips at ``0.0``:
         ``q_zero = max(0, q_0 - slope_lo * tau_0)``.
       - *Upper tail* (above ``tau_{K-1}`` up to ``1 - tol``): fits an exponential tail
         ``Q(tau) = q_{K-1} + theta * ln((1 - tau_{K-1}) / (1 - tau))`` passing through
         the last two knots ``(tau_{K-2}, q_{K-2})`` and ``(tau_{K-1}, q_{K-1})``:
         ``theta = (q_{K-1} - q_{K-2}) / ln((1 - tau_{K-2}) / (1 - tau_{K-1}))``.
         Evaluates ``q_upper = Q(1 - tol)`` and sets the integer support cap
         ``y_max = floor(q_upper + 0.5)`` (optionally capped at ``max_support``).
    3. **Continuous CDF evaluation at half-integers**:
       Defines integer demand via rounding: ``D = round(Q(U))`` for ``U ~ Uniform(0, 1)``,
       so ``P(D <= y) = P(Q(U) < y + 0.5)``. Evaluates ``cdf[y] = Q^{-1}(y + 0.5 - 1e-9)``
       for ``y = 0, 1, ..., y_max`` using piecewise-linear interpolation on ``[0, q_{K-1}]``
       and the exact exponential CDF ``1 - (1 - tau_{K-1}) * exp(-(x - q_{K-1}) / theta)``
       above ``q_{K-1}``, lumping all remaining tail mass at ``cdf[y_max] = 1.0``.
    4. **Differencing & support trimming**:
       Computes ``pmf[y] = max(0, cdf[y] - cdf[y - 1])``, slices from the first to the last
       positive-probability integer ``[first, last]``, and normalizes to sum to ``1.0``.

    Why it is needed
    ----------------
    Distribution-free quantile models (CatBoost ``MultiQuantile``, ``EmpiricalQuantileForecaster``)
    output a finite set of quantiles per day, whereas convolution (``D_1 + ... + D_L``) requires
    a full discrete PMF on the non-negative integers. Compiling this reconstruction in Numba
    allows converting hundreds of thousands of (item, day, horizon) quantile vectors per second.

    Where it is used
    ----------------
    - :func:`_quantiles_to_mean_batch_numba` (computing expected value ``mu`` from quantiles)
    - :func:`_ddlt_from_quantiles_row_numba` (building daily PMFs inside the quantile DDLT loop)
    - :func:`probabilistic_modeling.distributions.quantile.quantiles_to_pmf` (and ``quantiles_to_distribution``)
    """
    k_len = len(levels)
    # 1. Monotone rearrangement (sort) and clip at 0:
    q = np.sort(values.astype(np.float64))
    for i in range(k_len):
        if q[i] < 0.0:
            q[i] = 0.0

    # 2a. Linear lower-tail extrapolation down to tau = 0:
    slope_lo = (q[1] - q[0]) / (levels[1] - levels[0])
    q_zero = q[0] - slope_lo * levels[0]
    if q_zero < 0.0:
        q_zero = 0.0

    # 2b. Exponential upper-tail scale parameter theta from the last two quantile knots:
    theta = (q[k_len - 1] - q[k_len - 2]) / np.log((1.0 - levels[k_len - 2]) / (1.0 - levels[k_len - 1]))
    if tol < 1.0 - levels[k_len - 1]:
        q_upper = q[k_len - 1] + theta * np.log((1.0 - levels[k_len - 1]) / tol)
    else:
        q_upper = q[k_len - 1]

    y_max = int(np.floor(q_upper + 0.5))
    if max_support >= 0 and y_max > max_support:
        y_max = max_support
    if y_max < 0:
        y_max = 0

    # 3. Build piecewise-linear knots (tau = 0, tau_0, ..., tau_{K-1}) and evaluate CDF at y + 0.5 - eps:
    knots_q = np.empty(k_len + 1, dtype=np.float64)
    knots_tau = np.empty(k_len + 1, dtype=np.float64)
    knots_q[0] = q_zero
    knots_tau[0] = 0.0
    for i in range(k_len):
        knots_q[i + 1] = q[i]
        knots_tau[i + 1] = levels[i]

    x = np.arange(y_max + 1, dtype=np.float64) + 0.5 - _LEFT_LIMIT_EPS
    cdf = np.interp(x, knots_q, knots_tau)

    # Apply exponential upper tail for half-integers exceeding the highest quantile knot q_{K-1}:
    q_last = q[k_len - 1]
    tau_last = levels[k_len - 1]
    for i in range(y_max + 1):
        if x[i] > q_last:
            if theta > 0.0:
                cdf[i] = 1.0 - (1.0 - tau_last) * np.exp(-(x[i] - q_last) / theta)
            else:
                cdf[i] = 1.0
    # Lump remaining upper-tail mass (<= tol) into the final bin y_max:
    cdf[y_max] = 1.0

    # 4. Difference the discrete CDF to obtain the PMF and trim leading/trailing zeros:
    pmf_full = np.empty(y_max + 1, dtype=np.float64)
    prev = 0.0
    first = -1
    last = 0
    for i in range(y_max + 1):
        diff = cdf[i] - prev
        prev = cdf[i]
        if diff < 0.0:
            diff = 0.0
        pmf_full[i] = diff
        if diff > 0.0:
            if first == -1:
                first = i
            last = i

    if first == -1:
        out = np.ones(1, dtype=np.float64)
        return out, 0

    out = pmf_full[first : last + 1].copy()
    s = np.sum(out)
    if s > 0.0:
        out /= s
    return out, first


@njit(parallel=True, cache=True)
def _quantiles_to_mean_batch_numba(
    levels: np.ndarray,
    values_matrix: np.ndarray,
    tol: float = 1e-4,
    max_support: int = -1,
) -> np.ndarray:
    """
    Parallel Numba computation of the reconstructed discrete distribution mean ``E[D]``
    for every row of a ``(n_rows, n_quantiles)`` matrix.

    What it does
    ------------
    Uses ``numba.prange`` across rows to reconstruct each row's discrete PMF via
    :func:`_quantiles_to_pmf_numba` and compute ``sum_k (min_val + k) * pmf[k]`` without
    persisting the intermediate PMF arrays in memory.

    Why it is needed
    ----------------
    Quantile regression predicts quantiles (including the median ``q_0.5``), not the mean
    ``E[D]``. For skewed/intermittent count data, the median is often 0 while the mean is > 0,
    so point accuracy metrics (WMAPE, bias) require the expected value ``target_t_plus_<h>_mu``
    of the reconstructed distribution.

    Where it is used
    ----------------
    - :func:`probabilistic_modeling.distributions.quantile.quantiles_to_mean`
    - :meth:`CatBoostQuantileForecaster.predict` (populates ``target_t_plus_<h>_mu``)
    """
    n_rows = values_matrix.shape[0]
    means = np.empty(n_rows, dtype=np.float64)
    for i in prange(n_rows):
        pmf, min_val = _quantiles_to_pmf_numba(levels, values_matrix[i], tol, max_support)
        acc = 0.0
        for k in range(len(pmf)):
            acc += (min_val + k) * pmf[k]
        means[i] = acc
    return means


@njit(cache=True)
def _ddlt_from_quantiles_row_numba(
    lt_pmf: np.ndarray,
    lt_min: int,
    levels: np.ndarray,
    row_quantiles: np.ndarray,
    default_pmf: np.ndarray,
    default_min: int,
    tol: float = 1e-4,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Fused single-row kernel that converts one item-day's multi-horizon quantile matrix
    ``row_quantiles`` of shape ``(horizon, n_quantiles)`` into daily PMFs and computes
    its random-sum DDLT distribution ``(ddlt_pmf, ddlt_min)`` in one compiled call.

    What it does
    ------------
    1. For each forecast horizon ``h = 0, ..., horizon - 1``, reconstructs the daily PMF
       ``(pmf, min_val)`` from ``row_quantiles[h]`` using :func:`_quantiles_to_pmf_numba`.
    2. If the maximum replenishment window ``lt_max = max(L + R)`` exceeds ``horizon``,
       pads the remaining days ``h = horizon, ..., lt_max - 1`` with the series' historical
       fallback distribution ``(default_pmf, default_min)``.
    3. Calls :func:`_random_sum_aligned_numba` to compute the lead-time mixture of convolutions.

    Why it is needed
    ----------------
    Keeping both the quantile-to-PMF reconstruction for all ``H`` horizons and the random-sum
    convolution inside a single Numba call avoids creating ``H`` Python ``DiscreteDistribution``
    wrapper objects per row.

    Where it is used
    ----------------
    Called per row inside :meth:`DDLTCalculator.from_quantiles_group`.
    """
    horizon = row_quantiles.shape[0]
    lt_max = lt_min + len(lt_pmf) - 1
    total_days = lt_max if lt_max > horizon else horizon

    rv_pmfs = NumbaList()
    rv_mins = np.empty(total_days, dtype=np.int64)
    for h in range(horizon):
        pmf, min_val = _quantiles_to_pmf_numba(levels, row_quantiles[h], tol, -1)
        rv_pmfs.append(pmf)
        rv_mins[h] = min_val
    for h in range(horizon, total_days):
        rv_pmfs.append(default_pmf)
        rv_mins[h] = default_min

    return _random_sum_aligned_numba(lt_pmf, lt_min, rv_pmfs, rv_mins, 1, tail_tol)


# Hard cap on the number of support bins (50 million) when dynamically growing a Poisson/NB2
# PMF array in `_count_pmf_truncated_numba`, guarding against runaway allocations if extreme
# parameters are passed.
_COUNT_PMF_MAX_SUPPORT = 50_000_000


@njit(cache=True)
def _count_pmf_truncated_numba(mu: float, alpha: float, tol: float) -> np.ndarray:
    """
    Build a truncated Poisson(``mu``) or Negative-Binomial NB2(``mu``, ``alpha``) PMF on
    ``[0, K]`` directly in compiled Numba code, lumping the upper tail ``1 - F(K)`` into bin ``K``.

    What it does
    ------------
    - When ``alpha <= 0`` or ``alpha`` is ``NaN``, evaluates the **Poisson(mu)** PMF:
          log P(X = k) = k * log(mu) - mu - lgamma(k + 1)
    - When ``alpha > 0``, evaluates the **NB2(mu, alpha)** PMF with mean ``mu`` and variance
      ``mu * (1 + alpha * mu)``, matching SciPy's ``stats.nbinom(n, p)`` parameterization
      where ``n = 1 / alpha`` and ``p = 1 / (1 + mu * alpha)``:
          log P(X = k) = lgamma(k + n) - lgamma(n) - lgamma(k + 1) + n * log(p) + k * log(1 - p)
    - Pre-allocates an initial buffer of ``int(mu + 12 * sd + 32)`` bins (which covers almost
      all normal tolerances without reallocation) and dynamically doubles the buffer if ``k``
      exceeds capacity (for extreme heavy tails or ultra-tight ``tol``).
    - Stops at the smallest ``K`` where the cumulative sum ``cdf >= 1 - tol``, adds the
      remaining tail mass ``1 - cdf`` into the last bin ``out[K]`` (lumping, matching
      :func:`probabilistic_modeling.distributions.discrete._lump_upper_tail`), and normalizes.

    Why it is needed
    ----------------
    Previously, computing DDLT for a GLM required calling ``scipy.stats.nbinom.ppf`` and
    ``scipy.stats.nbinom.pmf`` in Python for every ``(item, day, horizon)`` — hundreds of
    thousands of SciPy object dispatches per store. Evaluating the PMF in log space via
    ``math.lgamma`` inside Numba eliminates SciPy overhead completely (~20x speedup on GLM DDLT)
    and prevents floating-point underflow when ``mu`` is large.

    Where it is used
    ----------------
    Called by :func:`_ddlt_from_glm_row_numba` (and tested against SciPy in ``tests/test_glm_ddlt_numba.py``).
    """
    if not (mu > 0.0):
        return np.ones(1, dtype=np.float64)
    use_nb = alpha > 0.0  # Evaluates to False when alpha is NaN or <= 0 (Poisson case)
    target = 1.0 - tol

    n = 0.0
    log_p_term = 0.0
    log_q = 0.0
    lg_n = 0.0
    log_mu = 0.0
    if use_nb:
        n = 1.0 / alpha
        p = 1.0 / (1.0 + mu * alpha)
        log_p_term = n * math.log(p)
        log_q = math.log1p(-p) if p < 1.0 else -np.inf
        lg_n = math.lgamma(n)
        sd = math.sqrt(mu * (1.0 + mu * alpha))
    else:
        log_mu = math.log(mu)
        sd = math.sqrt(mu)

    # Initial buffer size: mean + 12 standard deviations + 32 bins
    cap = int(mu + 12.0 * sd + 32.0)
    pmf = np.empty(cap, dtype=np.float64)
    cdf = 0.0
    k = 0
    while True:
        # Dynamically double buffer if heavy NB2 tail exceeds initial capacity:
        if k >= cap:
            if cap >= _COUNT_PMF_MAX_SUPPORT:
                k -= 1
                break
            new_cap = min(cap * 2, _COUNT_PMF_MAX_SUPPORT)
            grown = np.empty(new_cap, dtype=np.float64)
            grown[:cap] = pmf
            pmf = grown
            cap = new_cap
        if use_nb:
            lp = math.lgamma(k + n) - lg_n - math.lgamma(k + 1.0) + log_p_term
            if k > 0:
                lp += k * log_q
        else:
            lp = k * log_mu - mu - math.lgamma(k + 1.0)
        pk = math.exp(lp)
        pmf[k] = pk
        cdf += pk
        if cdf >= target:
            break
        k += 1

    out = pmf[: k + 1].copy()
    # Lump the truncated upper-tail probability mass 1 - F(K) into the last bin K:
    tail = 1.0 - cdf
    if tail > 0.0:
        out[k] += tail
    s = np.sum(out)
    if s > 0.0:
        out /= s
    return out


@njit(cache=True)
def _hurdle_count_pmf_truncated_numba(pi: float, lam: float, alpha: float, tol: float) -> np.ndarray:
    """
    Build a truncated Zero-Hurdle Poisson(``pi``, ``lam``) or Zero-Hurdle Negative-Binomial
    NB2(``pi``, ``lam``, ``alpha``) PMF on ``[0, K]`` directly in compiled Numba code.

    What it does
    ------------
    Models a two-part hurdle count distribution where:
    - ``P(Y = 0) = 1 - pi`` (with ``pi = P(Y > 0)`` in ``[0, 1]``), and
    - For ``k >= 1``, ``P(Y = k) = pi * p_base(k) / (1 - p_base(0))``, where ``p_base`` is the
      untruncated Poisson(``lam``) PMF (when ``alpha <= 0`` or ``NaN``) or NB2(``lam``, ``alpha``)
      PMF (when ``alpha > 0``).
    - If ``pi`` is ``NaN``, falls back to :func:`_count_pmf_truncated_numba` with mean ``lam``.
    - If ``pi <= 0``, returns a degenerate point mass ``[1.0]`` at ``0``.
    - If ``lam <= 1e-12`` (or ``1 - p_base(0) <= 1e-15``), all positive mass concentrates at
      ``k = 1``, returning ``[1 - pi, pi]`` on ``{0, 1}``.
    - Accumulates the zero-truncated conditional CDF ``cdf_pos = sum_{j=1}^k q_+(j)`` until
      ``cdf_pos >= 1 - tol`` (guaranteeing that the unconditional truncated upper tail is at most
      ``pi * tol <= tol`` and that the unconditional mean ``pi * lam / (1 - p_base(0))`` is
      preserved even when ``pi`` is very small), then lumps ``pi * (1 - cdf_pos)`` into bin ``K``.

    Where it is used
    ----------------
    Called by :func:`_ddlt_from_hurdle_glm_row_numba` inside :meth:`DDLTCalculator.from_glm_group`.
    """
    if math.isnan(pi):
        return _count_pmf_truncated_numba(lam, alpha, tol)
    if not (pi > 0.0):
        return np.ones(1, dtype=np.float64)
    if pi > 1.0:
        pi = 1.0
    if not (lam > 1e-12):
        out_deg = np.empty(2, dtype=np.float64)
        out_deg[0] = 1.0 - pi
        out_deg[1] = pi
        return out_deg

    use_nb = alpha > 0.0
    target = 1.0 - tol

    n = 0.0
    log_p_term = 0.0
    log_q = 0.0
    lg_n = 0.0
    log_lam = 0.0
    if use_nb:
        n = 1.0 / alpha
        p = 1.0 / (1.0 + lam * alpha)
        log_p_term = n * math.log(p)
        log_q = math.log1p(-p) if p < 1.0 else -np.inf
        lg_n = math.lgamma(n)
        sd = math.sqrt(lam * (1.0 + lam * alpha))
        pos_mass = -math.expm1(log_p_term)
    else:
        log_lam = math.log(lam)
        sd = math.sqrt(lam)
        pos_mass = -math.expm1(-lam)

    if not (pos_mass > 1e-15):
        out_deg = np.empty(2, dtype=np.float64)
        out_deg[0] = 1.0 - pi
        out_deg[1] = pi
        return out_deg

    log_pos_mass = math.log(pos_mass)
    cap = max(int(lam + 12.0 * sd + 32.0), 2)
    pmf = np.empty(cap, dtype=np.float64)
    pmf[0] = 1.0 - pi
    cdf_pos = 0.0
    k = 1
    while True:
        if k >= cap:
            if cap >= _COUNT_PMF_MAX_SUPPORT:
                k -= 1
                break
            new_cap = min(cap * 2, _COUNT_PMF_MAX_SUPPORT)
            grown = np.empty(new_cap, dtype=np.float64)
            grown[:cap] = pmf
            pmf = grown
            cap = new_cap
        if use_nb:
            lp = math.lgamma(k + n) - lg_n - math.lgamma(k + 1.0) + log_p_term + k * log_q - log_pos_mass
        else:
            lp = k * log_lam - lam - math.lgamma(k + 1.0) - log_pos_mass
        qk = math.exp(lp)
        pmf[k] = pi * qk
        cdf_pos += qk
        if cdf_pos >= target:
            break
        k += 1

    out = pmf[: k + 1].copy()
    tail_pos = 1.0 - cdf_pos
    if tail_pos > 0.0:
        out[k] += pi * tail_pos
    s = np.sum(out)
    if s > 0.0:
        out /= s
    return out


@njit(cache=True)
def _ddlt_from_glm_row_numba(
    lt_pmf: np.ndarray,
    lt_min: int,
    mu_row: np.ndarray,
    alpha_row: np.ndarray,
    default_pmf: np.ndarray,
    default_min: int,
    tol: float = DEFAULT_SUPPORT_TOL,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Fused single-row kernel that builds per-horizon Poisson/NB2 PMFs from GLM parameters
    ``(mu_row, alpha_row)`` and computes the random-sum DDLT distribution ``(ddlt_pmf, ddlt_min)``
    in one compiled call.

    What it does
    ------------
    1. For each forecast horizon ``h = 0, ..., horizon - 1``, builds the count PMF on ``[0, K_h]``
       from ``(mu_row[h], alpha_row[h])`` via :func:`_count_pmf_truncated_numba`.
    2. For any days beyond ``horizon`` up to ``lt_max = max(L + R)``, appends the series'
       historical fallback distribution ``(default_pmf, default_min)``.
    3. Calls :func:`_random_sum_aligned_numba` to compute the mixture of convolutions over ``L + R``.

    Why it is needed
    ----------------
    GLM counterpart of :func:`_ddlt_from_quantiles_row_numba`; keeps PMF construction and
    convolution entirely inside compiled Numba code without creating intermediate Python objects.

    Where it is used
    ----------------
    Called per row inside :meth:`DDLTCalculator.from_glm_group`.
    """
    horizon = len(mu_row)
    lt_max = lt_min + len(lt_pmf) - 1
    if lt_max <= 0:
        return np.ones(1, dtype=np.float64), 0
    total_days = lt_max if lt_max > horizon else horizon

    rv_pmfs = NumbaList()
    rv_mins = np.empty(total_days, dtype=np.int64)
    for h in range(horizon):
        rv_pmfs.append(_count_pmf_truncated_numba(mu_row[h], alpha_row[h], tol))
        rv_mins[h] = 0
    for h in range(horizon, total_days):
        rv_pmfs.append(default_pmf)
        rv_mins[h] = default_min

    return _random_sum_aligned_numba(lt_pmf, lt_min, rv_pmfs, rv_mins, 1, tail_tol)


@njit(cache=True)
def _ddlt_from_hurdle_glm_row_numba(
    lt_pmf: np.ndarray,
    lt_min: int,
    pi_row: np.ndarray,
    lam_row: np.ndarray,
    alpha_row: np.ndarray,
    default_pmf: np.ndarray,
    default_min: int,
    tol: float = DEFAULT_SUPPORT_TOL,
    tail_tol: float = DEFAULT_TAIL_TOL,
) -> tuple[np.ndarray, int]:
    """
    Fused single-row kernel that builds per-horizon Zero-Hurdle Poisson/NB2 PMFs from
    ``(pi_row, lam_row, alpha_row)`` and computes the random-sum DDLT distribution
    ``(ddlt_pmf, ddlt_min)`` in one compiled call.

    Where it is used
    ----------------
    Called per row inside :meth:`DDLTCalculator.from_glm_group` when hurdle GLM parameter
    columns (``<target>_pi`` and ``<target>_lambda``) are present.
    """
    horizon = len(lam_row)
    lt_max = lt_min + len(lt_pmf) - 1
    if lt_max <= 0:
        return np.ones(1, dtype=np.float64), 0
    total_days = lt_max if lt_max > horizon else horizon

    rv_pmfs = NumbaList()
    rv_mins = np.empty(total_days, dtype=np.int64)
    for h in range(horizon):
        rv_pmfs.append(_hurdle_count_pmf_truncated_numba(pi_row[h], lam_row[h], alpha_row[h], tol))
        rv_mins[h] = 0
    for h in range(horizon, total_days):
        rv_pmfs.append(default_pmf)
        rv_mins[h] = default_min

    return _random_sum_aligned_numba(lt_pmf, lt_min, rv_pmfs, rv_mins, 1, tail_tol)


# =============================================================================
# Section 5: High-Level DDLTCalculator Orchestrator
# =============================================================================

class DDLTCalculator:
    """
    Model-agnostic Demand-During-Lead-Time (DDLT) calculator.

    What it does
    ------------
    Combines multi-horizon daily demand forecasts in a DataFrame (represented either as
    predicted quantile columns ``target_t_plus_<h>_q<tau>`` or as GLM ``(mu, alpha)`` columns
    ``target_t_plus_<h>_mu`` / ``target_t_plus_<h>_alpha``) with a replenishment lead-time
    distribution ``L`` and review period ``R = cycle_time``.

    Depending on ``return_distribution``:
    - ``return_distribution=True``: attaches a ``demand_distribution`` column containing a
      :class:`DiscreteDistribution` object per row (used by budgeted newsvendor, calibration
      evaluation, or downstream simulation).
    - ``return_distribution=False``: directly evaluates the ``service_level`` quantile of each
      row's DDLT distribution in ``O(1)`` extra memory and attaches an integer ``target_inventory``
      column (without allocating :class:`DiscreteDistribution` objects).

    Why it is needed
    ----------------
    Decouples uncertainty propagation (convolving daily forecasts across random lead times,
    handling fallback distributions when ``max(L + R) > H``, and tightening support truncation
    for high service levels) from the specific forecasting model family.

    Where it is used
    ----------------
    - :meth:`GLMForecaster.compute_ddlt` (calls :meth:`from_glm_group`)
    - :meth:`CatBoostQuantileForecaster.compute_ddlt` (calls :meth:`from_quantiles_group`)
    - :meth:`EmpiricalQuantileForecaster.compute_ddlt` (calls :meth:`from_quantiles_group`)
    - :meth:`DiscreteTimeSeries.compute_lead_time_demand_glm` and
      :meth:`DiscreteTimeSeries.compute_lead_time_demand_catboost`
    """

    def __init__(
        self,
        lead_time: DiscreteDistribution | dict[Any, DiscreteDistribution],
        cycle_time: int = 0,
        lead_time_keys: list[str] | None = None,
        tol: float = DEFAULT_SUPPORT_TOL,
        tail_tol: float = DEFAULT_TAIL_TOL,
        max_service_level: float | None = None,
    ):
        if cycle_time < 0:
            raise ValueError("'cycle_time' must be non-negative.")
        if max_service_level is not None and not (0.0 <= float(max_service_level) <= 1.0):
            raise ValueError("'max_service_level' must be between 0 and 1.")
        self.lead_time = lead_time
        self.cycle_time = cycle_time
        self.lead_time_keys = list(lead_time_keys) if lead_time_keys else []
        self.tol = tol
        self.tail_tol = tail_tol
        self.max_service_level = max_service_level

    def effective_tol(self, n_terms: int, service_levels: np.ndarray | None = None) -> float:
        """
        Compute the per-day upper-tail support tolerance ``tol`` that keeps DDLT quantiles
        accurate up to the highest requested service level across ``n_terms = max(L + R)`` days.

        Uses the maximum of ``self.max_service_level`` and any finite values in ``service_levels``,
        then delegates to :func:`support_tol_for_service_level`. Never loosens ``self.tol``.
        """
        candidates = []
        if self.max_service_level is not None:
            candidates.append(float(self.max_service_level))
        if service_levels is not None and len(service_levels) > 0:
            finite = service_levels[np.isfinite(service_levels)]
            if len(finite) > 0:
                candidates.append(float(np.max(finite)))
        if not candidates:
            return float(self.tol)
        return support_tol_for_service_level(max(candidates), n_terms=n_terms, base_tol=float(self.tol))

    def _shifted_lead_time(self, lt: DiscreteDistribution) -> DiscreteDistribution:
        """
        Return a copy of the lead-time distribution ``L`` shifted by ``self.cycle_time`` (``R``)
        and split to unit bin width (`bin_size = 1`), representing the total protection window ``L + R``.
        """
        lt_with_cycle = lt.copy()
        lt_with_cycle.shift(self.cycle_time)
        if lt_with_cycle.min < 0:
            raise ValueError("Lead time plus cycle time must have non-negative support.")
        if lt_with_cycle.bin_size != 1:
            lt_with_cycle.split_bins()
        return lt_with_cycle

    @staticmethod
    def _build_default_demand(
        row: pd.Series, max_lt: int, horizon: int, tol: float = DEFAULT_SUPPORT_TOL
    ) -> DiscreteDistribution | None:
        """
        Construct a series' historical best-fit parametric distribution from row columns
        ``(best_fit, distr_param_0, distr_param_1, distr_param_2)`` when ``max_lt > horizon``.
        """
        if max_lt <= horizon:
            return None
        best_fit = row.get("best_fit")
        if pd.isna(best_fit) or best_fit is None:
            raise ValueError(
                f"Lead time + cycle time ({max_lt}) exceeds forecast horizon ({horizon}), "
                "and no fallback 'best_fit' distribution is present."
            )
        params = tuple(
            row[c]
            for c in ("distr_param_0", "distr_param_1", "distr_param_2")
            if c in row and not pd.isna(row[c])
        )
        return DiscreteDistribution.from_parametric(str(best_fit), params, tol=tol)

    def _default_demand_arrays(
        self, gdf: pd.DataFrame, max_lt: int, horizon: int, tol: float
    ) -> list[tuple[np.ndarray, int]]:
        """
        Build per-row ``(pmf, min)`` tuples of the historical best-fit fallback distribution
        used for days ``h >= horizon`` when ``max(L + R) > horizon``.

        Why it is needed
        ----------------
        When a prediction DataFrame has multiple test days per series (e.g. 28 test days x 3,049
        items = 85,372 rows), the fallback parameters ``(best_fit, distr_param_0..2)`` are static
        per series. Caching by ``(best_fit, params)`` tuple builds only 3,049 distributions
        instead of 85,372 and avoids slow ``DataFrame.iloc`` row slicing.
        """
        cols = ["best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
        present = [c for c in cols if c in gdf.columns]
        values = gdf[present].to_numpy(dtype=object)
        cache: dict[tuple, tuple[np.ndarray, int]] = {}
        out: list[tuple[np.ndarray, int]] = []
        for row in values:
            key = tuple(None if pd.isna(v) else v for v in row)
            hit = cache.get(key)
            if hit is None:
                distr = self._build_default_demand(pd.Series(row, index=present), max_lt, horizon, tol=tol)
                hit = (np.ascontiguousarray(distr.pmf, dtype=np.float64), int(distr.min))
                cache[key] = hit
            out.append(hit)
        return out

    def from_quantiles_group(
        self,
        gdf: pd.DataFrame,
        lead_time: DiscreteDistribution,
        target_cols: list[str],
        quantiles: tuple[float, ...] | list[float] | np.ndarray,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        period_col: str = "day",
    ) -> pd.DataFrame:
        """
        Compute DDLT distributions or ``target_inventory`` from multi-horizon quantile columns
        (``target_t_plus_<h>_q<tau>``) for a group of rows sharing a single ``lead_time`` distribution.

        What it does
        ------------
        1. Optionally filters ``gdf`` to the first decision period if ``pred_first_day=True``.
        2. Reshapes all ``horizon * len(quantiles)`` predicted quantile columns into a contiguous
           3D float64 tensor ``q_matrix`` of shape ``(n_rows, horizon, num_quantiles)``.
        3. Computes the effective support truncation tolerance ``tol_eff`` and prepares fallback
           distributions if ``max(L + R) > horizon``.
        4. Calls :func:`_ddlt_from_quantiles_row_numba` for each row and attaches either
           ``demand_distribution`` (:class:`DiscreteDistribution`) or ``target_inventory`` (`int`).
        """
        from probabilistic_modeling.models.catboost_quantile import quantile_col_name
        from probabilistic_modeling.distributions.quantile import validate_quantile_levels

        t0 = time.perf_counter()
        if pred_first_day:
            first_day = gdf[period_col].min()
            gdf = gdf[gdf[period_col] == first_day].copy()
        else:
            gdf = gdf.copy()

        levels = validate_quantile_levels(quantiles)
        horizon = len(target_cols)
        if horizon == 0:
            raise ValueError("At least one forecast horizon is required.")

        if not return_distribution:
            service_levels_arr = gdf["service_level"].to_numpy(dtype=float)
            if np.any((service_levels_arr < 0) | (service_levels_arr > 1)):
                raise ValueError("'service_levels' values must be between 0 and 1.")

        q_cols = [quantile_col_name(t, q) for t in target_cols for q in levels]
        missing_cols = [c for c in q_cols if c not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required to compute target inventory: {missing_cols}.")

        lt_with_cycle = self._shifted_lead_time(lead_time)
        lt_pmf = np.ascontiguousarray(lt_with_cycle.pmf, dtype=np.float64)
        lt_min = int(lt_with_cycle.min)
        needs_default = lt_with_cycle.max > horizon
        if needs_default:
            param_cols = ["best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
            missing_param = [c for c in param_cols if c not in gdf.columns]
            if missing_param:
                raise ValueError(f"Missing columns required to compute target inventory: {missing_param}.")

        n_terms = max(int(lt_with_cycle.max), 1)
        tol_eff = self.effective_tol(n_terms, None if return_distribution else service_levels_arr)
        logger.info(
            "Computing Numba DDLT from quantiles for %d rows (horizon=%d, quantiles=%d, L+R support=[%d, %d], tol=%.1e)...",
            len(gdf),
            horizon,
            len(levels),
            lt_with_cycle.min,
            lt_with_cycle.max,
            tol_eff,
        )

        num_q = len(levels)
        q_matrix = np.ascontiguousarray(
            gdf[q_cols].to_numpy(dtype=np.float64).reshape(len(gdf), horizon, num_q)
        )
        dummy_pmf = np.ones(1, dtype=np.float64)
        defaults = self._default_demand_arrays(gdf, int(lt_with_cycle.max), horizon, tol_eff) if needs_default else None

        results = []
        for row_num in range(len(gdf)):
            row_quantiles = q_matrix[row_num]
            if np.any(~np.isfinite(row_quantiles)):
                raise ValueError(f"Invalid CatBoost quantile forecast for row {row_num}: {row_quantiles}.")

            if needs_default:
                default_pmf, default_min = defaults[row_num]
            else:
                default_pmf = dummy_pmf
                default_min = 0

            ddlt_pmf, ddlt_min = _ddlt_from_quantiles_row_numba(
                lt_pmf, lt_min, levels, row_quantiles, default_pmf, default_min, tol_eff, self.tail_tol
            )
            if return_distribution:
                results.append(DiscreteDistribution(ddlt_pmf, ddlt_min, 1, truncation_tol=n_terms * tol_eff))
            else:
                q_val = _pmf_quantile_scalar_numba(ddlt_pmf, ddlt_min, 1, float(service_levels_arr[row_num]))
                results.append(int(q_val))

        if return_distribution:
            gdf["demand_distribution"] = results
        else:
            gdf["target_inventory"] = results
        logger.info("DDLT from quantiles completed for %d rows in %.2fs", len(gdf), time.perf_counter() - t0)
        return gdf

    def from_glm_group(
        self,
        gdf: pd.DataFrame,
        lead_time: DiscreteDistribution,
        target_cols: list[str],
        glm_mu_cols: list[str] | None = None,
        glm_alpha_cols: list[str] | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        period_col: str = "day",
        glm_pi_cols: list[str] | None = None,
        glm_lambda_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute DDLT distributions or ``target_inventory`` from GLM parameter columns
        (standard ``(mu, alpha)`` or hurdle ``(pi, lambda, alpha)``) for a group of rows
        sharing a single ``lead_time`` distribution.

        What it does
        ------------
        1. Extracts contiguous 2D arrays ``mu_arr`` and ``alpha_arr`` (plus ``pi_arr`` and
           ``lam_arr`` when hurdle columns ``<target>_pi`` and ``<target>_lambda`` are present)
           of shape ``(n_rows, horizon)``.
        2. Computes the effective truncation tolerance ``tol_eff`` and caches any historical
           fallback distributions needed when ``max(L + R) > horizon``.
        3. Calls the compiled :func:`_ddlt_from_glm_row_numba` (or
           :func:`_ddlt_from_hurdle_glm_row_numba`) kernel for each row and attaches
           either ``demand_distribution`` or ``target_inventory``.
        """
        t0 = time.perf_counter()
        if pred_first_day:
            first_day = gdf[period_col].min()
            gdf = gdf[gdf[period_col] == first_day].copy()
        else:
            gdf = gdf.copy()

        if glm_mu_cols is None:
            glm_mu_cols = [f"{col_name}_mu" for col_name in target_cols]
        if glm_alpha_cols is None:
            glm_alpha_cols = [f"{col_name}_alpha" for col_name in target_cols]
        horizon = len(glm_mu_cols)
        if horizon != len(glm_alpha_cols):
            raise ValueError("Number of 'mu' and 'alpha' columns do not match.")
        if horizon == 0:
            raise ValueError("At least one GLM forecast horizon is required.")

        default_pi_cols = [f"{col_name}_pi" for col_name in target_cols]
        default_lam_cols = [f"{col_name}_lambda" for col_name in target_cols]
        use_hurdle = (
            (glm_pi_cols is not None and glm_lambda_cols is not None)
            or all(c in gdf.columns for c in default_pi_cols + default_lam_cols)
        )
        if use_hurdle:
            if glm_pi_cols is None:
                glm_pi_cols = default_pi_cols
            if glm_lambda_cols is None:
                glm_lambda_cols = default_lam_cols
            if len(glm_pi_cols) != horizon or len(glm_lambda_cols) != horizon:
                raise ValueError("Number of 'pi', 'lambda', and 'mu' columns do not match.")

        if not return_distribution:
            service_levels_arr = gdf["service_level"].to_numpy(dtype=float)
            if np.any((service_levels_arr < 0) | (service_levels_arr > 1)):
                raise ValueError("'service_levels' values must be between 0 and 1.")

        lt_with_cycle = self._shifted_lead_time(lead_time)
        needs_default = lt_with_cycle.max > horizon
        param_cols = ["best_fit", "distr_param_0", "distr_param_1", "distr_param_2"]
        hurdle_req = (glm_pi_cols + glm_lambda_cols) if use_hurdle else []
        required_cols = glm_mu_cols + glm_alpha_cols + hurdle_req + (param_cols if needs_default else [])
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required to compute target inventory: {missing_cols}.")

        n_terms = max(int(lt_with_cycle.max), 1)
        tol_eff = self.effective_tol(n_terms, None if return_distribution else service_levels_arr)
        logger.info(
            "Computing DDLT from %sGLM parameters for %d rows (horizon=%d, L+R support=[%d, %d], tol=%.1e)...",
            "hurdle " if use_hurdle else "",
            len(gdf),
            horizon,
            lt_with_cycle.min,
            lt_with_cycle.max,
            tol_eff,
        )

        mu_arr = np.ascontiguousarray(gdf[glm_mu_cols].to_numpy(dtype=np.float64))
        alpha_arr = np.ascontiguousarray(gdf[glm_alpha_cols].to_numpy(dtype=np.float64))
        bad = ~np.isfinite(mu_arr) | (mu_arr < 0)
        if np.any(bad):
            row_num = int(np.argmax(bad.any(axis=1)))
            raise ValueError(f"Invalid GLM mean for row {row_num}: {mu_arr[row_num]}.")

        if use_hurdle:
            pi_arr = np.ascontiguousarray(gdf[glm_pi_cols].to_numpy(dtype=np.float64))
            lam_arr = np.ascontiguousarray(gdf[glm_lambda_cols].to_numpy(dtype=np.float64))
            bad_h = (~np.isnan(pi_arr) & ((pi_arr < 0.0) | (pi_arr > 1.0))) | ~np.isfinite(lam_arr) | (lam_arr < 0.0)
            if np.any(bad_h):
                row_num = int(np.argmax(bad_h.any(axis=1)))
                raise ValueError(
                    f"Invalid hurdle GLM parameters for row {row_num}: pi={pi_arr[row_num]}, lambda={lam_arr[row_num]}."
                )

        lt_pmf = np.ascontiguousarray(lt_with_cycle.pmf, dtype=np.float64)
        lt_min = int(lt_with_cycle.min)
        defaults = self._default_demand_arrays(gdf, int(lt_with_cycle.max), horizon, tol_eff) if needs_default else None
        dummy_pmf = np.ones(1, dtype=np.float64)

        results = []
        for row_num in range(len(gdf)):
            default_pmf, default_min = defaults[row_num] if defaults is not None else (dummy_pmf, 0)
            if use_hurdle:
                ddlt_pmf, ddlt_min = _ddlt_from_hurdle_glm_row_numba(
                    lt_pmf,
                    lt_min,
                    pi_arr[row_num],
                    lam_arr[row_num],
                    alpha_arr[row_num],
                    default_pmf,
                    default_min,
                    tol_eff,
                    self.tail_tol,
                )
            else:
                ddlt_pmf, ddlt_min = _ddlt_from_glm_row_numba(
                    lt_pmf, lt_min, mu_arr[row_num], alpha_arr[row_num], default_pmf, default_min, tol_eff, self.tail_tol
                )
            if return_distribution:
                results.append(DiscreteDistribution(ddlt_pmf, ddlt_min, 1, truncation_tol=n_terms * tol_eff))
            else:
                q_val = _pmf_quantile_scalar_numba(ddlt_pmf, ddlt_min, 1, float(service_levels_arr[row_num]))
                results.append(int(q_val))

        if return_distribution:
            gdf["demand_distribution"] = results
        else:
            gdf["target_inventory"] = results
        logger.info("DDLT from GLM completed for %d rows in %.2fs", len(gdf), time.perf_counter() - t0)
        return gdf
