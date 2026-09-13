"""Kernel density estimation: recovering a smooth probability density from samples, no model assumed.

A histogram estimates a distribution by binning, but its blocky, bin-placement-dependent shape throws
away information. KERNEL DENSITY ESTIMATION replaces each data point with a smooth little bump (the
KERNEL) and sums them: the estimate at x is the average of a kernel centred on every sample,

    f_hat(x) = (1 / n h) sum_i K((x - x_i) / h),

where h is the BANDWIDTH -- the width of each bump, the one crucial tuning knob. Too small and the
estimate is spiky and overfit; too large and real features wash out. KDE is nonparametric: it assumes
no functional form, only smoothness, and converges to the true density as n grows. It is the workhorse
of exploratory data analysis, the smooth cousin of the histogram, and the density model behind
mean-shift clustering and naive-Bayes with continuous features.

The kernel K is any symmetric density integrating to one -- Gaussian, Epanechnikov (the
variance-optimal choice), triangular, uniform, cosine. The bandwidth is usually chosen by a
plug-in rule that is optimal for near-Gaussian data: SILVERMAN'S rule h = 0.9 A n^(-1/5) with
A = min(std, IQR/1.34), or SCOTT'S h = std * n^(-1/5). This module estimates the density on a grid or
at arbitrary points, offers all those kernels and bandwidth rules, and an ADAPTIVE (variable-
bandwidth) estimator that widens the kernel in sparse regions.

Validated by the defining properties and against known distributions: the estimate is everywhere
non-negative and integrates to one (numerically), it recovers a standard normal and a bimodal mixture
so their KDE matches the true density in mean and shape, wider bandwidths are smoother (lower total
variation), the Silverman and Scott rules give sensible positive bandwidths that shrink as n grows,
and a leave-one-out likelihood peaks near the plug-in bandwidth. Pure stdlib; the nonparametric-density
companion to the histogram, Gaussian-mixture, and mean-shift tools."""

from __future__ import annotations

import math


SQRT2PI = math.sqrt(2 * math.pi)


# --- kernels (each integrates to 1, symmetric, unit-ish scale) ----------------
def gaussian_kernel(u):
    return math.exp(-0.5 * u * u) / SQRT2PI


def epanechnikov_kernel(u):
    return 0.75 * (1 - u * u) if abs(u) <= 1 else 0.0


def triangular_kernel(u):
    return (1 - abs(u)) if abs(u) <= 1 else 0.0


def uniform_kernel(u):
    return 0.5 if abs(u) <= 1 else 0.0


def cosine_kernel(u):
    return (math.pi / 4) * math.cos(math.pi * u / 2) if abs(u) <= 1 else 0.0


KERNELS = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
    "triangular": triangular_kernel,
    "uniform": uniform_kernel,
    "cosine": cosine_kernel,
}


# --- bandwidth selection -----------------------------------------------------
def _mean_std(data):
    n = len(data)
    m = sum(data) / n
    var = sum((x - m) ** 2 for x in data) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var)


def _iqr(data):
    s = sorted(data)
    n = len(s)

    def q(p):
        idx = p * (n - 1)
        lo = int(idx)
        frac = idx - lo
        if lo + 1 < n:
            return s[lo] * (1 - frac) + s[lo + 1] * frac
        return s[lo]

    return q(0.75) - q(0.25)


def silverman_bandwidth(data):
    """Silverman's rule of thumb: h = 0.9 * A * n^(-1/5), A = min(std, IQR/1.34)."""
    n = len(data)
    if n < 2:
        return 1.0
    _, std = _mean_std(data)
    iqr = _iqr(data)
    A = min(std, iqr / 1.34) if iqr > 0 else std
    if A <= 0:
        A = std if std > 0 else 1.0
    return 0.9 * A * n ** (-1 / 5)


def scott_bandwidth(data):
    """Scott's rule: h = std * n^(-1/5)."""
    n = len(data)
    if n < 2:
        return 1.0
    _, std = _mean_std(data)
    return (std if std > 0 else 1.0) * n ** (-1 / 5)


# --- the estimator -----------------------------------------------------------
class KDE:
    """A 1-D kernel density estimator."""

    def __init__(self, data, bandwidth=None, kernel="gaussian"):
        if len(data) == 0:
            raise ValueError("need at least one data point")
        self.data = list(data)
        self.n = len(data)
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel}")
        self.kernel_name = kernel
        self.kernel = KERNELS[kernel]
        if bandwidth is None:
            bandwidth = silverman_bandwidth(self.data)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.h = bandwidth

    def pdf(self, x):
        """Density estimate at a single point x."""
        h = self.h
        return sum(self.kernel((x - xi) / h) for xi in self.data) / (self.n * h)

    def evaluate(self, xs):
        """Density at each point in xs."""
        return [self.pdf(x) for x in xs]

    def integrate(self, lo, hi, steps=2000):
        """Numerically integrate the estimate over [lo, hi] (trapezoid)."""
        dx = (hi - lo) / steps
        total = 0.0
        prev = self.pdf(lo)
        for i in range(1, steps + 1):
            x = lo + i * dx
            cur = self.pdf(x)
            total += 0.5 * (prev + cur) * dx
            prev = cur
        return total

    def loo_log_likelihood(self):
        """Leave-one-out log-likelihood: sum_i log f_hat_{-i}(x_i). A bandwidth-quality score."""
        h = self.h
        total = 0.0
        for i, xi in enumerate(self.data):
            # density at xi using all points except i
            s = sum(self.kernel((xi - xj) / h) for j, xj in enumerate(self.data) if j != i)
            dens = s / ((self.n - 1) * h)
            if dens <= 0:
                return float("-inf")
            total += math.log(dens)
        return total


def adaptive_pdf(data, x, kernel="gaussian", alpha=0.5, pilot_bandwidth=None):
    """Adaptive (variable-bandwidth) KDE: the local bandwidth is scaled by (pilot density)^(-alpha),
    widening the kernel where the pilot estimate is sparse. Returns the density at x."""
    K = KERNELS[kernel]
    n = len(data)
    h0 = pilot_bandwidth if pilot_bandwidth else silverman_bandwidth(data)
    # pilot densities at each data point
    pilot = []
    for xi in data:
        s = sum(K((xi - xj) / h0) for xj in data) / (n * h0)
        pilot.append(max(s, 1e-12))
    g = math.exp(sum(math.log(p) for p in pilot) / n)  # geometric mean
    total = 0.0
    for xi, pi in zip(data, pilot):
        lam = (pi / g) ** (-alpha)
        hi = h0 * lam
        total += K((x - xi) / hi) / hi
    return total / n


def grid(lo, hi, n_points):
    """A uniform grid of n_points from lo to hi inclusive."""
    if n_points < 2:
        return [lo]
    step = (hi - lo) / (n_points - 1)
    return [lo + i * step for i in range(n_points)]
