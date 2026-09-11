"""Box-Muller: turning uniform randomness into a bell curve.

A raw random-number generator gives you uniform values in [0, 1). Almost every simulation --
Brownian motion, Monte-Carlo finance, noise models, machine-learning initialization -- needs
GAUSSIAN (normal) values instead. The Box-Muller transform (1958) converts a pair of independent
uniforms into a pair of independent standard normals with a little trigonometry:

    z0 = sqrt(-2 ln u1) cos(2 pi u2),
    z1 = sqrt(-2 ln u1) sin(2 pi u2),

which is exact -- it is the polar-coordinate change of variables that maps the uniform square
onto the 2D Gaussian, whose radius r has r^2 exponentially distributed (hence sqrt(-2 ln u)) and
whose angle is uniform (hence the cos/sin). Scaling by sigma and shifting by mu gives any normal
N(mu, sigma^2).

The trig calls are the slow part, so Marsaglia's POLAR method rejects to a uniform point in the
unit disc and reuses its coordinates, avoiding cos/sin entirely -- the same output distribution,
faster. This module implements both, generates single values and samples of N(mu, sigma^2),
and verifies the output with the sample mean and variance, the standard-normal moments (skewness
~0, kurtosis ~3), and the 68-95-99.7 rule. Pure stdlib (seeded LCG); the Gaussian-sampling
companion to the alias-method and reservoir notes.
"""

from __future__ import annotations

import math


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        """Uniform float in (0, 1] -- kept strictly positive so log() is safe."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return ((self.state >> 8) + 1) / (1 << 24)


def box_muller_pair(u1: float, u2: float):
    """The basic Box-Muller transform: map two uniforms in (0, 1] to two independent standard
    normals (z0, z1)."""
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    return r * math.cos(theta), r * math.sin(theta)


def standard_normals(n: int, seed: int = 1):
    """Generate n independent standard-normal values via Box-Muller (two per uniform pair)."""
    rng = _Rng(seed)
    out = []
    while len(out) < n:
        z0, z1 = box_muller_pair(rng.random(), rng.random())
        out.append(z0)
        if len(out) < n:
            out.append(z1)
    return out


def polar_normals(n: int, seed: int = 1):
    """Generate n standard normals via Marsaglia's polar method (rejection, no trig)."""
    rng = _Rng(seed)
    out = []
    while len(out) < n:
        # sample a point uniformly in the square [-1, 1]^2, reject outside the unit disc
        while True:
            x = 2.0 * rng.random() - 1.0
            y = 2.0 * rng.random() - 1.0
            s = x * x + y * y
            if 0.0 < s < 1.0:
                break
        factor = math.sqrt(-2.0 * math.log(s) / s)
        out.append(x * factor)
        if len(out) < n:
            out.append(y * factor)
    return out


def normal(mu: float = 0.0, sigma: float = 1.0, *, rng: _Rng = None):
    """Draw a single N(mu, sigma^2) value using a supplied RNG (creates one if none given)."""
    r = rng or _Rng(1)
    z, _ = box_muller_pair(r.random(), r.random())
    return mu + sigma * z


def sample(n: int, mu: float = 0.0, sigma: float = 1.0, seed: int = 1):
    """A sample of n values from N(mu, sigma^2)."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    return [mu + sigma * z for z in standard_normals(n, seed)]


# --- descriptive statistics (for validation) -------------------------------

def mean(data) -> float:
    return sum(data) / len(data)


def variance(data) -> float:
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def skewness(data) -> float:
    """Sample skewness (third standardized moment); ~0 for a symmetric distribution."""
    m = mean(data)
    s = math.sqrt(variance(data))
    if s == 0:
        return 0.0
    return sum(((x - m) / s) ** 3 for x in data) / len(data)


def kurtosis(data) -> float:
    """Sample kurtosis (fourth standardized moment); ~3 for a normal distribution."""
    m = mean(data)
    s = math.sqrt(variance(data))
    if s == 0:
        return 0.0
    return sum(((x - m) / s) ** 4 for x in data) / len(data)


def fraction_within(data, k: float) -> float:
    """Fraction of the sample within k standard deviations of the mean -- checks the
    68-95-99.7 rule (k=1,2,3)."""
    m = mean(data)
    s = math.sqrt(variance(data))
    return sum(1 for x in data if abs(x - m) <= k * s) / len(data)
