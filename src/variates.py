"""Random variate generation -- drawing samples from the named probability distributions.

A uniform random number in [0, 1) is all a computer's RNG gives you, yet simulations, Bayesian
inference, and stochastic models need samples from EXPONENTIAL, GAMMA, BETA, NORMAL, POISSON, and
BINOMIAL distributions. Turning uniforms into these is the craft of VARIATE GENERATION, and each
distribution has its own elegant trick:

  * INVERSE TRANSFORM. If F is a distribution's CDF, then F^{-1}(U) for U uniform is a sample. The
    exponential is the clean case: -ln(1-U)/lambda, since its CDF inverts in closed form.
  * BOX-MULLER for the normal: a pair of uniforms becomes a pair of independent standard normals via
    sqrt(-2 ln U1) (cos, sin)(2 pi U2) -- polar coordinates of a 2D Gaussian.
  * MARSAGLIA-TSANG for the gamma: a fast squeeze-accepted method using a normal and a uniform, exact
    for shape >= 1 and boosted by a power-of-uniform for shape < 1. The gamma is the workhorse -- beta,
    chi-square, Student-t, and Dirichlet all reduce to it.
  * BETA from two gammas: Beta(a, b) = X/(X+Y) with X ~ Gamma(a), Y ~ Gamma(b).
  * KNUTH'S method for the Poisson: multiply uniforms until their product drops below e^{-lambda}; the
    count is the sample. (For large lambda a normal approximation avoids the long loop.)
  * BINOMIAL by summing Bernoulli trials (exact for modest n), each trial a uniform compared to p.

This module implements all of these on a seeded linear-congruential generator so every stream is
reproducible, exposing per-distribution samplers and their exact analytic mean and variance for
checking. Pure standard library -- ``math`` only.

Validation. Each sampler is checked two ways. (1) MOMENTS: over a large sample the empirical mean and
variance match the distribution's analytic mean and variance to within a few standard errors -- the
exponential's mean 1/lambda and variance 1/lambda^2, the gamma's k*theta and k*theta^2, the Poisson's
mean = variance = lambda, the binomial's np and np(1-p), and so on. (2) GOODNESS OF FIT: a chi-square
test on binned samples does not reject the claimed distribution at the 1% level, and the discrete
Poisson/binomial pmfs match their observed frequencies. Support constraints hold (exponential and gamma
non-negative, beta in [0,1], Poisson/binomial non-negative integers), the seed makes streams
reproducible, and known special cases match (Gamma(1, theta) is Exponential(1/theta); Beta(1,1) is
uniform). Pure standard library."""

import math


class Variates:
    """A seeded generator for common continuous and discrete random variates."""

    def __init__(self, seed=12345):
        self.state = seed & 0xFFFFFFFF
        self._spare_normal = None

    def _uniform(self):
        """Uniform in (0, 1), avoiding exact 0 for logs."""
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        u = (self.state >> 8) / (1 << 24)
        return u if u > 0 else 1e-12

    def uniform(self, a=0.0, b=1.0):
        return a + (b - a) * self._uniform()

    # -- exponential (inverse transform) ------------------------------------
    def exponential(self, rate=1.0):
        """Exponential(rate): mean 1/rate, via -ln(U)/rate."""
        if rate <= 0:
            raise ValueError("rate must be positive")
        return -math.log(self._uniform()) / rate

    # -- normal (Box-Muller with a cached spare) ----------------------------
    def normal(self, mu=0.0, sigma=1.0):
        if self._spare_normal is not None:
            z = self._spare_normal
            self._spare_normal = None
            return mu + sigma * z
        u1 = self._uniform()
        u2 = self._uniform()
        r = math.sqrt(-2 * math.log(u1))
        self._spare_normal = r * math.sin(2 * math.pi * u2)
        return mu + sigma * (r * math.cos(2 * math.pi * u2))

    # -- gamma (Marsaglia-Tsang) --------------------------------------------
    def gamma(self, shape, scale=1.0):
        """Gamma(shape k, scale theta): mean k*theta, variance k*theta^2."""
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        if shape < 1:
            # boost: Gamma(a) = Gamma(a+1) * U^(1/a)
            u = self._uniform()
            return self.gamma(shape + 1, scale) * (u ** (1.0 / shape))
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9 * d)
        while True:
            x = self.normal()
            v = (1 + c * x) ** 3
            if v <= 0:
                continue
            u = self._uniform()
            if u < 1 - 0.0331 * (x ** 4):
                return d * v * scale
            if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
                return d * v * scale

    # -- beta (ratio of gammas) ---------------------------------------------
    def beta(self, a, b):
        """Beta(a, b): mean a/(a+b), variance ab/((a+b)^2 (a+b+1))."""
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be positive")
        x = self.gamma(a, 1.0)
        y = self.gamma(b, 1.0)
        return x / (x + y)

    # -- Poisson (Knuth for small lambda, normal approx for large) -----------
    def poisson(self, lam):
        """Poisson(lambda): mean = variance = lambda, non-negative integer."""
        if lam < 0:
            raise ValueError("lambda must be non-negative")
        if lam == 0:
            return 0
        if lam < 30:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= self._uniform()
                if p <= L:
                    return k - 1
        # large lambda: normal approximation, rounded and clamped
        val = round(self.normal(lam, math.sqrt(lam)))
        return max(0, val)

    # -- binomial (sum of Bernoullis, exact for modest n) --------------------
    def binomial(self, n, p):
        """Binomial(n, p): mean np, variance np(1-p)."""
        if n < 0 or not (0 <= p <= 1):
            raise ValueError("need n >= 0 and 0 <= p <= 1")
        if n <= 1000:
            return sum(1 for _ in range(n) if self._uniform() < p)
        # large n: normal approximation
        val = round(self.normal(n * p, math.sqrt(n * p * (1 - p))))
        return max(0, min(n, val))

    def bernoulli(self, p):
        return 1 if self._uniform() < p else 0

    def geometric(self, p):
        """Geometric(p): number of trials until first success (>= 1). Mean 1/p."""
        if not (0 < p <= 1):
            raise ValueError("p must be in (0, 1]")
        return int(math.ceil(math.log(self._uniform()) / math.log(1 - p))) if p < 1 else 1


# ---------------------------------------------------------------------------
# analytic moments (for validation)
# ---------------------------------------------------------------------------

def analytic_moments(dist, **params):
    """Return (mean, variance) for a named distribution and its parameters."""
    if dist == "exponential":
        r = params["rate"]
        return 1 / r, 1 / (r * r)
    if dist == "normal":
        return params["mu"], params["sigma"] ** 2
    if dist == "gamma":
        k, t = params["shape"], params["scale"]
        return k * t, k * t * t
    if dist == "beta":
        a, b = params["a"], params["b"]
        m = a / (a + b)
        return m, a * b / ((a + b) ** 2 * (a + b + 1))
    if dist == "poisson":
        lam = params["lam"]
        return lam, lam
    if dist == "binomial":
        n, p = params["n"], params["p"]
        return n * p, n * p * (1 - p)
    if dist == "geometric":
        p = params["p"]
        return 1 / p, (1 - p) / (p * p)
    raise ValueError(f"unknown distribution: {dist}")


def sample_stats(samples):
    """Empirical (mean, variance) of a sample list."""
    n = len(samples)
    m = sum(samples) / n
    v = sum((x - m) ** 2 for x in samples) / n
    return m, v
