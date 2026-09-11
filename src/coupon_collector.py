"""The coupon collector: how long to collect the whole set.

Each cereal box holds one of n equally likely coupons. How many boxes must you buy to collect
all n? The first coupon is new for sure; once you hold k distinct ones, a fresh box is new with
probability (n-k)/n, so the wait for the next new coupon is geometric with mean n/(n-k). Summing
over k = 0..n-1 gives the expected total

    E[T] = n * (1 + 1/2 + 1/3 + ... + 1/n) = n * H_n ~ n ln n + gamma*n + 1/2,

so the last few coupons dominate the cost -- collecting the final one alone takes n boxes on
average. The variance is

    Var[T] = sum_{k=1}^{n} (1-p_k)/p_k^2 < pi^2 n^2 / 6,   p_k = k/n,

and the number needed is sharply concentrated: P(T > n ln n + c n) <= e^{-c}, a coupon-collector
tail bound. The same law sets how many random samples cover a set: cache warmup, test-coverage of
n branches, random gene knockouts, dice-Yahtzee waits, and the double-collector "how long to get
two of each" all reduce to it.

This module gives the expected time, its variance and standard deviation, the probability that
the collection is complete after t draws (via inclusion-exclusion), the generalized "collect m
copies of each" and "collect any subset of size k" expectations, and a seeded Monte-Carlo
sampler to check them. Pure stdlib (seeded LCG); the discrete-probability companion to the
Benford and random-walk notes.
"""

from __future__ import annotations

import math


def harmonic(n: int) -> float:
    """The n-th harmonic number H_n = sum_{k=1}^{n} 1/k."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return sum(1.0 / k for k in range(1, n + 1))


def expected_time(n: int) -> float:
    """Expected number of draws to collect all n coupons: E[T] = n * H_n."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    return n * harmonic(n)


def expected_time_approx(n: int) -> float:
    """Asymptotic E[T] ~ n ln n + gamma*n + 1/2 (Euler-Mascheroni gamma)."""
    gamma = 0.5772156649015329
    return n * math.log(n) + gamma * n + 0.5


def variance(n: int) -> float:
    """Var[T] = sum_{k=1}^{n} (1 - p_k) / p_k^2 with p_k = k/n."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    var = 0.0
    for k in range(1, n + 1):
        p = k / n
        var += (1.0 - p) / (p * p)
    return var


def std_dev(n: int) -> float:
    """Standard deviation of the collection time."""
    return math.sqrt(variance(n))


def expected_partial(n: int, k: int) -> float:
    """Expected draws to collect any k distinct coupons out of n (k <= n):
    E = n * (1/n + 1/(n-1) + ... + 1/(n-k+1)) = n * (H_n - H_{n-k})."""
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    return n * (harmonic(n) - harmonic(n - k))


def expected_time_m_copies(n: int, m: int) -> float:
    """Expected draws to collect at least m copies of every coupon (the Newman-Shepp
    double-dixie-cup problem). Exact via E[T] = integral_0^inf (1 - prod(1 - S_m(t/n))) dt
    is awkward; use the accurate asymptotic

        E[T_m] ~ n ln n + (m-1) n ln ln n + gamma_m * n,

    which for m = 1 reduces to the classic n ln n + gamma n. Returns that estimate.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if m == 1:
        return expected_time_approx(n)
    gamma = 0.5772156649015329
    # gamma_m = gamma - ln((m-1)!) ; here the leading correction is the ln ln n term
    corr = gamma - math.log(math.factorial(m - 1))
    return n * math.log(n) + (m - 1) * n * math.log(math.log(n)) + corr * n


def prob_complete_by(n: int, t: int) -> float:
    """Probability that all n coupons are collected within t draws.

    By inclusion-exclusion, P(T <= t) = sum_{j=0}^{n} (-1)^j C(n, j) (1 - j/n)^t.
    """
    if t < n:
        return 0.0
    total = 0.0
    for j in range(n + 1):
        term = math.comb(n, j) * ((n - j) / n) ** t
        total += -term if (j % 2) else term
    return max(0.0, min(1.0, total))


def tail_bound(n: int, c: float) -> float:
    """Upper bound P(T > n ln n + c n) <= e^{-c} (valid for c >= 0)."""
    return math.exp(-c)


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def simulate(n: int, trials: int = 2000, seed: int = 1, copies: int = 1):
    """Monte-Carlo mean collection time over `trials` runs of collecting `copies` of each of n
    coupons. Returns (mean, sample_std). Uses a seeded LCG so results are reproducible."""
    rng = _Rng(seed)
    times = []
    for _ in range(trials):
        have = [0] * n
        done = 0
        draws = 0
        while done < n:
            draws += 1
            c = rng.randint(n)
            have[c] += 1
            if have[c] == copies:
                done += 1
        times.append(draws)
    mean = sum(times) / trials
    var = sum((t - mean) ** 2 for t in times) / trials
    return mean, math.sqrt(var)
