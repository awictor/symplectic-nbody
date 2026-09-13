"""Tests for variates: moments match analytic, chi-square GOF, support constraints, special cases."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from variates import Variates, analytic_moments, sample_stats  # noqa: E402


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def moments_ok(samples, mean, var, n):
    em, ev = sample_stats(samples)
    # standard error of the mean ~ sqrt(var/n); allow 4 SE
    se = math.sqrt(var / n) if var > 0 else 1e-6
    return abs(em - mean) < 5 * se and abs(ev - var) < 0.1 * var + 0.05


def chi_square_uniform_bins(samples, cdf, bins=10):
    """Chi-square GOF: bin samples by equal-probability CDF ranges; return the statistic."""
    n = len(samples)
    lo, hi = min(samples), max(samples)
    # bin by value range, expected via cdf differences
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    observed = [0] * bins
    for x in samples:
        b = min(bins - 1, int((x - lo) / (hi - lo) * bins)) if hi > lo else 0
        observed[b] += 1
    chi = 0.0
    for i in range(bins):
        p = cdf(edges[i + 1]) - cdf(edges[i])
        expected = n * p
        if expected >= 5:
            chi += (observed[i] - expected) ** 2 / expected
    return chi


def main():
    N = 40000

    # ---- 1. moments match analytic ----------------------------------------------------
    v = Variates(2024)
    tests = [
        ("exponential", lambda: v.exponential(2.0), dict(rate=2.0)),
        ("exponential", lambda: v.exponential(0.5), dict(rate=0.5)),
        ("normal", lambda: v.normal(5, 2), dict(mu=5, sigma=2)),
        ("gamma", lambda: v.gamma(3.0, 2.0), dict(shape=3.0, scale=2.0)),
        ("gamma", lambda: v.gamma(0.5, 4.0), dict(shape=0.5, scale=4.0)),  # shape < 1 boost path
        ("beta", lambda: v.beta(2.0, 5.0), dict(a=2.0, b=5.0)),
        ("poisson", lambda: v.poisson(4.0), dict(lam=4.0)),
        ("binomial", lambda: v.binomial(20, 0.3), dict(n=20, p=0.3)),
        ("geometric", lambda: v.geometric(0.25), dict(p=0.25)),
    ]
    for name, gen, params in tests:
        samples = [gen() for _ in range(N)]
        mean, var = analytic_moments(name, **params)
        check(f"{name} {params} moments match", moments_ok(samples, mean, var, N),
              f"emp {sample_stats(samples)} vs ({mean:.3f},{var:.3f})")

    # ---- 2. chi-square goodness-of-fit for the exponential ----------------------------
    v = Variates(7)
    rate = 1.5
    samples = [v.exponential(rate) for _ in range(N)]
    def exp_cdf(x):
        return 1 - math.exp(-rate * x) if x >= 0 else 0.0
    chi = chi_square_uniform_bins(samples, exp_cdf, bins=12)
    # ~11 dof, 1% critical ~ 24.7; be generous (bins with expected<5 dropped)
    check("exponential passes chi-square GOF", chi < 30, f"chi={chi:.2f}")

    # ---- 3. Poisson pmf matches observed frequencies ----------------------------------
    v = Variates(11)
    lam = 3.0
    samples = [v.poisson(lam) for _ in range(N)]
    counts = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    chi = 0.0
    for k in range(0, 12):
        pmf = math.exp(-lam) * lam ** k / math.factorial(k)
        expected = N * pmf
        if expected >= 5:
            chi += (counts.get(k, 0) - expected) ** 2 / expected
    check("Poisson pmf matches observed (chi-square)", chi < 30, f"chi={chi:.2f}")

    # ---- 4. binomial pmf matches observed ---------------------------------------------
    v = Variates(13)
    n, p = 10, 0.4
    samples = [v.binomial(n, p) for _ in range(N)]
    counts = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    chi = 0.0
    for k in range(n + 1):
        pmf = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        expected = N * pmf
        if expected >= 5:
            chi += (counts.get(k, 0) - expected) ** 2 / expected
    check("binomial pmf matches observed (chi-square)", chi < 30, f"chi={chi:.2f}")

    # ---- 5. support constraints -------------------------------------------------------
    v = Variates(5)
    check("exponential non-negative", all(v.exponential(1.0) >= 0 for _ in range(1000)))
    check("gamma non-negative", all(v.gamma(2.0, 1.0) >= 0 for _ in range(1000)))
    check("beta in [0,1]", all(0 <= v.beta(2, 3) <= 1 for _ in range(1000)))
    check("poisson non-negative integer",
          all(isinstance(x, int) and x >= 0 for x in (v.poisson(5) for _ in range(1000))))
    check("binomial in [0,n]", all(0 <= v.binomial(10, 0.5) <= 10 for _ in range(1000)))
    check("geometric >= 1", all(v.geometric(0.3) >= 1 for _ in range(1000)))

    # ---- 6. reproducibility -----------------------------------------------------------
    a = Variates(999)
    b = Variates(999)
    check("same seed -> identical stream",
          [a.gamma(2, 1) for _ in range(50)] == [b.gamma(2, 1) for _ in range(50)])
    c = Variates(1000)
    check("different seed -> different stream",
          [a.exponential(1) for _ in range(20)] != [c.exponential(1) for _ in range(20)])

    # ---- 7. special cases -------------------------------------------------------------
    # Gamma(1, theta) is Exponential(rate=1/theta): mean theta
    v = Variates(3)
    g1 = [v.gamma(1.0, 3.0) for _ in range(N)]
    check("Gamma(1, 3) mean ~ 3 (= exponential)", abs(sample_stats(g1)[0] - 3.0) < 0.1)
    # Beta(1,1) is uniform on [0,1]: mean 0.5, var 1/12
    b11 = [v.beta(1.0, 1.0) for _ in range(N)]
    m, var = sample_stats(b11)
    check("Beta(1,1) is uniform (mean 0.5, var 1/12)", abs(m - 0.5) < 0.02 and abs(var - 1 / 12) < 0.01)

    # ---- 8. large-lambda Poisson uses the approximation but stays sane -----------------
    v = Variates(17)
    big = [v.poisson(100.0) for _ in range(5000)]
    m, var = sample_stats(big)
    check("large-lambda Poisson mean ~ 100", abs(m - 100) < 5)
    check("large-lambda Poisson variance ~ 100", abs(var - 100) < 20)

    # ---- 9. parameter validation ------------------------------------------------------
    try:
        v.exponential(-1)
        check("bad rate raises", False)
    except ValueError:
        check("bad rate raises", True)
    try:
        v.beta(0, 1)
        check("bad beta param raises", False)
    except ValueError:
        check("bad beta param raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
