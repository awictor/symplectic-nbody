"""The Kelly criterion: how much to bet to grow fastest.

You have an edge -- a bet that pays b-to-1 and wins with probability p > 1/(b+1), so its
expected value is positive. How much of your bankroll should you stake each time? Bet too little
and you leave growth on the table; bet too much and a losing streak wipes you out. John Kelly's
answer (1956) maximizes the long-run exponential growth rate of the bankroll, and it is a fixed
FRACTION of your current wealth:

    f* = (p b - q) / b = p - q/b,     q = 1 - p,

the edge divided by the odds. Because you compound, the right objective is the expected log
return, and the growth rate at fraction f is

    g(f) = p ln(1 + b f) + q ln(1 - f),

a concave curve peaking at f*. Betting f* grows the bankroll faster in the long run than any
other constant fraction -- but the curve is asymmetric: overbetting past 2 f* actually loses
money (negative growth), while underbetting is safe but slow. Betting the full edge is
famously volatile, so practitioners use "fractional Kelly" (half-Kelly gives ~3/4 of the growth
at far less variance). The same log-optimal rule sizes positions in quantitative finance and
information-theoretic gambling (Kelly derived it from Shannon's channel capacity).

This module gives the optimal fraction, the growth rate at any fraction, the growth-optimal
value, the break-even (zero-growth) fraction, the doubling time, and a seeded Monte-Carlo of the
compounding bankroll that confirms f* wins. Pure stdlib; the log-growth companion to the
gambler's-ruin and Shannon notes.
"""

from __future__ import annotations

import math


def optimal_fraction(p: float, b: float = 1.0) -> float:
    """Kelly-optimal fraction of the bankroll to stake on a bet that wins with probability p and
    pays b-to-1: f* = (p(b+1) - 1)/b = p - (1-p)/b. Negative (do not bet) if the edge is
    negative; clamped to [0, 1]."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if b <= 0:
        raise ValueError("b (net odds) must be > 0")
    f = (p * (b + 1.0) - 1.0) / b
    return max(0.0, min(1.0, f))


def edge(p: float, b: float = 1.0) -> float:
    """Expected profit per unit staked: p b - (1-p). Positive means a favorable bet."""
    return p * b - (1.0 - p)


def growth_rate(f: float, p: float, b: float = 1.0) -> float:
    """Long-run exponential growth rate of the bankroll at bet fraction f:
    g(f) = p ln(1 + b f) + (1-p) ln(1 - f). Returns -inf if f makes ruin certain (f >= 1)."""
    q = 1.0 - p
    if f >= 1.0:
        return float("-inf")
    if f <= -1.0 / b:
        return float("-inf")
    g = 0.0
    if p > 0:
        g += p * math.log(1.0 + b * f)
    if q > 0:
        g += q * math.log(1.0 - f)
    return g


def optimal_growth_rate(p: float, b: float = 1.0) -> float:
    """The growth rate achieved by betting the Kelly fraction f*."""
    return growth_rate(optimal_fraction(p, b), p, b)


def break_even_fraction(p: float, b: float = 1.0) -> float:
    """The nonzero fraction at which the growth rate returns to zero (overbetting past this
    loses money). For b = 1 this is 2 f*; in general it solves g(f) = 0 numerically."""
    fstar = optimal_fraction(p, b)
    if fstar <= 0:
        return 0.0
    # g(0) = 0 and g(f*) > 0; g decreases back through 0 somewhere in (f*, 1). Bisect.
    lo, hi = fstar, 1.0 - 1e-12
    if growth_rate(hi, p, b) > 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if growth_rate(mid, p, b) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def doubling_time(p: float, b: float = 1.0, fraction: float = None) -> float:
    """Expected number of bets to double the bankroll at the given fraction (default: Kelly f*):
    ln 2 / g(f). Infinite if the growth rate is nonpositive."""
    f = optimal_fraction(p, b) if fraction is None else fraction
    g = growth_rate(f, p, b)
    return math.log(2.0) / g if g > 0 else float("inf")


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def simulate_growth(f: float, p: float, b: float = 1.0, bets: int = 1000,
                    trials: int = 2000, seed: int = 1) -> float:
    """Monte-Carlo mean per-bet log-growth rate from compounding a fraction f of the bankroll
    over `bets` bets, averaged across `trials`. Returns (1/bets) * mean(ln(final/initial)),
    which should match growth_rate(f, p, b)."""
    rng = _Rng(seed)
    total_log = 0.0
    for _ in range(trials):
        logw = 0.0
        for _ in range(bets):
            if rng.random() < p:
                logw += math.log(1.0 + b * f)
            else:
                logw += math.log(1.0 - f)
                if 1.0 - f <= 0:
                    logw = float("-inf")
                    break
        total_log += logw
    return total_log / (trials * bets)
