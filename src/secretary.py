"""The secretary problem: optimal stopping and the 1/e rule.

You interview n candidates one at a time in random order. After each you must accept or reject
on the spot -- no going back -- and you only care about landing the single best. What strategy
maximizes the chance of hiring the top candidate?

The optimal policy is a cutoff rule: reject the first r-1 candidates outright (a "look" phase),
remember the best of them, then accept the first later candidate who beats all seen so far. For
a threshold r the win probability is

    P(r) = (r-1)/n * sum_{i=r}^{n} 1/(i-1),

maximized near r ~ n/e, and as n grows both the optimal look-fraction and the win probability
tend to

    1/e ~ 0.3679.

So you should look (and reject) about 37% of the candidates, then leap at the next record --
and you land the very best about 37% of the time, remarkably, no matter how large n is. The
same optimal-stopping law governs "the best apartment while flat-hunting", online auctions and
the parking problem, and it is the seed of the theory of optimal stopping and prophet
inequalities.

This module gives the exact win probability for any cutoff, the optimal cutoff and its
probability, the 1/e asymptotics, the expected rank variant, and a seeded Monte-Carlo sampler
that plays the game to confirm the theory. Pure stdlib (seeded LCG); the optimal-stopping
companion to the coupon-collector and Benford notes.
"""

from __future__ import annotations

import math


def win_probability(n: int, r: int) -> float:
    """Probability of selecting the single best of n candidates using cutoff r: reject the
    first r-1, then take the first later candidate better than all before it.

        P(r) = (r-1)/n * sum_{i=r}^{n} 1/(i-1).

    r = 1 means "take the first candidate" (win probability 1/n)."""
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n")
    if r == 1:
        return 1.0 / n
    s = sum(1.0 / (i - 1) for i in range(r, n + 1))
    return (r - 1) / n * s


def optimal_cutoff(n: int):
    """The cutoff r (1..n) maximizing the win probability, and that probability, as (r, P)."""
    best_r, best_p = 1, win_probability(n, 1)
    for r in range(2, n + 1):
        p = win_probability(n, r)
        if p > best_p:
            best_r, best_p = r, p
    return best_r, best_p


def optimal_fraction(n: int) -> float:
    """The optimal look-fraction (r-1)/n -- the share of candidates rejected outright."""
    r, _ = optimal_cutoff(n)
    return (r - 1) / n


def asymptotic_probability() -> float:
    """The large-n limit of the win probability: 1/e."""
    return 1.0 / math.e


def asymptotic_fraction() -> float:
    """The large-n limit of the optimal look-fraction: 1/e."""
    return 1.0 / math.e


def expected_candidates_seen(n: int, r: int) -> float:
    """Expected number of candidates interviewed (including the look phase) before stopping.
    If no later candidate beats the look-phase best, all n are seen."""
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n")
    if r == 1:
        return 1.0  # no look phase: accept the very first candidate
    # P(stop exactly at position k), k = r..n: position k holds the running max of the first k
    # AND the overall max of the first k is not in the look phase... use the standard result
    # P(stop at k) = (r-1)/((k-1)k) for k >= r (prob k is a record beating the look best),
    # plus the residual mass of never stopping (all n seen).
    exp = 0.0
    stop_mass = 0.0
    for k in range(r, n + 1):
        p_stop = (r - 1) / ((k - 1) * k) if k > 1 else 0.0
        exp += k * p_stop
        stop_mass += p_stop
    exp += n * (1.0 - stop_mass)  # never stopped -> saw all n
    return exp


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def randint(self, k: int) -> int:
        return (self._next() >> 16) % k

    def shuffle(self, a):
        """In-place Fisher-Yates shuffle."""
        for i in range(len(a) - 1, 0, -1):
            j = self.randint(i + 1)
            a[i], a[j] = a[j], a[i]


def simulate(n: int, r: int, trials: int = 5000, seed: int = 1) -> float:
    """Monte-Carlo win rate: shuffle ranks 1..n (n = best), apply the cutoff-r rule, and count
    how often the chosen candidate is the overall best. Returns the empirical probability."""
    rng = _Rng(seed)
    wins = 0
    for _ in range(trials):
        order = list(range(1, n + 1))
        rng.shuffle(order)
        # look phase: best (highest rank) among first r-1
        look_best = max(order[:r - 1]) if r > 1 else 0
        chosen = order[-1]  # default: forced to take the last if none qualifies
        for k in range(r - 1, n):
            if order[k] > look_best:
                chosen = order[k]
                break
        if chosen == n:
            wins += 1
    return wins / trials
