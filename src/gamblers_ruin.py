"""Gambler's ruin: the walk that ends at a wall.

A gambler starts with i dollars and bets $1 at a time against a house, winning each round with
probability p and losing with 1-p. Play stops when the gambler is broke (0) or reaches a target
N. What is the chance of ruin, and how long does the game last? This is a random walk on
{0, 1, ..., N} with two absorbing walls, and its answers are exact.

For a FAIR game (p = 1/2) the ruin probability is beautifully simple,

    P(ruin | start i) = 1 - i/N,

so the chance of reaching N before 0 is just i/N -- your stake as a fraction of the total. The
expected number of rounds is i(N - i). For a BIASED game with q = 1-p and r = q/p != 1,

    P(ruin) = (r^i - r^N) / (1 - r^N)  ... written from the win side as (r^i-1)/(r^N-1) is the
    reach-N probability; a tiny edge against you makes ruin nearly certain as N grows.

Two lessons fall out. First, against an infinitely rich house (N -> infinity) with p <= 1/2 you
are ruined with probability 1 -- the origin of "the house always wins". Second, even in a fair
game, betting until you double your money is a coin flip, but betting until you go broke or hit
a distant target overwhelmingly ends in ruin if the target is far. The same absorbing-walk math
models fixation of a neutral or selected allele in a finite population (the Moran model),
sequential hypothesis tests, and queueing.

This module gives the exact ruin and reach-target probabilities and the expected game duration
for fair and biased games, the infinite-house limit, and a seeded Monte-Carlo sampler that
plays the walk to confirm the formulas. Pure stdlib; the absorbing-random-walk companion to the
Polya-walk and coupon-collector notes.
"""

from __future__ import annotations

import math


def ruin_probability(i: int, N: int, p: float = 0.5) -> float:
    """Probability of hitting 0 before N, starting from i, betting $1 with win probability p.

    Fair game (p = 1/2): 1 - i/N. Biased: (r^i - r^N)/(1 - r^N) with r = (1-p)/p."""
    if not 0 <= i <= N:
        raise ValueError("need 0 <= i <= N")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if i == 0:
        return 1.0
    if i == N:
        return 0.0
    if abs(p - 0.5) < 1e-15:
        return 1.0 - i / N
    r = (1.0 - p) / p
    return (r ** i - r ** N) / (1.0 - r ** N)


def reach_target_probability(i: int, N: int, p: float = 0.5) -> float:
    """Probability of reaching N before 0 (winning): 1 - ruin_probability."""
    return 1.0 - ruin_probability(i, N, p)


def expected_duration(i: int, N: int, p: float = 0.5) -> float:
    """Expected number of rounds until absorption (reaching 0 or N) from i.

    Fair game: i(N - i). Biased: i/(q-p) - N/(q-p) * (1 - r^i)/(1 - r^N), r = q/p."""
    if not 0 <= i <= N:
        raise ValueError("need 0 <= i <= N")
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if i == 0 or i == N:
        return 0.0
    if abs(p - 0.5) < 1e-15:
        return float(i * (N - i))
    q = 1.0 - p
    r = q / p
    return i / (q - p) - (N / (q - p)) * (1.0 - r ** i) / (1.0 - r ** N)


def ruin_probability_infinite_house(i: int, p: float = 0.5) -> float:
    """Ruin probability against an infinitely rich house (N -> infinity), starting from i.

    p <= 1/2: certain ruin (1.0). p > 1/2: (q/p)^i, the chance the downward drift wins first."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if i <= 0:
        return 1.0
    if p <= 0.5:
        return 1.0
    return ((1.0 - p) / p) ** i


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def simulate(i: int, N: int, p: float = 0.5, trials: int = 5000, seed: int = 1):
    """Monte-Carlo play of the walk. Returns (ruin_fraction, mean_duration) over `trials`
    games, each a $1 random walk from i absorbed at 0 or N."""
    rng = _Rng(seed)
    ruins = 0
    total_steps = 0
    for _ in range(trials):
        pos = i
        steps = 0
        while 0 < pos < N:
            pos += 1 if rng.random() < p else -1
            steps += 1
        if pos == 0:
            ruins += 1
        total_steps += steps
    return ruins / trials, total_steps / trials
