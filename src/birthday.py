"""The birthday problem: coincidences are far more common than they feel.

How many people must be in a room before two of them are more likely than not to share a
birthday? The intuition says "a lot" -- there are 365 days -- but the answer is only 23. The
reason is that k people make k(k-1)/2 pairs, and it is the number of pairs, growing like k^2,
that drives collisions, not k itself. The probability that all k birthdays are distinct is

    P(distinct) = prod_{i=0}^{k-1} (d - i) / d = d! / ((d-k)! d^k),

so the collision probability is 1 - that. With d = 365 it passes 1/2 at k = 23 and reaches 99.9%
by k = 70. In general a collision becomes likely once k ~ 1.177 sqrt(d) (the median), and the
expected number of draws until the first collision is ~ sqrt(pi d / 2).

This "square-root" law is exactly why it matters far beyond birthdays: it sets the difficulty
of the birthday attack on hash functions (a b-bit hash collides after ~2^(b/2) tries, not 2^b),
sizes hash tables and UUID spaces, and explains why shared-birthday and other coincidences feel
uncanny but are not. This module gives the exact collision and distinct probabilities, the
smallest group for a target collision probability, the median and expected first-collision
counts, the Poisson/exponential approximation for large d, and a seeded Monte-Carlo check. Pure
stdlib; the coincidence-counting companion to the coupon-collector and secretary notes.
"""

from __future__ import annotations

import math


def prob_distinct(k: int, days: int = 365) -> float:
    """Probability that k people all have distinct birthdays among `days` equally likely days.

        P = prod_{i=0}^{k-1} (days - i) / days.

    Zero once k > days (pigeonhole)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k > days:
        return 0.0
    p = 1.0
    for i in range(k):
        p *= (days - i) / days
    return p


def prob_collision(k: int, days: int = 365) -> float:
    """Probability that at least two of k people share a birthday: 1 - P(distinct)."""
    return 1.0 - prob_distinct(k, days)


def min_people_for(prob: float, days: int = 365) -> int:
    """Smallest group size k for which the collision probability is at least `prob`."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    if prob == 0.0:
        return 0
    k = 1
    while prob_collision(k, days) < prob:
        k += 1
        if k > days:  # certain collision by pigeonhole
            return days + 1
    return k


def median_collision(days: int = 365) -> float:
    """Approximate group size at which a collision is 50% likely: sqrt(2 days ln 2) ~ 1.177
    sqrt(days). For days = 365 this is ~22.5, matching the exact answer of 23."""
    return math.sqrt(2.0 * days * math.log(2.0))


def expected_first_collision(days: int = 365) -> float:
    """Expected number of people drawn until the first shared birthday (the birthday-attack
    Q-function), well approximated for large d by sqrt(pi days / 2)."""
    return math.sqrt(math.pi * days / 2.0)


def collision_approx(k: int, days: int = 365) -> float:
    """Poisson/exponential approximation to the collision probability:

        P ~ 1 - exp(-k(k-1) / (2 days)),

    from treating the k(k-1)/2 pairs as independent rare collision events. Accurate for
    days >> k."""
    return 1.0 - math.exp(-k * (k - 1) / (2.0 * days))


def hash_collision_tries(bits: int) -> float:
    """Expected number of random inputs before a b-bit hash function collides: ~sqrt(pi/2)
    2^(b/2), the birthday-attack cost. A 128-bit hash collides after ~2^64 tries, not 2^128,
    which is why collision resistance needs twice the bits of preimage resistance."""
    return math.sqrt(math.pi / 2.0) * 2.0 ** (bits / 2.0)


class _Rng:
    """Seeded LCG; high bits (an LCG's low bits are not random)."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def simulate_collision(k: int, days: int = 365, trials: int = 5000, seed: int = 1) -> float:
    """Monte-Carlo estimate of the collision probability for k people over `days` days."""
    rng = _Rng(seed)
    hits = 0
    for _ in range(trials):
        seen = set()
        collided = False
        for _ in range(k):
            b = rng.randint(days)
            if b in seen:
                collided = True
                break
            seen.add(b)
        if collided:
            hits += 1
    return hits / trials


def simulate_first_collision(days: int = 365, trials: int = 5000, seed: int = 1) -> float:
    """Monte-Carlo mean number of draws until the first collision over `days` days."""
    rng = _Rng(seed)
    total = 0
    for _ in range(trials):
        seen = set()
        draws = 0
        while True:
            draws += 1
            b = rng.randint(days)
            if b in seen:
                break
            seen.add(b)
        total += draws
    return total / trials
