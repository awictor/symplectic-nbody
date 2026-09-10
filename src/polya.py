"""Polya's random walk: a drunk man finds his way home, a drunk bird does not.

Start a random walker at the origin of an infinite lattice and let it step to a random nearest
neighbour forever. Will it ever return to where it started? Polya proved in 1921 that the
answer depends only on the dimension:

  1D and 2D: the walk is RECURRENT -- it returns to the origin with probability 1 (and in fact
             visits every site infinitely often),
  3D and up: the walk is TRANSIENT -- there is a nonzero chance it wanders off and never comes
             back.

Kakutani put it memorably: "a drunk man will find his way home, but a drunk bird may get lost
forever." The return probability in 3D is the Watson integral, p_return ~ 0.3405, so a 3D
walker escapes to infinity with probability ~0.66. Higher dimensions escape ever more easily.

The recurrence follows from whether the sum over all times of the probability of being at the
origin, sum_n P(at origin at step n), converges. That probability decays as n^(-d/2) (the
diffusive spreading over ~n^(d/2) sites), so the sum diverges for d <= 2 (recurrent) and
converges for d >= 3 (transient) -- the same n^(-d/2) that makes the Green's-function integral
finite only above two dimensions.

This module gives the return probability by dimension, the recurrence classification, the
expected number of origin visits (infinite when recurrent, finite = 1/(1-p) when transient),
the mean-square displacement of a walk, and a direct simulation of first-return, and
reproduces the certain-return in 1D/2D and the ~0.34 return probability in 3D. Pure stdlib
(seeded LCG, no random module); the stochastic-process companion to the diffusion and
percolation notes.
"""

from __future__ import annotations

import math

# Polya return probabilities by dimension (1,2 are exactly 1; 3+ from Watson-type integrals)
RETURN_PROBABILITY = {1: 1.0, 2: 1.0, 3: 0.340537, 4: 0.193206, 5: 0.135178}


class _Rng:
    """Seeded LCG so simulations are reproducible without the random module."""

    def __init__(self, seed: int = 1):
        self.state = seed & 0xFFFFFFFF

    def randint(self, k: int) -> int:
        # Use the HIGH bits: an LCG's low-order bits have very short periods (the lowest bit
        # of these constants just alternates), which would make randint(2) deterministic.
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 16) % k


def return_probability(dimension: int) -> float:
    """Probability a nearest-neighbour lattice walk ever returns to the origin, by dimension.
    1.0 for d <= 2 (recurrent); ~0.34 in 3D, falling in higher d (transient)."""
    if dimension <= 2:
        return 1.0
    return RETURN_PROBABILITY.get(dimension, RETURN_PROBABILITY[5] * (5.0 / dimension))


def is_recurrent(dimension: int) -> bool:
    """True if the walk is recurrent (returns with probability 1): d <= 2. False (transient)
    for d >= 3 -- Polya's theorem."""
    return dimension <= 2


def expected_visits(dimension: int) -> float:
    """Expected number of times the walk visits the origin (including the start), 1/(1-p).
    Infinite when recurrent (p=1), finite when transient. ~1.516 in 3D."""
    p = return_probability(dimension)
    if p >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - p)


def escape_probability(dimension: int) -> float:
    """Probability the walk NEVER returns to the origin, 1 - p_return. Zero for d <= 2,
    ~0.66 in 3D, rising with dimension."""
    return 1.0 - return_probability(dimension)


def mean_square_displacement(steps: int, step_length: float = 1.0) -> float:
    """Mean-square displacement of a random walk after n steps, <r^2> = n * a^2 (a = step
    length): the diffusive n-scaling, so rms distance grows as sqrt(n)."""
    return steps * step_length * step_length


def simulate_return_fraction(dimension: int, max_steps: int = 2000,
                             trials: int = 200, seed: int = 1) -> float:
    """Fraction of `trials` finite walks (up to max_steps) that return to the origin at least
    once. Approaches 1 for d<=2 as max_steps grows; saturates near p_return for d>=3."""
    returned = 0
    rng = _Rng(seed)
    for t in range(trials):
        pos = [0] * dimension
        came_back = False
        for _ in range(max_steps):
            axis = rng.randint(dimension)
            pos[axis] += 1 if rng.randint(2) == 0 else -1
            if all(x == 0 for x in pos):
                came_back = True
                break
        if came_back:
            returned += 1
    return returned / trials
