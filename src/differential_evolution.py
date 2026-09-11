"""Differential evolution: population optimization driven by vector differences.

Differential evolution (Storn & Price, 1997) is a population-based optimizer for continuous spaces
that, unlike genetic algorithms (which mutate bits) or particle swarms (which track velocities),
mutates by ADDING SCALED DIFFERENCES BETWEEN POPULATION MEMBERS. That self-referential step is its
signature: the spread of the population itself sets the mutation scale, so the search
automatically takes large steps when the population is dispersed (early, exploring) and small ones
as it converges (late, refining) -- no cooling schedule or velocity tuning required. It is a
remarkably robust black-box optimizer, often the first thing to try on a hard continuous problem.

The classic DE/rand/1/bin scheme, per target vector x_i each generation:

  MUTATE   -- pick three other distinct members a, b, c and form a donor v = a + F*(b - c),
              where F (~0.5-0.9) scales the difference vector.
  CROSSOVER-- build a trial by taking each coordinate from the donor with probability CR, else
              from x_i (binomial crossover), guaranteeing at least one donor coordinate.
  SELECT   -- keep whichever of the trial and x_i has the lower cost (greedy, elitist selection).

Because selection is greedy the best-so-far never worsens, and because mutation uses real
population differences DE needs no gradient and no assumptions about the landscape. This module
implements DE/rand/1/bin over a bounded box with bound reflection and a random-search baseline --
verified that it finds the global minimum of the Sphere, Rastrigin, and Rosenbrock benchmarks,
that the best cost decreases monotonically, that it beats random search at equal budget, that a
higher differential weight explores more, and that it solves a higher-dimensional problem. Pure
stdlib (its own RNG); the difference-vector companion to the genetic-algorithm and particle-swarm
notes."""

from __future__ import annotations

import math

from particle_swarm import sphere, rastrigin, rosenbrock     # reuse the standard benchmarks


class _Rng:
    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF

    def uniform(self, lo=0.0, hi=1.0):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return lo + (hi - lo) * ((self.state >> 8) / (1 << 24))     # high bits

    def randint(self, n):
        return int(self.uniform() * n) % n

    def sample3(self, n, exclude):
        """Three distinct indices in [0, n), all != exclude."""
        picks = []
        while len(picks) < 3:
            r = self.randint(n)
            if r != exclude and r not in picks:
                picks.append(r)
        return picks


def _reflect(x, lo, hi):
    """Keep a coordinate inside [lo, hi] by reflecting off the walls (better than clamping for DE)."""
    span = hi - lo
    if span <= 0:
        return lo
    while x < lo or x > hi:
        if x < lo:
            x = lo + (lo - x)
        elif x > hi:
            x = hi - (x - hi)
    return x


def minimize(cost_fn, bounds, pop_size=None, n_iter=200, F=0.7, CR=0.9, seed=0, track=False):
    """Minimize cost_fn over an axis-aligned box by DE/rand/1/bin.

    bounds  : list of (lo, hi) per dimension.
    F       : differential weight (scales the difference vector, ~0.5-0.9).
    CR      : crossover probability (fraction of coordinates taken from the donor).
    pop_size: population; defaults to max(10, 5*dim).
    Returns (best_vector, best_cost) or, with track=True, also a history dict."""
    rng = _Rng(seed)
    dim = len(bounds)
    if pop_size is None:
        pop_size = max(10, 5 * dim)

    pop = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(pop_size)]
    cost = [cost_fn(ind) for ind in pop]
    best_i = min(range(pop_size), key=lambda i: cost[i])
    best, best_cost = list(pop[best_i]), cost[best_i]
    history = [best_cost]

    for _ in range(n_iter):
        for i in range(pop_size):
            a, b, c = rng.sample3(pop_size, i)
            # DONOR: v = a + F*(b - c)
            donor = [pop[a][d] + F * (pop[b][d] - pop[c][d]) for d in range(dim)]
            # BINOMIAL CROSSOVER: at least one coordinate (jrand) always comes from the donor
            jrand = rng.randint(dim)
            trial = list(pop[i])
            for d in range(dim):
                if rng.uniform() < CR or d == jrand:
                    lo, hi = bounds[d]
                    trial[d] = _reflect(donor[d], lo, hi)
            # GREEDY SELECTION: keep the trial only if it is at least as good
            tc = cost_fn(trial)
            if tc <= cost[i]:
                pop[i] = trial
                cost[i] = tc
                if tc < best_cost:
                    best, best_cost = list(trial), tc
        history.append(best_cost)

    if track:
        return best, best_cost, {"history": history}
    return best, best_cost


def random_search(cost_fn, bounds, n_eval=6000, seed=0):
    """Baseline: sample n_eval uniform-random points in the box, keep the best."""
    rng = _Rng(seed)
    best, best_cost = None, math.inf
    for _ in range(n_eval):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        c = cost_fn(x)
        if c < best_cost:
            best, best_cost = x, c
    return best, best_cost


__all__ = ["minimize", "random_search", "sphere", "rastrigin", "rosenbrock"]
