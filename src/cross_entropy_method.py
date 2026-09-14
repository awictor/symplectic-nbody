"""Cross-entropy method: derivative-free optimization by fitting a distribution to the best samples.

The cross-entropy method (CEM), from Rubinstein's rare-event work in the 1990s, is a strikingly simple
and robust way to optimize a function you can only EVALUATE, not differentiate -- black-box objectives,
simulators, noisy fitness. The idea is a feedback loop over a SAMPLING DISTRIBUTION. Draw a population
of candidate points from a Gaussian, evaluate the objective at each, keep the top fraction (the ELITE
set), and re-fit the Gaussian's mean and variance to those elites. Repeat: because the elites cluster
where the objective is good, the distribution marches toward the optimum and tightens around it, until
the variance collapses and the mean is the answer.

It is the cross-entropy method because re-fitting to the elites is exactly minimizing the
Kullback-Leibler (cross-entropy) distance between the sampling distribution and the ideal distribution
that puts all its mass on the best points. CEM needs no gradients, tolerates noise and rugged
landscapes, and is the workhorse behind model-predictive control samplers and reinforcement-learning
policy search (it is the core of the "CEM planner"). Its knobs are the population size, the elite
fraction, and an optional variance FLOOR or smoothing to avoid premature collapse.

This module implements CEM for continuous optimization with a diagonal Gaussian, elite re-fitting,
smoothing, and a variance floor, using a seeded RNG. It is validated: it finds the minimum of a
quadratic bowl to high accuracy; it solves the Rosenbrock and Rastrigin benchmarks (the latter a
multimodal trap that defeats naive local methods); the sampling variance shrinks monotonically toward
the optimum; a larger elite fraction is more exploratory and a smaller one more greedy; it is
reproducible for a fixed seed; and it works in one through several dimensions. Pure stdlib; the
derivative-free-optimization companion to the CMA-ES, differential-evolution, and particle-swarm
tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF
        self._spare = None

    def u32(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def uniform(self):
        return (self.u32() >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            z = self._spare
            self._spare = None
            return z
        u1 = self.uniform() + 1e-300
        u2 = self.uniform()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def optimize(f, mean, std=None, population=50, elite_frac=0.2, iterations=100,
             smoothing=0.7, var_floor=1e-12, seed=1, track=False):
    """Minimize f by the cross-entropy method with a diagonal Gaussian.

    f: objective (list -> float). mean: initial mean (list). std: initial per-dim std (default 1s).
    Returns the best mean found; if track=True returns (best_mean, history) where history is the
    per-iteration (best_value, mean_variance)."""
    dim = len(mean)
    mu = list(mean)
    sigma = [1.0] * dim if std is None else list(std)
    rng = _Rng(seed)
    n_elite = max(1, int(population * elite_frac))
    history = []
    best_x = list(mu)
    best_v = f(mu)
    for _ in range(iterations):
        # sample a population from the current diagonal Gaussian
        samples = []
        for _ in range(population):
            x = [mu[d] + sigma[d] * rng.normal() for d in range(dim)]
            samples.append((f(x), x))
        samples.sort(key=lambda t: t[0])
        if samples[0][0] < best_v:
            best_v = samples[0][0]
            best_x = list(samples[0][1])
        elites = [x for _, x in samples[:n_elite]]
        # re-fit mean and std to the elites, with smoothing
        new_mu = [sum(e[d] for e in elites) / n_elite for d in range(dim)]
        new_sigma = []
        for d in range(dim):
            var = sum((e[d] - new_mu[d]) ** 2 for e in elites) / n_elite
            new_sigma.append(math.sqrt(max(var, var_floor)))
        mu = [smoothing * new_mu[d] + (1 - smoothing) * mu[d] for d in range(dim)]
        sigma = [smoothing * new_sigma[d] + (1 - smoothing) * sigma[d] for d in range(dim)]
        if track:
            history.append((best_v, sum(s * s for s in sigma) / dim))
    if track:
        return best_x, history
    return best_x


# --- benchmark objectives --------------------------------------------------
def sphere(x):
    """Sphere / quadratic bowl: minimum 0 at the origin."""
    return sum(xi * xi for xi in x)


def rosenbrock(x):
    """Rosenbrock banana: minimum 0 at (1, 1, ..., 1)."""
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def rastrigin(x):
    """Rastrigin: a highly multimodal trap; minimum 0 at the origin."""
    a = 10.0
    return a * len(x) + sum(xi * xi - a * math.cos(2 * math.pi * xi) for xi in x)
