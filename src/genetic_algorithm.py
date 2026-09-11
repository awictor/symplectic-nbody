"""Genetic algorithms: optimization by simulated evolution.

Where simulated annealing perturbs a SINGLE state, a genetic algorithm evolves a whole POPULATION,
letting good solutions breed. It is a direct metaphor for natural selection (Holland, 1975): each
candidate is an "individual" with a fitness; the fittest are selected to reproduce; CROSSOVER
recombines two parents' genes into offspring; and MUTATION injects random variation so the search
never fully converges to one point. Over generations the population's average fitness climbs and
good building blocks spread.

The loop, each generation:

  SELECT    -- pick parents biased toward high fitness. TOURNAMENT selection (take the best of k
               random individuals) is simple and robust, its pressure tuned by k.
  CROSSOVER -- with some probability, splice two parents (one-point for bit/real vectors) so a child
               inherits a mix; otherwise a parent passes through.
  MUTATE    -- flip bits / jitter genes with a small probability, the source of new material.
  ELITISM   -- copy the best few individuals unchanged, so the best-so-far never regresses.

Unlike gradient methods it needs no derivatives and, being population-based, explores many basins
at once -- good for rugged, discrete, or black-box landscapes. This module implements a generic GA
(tournament selection, one-point crossover, elitism) over both binary and real-valued genomes,
plus a 0/1 knapsack solver -- verified that it solves the OneMax bit problem to all-ones, maximizes
a multimodal real function, finds the known optimum of a small knapsack, that fitness improves
monotonically under elitism, and that it beats random search at equal budget. Pure stdlib (its own
RNG); the population-based companion to the simulated-annealing note."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF

    def uniform(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)     # high bits

    def randint(self, n):
        return int(self.uniform() * n) % n

    def gauss(self, sigma=1.0):
        u1 = max(1e-12, self.uniform())
        u2 = self.uniform()
        return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _tournament(pop, fitnesses, k, rng):
    """Return a copy of the best of k random individuals."""
    best_i = rng.randint(len(pop))
    for _ in range(k - 1):
        j = rng.randint(len(pop))
        if fitnesses[j] > fitnesses[best_i]:
            best_i = j
    ind = pop[best_i]
    return list(ind) if isinstance(ind, list) else ind


def _one_point_crossover(a, b, rng):
    """Splice two equal-length genomes at a random cut point."""
    n = len(a)
    if n < 2:
        return list(a), list(b)
    cut = 1 + rng.randint(n - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def genetic_algorithm(fitness_fn, init_fn, mutate_fn, n_generations=100, pop_size=60,
                      tournament_k=3, crossover_rate=0.8, elitism=2, seed=0, track=False):
    """Maximize fitness_fn by evolving a population.

    fitness_fn(individual) -> scalar (higher is better)
    init_fn(rng)           -> a random individual (a list genome)
    mutate_fn(individual, rng) -> a mutated copy
    Returns (best_individual, best_fitness) or, with track=True, also a history dict."""
    rng = _Rng(seed)
    pop = [init_fn(rng) for _ in range(pop_size)]
    fits = [fitness_fn(ind) for ind in pop]
    best_i = max(range(pop_size), key=lambda i: fits[i])
    best, best_fit = list(pop[best_i]), fits[best_i]
    best_history = [best_fit]
    mean_history = [sum(fits) / pop_size]

    for _ in range(n_generations):
        # elitism: carry the top individuals forward unchanged
        order = sorted(range(pop_size), key=lambda i: -fits[i])
        new_pop = [list(pop[order[e]]) for e in range(elitism)]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, tournament_k, rng)
            p2 = _tournament(pop, fits, tournament_k, rng)
            if rng.uniform() < crossover_rate:
                c1, c2 = _one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = list(p1), list(p2)
            new_pop.append(mutate_fn(c1, rng))
            if len(new_pop) < pop_size:
                new_pop.append(mutate_fn(c2, rng))
        pop = new_pop
        fits = [fitness_fn(ind) for ind in pop]
        gen_best_i = max(range(pop_size), key=lambda i: fits[i])
        if fits[gen_best_i] > best_fit:
            best, best_fit = list(pop[gen_best_i]), fits[gen_best_i]
        best_history.append(best_fit)
        mean_history.append(sum(fits) / pop_size)

    if track:
        return best, best_fit, {"best_history": best_history, "mean_history": mean_history}
    return best, best_fit


# --- ready-made genome factories -----------------------------------------
def binary_init(length):
    """An init_fn producing random 0/1 genomes of the given length."""
    return lambda rng: [rng.randint(2) for _ in range(length)]


def binary_mutate(flip_rate):
    """A mutate_fn flipping each bit with probability flip_rate."""
    def m(genome, rng):
        return [1 - g if rng.uniform() < flip_rate else g for g in genome]
    return m


def real_init(bounds):
    """An init_fn producing random real vectors within per-dimension (lo, hi) bounds."""
    return lambda rng: [lo + (hi - lo) * rng.uniform() for lo, hi in bounds]


def real_mutate(sigma, bounds):
    """A mutate_fn adding Gaussian jitter, clamped to bounds."""
    def m(genome, rng):
        out = []
        for i, g in enumerate(genome):
            v = g + rng.gauss(sigma)
            lo, hi = bounds[i]
            out.append(min(hi, max(lo, v)))
        return out
    return m


def solve_knapsack(weights, values, capacity, n_generations=200, pop_size=80, seed=0):
    """0/1 knapsack by GA: choose items maximizing value under a weight capacity.

    Returns (chosen_bits, total_value, total_weight). Over-capacity solutions are penalized."""
    n = len(weights)
    max_val = sum(values)

    def fitness(bits):
        w = sum(weights[i] for i in range(n) if bits[i])
        v = sum(values[i] for i in range(n) if bits[i])
        if w > capacity:
            return -(w - capacity) * max_val        # heavy penalty for infeasible
        return v

    best, best_fit = genetic_algorithm(fitness, binary_init(n), binary_mutate(1.0 / n),
                                        n_generations=n_generations, pop_size=pop_size, seed=seed)
    total_w = sum(weights[i] for i in range(n) if best[i])
    total_v = sum(values[i] for i in range(n) if best[i])
    return best, total_v, total_w
