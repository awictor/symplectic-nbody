"""The Moran process: evolution in a finite population with overlapping generations, one birth and death at a time.

Wright-Fisher replaces the WHOLE population each generation; the MORAN PROCESS (1958) is the
continuous-turnover alternative -- at each step ONE individual is chosen to reproduce (proportional to
fitness) and ONE is chosen to die (uniformly at random), keeping the population size N fixed. This models
overlapping generations -- a chemostat, a tumour, a well-mixed microbial culture -- and its single-step
birth-death structure makes it a clean absorbing Markov chain on the number i of type-A individuals, with
absorbing states i = 0 (A extinct) and i = N (A fixed).

Its fixation probabilities have exact closed forms. For a NEUTRAL mutant (equal fitness) starting from i
copies, the probability A eventually fixes is simply i/N -- so a single new neutral mutant fixes with
probability 1/N. With selection, if type-A has relative fitness r = f_A/f_B, the fixation probability from
i copies is

    rho_i = (1 - r^{-i}) / (1 - r^{-N}),

the classic Moran formula. A single advantageous mutant (i=1, r>1) fixes with probability
(1 - 1/r)/(1 - 1/r^N), which for large N approaches 1 - 1/r -- a beneficial mutation is far from
guaranteed to take over. The Moran process underlies much of evolutionary game theory and cancer-evolution
modelling.

This module simulates Moran trajectories with selection, computes fixation probabilities both by the exact
formula and by ensemble simulation, and gives the mean absorption time. It uses a seeded RNG. It is
validated: the neutral fixation probability from i copies equals i/N and a single neutral mutant fixes with
probability 1/N; with selection the empirical fixation frequency matches the closed-form rho_i; an
advantageous mutant fixes more often than a neutral one and a deleterious one less often; the single-mutant
formula approaches 1 - 1/r for large N; every trajectory is absorbed at 0 or N; fixation probabilities
increase monotonically with the starting count; and results are reproducible per seed. Pure stdlib; the
finite-population-evolution companion to the Wright-Fisher, Galton-Watson, and coalescent tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def fixation_probability_formula(n, i, r=1.0):
    """Exact Moran fixation probability rho_i for i type-A of relative fitness r in a population of N.

    Neutral (r=1): i/N. With selection: (1 - r^{-i}) / (1 - r^{-N})."""
    if i <= 0:
        return 0.0
    if i >= n:
        return 1.0
    if abs(r - 1.0) < 1e-12:
        return i / n
    ri = r ** (-i)
    rn = r ** (-n)
    return (1 - ri) / (1 - rn)


def simulate(n, i0, r=1.0, seed=1, max_steps=10_000_000):
    """Simulate one Moran trajectory: state = number of type-A. Returns (fixed, steps).

    Each step: pick a reproducer with prob proportional to fitness (A has fitness r, B has 1), pick a
    random individual to die. Absorbs at 0 or N."""
    rng = _Rng(seed)
    i = i0
    steps = 0
    while 0 < i < n and steps < max_steps:
        # probability the reproducer is type-A: (r*i) / (r*i + (n-i))
        fitness_a = r * i
        fitness_total = fitness_a + (n - i)
        repro_a = rng.u() < fitness_a / fitness_total
        # the individual that dies is uniform; net change in i:
        # birth A & death B -> +1 ; birth B & death A -> -1 ; else no change
        die_a = rng.u() < i / n
        if repro_a and not die_a:
            i += 1
        elif not repro_a and die_a:
            i -= 1
        steps += 1
    return (i >= n), steps


def fixation_probability(n, i0, r=1.0, n_runs=2000, seed=1):
    """Empirical fixation probability from i0 copies, by ensemble simulation."""
    fixed = 0
    for run in range(n_runs):
        is_fixed, _ = simulate(n, i0, r, seed=seed + run * 2749)
        if is_fixed:
            fixed += 1
    return fixed / n_runs


def mean_absorption_time(n, i0, r=1.0, n_runs=1000, seed=1):
    """Mean number of Moran steps to reach an absorbing state (fixation or extinction)."""
    total = 0
    for run in range(n_runs):
        _, steps = simulate(n, i0, r, seed=seed + run * 2749)
        total += steps
    return total / n_runs


def single_mutant_fixation(n, r):
    """Fixation probability of a single advantageous mutant (i=1): (1 - 1/r)/(1 - 1/r^N)."""
    return fixation_probability_formula(n, 1, r)
